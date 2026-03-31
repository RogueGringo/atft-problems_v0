#!/usr/bin/env python3
"""Harmonic Stack — spectral decomposition training script.

Two experiment modes:
  --stage prism  : Train wide shallow {0,1,3} transformer (Experiment 1)
  --stage full   : Freeze trained prism, add router + analyzers (Experiment 2)

Usage:
    python run_harmonic.py --stage prism --width 2048 --n_heads 16
    python run_harmonic.py --stage full  --prism_checkpoint results/.../model.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "topological-router"))

from ternary_linear import TernaryLinear
from harmonic_stack import HarmonicStack, HarmonicConfig, BandAnalyzer
from run_long import TextChunked, split_dataset, measure_perplexity, weight_stats
from topo_measures import effective_rank, spectral_gap, gini_fast


# ── Output directory ──────────────────────────────────────────────────────

def get_output_dir(stage: str, prism_layers: int, width: int) -> Path:
    d = Path(__file__).parent / "results" / f"harmonic_{stage}_{prism_layers}L_{width}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Topology measurement ──────────────────────────────────────────────────

def measure_topology_harmonic(model: HarmonicStack, sample_batch, device) -> dict:
    """Topology of hidden states from the prism blocks."""
    model.eval()
    with torch.no_grad():
        x = sample_batch[0][:1].to(device)
        _, _, hs = model(x, return_hidden=True)
    results = {}
    for i, h in enumerate(hs):
        h0 = h[0]   # (T, d)
        results[f"layer_{i}"] = {
            "eff_rank":    effective_rank(h0),
            "spectral_gap": spectral_gap(h0),
            "gini_sv":     gini_fast(torch.linalg.svdvals(h0.float().cpu()).numpy()),
        }
    keys = ["eff_rank", "spectral_gap", "gini_sv"]
    for k in keys:
        vals = [results[f"layer_{i}"][k] for i in range(len(hs))]
        results[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
    model.train()
    return results


# ── Per-layer crystal ─────────────────────────────────────────────────────

def per_layer_crystal(model: HarmonicStack) -> dict:
    """Per-TernaryLinear weight distribution (w0/w1/w3 fractions)."""
    crystals = {}
    for name, m in model.named_modules():
        if isinstance(m, TernaryLinear):
            wq = m.get_quantized_weight()
            total = wq.numel()
            if total == 0:
                continue
            crystals[name] = {
                "w0": (wq == 0).sum().item() / total,
                "w1": (wq == 1).sum().item() / total,
                "w3": (wq == 3).sum().item() / total,
                "n":  total,
            }
    return crystals


# ── Analyzer crystals ─────────────────────────────────────────────────────

def analyzer_crystals(model: HarmonicStack) -> dict | None:
    """Per-BandAnalyzer weight distribution.

    Reads model.band_analyzers[0].proj, [1].proj, [2].proj.
    Returns None if band_analyzers is not yet built.
    """
    if model.band_analyzers is None:
        return None
    band_names = ["void", "identity", "prime"]
    result = {}
    for i, (band, analyzer) in enumerate(zip(band_names, model.band_analyzers)):
        proj = analyzer.proj
        if isinstance(proj, TernaryLinear):
            wq = proj.get_quantized_weight()
            total = wq.numel()
            result[band] = {
                "w0": (wq == 0).sum().item() / total,
                "w1": (wq == 1).sum().item() / total,
                "w3": (wq == 3).sum().item() / total,
                "n":  total,
            }
        else:
            # Fallback for non-ternary projections
            w = proj.weight.detach().abs()
            result[band] = {"mean_abs": w.mean().item(), "n": w.numel()}
    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Harmonic Stack training")
    parser.add_argument("--stage", choices=["prism", "full"], default="prism")
    parser.add_argument("--width", type=int, default=2048,
                        help="n_embd for the prism (default 2048)")
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--prism_layers", type=int, default=1)
    parser.add_argument("--analyzer_width", type=int, default=512,
                        help="BandAnalyzer output width (stage=full only)")
    parser.add_argument("--prism_checkpoint", type=str, default=None,
                        help="Path to model.pt — required for stage=full")
    parser.add_argument("--dataset", default="wikitext",
                        choices=["tinystories", "wikitext", "kant", "sep"])
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=200000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ternary_decay", type=float, default=0.01,
                        help="Weight decay on ternary params (default 0.01)")
    args = parser.parse_args()

    if args.stage == "full" and args.prism_checkpoint is None:
        parser.error("--prism_checkpoint is required for --stage full")

    OUTPUT_DIR = get_output_dir(args.stage, args.prism_layers, args.width)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    full_ds = TextChunked(tokenizer, max_length=args.max_length,
                          n_samples=args.n_samples,
                          dataset_name=args.dataset)
    train_ds, test_ds = split_dataset(full_ds)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True, drop_last=True)
    sample_batch = next(iter(train_loader))

    # Model config
    config = HarmonicConfig(
        vocab_size    = tokenizer.vocab_size,
        block_size    = args.max_length,
        n_prism_layers= args.prism_layers,
        n_embd        = args.width,
        n_head        = args.n_heads,
        dropout       = 0.0,
        weight_set    = "013",
        analyzer_width= args.analyzer_width,
    )

    # ── Stage-specific model construction ────────────────────────────────

    if args.stage == "prism":
        model = HarmonicStack(config, stage="prism").to(device)
        params = model.count_params()
        print(f"\n{params['total']:,} params ({params['ternary_pct']:.1f}% ternary)")

        # Separate ternary and continuous params
        ternary_params = []
        continuous_params = []
        for name, p in model.named_parameters():
            is_ternary = False
            for mname, m in model.named_modules():
                if isinstance(m, TernaryLinear) and name.startswith(mname + "."):
                    is_ternary = True
                    break
            if is_ternary:
                ternary_params.append(p)
            else:
                continuous_params.append(p)

        optim_groups = []
        if ternary_params:
            optim_groups.append({"params": ternary_params,
                                 "weight_decay": args.ternary_decay})
            print(f"  Ternary params (L2={args.ternary_decay}): "
                  f"{sum(p.numel() for p in ternary_params):,}")
        optim_groups.append({"params": continuous_params, "weight_decay": 0.01})
        print(f"  Continuous params (L2=0.01): "
              f"{sum(p.numel() for p in continuous_params):,}")

    else:  # stage == "full"
        model = HarmonicStack(config, stage="full").to(device)

        # Load prism weights (strict=False — lm_head shape differs)
        ckpt_path = Path(args.prism_checkpoint)
        print(f"\nLoading prism checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing keys:    {missing}")
        print(f"  Unexpected keys: {unexpected}")

        # Wire up router + analyzers
        prism_ternary = model.get_last_prism_ternary()
        if prism_ternary is None:
            raise RuntimeError("Could not find last prism TernaryLinear layer.")
        model.build_stage2(prism_ternary)
        model = model.to(device)   # move newly created modules to device

        # Band sizes
        sizes = model.router.band_sizes()
        print(f"  Router band sizes: void={sizes['void']} "
              f"identity={sizes['identity']} prime={sizes['prime']}")

        params = model.count_params()
        print(f"\n{params['total']:,} params ({params['ternary_pct']:.1f}% ternary)")

        # Only train unfrozen params (analyzers + new lm_head)
        ternary_params = []
        continuous_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_ternary = False
            for mname, m in model.named_modules():
                if isinstance(m, TernaryLinear) and name.startswith(mname + "."):
                    is_ternary = True
                    break
            if is_ternary:
                ternary_params.append(p)
            else:
                continuous_params.append(p)

        optim_groups = []
        if ternary_params:
            optim_groups.append({"params": ternary_params,
                                 "weight_decay": args.ternary_decay})
            print(f"  Trainable ternary params (L2={args.ternary_decay}): "
                  f"{sum(p.numel() for p in ternary_params):,}")
        optim_groups.append({"params": continuous_params, "weight_decay": 0.01})
        print(f"  Trainable continuous params (L2=0.01): "
              f"{sum(p.numel() for p in continuous_params):,}")

    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)

    # LR warmup (2000 steps) + cosine decay
    warmup_steps = min(2000, args.max_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return 0.05 + 0.95 * (1 + np.cos(np.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    accum = max(1, args.effective_batch // args.batch_size)

    # Training log
    init_wdist = weight_stats(model)
    log = {
        "stage": args.stage,
        "config": vars(args),
        "params": params,
        "experiment": {
            "init_weight_dist": init_wdist,
            "weight_decay_ternary": args.ternary_decay,
            "weight_decay_continuous": 0.01,
            "warmup_steps": warmup_steps,
            "prism_checkpoint": args.prism_checkpoint,
        },
        "checkpoints": [],
        "steps": [],
    }

    step = 0
    start = time.time()
    prev_eff_rank = None
    compression_stalled = 0

    print(f"\nTraining {args.max_steps} steps | stage={args.stage} | "
          f"accum={accum} | warmup={warmup_steps}\n")
    model.train()
    optimizer.zero_grad()

    while step < args.max_steps:
        for bx, by in train_loader:
            if step >= args.max_steps:
                break

            bx, by = bx.to(device), by.to(device)
            _, loss = model(bx, targets=by)
            (loss / accum).backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(continuous_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Quick log every 200 steps
            if step % 200 == 0:
                elapsed = time.time() - start
                ppl = torch.exp(loss).item()
                ppl_s = f"{ppl:.1f}" if ppl < 1e5 else f"{ppl:.0e}"
                log["steps"].append({"step": step, "loss": loss.item(),
                                     "ppl": min(ppl, 1e7), "elapsed": elapsed})
                print(f"  {step:6d} | loss {loss.item():.4f} | ppl {ppl_s:>10s} | "
                      f"{elapsed:.0f}s", flush=True)

            # Full checkpoint every 1000 steps
            if step > 0 and step % 1000 == 0:
                train_ppl = measure_perplexity(model, train_loader, device)
                test_ppl  = measure_perplexity(model, test_loader,  device)
                topo      = measure_topology_harmonic(model, sample_batch, device)
                wstats    = weight_stats(model)
                plc       = per_layer_crystal(model)
                gen_gap   = (test_ppl["ppl"] / train_ppl["ppl"]
                             if train_ppl["ppl"] > 0 else 999.0)

                ckpt = {
                    "step":       step,
                    "train_ppl":  train_ppl,
                    "test_ppl":   test_ppl,
                    "gen_gap":    gen_gap,
                    "topology": {
                        "eff_rank":     topo["mean_eff_rank"],
                        "spectral_gap": topo["mean_spectral_gap"],
                        "gini_sv":      topo["mean_gini_sv"],
                    },
                    "weight_dist":     wstats,
                    "per_layer_crystal": plc,
                    "elapsed":    time.time() - start,
                }

                if args.stage == "full":
                    ckpt["analyzer_crystals"] = analyzer_crystals(model)

                log["checkpoints"].append(ckpt)

                # Compression tracking
                curr_rank  = topo["mean_eff_rank"]
                rank_delta = (prev_eff_rank - curr_rank) if prev_eff_rank is not None else 0.0
                if prev_eff_rank is not None and abs(rank_delta) < 1.0:
                    compression_stalled += 1
                else:
                    compression_stalled = 0
                prev_eff_rank = curr_rank

                print(f"\n    CKPT {step:6d} | train {train_ppl['ppl']:.1f} | "
                      f"test {test_ppl['ppl']:.1f} | gap {gen_gap:.3f}")
                print(f"    eff_rank {curr_rank:.1f} (Δ{rank_delta:+.1f}) | "
                      f"spec_gap {topo['mean_spectral_gap']:.1f} | "
                      f"gini {topo['mean_gini_sv']:.3f}")
                print(f"    weights: 0={wstats['zero']:.3f} 1={wstats['one']:.3f} "
                      f"3={wstats['three']:.3f}")
                if args.stage == "full" and ckpt["analyzer_crystals"]:
                    ac = ckpt["analyzer_crystals"]
                    for band in ["void", "identity", "prime"]:
                        if band in ac and "w0" in ac[band]:
                            b = ac[band]
                            print(f"    analyzer[{band}]: "
                                  f"0={b['w0']:.3f} 1={b['w1']:.3f} 3={b['w3']:.3f}")
                if compression_stalled >= 3:
                    print(f"    ** COMPRESSION SATURATED "
                          f"(stalled {compression_stalled} ckpts) **")
                print(flush=True)

                # Save intermediate log
                with open(OUTPUT_DIR / "training_log.json", "w") as f:
                    json.dump(log, f, indent=2, default=str)

            step += 1

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed   = time.time() - start
    train_ppl = measure_perplexity(model, train_loader, device)
    test_ppl  = measure_perplexity(model, test_loader,  device)
    topo      = measure_topology_harmonic(model, sample_batch, device)
    wstats    = weight_stats(model)
    plc       = per_layer_crystal(model)

    # Generations
    model.eval()
    prompts = ["Once upon a time", "The little dog",
               "She was happy because", "He looked at the big",
               "They went to the park"]
    gens = []
    for prompt in prompts:
        toks = tokenizer.encode(prompt, return_tensors="pt").to(device)
        out  = model.generate(toks, max_new=80, temperature=0.8)
        text = tokenizer.decode(out[0].cpu(), skip_special_tokens=True)
        gens.append(text)
        print(f"  > {text[:120]}")

    final_entry = {
        "elapsed":   elapsed,
        "train_ppl": train_ppl,
        "test_ppl":  test_ppl,
        "gen_gap":   test_ppl["ppl"] / train_ppl["ppl"],
        "topology": {
            "eff_rank":     topo["mean_eff_rank"],
            "spectral_gap": topo["mean_spectral_gap"],
            "gini_sv":      topo["mean_gini_sv"],
        },
        "weight_dist":      wstats,
        "per_layer_crystal": plc,
        "generations":      gens,
    }
    if args.stage == "full":
        final_entry["analyzer_crystals"] = analyzer_crystals(model)
        final_entry["band_sizes"] = model.router.band_sizes()

    log["final"] = final_entry

    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")

    print(f"\n{'='*60}")
    print(f"  Harmonic Stack [{args.stage}] — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  train_ppl={train_ppl['ppl']:.1f} | test_ppl={test_ppl['ppl']:.1f}")
    print(f"  gen_gap={test_ppl['ppl']/train_ppl['ppl']:.3f}")
    print(f"  eff_rank={topo['mean_eff_rank']:.1f} | "
          f"spectral_gap={topo['mean_spectral_gap']:.1f}")
    print(f"  weights: 0={wstats['zero']:.3f} 1={wstats['one']:.3f} "
          f"3={wstats['three']:.3f}")
    if args.stage == "full":
        sizes = model.router.band_sizes()
        print(f"  bands: void={sizes['void']} identity={sizes['identity']} "
              f"prime={sizes['prime']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
