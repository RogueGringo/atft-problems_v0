#!/usr/bin/env python3
"""Long run — {0,1,3} at 50K steps. Walk the compression vector.

The common denominator across all discrete weight experiments:
  - Eff rank DECREASES (compresses, builds hierarchy)
  - Spectral gap INCREASES (structures, differentiates)
  - Gen gap stays ≤ 1.0 (generalizes, doesn't memorize)
  - Converges iteratively (has fixed point)

{0,1,3} walks this vector fastest. This run trains until the
compression saturates — where the architecture says "I've found
all the structure there is."

Measurements every 1000 steps. Full topology battery.

Usage:
    python run_long.py
    python run_long.py --max_steps 100000
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

def get_output_dir(size="small"):
    d = Path(__file__).parent / "results" / f"long_run_{size}"
    d.mkdir(parents=True, exist_ok=True)
    return d

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "topological-router"))

from ternary_linear import TernaryLinear, TernaryMutator, BitFlipLinear, BitFlipEngine
from ternary_transformer import build_model
from topo_measures import effective_rank, spectral_gap, gini_fast
from measure import zero_mask_topology, iterative_inference


# ── Dataset ───────────────────────────────────────────────────────────────

class TextChunked(Dataset):
    """Chunked text dataset — supports TinyStories and WikiText."""

    def __init__(self, tokenizer, max_length=256, n_samples=200000,
                 dataset_name="tinystories"):
        from datasets import load_dataset

        if dataset_name == "wikitext":
            print(f"Loading WikiText-103 ({n_samples} samples)...")
            ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                              split="train", streaming=False)
            ds = ds.select(range(min(n_samples, len(ds))))
        elif dataset_name == "kant":
            print("Loading Kant's Critique of Pure Reason...")
            kant_path = Path(__file__).parent / "data" / "kant_critique.txt"
            text = kant_path.read_text()
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
            ds = [{"text": p} for p in paragraphs]
            print(f"  {len(ds)} paragraphs")
        elif dataset_name == "sep":
            print(f"Loading Stanford Encyclopedia of Philosophy ({n_samples} entries)...")
            ds = load_dataset("AiresPucrs/stanford-encyclopedia-philosophy",
                              split="train", streaming=False)
            ds = ds.select(range(min(n_samples, len(ds))))
        elif dataset_name == "animalfarm":
            print("Loading Animal Farm (surface-depth mismatch test)...")
            af_path = Path(__file__).parent / "data" / "animal_farm.txt"
            text = af_path.read_text()
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
            ds = [{"text": p} for p in paragraphs]
            print(f"  {len(ds)} paragraphs")
        elif dataset_name == "chinese":
            print(f"Loading Chinese Wikipedia ({n_samples} samples)...")
            ds = load_dataset("wikimedia/wikipedia", "20231101.zh",
                              split="train", streaming=False)
            ds = ds.select(range(min(n_samples, len(ds))))
        elif dataset_name == "code":
            print(f"Loading Python code (The Stack, {n_samples} samples)...")
            ds = load_dataset("bigcode/the-stack-smol", "data/python",
                              split="train", streaming=False)
            ds = ds.select(range(min(n_samples, len(ds))))
        elif dataset_name == "korean":
            print(f"Loading Korean Wikipedia ({n_samples} samples)...")
            ds = load_dataset("wikimedia/wikipedia", "20231101.ko",
                              split="train", streaming=False)
            ds = ds.select(range(min(n_samples, len(ds))))
        else:
            print(f"Loading TinyStories ({n_samples} samples)...")
            ds = load_dataset("roneneldan/TinyStories", split="train",
                              streaming=False)
            ds = ds.select(range(min(n_samples, len(ds))))

        self.tokens = []
        for item in ds:
            text = item.get("text", "")
            if not text or len(text.strip()) < 10:
                continue
            toks = tokenizer.encode(text, add_special_tokens=True)
            if len(toks) >= max_length:
                for i in range(0, len(toks) - max_length, max_length):
                    self.tokens.append(toks[i:i + max_length])
            elif len(toks) > 32:
                toks = toks + [tokenizer.pad_token_id or 0] * (max_length - len(toks))
                self.tokens.append(toks[:max_length])

        print(f"  {len(self.tokens)} chunks of length {max_length}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        toks = torch.tensor(self.tokens[idx], dtype=torch.long)
        return toks[:-1], toks[1:]


def split_dataset(dataset, train_frac=0.9):
    n = len(dataset)
    n_train = int(n * train_frac)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    return Subset(dataset, indices[:n_train].tolist()), Subset(dataset, indices[n_train:].tolist())


# ── Measurements ──────────────────────────────────────────────────────────

def measure_perplexity(model, dataloader, device, max_batches=100):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    model.train()
    avg = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return {"loss": avg, "ppl": min(float(np.exp(avg)), 1e7)}


def measure_topology(model, sample_batch, device):
    model.eval()
    with torch.no_grad():
        x = sample_batch[0][:1].to(device)
        _, _, hs = model(x, return_hidden=True)
    results = {}
    for i, h in enumerate(hs):
        h0 = h[0]
        results[f"layer_{i}"] = {
            "eff_rank": effective_rank(h0),
            "spectral_gap": spectral_gap(h0),
            "gini_sv": gini_fast(torch.linalg.svdvals(h0.float().cpu()).numpy()),
        }
    for k in ["eff_rank", "spectral_gap", "gini_sv"]:
        vals = [results[f"layer_{i}"][k] for i in range(len(hs))]
        results[f"mean_{k}"] = float(np.mean(vals))
    model.train()
    return results


def weight_stats(model):
    total_zero = 0
    total_one = 0
    total_three = 0
    total = 0
    for _, m in model.named_modules():
        if isinstance(m, (TernaryLinear, BitFlipLinear)):
            wq = m.get_quantized_weight()
            total_zero += (wq == 0).sum().item()
            total_one += (wq == 1).sum().item()
            total_three += (wq == 3).sum().item()
            total += wq.numel()
    if total == 0:
        return {"zero": 0, "one": 0, "three": 0}
    return {
        "zero": total_zero / total,
        "one": total_one / total,
        "three": total_three / total,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--size", default="small",
                        choices=["small", "medium", "deep"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=200000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="tinystories",
                        choices=["tinystories", "wikitext", "kant", "sep", "korean", "animalfarm", "chinese", "code"],
                        help="Dataset: tinystories, wikitext, kant, sep, korean, animalfarm, chinese, code")
    parser.add_argument("--init_mode", type=str, default=None,
                        choices=["mixed", "void", "identity", "uniform"],
                        help="Ternary weight init. None=auto (deep→mixed). "
                             "'void'=all-zero, network must self-organize.")
    # Mutation engine — gradient-informed promotion/demotion of ternary weights
    parser.add_argument("--mutate", action="store_true",
                        help="Enable gradient-informed weight mutation. "
                             "Promotes high-gradient zeros to 3, demotes "
                             "low-gradient actives to 0.")
    parser.add_argument("--promote_frac", type=float, default=0.005,
                        help="Fraction of zeros to promote per cycle (default 0.5%%)")
    parser.add_argument("--demote_frac", type=float, default=0.002,
                        help="Fraction of actives to demote per cycle (default 0.2%%)")
    parser.add_argument("--mutate_cycle", type=int, default=500,
                        help="Steps between mutation cycles (default 500)")
    parser.add_argument("--mutate_warmup", type=int, default=2000,
                        help="No mutation before this step (default 2000)")
    parser.add_argument("--ternary_decay", type=float, default=0.0,
                        help="Weight decay on ternary params (default 0). "
                             "Non-zero = selection pressure that prunes weak "
                             "connections. Combine with --mutate for evolution.")
    # BitFlip: Quake III-style discrete training (replaces STE entirely)
    parser.add_argument("--bitflip", action="store_true",
                        help="Use BitFlip discrete training instead of STE. "
                             "No continuous latent weights. Gradient → bit flips.")
    parser.add_argument("--flip_pct", type=float, default=0.001,
                        help="The magic constant: fraction of weights to flip "
                             "per cycle (default 0.1%%)")
    parser.add_argument("--flip_cycle", type=int, default=100,
                        help="Steps between flip cycles (default 100)")
    parser.add_argument("--flip_warmup", type=int, default=500,
                        help="No flips before this many optim steps (default 500)")
    parser.add_argument("--flip_cooldown", type=int, default=0,
                        help="Stop flips this many steps before end (default 0). "
                             "e.g. 10000 = no flips in final 10K steps.")
    parser.add_argument("--flip_gravity", type=float, default=0.0,
                        help="Discrete L2: bias toward zero. Demotions boosted, "
                             "promotions dampened. 0=symmetric, 1=2x demotion bias.")
    args = parser.parse_args()

    # Distinct output dir per init_mode so runs don't overwrite each other
    init_suffix = f"_{args.init_mode}" if args.init_mode else ""
    mutate_suffix = "_mutate" if args.mutate else ""
    decay_suffix = f"_decay{args.ternary_decay}" if args.ternary_decay > 0 else ""
    bitflip_suffix = "_bitflip" if args.bitflip else ""
    OUTPUT_DIR = get_output_dir(f"{args.size}{init_suffix}{mutate_suffix}{decay_suffix}{bitflip_suffix}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset — bigger for longer training
    full_ds = TextChunked(tokenizer, max_length=args.max_length,
                          n_samples=args.n_samples,
                          dataset_name=args.dataset)
    train_ds, test_ds = split_dataset(full_ds)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=True)
    sample_batch = next(iter(train_loader))

    # Model
    ws = "bitflip" if args.bitflip else "013"
    model = build_model(size=args.size, weight_set=ws,
                        vocab_size=tokenizer.vocab_size,
                        init_mode=args.init_mode).to(device)
    params = model.count_params()
    print(f"\n{params['total']:,} params ({params['ternary_pct']:.1f}% ternary)")

    # Separate param groups based on layer type
    from ternary_linear import TernaryLinear, BitFlipLinear
    discrete_params = []  # TernaryLinear OR BitFlipLinear weights
    bitflip_weights = []  # BitFlipLinear weights specifically (excluded from optimizer)
    continuous_params = []
    for name, p in model.named_parameters():
        is_discrete = False
        is_bitflip_weight = False
        for mname, m in model.named_modules():
            if isinstance(m, (TernaryLinear, BitFlipLinear)) and name.startswith(mname):
                is_discrete = True
                # BitFlipLinear weights (not biases) are never optimized
                if isinstance(m, BitFlipLinear) and name == f"{mname}.weight":
                    is_bitflip_weight = True
                break
        if is_bitflip_weight:
            bitflip_weights.append(p)
            p.requires_grad_(True)  # still need grad for BitFlipEngine
        elif is_discrete:
            discrete_params.append(p)
        else:
            continuous_params.append(p)

    # Optimizer: bitflip weights excluded entirely — trained by BitFlipEngine
    optim_groups = []
    if discrete_params:
        td_label = "FREE" if args.ternary_decay == 0 else f"L2={args.ternary_decay}"
        optim_groups.append({"params": discrete_params, "weight_decay": args.ternary_decay})
        print(f"  Ternary params ({td_label}): {sum(p.numel() for p in discrete_params):,}")
    if bitflip_weights:
        print(f"  BitFlip params (engine-only): {sum(p.numel() for p in bitflip_weights):,}")
    optim_groups.append({"params": continuous_params, "weight_decay": 0.01})
    print(f"  Continuous params (L2=0.01):  {sum(p.numel() for p in continuous_params):,}")

    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)

    # QuakeFlip for STE mode — four-direction boundary crossing
    mutator = None
    if args.mutate and not args.bitflip:
        mutator = TernaryMutator(
            model,
            flip_pct=args.promote_frac,  # reuse promote_frac as flip_pct
            cycle_steps=args.mutate_cycle,
            warmup_steps=args.mutate_warmup,
        )
        print(f"  QuakeFlip (STE): flip_pct={args.promote_frac:.4f} "
              f"cycle={args.mutate_cycle} warmup={args.mutate_warmup}")

    # BitFlip engine (replaces STE entirely)
    flipper = None
    if args.bitflip:
        flipper = BitFlipEngine(
            model,
            flip_pct=args.flip_pct,
            cycle_steps=args.flip_cycle,
            warmup_steps=args.flip_warmup,
            gravity=args.flip_gravity,
        )
        print(f"  BitFlip: magic_constant={args.flip_pct:.4f} "
              f"cycle={args.flip_cycle} warmup={args.flip_warmup} "
              f"gravity={args.flip_gravity:.2f}")

    # LR warmup + cosine decay
    warmup_steps = min(2000, args.max_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # linear warmup
        # cosine decay after warmup
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return 0.05 + 0.95 * (1 + np.cos(np.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    accum = max(1, args.effective_batch // args.batch_size)

    # Annotate experiment for cross-run synthesis
    init_mode_actual = args.init_mode or ("mixed" if args.size == "deep" else "uniform")
    init_wdist = weight_stats(model)

    log = {
        "weight_set": "013",
        "config": vars(args),
        "params": params,
        "experiment": {
            "init_mode": init_mode_actual,
            "init_weight_dist": init_wdist,
            "weight_decay_ternary": args.ternary_decay,
            "weight_decay_continuous": 0.01,
            "grad_clip_ternary": False,
            "grad_clip_continuous": 1.0,
            "mutator_enabled": args.mutate,
            "mutator_config": {
                "promote_frac": args.promote_frac,
                "demote_frac": args.demote_frac,
                "cycle_steps": args.mutate_cycle,
                "warmup_steps": args.mutate_warmup,
            } if args.mutate else None,
            "hypothesis": (
                "Mutation engine: STE can't cross quantization boundaries. "
                "Gradient-informed promotion (0→3) and demotion (active→0) "
                "lets the network discover its own weight ratio. "
                "Compare to all prior runs where ratios froze at init."
                if args.mutate else
                "void init: model must self-organize from zero. "
                "Every connection earned by gradient pressure."
                if init_mode_actual == "void" else
                f"init_mode={init_mode_actual}: "
                "track weight ratio evolution and compression dynamics."
            ),
        },
        "checkpoints": [],
        "steps": [],
    }

    step = 0
    start = time.time()
    prev_eff_rank = None
    compression_stalled = 0

    print(f"\nTraining {args.max_steps} steps. Checkpoint every 1000.\n")
    model.train()
    optimizer.zero_grad()

    while step < args.max_steps:
        for bx, by in train_loader:
            if step >= args.max_steps:
                break

            bx, by = bx.to(device), by.to(device)
            _, loss = model(bx, targets=by)
            (loss / accum).backward()

            # Accumulate gradient magnitudes for mutation/bitflip engines
            if mutator is not None:
                mutator.accumulate()
            if flipper is not None:
                flipper.accumulate()

            if (step + 1) % accum == 0:
                # Only clip continuous params for stability
                torch.nn.utils.clip_grad_norm_(continuous_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # QuakeFlip for STE mode — with cooldown
                if mutator is not None and step < (args.max_steps - args.flip_cooldown):
                    mstats = mutator.maybe_mutate(step)
                    if mstats is not None:
                        print(f"    QFLIP {step}: 0→1={mstats['0→1']} 1→3={mstats['1→3']} "
                              f"3→1={mstats['3→1']} 1→0={mstats['1→0']} "
                              f"total={mstats['total']}", flush=True)

                # BitFlip engine (Quake mode) — with cooldown
                if flipper is not None and step < (args.max_steps - args.flip_cooldown):
                    fstats = flipper.maybe_flip(step)
                    if fstats is not None:
                        print(f"    FLIP {step}: 0→1={fstats['0→1']} 1→3={fstats['1→3']} "
                              f"3→1={fstats['3→1']} 1→0={fstats['1→0']} "
                              f"total={fstats['total']}", flush=True)
                    if step >= (args.max_steps - args.flip_cooldown - 200) and \
                       step < (args.max_steps - args.flip_cooldown):
                        if step % 1000 == 0:
                            print(f"    ** COOLDOWN starts at step "
                                  f"{args.max_steps - args.flip_cooldown} **", flush=True)

            # Quick log
            if step % 200 == 0:
                elapsed = time.time() - start
                ppl = torch.exp(loss).item()
                ppl_s = f"{ppl:.1f}" if ppl < 1e5 else f"{ppl:.0e}"
                log["steps"].append({"step": step, "loss": loss.item(),
                                     "ppl": min(ppl, 1e7), "elapsed": elapsed})
                print(f"  {step:6d} | loss {loss.item():.4f} | ppl {ppl_s:>10s} | "
                      f"{elapsed:.0f}s", flush=True)

            # Full checkpoint
            if step > 0 and step % 1000 == 0:
                train_ppl = measure_perplexity(model, train_loader, device)
                test_ppl = measure_perplexity(model, test_loader, device)
                topo = measure_topology(model, sample_batch, device)
                wstats = weight_stats(model)
                gen_gap = test_ppl["ppl"] / train_ppl["ppl"] if train_ppl["ppl"] > 0 else 999

                ckpt = {
                    "step": step,
                    "train_ppl": train_ppl,
                    "test_ppl": test_ppl,
                    "gen_gap": gen_gap,
                    "eff_rank": topo["mean_eff_rank"],
                    "spectral_gap": topo["mean_spectral_gap"],
                    "gini_sv": topo["mean_gini_sv"],
                    "weight_dist": wstats,
                    "elapsed": time.time() - start,
                }
                if mutator is not None:
                    recent = [m for m in mutator.history if m["step"] > step - 1000]
                    ckpt["mutations"] = {
                        "total_flips": sum(m.get("total", 0) for m in recent),
                        "cycles_this_ckpt": len(recent),
                    }
                log["checkpoints"].append(ckpt)

                # Compression tracking
                curr_rank = topo["mean_eff_rank"]
                rank_delta = (prev_eff_rank - curr_rank) if prev_eff_rank else 0
                if prev_eff_rank and abs(rank_delta) < 1.0:
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
                if compression_stalled >= 3:
                    print(f"    ** COMPRESSION SATURATED (stalled {compression_stalled} ckpts) **")
                print(flush=True)

                # Generate samples at key milestones
                if step % 5000 == 0:
                    model.eval()
                    for prompt in ["Once upon a time", "The little dog"]:
                        toks = tokenizer.encode(prompt, return_tensors="pt").to(device)
                        out = model.generate(toks, max_new=60, temperature=0.8)
                        text = tokenizer.decode(out[0].cpu(), skip_special_tokens=True)
                        print(f"    > {text[:120]}")
                    model.train()

                # Save intermediate
                with open(OUTPUT_DIR / "training_log.json", "w") as f:
                    json.dump(log, f, indent=2, default=str)

            step += 1

    # ── Final ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    train_ppl = measure_perplexity(model, train_loader, device)
    test_ppl = measure_perplexity(model, test_loader, device)
    topo = measure_topology(model, sample_batch, device)

    # Iterative inference
    toks = tokenizer.encode("Once upon a time", return_tensors="pt").to(device)
    iter_result = iterative_inference(model, toks, max_iter=20)

    # Zero topology
    zt = None
    for _, m in model.named_modules():
        if isinstance(m, (TernaryLinear, BitFlipLinear)):
            zt = zero_mask_topology(m.get_quantized_weight())
            break

    # Generations
    model.eval()
    gens = []
    for p in ["Once upon a time", "The dog ran to the", "She was happy because",
              "He looked at the big", "They went to the park"]:
        toks = tokenizer.encode(p, return_tensors="pt").to(device)
        out = model.generate(toks, max_new=80, temperature=0.8)
        text = tokenizer.decode(out[0].cpu(), skip_special_tokens=True)
        gens.append(text)
        print(f"  > {text[:120]}")

    log["final"] = {
        "elapsed": elapsed,
        "train_ppl": train_ppl,
        "test_ppl": test_ppl,
        "gen_gap": test_ppl["ppl"] / train_ppl["ppl"],
        "topology": {
            "eff_rank": topo["mean_eff_rank"],
            "spectral_gap": topo["mean_spectral_gap"],
            "gini_sv": topo["mean_gini_sv"],
        },
        "iterative": {
            "converged": iter_result["converged"],
            "n_iters": iter_result["n_iterations"],
        },
        "zero_topology": zt,
        "weight_dist": weight_stats(model),
        "generations": gens,
    }

    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")

    print(f"\n{'='*60}")
    print(f"  {0,1,3} LONG RUN — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  train_ppl={train_ppl['ppl']:.1f} | test_ppl={test_ppl['ppl']:.1f}")
    print(f"  gen_gap={test_ppl['ppl']/train_ppl['ppl']:.3f}")
    print(f"  eff_rank={topo['mean_eff_rank']:.1f} | spectral_gap={topo['mean_spectral_gap']:.1f}")
    print(f"  iter_converge={iter_result['converged']} ({iter_result['n_iterations']} iters)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
