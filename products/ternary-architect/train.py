#!/usr/bin/env python3
"""Training loop for {0,1,3} transformer experiments.

Trains on TinyStories with full topology measurement at checkpoints.
Tracks training-as-filtration: which weights commit early (structure)
vs oscillate (noise being filtered).

Usage:
    python train.py --weight_set 013 --size small --max_steps 5000
    python train.py --weight_set n101 --size small --max_steps 5000  # BitNet baseline
    python train.py --weight_set fp16 --size small --max_steps 5000  # fp16 baseline
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ternary_linear import TernaryLinear, quantize_to_set
from ternary_transformer import build_model, GPTConfig

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────

class TinyStoriesDataset(Dataset):
    """Streams TinyStories from HuggingFace, tokenizes, and chunks."""

    def __init__(self, tokenizer, max_length: int = 512, n_samples: int = 50000):
        from datasets import load_dataset
        print(f"Loading TinyStories ({n_samples} samples)...")
        ds = load_dataset("roneneldan/TinyStories", split="train",
                          streaming=False)
        # Take a subset for speed
        ds = ds.select(range(min(n_samples, len(ds))))

        self.tokens = []
        for item in ds:
            toks = tokenizer.encode(item["text"], add_special_tokens=True)
            if len(toks) >= max_length:
                # Chunk into max_length segments
                for i in range(0, len(toks) - max_length, max_length):
                    self.tokens.append(toks[i:i + max_length])
            elif len(toks) > 32:
                # Pad short sequences
                toks = toks + [tokenizer.pad_token_id or 0] * (max_length - len(toks))
                self.tokens.append(toks[:max_length])

        print(f"  {len(self.tokens)} training chunks of length {max_length}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        toks = torch.tensor(self.tokens[idx], dtype=torch.long)
        return toks[:-1], toks[1:]  # input, target


# ── Topology measurements ─────────────────────────────────────────────────

def measure_hidden_topology(model, sample_batch, device):
    """Run the full topology battery on hidden states."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "topological-router"))
    from topo_measures import effective_rank, spectral_gap, gini_fast, h0_gini

    model.eval()
    with torch.no_grad():
        x = sample_batch[0][:1].to(device)  # single example
        _, _, hidden_states = model(x, return_hidden=True)

    results = {}
    for i, hs in enumerate(hidden_states):
        h = hs[0]  # (T, d)
        results[f"layer_{i}"] = {
            "eff_rank": effective_rank(h),
            "spectral_gap": spectral_gap(h),
            "gini_sv": gini_fast(torch.linalg.svdvals(h.float().cpu()).numpy()),
            "h0_gini": h0_gini(h.cpu().numpy(), max_n=200),
        }

    # Average across layers
    keys = ["eff_rank", "spectral_gap", "gini_sv", "h0_gini"]
    for k in keys:
        vals = [results[f"layer_{i}"][k] for i in range(len(hidden_states))]
        results[f"mean_{k}"] = float(np.mean(vals))
        results[f"std_{k}"] = float(np.std(vals))

    model.train()
    return results


def measure_weight_topology(model):
    """Analyze the topology of the weight matrices themselves."""
    results = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            dist = module.weight_distribution()
            results[name] = {
                "distribution": dist,
            }

    # Global distribution across all ternary layers
    all_dists = {}
    count = 0
    for name, data in results.items():
        for k, v in data["distribution"].items():
            all_dists[k] = all_dists.get(k, 0.0) + v
        count += 1
    if count > 0:
        for k in all_dists:
            all_dists[k] /= count
    results["global_distribution"] = all_dists

    return results


def snapshot_weights(model) -> dict[str, torch.Tensor]:
    """Snapshot all quantized ternary weights for persistence tracking."""
    snap = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            snap[name] = module.get_quantized_weight().cpu().clone()
    return snap


def compute_weight_stability(snapshots: list[dict[str, torch.Tensor]]) -> dict:
    """Analyze how many weights have settled to their final value.

    For each weight position, track when it last changed value.
    Stable = same value for last N snapshots.
    """
    if len(snapshots) < 2:
        return {"stability": 1.0, "n_snapshots": len(snapshots)}

    latest = snapshots[-1]
    results = {}
    total_stable = 0
    total_weights = 0

    for name in latest:
        # Count how many checkpoints back this weight has been the same
        current = latest[name]
        stable_count = 0
        for i in range(len(snapshots) - 2, -1, -1):
            if name in snapshots[i] and torch.equal(snapshots[i][name], current):
                stable_count += 1
            else:
                break

        n = current.numel()
        # Per-element stability: fraction unchanged from previous
        if len(snapshots) >= 2 and name in snapshots[-2]:
            unchanged = (current == snapshots[-2][name]).float().mean().item()
        else:
            unchanged = 1.0

        results[name] = {
            "consecutive_stable": stable_count,
            "pct_unchanged": unchanged,
        }
        total_stable += unchanged * n
        total_weights += n

    results["global_stability"] = total_stable / total_weights if total_weights > 0 else 1.0
    return results


# ── Training ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    dataset = TinyStoriesDataset(tokenizer, max_length=args.max_length,
                                 n_samples=args.n_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    # Model
    model = build_model(size=args.size, weight_set=args.weight_set,
                        vocab_size=tokenizer.vocab_size)
    model = model.to(device)

    params = model.count_params()
    print(f"\nModel: {args.weight_set} / {args.size}")
    print(f"  Total params:   {params['total']:,}")
    print(f"  Ternary params: {params['ternary']:,} ({params['ternary_pct']:.1f}%)")
    print(f"  Continuous:     {params['continuous']:,}")

    # Optimizer — acts on latent weights (fp32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    # Gradient accumulation
    accum_steps = max(1, args.effective_batch // args.batch_size)

    # ── Training state ────────────────────────────────────────────────────
    log = {
        "config": {
            "weight_set": args.weight_set,
            "size": args.size,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "effective_batch": args.effective_batch,
            "max_length": args.max_length,
            "n_samples": args.n_samples,
            "params": params,
        },
        "steps": [],
        "checkpoints": [],
        "weight_snapshots_meta": [],
    }

    weight_snapshots = []
    sample_batch = next(iter(loader))  # keep one batch for measurement
    step = 0
    start_time = time.time()

    print(f"\nTraining for {args.max_steps} steps (accum={accum_steps})...\n")

    model.train()
    optimizer.zero_grad()

    while step < args.max_steps:
        for batch_x, batch_y in loader:
            if step >= args.max_steps:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            _, loss = model(batch_x, targets=batch_y)
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ── Logging ───────────────────────────────────────────────
            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                lr_now = scheduler.get_last_lr()[0]
                ppl = torch.exp(loss * accum_steps).item()

                step_data = {
                    "step": step,
                    "loss": (loss * accum_steps).item(),
                    "ppl": min(ppl, 1e6),
                    "lr": lr_now,
                    "elapsed": elapsed,
                }

                # Weight distribution (cheap)
                if args.weight_set != "fp16":
                    w_topo = measure_weight_topology(model)
                    step_data["weight_dist"] = w_topo.get("global_distribution", {})

                log["steps"].append(step_data)
                ppl_str = f"{ppl:.1f}" if ppl < 1e5 else "inf"
                print(f"  step {step:5d} | loss {(loss*accum_steps).item():.4f} | "
                      f"ppl {ppl_str:>8s} | lr {lr_now:.2e} | {elapsed:.0f}s")

            # ── Checkpoint with measurements ──────────────────────────
            if step > 0 and step % args.ckpt_every == 0:
                print(f"\n  [Checkpoint @ step {step}]")

                # Hidden state topology
                hs_topo = measure_hidden_topology(model, sample_batch, device)
                print(f"    eff_rank: {hs_topo['mean_eff_rank']:.1f} | "
                      f"spectral_gap: {hs_topo['mean_spectral_gap']:.2f} | "
                      f"gini_sv: {hs_topo['mean_gini_sv']:.3f}")

                # Weight snapshot for persistence tracking
                snap = snapshot_weights(model)
                weight_snapshots.append(snap)
                stability = compute_weight_stability(weight_snapshots)

                ckpt_data = {
                    "step": step,
                    "hidden_topology": hs_topo,
                    "weight_stability": {
                        "global": stability.get("global_stability", 1.0),
                    },
                }
                log["checkpoints"].append(ckpt_data)
                log["weight_snapshots_meta"].append({
                    "step": step,
                    "global_stability": stability.get("global_stability", 1.0),
                })

                print(f"    weight stability: {stability.get('global_stability', 1.0):.3f}")
                print()

                model.train()

            step += 1

    # ── Final measurements ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.0f}s")

    print("\nFinal measurements...")
    final_topo = measure_hidden_topology(model, sample_batch, device)
    final_weights = measure_weight_topology(model)

    # Final weight snapshot
    final_snap = snapshot_weights(model)
    weight_snapshots.append(final_snap)
    final_stability = compute_weight_stability(weight_snapshots)

    log["final"] = {
        "elapsed": elapsed,
        "hidden_topology": final_topo,
        "weight_topology": final_weights,
        "weight_stability": final_stability.get("global_stability", 1.0),
    }

    # ── Generation samples ────────────────────────────────────────────────
    print("\nSample generations:")
    prompts = ["Once upon a time", "The little dog", "She looked at the"]
    model.eval()
    generations = []
    for p in prompts:
        toks = tokenizer.encode(p, return_tensors="pt").to(device)
        out = model.generate(toks, max_new=60, temperature=0.8)
        text = tokenizer.decode(out[0].cpu(), skip_special_tokens=True)
        print(f"  > {text[:120]}")
        generations.append(text)
    log["generations"] = generations

    # ── Save ──────────────────────────────────────────────────────────────
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{args.weight_set}_{args.size}_{ts}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save log
    with open(run_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    # Save model weights
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Save final weight snapshots for zero-pattern analysis
    if weight_snapshots:
        torch.save(weight_snapshots, run_dir / "weight_snapshots.pt")

    print(f"\nResults saved to {run_dir}")
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.weight_set} / {args.size}")
    print(f"  Final loss:      {log['steps'][-1]['loss']:.4f}")
    print(f"  Final ppl:       {log['steps'][-1]['ppl']:.1f}")
    print(f"  Eff rank (mean): {final_topo['mean_eff_rank']:.1f}")
    print(f"  Spectral gap:    {final_topo['mean_spectral_gap']:.2f}")
    print(f"  Gini SV:         {final_topo['mean_gini_sv']:.3f}")
    print(f"  Weight stability:{final_stability.get('global_stability', 1.0):.3f}")
    if args.weight_set != "fp16":
        print(f"  Weight dist:     {final_weights.get('global_distribution', {})}")
    print(f"{'='*60}")

    return log


def main():
    parser = argparse.ArgumentParser(description="Train {0,1,3} transformer")
    parser.add_argument("--weight_set", type=str, default="013",
                        choices=["013", "n101", "012", "015", "017", "0123", "fp16"])
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium"])
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=500)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
