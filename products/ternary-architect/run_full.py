#!/usr/bin/env python3
"""Full experimental campaign — the real work.

1. Train 5 weight sets for 5000 steps each (long enough to see generalization)
2. Track training speed per step (does ternary get faster as weights crystallize?)
3. Generalization test: train/test split, measure overfitting gap
4. Full measurement battery on each model

Weight sets:
  013  — {0, 1, 3} (the thesis: void + unit + prime)
  012  — {0, 1, 2} (fast learner: void + unit + bisection)
  0123 — {0, 1, 2, 3} (full basis: void + unit + bisection + triangulation)
  n101 — {-1, 0, 1} (BitNet: sign + void)
  fp16 — continuous baseline

Usage:
    python run_full.py              # full campaign
    python run_full.py --steps 2000 # shorter
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
from torch.utils.data import Dataset, DataLoader, Subset

OUTPUT_DIR = Path(__file__).parent / "results" / "full_campaign"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "topological-router"))

from ternary_linear import TernaryLinear, quantize_to_set
from ternary_transformer import build_model
from topo_measures import effective_rank, spectral_gap, gini_fast, h0_gini
from measure import zero_mask_topology, zero_mask_ablation, iterative_inference


# ── Dataset with train/test split ─────────────────────────────────────────

class TinyStoriesChunked(Dataset):
    """TinyStories chunked into fixed-length training sequences."""

    def __init__(self, tokenizer, max_length: int = 256, n_samples: int = 50000):
        from datasets import load_dataset
        print(f"Loading TinyStories ({n_samples} samples)...")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
        ds = ds.select(range(min(n_samples, len(ds))))

        self.tokens = []
        for item in ds:
            toks = tokenizer.encode(item["text"], add_special_tokens=True)
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
    """Split into train/test for generalization measurement."""
    n = len(dataset)
    n_train = int(n * train_frac)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx = indices[:n_train].tolist()
    test_idx = indices[n_train:].tolist()
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


# ── Measurement functions ─────────────────────────────────────────────────

def measure_perplexity(model, dataloader, device, max_batches=50):
    """Compute perplexity on a dataset."""
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
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return {"loss": avg_loss, "ppl": min(float(np.exp(avg_loss)), 1e7)}


def measure_topology(model, sample_batch, device):
    """Quick topology measurement on hidden states."""
    model.eval()
    with torch.no_grad():
        x = sample_batch[0][:1].to(device)
        _, _, hidden_states = model(x, return_hidden=True)

    results = {}
    for i, hs in enumerate(hidden_states):
        h = hs[0]
        results[f"layer_{i}"] = {
            "eff_rank": effective_rank(h),
            "spectral_gap": spectral_gap(h),
            "gini_sv": gini_fast(torch.linalg.svdvals(h.float().cpu()).numpy()),
        }

    keys = ["eff_rank", "spectral_gap", "gini_sv"]
    for k in keys:
        vals = [results[f"layer_{i}"][k] for i in range(len(hidden_states))]
        results[f"mean_{k}"] = float(np.mean(vals))

    model.train()
    return results


def measure_weight_stats(model):
    """Weight distribution and sparsity stats."""
    stats = {}
    total_zero = 0
    total_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            dist = module.weight_distribution()
            stats[name] = dist
            w_q = module.get_quantized_weight()
            total_zero += (w_q == 0).sum().item()
            total_weights += w_q.numel()

    global_dist = {}
    count = 0
    for name, dist in stats.items():
        for k, v in dist.items():
            global_dist[k] = global_dist.get(k, 0.0) + v
        count += 1
    if count > 0:
        for k in global_dist:
            global_dist[k] /= count

    return {
        "per_layer": stats,
        "global": global_dist,
        "sparsity": total_zero / total_weights if total_weights > 0 else 0,
    }


def snapshot_quantized(model) -> dict[str, torch.Tensor]:
    """Snapshot quantized weights."""
    snap = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            snap[name] = module.get_quantized_weight().cpu().clone()
    return snap


# ── Training one model ────────────────────────────────────────────────────

def train_one(weight_set: str, train_loader, test_loader, sample_batch,
              device, args) -> dict:
    """Train a single model and collect all measurements."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(size=args.size, weight_set=weight_set,
                        vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    params = model.count_params()

    print(f"\n{'='*60}")
    print(f"  {weight_set.upper()} | {params['total']:,} params | "
          f"{params['ternary_pct']:.1f}% ternary")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)
    accum_steps = max(1, args.effective_batch // args.batch_size)

    # ── Logging structures ────────────────────────────────────────────
    log = {
        "weight_set": weight_set,
        "params": params,
        "steps": [],
        "checkpoints": [],
        "step_times": [],  # per-step timing for speed analysis
    }

    weight_snapshots = []
    step = 0
    start = time.time()
    step_start = time.time()

    model.train()
    optimizer.zero_grad()

    while step < args.max_steps:
        for batch_x, batch_y in train_loader:
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

            # Per-step timing
            step_end = time.time()
            step_time = step_end - step_start
            step_start = step_end

            if step % 10 == 0:
                log["step_times"].append({
                    "step": step,
                    "ms": step_time * 1000,
                })

            # Logging
            if step % args.log_every == 0:
                elapsed = time.time() - start
                ppl = torch.exp(loss * accum_steps).item()
                step_data = {
                    "step": step,
                    "loss": (loss * accum_steps).item(),
                    "ppl": min(ppl, 1e7),
                    "elapsed": elapsed,
                }
                log["steps"].append(step_data)

                ppl_s = f"{ppl:.1f}" if ppl < 1e5 else f"{ppl:.0e}"
                print(f"  {step:5d} | loss {(loss*accum_steps).item():.4f} | "
                      f"ppl {ppl_s:>10s} | {elapsed:.0f}s")

            # Checkpoint
            if step > 0 and step % args.ckpt_every == 0:
                # Train + test perplexity
                train_ppl = measure_perplexity(model, train_loader, device)
                test_ppl = measure_perplexity(model, test_loader, device)

                # Topology
                topo = measure_topology(model, sample_batch, device)

                # Weight stats
                w_stats = measure_weight_stats(model) if weight_set != "fp16" else {}

                # Snapshot for persistence
                if weight_set != "fp16":
                    snap = snapshot_quantized(model)
                    weight_snapshots.append(snap)

                # Weight stability
                stability = 1.0
                if len(weight_snapshots) >= 2:
                    curr = weight_snapshots[-1]
                    prev = weight_snapshots[-2]
                    total_same = 0
                    total_n = 0
                    for name in curr:
                        if name in prev:
                            total_same += (curr[name] == prev[name]).float().sum().item()
                            total_n += curr[name].numel()
                    stability = total_same / total_n if total_n > 0 else 1.0

                ckpt = {
                    "step": step,
                    "train_ppl": train_ppl,
                    "test_ppl": test_ppl,
                    "gen_gap": test_ppl["ppl"] / train_ppl["ppl"] if train_ppl["ppl"] > 0 else float("inf"),
                    "topology": {
                        "eff_rank": topo["mean_eff_rank"],
                        "spectral_gap": topo["mean_spectral_gap"],
                        "gini_sv": topo["mean_gini_sv"],
                    },
                    "weight_stability": stability,
                    "sparsity": w_stats.get("sparsity", None),
                    "weight_dist": w_stats.get("global", None),
                }
                log["checkpoints"].append(ckpt)

                gen_gap_s = f"{ckpt['gen_gap']:.2f}x"
                print(f"    CKPT | train_ppl {train_ppl['ppl']:.1f} | "
                      f"test_ppl {test_ppl['ppl']:.1f} | "
                      f"gen_gap {gen_gap_s} | "
                      f"eff_rank {topo['mean_eff_rank']:.1f} | "
                      f"stability {stability:.4f}")

                model.train()

            step += 1

    # ── Final measurements ────────────────────────────────────────────
    elapsed = time.time() - start

    train_ppl = measure_perplexity(model, train_loader, device)
    test_ppl = measure_perplexity(model, test_loader, device)
    topo = measure_topology(model, sample_batch, device)

    # Iterative inference
    test_text = "Once upon a time"
    toks = tokenizer.encode(test_text, return_tensors="pt").to(device)
    iter_result = iterative_inference(model, toks, max_iter=20)

    # Zero topology + ablation
    zero_topo = None
    zero_abl = None
    if weight_set != "fp16":
        w_stats = measure_weight_stats(model)
        # Zero topology on largest layer
        for name, module in model.named_modules():
            if isinstance(module, TernaryLinear):
                w_q = module.get_quantized_weight()
                zero_topo = zero_mask_topology(w_q)
                break
        zero_abl = zero_mask_ablation(model, tokenizer,
            "Once upon a time there was a little girl who loved to play", device)

    # Generation samples
    model.eval()
    generations = []
    for prompt in ["Once upon a time", "The dog ran to", "She was happy because"]:
        toks = tokenizer.encode(prompt, return_tensors="pt").to(device)
        out = model.generate(toks, max_new=80, temperature=0.8)
        text = tokenizer.decode(out[0].cpu(), skip_special_tokens=True)
        generations.append(text)

    # Step time analysis
    step_times = [s["ms"] for s in log["step_times"]]
    early_times = step_times[:len(step_times)//4] if step_times else [0]
    late_times = step_times[-len(step_times)//4:] if step_times else [0]

    log["final"] = {
        "elapsed": elapsed,
        "train_ppl": train_ppl,
        "test_ppl": test_ppl,
        "gen_gap": test_ppl["ppl"] / train_ppl["ppl"] if train_ppl["ppl"] > 0 else float("inf"),
        "topology": {
            "eff_rank": topo["mean_eff_rank"],
            "spectral_gap": topo["mean_spectral_gap"],
            "gini_sv": topo["mean_gini_sv"],
        },
        "iterative_inference": {
            "converged": iter_result["converged"],
            "n_iterations": iter_result["n_iterations"],
            "final_delta": iter_result.get("final_delta"),
        },
        "zero_topology": zero_topo,
        "zero_ablation": zero_abl,
        "generations": generations,
        "step_timing": {
            "early_avg_ms": float(np.mean(early_times)),
            "late_avg_ms": float(np.mean(late_times)),
            "speedup_ratio": float(np.mean(early_times) / np.mean(late_times))
                if np.mean(late_times) > 0 else 1.0,
        },
    }

    # Save
    run_dir = OUTPUT_DIR / weight_set
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    torch.save(model.state_dict(), run_dir / "model.pt")
    if weight_snapshots:
        torch.save(weight_snapshots, run_dir / "weight_snapshots.pt")

    print(f"\n  {weight_set} DONE in {elapsed:.0f}s")
    print(f"  train_ppl={train_ppl['ppl']:.1f} | test_ppl={test_ppl['ppl']:.1f} | "
          f"gen_gap={log['final']['gen_gap']:.2f}x")
    print(f"  eff_rank={topo['mean_eff_rank']:.1f} | "
          f"spectral_gap={topo['mean_spectral_gap']:.2f}")
    print(f"  iter_converge={iter_result['converged']} ({iter_result['n_iterations']} iters)")
    print(f"  step_time: early={np.mean(early_times):.1f}ms late={np.mean(late_times):.1f}ms "
          f"ratio={log['final']['step_timing']['speedup_ratio']:.2f}")
    for g in generations[:2]:
        print(f"  > {g[:100]}")

    # Cleanup GPU
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return log


# ── Main campaign ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_sets", nargs="+",
                        default=["013", "012", "0123", "n101", "fp16"])
    parser.add_argument("--size", default="small")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=80000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=500)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset with train/test split
    full_dataset = TinyStoriesChunked(tokenizer, max_length=args.max_length,
                                      n_samples=args.n_samples)
    train_ds, test_ds = split_dataset(full_dataset, train_frac=0.9)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=True)

    # Keep a fixed sample batch for topology measurement
    sample_batch = next(iter(train_loader))

    # ── Run all experiments ───────────────────────────────────────────
    campaign_start = time.time()
    all_logs = {}

    for ws in args.weight_sets:
        log = train_one(ws, train_loader, test_loader, sample_batch, device, args)
        all_logs[ws] = log

    # ── Comparison summary ────────────────────────────────────────────
    campaign_elapsed = time.time() - campaign_start

    print(f"\n\n{'='*80}")
    print(f"  FULL CAMPAIGN RESULTS — {campaign_elapsed:.0f}s ({campaign_elapsed/60:.1f} min)")
    print(f"{'='*80}\n")

    # Build comparison table
    headers = args.weight_sets
    print(f"{'Metric':<28s}", end="")
    for ws in headers:
        print(f" {ws:>10s}", end="")
    print()
    print("-" * (28 + 11 * len(headers)))

    rows = [
        ("Train PPL", lambda l: l["final"]["train_ppl"]["ppl"]),
        ("Test PPL", lambda l: l["final"]["test_ppl"]["ppl"]),
        ("Gen Gap (test/train)", lambda l: l["final"]["gen_gap"]),
        ("Eff Rank", lambda l: l["final"]["topology"]["eff_rank"]),
        ("Spectral Gap", lambda l: l["final"]["topology"]["spectral_gap"]),
        ("Gini SV", lambda l: l["final"]["topology"]["gini_sv"]),
        ("Iter Converged", lambda l: l["final"]["iterative_inference"]["converged"]),
        ("Iter Count", lambda l: l["final"]["iterative_inference"]["n_iterations"]),
        ("Early step (ms)", lambda l: l["final"]["step_timing"]["early_avg_ms"]),
        ("Late step (ms)", lambda l: l["final"]["step_timing"]["late_avg_ms"]),
        ("Speed ratio (early/late)", lambda l: l["final"]["step_timing"]["speedup_ratio"]),
    ]

    # Add zero-topology rows only for ternary
    rows.extend([
        ("Zero abl deg%", lambda l: (l["final"].get("zero_ablation") or {}).get("degradation_pct", "N/A")),
    ])

    for label, fn in rows:
        print(f"{label:<28s}", end="")
        for ws in headers:
            if ws in all_logs:
                try:
                    val = fn(all_logs[ws])
                    if isinstance(val, bool):
                        print(f" {'YES':>10s}" if val else f" {'NO':>10s}", end="")
                    elif isinstance(val, float):
                        if abs(val) > 1000:
                            print(f" {val:>10.0f}", end="")
                        else:
                            print(f" {val:>10.3f}", end="")
                    else:
                        print(f" {str(val):>10s}", end="")
                except Exception:
                    print(f" {'ERR':>10s}", end="")
            else:
                print(f" {'---':>10s}", end="")
        print()

    # Save full comparison
    summary = {
        "config": {
            "weight_sets": args.weight_sets,
            "size": args.size,
            "max_steps": args.max_steps,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
        },
        "elapsed": campaign_elapsed,
        "results": {},
    }
    for ws in args.weight_sets:
        if ws in all_logs:
            summary["results"][ws] = all_logs[ws]["final"]

    with open(OUTPUT_DIR / "campaign_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"\nKey questions answered:")
    print(f"  1. Does {'{0,1,3}'} catch up to fp16 with more training?")
    print(f"  2. Which weight set generalizes best (lowest gen gap)?")
    print(f"  3. Does training get faster for ternary (speed ratio > 1)?")
    print(f"  4. Does {'{0,1,2,3}'} get the best of both worlds?")


if __name__ == "__main__":
    main()
