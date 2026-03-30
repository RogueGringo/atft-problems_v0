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

from ternary_linear import TernaryLinear
from ternary_transformer import build_model
from topo_measures import effective_rank, spectral_gap, gini_fast
from measure import zero_mask_topology, iterative_inference


# ── Dataset ───────────────────────────────────────────────────────────────

class TinyStoriesChunked(Dataset):
    def __init__(self, tokenizer, max_length=256, n_samples=200000):
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
        if isinstance(m, TernaryLinear):
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
    args = parser.parse_args()

    OUTPUT_DIR = get_output_dir(args.size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset — bigger for longer training
    full_ds = TinyStoriesChunked(tokenizer, max_length=args.max_length,
                                  n_samples=args.n_samples)
    train_ds, test_ds = split_dataset(full_ds)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=True)
    sample_batch = next(iter(train_loader))

    # Model
    model = build_model(size=args.size, weight_set="013",
                        vocab_size=tokenizer.vocab_size).to(device)
    params = model.count_params()
    print(f"\n{params['total']:,} params ({params['ternary_pct']:.1f}% ternary)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # LR warmup + cosine decay — critical for deep {0,1,3} networks.
    # The STE needs microscopic steps while the identity highways form.
    warmup_steps = min(2000, args.max_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # linear warmup
        # cosine decay after warmup
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return 0.05 + 0.95 * (1 + np.cos(np.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    accum = max(1, args.effective_batch // args.batch_size)

    log = {
        "weight_set": "013",
        "config": vars(args),
        "params": params,
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

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

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
    for _, m in model.named_modules():
        if isinstance(m, TernaryLinear):
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
