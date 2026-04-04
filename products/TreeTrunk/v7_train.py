#!/usr/bin/env python3
"""v7 Cross-Domain Training — Force the manifold to discover universal truth.

Stratified NQ + GSM8K training through the locked TopologicalTrainer.
The sheaf gradient is equally informed by factual retrieval AND
mathematical deduction. The only way to minimize loss across both
domains is to learn the invariant geometric structure of logic itself.

Usage:
    python3 v7_train.py --steps 1000 --batch 8 --lr 5e-4
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from v6_topological_trainer import TextFeatureMap, TopologicalTrainer
from v7_universal_sampler import NQSampler, GSM8KSampler, UniversalTripletSampler

CHECKPOINT_DIR = Path(__file__).parent / "results" / "v7_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train(
    n_steps: int = 1000,
    batch_size: int = 8,
    lr: float = 5e-4,
    d_emb: int = 128,
    stalk_dim: int = 8,
    k: int = 4,
    margin: float = 0.5,
    log_every: int = 10,
    save_every: int = 200,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\nv7 Cross-Domain Training — {n_steps} steps, batch={batch_size}, lr={lr}")
    print(f"  Device: {device}")

    # ── Data: stratified NQ + GSM8K ───────────────────────────────────
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    nq_sampler = NQSampler(encoder, max_cloud=12)
    gsm8k_sampler = GSM8KSampler(encoder, max_cloud=12)
    sampler = UniversalTripletSampler([nq_sampler, gsm8k_sampler])

    # ── Model: fresh init (no v6 checkpoint — clean cross-domain) ─────
    feat_map = TextFeatureMap(
        out_dim=d_emb, vocab_size=1, backbone_dim=384
    ).to(device)

    trainer = TopologicalTrainer(
        feature_map=feat_map,
        d_emb=d_emb,
        stalk_dim=stalk_dim,
        k=k,
        margin=margin,
        lr=lr,
    ).to(device)

    total_params = sum(p.numel() for p in trainer.parameters())
    print(f"  Parameters: {total_params:,} (fresh init)")

    # ── Training ──────────────────────────────────────────────────────
    header = (f"{'Step':>6s}  {'Loss':>8s}  {'GapPos':>8s}  {'GapNeg':>8s}  "
              f"{'Δ(N-P)':>7s}  {'α':>6s}  {'β':>6s}  "
              f"{'NQ':>4s} {'GSM':>4s}  {'t/s':>5s}")
    print(f"\n{header}")
    print("─" * 80)

    history = []
    t0 = time.time()

    for step in range(n_steps):
        step_t0 = time.time()

        trainer.optimizer.zero_grad()
        batch_loss = 0.0
        batch_gap_pos = 0.0
        batch_gap_neg = 0.0
        batch_alpha = 0.0
        batch_beta = 0.0
        domain_counts = {"nq": 0, "gsm8k": 0}

        # Stratified batch
        batch = sampler.sample_batch(batch_size)

        for q_np, pos_np, neg_np, meta in batch:
            domain_counts[meta["domain"]] = domain_counts.get(meta["domain"], 0) + 1

            q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
            pos_t = torch.tensor(pos_np, dtype=torch.float32, device=device)
            neg_t = torch.tensor(neg_np, dtype=torch.float32, device=device)

            q_emb = trainer.feature_map(q_t)
            pos_emb = trainer.feature_map(pos_t)
            neg_emb = trainer.feature_map(neg_t)

            sheaf_loss, info = trainer.sheaf_loss_fn(q_emb, pos_emb, neg_emb)

            alpha, beta = trainer.gauge_router(q_emb)
            router_reg = 0.01 * ((alpha - 0.7) ** 2 + (beta - 0.3) ** 2)
            loss = sheaf_loss + router_reg

            (loss / batch_size).backward()

            batch_loss += loss.item() / batch_size
            batch_gap_pos += info["gap_pos"] / batch_size
            batch_gap_neg += info["gap_neg"] / batch_size
            batch_alpha += alpha.item() / batch_size
            batch_beta += beta.item() / batch_size

        trainer.optimizer.step()
        step_time = time.time() - step_t0
        gap_delta = batch_gap_neg - batch_gap_pos

        record = {
            "step": step,
            "loss": batch_loss,
            "gap_pos": batch_gap_pos,
            "gap_neg": batch_gap_neg,
            "gap_delta": gap_delta,
            "alpha": batch_alpha,
            "beta": batch_beta,
            "nq_count": domain_counts.get("nq", 0),
            "gsm8k_count": domain_counts.get("gsm8k", 0),
            "step_time": step_time,
        }
        history.append(record)

        if step % log_every == 0 or step == n_steps - 1:
            print(f"{step:>6d}  {batch_loss:>8.4f}  {batch_gap_pos:>8.4f}  "
                  f"{batch_gap_neg:>8.4f}  {gap_delta:>+7.3f}  "
                  f"{batch_alpha:>6.3f}  {batch_beta:>6.3f}  "
                  f"{domain_counts.get('nq',0):>4d} {domain_counts.get('gsm8k',0):>4d}  "
                  f"{step_time:>4.1f}s")

        if save_every > 0 and (step + 1) % save_every == 0:
            ckpt = CHECKPOINT_DIR / f"v7_step{step+1}.pt"
            torch.save(trainer.state_dict(), ckpt)
            print(f"  → checkpoint: {ckpt.name}")

    elapsed = time.time() - t0

    # ── Per-domain analysis ───────────────────────────────────────────
    # Compute rolling averages for last 50 steps
    last_50 = history[-50:]
    avg_loss = np.mean([r["loss"] for r in last_50])
    avg_gap_pos = np.mean([r["gap_pos"] for r in last_50])
    avg_gap_neg = np.mean([r["gap_neg"] for r in last_50])
    avg_delta = np.mean([r["gap_delta"] for r in last_50])

    first = history[0]
    last = history[-1]

    print(f"\n{'='*80}")
    print(f"  v7 CROSS-DOMAIN TRAINING COMPLETE — {n_steps} steps in {elapsed:.0f}s")
    print(f"{'='*80}")
    print(f"  Loss:     {first['loss']:.4f} → {last['loss']:.4f}  "
          f"(last-50 avg: {avg_loss:.4f})")
    print(f"  GapPos:   {first['gap_pos']:.4f} → {last['gap_pos']:.4f}  "
          f"(last-50 avg: {avg_gap_pos:.4f})")
    print(f"  GapNeg:   {first['gap_neg']:.4f} → {last['gap_neg']:.4f}  "
          f"(last-50 avg: {avg_gap_neg:.4f})")
    print(f"  Δ(N-P):   {first['gap_delta']:+.3f} → {last['gap_delta']:+.3f}  "
          f"(last-50 avg: {avg_delta:+.4f})")
    print(f"  α:        {first['alpha']:.4f} → {last['alpha']:.4f}")
    print(f"  β:        {first['beta']:.4f} → {last['beta']:.4f}")
    print(f"{'='*80}")

    # Save
    with open(CHECKPOINT_DIR / "v7_training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    final = CHECKPOINT_DIR / "v7_trainer_final.pt"
    torch.save(trainer.state_dict(), final)
    print(f"  History: {CHECKPOINT_DIR / 'v7_training_history.json'}")
    print(f"  Model:   {final}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(n_steps=args.steps, batch_size=args.batch, lr=args.lr,
          device_str=args.device)
