#!/usr/bin/env python3
"""v6 Training Loop — Warp the manifold with real data.

Feeds NQ (Context, Query, Target) triplets through the TopologicalTrainer.
The SheafTopologyLoss drives:
  gap_pos → 0  (correct answers become topologically coherent)
  gap_neg → ∞  (wrong answers shatter the manifold)

The DynamicGaugeRouter learns per-query α/β ratios, starting from
the v5 empirical prior (0.70, 0.30).

Data pipeline:
  1. Load pre-computed NQ chunk embeddings (384-dim, memory-mapped)
  2. Load sentence-transformer for on-the-fly question encoding
  3. For each triplet:
     - q_cloud:   question → sliding-window embedding → (n_q, 384)
     - pos_cloud:  sample chunks from correct article → (n_pos, 384)
     - neg_cloud:  sample chunks from wrong article → (n_neg, 384)
  4. TextFeatureMap projects 384 → 128
  5. SheafTopologyLoss computes contrastive spectral gap
  6. Backward + optimizer step

Usage:
    python3 v6_train.py --steps 500 --batch 4 --lr 1e-3

Requires: results/v5_nq_base_map/ from v5_nq_ingest.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from v6_topological_trainer import (
    TextFeatureMap,
    TopologicalTrainer,
)

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
CHECKPOINT_DIR = Path(__file__).parent / "results" / "v6_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class NQTripletSampler:
    """Samples (query, positive, hard_negative) triplets from pre-mined cache.

    Hard negatives are chunks from WRONG articles that have HIGH cosine
    similarity to the query — the exact semantic paradoxes that force the
    manifold to warp factual truth into topological coherence.

    Query: question text encoded on-the-fly with sliding windows.
    Positive: pre-computed chunk embeddings from correct article.
    Negative: pre-mined FAISS hard negatives (high cosine, wrong article).
    """

    def __init__(self, max_cloud_size: int = 12):
        from sentence_transformers import SentenceTransformer

        print("Loading NQ triplet sampler (hard negative mode)...")

        # Memory-map embeddings (3.59GB — stays on disk, loads pages on access)
        self.embeddings = np.load(
            BASE_MAP_DIR / "embeddings.npy", mmap_mode="r"
        )

        # Load pre-mined hard negatives cache
        cache_path = CHECKPOINT_DIR / "hard_negatives.json"
        with open(cache_path) as f:
            self.cache = json.load(f)

        # Filter to entries with both positives and hard negatives
        self.cache = [c for c in self.cache
                      if c["pos_chunk_ids"] and c["hard_neg_chunks"]]

        self.max_cloud = max_cloud_size
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.rng = np.random.default_rng(42)

        avg_negs = np.mean([len(c["hard_neg_chunks"]) for c in self.cache])
        print(f"  {len(self.cache):,} cached triplets with hard negatives")
        print(f"  Avg hard negatives per query: {avg_negs:.1f}")
        print(f"  Embeddings: {self.embeddings.shape} (memory-mapped)")

    def _embed_question(self, text: str) -> np.ndarray:
        """Sliding-window embedding of question text."""
        words = text.split()
        window, stride = 5, 3
        if len(words) <= window:
            chunks = [text]
        else:
            chunks = [" ".join(words[i:i + window])
                      for i in range(0, len(words) - window + 1, stride)]
        return self.encoder.encode(chunks, convert_to_numpy=True,
                                   show_progress_bar=False)

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Sample one (q_cloud, pos_cloud, hard_neg_cloud) triplet.

        Returns numpy arrays (n_q, 384), (n_pos, 384), (n_neg, 384) + metadata.
        """
        entry = self.cache[self.rng.integers(len(self.cache))]

        # Query cloud: encode question on-the-fly
        q_cloud = self._embed_question(entry["question"])

        # Positive: sample from correct article chunks
        pos_ids = entry["pos_chunk_ids"]
        if len(pos_ids) > self.max_cloud:
            pos_ids = self.rng.choice(pos_ids, self.max_cloud, replace=False).tolist()
        pos_cloud = np.array(self.embeddings[pos_ids])

        # Hard negative: sample from pre-mined FAISS collisions
        neg_entries = entry["hard_neg_chunks"]
        neg_ids = [n["chunk_idx"] for n in neg_entries]
        if len(neg_ids) > self.max_cloud:
            neg_ids = self.rng.choice(neg_ids, self.max_cloud, replace=False).tolist()
        neg_cloud = np.array(self.embeddings[neg_ids])

        meta = {
            "question": entry["question"][:80],
            "answer": entry.get("answer", "")[:40],
            "pos_article": entry["title"][:40],
            "top_neg_cosine": neg_entries[0]["cosine_sim"],
            "n_q": len(q_cloud),
            "n_pos": len(pos_cloud),
            "n_neg": len(neg_cloud),
        }
        return q_cloud, pos_cloud, neg_cloud, meta


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train(
    n_steps: int = 500,
    batch_size: int = 4,
    lr: float = 1e-3,
    d_emb: int = 128,
    stalk_dim: int = 8,
    k: int = 4,
    margin: float = 0.5,
    log_every: int = 10,
    save_every: int = 100,
    device_str: str = "cuda",
):
    """Main training loop."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\nv6 Training — {n_steps} steps, batch={batch_size}, lr={lr}")
    print(f"  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    sampler = NQTripletSampler(max_cloud_size=12)

    # ── Model ─────────────────────────────────────────────────────────
    # backbone_dim=384 (sentence-transformer output), out_dim=128
    feat_map = TextFeatureMap(
        out_dim=d_emb,
        vocab_size=1,  # unused — we pass float embeddings directly
        backbone_dim=384,
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
    print(f"  Parameters: {total_params:,}")

    # ── Training ──────────────────────────────────────────────────────
    print(f"\n{'Step':>6s}  {'Loss':>8s}  {'GapPos':>8s}  {'GapNeg':>8s}  "
          f"{'Δ(N-P)':>7s}  {'α':>6s}  {'β':>6s}  {'t/s':>5s}")
    print("─" * 70)

    history = []
    t0 = time.time()

    for step in range(n_steps):
        step_t0 = time.time()

        # Accumulate gradients over mini-batch
        trainer.optimizer.zero_grad()
        batch_loss = 0.0
        batch_gap_pos = 0.0
        batch_gap_neg = 0.0
        batch_alpha = 0.0
        batch_beta = 0.0

        for _ in range(batch_size):
            q_np, pos_np, neg_np, meta = sampler.sample()

            # Convert to torch tensors
            q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
            pos_t = torch.tensor(pos_np, dtype=torch.float32, device=device)
            neg_t = torch.tensor(neg_np, dtype=torch.float32, device=device)

            # Forward through feature map
            q_emb = trainer.feature_map(q_t)
            pos_emb = trainer.feature_map(pos_t)
            neg_emb = trainer.feature_map(neg_t)

            # Sheaf loss (relative triplet: relu(gap_pos - gap_neg + margin))
            sheaf_loss, info = trainer.sheaf_loss_fn(q_emb, pos_emb, neg_emb)

            # Gauge router — learns α/β for inference. NOT gated into loss
            # (gating caused mode collapse: router zeroed β to kill loss).
            # Instead: sheaf loss flows independently, router gets soft
            # regularization toward v5 prior.
            alpha, beta = trainer.gauge_router(q_emb)
            router_reg = 0.01 * ((alpha - 0.7) ** 2 + (beta - 0.3) ** 2)
            loss = sheaf_loss + router_reg

            # Scale loss for gradient accumulation
            (loss / batch_size).backward()

            batch_loss += loss.item() / batch_size
            batch_gap_pos += info["gap_pos"] / batch_size
            batch_gap_neg += info["gap_neg"] / batch_size
            batch_alpha += alpha.item() / batch_size
            batch_beta += beta.item() / batch_size

        # Optimizer step
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
            "step_time": step_time,
        }
        history.append(record)

        if step % log_every == 0 or step == n_steps - 1:
            print(f"{step:>6d}  {batch_loss:>8.4f}  {batch_gap_pos:>8.4f}  "
                  f"{batch_gap_neg:>8.4f}  {gap_delta:>+7.3f}  "
                  f"{batch_alpha:>6.3f}  {batch_beta:>6.3f}  "
                  f"{step_time:>4.1f}s")

        if save_every > 0 and (step + 1) % save_every == 0:
            ckpt_path = CHECKPOINT_DIR / f"trainer_step{step+1}.pt"
            torch.save(trainer.state_dict(), ckpt_path)
            print(f"  → checkpoint: {ckpt_path.name}")

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE — {n_steps} steps in {elapsed:.0f}s "
          f"({elapsed/n_steps:.2f}s/step)")
    print(f"{'='*65}")

    first = history[0]
    last = history[-1]
    print(f"  Loss:    {first['loss']:.4f} → {last['loss']:.4f}")
    print(f"  GapPos:  {first['gap_pos']:.4f} → {last['gap_pos']:.4f}")
    print(f"  GapNeg:  {first['gap_neg']:.4f} → {last['gap_neg']:.4f}")
    print(f"  α:       {first['alpha']:.4f} → {last['alpha']:.4f}")
    print(f"  β:       {first['beta']:.4f} → {last['beta']:.4f}")

    # Save history
    with open(CHECKPOINT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History: {CHECKPOINT_DIR / 'training_history.json'}")

    # Save final model
    final_path = CHECKPOINT_DIR / "trainer_final.pt"
    torch.save(trainer.state_dict(), final_path)
    print(f"  Model:   {final_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(n_steps=args.steps, batch_size=args.batch, lr=args.lr,
          device_str=args.device)
