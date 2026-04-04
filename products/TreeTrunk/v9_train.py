#!/usr/bin/env python3
"""v9 Training — Typed dependency sheaf on pre-compiled graphs.

Loads pre-compiled NQ + GSM8K graphs from disk (zero spaCy overhead).
Trains DependencyGraphSheafLaplacian with 52 typed restriction maps.
Stratified batches: equal NQ and GSM8K per step.

The sheaf learns different geometric rotations for:
  - nsubj (who acts) vs dobj (who receives)
  - op_mul (multiplication) vs step_next (deductive sequence)
  - prep (spatial/temporal relation) vs amod (attribute modification)

All under the same spectral gap loss.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from v9_causal_graph import DependencyGraphSheafLaplacian, TypedSheafTopologyLoss

GRAPH_DIR = Path(__file__).parent / "results" / "v9_compiled_graphs"
CHECKPOINT_DIR = Path(__file__).parent / "results" / "v9_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_compiled_graphs():
    """Load pre-compiled graph triplets from disk."""
    print("Loading compiled graphs...")
    nq = torch.load(GRAPH_DIR / "nq_graphs.pt", weights_only=False)
    gsm8k = torch.load(GRAPH_DIR / "gsm8k_graphs.pt", weights_only=False)
    print(f"  NQ:    {len(nq):,} triplets")
    print(f"  GSM8K: {len(gsm8k):,} triplets")
    return nq, gsm8k


def encode_nodes(node_texts: list[str], encoder, device) -> torch.Tensor:
    """Encode node texts to 384-dim embeddings."""
    if not node_texts:
        return torch.zeros(1, 384, device=device)
    embs = encoder.encode(node_texts, convert_to_numpy=True, show_progress_bar=False)
    return torch.tensor(embs, dtype=torch.float32, device=device)


def graph_to_typed_edges(graph_dict: dict) -> list[tuple[int, int, str]]:
    """Convert compiled graph tensors back to typed edge list for the sheaf."""
    edge_index = graph_dict["edge_index"]  # (n_edges, 2)
    edge_type_ids = graph_dict["edge_type_ids"]  # (n_edges,)

    if edge_index.shape[0] == 0:
        return []

    # We pass integer type IDs as strings — the sheaf's ModuleDict uses string keys
    edges = []
    for i in range(edge_index.shape[0]):
        src = int(edge_index[i, 0])
        dst = int(edge_index[i, 1])
        etype = str(int(edge_type_ids[i]))
        edges.append((src, dst, etype))
    return edges


def train(
    n_steps: int = 500,
    batch_size: int = 8,
    lr: float = 1e-3,
    stalk_dim: int = 8,
    margin: float = 0.5,
    log_every: int = 10,
    save_every: int = 200,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\nv9 Typed Sheaf Training — {n_steps} steps, batch={batch_size}")
    print(f"  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    nq_graphs, gsm8k_graphs = load_compiled_graphs()

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Model ─────────────────────────────────────────────────────────
    loss_fn = TypedSheafTopologyLoss(
        d_in=384, stalk_dim=stalk_dim, margin=margin
    ).to(device)

    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=lr)
    rng = np.random.default_rng(42)

    total_params = sum(p.numel() for p in loss_fn.parameters())
    print(f"  Initial params: {total_params:,}")

    # ── Training ──────────────────────────────────────────────────────
    print(f"\n{'Step':>6s}  {'Loss':>8s}  {'GPos':>8s}  {'GNeg':>8s}  "
          f"{'Δ(N-P)':>8s}  {'NQ':>3s} {'GSM':>3s}  {'Maps':>4s}  {'t':>5s}")
    print("─" * 72)

    history = []
    t0 = time.time()

    for step in range(n_steps):
        step_t0 = time.time()
        optimizer.zero_grad()

        batch_loss = 0.0
        batch_gp = 0.0
        batch_gn = 0.0
        n_nq = 0
        n_gsm = 0

        # Stratified batch: half NQ, half GSM8K
        half = batch_size // 2

        for b in range(batch_size):
            # Pick domain
            if b < half:
                pool = nq_graphs
                n_nq += 1
            else:
                pool = gsm8k_graphs
                n_gsm += 1

            entry = pool[rng.integers(len(pool))]

            # Pick one positive and one negative graph
            pos_g = entry["pos_graphs"][rng.integers(len(entry["pos_graphs"]))]
            neg_g = entry["neg_graphs"][rng.integers(len(entry["neg_graphs"]))]

            # Merge query + positive into one graph
            q_g = entry["q_graph"]
            q_nodes = q_g["node_texts"]
            pos_nodes = pos_g["node_texts"]
            neg_nodes = neg_g["node_texts"]

            # Encode nodes
            pos_all_nodes = q_nodes + pos_nodes
            neg_all_nodes = q_nodes + neg_nodes

            pos_emb = encode_nodes(pos_all_nodes, encoder, device)
            neg_emb = encode_nodes(neg_all_nodes, encoder, device)

            # Build merged edge lists (offset pos/neg node indices by len(q_nodes))
            n_q = len(q_nodes)
            q_edges = graph_to_typed_edges(q_g)

            pos_edges_raw = graph_to_typed_edges(pos_g)
            pos_edges = list(q_edges) + [(s + n_q, d + n_q, t) for s, d, t in pos_edges_raw]
            # Bridge: connect last query node to first pos node
            if n_q > 0 and len(pos_nodes) > 0:
                pos_edges.append((n_q - 1, n_q, "0"))  # bridge edge (type 0)

            neg_edges_raw = graph_to_typed_edges(neg_g)
            neg_edges = list(q_edges) + [(s + n_q, d + n_q, t) for s, d, t in neg_edges_raw]
            if n_q > 0 and len(neg_nodes) > 0:
                neg_edges.append((n_q - 1, n_q, "0"))

            # Forward
            loss, info = loss_fn(pos_emb, pos_edges, neg_emb, neg_edges)
            (loss / batch_size).backward()

            batch_loss += info["loss"] / batch_size
            batch_gp += info["gap_pos"] / batch_size
            batch_gn += info["gap_neg"] / batch_size

        optimizer.step()
        step_time = time.time() - step_t0
        delta = batch_gn - batch_gp
        n_maps = len(loss_fn.sheaf.restriction_maps)

        record = {
            "step": step, "loss": batch_loss,
            "gap_pos": batch_gp, "gap_neg": batch_gn,
            "gap_delta": delta, "n_maps": n_maps,
            "step_time": step_time,
        }
        history.append(record)

        if step % log_every == 0 or step == n_steps - 1:
            print(f"{step:>6d}  {batch_loss:>8.4f}  {batch_gp:>8.4f}  "
                  f"{batch_gn:>8.4f}  {delta:>+8.4f}  "
                  f"{n_nq:>3d} {n_gsm:>3d}  {n_maps:>4d}  "
                  f"{step_time:>4.1f}s")

        if save_every > 0 and (step + 1) % save_every == 0:
            ckpt = CHECKPOINT_DIR / f"v9_step{step+1}.pt"
            torch.save(loss_fn.state_dict(), ckpt)
            print(f"  → checkpoint: {ckpt.name}")

    elapsed = time.time() - t0

    first, last = history[0], history[-1]
    last50 = history[-50:]
    avg_delta = np.mean([r["gap_delta"] for r in last50])

    print(f"\n{'='*72}")
    print(f"  v9 TRAINING COMPLETE — {n_steps} steps in {elapsed:.0f}s")
    print(f"{'='*72}")
    print(f"  Loss:    {first['loss']:.4f} → {last['loss']:.4f}")
    print(f"  GapPos:  {first['gap_pos']:.4f} → {last['gap_pos']:.4f}")
    print(f"  GapNeg:  {first['gap_neg']:.4f} → {last['gap_neg']:.4f}")
    print(f"  Δ(N-P):  {first['gap_delta']:+.4f} → {last['gap_delta']:+.4f}  "
          f"(last-50 avg: {avg_delta:+.4f})")
    print(f"  Maps:    {last['n_maps']} typed restriction matrices")
    print(f"{'='*72}")

    with open(CHECKPOINT_DIR / "v9_history.json", "w") as f:
        json.dump(history, f, indent=2)
    final = CHECKPOINT_DIR / "v9_final.pt"
    torch.save(loss_fn.state_dict(), final)
    print(f"  Model: {final}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(n_steps=args.steps, batch_size=args.batch, lr=args.lr)
