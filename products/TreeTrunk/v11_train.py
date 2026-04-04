#!/usr/bin/env python3
"""v11 Cross-Domain Trainer — The Isomorphism Crucible.

Trains ONE DependencyGraphSheafLaplacian on all 4 compiled domains
(NQ, GSM8K, MBPP, Telemetry) simultaneously. The 122 restriction maps
compete for geometric territory in the 128-dim manifold.

If the Universal Machine Code hypothesis holds, conceptually equivalent
edge types from different notations will converge to similar matrices.

Stratified batching: each batch contains equal samples from all 4 domains,
forcing the shared projection to find domain-invariant geometry.

Output: v11_checkpoints/v11_universal_manifold.pt
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from v9_causal_graph import TypedSheafTopologyLoss

GRAPH_DIR = Path(__file__).parent / "results" / "v9_compiled_graphs"
CHECKPOINT_DIR = Path(__file__).parent / "results" / "v11_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Pre-encoder — encode all unique node texts once
# ══════════════════════════════════════════════════════════════════════════════

class NodeEncoder:
    """Pre-encodes all unique node texts from compiled graphs.

    Training becomes pure tensor lookup — no encoder calls.
    """

    def __init__(self, device):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.device = device
        self.cache: dict[str, torch.Tensor] = {}

    def encode_all_graphs(self, all_graphs: list[dict]):
        """Collect all unique node texts from triplets and encode once."""
        unique_texts = set()

        def collect(graph_tensors: dict):
            for text in graph_tensors["node_texts"]:
                unique_texts.add(text)

        for entry in all_graphs:
            collect(entry["q_graph"])
            for g in entry["pos_graphs"]:
                collect(g)
            for g in entry["neg_graphs"]:
                collect(g)

        unique_list = sorted(unique_texts)
        print(f"  Encoding {len(unique_list):,} unique node texts...")
        t0 = time.time()
        embs = self.encoder.encode(
            unique_list, convert_to_numpy=True,
            show_progress_bar=True, batch_size=512
        )
        print(f"  Encoded in {time.time()-t0:.1f}s")

        for text, emb in zip(unique_list, embs):
            self.cache[text] = torch.tensor(emb, dtype=torch.float32, device=self.device)

    def graph_to_embedding(self, graph_tensors: dict) -> torch.Tensor:
        """Look up embeddings for all nodes in a graph."""
        texts = graph_tensors["node_texts"]
        if not texts:
            return torch.zeros(1, 384, device=self.device)
        return torch.stack([self.cache[t] for t in texts])


# ══════════════════════════════════════════════════════════════════════════════
# Graph utilities
# ══════════════════════════════════════════════════════════════════════════════

def edges_from_tensors(graph_tensors: dict) -> list[tuple[int, int, str]]:
    """Convert edge tensors back to typed edge list."""
    ei = graph_tensors["edge_index"]
    et = graph_tensors["edge_type_ids"]
    if ei.shape[0] == 0:
        return []
    edges = []
    for i in range(ei.shape[0]):
        edges.append((int(ei[i, 0]), int(ei[i, 1]), str(int(et[i]))))
    return edges


def merge_triplet_graphs(
    q_tensors: dict, other_tensors: dict
) -> tuple[list, int]:
    """Merge query graph + one pos/neg graph. Returns (edges, n_q)."""
    q_nodes = q_tensors["node_texts"]
    o_nodes = other_tensors["node_texts"]
    n_q = len(q_nodes)

    q_edges = edges_from_tensors(q_tensors)
    o_edges_raw = edges_from_tensors(other_tensors)
    # Offset other graph's indices
    merged_edges = list(q_edges) + [(s + n_q, d + n_q, t) for s, d, t in o_edges_raw]
    # Bridge edge (type "bridge")
    if n_q > 0 and len(o_nodes) > 0:
        merged_edges.append((n_q - 1, n_q, "bridge"))

    return merged_edges, n_q


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train(
    n_steps: int = 1000,
    batch_size: int = 8,
    lr: float = 1e-3,
    stalk_dim: int = 8,
    margin: float = 0.5,
    log_every: int = 20,
    save_every: int = 250,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\nv11 Cross-Domain Training — {n_steps} steps, batch={batch_size}")
    print(f"  Device: {device}")

    # ── Load all 4 domains ────────────────────────────────────────────
    print(f"\nLoading all 4 compiled domains...")
    nq = torch.load(GRAPH_DIR / "nq_graphs.pt", weights_only=False)
    gsm8k = torch.load(GRAPH_DIR / "gsm8k_graphs.pt", weights_only=False)
    mbpp = torch.load(GRAPH_DIR / "mbpp_graphs.pt", weights_only=False)
    tel = torch.load(GRAPH_DIR / "telemetry_graphs.pt", weights_only=False)

    domains = {
        "nq": nq,
        "gsm8k": gsm8k,
        "mbpp": mbpp,
        "telemetry": tel,
    }
    print(f"  NQ:        {len(nq):,}")
    print(f"  GSM8K:     {len(gsm8k):,}")
    print(f"  MBPP:      {len(mbpp):,}")
    print(f"  Telemetry: {len(tel):,}")
    total = sum(len(v) for v in domains.values())
    print(f"  Total:     {total:,} triplets")

    # ── Pre-encode all unique node texts ──────────────────────────────
    print(f"\nPre-encoding node texts (one-time cost)...")
    node_enc = NodeEncoder(device)
    all_graphs = nq + gsm8k + mbpp + tel
    node_enc.encode_all_graphs(all_graphs)

    # ── Load registry for edge type count ─────────────────────────────
    with open(GRAPH_DIR / "v9_edge_registry.json") as f:
        registry = json.load(f)
    n_edge_types = registry["total_types"]
    print(f"  {n_edge_types} edge types in registry")

    # ── Model ─────────────────────────────────────────────────────────
    loss_fn = TypedSheafTopologyLoss(
        d_in=384, stalk_dim=stalk_dim, margin=margin
    ).to(device)

    # Pre-instantiate ALL 122 restriction maps from the registry
    # (plus "bridge" for cross-graph connections)
    for i in range(n_edge_types):
        loss_fn.sheaf._get_restriction(str(i))
    loss_fn.sheaf._get_restriction("bridge")

    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=lr)
    rng = np.random.default_rng(42)

    total_params = sum(p.numel() for p in loss_fn.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Restriction maps: {len(loss_fn.sheaf.restriction_maps)}")

    # ── Training ──────────────────────────────────────────────────────
    domain_names = list(domains.keys())
    per_domain = max(1, batch_size // len(domain_names))

    print(f"\n  Stratified batch: {per_domain} per domain × {len(domain_names)} domains")
    print(f"\n  {'Step':>6s}  {'Loss':>8s}  {'GPos':>8s}  {'GNeg':>8s}  "
          f"{'Δ(N-P)':>8s}  {'t':>5s}")
    print("  " + "─" * 60)

    history = []
    t0 = time.time()

    for step in range(n_steps):
        step_t0 = time.time()
        optimizer.zero_grad()

        batch_loss = 0.0
        batch_gp = 0.0
        batch_gn = 0.0
        count = 0

        # Stratified: equal samples from each domain
        for domain_name in domain_names:
            pool = domains[domain_name]
            for _ in range(per_domain):
                entry = pool[rng.integers(len(pool))]

                pos_g = entry["pos_graphs"][rng.integers(len(entry["pos_graphs"]))]
                neg_g = entry["neg_graphs"][rng.integers(len(entry["neg_graphs"]))]

                q_t = entry["q_graph"]

                # Build pos: q + pos merged
                pos_edges, _ = merge_triplet_graphs(q_t, pos_g)
                pos_all_texts = q_t["node_texts"] + pos_g["node_texts"]
                pos_graph_dict = {"node_texts": pos_all_texts}
                pos_emb = node_enc.graph_to_embedding(pos_graph_dict)

                # Build neg: q + neg merged
                neg_edges, _ = merge_triplet_graphs(q_t, neg_g)
                neg_all_texts = q_t["node_texts"] + neg_g["node_texts"]
                neg_graph_dict = {"node_texts": neg_all_texts}
                neg_emb = node_enc.graph_to_embedding(neg_graph_dict)

                loss, info = loss_fn(pos_emb, pos_edges, neg_emb, neg_edges)
                (loss / (per_domain * len(domain_names))).backward()

                batch_loss += info["loss"]
                batch_gp += info["gap_pos"]
                batch_gn += info["gap_neg"]
                count += 1

        optimizer.step()
        step_time = time.time() - step_t0

        avg_loss = batch_loss / max(count, 1)
        avg_gp = batch_gp / max(count, 1)
        avg_gn = batch_gn / max(count, 1)
        delta = avg_gn - avg_gp

        record = {
            "step": step, "loss": avg_loss,
            "gap_pos": avg_gp, "gap_neg": avg_gn,
            "gap_delta": delta, "step_time": step_time,
        }
        history.append(record)

        if step % log_every == 0 or step == n_steps - 1:
            print(f"  {step:>6d}  {avg_loss:>8.4f}  {avg_gp:>8.4f}  "
                  f"{avg_gn:>8.4f}  {delta:>+8.4f}  {step_time:>4.1f}s")

        if save_every > 0 and (step + 1) % save_every == 0:
            ckpt = CHECKPOINT_DIR / f"v11_step{step+1}.pt"
            torch.save(loss_fn.state_dict(), ckpt)
            print(f"    → checkpoint: {ckpt.name}")

    elapsed = time.time() - t0

    # Save final
    final = CHECKPOINT_DIR / "v11_universal_manifold.pt"
    torch.save(loss_fn.state_dict(), final)
    with open(CHECKPOINT_DIR / "v11_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Summary
    last50 = history[-50:]
    avg_delta = np.mean([r["gap_delta"] for r in last50])

    print(f"\n{'='*65}")
    print(f"  v11 CROSS-DOMAIN TRAINING COMPLETE — {elapsed:.0f}s")
    print(f"{'='*65}")
    print(f"  Loss:     {history[0]['loss']:.4f} → {history[-1]['loss']:.4f}")
    print(f"  GapPos:   {history[0]['gap_pos']:.4f} → {history[-1]['gap_pos']:.4f}")
    print(f"  GapNeg:   {history[0]['gap_neg']:.4f} → {history[-1]['gap_neg']:.4f}")
    print(f"  Δ(N-P):   {history[0]['gap_delta']:+.4f} → {history[-1]['gap_delta']:+.4f}")
    print(f"            (last-50 avg: {avg_delta:+.4f})")
    print(f"  Model: {final}")
    print(f"{'='*65}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(n_steps=args.steps, batch_size=args.batch, lr=args.lr)
