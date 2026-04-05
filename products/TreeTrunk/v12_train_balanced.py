#!/usr/bin/env python3
"""v12 Balanced Crystallization — 10K steps, edge-balanced sampling.

Solves the v11 edge starvation problem. English syntax (nsubj, prep, det)
dominates the dataset, starving code/math/physics edges of gradient updates.

EdgeBalancedSampler guarantees each of the 122 restriction maps receives
equal training exposure by round-robin over edge types instead of
uniform triplet sampling.

Continuous isomorphism tracking: every 1000 steps, measure Frobenius
distances between known conceptual equivalents. Watch the matrices
crystallize into their final geometry.

Target: step_next ↔ t_next should drop from 0.074 to <0.01
        (math deduction == physical time)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from v9_causal_graph import TypedSheafTopologyLoss

GRAPH_DIR = Path(__file__).parent / "results" / "v9_compiled_graphs"
CHECKPOINT_DIR = Path(__file__).parent / "results" / "v12_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Edge-Balanced Sampler
# ══════════════════════════════════════════════════════════════════════════════

class EdgeBalancedSampler:
    """Round-robin over edge types, not triplets.

    For each step, picks an edge type (cycling through all 122) then
    samples a triplet that contains that edge type. This guarantees
    every restriction map gets equal gradient exposure.
    """

    def __init__(self, all_triplets: list[dict], n_edge_types: int, rng):
        self.triplets = all_triplets
        self.rng = rng
        self.n_edge_types = n_edge_types

        # Build inverted index: edge_type_id → list of triplet indices containing it
        print(f"  Building edge-type inventory...")
        self.edge_to_triplets: dict[int, list[int]] = defaultdict(list)

        for t_idx, triplet in enumerate(all_triplets):
            # Collect edge types from q, pos, neg graphs
            seen = set()
            for graph_key in ["q_graph", "pos_graphs", "neg_graphs"]:
                graphs = triplet[graph_key]
                if isinstance(graphs, dict):
                    graphs = [graphs]
                for g in graphs:
                    et = g["edge_type_ids"]
                    if et.numel() > 0:
                        for t in et.unique().tolist():
                            seen.add(t)
            for edge_type in seen:
                self.edge_to_triplets[edge_type].append(t_idx)

        # Report edge frequencies
        freqs = {et: len(idxs) for et, idxs in self.edge_to_triplets.items()}
        min_freq = min(freqs.values())
        max_freq = max(freqs.values())
        print(f"  Edge type coverage: {len(self.edge_to_triplets)} types")
        print(f"  Frequency range: {min_freq} → {max_freq} triplets")
        print(f"  Max/min ratio: {max_freq/max(min_freq,1):.0f}x (before balancing)")

        # Cycle through edge types that have at least some triplets
        self.active_edge_types = sorted(self.edge_to_triplets.keys())
        self._cursor = 0

    def sample(self) -> dict:
        """Sample one triplet via edge-type round-robin."""
        # Pick an edge type (round-robin + random offset for variety)
        etype = self.active_edge_types[self._cursor % len(self.active_edge_types)]
        self._cursor += 1
        # Sample a triplet containing that edge type
        triplet_idx = self.rng.choice(self.edge_to_triplets[etype])
        return self.triplets[triplet_idx]


# ══════════════════════════════════════════════════════════════════════════════
# Pre-encoder
# ══════════════════════════════════════════════════════════════════════════════

class NodeEncoder:
    def __init__(self, device):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.device = device
        self.cache: dict[str, torch.Tensor] = {}

    def encode_all(self, triplets):
        unique_texts = set()
        for entry in triplets:
            for text in entry["q_graph"]["node_texts"]:
                unique_texts.add(text)
            for g in entry["pos_graphs"]:
                for text in g["node_texts"]:
                    unique_texts.add(text)
            for g in entry["neg_graphs"]:
                for text in g["node_texts"]:
                    unique_texts.add(text)

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

    def lookup(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(1, 384, device=self.device)
        return torch.stack([self.cache[t] for t in texts])


# ══════════════════════════════════════════════════════════════════════════════
# Graph utilities
# ══════════════════════════════════════════════════════════════════════════════

def edges_from_tensors(graph_tensors: dict) -> list[tuple[int, int, str]]:
    ei = graph_tensors["edge_index"]
    et = graph_tensors["edge_type_ids"]
    if ei.shape[0] == 0:
        return []
    return [(int(ei[i, 0]), int(ei[i, 1]), str(int(et[i])))
            for i in range(ei.shape[0])]


def merge_and_encode(q_t, other_t, node_enc, device):
    """Merge query + other graph, return (embeddings, edges)."""
    q_nodes = q_t["node_texts"]
    o_nodes = other_t["node_texts"]
    n_q = len(q_nodes)

    q_edges = edges_from_tensors(q_t)
    o_edges_raw = edges_from_tensors(other_t)
    merged_edges = list(q_edges) + [(s + n_q, d + n_q, t) for s, d, t in o_edges_raw]
    if n_q > 0 and len(o_nodes) > 0:
        merged_edges.append((n_q - 1, n_q, "bridge"))

    all_texts = q_nodes + o_nodes
    emb = node_enc.lookup(all_texts)
    return emb, merged_edges


# ══════════════════════════════════════════════════════════════════════════════
# Isomorphism tracking (continuous)
# ══════════════════════════════════════════════════════════════════════════════

def frob_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    """Normalized Frobenius distance between two matrices."""
    A_n = A / (A.norm('fro') + 1e-12)
    B_n = B / (B.norm('fro') + 1e-12)
    return float((A_n - B_n).norm('fro').item())


def measure_isomorphism(loss_fn, registry) -> dict:
    """Compute Frobenius distances for tracked conceptual pairs."""
    name_to_id = registry["name_to_id"]

    def get_matrix(name):
        if name not in name_to_id:
            return None
        mid = str(name_to_id[name])
        if mid not in loss_fn.sheaf.restriction_maps:
            return None
        return loss_fn.sheaf.restriction_maps[mid].weight.detach()

    pairs = [
        ("step_next", "t_next"),                         # math ↔ physics
        ("code_If_test", "code_Compare_ops"),            # code internal
        ("operand_left", "operand_right"),               # math symmetry
        ("code_While_body", "acl"),                       # code ↔ english
        ("code_BinOp_left", "code_BinOp_right"),         # code symmetry
        ("advcl", "code_If_test"),                       # english ↔ code conditional
        ("conj", "step_next"),                            # english ↔ math sequence
        ("dobj", "operand_right"),                        # english ↔ math object
    ]

    distances = {}
    for a, b in pairs:
        ma = get_matrix(a)
        mb = get_matrix(b)
        if ma is not None and mb is not None:
            distances[f"{a}_vs_{b}"] = round(frob_distance(ma, mb), 4)
        else:
            distances[f"{a}_vs_{b}"] = None

    # Also compute a random baseline
    import random as _random
    keys = [k for k in loss_fn.sheaf.restriction_maps.keys()]
    _random.shuffle(keys)
    baseline_dists = []
    for i in range(0, min(50, len(keys) - 1), 2):
        ma = loss_fn.sheaf.restriction_maps[keys[i]].weight.detach()
        mb = loss_fn.sheaf.restriction_maps[keys[i+1]].weight.detach()
        baseline_dists.append(frob_distance(ma, mb))
    distances["_random_baseline_mean"] = round(float(np.mean(baseline_dists)), 4)
    return distances


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def train(
    n_steps: int = 10000,
    batch_size: int = 8,
    lr: float = 1e-3,
    stalk_dim: int = 8,
    margin: float = 0.5,
    log_every: int = 50,
    track_every: int = 1000,
    save_every: int = 2000,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\nv12 Balanced Crystallization — {n_steps} steps, batch={batch_size}")
    print(f"  Device: {device}")

    # Load all 4 domains
    print(f"\nLoading all 4 compiled domains...")
    all_triplets = []
    all_triplets.extend(torch.load(GRAPH_DIR / "nq_graphs.pt", weights_only=False))
    all_triplets.extend(torch.load(GRAPH_DIR / "gsm8k_graphs.pt", weights_only=False))
    all_triplets.extend(torch.load(GRAPH_DIR / "mbpp_graphs.pt", weights_only=False))
    all_triplets.extend(torch.load(GRAPH_DIR / "telemetry_graphs.pt", weights_only=False))
    print(f"  Total: {len(all_triplets):,} triplets")

    # Pre-encode
    print(f"\nPre-encoding...")
    node_enc = NodeEncoder(device)
    node_enc.encode_all(all_triplets)

    # Registry
    with open(GRAPH_DIR / "v9_edge_registry.json") as f:
        registry = json.load(f)
    n_edge_types = registry["total_types"]

    # Sampler
    rng = np.random.default_rng(42)
    print(f"\nBuilding EdgeBalancedSampler...")
    sampler = EdgeBalancedSampler(all_triplets, n_edge_types, rng)

    # Model
    loss_fn = TypedSheafTopologyLoss(
        d_in=384, stalk_dim=stalk_dim, margin=margin
    ).to(device)
    for i in range(n_edge_types):
        loss_fn.sheaf._get_restriction(str(i))
    loss_fn.sheaf._get_restriction("bridge")

    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=lr)
    total_params = sum(p.numel() for p in loss_fn.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Restriction maps: {len(loss_fn.sheaf.restriction_maps)}")

    # Training
    print(f"\n  {'Step':>6s}  {'Loss':>7s}  {'GPos':>7s}  {'GNeg':>7s}  "
          f"{'Δ(N-P)':>8s}  {'t':>5s}")
    print("  " + "─" * 55)

    history = []
    iso_trajectory = []
    t0 = time.time()

    for step in range(n_steps):
        step_t0 = time.time()
        optimizer.zero_grad()

        batch_loss = 0.0
        batch_gp = 0.0
        batch_gn = 0.0

        for _ in range(batch_size):
            entry = sampler.sample()
            pos_g = entry["pos_graphs"][rng.integers(len(entry["pos_graphs"]))]
            neg_g = entry["neg_graphs"][rng.integers(len(entry["neg_graphs"]))]
            q_t = entry["q_graph"]

            pos_emb, pos_edges = merge_and_encode(q_t, pos_g, node_enc, device)
            neg_emb, neg_edges = merge_and_encode(q_t, neg_g, node_enc, device)

            loss, info = loss_fn(pos_emb, pos_edges, neg_emb, neg_edges)
            (loss / batch_size).backward()

            batch_loss += info["loss"]
            batch_gp += info["gap_pos"]
            batch_gn += info["gap_neg"]

        optimizer.step()
        step_time = time.time() - step_t0

        avg_loss = batch_loss / batch_size
        avg_gp = batch_gp / batch_size
        avg_gn = batch_gn / batch_size
        delta = avg_gn - avg_gp

        history.append({
            "step": step, "loss": avg_loss,
            "gap_pos": avg_gp, "gap_neg": avg_gn,
            "gap_delta": delta, "step_time": step_time,
        })

        if step % log_every == 0 or step == n_steps - 1:
            print(f"  {step:>6d}  {avg_loss:>7.4f}  {avg_gp:>7.4f}  "
                  f"{avg_gn:>7.4f}  {delta:>+8.4f}  {step_time:>4.1f}s")

        # Isomorphism tracking
        if (step + 1) % track_every == 0:
            distances = measure_isomorphism(loss_fn, registry)
            distances["step"] = step + 1
            iso_trajectory.append(distances)
            print(f"    ISO@{step+1}: "
                  f"step_next↔t_next={distances.get('step_next_vs_t_next', 'N/A'):.4f}  "
                  f"baseline={distances.get('_random_baseline_mean', 0):.4f}  "
                  f"advcl↔code_If={distances.get('advcl_vs_code_If_test', 0):.4f}")

        if save_every > 0 and (step + 1) % save_every == 0:
            ckpt = CHECKPOINT_DIR / f"v12_step{step+1}.pt"
            torch.save(loss_fn.state_dict(), ckpt)
            print(f"    → checkpoint: {ckpt.name}")

    elapsed = time.time() - t0

    # Save final
    final = CHECKPOINT_DIR / "v12_universal_manifold.pt"
    torch.save(loss_fn.state_dict(), final)
    with open(CHECKPOINT_DIR / "v12_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(CHECKPOINT_DIR / "v12_iso_trajectory.json", "w") as f:
        json.dump(iso_trajectory, f, indent=2)

    last100 = history[-100:]
    avg_delta = float(np.mean([r["gap_delta"] for r in last100]))

    print(f"\n{'='*70}")
    print(f"  v12 CRYSTALLIZATION COMPLETE — {elapsed:.0f}s ({elapsed/60:.0f}m)")
    print(f"{'='*70}")
    print(f"  Loss:     {history[0]['loss']:.4f} → {history[-1]['loss']:.4f}")
    print(f"  GapPos:   {history[0]['gap_pos']:.4f} → {history[-1]['gap_pos']:.4f}")
    print(f"  GapNeg:   {history[0]['gap_neg']:.4f} → {history[-1]['gap_neg']:.4f}")
    print(f"  Δ(N-P):   last-100 avg: {avg_delta:+.4f}")
    print(f"\n  Isomorphism trajectory:")
    if iso_trajectory:
        first = iso_trajectory[0]
        last = iso_trajectory[-1]
        print(f"    step_next↔t_next: {first.get('step_next_vs_t_next',0):.4f} → {last.get('step_next_vs_t_next',0):.4f}")
        print(f"    advcl↔code_If_test: {first.get('advcl_vs_code_If_test',0):.4f} → {last.get('advcl_vs_code_If_test',0):.4f}")
        print(f"    baseline:        {first.get('_random_baseline_mean',0):.4f} → {last.get('_random_baseline_mean',0):.4f}")
    print(f"\n  Model: {final}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(n_steps=args.steps, batch_size=args.batch, lr=args.lr)
