#!/usr/bin/env python3
"""P vs NP v5: Frustration Loops via Implication Graph H₁.

v1-v4 measured LOCAL structure (H₀). All failed.
The SAT transition is GLOBAL: contradiction loops appearing.

v5: Build the implication graph, compute H₁ (loops), detect when
unavoidable frustration cycles first appear.

The implication graph:
  - Nodes: 2N literals (x₁, ¬x₁, x₂, ¬x₂, ..., xₙ, ¬xₙ)
  - For each clause (a ∨ b ∨ c), add implications:
    ¬a → b∨c (if a is false, b or c must be true)
    In 2-SAT: ¬a → b means "if not a then b"
    In 3-SAT: we use the RESOLUTION structure — pairs of clauses
    that share a variable with opposite polarity create implications

  Simpler approach: build the CONFLICT GRAPH
  - Nodes: N variables
  - Edge between i and j if they appear in a clause with OPPOSITE polarities
    (one positive, one negative) — meaning setting one forces the other
  - Weight = number of such conflicting clauses

  At low α: sparse conflict graph, few cycles
  At α_c: conflict graph becomes "rigid" — cycles unavoidable
  At high α: dense, many short cycles

  H₁ persistence on the conflict graph detects loop formation.

GPU pipeline:
  - Build conflict adjacency matrix on GPU (batched over instances)
  - Compute shortest-path distances (Floyd-Warshall or BFS)
  - H₁ via Rips complex on the conflict graph embedding
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

COLORS = {"gold": "#c5a03f", "teal": "#45a8b0", "red": "#e94560",
          "purple": "#9b59b6", "bg": "#0f0d08", "text": "#d6d0be",
          "muted": "#817a66"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.facecolor": COLORS["bg"], "figure.facecolor": COLORS["bg"],
    "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "axes.edgecolor": COLORS["muted"], "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})

OUTPUT_DIR = Path("problems/p-vs-np/results")
FIG_DIR = Path("problems/p-vs-np/results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_3sat(n_vars, alpha, seed):
    rng = np.random.default_rng(seed)
    n_clauses = int(alpha * n_vars)
    var_indices = np.zeros((n_clauses, 3), dtype=np.int32)
    negations = np.zeros((n_clauses, 3), dtype=bool)
    for i in range(n_clauses):
        var_indices[i] = rng.choice(n_vars, size=3, replace=False)
        negations[i] = rng.random(3) < 0.5
    return var_indices, negations, n_clauses


def build_literal_conflict_graph(var_indices, negations, n_vars):
    """Build conflict graph on 2N literal nodes.

    Nodes: 0..2N-1 where node 2i = variable i positive, 2i+1 = variable i negative.

    For each clause (l₁ ∨ l₂ ∨ l₃), the implication structure is:
      ¬l₁ → (l₂ ∨ l₃): if l₁ is false, at least one of l₂, l₃ must be true
      For 2-SAT reduction: ¬l₁ → l₂ AND ¬l₁ → l₃ (overapproximation)

    Edge from literal node a to literal node b means "a implies b".
    We build the adjacency matrix of this directed implication graph,
    then symmetrize for undirected H₁ analysis.

    Returns: (2N, 2N) adjacency matrix
    """
    n_lit = 2 * n_vars
    adj = np.zeros((n_lit, n_lit), dtype=np.float32)

    for c_idx in range(len(var_indices)):
        lits = []
        for k in range(3):
            v = var_indices[c_idx, k]
            if negations[c_idx, k]:
                lits.append(2 * v + 1)  # negated → odd node
            else:
                lits.append(2 * v)      # positive → even node

        # For each literal in the clause, its negation implies the other two
        for i in range(3):
            neg_lit = lits[i] ^ 1  # flip the polarity bit
            for j in range(3):
                if i != j:
                    adj[neg_lit, lits[j]] += 1.0

    # Symmetrize for undirected analysis
    adj = adj + adj.T

    return adj


def conflict_graph_features(adj, n_vars):
    """Extract topological features from the literal conflict graph.

    Returns per-instance features:
      - n_edges: number of non-zero entries
      - mean_degree: average node degree
      - clustering: mean local clustering coefficient
      - n_triangles: number of triangles (proxy for H₁ generators)
      - spectral_gap: λ₂ - λ₁ of graph Laplacian (connectivity measure)
      - algebraic_connectivity: λ₂ (Fiedler value, 0 = disconnected)
    """
    n_lit = 2 * n_vars
    degree = adj.sum(axis=1)
    n_edges = int((adj > 0).sum()) // 2
    mean_deg = float(degree.mean())

    # Triangles via A³ trace
    adj_binary = (adj > 0).astype(np.float32)
    A2 = adj_binary @ adj_binary
    A3_diag = (A2 * adj_binary).sum()  # trace of A³
    n_triangles = int(A3_diag) // 6  # each triangle counted 6 times

    # Graph Laplacian eigenvalues (smallest few)
    L = np.diag(degree) - adj
    try:
        eigs = np.sort(np.linalg.eigvalsh(L))
        spectral_gap = float(eigs[1] - eigs[0]) if len(eigs) > 1 else 0
        algebraic_conn = float(eigs[1]) if len(eigs) > 1 else 0
    except Exception:
        spectral_gap = 0
        algebraic_conn = 0

    return {
        "n_edges": n_edges,
        "mean_degree": mean_deg,
        "n_triangles": n_triangles,
        "spectral_gap": spectral_gap,
        "algebraic_connectivity": algebraic_conn,
    }


def spectral_embedding_persistence(adj, d=10, max_points=300):
    """H₁-sensitive persistence via spectral embedding of conflict graph.

    Embed the 2N literal nodes using the d smallest non-trivial eigenvectors
    of the graph Laplacian. Then run Rips persistence on the embedding.

    H₁ features in the Rips complex correspond to loops in the conflict graph.
    """
    n = adj.shape[0]
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj

    # Spectral embedding
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        d_actual = min(d, n - 2)
        embedding = eigenvectors[:, 1:d_actual + 1]  # skip constant eigenvector
    except Exception:
        return np.array([]), np.array([]), 0

    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
        embedding = embedding[idx]
        n = max_points

    # GPU pairwise distances
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts = torch.tensor(embedding, dtype=torch.float32, device=device)
    dists = torch.cdist(pts, pts).cpu().numpy()

    # H₀ persistence (connected components)
    parent = list(range(n))
    rank_uf = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dists[i, j], i, j))
    edges.sort()

    h0_bars = []
    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank_uf[ri] < rank_uf[rj]: ri, rj = rj, ri
            parent[rj] = ri
            if rank_uf[ri] == rank_uf[rj]: rank_uf[ri] += 1
            h0_bars.append(float(dist))

    h0_bars = np.array(h0_bars)

    # H₁ proxy: count triangles in the Rips complex at various scales
    # A triangle (i,j,k) exists at scale ε if all three distances ≤ ε
    # H₁ = loops that are NOT boundaries of triangles
    # Proxy: ratio of edges to triangles at the onset scale
    # More triangles relative to edges = more H₁ killed = simpler topology
    # Fewer triangles relative to edges = more surviving H₁ = more frustration loops

    if len(h0_bars) > 0:
        onset_scale = float(np.percentile(h0_bars, 50))  # median merge distance
    else:
        onset_scale = 0

    # Count triangles at onset scale
    n_edges_at_onset = int(np.sum(dists[np.triu_indices(n, k=1)] <= onset_scale))
    n_triangles_at_onset = 0
    if onset_scale > 0 and n <= 300:
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] > onset_scale:
                    continue
                for k in range(j + 1, n):
                    if dists[i, k] <= onset_scale and dists[j, k] <= onset_scale:
                        n_triangles_at_onset += 1

    # H₁ proxy: edges - triangles at onset (more excess edges = more loops)
    h1_proxy = n_edges_at_onset - n_triangles_at_onset

    return h0_bars, h1_proxy, n_triangles_at_onset


def gini(values):
    n = len(values)
    if n <= 1 or np.sum(values) == 0:
        return 0.0
    sorted_v = np.sort(values)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_v)) / (n * np.sum(sorted_v)) - (n + 1.0) / n)


def main():
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  P vs NP v5: FRUSTRATION LOOPS (H₁ via Implication Graph)")
    print("  The transition is in the loops, not the components.")
    print(f"  {timestamp}")
    print("=" * 70)

    N_VARS = 60  # 2N=120 literal nodes; keep small for triangle counting
    ALPHA_GRID = np.arange(2.0, 7.0, 0.25)
    N_INSTANCES = 20
    results = []

    total_t0 = time.time()

    for alpha in ALPHA_GRID:
        alpha_t0 = time.time()
        onset_scales = []
        gini_values = []
        h1_proxies = []
        triangle_counts = []
        graph_features_list = []

        for inst in range(N_INSTANCES):
            seed = int(alpha * 10000) + inst
            vi, neg, nc = generate_3sat(N_VARS, alpha, seed)

            # Build literal conflict graph
            adj = build_literal_conflict_graph(vi, neg, N_VARS)

            # Graph-level features
            gf = conflict_graph_features(adj, N_VARS)
            graph_features_list.append(gf)

            # Spectral embedding + persistence
            h0_bars, h1_proxy, n_tri = spectral_embedding_persistence(adj, d=8, max_points=120)

            if len(h0_bars) > 0:
                onset = float(np.percentile(h0_bars, 95))
                g = gini(h0_bars)
            else:
                onset = 0
                g = 0

            onset_scales.append(onset)
            gini_values.append(g)
            h1_proxies.append(h1_proxy)
            triangle_counts.append(n_tri)

        mean_onset = float(np.mean(onset_scales))
        std_onset = float(np.std(onset_scales))
        mean_gini = float(np.mean(gini_values))
        mean_h1 = float(np.mean(h1_proxies))
        mean_tri = float(np.mean(triangle_counts))
        mean_alg_conn = float(np.mean([gf["algebraic_connectivity"] for gf in graph_features_list]))
        mean_graph_tri = float(np.mean([gf["n_triangles"] for gf in graph_features_list]))
        elapsed = time.time() - alpha_t0

        results.append({
            "alpha": float(alpha),
            "mean_onset": mean_onset,
            "std_onset": std_onset,
            "mean_gini": mean_gini,
            "mean_h1_proxy": mean_h1,
            "mean_rips_triangles": mean_tri,
            "mean_algebraic_connectivity": mean_alg_conn,
            "mean_graph_triangles": mean_graph_tri,
            "elapsed_s": round(elapsed, 1),
        })

        print(f"  α={alpha:.2f}: ε*={mean_onset:.4f} G={mean_gini:.4f} "
              f"H₁≈{mean_h1:.0f} tri={mean_tri:.0f} "
              f"λ₂={mean_alg_conn:.2f} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_t0

    # ── Analysis ──
    alphas = np.array([r["alpha"] for r in results])
    onsets = np.array([r["mean_onset"] for r in results])
    ginis_arr = np.array([r["mean_gini"] for r in results])
    h1s = np.array([r["mean_h1_proxy"] for r in results])
    alg_conns = np.array([r["mean_algebraic_connectivity"] for r in results])
    graph_tris = np.array([r["mean_graph_triangles"] for r in results])

    d_onset = np.gradient(onsets, alphas)
    d_h1 = np.gradient(h1s, alphas)
    d_conn = np.gradient(alg_conns, alphas)

    max_onset_idx = np.argmax(np.abs(d_onset))
    max_h1_idx = np.argmax(np.abs(d_h1))
    max_conn_idx = np.argmax(np.abs(d_conn))

    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")
    print(f"  ε* max |d/dα| at α = {alphas[max_onset_idx]:.2f}")
    print(f"  H₁ proxy max |d/dα| at α = {alphas[max_h1_idx]:.2f}")
    print(f"  λ₂ max |d/dα| at α = {alphas[max_conn_idx]:.2f}")
    print(f"  Known α_c ≈ 4.27")

    transitions = [alphas[max_onset_idx], alphas[max_h1_idx], alphas[max_conn_idx]]
    detected = any(3.5 <= t <= 5.5 for t in transitions)
    verdict = "PASS" if detected else "FAIL"
    print(f"  VERDICT: {verdict}")

    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax in axes.flat:
        ax.axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27")
        ax.grid(True, alpha=0.15)

    axes[0, 0].errorbar(alphas, onsets, yerr=[r["std_onset"] for r in results],
                         color=COLORS["teal"], linewidth=2, marker="o", markersize=4, capsize=2)
    axes[0, 0].set_ylabel("ε*"); axes[0, 0].set_title("Onset Scale (conflict graph embedding)", color=COLORS["teal"])
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(alphas, h1s, "-s", color=COLORS["gold"], linewidth=2, markersize=4)
    axes[0, 1].set_ylabel("H₁ proxy (excess edges over triangles)")
    axes[0, 1].set_title("Frustration Loops", color=COLORS["gold"])
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(alphas, alg_conns, "-^", color=COLORS["purple"], linewidth=2, markersize=4)
    axes[1, 0].set_xlabel("α"); axes[1, 0].set_ylabel("λ₂ (algebraic connectivity)")
    axes[1, 0].set_title("Graph Connectivity", color=COLORS["purple"])
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(alphas, graph_tris, "-d", color=COLORS["red"], linewidth=2, markersize=4)
    axes[1, 1].set_xlabel("α"); axes[1, 1].set_ylabel("Graph triangles")
    axes[1, 1].set_title("Implication Graph Triangles", color=COLORS["red"])
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"P vs NP v5: Frustration Loops (N={N_VARS}, {N_INSTANCES} inst/α)",
                 color=COLORS["text"], fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sat_phase_transition_v5.png")
    plt.close(fig)

    output = {
        "experiment": "sat_topology_v5",
        "feature_map": "literal_conflict_graph_H1",
        "n_vars": N_VARS, "n_instances_per_alpha": N_INSTANCES,
        "results": results,
        "transitions": {
            "onset": float(alphas[max_onset_idx]),
            "h1_proxy": float(alphas[max_h1_idx]),
            "algebraic_connectivity": float(alphas[max_conn_idx]),
        },
        "verdict": verdict,
        "total_time_s": round(total_elapsed, 1),
    }
    with open(OUTPUT_DIR / "sat_topology_v5.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Total: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
