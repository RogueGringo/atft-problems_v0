#!/usr/bin/env python3
"""P vs NP: SAT Phase Transition via Topological Detection.

No solver. No search. Just topology.

The 3-SAT phase transition at α_c ≈ 4.27 (clause-to-variable ratio) is
structurally identical to the confinement-deconfinement transition in
gauge theory. This script measures it topologically:

1. Generate random 3-SAT instances at varying α
2. Translate each instance to a point cloud (spectral embedding of clause-variable graph)
3. Compute H₀ persistence on GPU (batched pairwise distances + Union-Find)
4. Extract onset scale ε*(α) and Gini trajectory
5. Detect the phase transition as a topological waypoint

The GPU computes the shape of the problem. The shape tells you the hardness.

No differential equations. No sequential search. Just parallel linear algebra.
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
from scipy.spatial.distance import pdist

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


def generate_random_3sat(n_vars: int, alpha: float, seed: int) -> np.ndarray:
    """Generate a random 3-SAT instance.

    Args:
        n_vars: Number of Boolean variables
        alpha: Clause-to-variable ratio (n_clauses = alpha * n_vars)
        seed: Random seed

    Returns:
        (n_clauses, 3) array where each entry is a signed variable index.
        Positive = variable, negative = negated variable.
    """
    rng = np.random.default_rng(seed)
    n_clauses = int(alpha * n_vars)
    clauses = np.zeros((n_clauses, 3), dtype=np.int32)
    for i in range(n_clauses):
        # Pick 3 distinct variables
        vars_chosen = rng.choice(n_vars, size=3, replace=False) + 1
        # Randomly negate each
        signs = rng.choice([-1, 1], size=3)
        clauses[i] = vars_chosen * signs
    return clauses


def sat_to_point_cloud(clauses: np.ndarray, n_vars: int) -> np.ndarray:
    """Translate SAT instance to point cloud via spectral embedding.

    Builds the clause-variable bipartite graph, computes the graph Laplacian,
    and uses the first d eigenvectors as coordinates.

    Each node (variable or clause) becomes a point in R^d.
    """
    n_clauses = len(clauses)
    n_total = n_vars + n_clauses

    # Build adjacency matrix of bipartite graph
    # Variables: nodes 0..n_vars-1
    # Clauses: nodes n_vars..n_total-1
    adj = np.zeros((n_total, n_total))
    for c_idx, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1  # 0-indexed
            clause_node = n_vars + c_idx
            # Weight: +1 for positive literal, -1 for negative (signed embedding)
            weight = 1.0 if lit > 0 else 0.5  # asymmetric to encode polarity
            adj[var_idx, clause_node] = weight
            adj[clause_node, var_idx] = weight

    # Graph Laplacian
    degree = np.diag(adj.sum(axis=1))
    L = degree - adj

    # Spectral embedding: use eigenvectors 1..d (skip the constant eigenvector)
    d = min(10, n_total - 2)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    embedding = eigenvectors[:, 1:d+1]  # skip first (constant)

    return embedding


def gpu_h0_persistence(points: np.ndarray, max_points: int = 500) -> np.ndarray:
    """H₀ persistence via GPU-accelerated pairwise distances + CPU Union-Find.

    GPU computes the distance matrix. CPU runs Union-Find (sequential but fast).
    """
    n = len(points)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
        points = points[idx]
        n = max_points

    # GPU pairwise distances
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts_t = torch.tensor(points, dtype=torch.float32, device=device)
    dists_t = torch.cdist(pts_t, pts_t)

    # Extract upper triangle on GPU, sort on GPU
    mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    flat_dists = dists_t[mask]
    sorted_dists, sorted_indices = torch.sort(flat_dists)

    # Map flat index back to (i,j) pairs
    rows, cols = torch.where(mask)
    sorted_i = rows[sorted_indices].cpu().numpy()
    sorted_j = cols[sorted_indices].cpu().numpy()
    sorted_d = sorted_dists.cpu().numpy()

    # Union-Find on CPU
    parent = list(range(n))
    rank_uf = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bars = []
    for k in range(len(sorted_d)):
        i, j, d = int(sorted_i[k]), int(sorted_j[k]), float(sorted_d[k])
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank_uf[ri] < rank_uf[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank_uf[ri] == rank_uf[rj]:
                rank_uf[ri] += 1
            bars.append(d)

    return np.array(bars)


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
    print("  P vs NP: SAT PHASE TRANSITION VIA TOPOLOGY")
    print(f"  No solver. No search. Just topology on GPU.")
    print(f"  {timestamp}")
    print("=" * 70)

    N_VARS = 50
    ALPHA_GRID = np.arange(2.0, 6.5, 0.25)
    N_INSTANCES = 20  # instances per alpha
    results = []

    total_t0 = time.time()

    for alpha in ALPHA_GRID:
        alpha_t0 = time.time()
        onset_scales = []
        gini_values = []

        for inst in range(N_INSTANCES):
            seed = int(alpha * 1000) + inst

            # Generate
            clauses = generate_random_3sat(N_VARS, alpha, seed)

            # Translate to point cloud
            points = sat_to_point_cloud(clauses, N_VARS)

            # GPU H₀ persistence
            bars = gpu_h0_persistence(points)

            if len(bars) > 0:
                onset = float(np.percentile(bars, 95))
                g = gini(bars)
            else:
                onset = 0.0
                g = 0.0

            onset_scales.append(onset)
            gini_values.append(g)

        mean_onset = float(np.mean(onset_scales))
        std_onset = float(np.std(onset_scales))
        mean_gini = float(np.mean(gini_values))
        elapsed = time.time() - alpha_t0

        results.append({
            "alpha": float(alpha),
            "n_instances": N_INSTANCES,
            "mean_onset": mean_onset,
            "std_onset": std_onset,
            "mean_gini": mean_gini,
            "elapsed_s": round(elapsed, 1),
        })

        print(f"  alpha={alpha:.2f}: onset={mean_onset:.4f}+/-{std_onset:.4f} "
              f"G={mean_gini:.4f} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_t0

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")

    alphas = np.array([r["alpha"] for r in results])
    onsets = np.array([r["mean_onset"] for r in results])
    ginis = np.array([r["mean_gini"] for r in results])

    # Find maximum derivative of onset scale (the transition)
    d_onset = np.gradient(onsets, alphas)
    max_deriv_idx = np.argmax(np.abs(d_onset))
    transition_alpha = alphas[max_deriv_idx]
    max_deriv = d_onset[max_deriv_idx]

    print(f"  Maximum |dε*/dα| at α = {transition_alpha:.2f} (derivative = {max_deriv:.4f})")
    print(f"  Known phase transition: α_c ≈ 4.27")
    print(f"  Distance from known: {abs(transition_alpha - 4.27):.2f}")

    detected = 3.5 <= transition_alpha <= 5.0
    verdict = "PASS" if detected else "FAIL"
    print(f"\n  VERDICT: {verdict}")

    # ── Figures ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.errorbar(alphas, onsets, yerr=[r["std_onset"] for r in results],
                 color=COLORS["teal"], linewidth=2, markersize=6, marker="o",
                 capsize=3, label="Mean onset ε*(α)")
    ax1.axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27 (known)")
    ax1.axvline(transition_alpha, color=COLORS["gold"], linestyle="--", alpha=0.7,
                label=f"ATFT transition: α = {transition_alpha:.2f}")
    ax1.set_xlabel("α (clause-to-variable ratio)")
    ax1.set_ylabel("ε* (topological onset scale)")
    ax1.set_title("SAT Phase Transition: Onset Scale", color=COLORS["gold"])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.15)

    ax2.plot(alphas, ginis, "-s", color=COLORS["gold"], linewidth=2, markersize=6)
    ax2.axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27")
    ax2.set_xlabel("α (clause-to-variable ratio)")
    ax2.set_ylabel("Gini coefficient")
    ax2.set_title("SAT Phase Transition: Gini Trajectory", color=COLORS["gold"])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.15)

    fig.suptitle(f"P vs NP: 3-SAT Topological Phase Transition (N={N_VARS}, {N_INSTANCES} instances/α)",
                 color=COLORS["text"], fontsize=14)
    fig.savefig(FIG_DIR / "sat_phase_transition.png")
    plt.close(fig)

    # ── Save ──
    output = {
        "experiment": "sat_topology",
        "n_vars": N_VARS,
        "n_instances_per_alpha": N_INSTANCES,
        "alpha_grid": ALPHA_GRID.tolist(),
        "results": results,
        "transition_alpha": float(transition_alpha),
        "max_derivative": float(max_deriv),
        "known_alpha_c": 4.27,
        "distance_from_known": float(abs(transition_alpha - 4.27)),
        "verdict": verdict,
        "total_time_s": round(total_elapsed, 1),
        "timestamp": timestamp,
    }

    with open(OUTPUT_DIR / "sat_topology.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results: {OUTPUT_DIR / 'sat_topology.json'}")
    print(f"  Figure: {FIG_DIR / 'sat_phase_transition.png'}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
