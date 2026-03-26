#!/usr/bin/env python3
"""P vs NP v3: Variable-Centric Constraint Geometry.

v1 FAIL: measured problem graph topology (monotone, wrong object)
v2 FAIL: random probes see 7/8 at all α (blind to correlations)

v3: measure CONSTRAINT PRESSURE on each variable — the local curvature.
For each variable i, compute:
  - positive_fraction: fraction of appearances where i is positive
  - degree: number of clauses containing i (normalized)
  - co-occurrence: mean number of shared clauses with other variables in same clauses
  - frustration: if i appears positive in clauses that share variables appearing negative

Each variable becomes a point in R^d (d = feature dimension).
The point cloud of variables should restructure at α_c.

At low α: variables are loosely constrained, spread in feature space
At α_c: constraint graph rigidifies, variables cluster into forced groups
At high α: over-determined, all variables heavily constrained and frustrated

This is the SAT analogue of action density at each lattice site.
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


def generate_3sat(n_vars: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    n_clauses = int(alpha * n_vars)
    var_indices = np.zeros((n_clauses, 3), dtype=np.int32)
    negations = np.zeros((n_clauses, 3), dtype=bool)
    for i in range(n_clauses):
        var_indices[i] = rng.choice(n_vars, size=3, replace=False)
        negations[i] = rng.random(3) < 0.5
    return var_indices, negations, n_clauses


def variable_feature_map(var_indices, negations, n_vars, n_clauses):
    """Compute per-variable constraint geometry features.

    Returns (n_vars, d) feature matrix where d=6:
      0: degree (normalized by max possible = n_clauses)
      1: positive_fraction (fraction of appearances as positive literal)
      2: mean_co_degree (mean clause-sharing count with co-occurring variables)
      3: polarity_tension (|positive - negative| / total, high = forced, low = frustrated)
      4: clause_density (fraction of clauses this var appears in)
      5: local_connectivity (number of distinct variables in shared clauses / n_vars)
    """
    # Per-variable statistics
    degree = np.zeros(n_vars)
    positive_count = np.zeros(n_vars)
    var_to_clauses = [set() for _ in range(n_vars)]

    for c_idx in range(n_clauses):
        for lit_idx in range(3):
            v = var_indices[c_idx, lit_idx]
            degree[v] += 1
            if not negations[c_idx, lit_idx]:
                positive_count[v] += 1
            var_to_clauses[v].add(c_idx)

    features = np.zeros((n_vars, 6))

    max_degree = max(degree.max(), 1)
    features[:, 0] = degree / max_degree  # normalized degree

    # Positive fraction (avoid div by 0)
    safe_degree = np.maximum(degree, 1)
    features[:, 1] = positive_count / safe_degree

    # Polarity tension: |pos - neg| / total (1 = forced, 0 = balanced/frustrated)
    neg_count = degree - positive_count
    features[:, 3] = np.abs(positive_count - neg_count) / safe_degree

    # Clause density
    features[:, 4] = degree / max(n_clauses, 1)

    # Co-occurrence statistics
    for v in range(n_vars):
        if not var_to_clauses[v]:
            continue
        co_vars = set()
        for c in var_to_clauses[v]:
            for lit_idx in range(3):
                other = var_indices[c, lit_idx]
                if other != v:
                    co_vars.add(other)

        # Mean co-degree: average clause-sharing with co-occurring variables
        if co_vars:
            co_degrees = []
            for u in co_vars:
                shared = len(var_to_clauses[v] & var_to_clauses[u])
                co_degrees.append(shared)
            features[v, 2] = np.mean(co_degrees) / max_degree
        features[v, 5] = len(co_vars) / max(n_vars - 1, 1)  # local connectivity

    return features


def gpu_persistence(points, max_points=400):
    n = len(points)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
        points = points[idx]
        n = max_points

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts = torch.tensor(points, dtype=torch.float32, device=device)
    dists = torch.cdist(pts, pts)

    mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    flat_dists = dists[mask]
    sorted_dists, sorted_indices = torch.sort(flat_dists)

    rows, cols = torch.where(mask)
    si = rows[sorted_indices].cpu().numpy()
    sj = cols[sorted_indices].cpu().numpy()
    sd = sorted_dists.cpu().numpy()

    parent = list(range(n))
    rank_uf = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bars = []
    for k in range(len(sd)):
        ri, rj = find(int(si[k])), find(int(sj[k]))
        if ri != rj:
            if rank_uf[ri] < rank_uf[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank_uf[ri] == rank_uf[rj]:
                rank_uf[ri] += 1
            bars.append(float(sd[k]))

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
    print("  P vs NP v3: VARIABLE-CENTRIC CONSTRAINT GEOMETRY")
    print("  Local curvature at each variable site.")
    print(f"  {timestamp}")
    print("=" * 70)

    N_VARS = 100
    ALPHA_GRID = np.arange(2.0, 7.5, 0.25)
    N_INSTANCES = 30
    results = []

    total_t0 = time.time()

    for alpha in ALPHA_GRID:
        alpha_t0 = time.time()
        onset_scales = []
        gini_values = []
        mean_tensions = []
        mean_connectivities = []

        for inst in range(N_INSTANCES):
            seed = int(alpha * 10000) + inst
            vi, neg, nc = generate_3sat(N_VARS, alpha, seed)

            # Variable feature map
            features = variable_feature_map(vi, neg, N_VARS, nc)

            # Record mean tension and connectivity
            mean_tensions.append(float(np.mean(features[:, 3])))
            mean_connectivities.append(float(np.mean(features[:, 5])))

            # H₀ persistence on variable point cloud
            bars = gpu_persistence(features)

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
        mean_tension = float(np.mean(mean_tensions))
        mean_conn = float(np.mean(mean_connectivities))
        elapsed = time.time() - alpha_t0

        results.append({
            "alpha": float(alpha),
            "mean_onset": mean_onset,
            "std_onset": std_onset,
            "mean_gini": mean_gini,
            "mean_tension": mean_tension,
            "mean_connectivity": mean_conn,
            "elapsed_s": round(elapsed, 1),
        })

        print(f"  α={alpha:.2f}: ε*={mean_onset:.4f}±{std_onset:.4f} "
              f"G={mean_gini:.4f} tension={mean_tension:.3f} "
              f"conn={mean_conn:.3f} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_t0

    # ── Analysis ──
    alphas = np.array([r["alpha"] for r in results])
    onsets = np.array([r["mean_onset"] for r in results])
    ginis = np.array([r["mean_gini"] for r in results])
    tensions = np.array([r["mean_tension"] for r in results])
    conns = np.array([r["mean_connectivity"] for r in results])

    # Derivative analysis
    d_onset = np.gradient(onsets, alphas)
    d_gini = np.gradient(ginis, alphas)
    d_tension = np.gradient(tensions, alphas)

    max_onset_idx = np.argmax(np.abs(d_onset))
    max_gini_idx = np.argmax(np.abs(d_gini))
    max_tension_idx = np.argmax(np.abs(d_tension))

    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")
    print(f"  ε* max |d/dα| at α = {alphas[max_onset_idx]:.2f}")
    print(f"  Gini max |d/dα| at α = {alphas[max_gini_idx]:.2f}")
    print(f"  Tension max |d/dα| at α = {alphas[max_tension_idx]:.2f}")
    print(f"  Known α_c ≈ 4.27")

    all_transitions = [alphas[max_onset_idx], alphas[max_gini_idx], alphas[max_tension_idx]]
    detected = any(3.5 <= t <= 5.5 for t in all_transitions)
    verdict = "PASS" if detected else "FAIL"
    print(f"  VERDICT: {verdict}")

    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax in axes.flat:
        ax.axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27")

    axes[0, 0].errorbar(alphas, onsets, yerr=[r["std_onset"] for r in results],
                         color=COLORS["teal"], linewidth=2, marker="o", markersize=4, capsize=2)
    axes[0, 0].set_ylabel("ε* (onset scale)")
    axes[0, 0].set_title("Topological Onset Scale", color=COLORS["teal"])
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.15)

    axes[0, 1].plot(alphas, ginis, "-s", color=COLORS["gold"], linewidth=2, markersize=4)
    axes[0, 1].set_ylabel("Gini")
    axes[0, 1].set_title("Persistence Hierarchy", color=COLORS["gold"])
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.15)

    axes[1, 0].plot(alphas, tensions, "-^", color=COLORS["purple"], linewidth=2, markersize=4)
    axes[1, 0].set_ylabel("Mean polarity tension")
    axes[1, 0].set_title("Variable Frustration", color=COLORS["purple"])
    axes[1, 0].set_xlabel("α")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.15)

    axes[1, 1].plot(alphas, conns, "-d", color=COLORS["red"], linewidth=2, markersize=4)
    axes[1, 1].set_ylabel("Mean local connectivity")
    axes[1, 1].set_title("Constraint Graph Density", color=COLORS["red"])
    axes[1, 1].set_xlabel("α")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.15)

    fig.suptitle(
        f"P vs NP v3: Variable Constraint Geometry (N={N_VARS}, {N_INSTANCES} inst/α)",
        color=COLORS["text"], fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sat_phase_transition_v3.png")
    plt.close(fig)

    # ── Save ──
    output = {
        "experiment": "sat_topology_v3",
        "feature_map": "variable_constraint_geometry",
        "n_vars": N_VARS,
        "n_instances_per_alpha": N_INSTANCES,
        "results": results,
        "transition_onset": float(alphas[max_onset_idx]),
        "transition_gini": float(alphas[max_gini_idx]),
        "transition_tension": float(alphas[max_tension_idx]),
        "verdict": verdict,
        "total_time_s": round(total_elapsed, 1),
        "timestamp": timestamp,
    }
    with open(OUTPUT_DIR / "sat_topology_v3.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Total: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
