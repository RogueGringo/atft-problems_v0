#!/usr/bin/env python3
"""P vs NP v2: Random Probe Feature Map.

v1 FAILED because it measured the topology of the PROBLEM (clause-variable graph).
The graph grows monotonically with α. Nothing to detect.

v2 measures the topology of the SOLUTION SPACE by probing it:
1. Generate K random assignments (probes)
2. For each probe: compute clause satisfaction pattern → binary vector in {0,1}^m
3. Point cloud = K points in R^m (satisfaction patterns)
4. Metric = Hamming distance between patterns
5. H₀ persistence detects when the solution space FRAGMENTS

Below α_c: probes cluster (many satisfy most clauses) → one component
Above α_c: probes spread (nothing satisfies everything) → no structure
AT α_c: cluster shatters → onset scale discontinuity

GPU: assignments × incidence = satisfaction patterns. Batched matmul.
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
    """Generate random 3-SAT. Returns (clauses, var_indices, negations).

    clauses: (n_clauses, 3) array of variable indices (0-based)
    negations: (n_clauses, 3) boolean array (True = negated)
    """
    rng = np.random.default_rng(seed)
    n_clauses = int(alpha * n_vars)
    var_indices = np.zeros((n_clauses, 3), dtype=np.int32)
    negations = np.zeros((n_clauses, 3), dtype=bool)
    for i in range(n_clauses):
        var_indices[i] = rng.choice(n_vars, size=3, replace=False)
        negations[i] = rng.random(3) < 0.5
    return var_indices, negations, n_clauses


def probe_solution_space_gpu(var_indices, negations, n_vars, n_clauses,
                              n_probes=500, seed=42):
    """GPU-accelerated random probing of the solution space.

    Generates n_probes random assignments, evaluates all clauses in parallel.
    Returns: satisfaction_matrix (n_probes, n_clauses) — fraction of literals
    satisfied in each clause by each probe.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    # Random assignments: (n_probes, n_vars) boolean
    assignments = torch.tensor(
        rng.integers(0, 2, size=(n_probes, n_vars)),
        dtype=torch.float32, device=device
    )

    # Clause evaluation on GPU
    # For each clause c with variables (v1, v2, v3) and negations (n1, n2, n3):
    # literal_i = assignment[v_i] XOR n_i
    # clause satisfied = OR(literal_1, literal_2, literal_3)

    vi = torch.tensor(var_indices, dtype=torch.long, device=device)  # (m, 3)
    neg = torch.tensor(negations, dtype=torch.float32, device=device)  # (m, 3)

    # Gather variable values for all probes × all clauses × 3 literals
    # assignments: (K, N), vi: (m, 3) → need (K, m, 3)
    lit_vals = assignments[:, vi]  # (K, m, 3) — variable values at clause positions

    # Apply negation: XOR with negation mask
    lit_vals = torch.abs(lit_vals - neg.unsqueeze(0))  # (K, m, 3) — literal values

    # Clause satisfaction: OR over 3 literals = max
    clause_sat = lit_vals.max(dim=2).values  # (K, m) — 1 if clause satisfied, 0 if not

    # Fraction of clauses satisfied per probe
    frac_sat = clause_sat.mean(dim=1)  # (K,) — overall satisfaction fraction

    # The point cloud: each probe is a point in R^m (clause satisfaction pattern)
    # Use the continuous values (fraction of literals true per clause) for richer metric
    # Actually, use the satisfaction PATTERN as the point cloud
    return clause_sat.cpu().numpy(), frac_sat.cpu().numpy()


def hamming_persistence(patterns, max_points=400):
    """H₀ persistence using Hamming distance on binary satisfaction patterns."""
    n = len(patterns)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
        patterns = patterns[idx]
        n = max_points

    # GPU Hamming distances (binary → sum of XOR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = torch.tensor(patterns, dtype=torch.float32, device=device)
    # Hamming = number of positions that differ / total positions
    # For binary vectors: hamming(a,b) = mean(|a-b|)
    dists = torch.cdist(p, p, p=1) / p.shape[1]  # L1 normalized = Hamming
    dists = dists.cpu().numpy()

    # Union-Find
    parent = list(range(n))
    rank_uf = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Sort all pairs by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dists[i, j], i, j))
    edges.sort()

    bars = []
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank_uf[ri] < rank_uf[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank_uf[ri] == rank_uf[rj]:
                rank_uf[ri] += 1
            bars.append(float(d))

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
    print("  P vs NP v2: RANDOM PROBE FEATURE MAP")
    print("  Probing solution space topology. No solver.")
    print(f"  {timestamp}")
    print("=" * 70)

    N_VARS = 80
    N_PROBES = 500
    ALPHA_GRID = np.arange(2.0, 7.0, 0.25)
    N_INSTANCES = 20
    results = []

    total_t0 = time.time()

    for alpha in ALPHA_GRID:
        alpha_t0 = time.time()
        onset_scales = []
        gini_values = []
        mean_sats = []

        for inst in range(N_INSTANCES):
            seed = int(alpha * 10000) + inst

            vi, neg, n_clauses = generate_3sat(N_VARS, alpha, seed)
            patterns, frac_sat = probe_solution_space_gpu(
                vi, neg, N_VARS, n_clauses, n_probes=N_PROBES, seed=seed + 99999
            )

            mean_sats.append(float(np.mean(frac_sat)))

            bars = hamming_persistence(patterns, max_points=300)

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
        mean_sat_frac = float(np.mean(mean_sats))
        elapsed = time.time() - alpha_t0

        results.append({
            "alpha": float(alpha),
            "mean_onset": mean_onset,
            "std_onset": std_onset,
            "mean_gini": mean_gini,
            "mean_sat_fraction": mean_sat_frac,
            "elapsed_s": round(elapsed, 1),
        })

        print(f"  α={alpha:.2f}: onset={mean_onset:.4f}±{std_onset:.4f} "
              f"G={mean_gini:.4f} sat={mean_sat_frac:.3f} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_t0

    # ── Analysis ──
    alphas = np.array([r["alpha"] for r in results])
    onsets = np.array([r["mean_onset"] for r in results])
    ginis = np.array([r["mean_gini"] for r in results])
    sat_fracs = np.array([r["mean_sat_fraction"] for r in results])

    # Derivative analysis
    d_onset = np.gradient(onsets, alphas)
    d_gini = np.gradient(ginis, alphas)

    max_onset_deriv_idx = np.argmax(np.abs(d_onset))
    max_gini_deriv_idx = np.argmax(np.abs(d_gini))

    transition_onset = alphas[max_onset_deriv_idx]
    transition_gini = alphas[max_gini_deriv_idx]

    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")
    print(f"  Onset ε* max |derivative| at α = {transition_onset:.2f}")
    print(f"  Gini max |derivative| at α = {transition_gini:.2f}")
    print(f"  Known α_c ≈ 4.27")
    print(f"  Satisfaction fraction crosses 0.875 (7/8) at α ≈ "
          f"{alphas[np.argmin(np.abs(sat_fracs - 0.875))]:.2f}")

    # Detect transition in onset, gini, or sat fraction
    detected = any(3.5 <= t <= 5.5 for t in [transition_onset, transition_gini])
    verdict = "PASS" if detected else "FAIL"
    print(f"\n  VERDICT: {verdict}")

    # ── Figures ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.errorbar(alphas, onsets, yerr=[r["std_onset"] for r in results],
                color=COLORS["teal"], linewidth=2, marker="o", markersize=5, capsize=2)
    ax.axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27")
    ax.set_xlabel("α (clause/variable ratio)")
    ax.set_ylabel("ε* (onset scale)")
    ax.set_title("Solution Space Onset Scale", color=COLORS["teal"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    ax.plot(alphas, ginis, "-s", color=COLORS["gold"], linewidth=2, markersize=5)
    ax.axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27")
    ax.set_xlabel("α")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Persistence Hierarchy", color=COLORS["gold"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    ax = axes[2]
    ax.plot(alphas, sat_fracs, "-^", color=COLORS["red"], linewidth=2, markersize=5)
    ax.axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27")
    ax.axhline(7/8, color=COLORS["muted"], linestyle=":", alpha=0.5, label="7/8 (random baseline)")
    ax.set_xlabel("α")
    ax.set_ylabel("Mean clause satisfaction")
    ax.set_title("Random Probe Satisfaction", color=COLORS["red"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"P vs NP v2: Solution Space Probing (N={N_VARS}, {N_PROBES} probes, {N_INSTANCES} inst/α)",
        color=COLORS["text"], fontsize=14)
    fig.savefig(FIG_DIR / "sat_phase_transition_v2.png")
    plt.close(fig)

    # ── Save ──
    output = {
        "experiment": "sat_topology_v2",
        "feature_map": "random_probe_satisfaction_pattern",
        "n_vars": N_VARS, "n_probes": N_PROBES,
        "n_instances_per_alpha": N_INSTANCES,
        "results": results,
        "transition_onset_alpha": float(transition_onset),
        "transition_gini_alpha": float(transition_gini),
        "verdict": verdict,
        "total_time_s": round(total_elapsed, 1),
        "timestamp": timestamp,
    }
    with open(OUTPUT_DIR / "sat_topology_v2.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Total: {total_elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
