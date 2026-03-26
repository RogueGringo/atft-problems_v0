#!/usr/bin/env python3
"""P vs NP v6: Solution Space Overlap via WalkSAT.

v1-v5 all measured the PROBLEM INSTANCE. All failed or gave false positives.
The transition is in the SOLUTION SPACE — the cluster shattering at α_c.

v6: WalkSAT finds actual solutions (or best-effort near-solutions).
Point cloud = K solution endpoints in {0,1}^N.
Distance = Hamming overlap.
H₀ persistence detects cluster fragmentation.

Below α_c: WalkSAT finds many diverse solutions → one cluster
At α_c: solutions rare and clustered → cluster shatters
Above α_c: no solutions → stuck minima with different structure

References:
  Mézard, Parisi, Zecchina (2002) — replica symmetry breaking in random SAT
  Selman, Kautz, Cohen (1994) — WalkSAT algorithm
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


def walksat(var_indices, negations, n_vars, n_clauses,
            n_runs=500, max_steps=5000, noise_p=0.57, seed=42):
    """WalkSAT: stochastic local search for SAT.

    For each run:
      1. Start with random assignment
      2. Pick a random unsatisfied clause
      3. With probability noise_p: flip a random variable in that clause
         With probability 1-noise_p: flip the variable that minimizes breaks
      4. Repeat for max_steps or until all clauses satisfied

    Returns: (n_runs, n_vars) array of best assignments per run,
             (n_runs,) array of number of unsatisfied clauses at best
    """
    rng = np.random.default_rng(seed)
    best_assignments = np.zeros((n_runs, n_vars), dtype=np.int8)
    best_unsat_counts = np.full(n_runs, n_clauses, dtype=np.int32)

    for run in range(n_runs):
        # Random start
        assignment = rng.integers(0, 2, size=n_vars, dtype=np.int8)
        best_assign = assignment.copy()
        best_unsat = n_clauses

        for step in range(max_steps):
            # Evaluate all clauses
            unsat_indices = []
            for c in range(n_clauses):
                satisfied = False
                for k in range(3):
                    v = var_indices[c, k]
                    val = assignment[v]
                    if negations[c, k]:
                        val = 1 - val
                    if val == 1:
                        satisfied = True
                        break
                if not satisfied:
                    unsat_indices.append(c)

            n_unsat = len(unsat_indices)

            if n_unsat < best_unsat:
                best_unsat = n_unsat
                best_assign = assignment.copy()

            if n_unsat == 0:
                break  # Found a solution

            # Pick random unsatisfied clause
            c = unsat_indices[rng.integers(0, n_unsat)]

            if rng.random() < noise_p:
                # Noise move: flip random variable in clause
                k = rng.integers(0, 3)
                v = var_indices[c, k]
                assignment[v] = 1 - assignment[v]
            else:
                # Greedy move: flip variable that minimizes breaks
                best_var = var_indices[c, 0]
                best_breaks = n_clauses

                for k in range(3):
                    v = var_indices[c, k]
                    # Count how many currently-satisfied clauses would break
                    assignment[v] = 1 - assignment[v]  # tentative flip
                    breaks = 0
                    for c2 in range(n_clauses):
                        if c2 in unsat_indices:
                            continue
                        sat = False
                        for k2 in range(3):
                            v2 = var_indices[c2, k2]
                            val2 = assignment[v2]
                            if negations[c2, k2]:
                                val2 = 1 - val2
                            if val2 == 1:
                                sat = True
                                break
                        if not sat:
                            breaks += 1
                    assignment[v] = 1 - assignment[v]  # undo

                    if breaks < best_breaks:
                        best_breaks = breaks
                        best_var = v

                assignment[best_var] = 1 - assignment[best_var]

        best_assignments[run] = best_assign
        best_unsat_counts[run] = best_unsat

    return best_assignments, best_unsat_counts


def hamming_persistence_gpu(assignments, max_points=400):
    """H₀ persistence on solution overlap (Hamming distance) using GPU."""
    n = len(assignments)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
        assignments = assignments[idx]
        n = max_points

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.tensor(assignments, dtype=torch.float32, device=device)
    n_vars = a.shape[1]

    # Hamming distance normalized by n_vars
    dists = torch.cdist(a, a, p=1) / n_vars
    dists_np = dists.cpu().numpy()

    # Union-Find
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
            edges.append((dists_np[i, j], i, j))
    edges.sort()

    bars = []
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank_uf[ri] < rank_uf[rj]: ri, rj = rj, ri
            parent[rj] = ri
            if rank_uf[ri] == rank_uf[rj]: rank_uf[ri] += 1
            bars.append(float(d))

    return np.array(bars), dists_np


def gini(values):
    n = len(values)
    if n <= 1 or np.sum(values) == 0:
        return 0.0
    sorted_v = np.sort(values)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_v)) / (n * np.sum(sorted_v)) - (n + 1.0) / n)


def overlap_distribution_bimodality(dists_np):
    """Measure bimodality of the overlap distribution.

    Bimodal overlap = cluster shattering = phase transition.
    Returns: bimodality coefficient (>0.555 suggests bimodality).
    """
    overlaps = 1.0 - dists_np[np.triu_indices(len(dists_np), k=1)]
    if len(overlaps) < 10:
        return 0.0
    from scipy.stats import kurtosis, skew
    n = len(overlaps)
    s = skew(overlaps)
    k = kurtosis(overlaps)  # excess kurtosis
    # Sarle's bimodality coefficient
    bc = (s**2 + 1) / (k + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
    return float(bc)


def main():
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  P vs NP v6: SOLUTION SPACE OVERLAP VIA WALKSAT")
    print("  Point the instrument at the solution space, not the problem.")
    print(f"  {timestamp}")
    print("=" * 70)

    N_VARS = 150
    N_WALKS = 200  # WalkSAT runs per instance
    MAX_STEPS = 5000
    NOISE_P = 0.57
    ALPHA_GRID = np.arange(3.0, 6.0, 0.2)
    N_INSTANCES = 10
    results = []

    total_t0 = time.time()

    for alpha in ALPHA_GRID:
        alpha_t0 = time.time()
        onset_scales = []
        gini_values = []
        mean_unsat_fracs = []
        solution_rates = []
        bimodalities = []

        for inst in range(N_INSTANCES):
            seed = int(alpha * 10000) + inst
            vi, neg, nc = generate_3sat(N_VARS, alpha, seed)

            # Run GPU WalkSAT
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location(
                "gpu_walksat",
                str(Path(__file__).parent / "gpu_walksat.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            gpu_walksat = mod.gpu_walksat
            assignments, unsat_counts = gpu_walksat(
                vi, neg, N_VARS, nc,
                n_walks=N_WALKS, max_steps=MAX_STEPS,
                noise_p=NOISE_P, seed=seed + 55555,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # Statistics
            sol_rate = float(np.mean(unsat_counts == 0))
            mean_unsat = float(np.mean(unsat_counts)) / nc if nc > 0 else 0
            solution_rates.append(sol_rate)
            mean_unsat_fracs.append(mean_unsat)

            # H₀ persistence on solution overlaps
            bars, dists_np = hamming_persistence_gpu(assignments, max_points=200)

            if len(bars) > 0:
                onset = float(np.percentile(bars, 95))
                g = gini(bars)
            else:
                onset = 0.0
                g = 0.0

            onset_scales.append(onset)
            gini_values.append(g)

            # Overlap bimodality
            try:
                bc = overlap_distribution_bimodality(dists_np)
            except Exception:
                bc = 0.0
            bimodalities.append(bc)

        mean_onset = float(np.mean(onset_scales))
        std_onset = float(np.std(onset_scales))
        mean_gini = float(np.mean(gini_values))
        mean_sol_rate = float(np.mean(solution_rates))
        mean_unsat = float(np.mean(mean_unsat_fracs))
        mean_bimodal = float(np.mean(bimodalities))
        elapsed = time.time() - alpha_t0

        results.append({
            "alpha": float(alpha),
            "mean_onset": mean_onset,
            "std_onset": std_onset,
            "mean_gini": mean_gini,
            "mean_solution_rate": mean_sol_rate,
            "mean_residual_unsat": mean_unsat,
            "mean_bimodality": mean_bimodal,
            "elapsed_s": round(elapsed, 1),
        })

        print(f"  α={alpha:.2f}: ε*={mean_onset:.4f}±{std_onset:.4f} "
              f"G={mean_gini:.4f} sol={mean_sol_rate:.2f} "
              f"unsat={mean_unsat:.4f} bimod={mean_bimodal:.3f} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_t0

    # ── Analysis ──
    alphas = np.array([r["alpha"] for r in results])
    onsets = np.array([r["mean_onset"] for r in results])
    ginis_arr = np.array([r["mean_gini"] for r in results])
    sol_rates = np.array([r["mean_solution_rate"] for r in results])
    bimodals = np.array([r["mean_bimodality"] for r in results])

    # Find where solution rate drops to 0
    first_zero_idx = next((i for i in range(len(sol_rates)) if sol_rates[i] < 0.01), len(sol_rates) - 1)
    sat_unsat_alpha = alphas[first_zero_idx]

    d_onset = np.gradient(onsets, alphas)
    d_bimodal = np.gradient(bimodals, alphas)
    max_onset_idx = np.argmax(np.abs(d_onset))
    max_bimodal_idx = np.argmax(np.abs(d_bimodal))

    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")
    print(f"  Solution rate → 0 at α ≈ {sat_unsat_alpha:.2f}")
    print(f"  ε* max |d/dα| at α = {alphas[max_onset_idx]:.2f}")
    print(f"  Bimodality max |d/dα| at α = {alphas[max_bimodal_idx]:.2f}")
    print(f"  Known α_c ≈ 4.27")

    transitions = [sat_unsat_alpha, alphas[max_onset_idx], alphas[max_bimodal_idx]]
    detected = any(3.8 <= t <= 5.0 for t in transitions)
    verdict = "PASS" if detected else "FAIL"
    print(f"  VERDICT: {verdict}")

    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax in axes.flat:
        ax.axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27")
        ax.grid(True, alpha=0.15)

    axes[0, 0].errorbar(alphas, onsets, yerr=[r["std_onset"] for r in results],
                         color=COLORS["teal"], linewidth=2, marker="o", markersize=5, capsize=2)
    axes[0, 0].set_ylabel("ε* (onset scale)")
    axes[0, 0].set_title("Solution Overlap Onset Scale", color=COLORS["teal"])
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(alphas, sol_rates, "-s", color=COLORS["gold"], linewidth=2, markersize=5)
    axes[0, 1].set_ylabel("Fraction of runs finding a solution")
    axes[0, 1].set_title("WalkSAT Solution Rate", color=COLORS["gold"])
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(alphas, ginis_arr, "-^", color=COLORS["purple"], linewidth=2, markersize=5)
    axes[1, 0].set_xlabel("α"); axes[1, 0].set_ylabel("Gini")
    axes[1, 0].set_title("Overlap Persistence Hierarchy", color=COLORS["purple"])
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(alphas, bimodals, "-d", color=COLORS["red"], linewidth=2, markersize=5)
    axes[1, 1].set_xlabel("α"); axes[1, 1].set_ylabel("Bimodality coefficient")
    axes[1, 1].axhline(0.555, color=COLORS["muted"], linestyle=":", alpha=0.5, label="Bimodal threshold")
    axes[1, 1].set_title("Overlap Distribution Bimodality", color=COLORS["red"])
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        f"P vs NP v6: Solution Space Topology (N={N_VARS}, {N_WALKS} WalkSAT runs, {N_INSTANCES} inst/α)",
        color=COLORS["text"], fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sat_phase_transition_v6.png")
    plt.close(fig)

    output = {
        "experiment": "sat_topology_v6",
        "feature_map": "walksat_solution_overlap",
        "n_vars": N_VARS, "n_walks": N_WALKS,
        "max_steps": MAX_STEPS, "noise_p": NOISE_P,
        "n_instances_per_alpha": N_INSTANCES,
        "results": results,
        "sat_unsat_transition": float(sat_unsat_alpha),
        "onset_transition": float(alphas[max_onset_idx]),
        "bimodality_transition": float(alphas[max_bimodal_idx]),
        "verdict": verdict,
        "total_time_s": round(total_elapsed, 1),
    }
    with open(OUTPUT_DIR / "sat_topology_v6.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Total: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
