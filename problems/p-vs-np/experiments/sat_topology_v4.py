#!/usr/bin/env python3
"""P vs NP v4: GPU-Parallel Greedy Walk Endpoints.

v1-v3 FAILED: all measured problem structure (monotone).
The transition is in the SOLUTION SPACE, not the problem graph.

v4: probe the solution landscape via parallel greedy walks on GPU.
  1. Start K=1000 random assignments in parallel (GPU tensor)
  2. Each walk: flip the variable that fixes the most unsatisfied clauses
  3. Run T=100 steps per walk (GPU batched argmax)
  4. The ENDPOINTS form the point cloud in Hamming space
  5. H₀ persistence on endpoints

Below α_c: walks converge to solutions → endpoints cluster → few H₀ components
Above α_c: walks get stuck at local minima → endpoints scatter → many components
AT α_c: the cluster fragments → onset scale discontinuity

This is walkSAT used as a PROBE, not a solver.
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


def generate_3sat_tensors(n_vars, alpha, seed, device="cuda"):
    """Generate 3-SAT as GPU tensors for batched evaluation."""
    rng = np.random.default_rng(seed)
    n_clauses = int(alpha * n_vars)
    var_indices = rng.choice(n_vars, size=(n_clauses, 3), replace=True)
    # Ensure no duplicate vars in same clause
    for i in range(n_clauses):
        while len(set(var_indices[i])) < 3:
            var_indices[i] = rng.choice(n_vars, size=3, replace=False)
    negations = rng.random((n_clauses, 3)) < 0.5

    vi = torch.tensor(var_indices, dtype=torch.long, device=device)
    neg = torch.tensor(negations, dtype=torch.float32, device=device)
    return vi, neg, n_clauses


def evaluate_clauses(assignments, vi, neg):
    """Evaluate all clauses for all assignments in parallel.

    assignments: (K, N) float tensor (0 or 1)
    vi: (m, 3) long tensor — variable indices per clause
    neg: (m, 3) float tensor — negation mask

    Returns: (K, m) float tensor — 1 if clause satisfied, 0 if not
    """
    lit_vals = assignments[:, vi]  # (K, m, 3)
    lit_vals = torch.abs(lit_vals - neg.unsqueeze(0))  # apply negation
    return lit_vals.max(dim=2).values  # (K, m) — clause satisfaction


def count_unsat_per_var(assignments, vi, neg, n_vars):
    """For each walk and each variable, count unsatisfied clauses it appears in.

    Returns: (K, N) tensor — number of unsatisfied clauses involving each variable
    """
    K = assignments.shape[0]
    clause_sat = evaluate_clauses(assignments, vi, neg)  # (K, m)
    unsat = 1.0 - clause_sat  # (K, m) — 1 if unsatisfied

    # For each variable, sum unsatisfied clauses containing it
    var_unsat = torch.zeros(K, n_vars, device=assignments.device)
    for lit_idx in range(3):
        # vi[:, lit_idx] gives the variable index for each clause's lit_idx-th literal
        # unsat[:, :] gives whether each clause is unsatisfied
        # We want to scatter-add unsat counts to the variable indices
        var_idx_expanded = vi[:, lit_idx].unsqueeze(0).expand(K, -1)  # (K, m)
        var_unsat.scatter_add_(1, var_idx_expanded, unsat)

    return var_unsat


def parallel_greedy_walks(n_vars, vi, neg, n_walks=500, n_steps=100, seed=42):
    """Run parallel greedy walks on GPU.

    Each walk starts from a random assignment and greedily flips the variable
    in the most unsatisfied clauses.

    Returns: endpoints (n_walks, n_vars) float tensor
    """
    device = vi.device
    rng = np.random.default_rng(seed)

    # Random starting assignments
    assignments = torch.tensor(
        rng.integers(0, 2, size=(n_walks, n_vars)),
        dtype=torch.float32, device=device
    )

    for step in range(n_steps):
        # Count unsatisfied clauses per variable per walk
        var_unsat = count_unsat_per_var(assignments, vi, neg, n_vars)

        # Greedy: flip the variable with most unsatisfied clauses
        flip_var = var_unsat.argmax(dim=1)  # (K,) — which var to flip per walk

        # Only flip if there are unsatisfied clauses
        max_unsat = var_unsat.max(dim=1).values
        should_flip = max_unsat > 0

        # Flip
        for k in range(n_walks):
            if should_flip[k]:
                v = flip_var[k].item()
                assignments[k, v] = 1.0 - assignments[k, v]

    return assignments


def endpoint_persistence(endpoints, max_points=400):
    """H₀ persistence on walk endpoints using Hamming distance."""
    n = endpoints.shape[0]
    device = endpoints.device

    if n > max_points:
        idx = torch.randperm(n, device=device)[:max_points]
        endpoints = endpoints[idx]
        n = max_points

    # Hamming distances on GPU
    dists = torch.cdist(endpoints.float(), endpoints.float(), p=1) / endpoints.shape[1]
    dists_np = dists.cpu().numpy()

    # Union-Find on CPU
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
    print("  P vs NP v4: GPU-PARALLEL GREEDY WALK ENDPOINTS")
    print("  Probe the solution landscape, not the problem structure.")
    print(f"  {timestamp}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    N_VARS = 100
    N_WALKS = 500
    N_STEPS = 200
    ALPHA_GRID = np.arange(2.0, 7.0, 0.25)
    N_INSTANCES = 20
    results = []

    total_t0 = time.time()

    for alpha in ALPHA_GRID:
        alpha_t0 = time.time()
        onset_scales = []
        gini_values = []
        mean_unsat_fracs = []

        for inst in range(N_INSTANCES):
            seed = int(alpha * 10000) + inst
            vi, neg, nc = generate_3sat_tensors(N_VARS, alpha, seed, device)

            # Run parallel greedy walks
            endpoints = parallel_greedy_walks(N_VARS, vi, neg,
                                              n_walks=N_WALKS, n_steps=N_STEPS,
                                              seed=seed + 77777)

            # Measure: how many clauses still unsatisfied at endpoints?
            clause_sat = evaluate_clauses(endpoints, vi, neg)
            unsat_frac = 1.0 - clause_sat.mean(dim=1)  # (K,) per-walk unsat fraction
            mean_unsat = float(unsat_frac.mean().item())
            mean_unsat_fracs.append(mean_unsat)

            # H₀ persistence on endpoints
            bars = endpoint_persistence(endpoints, max_points=300)

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
        mean_unsat = float(np.mean(mean_unsat_fracs))
        elapsed = time.time() - alpha_t0

        results.append({
            "alpha": float(alpha),
            "mean_onset": mean_onset,
            "std_onset": std_onset,
            "mean_gini": mean_gini,
            "mean_residual_unsat": mean_unsat,
            "elapsed_s": round(elapsed, 1),
        })

        print(f"  α={alpha:.2f}: ε*={mean_onset:.4f}±{std_onset:.4f} "
              f"G={mean_gini:.4f} unsat={mean_unsat:.4f} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_t0

    # ── Analysis ──
    alphas = np.array([r["alpha"] for r in results])
    onsets = np.array([r["mean_onset"] for r in results])
    ginis_arr = np.array([r["mean_gini"] for r in results])
    unsats = np.array([r["mean_residual_unsat"] for r in results])

    d_onset = np.gradient(onsets, alphas)
    d_unsat = np.gradient(unsats, alphas)

    max_onset_idx = np.argmax(np.abs(d_onset))
    max_unsat_idx = np.argmax(np.abs(d_unsat))

    # Find where residual unsat crosses from ~0 to >0
    first_unsat_idx = next((i for i in range(len(unsats)) if unsats[i] > 0.005), len(unsats) - 1)
    sat_transition = alphas[first_unsat_idx]

    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")
    print(f"  ε* max |d/dα| at α = {alphas[max_onset_idx]:.2f}")
    print(f"  Residual unsat > 0 first at α = {sat_transition:.2f}")
    print(f"  Residual unsat max |d/dα| at α = {alphas[max_unsat_idx]:.2f}")
    print(f"  Known α_c ≈ 4.27")

    detected = any(3.5 <= t <= 5.5 for t in [alphas[max_onset_idx], sat_transition, alphas[max_unsat_idx]])
    verdict = "PASS" if detected else "FAIL"
    print(f"  VERDICT: {verdict}")

    # ── Figures ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].errorbar(alphas, onsets, yerr=[r["std_onset"] for r in results],
                     color=COLORS["teal"], linewidth=2, marker="o", markersize=5, capsize=2)
    axes[0].axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27")
    axes[0].set_xlabel("α"); axes[0].set_ylabel("ε*")
    axes[0].set_title("Walk Endpoint Onset Scale", color=COLORS["teal"])
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.15)

    axes[1].plot(alphas, ginis_arr, "-s", color=COLORS["gold"], linewidth=2, markersize=5)
    axes[1].axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27")
    axes[1].set_xlabel("α"); axes[1].set_ylabel("Gini")
    axes[1].set_title("Endpoint Persistence Hierarchy", color=COLORS["gold"])
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.15)

    axes[2].plot(alphas, unsats, "-^", color=COLORS["red"], linewidth=2, markersize=5)
    axes[2].axvline(4.27, color=COLORS["red"], linestyle="--", alpha=0.7, label="α_c ≈ 4.27")
    axes[2].set_xlabel("α"); axes[2].set_ylabel("Mean residual unsat fraction")
    axes[2].set_title("Greedy Walk Residual", color=COLORS["red"])
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.15)

    fig.suptitle(f"P vs NP v4: Greedy Walk Endpoints (N={N_VARS}, {N_WALKS} walks × {N_STEPS} steps)",
                 color=COLORS["text"], fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sat_phase_transition_v4.png")
    plt.close(fig)

    output = {
        "experiment": "sat_topology_v4",
        "feature_map": "greedy_walk_endpoints",
        "n_vars": N_VARS, "n_walks": N_WALKS, "n_steps": N_STEPS,
        "n_instances_per_alpha": N_INSTANCES,
        "results": results,
        "sat_transition_alpha": float(sat_transition),
        "verdict": verdict,
        "total_time_s": round(total_elapsed, 1),
    }
    with open(OUTPUT_DIR / "sat_topology_v4.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Total: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
