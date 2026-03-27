#!/usr/bin/env python3
"""SAT Hardness API — Proof of Concept.

THE PRODUCT TEST: Does the topological Gini score predict actual solve time?

Protocol:
1. Generate 500 random 3-SAT instances across α = [3.0, 5.5]
2. For each instance:
   a. Compute topological score (GPU WalkSAT → overlap persistence → Gini)
   b. Solve with pysat (Glucose4) — measure wall clock time
3. Correlate: Gini score vs solve time

If Pearson r > 0.7: THE PRODUCT IS VIABLE.
If r < 0.3: back to research.
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
from scipy.stats import pearsonr, spearmanr

COLORS = {"gold": "#c5a03f", "teal": "#45a8b0", "red": "#e94560",
          "bg": "#0f0d08", "text": "#d6d0be", "muted": "#817a66"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.facecolor": COLORS["bg"], "figure.facecolor": COLORS["bg"],
    "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "axes.edgecolor": COLORS["muted"], "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})

OUTPUT_DIR = Path("products/sat-hardness-api/results")
FIG_DIR = Path("products/sat-hardness-api/results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_3sat_dimacs(n_vars, alpha, seed):
    """Generate random 3-SAT and return both numpy arrays and DIMACS clauses."""
    rng = np.random.default_rng(seed)
    n_clauses = int(alpha * n_vars)
    clauses_np = np.zeros((n_clauses, 3), dtype=np.int32)
    negations = np.zeros((n_clauses, 3), dtype=bool)
    dimacs_clauses = []

    for i in range(n_clauses):
        vars_chosen = rng.choice(n_vars, size=3, replace=False)
        negs = rng.random(3) < 0.5
        clauses_np[i] = vars_chosen
        negations[i] = negs

        # DIMACS format: positive var = var+1, negative = -(var+1)
        clause = []
        for v, n in zip(vars_chosen, negs):
            lit = (v + 1) if not n else -(v + 1)
            clause.append(int(lit))
        dimacs_clauses.append(clause)

    return clauses_np, negations, n_clauses, dimacs_clauses


def solve_with_pysat(dimacs_clauses, n_vars, timeout=30.0):
    """Solve with Glucose4 via pysat. Returns (sat, solve_time_ms)."""
    from pysat.solvers import Glucose4

    solver = Glucose4()
    for clause in dimacs_clauses:
        solver.add_clause(clause)

    t0 = time.time()
    result = solver.solve()
    solve_time = (time.time() - t0) * 1000  # ms

    solver.delete()
    return result, solve_time


def compute_gini_score(clauses_np, negations, n_vars, n_clauses,
                        n_walks=200, max_steps=3000, seed=42):
    """Compute topological Gini score via GPU WalkSAT + overlap persistence."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gw", str(Path(__file__).parent.parent.parent /
                   "problems/p-vs-np/experiments/gpu_walksat.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assignments, unsat = mod.gpu_walksat(
        clauses_np, negations, n_vars, n_clauses,
        n_walks=n_walks, max_steps=max_steps,
        noise_p=0.57, seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # H₀ persistence on overlap
    n = min(len(assignments), 200)
    if len(assignments) > n:
        idx = np.random.default_rng(42).choice(len(assignments), n, replace=False)
        assignments = assignments[idx]

    a = torch.tensor(assignments, dtype=torch.float32, device="cuda")
    dists = (torch.cdist(a, a, p=1) / a.shape[1]).cpu().numpy()

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = sorted([(dists[i, j], i, j) for i in range(n) for j in range(i + 1, n)])
    bars = []
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank[ri] < rank[rj]: ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]: rank[ri] += 1
            bars.append(d)

    bars = np.array(bars) if bars else np.array([0.0])

    # Gini
    nn = len(bars)
    if nn <= 1 or np.sum(bars) == 0:
        return 0.0, float(np.mean(unsat))
    sv = np.sort(bars)
    idx_arr = np.arange(1, nn + 1, dtype=np.float64)
    gini = float((2 * np.sum(idx_arr * sv)) / (nn * np.sum(sv)) - (nn + 1) / nn)

    return gini, float(np.mean(unsat))


def main():
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  SAT HARDNESS API — PRODUCT VIABILITY TEST")
    print("  Does Gini predict solve time?")
    print(f"  {timestamp}")
    print("=" * 70)

    N_VARS = 100
    N_INSTANCES = 300
    ALPHA_RANGE = (3.0, 5.5)

    rng = np.random.default_rng(42)
    alphas = rng.uniform(*ALPHA_RANGE, size=N_INSTANCES)

    results = []
    total_t0 = time.time()

    for i, alpha in enumerate(alphas):
        seed = 100000 + i
        vi, neg, nc, dimacs = generate_3sat_dimacs(N_VARS, alpha, seed)

        # Topology score
        t0 = time.time()
        gini_score, mean_unsat = compute_gini_score(vi, neg, N_VARS, nc, seed=seed + 99)
        topo_time = (time.time() - t0) * 1000

        # Actual solve
        sat, solve_time = solve_with_pysat(dimacs, N_VARS, timeout=30.0)

        results.append({
            "instance": i,
            "alpha": float(alpha),
            "gini": gini_score,
            "mean_unsat": mean_unsat,
            "sat": sat,
            "solve_time_ms": solve_time,
            "topo_time_ms": topo_time,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_INSTANCES}: α={alpha:.2f} G={gini_score:.4f} "
                  f"solve={solve_time:.1f}ms sat={sat} topo={topo_time:.0f}ms")

    total_elapsed = time.time() - total_t0

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("  CORRELATION ANALYSIS")
    print(f"{'='*70}")

    ginis = np.array([r["gini"] for r in results])
    solve_times = np.array([r["solve_time_ms"] for r in results])
    alphas_arr = np.array([r["alpha"] for r in results])
    sats = np.array([r["sat"] for r in results])

    # Filter to instances that took non-trivial time
    # (very easy instances solve in <1ms regardless of topology)
    mask = solve_times > 0.1
    g_filt = ginis[mask]
    s_filt = solve_times[mask]

    if len(g_filt) > 10:
        r_pearson, p_pearson = pearsonr(g_filt, np.log1p(s_filt))
        r_spearman, p_spearman = spearmanr(g_filt, s_filt)
    else:
        r_pearson, p_pearson = 0, 1
        r_spearman, p_spearman = 0, 1

    # Also correlate alpha with solve time (baseline)
    r_alpha, p_alpha = pearsonr(alphas_arr[mask], np.log1p(s_filt))

    print(f"  N instances: {N_INSTANCES} ({mask.sum()} with solve_time > 0.1ms)")
    print(f"  Gini vs log(solve_time): Pearson r = {r_pearson:.4f} (p={p_pearson:.2e})")
    print(f"  Gini vs solve_time: Spearman ρ = {r_spearman:.4f} (p={p_spearman:.2e})")
    print(f"  Alpha vs log(solve_time): Pearson r = {r_alpha:.4f} (baseline)")
    print(f"  SAT fraction: {sats.mean():.2f}")
    print(f"  Mean topo time: {np.mean([r['topo_time_ms'] for r in results]):.0f}ms")
    print(f"  Mean solve time: {np.mean(solve_times):.1f}ms")

    if abs(r_spearman) > 0.7:
        verdict = "PRODUCT VIABLE — Gini predicts hardness"
    elif abs(r_spearman) > 0.4:
        verdict = "PROMISING — moderate correlation, needs refinement"
    elif abs(r_spearman) > 0.2:
        verdict = "WEAK — topology sees something but not enough for product"
    else:
        verdict = "NO CORRELATION — back to research"

    print(f"\n  VERDICT: {verdict}")

    # ── Figures ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Gini vs solve time
    ax = axes[0]
    colors = [COLORS["teal"] if s else COLORS["red"] for s in sats]
    ax.scatter(ginis, np.log1p(solve_times), c=colors, s=15, alpha=0.6)
    ax.set_xlabel("Topological Gini Score")
    ax.set_ylabel("log(1 + solve_time_ms)")
    ax.set_title(f"Gini vs Solve Time (r={r_pearson:.3f})", color=COLORS["gold"])
    ax.grid(True, alpha=0.15)

    # Alpha vs solve time (baseline)
    ax = axes[1]
    ax.scatter(alphas_arr, np.log1p(solve_times), c=colors, s=15, alpha=0.6)
    ax.set_xlabel("α (clause/variable ratio)")
    ax.set_ylabel("log(1 + solve_time_ms)")
    ax.set_title(f"α vs Solve Time (r={r_alpha:.3f})", color=COLORS["teal"])
    ax.axvline(4.27, color=COLORS["red"], ls="--", alpha=0.5, label="α_c")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # Gini distribution by SAT/UNSAT
    ax = axes[2]
    sat_ginis = ginis[sats]
    unsat_ginis = ginis[~sats]
    if len(sat_ginis) > 0:
        ax.hist(sat_ginis, bins=30, alpha=0.7, color=COLORS["teal"], label=f"SAT (n={len(sat_ginis)})")
    if len(unsat_ginis) > 0:
        ax.hist(unsat_ginis, bins=30, alpha=0.7, color=COLORS["red"], label=f"UNSAT (n={len(unsat_ginis)})")
    ax.set_xlabel("Gini Score")
    ax.set_ylabel("Count")
    ax.set_title("Gini Distribution by Satisfiability", color=COLORS["gold"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    fig.suptitle(f"SAT Hardness API: Product Viability (N={N_VARS}, {N_INSTANCES} instances)",
                 color=COLORS["text"], fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hardness_correlation.png")
    plt.close(fig)

    # Save
    output = {
        "experiment": "hardness_correlation",
        "n_vars": N_VARS, "n_instances": N_INSTANCES,
        "pearson_r_gini_solve": float(r_pearson),
        "spearman_r_gini_solve": float(r_spearman),
        "pearson_r_alpha_solve": float(r_alpha),
        "verdict": verdict,
        "sat_fraction": float(sats.mean()),
        "mean_topo_time_ms": float(np.mean([r["topo_time_ms"] for r in results])),
        "mean_solve_time_ms": float(np.mean(solve_times)),
        "total_time_s": round(total_elapsed, 1),
    }
    with open(OUTPUT_DIR / "hardness_correlation.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Total: {total_elapsed:.0f}s")
    print(f"  Figure: {FIG_DIR / 'hardness_correlation.png'}")


if __name__ == "__main__":
    main()
