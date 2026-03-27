#!/usr/bin/env python3
"""SAT Classification ROC Curve.

Quantifies how well the topological Gini score separates SAT from UNSAT
instances near the phase transition (α ∈ [3.5, 5.0]).

Protocol:
1. Generate 200 random 3-SAT instances at N=100, α ∈ [3.5, 5.0]
2. Compute Gini score (GPU WalkSAT + H₀ persistence)
3. Solve with pysat (Glucose4) — record SAT/UNSAT ground truth
4. Compute ROC curve, AUC, optimal threshold (Youden's J)
5. Report: AUC, accuracy, precision, recall at optimal threshold

Verdict thresholds:
  AUC > 0.90 → production-ready classifier
  AUC > 0.85 → strong classifier
  AUC < 0.70 → insufficient for classification task
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
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ── Styling (matches hardness_correlation.py) ────────────────────────────────
COLORS = {
    "gold":  "#c5a03f",
    "teal":  "#45a8b0",
    "red":   "#e94560",
    "bg":    "#0f0d08",
    "text":  "#d6d0be",
    "muted": "#817a66",
    "green": "#4caf7d",
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.facecolor":   COLORS["bg"],
    "figure.facecolor": COLORS["bg"],
    "text.color":       COLORS["text"],
    "axes.labelcolor":  COLORS["text"],
    "xtick.color":      COLORS["muted"],
    "ytick.color":      COLORS["muted"],
    "axes.edgecolor":   COLORS["muted"],
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
    "savefig.dpi":      200,
})

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("products/sat-hardness-api/results")
FIG_DIR    = Path("products/sat-hardness-api/results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── 3-SAT generation ─────────────────────────────────────────────────────────

def generate_3sat(n_vars: int, alpha: float, seed: int):
    """Generate random 3-SAT.  Returns (clauses_np, negations, n_clauses, dimacs)."""
    rng = np.random.default_rng(seed)
    n_clauses = int(alpha * n_vars)
    clauses_np = np.zeros((n_clauses, 3), dtype=np.int32)
    negations  = np.zeros((n_clauses, 3), dtype=bool)
    dimacs     = []
    for i in range(n_clauses):
        vs  = rng.choice(n_vars, size=3, replace=False)
        negs = rng.random(3) < 0.5
        clauses_np[i] = vs
        negations[i]  = negs
        clause = [int(v + 1) if not n else int(-(v + 1)) for v, n in zip(vs, negs)]
        dimacs.append(clause)
    return clauses_np, negations, n_clauses, dimacs


# ── pysat solver ─────────────────────────────────────────────────────────────

def solve_with_pysat(dimacs_clauses, n_vars):
    """Returns (is_sat: bool, solve_time_ms: float)."""
    from pysat.solvers import Glucose4
    solver = Glucose4()
    for clause in dimacs_clauses:
        solver.add_clause(clause)
    t0 = time.time()
    result = solver.solve()
    solve_time = (time.time() - t0) * 1000
    solver.delete()
    return bool(result), solve_time


# ── Gini score ───────────────────────────────────────────────────────────────

def compute_gini(clauses_np, negations, n_vars, n_clauses,
                 n_walks: int = 200, max_steps: int = 3000, seed: int = 42):
    """Compute topological Gini via GPU WalkSAT + H₀ persistence."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gw",
        str(Path(__file__).parent.parent.parent
            / "problems/p-vs-np/experiments/gpu_walksat.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assignments, unsat = mod.gpu_walksat(
        clauses_np, negations, n_vars, n_clauses,
        n_walks=n_walks, max_steps=max_steps,
        noise_p=0.57, seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Sub-sample to ≤ 200 assignments for H₀ persistence
    n = min(len(assignments), 200)
    if len(assignments) > n:
        idx = np.random.default_rng(42).choice(len(assignments), n, replace=False)
        assignments = assignments[idx]

    a = torch.tensor(assignments, dtype=torch.float32,
                     device="cuda" if torch.cuda.is_available() else "cpu")
    dists = (torch.cdist(a, a, p=1) / a.shape[1]).cpu().numpy()

    # Union-Find for H₀ persistence
    parent = list(range(n))
    rank   = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = sorted(
        [(dists[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
    )
    bars = []
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1
            bars.append(d)

    bars = np.array(bars) if bars else np.array([0.0])

    nn = len(bars)
    if nn <= 1 or np.sum(bars) == 0:
        return 0.0

    sv      = np.sort(bars)
    idx_arr = np.arange(1, nn + 1, dtype=np.float64)
    gini    = float(
        (2 * np.sum(idx_arr * sv)) / (nn * np.sum(sv)) - (nn + 1) / nn
    )
    return gini


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  SAT CLASSIFICATION ROC CURVE")
    print("  Gini as classifier: SAT vs UNSAT (near phase transition)")
    print(f"  {timestamp}")
    print("=" * 70)

    N_VARS     = 100
    N_INSTANCES = 200
    ALPHA_RANGE = (3.5, 5.0)    # near transition — hardest instances

    rng    = np.random.default_rng(7)
    alphas = rng.uniform(*ALPHA_RANGE, size=N_INSTANCES)

    records    = []
    total_t0   = time.time()

    for i, alpha in enumerate(alphas):
        seed = 200000 + i
        clauses_np, negations, n_clauses, dimacs = generate_3sat(N_VARS, alpha, seed)

        # Ground truth
        is_sat, solve_ms = solve_with_pysat(dimacs, N_VARS)

        # Topological score
        t0   = time.time()
        gini = compute_gini(clauses_np, negations, N_VARS, n_clauses, seed=seed + 77)
        topo_ms = (time.time() - t0) * 1000

        records.append({
            "instance":      i,
            "alpha":         float(alpha),
            "gini":          gini,
            "sat":           is_sat,
            "solve_time_ms": solve_ms,
            "topo_time_ms":  topo_ms,
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - total_t0
            print(f"  {i+1}/{N_INSTANCES}: α={alpha:.2f}  G={gini:.4f}  "
                  f"SAT={is_sat}  solve={solve_ms:.1f}ms  topo={topo_ms:.0f}ms  "
                  f"[{elapsed:.0f}s total]")

    total_elapsed = time.time() - total_t0

    # ── ROC analysis ─────────────────────────────────────────────────────────
    ginis  = np.array([r["gini"]  for r in records])
    labels = np.array([r["sat"]   for r in records], dtype=int)  # 1=SAT, 0=UNSAT

    print(f"\n{'='*70}")
    print("  ROC ANALYSIS")
    print(f"{'='*70}")
    print(f"  N instances:  {N_INSTANCES}")
    print(f"  SAT fraction: {labels.mean():.3f}  ({labels.sum()} SAT / {(~labels.astype(bool)).sum()} UNSAT)")
    print(f"  Gini range:   [{ginis.min():.4f}, {ginis.max():.4f}]")
    print(f"  Gini mean:    SAT={ginis[labels==1].mean():.4f}  UNSAT={ginis[labels==0].mean():.4f}")

    # ROC curve — higher Gini → predict SAT (adjust if empirically inverted)
    fpr, tpr, thresholds = roc_curve(labels, ginis)
    auc = roc_auc_score(labels, ginis)

    # Youden's J: maximise TPR - FPR
    j_stat   = tpr - fpr
    best_idx = int(np.argmax(j_stat))
    best_thr = float(thresholds[best_idx])
    best_fpr = float(fpr[best_idx])
    best_tpr = float(tpr[best_idx])

    # Classification metrics at optimal threshold
    y_pred = (ginis >= best_thr).astype(int)
    acc  = accuracy_score(labels, y_pred)
    prec = precision_score(labels, y_pred, zero_division=0)
    rec  = recall_score(labels, y_pred, zero_division=0)
    cm   = confusion_matrix(labels, y_pred)

    print(f"\n  AUC:              {auc:.4f}")
    print(f"  Optimal threshold (Youden's J): {best_thr:.4f}")
    print(f"    TPR: {best_tpr:.3f}  FPR: {best_fpr:.3f}  J: {best_tpr-best_fpr:.3f}")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}  (SAT prediction)")
    print(f"  Recall:    {rec:.3f}  (SAT recall)")
    print(f"  Confusion matrix:\n    {cm}")

    if auc > 0.90:
        verdict = "PRODUCTION-READY — AUC > 0.90, strong SAT/UNSAT classifier"
    elif auc > 0.85:
        verdict = "STRONG CLASSIFIER — AUC > 0.85"
    elif auc > 0.70:
        verdict = "MODERATE CLASSIFIER — AUC > 0.70, usable with caveats"
    else:
        verdict = "WEAK CLASSIFIER — AUC < 0.70, insufficient for classification"

    print(f"\n  VERDICT: {verdict}")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: ROC curve ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(fpr, tpr, color=COLORS["gold"], lw=2.0,
            label=f"Gini classifier (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], lw=1.0, label="Random")
    ax.scatter([best_fpr], [best_tpr], color=COLORS["red"], s=80, zorder=5,
               label=f"Youden's J @ {best_thr:.3f}\n(TPR={best_tpr:.2f}, FPR={best_fpr:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Gini → SAT/UNSAT", color=COLORS["gold"])
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.15)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # AUC text box
    ax.text(0.60, 0.10,
            f"AUC = {auc:.4f}\nAcc = {acc:.3f}\nPrec = {prec:.3f}\nRec = {rec:.3f}",
            transform=ax.transAxes, fontsize=10,
            color=COLORS["text"],
            bbox=dict(boxstyle="round", fc=COLORS["bg"], ec=COLORS["muted"], alpha=0.8))

    # ── Panel 2: Gini histograms by SAT/UNSAT ────────────────────────────────
    ax = axes[1]
    sat_ginis   = ginis[labels == 1]
    unsat_ginis = ginis[labels == 0]
    bins = np.linspace(ginis.min(), ginis.max(), 35)

    if len(sat_ginis) > 0:
        ax.hist(sat_ginis, bins=bins, alpha=0.70, color=COLORS["teal"],
                label=f"SAT (n={len(sat_ginis)})", density=True)
    if len(unsat_ginis) > 0:
        ax.hist(unsat_ginis, bins=bins, alpha=0.70, color=COLORS["red"],
                label=f"UNSAT (n={len(unsat_ginis)})", density=True)

    ax.axvline(best_thr, color=COLORS["gold"], ls="--", lw=1.5,
               label=f"Threshold = {best_thr:.3f}")
    ax.set_xlabel("Gini Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Gini Distribution by Satisfiability\n(α ∈ [{ALPHA_RANGE[0]}, {ALPHA_RANGE[1]}], N={N_VARS})",
                 color=COLORS["gold"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"SAT Hardness API: Classification ROC  —  {N_INSTANCES} instances near transition",
        color=COLORS["text"], fontsize=13,
    )
    fig.tight_layout()

    fig_path = FIG_DIR / "sat_roc.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\n  Figure: {fig_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "experiment":          "classification_roc",
        "timestamp":           timestamp,
        "n_vars":              N_VARS,
        "n_instances":         N_INSTANCES,
        "alpha_range":         list(ALPHA_RANGE),
        "sat_fraction":        float(labels.mean()),
        "auc":                 float(auc),
        "optimal_threshold":   float(best_thr),
        "tpr_at_optimal":      float(best_tpr),
        "fpr_at_optimal":      float(best_fpr),
        "youdens_j":           float(best_tpr - best_fpr),
        "accuracy":            float(acc),
        "precision":           float(prec),
        "recall":              float(rec),
        "confusion_matrix":    cm.tolist(),
        "verdict":             verdict,
        "mean_topo_time_ms":   float(np.mean([r["topo_time_ms"] for r in records])),
        "mean_solve_time_ms":  float(np.mean([r["solve_time_ms"] for r in records])),
        "total_time_s":        round(total_elapsed, 1),
        "gini_mean_sat":       float(ginis[labels == 1].mean()) if labels.sum() > 0 else None,
        "gini_mean_unsat":     float(ginis[labels == 0].mean()) if (labels == 0).sum() > 0 else None,
        "instances":           records,
    }

    json_path = OUTPUT_DIR / "classification_roc.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results: {json_path}")

    print(f"\n  Total elapsed: {total_elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
