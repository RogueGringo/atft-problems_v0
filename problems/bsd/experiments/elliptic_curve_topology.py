"""BSD Experiment: Elliptic Curve Topology vs Rank.

Tests whether the topological signature of a 2D elliptic curve point cloud
correlates with the arithmetic rank of the curve (Birch & Swinnerton-Dyer).

Approach
--------
1. Sample N real points from y^2 = x^3 + ax + b in R^2.
2. Compute H_0 persistent homology via Vietoris-Rips (union-find, O(N^2 log N)).
3. Extract: number of long-lived bars, Gini coefficient of lifetimes, area under
   Betti-0 curve, max lifetime.
4. Compare across curves with known arithmetic rank 0, 1, 2.

Hypothesis
----------
Higher-rank curves should produce more long-lived H_0 features because the
independent Mordell-Weil generators create structural variation in the real
component of the curve.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Make JTopo importable
JTOPO_ROOT = Path("/home/wb1/Desktop/Dev/JTopo")
if str(JTOPO_ROOT) not in sys.path:
    sys.path.insert(0, str(JTOPO_ROOT))


# ---------------------------------------------------------------------------
# Elliptic curve sampler
# ---------------------------------------------------------------------------

def sample_elliptic_curve(
    a: float,
    b: float,
    n_points: int = 500,
    x_range: tuple[float, float] = (-10.0, 10.0),
    seed: int = 42,
) -> np.ndarray:
    """Sample real points from y^2 = x^3 + ax + b.

    Returns array of shape (n_points, 2).  Points where y^2 < 0 are rejected,
    so we sample until we have n_points.  Both branches (y > 0, y < 0) are
    included uniformly.
    """
    rng = np.random.default_rng(seed)
    points: list[list[float]] = []
    # Use a grid-first pass for efficiency, then random fill if needed
    x_grid = np.linspace(x_range[0], x_range[1], 4 * n_points)
    for x in x_grid:
        y2 = x ** 3 + a * x + b
        if y2 >= 0:
            y = float(np.sqrt(y2))
            sign = rng.choice([-1, 1])
            points.append([float(x), sign * y])
            if len(points) >= n_points:
                break

    # Random fallback if grid didn't fill enough
    attempts = 0
    while len(points) < n_points and attempts < 10 * n_points:
        x = rng.uniform(x_range[0], x_range[1])
        y2 = x ** 3 + a * x + b
        if y2 >= 0:
            y = float(np.sqrt(y2))
            sign = rng.choice([-1, 1])
            points.append([float(x), sign * y])
        attempts += 1

    pts = np.array(points[:n_points], dtype=np.float64)
    return pts


# ---------------------------------------------------------------------------
# H_0 persistent homology via union-find on VR complex
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Returns True if a merge happened (different components)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True


def h0_persistence_2d(points: np.ndarray) -> np.ndarray:
    """H_0 persistent homology for a 2D point cloud via Vietoris-Rips.

    Returns (n_finite_bars, 2) array of (birth, death) pairs.
    Each point is born at epsilon=0.  Two components merge when an edge of
    length d is added; the younger component dies at d.
    The immortal component (last surviving) is excluded.

    Algorithm: sort all pairwise edges by length; process with union-find.
    O(N^2 log N).
    """
    n = len(points)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Pairwise distances — use condensed form
    dists = pdist(points, metric="euclidean")
    # Build edge list (i, j, dist)
    idx = np.triu_indices(n, k=1)
    i_arr, j_arr = idx
    # Sort edges by distance
    order = np.argsort(dists)
    i_sorted = i_arr[order]
    j_sorted = j_arr[order]
    d_sorted = dists[order]

    uf = _UnionFind(n)
    bars: list[tuple[float, float]] = []

    for i, j, d in zip(i_sorted, j_sorted, d_sorted):
        if uf.union(i, j):
            # Younger component (born at 0) dies at d
            bars.append((0.0, float(d)))
            if uf.n_components == 1:
                break

    return np.array(bars, dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def gini_coefficient(lifetimes: np.ndarray) -> float:
    """Gini coefficient of the persistence lifetime distribution."""
    lt = lifetimes[np.isfinite(lifetimes)]
    lt = lt[lt > 0]
    if len(lt) == 0:
        return 0.0
    lt = np.sort(lt)
    n = len(lt)
    cumsum = np.cumsum(lt)
    return float((2 * np.sum((np.arange(1, n + 1)) * lt) - (n + 1) * cumsum[-1]) / (n * cumsum[-1]))


@dataclass
class TopologicalFeatures:
    curve_name: str
    rank: int
    a: float
    b: float
    n_points: int
    n_bars: int                    # finite H0 bars
    max_lifetime: float            # longest finite bar
    median_lifetime: float
    mean_lifetime: float
    n_long_bars: int               # bars > median
    gini: float                    # Gini of lifetimes
    betti_auc: float               # area under Betti-0 curve (trapezoid)
    sample_time_s: float
    ph_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_features(
    curve_name: str,
    rank: int,
    a: float,
    b: float,
    n_points: int = 500,
) -> TopologicalFeatures:
    t0 = time.perf_counter()
    pts = sample_elliptic_curve(a, b, n_points=n_points)
    t_sample = time.perf_counter() - t0

    # Normalize point cloud to unit scale for fair comparison
    pts_centered = pts - pts.mean(axis=0)
    scale = pts_centered.std()
    if scale > 0:
        pts_norm = pts_centered / scale
    else:
        pts_norm = pts_centered

    t1 = time.perf_counter()
    bars = h0_persistence_2d(pts_norm)
    t_ph = time.perf_counter() - t1

    if len(bars) == 0:
        lifetimes = np.array([0.0])
    else:
        lifetimes = bars[:, 1] - bars[:, 0]

    max_lt = float(np.max(lifetimes))
    med_lt = float(np.median(lifetimes))
    mean_lt = float(np.mean(lifetimes))
    n_long = int(np.sum(lifetimes > med_lt))
    gin = gini_coefficient(lifetimes)

    # Betti-0 curve: at scale eps, Betti = 1 + #{bars with death > eps}
    eps_grid = np.linspace(0, max_lt * 1.05, 200)
    betti_vals = np.array([1 + int(np.sum(bars[:, 1] > eps)) for eps in eps_grid], dtype=float)
    betti_auc = float(np.trapezoid(betti_vals, eps_grid))

    return TopologicalFeatures(
        curve_name=curve_name,
        rank=rank,
        a=float(a),
        b=float(b),
        n_points=len(pts),
        n_bars=len(bars),
        max_lifetime=max_lt,
        median_lifetime=med_lt,
        mean_lifetime=mean_lt,
        n_long_bars=n_long,
        gini=gin,
        betti_auc=betti_auc,
        sample_time_s=t_sample,
        ph_time_s=t_ph,
    )


# ---------------------------------------------------------------------------
# Curves with known rank
# ---------------------------------------------------------------------------

CURVES: list[dict[str, Any]] = [
    # Rank 0
    {"name": "y²=x³-x",         "rank": 0, "a": -1,  "b": 0,  "note": "conductor 32"},
    {"name": "y²=x³+x+1",       "rank": 0, "a":  1,  "b": 1,  "note": "rank 0"},
    {"name": "y²=x³+1",         "rank": 0, "a":  0,  "b": 1,  "note": "j=0, rank 0"},
    {"name": "y²=x³-2",         "rank": 0, "a":  0,  "b": -2, "note": "rank 0"},
    # Rank 1
    {"name": "y²=x³-x+1",       "rank": 1, "a": -1,  "b": 1,  "note": "rank 1"},
    {"name": "y²=x³+x-1",       "rank": 1, "a":  1,  "b": -1, "note": "rank 1"},
    {"name": "y²=x³-2x+1",      "rank": 1, "a": -2,  "b": 1,  "note": "rank 1"},
    {"name": "y²=x³+3x-5",      "rank": 1, "a":  3,  "b": -5, "note": "rank 1"},
    # Rank 2
    {"name": "y²=x³-7x+6",      "rank": 2, "a": -7,  "b": 6,  "note": "rank 2, generators known"},
    {"name": "y²=x³-5x+4",      "rank": 2, "a": -5,  "b": 4,  "note": "rank 2"},
    {"name": "y²=x³-8x+6",      "rank": 2, "a": -8,  "b": 6,  "note": "rank 2"},
    {"name": "y²=x³-10x+10",    "rank": 2, "a": -10, "b": 10, "note": "rank 2"},
]


# ---------------------------------------------------------------------------
# Plotting — JTopo style
# ---------------------------------------------------------------------------

JTOPO_COLORS = {
    0: "#4393c3",   # blue  — rank 0
    1: "#d6604d",   # red   — rank 1
    2: "#4dac26",   # green — rank 2
}
RANK_LABELS = {0: "Rank 0", 1: "Rank 1", 2: "Rank 2"}


def plot_results(features: list[TopologicalFeatures], save_path: Path) -> None:
    """Four-panel JTopo-styled figure."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "BSD Experiment: Elliptic Curve Topology vs Arithmetic Rank\n"
        r"$y^2 = x^3 + ax + b$ — H$_0$ Persistent Homology",
        fontsize=14, fontweight="bold", y=0.98,
    )

    ranks = [f.rank for f in features]
    max_lts = [f.max_lifetime for f in features]
    n_longs = [f.n_long_bars for f in features]
    ginis = [f.gini for f in features]
    betaaucs = [f.betti_auc for f in features]
    colors = [JTOPO_COLORS[r] for r in ranks]
    names = [f.curve_name for f in features]

    # ---- Panel A: Max lifetime by curve ----
    ax = axes[0, 0]
    x_pos = np.arange(len(features))
    bars_a = ax.bar(x_pos, max_lts, color=colors, edgecolor="white", linewidth=0.8, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace("y²=", "") for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Max H₀ Lifetime")
    ax.set_title("Panel A: Max Persistence Lifetime")
    # Legend patches
    from matplotlib.patches import Patch
    legend_elems = [Patch(color=JTOPO_COLORS[r], label=RANK_LABELS[r]) for r in sorted(JTOPO_COLORS)]
    ax.legend(handles=legend_elems, fontsize=9)

    # ---- Panel B: Long-lived bars count ----
    ax = axes[0, 1]
    ax.bar(x_pos, n_longs, color=colors, edgecolor="white", linewidth=0.8, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace("y²=", "") for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count (lifetime > median)")
    ax.set_title("Panel B: Long-Lived H₀ Bars")
    ax.legend(handles=legend_elems, fontsize=9)

    # ---- Panel C: Gini coefficient ----
    ax = axes[1, 0]
    ax.bar(x_pos, ginis, color=colors, edgecolor="white", linewidth=0.8, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace("y²=", "") for n in names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Panel C: Gini of Lifetime Distribution")
    ax.legend(handles=legend_elems, fontsize=9)

    # ---- Panel D: Rank vs mean feature (scatter with mean ± std) ----
    ax = axes[1, 1]
    for rank_val in [0, 1, 2]:
        subset = [f for f in features if f.rank == rank_val]
        auc_vals = np.array([f.betti_auc for f in subset])
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(subset))
        ax.scatter(
            np.full(len(subset), rank_val) + jitter,
            auc_vals,
            color=JTOPO_COLORS[rank_val],
            s=80, alpha=0.8, zorder=4, label=RANK_LABELS[rank_val],
        )
        if len(auc_vals) > 0:
            ax.hlines(
                np.mean(auc_vals),
                rank_val - 0.25, rank_val + 0.25,
                color=JTOPO_COLORS[rank_val],
                linewidth=2.5, zorder=5,
            )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Rank 0", "Rank 1", "Rank 2"])
    ax.set_ylabel("Betti-0 AUC")
    ax.set_title("Panel D: AUC vs Arithmetic Rank\n(horizontal = group mean)")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {save_path}")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def rank_correlation(features: list[TopologicalFeatures], metric: str) -> dict[str, float]:
    """Spearman rank correlation between arithmetic rank and a topo metric."""
    from scipy.stats import spearmanr, pearsonr
    ranks = np.array([f.rank for f in features])
    vals = np.array([getattr(f, metric) for f in features])
    sp_r, sp_p = spearmanr(ranks, vals)
    pe_r, pe_p = pearsonr(ranks, vals)
    return {
        "spearman_r": float(sp_r),
        "spearman_p": float(sp_p),
        "pearson_r": float(pe_r),
        "pearson_p": float(pe_p),
    }


def summarize_by_rank(features: list[TopologicalFeatures], metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for rank_val in [0, 1, 2]:
        subset = [getattr(f, metric) for f in features if f.rank == rank_val]
        if subset:
            arr = np.array(subset)
            out[f"rank_{rank_val}"] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(subset),
            }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("BSD Experiment: Elliptic Curve Topology vs Rank")
    print("=" * 60)

    N_POINTS = 500
    features: list[TopologicalFeatures] = []

    for curve in CURVES:
        name = curve["name"]
        rank = curve["rank"]
        a = curve["a"]
        b = curve["b"]
        note = curve.get("note", "")
        print(f"\n  Processing {name}  (rank={rank}, {note}) ...")

        feat = extract_features(name, rank, a, b, n_points=N_POINTS)
        features.append(feat)

        print(
            f"    n_points={feat.n_points}  bars={feat.n_bars}"
            f"  max_lt={feat.max_lifetime:.4f}"
            f"  n_long={feat.n_long_bars}"
            f"  gini={feat.gini:.4f}"
            f"  betti_auc={feat.betti_auc:.4f}"
            f"  [sample={feat.sample_time_s*1e3:.1f}ms, ph={feat.ph_time_s*1e3:.1f}ms]"
        )

    # --- Statistical correlation ---
    print("\n" + "-" * 60)
    print("Rank correlation: arithmetic rank vs topological features")
    print("-" * 60)

    correlations: dict[str, dict] = {}
    for metric in ["max_lifetime", "n_long_bars", "gini", "betti_auc", "mean_lifetime"]:
        corr = rank_correlation(features, metric)
        correlations[metric] = corr
        print(
            f"  {metric:<20s}: "
            f"Spearman r={corr['spearman_r']:+.3f} (p={corr['spearman_p']:.4f})  "
            f"Pearson r={corr['pearson_r']:+.3f} (p={corr['pearson_p']:.4f})"
        )

    # --- Per-rank summary ---
    print("\n" + "-" * 60)
    print("Per-rank mean ± std")
    print("-" * 60)
    summaries: dict[str, dict] = {}
    for metric in ["max_lifetime", "n_long_bars", "gini", "betti_auc"]:
        s = summarize_by_rank(features, metric)
        summaries[metric] = s
        parts = []
        for rk in ["rank_0", "rank_1", "rank_2"]:
            if rk in s:
                parts.append(f"{rk}: {s[rk]['mean']:.3f}±{s[rk]['std']:.3f}")
        print(f"  {metric:<20s}: {' | '.join(parts)}")

    # --- STATUS ---
    # Primary metric: does betti_auc increase with rank?
    auc_trend = [
        np.mean([f.betti_auc for f in features if f.rank == r])
        for r in [0, 1, 2]
    ]
    is_monotone = all(auc_trend[i] <= auc_trend[i + 1] for i in range(len(auc_trend) - 1))
    sp_r_auc = correlations["betti_auc"]["spearman_r"]
    sp_p_auc = correlations["betti_auc"]["spearman_p"]

    if sp_r_auc > 0.4 and sp_p_auc < 0.10:
        status = "POSITIVE_CORRELATION"
        verdict = "Positive correlation detected: higher-rank curves show larger Betti-0 AUC."
    elif sp_r_auc > 0.2:
        status = "WEAK_POSITIVE"
        verdict = "Weak positive trend observed but not statistically significant."
    elif sp_r_auc < -0.2:
        status = "NEGATIVE_CORRELATION"
        verdict = "Negative correlation — higher-rank curves show smaller AUC (unexpected)."
    else:
        status = "NO_CORRELATION"
        verdict = "No clear correlation between arithmetic rank and topological features."

    monotone_str = "yes" if is_monotone else "no"
    print(f"\n{'='*60}")
    print(f"STATUS: {status}")
    print(f"Verdict: {verdict}")
    print(f"AUC trend monotone (rank 0→1→2): {monotone_str}  {[f'{v:.3f}' for v in auc_trend]}")
    print(f"{'='*60}")

    # --- Save results ---
    output = {
        "experiment": "bsd_elliptic_curve_topology",
        "status": status,
        "verdict": verdict,
        "n_curves": len(CURVES),
        "n_points_per_curve": N_POINTS,
        "auc_monotone_rank_0_1_2": is_monotone,
        "auc_trend_by_rank": {f"rank_{r}": float(v) for r, v in zip([0, 1, 2], auc_trend)},
        "correlations": correlations,
        "summaries_by_rank": summaries,
        "curves": [f.to_dict() for f in features],
    }

    out_path = RESULTS_DIR / "elliptic_topology.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved: {out_path}")

    # --- Plot ---
    plot_results(features, FIGURES_DIR / "bsd_rank_topology.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
