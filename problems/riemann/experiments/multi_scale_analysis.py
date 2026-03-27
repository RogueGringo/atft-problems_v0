#!/usr/bin/env python3
"""Multi-Scale Zeta Analysis.

The ATFT framework encodes arithmetic structure via holonomy of a sheaf
Laplacian on a Rips complex built from the unfolded zeta zeros.  Previous
experiments fixed ε = 3.0.  This experiment sweeps ε over the full range
[0.5, 5.0] to reveal WHERE in scale-space the arithmetic signal lives.

Protocol:
1. Load 1000 zeta zeros (Odlyzko) + 1000 GUE + 1000 Random
2. Unfold all three to mean spacing = 1
3. For each ε ∈ [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
   a. Count Rips edges (density proxy)
   b. Compute S(σ=0.5, ε) = Σ smallest 20 eigenvalues (MatFree engine)
      for Zeta and GUE
   c. Arithmetic premium = S_zeta / S_gue  (or difference if preferred)
4. Plot premium vs ε
5. Save figure + JSON

If the premium peaks at a specific ε, that is the natural scale of the
arithmetic structure — the geometric "resonance length" of the zeta zeros.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── JTopo package path ────────────────────────────────────────────────────────
JTOPO_ROOT = Path(__file__).resolve().parents[4] / "JTopo"
sys.path.insert(0, str(JTOPO_ROOT))

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.gue import GUESource
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.matfree_sheaf_laplacian import MatFreeSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

# ── Styling (JTopo dark theme) ────────────────────────────────────────────────
COLORS = {
    "gold":  "#c5a03f",
    "teal":  "#45a8b0",
    "red":   "#e94560",
    "blue":  "#5c8dd4",
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

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIG_DIR     = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = JTOPO_ROOT / "data" / "odlyzko_zeros.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_rips_edges(pts: np.ndarray, epsilon: float) -> int:
    """Count edges in the Vietoris-Rips complex at scale ε."""
    n = len(pts)
    dists = np.abs(pts[:, None] - pts[None, :])
    edges = int(np.sum(dists < epsilon)) - n  # subtract diagonal
    return edges // 2                          # undirected


def load_and_unfold_data(n: int):
    """Load + unfold zeta, GUE, and random Poisson point clouds."""
    # Zeta zeros
    src   = ZetaZerosSource(DATA_PATH)
    cloud = src.generate(n)
    zeta  = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]

    # GUE (Dumitriu-Edelman tridiagonal, σ=1)
    gue_cloud = GUESource(seed=42).generate(n)
    gue       = SpectralUnfolding(method="rank").transform(gue_cloud).points[:, 0]
    # Scale GUE to mean spacing = 1  (rank-unfold already does this)
    ms = float(np.mean(np.diff(np.sort(gue))))
    if ms > 0:
        gue = gue / ms

    # Random (Poisson) — uniform on same range as zeta
    rng    = np.random.default_rng(0)
    z_min, z_max = float(zeta.min()), float(zeta.max())
    rand   = np.sort(rng.uniform(z_min, z_max, n))

    return zeta, gue, rand


def compute_spectral_sum(zeros: np.ndarray, epsilon: float,
                         builder: TransportMapBuilder, k_eig: int = 20) -> float:
    """S(ε) = Σ k smallest eigenvalues of the MatFree sheaf Laplacian."""
    lap  = MatFreeSheafLaplacian(builder, zeros, transport_mode="superposition")
    eigs = lap.smallest_eigenvalues(epsilon, k=k_eig)
    return float(np.sum(eigs))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  ATFT MULTI-SCALE ZETA ANALYSIS")
    print("  Scale dependence of the arithmetic premium")
    print(f"  {timestamp}")
    print("=" * 70)

    N       = 1000
    K       = 200
    SIGMA   = 0.5
    K_EIG   = 20
    EPSILONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    print(f"\n  N={N} zeros, K={K}, σ={SIGMA}, k_eig={K_EIG}")
    print(f"  ε grid: {EPSILONS}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading and unfolding point clouds...")
    zeta, gue, rand = load_and_unfold_data(N)
    print(f"  Zeta: mean_spacing={np.mean(np.diff(np.sort(zeta))):.4f}  "
          f"range=[{zeta.min():.1f}, {zeta.max():.1f}]")
    print(f"  GUE:  mean_spacing={np.mean(np.diff(np.sort(gue))):.4f}")
    print(f"  Rand: mean_spacing={np.mean(np.diff(np.sort(rand))):.4f}")

    # ── Build transport (shared builder, K+primes fixed) ─────────────────────
    builder = TransportMapBuilder(K=K, sigma=SIGMA)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    sweep_results = []

    for eps in EPSILONS:
        print(f"\n  ε = {eps}")

        # Edge counts
        n_edges_zeta = count_rips_edges(zeta, eps)
        n_edges_gue  = count_rips_edges(gue, eps)
        n_edges_rand = count_rips_edges(rand, eps)
        print(f"    edges: zeta={n_edges_zeta}  gue={n_edges_gue}  rand={n_edges_rand}")

        # Zeta spectral sum
        print(f"    Computing S_zeta...")
        t0      = time.time()
        s_zeta  = compute_spectral_sum(zeta, eps, builder, K_EIG)
        t_zeta  = time.time() - t0
        print(f"    S_zeta  = {s_zeta:.6f}  ({t_zeta:.1f}s)")

        # GUE spectral sum
        print(f"    Computing S_gue...")
        t0     = time.time()
        s_gue  = compute_spectral_sum(gue, eps, builder, K_EIG)
        t_gue  = time.time() - t0
        print(f"    S_gue   = {s_gue:.6f}  ({t_gue:.1f}s)")

        # Premium
        if s_gue > 1e-15:
            premium_ratio = s_zeta / s_gue
        else:
            premium_ratio = float("nan")

        premium_diff = s_zeta - s_gue

        print(f"    Premium: ratio={premium_ratio:.4f}  diff={premium_diff:.6f}")
        sys.stdout.flush()

        sweep_results.append({
            "epsilon":         eps,
            "n_edges_zeta":    n_edges_zeta,
            "n_edges_gue":     n_edges_gue,
            "n_edges_rand":    n_edges_rand,
            "s_zeta":          s_zeta,
            "s_gue":           s_gue,
            "premium_ratio":   premium_ratio,
            "premium_diff":    premium_diff,
            "time_zeta_s":     round(t_zeta, 2),
            "time_gue_s":      round(t_gue, 2),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SCALE DEPENDENCE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'ε':>5}  {'edges(Z)':>10}  {'S_zeta':>12}  {'S_gue':>12}  {'Premium':>10}")
    for r in sweep_results:
        print(f"  {r['epsilon']:>5.1f}  {r['n_edges_zeta']:>10}  "
              f"{r['s_zeta']:>12.5f}  {r['s_gue']:>12.5f}  "
              f"{r['premium_ratio']:>10.4f}")

    # Peak premium
    valid = [r for r in sweep_results if not np.isnan(r["premium_ratio"])]
    if valid:
        peak = max(valid, key=lambda r: abs(r["premium_ratio"] - 1.0))
        print(f"\n  Peak arithmetic premium at ε = {peak['epsilon']:.1f}")
        print(f"    S_zeta / S_gue = {peak['premium_ratio']:.4f}")
        print(f"    (edges_zeta={peak['n_edges_zeta']})")

    # ── Figure ────────────────────────────────────────────────────────────────
    eps_arr      = np.array([r["epsilon"]       for r in sweep_results])
    prem_ratio   = np.array([r["premium_ratio"]  for r in sweep_results])
    prem_diff    = np.array([r["premium_diff"]   for r in sweep_results])
    s_zeta_arr   = np.array([r["s_zeta"]         for r in sweep_results])
    s_gue_arr    = np.array([r["s_gue"]          for r in sweep_results])
    edges_zeta   = np.array([r["n_edges_zeta"]   for r in sweep_results])
    edges_gue    = np.array([r["n_edges_gue"]    for r in sweep_results])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel 1: S(ε) for Zeta and GUE ───────────────────────────────────────
    ax = axes[0]
    ax.plot(eps_arr, s_zeta_arr, "o-", color=COLORS["gold"],  lw=2.0,
            markersize=7, label="Zeta zeros")
    ax.plot(eps_arr, s_gue_arr,  "s--", color=COLORS["teal"], lw=2.0,
            markersize=7, label="GUE")
    ax.set_xlabel("ε (Rips scale)")
    ax.set_ylabel("S(σ=0.5, ε)  =  Σ λₖ")
    ax.set_title("Spectral Sum vs Scale", color=COLORS["gold"])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.set_yscale("log")

    # ── Panel 2: Arithmetic premium (ratio) ───────────────────────────────────
    ax = axes[1]
    # Shade premium region
    ax.fill_between(eps_arr, 1.0, prem_ratio,
                    where=prem_ratio >= 1.0,
                    color=COLORS["gold"], alpha=0.25, label="Zeta > GUE")
    ax.fill_between(eps_arr, prem_ratio, 1.0,
                    where=prem_ratio < 1.0,
                    color=COLORS["teal"], alpha=0.25, label="GUE > Zeta")
    ax.plot(eps_arr, prem_ratio, "o-", color=COLORS["gold"], lw=2.0,
            markersize=8, label="S_zeta / S_gue")
    ax.axhline(1.0, color=COLORS["muted"], ls="--", lw=1.0, label="No premium")
    if valid:
        peak_eps = peak["epsilon"]
        ax.axvline(peak_eps, color=COLORS["red"], ls=":", lw=1.5,
                   label=f"Peak ε = {peak_eps:.1f}")
    ax.set_xlabel("ε (Rips scale)")
    ax.set_ylabel("S_zeta / S_gue")
    ax.set_title("Arithmetic Premium vs Scale", color=COLORS["gold"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # ── Panel 3: Edge density ─────────────────────────────────────────────────
    ax = axes[2]
    ax.semilogy(eps_arr, edges_zeta, "o-", color=COLORS["gold"],
                lw=2.0, markersize=7, label="Zeta")
    ax.semilogy(eps_arr, edges_gue,  "s--", color=COLORS["teal"],
                lw=2.0, markersize=7, label="GUE")
    ax.set_xlabel("ε (Rips scale)")
    ax.set_ylabel("Rips edges (log scale)")
    ax.set_title("Complex Density vs Scale", color=COLORS["gold"])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"ATFT Multi-Scale Analysis: N={N} zeros, K={K}, σ={SIGMA}",
        color=COLORS["text"], fontsize=13,
    )
    fig.tight_layout()

    fig_path = FIG_DIR / "multi_scale_premium.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\n  Figure: {fig_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "experiment":  "multi_scale_analysis",
        "timestamp":   timestamp,
        "N":           N,
        "K":           K,
        "sigma":       SIGMA,
        "k_eig":       K_EIG,
        "epsilon_grid": EPSILONS,
        "peak_epsilon": peak["epsilon"] if valid else None,
        "peak_premium_ratio": peak["premium_ratio"] if valid else None,
        "sweep":       sweep_results,
    }

    json_path = RESULTS_DIR / "multi_scale_analysis.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results: {json_path}")

    total_elapsed = sum(r["time_zeta_s"] + r["time_gue_s"] for r in sweep_results)
    print(f"  Compute time: {total_elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
