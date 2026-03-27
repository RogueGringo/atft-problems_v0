#!/usr/bin/env python3
"""Navier-Stokes: Taylor-Green Vortex Topology.

Simulate the Taylor-Green vortex and track topological evolution of the
vortex point cloud (high-vorticity skeleton) across time and Reynolds number.

Method:
  - Pseudospectral DNS on GPU (PyTorch FFT), N=64³ grid, periodic [0,2π]³
  - Vorticity field via spectral differentiation
  - Integrating-factor Euler (viscous term exact, nonlinear explicit)
  - Spectral filter (implicit SGS/LES): exp(-alpha*(|k|/k_max)^8) per step
    stabilizes underresolved high-Re runs without killing large-scale physics
  - H₀ persistence (union-find on Euclidean distance) on vortex cloud
  - Topological observables: ε*(t), Gini(t), component count N₀(t)
  - Sweep: Re ∈ {100, 500, 1000, 2000}

Solver notes:
  The integrating factor handles viscosity exactly: u_hat *= exp(-ν|k|²Δt)
  before adding the nonlinear contribution. The spectral filter damps
  super-dealiased modes that would otherwise alias into instability at high Re.
  This is equivalent to implicit LES (large-eddy simulation) at N=64.

Physics:
  u(x,y,z,0) =  sin(x)cos(y)cos(z)
  v(x,y,z,0) = -cos(x)sin(y)cos(z)
  w(x,y,z,0) = 0
  Domain: [0, 2π]³, periodic
  ν = 1/Re

Expected behavior:
  Low Re:  smooth viscous decay, ε* decreases monotonically, Gini low
  High Re: slower decay, residual turbulence at late time, ε* higher,
           more complex vortex structures captured in topology
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

# ── Constants ──────────────────────────────────────────────────────────────
PI = float(np.pi)
N = 64           # Grid resolution (N³)
T_END = 4.0      # End time
SAVE_INTERVAL = 0.20   # Save topology every 0.2 time units
VORTEX_THRESHOLD_FRAC = 0.50   # Points where |ω| > this fraction of max|ω|
MAX_CLOUD_POINTS = 1200        # Cap vortex cloud for PH speed
RE_VALUES = [100, 500, 1000, 2000]

# Per-Re time steps: smaller dt at high Re to control nonlinear CFL
DT_FOR_RE = {100: 0.005, 500: 0.002, 1000: 0.001, 2000: 0.001}

# Spectral filter parameter: exp(-FILTER_ALPHA * (|k|/k_max)^8)
# alpha=1.0: very mild, only damps modes at k>0.9*k_max
FILTER_ALPHA = 1.0

# ── JTopo-style colors ─────────────────────────────────────────────────────
COLORS = {
    "gold":   "#c5a03f",
    "teal":   "#45a8b0",
    "red":    "#e94560",
    "purple": "#9b59b6",
    "green":  "#2ecc71",
    "bg":     "#0f0d08",
    "text":   "#d6d0be",
    "muted":  "#817a66",
}
RE_PALETTE = [COLORS["teal"], COLORS["gold"], COLORS["red"], COLORS["purple"]]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.facecolor": COLORS["bg"],
    "figure.facecolor": COLORS["bg"],
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "axes.edgecolor": COLORS["muted"],
    "grid.color": COLORS["muted"],
    "grid.alpha": 0.15,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

OUTPUT_DIR = Path("problems/navier-stokes/results")
FIG_DIR    = Path("problems/navier-stokes/results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Spectral infrastructure ────────────────────────────────────────────────

def build_grid(n: int, device: str) -> tuple:
    """Build wavenumber arrays, dealiasing mask, and spectral filter."""
    k = torch.fft.fftfreq(n, d=1.0 / n, device=device, dtype=torch.float64)
    KX, KY, KZ = torch.meshgrid(k, k, k, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0  # avoid /0 in pressure projection

    # 2/3 dealiasing mask
    cutoff = n // 3
    mask = (torch.abs(KX) <= cutoff) & (torch.abs(KY) <= cutoff) & (torch.abs(KZ) <= cutoff)

    # Spectral filter: damps modes near Nyquist without touching resolved scales
    k_max = float(n // 2)
    K_abs = torch.sqrt(K2.clamp(min=0.0))
    sf = torch.exp(-FILTER_ALPHA * (K_abs / k_max) ** 8).to(torch.complex128)

    return KX, KY, KZ, K2, mask, sf


def taylor_green_ic(n: int, device: str) -> tuple:
    """Initialize Taylor-Green vortex on N³ periodic grid [0, 2π]³."""
    x = torch.linspace(0.0, 2.0 * PI, n + 1, device=device, dtype=torch.float64)[:-1]
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w = torch.zeros_like(u)
    return u, v, w


# ── Nonlinear term (advection) ─────────────────────────────────────────────

def nonlinear_term(u_hat: torch.Tensor, v_hat: torch.Tensor, w_hat: torch.Tensor,
                   KX: torch.Tensor, KY: torch.Tensor, KZ: torch.Tensor,
                   K2: torch.Tensor, mask: torch.Tensor,
                   ) -> tuple:
    """Compute dealiased -∇p projected advection term.

    Returns (-Nu_hat, -Nv_hat, -Nw_hat) where N = (u·∇)u.
    The result is projected onto the divergence-free manifold.
    """
    uh_d = u_hat * mask
    vh_d = v_hat * mask
    wh_d = w_hat * mask

    # Physical velocities from dealiased modes
    u = torch.fft.ifftn(uh_d).real
    v = torch.fft.ifftn(vh_d).real
    w = torch.fft.ifftn(wh_d).real

    # Spectral gradients
    dudx = torch.fft.ifftn(1j * KX * uh_d).real
    dudy = torch.fft.ifftn(1j * KY * uh_d).real
    dudz = torch.fft.ifftn(1j * KZ * uh_d).real
    dvdx = torch.fft.ifftn(1j * KX * vh_d).real
    dvdy = torch.fft.ifftn(1j * KY * vh_d).real
    dvdz = torch.fft.ifftn(1j * KZ * vh_d).real
    dwdx = torch.fft.ifftn(1j * KX * wh_d).real
    dwdy = torch.fft.ifftn(1j * KY * wh_d).real
    dwdz = torch.fft.ifftn(1j * KZ * wh_d).real

    # Advection
    Nu = u * dudx + v * dudy + w * dudz
    Nv = u * dvdx + v * dvdy + w * dvdz
    Nw = u * dwdx + v * dwdy + w * dwdz

    # Spectral transform with dealiasing mask applied
    Nu_hat = torch.fft.fftn(Nu) * mask
    Nv_hat = torch.fft.fftn(Nv) * mask
    Nw_hat = torch.fft.fftn(Nw) * mask

    # Leray projection: remove divergent part
    div = 1j * KX * Nu_hat + 1j * KY * Nv_hat + 1j * KZ * Nw_hat
    Nu_hat -= 1j * KX * div / K2
    Nv_hat -= 1j * KY * div / K2
    Nw_hat -= 1j * KZ * div / K2

    return -Nu_hat, -Nv_hat, -Nw_hat


# ── Integrating-Factor Euler time step ────────────────────────────────────

def if_euler_step(u_hat: torch.Tensor, v_hat: torch.Tensor, w_hat: torch.Tensor,
                  E: torch.Tensor,  # exp(-ν K² Δt) * spectral_filter
                  dt: float,
                  KX: torch.Tensor, KY: torch.Tensor, KZ: torch.Tensor,
                  K2: torch.Tensor, mask: torch.Tensor,
                  ) -> tuple:
    """Integrating-factor Euler step: exact viscous decay + explicit nonlinear.

    u_hat_{n+1} = E * u_hat_n + dt * N(u_hat_n)

    where E = exp(-ν K² Δt) * spectral_filter absorbs both viscosity and
    high-wavenumber stabilization in a single elementwise multiplication.
    The nonlinear term is divergence-free projected at each step.
    """
    Nu_hat, Nv_hat, Nw_hat = nonlinear_term(u_hat, v_hat, w_hat, KX, KY, KZ, K2, mask)

    # Integrating factor step: viscous decay (exact) + nonlinear contribution
    # The nonlinear term is already divergence-free projected; no extra projection needed.
    u_new = u_hat * E + dt * Nu_hat
    v_new = v_hat * E + dt * Nv_hat
    w_new = w_hat * E + dt * Nw_hat

    return u_new, v_new, w_new


# ── Vortex Cloud Extraction ─────────────────────────────────────────────────

def spectral_curl(u_hat: torch.Tensor, v_hat: torch.Tensor, w_hat: torch.Tensor,
                  KX: torch.Tensor, KY: torch.Tensor, KZ: torch.Tensor,
                  ) -> tuple:
    """Compute ω = ∇×u via spectral differentiation, return physical components."""
    wx = torch.fft.ifftn(1j * KY * w_hat - 1j * KZ * v_hat).real
    wy = torch.fft.ifftn(1j * KZ * u_hat - 1j * KX * w_hat).real
    wz = torch.fft.ifftn(1j * KX * v_hat - 1j * KY * u_hat).real
    return wx, wy, wz


def extract_vortex_cloud(wx: torch.Tensor, wy: torch.Tensor, wz: torch.Tensor,
                          threshold_frac: float = VORTEX_THRESHOLD_FRAC,
                          max_points: int = MAX_CLOUD_POINTS,
                          ) -> np.ndarray:
    """Extract grid positions where |ω| > threshold_frac * max(|ω|).

    Returns (M, 3) float32 array of vortex positions (grid-index units).
    """
    omega_mag = torch.sqrt(wx**2 + wy**2 + wz**2)
    omega_max = float(omega_mag.max().item())
    if omega_max < 1e-12:
        return np.zeros((0, 3), dtype=np.float32)

    mask_vx = omega_mag > (threshold_frac * omega_max)
    indices = torch.nonzero(mask_vx, as_tuple=False).float()

    if indices.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pts = indices.cpu().numpy().astype(np.float32)
    if len(pts) > max_points:
        rng = np.random.default_rng(seed=7)
        idx = rng.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    return pts


# ── H₀ Persistent Homology ─────────────────────────────────────────────────

def h0_persistence(pts: np.ndarray) -> np.ndarray:
    """Union-find H₀ persistence on Euclidean point cloud.

    Returns sorted array of death times (the N-1 edge weights merging N components).
    """
    n = len(pts)
    if n < 2:
        return np.array([], dtype=np.float64)

    # GPU pairwise distances for large clouds
    if n > 200:
        t = torch.tensor(pts, dtype=torch.float32, device=DEVICE)
        dists = torch.cdist(t, t).cpu().numpy()
    else:
        from scipy.spatial.distance import cdist
        dists = cdist(pts, pts, metric="euclidean").astype(np.float32)

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    triu_i, triu_j = np.triu_indices(n, k=1)
    edge_d = dists[triu_i, triu_j]
    order = np.argsort(edge_d)

    death_times = []
    for idx in order:
        d = float(edge_d[idx])
        ri, rj = find(int(triu_i[idx])), find(int(triu_j[idx]))
        if ri != rj:
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1
            death_times.append(d)

    return np.array(death_times, dtype=np.float64)


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array."""
    n = len(values)
    if n <= 1 or np.sum(values) == 0:
        return 0.0
    s = np.sort(values)
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * s)) / (n * np.sum(s)) - (n + 1.0) / n)


def topology_from_cloud(pts: np.ndarray) -> dict:
    """Compute H₀ topological summary from vortex point cloud."""
    if len(pts) < 2:
        return {
            "n_components": int(len(pts)),
            "epsilon_star": 0.0,
            "gini": 0.0,
            "n_points": int(len(pts)),
        }

    bars = h0_persistence(pts)

    if len(bars) > 0:
        eps_star = float(np.percentile(bars, 95))
        g = gini(bars)
        n_components = max(1, int(np.sum(bars > eps_star)))
    else:
        eps_star = 0.0
        g = 0.0
        n_components = 1

    return {
        "n_components": n_components,
        "epsilon_star": eps_star,
        "gini": g,
        "n_points": int(len(pts)),
    }


# ── Simulation ─────────────────────────────────────────────────────────────

def run_simulation(re: float) -> dict:
    """Run Taylor-Green DNS and collect topological observables.

    Uses integrating-factor Euler with spectral LES filter for stability
    at all Re on N=64 grid.
    """
    nu = 1.0 / re
    dt = DT_FOR_RE.get(int(re), 0.005)
    n_steps = int(round(T_END / dt))
    save_every = max(1, int(round(SAVE_INTERVAL / dt)))

    print(f"\n  [Re={re:.0f}] ν={nu:.5f} | dt={dt} | n_steps={n_steps} | "
          f"saves={n_steps // save_every + 1}")

    KX, KY, KZ, K2, mask, sf = build_grid(N, DEVICE)

    # Integrating factor: E = exp(-ν K² Δt) * spectral_filter
    E = torch.exp(-nu * K2 * dt).to(torch.complex128) * sf

    u, v, w = taylor_green_ic(N, DEVICE)
    u_hat = torch.fft.fftn(u.to(torch.complex128)) * mask
    v_hat = torch.fft.fftn(v.to(torch.complex128)) * mask
    w_hat = torch.fft.fftn(w.to(torch.complex128)) * mask

    times: list[float] = []
    n_components_list: list[int] = []
    epsilon_star_list: list[float] = []
    gini_list: list[float] = []
    n_points_list: list[int] = []
    enstrophy_list: list[float] = []

    t0 = time.time()

    for step in range(n_steps + 1):
        current_t = step * dt

        if step % save_every == 0:
            # Vorticity
            wx, wy, wz = spectral_curl(u_hat, v_hat, w_hat, KX, KY, KZ)
            omega_mag = torch.sqrt(wx**2 + wy**2 + wz**2)
            omega_max_val = float(omega_mag.max().item())

            if not np.isfinite(omega_max_val):
                print(f"    t={current_t:.2f} | DIVERGED — stopping")
                break

            enstrophy = float(0.5 * torch.mean(omega_mag**2).item())
            enstrophy_list.append(enstrophy)

            # Vortex cloud
            pts = extract_vortex_cloud(wx, wy, wz)
            topo = topology_from_cloud(pts)

            times.append(float(current_t))
            n_components_list.append(topo["n_components"])
            epsilon_star_list.append(topo["epsilon_star"])
            gini_list.append(topo["gini"])
            n_points_list.append(topo["n_points"])

            elapsed = time.time() - t0
            print(f"    t={current_t:.2f} | E={enstrophy:.4f} | "
                  f"pts={topo['n_points']} | ε*={topo['epsilon_star']:.3f} | "
                  f"G={topo['gini']:.3f} | N₀={topo['n_components']} | "
                  f"({elapsed:.1f}s)")

        if step < n_steps:
            u_hat, v_hat, w_hat = if_euler_step(
                u_hat, v_hat, w_hat, E, dt, KX, KY, KZ, K2, mask
            )

    total_time = time.time() - t0
    print(f"  [Re={re:.0f}] Done in {total_time:.1f}s ({len(times)} snapshots)")

    return {
        "re": float(re),
        "nu": float(nu),
        "n": N,
        "dt": dt,
        "n_steps": n_steps,
        "save_every": save_every,
        "t_end": T_END,
        "times": times,
        "n_components": n_components_list,
        "epsilon_star": epsilon_star_list,
        "gini": gini_list,
        "n_points": n_points_list,
        "enstrophy": enstrophy_list,
        "total_time_s": round(total_time, 1),
    }


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze_re_dependence(all_results: list[dict]) -> dict:
    """Summarize Re-dependent topological behavior."""
    re_summary = []
    for r in all_results:
        times = np.array(r["times"])
        eps = np.array(r["epsilon_star"])
        gini_arr = np.array(r["gini"])
        ens = np.array(r["enstrophy"])

        half = max(1, len(times) // 2)
        mean_eps_late = float(np.mean(eps[half:]))
        std_eps_late  = float(np.std(eps[half:]))
        mean_gini_late = float(np.mean(gini_arr[half:]))
        peak_t = float(times[int(np.argmax(ens))]) if len(ens) > 0 else 0.0
        variability = float(np.std(eps) / (np.mean(eps) + 1e-10))

        re_summary.append({
            "re": r["re"],
            "mean_eps_late": mean_eps_late,
            "std_eps_late": std_eps_late,
            "mean_gini_late": mean_gini_late,
            "enstrophy_peak_t": peak_t,
            "eps_variability": variability,
            "final_enstrophy": float(ens[-1]) if len(ens) > 0 else 0.0,
        })

    re_arr = np.array([s["re"] for s in re_summary], dtype=float)
    eps_arr = np.array([s["mean_eps_late"] for s in re_summary], dtype=float)
    gini_arr2 = np.array([s["mean_gini_late"] for s in re_summary], dtype=float)
    var_arr = np.array([s["eps_variability"] for s in re_summary], dtype=float)
    ens_arr = np.array([s["final_enstrophy"] for s in re_summary], dtype=float)

    def safe_corr(a, b):
        if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    corr_eps_re  = safe_corr(re_arr, eps_arr)
    corr_gini_re = safe_corr(re_arr, gini_arr2)
    corr_var_re  = safe_corr(re_arr, var_arr)
    corr_ens_re  = safe_corr(re_arr, ens_arr)

    # Detect Re-dependent behavior:
    # ε* or Gini or final enstrophy should correlate with Re
    re_dependent = (
        abs(corr_eps_re) > 0.3 or
        abs(corr_gini_re) > 0.3 or
        abs(corr_var_re) > 0.3 or
        abs(corr_ens_re) > 0.3
    )

    return {
        "re_summary": re_summary,
        "corr_eps_re": corr_eps_re,
        "corr_gini_re": corr_gini_re,
        "corr_variability_re": corr_var_re,
        "corr_final_enstrophy_re": corr_ens_re,
        "re_dependent_behavior": re_dependent,
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(all_results: list[dict], fig_path: Path) -> None:
    """4-panel figure: vortex count, ε*(t), Gini(t), cross-Re comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: vortex component count N₀(t)
    ax = axes[0, 0]
    for res, color in zip(all_results, RE_PALETTE):
        t = res["times"]
        ax.plot(t, res["n_components"], "-o", color=color, linewidth=2,
                markersize=3, label=f"Re={res['re']:.0f}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("N₀(t)  [H₀ components]")
    ax.set_title("Vortex Component Count vs Time", color=COLORS["teal"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # Panel 2: ε*(t) onset scale
    ax = axes[0, 1]
    for res, color in zip(all_results, RE_PALETTE):
        t = res["times"]
        ax.plot(t, res["epsilon_star"], "-s", color=color, linewidth=2,
                markersize=3, label=f"Re={res['re']:.0f}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("ε*(t)  [vortex coherence length]")
    ax.set_title("Onset Scale ε*(t) vs Time", color=COLORS["gold"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # Panel 3: Gini(t)
    ax = axes[1, 0]
    for res, color in zip(all_results, RE_PALETTE):
        t = res["times"]
        ax.plot(t, res["gini"], "-^", color=color, linewidth=2,
                markersize=3, label=f"Re={res['re']:.0f}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Gini(t)  [persistence hierarchy]")
    ax.set_title("Gini Coefficient vs Time", color=COLORS["purple"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # Panel 4: Cross-Re mean ε* and Gini at late time
    ax = axes[1, 1]
    re_vals = [r["re"] for r in all_results]
    half = lambda r: max(1, len(r["epsilon_star"]) // 2)
    late_eps  = [float(np.mean(r["epsilon_star"][half(r):])) for r in all_results]
    late_gini = [float(np.mean(r["gini"][half(r):])) for r in all_results]
    std_eps   = [float(np.std(r["epsilon_star"][half(r):])) for r in all_results]

    ax2 = ax.twinx()
    ax.errorbar(re_vals, late_eps, yerr=std_eps, fmt="-o", color=COLORS["teal"],
                linewidth=2, markersize=7, capsize=4, label="ε* mean (left)")
    ax2.plot(re_vals, late_gini, "-s", color=COLORS["gold"], linewidth=2,
             markersize=7, label="Gini mean (right)")
    ax.set_xlabel("Reynolds Number Re")
    ax.set_ylabel("Mean ε* (late time)", color=COLORS["teal"])
    ax2.set_ylabel("Mean Gini (late time)", color=COLORS["gold"])
    ax.set_title("Re-Dependent Topological Behavior", color=COLORS["red"])
    ax.set_xscale("log")
    ax.tick_params(axis="y", labelcolor=COLORS["teal"])
    ax2.tick_params(axis="y", labelcolor=COLORS["gold"])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"Navier-Stokes: Taylor-Green Vortex Topology  "
        f"(N={N}³, t∈[0,{T_END:.1f}], IF-Euler + spectral LES filter)",
        color=COLORS["text"], fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_path, facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  NAVIER-STOKES: TAYLOR-GREEN VORTEX TOPOLOGY")
    print("  Tracking topological evolution of the vortex skeleton")
    print(f"  {timestamp}")
    print(f"  Device: {DEVICE} | N={N}³ | T_END={T_END} | Re values: {RE_VALUES}")
    print("=" * 70)

    wall_t0 = time.time()
    all_results = []

    for re in RE_VALUES:
        result = run_simulation(float(re))
        all_results.append(result)

    total_wall = time.time() - wall_t0

    print("\n" + "=" * 70)
    print("  ANALYSIS: Re-Dependent Topological Behavior")
    print("=" * 70)
    analysis = analyze_re_dependence(all_results)

    for s in analysis["re_summary"]:
        print(f"  Re={s['re']:.0f}: ε*_late={s['mean_eps_late']:.3f}±{s['std_eps_late']:.3f} "
              f"Gini_late={s['mean_gini_late']:.3f} "
              f"var={s['eps_variability']:.3f} "
              f"ens_final={s['final_enstrophy']:.4f}")

    print(f"\n  Correlation(ε*, Re)          = {analysis['corr_eps_re']:.3f}")
    print(f"  Correlation(Gini, Re)        = {analysis['corr_gini_re']:.3f}")
    print(f"  Correlation(variability, Re) = {analysis['corr_variability_re']:.3f}")
    print(f"  Correlation(final_ens, Re)   = {analysis['corr_final_enstrophy_re']:.3f}")

    status = "RE-DEPENDENT BEHAVIOR DETECTED" if analysis["re_dependent_behavior"] else "NO CLEAR RE DEPENDENCE"
    print(f"\n  STATUS: {status}")

    # Plot
    fig_path = FIG_DIR / "ns_vortex_topology.png"
    plot_results(all_results, fig_path)

    # Save JSON
    output = {
        "experiment": "taylor_green_topology",
        "timestamp": timestamp,
        "device": DEVICE,
        "grid_n": N,
        "t_end": T_END,
        "save_interval": SAVE_INTERVAL,
        "dt_for_re": DT_FOR_RE,
        "filter_alpha": FILTER_ALPHA,
        "vortex_threshold_frac": VORTEX_THRESHOLD_FRAC,
        "solver": "Integrating-factor Euler + 2/3 dealiasing + spectral LES filter",
        "re_values": RE_VALUES,
        "simulations": all_results,
        "analysis": analysis,
        "total_wall_time_s": round(total_wall, 1),
        "status": "RE-DEPENDENT" if analysis["re_dependent_behavior"] else "INCONCLUSIVE",
    }

    json_path = OUTPUT_DIR / "taylor_green_topology.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON saved: {json_path}")
    print(f"\n  Total wall time: {total_wall:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
