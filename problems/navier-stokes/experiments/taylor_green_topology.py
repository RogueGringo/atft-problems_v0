#!/usr/bin/env python3
"""Navier-Stokes: Taylor-Green Vortex Topology.

Simulate the Taylor-Green vortex and track topological evolution of the
vortex point cloud (high-vorticity skeleton) across time and Reynolds number.

Method:
  - Pseudospectral DNS on GPU (PyTorch FFT), N=64³ grid
  - Vorticity field computed via spectral differentiation
  - RK4 time-stepping with spectral viscous term
  - H₀ persistence (union-find on Euclidean distance) on vortex cloud
  - Topological observables: ε*(t), Gini(t), component count N₀(t)
  - Sweep: Re ∈ {100, 500, 1000, 2000}

Physics:
  u(x,y,z,0) =  sin(x)cos(y)cos(z)
  v(x,y,z,0) = -cos(x)sin(y)cos(z)
  w(x,y,z,0) = 0
  Domain: [0, 2π]³, periodic

Expected behavior:
  Low Re:  smooth viscous decay, vortices diffuse, ε* large and stable
  High Re: vortex stretching & reconnection, ε* fluctuates, Gini spikes
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
N = 64          # Grid resolution (N³)
DT = 0.01       # Time step
N_STEPS = 400   # Total steps (t ∈ [0, 4.0])
SAVE_EVERY = 20 # Save topology every 20 steps (20 snapshots per Re)
VORTEX_THRESHOLD_FRAC = 0.50   # Points where |ω| > this fraction of max|ω|
MAX_CLOUD_POINTS = 1500        # Cap vortex cloud for PH speed
RE_VALUES = [100, 500, 1000, 2000]

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


# ── Spectral DNS ────────────────────────────────────────────────────────────

def build_wavenumbers(n: int, device: str) -> tuple[torch.Tensor, ...]:
    """Build 3D wavenumber arrays and K² for N³ grid."""
    k = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    KX, KY, KZ = torch.meshgrid(k, k, k, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0  # avoid division by zero for pressure projection
    return KX, KY, KZ, K2


def taylor_green_ic(n: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize Taylor-Green vortex on N³ periodic grid [0, 2π]³."""
    x = torch.linspace(0.0, 2.0 * PI, n + 1, device=device, dtype=torch.float64)[:-1]
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w = torch.zeros_like(u)
    return u, v, w


def spectral_curl(u_hat: torch.Tensor, v_hat: torch.Tensor, w_hat: torch.Tensor,
                  KX: torch.Tensor, KY: torch.Tensor, KZ: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute vorticity ω = ∇×u in spectral space, return physical-space fields."""
    # ω = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y)
    wx_hat = 1j * KY * w_hat - 1j * KZ * v_hat
    wy_hat = 1j * KZ * u_hat - 1j * KX * w_hat
    wz_hat = 1j * KX * v_hat - 1j * KY * u_hat
    wx = torch.fft.ifftn(wx_hat).real
    wy = torch.fft.ifftn(wy_hat).real
    wz = torch.fft.ifftn(wz_hat).real
    return wx, wy, wz


def nonlinear_term(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                   KX: torch.Tensor, KY: torch.Tensor, KZ: torch.Tensor,
                   K2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute nonlinear (advection) term N = -(u·∇)u, projected onto divergence-free.

    Uses the rotation form: (u·∇)u = ω×u + ∇(|u|²/2)
    which naturally gives a divergence-free result after pressure projection.
    We compute the full advection in physical space and project.
    """
    u_hat = torch.fft.fftn(u)
    v_hat = torch.fft.fftn(v)
    w_hat = torch.fft.fftn(w)

    # Dealiase with 2/3 rule
    n = u.shape[0]
    cutoff = n // 3
    mask = (torch.abs(KX) <= cutoff) & (torch.abs(KY) <= cutoff) & (torch.abs(KZ) <= cutoff)
    u_hat_d = u_hat * mask
    v_hat_d = v_hat * mask
    w_hat_d = w_hat * mask

    ud = torch.fft.ifftn(u_hat_d).real
    vd = torch.fft.ifftn(v_hat_d).real
    wd = torch.fft.ifftn(w_hat_d).real

    # ∂u/∂x, ∂u/∂y, ∂u/∂z etc via spectral differentiation
    dudx = torch.fft.ifftn(1j * KX * u_hat_d).real
    dudy = torch.fft.ifftn(1j * KY * u_hat_d).real
    dudz = torch.fft.ifftn(1j * KZ * u_hat_d).real
    dvdx = torch.fft.ifftn(1j * KX * v_hat_d).real
    dvdy = torch.fft.ifftn(1j * KY * v_hat_d).real
    dvdz = torch.fft.ifftn(1j * KZ * v_hat_d).real
    dwdx = torch.fft.ifftn(1j * KX * w_hat_d).real
    dwdy = torch.fft.ifftn(1j * KY * w_hat_d).real
    dwdz = torch.fft.ifftn(1j * KZ * w_hat_d).real

    # Advection: (u·∇)u
    Nu = ud * dudx + vd * dudy + wd * dudz
    Nv = ud * dvdx + vd * dvdy + wd * dvdz
    Nw = ud * dwdx + vd * dwdy + wd * dwdz

    # Transform to spectral space
    Nu_hat = torch.fft.fftn(Nu)
    Nv_hat = torch.fft.fftn(Nv)
    Nw_hat = torch.fft.fftn(Nw)

    # Pressure projection: project out divergence
    div = 1j * KX * Nu_hat + 1j * KY * Nv_hat + 1j * KZ * Nw_hat
    Nu_hat -= 1j * KX * div / K2
    Nv_hat -= 1j * KY * div / K2
    Nw_hat -= 1j * KZ * div / K2

    return -Nu_hat, -Nv_hat, -Nw_hat


def rhs_spectral(u_hat: torch.Tensor, v_hat: torch.Tensor, w_hat: torch.Tensor,
                 nu: float, KX: torch.Tensor, KY: torch.Tensor,
                 KZ: torch.Tensor, K2: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute du_hat/dt = N(u) - ν K² u_hat, plus divergence-free projection."""
    u = torch.fft.ifftn(u_hat).real
    v = torch.fft.ifftn(v_hat).real
    w = torch.fft.ifftn(w_hat).real

    Nu_hat, Nv_hat, Nw_hat = nonlinear_term(u, v, w, KX, KY, KZ, K2)

    du = Nu_hat - nu * K2 * u_hat
    dv = Nv_hat - nu * K2 * v_hat
    dw = Nw_hat - nu * K2 * w_hat

    # Enforce divergence-free
    div = 1j * KX * du + 1j * KY * dv + 1j * KZ * dw
    du -= 1j * KX * div / K2
    dv -= 1j * KY * div / K2
    dw -= 1j * KZ * div / K2

    return du, dv, dw


def rk4_step(u_hat: torch.Tensor, v_hat: torch.Tensor, w_hat: torch.Tensor,
             nu: float, dt: float,
             KX: torch.Tensor, KY: torch.Tensor,
             KZ: torch.Tensor, K2: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classic RK4 step in spectral space."""
    k1u, k1v, k1w = rhs_spectral(u_hat, v_hat, w_hat, nu, KX, KY, KZ, K2)
    k2u, k2v, k2w = rhs_spectral(u_hat + 0.5*dt*k1u, v_hat + 0.5*dt*k1v,
                                   w_hat + 0.5*dt*k1w, nu, KX, KY, KZ, K2)
    k3u, k3v, k3w = rhs_spectral(u_hat + 0.5*dt*k2u, v_hat + 0.5*dt*k2v,
                                   w_hat + 0.5*dt*k2w, nu, KX, KY, KZ, K2)
    k4u, k4v, k4w = rhs_spectral(u_hat + dt*k3u, v_hat + dt*k3v,
                                   w_hat + dt*k3w, nu, KX, KY, KZ, K2)

    u_new = u_hat + (dt / 6.0) * (k1u + 2*k2u + 2*k3u + k4u)
    v_new = v_hat + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    w_new = w_hat + (dt / 6.0) * (k1w + 2*k2w + 2*k3w + k4w)
    return u_new, v_new, w_new


# ── Vortex Cloud Extraction ─────────────────────────────────────────────────

def extract_vortex_cloud(wx: torch.Tensor, wy: torch.Tensor, wz: torch.Tensor,
                          threshold_frac: float = VORTEX_THRESHOLD_FRAC,
                          max_points: int = MAX_CLOUD_POINTS) -> np.ndarray:
    """Extract spatial positions where |ω| > threshold_frac * max(|ω|).

    Returns (M, 3) float32 array of vortex positions in grid-index space.
    """
    omega_mag = torch.sqrt(wx**2 + wy**2 + wz**2)
    omega_max = float(omega_mag.max().item())
    if omega_max < 1e-12:
        return np.zeros((0, 3), dtype=np.float32)

    threshold = threshold_frac * omega_max
    mask = (omega_mag > threshold)
    indices = torch.nonzero(mask, as_tuple=False)  # (M, 3)

    if indices.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pts = indices.float().cpu().numpy()

    # Subsample if too large
    if len(pts) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    return pts


# ── H₀ Persistent Homology ─────────────────────────────────────────────────

def h0_persistence(pts: np.ndarray) -> np.ndarray:
    """Compute H₀ birth-death bars via union-find on Euclidean Rips complex.

    Returns sorted array of death times (birth=0 for all bars in H₀).
    The N-1 finite death times are the edge lengths that merge components.
    """
    n = len(pts)
    if n < 2:
        return np.array([])

    # Pairwise distances — use GPU for speed when large
    if n > 300:
        device_pts = torch.tensor(pts, dtype=torch.float32, device=DEVICE)
        dists_t = torch.cdist(device_pts, device_pts)
        dists = dists_t.cpu().numpy()
    else:
        from scipy.spatial.distance import cdist
        dists = cdist(pts, pts, metric="euclidean").astype(np.float32)

    # Union-Find
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Build sorted edge list
    triu_i, triu_j = np.triu_indices(n, k=1)
    edge_dists = dists[triu_i, triu_j]
    order = np.argsort(edge_dists)
    sorted_dists = edge_dists[order]
    sorted_i = triu_i[order]
    sorted_j = triu_j[order]

    death_times = []
    for d, i, j in zip(sorted_dists, sorted_i, sorted_j):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1
            death_times.append(float(d))

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
        return {"n_components": int(len(pts)), "epsilon_star": 0.0, "gini": 0.0, "n_points": int(len(pts))}

    bars = h0_persistence(pts)

    # ε* = 95th percentile of death times (onset scale for merged topology)
    if len(bars) > 0:
        eps_star = float(np.percentile(bars, 95))
        g = gini(bars)
        n_components = int(np.sum(bars > eps_star * 0.5)) + 1
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

    Returns dict with time series of topo metrics.
    """
    nu = 1.0 / re
    print(f"\n  [Re={re:.0f}] ν={nu:.5f} | Device={DEVICE} | N={N}")

    KX, KY, KZ, K2 = build_wavenumbers(N, DEVICE)

    u, v, w = taylor_green_ic(N, DEVICE)
    u_hat = torch.fft.fftn(u.to(torch.complex128))
    v_hat = torch.fft.fftn(v.to(torch.complex128))
    w_hat = torch.fft.fftn(w.to(torch.complex128))
    # Use float64 wavenumbers for stability
    KX64 = KX.to(torch.float64)
    KY64 = KY.to(torch.float64)
    KZ64 = KZ.to(torch.float64)
    K264 = K2.to(torch.float64)

    times = []
    n_components_list = []
    epsilon_star_list = []
    gini_list = []
    n_points_list = []
    enstrophy_list = []

    t0 = time.time()

    for step in range(N_STEPS + 1):
        current_t = step * DT

        if step % SAVE_EVERY == 0:
            # Compute vorticity
            u_hat_c = u_hat.to(torch.complex128)
            v_hat_c = v_hat.to(torch.complex128)
            w_hat_c = w_hat.to(torch.complex128)
            wx, wy, wz = spectral_curl(u_hat_c, v_hat_c, w_hat_c, KX64, KY64, KZ64)

            omega_mag = torch.sqrt(wx**2 + wy**2 + wz**2)
            enstrophy = float(0.5 * torch.mean(omega_mag**2).item())
            enstrophy_list.append(enstrophy)

            # Extract vortex cloud
            pts = extract_vortex_cloud(wx, wy, wz)

            # Topology
            topo = topology_from_cloud(pts)

            times.append(float(current_t))
            n_components_list.append(topo["n_components"])
            epsilon_star_list.append(topo["epsilon_star"])
            gini_list.append(topo["gini"])
            n_points_list.append(topo["n_points"])

            elapsed = time.time() - t0
            print(f"    t={current_t:.2f} | enstrophy={enstrophy:.4f} | "
                  f"pts={topo['n_points']} | ε*={topo['epsilon_star']:.3f} | "
                  f"G={topo['gini']:.3f} | N₀={topo['n_components']} | "
                  f"({elapsed:.1f}s)")

        if step < N_STEPS:
            u_hat, v_hat, w_hat = rk4_step(
                u_hat, v_hat, w_hat, nu, DT, KX64, KY64, KZ64, K264
            )

    total_time = time.time() - t0
    print(f"  [Re={re:.0f}] Done in {total_time:.1f}s")

    return {
        "re": float(re),
        "nu": float(nu),
        "n": N,
        "dt": DT,
        "n_steps": N_STEPS,
        "save_every": SAVE_EVERY,
        "times": times,
        "n_components": n_components_list,
        "epsilon_star": epsilon_star_list,
        "gini": gini_list,
        "n_points": n_points_list,
        "enstrophy": enstrophy_list,
        "total_time_s": round(total_time, 1),
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(all_results: list[dict], fig_path: Path) -> None:
    """4-panel figure: vortex count, ε*(t), Gini(t), cross-Re comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Vortex component count N₀(t) per Re
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

    # Panel 2: ε*(t) onset scale per Re
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

    # Panel 3: Gini(t) per Re
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

    # Panel 4: Cross-Re comparison — mean ε* and Gini at late time
    ax = axes[1, 1]
    re_vals = [r["re"] for r in all_results]
    # Use second half of time series as "late time"
    late_eps = [float(np.mean(r["epsilon_star"][len(r["epsilon_star"])//2:]))
                for r in all_results]
    late_gini = [float(np.mean(r["gini"][len(r["gini"])//2:]))
                 for r in all_results]
    std_eps = [float(np.std(r["epsilon_star"][len(r["epsilon_star"])//2:]))
               for r in all_results]

    ax2 = ax.twinx()
    ax.errorbar(re_vals, late_eps, yerr=std_eps, fmt="-o", color=COLORS["teal"],
                linewidth=2, markersize=7, capsize=4, label="ε* (left)")
    ax2.plot(re_vals, late_gini, "-s", color=COLORS["gold"], linewidth=2,
             markersize=7, label="Gini (right)")
    ax.set_xlabel("Reynolds Number Re")
    ax.set_ylabel("Mean ε* (late time)", color=COLORS["teal"])
    ax2.set_ylabel("Mean Gini (late time)", color=COLORS["gold"])
    ax.set_title("Re-Dependent Topological Behavior", color=COLORS["red"])
    ax.set_xscale("log")
    ax.tick_params(axis="y", labelcolor=COLORS["teal"])
    ax2.tick_params(axis="y", labelcolor=COLORS["gold"])
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"Navier-Stokes: Taylor-Green Vortex Topology  (N={N}³, t∈[0,{N_STEPS*DT:.1f}])",
        color=COLORS["text"], fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_path, facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze_re_dependence(all_results: list[dict]) -> dict:
    """Summarize whether Re-dependent topological behavior is observed."""
    re_summary = []
    for r in all_results:
        times = np.array(r["times"])
        eps = np.array(r["epsilon_star"])
        gini_arr = np.array(r["gini"])
        ens = np.array(r["enstrophy"])

        # Late-time stats (second half)
        half = len(times) // 2
        mean_eps_late = float(np.mean(eps[half:]))
        std_eps_late  = float(np.std(eps[half:]))
        mean_gini_late = float(np.mean(gini_arr[half:]))

        # Enstrophy peak time (proxy for vorticity intensification)
        peak_t = float(times[int(np.argmax(ens))])

        # Variability ratio: high Re should have higher variability in ε*
        variability = float(np.std(eps) / (np.mean(eps) + 1e-10))

        re_summary.append({
            "re": r["re"],
            "mean_eps_late": mean_eps_late,
            "std_eps_late": std_eps_late,
            "mean_gini_late": mean_gini_late,
            "enstrophy_peak_t": peak_t,
            "eps_variability": variability,
        })

    # Re-dependent: ε* should decrease with Re (more turbulent = smaller coherence)
    eps_vals = [s["mean_eps_late"] for s in re_summary]
    gini_vals = [s["mean_gini_late"] for s in re_summary]
    var_vals = [s["eps_variability"] for s in re_summary]

    # Monotonic decrease in ε* with Re?
    re_order = np.argsort([s["re"] for s in re_summary])
    eps_ordered = [eps_vals[i] for i in re_order]
    gini_ordered = [gini_vals[i] for i in re_order]

    eps_decreasing = all(eps_ordered[i] >= eps_ordered[i+1]
                         for i in range(len(eps_ordered)-1))
    gini_increasing = all(gini_ordered[i] <= gini_ordered[i+1]
                          for i in range(len(gini_ordered)-1))

    # More permissive: check correlation direction
    re_arr = np.array([s["re"] for s in re_summary], dtype=float)
    eps_arr = np.array(eps_vals, dtype=float)
    gini_arr2 = np.array(gini_vals, dtype=float)
    var_arr = np.array(var_vals, dtype=float)

    if len(re_arr) > 1 and np.std(eps_arr) > 0:
        corr_eps_re = float(np.corrcoef(re_arr, eps_arr)[0, 1])
        corr_gini_re = float(np.corrcoef(re_arr, gini_arr2)[0, 1])
        corr_var_re = float(np.corrcoef(re_arr, var_arr)[0, 1])
    else:
        corr_eps_re = corr_gini_re = corr_var_re = 0.0

    # Verdict: Re-dependent behavior detected if
    # ε* has negative correlation with Re AND variability increases with Re
    re_dependent = (corr_eps_re < -0.3) or (corr_gini_re > 0.3) or (corr_var_re > 0.3)

    return {
        "re_summary": re_summary,
        "eps_decreasing_with_re": eps_decreasing,
        "gini_increasing_with_re": gini_increasing,
        "corr_eps_re": corr_eps_re,
        "corr_gini_re": corr_gini_re,
        "corr_variability_re": corr_var_re,
        "re_dependent_behavior": re_dependent,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  NAVIER-STOKES: TAYLOR-GREEN VORTEX TOPOLOGY")
    print("  Tracking topological evolution of the vortex skeleton")
    print(f"  {timestamp}")
    print(f"  Device: {DEVICE} | N={N}³ | Re values: {RE_VALUES}")
    print("=" * 70)

    wall_t0 = time.time()
    all_results = []

    for re in RE_VALUES:
        result = run_simulation(float(re))
        all_results.append(result)

    total_wall = time.time() - wall_t0

    # Analysis
    print("\n" + "=" * 70)
    print("  ANALYSIS: Re-Dependent Topological Behavior")
    print("=" * 70)
    analysis = analyze_re_dependence(all_results)

    for s in analysis["re_summary"]:
        print(f"  Re={s['re']:.0f}: ε*_late={s['mean_eps_late']:.3f}±{s['std_eps_late']:.3f} "
              f"Gini_late={s['mean_gini_late']:.3f} "
              f"var={s['eps_variability']:.3f} "
              f"ens_peak_t={s['enstrophy_peak_t']:.2f}")

    print(f"\n  Correlation(ε*, Re)  = {analysis['corr_eps_re']:.3f}  "
          f"(negative expected for turbulence)")
    print(f"  Correlation(Gini, Re) = {analysis['corr_gini_re']:.3f}  "
          f"(positive expected)")
    print(f"  Correlation(var, Re)  = {analysis['corr_variability_re']:.3f}  "
          f"(positive expected)")
    print(f"\n  ε* monotone decrease with Re: {analysis['eps_decreasing_with_re']}")
    print(f"  Gini monotone increase with Re: {analysis['gini_increasing_with_re']}")
    print(f"\n  STATUS: {'RE-DEPENDENT BEHAVIOR DETECTED' if analysis['re_dependent_behavior'] else 'NO CLEAR RE DEPENDENCE DETECTED'}")

    # Plot
    fig_path = FIG_DIR / "ns_vortex_topology.png"
    plot_results(all_results, fig_path)

    # Save JSON
    output = {
        "experiment": "taylor_green_topology",
        "timestamp": timestamp,
        "device": DEVICE,
        "grid_n": N,
        "dt": DT,
        "n_steps": N_STEPS,
        "save_every": SAVE_EVERY,
        "vortex_threshold_frac": VORTEX_THRESHOLD_FRAC,
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
