"""Simple density-field void finder.

Algorithm (per spec):
  1. Smooth the density field at ~10 Mpc with a Gaussian filter.
  2. Find local minima via scipy.ndimage.minimum_filter.
  3. For each minimum, grow a spherical region in 1-cell steps until either
     the enclosed density contrast delta_eff < threshold (default -0.2) fails
     OR max_radius is reached. Final radius = last step where delta_eff held.
  4. Rank by depth * radius and return top K.

v1 intentionally avoids watershed approaches (VIDE, ZOBOV). See spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter


@dataclass
class VoidCandidate:
    """A candidate void with its center, effective radius, and depth."""
    center_mpc: tuple[float, float, float]
    radius_mpc: float
    delta_eff: float
    n_halos_enclosed_estimate: int  # rough count; full count done downstream
    score: float                    # depth * radius for ranking


def _local_minima_indices(
    rho_smooth: np.ndarray,
    *,
    neighborhood_size: int = 3,
) -> np.ndarray:
    """Return (M, 3) integer grid indices where rho_smooth is a local minimum."""
    mins = minimum_filter(rho_smooth, size=neighborhood_size)
    mask = (rho_smooth == mins)
    mins_wide = minimum_filter(rho_smooth, size=max(neighborhood_size + 2, 5))
    mask &= (rho_smooth <= mins_wide + 1e-12)
    idx = np.argwhere(mask)
    return idx


def _grow_sphere(
    rho: np.ndarray,
    center_cell: np.ndarray,
    *,
    mean_rho: float,
    cell_mpc: float,
    delta_threshold: float,
    max_radius_cells: int,
) -> tuple[int, float, int]:
    """Grow a sphere in 1-cell radial steps around center_cell in rho.

    Returns:
      best_r_cells: final radius in cells where delta_eff < delta_threshold still held.
                    0 if even a 1-cell radius fails the threshold.
      delta_eff:    enclosed density contrast at best_r_cells.
      n_cells:      cell count inside best_r_cells.
    """
    n = rho.shape[0]
    half = max_radius_cells
    rng = np.arange(-half, half + 1)
    dx, dy, dz = np.meshgrid(rng, rng, rng, indexing="ij")
    dist2 = dx * dx + dy * dy + dz * dz

    cx, cy, cz = int(center_cell[0]), int(center_cell[1]), int(center_cell[2])
    ix = (cx + rng) % n
    iy = (cy + rng) % n
    iz = (cz + rng) % n
    sub = rho[np.ix_(ix, iy, iz)]

    best_r = 0
    best_delta = 0.0
    best_n = 0
    for r in range(1, max_radius_cells + 1):
        mask = dist2 < r * r
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        mean_inside = float(sub[mask].mean())
        delta_eff = (mean_inside / mean_rho) - 1.0 if mean_rho > 0 else 0.0
        if delta_eff < delta_threshold:
            best_r = r
            best_delta = delta_eff
            best_n = n_cells
        else:
            break
    return best_r, best_delta, best_n


def find_voids(
    *,
    rho: np.ndarray,
    box_mpc: float,
    smoothing_mpc: float = 10.0,
    delta_threshold: float = -0.2,
    max_radius_mpc: float = 100.0,
    k_top: int = 5,
) -> list[VoidCandidate]:
    """Locate up to k_top void candidates in the density grid rho.

    Args:
      rho:              (N, N, N) float64 density grid (mass per cell, arbitrary norm).
      box_mpc:          cube side length in Mpc.
      smoothing_mpc:    Gaussian smoothing scale for the minimum finder.
      delta_threshold:  density contrast threshold (default -0.2 per KBC).
      max_radius_mpc:   cap on sphere growth.
      k_top:            return at most k_top candidates, sorted by depth*radius.
    """
    n_grid = rho.shape[0]
    cell_mpc = box_mpc / n_grid
    sigma_cells = smoothing_mpc / cell_mpc
    rho_smooth = gaussian_filter(rho, sigma=sigma_cells, mode="wrap")

    mean_rho = float(rho.mean())
    if mean_rho <= 0.0:
        return []

    minima = _local_minima_indices(rho_smooth)
    max_radius_cells = max(1, int(np.ceil(max_radius_mpc / cell_mpc)))

    candidates: list[VoidCandidate] = []
    for m in minima:
        r_cells, delta_eff, n_cells = _grow_sphere(
            rho, m,
            mean_rho=mean_rho,
            cell_mpc=cell_mpc,
            delta_threshold=delta_threshold,
            max_radius_cells=max_radius_cells,
        )
        if r_cells <= 0:
            continue
        radius_mpc = r_cells * cell_mpc
        depth = max(-delta_eff, 0.0)
        score = depth * radius_mpc
        candidates.append(VoidCandidate(
            center_mpc=(
                float((m[0] + 0.5) * cell_mpc),
                float((m[1] + 0.5) * cell_mpc),
                float((m[2] + 0.5) * cell_mpc),
            ),
            radius_mpc=radius_mpc,
            delta_eff=float(delta_eff),
            n_halos_enclosed_estimate=int(round(n_cells * mean_rho * (1.0 + delta_eff))),
            score=float(score),
        ))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:k_top]
