"""Synthetic LTB-family cosmic-web generator.

Produces a uniform Poisson point cloud with a radial density suppression
implementing a top-hat void of depth delta and radius R, centered at the
box midpoint. Each point is typed by local density via k-NN estimate:
  lowest tercile      -> VOID
  middle lower third  -> WALL
  middle upper third  -> FILAMENT
  highest tercile     -> NODE
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from problems.hubble_tension_web.types import Environment, LocalCosmicWeb, VoidParameters


def generate_synthetic_void(
    params: VoidParameters,
    n_points: int,
    box_mpc: float,
    rng_seed: int = 0,
) -> LocalCosmicWeb:
    rng = np.random.default_rng(rng_seed)

    center = np.full(3, box_mpc / 2.0)
    R_outer = box_mpc / 2.0  # confine to inscribed sphere so density formulas are consistent

    # 1. Rejection-sample points uniform in the sphere of radius R_outer.
    #    Apply void suppression inside params.R_mpc.
    pts: list[np.ndarray] = []
    batch_size = max(n_points * 10, 10_000)
    while len(pts) < n_points:
        candidates = rng.uniform(0.0, box_mpc, size=(batch_size, 3))
        r = np.linalg.norm(candidates - center, axis=1)

        # Keep only within the inscribed sphere
        in_sphere = r < R_outer
        candidates = candidates[in_sphere]
        r = r[in_sphere]

        # 2. Rejection: inside void radius, keep with probability (1 + delta).
        in_void = r < params.R_mpc
        keep_prob = np.where(in_void, 1.0 + params.delta, 1.0)
        keep_prob = np.clip(keep_prob, 0.0, 1.0)
        u = rng.uniform(0.0, 1.0, size=len(candidates))
        accepted = candidates[u < keep_prob]
        pts.extend(accepted.tolist())

    positions = np.array(pts[:n_points])

    # 3. Environment typing by local density (k-NN inverse mean distance).
    tree = cKDTree(positions)
    k = 8
    dists, _ = tree.query(positions, k=k + 1)
    local_density = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-9)
    quartiles = np.quantile(local_density, [0.25, 0.5, 0.75])
    env_for = np.where(
        local_density < quartiles[0], Environment.VOID,
        np.where(
            local_density < quartiles[1], Environment.WALL,
            np.where(local_density < quartiles[2], Environment.FILAMENT, Environment.NODE),
        ),
    )
    environments = env_for.tolist()

    return LocalCosmicWeb(positions=positions, environments=environments)
