"""Typed sheaf Laplacian with density-gradient stalks.

Stalk layout (stalk_dim=8):
  coords 0-2 : unit density-gradient direction ĝ_v
  coords 3-6 : environment one-hot (void, wall, filament, node)
  coord  7   : pad (fixed at 0)

Edge restriction maps (oriented edge src→dst, edge type t = (env_src, env_dst)):
  R_src^t = I_8
  R_dst^t = λ^t · (Rot_3(ĝ_src → ĝ_dst) ⊕ P^t_4 ⊕ I_1)

where:
  Rot_3(a → b) is the Rodrigues rotation sending unit vector a to unit vector b,
  with parallel/antiparallel edge-case handling.
  P^t_4 is the 4x4 permutation swapping env_src one-hot with env_dst one-hot
  (identity when env_src == env_dst).
  λ^t is the EDGE_TYPE_LAMBDA prefactor (ordinal physical prior).

L_F = δ^T δ for the coboundary δ. PSD and symmetric by construction.

This module currently holds:
  - Task-5 additions: STALK_DIM, _ENV_INDEX, EDGE_TYPE_LAMBDA, build_stalk_init.
  - Phase-1 legacy: _stable_orthogonal, _seed_from_etype, _ENV_WEIGHTS, and a
    typed_sheaf_laplacian that uses random-orthogonal R_dst. Task 6 replaces
    that legacy Laplacian with the structured (Rot_3 ⊕ P^t_4 ⊕ I_1) form.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import KDTree

from problems.hubble_tension_web.types import Environment, LocalCosmicWeb

STALK_DIM: int = 8
GRADIENT_FLOOR: float = 1e-9
_ENV_INDEX: Dict[str, int] = {e.value: i for i, e in enumerate(Environment)}
# _ENV_INDEX: {"void":0, "wall":1, "filament":2, "node":3}

EDGE_TYPE_LAMBDA: Dict[Tuple[str, str], float] = {
    ("void", "void"):         1.0,
    ("wall", "wall"):         1.0,
    ("filament", "filament"): 1.0,
    ("node", "node"):         1.0,
    ("void", "wall"):         1.5, ("wall", "void"):         1.5,
    ("void", "filament"):     1.8, ("filament", "void"):     1.8,
    ("void", "node"):         2.2, ("node", "void"):         2.2,
    ("wall", "filament"):     1.2, ("filament", "wall"):     1.2,
    ("wall", "node"):         1.5, ("node", "wall"):         1.5,
    ("filament", "node"):     1.1, ("node", "filament"):     1.1,
}
# Ordinal physical prior, not calibrated. See REWORK spec §3.3.


def _lambda_for_etype(etype: str) -> float:
    """etype is 'env_src-env_dst'; look up in EDGE_TYPE_LAMBDA."""
    src, dst = etype.split("-", 1)
    return EDGE_TYPE_LAMBDA[(src, dst)]


def build_stalk_init(
    web: LocalCosmicWeb,
    *,
    h_mpc: float | None = None,
    k_density: int = 8,
) -> Tuple[np.ndarray, List[bool]]:
    """Construct initial stalks: (N, STALK_DIM) array + per-node degeneracy flags.

    Density estimate: simple k-NN distance inverse (like the v1 synthetic typing).
    Gradient: finite-difference weighted by 1/|Δx|² over the same k-NN neighborhood.
    If |∇ρ| < GRADIENT_FLOOR, stalk coords 0-2 default to ê_z and the node is flagged.

    h_mpc is accepted for future smoothing-length use; currently ignored in favor
    of the KDTree k_density-neighbor estimate. Left in the signature so callers
    can pass it once we revisit kernel density estimation.
    """
    N = web.positions.shape[0]
    stalks = np.zeros((N, STALK_DIM), dtype=np.float64)
    flags: List[bool] = [False] * N

    tree = KDTree(web.positions)
    dists, nbr_idx = tree.query(web.positions, k=k_density + 1)
    dists = dists[:, 1:]
    nbr_idx = nbr_idx[:, 1:]

    mean_d = dists.mean(axis=1)
    rho = 1.0 / (mean_d + 1e-12)

    for v in range(N):
        dx = web.positions[nbr_idx[v]] - web.positions[v]
        d_sq = np.sum(dx * dx, axis=1)
        d_sq = np.maximum(d_sq, 1e-12)
        d_rho = rho[nbr_idx[v]] - rho[v]
        grad = (d_rho[:, None] * dx / d_sq[:, None]).sum(axis=0)
        g_norm = float(np.linalg.norm(grad))
        if g_norm < GRADIENT_FLOOR:
            stalks[v, 0:3] = np.array([0.0, 0.0, 1.0])
            flags[v] = True
        else:
            stalks[v, 0:3] = grad / g_norm

        env_val = web.environments[v].value
        stalks[v, 3 + _ENV_INDEX[env_val]] = 1.0

    return stalks, flags


def _stable_orthogonal(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(M)
    return Q


def _seed_from_etype(etype: str) -> int:
    h = 0
    for c in etype:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    return h


# Ordinal environment weights for the lambda prefactor.
_ENV_WEIGHTS = {"void": 1.0, "wall": 2.0, "filament": 3.0, "node": 4.0}


def typed_sheaf_laplacian(
    positions: np.ndarray,
    n: int,
    edges: List[Tuple[int, int, str]],
    stalk_dim: int = 8,
    rng_seed: int = 0,
) -> np.ndarray:
    # 1. Asymmetric per-edge-type restriction matrices.
    R_src_cache: Dict[str, np.ndarray] = {}
    R_dst_cache: Dict[str, np.ndarray] = {}

    for _, _, etype in edges:
        if etype not in R_src_cache:
            R_src_cache[etype] = np.eye(stalk_dim)

            seed = _seed_from_etype(etype) ^ rng_seed
            Q = _stable_orthogonal(seed, stalk_dim)

            lam = 1.0
            if "-" in etype:
                parts = etype.split("-")
                if len(parts) == 2:
                    src_w = _ENV_WEIGHTS.get(parts[0], 1.0)
                    dst_w = _ENV_WEIGHTS.get(parts[1], 1.0)
                    lam = dst_w / src_w

            R_dst_cache[etype] = lam * Q

    # 2. Coboundary delta: (m*stalk_dim, n*stalk_dim)
    m = len(edges)
    delta = np.zeros((m * stalk_dim, n * stalk_dim))
    for eidx, (s, d, etype) in enumerate(edges):
        R_src = R_src_cache[etype]
        R_dst = R_dst_cache[etype]

        row = slice(eidx * stalk_dim, (eidx + 1) * stalk_dim)
        col_s = slice(s * stalk_dim, (s + 1) * stalk_dim)
        col_d = slice(d * stalk_dim, (d + 1) * stalk_dim)

        delta[row, col_s] = -R_src
        delta[row, col_d] = R_dst

    # 3. L = delta^T delta, symmetrized to kill float asymmetry.
    L = delta.T @ delta
    L = 0.5 * (L + L.T)
    return L
