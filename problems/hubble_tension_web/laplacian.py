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
  - Task-6 (landed): STALK_DIM, _ENV_INDEX, EDGE_TYPE_LAMBDA, build_stalk_init,
    _rodrigues_rotation, _env_permutation_4x4, _R_dst_for_edge, and the
    structured typed_sheaf_laplacian with R_dst^t = λ^t · (Rot_3 ⊕ P^t_4 ⊕ I_1).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse as _sparse
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
    k_density = min(k_density, N - 1)
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


def _rodrigues_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix sending unit vector a to unit vector b.

    Edge cases:
      parallel   (a ≈ b):              return I_3
      antiparallel (a ≈ -b):           π-rotation about a deterministic perpendicular axis
      generic:                         standard Rodrigues formula.
    """
    dot = float(np.dot(a, b))
    if dot > 1.0 - 1e-9:
        return np.eye(3)
    if dot < -1.0 + 1e-9:
        ez = np.array([0.0, 0.0, 1.0])
        ex = np.array([1.0, 0.0, 0.0])
        axis = np.cross(ez, a)
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(ex, a)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = dot
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _env_permutation_4x4(env_src: str, env_dst: str) -> np.ndarray:
    """4x4 permutation swapping env_src one-hot with env_dst one-hot.

    Identity when env_src == env_dst. Otherwise, swaps the two corresponding
    one-hot coordinates; leaves the other two coordinates as identity.
    """
    P = np.eye(4)
    i = _ENV_INDEX[env_src]
    j = _ENV_INDEX[env_dst]
    if i == j:
        return P
    P[i, i] = 0.0
    P[j, j] = 0.0
    P[i, j] = 1.0
    P[j, i] = 1.0
    return P


def _R_dst_for_edge(
    g_src: np.ndarray,
    g_dst: np.ndarray,
    env_src: str,
    env_dst: str,
) -> np.ndarray:
    """Construct R_dst^t = λ^t · (Rot ⊕ P ⊕ I_1) as an (8,8) block-diagonal matrix."""
    lam = EDGE_TYPE_LAMBDA[(env_src, env_dst)]
    rot = _rodrigues_rotation(g_src, g_dst)
    perm = _env_permutation_4x4(env_src, env_dst)
    R = np.zeros((STALK_DIM, STALK_DIM))
    R[0:3, 0:3] = rot
    R[3:7, 3:7] = perm
    R[7, 7] = 1.0
    return lam * R


def typed_sheaf_laplacian(
    *,
    positions: np.ndarray,
    n: int,
    edges: List[Tuple[int, int, str]],
    stalk_dim: int = STALK_DIM,
    rng_seed: int = 0,              # unused; kept for signature compatibility
    environments: List[Environment] | None = None,
) -> "_sparse.csr_matrix":
    """Assemble L_F = δ^T δ with density-gradient typed restriction maps.

    R_src^t = I_8, R_dst^t = λ^t · (Rot_3(ĝ_s→ĝ_d) ⊕ P^t_4 ⊕ I_1).

    `environments` is optional: if omitted, per-node environments are reconstructed
    from the edge-type strings (each edge carries 'env_src-env_dst' in src<dst index
    order, so env_of[s] = parts[0], env_of[d] = parts[1] is unambiguous).

    Requires stalk_dim == STALK_DIM == 8. The gradient/env layout is not adjustable.
    """
    if stalk_dim != STALK_DIM:
        raise ValueError(
            f"typed_sheaf_laplacian requires stalk_dim={STALK_DIM}; got {stalk_dim}. "
            "The gradient/env layout is not adjustable."
        )

    if environments is None:
        env_of: List[str | None] = [None] * n
        for s, d, etype in edges:
            e_s, e_d = etype.split("-", 1)
            env_of[s] = e_s
            env_of[d] = e_d
        if any(e is None for e in env_of):
            raise ValueError(
                "Could not infer environments for all nodes from edges; "
                "pass environments=web.environments explicitly."
            )
        env_values = env_of                                          # type: ignore[assignment]
    else:
        env_values = [e.value for e in environments]

    from problems.hubble_tension_web.types import LocalCosmicWeb as _LCW
    envs_enum = [Environment(v) for v in env_values]
    web = _LCW(positions=positions, environments=envs_enum)
    stalks, _flags = build_stalk_init(web)
    g = stalks[:, 0:3]

    m = len(edges)
    # Sparse coboundary assembly: each delta row has exactly 2*STALK_DIM=16
    # nonzeros (the -I block on col_s and the R_dst block on col_d). lil_matrix
    # allows efficient row-slice assignment during construction; we convert to
    # csr before the transpose/matmul.
    delta = _sparse.lil_matrix(
        (m * STALK_DIM, n * STALK_DIM), dtype=np.float64,
    )
    neg_I = -np.eye(STALK_DIM)
    for e_idx, (s, d, etype) in enumerate(edges):
        env_s, env_d = etype.split("-", 1)
        R_dst = _R_dst_for_edge(g[s], g[d], env_s, env_d)
        row0 = e_idx * STALK_DIM
        col_s0 = s * STALK_DIM
        col_d0 = d * STALK_DIM
        delta[row0:row0 + STALK_DIM, col_s0:col_s0 + STALK_DIM] = neg_I
        delta[row0:row0 + STALK_DIM, col_d0:col_d0 + STALK_DIM] = R_dst

    delta_csr = delta.tocsr()
    L = (delta_csr.T @ delta_csr).tocsr()
    L = (0.5 * (L + L.T)).tocsr()
    return L
