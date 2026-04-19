"""Typed sheaf Laplacian on the environment-typed cosmic-web k-NN graph.

Per-edge-type orthogonal restriction matrices, deterministically seeded by the
edge-type string. Standard construction:  L = delta^T delta, where delta is
the signed coboundary applying the edge's restriction on each endpoint block.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


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


def typed_sheaf_laplacian(
    positions: np.ndarray,
    n: int,
    edges: List[Tuple[int, int, str]],
    stalk_dim: int = 8,
    rng_seed: int = 0,
) -> np.ndarray:
    # 1. Per-edge-type orthogonal restriction matrices.
    R_cache: Dict[str, np.ndarray] = {}
    for _, _, etype in edges:
        if etype not in R_cache:
            R_cache[etype] = _stable_orthogonal(_seed_from_etype(etype) ^ rng_seed, stalk_dim)

    # 2. Coboundary delta: (m*stalk_dim, n*stalk_dim)
    m = len(edges)
    delta = np.zeros((m * stalk_dim, n * stalk_dim))
    for eidx, (s, d, etype) in enumerate(edges):
        R = R_cache[etype]
        row = slice(eidx * stalk_dim, (eidx + 1) * stalk_dim)
        col_s = slice(s * stalk_dim, (s + 1) * stalk_dim)
        col_d = slice(d * stalk_dim, (d + 1) * stalk_dim)
        delta[row, col_s] = -R
        delta[row, col_d] = R

    # 3. L = delta^T delta, symmetrized to kill float asymmetry.
    L = delta.T @ delta
    L = 0.5 * (L + L.T)
    return L
