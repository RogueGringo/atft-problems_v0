"""Typed sheaf Laplacian on the environment-typed cosmic-web k-NN graph.

Phase-1 rework: introduces asymmetric per-edge-type restriction matrices to
break the symmetric cancellation flaw (R_src = -R_dst = orthogonal) that
made v1 provably a graph-Laplacian no-op.

  R_src^t = I
  R_dst^t = lambda^t * Q^t

where Q^t is a deterministically-seeded orthogonal matrix per edge type and
lambda^t = dst_env_weight / src_env_weight is an ordinal transition prefactor
reflecting the physical distinctness of environment transitions.

Known limitations vs the full Opus rework spec:
  - Stalks remain random-orthogonal, not structured 3-grad + 4-env + 1-pad.
  - Edge types are now ordered pairs via graph.oriented_edge_type_for_pair,
    so void->wall and wall->void are distinct etype strings. The Laplacian
    rewrite in Task 6 consumes this ordering for structured R_dst.
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
