"""Spectrum + persistent-Betti summary of a typed sheaf Laplacian.

β0 = graph connected component count via union-find.
β1 = graph cycle space dim = len(edges) - n_nodes + β0.
spectrum = first k_spec smallest eigenvalues of L_F (dense eigvalsh).
lambda_min = smallest non-zero eigenvalue (spectral gap).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from problems.hubble_tension_web.types import SpectralSummary


def _connected_components(n: int, edges: List[Tuple[int, int, str]]) -> int:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for s, d, _ in edges:
        union(s, d)

    return len({find(x) for x in range(n)})


def summarize_spectrum(
    *,
    L: np.ndarray,
    n_nodes: int,
    edges: List[Tuple[int, int, str]],
    k_spec: int = 16,
    zero_tol: float = 1e-6,
) -> SpectralSummary:
    w = np.linalg.eigvalsh(L)
    w = np.sort(w)
    spectrum = w[:k_spec].copy()

    beta0 = _connected_components(n_nodes, edges)
    beta1 = max(len(edges) - n_nodes + beta0, 0)

    nonzero = w[w > zero_tol]
    lambda_min = float(nonzero[0]) if nonzero.size > 0 else float(zero_tol)

    return SpectralSummary(
        spectrum=spectrum,
        beta0=int(beta0),
        beta1=int(beta1),
        lambda_min=lambda_min,
    )
