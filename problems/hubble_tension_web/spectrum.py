"""Spectrum + persistent-Betti summary of a typed sheaf Laplacian.

β0: connected-component count of the graph backbone (union-find).
β1: PERSISTENT first Betti number via Vietoris-Rips filtration on node positions,
    lifetime-thresholded at τ_persist · ℓ̄ (ℓ̄ = mean k-NN edge length).
spectrum: first k_spec smallest eigenvalues via dense eigvalsh.
λ_min: smallest non-zero eigenvalue (spectral gap).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import warnings

from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy import sparse as _sparse

try:
    from ripser import ripser as _ripser
except ImportError as _err:  # pragma: no cover
    _ripser = None
    _RIPSER_ERR = _err
else:
    _RIPSER_ERR = None

from problems.hubble_tension_web.types import SpectralSummary

TAU_PERSIST: float = 1.5   # lifetime multiplier; persistent H1 classes must satisfy
                           # death - birth > TAU_PERSIST * ell_bar.
TAU_MAX: float = 6.0       # VR filtration cap: eps_max = TAU_MAX * ell_bar.


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


def _mean_knn_edge_length(edges: List[Tuple[int, int, str]], positions: np.ndarray) -> float:
    if not edges:
        return 1.0
    lengths = [float(np.linalg.norm(positions[s] - positions[d])) for s, d, _ in edges]
    return float(np.mean(lengths))


def persistent_beta1(
    positions: np.ndarray,
    *,
    tau_persist: float = TAU_PERSIST,
    tau_max: float = TAU_MAX,
    ell_bar: float | None = None,
    edges_for_ell: List[Tuple[int, int, str]] | None = None,
) -> int:
    """Compute β1_persistent via VR filtration.

    ell_bar: if not given, estimate from edges_for_ell; else from a single nearest-neighbor
             pass on the positions.
    """
    if _ripser is None:
        raise RuntimeError(
            f"ripser not installed: {_RIPSER_ERR}. "
            "`pip install ripser` or fall back to gudhi."
        )
    if ell_bar is None:
        if edges_for_ell is not None and len(edges_for_ell) > 0:
            ell_bar = _mean_knn_edge_length(edges_for_ell, positions)
        else:
            from scipy.spatial import KDTree
            tree = KDTree(positions)
            sample = min(100, len(positions))
            d, _ = tree.query(positions[:sample], k=2)
            ell_bar = float(d[:, 1].mean())

    thresh = tau_max * ell_bar
    result = _ripser(positions, maxdim=1, thresh=thresh)
    dgm1 = result["dgms"][1]
    if dgm1.size == 0:
        return 0
    # Classes whose true death exceeds the VR cap appear with death=inf; they are MORE
    # persistent than classes that die under thresh, not less. Treat them as having
    # lifetime `thresh - birth` (a strict lower bound on their true lifetime) so we
    # don't drop genuine global cycles just because the filtration was capped.
    births = dgm1[:, 0]
    deaths = np.where(np.isfinite(dgm1[:, 1]), dgm1[:, 1], thresh)
    lifetimes = deaths - births
    return int(np.sum(lifetimes > tau_persist * ell_bar))


def summarize_spectrum(
    *,
    L,
    n_nodes: int,
    edges: List[Tuple[int, int, str]],
    positions: np.ndarray,
    k_spec: int = 16,
    zero_tol: float = 1e-6,
    tau_persist: float = TAU_PERSIST,
    tau_max: float = TAU_MAX,
) -> SpectralSummary:
    # Accept dense ndarray or scipy.sparse matrix.
    k_arnoldi = k_spec + 4
    # Typed sheaf Laplacians have triple-degenerate eigenvalues from the
    # stalk structure; default ARPACK Krylov subspace (ncv = 2k+1) is too
    # narrow to resolve them reliably. ncv = 3*k_arnoldi covers the clusters.
    ncv = max(3 * k_arnoldi, 40)

    w_all: np.ndarray
    if _sparse.issparse(L):
        # eigsh requires k < n; if the matrix is tiny and k_spec+4 >= n,
        # just do a dense solve. Also guard ncv >= n.
        n_dim = L.shape[0]
        if k_arnoldi >= n_dim or ncv >= n_dim:
            w_all = np.sort(np.linalg.eigvalsh(L.toarray()))
        else:
            try:
                # Shift-invert at small negative sigma: L is PSD (singular),
                # so sigma=0 would blow up the LU factor; sigma=-1e-6 is
                # below the kernel but still concentrates convergence on
                # the bottom of the spectrum.
                w, _ = eigsh(
                    L, k=k_arnoldi, sigma=-1e-6, which="LM", tol=1e-8, ncv=ncv,
                )
                w_all = np.sort(w)
            except ArpackNoConvergence:
                warnings.warn(
                    "eigsh failed to converge (likely degenerate kernel); "
                    "falling back to dense eigvalsh. This is slow for large L.",
                    UserWarning,
                    stacklevel=2,
                )
                w_all = np.sort(np.linalg.eigvalsh(L.toarray()))
    else:
        # Dense input — preserve the exact old behavior for legacy callers.
        w_all = np.sort(np.linalg.eigvalsh(L))

    spectrum = w_all[:k_spec].copy()

    beta0 = _connected_components(n_nodes, edges)
    beta1 = persistent_beta1(
        positions,
        tau_persist=tau_persist,
        tau_max=tau_max,
        edges_for_ell=edges,
    )

    nonzero = w_all[w_all > zero_tol]
    lambda_min = float(nonzero[0]) if nonzero.size > 0 else float(zero_tol)

    return SpectralSummary(
        spectrum=spectrum,
        beta0=int(beta0),
        beta1=int(beta1),
        lambda_min=lambda_min,
    )
