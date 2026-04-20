"""Accuracy and fallback tests for the Arnoldi-based summarize_spectrum.

Contract (spec 2026-04-20 §Step 2):
  - spectrum[:k_spec] matches the dense eigvalsh result to rel < 1e-6.
  - lambda_min (smallest nonzero eigenvalue) matches to rel < 1e-9.
  - ArpackNoConvergence triggers dense fallback with a UserWarning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _build_small_web(seed: int, n: int = 60):
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 10, size=(n, 3))
    envs = rng.choice(list(Environment), size=n).tolist()
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n_, edges = build_typed_graph(web, k=6)
    return web, n_, edges


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_eigsh_bottom_k_matches_dense(seed: int) -> None:
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    web, n, edges = _build_small_web(seed, n=60)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments,
    )

    L_dense = L.toarray()
    w_dense = np.sort(np.linalg.eigvalsh(L_dense))
    bottom_ref = w_dense[:16]

    summary = summarize_spectrum(
        L=L, n_nodes=n, edges=edges, positions=web.positions, k_spec=16,
    )
    bottom_got = np.sort(summary.spectrum)

    # rel < 1e-6 bound per spec §Step 2. Use atol=1e-10 for near-zero kernel
    # eigenvalues where "relative" is meaningless.
    np.testing.assert_allclose(bottom_got, bottom_ref, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_eigsh_lambda_min_matches_dense(seed: int) -> None:
    """Tighter rel<1e-9 contract on the spectral gap (f_topo depends on it)."""
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    web, n, edges = _build_small_web(seed, n=60)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments,
    )

    L_dense = L.toarray()
    w_dense = np.sort(np.linalg.eigvalsh(L_dense))
    nonzero = w_dense[w_dense > 1e-6]
    lambda_min_ref = float(nonzero[0]) if nonzero.size > 0 else 1e-6

    summary = summarize_spectrum(
        L=L, n_nodes=n, edges=edges, positions=web.positions, k_spec=16,
    )
    rel = abs(summary.lambda_min - lambda_min_ref) / max(abs(lambda_min_ref), 1e-24)
    assert rel < 1e-9, (
        f"lambda_min rel error {rel:.2e} exceeds spec bound 1e-9; "
        f"got {summary.lambda_min}, ref {lambda_min_ref}"
    )


def test_eigsh_fallback_to_dense_on_arpack_failure(monkeypatch) -> None:
    """If eigsh raises ArpackNoConvergence, summarize_spectrum falls back to dense.

    Monkeypatch scipy.sparse.linalg.eigsh to raise ArpackNoConvergence, then
    confirm the function still returns a valid SpectralSummary (via the dense
    path) and emits a UserWarning.
    """
    import scipy.sparse.linalg as _spla
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum
    from problems.hubble_tension_web.types import SpectralSummary

    web, n, edges = _build_small_web(0, n=30)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments,
    )

    def _always_fail(*args, **kwargs):
        # ArpackNoConvergence requires (msg, eigenvalues, eigenvectors)
        raise _spla.ArpackNoConvergence("forced failure for test", np.array([]), np.array([]))

    import problems.hubble_tension_web.spectrum as _spec_mod
    monkeypatch.setattr(_spec_mod, "eigsh", _always_fail)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = summarize_spectrum(
            L=L, n_nodes=n, edges=edges, positions=web.positions, k_spec=8,
        )

    assert isinstance(summary, SpectralSummary)
    assert summary.spectrum.shape == (8,)
    assert summary.lambda_min > 0
    fallback_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "eigsh" in str(w.message).lower()
    ]
    assert len(fallback_warnings) >= 1, (
        f"expected UserWarning mentioning eigsh fallback; got {[str(w.message) for w in caught]}"
    )
