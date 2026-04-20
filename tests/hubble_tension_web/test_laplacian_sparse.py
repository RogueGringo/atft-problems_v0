"""Sparse-coboundary equivalence tests for typed_sheaf_laplacian.

Contract (spec 2026-04-20 §Step 1):
  typed_sheaf_laplacian(...) must return scipy.sparse.csr_matrix whose .toarray()
  value matches the dense reference implementation to rel < 1e-12 on at least
  three independent random webs. The dense reference is computed inline here
  (copied from the pre-Task-1 implementation) so the test does not depend on the
  module under change.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest
from scipy import sparse


def _dense_reference_laplacian(positions, n, edges, environments):
    """Reference dense assembly — mirrors the pre-Task-1 typed_sheaf_laplacian body.

    Uses the CURRENT (post-Task-1) helpers (_R_dst_for_edge, build_stalk_init,
    STALK_DIM) because those are math-frozen and their outputs are bit-stable.
    """
    from problems.hubble_tension_web.laplacian import (
        STALK_DIM, _R_dst_for_edge, build_stalk_init,
    )
    from problems.hubble_tension_web.types import Environment, LocalCosmicWeb

    web = LocalCosmicWeb(positions=positions, environments=environments)
    stalks, _ = build_stalk_init(web)
    g = stalks[:, 0:3]

    m = len(edges)
    delta = np.zeros((m * STALK_DIM, n * STALK_DIM), dtype=np.float64)
    for e_idx, (s, d, etype) in enumerate(edges):
        env_s, env_d = etype.split("-", 1)
        R_dst = _R_dst_for_edge(g[s], g[d], env_s, env_d)
        row = slice(e_idx * STALK_DIM, (e_idx + 1) * STALK_DIM)
        col_s = slice(s * STALK_DIM, (s + 1) * STALK_DIM)
        col_d = slice(d * STALK_DIM, (d + 1) * STALK_DIM)
        delta[row, col_s] = -np.eye(STALK_DIM)
        delta[row, col_d] = R_dst
    L = delta.T @ delta
    return 0.5 * (L + L.T)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_sparse_laplacian_matches_dense(seed: int) -> None:
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian

    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 10, size=(40, 3))
    envs = rng.choice(list(Environment), size=40).tolist()
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)

    L_sparse = typed_sheaf_laplacian(
        positions=positions, n=n, edges=edges, stalk_dim=8,
        environments=web.environments,
    )
    assert sparse.issparse(L_sparse), "typed_sheaf_laplacian must return a sparse matrix"
    assert L_sparse.format == "csr", f"expected csr format, got {L_sparse.format}"

    L_dense_from_sparse = L_sparse.toarray()
    L_ref = _dense_reference_laplacian(positions, n, edges, envs)

    assert L_dense_from_sparse.shape == L_ref.shape
    np.testing.assert_allclose(L_dense_from_sparse, L_ref, rtol=1e-12, atol=1e-12)


def test_sparse_laplacian_density_is_approximately_two_over_n() -> None:
    """Spec §Step 1 note: each delta row has exactly 2*STALK_DIM=16 nonzeros.

    After L = delta.T @ delta, the density of L is bounded above by the density
    of delta^2 (order of magnitude). We check it is << 1 to catch accidental
    densification (e.g. someone doing `.toarray()` internally).
    """
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian

    rng = np.random.default_rng(123)
    positions = rng.uniform(0, 10, size=(80, 3))
    envs = rng.choice(list(Environment), size=80).tolist()
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)

    L = typed_sheaf_laplacian(
        positions=positions, n=n, edges=edges, stalk_dim=8,
        environments=web.environments,
    )
    total = L.shape[0] * L.shape[1]
    density = L.nnz / total
    assert density < 0.10, (
        f"L density {density:.3f} suggests accidental densification; expected < 0.10"
    )
