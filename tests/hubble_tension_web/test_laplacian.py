import numpy as np
import pytest


def test_typed_sheaf_laplacian_is_symmetric_psd():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 10, size=(40, 3))
    envs = rng.choice(list(Environment), size=40).tolist()
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=4, rng_seed=0)
    assert np.allclose(L, L.T, atol=1e-8)
    w = np.linalg.eigvalsh(L)
    assert w.min() > -1e-8


def test_laplacian_dimension_is_n_times_stalk_dim():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 5, size=(12, 3))
    envs = [Environment.VOID] * 12
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=3)
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=5, rng_seed=0)
    assert L.shape == (12 * 5, 12 * 5)
