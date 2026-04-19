import numpy as np
import pytest


def test_summarize_spectrum_returns_spectral_summary():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment, SpectralSummary
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 10, size=(50, 3))
    envs = [Environment.VOID if i < 25 else Environment.WALL for i in range(50)]
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=8, rng_seed=0)

    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, k_spec=16)

    assert isinstance(summary, SpectralSummary)
    assert summary.spectrum.shape == (16,)
    assert summary.beta0 >= 1
    assert summary.beta1 >= 0
    assert summary.lambda_min > 0


def test_two_disconnected_clusters_give_beta0_at_least_two():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    b = np.array([[100, 0, 0], [101, 0, 0], [100, 1, 0]], dtype=float)
    positions = np.vstack([a, b])
    envs = [Environment.VOID] * 6
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=2)
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=8, rng_seed=0)
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, k_spec=8)
    assert summary.beta0 >= 2
