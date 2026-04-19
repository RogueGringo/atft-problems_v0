import numpy as np
import pytest


def _setup_small_web(rng_seed: int, n: int = 50):
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    rng = np.random.default_rng(rng_seed)
    positions = rng.uniform(0, 10, size=(n, 3))
    envs = [Environment.VOID if i < n // 2 else Environment.WALL for i in range(n)]
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n_, edges = build_typed_graph(web, k=6)
    return web, n_, edges


def test_summarize_spectrum_returns_spectral_summary():
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum
    from problems.hubble_tension_web.types import SpectralSummary

    web, n, edges = _setup_small_web(0, n=50)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges,
        stalk_dim=STALK_DIM, environments=web.environments,
    )
    summary = summarize_spectrum(
        L=L, n_nodes=n, edges=edges, positions=web.positions, k_spec=16,
    )
    assert isinstance(summary, SpectralSummary)
    assert summary.spectrum.shape == (16,)
    assert summary.beta0 >= 1
    assert summary.beta1 >= 0
    assert summary.lambda_min > 0


def test_two_disconnected_clusters_give_beta0_at_least_two():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    b = np.array([[100, 0, 0], [101, 0, 0], [100, 1, 0]], dtype=float)
    positions = np.vstack([a, b])
    envs = [Environment.VOID] * 6
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=2)
    L = typed_sheaf_laplacian(
        positions=positions, n=n, edges=edges,
        stalk_dim=STALK_DIM, environments=envs,
    )
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, positions=positions, k_spec=8)
    assert summary.beta0 >= 2


def test_beta1_persistent_small_on_homogeneous_cloud():
    """Uniform Poisson cloud should give a small β1_persistent (finite-sample noise floor).

    Spec I2: β1_persistent/N < 0.05 on N=1000 uniform Poisson.
    """
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    N = 1000
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, size=(N, 3))
    envs = [Environment.VOID] * N
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=8)
    L = typed_sheaf_laplacian(
        positions=positions, n=n, edges=edges,
        stalk_dim=STALK_DIM, environments=envs,
    )
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, positions=positions, k_spec=8)
    assert summary.beta1 / N < 0.05, (
        f"beta1_persistent={summary.beta1} / N={N} = {summary.beta1/N:.4f} >= 0.05. "
        "Uniform Poisson cloud should produce O(1) noise floor, not a large β1."
    )


def test_beta1_persistent_nonzero_on_ring_cloud():
    """Points sampled on a ring should produce at least one persistent H1 class."""
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    N = 300
    rng = np.random.default_rng(42)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    positions = np.stack([
        5.0 * np.cos(theta) + 0.05 * rng.standard_normal(N),
        5.0 * np.sin(theta) + 0.05 * rng.standard_normal(N),
        0.05 * rng.standard_normal(N),
    ], axis=1)
    envs = [Environment.VOID] * N
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)
    L = typed_sheaf_laplacian(
        positions=positions, n=n, edges=edges,
        stalk_dim=STALK_DIM, environments=envs,
    )
    summary = summarize_spectrum(L=L, n_nodes=n, edges=edges, positions=positions, k_spec=8)
    assert summary.beta1 >= 1, (
        f"Ring cloud should give at least one persistent H1 class; got beta1={summary.beta1}."
    )
