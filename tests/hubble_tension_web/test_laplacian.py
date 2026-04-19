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


def test_typed_vs_untyped_spectrum_differs():
    """I1 regression guard: the Laplacian spectrum MUST respond to edge typing.

    Build the same graph twice: once with all-identical environment types (untyped
    collapse), once with a mix. Median eigenvalue must differ by > 1e-3 relative.
    """
    import numpy as np
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian

    rng = np.random.default_rng(7)
    positions = rng.uniform(0, 10, size=(60, 3))

    envs_typed = rng.choice(list(Environment), size=60).tolist()
    envs_untyped = [Environment.VOID] * 60

    web_typed = LocalCosmicWeb(positions=positions, environments=envs_typed)
    web_untyped = LocalCosmicWeb(positions=positions, environments=envs_untyped)

    n_t, edges_t = build_typed_graph(web_typed, k=6)
    n_u, edges_u = build_typed_graph(web_untyped, k=6)

    L_typed   = typed_sheaf_laplacian(positions=positions, n=n_t, edges=edges_t, stalk_dim=8, rng_seed=0)
    L_untyped = typed_sheaf_laplacian(positions=positions, n=n_u, edges=edges_u, stalk_dim=8, rng_seed=0)

    w_t = np.sort(np.linalg.eigvalsh(L_typed))
    w_u = np.sort(np.linalg.eigvalsh(L_untyped))

    med_t = float(np.median(w_t))
    med_u = float(np.median(w_u))
    rel = abs(med_t - med_u) / max(abs(med_u), 1e-12)
    assert rel > 1e-3, (
        f"Typed and untyped Laplacian medians indistinguishable (rel={rel:.2e}). "
        f"This is the v1 no-op regression guard — the Laplacian must respond to typing."
    )


def test_nullity_drops_under_typing():
    """Corollary of I1: nullity(L_F_typed) < stalk_dim * β₀ when Rot or P is nontrivial.

    In untyped collapse, nullspace dim = stalk_dim · β₀ (constant sections lift per-dimension).
    In the typed case with at least one edge having non-identity Rot or P, the nullity drops
    (generic constant sections are no longer in the kernel).
    """
    import numpy as np
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian

    stalk_dim = 8
    rng = np.random.default_rng(11)
    positions = rng.uniform(0, 10, size=(40, 3))
    envs = rng.choice(list(Environment), size=40).tolist()
    # Ensure at least two environments present so typing is nontrivial.
    if len({e for e in envs}) < 2:
        envs[0] = Environment.VOID
        envs[1] = Environment.NODE
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=stalk_dim, rng_seed=0)

    w = np.linalg.eigvalsh(L)
    # Graph is almost certainly connected at k=6, N=40, so β₀ = 1.
    # Untyped expectation: nullity = stalk_dim * 1 = 8.
    # Typed expectation: nullity < 8.
    nullity = int(np.sum(w < 1e-6))
    assert nullity < stalk_dim, (
        f"nullity(L_F_typed)={nullity} did not drop below stalk_dim={stalk_dim}; "
        f"typing is not producing nontrivial restriction maps."
    )
