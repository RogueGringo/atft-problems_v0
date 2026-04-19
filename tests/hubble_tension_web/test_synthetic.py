import numpy as np
import pytest


def test_generate_synthetic_void_returns_local_cosmic_web():
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.types import VoidParameters, Environment, LocalCosmicWeb
    params = VoidParameters(delta=-0.2, R_mpc=300.0)
    web = generate_synthetic_void(params, n_points=2000, box_mpc=800.0, rng_seed=42)
    assert isinstance(web, LocalCosmicWeb)
    assert web.positions.shape == (2000, 3)
    assert len(web.environments) == 2000
    assert all(isinstance(e, Environment) for e in web.environments)


def test_void_depth_reflected_in_inner_density():
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.types import VoidParameters
    params = VoidParameters(delta=-0.3, R_mpc=200.0)
    web = generate_synthetic_void(params, n_points=5000, box_mpc=800.0, rng_seed=0)
    r = np.linalg.norm(web.positions - 400.0, axis=1)   # center at box midpoint
    inner = (r < 200.0).sum()
    outer = (r > 300.0).sum()
    # Inner density per unit volume should be ~ (1 + delta) = 0.7 of outer
    inner_density = inner / ((4/3) * np.pi * 200.0**3)
    outer_density = outer / ((4/3) * np.pi * (400.0**3 - 300.0**3))
    ratio = inner_density / outer_density
    assert 0.55 < ratio < 0.85   # tolerance for Poisson fluctuation


def test_smooth_limit_generator_has_no_void():
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.types import VoidParameters
    params = VoidParameters(delta=0.0, R_mpc=200.0)   # no under-density
    web = generate_synthetic_void(params, n_points=3000, box_mpc=800.0, rng_seed=1)
    r = np.linalg.norm(web.positions - 400.0, axis=1)
    inner = (r < 200.0).sum()
    outer = (r > 300.0).sum()
    inner_density = inner / ((4/3) * np.pi * 200.0**3)
    outer_density = outer / ((4/3) * np.pi * (400.0**3 - 300.0**3))
    assert 0.85 < inner_density / outer_density < 1.15
