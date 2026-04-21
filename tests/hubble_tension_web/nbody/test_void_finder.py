"""Tests for the local-minimum sphere-growth void finder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from problems.hubble_tension_web.nbody import tidal_tensor, void_finder
from problems.hubble_tension_web.nbody.mdpl2_fetch import load_halo_catalog


FIXTURE = Path(__file__).parent / "fixtures" / "mini_mdpl2.parquet"


def test_find_planted_void_in_synthetic_grid():
    """Place halos in a uniform background + excluded sphere; void finder must
    locate the sphere center within 2 grid cells."""
    rng = np.random.default_rng(0)
    box_mpc = 50.0
    n_grid = 32
    positions = rng.uniform(0.0, box_mpc, size=(4000, 3))
    r = np.linalg.norm(positions - 25.0, axis=1)
    positions = positions[r > 10.0]
    masses = np.ones(positions.shape[0])
    rho = tidal_tensor.cic_deposit(positions, masses, n_grid=n_grid, box_mpc=box_mpc)

    voids = void_finder.find_voids(
        rho=rho, box_mpc=box_mpc,
        smoothing_mpc=5.0, delta_threshold=-0.2, max_radius_mpc=20.0,
        k_top=5,
    )
    assert len(voids) >= 1
    best = voids[0]
    center_err = np.linalg.norm(np.array(best.center_mpc) - 25.0)
    cell_mpc = box_mpc / n_grid
    assert center_err < 2.0 * cell_mpc, (
        f"void center off by {center_err:.2f} Mpc (> 2 cells = {2*cell_mpc:.2f} Mpc)"
    )
    assert best.delta_eff < 0.0
    assert best.radius_mpc >= 5.0


def test_find_voids_returns_at_most_k():
    rho = np.ones((16, 16, 16), dtype=np.float64)
    rho[4, 4, 4] = 0.01
    rho[10, 10, 10] = 0.01
    rho[4, 10, 4] = 0.01
    voids = void_finder.find_voids(
        rho=rho, box_mpc=16.0,
        smoothing_mpc=0.5, delta_threshold=-0.2, max_radius_mpc=5.0,
        k_top=2,
    )
    assert len(voids) <= 2


def test_find_voids_on_fixture_returns_nonempty():
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    rho = tidal_tensor.cic_deposit(
        halos.positions, halos.masses, n_grid=32, box_mpc=halos.box_mpc,
    )
    voids = void_finder.find_voids(
        rho=rho, box_mpc=halos.box_mpc,
        smoothing_mpc=4.0, delta_threshold=-0.1, max_radius_mpc=20.0,
        k_top=5,
    )
    assert len(voids) >= 1, "committed fixture has a planted void; finder must locate it"


def test_voidcandidate_fields_populated():
    rho = np.ones((8, 8, 8), dtype=np.float64)
    rho[4, 4, 4] = 0.01
    voids = void_finder.find_voids(
        rho=rho, box_mpc=8.0,
        smoothing_mpc=0.5, delta_threshold=-0.2, max_radius_mpc=3.0,
        k_top=1,
    )
    assert len(voids) == 1
    v = voids[0]
    assert len(v.center_mpc) == 3
    assert v.radius_mpc > 0
    assert v.delta_eff <= 0.0
    assert v.n_halos_enclosed_estimate >= 0
