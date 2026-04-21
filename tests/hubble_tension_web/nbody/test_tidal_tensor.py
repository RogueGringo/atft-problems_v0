"""Tests for tidal-tensor T-web classification.

Three-tier coverage:
1. CIC kernel unit: point mass at cell center deposits mass 1 at that cell.
2. Spherical-void: interior cells see all eigvals positive => Environment.VOID.
3. End-to-end on the committed fixture: void-center cell classified as VOID.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from problems.hubble_tension_web.nbody import tidal_tensor
from problems.hubble_tension_web.nbody.mdpl2_fetch import load_halo_catalog
from problems.hubble_tension_web.types import Environment


FIXTURE = Path(__file__).parent / "fixtures" / "mini_mdpl2.parquet"


def test_cic_deposit_conserves_total_mass():
    positions = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
    masses = np.array([1.0])
    rho = tidal_tensor.cic_deposit(positions, masses, n_grid=16, box_mpc=10.0)
    assert np.isclose(rho.sum(), 1.0, atol=1e-9)


def test_cic_deposit_single_point_at_cell_center():
    """A point exactly at a cell center goes entirely into that cell."""
    positions = np.array([[3.5, 3.5, 3.5]], dtype=np.float64)
    masses = np.array([1.0])
    rho = tidal_tensor.cic_deposit(positions, masses, n_grid=8, box_mpc=8.0)
    assert np.isclose(rho[3, 3, 3], 1.0, atol=1e-9)
    total_off_cell = rho.sum() - rho[3, 3, 3]
    assert np.isclose(total_off_cell, 0.0, atol=1e-9)


def test_classify_spherical_void_center_is_void():
    """Planted void: an empty interior sphere => all eigvals of tidal tensor positive
    at interior cells => Environment.VOID."""
    rng = np.random.default_rng(0)
    n = 2000
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.uniform(20.0, 24.0, size=n)
    positions = 25.0 + u * r[:, None]
    masses = np.ones(n)
    env_grid, _ = tidal_tensor.classify(
        positions=positions, masses=masses, n_grid=32, box_mpc=50.0, lambda_th=0.0,
    )
    assert env_grid[16, 16, 16] == 0, (
        f"center cell of a spherical void should be VOID; got code {env_grid[16,16,16]}"
    )


def test_classify_returns_valid_codes_only():
    """Every cell code is in {0, 1, 2, 3}."""
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    assert env_grid.dtype == np.uint8
    assert env_grid.min() >= 0 and env_grid.max() <= 3


def test_code_to_env_lookup_is_consistent():
    """The CODE_TO_ENV table must return genuine Environment enum members."""
    assert tidal_tensor.CODE_TO_ENV[0] is Environment.VOID
    assert tidal_tensor.CODE_TO_ENV[1] is Environment.WALL
    assert tidal_tensor.CODE_TO_ENV[2] is Environment.FILAMENT
    assert tidal_tensor.CODE_TO_ENV[3] is Environment.NODE


def test_fixture_void_center_classified_as_void():
    """The committed fixture has a planted void at (25, 25, 25). The cell
    containing that point must be classified Environment.VOID."""
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, meta = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    cx = int(25.0 / halos.box_mpc * 32)
    cy, cz = cx, cx
    assert env_grid[cx, cy, cz] == 0, (
        f"fixture void center classified as code {env_grid[cx,cy,cz]} "
        f"(expected 0/VOID); meta={meta}"
    )
