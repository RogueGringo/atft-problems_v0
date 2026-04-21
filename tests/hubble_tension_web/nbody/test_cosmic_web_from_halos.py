"""Tests for cosmic-web assembly from halos + T-web + void candidate."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from problems.hubble_tension_web.nbody import (
    cosmic_web_from_halos,
    tidal_tensor,
    void_finder,
)
from problems.hubble_tension_web.nbody.mdpl2_fetch import load_halo_catalog
from problems.hubble_tension_web.types import (
    Environment,
    LocalCosmicWeb,
    VoidParameters,
)


FIXTURE = Path(__file__).parent / "fixtures" / "mini_mdpl2.parquet"


def test_assemble_from_candidate_returns_pair():
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    candidate = void_finder.VoidCandidate(
        center_mpc=(25.0, 25.0, 25.0),
        radius_mpc=15.0,
        delta_eff=-0.3,
        n_halos_enclosed_estimate=5,
        score=4.5,
    )
    web, params = cosmic_web_from_halos.assemble(
        halos=halos, env_grid=env_grid, candidate=candidate,
    )
    assert isinstance(web, LocalCosmicWeb)
    assert isinstance(params, VoidParameters)
    assert params.delta == candidate.delta_eff
    assert params.R_mpc == candidate.radius_mpc


def test_positions_are_void_center_relative():
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    candidate = void_finder.VoidCandidate(
        center_mpc=(25.0, 25.0, 25.0),
        radius_mpc=15.0,
        delta_eff=-0.3,
        n_halos_enclosed_estimate=5,
        score=4.5,
    )
    web, _ = cosmic_web_from_halos.assemble(
        halos=halos, env_grid=env_grid, candidate=candidate,
    )
    r = np.linalg.norm(web.positions, axis=1)
    assert np.all(r <= candidate.radius_mpc + 1e-6), (
        f"halos must be within R={candidate.radius_mpc}; max r={r.max()}"
    )


def test_environments_are_enum_instances():
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    candidate = void_finder.VoidCandidate(
        center_mpc=(25.0, 25.0, 25.0),
        radius_mpc=15.0,
        delta_eff=-0.3,
        n_halos_enclosed_estimate=5,
        score=4.5,
    )
    web, _ = cosmic_web_from_halos.assemble(
        halos=halos, env_grid=env_grid, candidate=candidate,
    )
    assert all(isinstance(e, Environment) for e in web.environments)


def test_empty_void_returns_empty_web_gracefully():
    """A void with radius 0.1 Mpc (smaller than one cell) may contain zero halos.
    Assembler should return an empty LocalCosmicWeb without crashing."""
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    candidate = void_finder.VoidCandidate(
        center_mpc=(49.0, 49.0, 49.0),
        radius_mpc=0.1,
        delta_eff=-0.5,
        n_halos_enclosed_estimate=0,
        score=0.05,
    )
    web, params = cosmic_web_from_halos.assemble(
        halos=halos, env_grid=env_grid, candidate=candidate,
    )
    assert web.positions.shape[0] == 0
    assert len(web.environments) == 0
    assert params.R_mpc == 0.1


def test_void_params_signs_respected():
    """delta_eff must be <= 0 (VoidParameters validator enforces this)."""
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    import pytest
    bad = void_finder.VoidCandidate(
        center_mpc=(25.0, 25.0, 25.0),
        radius_mpc=5.0,
        delta_eff=0.3,
        n_halos_enclosed_estimate=10,
        score=1.5,
    )
    with pytest.raises(ValueError, match="delta"):
        cosmic_web_from_halos.assemble(
            halos=halos, env_grid=env_grid, candidate=bad,
        )
