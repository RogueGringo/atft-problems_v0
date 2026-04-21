"""Tests for the Parquet halo-cache reader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from problems.hubble_tension_web.nbody import NBodyDataNotAvailable
from problems.hubble_tension_web.nbody import mdpl2_fetch


FIXTURE = Path(__file__).parent / "fixtures" / "mini_mdpl2.parquet"


def test_fixture_is_committed_and_readable():
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    df = pd.read_parquet(FIXTURE)
    for col in mdpl2_fetch.EXPECTED_COLUMNS:
        assert col in df.columns, f"schema contract broken: {col} missing"


def test_load_halo_catalog_returns_ndarrays():
    halos = mdpl2_fetch.load_halo_catalog(FIXTURE, mass_cut=0.0)
    assert halos.positions.shape[1] == 3
    assert halos.positions.dtype == np.float64  # promoted from float32 internally
    assert halos.masses.ndim == 1
    assert halos.positions.shape[0] == halos.masses.shape[0]
    assert halos.box_mpc > 0


def test_mass_cut_applied():
    halos_all = mdpl2_fetch.load_halo_catalog(FIXTURE, mass_cut=0.0)
    halos_cut = mdpl2_fetch.load_halo_catalog(FIXTURE, mass_cut=1e12)
    # Every retained halo must be above the cut.
    assert np.all(halos_cut.masses >= 1e12)
    # The cut is not a no-op (log-normal sigma=0.3 around 10^12 gives ~50% below).
    assert halos_cut.positions.shape[0] < halos_all.positions.shape[0]


def test_missing_file_raises_nbody_data_not_available(tmp_path):
    with pytest.raises(NBodyDataNotAvailable):
        mdpl2_fetch.load_halo_catalog(tmp_path / "does_not_exist.parquet", mass_cut=0.0)


def test_schema_violation_raises(tmp_path):
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"x": [1.0], "y": [2.0]}).to_parquet(bad)  # missing z, mvir
    with pytest.raises(ValueError, match="schema"):
        mdpl2_fetch.load_halo_catalog(bad, mass_cut=0.0)


def test_network_fetch_is_stubbed():
    with pytest.raises(NotImplementedError, match="CosmoSim"):
        mdpl2_fetch.fetch_from_network(url="https://cosmosim.example/mdpl2/z0_500Mpc.dat")


from unittest.mock import patch


def test_network_fetch_delegates_when_enabled(tmp_path, monkeypatch):
    """With ATFT_MDPL2_DOWNLOAD_ENABLED='1', fetch_from_network defers to
    mdpl2_download.fetch_sub_box rather than raising NotImplementedError."""
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")

    dest = tmp_path / "deleg.parquet"
    with patch(
        "problems.hubble_tension_web.nbody.mdpl2_download.fetch_sub_box",
        return_value=dest,
    ) as mock_fetch:
        result = mdpl2_fetch.fetch_from_network(
            url="https://cosmosim.example/mdpl2/z0_500Mpc.dat",
            dest=dest,
        )
    assert result == dest
    mock_fetch.assert_called_once()
    _, kwargs = mock_fetch.call_args
    assert kwargs.get("dest") == dest


def test_network_fetch_still_stubbed_when_disabled(monkeypatch):
    """Redundant with test_network_fetch_is_stubbed but pins the env-off branch."""
    monkeypatch.delenv("ATFT_MDPL2_DOWNLOAD_ENABLED", raising=False)
    with pytest.raises(NotImplementedError, match="CosmoSim"):
        mdpl2_fetch.fetch_from_network(url="https://cosmosim.example/mdpl2/z0_500Mpc.dat")
