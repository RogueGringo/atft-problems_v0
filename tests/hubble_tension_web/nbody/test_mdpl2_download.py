"""Tests for the CosmoSim MDPL2 download layer.

All network I/O is mocked. The live path is exercised manually by the maintainer
with ATFT_MDPL2_DOWNLOAD_ENABLED=1 and a real ATFT_SCISERVER_TOKEN; not in CI.
"""
from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from problems.hubble_tension_web.nbody import mdpl2_download
from problems.hubble_tension_web.nbody import mdpl2_fetch


# Sample CosmoSim CSV response (h-units, Planck h=0.6777).
# Row 1: exact h-to-physical round-trip anchor: 677.7 h^-1 Mpc -> 999.85... Mpc.
# Row 2: zero-position anchor (must stay zero under h-division).
# Row 3: mid-mass halo so the mass_cut / mvir_phys relationship is exercised.
MOCK_CSV = """x,y,z,Mvir
677.7,0.0,0.0,6.777e11
0.0,0.0,0.0,1.3554e12
100.0,200.0,300.0,2e12
"""


def test_fetch_sub_box_writes_parquet_with_correct_schema(tmp_path, monkeypatch):
    """Mocked HTTP round-trip produces Parquet matching EXPECTED_COLUMNS."""
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")
    monkeypatch.setenv("ATFT_SCISERVER_TOKEN", "fake-token")

    dest = tmp_path / "mdpl2_z0_500Mpc.parquet"

    # Mock the three-phase HTTP: submit -> poll -> fetch CSV.
    with patch.object(mdpl2_download, "_submit_sql_job", return_value="job-123"), \
         patch.object(mdpl2_download, "_poll_until_done", return_value=None), \
         patch.object(mdpl2_download, "_download_csv", return_value=MOCK_CSV):
        produced = mdpl2_download.fetch_sub_box(dest=dest, sub_box_mpc=500.0)

    assert produced == dest
    assert dest.exists(), f"expected Parquet at {dest}"

    df = pd.read_parquet(dest)
    for col in mdpl2_fetch.EXPECTED_COLUMNS:
        assert col in df.columns, f"schema contract broken: {col} missing"
    assert len(df) == 3


def test_fetch_sub_box_applies_h_conversion(tmp_path, monkeypatch):
    """Input x=677.7 h^-1 Mpc, h=0.6777 -> output x close to 1000 Mpc.

    Exact: 677.7 / 0.6777 = 1000.0 (the spec's round-trip anchor).
    """
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")
    monkeypatch.setenv("ATFT_SCISERVER_TOKEN", "fake-token")
    dest = tmp_path / "mdpl2.parquet"

    with patch.object(mdpl2_download, "_submit_sql_job", return_value="job-123"), \
         patch.object(mdpl2_download, "_poll_until_done", return_value=None), \
         patch.object(mdpl2_download, "_download_csv", return_value=MOCK_CSV):
        mdpl2_download.fetch_sub_box(dest=dest, sub_box_mpc=500.0)

    df = pd.read_parquet(dest)

    # Row 1: x=677.7 h^-1 Mpc -> 1000.0 Mpc, mvir=6.777e11 h^-1 M_sun -> 1e12 M_sun.
    # h_MDPL2 = 0.6777 exactly; 677.7 / 0.6777 = 1000.0 to machine precision.
    h = mdpl2_download.MDPL2_H
    assert h == pytest.approx(0.6777, abs=1e-12)

    np.testing.assert_allclose(df.loc[0, "x"], 677.7 / h, atol=1e-9)
    assert abs(df.loc[0, "x"] - 1000.0) < 1e-9, (
        f"h-conversion anchor off: got x={df.loc[0, 'x']}, expected 1000.0"
    )
    assert df.loc[0, "y"] == 0.0
    assert df.loc[0, "z"] == 0.0
    assert abs(df.loc[0, "mvir"] - 1e12) < 1e-3, (
        f"h-conversion mvir anchor off: got {df.loc[0, 'mvir']}, expected 1e12"
    )

    # Row 2 is at origin; must stay at origin.
    assert df.loc[1, "x"] == 0.0 and df.loc[1, "y"] == 0.0 and df.loc[1, "z"] == 0.0


def test_fetch_sub_box_gated_on_enable_flag(tmp_path, monkeypatch):
    """With ATFT_MDPL2_DOWNLOAD_ENABLED != '1', fetch_sub_box must refuse."""
    monkeypatch.delenv("ATFT_MDPL2_DOWNLOAD_ENABLED", raising=False)
    from problems.hubble_tension_web.nbody import NBodyDataNotAvailable
    with pytest.raises(NBodyDataNotAvailable, match="ATFT_MDPL2_DOWNLOAD_ENABLED"):
        mdpl2_download.fetch_sub_box(dest=tmp_path / "x.parquet")


def test_fetch_sub_box_requires_token_when_enabled(tmp_path, monkeypatch):
    """With ATFT_MDPL2_DOWNLOAD_ENABLED='1' but no token, must raise with a clear message."""
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")
    monkeypatch.delenv("ATFT_SCISERVER_TOKEN", raising=False)
    from problems.hubble_tension_web.nbody import NBodyDataNotAvailable
    with pytest.raises(NBodyDataNotAvailable, match="ATFT_SCISERVER_TOKEN"):
        mdpl2_download.fetch_sub_box(dest=tmp_path / "x.parquet")


def test_fetch_sub_box_applies_sub_box_filter_via_sql(monkeypatch):
    """The sub_box_mpc argument flows into the SQL body as an h^-1 Mpc bound."""
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")
    monkeypatch.setenv("ATFT_SCISERVER_TOKEN", "fake-token")

    captured_sql: list[str] = []

    def fake_submit(*, url: str, sql: str, token: str) -> str:
        captured_sql.append(sql)
        return "job-xyz"

    with patch.object(mdpl2_download, "_submit_sql_job", side_effect=fake_submit), \
         patch.object(mdpl2_download, "_poll_until_done", return_value=None), \
         patch.object(mdpl2_download, "_download_csv", return_value=MOCK_CSV):
        # Use an in-memory dest via tmp-dir equivalent: we only care the SQL was built.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            mdpl2_download.fetch_sub_box(dest=Path(td) / "x.parquet", sub_box_mpc=500.0)

    assert captured_sql, "SQL job was never submitted"
    sql = captured_sql[0]
    # Sub-box filter in h^-1 Mpc: 500 Mpc * h = 500 * 0.6777 = 338.85 h^-1 Mpc.
    # We don't pin an exact format; just that the bound appears.
    assert "338" in sql, f"expected h-unit sub-box bound in SQL, got: {sql!r}"
    # Host-halo filter per spec § three-sharp-findings + standard Rockstar idiom.
    assert "pid" in sql, f"expected pid=-1 host-halo filter in SQL, got: {sql!r}"


def test_override_sql_env_var(tmp_path, monkeypatch):
    """ATFT_MDPL2_SQL wholesale-overrides the query string."""
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")
    monkeypatch.setenv("ATFT_SCISERVER_TOKEN", "fake-token")
    monkeypatch.setenv("ATFT_MDPL2_SQL", "SELECT x, y, z, Mvir FROM custom_table WHERE 1=1")

    captured: list[str] = []

    def fake_submit(*, url: str, sql: str, token: str) -> str:
        captured.append(sql)
        return "j"

    with patch.object(mdpl2_download, "_submit_sql_job", side_effect=fake_submit), \
         patch.object(mdpl2_download, "_poll_until_done", return_value=None), \
         patch.object(mdpl2_download, "_download_csv", return_value=MOCK_CSV):
        mdpl2_download.fetch_sub_box(dest=tmp_path / "x.parquet", sub_box_mpc=500.0)

    assert captured[0] == "SELECT x, y, z, Mvir FROM custom_table WHERE 1=1"
