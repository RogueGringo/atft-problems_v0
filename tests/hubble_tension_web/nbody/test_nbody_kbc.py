"""Smoke test for the end-to-end nbody_kbc experiment on the fixture."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


FIXTURE = Path("tests/hubble_tension_web/nbody/fixtures/mini_mdpl2.parquet")
OUT = Path("problems/hubble_tension_web/results/nbody_kbc.json")


def test_nbody_kbc_runs_on_fixture(tmp_path, monkeypatch):
    """Run the experiment module with the fixture Parquet as the cache file."""
    out_path = tmp_path / "nbody_kbc.json"
    env_overrides = {
        "ATFT_NBODY_CACHE_FILE": str(FIXTURE.resolve()),
        "ATFT_NBODY_GRID": "32",
        "ATFT_NBODY_OUTPUT_JSON": str(out_path),
        "ATFT_NBODY_K_VOIDS": "3",
    }
    import os
    env = os.environ.copy()
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-m", "problems.hubble_tension_web.experiments.nbody_kbc"],
        env=env, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout:\n{result.stdout.decode(errors='replace')}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    assert "grid_N" in data and data["grid_N"] == 32
    assert "K" in data and data["K"] == 3
    assert "voids" in data and isinstance(data["voids"], list)
    assert "beta1_distribution" in data
    dist = data["beta1_distribution"]
    assert "count_nonzero" in dist and "count_total" in dist
    assert dist["count_total"] == len(data["voids"])

    if data["voids"]:
        v0 = data["voids"][0]
        for field in ("idx", "center_mpc", "N_halos", "delta_eff", "R_eff_mpc",
                      "beta0", "beta1_persistent", "lambda_min",
                      "delta_H0_total", "kinematic_term", "topological_term",
                      "f_topo_at_alpha_1", "ltb_anchor_at_delta_R", "y_residual"):
            assert field in v0, f"missing per-void field: {field}"
        assert isinstance(v0["beta1_persistent"], int)
        assert v0["beta1_persistent"] >= 0  # 0 is a valid outcome per spec
        assert isinstance(v0["f_topo_at_alpha_1"], float)
        assert isinstance(v0["ltb_anchor_at_delta_R"], float)
        assert isinstance(v0["y_residual"], float)
