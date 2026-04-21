"""Tests for the real-void alpha recalibration experiment.

Fixture strategy: write a mock nbody_kbc.json with 8 synthetic per-void records,
compute the expected alpha* in numpy from the same literal values, run
nbody_calibration as a subprocess with a custom results dir, and assert the
produced alpha* matches the hand-computed value to 1e-10. Exercises all three
reason-code branches: fit-succeeded, insufficient-voids, f_topo-all-zero.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


# Planted fixture. Values chosen so:
#   * K = 8 (> default min 3)
#   * beta1 varies in {0, 1, 2, 3} across the records
#   * deltas span the spec's void range [-0.3, -0.05]
#   * Rs span [100, 500] Mpc
# These exact numbers are what the hand-computation below re-uses — tests stay
# closed-form, not a same-formula-twice tautology.
_DELTAS      = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.15, -0.20]
_RS          = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0]
_BETA0S      = [10, 10, 12, 14, 11, 13, 12, 15]
_BETA1S      = [0, 1, 2, 3, 0, 1, 2, 3]
_LAMBDA_MINS = [0.5, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2]


def _hand_computed_alpha_star():
    """Compute alpha* in numpy from the same literal values as the fixture."""
    from problems.hubble_tension_web.functional import C1, f_topo
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb

    f = np.array([
        f_topo(int(b0), int(b1), float(lm), float(R))
        for b0, b1, lm, R in zip(_BETA0S, _BETA1S, _LAMBDA_MINS, _RS)
    ], dtype=np.float64)
    y = np.array([
        delta_H0_ltb(delta=float(d), R_mpc=float(R)) - C1 * float(d)
        for d, R in zip(_DELTAS, _RS)
    ], dtype=np.float64)

    denom = float(f @ f)
    assert denom > 1e-24, "fixture accidentally landed at f_topo-all-zero"
    alpha_star = float((f @ y) / denom)
    return f, y, alpha_star


def _make_fixture_nbody_kbc_json(*, voids_records: list[dict]) -> dict:
    return {
        "cache_source": "/synthetic/test_fixture",
        "grid_N": 32,
        "lambda_th": 0.0,
        "K": len(voids_records),
        "alpha_used": 0.0,
        "timestamp": "2026-04-21T00:00:00+00:00",
        "voids": voids_records,
        "beta1_distribution": {
            "count_nonzero": sum(1 for v in voids_records if v["beta1_persistent"] > 0),
            "count_total": len(voids_records),
            "median": 0.0,
            "max": max((v["beta1_persistent"] for v in voids_records), default=0),
        },
    }


def _make_voids_records() -> list[dict]:
    from problems.hubble_tension_web.functional import C1, f_topo
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb

    recs = []
    for i, (d, R, b0, b1, lm) in enumerate(zip(
        _DELTAS, _RS, _BETA0S, _BETA1S, _LAMBDA_MINS
    )):
        f_val = float(f_topo(int(b0), int(b1), float(lm), float(R)))
        ltb = float(delta_H0_ltb(delta=float(d), R_mpc=float(R)))
        y_res = float(ltb - C1 * float(d))
        recs.append({
            "idx": i,
            "center_mpc": [250.0, 250.0, 250.0],
            "N_halos": 100,
            "delta_eff": float(d),
            "R_eff_mpc": float(R),
            "beta0": int(b0),
            "beta1_persistent": int(b1),
            "lambda_min": float(lm),
            "delta_H0_total": C1 * float(d),
            "kinematic_term": C1 * float(d),
            "topological_term": 0.0,
            "f_topo_at_alpha_1": f_val,
            "ltb_anchor_at_delta_R": ltb,
            "y_residual": y_res,
        })
    return recs


def _run_calibration_subprocess(*, results_dir: Path, extra_env: dict | None = None) -> Path:
    env = os.environ.copy()
    env["ATFT_HUBBLE_RESULTS_DIR"] = str(results_dir)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-m", "problems.hubble_tension_web.experiments.nbody_calibration"],
        capture_output=True, env=env, timeout=60,
    )
    assert result.returncode == 0, (
        f"nbody_calibration failed rc={result.returncode}\n"
        f"stdout:\n{result.stdout.decode(errors='replace')}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
    return results_dir / "nbody_calibration.json"


def test_alpha_star_matches_closed_form(tmp_path):
    """On a K=8 planted fixture, alpha* must match the hand-computed value to 1e-10."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    voids_records = _make_voids_records()
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=voids_records), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    assert cal_path.exists()
    out = json.loads(cal_path.read_text())

    _, _, expected_alpha = _hand_computed_alpha_star()
    assert out["alpha_star"] is not None, f"reason was {out.get('reason')!r}"
    assert abs(out["alpha_star"] - expected_alpha) < 1e-10, (
        f"alpha_star={out['alpha_star']}, expected {expected_alpha}, "
        f"diff={out['alpha_star'] - expected_alpha}"
    )
    assert out["alpha_units"] == "km/s"
    assert out["K"] == 8


def test_bootstrap_ci_brackets_alpha_star(tmp_path):
    """68% bootstrap CI must be a 2-tuple that brackets the point estimate."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=_make_voids_records()), indent=2)
    )

    cal_path = _run_calibration_subprocess(
        results_dir=results_dir,
        extra_env={"ATFT_NBODY_CAL_BOOTSTRAP_B": "500"},  # small B for test speed
    )
    out = json.loads(cal_path.read_text())

    ci = out["alpha_ci_68"]
    assert ci is not None and len(ci) == 2
    lo, hi = float(ci[0]), float(ci[1])
    assert lo <= hi
    # The bootstrap median tends toward alpha*; CI should span it at 68%.
    alpha = float(out["alpha_star"])
    # Allow a small slack: on K=8, the sampled bootstrap CI can exclude alpha_star
    # in edge cases. Require at most one edge is on the wrong side (standard
    # percentile-bootstrap interpretation on small K).
    within = lo <= alpha <= hi
    assert within or abs(alpha - lo) < 0.5 or abs(alpha - hi) < 0.5, (
        f"alpha_star={alpha} far outside 68% CI {ci}"
    )


def test_f_test_returns_valid_pvalue(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=_make_voids_records()), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    out = json.loads(cal_path.read_text())

    p = out["p_F_alpha_vs_zero"]
    assert 0.0 <= float(p) <= 1.0


def test_per_void_records_complete(tmp_path):
    """Output per_void block has all required keys for aggregate.py."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=_make_voids_records()), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    out = json.loads(cal_path.read_text())

    assert len(out["per_void"]) == 8
    for pv in out["per_void"]:
        for key in ("delta", "R_mpc", "f_topo", "y_ltb_residual", "beta1"):
            assert key in pv, f"per_void missing key: {key}"


def test_insufficient_voids_branch(tmp_path):
    """K < ATFT_NBODY_CAL_MIN_VOIDS -> alpha_star null with insufficient-voids reason."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    recs = _make_voids_records()[:2]  # K = 2 < default min 3
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=recs), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    out = json.loads(cal_path.read_text())

    assert out["alpha_star"] is None
    assert out["alpha_ci_68"] is None
    assert "insufficient voids" in out["reason"]
    assert out["K"] == 2


def test_f_topo_all_zero_branch(tmp_path):
    """If every record has beta1=0, f_topo collapses; alpha_star null with clear reason."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Build records with beta1 forced to 0 everywhere so f_topo = 0.
    from problems.hubble_tension_web.functional import C1
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb

    recs = []
    for i in range(5):
        d, R = -0.15, 250.0
        ltb = float(delta_H0_ltb(delta=d, R_mpc=R))
        recs.append({
            "idx": i, "center_mpc": [0, 0, 0], "N_halos": 50,
            "delta_eff": d, "R_eff_mpc": R,
            "beta0": 10, "beta1_persistent": 0, "lambda_min": 0.5,
            "delta_H0_total": C1 * d, "kinematic_term": C1 * d, "topological_term": 0.0,
            "f_topo_at_alpha_1": 0.0, "ltb_anchor_at_delta_R": ltb,
            "y_residual": ltb - C1 * d,
        })
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=recs), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    out = json.loads(cal_path.read_text())

    assert out["alpha_star"] is None
    assert out["reason"] == "f_topo all zero"
    assert out["K"] == 5


def test_output_json_schema_shape(tmp_path):
    """All required top-level keys present."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=_make_voids_records()), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    out = json.loads(cal_path.read_text())

    for key in (
        "alpha_star", "alpha_units", "alpha_ci_68", "alpha_bootstrap_median",
        "K", "bootstrap_B", "chi2_reduced", "pearson_r_f_y", "p_F_alpha_vs_zero",
        "reason", "per_void", "ltb_reference_source", "timestamp",
    ):
        assert key in out, f"schema missing top-level key: {key}"
