"""End-to-end smoke test of the REWORK pipeline.

Runs all four scripts in order, asserts artifacts exist, and checks signed-band contract.
"""
import json
from pathlib import Path
import subprocess
import sys


def run(mod: str) -> None:
    result = subprocess.run([sys.executable, "-m", mod], check=False, capture_output=True)
    assert result.returncode == 0, f"{mod} failed: {result.stderr.decode(errors='replace')}"


def test_full_pipeline_runs_end_to_end():
    run("problems.hubble_tension_web.experiments.analytical_reduction")
    run("problems.hubble_tension_web.experiments.sim_calibration")
    run("problems.hubble_tension_web.experiments.kbc_crosscheck")
    run("problems.hubble_tension_web.experiments.aggregate")

    results = Path("problems/hubble_tension_web/results")
    assert (results / "analytical_reduction.json").exists()
    assert (results / "sim_calibration.json").exists()
    assert (results / "kbc_crosscheck.json").exists()
    assert (results / "REPORT.md").exists()

    for fname in ["analytical_reduction.json", "sim_calibration.json", "kbc_crosscheck.json"]:
        json.loads((results / fname).read_text())


def test_kbc_cross_check_sign_is_correct():
    """Signed-band contract: KBC void must produce delta_H0 with the right sign (positive)."""
    results = Path("problems/hubble_tension_web/results")
    if not (results / "kbc_crosscheck.json").exists():
        run("problems.hubble_tension_web.experiments.sim_calibration")
        run("problems.hubble_tension_web.experiments.kbc_crosscheck")
    kbc = json.loads((results / "kbc_crosscheck.json").read_text())
    assert kbc["kinematic_term"] > 0, (
        f"KBC kinematic term must be > 0 (delta=-0.2, c1<0 => c1*delta > 0); "
        f"got {kbc['kinematic_term']}. Sign regression."
    )
    # delta_H0 total may be above or below the positive band; but sign must not be a SIGN ERROR
    # UNLESS alpha*f_topo is large-negative, which would be a calibration pathology flagged
    # in verdict.
    assert "SIGN ERROR" not in kbc["verdict"], (
        f"KBC verdict reports SIGN ERROR: {kbc['verdict']}"
    )


def test_analytical_reduction_tautology_residual_small():
    """Regression guard: kinematic_term = C1*delta to machine precision."""
    results = Path("problems/hubble_tension_web/results")
    if not (results / "analytical_reduction.json").exists():
        run("problems.hubble_tension_web.experiments.analytical_reduction")
    data = json.loads((results / "analytical_reduction.json").read_text())
    for r in data["records"]:
        assert abs(r["kin_tautology_residual"]) < 1e-6, (
            f"kin tautology residual too large at delta={r['delta']}: {r['kin_tautology_residual']}"
        )
