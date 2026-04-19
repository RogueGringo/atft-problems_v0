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
