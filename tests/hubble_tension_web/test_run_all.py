"""Smoke test for the concurrent experiment runner.

Verifies that `python -m problems.hubble_tension_web.experiments.run_all`
dispatches all three experiments + aggregate, writes the expected
artifacts, and stays under a conservative wall-time ceiling.

Takes ~30-60s; excluded from the fast suite via the test_pipeline-style
pattern (pytest --ignore=... at caller's discretion).
"""
import json
import subprocess
import sys
import time
from pathlib import Path


def test_run_all_wall_time_under_ceiling():
    """Concurrent pipeline finishes under 120s (4x headroom over the ~30s target)."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "problems.hubble_tension_web.experiments.run_all"],
        capture_output=True,
    )
    elapsed = time.perf_counter() - t0

    assert result.returncode == 0, (
        f"run_all exited rc={result.returncode}\nstdout:\n{result.stdout.decode(errors='replace')}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
    assert elapsed < 120.0, (
        f"run_all took {elapsed:.1f}s, exceeded 120s ceiling. "
        f"Post-perf-rework target is ~30s; 4x headroom should absorb CI jitter."
    )

    results = Path("problems/hubble_tension_web/results")
    for name in ("analytical_reduction.json", "sim_calibration.json",
                 "kbc_crosscheck.json", "REPORT.md"):
        assert (results / name).exists(), f"missing artifact: {name}"

    # JSON integrity
    for name in ("analytical_reduction.json", "sim_calibration.json",
                 "kbc_crosscheck.json"):
        json.loads((results / name).read_text())


def test_run_all_preserves_signed_contract():
    """The concurrent pipeline must produce the same signed KBC result as
    the sequential path (positive kinematic_term, no SIGN ERROR verdict).

    Relies on test_run_all_wall_time_under_ceiling having run first; if the
    artifact is missing, run the runner inline to produce it.
    """
    results = Path("problems/hubble_tension_web/results")
    kbc_path = results / "kbc_crosscheck.json"
    if not kbc_path.exists():
        subprocess.run(
            [sys.executable, "-m", "problems.hubble_tension_web.experiments.run_all"],
            check=True,
        )

    kbc = json.loads(kbc_path.read_text())
    assert kbc["kinematic_term"] > 0, (
        f"signed contract violated: kinematic_term={kbc['kinematic_term']}"
    )
    assert "SIGN ERROR" not in kbc["verdict"], (
        f"verdict contains SIGN ERROR: {kbc['verdict']}"
    )


def test_run_all_skips_nbody_when_cache_absent(tmp_path):
    """run_all must exit 0 and run the three synthetic experiments even when
    no nbody cache exists. The nbody step is opt-in, not load-bearing."""
    import os
    env = os.environ.copy()
    # Ensure the cache lookup points at a definitely-empty dir.
    env["ATFT_DATA_CACHE"] = str(tmp_path / "nonexistent")
    env.pop("ATFT_NBODY_CACHE_FILE", None)  # defensively clear
    result = subprocess.run(
        [sys.executable, "-m", "problems.hubble_tension_web.experiments.run_all"],
        capture_output=True, env=env, timeout=180,
    )
    assert result.returncode == 0, (
        f"run_all must survive a missing nbody cache. stderr:\n"
        f"{result.stderr.decode(errors='replace')}"
    )
    combined = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
    assert "nbody" in combined.lower(), (
        "run_all should mention nbody in output when skipping it"
    )
    # All three synthetic outputs must still exist.
    results = Path("problems/hubble_tension_web/results")
    for name in ("analytical_reduction.json", "sim_calibration.json",
                 "kbc_crosscheck.json", "REPORT.md"):
        assert (results / name).exists(), f"synthetic artifact missing: {name}"
