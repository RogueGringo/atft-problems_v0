"""Concurrent runner for the three independent experiments + aggregate.

analytical_reduction, sim_calibration, kbc_crosscheck are independent
processes - they share no state and write to different result files.
Dispatch them concurrently via subprocess.Popen, wait for all to finish,
then run aggregate (which depends on all three JSON outputs).

Usage:
    python -m problems.hubble_tension_web.experiments.run_all

Expected wall time on 8-core Snapdragon X Plus (post-perf-rework):
    sequential baseline (T=0):    ~6123 s  (~102 min)
    sequential after perf rework:    ~58 s
    concurrent after perf rework:    ~30 s  (max of the three + aggregate)

The concurrent wall time is lower-bounded by max(analytical, sim_cal, kbc)
plus a small aggregate step. On our measured numbers:
    max(29s analytical, 20s sim_cal, 7s kbc) + ~1s aggregate ~= 30s.

Note on oversubscription: sim_calibration internally uses
multiprocessing.Pool with cpu_count() workers. When this runner launches
three subprocesses concurrently, the inner pool can oversubscribe the
CPU. In practice the OS schedules the extra workers via time-slicing
with only modest overhead (most sim_calibration worker time is in
scipy's BLAS, which yields the GIL). If you see degradation, add an
env var like HUBBLE_PARALLEL=0 to sim_calibration and fall back to the
sequential scan when set - not implemented here because the simple
concurrent path already comfortably hits the <30s target.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    "problems.hubble_tension_web.experiments.analytical_reduction",
    "problems.hubble_tension_web.experiments.sim_calibration",
    "problems.hubble_tension_web.experiments.kbc_crosscheck",
]
AGGREGATE = "problems.hubble_tension_web.experiments.aggregate"


def _env_for(mod: str) -> dict:
    """Per-experiment env overrides to avoid CPU oversubscription.

    sim_calibration spawns an inner multiprocessing.Pool. When this runner
    launches it concurrently with analytical + kbc, the inner pool must be
    clamped so those other two processes have a core to land on. Reserve 2
    cores: one for analytical, one for kbc + OS.
    """
    env = os.environ.copy()
    # Pin BLAS/OMP thread counts so concurrent experiments don't each
    # spin up cpu_count() BLAS threads and fight for the same cores.
    # Each experiment uses 1 BLAS thread; parallelism is delivered by
    # the run_all dispatch (outer) and sim_calibration's Pool (inner).
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    if mod.endswith("sim_calibration"):
        total = os.cpu_count() or 4
        # Leave 2 cores for analytical + kbc + OS.
        env["HUBBLE_POOL_WORKERS"] = str(max(total - 2, 2))
    return env


def main() -> None:
    t0 = time.perf_counter()

    procs = [
        subprocess.Popen(
            [sys.executable, "-m", mod],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_env_for(mod),
        )
        for mod in EXPERIMENTS
    ]

    failures: list[tuple[str, int, str]] = []
    for proc, mod in zip(procs, EXPERIMENTS):
        out, _ = proc.communicate()
        if proc.returncode != 0:
            failures.append((mod, proc.returncode, out.decode(errors="replace")))

    t_parallel = time.perf_counter() - t0

    if failures:
        for mod, rc, out in failures:
            print(f"=== {mod} FAILED rc={rc} ===", file=sys.stderr)
            print(out, file=sys.stderr)
        sys.exit(1)

    agg_t0 = time.perf_counter()
    rc = subprocess.run([sys.executable, "-m", AGGREGATE]).returncode
    t_aggregate = time.perf_counter() - agg_t0
    if rc != 0:
        print(f"aggregate failed rc={rc}", file=sys.stderr)
        sys.exit(1)

    t_total = time.perf_counter() - t0

    print(f"parallel phase (3 experiments concurrent): {t_parallel:.2f}s")
    print(f"aggregate:                                  {t_aggregate:.2f}s")
    print(f"total wall time:                            {t_total:.2f}s")

    results_dir = Path(__file__).parent.parent / "results"
    for name in ("analytical_reduction.json", "sim_calibration.json",
                 "kbc_crosscheck.json", "REPORT.md"):
        path = results_dir / name
        if not path.exists():
            print(f"WARNING: expected output missing: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
