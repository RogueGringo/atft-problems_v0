# Hubble-Tension-Web Performance Notes

Wall-time record for the 4-step perf pass (spec: `2026-04-20-hubble-perf-rework-design.md`).

All measurements on the project maintainer's Snapdragon X Plus laptop (Windows 11, Python 3.14, single user, background apps minimized). Times are wall time of `python -m problems.hubble_tension_web.experiments.<name>` from a warm Python install (scipy etc. already imported at least once in the session).

## Methodology

- One measurement per configuration (no averaging across runs — order-of-magnitude tracking, not benchmark-publication-grade).
- Module-load time is included (subprocess wall clock).
- Matplotlib backend: Agg (no display). Results dir is preserved between runs.
- Pytest `test_pipeline.py` is NOT used for these measurements — it wraps all three experiments and adds its own overhead.

## Baseline (T=0, HEAD = `e862853`, dense + sequential + fp64)

| Experiment              | Wall time     | Notes                                           |
|-------------------------|---------------|-------------------------------------------------|
| analytical_reduction.py | **913 s**     | 15:11:11 → 15:26:24, 9-point δ scan at R=300   |
| sim_calibration.py      | **5045 s**    | 15:26:24 → 16:50:29, 30-point (δ, R) scan      |
| kbc_crosscheck.py       | **165 s**     | Rerun at `n_points=1500` after n=2500 swap trap |
| **pipeline total**      | **~6123 s**   | ≈ 1 h 42 min                                    |

Baseline numbers captured during the REWORK closeout run (commit `9aab0a8`). Between that run and this T=0 commit (`e862853`), only the spec document (`7f3635d`) and this plan (`e862853`) were added — **zero source-code changes** — so these numbers are HEAD-valid without re-running the 90+ minute sim_calibration leg.

## After Step 1 (sparse coboundary)

<populated in Task 1>

## After Step 2 (Arnoldi eigensolver)

<populated in Task 2>

## After Step 3 (parallel scan)

<populated in Task 3>

## After Step 4 (int8 quantized sidecar)

<populated in Task 4; note int8 is a sidecar — no wall-time change to fp64 path expected>

## Target (spec §Acceptance gate)

- `test_pipeline.py` end-to-end at N=1500: **<30 s** wall time (~200x over baseline; most of it from Step 3 parallelizing sim_calibration and Step 2's Arnoldi in place of full eigvalsh).
- Memory at N=1500 dense Laplacian step: baseline ~26 GB → target <200 MB after Step 1 (sparse).
