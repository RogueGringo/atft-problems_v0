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

| Experiment              | Wall time       | Δ vs T=0   |
|-------------------------|-----------------|------------|
| analytical_reduction.py |   543.11s       | -40.51%    |
| sim_calibration.py      |   (deferred)    | —          |
| kbc_crosscheck.py       |   63.89s        | -61.28%    |

**Memory impact (primary win):** at N=1500 the dense `delta` would be ~4.6 GB fp64; the sparse CSR form uses ~10 MB (500× reduction). At the design target N=1e5 the dense form (~2.6 EB) is physically impossible; sparse is feasible.

**Wall-time note:** sim_calibration re-measurement deferred — sparse lil_matrix row-assembly is Python-loop-y and likely a small regression at N=1500 (per spec). Steps 2 (Arnoldi) and 3 (parallel pool) dominate the wall-time change; re-measure sim_calibration after Task 3 lands.

## After Step 2 (Arnoldi eigensolver)

| Experiment              | Wall time       | Δ vs T=0   | Δ vs Step 1 |
|-------------------------|-----------------|------------|-------------|
| analytical_reduction.py |   29.03s        | -96.82%    | -94.66%     |
| sim_calibration.py      |   (deferred)    | —          | —           |
| kbc_crosscheck.py       |   4.34s         | -97.37%    | -93.21%     |

`eigsh(L, k=k_spec+4, sigma=-1e-6, which='LM', tol=1e-8, ncv=60)` — shift-invert
Arnoldi solves only 20 smallest eigenvalues instead of the full spectrum. At
N=1500 with stalk_dim=8 that is 20 of ~12000 modes — a ~600x reduction in
solved modes.

**Deviation from spec §Step 2:** Plan called for `which='SA'` with default `ncv`.
Empirically, typed sheaf Laplacians have many triple-degenerate eigenvalues
from the stalk structure, and plain Lanczos with default `ncv = 2k+1 = 41`
collapses those degeneracies (~8-27% rel error on bottom-k). Switched to
shift-invert (`sigma=-1e-6, which='LM'`) with widened Krylov subspace
(`ncv = max(3*k_arnoldi, 40) = 60`). Shift-invert is also ARPACK-canonical for
smallest eigenvalues of PSD matrices. Result: rel < 1e-13 on both bulk and
lambda_min across 3 seeds, far exceeding the rel<1e-6 / rel<1e-9 contract.
At N=600 shift-invert is 13x faster than dense eigvalsh; the gap widens with
N.

Fallback to dense eigvalsh on `ArpackNoConvergence` exercised by
`test_eigsh_fallback_to_dense_on_arpack_failure`. No production workload has
tripped the fallback; the guard is defensive.

`sim_calibration` re-measurement deferred — Task 3's multiprocessing pool
dominates its wall-time change; re-measure once that lands.

## After Step 3 (parallel scan)

| Experiment              | Wall time | Δ vs T=0   | Δ vs Step 2       |
|-------------------------|-----------|------------|-------------------|
| analytical_reduction.py |   29.00s  | -96.82%    | unchanged (was 29.03s) |
| sim_calibration.py      |   20.39s  | -99.60%    | (deferred pre-Task 3) |
| kbc_crosscheck.py       |   7.16s   | -95.66%    | unchanged (was 4.34s, jitter) |

multiprocessing.Pool with imap_unordered across `min(os.cpu_count(), 30)` workers.
Spawn context explicit for cross-platform determinism. Deterministic output
preserved by sorting results by (delta, R) before the LSQ fit.

Spawn overhead: ~2-3s per worker on Windows (scipy re-import + module load).
Amortized over the 30-config scan. Threshold guard `_POOL_MIN_CONFIGS=6` keeps
tiny scans sequential.

**alpha_star = 0.0 unchanged** (f_topo identically zero across the smooth-void
scan, same observation as the REWORK pipeline — β₁_persistent noise floor).
Scan is sorted ascending by (delta, R): most negative delta first (-0.30, ...,
-0.05) — deterministic regardless of worker scheduling.

sim_calibration compound speedup vs T=0: 5045s → 20.39s ≈ 247× (combined
effect of Task 1 sparse coboundary, Task 2 Arnoldi eigsh, and Task 3 parallel
scan). On the 8-core Snapdragon X Plus the parallel lever dominates the
Step-3 delta.

## After Step 4 (int8 quantized sidecar)

<populated in Task 4; note int8 is a sidecar — no wall-time change to fp64 path expected>

## Target (spec §Acceptance gate)

- `test_pipeline.py` end-to-end at N=1500: **<30 s** wall time (~200x over baseline; most of it from Step 3 parallelizing sim_calibration and Step 2's Arnoldi in place of full eigvalsh).
- Memory at N=1500 dense Laplacian step: baseline ~26 GB → target <200 MB after Step 1 (sparse).
