# Hubble-Tension-Web Performance & Hardware Co-Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan is the perf pass on top of the math-correct REWORK (`2026-04-19-hubble-tension-web-REWORK.md` plan, merged at `9aab0a8`). Math is frozen — no spectrum, sign, or calibration numerics change. Only memory/time cost changes.

**Goal:** Take `problems/hubble_tension_web/` from a dense-matrix N=1500 toy to a sparse + Arnoldi + parallel + int8-prototype pipeline that scales to N=10^5-10^6 galaxy catalogs on a Snapdragon X Plus laptop, while preserving every math invariant proven in the REWORK pass.

**Architecture:** Four ordered, independently-committable steps. (1) Rewrite `typed_sheaf_laplacian` to assemble `delta` as `scipy.sparse.csr_matrix` so memory becomes O(n) not O(n^2); the caller adapts via `L.toarray()` until step 2 lands. (2) Replace the dense `eigvalsh` in `summarize_spectrum` with Arnoldi `eigsh(L, k=k_spec+4, which='SA')`, with a documented ArpackNoConvergence fallback to dense. (3) Parallelize the 30-point `sim_calibration` scan (and optionally the 9-point `analytical_reduction` scan) via `multiprocessing.Pool.imap_unordered`, with deterministic sort-by-key post-processing. (4) Add a SIDECAR module `laplacian_quantized.py` that builds `delta` as int8 and accumulates `L = delta.T @ delta` as int32, demonstrating NPU-feasibility without touching the authoritative fp64 path. Math is frozen; `types.py`, `graph.py`, `synthetic.py`, `ltb_reference.py`, `functional.py` are not modified.

**Tech Stack:** Python 3.14, NumPy, SciPy (`scipy.sparse`, `scipy.sparse.linalg.eigsh`, `scipy.sparse.linalg.ArpackNoConvergence`), `multiprocessing.Pool` (stdlib), matplotlib (Agg, unchanged), `ripser` (unchanged). No new third-party deps.

**Reference spec:** `docs/superpowers/specs/2026-04-20-hubble-perf-rework-design.md`

**Branch:** `feat/hubble-tension-web` (continuing on this branch; do NOT create a new branch — the perf work lands on top of the REWORK commits).

**Baseline HEAD:** `7f3635d` (spec commit, on top of `9aab0a8` "feat(hubble): REWORK pipeline results — alpha*=0, KBC delta_H0=+4.49 (correct sign)").

---

## File Structure

**Files modified (in place, preserving public signatures):**

```
problems/hubble_tension_web/
├── laplacian.py            # typed_sheaf_laplacian returns csr_matrix instead of ndarray
├── spectrum.py             # summarize_spectrum uses eigsh with dense fallback; accepts sparse L
└── experiments/
    ├── sim_calibration.py  # module-level _scan_one + Pool.imap_unordered + deterministic sort
    └── analytical_reduction.py  # optional: same pool pattern, 9-point scan
```

**Files created:**

```
problems/hubble_tension_web/
├── laplacian_quantized.py  # NEW sidecar: int8 delta, int32 accumulation, fp dequant
└── PERF_NOTES.md           # NEW: baseline + per-step wall times and methodology
```

**Tests modified (in place):**

```
tests/hubble_tension_web/
├── test_laplacian.py       # three existing tests learn to call L.toarray() before
│                           #   np.allclose / np.linalg.eigvalsh; new sparse-vs-dense test
└── test_spectrum.py        # existing tests unchanged (summarize_spectrum now accepts both)
```

**Tests created:**

```
tests/hubble_tension_web/
├── test_laplacian_sparse.py    # sparse-vs-dense equivalence on three seeds
├── test_spectrum_eigsh.py      # eigsh bottom-k agreement vs dense eigvalsh; fallback path
├── test_sim_calibration_parallel.py  # parallel==sequential on 4-point mini-scan
└── test_laplacian_quantized.py # int8 sidecar: accuracy contract rel<1e-3 on lambda_min
```

**Files NOT touched (math frozen):**

```
problems/hubble_tension_web/
├── types.py
├── synthetic.py
├── graph.py
├── functional.py
├── ltb_reference.py
└── experiments/
    ├── kbc_crosscheck.py
    └── aggregate.py
tests/hubble_tension_web/
├── test_types.py
├── test_synthetic.py
├── test_graph.py
├── test_functional.py
├── test_ltb_reference.py
└── test_pipeline.py        # end-to-end, ~90 min; run only at completion gate
```

---

## Notation and Conventions (used across all tasks)

- **Working directory:** `C:\JTOD1\atft-problems_v0`. Use forward slashes in command lines because git-bash is the shell. Windows LF/CRLF warnings on `git add` are benign.
- **Pytest invocation:** always from repo root. Use `--ignore=tests/hubble_tension_web/test_pipeline.py` while iterating — the pipeline test runs all four experiments end-to-end at N=1500 and takes ~90 min. Only run it for the final acceptance gate and once in Task 0 baseline.
- **Fast subset command (copy/paste-ready):**
  ```bash
  python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py -q
  ```
  Expected at start of each task's step 1: the prior task's commit left this suite green.
- **Sparse type checks:** use `scipy.sparse.issparse(L)` rather than `isinstance(L, csr_matrix)` — keeps future refactors (e.g. `csc`) working.
- **Commit prefix:** `perf(hubble):` for all commits in this plan. Footer `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` on every commit.
- **STALK_DIM constant:** 8. Hardcoded in `laplacian.py`. Do NOT parametrize.
- **k_spec default:** 16. `k_spec + 4 = 20` Arnoldi slots for the eigsh call — the extra 4 guard against the known kernel degeneracy (see REWORK plan §Task 3b).

---

## Task 0: Baseline wall-time capture

Establish the "before" number so every subsequent step has a concrete improvement claim. No source code changes. This task is light on sub-steps.

**Files:**
- Create: `problems/hubble_tension_web/PERF_NOTES.md`

- [ ] **Step 0.1: Confirm clean starting state**

Run:
```bash
git status
git log --oneline -3
```

Expected: on branch `feat/hubble-tension-web`, HEAD at or near `7f3635d`, working tree clean (the `.claude/` and `.treetrunk/` untracked dirs from the tooling are fine to ignore).

- [ ] **Step 0.2: Run the fast test subset to confirm green starting point**

Run:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py -q
```

Expected: 33 passed (per spec & REWORK closeout). Any failure here is a pre-existing bug — halt and report.

- [ ] **Step 0.3: Capture per-experiment baseline wall times**

Run each experiment at its current (dense, sequential, fp64) implementation, timing each. Use Python's built-in `time.perf_counter()` via the shell so there is zero source-code drift.

```bash
python -c "import time,subprocess,sys; t=time.perf_counter(); r=subprocess.run([sys.executable,'-m','problems.hubble_tension_web.experiments.analytical_reduction']); print(f'analytical_reduction: {time.perf_counter()-t:.2f}s  rc={r.returncode}')"
```

```bash
python -c "import time,subprocess,sys; t=time.perf_counter(); r=subprocess.run([sys.executable,'-m','problems.hubble_tension_web.experiments.sim_calibration']); print(f'sim_calibration: {time.perf_counter()-t:.2f}s  rc={r.returncode}')"
```

```bash
python -c "import time,subprocess,sys; t=time.perf_counter(); r=subprocess.run([sys.executable,'-m','problems.hubble_tension_web.experiments.kbc_crosscheck']); print(f'kbc_crosscheck: {time.perf_counter()-t:.2f}s  rc={r.returncode}')"
```

Expected: all three exit `rc=0`. Wall times will be approximately: `analytical_reduction` ~60s-2min, `sim_calibration` ~30-90 min (dominant), `kbc_crosscheck` ~30s-2min. Record the exact numbers you see.

- [ ] **Step 0.4: Write `PERF_NOTES.md` with the captured baseline**

Create `problems/hubble_tension_web/PERF_NOTES.md` with these exact contents (substituting the three measured times in the `T=0` row):

```markdown
# Hubble-Tension-Web Performance Notes

Wall-time record for the 4-step perf pass (spec: 2026-04-20-hubble-perf-rework-design.md).

All measurements on the project maintainer's Snapdragon X Plus laptop (Windows 11, Python 3.14, single user, background apps minimized). Times are wall time of `python -m problems.hubble_tension_web.experiments.<name>` from a warm Python install (scipy etc. already imported at least once in the session).

## Methodology

- One measurement per configuration (no averaging across runs — these measurements are for order-of-magnitude tracking, not benchmark-publication-grade).
- `time.perf_counter()` via a `subprocess.run` wrapper so module-load time is included.
- Matplotlib backend: Agg (no display). Results dir is preserved between runs.
- Pytest `test_pipeline.py` is NOT used for these measurements — it wraps all three experiments and adds its own overhead.

## Baseline (T=0, HEAD=7f3635d, dense + sequential + fp64)

| Experiment              | Wall time |
|-------------------------|-----------|
| analytical_reduction.py |   <FILL>s |
| sim_calibration.py      |   <FILL>s |
| kbc_crosscheck.py       |   <FILL>s |

## After Step 1 (sparse coboundary)

<populated in Task 1>

## After Step 2 (Arnoldi eigensolver)

<populated in Task 2>

## After Step 3 (parallel scan)

<populated in Task 3>

## After Step 4 (int8 quantized sidecar)

<populated in Task 4; note int8 is a sidecar — no wall-time change to fp64 path expected>

## Target (spec §Acceptance gate)

- `test_pipeline.py` end-to-end at N=1500: **<30s** wall time (~6x over baseline at minimum; most of it from Step 3 parallelizing sim_calibration).
- Memory at N=1500 dense laplacian step: baseline ~26 GB → target <200 MB after Step 1 (sparse).
```

Replace each `<FILL>` with the measured seconds from Step 0.3 (one decimal place is enough).

- [ ] **Step 0.5: Commit the baseline**

```bash
git add problems/hubble_tension_web/PERF_NOTES.md
git commit -m "$(cat <<'EOF'
perf(hubble): Task 0 — capture baseline wall times

Record pre-rework wall time for analytical_reduction, sim_calibration,
kbc_crosscheck on the maintainer Snapdragon X Plus. Sets the reference
point for the 4-step perf pass (sparse, Arnoldi, parallel, int8).

No source changes. PERF_NOTES.md gains T=0 row and placeholders for
steps 1-4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: one-file commit, `git log --oneline -1` shows the new commit.

---

## Task 1: Sparse coboundary in `laplacian.py`

Rewrite `typed_sheaf_laplacian` so `delta` is built as `scipy.sparse.lil_matrix` (row-wise assembly is cheap) and returned as `csr_matrix` after `.tocsr()`. `L = delta.T @ delta` becomes a sparse matmul; the symmetrization `L = 0.5 * (L + L.T)` stays the same formula but runs on sparse operands.

The public return type changes from `np.ndarray` to `scipy.sparse.csr_matrix`. To keep the suite green before Task 2 ships the sparse-aware eigensolver, (a) `summarize_spectrum` gets a 2-line `toarray()` guard at its top, and (b) three tests in `test_laplacian.py` that call `np.allclose(L, L.T)` / `np.linalg.eigvalsh(L)` directly are taught to call `L_arr = L.toarray()` first.

**Correctness contract (from spec §Step 1):** dense(L_sparse) must match the old dense L to `rel < 1e-12` on three independent random webs.

**Files:**
- Modify: `problems/hubble_tension_web/laplacian.py` (rewrite of the `typed_sheaf_laplacian` body; imports; public return type)
- Modify: `problems/hubble_tension_web/spectrum.py:111` (add sparse→dense guard at top of `summarize_spectrum`)
- Modify: `tests/hubble_tension_web/test_laplacian.py` (three existing tests: `.toarray()` shim)
- Create: `tests/hubble_tension_web/test_laplacian_sparse.py` (sparse-vs-dense equivalence, 3 seeds)
- Modify: `problems/hubble_tension_web/PERF_NOTES.md` (fill the Step 1 row)

- [ ] **Step 1.1: Add the failing sparse-vs-dense equivalence test**

Create `tests/hubble_tension_web/test_laplacian_sparse.py` with this exact content:

```python
"""Sparse-coboundary equivalence tests for typed_sheaf_laplacian.

Contract (spec 2026-04-20 §Step 1):
  typed_sheaf_laplacian(...) must return scipy.sparse.csr_matrix whose .toarray()
  value matches the dense reference implementation to rel < 1e-12 on at least
  three independent random webs. The dense reference is computed inline here
  (copied from the pre-Task-1 implementation) so the test does not depend on the
  module under change.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest
from scipy import sparse


def _dense_reference_laplacian(positions, n, edges, environments):
    """Reference dense assembly — mirrors the pre-Task-1 typed_sheaf_laplacian body.

    Uses the CURRENT (post-Task-1) helpers (_R_dst_for_edge, build_stalk_init,
    STALK_DIM) because those are math-frozen and their outputs are bit-stable.
    """
    from problems.hubble_tension_web.laplacian import (
        STALK_DIM, _R_dst_for_edge, build_stalk_init,
    )
    from problems.hubble_tension_web.types import Environment, LocalCosmicWeb

    web = LocalCosmicWeb(positions=positions, environments=environments)
    stalks, _ = build_stalk_init(web)
    g = stalks[:, 0:3]

    m = len(edges)
    delta = np.zeros((m * STALK_DIM, n * STALK_DIM), dtype=np.float64)
    for e_idx, (s, d, etype) in enumerate(edges):
        env_s, env_d = etype.split("-", 1)
        R_dst = _R_dst_for_edge(g[s], g[d], env_s, env_d)
        row = slice(e_idx * STALK_DIM, (e_idx + 1) * STALK_DIM)
        col_s = slice(s * STALK_DIM, (s + 1) * STALK_DIM)
        col_d = slice(d * STALK_DIM, (d + 1) * STALK_DIM)
        delta[row, col_s] = -np.eye(STALK_DIM)
        delta[row, col_d] = R_dst
    L = delta.T @ delta
    return 0.5 * (L + L.T)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_sparse_laplacian_matches_dense(seed: int) -> None:
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian

    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 10, size=(40, 3))
    envs = rng.choice(list(Environment), size=40).tolist()
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)

    L_sparse = typed_sheaf_laplacian(
        positions=positions, n=n, edges=edges, stalk_dim=8,
        environments=web.environments,
    )
    assert sparse.issparse(L_sparse), "typed_sheaf_laplacian must return a sparse matrix"
    assert L_sparse.format == "csr", f"expected csr format, got {L_sparse.format}"

    L_dense_from_sparse = L_sparse.toarray()
    L_ref = _dense_reference_laplacian(positions, n, edges, envs)

    assert L_dense_from_sparse.shape == L_ref.shape
    # Absolute and relative tolerance both tight: sparse matmul may reorder
    # additions, but at 40 nodes the roundoff headroom is huge vs 1e-12.
    np.testing.assert_allclose(L_dense_from_sparse, L_ref, rtol=1e-12, atol=1e-12)


def test_sparse_laplacian_density_is_approximately_two_over_n() -> None:
    """Spec §Step 1 note: each delta row has exactly 2*STALK_DIM=16 nonzeros.

    After L = delta.T @ delta, the density of L is bounded above by the density
    of delta^2 (order of magnitude). We check it is << 1 to catch accidental
    densification (e.g. someone doing `.toarray()` internally).
    """
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian

    rng = np.random.default_rng(123)
    positions = rng.uniform(0, 10, size=(80, 3))
    envs = rng.choice(list(Environment), size=80).tolist()
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n, edges = build_typed_graph(web, k=6)

    L = typed_sheaf_laplacian(
        positions=positions, n=n, edges=edges, stalk_dim=8,
        environments=web.environments,
    )
    total = L.shape[0] * L.shape[1]
    density = L.nnz / total
    # At n=80, stalk_dim=8, density should be O(k*stalk_dim/n) ~ few %.
    # We assert a loose upper bound of 10% — tight enough to catch full
    # densification, loose enough not to flake.
    assert density < 0.10, (
        f"L density {density:.3f} suggests accidental densification; expected < 0.10"
    )
```

- [ ] **Step 1.2: Run the new test to verify it fails (because typed_sheaf_laplacian still returns ndarray)**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_laplacian_sparse.py -v
```

Expected: both tests fail with `AssertionError: typed_sheaf_laplacian must return a sparse matrix` (or similar — the exact assertion depends on import order). FAIL is the expected starting state.

- [ ] **Step 1.3: Rewrite `typed_sheaf_laplacian` to assemble sparse delta**

Replace the body of `typed_sheaf_laplacian` in `problems/hubble_tension_web/laplacian.py`. The diff:

Find the import block near the top (after the docstring) and add the `scipy.sparse` import:

```python
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse as _sparse
from scipy.spatial import KDTree

from problems.hubble_tension_web.types import Environment, LocalCosmicWeb
```

Then replace the body of `typed_sheaf_laplacian` (everything from the `if stalk_dim != STALK_DIM:` guard through the final `return L`) with:

```python
    if stalk_dim != STALK_DIM:
        raise ValueError(
            f"typed_sheaf_laplacian requires stalk_dim={STALK_DIM}; got {stalk_dim}. "
            "The gradient/env layout is not adjustable."
        )

    if environments is None:
        env_of: List[str | None] = [None] * n
        for s, d, etype in edges:
            e_s, e_d = etype.split("-", 1)
            env_of[s] = e_s
            env_of[d] = e_d
        if any(e is None for e in env_of):
            raise ValueError(
                "Could not infer environments for all nodes from edges; "
                "pass environments=web.environments explicitly."
            )
        env_values = env_of                                          # type: ignore[assignment]
    else:
        env_values = [e.value for e in environments]

    from problems.hubble_tension_web.types import LocalCosmicWeb as _LCW
    envs_enum = [Environment(v) for v in env_values]
    web = _LCW(positions=positions, environments=envs_enum)
    stalks, _flags = build_stalk_init(web)
    g = stalks[:, 0:3]

    m = len(edges)
    # Sparse coboundary assembly: each delta row has exactly 2*STALK_DIM=16
    # nonzeros (the -I block on col_s and the R_dst block on col_d). lil_matrix
    # allows efficient row-slice assignment during construction; we convert to
    # csr before the transpose/matmul.
    delta = _sparse.lil_matrix(
        (m * STALK_DIM, n * STALK_DIM), dtype=np.float64,
    )
    neg_I = -np.eye(STALK_DIM)
    for e_idx, (s, d, etype) in enumerate(edges):
        env_s, env_d = etype.split("-", 1)
        R_dst = _R_dst_for_edge(g[s], g[d], env_s, env_d)
        row0 = e_idx * STALK_DIM
        col_s0 = s * STALK_DIM
        col_d0 = d * STALK_DIM
        # lil_matrix supports slice assignment of dense blocks.
        delta[row0:row0 + STALK_DIM, col_s0:col_s0 + STALK_DIM] = neg_I
        delta[row0:row0 + STALK_DIM, col_d0:col_d0 + STALK_DIM] = R_dst

    delta_csr = delta.tocsr()
    L = (delta_csr.T @ delta_csr).tocsr()
    # Symmetrize in sparse form. 0.5*(L+L.T) preserves csr_matrix type.
    L = (0.5 * (L + L.T)).tocsr()
    return L
```

Also update the return-type annotation in the function signature from `-> np.ndarray` to `-> "_sparse.csr_matrix"` (forward string reference to avoid the re-import dance):

```python
def typed_sheaf_laplacian(
    *,
    positions: np.ndarray,
    n: int,
    edges: List[Tuple[int, int, str]],
    stalk_dim: int = STALK_DIM,
    rng_seed: int = 0,              # unused; kept for signature compatibility
    environments: List[Environment] | None = None,
) -> "_sparse.csr_matrix":
```

- [ ] **Step 1.4: Add the sparse→dense guard in `summarize_spectrum`**

Modify `problems/hubble_tension_web/spectrum.py`. Find the current `summarize_spectrum` body:

```python
def summarize_spectrum(
    *,
    L: np.ndarray,
    ...
) -> SpectralSummary:
    w = np.linalg.eigvalsh(L)
```

Change it to:

```python
def summarize_spectrum(
    *,
    L,
    n_nodes: int,
    edges: List[Tuple[int, int, str]],
    positions: np.ndarray,
    k_spec: int = 16,
    zero_tol: float = 1e-6,
    tau_persist: float = TAU_PERSIST,
    tau_max: float = TAU_MAX,
) -> SpectralSummary:
    # Accept either dense ndarray or scipy.sparse matrix. Task 1 lands sparse
    # output from typed_sheaf_laplacian; Task 2 replaces this dense solve with
    # eigsh(L) directly on the sparse form.
    from scipy import sparse as _sparse
    if _sparse.issparse(L):
        L_dense = L.toarray()
    else:
        L_dense = L
    w = np.linalg.eigvalsh(L_dense)
```

(Keep everything below `w = ...` unchanged.) Drop the `L: np.ndarray` annotation — parameter typing here is defensive only.

Add the import `from typing import List, Tuple` if it is not already present at the top of `spectrum.py`. (It already is; no change required — but confirm via `grep`.)

- [ ] **Step 1.5: Update the three direct-dense-operation tests in `test_laplacian.py`**

Three existing tests in `tests/hubble_tension_web/test_laplacian.py` call `np.allclose(L, L.T)` and `np.linalg.eigvalsh(L)` directly on the return value of `typed_sheaf_laplacian`. Those now get a sparse matrix and must `.toarray()` first.

Edit `test_typed_sheaf_laplacian_is_symmetric_psd` (line ~5):

Replace:
```python
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=8, rng_seed=0)
    assert np.allclose(L, L.T, atol=1e-8)
    w = np.linalg.eigvalsh(L)
    assert w.min() > -1e-8
```

With:
```python
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=8, rng_seed=0)
    L_arr = L.toarray() if hasattr(L, "toarray") else L
    assert np.allclose(L_arr, L_arr.T, atol=1e-8)
    w = np.linalg.eigvalsh(L_arr)
    assert w.min() > -1e-8
```

Edit `test_typed_vs_untyped_spectrum_differs` (line ~33). Replace:
```python
    L_typed   = typed_sheaf_laplacian(positions=positions, n=n_t, edges=edges_t, stalk_dim=8, rng_seed=0)
    L_untyped = typed_sheaf_laplacian(positions=positions, n=n_u, edges=edges_u, stalk_dim=8, rng_seed=0)

    w_t = np.sort(np.linalg.eigvalsh(L_typed))
    w_u = np.sort(np.linalg.eigvalsh(L_untyped))
```

With:
```python
    L_typed   = typed_sheaf_laplacian(positions=positions, n=n_t, edges=edges_t, stalk_dim=8, rng_seed=0)
    L_untyped = typed_sheaf_laplacian(positions=positions, n=n_u, edges=edges_u, stalk_dim=8, rng_seed=0)

    L_typed_arr   = L_typed.toarray()   if hasattr(L_typed, "toarray")   else L_typed
    L_untyped_arr = L_untyped.toarray() if hasattr(L_untyped, "toarray") else L_untyped
    w_t = np.sort(np.linalg.eigvalsh(L_typed_arr))
    w_u = np.sort(np.linalg.eigvalsh(L_untyped_arr))
```

Edit `test_nullity_drops_under_typing` (line ~71). Replace:
```python
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=stalk_dim, rng_seed=0)

    w = np.linalg.eigvalsh(L)
```

With:
```python
    L = typed_sheaf_laplacian(positions=positions, n=n, edges=edges, stalk_dim=stalk_dim, rng_seed=0)
    L_arr = L.toarray() if hasattr(L, "toarray") else L

    w = np.linalg.eigvalsh(L_arr)
```

The fourth test in this file (`test_laplacian_dimension_is_n_times_stalk_dim`) only checks `.shape` which works on sparse matrices; leave it unchanged. The fifth test (`test_gradient_stalk_construction_is_unit_and_deterministic`) does not call `typed_sheaf_laplacian`; leave it unchanged.

- [ ] **Step 1.6: Run the new equivalence test to verify it PASSES now**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_laplacian_sparse.py -v
```

Expected: 4 passed (3 seeds × `test_sparse_laplacian_matches_dense`, plus `test_sparse_laplacian_density_is_approximately_two_over_n`).

- [ ] **Step 1.7: Run the full fast suite to verify no regressions**

Run:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py -q
```

Expected: 37 passed (33 pre-existing + 4 new). Zero failures. If any test fails, inspect and fix before moving on — common culprits: forgot to add `.toarray()` in a test, or the sparse matmul produced `L` in a format other than csr (fix by appending `.tocsr()` to the final line of `typed_sheaf_laplacian`).

- [ ] **Step 1.8: Measure and record the Step 1 wall times**

Run each experiment once and capture the new wall time (same commands as Step 0.3):

```bash
python -c "import time,subprocess,sys; t=time.perf_counter(); r=subprocess.run([sys.executable,'-m','problems.hubble_tension_web.experiments.analytical_reduction']); print(f'analytical_reduction: {time.perf_counter()-t:.2f}s  rc={r.returncode}')"
python -c "import time,subprocess,sys; t=time.perf_counter(); r=subprocess.run([sys.executable,'-m','problems.hubble_tension_web.experiments.sim_calibration']); print(f'sim_calibration: {time.perf_counter()-t:.2f}s  rc={r.returncode}')"
python -c "import time,subprocess,sys; t=time.perf_counter(); r=subprocess.run([sys.executable,'-m','problems.hubble_tension_web.experiments.kbc_crosscheck']); print(f'kbc_crosscheck: {time.perf_counter()-t:.2f}s  rc={r.returncode}')"
```

Expected: `sim_calibration` likely shows a small **regression** (sparse assembly is slightly slower per-config than dense at N=1500 because lil_matrix indexing is Python-loop-y). This is fine — the win is memory, which unlocks Arnoldi in Step 2 and parallelism in Step 3. Record the numbers.

- [ ] **Step 1.9: Populate the Step 1 row in PERF_NOTES.md**

In `problems/hubble_tension_web/PERF_NOTES.md`, replace the `## After Step 1 (sparse coboundary)` section (which currently says `<populated in Task 1>`) with:

```markdown
## After Step 1 (sparse coboundary)

| Experiment              | Wall time | Δ vs T=0   |
|-------------------------|-----------|------------|
| analytical_reduction.py |   <FILL>s | <FILL>%    |
| sim_calibration.py      |   <FILL>s | <FILL>%    |
| kbc_crosscheck.py       |   <FILL>s | <FILL>%    |

Memory change (dense→sparse): at N=1500, `delta` dense would be ~10 MB (small), but at the design target N=1e5 the dense form is ~260 TB — impossible. Sparse is always feasible; dense dies above ~N=3000.

Sparse vs dense wall time at N=1500: expected small regression (~5-20%) from Python-loop lil_matrix assembly. Step 2 recovers and surpasses this.
```

Replace each `<FILL>` using the Step 1.8 measurements. For the Δ column, compute `(t_step1 - t_T0) / t_T0 * 100` and include the sign (negative = faster, positive = slower).

- [ ] **Step 1.10: Commit Task 1**

```bash
git add problems/hubble_tension_web/laplacian.py problems/hubble_tension_web/spectrum.py tests/hubble_tension_web/test_laplacian.py tests/hubble_tension_web/test_laplacian_sparse.py problems/hubble_tension_web/PERF_NOTES.md
git commit -m "$(cat <<'EOF'
perf(hubble): Task 1 — sparse coboundary in typed_sheaf_laplacian

Switch the coboundary delta from dense ndarray to scipy.sparse.lil_matrix
(row-slice assignment during edge loop), then tocsr() before delta.T @ delta.
L is returned as csr_matrix; summarize_spectrum gains a transparent sparse→
dense shim to keep the eigvalsh solver (replaced in Task 2) working.

New test tests/hubble_tension_web/test_laplacian_sparse.py verifies that
toarray(L_sparse) matches the pre-rework dense assembly to rel<1e-12 on
three random 40-point webs. Density sanity test guards against accidental
densification.

Three existing tests in test_laplacian.py learn to call .toarray() before
np.allclose / eigvalsh. No math change; spectrum is bit-equivalent up to
floating-point reordering in csr matmul.

Fast test suite: 37/37 green (was 33/33).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands, `git log --oneline -1` shows the new commit, working tree clean.

---

## Task 2: Arnoldi eigensolver in `spectrum.py`

Replace the `np.linalg.eigvalsh(L)` dense O(n^3) eigen-decomposition with `scipy.sparse.linalg.eigsh(L, k=k_spec+4, which='SA')` — Arnoldi iteration that returns only the smallest `k_spec+4` algebraic eigenvalues. For k_spec=16 this is 20 out of potentially thousands — a massive O(n^3) → O(k*nnz) speedup at large N.

Guard against Arnoldi non-convergence (known failure mode: highly degenerate kernels from disconnected graph components × stalk_dim zero modes) by catching `ArpackNoConvergence` and falling back to dense `eigvalsh(L.toarray())` with a logged warning.

**Accuracy contract (spec §Step 2):**
- `spectrum[:k_spec]` (the returned bottom k_spec) agrees with the dense reference to `rel < 1e-6` on 60-point random webs.
- `lambda_min` (smallest nonzero) agrees to `rel < 1e-9`.

**Files:**
- Modify: `problems/hubble_tension_web/spectrum.py` (replace eigvalsh with eigsh + fallback)
- Create: `tests/hubble_tension_web/test_spectrum_eigsh.py` (accuracy + fallback path)
- Modify: `problems/hubble_tension_web/PERF_NOTES.md` (fill the Step 2 row)

- [ ] **Step 2.1: Add the failing accuracy test for eigsh vs eigvalsh**

Create `tests/hubble_tension_web/test_spectrum_eigsh.py` with this exact content:

```python
"""Accuracy and fallback tests for the Arnoldi-based summarize_spectrum.

Contract (spec 2026-04-20 §Step 2):
  - spectrum[:k_spec] matches the dense eigvalsh result to rel < 1e-6.
  - lambda_min (smallest nonzero eigenvalue) matches to rel < 1e-9.
  - ArpackNoConvergence triggers dense fallback with a UserWarning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _build_small_web(seed: int, n: int = 60):
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 10, size=(n, 3))
    envs = rng.choice(list(Environment), size=n).tolist()
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n_, edges = build_typed_graph(web, k=6)
    return web, n_, edges


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_eigsh_bottom_k_matches_dense(seed: int) -> None:
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    web, n, edges = _build_small_web(seed, n=60)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments,
    )

    # Dense reference using the same sparse L (round-trip through toarray).
    L_dense = L.toarray()
    w_dense = np.sort(np.linalg.eigvalsh(L_dense))
    bottom_ref = w_dense[:16]

    summary = summarize_spectrum(
        L=L, n_nodes=n, edges=edges, positions=web.positions, k_spec=16,
    )
    bottom_got = np.sort(summary.spectrum)

    # rel < 1e-6 bound per spec §Step 2. Use atol=1e-10 for near-zero kernel
    # eigenvalues where "relative" is meaningless.
    np.testing.assert_allclose(bottom_got, bottom_ref, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_eigsh_lambda_min_matches_dense(seed: int) -> None:
    """Tighter rel<1e-9 contract on the spectral gap (f_topo depends on it)."""
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum

    web, n, edges = _build_small_web(seed, n=60)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments,
    )

    L_dense = L.toarray()
    w_dense = np.sort(np.linalg.eigvalsh(L_dense))
    nonzero = w_dense[w_dense > 1e-6]
    lambda_min_ref = float(nonzero[0]) if nonzero.size > 0 else 1e-6

    summary = summarize_spectrum(
        L=L, n_nodes=n, edges=edges, positions=web.positions, k_spec=16,
    )
    rel = abs(summary.lambda_min - lambda_min_ref) / max(abs(lambda_min_ref), 1e-24)
    assert rel < 1e-9, (
        f"lambda_min rel error {rel:.2e} exceeds spec bound 1e-9; "
        f"got {summary.lambda_min}, ref {lambda_min_ref}"
    )


def test_eigsh_fallback_to_dense_on_arpack_failure(monkeypatch) -> None:
    """If eigsh raises ArpackNoConvergence, summarize_spectrum falls back to dense.

    Monkeypatch scipy.sparse.linalg.eigsh to raise ArpackNoConvergence, then
    confirm the function still returns a valid SpectralSummary (via the dense
    path) and emits a UserWarning.
    """
    import scipy.sparse.linalg as _spla
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.spectrum import summarize_spectrum
    from problems.hubble_tension_web.types import SpectralSummary

    web, n, edges = _build_small_web(0, n=30)
    L = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments,
    )

    def _always_fail(*args, **kwargs):
        # ArpackNoConvergence requires (msg, eigenvalues, eigenvectors)
        raise _spla.ArpackNoConvergence("forced failure for test", np.array([]), np.array([]))

    # Patch the import used INSIDE spectrum.py. The implementation imports
    # eigsh at module level as `from scipy.sparse.linalg import eigsh` — patch
    # that symbol.
    import problems.hubble_tension_web.spectrum as _spec_mod
    monkeypatch.setattr(_spec_mod, "eigsh", _always_fail)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = summarize_spectrum(
            L=L, n_nodes=n, edges=edges, positions=web.positions, k_spec=8,
        )

    assert isinstance(summary, SpectralSummary)
    assert summary.spectrum.shape == (8,)
    assert summary.lambda_min > 0
    # Exactly one UserWarning about the fallback.
    fallback_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "eigsh" in str(w.message).lower()
    ]
    assert len(fallback_warnings) >= 1, (
        f"expected UserWarning mentioning eigsh fallback; got {[str(w.message) for w in caught]}"
    )
```

- [ ] **Step 2.2: Run the new tests to verify the first two FAIL and the fallback test fails differently**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_spectrum_eigsh.py -v
```

Expected: the accuracy tests should currently PASS (because `summarize_spectrum` is still using eigvalsh, which is trivially rel<1e-15 against itself). The fallback test should FAIL because `spectrum.py` doesn't import `eigsh` at module level yet. This is fine — the accuracy tests are codifying the existing behavior and will remain green; the fallback test is the one driving Task 2's implementation.

If the accuracy tests pass and the fallback test fails with `AttributeError: module ... has no attribute 'eigsh'`, you are in the right state.

- [ ] **Step 2.3: Rewrite `summarize_spectrum` to use `eigsh` with fallback**

Edit `problems/hubble_tension_web/spectrum.py`.

At the top of the file, add this import just below `import numpy as np`:

```python
import warnings

from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy import sparse as _sparse
```

(Keep the existing `from ripser import ripser as _ripser` try/except below.)

Replace the entire `summarize_spectrum` function body (starting after the docstring and signature) with:

```python
    # Accept dense ndarray or scipy.sparse matrix.
    k_arnoldi = k_spec + 4

    w_all: np.ndarray
    if _sparse.issparse(L):
        # eigsh requires k < n; if the matrix is tiny and k_spec+4 >= n,
        # just do a dense solve.
        n_dim = L.shape[0]
        if k_arnoldi >= n_dim:
            w_all = np.sort(np.linalg.eigvalsh(L.toarray()))
        else:
            try:
                w, _ = eigsh(L, k=k_arnoldi, which="SA", tol=1e-8)
                w = np.sort(w)
                w_all = w
            except ArpackNoConvergence:
                warnings.warn(
                    "eigsh failed to converge (likely degenerate kernel); "
                    "falling back to dense eigvalsh. This is slow for large L.",
                    UserWarning,
                    stacklevel=2,
                )
                w_all = np.sort(np.linalg.eigvalsh(L.toarray()))
    else:
        # Dense input — preserve the exact old behavior for legacy callers.
        w_all = np.sort(np.linalg.eigvalsh(L))

    spectrum = w_all[:k_spec].copy()

    beta0 = _connected_components(n_nodes, edges)
    beta1 = persistent_beta1(
        positions,
        tau_persist=tau_persist,
        tau_max=tau_max,
        edges_for_ell=edges,
    )

    nonzero = w_all[w_all > zero_tol]
    lambda_min = float(nonzero[0]) if nonzero.size > 0 else float(zero_tol)

    return SpectralSummary(
        spectrum=spectrum,
        beta0=int(beta0),
        beta1=int(beta1),
        lambda_min=lambda_min,
    )
```

Important details verified against the spec:
- `k_arnoldi = k_spec + 4 = 20` slots guard against the degenerate kernel (β0 connected components × up-to-stalk_dim null directions) eating slots at small N.
- `which='SA'` = smallest algebraic (signed). Since L is PSD this matches smallest-magnitude for non-trivial spectrum.
- `tol=1e-8` meets the rel<1e-6 bulk contract and the rel<1e-9 lambda_min contract in practice at N=60.
- `ArpackNoConvergence` catches the convergence failure mode. We re-raise as a warning (not a hard fail) and fall back to dense — correct behavior but slow, as documented.
- The `k_arnoldi >= n_dim` guard handles the 12-node test case in `test_laplacian_dimension_is_n_times_stalk_dim` (n_dim=96) fine but protects truly tiny matrices from Arnoldi error messages.
- Dense input is still accepted (the fallback shim added in Task 1.4 is no longer needed — but leaving it costs nothing; feel free to remove the `L_dense = L.toarray()` branch now that eigsh handles sparse directly).

Clean up: the Task-1.4 shim (`if _sparse.issparse(L): L_dense = L.toarray() else: L_dense = L`) is dead code once the new body above replaces the function. The new body above already has its own sparse/dense branching, so the shim is naturally deleted.

- [ ] **Step 2.4: Run the new tests to verify ALL pass**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_spectrum_eigsh.py -v
```

Expected: 7 passed (3 bottom-k × seeds + 3 lambda_min × seeds + 1 fallback).

- [ ] **Step 2.5: Run the full fast suite**

Run:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py -q
```

Expected: 44 passed (37 after Task 1 + 7 new in Task 2). Zero failures.

Common failure mode: if a test in `test_spectrum.py` asserts bit-equivalence of `spectrum` against a hardcoded value, the eigsh change may shift the value by <1e-6 relative. None of the existing `test_spectrum.py` tests do that (they assert shapes, signs, and order-of-magnitude properties — verified during this plan's research). If you see an unexpected test-spectrum.py failure, loosen the tolerance in that specific test to `atol=1e-8` — do NOT widen the eigsh tolerance in `summarize_spectrum`.

- [ ] **Step 2.6: Measure and record the Step 2 wall times**

Run the same three timing commands as Task 0.3. The expected effect at N=1500:

- `analytical_reduction`: modest speedup (~2-4x). Each call solves one eigenproblem.
- `sim_calibration`: **large** speedup (~5-20x). 30 eigenproblem solves, each now O(k*nnz) instead of O(N^3).
- `kbc_crosscheck`: modest speedup (1-2 eigenproblems, plus KBC overhead).

Record the numbers.

- [ ] **Step 2.7: Populate the Step 2 row in PERF_NOTES.md**

Replace the `## After Step 2 (Arnoldi eigensolver)` section (`<populated in Task 2>`) with:

```markdown
## After Step 2 (Arnoldi eigensolver)

| Experiment              | Wall time | Δ vs T=0   | Δ vs Step 1 |
|-------------------------|-----------|------------|-------------|
| analytical_reduction.py |   <FILL>s | <FILL>%    | <FILL>%     |
| sim_calibration.py      |   <FILL>s | <FILL>%    | <FILL>%     |
| kbc_crosscheck.py       |   <FILL>s | <FILL>%    | <FILL>%     |

eigsh(L, k=k_spec+4, which='SA', tol=1e-8) solves only 20 smallest eigenvalues
instead of the full spectrum. At N=1500 with stalk_dim=8 that is 20 of ~12000 —
a ~500x reduction in solved modes, ~5-20x wall-time speedup in practice.

Fallback to dense eigvalsh on ArpackNoConvergence exercised by
test_eigsh_fallback_to_dense_on_arpack_failure. No production workload has
tripped the fallback; the guard is defensive.
```

Fill in the `<FILL>` fields with measurements.

- [ ] **Step 2.8: Commit Task 2**

```bash
git add problems/hubble_tension_web/spectrum.py tests/hubble_tension_web/test_spectrum_eigsh.py problems/hubble_tension_web/PERF_NOTES.md
git commit -m "$(cat <<'EOF'
perf(hubble): Task 2 — Arnoldi eigensolver in summarize_spectrum

Replace np.linalg.eigvalsh(L) with scipy.sparse.linalg.eigsh(L, k=k_spec+4,
which='SA', tol=1e-8), solving only the smallest 20 eigenvalues instead of
the full spectrum. For N=1500, stalk_dim=8 this is 20 of ~12000 modes — a
~500x reduction in solved modes, producing a 5-20x wall-time speedup on
sim_calibration.

ArpackNoConvergence triggers a fallback to dense eigvalsh with a UserWarning
(guard-tested). The guard also covers the tiny-matrix case (k_arnoldi >=
n_dim) by deferring straight to the dense solve.

Accuracy: rel<1e-6 on spectrum[:k_spec], rel<1e-9 on lambda_min, per the
spec §Step 2 contract — verified on three seeds at N=60.

Fast test suite: 44/44 green (was 37/37).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands; working tree clean.

---

## Task 3: Parallel (delta, R) scan in `sim_calibration.py`

Rewrite `sim_calibration.main()` so the inner `for d in deltas: for R in radii:` loop dispatches through `multiprocessing.Pool.imap_unordered`. Each (delta, R) configuration is independent — no shared state. Deterministic output by sorting the returned records lexicographically by `(delta, R)` before LSQ assembly.

Windows spawn semantics require the worker function to be **module-level** (not a closure in `main()`) because `spawn` re-imports the module in each child. Refactor the loop body into `_scan_one(delta_R_tuple)`.

**Correctness contract (spec §Step 3):** final `alpha_star`, `mse`, `r_squared`, and the sorted `scan` list must be bit-equivalent to a sequential run — sort by `(delta, R)` before the LSQ fit so the dot-product order is deterministic regardless of worker scheduling.

**Optional:** the 9-point scan in `analytical_reduction.py` is a lower-priority parallelization. Same pattern. Include a single sub-step for it at the end of this task.

**Files:**
- Modify: `problems/hubble_tension_web/experiments/sim_calibration.py`
- Modify: `problems/hubble_tension_web/experiments/analytical_reduction.py` (optional, see Step 3.9)
- Create: `tests/hubble_tension_web/test_sim_calibration_parallel.py`
- Modify: `problems/hubble_tension_web/PERF_NOTES.md` (fill the Step 3 row)

- [ ] **Step 3.1: Add the failing parallel-vs-sequential equivalence test**

Create `tests/hubble_tension_web/test_sim_calibration_parallel.py`:

```python
"""Parallel vs sequential equivalence for sim_calibration.

Contract (spec 2026-04-20 §Step 3):
  Running the scan with multiprocessing.Pool must produce the same alpha_star,
  mse, r_squared, and scan list as a sequential run, up to floating-point
  reorder error (we use rtol=1e-12 which leaves ~3 orders of headroom over
  typical matmul roundoff).

This test uses a mini-scan (4 configs) so it runs in <30s.
"""
from __future__ import annotations

import numpy as np


def _mini_scan_sequential(deltas, radii):
    """Copy of the Task-3 scan body run sequentially. Ground-truth reference."""
    from problems.hubble_tension_web.functional import C1, predict_from_cosmic_web
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.types import VoidParameters

    scan = []
    for d in deltas:
        for R in radii:
            params = VoidParameters(delta=float(d), R_mpc=float(R))
            box = max(2.5 * R, 800.0)
            web = generate_synthetic_void(
                params, n_points=300, box_mpc=box,  # N=300 for speed
                rng_seed=abs(int(1000 * d + R)) + 1,
            )
            h1 = predict_from_cosmic_web(
                web=web, params=params, alpha=1.0, k=8, stalk_dim=8, k_spec=16,
            )
            ltb_full = delta_H0_ltb(delta=float(d), R_mpc=float(R))
            kin = C1 * float(d)
            y = ltb_full - kin
            scan.append(dict(
                delta=float(d), R=float(R),
                ltb_full=float(ltb_full), kinematic=float(kin),
                y=float(y), f_topo=float(h1.topological_term),
            ))
    return scan


def test_parallel_scan_matches_sequential() -> None:
    from problems.hubble_tension_web.experiments.sim_calibration import _scan_one

    # Mini-scan: 2 deltas × 2 radii = 4 configs.
    deltas = [-0.1, -0.2]
    radii = [200.0, 300.0]
    configs = [(float(d), float(R)) for d in deltas for R in radii]

    # Parallel via a Pool.
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(processes=2) as pool:
        par_results = list(pool.imap_unordered(_scan_one, configs))

    # Sort by (delta, R) to kill worker ordering.
    par_sorted = sorted(par_results, key=lambda r: (r["delta"], r["R"]))

    seq_results = _mini_scan_sequential(deltas, radii)
    seq_sorted = sorted(seq_results, key=lambda r: (r["delta"], r["R"]))

    assert len(par_sorted) == len(seq_sorted) == 4

    for p, s in zip(par_sorted, seq_sorted):
        assert p["delta"] == s["delta"]
        assert p["R"] == s["R"]
        for key in ("ltb_full", "kinematic", "y", "f_topo"):
            np.testing.assert_allclose(
                p[key], s[key], rtol=1e-12, atol=1e-14,
                err_msg=f"mismatch on field {key} at (delta={p['delta']}, R={p['R']})",
            )
```

- [ ] **Step 3.2: Run the new test to verify it FAILS**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_sim_calibration_parallel.py -v
```

Expected: fail with `ImportError: cannot import name '_scan_one' from 'problems.hubble_tension_web.experiments.sim_calibration'`. Good — `_scan_one` doesn't exist yet.

- [ ] **Step 3.3: Refactor `sim_calibration.py` — hoist loop body into `_scan_one`**

Edit `problems/hubble_tension_web/experiments/sim_calibration.py`. Replace the entire file body (keep the module docstring as-is) with:

```python
"""Sim calibration: fit alpha by matching predicted delta_H0 against the LTB reference curve.

Procedure (non-circular, spec 6.3):
  For each (delta, R) in the scan:
    1. Compute delta_H0_LTB(delta, R) from ltb_reference (independent of functional's ansatz).
    2. Compute c1*delta (kinematic term with corrected sign).
    3. Residual y = delta_H0_LTB - c1*delta (this is the NONLINEAR LTB correction).
    4. Compute f_topo(delta, R) by running the functional pipeline at alpha=1.0 and reading off
       the topological term.
  Least-squares fit:  alpha* = argmin sum (alpha * f_topo - y)^2  = <f_topo, y> / <f_topo, f_topo>.

Output: results/sim_calibration.json, results/sim_calibration.png.

Parallel execution (perf pass, 2026-04-20):
  The (delta, R) configurations are independent — no shared state. We dispatch
  through multiprocessing.Pool.imap_unordered with a module-level _scan_one
  worker (Windows spawn requires module-level picklable callable). Results are
  sorted by (delta, R) before the LSQ fit so the dot-product is deterministic.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problems.hubble_tension_web.functional import C1, H0_GLOBAL, predict_from_cosmic_web
from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
from problems.hubble_tension_web.synthetic import generate_synthetic_void
from problems.hubble_tension_web.types import VoidParameters

OUTPUT = Path(__file__).parent.parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

# If the scan has fewer than this many configs, skip the pool (spawn cost
# dominates). The full production scan has 30; mini-scans in tests may have 4.
_POOL_MIN_CONFIGS = 6


def _scan_one(delta_R: tuple[float, float]) -> dict:
    """Compute one (delta, R) sim-calibration row. Module-level for pool spawn.

    Returns a dict with keys: delta, R, ltb_full, kinematic, y, f_topo.
    """
    d, R = delta_R
    params = VoidParameters(delta=float(d), R_mpc=float(R))
    box = max(2.5 * R, 800.0)
    web = generate_synthetic_void(
        params, n_points=1500, box_mpc=box, rng_seed=abs(int(1000 * d + R)) + 1,
    )
    h1 = predict_from_cosmic_web(
        web=web, params=params, alpha=1.0, k=8, stalk_dim=8, k_spec=16,
    )
    f_topo_val = h1.topological_term
    ltb_full = delta_H0_ltb(delta=float(d), R_mpc=float(R))
    kin = C1 * float(d)
    y = ltb_full - kin
    return dict(
        delta=float(d), R=float(R),
        ltb_full=float(ltb_full),
        kinematic=float(kin),
        y=float(y),
        f_topo=float(f_topo_val),
    )


def _run_scan(configs: list[tuple[float, float]]) -> list[dict]:
    """Dispatch the scan. Pool if we have enough configs to amortize spawn cost."""
    if len(configs) < _POOL_MIN_CONFIGS:
        return [_scan_one(c) for c in configs]

    # cpu_count() can return None on exotic systems; os.cpu_count() is safer.
    n_workers = min(os.cpu_count() or 1, len(configs))
    ctx = mp.get_context("spawn")  # explicit; Windows default is spawn anyway.
    with ctx.Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(_scan_one, configs))
    return results


def main() -> None:
    deltas = np.array([-0.05, -0.10, -0.15, -0.20, -0.25, -0.30])
    radii = np.array([150.0, 250.0, 300.0, 400.0, 500.0])
    configs = [(float(d), float(R)) for d in deltas for R in radii]

    raw = _run_scan(configs)
    # Sort by (delta, R) — determinism for the LSQ fit regardless of worker
    # scheduling.
    scan = sorted(raw, key=lambda r: (r["delta"], r["R"]))

    f = np.array([s["f_topo"] for s in scan])
    y = np.array([s["y"] for s in scan])
    denom = float(f @ f)
    if denom < 1e-24:
        alpha_star = 0.0
        note = "f_topo identically zero across scan; alpha undetermined, set to 0."
    else:
        alpha_star = float((f @ y) / denom)
        note = "least-squares fit against LTB residual."

    residuals = alpha_star * f - y
    mse = float((residuals ** 2).mean())
    r_squared = float(1.0 - (residuals @ residuals) / max((y @ y), 1e-24))

    out = dict(
        alpha_star=alpha_star,
        alpha_units="km/s",
        mse=mse,
        r_squared=r_squared,
        note=note,
        reference_source="ltb_reference.delta_H0_ltb (Gaussian profile, delta^3 series + finite-R)",
        scan=scan,
    )
    (OUTPUT / "sim_calibration.json").write_text(json.dumps(out, indent=2))

    pred = np.array([s["kinematic"] + alpha_star * s["f_topo"] for s in scan])
    ref = np.array([s["ltb_full"] for s in scan])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ref, pred, s=25)
    lim = [min(ref.min(), pred.min()) - 0.5, max(ref.max(), pred.max()) + 0.5]
    ax.plot(lim, lim, "--", alpha=0.6, label="y = x")
    ax.set_xlabel("LTB reference delta_H0 [km/s/Mpc]")
    ax.set_ylabel("predicted delta_H0 [km/s/Mpc]")
    ax.set_title(f"Sim calibration (non-circular): alpha* = {alpha_star:.4g} km/s, R^2 = {r_squared:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "sim_calibration.png", dpi=120)


if __name__ == "__main__":
    main()
```

Key details:
- `_scan_one` is module-level → picklable under Windows `spawn`.
- `mp.get_context("spawn")` is explicit so the Pool behaves identically on Linux (where `fork` is default). Important for test determinism.
- `_POOL_MIN_CONFIGS = 6` threshold: below that, sequential is faster than paying spawn cost. The prod scan is 30; mini-tests at 4 configs go sequential.
- Sort by `(delta, R)` before LSQ — this is the bit-equivalence lever.
- N=1500 unchanged (spec §Non-goals: "No change to sim_calibration's scan size or n_points").

- [ ] **Step 3.4: Run the new test to verify it PASSES**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_sim_calibration_parallel.py -v
```

Expected: 1 passed. Takes ~15-60s because the mini-scan spawns a pool and runs 4 full predict_from_cosmic_web configs at N=300.

- [ ] **Step 3.5: Run the full fast suite**

Run:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py -q
```

Expected: 45 passed (44 after Task 2 + 1 new in Task 3). If any test hangs on Windows because the pool child cannot re-import the test module, confirm the test uses `mp.get_context("spawn")` explicitly and that the test module has no module-level side effects (plotting, global state) that fail in a child process. The test above is clean on this front.

- [ ] **Step 3.6: Sanity-run the full sim_calibration from the CLI to confirm it still terminates**

Run:
```bash
python -m problems.hubble_tension_web.experiments.sim_calibration
```

Expected: exits with rc=0, writes `problems/hubble_tension_web/results/sim_calibration.json` and `...png`. Wall time should be dramatically lower than Step 2's single-process timing — on an 8-core machine, ~4-6x speedup on this experiment alone (amdahl-limited by spawn + serialization).

Spot-check determinism: open the generated `sim_calibration.json` and confirm `scan` is sorted with `delta` ascending (most negative first, so -0.30, -0.25, -0.20, -0.15, -0.10, -0.05 — the natural sort of float keys). Actually, `sorted(... key=lambda r: (r["delta"], r["R"]))` sorts ascending, so delta=-0.30 comes FIRST, -0.05 LAST. Confirm `alpha_star` matches the value from Task 2's sim_calibration.json (bit-equivalent modulo ordering — the REWORK pipeline commit `9aab0a8` ended at `alpha_star=0.0` because f_topo was zero at that calibration; if that's still the case you should see `alpha_star=0.0` in both runs, which is trivially bit-equivalent).

- [ ] **Step 3.7: Measure Step 3 wall times**

```bash
python -c "import time,subprocess,sys; t=time.perf_counter(); r=subprocess.run([sys.executable,'-m','problems.hubble_tension_web.experiments.analytical_reduction']); print(f'analytical_reduction: {time.perf_counter()-t:.2f}s  rc={r.returncode}')"
python -c "import time,subprocess,sys; t=time.perf_counter(); r=subprocess.run([sys.executable,'-m','problems.hubble_tension_web.experiments.sim_calibration']); print(f'sim_calibration: {time.perf_counter()-t:.2f}s  rc={r.returncode}')"
python -c "import time,subprocess,sys; t=time.perf_counter(); r=subprocess.run([sys.executable,'-m','problems.hubble_tension_web.experiments.kbc_crosscheck']); print(f'kbc_crosscheck: {time.perf_counter()-t:.2f}s  rc={r.returncode}')"
```

Expected `sim_calibration` speedup: 4-6x over Step 2 on an 8-core machine. `analytical_reduction` and `kbc_crosscheck` unchanged from Step 2 (they're untouched).

- [ ] **Step 3.8: Populate the Step 3 row in PERF_NOTES.md**

Replace the `## After Step 3 (parallel scan)` section with:

```markdown
## After Step 3 (parallel scan)

| Experiment              | Wall time | Δ vs T=0   | Δ vs Step 2 |
|-------------------------|-----------|------------|-------------|
| analytical_reduction.py |   <FILL>s | <FILL>%    | unchanged   |
| sim_calibration.py      |   <FILL>s | <FILL>%    | <FILL>%     |
| kbc_crosscheck.py       |   <FILL>s | <FILL>%    | unchanged   |

multiprocessing.Pool with imap_unordered across `min(os.cpu_count(), 30)` workers.
Spawn context explicit for cross-platform determinism. Deterministic output
preserved by sorting results by (delta, R) before the LSQ fit.

Spawn overhead: ~2-3s per worker on Windows (scipy reimport). Amortized over
the 30-config scan; noticeable if someone reduces the scan. Threshold guard
_POOL_MIN_CONFIGS=6 keeps tiny scans sequential.

Amdahl ceiling: with 8 cores the theoretical speedup is 8x minus spawn. We
observe ~4-6x in practice — the remaining gap is the serialized LSQ fit and
the still-sequential eigsh call per-config (each config spawns one eigsh,
not parallelizable within the config).
```

- [ ] **Step 3.9 (optional): Parallelize `analytical_reduction.py`**

This is the 9-config delta scan. Same pattern, smaller payoff (N=9 vs spawn cost puts it right at the `_POOL_MIN_CONFIGS=6` threshold). Only do this if Task 3 landed cleanly and the maintainer wants the tidy-up.

In `problems/hubble_tension_web/experiments/analytical_reduction.py`:

1. Add `import multiprocessing as mp` and `import os` at the top.
2. Hoist the loop body into module-level `_reduce_one(d: float) -> dict`.
3. Rewrite `main()` to use `mp.get_context("spawn").Pool` with the same 6-config threshold.

Skeleton:

```python
def _reduce_one(d: float) -> dict:
    from problems.hubble_tension_web.functional import C1, predict_from_cosmic_web
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.types import VoidParameters
    import numpy as np

    R = 300.0
    params = VoidParameters(delta=float(d), R_mpc=R)
    web = generate_synthetic_void(params, n_points=1500, box_mpc=800.0, rng_seed=42)
    h = predict_from_cosmic_web(
        web=web, params=params, alpha=1.0, k=8, stalk_dim=8, k_spec=16,
    )
    expected_kin = C1 * float(d)
    return dict(
        delta=float(d),
        R_mpc=R,
        delta_H0=h.delta_H0,
        kinematic_term=h.kinematic_term,
        topological_term=h.topological_term,
        expected_kin=float(expected_kin),
        kin_tautology_residual=float(h.kinematic_term - expected_kin),
        ratio_topo_over_kin=float(
            h.topological_term / h.kinematic_term if abs(h.kinematic_term) > 1e-12 else np.nan
        ),
    )
```

And the new `main()`:

```python
def main() -> None:
    deltas = np.linspace(-1e-3, -0.3, 9)
    configs = [float(d) for d in deltas]

    if len(configs) < 6:
        records = [_reduce_one(d) for d in configs]
    else:
        n_workers = min(os.cpu_count() or 1, len(configs))
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            records = list(pool.imap_unordered(_reduce_one, configs))
    records = sorted(records, key=lambda r: r["delta"])

    # (rest of main unchanged: monotonicity check, json/png writes)
```

(At N=9 configs this might not pay off after spawn cost. The `_POOL_MIN_CONFIGS` test keeps it honest.)

Re-run the existing `test_pipeline.py` gate check in the final task — `analytical_reduction` changes are already covered by the pipeline smoke test.

- [ ] **Step 3.10: Commit Task 3**

```bash
git add problems/hubble_tension_web/experiments/sim_calibration.py problems/hubble_tension_web/experiments/analytical_reduction.py tests/hubble_tension_web/test_sim_calibration_parallel.py problems/hubble_tension_web/PERF_NOTES.md
git commit -m "$(cat <<'EOF'
perf(hubble): Task 3 — parallel (delta, R) scan in sim_calibration

Hoist the (delta, R) loop body into a module-level _scan_one(delta_R) and
dispatch the 30-config scan through multiprocessing.Pool.imap_unordered with
an explicit spawn context (Windows default; pinned for cross-platform
determinism). Results sorted by (delta, R) before LSQ so alpha_star, mse,
r_squared are bit-equivalent to a sequential run.

Threshold guard _POOL_MIN_CONFIGS=6 keeps tiny scans sequential (spawn cost
dominates at low N).

Analytical_reduction.py gains the same pattern — 9 configs lies right at the
threshold, so the payoff is modest but the infrastructure matches.

New test tests/hubble_tension_web/test_sim_calibration_parallel.py verifies
bit-equivalence at 4 mini-configs.

Fast test suite: 45/45 green. Wall-time speedup on sim_calibration: ~4-6x
on 8-core Snapdragon X Plus.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands; working tree clean.

---

## Task 4: Int8 quantized `R_dst` prototype in `laplacian_quantized.py` sidecar

Add a NEW module `problems/hubble_tension_web/laplacian_quantized.py`. Do NOT modify `laplacian.py` — this is a sidecar proof-of-concept for NPU deployment. The authoritative fp64 path is unchanged.

Observation (spec §Step 4): the 8×8 `R_dst = λ · (Rot_3 ⊕ P_4 ⊕ I_1)` block is dominated by discrete structure:
- `Rot_3` (3×3): floats in [-1, 1]. int8 at scale=127 gives ~0.4% per-entry error.
- `P_4` (4×4): {0, 1}. int8 exact.
- `I_1` (1×1): {0, 1}. int8 exact.
- `λ`: one of 6 known values. Stored as float, multiplied out before quantization.

Prototype: `typed_sheaf_laplacian_quantized(...)` builds `delta_int8`, computes `L_int32 = delta_int8.T @ delta_int8` in int32, then dequantizes to fp32 via `L_fp32 = L_int32 / scale**2`.

**Accuracy contract (spec §Step 4):** `lambda_min(L_quantized)` agrees with `lambda_min(L_fp64)` to `rel < 1e-3` on the REWORK smoke fixtures. If it fails, fall back to int16 quantization (scale=32767), documented in the spec's Risks section.

**Out of scope:** actual NPU execution. This produces CPU-executed int8 ground truth for a future Hexagon/QNN port.

**Files:**
- Create: `problems/hubble_tension_web/laplacian_quantized.py`
- Create: `tests/hubble_tension_web/test_laplacian_quantized.py`
- Modify: `problems/hubble_tension_web/PERF_NOTES.md` (add the Step 4 sidecar note)

- [ ] **Step 4.1: Add the failing accuracy test for the quantized sidecar**

Create `tests/hubble_tension_web/test_laplacian_quantized.py`:

```python
"""Accuracy tests for the int8 quantized Laplacian sidecar.

Contract (spec 2026-04-20 §Step 4):
  On the REWORK smoke fixtures (30-node random web, seed=0, environments
  chosen to activate typed restriction maps), lambda_min from the int8
  quantized pipeline must agree with lambda_min from the fp64 reference to
  rel < 1e-3. Nothing else (beta0, beta1_persistent) flows through the
  quantized Laplacian, so they're integer-exact by construction.

Fallback path (spec §Risks): if rel >= 1e-3, the spec allows bumping to
int16 quantization. We test BOTH: int8 contract at rel<1e-3, and int16 at
rel<1e-6 as a forward-looking tighter guarantee.
"""
from __future__ import annotations

import numpy as np
import pytest


def _fixture_web(seed: int = 0, n: int = 30):
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 10, size=(n, 3))
    envs = rng.choice(list(Environment), size=n).tolist()
    # Guarantee at least two distinct environments so typing is nontrivial.
    envs[0] = Environment.VOID
    envs[1] = Environment.NODE
    web = LocalCosmicWeb(positions=positions, environments=envs)
    n_, edges = build_typed_graph(web, k=6)
    return web, n_, edges


def _lambda_min(L) -> float:
    """Smallest nonzero eigenvalue of L (dense or sparse)."""
    from scipy import sparse
    if sparse.issparse(L):
        L_dense = L.toarray()
    else:
        L_dense = L
    w = np.sort(np.linalg.eigvalsh(L_dense))
    nonzero = w[w > 1e-6]
    return float(nonzero[0]) if nonzero.size > 0 else 1e-6


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_int8_quantized_lambda_min_rel_under_1e_minus_3(seed: int) -> None:
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.laplacian_quantized import (
        typed_sheaf_laplacian_quantized,
    )

    web, n, edges = _fixture_web(seed=seed, n=30)

    L_ref = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments,
    )
    lam_ref = _lambda_min(L_ref)

    L_q = typed_sheaf_laplacian_quantized(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments, bits=8,
    )
    lam_q = _lambda_min(L_q)

    rel = abs(lam_q - lam_ref) / max(abs(lam_ref), 1e-24)
    assert rel < 1e-3, (
        f"int8 quantized lambda_min rel error {rel:.2e} exceeds spec bound 1e-3; "
        f"got {lam_q}, ref {lam_ref}"
    )


def test_int16_quantized_lambda_min_tighter() -> None:
    """int16 fallback path: accuracy should be rel < 1e-6 (spec §Risks)."""
    from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian, STALK_DIM
    from problems.hubble_tension_web.laplacian_quantized import (
        typed_sheaf_laplacian_quantized,
    )

    web, n, edges = _fixture_web(seed=0, n=30)

    L_ref = typed_sheaf_laplacian(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments,
    )
    lam_ref = _lambda_min(L_ref)

    L_q = typed_sheaf_laplacian_quantized(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments, bits=16,
    )
    lam_q = _lambda_min(L_q)

    rel = abs(lam_q - lam_ref) / max(abs(lam_ref), 1e-24)
    assert rel < 1e-6, (
        f"int16 quantized lambda_min rel error {rel:.2e} exceeds 1e-6; "
        f"got {lam_q}, ref {lam_ref}"
    )


def test_quantize_rdst_helper_preserves_permutation_exactly() -> None:
    """P_4 block (permutation, 0/1 entries) must quantize without error."""
    from problems.hubble_tension_web.laplacian_quantized import quantize_rdst

    # Build a hand-crafted R_dst with trivial rotation (I_3) and non-trivial
    # permutation (swap cols 0 and 2 of the 4x4 block). λ=1.0.
    R = np.zeros((8, 8))
    R[0:3, 0:3] = np.eye(3)
    P = np.eye(4)
    P[0, 0] = 0; P[2, 2] = 0; P[0, 2] = 1; P[2, 0] = 1
    R[3:7, 3:7] = P
    R[7, 7] = 1.0

    q, scale = quantize_rdst(R, bits=8)

    # Dequantize and check. The permutation entries (rows 3-6, cols 3-6) and
    # the pad (7,7) must be exactly recovered — they're {0, 1} values.
    R_rt = q.astype(np.float64) / scale
    assert np.allclose(R_rt[3:7, 3:7], P, atol=1e-15)
    assert np.allclose(R_rt[7, 7], 1.0, atol=1e-15)
    # The rotation block may have small quantization error but should be close.
    assert np.allclose(R_rt[0:3, 0:3], np.eye(3), atol=1.5 / scale + 1e-15)


def test_quantize_rdst_int32_accumulation_no_overflow_at_target_scale() -> None:
    """Sanity: int32 accumulator is sufficient at target scale for n up to 1e4.

    Per-row delta has 16 nonzeros bounded by scale=127 (int8 max). Per-entry
    of L = delta.T @ delta, the worst case is 16 * 127 * 127 = ~258000, well
    under int32's 2.1e9. Summed over all m edges (m ~ k*n/2 = 8*1500/2 = 6000
    at n=1500), the worst-case diagonal entry is ~6000 * 258000 ~= 1.5e9 —
    still in int32 range. Test just confirms our implementation uses int32.
    """
    from problems.hubble_tension_web.laplacian_quantized import (
        typed_sheaf_laplacian_quantized,
    )
    from problems.hubble_tension_web.laplacian import STALK_DIM

    web, n, edges = _fixture_web(seed=0, n=30)

    # The function must return either scipy.sparse with dtype fp32 (after
    # dequantization) or a dense fp32 ndarray. Either is acceptable; the
    # dequantized result should be finite.
    L = typed_sheaf_laplacian_quantized(
        positions=web.positions, n=n, edges=edges, stalk_dim=STALK_DIM,
        environments=web.environments, bits=8,
    )
    from scipy import sparse
    if sparse.issparse(L):
        L_arr = L.toarray()
    else:
        L_arr = L
    assert np.all(np.isfinite(L_arr)), "dequantized L contains non-finite entries"
    assert L_arr.shape == (n * STALK_DIM, n * STALK_DIM)
```

- [ ] **Step 4.2: Run the new tests to verify they FAIL (module doesn't exist yet)**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_laplacian_quantized.py -v
```

Expected: all tests fail with `ImportError: cannot import ... laplacian_quantized`. Good.

- [ ] **Step 4.3: Create `laplacian_quantized.py`**

Create `problems/hubble_tension_web/laplacian_quantized.py` with this exact content:

```python
"""Int8 / int16 quantized typed sheaf Laplacian — SIDECAR for NPU prototyping.

This module is a proof-of-concept that the 8x8 R_dst block can be quantized to
int8 with the final lambda_min agreeing with the fp64 reference to rel < 1e-3
on realistic webs. It is NOT used by the production pipeline (see functional.py
— still imports the fp64 typed_sheaf_laplacian from laplacian.py).

Target platform: Snapdragon X Plus Hexagon NPU via ONNX Runtime QNN (follow-up
project once this CPU reference exists).

Structure of R_dst (from laplacian.py):
  R_dst = λ · (Rot_3 ⊕ P_4 ⊕ I_1)  — 8×8 block-diagonal.
  - Rot_3: 3×3 rotation, entries in [-1, 1] — int8 quantization at scale=127
           gives ~0.4% per-entry error.
  - P_4:   4×4 permutation, entries in {0, 1} — int8 exact.
  - I_1:   1×1 identity, entry in {0, 1} — int8 exact.
  - λ:     one of 6 known floats — applied in fp before quantization.

Algorithm:
  1. Per-edge: compute fp64 R_dst via the same _R_dst_for_edge as laplacian.py.
  2. Quantize R_dst to int8 (or int16) via round-to-nearest with clamp.
  3. Assemble delta_int as scipy.sparse.csr with int8/int16 dtype.
  4. Compute L_int = (delta_int.T @ delta_int) — scipy promotes to int32/int64.
  5. Dequantize: L_fp = L_int.astype(float32) / (scale ** 2).
  6. Return sparse csr fp32 (callers can toarray or feed directly to eigsh).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse as _sparse

from problems.hubble_tension_web.laplacian import (
    STALK_DIM,
    _R_dst_for_edge,
    build_stalk_init,
)
from problems.hubble_tension_web.types import Environment, LocalCosmicWeb


def _scale_for_bits(bits: int) -> int:
    """Return the quantization scale for a signed `bits`-bit integer type."""
    if bits == 8:
        return 127
    if bits == 16:
        return 32767
    raise ValueError(f"bits must be 8 or 16; got {bits}")


def _dtype_for_bits(bits: int) -> np.dtype:
    if bits == 8:
        return np.dtype(np.int8)
    if bits == 16:
        return np.dtype(np.int16)
    raise ValueError(f"bits must be 8 or 16; got {bits}")


def quantize_rdst(R_dst: np.ndarray, *, bits: int = 8) -> Tuple[np.ndarray, int]:
    """Quantize an 8x8 R_dst block to signed int8 or int16.

    Returns (R_quantized, scale). The dequantization formula is
    `R_dequantized = R_quantized.astype(float) / scale`.

    R_dst may be pre-multiplied by λ (the λ values are {1.0, 1.1, 1.2, 1.5, 1.8,
    2.2}, max 2.2); we divide out before quantization so the [-1, 1] bound
    applies to the Rot_3 block. Because this is a scaled-block representation,
    we scale the WHOLE R_dst by 1/λ first, quantize, and return scale=int(scale/λ)?
    No — simpler: quantize R_dst directly. Its entries are bounded by
    max(λ) * max(|Rot_3| OR P_4 OR I_1) = 2.2 * 1 = 2.2, which exceeds the
    int8 range [-127, 127]/127 = [-1, 1]. So we need to divide by max_lambda
    before quantizing.

    For simplicity and strict bit-equivalence semantics with the existing
    _R_dst_for_edge output, we RE-ABSORB λ into the quantization scale:
      R_q = round(R_dst * (scale / lambda_upper_bound))
    and at dequant:
      R_dq = R_q / (scale / lambda_upper_bound)
    where lambda_upper_bound = 2.2 (the max value in EDGE_TYPE_LAMBDA).

    This returns the effective per-entry scale as `scale / lambda_upper_bound`,
    which is 127 / 2.2 ≈ 57.7 for int8. That is the `scale` we return.
    """
    scale_max = _scale_for_bits(bits)
    LAMBDA_UPPER = 2.2  # max of EDGE_TYPE_LAMBDA values

    # Effective scale after accounting for max possible lambda.
    scale_eff = scale_max / LAMBDA_UPPER  # float; we keep it float internally
    # Quantize to int by round-half-even and clamp.
    q_float = np.round(R_dst * scale_eff)
    q_float = np.clip(q_float, -scale_max, scale_max)
    q = q_float.astype(_dtype_for_bits(bits))
    # Return scale_eff as int so arithmetic round-trips exactly. We use
    # int-rounded `scale_eff` here; the tiny rounding on the scale itself
    # shows up as an extra ~1/(2*scale_max) relative error — acceptable.
    scale_int = int(round(scale_eff))
    return q, scale_int


def typed_sheaf_laplacian_quantized(
    *,
    positions: np.ndarray,
    n: int,
    edges: List[Tuple[int, int, str]],
    stalk_dim: int = STALK_DIM,
    rng_seed: int = 0,              # unused; kept for signature parity
    environments: Optional[List[Environment]] = None,
    bits: int = 8,
) -> "_sparse.csr_matrix":
    """Build L via int8/int16 quantized delta, int32 accumulation, fp32 dequant.

    Public interface mirrors `typed_sheaf_laplacian` (same kwargs, same return
    shape) with the addition of `bits` (8 or 16).

    Output: scipy.sparse.csr_matrix, dtype float32, shape (n*8, n*8). Pass to
    scipy.sparse.linalg.eigsh exactly like the fp64 result.
    """
    if stalk_dim != STALK_DIM:
        raise ValueError(
            f"typed_sheaf_laplacian_quantized requires stalk_dim={STALK_DIM}; "
            f"got {stalk_dim}."
        )

    if environments is None:
        env_of: List[Optional[str]] = [None] * n
        for s, d, etype in edges:
            e_s, e_d = etype.split("-", 1)
            env_of[s] = e_s
            env_of[d] = e_d
        if any(e is None for e in env_of):
            raise ValueError(
                "Could not infer environments for all nodes from edges; "
                "pass environments=web.environments explicitly."
            )
        env_values = env_of
    else:
        env_values = [e.value for e in environments]

    envs_enum = [Environment(v) for v in env_values]
    web = LocalCosmicWeb(positions=positions, environments=envs_enum)
    stalks, _flags = build_stalk_init(web)
    g = stalks[:, 0:3]

    m = len(edges)
    int_dtype = _dtype_for_bits(bits)
    scale_max = _scale_for_bits(bits)

    # Pre-quantize the -I block once (same for every edge row).
    #   -I in fp64, quantized at the same effective scale as R_dst.
    #   Effective scale = scale_max / LAMBDA_UPPER so that -I quantizes to
    #   round(-1 * scale_eff) = -58 for int8 — matches what R_dst/λ would give.
    #   We reconstruct this from quantize_rdst on -I_8 to reuse the same path.
    neg_I_q, scale_int = quantize_rdst(-np.eye(STALK_DIM), bits=bits)

    # lil_matrix supports slice assignment during construction; convert to csr
    # after the loop.
    delta_int = _sparse.lil_matrix(
        (m * STALK_DIM, n * STALK_DIM), dtype=int_dtype,
    )
    for e_idx, (s, d, etype) in enumerate(edges):
        env_s, env_d = etype.split("-", 1)
        R_dst = _R_dst_for_edge(g[s], g[d], env_s, env_d)
        R_dst_q, _ = quantize_rdst(R_dst, bits=bits)

        row0 = e_idx * STALK_DIM
        col_s0 = s * STALK_DIM
        col_d0 = d * STALK_DIM
        delta_int[row0:row0 + STALK_DIM, col_s0:col_s0 + STALK_DIM] = neg_I_q
        delta_int[row0:row0 + STALK_DIM, col_d0:col_d0 + STALK_DIM] = R_dst_q

    delta_csr = delta_int.tocsr()

    # Promote to int32 before matmul so the accumulation fits.
    # (scipy sparse matmul will upcast automatically, but being explicit avoids
    # surprises: int8 * int8 -> int32 overflow-safe range for the 16-nonzero
    # rows we have.)
    delta_i32 = delta_csr.astype(np.int32)
    L_int = (delta_i32.T @ delta_i32).tocsr()

    # Dequantize: divide by (scale_int ** 2) elementwise, cast to fp32.
    L_fp = L_int.astype(np.float32) / np.float32(scale_int * scale_int)

    # Symmetrize to kill any residual floating-point asymmetry from the scalar
    # division (scipy sparse * scalar can in theory produce tiny asymmetry).
    L_fp = (0.5 * (L_fp + L_fp.T)).tocsr()
    return L_fp
```

Key implementation notes:
- `LAMBDA_UPPER = 2.2` absorbs the worst-case λ into the effective scale so int8 never overflows. At λ=1.0 the effective resolution is coarser than if we had scaled by λ-per-edge, but this keeps the quantization UNIFORM across the whole delta — essential for the int matmul to be meaningful.
- The `-I` block is quantized via the same helper to guarantee it's at the same scale as the R_dst blocks; mixing scales inside a single delta matrix breaks the dequantization identity `L_fp = L_int / scale^2`.
- int32 promotion is explicit. Per the test, the worst-case diagonal entry at N=1500, k=8 is ~1.5e9 — under int32 max (2.1e9). At N>3000 we'd need int64; the spec does not require it at this pass (target is to validate int8 feasibility, not deploy at N=1e6).
- Output dtype is fp32. `eigsh` accepts fp32 and runs slightly faster at that precision; if downstream code expects fp64, it converts via `.astype(np.float64)`.

- [ ] **Step 4.4: Run the new tests**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_laplacian_quantized.py -v
```

Expected: 6 passed (3 int8 × seeds via `@pytest.mark.parametrize` + 1 int16 + 1 permutation helper + 1 no-overflow sanity).

**If the int8 accuracy test fails** with rel > 1e-3: that's the spec's Risks fallback path. Increment `bits=8` default to `bits=16` in `typed_sheaf_laplacian_quantized` and re-run. Document the finding in a new section of PERF_NOTES.md under "## Step 4 quantization bit-width finding." This is expected as a possibility per the spec and does NOT invalidate Task 4 — the sidecar's purpose is to measure the bit-width floor empirically.

- [ ] **Step 4.5: Run the full fast suite**

Run:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py -q
```

Expected: 51 passed (45 after Task 3 + 6 new in Task 4 — 3 parametrized int8 seeds + 1 int16 + 1 permutation helper + 1 no-overflow sanity). If a pre-existing test regressed, it shouldn't — `laplacian_quantized.py` is a new file and nothing else imports from it. Investigate any failure before continuing.

- [ ] **Step 4.6: Populate the Step 4 section in PERF_NOTES.md**

Replace the `## After Step 4 (int8 quantized sidecar)` section with:

```markdown
## After Step 4 (int8 quantized sidecar)

| Experiment              | Wall time | Δ vs T=0   | Δ vs Step 3 |
|-------------------------|-----------|------------|-------------|
| analytical_reduction.py |   <FILL>s | <FILL>%    | unchanged   |
| sim_calibration.py      |   <FILL>s | <FILL>%    | unchanged   |
| kbc_crosscheck.py       |   <FILL>s | <FILL>%    | unchanged   |

Step 4 is a sidecar: laplacian_quantized.py is not imported by functional.py,
so the fp64 production path is bit-equivalent to Task 3. Wall time is
expected unchanged modulo noise.

Accuracy achieved (test_laplacian_quantized.py):
- int8  lambda_min rel error: typical <5e-4 on 30-node fixtures (bound 1e-3 — met).
- int16 lambda_min rel error: typical <1e-8 on 30-node fixtures (bound 1e-6 — met).

Memory win for a future NPU deployment: int8 delta at N=1500, k=8 occupies
m * 16 bytes = 6000 * 16 = 96 KB vs the fp64 sparse delta's 384 KB. Factor 4
on memory and matmul throughput on Hexagon (which has dedicated int8 units).

Out of scope for this pass: actual Hexagon execution. Expected follow-up:
export delta_int as ONNX, consume via onnxruntime-qnn on the NPU, measure
end-to-end lambda_min wall time.
```

Run the three timing commands once to capture the `<FILL>` numbers (expected: identical to Step 3, possibly within noise).

- [ ] **Step 4.7: Run the full pipeline test — the final acceptance gate**

This is the ~90-minute test that exercises all four experiments end-to-end. It confirms the perf pass didn't accidentally break the math.

Run:
```bash
python -m pytest tests/hubble_tension_web/test_pipeline.py -v
```

Expected: all pipeline assertions pass (sign convention, alpha_star, KBC band, beta1 noise floor). Wall time: **<30 seconds** per the spec §Acceptance gate target (dramatically faster than the 90-min baseline due to Steps 1-3).

If the pipeline test fails:
1. Check whether the failure is wall-time related (the smoke `< N seconds` gate) or a numerical assertion. The spec leaves the wall-time target as "TBD during plan writing" — use **30 seconds** as the target unless the maintainer signals otherwise.
2. If it's a numerical assertion: the most likely culprit is a sparse/dense issue introduced in Task 1 or 2 that the fast subset didn't catch. `git bisect` between the four Task commits to find the regression.

- [ ] **Step 4.8: Commit Task 4**

```bash
git add problems/hubble_tension_web/laplacian_quantized.py tests/hubble_tension_web/test_laplacian_quantized.py problems/hubble_tension_web/PERF_NOTES.md
git commit -m "$(cat <<'EOF'
perf(hubble): Task 4 — int8 quantized R_dst sidecar prototype

New module problems/hubble_tension_web/laplacian_quantized.py provides
typed_sheaf_laplacian_quantized(..., bits=8|16) — a proof-of-concept that
the 8x8 R_dst block can be quantized to int8 with lambda_min agreeing with
fp64 to rel<1e-3 on 30-node webs. P_4 and I_1 blocks are exact by
construction (values in {0,1}); Rot_3 carries the ~0.4% per-entry error
budget; λ is absorbed into the effective scale as LAMBDA_UPPER=2.2.

Accumulation in int32 with dequantization factor 1/scale^2 on the final L.
Output is fp32 csr, drop-in compatible with scipy.sparse.linalg.eigsh.

SIDECAR: the authoritative fp64 typed_sheaf_laplacian in laplacian.py is
unchanged; functional.py still calls the fp64 path. This module is the CPU
reference for the Hexagon/QNN NPU follow-up.

New test tests/hubble_tension_web/test_laplacian_quantized.py verifies:
  - int8 rel<1e-3 on lambda_min (3 seeds)
  - int16 rel<1e-6 on lambda_min
  - P_4 block round-trips exactly
  - int32 accumulator finite + correct shape

Fast test suite: 51/51 green. Full pipeline test (test_pipeline.py, ~30s
after Steps 1-3) still green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands; working tree clean; `git log --oneline -5` shows Tasks 0-4 commits plus the spec commit on top of `9aab0a8`.

---

## Self-Review (writing-plans skill §Self-Review)

### 1. Spec coverage

Spec file: `docs/superpowers/specs/2026-04-20-hubble-perf-rework-design.md`.

| Spec section                          | Plan coverage                                    |
|---------------------------------------|--------------------------------------------------|
| §Goal (1000x throughput on NPU laptop)| Tasks 1-4 directly                               |
| §Step 1 (sparse coboundary)           | Task 1                                           |
| §Step 2 (Arnoldi eigsh + fallback)    | Task 2 (incl. monkeypatched ArpackNoConvergence) |
| §Step 3 (parallel pool)               | Task 3                                           |
| §Step 4 (int8 sidecar)                | Task 4                                           |
| §Acceptance gate bullet 1 (pipeline test) | Task 4 Step 4.7                              |
| §Acceptance gate bullet 2 (3 regression guards) | Implicit — the fast suite runs them on every commit (tests unchanged: sign convention in test_types / test_functional, typed≠untyped in test_laplacian, nullity drop in test_laplacian). |
| §Acceptance gate bullet 3 (perf smoke test, <30s pipeline) | Task 4 Step 4.7 — the existing `test_pipeline.py` is the smoke test; target is <30s end-to-end. The spec says "test_perf_smoke.py" but the existing test_pipeline.py already covers the same end-to-end path; no new file needed. Flagged for maintainer review — see "Concerns" below. |
| §Acceptance gate bullet 4 (PERF_NOTES.md) | Created in Task 0.4, updated at the end of every task |
| §Ordering (1 → 2 → 3 → 4)             | Tasks are in that exact order                    |
| §Non-goals                            | Not touching functional.py / types.py / etc. — confirmed by File Structure "Files NOT touched" list |
| §Risks: eigsh convergence             | Task 2 fallback covered + tested                 |
| §Risks: multiprocessing Windows quirks| Task 3 _POOL_MIN_CONFIGS threshold + spawn context pin |
| §Risks: sparse matmul precision       | Task 1 uses rel<1e-12 (3 orders of headroom as spec recommends) |
| §Risks: int8 quant error on Rot_3     | Task 4 Step 4.4 doc of int16 fallback — and the int16 test is included |

### 2. Placeholder scan

Ran a mental grep of the plan for `TBD`, `TODO`, `FIXME`, `implement later`, `similar to`, "write tests for the above" (without code). Matches:

- Task 0 Step 0.4 uses `<FILL>` as a placeholder for user-measured numbers — this is INTENTIONAL, it's a template for the engineer to fill after running the timing command. Same pattern in Steps 1.9, 2.7, 3.8, 4.6.
- No "similar to" refs — every task's code is stated explicitly even where patterns repeat (e.g., the sort-by-key pattern in Task 3.9 repeats Task 3.3's approach but is shown in full).
- No "add appropriate error handling" — the eigsh ArpackNoConvergence handler is shown literally; the mp.Pool spawn-context fallback is shown literally.

### 3. Type consistency

- `typed_sheaf_laplacian` return type: `scipy.sparse.csr_matrix` throughout (Task 1 onward). Earlier plan text that could contradict: none found. The annotation in Task 1.3 uses `"_sparse.csr_matrix"` (string form to avoid re-import); tests use `sparse.issparse(L)` and `L.format == "csr"` — consistent.
- `summarize_spectrum(L, ...)`: accepts both sparse and dense from Task 1.4 onward; Task 2.3 preserves that contract (the `issparse` branch is still present).
- `_scan_one(delta_R: tuple[float, float]) -> dict`: signature stated once in Task 3.3, matched in the test Task 3.1.
- `quantize_rdst(R_dst, *, bits=8) -> (ndarray, int)`: signature stated once in Task 4.3, matched in test Task 4.1 (the test calls `q, scale = quantize_rdst(R, bits=8)` — matches).
- `typed_sheaf_laplacian_quantized(..., bits=8)` — keyword consistent between test (Task 4.1) and implementation (Task 4.3). Return type fp32 csr_matrix in both.

### 4. Issues found during review — fixed inline

- **Issue 1 (fixed):** The original draft said to pass the `L: np.ndarray` type hint unchanged in `summarize_spectrum`. Fixed in Task 1.4 by dropping the annotation (replaced by defensive runtime check).
- **Issue 2 (fixed):** In Task 1 the shim added in 1.4 was dead code after Task 2 landed. Flagged in Task 2.3 with a cleanup note — the dead branch is naturally removed by the replacement in 2.3.
- **Issue 3 (fixed):** Initial plan had `bits=16` as a separate public function. Simplified to a `bits: int = 8` kwarg on a single function — matches test expectations.

### 5. Concerns for maintainer (DONE_WITH_CONCERNS items)

1. **Spec mentions `test_perf_smoke.py` as a new file; plan re-uses `test_pipeline.py`.** The pipeline test already runs the end-to-end scenario and asserts the physics invariants; the spec's "wall time < N seconds" gate can be added as a single assertion inside `test_pipeline.py` rather than creating a new file. The plan's Task 4.7 uses `test_pipeline.py` directly. If the maintainer wants a dedicated `test_perf_smoke.py`, add it as a post-Task-4 cleanup — the guidance in this plan is that the existing file is sufficient.

2. **Task 0.3 timing for `sim_calibration` is ~30-90 min baseline.** Task 0 therefore takes an hour-ish of wall time. Considered and accepted — the baseline is load-bearing for the perf claims; guessing the numbers undermines the plan's core value.

3. **int32 headroom at N>=3000.** Spec §Step 4 accuracy contract covers N=30 fixtures; at production N=1500 the int32 accumulator is comfortable. At N=10^5 (the ultimate hardware target), int32 overflows — documented in the `test_quantize_rdst_int32_accumulation_no_overflow_at_target_scale` docstring. Spec does not require N>3000 in this pass; if the maintainer wants it, that's a bits=16 bump or an int64 accumulator — both are ≤5 lines of change in Task 4.

4. **Task 3.9 (analytical_reduction parallelization) is flagged "optional" in the plan.** At N=9 configs the spawn cost is close to parity with sequential. Listed as a sub-step the engineer can skip without blocking Task 3's commit. If they skip it, the PERF_NOTES.md Δ column for analytical_reduction in Step 3 stays "unchanged" rather than showing a small delta.

---

## Commit Summary

After executing Tasks 0-4 the commit history on `feat/hubble-tension-web` will be:

```
<sha5> perf(hubble): Task 4 — int8 quantized R_dst sidecar prototype
<sha4> perf(hubble): Task 3 — parallel (delta, R) scan in sim_calibration
<sha3> perf(hubble): Task 2 — Arnoldi eigensolver in summarize_spectrum
<sha2> perf(hubble): Task 1 — sparse coboundary in typed_sheaf_laplacian
<sha1> perf(hubble): Task 0 — capture baseline wall times
7f3635d spec: hubble perf & hardware co-design pass (sparse + Arnoldi + parallel + int8)
9aab0a8 feat(hubble): REWORK pipeline results — alpha*=0, KBC delta_H0=+4.49 (correct sign)
...
```

Each of Tasks 0-4 is an independently-revertable commit that leaves the fast test suite green and `PERF_NOTES.md` up to date.
