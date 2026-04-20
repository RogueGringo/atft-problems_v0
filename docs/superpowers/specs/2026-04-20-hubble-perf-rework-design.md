# Hubble-Tension-Web: Performance & Hardware Co-Design Pass

**Date:** 2026-04-20
**Branch target:** `feat/hubble-perf-rework` (new branch off `feat/hubble-tension-web`, i.e. atop `9aab0a8`)
**Predecessor:** the math-rework (`2026-04-19-hubble-tension-web-REWORK-design.md` and its plan). The math is now correct and tested (33/33 non-pipeline green; full pipeline green with +4.49 km/s/Mpc signed KBC verdict). This pass is **optimization and hardware co-design**, not a math change. Behavior must remain bit-equivalent (to numerical tolerance) for steps 1-3, and within a clearly-bounded error envelope for step 4.

## Goal

Take the hubble_tension_web pipeline from "runs on a single 1500-point spherical-cow synthetic void in ~2-3 min" to "runs on real N-body galaxy catalogs at 10^5 - 10^6 points in comparable wall time" — a ~1000x throughput target on the user's Snapdragon X Plus laptop.

## Motivation

The REWORK pipeline proved the math is honest. The remaining cliff is wall time:

- `laplacian.py` builds a dense `delta` coboundary of shape `(m * STALK_DIM, n * STALK_DIM)` — for N=2500 points, k=8 neighbors, stalk_dim=8 that's (160000, 20000) at ~26 GB fp64. We already hit the swap wall at N=2500; anything resembling a real galaxy catalog (N >> 10^5) is mathematically impossible in dense form.
- `spectrum.py` calls `np.linalg.eigvalsh(L)` to get the smallest 16 eigenvalues out of thousands. O(n^3) operations to throw away >99% of them.
- `experiments/sim_calibration.py` runs a 30-point (delta, R) scan sequentially — no use of the 8 available CPU cores.
- **Discrete-structure opportunity:** most of `R_dst = λ · (Rot_3 ⊕ P_4 ⊕ I_1)` is literally integer/Boolean. `P_4` is a permutation; `I_1` is {0,1}; `λ` is a 6-value lookup table (log2(6) ≈ 3 bits). Only the 3×3 `Rot_3` block carries continuous data. The problem's apparent float-cost is largely an artifact of how we encoded it.

## Design (4 steps, ordered)

### Step 1 — Sparse coboundary in `laplacian.py`

Rewrite `typed_sheaf_laplacian` to assemble `delta` as `scipy.sparse.csr_matrix` (or `lil_matrix` during construction, then `.tocsr()`). Compute `L = delta.T @ delta` with sparse matmul and symmetrize via `L = 0.5 * (L + L.T)` still in sparse form.

Key invariant: each row of `delta` has exactly `2 * STALK_DIM = 16` nonzeros (the `-I` block on col_s and the `R_dst` block on col_d). Density ≈ 16 / (n * STALK_DIM) = 16 / (8n) = 2/n. At n=1500 this is 0.13% nonzero — a ~750× memory win. At n=100000 it becomes a ~50000× win.

**Behavior contract:** `L` as a NumPy dense matrix must be bit-equivalent (to 1e-12) to the output of the current implementation on the same inputs. Test added: `test_sparse_laplacian_matches_dense` parameterized over 3 small random webs. The public return type of `typed_sheaf_laplacian` changes from `np.ndarray` to `scipy.sparse.csr_matrix`; callers in `spectrum.py` adapt (trivially — `eigvalsh` wants dense, `eigsh` wants sparse, and Step 2 changes which we call).

**Blast radius:** `laplacian.py` (public signature preserved), `spectrum.py` (eigensolver choice), `tests/hubble_tension_web/test_laplacian.py` (new equivalence test).

### Step 2 — Arnoldi eigensolver in `spectrum.py`

Replace `np.linalg.eigvalsh(L)` with `scipy.sparse.linalg.eigsh(L, k=k_spec + 4, which='SA', ...)` (SA = smallest algebraic). Take the bottom `k_spec` as before, and derive `lambda_min` as the smallest non-zero from the returned set (the extra 4 eigenvalues guard against the known-degenerate kernel eating up the slots at small graph sizes).

**Accuracy contract:** the returned `spectrum[:k_spec]` must agree with the dense result to `rel < 1e-6` on randomized 60-point webs — tighter than fp32 but looser than bit-equivalent. `lambda_min` must agree to `rel < 1e-9` (it's the physically important one and matters for `f_topo`). Test added: `test_eigsh_vs_eigvalsh_small_bottom`.

**Blast radius:** `spectrum.py` only.

**Fallback:** if `eigsh` fails to converge for pathological cases (disconnected components with stalk_dim * beta_0 zero modes), fall back to dense on that call with a logged warning.

### Step 3 — Parallel (delta, R) scan in `sim_calibration.py`

Wrap the scan loop in `multiprocessing.Pool(cpu_count)`. Each worker runs one `(delta, R)` configuration independently — no shared state. Results flow back via `pool.imap_unordered` and are assembled in the parent.

**Correctness contract:** the final `alpha_star`, `mse`, `r_squared`, and `scan` list must be bit-equivalent to a sequential run (sort results by `(delta, R)` before the LSQ fit so ordering is deterministic). Test added: `test_parallel_scan_matches_sequential` using a 4-point (reduced) scan so the unit test runs in seconds.

**Blast radius:** `experiments/sim_calibration.py` only. Windows process-spawn needs `if __name__ == "__main__":` already present.

**Optional extension:** also parallelize the 9-point delta scan in `analytical_reduction.py`; easy win if Step 3 lands cleanly.

### Step 4 — Int8 quantized R_dst prototype (`laplacian_quantized.py` sidecar)

Build a SIDECAR module `laplacian_quantized.py` — do NOT modify the authoritative `laplacian.py`. This is a prototype for eventual NPU deployment.

Observation: the 8x8 `R_dst = λ · (Rot_3 ⊕ P_4 ⊕ I_1)` block decomposes as:

- **3x3 Rot_3:** entries bounded in [-1, 1]. int8 quantization with scale=127 gives ~0.4% relative error per entry.
- **4x4 P_4:** entries in {0, 1}. int8 exact.
- **1x1 I_1:** entry in {0, 1}. int8 exact.
- **λ scalar:** one of 6 known values — pre-quantized as an int16 (or fp16) lookup.

Prototype: `typed_sheaf_laplacian_quantized(positions, n, edges, ...)` that builds `delta` as int8 tensors, accumulates `L = delta.T @ delta` in int32 (no overflow up to n=10^6 at stalk_dim=8 for bounded restrictions), and dequantizes to fp32 for the final eigenvalue step.

**Accuracy contract:** on the REWORK smoke fixtures (analytical and a 30-point sim_cal subset), the quantized pipeline's `lambda_min` and `f_topo` values must agree with the fp64 reference to `rel < 1e-3`. `beta_0` and `beta_1_persistent` are integer counts and must agree exactly (both come from graph/ripser, not the quantized Laplacian).

**Out of scope for step 4:** actual NPU execution. This step produces a proof-of-concept that demonstrates int8 viability on the CPU; targeting Hexagon via ONNX Runtime QNN is a separate follow-up project once the quantized reference exists as ground truth.

## Acceptance gate

After all 4 steps:

1. Full pipeline test (`test_pipeline.py` suite) continues to pass end-to-end.
2. The three permanent regression guards from the REWORK (sign convention, typed ≠ untyped spectrum, nullity drop) continue to pass — these are invariant and unchanged by this pass.
3. New performance smoke test: `test_perf_smoke.py` runs the full pipeline at N=1500 and asserts wall time < **N seconds** (target TBD during plan writing — expect ~30s from the current ~3min as a reasonable goal). This is the test that catches regressions if someone reintroduces dense code paths.
4. A brief `PERF_NOTES.md` in `problems/hubble_tension_web/` documenting the wall-time improvement per step and the measurement methodology.

## Ordering

1 -> 2 (sparse input unlocks cheap Arnoldi).
3 is orthogonal to 1 and 2 — can be developed in parallel but lands later so that parallel workers inherit the sparse speedup per-config.
4 is parallel to 1/2/3 (separate module) but is best landed after 1 so the discrete structure-awareness in `R_dst_for_edge` is shared between the fp64 and int8 paths.

Sequence: 1, 2, 3, 4 in commit order (each step its own commit + green tests).

## Non-goals

- No change to `functional.py`, `types.py`, `graph.py`, `synthetic.py`, `ltb_reference.py` — math is frozen.
- No change to ripser calls in `spectrum.persistent_beta1` — ripser's own internals are outside our control; if it becomes the bottleneck, that's a separate optimization project (alpha complex, witness complex, sparse ripser, etc.).
- No migration to GPU (torch/JAX). Pure CPU + NPU target.
- No change to `sim_calibration`'s scan size or `n_points` per config in this pass — those are calibration-quality decisions, not perf decisions.

## Risks & fallbacks

- **eigsh convergence:** Arnoldi can fail on highly-degenerate kernels. Fallback: dense eigvalsh at that call site with a warning. Guard-tested.
- **Multiprocessing Windows quirks:** child-process import of scipy can be slow (~2s spawn overhead). Amortized across 30 configs but noticeable at small scan sizes. Fallback: threshold — use pool only if scan size > 6.
- **Sparse matmul precision:** `csr.T @ csr` uses float accumulation that can accumulate differently than dense. The 1e-12 tolerance in the equivalence test gives ~3 orders of headroom vs typical floating-point error; if we see failures at 1e-10, widen to 1e-10 and document.
- **int8 quantization on Rot_3:** the ~0.4% per-entry error can amplify through large products. If the rel < 1e-3 accuracy contract fails at step 4, bump to int16 quantization (still a 2-4× speedup on Hexagon, lossless for practical purposes).
