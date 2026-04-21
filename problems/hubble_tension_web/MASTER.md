# `hubble_tension_web` — Master Knowledge Doc

**Last updated:** 2026-04-21 · **Master branch HEAD:** `23620db` (merge of `feat/nbody-ingestion`)
**Authoritative spec tree:** `docs/superpowers/{specs,plans}/` · **This doc is a navigator, not a replacement for the specs.**
**Upstream framework:** `docs/THESIS_014_UNIFIED.md` — the ATFT doctrine this project specializes.

Audiences in one pass:
- **New contributor** — §2 quick-start, §4 architecture, §10 extension points.
- **Reviewing researcher** — §3 thesis, §6 empirical state, §9 surprise findings.
- **Future maintainer** — §11 branch topology, §12 specs, §13 deferred v2, §9 traps.

---

## Table of Contents

- [§1. TL;DR — 60-second pitch](#1-tldr)
- [§2. Quick-start](#2-quick-start)
- [§3. The thesis](#3-the-thesis)
- [§4. Architecture](#4-architecture)
- [§5. Pipeline step by step](#5-pipeline-step-by-step)
- [§6. Current empirical state](#6-current-empirical-state)
- [§7. Operations](#7-operations)
- [§8. Test suite topology](#8-test-suite-topology)
- [§9. Surprise findings worth knowing](#9-surprise-findings-worth-knowing)
- [§10. Extension points](#10-extension-points)
- [§11. Branch merge topology on master](#11-branch-merge-topology-on-master)
- [§12. Spec plan inventory](#12-spec-plan-inventory)
- [§13. Deferred v2 items](#13-deferred-v2-items)
- [§14. Glossary](#14-glossary)
- [§15. Ownership and contact](#15-ownership-and-contact)
- [Appendix A recommended reading order](#appendix-a-recommended-reading-order)

---

## 1. TL;DR

The project tests one claim: **the Hubble tension is a scale-mismatch artifact of discrete cosmic-web topology**. Given a local point cloud of halos with environment labels (void / wall / filament / node), a *typed sheaf Laplacian* extracts topological invariants (β₀, β₁, λ_min) whose combination `α · f_topo + c₁ · δ` should match the observed local-vs-global H₀ gap.

Status as of `23620db`:
- Math is bit-equivalent to reference (1e-12 sparse/dense, 1e-13 Arnoldi/dense).
- **α* = 0 on every synthetic scan** because β₁_persistent = 0 on smooth LTB-family voids — the topology contributes nothing when the input is a featureless hole. This is the **expected** null result; the physics test awaits real N-body ingest.
- N-body pipeline (MDPL2 parquet → T-web → void finder → pipeline) is wired and smoke-tested on a committed 2724-halo fixture. Real MDPL2 credentials stubbed in v1.
- End-to-end wall time: **6123 s baseline → 35 s concurrent runner** (≈175× on 8-core Snapdragon X Plus).

---

## 2. Quick-start

### Prerequisites

- **Python 3.14** (per PERF_NOTES.md baselines; 3.11+ should work but untested here).
- No `requirements.txt` exists — dependencies are installed manually:

```bash
pip install numpy scipy pandas pyarrow matplotlib ripser
```

`ripser` pulls a C++ extension on first install (~30 s on Windows MSVC; ships prebuilt on ARM Linux via pip).

### Commands

From repo root:

```bash
# 1. Fast unit suite (~10s, no subprocess, no pipeline)
python -m pytest tests/hubble_tension_web/ \
  --ignore=tests/hubble_tension_web/test_pipeline.py \
  --ignore=tests/hubble_tension_web/test_run_all.py \
  -q --timeout=60

# 2. End-to-end run: 3 concurrent synthetic experiments + aggregate + conditional nbody
python -m problems.hubble_tension_web.experiments.run_all

# 3. Full pipeline smoke (sequential, ~90s)
python -m pytest tests/hubble_tension_web/test_pipeline.py -v
```

Outputs land in `problems/hubble_tension_web/results/`:
- `analytical_reduction.{json,png}`, `sim_calibration.{json,png}`, `kbc_crosscheck.json`, `REPORT.md`
- `nbody_kbc.{json,png}` (conditional — only if the MDPL2 cache is present)

The public entry point for any custom experiment is `problems/hubble_tension_web/functional.py::predict_from_cosmic_web`. Build a `LocalCosmicWeb` + `VoidParameters`, pass them in, get a `HubbleShift` back.

---

## 3. The thesis

**Ansatz.** For a local underdensity (void) of contrast δ and effective radius R, the locally-inferred Hubble constant departs from the global value by

```
ΔH₀(β₀, β₁, δ, R) = c₁ · δ  +  α · f_topo(β₀, β₁, λ_min, R)

c₁      = -H₀_GLOBAL / 3                      [km/s/Mpc; negative]
f_topo  = (β₁ / max(β₀, 1)) · (1 / max(λ_min, 1e-6)) · (1 / R)
α       [km/s]                                # one free coefficient; units pinned
β₀, β₁  = persistent topological invariants of the typed sheaf Laplacian spectrum
λ_min   = spectral gap
```

- The kinematic coefficient `c₁ = −H₀/3` is **asserted from LTB linear theory** (Garcia-Bellido & Haugbølle 2008), not derived from `spec(L_F)`. `ltb_reference.py` is the independent calibration target that `sim_calibration.py` fits α against.
- `f_topo` depends only on eigenvalue *ratios*, so absolute density units (and the 4πG prefactor) drop out — we solve `-k² φ̂ = ρ̂` directly.
- The sign convention is pinned: a void (δ<0) implies `c₁·δ > 0` implies ΔH₀ > 0 implies local H₀ exceeds global, matching the observed tension direction.

**Why a typed sheaf Laplacian (not the graph Laplacian).** Motivated by **Čech–de Rham**: on a cosheaf with **asymmetric restriction maps**, discrete Čech cohomology and continuous de Rham cohomology compute the same invariant. The v1 implementation used symmetric orthogonal restriction maps (R_src = −R_dst = Q orthogonal), which provably collapses to `L_F ≡ L_graph ⊗ I_stalk_dim` — a mathematical no-op. The v2 rework uses

```
R_src^t = I_8
R_dst^t = λ^t · (Rot_3(ĝ_src → ĝ_dst) ⊕ P^t_4 ⊕ I_1)
```

(a block-diagonal 8×8 matrix: a 3×3 rotation aligning density gradients, a 4×4 permutation on the env one-hot, a 1×1 identity on the pad coord) where the restriction map per oriented edge type carries a genuinely asymmetric, environment-aware structure. `EDGE_TYPE_LAMBDA` is a 16-entry table of λ values (6 distinct scalars, ordinal physical prior — not calibrated).

See `docs/superpowers/specs/2026-04-19-hubble-tension-web-REWORK-design.md` for the full rework derivation including the no-op proof. For the upstream ATFT framework this project specializes, see `docs/THESIS_014_UNIFIED.md`.

---

## 4. Architecture

### Module dependency graph

```
types.py  (LocalCosmicWeb, VoidParameters, HubbleShift, SpectralSummary, Environment)
  ^ ^ ^ ^
  | | | +-- synthetic.py, graph.py, laplacian.py, spectrum.py, nbody/*, functional.py
  | |
  | +------ ltb_reference.py  (imports NOTHING from the package — circularity guard)
  |
  +-------- laplacian_quantized.py  (SIDECAR — verified no non-test imports)

functional.py → types, graph, laplacian, spectrum       # the public wrapper

experiments/
├── analytical_reduction.py   → functional, synthetic, types
├── sim_calibration.py        → functional, ltb_reference, synthetic, types
├── kbc_crosscheck.py         → functional, synthetic, types
├── nbody_kbc.py              → functional, nbody/* (graph, laplacian, spectrum via lazy imports)
├── aggregate.py              → (no package imports; reads the 3 synthetic JSONs)
└── run_all.py                → subprocess-launches the 4 experiments above

nbody/
├── __init__.py               (constants, NBodyDataNotAvailable)
├── mdpl2_fetch.py            → nbody; pandas/pyarrow Parquet reader
├── tidal_tensor.py           → types
├── void_finder.py            (pure density-grid algorithm; scipy.ndimage only)
└── cosmic_web_from_halos.py  → nbody.{mdpl2_fetch, tidal_tensor, void_finder}, types
```

### Type flow through `predict_from_cosmic_web`

```
LocalCosmicWeb(positions: (N,3) float64, environments: list[Environment])
VoidParameters(delta ≤ 0, R_mpc > 0)
       │
       ▼ build_typed_graph(web, k=8)                                [graph.py]
n: int,  edges: list[tuple[int,int,str]]           # str = "env_src-env_dst"
       │
       ▼ typed_sheaf_laplacian(positions, n, edges, ...)            [laplacian.py]
L: scipy.sparse.csr_matrix, shape (n·8, n·8), float64
       │
       ▼ summarize_spectrum(L, n, edges, positions, k_spec=16)      [spectrum.py]
SpectralSummary(spectrum: (16,), beta0: int, beta1: int, lambda_min: float)
       │
       ▼ kappa_operator(summary, delta, R, alpha)                   [functional.py]
HubbleShift(delta_H0, kinematic_term, topological_term, delta)
       └─ __post_init__ enforces (sum consistency AND void-sign regression guard)
```

`stalk_dim = 8 = 3 (unit density-gradient) + 4 (env one-hot) + 1 (pad)`. It is pinned; any other value raises.

---

## 5. Pipeline step by step

Following one `predict_from_cosmic_web(web, params, alpha=…, k=8, k_spec=16)` call:

1. **k-NN graph** (`graph.build_typed_graph`): `scipy.spatial.KDTree` gives k neighbors per node. Each undirected edge `{u,v}` is stored once as `(s=min(u,v), d=max(u,v), "env_s-env_d")`. Edge types are **order-sensitive** — `"void-wall"` does not equal `"wall-void"`. `EDGE_TYPES` has 16 entries for the 4-env enum.
2. **Stalk initialization** (`laplacian.build_stalk_init`): for each node, compute a unit density-gradient ĝ via KDTree-weighted finite differences of local density `ρ ≈ 1/mean_nn_dist`. Coords 0–2 = ĝ; coords 3–6 = one-hot over `{void, wall, filament, node}`; coord 7 = 0 (pad). Degenerate gradients (|∇ρ| < 1e-9) fall back to `ê_z` with a `flag=True`. `k_density = min(k_density, N-1)` clamp prevents KDTree's `k+1` sentinel from going out of bounds at small N.
3. **Sparse coboundary assembly** (`laplacian.typed_sheaf_laplacian`): for each oriented edge, write `-I_8` on the `col_s` block and `R_dst = λ·(Rot_3 ⊕ P_4 ⊕ I_1)` on the `col_d` block of a `(m·8, n·8)` sparse LIL matrix, then `tocsr()`, then `L = δᵀ δ`, symmetrized `L = 0.5(L + Lᵀ)` to kill matmul float asymmetry. Output is `scipy.sparse.csr_matrix`.
4. **Spectrum** (`spectrum.summarize_spectrum`): shift-invert Arnoldi on the sparse L:

   ```python
   eigsh(L, k=k_spec+4, sigma=-1e-6, which="LM",
         tol=1e-8, ncv=max(3*(k_spec+4), 40), v0=deterministic_normal)
   ```

   Fallback to dense `eigvalsh(L.toarray())` on `ArpackNoConvergence` (guard-tested, never tripped in production). `lambda_min` = smallest eigenvalue above `zero_tol = 1e-6`.
5. **β₀, β₁** (`spectrum._connected_components`, `spectrum.persistent_beta1`): β₀ via union-find on the backbone graph. β₁ via `ripser(positions, maxdim=1, thresh=τ_max·ℓ̄)` with `τ_max = 6.0`, lifetime threshold `τ_persist·ℓ̄` where `τ_persist = 1.5` and `ℓ̄` is the mean k-NN edge length. **Critical detail:** classes with `death = inf` (survive past the VR cap) have their lifetime clamped to `thresh − birth` (a strict lower bound), NOT zeroed out.
6. **κ operator** (`functional.kappa_operator`): `kinematic = C1·δ`, `topological = α · f_topo`, return `HubbleShift(kinematic + topological, kinematic, topological, delta)`. `HubbleShift.__post_init__` enforces sum consistency and the §3 sign convention (rejects the v1 bug pattern `kinematic<0 with δ<0 and |topological|≈0`). See §9.2 for the separate T-web sign flip.

---

## 6. Current empirical state

| Quantity | Value | Notes |
|---|---|---|
| α* (synthetic 30-point scan) | **0.0 km/s** | f_topo is identically 0 because β₁_persistent is 0 on smooth LTB voids; LSQ has no signal to fit |
| KBC cross-check at δ=-0.2, R=300 Mpc | ΔH₀ = **+4.49 km/s/Mpc** | kinematic +4.49, topological 0.0, **correct sign**, ABOVE `[+1, +3]` literature band |
| LTB reference at KBC params | +5.28 km/s/Mpc | Gaussian-profile heuristic; matches the kinematic neighborhood |
| Analytical tautology residual | **0.0 exactly** | `kinematic_term - C1·δ` at every scan point |
| N-body fixture β₁ distribution | `count_nonzero=0, count_total=2, max=0` | Expected null on a toy Poisson-background plus one planted smooth void |

**Reading of the null result.** The full chain is mechanically correct (sign, assembly, eigensolver, persistence filter), but synthetic LTB voids have no sub-structure. `β₁_persistent` stays at the VR noise floor, so `f_topo` is identically 0, α* is undetermined, and the honest result is that **topology contributes nothing** for this input class. The physics test of the ATFT thesis — "do real voids have β₁ > 0?" — requires real MDPL2 halos. The pipeline is now wired to accept them.

**Reproducing the KBC headline number:**

```bash
python -m problems.hubble_tension_web.experiments.kbc_crosscheck
python -c "import json; d=json.load(open('problems/hubble_tension_web/results/kbc_crosscheck.json')); print({k: d[k] for k in ('delta_H0','kinematic_term','topological_term','verdict')})"
```

Expected: `{'delta_H0': 4.4933…, 'kinematic_term': 4.4933…, 'topological_term': 0.0, 'verdict': 'ABOVE band …'}`.

---

## 7. Operations

### Canonical commands

```bash
# Full run (concurrent): ~30-35s wall time
python -m problems.hubble_tension_web.experiments.run_all

# Individual experiments (rarely needed — run_all launches them)
python -m problems.hubble_tension_web.experiments.analytical_reduction
python -m problems.hubble_tension_web.experiments.sim_calibration
python -m problems.hubble_tension_web.experiments.kbc_crosscheck
python -m problems.hubble_tension_web.experiments.nbody_kbc    # needs env or cache
python -m problems.hubble_tension_web.experiments.aggregate     # regen REPORT.md
```

### Env var catalog

Commonly twisted:

| Var | Default | Effect |
|---|---|---|
| `HUBBLE_POOL_WORKERS` | `os.cpu_count()` | Caps `sim_calibration`'s inner `mp.Pool`. `run_all._env_for` sets it to `cpu_count-2` to leave cores for the other concurrent experiments. |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | `"1"` (set by `run_all._env_for`) | BLAS thread pin; prevents 3 concurrent subprocesses each spawning `cpu_count()` BLAS threads. |
| `ATFT_DATA_CACHE` | `~/.cache/atft` | Root cache dir. Probed as `$ATFT_DATA_CACHE/mdpl2/*.parquet`. |
| `ATFT_NBODY_CACHE_FILE` | (unset) | Explicit Parquet path; wins over the cache probe. |

N-body fine-tuning (read in `experiments/nbody_kbc.py`): `ATFT_NBODY_GRID` (default 128; **256³ exceeds 3 GB — deferred**), `ATFT_NBODY_LAMBDA_TH` (default 0.0), `ATFT_NBODY_K_VOIDS` (default 5), `ATFT_NBODY_MASS_CUT` (default 0.0, M_sun), `ATFT_NBODY_OUTPUT_JSON` (default `results/nbody_kbc.json`).

### Wall-time record (`PERF_NOTES.md`, Snapdragon X Plus, Python 3.14)

| Stage | analytical_reduction | sim_calibration | kbc_crosscheck | pipeline total |
|---|---|---|---|---|
| T=0 baseline (dense, sequential) | 913 s | 5045 s | 165 s | ~6123 s (~102 min) |
| Step 1 sparse | 543 s | deferred | 64 s | — |
| Step 2 Arnoldi | 29 s | deferred | 4 s | — |
| Step 3 parallel + concurrent runner | 29 s | 20 s | 7 s | **~30 s** |

sim_calibration compound speedup: **247×** (5045 → 20 s).

### Artifacts written by the pipeline

| Path | Writer |
|---|---|
| `results/analytical_reduction.{json,png}` | `experiments/analytical_reduction.py` |
| `results/sim_calibration.{json,png}` | `experiments/sim_calibration.py` |
| `results/kbc_crosscheck.json` | `experiments/kbc_crosscheck.py` |
| `results/nbody_kbc.{json,png}` | `experiments/nbody_kbc.py` (conditional) |
| `results/REPORT.md` | `experiments/aggregate.py` |
| `results/v1_superseded/*` | Frozen archive of pre-REWORK artifacts |

---

## 8. Test suite topology

**76 collected items** in the fast suite (`73 passed + 3 xfailed`) across 18 test files. **6 items** in the slow suite (`test_pipeline.py` + `test_run_all.py`). 1-to-1 alignment between production module and test file.

| File | Count | Lens |
|---|---|---|
| `test_types.py` | 6 | dataclass invariants plus sign-guard |
| `test_functional.py` | 6 | c₁ = -H₀/3, α units, sign convention |
| `test_laplacian.py` | 5 | PSD/symmetric, typed vs untyped, nullity drop |
| `test_ltb_reference.py` | 5 | leading order, functional-independent (grep check) |
| `test_laplacian_quantized.py` | 5 | sidecar int8/int16 (3 xfailed for int8 floor) |
| `test_graph.py` | 4 | ordered edge types, `EDGE_TYPES` = 16 |
| `test_spectrum.py` | 4 | β₁ small on Poisson, nonzero on ring |
| `test_spectrum_eigsh.py` | 3 | eigsh vs dense, ArpackNoConvergence fallback |
| `test_synthetic.py` | 3 | void depth, smooth limit |
| `test_pipeline.py` | 3 (slow) | end-to-end, KBC sign, tautology residual |
| `test_run_all.py` | 3 (slow) | wall time, signed contract, skip-nbody-when-absent |
| `test_laplacian_sparse.py` | 2 | sparse L equals dense to 1e-12 |
| `test_sim_calibration_parallel.py` | 1 | parallel scan bit-equivalent to sequential |
| `nbody/test_mdpl2_fetch.py` | 6 | schema contract, stubbed network |
| `nbody/test_tidal_tensor.py` | 6 | CIC, T-web classification |
| `nbody/test_cosmic_web_from_halos.py` | 5 | assembly, void-sign respected |
| `nbody/test_void_finder.py` | 4 | planted void recovered within 2 cells |
| `nbody/test_nbody_kbc.py` | 1 | subprocess smoke on fixture |

### Non-obvious gates worth calling out

Most acceptance checks are transparent from their test names. These four encode invariants that would escape a name-only scan:

- `test_ltb_reference_is_functional_independent` — **grep-based** static check that `ltb_reference.py` imports nothing from `functional/laplacian/spectrum`, preserving the non-circularity of the calibration target.
- `test_int8_quantized_lambda_min_rel_under_1e_minus_3` — **xfail(strict=True)** × 3 seeds; documents the empirical ~1.3% int8 uniform-scale floor (see §9.3). Companion `test_int16_quantized_lambda_min_meets_spec_bound` actually passes at rel<1e-6.
- `test_run_all_preserves_signed_contract` — ensures the concurrent-runner KBC output keeps `kinematic > 0` and doesn't degrade to a SIGN ERROR verdict under the parallel subprocess path.
- `test_typed_vs_untyped_spectrum_differs` + `test_nullity_drops_under_typing` — the pair that guards against a silent regression back to the v1 no-op Laplacian (see §3).

---

## 9. Surprise findings worth knowing

These are non-obvious gotchas that are NOT in the specs — captured only by trial and error during implementation. Each has a 2-sentence derivation so a future reader does not have to re-live the trap.

### 9.1 The ARPACK v0 trap (`fix(hubble): deterministic ARPACK starting vector`, `7a63468`)

**Symptom.** `test_eigsh_bottom_k_matches_dense[seed=7]` flaked under load; ARPACK returned a shifted bottom-k.

**Root cause.** Our typed sheaf Laplacian has **triple-degenerate eigenvalue clusters** from the `Rot_3 ⊕ P_4 ⊕ I_1` block structure. ARPACK's default Lanczos `ncv = 2k+1 = 41` Krylov subspace is too narrow to resolve them reliably; its default random `v0` also has a random projection onto the degenerate subspace, so different runs recover different representatives.

**Why the naive fix is wrong.** `v0 = np.ones(n)/sqrt(n)` is deterministic but **orthogonal to many of the degenerate stalk eigenmodes** (the ones-vector lies along the graph-Laplacian null space). All 3 seeds fail under that fix.

**Correct fix.** `v0 = rng(seed=0xA7CE1D).standard_normal(n); v0 /= norm`. Deterministic AND generic — a Gaussian vector has nonzero projection onto every eigenmode with probability 1.

**Lesson (worth a mantra).** **Deterministic does not equal structured.** Reproducibility and genericity are independent axes.

### 9.2 The T-web sign flip (`docs(nbody): clarify T_ij sign-flip convention`, `42afc31`)

**Strict physics.** Poisson `∇²φ = ρ` gives `φ̂ = -ρ̂/k²`. Tidal tensor `T_ij = ∂_i∂_j φ` gives `T̂_ij = -k_i k_j φ̂`. Combined: `T̂_ij = +k_i k_j ρ̂ / k²`. Trace recovers `ρ`. At a density peak, all eigvals positive gives Forero-Romero "3 positive means NODE".

**Our code.** `_tidal_tensor_fft` uses `T̂_ij = +k_i k_j φ̂` (drops the minus). This **flips the sign of T** everywhere, so 3 positive eigvals now occur at voids (density minima), which matches our `CODE_TO_ENV = (VOID, WALL, FILAMENT, NODE)` indexing.

**Cost.** `Tr(T)` does not equal ρ under our convention. **Benefit.** The code `3 - n_positive` directly indexes `CODE_TO_ENV` — no inversion table. We never use the trace; classification reads only eigenvalue signs.

**Lesson.** When two conventions disagree, pick the one that collapses more downstream code, and document the flip loudly.

### 9.3 The int8 uniform-scale floor (`perf(hubble): Task 4 — int8/int16 quantized R_dst sidecar`, `6a426a9`)

**Finding.** Uniform per-tensor int8 quantization of `R_dst = λ·(Rot_3 ⊕ P_4 ⊕ I_1)` with `scale = 127/max(λ) = 127/2.2 ≈ 57` gives **rel ≈ 1.3%** on λ_min (vs. spec contract `rel < 1e-3`). Int16 passes `rel < 1e-6` comfortably.

**Why int8 misses.** P_4 and I_1 blocks are exact (entries in {0,1}), but Rot_3's 9 floats each carry ~0.4% per-entry quantization error. Compounded across the matmul `L = δᵀδ`, the smallest eigenvalue picks up ~1–2% error — just above the contract.

**Implication for Hexagon NPU plans.** The 4× int8-over-int16 throughput win is **not available** under uniform per-tensor scale. Recovering int8 would need either **per-channel scales** or **mixed precision** (int8 Rot, int16 P/λ). Deferred to post-v1 NPU work.

**Secondary finding (accumulator width).** int16 delta squared, summed over ~12 incident edges per node, reaches ~1.3e10 — past int32's 2.1e9 ceiling. The accumulator is int32 for bits=8, int64 for bits=16. Without the int64 promotion, int16 would silently overflow and report rel ≈ 94%.

### 9.4 The CIC anchor shift (`feat(nbody): T-web classifier`, `767d04e`)

Cell `(i,j,k)` has its **center** at position `((i+0.5)·cell, ...)`, not at `(i·cell, ...)`. `cic_deposit` applies `scaled = positions/cell - 0.5` so a particle at a cell center gets `frac = 0` and deposits all its mass into that one cell. `lookup_env_at_positions` uses `floor(position/cell)`, consistent with the same convention. Regression caught by `test_cic_deposit_single_point_at_cell_center`; the wrong convention would silently bias every T-web classification by half a cell.

### 9.5 The VR-filtration `inf`-death clamp (`feat(hubble): Task 2 — persistent β1`, `d0b16d1`)

**Trap.** Ripser's persistence diagram returns `(birth, inf)` for 1-cycles that survive past the filtration threshold. These are **the most persistent cycles** — real topological features — but naive filtering `lifetime = death - birth` zeros them out.

**Fix.** Replace `inf` with `thresh = τ_max·ℓ̄` so `lifetime = thresh - birth` is a strict lower bound on the true lifetime, preserving the most-persistent classes in the `lifetime > τ_persist·ℓ̄` gate.

### 9.6 The analytical-reduction honesty patch (`docs(hubble): review follow-up`, `2d5c9d2`)

**Original claim.** `analytical_reduction.py` primary assertion: "topological_term shrinks relative to kinematic_term as δ → 0 (β₁ noise-floor only)."

**Opus pre-merge review finding.** Across the scan, `β₁_persistent = 0` at every δ, so `topological_term = α · 0 = 0` by **arithmetic**, not by any demonstrated homogenization limit. The assertion read as a non-trivial physics check but was vacuously satisfied.

**Patch.** Renamed `primary_assertion` to `primary_observation`; honestly states "smooth-void regime; β₁ at noise floor; kinematic dominates; sub-void fixture needed for a real limit test." Tautology residual test (secondary) is still a meaningful sign-regression guard.

---

## 10. Extension points

| Task | Where | Notes |
|---|---|---|
| Add a new environment class | `types.Environment` plus `laplacian._ENV_INDEX` plus `laplacian.EDGE_TYPE_LAMBDA` | STALK_DIM = 8 assumes exactly 4 envs (3 grad + 4 one-hot + 1 pad). Adding a 5th env needs STALK_DIM bump plus revalidation of every Rot/perm. |
| Swap the void finder | New sibling in `nbody/` returning `VoidCandidate` | `cosmic_web_from_halos.assemble` is agnostic to provenance. |
| Swap the calibration target | New sibling in `problems/hubble_tension_web/` exposing `delta_H0_<name>(*, delta, R_mpc) -> float` | Update the import in `sim_calibration.py`. Must **not** import from `functional/laplacian/spectrum` (enforced by grep test). |
| Swap the Laplacian | Any module exposing `(positions, n, edges, stalk_dim, rng_seed, environments) -> csr_matrix` | `laplacian_quantized.py` is the canonical sidecar template. |
| Reweight restriction maps | `laplacian.EDGE_TYPE_LAMBDA` dict | Keyed by ordered `Tuple[str,str]` — one table edit. |
| Add a new data source | Anything returning `LocalCosmicWeb` | Synthetic and nbody-cosmic-web are the two templates. |

---

## 11. Branch merge topology on master

```
23620db  Merge feat/nbody-ingestion — MDPL2 + T-web + real-void β₁ test (11 commits)
349a8da  perf(hubble): concurrent runner — 175× over baseline
f8f4cf2  Merge feat/hubble-perf-rework — sparse + Arnoldi + parallel + int8 sidecar
7efce59  Merge feat/hubble-tension-web — typed sheaf-Laplacian REWORK (38 commits)
63202be  Add copyright + ownership to README
```

Remote branches preserved on origin (`RogueGringo/atft-problems_v0`):
- `master` (trunk)
- `feat/hubble-tension-web` (REWORK history)
- `feat/hubble-perf-rework` (perf-pass history)
- `feat/nbody-ingestion` (nbody history)
- `stage/dual-gpu-rig` (clean fork-off-master staging point for the GTX+RTX home rig)

---

## 12. Spec plan inventory

All under `docs/superpowers/`:

| Document | Purpose |
|---|---|
| `specs/2026-04-19-hubble-tension-web-design.md` | Original design — functional ansatz, 3-leg validation protocol. |
| `plans/2026-04-19-hubble-tension-web.md` | v1 implementation plan. **Superseded** by REWORK below. |
| `specs/2026-04-19-hubble-tension-web-REWORK-design.md` | Math rework — no-op proof for v1 Laplacian, asymmetric R_dst, persistent β₁, signed c₁. |
| `plans/2026-04-19-hubble-tension-web-REWORK.md` | 12-task rework plan. |
| `specs/2026-04-20-hubble-perf-rework-design.md` | Perf plus hardware co-design — 4 steps (sparse, Arnoldi, parallel, int8 sidecar). |
| `plans/2026-04-20-hubble-perf-rework.md` | Perf rework plan. |
| `specs/2026-04-20-nbody-ingestion-design.md` | MDPL2 plus T-web plus real-void β₁ test. |
| `plans/2026-04-20-nbody-ingestion.md` | N-body ingestion plan (7 tasks including scaffold). |

---

## 13. Deferred v2 items

Explicitly-punted work, all documented with rationale in their respective specs:

- **α recalibration against real voids.** v1 uses α from the smooth-void fit (which lands at 0). Meaningful recalibration requires β₁ > 0 on real data; deferred until N-body runs confirm that.
- **VIDE / ZOBOV watershed void finders.** v1 uses a simple Gaussian-smoothed local-min plus sphere-growth finder. Fine for ~10⁴-point catalogs; may need upgrading for 10⁶+.
- **IllustrisTNG baryonic catalogs.** v1 targets MDPL2 (DM-only halos). IllustrisTNG adds galaxy catalogs and gas physics; different ingest path, richer sub-structure.
- **SDSS / 2MRS observational ingestion, including RSD.** Real sky data, with redshift-space distortions plus survey masks. Would need a dedicated ingest plus observational-systematics pass. v1 uses real-space positions from MDPL2 directly.
- **256³ T-web grid.** Peak memory ~3 GB exceeds 16 GB laptop budget once other processes are running. Default is 128³; `ATFT_NBODY_GRID=256` override available for rigs with headroom.
- **Full 1 Gpc³ MDPL2 analysis.** v1 targets a 500 Mpc sub-box for memory tractability. Full-box analysis needs slab-by-slab streaming.
- **NPU int8 deployment.** The Task-4 sidecar proved uniform int8 does not meet the accuracy contract (~1.3% vs target 0.1%). Per-channel scales or mixed precision needed; Hexagon QNN port is the follow-on project.

---

## 14. Glossary

- **ATFT** — Adaptive Topological Field Theory; the project's working framework treating physical observables as scale-slices of discrete topological evolution. See `docs/THESIS_014_UNIFIED.md` for the umbrella.
- **Čech–de Rham isomorphism** — the statement that on a well-constructed (co)sheaf, discrete Čech cohomology and continuous de Rham cohomology compute the same invariants. Motivates the asymmetric R_dst design.
- **CIC** — Cloud-In-Cell; the trilinear mass-assignment kernel used by `tidal_tensor.cic_deposit`.
- **`f_topo`** — the topology scalar `(β₁ / max(β₀,1)) · (1 / max(λ_min, 1e-6)) · (1/R)`; identically 0 on smooth voids by construction, so ΔH₀ reduces to `c₁·δ` when β₁ = 0.
- **KBC void** — the ~300 Mpc local under-density identified by Keenan, Barger and Cowie (2013); our benchmark real-universe void.
- **LTB** — Lemaître-Tolman-Bondi; the exact spherically-symmetric GR solution for a local under-density, used as the linear-order reference.
- **Persistent β₁** — the number of 1-cycles in a Vietoris-Rips filtration that survive longer than a threshold `τ_persist·ℓ̄`. Our "topological signal" on smooth voids is 0; on ring-like sub-structure it is at least 1.
- **`Rot_3 ⊕ P_4 ⊕ I_1`** — the block-diagonal 8×8 structure of R_dst: a 3×3 rotation aligning density gradients, a 4×4 permutation on the env one-hot, a 1×1 identity on the pad coord.
- **Sheaf Laplacian** — `L_F = δᵀδ` where `δ` is the sheaf coboundary on a typed graph. Lives in an `(n·stalk_dim)` by `(n·stalk_dim)` real matrix.
- **T-web** — tidal-tensor-based environment classifier. Count of positive eigenvalues of `∂_i∂_j φ` maps to VOID/WALL/FILAMENT/NODE.
- **Stalk** — the per-node vector space of the sheaf. Ours is 8-dimensional (3 grad + 4 env one-hot + 1 pad).

---

## 15. Ownership and contact

This repository is **© 2025–2026 Blake Jones (b.jones@jtech.ai), all rights reserved.** See repo-root `README.md` for full terms. This is private intellectual property; do not redistribute.

- **Maintainer:** Blake Jones (`git log --format='%aN <%aE>'` for the authoritative list).
- **Questions / issues:** b.jones@jtech.ai.
- **External researchers:** contact the maintainer before redistributing or citing pre-publication results from this codebase.

---

## Appendix A recommended reading order

For a new contributor (≈1 hour):

1. `problems/hubble_tension_web/MASTER.md` — this doc.
2. `problems/hubble_tension_web/types.py` — every module imports from this.
3. `problems/hubble_tension_web/functional.py` — public entry; 20-line orchestration.
4. `problems/hubble_tension_web/graph.py` — the oriented-edge convention.
5. `problems/hubble_tension_web/laplacian.py` — highest concept density.
6. `problems/hubble_tension_web/spectrum.py` — eigsh tuning plus ripser persistence.
7. `problems/hubble_tension_web/experiments/sim_calibration.py` — richest experiment, deterministic LSQ.
8. `docs/superpowers/specs/2026-04-19-hubble-tension-web-REWORK-design.md` — the "why v2 looks the way it does."

Aside-reading as you hit the relevant code: `ltb_reference.py` (if you touch calibration), `experiments/run_all.py` (one-liner entry point, ~150 lines), `nbody/tidal_tensor.py` (if you touch the N-body path — the sign-flip note is essential).

For a reviewing researcher (≈2 hours): the list above plus `docs/THESIS_014_UNIFIED.md` (framework context), every spec plus plan under `docs/superpowers/`, sections 3 and 9 of this doc, then inspect `test_laplacian.py`, `test_spectrum_eigsh.py`, and `test_ltb_reference.py` to see the invariants enforced as tests.

For a future maintainer: start at section 9 Surprise findings and section 13 Deferred v2 items. The working code teaches the rest.
