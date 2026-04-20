# N-Body Ingestion: MDPL2 Halos → β₁ Test on Real Voids

**Date:** 2026-04-20
**Branch:** `feat/nbody-ingestion` (off master HEAD `349a8da`)
**Predecessor:** master — REWORK (typed sheaf Laplacian) + perf pass (sparse + Arnoldi + parallel + concurrent runner) + 105× end-to-end speedup.

## Goal

Replace `synthetic.generate_synthetic_void` with real N-body data and test the **central ATFT Hubble-tension thesis**: that real cosmic voids — unlike smooth LTB-family synthetics — have internal substructure (sub-voids, filaments, walls within walls) that produces persistent 1-cycles, i.e. `β₁_persistent > 0`. If so, the topological term `α · f_topo` becomes non-zero and the functional can meaningfully contribute to ΔH₀ beyond the kinematic `c₁·δ`.

On synthetic voids the full pipeline proved (merged in `f8f4cf2`): α* = 0, β₁ = 0, ΔH₀ = kinematic-only = +4.49 km/s/Mpc at KBC params. The missing ingredient was real substructure. This spec plumbs it in.

## The three locked constraints

1. **Data source: MDPL2 halos** (DM-only, 1 Gpc³ box, Rockstar halo catalog, Planck cosmology, z=0 snapshot). No-auth public download via CosmoSim. Pristine 3D geometric ground truth — no observational distortions.
2. **Environment classification: T-web (tidal tensor eigenvalues).** Count of positive eigenvalues of the tidal tensor `T_ij = ∂_i ∂_j φ` (where φ is the gravitational potential) maps 1:1 to our `Environment` enum:
   - 3 positive → `Environment.VOID`
   - 2 positive → `Environment.WALL`
   - 1 positive → `Environment.FILAMENT`
   - 0 positive → `Environment.NODE`
3. **Physics test: real-void β₁ observation.** Extract actual KBC-like voids (R ≈ 300 Mpc, δ ≈ -0.2) from the MDPL2 volume, compute persistent homology on the halos inside, and check whether `β₁_persistent > 0`.

## New module layout

```
problems/hubble_tension_web/nbody/                 # NEW package
├── __init__.py
├── mdpl2_fetch.py          # one-shot downloader + cache; user runs once
├── tidal_tensor.py         # density-field → potential → tidal-tensor → T-web classifier
├── void_finder.py          # locate KBC-like voids in the classified volume
├── cosmic_web_from_halos.py  # assemble a LocalCosmicWeb from halos + T-web labels
└── README.md               # data provenance + cached file policy
```

New experiment:

```
problems/hubble_tension_web/experiments/
└── nbody_kbc.py            # runs the full pipeline on MDPL2 voids
```

Tests:

```
tests/hubble_tension_web/nbody/
├── test_tidal_tensor.py    # synthetic-field unit tests (e.g. single spherical void → all interior cells classified VOID)
├── test_void_finder.py     # synthetic-field unit tests (known void at known location)
├── test_cosmic_web_from_halos.py  # LocalCosmicWeb round-trip with known labels
└── test_nbody_kbc.py       # end-to-end smoke test on a small cached fixture
```

The **authoritative fp64 production path is untouched**: `functional.predict_from_cosmic_web` keeps its signature. Real-data feed just constructs a `LocalCosmicWeb` from MDPL2 halos and a `VoidParameters(δ, R_mpc)` from the detected void's effective density/radius, and hands both to `predict_from_cosmic_web`.

## Key design choices

### Data acquisition (mdpl2_fetch.py)

- **What to download:** Rockstar halo catalog for a single z=0 snapshot. The full catalog is ~tens of GB. For v1, we download **a single sub-box** (e.g., 500 Mpc cube) to stay under ~5 GB and under typical Python memory.
- **Source URL:** CosmoSim MDPL2 public endpoint (well-known public URL; resolve at implementation time — do not hardcode a guessed URL in the spec).
- **Cache:** Download to `~/.cache/atft/mdpl2/` by default; override via env var `ATFT_DATA_CACHE`. Halo catalog stored as a Parquet file after one-time ingestion from the raw format (saves ~5× on I/O after first run).
- **Mass cut:** Down-select to halos with `M_vir > 10^11.5 M_sun/h` (galaxy-hosting halos, keeps N tractable).
- **Network policy:** The module raises `NBodyDataNotAvailable` if the cache is cold AND network is unavailable. The full experiment suite (including `test_nbody_kbc.py`) can run offline once the cache is populated. CI runs will never touch the network — they use a tiny synthetic fixture that mimics the cache format.

### T-web classification (tidal_tensor.py)

- **Grid:** 256³ cells over the sub-box (~2 Mpc per cell at 500 Mpc box). Memory budget: `256³ × 8 bytes × ~5 arrays = ~700 MB`. Acceptable.
- **Density field:** CIC (cloud-in-cell) mass assignment from halo catalog. `numpy.add.at` or a small Cython/numba kernel if it's too slow.
- **Potential:** Solve Poisson via FFT (`np.fft.fftn`). Divide by `-k²`, zero the DC mode.
- **Tidal tensor:** Second derivatives via `-k_i k_j φ_hat` in Fourier space, inverse FFT, stack to a `(256, 256, 256, 3, 3)` array (~3 GB peak — might need to process slab-by-slab if memory tight).
- **Eigenvalues:** `np.linalg.eigvalsh` on each 3×3, threshold at 0.0 (or a small positive tidal-smoothing floor like `λ_th = 0.2` for robustness per Hahn+2007). Environment = 3 − `count(λ > λ_th)`. We ship `λ_th = 0.0` as v1; tunable via config.
- **Output:** `(256, 256, 256)` uint8 array of Environment codes. Persisted to disk alongside the halo cache.

### Void finder (void_finder.py)

- **Algorithm:** Simple but robust — find local minima of the density field (Gaussian-smoothed at ~10 Mpc). For each local minimum, grow a spherical region outward until the enclosed density contrast `δ_eff` crosses a threshold (-0.2 matches the KBC ansatz; also try -0.1 and -0.3).
- **Rank by depth × radius.** Return top N candidate voids.
- **No external tool (VIDE, ZOBOV) in v1** — those are heavy installs and the goal is a self-contained reference. If v1's voids aren't convincing, document and defer to a v2 that uses VIDE.

### Cosmic web assembly (cosmic_web_from_halos.py)

- Pick one candidate void from void_finder.
- Extract all halos within the void's effective radius.
- Look up each halo's Environment from the T-web grid (nearest-cell assignment).
- Build `LocalCosmicWeb(positions=halo_coords_relative_to_void_center, environments=env_list)`.
- Build `VoidParameters(delta=δ_eff, R_mpc=R_eff)`.
- Return `(LocalCosmicWeb, VoidParameters)`.

### Experiment (experiments/nbody_kbc.py)

```
1. Load MDPL2 sub-box from cache (or raise NBodyDataNotAvailable).
2. Compute T-web classification (cached to disk after first run per sub-box + λ_th).
3. Find top K candidate voids (K=5 for v1).
4. For each candidate void:
   a. Assemble LocalCosmicWeb + VoidParameters.
   b. Run predict_from_cosmic_web(alpha=alpha_star_from_smooth_calibration, ...).
   c. Record: N_halos, δ_eff, R_eff, β₀, β₁_persistent, λ_min, ΔH₀_total, kinematic, topological.
5. Write results/nbody_kbc.json + results/nbody_kbc.png (β₁ distribution across voids).
6. Key question the JSON answers: for how many voids does β₁_persistent > 0?
```

α* to use: `0.0` from smooth-void calibration is uninformative (`α·0 = 0`). For real voids with `β₁ > 0` we'd want to *re-calibrate*. **Defer recalibration to v2** — v1 just reports β₁ and leaves α at the smooth-void value; the headline number is the count of voids with nonzero β₁.

## Integration contract

- `functional.py`, `laplacian.py`, `spectrum.py`, `types.py`, `graph.py`, `ltb_reference.py` — **zero changes** in this pass. All load-bearing math and perf work is preserved.
- `synthetic.py` stays as the canonical synthetic generator; `nbody/cosmic_web_from_halos.py` is the real-data alternative.
- The new `experiments/nbody_kbc.py` is added to `experiments/run_all.py` OPTIONALLY — only if the MDPL2 cache is present. If absent, `run_all.py` skips it with an informational message and still runs the three synthetic experiments.

## Acceptance gate

- All existing tests (52+ including the concurrent runner smoke test) continue to pass.
- New test suite (~10-15 tests across `tests/hubble_tension_web/nbody/`) passes, including:
  - `test_tidal_tensor`: single spherical void produces correct Environment codes at sampled cells.
  - `test_void_finder`: known synthetic density-dip is located to within 1 grid cell.
  - `test_nbody_kbc`: end-to-end runs on a tiny fixture (<50 halos) in <5 s; writes JSON with finite β₁ value (may be 0 — that is valid output).
- `nbody_kbc.py` runs against the MDPL2 cache and produces `results/nbody_kbc.json` with a `β₁_distribution` field showing at least one void's β₁ count per candidate.

## Out of scope (deferred)

- VIDE / ZOBOV-based watershed void finding.
- IllustrisTNG galaxy catalogs (we want halos first, galaxies second).
- SDSS/2MRS observational ingestion.
- Recalibrating α* against the real-void LTB residual (needs β₁ > 0 first).
- Redshift-space vs real-space distinction (MDPL2 gives real-space positions; we use them directly).
- Full-box 1 Gpc³ analysis (500 Mpc sub-box for v1; scale up later).
- NPU int8/int16 deployment (the Task-4 sidecar is shelved until the math-at-scale works first).

## Risks & fallbacks

- **MDPL2 access:** CosmoSim may throttle or change URLs. Fallback: use the MDPL2 halo catalog mirror at Rockstar's Bolshoi server or a SciServer account. Document the fallback when implementation hits the issue.
- **Memory on 256³ T-web:** if we blow memory, drop to 128³ and document the coarser resolution. v2 can reintroduce slab-by-slab processing.
- **Nothing-is-a-void:** the v1 sphere-growing finder may produce spurious candidates in high-density regions. Mitigation: filter by `δ_eff < -0.1` and `R_eff > 50 Mpc`. If all candidates fail the filter, fall back to manually-specified void centers from a published MDPL2 void catalog (document the source).
- **β₁ = 0 on all real voids:** would be a genuine null result. The spec is honest about this possibility — we report it and decide whether to widen the persistence threshold `τ_persist` or try deeper voids.

## What success looks like (headline for REPORT.md)

> Across K candidate MDPL2 voids with R ≈ 300 Mpc and δ ≈ -0.2, β₁_persistent was non-zero in `<N/K>` cases (median `<β₁_median>`, max `<β₁_max>`). The topology of real voids is **not** LTB-smooth; sub-structure carries measurable 1-cycles that the typed sheaf Laplacian picks up. This motivates a v2 recalibration of α* against real voids.

Or, if null:

> Across K candidate MDPL2 voids, β₁_persistent = 0 uniformly — consistent with the smooth-void finding. Either (a) the persistence threshold is too coarse for the sub-void scale present in MDPL2 at this resolution, or (b) sub-structure density contrasts are too shallow to survive the τ·ℓ̄ lifetime cut. Follow-ups: tune τ_persist, raise grid resolution, or migrate to IllustrisTNG where baryonic physics seeds richer sub-structure.
