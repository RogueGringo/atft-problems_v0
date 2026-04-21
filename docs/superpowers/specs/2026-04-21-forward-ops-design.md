# Forward Ops: Real MDPL2 β₁ Test + α Recalibration

**Date:** 2026-04-21
**Branch:** `feat/forward-ops` (off master HEAD `da81c26`, one commit past the nbody merge)
**Predecessors:** REWORK math merge (`f8f4cf2`), perf pass (`349a8da`), N-body ingestion merge (`23620db`), MASTER.md doc (`da81c26`).

## Goal

Close the single biggest open question in MASTER.md §6: **do real MDPL2 voids have `β₁_persistent > 0`**, and if so, what is α* when fit against the real-void LTB residual? Produces the first quantitative ATFT prediction with uncertainty that can be compared to the KBC literature band [+1, +3] km/s/Mpc.

Bundle A (real-data ingest) + bundle B (α recalibration on real voids), per the Phase-1 discovery lock-in.

## Thesis check this addresses

MASTER.md §3: the ATFT ansatz `ΔH₀ = c₁·δ + α·f_topo` currently predicts ΔH₀ = +4.49 km/s/Mpc at KBC params (kinematic-only, since `f_topo ≡ 0` on smooth LTB synthetics). MASTER.md §6 honestly reports "**topology contributes nothing** for this input class." This spec moves us past that honest null to an empirical test: once MDPL2 halos land in the pipeline, `β₁_persistent` becomes non-trivial (sub-voids, filaments, wall fragments inside each candidate void), `f_topo > 0`, and the 1D LSQ that currently collapses to α* = 0 becomes meaningful.

## Three locked constraints (per Phase-1 discovery + Phase-3 lock)

1. **Data path**: CosmoSim SQL sub-box pull (primary); API-token hlist download (fallback); manual `ATFT_NBODY_CACHE_FILE` (escape hatch).
2. **Opt-in network gate**: `ATFT_MDPL2_DOWNLOAD_ENABLED="0"` default. No network traffic unless explicitly enabled. Preserves CI + `test_run_all_skips_nbody_when_cache_absent`.
3. **Math scope**: pragmatic middle — bootstrap α* CI + F-test vs α=0, in one new `experiments/nbody_calibration.py`. **Defer** numerical-LTB integrator gate to v1.5 unless α* from bootstrap turns out suspicious.

## Three sharp findings from the swarm that shape the design

1. **No published "KBC-in-MDPL2" ground-truth void exists.** The Haslbauer-Banik-Kroupa 2020 paper used MXXL (4.1 Gpc box), not MDPL2 (1 h⁻¹ Gpc). First real run is structural: `find_voids` on a 500 Mpc sub-box, keep top-K with `δ_eff ∈ [-0.3, -0.1]` and `R_eff ∈ [40, 300]` Mpc, cross-check against Baldi+2020 Sparkling void-size-function shape. Success = β₁ distribution is *non-trivial*, not a coordinate match.

2. **Little-h unit trap.** CosmoSim columns are in `h⁻¹ Mpc` and `h⁻¹ M_sun` with MDPL2 Planck cosmology `h = 0.6777`. Our existing schema contract (`nbody/mdpl2_fetch.py:18`: `EXPECTED_COLUMNS = ("x","y","z","mvir")`) expects pure Mpc/M_sun. **Fix: convert at write-time in `mdpl2_download.py`** so the cached Parquet always matches the contract — `load_halo_catalog` stays ignorant of h. Forgetting this shrinks distances 32% and silently wrecks void-radius comparisons.

3. **Fit denominator degeneracy is a feature, not a bug.** `sim_calibration.py`'s honest `alpha_star = 0.0` short-circuit when `f @ f < 1e-24` is exactly the regression guard we want: if `nbody_calibration.py` also returns α* = 0, it means MDPL2 voids *also* have β₁ ≡ 0 at our current resolution — a real null result, not a pipeline bug. The F-test + bootstrap CI will distinguish "truly zero" from "positive but uncertain."

## New module layout

**Created:**
```
problems/hubble_tension_web/
├── nbody/
│   └── mdpl2_download.py         # NEW: CosmoSim SQL pull, auth, h-conversion, Parquet write
└── experiments/
    └── nbody_calibration.py      # NEW: 1D LSQ + bootstrap CI + F-test on nbody_kbc.json voids
```

**Modified:**
```
problems/hubble_tension_web/
├── nbody/
│   ├── mdpl2_fetch.py            # fetch_from_network: delegate to mdpl2_download
│   └── README.md                 # document download opt-in + env vars
├── experiments/
│   ├── nbody_kbc.py              # emit per-void `f_topo_at_alpha_1` + `ltb_anchor` so calibration can reuse without recompute
│   ├── run_all.py                # add 4th conditional leg (nbody_calibration) after nbody_kbc
│   └── aggregate.py              # new "Leg 4: Real-Void Calibration" section gated on nbody_calibration.json
└── MASTER.md                     # update §6 (real-data columns), §13 (demote recalibration from deferred), §12 (add this spec)
```

**Tests:**
```
tests/hubble_tension_web/
├── nbody/
│   └── test_mdpl2_download.py    # NEW: mock urllib/requests; assert Parquet schema + h conversion
├── test_nbody_calibration.py     # NEW: mini-fixture with 6-8 synthetic records + planted β₁>0; asserts bootstrap + F-test output shape
├── test_run_all.py               # +2 tests: triggers-calibration-when-voids-exist, skips-calibration-when-no-voids
└── nbody/test_mdpl2_fetch.py     # adjust `test_network_fetch_is_stubbed` match (stub raises only when `ATFT_MDPL2_DOWNLOAD_ENABLED != "1"`)
```

**Frozen** (do NOT touch): `types.py`, `functional.py`, `graph.py`, `laplacian.py`, `spectrum.py`, `synthetic.py`, `ltb_reference.py`, `laplacian_quantized.py`, `nbody/{tidal_tensor,void_finder,cosmic_web_from_halos,__init__}.py`, `experiments/{analytical_reduction,sim_calibration,kbc_crosscheck}.py`, all math-path tests.

## Env var surface (5 new, all opt-in with safe defaults)

| Var | Default | Effect |
|---|---|---|
| `ATFT_MDPL2_DOWNLOAD_ENABLED` | `"0"` | Must be `"1"` to actually hit network. Default preserves CI + the "skip nbody when absent" test. |
| `ATFT_MDPL2_URL` | (unset) | Override the CosmoSim URL / SQL endpoint. Useful for mirrors or testing against a local HTTP stub. |
| `ATFT_SCISERVER_TOKEN` | (unset) | Authorization token for CosmoSim / SciServer. No default — the download layer raises `NBodyDataNotAvailable` with a clear error if unset AND `_ENABLED = "1"`. |
| `ATFT_NBODY_CAL_MIN_VOIDS` | `3` | Minimum K of real voids required before attempting the α fit. Below this, `nbody_calibration.py` emits a JSON with `alpha_star = null, reason = "insufficient voids (K<3)"`. |
| `ATFT_NBODY_CAL_BOOTSTRAP_B` | `2000` | Bootstrap resample count. |

Existing `ATFT_NBODY_*` family and `ATFT_DATA_CACHE` unchanged.

## Math contract for `nbody_calibration.py`

Inputs per void `k ∈ {1..K}`: measured `(δ_k, R_k, β₀_k, β₁_k, λ_min_k)` extracted from `results/nbody_kbc.json::voids[k]`.

```python
# Per void
f_k  = (β₁_k / max(β₀_k, 1)) · (1 / max(λ_min_k, 1e-6)) · (1 / R_k)  # 1/Mpc
y_k  = ltb_reference.delta_H0_ltb(delta=δ_k, R_mpc=R_k) - C1 * δ_k    # km/s/Mpc

# Unweighted closed-form 1D LSQ
α*   = (Σ f_k·y_k) / (Σ f_k²)                                         # km/s

# Bootstrap (B = ATFT_NBODY_CAL_BOOTSTRAP_B, default 2000)
for b in range(B):
    idx = rng.integers(K, size=K)                 # resample with replacement
    α_b = (Σ f[idx]·y[idx]) / (Σ f[idx]²)
α_16, α_50, α_84 = percentile(α_bs, [16, 50, 84])

# F-test nested: M0 = kinematic only (0 free params), M1 = kinematic + α·f_topo (1 free)
RSS_0 = Σ (C1·δ_k - ΔH₀_LTB(δ_k, R_k))²    # all residual assigned to nonlinearity
RSS_1 = Σ (α*·f_k - y_k)²
F     = (RSS_0 - RSS_1) / (RSS_1 / (K - 1))
p_F   = 1 - scipy.stats.f.cdf(F, 1, K - 1)
```

Output `results/nbody_calibration.json` schema:

```json
{
  "alpha_star": <float | null>,
  "alpha_units": "km/s",
  "alpha_ci_68": [<float>, <float>] | null,
  "alpha_bootstrap_median": <float> | null,
  "K": <int>,
  "bootstrap_B": <int>,
  "chi2_reduced": <float>,
  "pearson_r_f_y": <float>,
  "p_F_alpha_vs_zero": <float>,
  "reason": "fit succeeded" | "insufficient voids (K<min)" | "f_topo all zero",
  "per_void": [ {"delta": ..., "R_mpc": ..., "f_topo": ..., "y_ltb_residual": ..., "beta1": ...}, ... ],
  "ltb_reference_source": "ltb_reference.delta_H0_ltb (Gaussian profile heuristic)",
  "timestamp": "<ISO-8601 UTC>"
}
```

**No external LTB numerical integrator in v1** — spec §6.2.a fallback stays available but is NOT wired into the calibration loop unless `alpha_star` comes out pathological (`|α*| > 10^4 km/s` or `p_F > 0.5`), in which case `nbody_calibration.py` sets `reason = "LTB heuristic may be stretched — see spec §6.2.a"` and the verdict downgrades but the experiment does not fail.

## run_all.py integration

Extend the existing conditional-launch block after `nbody_kbc`:

```python
# After nbody_kbc (existing, unchanged)
# NEW: conditional nbody_calibration
if (results_dir / "nbody_kbc.json").exists():
    nbody_kbc_data = json.loads((results_dir / "nbody_kbc.json").read_text())
    n_voids = len(nbody_kbc_data.get("voids", []))
    min_voids = int(os.environ.get("ATFT_NBODY_CAL_MIN_VOIDS", "3"))
    if n_voids >= min_voids:
        print(f"nbody_calibration: launching with K={n_voids} voids")
        subprocess.run([sys.executable, "-m", "problems.hubble_tension_web.experiments.nbody_calibration"], env=env_for_calibration)
    else:
        print(f"nbody_calibration: only {n_voids} voids (< {min_voids}); skipping.")
# Aggregate (unchanged; already picks up nbody_calibration.json conditionally)
```

## aggregate.py integration

Conditional new section, gated on file existence:

```python
if (OUTPUT / "nbody_calibration.json").exists():
    cal = json.loads(...)
    lines.extend([
        "",
        "## Leg 4: Real-Void α Calibration",
        "",
        f"- K real voids: {cal['K']}",
        f"- α* = **{cal['alpha_star']:.4g} km/s** (68% CI: {cal['alpha_ci_68']})" if cal['alpha_star'] else "- α*: undetermined",
        f"- F-test p-value (α vs 0): {cal['p_F_alpha_vs_zero']:.3g}",
        f"- χ²_reduced: {cal['chi2_reduced']:.3g}, Pearson r(f, y): {cal['pearson_r_f_y']:.3g}",
        "",
    ])
```

## Acceptance gates

- All existing **73 passing + 3 xfailed** fast-suite tests continue to pass (frozen core math).
- New tests:
  - `test_mdpl2_download.py`: mocks HTTP, asserts `(x, y, z, mvir)` Parquet in pure Mpc/M_sun (verifies h-conversion happened), hash-stable output, schema contract preserved.
  - `test_nbody_calibration.py`: on a K=8 synthetic-records mini-fixture (with planted β₁ ∈ {0, 1, 2, 3}), verifies α* finite, bootstrap CI brackets α*, F-test returns valid p-value, per-void records complete.
  - `test_run_all.py`: 2 new tests for the calibration conditional leg.
- **Stubbed-download regression**: `ATFT_MDPL2_DOWNLOAD_ENABLED` unset or `"0"` → `fetch_from_network` still raises `NotImplementedError` with the existing README pointer message. Preserves `test_network_fetch_is_stubbed`.
- **End-to-end sanity**: `run_all.py` with `ATFT_NBODY_CACHE_FILE=<committed fixture>` produces both `nbody_kbc.json` and `nbody_calibration.json` without network access, the latter with `reason = "insufficient voids (K<3)"` or `"f_topo all zero"` depending on fixture outcome. Aggregate picks up both.

## Out of scope (deferred to v2+)

- **Numerical LTB integrator** (`ltb_numerical.py` from swarm 2's proposal). Available as follow-up if α* gates trip.
- **`calibration_stats.py` as a shared utility**. v1 keeps the bootstrap/F-test inside `nbody_calibration.py`; a refactor to a shared util is v2 if a second calibrator ever wants the same primitives.
- **Per-channel int8 quantization** for NPU (still deferred from perf pass).
- **Full 1 Gpc³ MDPL2 pull** (500 Mpc sub-box only).
- **RSD / observational-space treatment** (real-space positions from MDPL2 are good enough for v1).
- **IllustrisTNG baryonic catalogs** (still DM-only).

## Risks

- **CosmoSim SQL job queue latency** can be 10-30 min during peak. Document that the first `ATFT_MDPL2_DOWNLOAD_ENABLED=1` run may take a coffee break; subsequent runs hit the cache.
- **SQL schema drift**: if CosmoSim renames `mvir` to `Mvir_all` or similar, the download breaks. Put the column-alias map in `mdpl2_download.py` and pin the MDPL2 API version used.
- **Single-snapshot periodic-box-edge halos**: if `find_voids` picks a candidate with center within `R_eff` of the sub-box boundary, the void is truncated. Filter candidates by `center ± R_eff ⊂ (0, 500)` before passing to calibration.
- **K < ATFT_NBODY_CAL_MIN_VOIDS**: expected on first runs at small sub-boxes. The `reason = "insufficient voids (K<3)"` branch is the honest output, not a failure.
- **Numerical heuristic stretch**: documented in the schema; won't silently poison α*.

## Success criteria for v1 landing

1. Fast suite: **73 passed + 3 xfailed + N new passing** (new N ≈ 8-12 tests).
2. `run_all.py` with fixture cache: writes `nbody_kbc.json` AND `nbody_calibration.json`; both pass their tests; aggregate REPORT.md mentions both.
3. Opt-in path verified: `ATFT_MDPL2_DOWNLOAD_ENABLED=1 ATFT_SCISERVER_TOKEN=<real>` fetches a 500 Mpc MDPL2 sub-box, converts to Parquet, pipeline runs end-to-end (manual validation — not in CI).
4. If β₁ > 0 on real MDPL2 voids: α* is finite with a bootstrap CI; p_F_alpha_vs_zero is a real number; MASTER.md §6 gets a new "real-data" row.
5. If β₁ = 0 on all real voids: honest null reported in JSON (`"f_topo all zero"`), and the result motivates either (a) higher-resolution tidal-tensor grid (`ATFT_NBODY_GRID=256`), (b) larger sub-box (v2 scale-up), or (c) IllustrisTNG (richer sub-structure, v2 data source).

Either outcome is a v1 success. The ATFT thesis is served either by a measured α* > 0 with uncertainty OR a measured α* = 0 with a bound.
