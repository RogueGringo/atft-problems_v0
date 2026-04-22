# N-Body Halo Ingestion

See `docs/superpowers/specs/2026-04-20-nbody-ingestion-design.md` for the nbody-pipeline design, and `docs/superpowers/specs/2026-04-21-forward-ops-design.md` for the real-MDPL2 + α-recalibration bundle.

## Data provenance

v1 ships with a synthetic test fixture only
(`tests/hubble_tension_web/nbody/fixtures/mini_mdpl2.parquet`).

**Opt-in real MDPL2 download (forward-ops).** With `ATFT_MDPL2_DOWNLOAD_ENABLED=1`
and `ATFT_SCISERVER_TOKEN=<your SciServer/CosmoSim token>` exported, the
`mdpl2_fetch.fetch_from_network` path delegates to `mdpl2_download.fetch_sub_box`,
which submits a TAP SQL async job against the CosmoSim Rockstar z=0 catalog
(`MDPL2.Rockstar` table, `pid=-1` host-halo filter, 500 Mpc sub-box from the
origin), polls until complete (typically 10-30 min), downloads the CSV result,
applies little-h conversion (`h=0.6777` → pure Mpc/M_sun), and writes Parquet
matching the `("x","y","z","mvir")` schema contract.

The CosmoSim TAP endpoint shape (`/async` POST, `/phase` polling, `/results/result`
GET) is **UNVERIFIED** as of the first live run. Both endpoint base and SQL body
are env-overridable (`ATFT_MDPL2_URL`, `ATFT_MDPL2_SQL`) so the maintainer can
patch without a code change if the production protocol has drifted.

**Manual fallback.** If the SQL path is unavailable:
1. Obtain a Rockstar z=0 halo catalog for a 500 Mpc MDPL2 sub-box manually
   (https://www.cosmosim.org, SciServer login).
2. Convert to Parquet with columns `x, y, z, mvir` (float64, **pure Mpc/M_sun**,
   NOT h⁻¹ units — apply `/ 0.6777` yourself).
3. Place at `~/.cache/atft/mdpl2/mdpl2_z0_500Mpc.parquet` or the path given by
   `$ATFT_DATA_CACHE`.

**CI guarantee.** With `ATFT_MDPL2_DOWNLOAD_ENABLED` unset (the default),
`fetch_from_network` raises `NotImplementedError("CosmoSim …")` — no network
traffic in CI, `test_network_fetch_is_stubbed` continues to pass.

## Fixture regeneration

```bash
python -m tests.hubble_tension_web.nbody.fixtures.generate_fixture
```
Commit the result.

## Configuration

- `ATFT_DATA_CACHE`: override cache directory (default `~/.cache/atft/`)
- `ATFT_NBODY_GRID`: override T-web grid size (default 128; must be power of 2)
- `ATFT_NBODY_LAMBDA_TH`: override tidal eigenvalue threshold (default 0.0)
- `ATFT_MDPL2_DOWNLOAD_ENABLED`: `"1"` enables the live CosmoSim SQL path (default `"0"`)
- `ATFT_SCISERVER_TOKEN`: required when `_ENABLED="1"` (no default)
- `ATFT_MDPL2_URL`: override the TAP endpoint (default `https://www.cosmosim.org/tap`)
- `ATFT_MDPL2_SQL`: wholesale-override the SQL body (default is built in `mdpl2_download._build_default_sql`)
- `ATFT_NBODY_CAL_MIN_VOIDS`: min K real voids before α fit is attempted (default `3`)
- `ATFT_NBODY_CAL_BOOTSTRAP_B`: bootstrap resamples (default `2000`)
- `ATFT_HUBBLE_RESULTS_DIR`: override the results directory (used by `nbody_calibration.py` tests)
