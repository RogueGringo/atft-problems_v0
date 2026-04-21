# N-Body Halo Ingestion

See `docs/superpowers/specs/2026-04-20-nbody-ingestion-design.md` for full design.

## Data provenance

v1 ships with a synthetic test fixture only
(`tests/hubble_tension_web/nbody/fixtures/mini_mdpl2.parquet`).
The real MDPL2 fetch path in `mdpl2_fetch.py` raises
`NotImplementedError` because the CosmoSim download requires either
a direct URL that the maintainer has not yet confirmed or a SciServer
account.

To enable real MDPL2 data for `nbody_kbc.py`:
1. Obtain a Rockstar z=0 halo catalog for a 500 Mpc MDPL2 sub-box
   (see https://www.cosmosim.org for access; SciServer login typically
   required).
2. Convert to Parquet with columns `x, y, z, mvir` (float32, Mpc/M_sun).
3. Place at `~/.cache/atft/mdpl2/mdpl2_z0_500Mpc.parquet` or the path
   given by `ATFT_DATA_CACHE`.

## Fixture regeneration

```bash
python -m tests.hubble_tension_web.nbody.fixtures.generate_fixture
```
Commit the result.

## Configuration

- `ATFT_DATA_CACHE`: override cache directory (default `~/.cache/atft/`)
- `ATFT_NBODY_GRID`: override T-web grid size (default 128; must be power of 2)
- `ATFT_NBODY_LAMBDA_TH`: override tidal eigenvalue threshold (default 0.0)
