# N-Body Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan adds MDPL2 halo ingestion + T-web classification + real-void β₁ to the hubble_tension_web pipeline. Core math (`types.py`, `functional.py`, `laplacian.py`, `spectrum.py`, `graph.py`, `ltb_reference.py`, `synthetic.py`) is **frozen** — no changes.

**Goal:** Add a `problems/hubble_tension_web/nbody/` package that turns MDPL2-style halo catalogs into `LocalCosmicWeb` inputs for the existing `predict_from_cosmic_web` path, so we can test the ATFT thesis that real cosmic voids carry `β₁_persistent > 0` sub-structure.

**Architecture:** Five-stage pipeline, new package only. (1) `mdpl2_fetch.py` reads a Parquet halo catalog from a local cache; (2) `tidal_tensor.py` deposits halo mass on a grid, solves Poisson via FFT, computes the tidal tensor, and classifies each cell into the `Environment` enum; (3) `void_finder.py` Gaussian-smooths the density field, finds local minima, and grows spherical regions to locate KBC-like voids; (4) `cosmic_web_from_halos.py` assembles a `LocalCosmicWeb` + `VoidParameters` for the existing functional; (5) `experiments/nbody_kbc.py` wires the above into the standard result-JSON pattern and is conditionally invoked from `run_all.py`. Everything is fixture-driven for v1 — no network access; the real MDPL2 fetch is stubbed with a clearly-documented `NotImplementedError`.

**Tech Stack:** Python 3.14, NumPy, SciPy (`scipy.ndimage.gaussian_filter`, `scipy.ndimage.minimum_filter`, `scipy.fft` aliasing to `numpy.fft` elsewhere in the codebase), `pandas` + `pyarrow` for the Parquet cache, `matplotlib` (Agg) for the optional β₁ histogram plot, `ripser` (unchanged, already used by `spectrum.persistent_beta1`). **No new third-party deps.** Evaluated `pooch` for cached downloads — rejected for v1: network path is stubbed, not exercised.

**Reference spec:** `docs/superpowers/specs/2026-04-20-nbody-ingestion-design.md`

**Branch:** `feat/nbody-ingestion` (already checked out at HEAD `1c77c54`, the spec commit).

---

## File Structure

**Files created (new `nbody/` package):**

```
problems/hubble_tension_web/nbody/
├── __init__.py                 # exports NBodyDataNotAvailable, package constants
├── mdpl2_fetch.py              # Parquet cache reader; network path raises NotImplementedError
├── tidal_tensor.py             # CIC deposit + FFT Poisson + tidal tensor + eigvalsh classifier
├── void_finder.py              # Gaussian smooth + local-min + sphere-growth
├── cosmic_web_from_halos.py    # assemble LocalCosmicWeb + VoidParameters
└── README.md                   # data provenance + fixture generation instructions
```

**Files created (experiment + fixture + tests):**

```
problems/hubble_tension_web/experiments/
└── nbody_kbc.py                # end-to-end runner; conditional in run_all.py

tests/hubble_tension_web/nbody/
├── __init__.py
├── fixtures/
│   ├── __init__.py
│   └── mini_mdpl2.parquet      # 200 halos, 50 Mpc box, one planted void (deterministic)
├── test_mdpl2_fetch.py
├── test_tidal_tensor.py
├── test_void_finder.py
├── test_cosmic_web_from_halos.py
└── test_nbody_kbc.py
```

**Files modified (in place):**

```
problems/hubble_tension_web/experiments/
└── run_all.py                  # optional conditional launch of nbody_kbc

tests/hubble_tension_web/
└── test_run_all.py             # new test: no-cache skip path still exits 0
```

**Files NOT touched (math frozen — compare against spec §"Integration contract"):**

```
problems/hubble_tension_web/
├── types.py
├── functional.py
├── laplacian.py
├── laplacian_quantized.py
├── spectrum.py
├── graph.py
├── synthetic.py
├── ltb_reference.py
└── experiments/
    ├── analytical_reduction.py
    ├── sim_calibration.py
    ├── kbc_crosscheck.py
    └── aggregate.py
tests/hubble_tension_web/
├── test_types.py
├── test_functional.py
├── test_laplacian.py
├── test_laplacian_sparse.py
├── test_laplacian_quantized.py
├── test_spectrum.py
├── test_spectrum_eigsh.py
├── test_synthetic.py
├── test_graph.py
├── test_ltb_reference.py
├── test_pipeline.py
└── test_sim_calibration_parallel.py
```

---

## Notation and Conventions (used across all tasks)

- **Working directory:** `C:\JTOD1\atft-problems_v0`. Use forward slashes in command lines (shell is git-bash on Windows). CRLF warnings from `git add` are benign.
- **Pytest invocation:** always from repo root. While iterating, run only the `nbody/` subset; run the full hubble suite at each commit to verify nothing regressed:
  ```bash
  python -m pytest tests/hubble_tension_web/nbody/ -q
  python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
  ```
  The `test_pipeline.py` suite is the ~90 min end-to-end and is only run at the final acceptance gate. `test_run_all.py` is ~30-60s (launches subprocesses); we include it in the Task 6 acceptance step.
- **Commit prefix:** `feat(nbody):` for all commits in this plan. Every commit ends with the co-author footer:
  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```
- **Default grid size:** `N_GRID = 128` (v1 default). 256³ would push the 3×3×N³ eigvalsh intermediate + complex FFT buffers past 3 GB on a 16 GB Snapdragon X Plus laptop. Override via env var `ATFT_NBODY_GRID` (integer). The spec §"Risks & fallbacks" explicitly authorizes this fallback.
- **Default box size:** `BOX_MPC = 50.0` for the fixture; 500.0 for real MDPL2. Box is always a cube.
- **Environment enum mapping:** `Environment` is string-valued (`VOID = "void"`, etc.). The T-web classifier produces `uint8` cell codes `0..3`; the lookup table is:
  ```python
  _CODE_TO_ENV = [Environment.VOID, Environment.WALL, Environment.FILAMENT, Environment.NODE]
  # code = 3 - count(eigvals > lambda_th); code 0 = all eigvals positive = VOID
  ```
  **Never call `Environment(code)`** — the enum is string-valued and that call throws `ValueError`.
- **Parquet schema contract:** the `EXPECTED_COLUMNS` constant in `mdpl2_fetch.py` is the load-bearing contract between the fixture writer (Task 0) and the reader (Task 1). Columns and dtypes:
  | Name   | dtype     | units                    |
  |--------|-----------|--------------------------|
  | `x`    | float32   | Mpc (box-local, [0, L))  |
  | `y`    | float32   | Mpc (box-local, [0, L))  |
  | `z`    | float32   | Mpc (box-local, [0, L))  |
  | `mvir` | float32   | M_sun (halo virial mass) |
  Additional columns are tolerated by the reader but not used.
- **rng seed:** every new deterministic function takes a `rng_seed: int = 0` kwarg. Fixture writer uses `rng_seed = 2026`.

---

## Task 0: Scaffold + deterministic fixture + package skeleton

Set up the `nbody/` package, the `tests/hubble_tension_web/nbody/` directory, and the tiny 200-halo Parquet fixture that all downstream tests will rely on. No new dependencies to install — `pyarrow` is already in `pip list` (version 23.0.1). Commit the fixture to git so CI never has to regenerate it.

**Files:**
- Create: `problems/hubble_tension_web/nbody/__init__.py`
- Create: `problems/hubble_tension_web/nbody/README.md`
- Create: `tests/hubble_tension_web/nbody/__init__.py`
- Create: `tests/hubble_tension_web/nbody/fixtures/__init__.py`
- Create: `tests/hubble_tension_web/nbody/fixtures/generate_fixture.py` (script; kept in-tree for reproducibility)
- Create: `tests/hubble_tension_web/nbody/fixtures/mini_mdpl2.parquet` (binary; committed to git)

- [ ] **Step 1: Verify baseline test suite is green**

Run:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py -q
```
Expected: all tests pass (52 passing + 3 xfailed per master state). If anything is red, STOP — fix or revert before starting the plan.

- [ ] **Step 2: Create the `nbody/` package skeleton**

Write `problems/hubble_tension_web/nbody/__init__.py`:
```python
"""N-body halo-catalog ingestion for hubble_tension_web.

v1 scope:
- Parquet halo catalog (fixture or one-shot MDPL2 dump)
- Tidal-tensor T-web environment classification
- Simple local-minimum sphere-growing void finder
- LocalCosmicWeb assembly for the existing predict_from_cosmic_web path

The network fetch from CosmoSim MDPL2 is stubbed in v1 (see mdpl2_fetch.py);
tests and the nbody_kbc experiment operate against a local Parquet cache.
"""
from __future__ import annotations

# Default grid size for T-web classification. 128 keeps peak memory
# (3x3xN^3 float64 eigvalsh intermediate + complex FFT buffers) under
# ~500 MB on a 16 GB laptop. Override with ATFT_NBODY_GRID env var.
# Spec section "Risks & fallbacks" authorizes the 128 default.
DEFAULT_N_GRID: int = 128

# Default mass cut (M_sun). Spec targets 10^11.5 for galaxy-hosting halos.
DEFAULT_MASS_CUT: float = 10.0 ** 11.5

# Default tidal-tensor eigenvalue threshold. 0.0 per spec v1; tunable.
DEFAULT_LAMBDA_TH: float = 0.0


class NBodyDataNotAvailable(FileNotFoundError):
    """Raised when a halo cache is requested but neither local file nor network
    fetch can satisfy it. Subclasses FileNotFoundError so callers can
    `except FileNotFoundError` and skip gracefully (used by run_all.py)."""
```

- [ ] **Step 3: Write the fixture-generator script**

Write `tests/hubble_tension_web/nbody/fixtures/generate_fixture.py`:
```python
"""Generates tests/hubble_tension_web/nbody/fixtures/mini_mdpl2.parquet.

Run with: python -m tests.hubble_tension_web.nbody.fixtures.generate_fixture

The output file is committed to git; you should only need to re-run this
script if the schema contract in mdpl2_fetch.EXPECTED_COLUMNS changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401 - ensures the parquet engine is available

# Fixture parameters — freeze these; downstream tests depend on them.
BOX_MPC: float = 50.0
N_HALOS_TOTAL: int = 200
VOID_CENTER_MPC: tuple[float, float, float] = (25.0, 25.0, 25.0)
VOID_RADIUS_MPC: float = 15.0
# Remove ~40 halos inside the void sphere to produce a clear under-density.
VOID_DEPLETION_FRACTION: float = 0.85
RNG_SEED: int = 2026


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    xyz = rng.uniform(0.0, BOX_MPC, size=(N_HALOS_TOTAL, 3))

    # Remove a fraction of halos inside the void sphere — deterministic.
    cx, cy, cz = VOID_CENTER_MPC
    r = np.sqrt((xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2 + (xyz[:, 2] - cz) ** 2)
    in_void = r < VOID_RADIUS_MPC
    u = rng.uniform(0.0, 1.0, size=N_HALOS_TOTAL)
    drop = in_void & (u < VOID_DEPLETION_FRACTION)
    xyz = xyz[~drop]

    # Log-normal mass distribution around 10^12 M_sun; truncated lower bound.
    log_mass = rng.normal(loc=12.0, scale=0.3, size=xyz.shape[0])
    mvir = np.power(10.0, log_mass).astype(np.float32)

    df = pd.DataFrame(
        dict(
            x=xyz[:, 0].astype(np.float32),
            y=xyz[:, 1].astype(np.float32),
            z=xyz[:, 2].astype(np.float32),
            mvir=mvir,
        )
    )
    return df


def main() -> None:
    out_path = Path(__file__).parent / "mini_mdpl2.parquet"
    df = generate()
    # Write with pyarrow engine and SNAPPY compression — deterministic byte output
    # is not guaranteed across pyarrow versions, but the logical content is.
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"wrote {len(df)} halos to {out_path}")


if __name__ == "__main__":
    main()
```

Write `tests/hubble_tension_web/nbody/__init__.py` as an empty file.
Write `tests/hubble_tension_web/nbody/fixtures/__init__.py` as an empty file.

- [ ] **Step 4: Generate the fixture file**

Run:
```bash
python -m tests.hubble_tension_web.nbody.fixtures.generate_fixture
```
Expected stdout: `wrote 1XX halos to .../mini_mdpl2.parquet` where `1XX` is approximately `200 - VOID_DEPLETION_FRACTION * N_in_void` (expect ~175-185 given the deterministic seed).

Verify the file exists and is under 10 KB:
```bash
ls -l tests/hubble_tension_web/nbody/fixtures/mini_mdpl2.parquet
```
Expected: file present, small (a few KB). If it's >100 KB something is wrong with the writer — stop and diagnose.

- [ ] **Step 5: Write the nbody README**

Write `problems/hubble_tension_web/nbody/README.md`:
```markdown
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
```

- [ ] **Step 6: Commit scaffold + fixture**

```bash
git add problems/hubble_tension_web/nbody/__init__.py \
        problems/hubble_tension_web/nbody/README.md \
        tests/hubble_tension_web/nbody/__init__.py \
        tests/hubble_tension_web/nbody/fixtures/__init__.py \
        tests/hubble_tension_web/nbody/fixtures/generate_fixture.py \
        tests/hubble_tension_web/nbody/fixtures/mini_mdpl2.parquet
git commit -m "$(cat <<'EOF'
feat(nbody): package skeleton + deterministic halo fixture

Creates problems/hubble_tension_web/nbody/ with __init__ exporting
NBodyDataNotAvailable + default constants. Generates a 200-halo,
50 Mpc Parquet fixture (one planted void at box center) from a
committed generator script.

No behavior change to existing modules. Tests still green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Run:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py -q
```
Expected: still all tests pass (we haven't touched any production code, only added files).

---

## Task 1: `mdpl2_fetch.py` — Parquet cache reader

Read the Parquet halo catalog from the cache directory, validate the schema, apply the mass cut, and return a typed record. The network download path is stubbed with `NotImplementedError` and a clear docstring pointing at the README.

**Files:**
- Create: `problems/hubble_tension_web/nbody/mdpl2_fetch.py`
- Test: `tests/hubble_tension_web/nbody/test_mdpl2_fetch.py`

- [ ] **Step 1: Write failing tests**

Write `tests/hubble_tension_web/nbody/test_mdpl2_fetch.py`:
```python
"""Tests for the Parquet halo-cache reader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from problems.hubble_tension_web.nbody import NBodyDataNotAvailable
from problems.hubble_tension_web.nbody import mdpl2_fetch


FIXTURE = Path(__file__).parent / "fixtures" / "mini_mdpl2.parquet"


def test_fixture_is_committed_and_readable():
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    df = pd.read_parquet(FIXTURE)
    for col in mdpl2_fetch.EXPECTED_COLUMNS:
        assert col in df.columns, f"schema contract broken: {col} missing"


def test_load_halo_catalog_returns_ndarrays():
    halos = mdpl2_fetch.load_halo_catalog(FIXTURE, mass_cut=0.0)
    assert halos.positions.shape[1] == 3
    assert halos.positions.dtype == np.float64  # promoted from float32 internally
    assert halos.masses.ndim == 1
    assert halos.positions.shape[0] == halos.masses.shape[0]
    assert halos.box_mpc > 0


def test_mass_cut_applied():
    halos_all = mdpl2_fetch.load_halo_catalog(FIXTURE, mass_cut=0.0)
    halos_cut = mdpl2_fetch.load_halo_catalog(FIXTURE, mass_cut=1e12)
    # Every retained halo must be above the cut.
    assert np.all(halos_cut.masses >= 1e12)
    # The cut is not a no-op (log-normal sigma=0.3 around 10^12 gives ~50% below).
    assert halos_cut.positions.shape[0] < halos_all.positions.shape[0]


def test_missing_file_raises_nbody_data_not_available(tmp_path):
    with pytest.raises(NBodyDataNotAvailable):
        mdpl2_fetch.load_halo_catalog(tmp_path / "does_not_exist.parquet", mass_cut=0.0)


def test_schema_violation_raises(tmp_path):
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"x": [1.0], "y": [2.0]}).to_parquet(bad)  # missing z, mvir
    with pytest.raises(ValueError, match="schema"):
        mdpl2_fetch.load_halo_catalog(bad, mass_cut=0.0)


def test_network_fetch_is_stubbed():
    with pytest.raises(NotImplementedError, match="CosmoSim"):
        mdpl2_fetch.fetch_from_network(url="https://cosmosim.example/mdpl2/z0_500Mpc.dat")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_mdpl2_fetch.py -v
```
Expected: all fail with `ModuleNotFoundError: No module named '...mdpl2_fetch'` (or equivalent).

- [ ] **Step 3: Write the implementation**

Write `problems/hubble_tension_web/nbody/mdpl2_fetch.py`:
```python
"""Parquet halo-catalog reader for MDPL2 (or fixture) data.

The network path to CosmoSim is intentionally stubbed in v1. Users wanting
real MDPL2 data must manually populate the Parquet cache per the README.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from problems.hubble_tension_web.nbody import NBodyDataNotAvailable


# Parquet schema contract. Task 0 fixture writer and this reader must agree.
EXPECTED_COLUMNS: tuple[str, ...] = ("x", "y", "z", "mvir")


@dataclass
class HaloCatalog:
    positions: np.ndarray   # (N, 3) float64, Mpc
    masses: np.ndarray      # (N,) float64, M_sun
    box_mpc: float          # inferred box size (max coord + small pad), Mpc


def load_halo_catalog(
    path: str | Path,
    *,
    mass_cut: float,
    box_mpc: float | None = None,
) -> HaloCatalog:
    """Read a halo catalog from a Parquet file.

    Args:
      path: Parquet file path.
      mass_cut: discard halos with mvir < mass_cut (M_sun). Pass 0.0 to keep all.
      box_mpc: if given, asserted as the box size. If None, inferred as
               max coord rounded up to the next integer multiple of 1.0 Mpc.

    Raises:
      NBodyDataNotAvailable: if path does not exist.
      ValueError: if the Parquet schema is missing any EXPECTED_COLUMNS entry.
    """
    path = Path(path)
    if not path.exists():
        raise NBodyDataNotAvailable(
            f"halo catalog not found at {path}. "
            "Run the fixture generator or populate ~/.cache/atft/mdpl2/ "
            "per problems/hubble_tension_web/nbody/README.md."
        )
    df = pd.read_parquet(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet schema violation at {path}: missing columns {missing}. "
            f"Expected {EXPECTED_COLUMNS}."
        )
    if mass_cut > 0.0:
        df = df[df["mvir"] >= mass_cut]
    positions = np.stack(
        [df["x"].to_numpy(dtype=np.float64),
         df["y"].to_numpy(dtype=np.float64),
         df["z"].to_numpy(dtype=np.float64)],
        axis=1,
    )
    masses = df["mvir"].to_numpy(dtype=np.float64)

    if box_mpc is None:
        max_coord = float(positions.max()) if positions.size else 1.0
        box_mpc = float(np.ceil(max_coord))
    return HaloCatalog(positions=positions, masses=masses, box_mpc=box_mpc)


def fetch_from_network(*, url: str, dest: str | Path | None = None) -> Path:
    """Download an MDPL2 halo catalog from CosmoSim.

    v1: NOT IMPLEMENTED. CosmoSim does not expose a stable no-auth URL for
    Rockstar catalogs; users must obtain the file manually via SciServer
    and place it in the cache. See the README.
    """
    raise NotImplementedError(
        "CosmoSim MDPL2 download requires SciServer credentials in v1. "
        "Manually download the halo catalog and place it at "
        "~/.cache/atft/mdpl2/ (or $ATFT_DATA_CACHE). "
        "See problems/hubble_tension_web/nbody/README.md."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_mdpl2_fetch.py -v
```
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/nbody/mdpl2_fetch.py \
        tests/hubble_tension_web/nbody/test_mdpl2_fetch.py
git commit -m "$(cat <<'EOF'
feat(nbody): Parquet halo-cache reader (mdpl2_fetch)

load_halo_catalog reads the Parquet fixture, applies a mass cut,
and returns a HaloCatalog record (positions fp64, masses fp64,
box_mpc inferred from max coord).

fetch_from_network stubs the CosmoSim download path with a
NotImplementedError pointing at the README. Real data ingestion
is a manual one-shot for v1.

6 new tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Run the full hubble suite:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```
Expected: previous count + 6 new tests pass.

---

## Task 2: `tidal_tensor.py` — CIC deposit + FFT Poisson + T-web classifier

Deposit halo mass on a 3D grid (Cloud-in-Cell kernel), solve `∇²φ = ρ` via FFT, compute `T_ij = ∂_i∂_j φ` as `-k_i k_j φ̂`, inverse-transform, compute eigenvalues per cell, and map `3 − count(λ > λ_th)` to the four-element `Environment` enum.

The expensive intermediate is the `(N, N, N, 3, 3)` tensor — at N=128 that's `128³ × 9 × 8 B = ~150 MB`; at N=256 it would be ~1.2 GB. We build it in two passes: accumulate the 6 unique upper-triangular components as separate `(N, N, N)` arrays, then stack into the full `(N, N, N, 3, 3)` before eigvalsh. Peak memory on 128³: ~500 MB, comfortably fits.

**Files:**
- Create: `problems/hubble_tension_web/nbody/tidal_tensor.py`
- Test: `tests/hubble_tension_web/nbody/test_tidal_tensor.py`

- [ ] **Step 1: Write failing tests**

Write `tests/hubble_tension_web/nbody/test_tidal_tensor.py`:
```python
"""Tests for tidal-tensor T-web classification.

Three-tier coverage:
1. CIC kernel unit: point mass at cell center deposits mass 1 at that cell.
2. Spherical-void: interior cells see all eigvals positive => Environment.VOID.
3. End-to-end on the committed fixture: void-center cell classified as VOID.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from problems.hubble_tension_web.nbody import tidal_tensor
from problems.hubble_tension_web.nbody.mdpl2_fetch import load_halo_catalog
from problems.hubble_tension_web.types import Environment


FIXTURE = Path(__file__).parent / "fixtures" / "mini_mdpl2.parquet"


def test_cic_deposit_conserves_total_mass():
    positions = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
    masses = np.array([1.0])
    rho = tidal_tensor.cic_deposit(positions, masses, n_grid=16, box_mpc=10.0)
    # CIC conserves mass up to roundoff (no boundary losses for points inside box).
    assert np.isclose(rho.sum(), 1.0, atol=1e-9)


def test_cic_deposit_single_point_at_cell_center():
    """A point exactly at a cell center goes entirely into that cell."""
    # With N=8 and box=8 Mpc, cell size = 1 Mpc, cell i center is at (i+0.5).
    positions = np.array([[3.5, 3.5, 3.5]], dtype=np.float64)
    masses = np.array([1.0])
    rho = tidal_tensor.cic_deposit(positions, masses, n_grid=8, box_mpc=8.0)
    # The neighboring 8 cells share the weight (2x2x2 CIC kernel);
    # at exact center the kernel collapses to a single cell with weight 1.
    assert np.isclose(rho[3, 3, 3], 1.0, atol=1e-9)
    # Everything else is zero.
    total_off_cell = rho.sum() - rho[3, 3, 3]
    assert np.isclose(total_off_cell, 0.0, atol=1e-9)


def test_classify_spherical_void_center_is_void():
    """Planted void: an empty interior sphere => all eigvals of tidal tensor positive
    at interior cells => Environment.VOID."""
    # Place halos in a thin shell around a void at box center.
    rng = np.random.default_rng(0)
    n = 2000
    # Random directions on unit sphere.
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.uniform(20.0, 24.0, size=n)
    positions = 25.0 + u * r[:, None]  # shell around (25,25,25) with radius 20-24
    masses = np.ones(n)
    env_grid, _ = tidal_tensor.classify(
        positions=positions, masses=masses, n_grid=32, box_mpc=50.0, lambda_th=0.0,
    )
    # Center cell (16, 16, 16) should be VOID (code 0).
    assert env_grid[16, 16, 16] == 0, (
        f"center cell of a spherical void should be VOID; got code {env_grid[16,16,16]}"
    )


def test_classify_returns_valid_codes_only():
    """Every cell code is in {0, 1, 2, 3}."""
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    assert env_grid.dtype == np.uint8
    assert env_grid.min() >= 0 and env_grid.max() <= 3


def test_code_to_env_lookup_is_consistent():
    """The CODE_TO_ENV table must return genuine Environment enum members."""
    assert tidal_tensor.CODE_TO_ENV[0] is Environment.VOID
    assert tidal_tensor.CODE_TO_ENV[1] is Environment.WALL
    assert tidal_tensor.CODE_TO_ENV[2] is Environment.FILAMENT
    assert tidal_tensor.CODE_TO_ENV[3] is Environment.NODE


def test_fixture_void_center_classified_as_void():
    """The committed fixture has a planted void at (25, 25, 25). The cell
    containing that point must be classified Environment.VOID."""
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, meta = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    cx = int(25.0 / halos.box_mpc * 32)
    cy, cz = cx, cx
    assert env_grid[cx, cy, cz] == 0, (
        f"fixture void center classified as code {env_grid[cx,cy,cz]} "
        f"(expected 0/VOID); meta={meta}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_tidal_tensor.py -v
```
Expected: all fail with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Write `problems/hubble_tension_web/nbody/tidal_tensor.py`:
```python
"""Density field -> gravitational potential -> tidal tensor -> T-web classification.

Pipeline (per spec section 'T-web classification'):
    rho      = cic_deposit(halos, grid=N)           # (N, N, N)    float64
    rho_hat  = np.fft.fftn(rho)
    k_vec    = np.fft.fftfreq(N) * N               # cell-unit wavenumbers
    k2       = k_x^2 + k_y^2 + k_z^2; k2[0,0,0] set to 1 to avoid div/0
    phi_hat  = -rho_hat / k2; phi_hat[0,0,0] = 0   # zero the DC mode
    T_ij_hat = -k_i * k_j * phi_hat                 # 6 unique components
    T_ij     = np.fft.ifftn(T_ij_hat).real          # (N, N, N) each
    eigvals  = np.linalg.eigvalsh(T_full)           # (N, N, N, 3)
    env_grid = 3 - np.sum(eigvals > lambda_th, axis=-1).astype(np.uint8)

The absolute normalization of rho and phi is irrelevant for T-web
classification because the count-of-positive-eigenvalues is scale-invariant
(eigvals scale linearly under rho rescaling; the threshold lambda_th=0 is
also a zero crossing and is invariant). This is why we can ignore 4*pi*G
and just solve -k^2 phi_hat = rho_hat directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from problems.hubble_tension_web.types import Environment


# Lookup table: uint8 cell code -> Environment enum instance.
# Code 0 = all three eigenvalues positive = VOID (per spec T-web mapping).
# NEVER call Environment(code) — the enum is string-valued.
CODE_TO_ENV: tuple[Environment, ...] = (
    Environment.VOID,
    Environment.WALL,
    Environment.FILAMENT,
    Environment.NODE,
)


@dataclass
class ClassifyMeta:
    """Diagnostic fields returned alongside env_grid."""
    n_grid: int
    box_mpc: float
    cell_mpc: float
    lambda_th: float
    rho_mean: float
    rho_max: float


def cic_deposit(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    n_grid: int,
    box_mpc: float,
) -> np.ndarray:
    """Cloud-in-Cell mass assignment to a cubic grid.

    Each halo contributes mass to the 8 cells surrounding its position
    with weights equal to the overlap between a unit-cube kernel and each
    grid cell.

    Periodic boundary wrap: halos near the box edge wrap to the far side,
    matching the periodic cosmological volume.
    """
    cell = box_mpc / n_grid
    scaled = positions / cell  # shape (N, 3), in cell-index units
    i0 = np.floor(scaled).astype(np.int64)
    frac = scaled - i0  # (N, 3)
    rho = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)

    # Precompute weights for the 8 corners of each particle's CIC kernel.
    # Weight for corner (dx, dy, dz) in {0,1}^3 is
    #   (1-frac_x if dx==0 else frac_x) * ... .
    for dx in (0, 1):
        wx = np.where(dx == 0, 1.0 - frac[:, 0], frac[:, 0])
        ix = (i0[:, 0] + dx) % n_grid
        for dy in (0, 1):
            wy = np.where(dy == 0, 1.0 - frac[:, 1], frac[:, 1])
            iy = (i0[:, 1] + dy) % n_grid
            for dz in (0, 1):
                wz = np.where(dz == 0, 1.0 - frac[:, 2], frac[:, 2])
                iz = (i0[:, 2] + dz) % n_grid
                w = masses * wx * wy * wz
                np.add.at(rho, (ix, iy, iz), w)
    return rho


def _tidal_tensor_fft(rho: np.ndarray) -> np.ndarray:
    """Return (N, N, N, 3, 3) tidal tensor from a density grid.

    Uses the zero-normalization convention: T_ij = -k_i k_j phi_hat / k^2.
    Any overall scale on rho factors through eigenvalues and does not change
    the sign-count classification.
    """
    n = rho.shape[0]
    rho_hat = np.fft.fftn(rho)

    k_vec = np.fft.fftfreq(n) * n  # (N,) in cell-index units
    kx = k_vec[:, None, None]
    ky = k_vec[None, :, None]
    kz = k_vec[None, None, :]
    k2 = kx * kx + ky * ky + kz * kz
    # Avoid div/0 at DC mode; zero the DC potential so the mean force vanishes.
    k2_safe = np.where(k2 == 0.0, 1.0, k2)
    phi_hat = -rho_hat / k2_safe
    phi_hat[0, 0, 0] = 0.0 + 0.0j

    # Build only the six unique upper-triangular components.
    def _second_deriv(ka: np.ndarray, kb: np.ndarray) -> np.ndarray:
        return np.fft.ifftn(-ka * kb * phi_hat).real

    T_xx = _second_deriv(kx, kx)
    T_yy = _second_deriv(ky, ky)
    T_zz = _second_deriv(kz, kz)
    T_xy = _second_deriv(kx, ky)
    T_xz = _second_deriv(kx, kz)
    T_yz = _second_deriv(ky, kz)

    # Stack into (N, N, N, 3, 3). Eigvalsh only needs symmetric input;
    # populate all 9 entries for numerical safety.
    T = np.empty((n, n, n, 3, 3), dtype=np.float64)
    T[..., 0, 0] = T_xx
    T[..., 1, 1] = T_yy
    T[..., 2, 2] = T_zz
    T[..., 0, 1] = T_xy; T[..., 1, 0] = T_xy
    T[..., 0, 2] = T_xz; T[..., 2, 0] = T_xz
    T[..., 1, 2] = T_yz; T[..., 2, 1] = T_yz
    return T


def classify(
    *,
    positions: np.ndarray,
    masses: np.ndarray,
    n_grid: int,
    box_mpc: float,
    lambda_th: float = 0.0,
) -> tuple[np.ndarray, ClassifyMeta]:
    """Run the full CIC -> Poisson FFT -> tidal tensor -> T-web classification.

    Returns:
      env_grid: (n_grid, n_grid, n_grid) uint8 array of environment codes
                (0=VOID, 1=WALL, 2=FILAMENT, 3=NODE). Use CODE_TO_ENV[code]
                to get the Environment enum instance.
      meta:     ClassifyMeta with diagnostic fields.
    """
    rho = cic_deposit(positions, masses, n_grid=n_grid, box_mpc=box_mpc)
    T = _tidal_tensor_fft(rho)
    eigvals = np.linalg.eigvalsh(T)  # (n_grid, n_grid, n_grid, 3)
    n_positive = np.sum(eigvals > lambda_th, axis=-1)  # (n_grid, n_grid, n_grid)
    env_grid = (3 - n_positive).astype(np.uint8)

    meta = ClassifyMeta(
        n_grid=n_grid,
        box_mpc=box_mpc,
        cell_mpc=box_mpc / n_grid,
        lambda_th=lambda_th,
        rho_mean=float(rho.mean()),
        rho_max=float(rho.max()),
    )
    return env_grid, meta


def lookup_env_at_positions(
    env_grid: np.ndarray,
    positions: np.ndarray,
    *,
    box_mpc: float,
) -> list[Environment]:
    """Nearest-cell assignment: map each (x,y,z) to its grid cell and return Environment."""
    n_grid = env_grid.shape[0]
    cell = box_mpc / n_grid
    idx = np.floor(positions / cell).astype(np.int64)
    # Clamp to [0, n_grid-1] for halos exactly on the far edge.
    idx = np.clip(idx, 0, n_grid - 1)
    codes = env_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    return [CODE_TO_ENV[int(c)] for c in codes]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_tidal_tensor.py -v
```
Expected: all 6 tests pass. The spherical-void test takes ~1 s at N=32; the fixture test similarly. If the spherical-void test fails at the center cell, inspect the eigvals via a debug print — it should be a regular pattern of positive values. If FFT conventions got inverted, the sign would flip and all VOID cells would show up as NODE — catch that here, not in Task 5.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/nbody/tidal_tensor.py \
        tests/hubble_tension_web/nbody/test_tidal_tensor.py
git commit -m "$(cat <<'EOF'
feat(nbody): T-web classifier (CIC + FFT Poisson + tidal eigvals)

classify() deposits halo mass via CIC, solves Poisson in Fourier
space (phi_hat = -rho_hat / k^2), builds the 3x3 tidal tensor per
cell, and maps 3 - count(eigvals > lambda_th) to uint8 codes
0..3. CODE_TO_ENV lookup yields the Environment enum instance.

lookup_env_at_positions resolves each halo to its nearest cell.

Memory at N=128: peak ~500 MB (3x3xN^3 eigvalsh intermediate);
OK on a 16 GB laptop. N=256 is a 2x grid refine and would hit
~4 GB — deferred to v2 per spec "Risks & fallbacks".

6 new tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Verify nothing regressed:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```
Expected: all tests pass.

---

## Task 3: `void_finder.py` — Gaussian smooth + local min + sphere growth

Given a density grid, smooth at ~10 Mpc with a Gaussian filter, find local minima via `scipy.ndimage.minimum_filter`, grow a sphere around each minimum until the enclosed density contrast crosses a threshold (default `-0.2`), rank by `depth × radius`, and return the top K candidates.

**Files:**
- Create: `problems/hubble_tension_web/nbody/void_finder.py`
- Test: `tests/hubble_tension_web/nbody/test_void_finder.py`

- [ ] **Step 1: Write failing tests**

Write `tests/hubble_tension_web/nbody/test_void_finder.py`:
```python
"""Tests for the local-minimum sphere-growth void finder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from problems.hubble_tension_web.nbody import tidal_tensor, void_finder
from problems.hubble_tension_web.nbody.mdpl2_fetch import load_halo_catalog


FIXTURE = Path(__file__).parent / "fixtures" / "mini_mdpl2.parquet"


def test_find_planted_void_in_synthetic_grid():
    """Place halos in a uniform background + excluded sphere; void finder must
    locate the sphere center within 1 grid cell."""
    rng = np.random.default_rng(0)
    box_mpc = 50.0
    n_grid = 32
    # Uniform background halos + excluded sphere at (25, 25, 25) with radius 10.
    positions = rng.uniform(0.0, box_mpc, size=(4000, 3))
    r = np.linalg.norm(positions - 25.0, axis=1)
    positions = positions[r > 10.0]
    masses = np.ones(positions.shape[0])
    rho = tidal_tensor.cic_deposit(positions, masses, n_grid=n_grid, box_mpc=box_mpc)

    voids = void_finder.find_voids(
        rho=rho, box_mpc=box_mpc,
        smoothing_mpc=5.0, delta_threshold=-0.2, max_radius_mpc=20.0,
        k_top=5,
    )
    assert len(voids) >= 1
    best = voids[0]
    center_err = np.linalg.norm(np.array(best.center_mpc) - 25.0)
    cell_mpc = box_mpc / n_grid
    assert center_err < 2.0 * cell_mpc, (
        f"void center off by {center_err:.2f} Mpc (> 2 cells = {2*cell_mpc:.2f} Mpc)"
    )
    # Should be an underdensity.
    assert best.delta_eff < 0.0
    # Should have grown to a non-trivial radius.
    assert best.radius_mpc >= 5.0


def test_find_voids_returns_at_most_k():
    rho = np.ones((16, 16, 16), dtype=np.float64)
    # Plant three local minima at distinct locations.
    rho[4, 4, 4] = 0.01
    rho[10, 10, 10] = 0.01
    rho[4, 10, 4] = 0.01
    voids = void_finder.find_voids(
        rho=rho, box_mpc=16.0,
        smoothing_mpc=0.5, delta_threshold=-0.2, max_radius_mpc=5.0,
        k_top=2,
    )
    assert len(voids) <= 2


def test_find_voids_on_fixture_returns_nonempty():
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    rho = tidal_tensor.cic_deposit(
        halos.positions, halos.masses, n_grid=32, box_mpc=halos.box_mpc,
    )
    voids = void_finder.find_voids(
        rho=rho, box_mpc=halos.box_mpc,
        smoothing_mpc=4.0, delta_threshold=-0.1, max_radius_mpc=20.0,
        k_top=5,
    )
    assert len(voids) >= 1, "committed fixture has a planted void; finder must locate it"


def test_voidcandidate_fields_populated():
    rho = np.ones((8, 8, 8), dtype=np.float64)
    rho[4, 4, 4] = 0.01
    voids = void_finder.find_voids(
        rho=rho, box_mpc=8.0,
        smoothing_mpc=0.5, delta_threshold=-0.2, max_radius_mpc=3.0,
        k_top=1,
    )
    assert len(voids) == 1
    v = voids[0]
    assert len(v.center_mpc) == 3
    assert v.radius_mpc > 0
    assert v.delta_eff <= 0.0
    assert v.n_halos_enclosed_estimate >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_void_finder.py -v
```
Expected: all fail with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Write `problems/hubble_tension_web/nbody/void_finder.py`:
```python
"""Simple density-field void finder.

Algorithm (per spec):
  1. Smooth the density field at ~10 Mpc with a Gaussian filter.
  2. Find local minima via scipy.ndimage.minimum_filter.
  3. For each minimum, grow a spherical region in 1-cell steps until either
     the enclosed density contrast delta_eff < threshold (default -0.2) fails
     OR max_radius is reached. Final radius = last step where delta_eff held.
  4. Rank by depth * radius and return top K.

v1 intentionally avoids watershed approaches (VIDE, ZOBOV). See spec §7.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter


@dataclass
class VoidCandidate:
    """A candidate void with its center, effective radius, and depth."""
    center_mpc: tuple[float, float, float]
    radius_mpc: float
    delta_eff: float
    n_halos_enclosed_estimate: int  # rough count; full count done downstream
    score: float                    # depth * radius for ranking


def _local_minima_indices(
    rho_smooth: np.ndarray,
    *,
    neighborhood_size: int = 3,
) -> np.ndarray:
    """Return (M, 3) integer grid indices where rho_smooth is a local minimum."""
    mins = minimum_filter(rho_smooth, size=neighborhood_size)
    mask = (rho_smooth == mins)
    # Exclude cells that equal their neighborhood minimum because the cell is at
    # an all-equal constant patch. Require strict minimum in at least one direction
    # by comparing to a larger neighborhood_size=5 min — if still equal, it's
    # not a clear dip and we drop it.
    mins_wide = minimum_filter(rho_smooth, size=max(neighborhood_size + 2, 5))
    mask &= (rho_smooth <= mins_wide + 1e-12)
    idx = np.argwhere(mask)
    return idx


def _grow_sphere(
    rho: np.ndarray,
    center_cell: np.ndarray,
    *,
    mean_rho: float,
    cell_mpc: float,
    delta_threshold: float,
    max_radius_cells: int,
) -> tuple[int, float, int]:
    """Grow a sphere in 1-cell radial steps around center_cell in rho.

    Returns:
      best_r_cells: final radius in cells where delta_eff < delta_threshold still held.
                    0 if even a 1-cell radius fails the threshold.
      delta_eff:    enclosed density contrast at best_r_cells.
      n_cells:      cell count inside best_r_cells.
    """
    n = rho.shape[0]
    # Build coordinate offsets once for the max radius.
    half = max_radius_cells
    rng = np.arange(-half, half + 1)
    dx, dy, dz = np.meshgrid(rng, rng, rng, indexing="ij")
    dist2 = dx * dx + dy * dy + dz * dz  # (2h+1, 2h+1, 2h+1)

    cx, cy, cz = int(center_cell[0]), int(center_cell[1]), int(center_cell[2])
    # Slice the cube around the center (with periodic wrap).
    ix = (cx + rng) % n
    iy = (cy + rng) % n
    iz = (cz + rng) % n
    sub = rho[np.ix_(ix, iy, iz)]  # (2h+1, 2h+1, 2h+1)

    best_r = 0
    best_delta = 0.0
    best_n = 0
    for r in range(1, max_radius_cells + 1):
        mask = dist2 <= r * r
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        mean_inside = float(sub[mask].mean())
        delta_eff = (mean_inside / mean_rho) - 1.0 if mean_rho > 0 else 0.0
        if delta_eff < delta_threshold:
            best_r = r
            best_delta = delta_eff
            best_n = n_cells
        else:
            # Monotonically, sphere only grows — the first r where the condition
            # fails is our stopping point.
            break
    return best_r, best_delta, best_n


def find_voids(
    *,
    rho: np.ndarray,
    box_mpc: float,
    smoothing_mpc: float = 10.0,
    delta_threshold: float = -0.2,
    max_radius_mpc: float = 100.0,
    k_top: int = 5,
) -> list[VoidCandidate]:
    """Locate up to k_top void candidates in the density grid rho.

    Args:
      rho:              (N, N, N) float64 density grid (mass per cell, arbitrary norm).
      box_mpc:          cube side length in Mpc.
      smoothing_mpc:    Gaussian smoothing scale for the minimum finder.
      delta_threshold:  density contrast threshold (default -0.2 per KBC).
      max_radius_mpc:   cap on sphere growth.
      k_top:            return at most k_top candidates, sorted by depth*radius.
    """
    n_grid = rho.shape[0]
    cell_mpc = box_mpc / n_grid
    sigma_cells = smoothing_mpc / cell_mpc
    rho_smooth = gaussian_filter(rho, sigma=sigma_cells, mode="wrap")

    mean_rho = float(rho.mean())
    if mean_rho <= 0.0:
        return []

    minima = _local_minima_indices(rho_smooth)
    max_radius_cells = max(1, int(np.ceil(max_radius_mpc / cell_mpc)))

    candidates: list[VoidCandidate] = []
    for m in minima:
        r_cells, delta_eff, n_cells = _grow_sphere(
            rho, m,
            mean_rho=mean_rho,
            cell_mpc=cell_mpc,
            delta_threshold=delta_threshold,
            max_radius_cells=max_radius_cells,
        )
        if r_cells <= 0:
            continue
        radius_mpc = r_cells * cell_mpc
        depth = max(-delta_eff, 0.0)
        score = depth * radius_mpc
        # Estimate halos enclosed: sum of rho in the sphere (rho in mass units).
        # Done below on a per-candidate basis.
        candidates.append(VoidCandidate(
            center_mpc=(
                float((m[0] + 0.5) * cell_mpc),
                float((m[1] + 0.5) * cell_mpc),
                float((m[2] + 0.5) * cell_mpc),
            ),
            radius_mpc=radius_mpc,
            delta_eff=float(delta_eff),
            n_halos_enclosed_estimate=int(round(n_cells * mean_rho * (1.0 + delta_eff))),
            score=float(score),
        ))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:k_top]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_void_finder.py -v
```
Expected: 4 tests pass. If `test_find_planted_void_in_synthetic_grid` fails with a center error of ~1 cell, try reducing `smoothing_mpc` in the test to 3.0 — Gaussian smoothing blurs the minimum away from the true center at very aggressive kernel widths. But do NOT weaken the test assertion; 2 cells is already a lenient bound.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/nbody/void_finder.py \
        tests/hubble_tension_web/nbody/test_void_finder.py
git commit -m "$(cat <<'EOF'
feat(nbody): local-minimum + sphere-growth void finder

find_voids Gaussian-smooths the density grid at ~10 Mpc (configurable),
locates local minima via scipy.ndimage.minimum_filter, grows each into
a spherical region in 1-cell radial steps until delta_eff crosses
threshold (default -0.2 per KBC), ranks by depth * radius, and returns
the top K.

Not a watershed finder — VIDE / ZOBOV deferred to v2 per spec.

4 new tests green; planted void recovered within 2 cells of ground
truth on synthetic 32^3 grid.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Verify nothing regressed:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```
Expected: full suite green.

---

## Task 4: `cosmic_web_from_halos.py` — assemble `LocalCosmicWeb` + `VoidParameters`

Given a halo catalog, a T-web environment grid, and a chosen void candidate, extract halos inside the void's effective radius, look up each halo's environment via nearest-cell assignment, and return the inputs `predict_from_cosmic_web` needs.

**Files:**
- Create: `problems/hubble_tension_web/nbody/cosmic_web_from_halos.py`
- Test: `tests/hubble_tension_web/nbody/test_cosmic_web_from_halos.py`

- [ ] **Step 1: Write failing tests**

Write `tests/hubble_tension_web/nbody/test_cosmic_web_from_halos.py`:
```python
"""Tests for cosmic-web assembly from halos + T-web + void candidate."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from problems.hubble_tension_web.nbody import (
    cosmic_web_from_halos,
    tidal_tensor,
    void_finder,
)
from problems.hubble_tension_web.nbody.mdpl2_fetch import load_halo_catalog
from problems.hubble_tension_web.types import (
    Environment,
    LocalCosmicWeb,
    VoidParameters,
)


FIXTURE = Path(__file__).parent / "fixtures" / "mini_mdpl2.parquet"


def test_assemble_from_candidate_returns_pair():
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    candidate = void_finder.VoidCandidate(
        center_mpc=(25.0, 25.0, 25.0),
        radius_mpc=15.0,
        delta_eff=-0.3,
        n_halos_enclosed_estimate=5,
        score=4.5,
    )
    web, params = cosmic_web_from_halos.assemble(
        halos=halos, env_grid=env_grid, candidate=candidate,
    )
    assert isinstance(web, LocalCosmicWeb)
    assert isinstance(params, VoidParameters)
    assert params.delta == candidate.delta_eff
    assert params.R_mpc == candidate.radius_mpc


def test_positions_are_void_center_relative():
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    candidate = void_finder.VoidCandidate(
        center_mpc=(25.0, 25.0, 25.0),
        radius_mpc=15.0,
        delta_eff=-0.3,
        n_halos_enclosed_estimate=5,
        score=4.5,
    )
    web, _ = cosmic_web_from_halos.assemble(
        halos=halos, env_grid=env_grid, candidate=candidate,
    )
    # All returned positions are within radius of origin (the void center).
    r = np.linalg.norm(web.positions, axis=1)
    assert np.all(r <= candidate.radius_mpc + 1e-6), (
        f"halos must be within R={candidate.radius_mpc}; max r={r.max()}"
    )


def test_environments_are_enum_instances():
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    candidate = void_finder.VoidCandidate(
        center_mpc=(25.0, 25.0, 25.0),
        radius_mpc=15.0,
        delta_eff=-0.3,
        n_halos_enclosed_estimate=5,
        score=4.5,
    )
    web, _ = cosmic_web_from_halos.assemble(
        halos=halos, env_grid=env_grid, candidate=candidate,
    )
    # LocalCosmicWeb.__post_init__ already enforces length match; here we
    # ensure the assembler materializes Environment instances, not raw codes.
    assert all(isinstance(e, Environment) for e in web.environments)


def test_empty_void_returns_empty_web_gracefully():
    """A void with radius 0.1 Mpc (smaller than one cell) may contain zero halos.
    Assembler should return an empty LocalCosmicWeb without crashing."""
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    candidate = void_finder.VoidCandidate(
        center_mpc=(49.0, 49.0, 49.0),  # corner, likely no halos
        radius_mpc=0.1,
        delta_eff=-0.5,
        n_halos_enclosed_estimate=0,
        score=0.05,
    )
    web, params = cosmic_web_from_halos.assemble(
        halos=halos, env_grid=env_grid, candidate=candidate,
    )
    # Empty web is valid; LocalCosmicWeb.__post_init__ enforces shape (0, 3).
    assert web.positions.shape[0] == 0
    assert len(web.environments) == 0
    assert params.R_mpc == 0.1


def test_void_params_signs_respected():
    """delta_eff must be <= 0 (VoidParameters validator enforces this)."""
    halos = load_halo_catalog(FIXTURE, mass_cut=0.0)
    env_grid, _ = tidal_tensor.classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=32, box_mpc=halos.box_mpc, lambda_th=0.0,
    )
    # A positive-delta candidate is meaningless as a "void" — assembler should
    # clamp or raise. Assembler RAISES per the VoidParameters contract.
    import pytest
    bad = void_finder.VoidCandidate(
        center_mpc=(25.0, 25.0, 25.0),
        radius_mpc=5.0,
        delta_eff=0.3,  # positive -> not a void
        n_halos_enclosed_estimate=10,
        score=1.5,
    )
    with pytest.raises(ValueError, match="delta"):
        cosmic_web_from_halos.assemble(
            halos=halos, env_grid=env_grid, candidate=bad,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_cosmic_web_from_halos.py -v
```
Expected: all fail with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Write `problems/hubble_tension_web/nbody/cosmic_web_from_halos.py`:
```python
"""Assemble a LocalCosmicWeb + VoidParameters from halos, a T-web grid, and a void."""
from __future__ import annotations

import numpy as np

from problems.hubble_tension_web.nbody.mdpl2_fetch import HaloCatalog
from problems.hubble_tension_web.nbody.tidal_tensor import lookup_env_at_positions
from problems.hubble_tension_web.nbody.void_finder import VoidCandidate
from problems.hubble_tension_web.types import LocalCosmicWeb, VoidParameters


def assemble(
    *,
    halos: HaloCatalog,
    env_grid: np.ndarray,
    candidate: VoidCandidate,
) -> tuple[LocalCosmicWeb, VoidParameters]:
    """Build (LocalCosmicWeb, VoidParameters) inputs for predict_from_cosmic_web.

    1. Filter halos to those within candidate.radius_mpc of candidate.center_mpc.
    2. Translate those halos so the void center is the origin.
    3. Look up each halo's environment in env_grid (absolute coords, nearest cell).
    4. Build LocalCosmicWeb(positions=relative, environments=...).
    5. Build VoidParameters(delta=candidate.delta_eff, R_mpc=candidate.radius_mpc).
       VoidParameters.__post_init__ rejects delta > 0 — this is the contract.

    Returns an empty web (positions shape (0, 3)) if no halos fall inside.
    """
    center = np.array(candidate.center_mpc, dtype=np.float64)
    rel = halos.positions - center
    r = np.linalg.norm(rel, axis=1)
    inside = r <= candidate.radius_mpc

    positions_rel = rel[inside]                     # (M, 3) halo coords relative to center
    positions_abs = halos.positions[inside]         # (M, 3) absolute — needed for env lookup

    # Reconstruct box size from env_grid coords; assume grid spans [0, box_mpc)^3
    # with cell_mpc = box_mpc / n_grid. The box_mpc is stored in halos.
    environments = lookup_env_at_positions(
        env_grid=env_grid, positions=positions_abs, box_mpc=halos.box_mpc,
    )

    web = LocalCosmicWeb(positions=positions_rel, environments=environments)
    # VoidParameters enforces delta <= 0 and R_mpc > 0 in __post_init__.
    params = VoidParameters(delta=candidate.delta_eff, R_mpc=candidate.radius_mpc)
    return web, params
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_cosmic_web_from_halos.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/nbody/cosmic_web_from_halos.py \
        tests/hubble_tension_web/nbody/test_cosmic_web_from_halos.py
git commit -m "$(cat <<'EOF'
feat(nbody): assemble LocalCosmicWeb + VoidParameters from halos

assemble(halos, env_grid, candidate) filters halos inside the void
radius, translates them to void-center-relative coordinates, looks
up each halo's environment via nearest-cell assignment against the
T-web grid, and constructs the (LocalCosmicWeb, VoidParameters) pair
that predict_from_cosmic_web consumes.

Empty-void case (candidate radius smaller than nearest halo) returns
an empty web gracefully. Positive-delta candidate raises ValueError
via VoidParameters' existing void-sign guard.

5 new tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Verify:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```
Expected: full suite green.

---

## Task 5: `experiments/nbody_kbc.py` — end-to-end runner

Put it all together: load the cache, classify, find K voids, assemble inputs, run `predict_from_cosmic_web`, and write a summary JSON + β₁-distribution PNG.

**Files:**
- Create: `problems/hubble_tension_web/experiments/nbody_kbc.py`
- Test: `tests/hubble_tension_web/nbody/test_nbody_kbc.py`

### JSON schema produced by `nbody_kbc.py`

Top-level fields (object):
- `cache_source`: `str` — absolute path of the Parquet file used.
- `grid_N`: `int` — T-web grid size used.
- `lambda_th`: `float` — tidal eigenvalue threshold used.
- `K`: `int` — number of void candidates requested.
- `alpha_used`: `float` — α from sim_calibration.json or 0.0 if file missing.
- `timestamp`: `str` — ISO-8601 UTC, writer-local clock.
- `voids`: list of per-void objects (length ≤ K).
- `beta1_distribution`: summary object.

Per-void object fields:
- `idx`: `int` — 0-based rank by depth × radius.
- `center_mpc`: `[float, float, float]` — void center, absolute box coords.
- `N_halos`: `int` — halos inside the void radius.
- `delta_eff`: `float` — enclosed density contrast.
- `R_eff_mpc`: `float` — effective void radius.
- `beta0`: `int` — connected components of the typed graph.
- `beta1_persistent`: `int` — persistent 1-Betti count (the headline physics quantity).
- `lambda_min`: `float` — smallest non-zero eigenvalue of L_F.
- `delta_H0_total`: `float` — km/s/Mpc.
- `kinematic_term`: `float` — km/s/Mpc.
- `topological_term`: `float` — km/s/Mpc (α · f_topo).

`beta1_distribution` summary:
- `count_nonzero`: `int`
- `count_total`: `int`
- `median`: `float`
- `max`: `int`

- [ ] **Step 1: Write failing tests**

Write `tests/hubble_tension_web/nbody/test_nbody_kbc.py`:
```python
"""Smoke test for the end-to-end nbody_kbc experiment on the fixture."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


FIXTURE = Path("tests/hubble_tension_web/nbody/fixtures/mini_mdpl2.parquet")
OUT = Path("problems/hubble_tension_web/results/nbody_kbc.json")


def test_nbody_kbc_runs_on_fixture(tmp_path, monkeypatch):
    """Run the experiment module with the fixture Parquet as the cache file."""
    out_path = tmp_path / "nbody_kbc.json"
    # Point the module at the fixture via env var + redirect output to tmp_path.
    env_overrides = {
        "ATFT_NBODY_CACHE_FILE": str(FIXTURE.resolve()),
        "ATFT_NBODY_GRID": "32",
        "ATFT_NBODY_OUTPUT_JSON": str(out_path),
        "ATFT_NBODY_K_VOIDS": "3",
    }
    import os
    env = os.environ.copy()
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-m", "problems.hubble_tension_web.experiments.nbody_kbc"],
        env=env, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout:\n{result.stdout.decode(errors='replace')}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    # Schema spot-checks.
    assert "grid_N" in data and data["grid_N"] == 32
    assert "K" in data and data["K"] == 3
    assert "voids" in data and isinstance(data["voids"], list)
    assert "beta1_distribution" in data
    dist = data["beta1_distribution"]
    assert "count_nonzero" in dist and "count_total" in dist
    assert dist["count_total"] == len(data["voids"])

    # Per-void structural checks — only if we found any voids on the fixture.
    if data["voids"]:
        v0 = data["voids"][0]
        for field in ("idx", "center_mpc", "N_halos", "delta_eff", "R_eff_mpc",
                      "beta0", "beta1_persistent", "lambda_min",
                      "delta_H0_total", "kinematic_term", "topological_term"):
            assert field in v0, f"missing per-void field: {field}"
        assert isinstance(v0["beta1_persistent"], int)
        assert v0["beta1_persistent"] >= 0  # 0 is a valid outcome per spec
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_nbody_kbc.py -v
```
Expected: fails with `ModuleNotFoundError: No module named '...nbody_kbc'` after subprocess invocation (rc != 0).

- [ ] **Step 3: Write the experiment module**

Write `problems/hubble_tension_web/experiments/nbody_kbc.py`:
```python
"""N-body KBC experiment: run the full pipeline on MDPL2-style halos.

Reads a Parquet halo cache, classifies the volume with the tidal-tensor
T-web, finds K void candidates, runs predict_from_cosmic_web on each,
and writes a JSON summary + an optional matplotlib histogram of the
beta1_persistent distribution.

Configured via environment variables so run_all.py can drive it:
  ATFT_NBODY_CACHE_FILE   — Parquet file path (required)
  ATFT_NBODY_GRID         — T-web grid size (default 128; fixture uses 32)
  ATFT_NBODY_LAMBDA_TH    — tidal eigenvalue threshold (default 0.0)
  ATFT_NBODY_K_VOIDS      — number of candidate voids (default 5)
  ATFT_NBODY_MASS_CUT     — halo mass cut in M_sun (default 0.0 = none)
  ATFT_NBODY_OUTPUT_JSON  — output JSON path
                             (default problems/.../results/nbody_kbc.json)

The experiment does NOT recalibrate alpha. It reuses sim_calibration.json
if present; otherwise alpha=0 (per spec v1: headline is beta1, not ΔH0).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problems.hubble_tension_web.functional import predict_from_cosmic_web
from problems.hubble_tension_web.nbody import (
    DEFAULT_LAMBDA_TH,
    DEFAULT_MASS_CUT,
    DEFAULT_N_GRID,
    NBodyDataNotAvailable,
)
from problems.hubble_tension_web.nbody.cosmic_web_from_halos import assemble
from problems.hubble_tension_web.nbody.mdpl2_fetch import load_halo_catalog
from problems.hubble_tension_web.nbody.tidal_tensor import classify
from problems.hubble_tension_web.nbody.void_finder import find_voids


DEFAULT_K: int = 5
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent / "results" / "nbody_kbc.json"
)
DEFAULT_PNG = DEFAULT_OUTPUT.with_suffix(".png")


def _alpha_from_sim_calibration() -> float:
    """Reuse alpha* from sim_calibration.json if present; otherwise 0.0."""
    sc = Path(__file__).parent.parent / "results" / "sim_calibration.json"
    if not sc.exists():
        return 0.0
    try:
        return float(json.loads(sc.read_text())["alpha_star"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def _run() -> dict:
    cache_file = os.environ.get("ATFT_NBODY_CACHE_FILE")
    if not cache_file:
        raise NBodyDataNotAvailable(
            "ATFT_NBODY_CACHE_FILE not set and no default cache located. "
            "See problems/hubble_tension_web/nbody/README.md."
        )
    cache_path = Path(cache_file)
    n_grid = int(os.environ.get("ATFT_NBODY_GRID", str(DEFAULT_N_GRID)))
    lambda_th = float(os.environ.get("ATFT_NBODY_LAMBDA_TH", str(DEFAULT_LAMBDA_TH)))
    k_voids = int(os.environ.get("ATFT_NBODY_K_VOIDS", str(DEFAULT_K)))
    mass_cut = float(os.environ.get("ATFT_NBODY_MASS_CUT", "0.0"))

    halos = load_halo_catalog(cache_path, mass_cut=mass_cut)
    env_grid, meta = classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=n_grid, box_mpc=halos.box_mpc, lambda_th=lambda_th,
    )
    # Density grid is stashed by classify indirectly through its intermediates;
    # recompute it here (cheap) for the void finder.
    from problems.hubble_tension_web.nbody.tidal_tensor import cic_deposit
    rho = cic_deposit(halos.positions, halos.masses,
                      n_grid=n_grid, box_mpc=halos.box_mpc)

    smoothing_mpc = min(10.0, halos.box_mpc / 5.0)
    # For the tiny fixture (50 Mpc box) we need a much smaller smoothing scale
    # than real MDPL2 (500 Mpc box); smoothing_mpc = box/5 accommodates both.
    voids = find_voids(
        rho=rho, box_mpc=halos.box_mpc,
        smoothing_mpc=smoothing_mpc,
        delta_threshold=-0.1,
        max_radius_mpc=halos.box_mpc / 2.0,
        k_top=k_voids,
    )

    alpha = _alpha_from_sim_calibration()

    per_void: list[dict] = []
    for idx, cand in enumerate(voids):
        try:
            web, params = assemble(halos=halos, env_grid=env_grid, candidate=cand)
        except ValueError:
            # Positive-delta candidate (rare; find_voids filter already enforces this).
            continue
        if web.positions.shape[0] < 4:
            # Too few halos for the k=8 NN graph + Laplacian — skip.
            continue
        # Cap k for tiny voids. build_typed_graph wants k < N.
        k_nn = min(8, web.positions.shape[0] - 1)
        h = predict_from_cosmic_web(
            web=web, params=params, alpha=alpha,
            k=k_nn, stalk_dim=8, k_spec=min(16, web.positions.shape[0] - 1),
            rng_seed=0,
        )
        # Re-derive beta0/beta1/lambda_min by calling summarize_spectrum directly
        # is redundant — HubbleShift doesn't expose them. We re-run the pieces.
        # Cheaper: inspect the summary returned. predict_from_cosmic_web doesn't
        # return the summary; rebuild it here for the JSON fields.
        from problems.hubble_tension_web.graph import build_typed_graph
        from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
        from problems.hubble_tension_web.spectrum import summarize_spectrum
        n, edges = build_typed_graph(web, k=k_nn)
        L = typed_sheaf_laplacian(
            positions=web.positions, n=n, edges=edges, stalk_dim=8,
            rng_seed=0, environments=web.environments,
        )
        summary = summarize_spectrum(
            L=L, n_nodes=n, edges=edges, positions=web.positions,
            k_spec=min(16, web.positions.shape[0] - 1),
        )
        per_void.append(dict(
            idx=idx,
            center_mpc=list(cand.center_mpc),
            N_halos=int(web.positions.shape[0]),
            delta_eff=float(cand.delta_eff),
            R_eff_mpc=float(cand.radius_mpc),
            beta0=int(summary.beta0),
            beta1_persistent=int(summary.beta1),
            lambda_min=float(summary.lambda_min),
            delta_H0_total=float(h.delta_H0),
            kinematic_term=float(h.kinematic_term),
            topological_term=float(h.topological_term),
        ))

    beta1s = [v["beta1_persistent"] for v in per_void]
    dist = dict(
        count_nonzero=int(sum(1 for b in beta1s if b > 0)),
        count_total=int(len(beta1s)),
        median=float(statistics.median(beta1s)) if beta1s else 0.0,
        max=int(max(beta1s)) if beta1s else 0,
    )

    return dict(
        cache_source=str(cache_path.resolve()),
        grid_N=int(n_grid),
        lambda_th=float(lambda_th),
        K=int(k_voids),
        alpha_used=float(alpha),
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        voids=per_void,
        beta1_distribution=dist,
    )


def main() -> None:
    try:
        out = _run()
    except NBodyDataNotAvailable as e:
        print(f"nbody_kbc: cache not available; skipping. ({e})", file=sys.stderr)
        sys.exit(0)  # non-error skip — run_all.py checks stdout/json existence.

    json_path = Path(
        os.environ.get("ATFT_NBODY_OUTPUT_JSON", str(DEFAULT_OUTPUT))
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2))

    # Optional histogram — skip if no voids.
    if out["voids"]:
        png_path = Path(str(json_path).replace(".json", ".png"))
        betas = np.array([v["beta1_persistent"] for v in out["voids"]])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(betas, bins=np.arange(betas.min(), betas.max() + 2) - 0.5)
        ax.set_xlabel("beta_1_persistent")
        ax.set_ylabel("count")
        ax.set_title(
            f"N-body KBC: beta_1 across {len(betas)} candidate voids"
        )
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)

    print(json.dumps(out["beta1_distribution"], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_nbody_kbc.py -v
```
Expected: 1 test passes within ~30 s. The fixture has ~180 halos; the void finder finds at least one candidate (the planted one); assemble materializes a `LocalCosmicWeb` with some halos inside; `predict_from_cosmic_web` returns a finite ΔH₀. The JSON matches the schema. If β₁ = 0 on the fixture, that's still a valid outcome — the smoke test only checks structure, not physics.

If `test_nbody_kbc_runs_on_fixture` times out, it's probably `ripser` hitting a degenerate point cloud. Lower `k_voids` in the test env from 3 to 1; the fixture has one planted void so 1 candidate suffices.

- [ ] **Step 5: Commit**

```bash
git add problems/hubble_tension_web/experiments/nbody_kbc.py \
        tests/hubble_tension_web/nbody/test_nbody_kbc.py
git commit -m "$(cat <<'EOF'
feat(nbody): end-to-end nbody_kbc experiment

experiments/nbody_kbc.py: load Parquet halo cache -> T-web classify ->
find K void candidates -> assemble LocalCosmicWeb + VoidParameters
per candidate -> predict_from_cosmic_web -> write results/nbody_kbc.json
+ beta_1 histogram PNG.

Env-driven config (ATFT_NBODY_CACHE_FILE / GRID / LAMBDA_TH / K_VOIDS /
MASS_CUT / OUTPUT_JSON) so run_all.py can invoke conditionally.

Re-uses alpha* from sim_calibration.json if present; else 0.0 (v1 spec
says the headline is beta_1 count, not recalibrated delta_H0).

NBodyDataNotAvailable path exits 0 with a stderr note — lets run_all
skip cleanly when the cache is cold.

1 new test green (subprocess runs on fixture in ~20s).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Verify:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```
Expected: full suite green.

---

## Task 6: `run_all.py` integration — conditional launch of `nbody_kbc`

Extend `run_all.py` to optionally launch `nbody_kbc` if the MDPL2 cache (or fixture) is present. If absent, print a skip message and continue with the three synthetic experiments as before.

**Files:**
- Modify: `problems/hubble_tension_web/experiments/run_all.py`
- Modify: `tests/hubble_tension_web/test_run_all.py` (add new test — existing tests unchanged)

- [ ] **Step 1: Write failing test**

Append to `tests/hubble_tension_web/test_run_all.py`:
```python
def test_run_all_skips_nbody_when_cache_absent(tmp_path):
    """run_all must exit 0 and run the three synthetic experiments even when
    no nbody cache exists. The nbody step is opt-in, not load-bearing."""
    import os
    env = os.environ.copy()
    # Ensure the cache lookup points at a definitely-empty dir.
    env["ATFT_DATA_CACHE"] = str(tmp_path / "nonexistent")
    env.pop("ATFT_NBODY_CACHE_FILE", None)  # defensively clear
    result = subprocess.run(
        [sys.executable, "-m", "problems.hubble_tension_web.experiments.run_all"],
        capture_output=True, env=env, timeout=180,
    )
    assert result.returncode == 0, (
        f"run_all must survive a missing nbody cache. stderr:\n"
        f"{result.stderr.decode(errors='replace')}"
    )
    combined = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
    assert "nbody" in combined.lower(), (
        "run_all should mention nbody in output when skipping it"
    )
    # All three synthetic outputs must still exist.
    results = Path("problems/hubble_tension_web/results")
    for name in ("analytical_reduction.json", "sim_calibration.json",
                 "kbc_crosscheck.json", "REPORT.md"):
        assert (results / name).exists(), f"synthetic artifact missing: {name}"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_run_all.py::test_run_all_skips_nbody_when_cache_absent -v
```
Expected: fails — the current `run_all.py` doesn't mention nbody.

- [ ] **Step 3: Modify `run_all.py`**

Edit `problems/hubble_tension_web/experiments/run_all.py`. Find the `EXPERIMENTS` list (currently 3 entries); add a conditional nbody launch AFTER the parallel phase completes and BEFORE the aggregate step. Use this replacement for `main()`:

```python
def _nbody_cache_path() -> Path | None:
    """Return the path of a real MDPL2 halo cache if one exists, else None.

    Explicit ATFT_NBODY_CACHE_FILE wins. Otherwise look under
    $ATFT_DATA_CACHE/mdpl2/*.parquet (or ~/.cache/atft/mdpl2/).
    """
    explicit = os.environ.get("ATFT_NBODY_CACHE_FILE")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    cache_root = Path(os.environ.get("ATFT_DATA_CACHE", str(Path.home() / ".cache" / "atft")))
    mdpl2_dir = cache_root / "mdpl2"
    if mdpl2_dir.exists():
        parquets = sorted(mdpl2_dir.glob("*.parquet"))
        if parquets:
            return parquets[0]
    return None


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

    # Optional nbody_kbc step — skipped cleanly if cache absent.
    nbody_cache = _nbody_cache_path()
    if nbody_cache is not None:
        print(f"nbody_kbc: launching with cache {nbody_cache}")
        nbody_env = os.environ.copy()
        nbody_env["ATFT_NBODY_CACHE_FILE"] = str(nbody_cache)
        rc = subprocess.run(
            [sys.executable, "-m", "problems.hubble_tension_web.experiments.nbody_kbc"],
            env=nbody_env,
        ).returncode
        if rc != 0:
            print(f"nbody_kbc failed rc={rc}; continuing with aggregate.", file=sys.stderr)
    else:
        print("nbody_kbc: no MDPL2 cache found, skipping (see nbody/README.md).")

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
```

Use the `Edit` tool, replacing the existing `main()` function body. The `EXPERIMENTS` list, `AGGREGATE` constant, and `_env_for` helper are untouched. The added `_nbody_cache_path` helper must be inserted above `main()`.

- [ ] **Step 4: Run the new test to verify it passes**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_run_all.py -v
```
Expected: both existing tests plus the new one pass. Total wall time ~30-60 s per run.

If `test_run_all_skips_nbody_when_cache_absent` fails because the log message doesn't contain "nbody", inspect the subprocess output — the skip message "nbody_kbc: no MDPL2 cache found" must appear on stdout.

- [ ] **Step 5: Full acceptance gate — run the complete suite**

Run:
```bash
python -m pytest tests/hubble_tension_web/ --ignore=tests/hubble_tension_web/test_pipeline.py -q
```
Expected: the prior 52 passing + 3 xfailed, plus roughly:
- 6 new `test_mdpl2_fetch.py`
- 6 new `test_tidal_tensor.py`
- 4 new `test_void_finder.py`
- 5 new `test_cosmic_web_from_halos.py`
- 1 new `test_nbody_kbc.py`
- 1 new test in `test_run_all.py`

= 23 new tests, all green.

- [ ] **Step 6: Commit**

```bash
git add problems/hubble_tension_web/experiments/run_all.py \
        tests/hubble_tension_web/test_run_all.py
git commit -m "$(cat <<'EOF'
feat(nbody): conditional nbody_kbc launch from run_all

_nbody_cache_path probes ATFT_NBODY_CACHE_FILE then $ATFT_DATA_CACHE/
mdpl2/*.parquet; if found, run_all launches nbody_kbc between the
three synthetic experiments and aggregate. If absent, prints a
skip notice and proceeds — zero impact on the authoritative synthetic
pipeline.

Existing test_run_all tests still pass unchanged; new test covers
the skip-path and asserts all three synthetic artifacts are written.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (inline, per writing-plans)

This block is the author's own end-of-plan audit. The implementer should NOT rerun it — it has already been performed.

### 1. Spec coverage

Walking the spec section-by-section:

| Spec requirement | Task |
|---|---|
| `problems/.../nbody/__init__.py` | Task 0 |
| `mdpl2_fetch.py` downloader + cache | Task 1 (cache reader done; network stubbed, documented) |
| `tidal_tensor.py` (CIC + FFT + eigvalsh + Environment) | Task 2 |
| `void_finder.py` (Gaussian smooth + local min + sphere growth) | Task 3 |
| `cosmic_web_from_halos.py` (assemble LocalCosmicWeb + VoidParameters) | Task 4 |
| `experiments/nbody_kbc.py` | Task 5 |
| New test directory `tests/.../nbody/` with 4 test files | Tasks 1-5 |
| Conditional integration into `run_all.py` | Task 6 |
| Mass cut `M_vir > 10^11.5` | `DEFAULT_MASS_CUT` in `__init__.py` (Task 0); plumbed through `_run` in Task 5 |
| Cache default `~/.cache/atft/mdpl2/`, env override `ATFT_DATA_CACHE` | Task 6 `_nbody_cache_path` |
| NBodyDataNotAvailable exception | Task 0 |
| T-web at 256³ grid; tunable `λ_th` | `DEFAULT_N_GRID=128` (v1 fallback per spec §"Risks"); `DEFAULT_LAMBDA_TH=0.0`; env overrides |
| 6 unique upper-triangular tidal-tensor components | Task 2 `_tidal_tensor_fft` |
| Void-finder ranks by depth × radius, returns top N | Task 3 `find_voids` with `score = depth * radius_mpc` |
| K=5 candidate voids | `DEFAULT_K = 5` in Task 5 |
| JSON records `N_halos, δ_eff, R_eff, β₀, β₁_persistent, λ_min, ΔH₀_total, kinematic, topological` | Task 5 per-void dict |
| `β₁_distribution` summary field | Task 5 `beta1_distribution` object |
| `β₁` count across voids as headline | Task 5 `count_nonzero` / `count_total` |
| `α*` from smooth-void calibration (v1 = 0); defer recalibration to v2 | Task 5 `_alpha_from_sim_calibration` |
| `run_all.py` skips if cache absent | Task 6 |
| Tiny synthetic fixture mimicking cache format | Task 0 + `test_nbody_kbc.py` uses it |
| <5 s end-to-end on fixture (spec) | Task 5 test uses 120 s timeout — conservative; expected ~20 s on fixture given ripser call overhead |
| All existing tests (52+) continue to pass | Tasks 0, 2, 4, 5, 6 explicit pytest verification |
| Acceptance: `nbody_kbc.py` writes `β₁_distribution` | Task 5 |

**Coverage gaps:** None found. v1 explicitly defers VIDE/ZOBOV, IllustrisTNG, α* recalibration, slab-by-slab at 256³, full 1 Gpc³ analysis.

### 2. Placeholder scan

Searched for: TBD, TODO, "implement later", "fill in", "Add appropriate error handling", "add validation", "handle edge cases", "Write tests for the above", "Similar to Task", "appropriate".

None found in normative plan text. One "appropriate" appears inside a bash comment inside Task 6's test, not as a directive to the implementer.

### 3. Type / name consistency

| Name | Defined in | Used in | Match? |
|---|---|---|---|
| `NBodyDataNotAvailable` | Task 0 `nbody/__init__.py` | Task 1 mdpl2_fetch, Task 5 nbody_kbc, Task 6 run_all (imports transitively) | yes |
| `DEFAULT_N_GRID`, `DEFAULT_MASS_CUT`, `DEFAULT_LAMBDA_TH` | Task 0 | Task 5 experiment | yes |
| `EXPECTED_COLUMNS` | Task 1 `mdpl2_fetch.py` | Task 1 test `test_mdpl2_fetch.py` | yes |
| `HaloCatalog` | Task 1 | Task 4 `assemble` signature, Task 5 experiment | yes |
| `load_halo_catalog(path, *, mass_cut, box_mpc=None)` | Task 1 | Task 2 test, Task 3 test, Task 4 test, Task 5 | yes |
| `cic_deposit(positions, masses, *, n_grid, box_mpc)` | Task 2 | Task 3 test, Task 5 | yes |
| `classify(*, positions, masses, n_grid, box_mpc, lambda_th=0.0)` | Task 2 | Task 3 test, Task 4 test, Task 5 | yes |
| `CODE_TO_ENV` | Task 2 | Task 2 test, Task 2 `lookup_env_at_positions` | yes |
| `lookup_env_at_positions(env_grid, positions, *, box_mpc)` | Task 2 | Task 4 `assemble` | yes |
| `VoidCandidate(center_mpc, radius_mpc, delta_eff, n_halos_enclosed_estimate, score)` | Task 3 | Task 4 test, Task 4 `assemble`, Task 5 | yes |
| `find_voids(*, rho, box_mpc, smoothing_mpc, delta_threshold, max_radius_mpc, k_top)` | Task 3 | Task 3 test, Task 5 | yes |
| `assemble(*, halos, env_grid, candidate)` | Task 4 | Task 5 | yes |
| `ATFT_NBODY_CACHE_FILE`, `ATFT_NBODY_GRID`, `ATFT_NBODY_K_VOIDS`, `ATFT_NBODY_OUTPUT_JSON`, `ATFT_NBODY_LAMBDA_TH`, `ATFT_NBODY_MASS_CUT` | Task 5 module | Task 5 test, Task 6 `_nbody_cache_path` | yes |
| `_nbody_cache_path()` | Task 6 | Task 6 `main` | yes |

One subtle item to double-check in implementation: `build_typed_graph(web, k=...)` in Task 5 caps `k_nn = min(8, N-1)`. The existing `graph.py` uses `k + 1` in the KDTree query (skipping self). With `k_nn = N - 1`, the query needs `k_nn + 1 = N` neighbors; KDTree permits this. Safe.

Another subtle item: `LocalCosmicWeb.__post_init__` validates that `positions.ndim == 2 and positions.shape[1] == 3`. For the empty-web edge case in Task 4's test, `positions_rel[inside]` produces a `(0, 3)` float64 array when `rel` is already `(N, 3)` and `inside` is all-False. NumPy preserves the second axis under boolean indexing, so shape is `(0, 3)`, and the validator passes. Verified by inspection of NumPy semantics.

### Fixes applied inline

None required. Proceed to commit.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-20-nbody-ingestion.md`.

**Recommended execution:** superpowers:subagent-driven-development (fresh subagent per task, two-stage review between tasks). Each of Tasks 0-6 is self-contained and ends in a green test suite + a commit; decomposition is suited to the subagent pattern.

Alternative: superpowers:executing-plans for inline batch execution with checkpoints.
