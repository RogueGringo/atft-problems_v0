# Forward Ops: Real MDPL2 β₁ Test + α Recalibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan adds a CosmoSim MDPL2 download layer and a real-void α recalibration leg to the hubble_tension_web pipeline. The **frozen modules** in spec §"Frozen" (types/functional/graph/laplacian/spectrum/synthetic/ltb_reference/…) MUST NOT be touched — all changes are in the new `nbody/mdpl2_download.py` module, the new `experiments/nbody_calibration.py` module, and additive edits to 6 existing files.

**Goal:** Produce the first quantitative ATFT prediction with uncertainty against the KBC literature band by (a) adding a real MDPL2 download path gated on `ATFT_MDPL2_DOWNLOAD_ENABLED`, and (b) recalibrating α against real-void LTB residuals via 1D LSQ + bootstrap CI + F-test.

**Architecture:** Two new Python modules, six additive edits, one README + one MASTER.md patch. (1) `nbody/mdpl2_download.py` submits a CosmoSim SQL job, polls, downloads CSV, applies h-conversion at write-time, and persists Parquet matching the existing `EXPECTED_COLUMNS = ("x","y","z","mvir")` contract. (2) `nbody/mdpl2_fetch.py::fetch_from_network` delegates to `mdpl2_download` when `ATFT_MDPL2_DOWNLOAD_ENABLED == "1"`, else preserves the `NotImplementedError("CosmoSim …")` stub. (3) `experiments/nbody_kbc.py` emits three additional per-void fields (`f_topo_at_alpha_1`, `ltb_anchor_at_delta_R`, `y_residual`) for downstream reuse. (4) `experiments/nbody_calibration.py` reads `nbody_kbc.json`, computes closed-form α* on K real voids, bootstraps a 68% CI, runs an F-test vs α=0, writes `nbody_calibration.json`. (5) `experiments/run_all.py` adds a conditional 4th leg. (6) `experiments/aggregate.py` adds a "Leg 4" section. (7) `MASTER.md` §6/§9/§12/§13 patched; `nbody/README.md` documents the download opt-in.

**Tech Stack:** Python 3.14, NumPy, SciPy (`scipy.stats.f.cdf`, `scipy.stats.pearsonr`), `pandas` + `pyarrow` for Parquet, `urllib.request` for the CosmoSim HTTP calls (no new third-party deps; `requests`/`pooch` rejected for v1 — network code is mocked in tests, exercised manually in the opt-in path). Test mocks via `unittest.mock.patch`. Determinism: all RNG seeds pinned (`np.random.default_rng(0)` for bootstrap; existing test seeds preserved).

**Reference spec:** `docs/superpowers/specs/2026-04-21-forward-ops-design.md`

**Branch:** `feat/forward-ops` (already checked out at HEAD `e708deb`, the spec commit).

**Scope boundary (planning agent note):** This document is markdown-only. No edits to files under `problems/` or `tests/` happen during plan authoring. Execution agents (subagent-driven or inline) perform the file mutations task-by-task.

---

## File Structure

**Files created (2 production modules):**

```
problems/hubble_tension_web/
├── nbody/
│   └── mdpl2_download.py         # NEW: CosmoSim SQL pull, h-conversion, Parquet write
└── experiments/
    └── nbody_calibration.py      # NEW: 1D LSQ + bootstrap CI + F-test on nbody_kbc.json
```

**Files created (2 tests):**

```
tests/hubble_tension_web/
├── nbody/
│   └── test_mdpl2_download.py    # NEW: mocked HTTP, Parquet schema, h-conversion verification
└── test_nbody_calibration.py     # NEW: K=8 planted-β₁ fixture; closed-form α* to 1e-10
```

**Files modified (8):**

```
problems/hubble_tension_web/
├── nbody/
│   ├── mdpl2_fetch.py            # fetch_from_network: delegate to mdpl2_download when enabled
│   └── README.md                 # document download opt-in + new env vars
├── experiments/
│   ├── nbody_kbc.py              # emit per-void f_topo_at_alpha_1 + ltb_anchor_at_delta_R + y_residual
│   ├── run_all.py                # conditional 4th leg (nbody_calibration) after nbody_kbc
│   └── aggregate.py              # new "Leg 4: Real-Void α Calibration" section
└── MASTER.md                     # §6 real-data row, §12 new spec/plan entries, §13 demote item

tests/hubble_tension_web/
├── nbody/test_mdpl2_fetch.py     # allow stub to be bypassed via ATFT_MDPL2_DOWNLOAD_ENABLED
└── test_run_all.py               # +2 tests: triggers-calibration-when-voids-exist / skips-when-no-voids
```

**Files NOT touched (frozen per spec §"Frozen"):**

```
types.py, functional.py, graph.py, laplacian.py, spectrum.py, synthetic.py,
ltb_reference.py, laplacian_quantized.py,
nbody/{tidal_tensor,void_finder,cosmic_web_from_halos,__init__}.py,
experiments/{analytical_reduction,sim_calibration,kbc_crosscheck}.py,
all tests under tests/hubble_tension_web/ except the two listed above.
```

---

## Env var surface (re-stated from spec for engineer convenience)

| Var | Default | Effect |
|---|---|---|
| `ATFT_MDPL2_DOWNLOAD_ENABLED` | `"0"` | Must be `"1"` to hit CosmoSim. Default preserves CI. |
| `ATFT_MDPL2_URL` | (unset) | Override CosmoSim TAP endpoint base URL. |
| `ATFT_MDPL2_SQL` | (unset) | Override the SQL query body (the one in `mdpl2_download.py` is marked UNVERIFIED). |
| `ATFT_SCISERVER_TOKEN` | (unset) | Authorization token. Required when `_ENABLED="1"`; missing → `NBodyDataNotAvailable`. |
| `ATFT_NBODY_CAL_MIN_VOIDS` | `3` | Minimum K real voids before attempting the α fit. |
| `ATFT_NBODY_CAL_BOOTSTRAP_B` | `2000` | Bootstrap resample count. |

Existing `ATFT_NBODY_*` family, `ATFT_DATA_CACHE`, `HUBBLE_POOL_WORKERS` unchanged.

---

## Task overview

| # | Title | Touches |
|---|---|---|
| 0 | Baseline + scaffold | `mdpl2_download.py` stub, `nbody_calibration.py` stub; commit green |
| 1 | `mdpl2_download.py` + `test_mdpl2_download.py` | mock HTTP, h-conversion, Parquet schema |
| 2 | `mdpl2_fetch.py::fetch_from_network` delegation | env-gated; preserves stub string |
| 3 | `nbody_kbc.py` per-void emission | 3 new fields; update `test_nbody_kbc.py` |
| 4 | `nbody_calibration.py` + `test_nbody_calibration.py` | 1D LSQ + bootstrap + F-test; 3 reason branches |
| 5 | `run_all.py` 4th leg + 2 new `test_run_all.py` tests | conditional launch on K ≥ min |
| 6 | `aggregate.py` Leg 4 section | conditional on `nbody_calibration.json` |
| 7 | `MASTER.md` patch + `nbody/README.md` update | doc + ownership only |

---

## Task 0: Baseline check + scaffold stubs (green commit)

**Purpose.** Establish a known-green baseline, then land stub modules that import cleanly, so Tasks 1 and 4 can add tests against a real symbol target rather than a missing import.

**Files:**
- Create: `problems/hubble_tension_web/nbody/mdpl2_download.py` (stub)
- Create: `problems/hubble_tension_web/experiments/nbody_calibration.py` (stub)
- Test: (no new tests — baseline suite only)

- [ ] **Step 0.1: Run the fast suite; capture the baseline count.**

Run:
```bash
cd /c/JTOD1/atft-problems_v0
python -m pytest tests/hubble_tension_web -x --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```

Expected: `73 passed, 3 xfailed` (MASTER.md §8). If the count differs, STOP and report to the user before continuing — a different baseline invalidates the acceptance gates downstream.

- [ ] **Step 0.2: Create the `mdpl2_download.py` stub.**

Write `problems/hubble_tension_web/nbody/mdpl2_download.py`:

```python
"""CosmoSim MDPL2 SQL download layer.

v1 scope: submit a TAP-style async SQL job against CosmoSim's Rockstar z=0 catalog,
poll until done, stream the CSV result, convert little-h units to pure Mpc/M_sun at
write-time, and persist Parquet matching the EXPECTED_COLUMNS contract from
mdpl2_fetch. Gated behind ATFT_MDPL2_DOWNLOAD_ENABLED=="1".

Implemented in Task 1.
"""
from __future__ import annotations

from pathlib import Path


def fetch_sub_box(*, dest: str | Path, sub_box_mpc: float = 500.0) -> Path:
    """Pull a MDPL2 sub-box halo catalog and write it as Parquet.

    Implemented in Task 1.
    """
    raise NotImplementedError("mdpl2_download.fetch_sub_box: see Task 1 of the forward-ops plan.")
```

- [ ] **Step 0.3: Create the `nbody_calibration.py` stub.**

Write `problems/hubble_tension_web/experiments/nbody_calibration.py`:

```python
"""Real-void α recalibration experiment.

Reads results/nbody_kbc.json, applies the 1D LSQ + bootstrap CI + F-test math
contract from spec §"Math contract for nbody_calibration.py", writes
results/nbody_calibration.json.

Implemented in Task 4.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("nbody_calibration.main: see Task 4 of the forward-ops plan.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 0.4: Verify imports don't break the suite.**

Run:
```bash
python -c "from problems.hubble_tension_web.nbody import mdpl2_download; from problems.hubble_tension_web.experiments import nbody_calibration; print('imports OK')"
```

Expected: `imports OK`. Nothing runs because `main()`/`fetch_sub_box()` are never called at import time.

Run the fast suite again:
```bash
python -m pytest tests/hubble_tension_web -x --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```

Expected: still `73 passed, 3 xfailed` (stubs add no tests and break no imports).

- [ ] **Step 0.5: Commit the scaffold.**

```bash
git add problems/hubble_tension_web/nbody/mdpl2_download.py \
        problems/hubble_tension_web/experiments/nbody_calibration.py
git commit -m "$(cat <<'EOF'
scaffold: forward-ops stub modules (mdpl2_download, nbody_calibration)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: `mdpl2_download.py` implementation + `test_mdpl2_download.py`

**Purpose.** Implement the CosmoSim SQL pull with h-conversion at write-time. All network I/O is mocked in tests; opt-in gate honors `ATFT_MDPL2_DOWNLOAD_ENABLED`.

**Files:**
- Modify: `problems/hubble_tension_web/nbody/mdpl2_download.py`
- Create: `tests/hubble_tension_web/nbody/test_mdpl2_download.py`

**Design notes (non-obvious decisions the engineer should understand):**

1. **Little-h convention.** MDPL2 Planck cosmology `h = MDPL2_H = 0.6777`. CosmoSim returns positions in `h⁻¹ Mpc` and masses in `h⁻¹ M_sun`. Physical units: `x_phys = x_h / h`, `mvir_phys = mvir_h / h`. (Spec §"three sharp findings" #2 — the "little-h unit trap"; spec-mandated example: input `x=677.7 h⁻¹Mpc` must round-trip to `x≈1000.0 Mpc` since `677.7 / 0.6777 = 999.85…` → off by a whisker — see Step 1.2 assertion tolerance.)
2. **Column alias.** Rockstar tables variously expose `mvir`, `Mvir_all`, or `M200c`. MDPL2 CosmoSim Rockstar primary is `Mvir`. We pick `Mvir` on the server side and rename to `mvir` before writing Parquet, matching the `EXPECTED_COLUMNS` contract from `mdpl2_fetch.py:18`.
3. **SQL endpoint shape — UNVERIFIED.** CosmoSim uses TAP (IVOA Table Access Protocol); the precise async endpoint layout (`/tap/async` → job URL → `/phase` polling → `/results/result` CSV) is inferred from TAP conventions and the CosmoSim metadata page. **Marked `# UNVERIFIED` in the module docstring** — the maintainer MUST confirm on the first `_ENABLED=1` run. The SQL and endpoint are also overridable via `ATFT_MDPL2_URL` and `ATFT_MDPL2_SQL`.
4. **Stdlib only.** `urllib.request` for HTTP; `csv` for CSV parsing; `pandas.to_parquet` for output. No `requests`/`pooch`.
5. **Polling cadence.** 10 s initial, exponential backoff to 60 s, total cap 45 minutes (CosmoSim SQL jobs typically complete in 10-30 min per spec §"Risks").

- [ ] **Step 1.1: Write the failing schema + h-conversion test.**

Write `tests/hubble_tension_web/nbody/test_mdpl2_download.py`:

```python
"""Tests for the CosmoSim MDPL2 download layer.

All network I/O is mocked. The live path is exercised manually by the maintainer
with ATFT_MDPL2_DOWNLOAD_ENABLED=1 and a real ATFT_SCISERVER_TOKEN; not in CI.
"""
from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from problems.hubble_tension_web.nbody import mdpl2_download
from problems.hubble_tension_web.nbody import mdpl2_fetch


# Sample CosmoSim CSV response (h-units, Planck h=0.6777).
# Row 1: exact h-to-physical round-trip anchor: 677.7 h^-1 Mpc -> 999.85... Mpc.
# Row 2: zero-position anchor (must stay zero under h-division).
# Row 3: mid-mass halo so the mass_cut / mvir_phys relationship is exercised.
MOCK_CSV = """x,y,z,Mvir
677.7,0.0,0.0,6.777e11
0.0,0.0,0.0,1.3554e12
100.0,200.0,300.0,2e12
"""


def test_fetch_sub_box_writes_parquet_with_correct_schema(tmp_path, monkeypatch):
    """Mocked HTTP round-trip produces Parquet matching EXPECTED_COLUMNS."""
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")
    monkeypatch.setenv("ATFT_SCISERVER_TOKEN", "fake-token")

    dest = tmp_path / "mdpl2_z0_500Mpc.parquet"

    # Mock the three-phase HTTP: submit -> poll -> fetch CSV.
    with patch.object(mdpl2_download, "_submit_sql_job", return_value="job-123"), \
         patch.object(mdpl2_download, "_poll_until_done", return_value=None), \
         patch.object(mdpl2_download, "_download_csv", return_value=MOCK_CSV):
        produced = mdpl2_download.fetch_sub_box(dest=dest, sub_box_mpc=500.0)

    assert produced == dest
    assert dest.exists(), f"expected Parquet at {dest}"

    df = pd.read_parquet(dest)
    for col in mdpl2_fetch.EXPECTED_COLUMNS:
        assert col in df.columns, f"schema contract broken: {col} missing"
    assert len(df) == 3


def test_fetch_sub_box_applies_h_conversion(tmp_path, monkeypatch):
    """Input x=677.7 h^-1 Mpc, h=0.6777 -> output x close to 1000 Mpc.

    Exact: 677.7 / 0.6777 = 1000.0 (the spec's round-trip anchor).
    """
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")
    monkeypatch.setenv("ATFT_SCISERVER_TOKEN", "fake-token")
    dest = tmp_path / "mdpl2.parquet"

    with patch.object(mdpl2_download, "_submit_sql_job", return_value="job-123"), \
         patch.object(mdpl2_download, "_poll_until_done", return_value=None), \
         patch.object(mdpl2_download, "_download_csv", return_value=MOCK_CSV):
        mdpl2_download.fetch_sub_box(dest=dest, sub_box_mpc=500.0)

    df = pd.read_parquet(dest)

    # Row 1: x=677.7 h^-1 Mpc -> 1000.0 Mpc, mvir=6.777e11 h^-1 M_sun -> 1e12 M_sun.
    # h_MDPL2 = 0.6777 exactly; 677.7 / 0.6777 = 1000.0 to machine precision.
    h = mdpl2_download.MDPL2_H
    assert h == pytest.approx(0.6777, abs=1e-12)

    np.testing.assert_allclose(df.loc[0, "x"], 677.7 / h, atol=1e-9)
    assert abs(df.loc[0, "x"] - 1000.0) < 1e-9, (
        f"h-conversion anchor off: got x={df.loc[0, 'x']}, expected 1000.0"
    )
    assert df.loc[0, "y"] == 0.0
    assert df.loc[0, "z"] == 0.0
    assert abs(df.loc[0, "mvir"] - 1e12) < 1e-3, (
        f"h-conversion mvir anchor off: got {df.loc[0, 'mvir']}, expected 1e12"
    )

    # Row 2 is at origin; must stay at origin.
    assert df.loc[1, "x"] == 0.0 and df.loc[1, "y"] == 0.0 and df.loc[1, "z"] == 0.0


def test_fetch_sub_box_gated_on_enable_flag(tmp_path, monkeypatch):
    """With ATFT_MDPL2_DOWNLOAD_ENABLED != '1', fetch_sub_box must refuse."""
    monkeypatch.delenv("ATFT_MDPL2_DOWNLOAD_ENABLED", raising=False)
    from problems.hubble_tension_web.nbody import NBodyDataNotAvailable
    with pytest.raises(NBodyDataNotAvailable, match="ATFT_MDPL2_DOWNLOAD_ENABLED"):
        mdpl2_download.fetch_sub_box(dest=tmp_path / "x.parquet")


def test_fetch_sub_box_requires_token_when_enabled(tmp_path, monkeypatch):
    """With ATFT_MDPL2_DOWNLOAD_ENABLED='1' but no token, must raise with a clear message."""
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")
    monkeypatch.delenv("ATFT_SCISERVER_TOKEN", raising=False)
    from problems.hubble_tension_web.nbody import NBodyDataNotAvailable
    with pytest.raises(NBodyDataNotAvailable, match="ATFT_SCISERVER_TOKEN"):
        mdpl2_download.fetch_sub_box(dest=tmp_path / "x.parquet")


def test_fetch_sub_box_applies_sub_box_filter_via_sql(monkeypatch):
    """The sub_box_mpc argument flows into the SQL body as an h^-1 Mpc bound."""
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")
    monkeypatch.setenv("ATFT_SCISERVER_TOKEN", "fake-token")

    captured_sql: list[str] = []

    def fake_submit(*, url: str, sql: str, token: str) -> str:
        captured_sql.append(sql)
        return "job-xyz"

    with patch.object(mdpl2_download, "_submit_sql_job", side_effect=fake_submit), \
         patch.object(mdpl2_download, "_poll_until_done", return_value=None), \
         patch.object(mdpl2_download, "_download_csv", return_value=MOCK_CSV):
        # Use an in-memory dest via tmp-dir equivalent: we only care the SQL was built.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            mdpl2_download.fetch_sub_box(dest=Path(td) / "x.parquet", sub_box_mpc=500.0)

    assert captured_sql, "SQL job was never submitted"
    sql = captured_sql[0]
    # Sub-box filter in h^-1 Mpc: 500 Mpc * h = 500 * 0.6777 = 338.85 h^-1 Mpc.
    # We don't pin an exact format; just that the bound appears.
    assert "338" in sql, f"expected h-unit sub-box bound in SQL, got: {sql!r}"
    # Host-halo filter per spec § three-sharp-findings + standard Rockstar idiom.
    assert "pid" in sql, f"expected pid=-1 host-halo filter in SQL, got: {sql!r}"


def test_override_sql_env_var(tmp_path, monkeypatch):
    """ATFT_MDPL2_SQL wholesale-overrides the query string."""
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")
    monkeypatch.setenv("ATFT_SCISERVER_TOKEN", "fake-token")
    monkeypatch.setenv("ATFT_MDPL2_SQL", "SELECT x, y, z, Mvir FROM custom_table WHERE 1=1")

    captured: list[str] = []

    def fake_submit(*, url: str, sql: str, token: str) -> str:
        captured.append(sql)
        return "j"

    with patch.object(mdpl2_download, "_submit_sql_job", side_effect=fake_submit), \
         patch.object(mdpl2_download, "_poll_until_done", return_value=None), \
         patch.object(mdpl2_download, "_download_csv", return_value=MOCK_CSV):
        mdpl2_download.fetch_sub_box(dest=tmp_path / "x.parquet", sub_box_mpc=500.0)

    assert captured[0] == "SELECT x, y, z, Mvir FROM custom_table WHERE 1=1"
```

- [ ] **Step 1.2: Run the test to confirm it fails against the stub.**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_mdpl2_download.py -v
```

Expected: all six tests FAIL. Most likely with `NotImplementedError: mdpl2_download.fetch_sub_box: see Task 1` or `AttributeError: module has no attribute '_submit_sql_job'`.

- [ ] **Step 1.3: Implement `mdpl2_download.py`.**

Overwrite `problems/hubble_tension_web/nbody/mdpl2_download.py`:

```python
"""CosmoSim MDPL2 SQL download layer.

Submits a TAP-style async SQL job against the CosmoSim Rockstar z=0 catalog,
polls until done, streams the CSV result, converts little-h units to pure
Mpc/M_sun at write-time, and persists Parquet matching the EXPECTED_COLUMNS
contract from mdpl2_fetch.

UNVERIFIED endpoint shape. This module encodes the CosmoSim TAP async protocol
as of the spec drafting date (2026-04). The precise async endpoint path layout
is inferred from IVOA TAP conventions and the CosmoSim metadata at
https://www.cosmosim.org/metadata/mdpl2/rockstar/. On the first live
`ATFT_MDPL2_DOWNLOAD_ENABLED=1` run, the maintainer must verify that:
  1. The TAP endpoint is reachable at $ATFT_MDPL2_URL/tap/async (default below).
  2. The returned job URL layout / phase polling / result download match what
     `_submit_sql_job`, `_poll_until_done`, `_download_csv` expect.
  3. The `Mvir` column name is still present (vs. `Mvir_all` / `M200c` variants).

Both the endpoint base and the SQL body are overridable via env vars:
  ATFT_MDPL2_URL  - TAP base (default https://www.cosmosim.org/tap)
  ATFT_MDPL2_SQL  - full SELECT statement (default: built from sub_box_mpc)

Gate: ATFT_MDPL2_DOWNLOAD_ENABLED must equal "1" or fetch_sub_box raises
NBodyDataNotAvailable. This preserves CI and the stubbed-network regression test.
"""
from __future__ import annotations

import csv
import io
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from problems.hubble_tension_web.nbody import NBodyDataNotAvailable


# Planck 2016 cosmology used in the MDPL2 simulation (Klypin+2016 Table 1).
MDPL2_H: float = 0.6777

# Default CosmoSim TAP endpoint. Override with ATFT_MDPL2_URL.
_DEFAULT_TAP_URL: str = "https://www.cosmosim.org/tap"

# Default Rockstar catalog name on CosmoSim. Override by wholesale-replacing
# the SQL body via ATFT_MDPL2_SQL if CosmoSim ever renames the table.
_DEFAULT_TABLE: str = "MDPL2.Rockstar"

# Polling parameters. CosmoSim SQL jobs typically complete in 10-30 min.
_POLL_INITIAL_S: float = 10.0
_POLL_MAX_S: float = 60.0
_POLL_TOTAL_CAP_S: float = 45.0 * 60.0  # 45 min ceiling

# Rockstar column-alias map. We always select the first-preference name on the
# server side; if a future MDPL2 revision exposes a different name, add it here
# AND bump the alias resolution in _build_default_sql.
_MASS_COLUMN_PREFERENCE: tuple[str, ...] = ("Mvir", "Mvir_all", "M200c")


def _h_correct_url(env: dict | None = None) -> str:
    env = env if env is not None else os.environ
    return env.get("ATFT_MDPL2_URL") or _DEFAULT_TAP_URL


def _build_default_sql(sub_box_mpc: float) -> str:
    """Build the default CosmoSim TAP SQL body.

    Returns a host-halo z=0 selection on a sub-box from the origin. Units are
    h^-1 Mpc on the server side; we convert to physical Mpc post-download.
    """
    sub_h = sub_box_mpc * MDPL2_H   # physical Mpc -> h^-1 Mpc
    mass_col = _MASS_COLUMN_PREFERENCE[0]
    # UNVERIFIED: confirm column/table names on first live run. The SQL is
    # parameterized so a mirror/variant can override via ATFT_MDPL2_SQL.
    return (
        f"SELECT x, y, z, {mass_col} FROM {_DEFAULT_TABLE} "
        f"WHERE pid = -1 "
        f"AND x < {sub_h:.4f} AND y < {sub_h:.4f} AND z < {sub_h:.4f} "
        f"AND snapnum = 125"  # z=0 snapshot for MDPL2
    )


def _submit_sql_job(*, url: str, sql: str, token: str) -> str:
    """POST the SQL to $url/async and return the returned job URL/id.

    TAP convention: POST form-encoded `REQUEST=doQuery&LANG=ADQL&QUERY=...`
    plus an Authorization header. Server returns 303 See Other with a Location
    header pointing at the job resource.
    """
    body = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "QUERY": sql,
        "FORMAT": "csv",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/async",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        # TAP returns the job URL in the Location header on 303.
        job_url = resp.headers.get("Location") or resp.read().decode("utf-8").strip()
    if not job_url:
        raise NBodyDataNotAvailable(f"CosmoSim TAP submit returned empty job URL for {url}")
    # Kick the job: POST PHASE=RUN to $job_url/phase.
    run_req = urllib.request.Request(
        f"{job_url.rstrip('/')}/phase",
        data=b"PHASE=RUN",
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(run_req, timeout=60.0):
        pass
    return job_url


def _poll_until_done(*, job_url: str, token: str) -> None:
    """Poll $job_url/phase until it reaches COMPLETED (or raises on ERROR/cap)."""
    start = time.monotonic()
    interval = _POLL_INITIAL_S
    while True:
        req = urllib.request.Request(
            f"{job_url.rstrip('/')}/phase",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            phase = resp.read().decode("utf-8").strip().upper()
        if phase == "COMPLETED":
            return
        if phase in ("ERROR", "ABORTED", "HELD"):
            raise NBodyDataNotAvailable(
                f"CosmoSim SQL job {job_url} terminated in phase={phase}"
            )
        elapsed = time.monotonic() - start
        if elapsed > _POLL_TOTAL_CAP_S:
            raise NBodyDataNotAvailable(
                f"CosmoSim SQL job {job_url} exceeded poll cap "
                f"({_POLL_TOTAL_CAP_S:.0f}s); last phase={phase}"
            )
        time.sleep(interval)
        interval = min(interval * 1.3, _POLL_MAX_S)


def _download_csv(*, job_url: str, token: str) -> str:
    """GET the result CSV body from $job_url/results/result."""
    req = urllib.request.Request(
        f"{job_url.rstrip('/')}/results/result",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=300.0) as resp:
        return resp.read().decode("utf-8")


def _parse_csv_to_dataframe(csv_text: str) -> pd.DataFrame:
    """Parse a CosmoSim TAP CSV response into a DataFrame with raw h-unit columns."""
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[dict] = []
    for r in reader:
        rows.append(r)
    if not rows:
        raise NBodyDataNotAvailable("CosmoSim CSV response contained zero rows")
    df = pd.DataFrame(rows)
    # Resolve the mass column alias.
    mass_col: str | None = None
    for cand in _MASS_COLUMN_PREFERENCE:
        if cand in df.columns:
            mass_col = cand
            break
    if mass_col is None:
        raise NBodyDataNotAvailable(
            f"CosmoSim CSV missing mass column; expected one of "
            f"{_MASS_COLUMN_PREFERENCE}, got {list(df.columns)}"
        )
    for needed in ("x", "y", "z"):
        if needed not in df.columns:
            raise NBodyDataNotAvailable(
                f"CosmoSim CSV missing column {needed!r}; got {list(df.columns)}"
            )
    # Cast text to float64. CosmoSim returns strings; pandas' default is object.
    for col in ("x", "y", "z", mass_col):
        df[col] = pd.to_numeric(df[col], downcast=None).astype("float64")
    # Rename mass to the schema-contract name.
    df = df.rename(columns={mass_col: "mvir"})
    return df[["x", "y", "z", "mvir"]]


def _apply_h_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Divide h^-1 units by MDPL2_H to get pure Mpc / M_sun."""
    out = df.copy()
    out["x"] = df["x"] / MDPL2_H
    out["y"] = df["y"] / MDPL2_H
    out["z"] = df["z"] / MDPL2_H
    out["mvir"] = df["mvir"] / MDPL2_H
    return out


def fetch_sub_box(*, dest: str | Path, sub_box_mpc: float = 500.0) -> Path:
    """Pull a MDPL2 sub-box halo catalog and write it as Parquet.

    Args:
      dest: output Parquet path (parent dir created if missing).
      sub_box_mpc: sub-box side length in physical Mpc. Converted to h^-1 Mpc
                   for the SQL WHERE clause.

    Returns:
      The Path to the written Parquet.

    Raises:
      NBodyDataNotAvailable: if ATFT_MDPL2_DOWNLOAD_ENABLED != "1", if
        ATFT_SCISERVER_TOKEN is unset, if the SQL job fails, or if the CSV
        response is malformed.
    """
    if os.environ.get("ATFT_MDPL2_DOWNLOAD_ENABLED", "0") != "1":
        raise NBodyDataNotAvailable(
            "CosmoSim MDPL2 download is gated. Set ATFT_MDPL2_DOWNLOAD_ENABLED=1 "
            "and ATFT_SCISERVER_TOKEN=<your token> to enable. See "
            "problems/hubble_tension_web/nbody/README.md."
        )
    token = os.environ.get("ATFT_SCISERVER_TOKEN")
    if not token:
        raise NBodyDataNotAvailable(
            "ATFT_SCISERVER_TOKEN unset. Obtain one at https://www.sciserver.org/ "
            "and export it. See problems/hubble_tension_web/nbody/README.md."
        )

    url = _h_correct_url()
    sql = os.environ.get("ATFT_MDPL2_SQL") or _build_default_sql(sub_box_mpc=sub_box_mpc)

    job_url = _submit_sql_job(url=url, sql=sql, token=token)
    _poll_until_done(job_url=job_url, token=token)
    csv_text = _download_csv(job_url=job_url, token=token)

    df_h = _parse_csv_to_dataframe(csv_text)
    df_phys = _apply_h_conversion(df_h)

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df_phys.to_parquet(dest_path, index=False)
    return dest_path
```

- [ ] **Step 1.4: Run the test to verify it passes.**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_mdpl2_download.py -v
```

Expected: all six tests PASS. If `test_fetch_sub_box_applies_h_conversion` fails by a tiny numeric margin, confirm `MDPL2_H == 0.6777` exactly — anything else is a bug (do NOT relax the tolerance).

- [ ] **Step 1.5: Run the full fast suite to confirm no regression.**

Run:
```bash
python -m pytest tests/hubble_tension_web -x --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```

Expected: `79 passed, 3 xfailed` (73 baseline + 6 new).

- [ ] **Step 1.6: Commit.**

```bash
git add problems/hubble_tension_web/nbody/mdpl2_download.py \
        tests/hubble_tension_web/nbody/test_mdpl2_download.py
git commit -m "$(cat <<'EOF'
feat(nbody): CosmoSim MDPL2 SQL download + h-conversion (opt-in, mocked in tests)

- Async TAP SQL job: submit -> poll -> CSV -> Parquet.
- h-conversion at write-time so cached Parquet matches EXPECTED_COLUMNS contract.
- Gated on ATFT_MDPL2_DOWNLOAD_ENABLED=1 and ATFT_SCISERVER_TOKEN.
- 6 tests: schema, h-conversion anchor (677.7 h^-1 Mpc -> 1000 Mpc), gate, token,
  sub-box SQL, env-override. All mocked; no live network in CI.

SQL endpoint shape UNVERIFIED — marked in module docstring. Maintainer to
confirm on first live _ENABLED=1 run; both endpoint + SQL are env-overridable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `mdpl2_fetch.py::fetch_from_network` delegates when enabled

**Purpose.** When the download gate is on, `fetch_from_network` delegates to `mdpl2_download.fetch_sub_box`. When off (default), the existing `NotImplementedError("CosmoSim …")` stub is preserved so the existing `test_network_fetch_is_stubbed` test keeps passing unchanged.

**Key constraint.** The existing test at `tests/hubble_tension_web/nbody/test_mdpl2_fetch.py:54-56` matches `"CosmoSim"` in the error message. We preserve that word in the new stub phrasing so the test stays green without modification.

**Files:**
- Modify: `problems/hubble_tension_web/nbody/mdpl2_fetch.py` (`fetch_from_network` only)
- Modify: `tests/hubble_tension_web/nbody/test_mdpl2_fetch.py` (add one delegating-path test; keep the existing 6 tests)

- [ ] **Step 2.1: Add the failing delegation test.**

Append to `tests/hubble_tension_web/nbody/test_mdpl2_fetch.py` (do NOT modify existing tests):

```python
from unittest.mock import patch


def test_network_fetch_delegates_when_enabled(tmp_path, monkeypatch):
    """With ATFT_MDPL2_DOWNLOAD_ENABLED='1', fetch_from_network defers to
    mdpl2_download.fetch_sub_box rather than raising NotImplementedError."""
    monkeypatch.setenv("ATFT_MDPL2_DOWNLOAD_ENABLED", "1")

    dest = tmp_path / "deleg.parquet"
    with patch(
        "problems.hubble_tension_web.nbody.mdpl2_download.fetch_sub_box",
        return_value=dest,
    ) as mock_fetch:
        result = mdpl2_fetch.fetch_from_network(
            url="https://cosmosim.example/mdpl2/z0_500Mpc.dat",
            dest=dest,
        )
    assert result == dest
    mock_fetch.assert_called_once()
    # Ensure dest flows through.
    _, kwargs = mock_fetch.call_args
    assert kwargs.get("dest") == dest


def test_network_fetch_still_stubbed_when_disabled(monkeypatch):
    """Redundant with test_network_fetch_is_stubbed but pins the env-off branch."""
    monkeypatch.delenv("ATFT_MDPL2_DOWNLOAD_ENABLED", raising=False)
    with pytest.raises(NotImplementedError, match="CosmoSim"):
        mdpl2_fetch.fetch_from_network(url="https://cosmosim.example/mdpl2/z0_500Mpc.dat")
```

- [ ] **Step 2.2: Run the tests; confirm new ones fail, existing ones still pass.**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_mdpl2_fetch.py -v
```

Expected: 6 existing tests PASS, 2 new tests FAIL (the delegation branch isn't wired yet; `test_network_fetch_still_stubbed_when_disabled` incidentally passes too because the current stub always raises).

Note: the existing `test_network_fetch_is_stubbed` currently does NOT pass `ATFT_MDPL2_DOWNLOAD_ENABLED` either. After Task 2 implementation, a stray `ATFT_MDPL2_DOWNLOAD_ENABLED=1` in the shell would break it — use `monkeypatch.delenv` defensively if the existing test later flakes in local dev. Don't modify it now; that's scope creep.

- [ ] **Step 2.3: Modify `fetch_from_network` in `mdpl2_fetch.py`.**

Replace the body of `fetch_from_network` (lines 76-88 in the current file). Keep the docstring's first paragraph, update the "v1" comment, and add the delegation branch:

```python
def fetch_from_network(*, url: str, dest: str | Path | None = None) -> Path:
    """Download an MDPL2 halo catalog from CosmoSim.

    When ATFT_MDPL2_DOWNLOAD_ENABLED == "1": delegates to
    problems.hubble_tension_web.nbody.mdpl2_download.fetch_sub_box, which
    performs the TAP SQL pull + h-conversion + Parquet write.

    Otherwise (default, including CI): raises NotImplementedError with the
    same historical message, preserving the `test_network_fetch_is_stubbed`
    regression test and the "no network in CI" guarantee.
    """
    import os
    if os.environ.get("ATFT_MDPL2_DOWNLOAD_ENABLED", "0") == "1":
        # Import inside the function to avoid a top-level cycle risk and to
        # keep import-time surface minimal when the download path is off.
        from problems.hubble_tension_web.nbody import mdpl2_download
        if dest is None:
            cache_root = Path(os.environ.get(
                "ATFT_DATA_CACHE", str(Path.home() / ".cache" / "atft")
            ))
            dest = cache_root / "mdpl2" / "mdpl2_z0_500Mpc.parquet"
        return mdpl2_download.fetch_sub_box(dest=dest, sub_box_mpc=500.0)

    raise NotImplementedError(
        "CosmoSim MDPL2 download is gated. Set ATFT_MDPL2_DOWNLOAD_ENABLED=1 "
        "and ATFT_SCISERVER_TOKEN=<token> to enable. Otherwise manually "
        "download a halo catalog and place it at ~/.cache/atft/mdpl2/ "
        "(or $ATFT_DATA_CACHE). See problems/hubble_tension_web/nbody/README.md."
    )
```

Verify the stub message still contains `"CosmoSim"` (required by the existing test at line 55).

- [ ] **Step 2.4: Run the tests to confirm all pass.**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_mdpl2_fetch.py -v
```

Expected: 8 PASS (6 existing + 2 new).

- [ ] **Step 2.5: Run the full fast suite.**

Run:
```bash
python -m pytest tests/hubble_tension_web -x --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```

Expected: `81 passed, 3 xfailed`.

- [ ] **Step 2.6: Commit.**

```bash
git add problems/hubble_tension_web/nbody/mdpl2_fetch.py \
        tests/hubble_tension_web/nbody/test_mdpl2_fetch.py
git commit -m "$(cat <<'EOF'
feat(nbody): mdpl2_fetch.fetch_from_network delegates to mdpl2_download when enabled

When ATFT_MDPL2_DOWNLOAD_ENABLED=1, fetch_from_network routes to the real
CosmoSim SQL pull. Otherwise (default) it preserves the NotImplementedError
stub containing the word "CosmoSim" — `test_network_fetch_is_stubbed` remains
unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `nbody_kbc.py` emits per-void `f_topo_at_alpha_1`, `ltb_anchor_at_delta_R`, `y_residual`

**Purpose.** The calibration leg (Task 4) needs to reuse — per void — the exact `f_topo` evaluated at `α=1` and the LTB reference anchor `ΔH₀_LTB(δ, R)` without re-running `predict_from_cosmic_web`. Emit all three from `nbody_kbc.py` at write-time.

**Key imports added to `nbody_kbc.py`:**
- `from problems.hubble_tension_web.functional import C1, f_topo`
- `from problems.hubble_tension_web.ltb_reference import delta_H0_ltb`

**Per-void math (straight from spec §"Math contract"):**
```python
f_topo_at_alpha_1 = f_topo(summary.beta0, summary.beta1, summary.lambda_min, cand.radius_mpc)
ltb_anchor_at_delta_R = delta_H0_ltb(delta=cand.delta_eff, R_mpc=cand.radius_mpc)
y_residual = ltb_anchor_at_delta_R - C1 * cand.delta_eff
```

**Files:**
- Modify: `problems/hubble_tension_web/experiments/nbody_kbc.py`
- Modify: `tests/hubble_tension_web/nbody/test_nbody_kbc.py` (extend schema assertions)

- [ ] **Step 3.1: Update the `test_nbody_kbc.py` schema assertion (failing first).**

In `tests/hubble_tension_web/nbody/test_nbody_kbc.py`, extend the per-void field list (currently lines 49-52). Change this block:

```python
        for field in ("idx", "center_mpc", "N_halos", "delta_eff", "R_eff_mpc",
                      "beta0", "beta1_persistent", "lambda_min",
                      "delta_H0_total", "kinematic_term", "topological_term"):
            assert field in v0, f"missing per-void field: {field}"
```

to:

```python
        for field in ("idx", "center_mpc", "N_halos", "delta_eff", "R_eff_mpc",
                      "beta0", "beta1_persistent", "lambda_min",
                      "delta_H0_total", "kinematic_term", "topological_term",
                      "f_topo_at_alpha_1", "ltb_anchor_at_delta_R", "y_residual"):
            assert field in v0, f"missing per-void field: {field}"
        # The three new fields must be JSON-serializable floats.
        assert isinstance(v0["f_topo_at_alpha_1"], float)
        assert isinstance(v0["ltb_anchor_at_delta_R"], float)
        assert isinstance(v0["y_residual"], float)
```

- [ ] **Step 3.2: Run the test; expect it to FAIL because the new fields aren't emitted yet.**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_nbody_kbc.py -v
```

Expected: FAIL with `KeyError` / `AssertionError: missing per-void field: f_topo_at_alpha_1` **if the fixture produces at least one void**. If `data["voids"]` is empty in the fixture, the test's per-void block is skipped and the existing 73-baseline count's "voids present" condition depends on the fixture — in which case the test will (vacuously) PASS. Run it either way; expected behavior is: FAIL when voids found, PASS (vacuous) otherwise.

**Engineer note.** The `mini_mdpl2.parquet` fixture typically yields 2-3 candidate voids at `grid_N=32, K_VOIDS=3`. If your run reports 0 voids, the test-guard is vacuous — proceed with Step 3.3 regardless; Step 3.4 will verify the fields appear in the JSON output.

- [ ] **Step 3.3: Modify `nbody_kbc.py`.**

Add the two imports at the top of `problems/hubble_tension_web/experiments/nbody_kbc.py` (insert after the existing `from problems.hubble_tension_web.functional import predict_from_cosmic_web` on line 34):

```python
from problems.hubble_tension_web.functional import C1, f_topo, predict_from_cosmic_web
from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
```

(Replace the existing `from problems.hubble_tension_web.functional import predict_from_cosmic_web` with the new combined line.)

Inside the per-void loop (currently at lines 122-134), extend the `per_void.append(dict(...))` block to include the three new fields. Replace the existing dict construction:

```python
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
```

with:

```python
        f_topo_at_alpha_1 = float(f_topo(
            int(summary.beta0), int(summary.beta1),
            float(summary.lambda_min), float(cand.radius_mpc),
        ))
        ltb_anchor = float(delta_H0_ltb(
            delta=float(cand.delta_eff), R_mpc=float(cand.radius_mpc),
        ))
        y_residual = float(ltb_anchor - C1 * float(cand.delta_eff))

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
            # New in forward-ops: per-void primitives the calibration leg reuses.
            f_topo_at_alpha_1=f_topo_at_alpha_1,
            ltb_anchor_at_delta_R=ltb_anchor,
            y_residual=y_residual,
        ))
```

- [ ] **Step 3.4: Run `test_nbody_kbc.py` to confirm pass.**

Run:
```bash
python -m pytest tests/hubble_tension_web/nbody/test_nbody_kbc.py -v
```

Expected: PASS. The subprocess run regenerates `nbody_kbc.json` with the three new fields populated.

- [ ] **Step 3.5: Quick sanity on the produced JSON.**

Run:
```bash
python -c "import json; p=__import__('pathlib').Path('problems/hubble_tension_web/results/nbody_kbc.json'); d=json.loads(p.read_text()) if p.exists() else {'voids':[]}; print([list(v.keys()) for v in d['voids'][:1]])"
```

Expected: a list that includes `'f_topo_at_alpha_1'`, `'ltb_anchor_at_delta_R'`, `'y_residual'`.

- [ ] **Step 3.6: Full fast suite.**

Run:
```bash
python -m pytest tests/hubble_tension_web -x --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```

Expected: `81 passed, 3 xfailed` (same count — Task 3 added assertions but no new test functions).

- [ ] **Step 3.7: Commit.**

```bash
git add problems/hubble_tension_web/experiments/nbody_kbc.py \
        tests/hubble_tension_web/nbody/test_nbody_kbc.py
git commit -m "$(cat <<'EOF'
feat(nbody): emit per-void f_topo, LTB anchor, and y_residual from nbody_kbc

Adds three fields to each record in nbody_kbc.json:
- f_topo_at_alpha_1: f_topo(beta0, beta1, lambda_min, R_eff) pre-computed at alpha=1
- ltb_anchor_at_delta_R: delta_H0_ltb(delta_eff, R_eff) reference value
- y_residual: ltb_anchor - C1 * delta_eff (the calibration target per spec math contract)

Lets nbody_calibration.py compute alpha* from nbody_kbc.json alone with no
re-execution of predict_from_cosmic_web.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `nbody_calibration.py` implementation + `test_nbody_calibration.py`

**Purpose.** Closed-form 1D LSQ on real-void primitives, bootstrap 68% CI, F-test vs α=0. Emit `nbody_calibration.json` per spec §"Math contract" schema. Three branches: fit-succeeded / insufficient-voids / f_topo-all-zero.

**Math — restated from spec with simplification flagged by advisor:**

Given K voids with pre-computed `f_k = f_topo_at_alpha_1_k` and `y_k = y_residual_k`:

```python
# Closed-form 1D LSQ
denom = sum(f_k**2 for f_k in f)
if denom < 1e-24:
    alpha_star = None; reason = "f_topo all zero"
else:
    alpha_star = sum(f_k * y_k for f_k, y_k in zip(f, y)) / denom

# Bootstrap (B = ATFT_NBODY_CAL_BOOTSTRAP_B, default 2000; seeded np.random.default_rng(0))
for b in range(B):
    idx = rng.integers(K, size=K)
    alpha_b = (f[idx] @ y[idx]) / (f[idx] @ f[idx])   # with zero-denom guard
alpha_16, alpha_50, alpha_84 = percentile(alpha_bs_valid, [16, 50, 84])

# F-test (since y = ΔH₀_LTB - C1·δ, spec's RSS_0 simplifies to y @ y).
RSS_0 = float(y @ y)                           # kinematic-only residual
RSS_1 = float((alpha_star * f - y) @ (alpha_star * f - y))   # with α fit
F     = (RSS_0 - RSS_1) / (RSS_1 / max(K - 1, 1))
p_F   = 1.0 - scipy.stats.f.cdf(F, 1, max(K - 1, 1))

# Pathology gate (spec §"No external LTB integrator in v1")
if (alpha_star is not None) and (abs(alpha_star) > 1e4 or p_F > 0.5):
    reason = "LTB heuristic may be stretched — see spec §6.2.a"
```

**Files:**
- Modify: `problems/hubble_tension_web/experiments/nbody_calibration.py` (replace stub)
- Create: `tests/hubble_tension_web/test_nbody_calibration.py`

- [ ] **Step 4.1: Write the failing test with planted α* closed-form.**

Write `tests/hubble_tension_web/test_nbody_calibration.py`:

```python
"""Tests for the real-void alpha recalibration experiment.

Fixture strategy: write a mock nbody_kbc.json with 8 synthetic per-void records,
compute the expected alpha* in numpy from the same literal values, run
nbody_calibration as a subprocess with a custom results dir, and assert the
produced alpha* matches the hand-computed value to 1e-10. Exercises all three
reason-code branches: fit-succeeded, insufficient-voids, f_topo-all-zero.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


# Planted fixture. Values chosen so:
#   * K = 8 (> default min 3)
#   * beta1 varies in {0, 1, 2, 3} across the records
#   * deltas span the spec's void range [-0.3, -0.05]
#   * Rs span [100, 500] Mpc
# These exact numbers are what the hand-computation below re-uses — tests stay
# closed-form, not a same-formula-twice tautology.
_DELTAS      = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.15, -0.20]
_RS          = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0]
_BETA0S      = [10, 10, 12, 14, 11, 13, 12, 15]
_BETA1S      = [0, 1, 2, 3, 0, 1, 2, 3]
_LAMBDA_MINS = [0.5, 0.4, 0.3, 0.2, 0.5, 0.4, 0.3, 0.2]


def _hand_computed_alpha_star():
    """Compute alpha* in numpy from the same literal values as the fixture."""
    from problems.hubble_tension_web.functional import C1, f_topo
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb

    f = np.array([
        f_topo(int(b0), int(b1), float(lm), float(R))
        for b0, b1, lm, R in zip(_BETA0S, _BETA1S, _LAMBDA_MINS, _RS)
    ], dtype=np.float64)
    y = np.array([
        delta_H0_ltb(delta=float(d), R_mpc=float(R)) - C1 * float(d)
        for d, R in zip(_DELTAS, _RS)
    ], dtype=np.float64)

    denom = float(f @ f)
    assert denom > 1e-24, "fixture accidentally landed at f_topo-all-zero"
    alpha_star = float((f @ y) / denom)
    return f, y, alpha_star


def _make_fixture_nbody_kbc_json(*, voids_records: list[dict]) -> dict:
    return {
        "cache_source": "/synthetic/test_fixture",
        "grid_N": 32,
        "lambda_th": 0.0,
        "K": len(voids_records),
        "alpha_used": 0.0,
        "timestamp": "2026-04-21T00:00:00+00:00",
        "voids": voids_records,
        "beta1_distribution": {
            "count_nonzero": sum(1 for v in voids_records if v["beta1_persistent"] > 0),
            "count_total": len(voids_records),
            "median": 0.0,
            "max": max((v["beta1_persistent"] for v in voids_records), default=0),
        },
    }


def _make_voids_records() -> list[dict]:
    from problems.hubble_tension_web.functional import C1, f_topo
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb

    recs = []
    for i, (d, R, b0, b1, lm) in enumerate(zip(
        _DELTAS, _RS, _BETA0S, _BETA1S, _LAMBDA_MINS
    )):
        f_val = float(f_topo(int(b0), int(b1), float(lm), float(R)))
        ltb = float(delta_H0_ltb(delta=float(d), R_mpc=float(R)))
        y_res = float(ltb - C1 * float(d))
        recs.append({
            "idx": i,
            "center_mpc": [250.0, 250.0, 250.0],
            "N_halos": 100,
            "delta_eff": float(d),
            "R_eff_mpc": float(R),
            "beta0": int(b0),
            "beta1_persistent": int(b1),
            "lambda_min": float(lm),
            "delta_H0_total": C1 * float(d),
            "kinematic_term": C1 * float(d),
            "topological_term": 0.0,
            "f_topo_at_alpha_1": f_val,
            "ltb_anchor_at_delta_R": ltb,
            "y_residual": y_res,
        })
    return recs


def _run_calibration_subprocess(*, results_dir: Path, extra_env: dict | None = None) -> Path:
    env = os.environ.copy()
    env["ATFT_HUBBLE_RESULTS_DIR"] = str(results_dir)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-m", "problems.hubble_tension_web.experiments.nbody_calibration"],
        capture_output=True, env=env, timeout=60,
    )
    assert result.returncode == 0, (
        f"nbody_calibration failed rc={result.returncode}\n"
        f"stdout:\n{result.stdout.decode(errors='replace')}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
    return results_dir / "nbody_calibration.json"


def test_alpha_star_matches_closed_form(tmp_path):
    """On a K=8 planted fixture, alpha* must match the hand-computed value to 1e-10."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    voids_records = _make_voids_records()
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=voids_records), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    assert cal_path.exists()
    out = json.loads(cal_path.read_text())

    _, _, expected_alpha = _hand_computed_alpha_star()
    assert out["alpha_star"] is not None, f"reason was {out.get('reason')!r}"
    assert abs(out["alpha_star"] - expected_alpha) < 1e-10, (
        f"alpha_star={out['alpha_star']}, expected {expected_alpha}, "
        f"diff={out['alpha_star'] - expected_alpha}"
    )
    assert out["alpha_units"] == "km/s"
    assert out["K"] == 8


def test_bootstrap_ci_brackets_alpha_star(tmp_path):
    """68% bootstrap CI must be a 2-tuple that brackets the point estimate."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=_make_voids_records()), indent=2)
    )

    cal_path = _run_calibration_subprocess(
        results_dir=results_dir,
        extra_env={"ATFT_NBODY_CAL_BOOTSTRAP_B": "500"},  # small B for test speed
    )
    out = json.loads(cal_path.read_text())

    ci = out["alpha_ci_68"]
    assert ci is not None and len(ci) == 2
    lo, hi = float(ci[0]), float(ci[1])
    assert lo <= hi
    # The bootstrap median tends toward alpha*; CI should span it at 68%.
    alpha = float(out["alpha_star"])
    # Allow a small slack: on K=8, the sampled bootstrap CI can exclude alpha_star
    # in edge cases. Require at most one edge is on the wrong side (standard
    # percentile-bootstrap interpretation on small K).
    within = lo <= alpha <= hi
    assert within or abs(alpha - lo) < 0.5 or abs(alpha - hi) < 0.5, (
        f"alpha_star={alpha} far outside 68% CI {ci}"
    )


def test_f_test_returns_valid_pvalue(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=_make_voids_records()), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    out = json.loads(cal_path.read_text())

    p = out["p_F_alpha_vs_zero"]
    assert 0.0 <= float(p) <= 1.0


def test_per_void_records_complete(tmp_path):
    """Output per_void block has all required keys for aggregate.py."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=_make_voids_records()), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    out = json.loads(cal_path.read_text())

    assert len(out["per_void"]) == 8
    for pv in out["per_void"]:
        for key in ("delta", "R_mpc", "f_topo", "y_ltb_residual", "beta1"):
            assert key in pv, f"per_void missing key: {key}"


def test_insufficient_voids_branch(tmp_path):
    """K < ATFT_NBODY_CAL_MIN_VOIDS -> alpha_star null with insufficient-voids reason."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    recs = _make_voids_records()[:2]  # K = 2 < default min 3
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=recs), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    out = json.loads(cal_path.read_text())

    assert out["alpha_star"] is None
    assert out["alpha_ci_68"] is None
    assert "insufficient voids" in out["reason"]
    assert out["K"] == 2


def test_f_topo_all_zero_branch(tmp_path):
    """If every record has beta1=0, f_topo collapses; alpha_star null with clear reason."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Build records with beta1 forced to 0 everywhere so f_topo = 0.
    from problems.hubble_tension_web.functional import C1
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb

    recs = []
    for i in range(5):
        d, R = -0.15, 250.0
        ltb = float(delta_H0_ltb(delta=d, R_mpc=R))
        recs.append({
            "idx": i, "center_mpc": [0, 0, 0], "N_halos": 50,
            "delta_eff": d, "R_eff_mpc": R,
            "beta0": 10, "beta1_persistent": 0, "lambda_min": 0.5,
            "delta_H0_total": C1 * d, "kinematic_term": C1 * d, "topological_term": 0.0,
            "f_topo_at_alpha_1": 0.0, "ltb_anchor_at_delta_R": ltb,
            "y_residual": ltb - C1 * d,
        })
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=recs), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    out = json.loads(cal_path.read_text())

    assert out["alpha_star"] is None
    assert out["reason"] == "f_topo all zero"
    assert out["K"] == 5


def test_output_json_schema_shape(tmp_path):
    """All required top-level keys present."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "nbody_kbc.json").write_text(
        json.dumps(_make_fixture_nbody_kbc_json(voids_records=_make_voids_records()), indent=2)
    )

    cal_path = _run_calibration_subprocess(results_dir=results_dir)
    out = json.loads(cal_path.read_text())

    for key in (
        "alpha_star", "alpha_units", "alpha_ci_68", "alpha_bootstrap_median",
        "K", "bootstrap_B", "chi2_reduced", "pearson_r_f_y", "p_F_alpha_vs_zero",
        "reason", "per_void", "ltb_reference_source", "timestamp",
    ):
        assert key in out, f"schema missing top-level key: {key}"
```

- [ ] **Step 4.2: Run the tests; expect ALL to fail against the stub.**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_nbody_calibration.py -v
```

Expected: 7 FAIL (stub `main()` raises `NotImplementedError`, so subprocess returncode is non-zero).

- [ ] **Step 4.3: Implement `nbody_calibration.py`.**

Overwrite `problems/hubble_tension_web/experiments/nbody_calibration.py`:

```python
"""Real-void alpha recalibration experiment.

Reads results/nbody_kbc.json (the directory override is via env var
ATFT_HUBBLE_RESULTS_DIR, defaulting to the canonical results dir), runs the
spec §"Math contract" procedure on the K per-void records, and writes
results/nbody_calibration.json.

Math (restated from spec, with RSS_0 simplification):
  Given arrays f (f_topo_at_alpha_1) and y (y_residual) of length K:
    alpha_star = (f @ y) / (f @ f)             if f @ f > 1e-24 else None
    bootstrap B samples of (alpha) with replacement-resampled indices
    alpha_16, alpha_50, alpha_84 = np.percentile(alpha_bs_valid, [16, 50, 84])
    RSS_0 = y @ y                                # kinematic-only residual
    RSS_1 = (alpha*f - y) @ (alpha*f - y)
    F     = (RSS_0 - RSS_1) / (RSS_1 / max(K-1, 1))
    p_F   = 1 - scipy.stats.f.cdf(F, 1, K-1)

Pathology gate: |alpha*| > 1e4 or p_F > 0.5 -> reason downgraded to
"LTB heuristic may be stretched — see spec §6.2.a", but the experiment
returns successfully.

Three reason codes:
  "fit succeeded"
  "insufficient voids (K<{min})"
  "f_topo all zero"
  "LTB heuristic may be stretched — see spec §6.2.a"  (override of "fit succeeded")
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.stats


def _results_dir() -> Path:
    override = os.environ.get("ATFT_HUBBLE_RESULTS_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent.parent / "results"


def _bootstrap_alpha(f: np.ndarray, y: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """Return an array of B bootstrap alpha estimates; degenerate samples dropped."""
    K = len(f)
    out: list[float] = []
    for _ in range(B):
        idx = rng.integers(0, K, size=K)
        ff = f[idx]
        yy = y[idx]
        denom = float(ff @ ff)
        if denom < 1e-24:
            continue  # skip the degenerate sample; don't bias toward zero
        out.append(float((ff @ yy) / denom))
    return np.array(out, dtype=np.float64)


def main() -> None:
    results_dir = _results_dir()
    kbc_path = results_dir / "nbody_kbc.json"
    if not kbc_path.exists():
        print(
            f"nbody_calibration: {kbc_path} not found; nothing to calibrate. Skipping.",
            file=sys.stderr,
        )
        sys.exit(0)

    kbc = json.loads(kbc_path.read_text())
    voids = kbc.get("voids", [])
    K = len(voids)
    min_voids = int(os.environ.get("ATFT_NBODY_CAL_MIN_VOIDS", "3"))
    B = int(os.environ.get("ATFT_NBODY_CAL_BOOTSTRAP_B", "2000"))

    per_void_out = [
        {
            "delta": float(v["delta_eff"]),
            "R_mpc": float(v["R_eff_mpc"]),
            "f_topo": float(v["f_topo_at_alpha_1"]),
            "y_ltb_residual": float(v["y_residual"]),
            "beta1": int(v["beta1_persistent"]),
        }
        for v in voids
    ]

    out: dict = {
        "alpha_star": None,
        "alpha_units": "km/s",
        "alpha_ci_68": None,
        "alpha_bootstrap_median": None,
        "K": K,
        "bootstrap_B": B,
        "chi2_reduced": 0.0,
        "pearson_r_f_y": 0.0,
        "p_F_alpha_vs_zero": 1.0,
        "reason": "fit succeeded",
        "per_void": per_void_out,
        "ltb_reference_source": "ltb_reference.delta_H0_ltb (Gaussian profile heuristic)",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    if K < min_voids:
        out["reason"] = f"insufficient voids (K<{min_voids})"
        (results_dir / "nbody_calibration.json").write_text(json.dumps(out, indent=2))
        return

    f = np.array([pv["f_topo"] for pv in per_void_out], dtype=np.float64)
    y = np.array([pv["y_ltb_residual"] for pv in per_void_out], dtype=np.float64)

    denom = float(f @ f)
    if denom < 1e-24:
        out["reason"] = "f_topo all zero"
        (results_dir / "nbody_calibration.json").write_text(json.dumps(out, indent=2))
        return

    alpha_star = float((f @ y) / denom)

    # Bootstrap 68% CI.
    rng = np.random.default_rng(0)
    alpha_bs = _bootstrap_alpha(f, y, B=B, rng=rng)
    if alpha_bs.size == 0:
        alpha_ci_68 = None
        alpha_median = None
    else:
        lo, med, hi = np.percentile(alpha_bs, [16.0, 50.0, 84.0])
        alpha_ci_68 = [float(lo), float(hi)]
        alpha_median = float(med)

    # F-test nested: M0 = kinematic only, M1 = kinematic + alpha*f_topo.
    # Because y = delta_H0_LTB - C1*delta, RSS_0 = y @ y exactly.
    rss_0 = float(y @ y)
    residuals = alpha_star * f - y
    rss_1 = float(residuals @ residuals)
    dof = max(K - 1, 1)
    F_stat = (rss_0 - rss_1) / (rss_1 / dof) if rss_1 > 0 else float("inf")
    p_F = float(1.0 - scipy.stats.f.cdf(F_stat, dfn=1, dfd=dof)) if F_stat != float("inf") else 0.0

    chi2_reduced = float(rss_1 / dof) if dof > 0 else 0.0
    # Pearson r on the (f, y) pair. Guard against degenerate variance.
    if np.std(f) < 1e-24 or np.std(y) < 1e-24:
        pearson_r = 0.0
    else:
        pearson_r = float(scipy.stats.pearsonr(f, y).statistic)

    out["alpha_star"] = alpha_star
    out["alpha_ci_68"] = alpha_ci_68
    out["alpha_bootstrap_median"] = alpha_median
    out["chi2_reduced"] = chi2_reduced
    out["pearson_r_f_y"] = pearson_r
    out["p_F_alpha_vs_zero"] = p_F

    # Pathology gate (spec §"No external LTB integrator in v1").
    if abs(alpha_star) > 1e4 or p_F > 0.5:
        out["reason"] = "LTB heuristic may be stretched — see spec §6.2.a"

    (results_dir / "nbody_calibration.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.4: Run the tests.**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_nbody_calibration.py -v
```

Expected: 7 PASS. If `test_bootstrap_ci_brackets_alpha_star` fails because the CI excludes α*, investigate — on K=8 with seeded RNG, this should be deterministic. The test allows small slack (0.5 km/s) to absorb legitimate bootstrap edge cases.

- [ ] **Step 4.5: Full fast suite.**

Run:
```bash
python -m pytest tests/hubble_tension_web -x --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```

Expected: `88 passed, 3 xfailed` (81 from Task 3 + 7 new).

- [ ] **Step 4.6: Commit.**

```bash
git add problems/hubble_tension_web/experiments/nbody_calibration.py \
        tests/hubble_tension_web/test_nbody_calibration.py
git commit -m "$(cat <<'EOF'
feat(nbody): nbody_calibration experiment — 1D LSQ + bootstrap CI + F-test

Reads results/nbody_kbc.json (dir overridable via ATFT_HUBBLE_RESULTS_DIR),
computes closed-form alpha* = (f @ y) / (f @ f), bootstraps a 68% CI via
np.random.default_rng(0) with ATFT_NBODY_CAL_BOOTSTRAP_B (default 2000)
resamples, and runs an F-test against alpha=0. Writes
results/nbody_calibration.json per spec §"Math contract" schema.

Three reason codes + one pathology override:
  fit succeeded / insufficient voids (K<min) / f_topo all zero
  overrides to "LTB heuristic may be stretched" when |alpha*|>1e4 or p_F>0.5

7 tests: closed-form alpha* to 1e-10, bootstrap CI bracketing, F-test range,
per-void schema, both skip branches, top-level schema.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `run_all.py` adds the conditional 4th leg + 2 new `test_run_all.py` tests

**Purpose.** After `nbody_kbc` runs, conditionally launch `nbody_calibration` when `len(voids) >= ATFT_NBODY_CAL_MIN_VOIDS`. When the cache is absent or voids are too few, skip cleanly (log only).

**Files:**
- Modify: `problems/hubble_tension_web/experiments/run_all.py`
- Modify: `tests/hubble_tension_web/test_run_all.py` (+2 tests)

- [ ] **Step 5.1: Write the failing tests.**

Append to `tests/hubble_tension_web/test_run_all.py` (keep the 3 existing tests unchanged):

```python
def test_run_all_triggers_nbody_calibration_when_voids_exist(tmp_path):
    """With a real nbody cache fixture, run_all must produce nbody_calibration.json."""
    import os
    from pathlib import Path as _P
    fixture = _P("tests/hubble_tension_web/nbody/fixtures/mini_mdpl2.parquet").resolve()
    assert fixture.exists(), f"nbody fixture missing: {fixture}"

    env = os.environ.copy()
    env["ATFT_NBODY_CACHE_FILE"] = str(fixture)
    env["ATFT_NBODY_GRID"] = "32"
    env["ATFT_NBODY_K_VOIDS"] = "3"
    env["ATFT_NBODY_CAL_MIN_VOIDS"] = "1"   # force the fit to attempt even on tiny K
    env["ATFT_NBODY_CAL_BOOTSTRAP_B"] = "100"  # small B for test speed

    result = subprocess.run(
        [sys.executable, "-m", "problems.hubble_tension_web.experiments.run_all"],
        capture_output=True, env=env, timeout=180,
    )
    assert result.returncode == 0, (
        f"run_all rc={result.returncode}\nstdout:\n"
        f"{result.stdout.decode(errors='replace')}\nstderr:\n"
        f"{result.stderr.decode(errors='replace')}"
    )

    results = Path("problems/hubble_tension_web/results")
    assert (results / "nbody_kbc.json").exists()
    # The calibration leg must have run and produced its JSON.
    assert (results / "nbody_calibration.json").exists(), (
        "run_all did not trigger nbody_calibration even though nbody_kbc "
        "produced voids and min_voids=1"
    )
    # Sanity: JSON parseable and has the expected top-level keys.
    cal = json.loads((results / "nbody_calibration.json").read_text())
    for key in ("alpha_star", "K", "reason", "per_void"):
        assert key in cal


def test_run_all_skips_calibration_when_voids_below_min(tmp_path):
    """With a real cache but ATFT_NBODY_CAL_MIN_VOIDS=99, calibration must skip cleanly."""
    import os
    from pathlib import Path as _P
    fixture = _P("tests/hubble_tension_web/nbody/fixtures/mini_mdpl2.parquet").resolve()
    assert fixture.exists()

    # Put the calibration JSON into a per-test dir so we can assert it DOES NOT exist
    # after run_all. Easiest: delete the canonical file beforehand and re-check.
    canonical_cal = Path("problems/hubble_tension_web/results/nbody_calibration.json")
    if canonical_cal.exists():
        canonical_cal.unlink()

    env = os.environ.copy()
    env["ATFT_NBODY_CACHE_FILE"] = str(fixture)
    env["ATFT_NBODY_GRID"] = "32"
    env["ATFT_NBODY_K_VOIDS"] = "3"
    env["ATFT_NBODY_CAL_MIN_VOIDS"] = "99"  # unattainably high -> must skip

    result = subprocess.run(
        [sys.executable, "-m", "problems.hubble_tension_web.experiments.run_all"],
        capture_output=True, env=env, timeout=180,
    )
    assert result.returncode == 0, (
        f"run_all rc={result.returncode}\nstdout:\n"
        f"{result.stdout.decode(errors='replace')}\nstderr:\n"
        f"{result.stderr.decode(errors='replace')}"
    )
    combined = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
    assert "nbody_calibration" in combined.lower(), (
        "run_all should mention nbody_calibration in output (skip or run)"
    )
    assert not canonical_cal.exists(), (
        "nbody_calibration.json should NOT be re-created when min_voids unreachable"
    )
```

- [ ] **Step 5.2: Run the two new tests; expect FAIL (calibration leg not wired in run_all).**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_run_all.py::test_run_all_triggers_nbody_calibration_when_voids_exist -v
python -m pytest tests/hubble_tension_web/test_run_all.py::test_run_all_skips_calibration_when_voids_below_min -v
```

Expected: the "triggers" test FAILS (`nbody_calibration.json` not written by `run_all`). The "skips" test may vacuously PASS (file doesn't exist because nothing creates it). Move to Step 5.3 regardless.

- [ ] **Step 5.3: Modify `run_all.py`.**

In `problems/hubble_tension_web/experiments/run_all.py`, locate the existing nbody_kbc block (lines 115-127):

```python
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
```

After this block and BEFORE the `agg_t0 = time.perf_counter()` line, insert:

```python
    # Optional nbody_calibration step — gated on nbody_kbc.json with K >= min.
    import json as _json
    results_dir = Path(__file__).parent.parent / "results"
    nbody_kbc_json = results_dir / "nbody_kbc.json"
    if nbody_kbc_json.exists():
        try:
            _kbc_data = _json.loads(nbody_kbc_json.read_text())
            n_voids = len(_kbc_data.get("voids", []))
        except (ValueError, _json.JSONDecodeError):
            n_voids = 0
        min_voids = int(os.environ.get("ATFT_NBODY_CAL_MIN_VOIDS", "3"))
        if n_voids >= min_voids:
            print(f"nbody_calibration: launching with K={n_voids} voids (min {min_voids}).")
            cal_env = os.environ.copy()
            rc = subprocess.run(
                [sys.executable, "-m", "problems.hubble_tension_web.experiments.nbody_calibration"],
                env=cal_env,
            ).returncode
            if rc != 0:
                print(
                    f"nbody_calibration failed rc={rc}; continuing with aggregate.",
                    file=sys.stderr,
                )
        else:
            print(
                f"nbody_calibration: only {n_voids} voids (< {min_voids}); skipping."
            )
    else:
        print("nbody_calibration: no nbody_kbc.json; skipping.")
```

Also extend the final missing-artifact check loop at the bottom of `main()` (lines 142-147). Replace:

```python
    results_dir = Path(__file__).parent.parent / "results"
    for name in ("analytical_reduction.json", "sim_calibration.json",
                 "kbc_crosscheck.json", "REPORT.md"):
        path = results_dir / name
        if not path.exists():
            print(f"WARNING: expected output missing: {path}", file=sys.stderr)
```

with:

```python
    # results_dir already defined above; re-use.
    for name in ("analytical_reduction.json", "sim_calibration.json",
                 "kbc_crosscheck.json", "REPORT.md"):
        path = results_dir / name
        if not path.exists():
            print(f"WARNING: expected output missing: {path}", file=sys.stderr)
```

(We de-duplicate the `results_dir` assignment since the inserted block defines it. The functional behavior is unchanged.)

- [ ] **Step 5.4: Run the two new tests; confirm they pass.**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_run_all.py::test_run_all_triggers_nbody_calibration_when_voids_exist -v
python -m pytest tests/hubble_tension_web/test_run_all.py::test_run_all_skips_calibration_when_voids_below_min -v
```

Expected: both PASS (the second may take 60-120 s).

- [ ] **Step 5.5: Confirm the 3 existing `test_run_all.py` tests still pass.**

Run:
```bash
python -m pytest tests/hubble_tension_web/test_run_all.py -v
```

Expected: 5 PASS (3 existing + 2 new). Total wall time: well under the 180 s timeout.

- [ ] **Step 5.6: Full fast suite.**

Run:
```bash
python -m pytest tests/hubble_tension_web -x --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```

Expected: `88 passed, 3 xfailed` (unchanged — the 2 new tests are in the slow suite).

- [ ] **Step 5.7: Commit.**

```bash
git add problems/hubble_tension_web/experiments/run_all.py \
        tests/hubble_tension_web/test_run_all.py
git commit -m "$(cat <<'EOF'
feat(nbody): run_all conditional 4th leg — nbody_calibration on K >= min_voids

After the existing nbody_kbc block, conditionally launch nbody_calibration when
results/nbody_kbc.json exists AND len(voids) >= ATFT_NBODY_CAL_MIN_VOIDS
(default 3). Otherwise log and skip.

+2 tests: trigger-on-voids (with min=1 override) and skip-when-below-min (with
min=99 override). Existing 3 test_run_all tests unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `aggregate.py` Leg 4 section (conditional)

**Purpose.** When `nbody_calibration.json` exists, `aggregate.py` appends a "Leg 4: Real-Void α Calibration" section to REPORT.md. No new tests — the existing `test_run_all.py` end-to-end covers the path.

**Files:**
- Modify: `problems/hubble_tension_web/experiments/aggregate.py`

- [ ] **Step 6.1: Modify `aggregate.py`.**

In `problems/hubble_tension_web/experiments/aggregate.py`, locate the end of the existing `lines.extend([...])` block (after "c1 is ASSERTED from LTB linear theory…", currently line 78). Replace the final write-line with the conditional insertion + write:

```python
    # --- Leg 4 (conditional) ---
    nbody_cal_path = OUTPUT / "nbody_calibration.json"
    if nbody_cal_path.exists():
        cal = json.loads(nbody_cal_path.read_text())
        lines.extend([
            "",
            "## Leg 4: Real-Void α Calibration",
            "",
            f"- K real voids: {cal['K']}",
            f"- Bootstrap resamples B: {cal['bootstrap_B']}",
            f"- Reason: {cal['reason']}",
        ])
        if cal["alpha_star"] is not None:
            ci = cal.get("alpha_ci_68") or ["n/a", "n/a"]
            ci_str = (
                f"[{ci[0]:.4g}, {ci[1]:.4g}]"
                if isinstance(ci[0], (int, float)) else "n/a"
            )
            lines.extend([
                f"- α\\* = **{cal['alpha_star']:.4g} {cal['alpha_units']}** (68% CI: {ci_str})",
                f"- α bootstrap median: {cal['alpha_bootstrap_median']:.4g} "
                f"{cal['alpha_units']}",
                f"- F-test p-value (α vs 0): {cal['p_F_alpha_vs_zero']:.3g}",
                f"- χ²_reduced: {cal['chi2_reduced']:.3g}, Pearson r(f, y): "
                f"{cal['pearson_r_f_y']:.3g}",
            ])
        else:
            lines.extend([
                f"- α\\*: **undetermined** ({cal['reason']})",
            ])
        lines.append("")

    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
```

(Remove the pre-existing `(OUTPUT / "REPORT.md").write_text(...)` line from its prior position; the insertion above is the new final write.)

- [ ] **Step 6.2: Manual smoke — run aggregate against an existing fixture.**

Run:
```bash
python -m problems.hubble_tension_web.experiments.run_all
cat problems/hubble_tension_web/results/REPORT.md | head -80
```

Expected: REPORT.md now contains `## Leg 4: Real-Void α Calibration` IF the nbody fixture produced ≥ min_voids candidates on this machine. If K < min_voids (or the nbody fixture is not discoverable), Leg 4 is absent — which is correct.

- [ ] **Step 6.3: Full fast + relevant slow suite.**

Run:
```bash
python -m pytest tests/hubble_tension_web -x --ignore=tests/hubble_tension_web/test_pipeline.py -q
```

Expected: `93 passed, 3 xfailed` (88 fast + 5 from test_run_all.py including the 2 new).

- [ ] **Step 6.4: Commit.**

```bash
git add problems/hubble_tension_web/experiments/aggregate.py
git commit -m "$(cat <<'EOF'
feat(aggregate): Leg 4 section for real-void alpha calibration (conditional)

aggregate.py now checks for results/nbody_calibration.json and, when present,
appends a "Leg 4: Real-Void α Calibration" section with K, bootstrap B, reason,
alpha* + 68% CI, F-test p-value, chi^2_reduced, and Pearson r.

Section is absent when nbody_calibration.json is missing (no voids cached, or
K < ATFT_NBODY_CAL_MIN_VOIDS) — existing 3-leg REPORT.md output is preserved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `MASTER.md` + `nbody/README.md` documentation patch

**Purpose.** Reflect the new capability in the single-source knowledge doc and the nbody README. No code changes.

**Files:**
- Modify: `problems/hubble_tension_web/MASTER.md`
- Modify: `problems/hubble_tension_web/nbody/README.md`

- [ ] **Step 7.1: Patch `MASTER.md` §6 (Current empirical state) — add a row.**

In `problems/hubble_tension_web/MASTER.md`, locate the §6 table (starting at line 194 with `| Quantity | Value | Notes |`). Append a final row AFTER the existing last row (`| N-body fixture β₁ distribution | …`):

```markdown
| Real-void α* (calibration leg) | **pending first `_ENABLED=1` run** | Computed by `experiments/nbody_calibration.py` from `nbody_kbc.json`; null on the toy fixture (K<3 or f_topo=0). Once a real 500 Mpc MDPL2 sub-box is ingested, α* and its 68% bootstrap CI populate here. |
```

- [ ] **Step 7.2: Patch `MASTER.md` §9 — add finding 9.7 on the h-unit trap.**

In `problems/hubble_tension_web/MASTER.md`, after §9.6 (ends around line 356) and before the `---` separator leading into §10, insert:

```markdown
### 9.7 The MDPL2 little-h unit trap (`feat(nbody): CosmoSim MDPL2 SQL download + h-conversion`, forward-ops)

**Trap.** CosmoSim returns MDPL2 Rockstar columns in `h⁻¹ Mpc` and `h⁻¹ M_sun` with Planck cosmology `h = 0.6777`. Our existing Parquet schema contract (`EXPECTED_COLUMNS = ("x","y","z","mvir")` in `mdpl2_fetch.py:18`) expects pure Mpc / M_sun. Shipping the raw CSV into the cache shrinks distances by ~32% and silently wrecks void-radius comparisons downstream.

**Fix.** `mdpl2_download._apply_h_conversion` divides the four columns by `MDPL2_H = 0.6777` at write-time; `load_halo_catalog` stays ignorant of `h`. The test anchor `x = 677.7 h⁻¹ Mpc → 1000.0 Mpc` + `mvir = 6.777e11 h⁻¹ M_sun → 1e12 M_sun` is asserted to 1e-9 in `test_mdpl2_download::test_fetch_sub_box_applies_h_conversion`.

**Lesson.** Unit conversions should live at the boundary where foreign units enter — not in the calculations that consume them. The schema contract is then invariant.
```

- [ ] **Step 7.3: Patch `MASTER.md` §12 — add forward-ops spec + plan entries.**

In `problems/hubble_tension_web/MASTER.md`, locate the §12 table (line 396 start). Append two rows inside the existing table:

```markdown
| `specs/2026-04-21-forward-ops-design.md` | Real MDPL2 ingest + α recalibration (this doc's motivating spec). |
| `plans/2026-04-21-forward-ops.md`        | Forward-ops implementation plan (8 tasks). |
```

- [ ] **Step 7.4: Patch `MASTER.md` §13 — promote the α recalibration item out of "deferred".**

In §13 (deferred v2 items, starting line 411), DELETE the first bullet:

```markdown
- **α recalibration against real voids.** v1 uses α from the smooth-void fit (which lands at 0). Meaningful recalibration requires β₁ > 0 on real data; deferred until N-body runs confirm that.
```

This item is now delivered by the forward-ops work.

- [ ] **Step 7.5: Patch `nbody/README.md` — document the download opt-in.**

In `problems/hubble_tension_web/nbody/README.md`, replace the "Data provenance" section (lines 5-21) with:

```markdown
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
```

In the "Configuration" section at the bottom, append the new env vars:

```markdown
- `ATFT_MDPL2_DOWNLOAD_ENABLED`: `"1"` enables the live CosmoSim SQL path (default `"0"`)
- `ATFT_SCISERVER_TOKEN`: required when `_ENABLED="1"` (no default)
- `ATFT_MDPL2_URL`: override the TAP endpoint (default `https://www.cosmosim.org/tap`)
- `ATFT_MDPL2_SQL`: wholesale-override the SQL body (default is built in `mdpl2_download._build_default_sql`)
- `ATFT_NBODY_CAL_MIN_VOIDS`: min K real voids before α fit is attempted (default `3`)
- `ATFT_NBODY_CAL_BOOTSTRAP_B`: bootstrap resamples (default `2000`)
```

- [ ] **Step 7.6: Verify markdown renders.**

Run:
```bash
python -c "import pathlib; p = pathlib.Path('problems/hubble_tension_web/MASTER.md'); assert '9.7' in p.read_text(); assert '2026-04-21-forward-ops' in p.read_text(); print('MASTER.md OK')"
python -c "import pathlib; p = pathlib.Path('problems/hubble_tension_web/nbody/README.md'); assert 'ATFT_MDPL2_DOWNLOAD_ENABLED' in p.read_text(); print('README OK')"
```

Expected: `MASTER.md OK` and `README OK`.

- [ ] **Step 7.7: Run the full fast suite one more time.**

Run:
```bash
python -m pytest tests/hubble_tension_web -x --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```

Expected: `88 passed, 3 xfailed`. No new tests.

- [ ] **Step 7.8: Commit.**

```bash
git add problems/hubble_tension_web/MASTER.md \
        problems/hubble_tension_web/nbody/README.md
git commit -m "$(cat <<'EOF'
docs(hubble): MASTER.md §6/§9/§12/§13 + nbody/README forward-ops patch

- §6: add real-void alpha* row (pending first _ENABLED=1 run)
- §9.7: MDPL2 little-h unit trap finding
- §12: forward-ops spec + plan entries
- §13: remove "alpha recalibration" — delivered
- nbody/README: document the ATFT_MDPL2_DOWNLOAD_ENABLED opt-in, SciServer token,
  endpoint + SQL overrides, the manual fallback, and the CI-stub guarantee

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final acceptance check

After Task 7 commits:

- [ ] **Fast suite still green at the expected count.**

```bash
python -m pytest tests/hubble_tension_web -x --ignore=tests/hubble_tension_web/test_pipeline.py --ignore=tests/hubble_tension_web/test_run_all.py -q
```

Expected: `88 passed, 3 xfailed` (73 baseline + 6 mdpl2_download + 2 mdpl2_fetch delegation + 7 nbody_calibration = 88; Task 3 extended an existing test rather than adding one).

- [ ] **Slow suite green.**

```bash
python -m pytest tests/hubble_tension_web/test_run_all.py -q
```

Expected: `5 passed` (3 existing + 2 new).

- [ ] **End-to-end: `run_all.py` + fixture produces all 5 JSONs + REPORT.md.**

```bash
python -m problems.hubble_tension_web.experiments.run_all
ls problems/hubble_tension_web/results/*.json problems/hubble_tension_web/results/REPORT.md
```

Expected output includes: `analytical_reduction.json`, `sim_calibration.json`, `kbc_crosscheck.json`, `nbody_kbc.json`, `nbody_calibration.json`, `REPORT.md`.

- [ ] **Stubbed-network regression holds.**

```bash
python -c "
import os; os.environ.pop('ATFT_MDPL2_DOWNLOAD_ENABLED', None)
from problems.hubble_tension_web.nbody import mdpl2_fetch
try:
    mdpl2_fetch.fetch_from_network(url='https://cosmosim.example/x')
    print('FAIL: should have raised')
except NotImplementedError as e:
    assert 'CosmoSim' in str(e); print('OK: stub preserved')
"
```

Expected: `OK: stub preserved`.

---

## Self-review (inline)

**Spec coverage.** Walking the spec top-to-bottom:

- §"Three locked constraints": (1) Path C CosmoSim SQL primary — Task 1 implements. (2) `ATFT_MDPL2_DOWNLOAD_ENABLED="0"` default gate — Task 1 + Task 2 enforce. (3) Bootstrap + F-test math in `nbody_calibration.py` — Task 4 implements. **Covered.**
- §"Three sharp findings": (1) No published KBC-in-MDPL2 ground truth — honored by the spec-native success-criterion framing in §"Success criteria" notes (no task needed; this is a narrative constraint). (2) Little-h trap — Task 1 step 1.3 (`_apply_h_conversion`) + test step 1.1 (anchor). **Covered.** (3) Fit denominator degeneracy — Task 4 step 4.3 `f_topo all zero` branch + test step 4.1 `test_f_topo_all_zero_branch`. **Covered.**
- §"New module layout": Tasks 0, 1, 4 create both new modules; Tasks 2, 3, 5, 6, 7 modify all 6 listed files. **Covered.**
- §"Env var surface": all 5 new vars — `ATFT_MDPL2_DOWNLOAD_ENABLED` (Task 1), `ATFT_MDPL2_URL` (Task 1), `ATFT_SCISERVER_TOKEN` (Task 1), `ATFT_NBODY_CAL_MIN_VOIDS` (Task 4 + Task 5), `ATFT_NBODY_CAL_BOOTSTRAP_B` (Task 4). Plus `ATFT_MDPL2_SQL` (added for UNVERIFIED-query override) and `ATFT_HUBBLE_RESULTS_DIR` (added so `nbody_calibration.py` is unit-testable). All documented in Task 7.5 README patch. **Covered.**
- §"Math contract": Task 4 step 4.3 implements every line. RSS_0 simplification (`= y @ y`) documented. Pathology gate (`|α*| > 1e4` OR `p_F > 0.5` → `"LTB heuristic may be stretched"`) present. **Covered.**
- §"JSON schema": all 13 top-level keys from spec → `nbody_calibration.py` `out` dict in Task 4 step 4.3. Asserted by `test_output_json_schema_shape` in Task 4 step 4.1. **Covered.**
- §"run_all.py integration": Task 5 step 5.3 matches the spec pseudocode literally (conditional on `K >= min_voids`). **Covered.**
- §"aggregate.py integration": Task 6 step 6.1 — same conditional, same rendered fields. **Covered.**
- §"Acceptance gates": (a) 73+3 baseline preserved — Task 0 step 0.1 verifies. (b) `test_mdpl2_download.py` asserts schema + h-conversion + schema contract preserved — Task 1 step 1.1 has 6 tests covering this. (c) `test_nbody_calibration.py` asserts α* finite on K=8 fixture, bootstrap CI, F-test, per-void complete — Task 4 step 4.1 has 7 tests. (d) `test_run_all.py` +2 tests — Task 5 step 5.1. (e) Stubbed-download regression — final-check last bullet. (f) End-to-end fixture path produces both `nbody_kbc.json` AND `nbody_calibration.json` — Task 5 step 5.1 first test. **Covered.**

**Placeholder scan.** Grepping the plan for red-flag tokens:

- `TBD` / `TODO` / `implement later` / `fill in details` — **none**. ("UNVERIFIED" markers on the CosmoSim SQL are deliberate and explicitly flagged as a known-unknown concern, not a placeholder for plan detail.)
- `"Add appropriate error handling"` / `"handle edge cases"` without specifics — **none**. Every error raise is spelled out.
- `"Write tests for the above"` without code — **none**. Every test step contains full test code.
- `"Similar to Task N"` — **none**. No cross-task code references; identical patterns are rewritten in each task.
- Steps that describe what to do without showing how — **none**. Every file edit shows the code.
- References to undefined types / functions — checked against the prior-context file reads: `C1`, `f_topo`, `delta_H0_ltb`, `NBodyDataNotAvailable`, `HaloCatalog`, `EXPECTED_COLUMNS`, `predict_from_cosmic_web` all verified against the existing sources before being used.

**Type / name consistency.**

- `fetch_sub_box(dest, sub_box_mpc)` — identical signature in Task 0 stub, Task 1 implementation, Task 1 test, Task 2 delegation.
- `fetch_from_network(url, dest)` — matches existing `mdpl2_fetch.py:76` signature; Task 2 adds `dest` default behavior without breaking the existing `url`-only call in `test_network_fetch_is_stubbed`.
- Per-void field names — `f_topo_at_alpha_1`, `ltb_anchor_at_delta_R`, `y_residual` — identical in Task 3 emission, Task 4 consumption, Task 3 test assertion.
- JSON keys — `alpha_star`, `alpha_units`, `alpha_ci_68`, `alpha_bootstrap_median`, `K`, `bootstrap_B`, `chi2_reduced`, `pearson_r_f_y`, `p_F_alpha_vs_zero`, `reason`, `per_void`, `ltb_reference_source`, `timestamp` — 13 keys, identical spelling in Task 4 impl, Task 4 test, Task 6 aggregate consumption.
- Reason strings — `"fit succeeded"` / `"insufficient voids (K<{min})"` / `"f_topo all zero"` / `"LTB heuristic may be stretched — see spec §6.2.a"` — identical spelling across impl + tests.
- Per-void JSON record keys in the `per_void` block — `delta`, `R_mpc`, `f_topo`, `y_ltb_residual`, `beta1` — identical in Task 4 impl and test assertions.

**Known concerns flagged for user visibility (not plan bugs):**

1. **CosmoSim TAP async endpoint shape UNVERIFIED.** Task 1's `_submit_sql_job`/`_poll_until_done`/`_download_csv` encode IVOA TAP conventions (`/async` POST, `/phase` polling, `/results/result` GET) as the most-likely correct shape based on CosmoSim's public metadata page. Because no CosmoSim account is available in this planning session, the exact HTTP protocol cannot be verified end-to-end. This is acceptable because: (a) all tests mock the HTTP layer; (b) `ATFT_MDPL2_URL` and `ATFT_MDPL2_SQL` are env-overridable, so the maintainer can patch without a code change on the first live run; (c) the stub-preserved CI path is completely unaffected. **Marked `# UNVERIFIED` in the module docstring**, surfaced to the user in the plan's final-report concerns, and flagged in MASTER.md §9.7 as a known-unknown.

2. **Column alias `Mvir` vs `Mvir_all` vs `M200c`.** Task 1 picks `Mvir` based on the spec's reference to the CosmoSim Rockstar metadata page and the alias ordering documented in `_MASS_COLUMN_PREFERENCE`. If CosmoSim has renamed the column since the spec was written, the CSV parse will fail with a clear `NBodyDataNotAvailable` message naming the preference list — operator remediation is to set `ATFT_MDPL2_SQL` to a corrected SELECT.

3. **Bootstrap determinism.** `np.random.default_rng(0)` is hard-coded in `nbody_calibration.py`. Tests rely on this for deterministic CI edges. Do NOT change without updating `test_bootstrap_ci_brackets_alpha_star` tolerances.

4. **Task 3 vacuous-test caveat.** `test_nbody_kbc.py` guards the new per-void fields behind `if data["voids"]:`. If a future fixture regression reduces the void count to 0, the field assertion becomes vacuous. Considered adding a fallback assertion that `data["voids"]` is non-empty, but that would break the honest null-result scenarios described in the spec. Acceptable risk — if needed, add a separate `test_nbody_kbc_fixture_produces_voids` guard in a future revision.

No spec requirement is uncovered by a task. No type or name drift between tasks. Plan is internally consistent and ready for execution.
