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
