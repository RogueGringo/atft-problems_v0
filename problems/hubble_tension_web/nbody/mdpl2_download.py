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
