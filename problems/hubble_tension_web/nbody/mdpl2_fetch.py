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
