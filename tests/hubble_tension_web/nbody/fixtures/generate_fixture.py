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
# Remove ~85% of halos inside the void sphere to produce a clear under-density.
VOID_DEPLETION_FRACTION: float = 0.85
RNG_SEED: int = 2026


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    xyz = rng.uniform(0.0, BOX_MPC, size=(N_HALOS_TOTAL, 3))

    cx, cy, cz = VOID_CENTER_MPC
    r = np.sqrt((xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2 + (xyz[:, 2] - cz) ** 2)
    in_void = r < VOID_RADIUS_MPC
    u = rng.uniform(0.0, 1.0, size=N_HALOS_TOTAL)
    drop = in_void & (u < VOID_DEPLETION_FRACTION)
    xyz = xyz[~drop]

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
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"wrote {len(df)} halos to {out_path}")


if __name__ == "__main__":
    main()
