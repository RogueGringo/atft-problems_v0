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
       VoidParameters.__post_init__ rejects delta > 0 -- this is the contract.

    Returns an empty web (positions shape (0, 3)) if no halos fall inside.
    """
    center = np.array(candidate.center_mpc, dtype=np.float64)
    rel = halos.positions - center
    r = np.linalg.norm(rel, axis=1)
    inside = r <= candidate.radius_mpc

    positions_rel = rel[inside]
    positions_abs = halos.positions[inside]

    environments = lookup_env_at_positions(
        env_grid=env_grid, positions=positions_abs, box_mpc=halos.box_mpc,
    )

    web = LocalCosmicWeb(positions=positions_rel, environments=environments)
    params = VoidParameters(delta=candidate.delta_eff, R_mpc=candidate.radius_mpc)
    return web, params
