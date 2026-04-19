"""KBC cross-check: run the calibrated functional on KBC-literature parameters.

Signed band compare (REWORK spec 6.5): band is (+1.0, +3.0) km/s/Mpc.
A negative predicted delta_H0 at delta = -0.2 is an unambiguous
sign-convention failure.
"""
from __future__ import annotations

import json
from pathlib import Path

from problems.hubble_tension_web.functional import predict_from_cosmic_web
from problems.hubble_tension_web.synthetic import generate_synthetic_void
from problems.hubble_tension_web.types import VoidParameters

OUTPUT = Path(__file__).parent.parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

KBC_DELTA = -0.2
KBC_R_MPC = 300.0
# SIGNED band: positive delta_H0 expected for a void under corrected sign convention.
LITERATURE_BAND = (1.0, 3.0)


def verdict(value: float, band: tuple[float, float]) -> str:
    lo, hi = band
    if value < 0:
        return "SIGN ERROR - predicted delta_H0 < 0 for a void; sign convention violated"
    if value < lo:
        return "BELOW band - local-void hypothesis weak for KBC parameters"
    if value > hi:
        return "ABOVE band - topology implies a larger tension contribution than perturbative theory captures"
    return "WITHIN band - consistent with literature"


def main() -> None:
    calib = json.loads((OUTPUT / "sim_calibration.json").read_text())
    alpha_star = float(calib["alpha_star"])

    params = VoidParameters(delta=KBC_DELTA, R_mpc=KBC_R_MPC)
    web = generate_synthetic_void(params, n_points=2500, box_mpc=900.0, rng_seed=2025)
    h = predict_from_cosmic_web(
        web=web, params=params, alpha=alpha_star, k=8, stalk_dim=8, k_spec=16,
    )

    v = verdict(h.delta_H0, LITERATURE_BAND)

    out = dict(
        delta=KBC_DELTA,
        R_mpc=KBC_R_MPC,
        alpha_star=alpha_star,
        alpha_units="km/s",
        delta_H0=h.delta_H0,
        kinematic_term=h.kinematic_term,
        topological_term=h.topological_term,
        literature_band_signed=list(LITERATURE_BAND),
        verdict=v,
    )
    (OUTPUT / "kbc_crosscheck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
