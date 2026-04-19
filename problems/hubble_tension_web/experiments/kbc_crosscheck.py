"""KBC cross-check: run the calibrated functional on KBC-literature parameters.

Inputs:
  - delta, R from the Keenan-Barger-Cowie void literature (delta ~= -0.2, R ~= 300 Mpc)
  - alpha* from results/sim_calibration.json

Outputs:
  - delta_H0 prediction, kinematic and topological breakdown
  - Literature comparison band: [1.0, 3.0] km/s/Mpc
  - Verdict: within / above / below band
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
LITERATURE_BAND = (1.0, 3.0)   # km/s/Mpc, magnitude of ΔH0


def verdict(mag: float, band: tuple[float, float]) -> str:
    lo, hi = band
    if mag < lo:
        return "BELOW band — local-void hypothesis weak for KBC parameters"
    if mag > hi:
        return "ABOVE band — topology implies a larger tension contribution than perturbative theory captures"
    return "WITHIN band — consistent with literature"


def main() -> None:
    calib = json.loads((OUTPUT / "sim_calibration.json").read_text())
    alpha_star = float(calib["alpha_star"])

    params = VoidParameters(delta=KBC_DELTA, R_mpc=KBC_R_MPC)
    web = generate_synthetic_void(params, n_points=2500, box_mpc=900.0, rng_seed=2025)
    h = predict_from_cosmic_web(
        web=web, params=params, alpha=alpha_star, k=8, stalk_dim=8, k_spec=16
    )

    mag = abs(h.delta_H0)
    v = verdict(mag, LITERATURE_BAND)

    out = dict(
        delta=KBC_DELTA,
        R_mpc=KBC_R_MPC,
        alpha_star=alpha_star,
        delta_H0=h.delta_H0,
        kinematic_term=h.kinematic_term,
        topological_term=h.topological_term,
        literature_band=LITERATURE_BAND,
        verdict=v,
    )
    (OUTPUT / "kbc_crosscheck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
