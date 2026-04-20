"""Analytical reduction: verify the functional reduces to LTB kinematic in the smooth limit.

Primary observation (honest reading):
  Across this smooth-void scan (LTB-family synthetics), beta_1_persistent sits at
  the noise floor (= 0) for every delta, so topological_term = alpha * 0 = 0
  identically by arithmetic. The kinematic term alone carries delta_H0 throughout
  this scan. To contrast "beta_1 = 0 regime" vs "beta_1 > 0 regime" a separate
  fixture with real persistent 1-cycles (e.g. ring or torus-perturbed void) is
  needed; not included in this experiment.
Secondary (tautology / sign guard):
  kinematic_term = C1 * delta = -(H0/3) * delta to machine precision.
Tertiary:
  d(delta_H0)/d(-delta) > 0 at fixed R for delta < 0 (monotonicity). Trivially
  satisfied here because topological_term = 0 and kinematic_term is linear in
  delta with negative coefficient; documented for completeness.

This experiment no longer claims to DERIVE c1 from spec(L_F). See REWORK spec 5.1.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problems.hubble_tension_web.functional import C1, H0_GLOBAL, predict_from_cosmic_web
from problems.hubble_tension_web.synthetic import generate_synthetic_void
from problems.hubble_tension_web.types import VoidParameters

OUTPUT = Path(__file__).parent.parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    deltas = np.linspace(-1e-3, -0.3, 9)   # skip exact 0 to avoid VoidParameters guard
    R = 300.0
    records = []
    for d in deltas:
        params = VoidParameters(delta=float(d), R_mpc=R)
        web = generate_synthetic_void(
            params, n_points=1500, box_mpc=800.0, rng_seed=42,
        )
        h = predict_from_cosmic_web(
            web=web, params=params, alpha=1.0, k=8, stalk_dim=8, k_spec=16,
        )
        expected_kin = C1 * float(d)
        records.append(dict(
            delta=float(d),
            R_mpc=R,
            delta_H0=h.delta_H0,
            kinematic_term=h.kinematic_term,
            topological_term=h.topological_term,
            expected_kin=float(expected_kin),
            kin_tautology_residual=float(h.kinematic_term - expected_kin),
            ratio_topo_over_kin=float(
                h.topological_term / h.kinematic_term if abs(h.kinematic_term) > 1e-12 else np.nan
            ),
        ))

    by_absd = sorted(records, key=lambda r: abs(r["delta"]))
    deltaH0_sorted = [r["delta_H0"] for r in by_absd]
    monotone_nondec = all(
        deltaH0_sorted[i + 1] + 1e-6 >= deltaH0_sorted[i]
        for i in range(len(deltaH0_sorted) - 1)
    )

    max_taut = max(abs(r["kin_tautology_residual"]) for r in records)

    out = dict(
        primary_observation="beta1_persistent = 0 at every scan point (smooth-void regime); topological_term = 0 identically by arithmetic; kinematic_term alone carries delta_H0",
        secondary_assertion=f"max |kinematic_term - C1*delta| = {max_taut:.2e} (should be ~0)",
        tertiary_assertion=f"delta_H0 monotone non-decreasing in |delta|: {monotone_nondec} (trivially so since topo=0, kin linear in delta)",
        records=records,
    )
    (OUTPUT / "analytical_reduction.json").write_text(json.dumps(out, indent=2))

    d_arr = np.array([r["delta"] for r in records])
    k_arr = np.array([r["kinematic_term"] for r in records])
    t_arr = np.array([r["topological_term"] for r in records])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d_arr, k_arr, "o-", label="kinematic (c1*delta, corrected sign)")
    ax.plot(d_arr, C1 * d_arr, "--", label=f"C1*delta expected, C1={C1:.2f}")
    ax.plot(d_arr, t_arr, "s-", alpha=0.6, label="topological (alpha=1)")
    ax.set_xlabel("delta")
    ax.set_ylabel("delta_H0 component [km/s/Mpc]")
    ax.legend()
    ax.set_title("Analytical reduction - kinematic = -H0*delta/3, topological bounded")
    fig.tight_layout()
    fig.savefig(OUTPUT / "analytical_reduction.png", dpi=120)


if __name__ == "__main__":
    main()
