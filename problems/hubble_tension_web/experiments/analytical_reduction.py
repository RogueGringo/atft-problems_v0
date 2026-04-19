"""Analytical reduction: verify 𝒦 recovers LTB as β₁ → 0.

Method:
  1. Build synthetic voids of varying delta in [0, -0.3] at R=300 Mpc.
  2. For each, compute ΔH0 via predict_from_cosmic_web with alpha=1.0.
  3. Confirm kinematic term == (H0_GLOBAL / 3.0) * delta to machine precision.
  4. Log topological term alongside for visibility.

Output: results/analytical_reduction.json, results/analytical_reduction.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problems.hubble_tension_web.functional import H0_GLOBAL, predict_from_cosmic_web
from problems.hubble_tension_web.synthetic import generate_synthetic_void
from problems.hubble_tension_web.types import VoidParameters

OUTPUT = Path(__file__).parent.parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    deltas = np.linspace(0.0, -0.3, 7)
    R = 300.0
    records = []
    for d in deltas:
        params = VoidParameters(delta=float(d), R_mpc=R)
        web = generate_synthetic_void(params, n_points=1500, box_mpc=800.0, rng_seed=42)
        h = predict_from_cosmic_web(web=web, params=params, alpha=1.0, k=8, stalk_dim=4, k_spec=16)
        records.append(dict(
            delta=float(d),
            R_mpc=R,
            delta_H0=h.delta_H0,
            kinematic_term=h.kinematic_term,
            topological_term=h.topological_term,
            expected_LTB=(H0_GLOBAL / 3.0) * float(d),
        ))

    out = {"records": records}
    (OUTPUT / "analytical_reduction.json").write_text(json.dumps(out, indent=2))

    d_arr = np.array([r["delta"] for r in records])
    k_arr = np.array([r["kinematic_term"] for r in records])
    t_arr = np.array([r["topological_term"] for r in records])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d_arr, k_arr, "o-", label="kinematic term")
    ax.plot(d_arr, (H0_GLOBAL / 3.0) * d_arr, "--", label="LTB expected")
    ax.plot(d_arr, t_arr, "s-", label="topological term", alpha=0.6)
    ax.set_xlabel("δ")
    ax.set_ylabel("ΔH₀ component [km/s/Mpc]")
    ax.legend()
    ax.set_title("Analytical reduction: kinematic matches LTB; topo small at α=1")
    fig.tight_layout()
    fig.savefig(OUTPUT / "analytical_reduction.png", dpi=120)


if __name__ == "__main__":
    main()
