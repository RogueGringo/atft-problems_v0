"""Sim calibration: fit alpha by matching predicted ΔH0 to a reference curve.

Reference curve (literature-grounded stand-in for IllustrisTNG / MDPL2 output):
  ΔH0_ref(delta, R) = (H0_GLOBAL / 3) * delta * exp(-((R - 300) / 200)^2)

This encodes the KBC-void literature's ~1-3 km/s/Mpc shift at R ≈ 300 Mpc,
delta ≈ -0.2. Real public-snapshot ingestion is deferred to a sequel task.

Output: results/sim_calibration.json, results/sim_calibration.png.
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


def reference_delta_H0(delta: float, R: float) -> float:
    return (H0_GLOBAL / 3.0) * delta * np.exp(-((R - 300.0) / 200.0) ** 2)


def loss_for_alpha(alpha: float, scan: list[dict]) -> float:
    residuals = []
    for rec in scan:
        h_kin = (H0_GLOBAL / 3.0) * rec["delta"]
        h_topo = alpha * rec["f_topo"]
        pred = h_kin + h_topo
        residuals.append(pred - rec["ref"])
    return float(np.mean(np.array(residuals) ** 2))


def main() -> None:
    deltas = np.array([-0.05, -0.10, -0.15, -0.20, -0.25, -0.30])
    radii = np.array([150.0, 250.0, 300.0, 400.0, 500.0])

    scan: list[dict] = []
    for d in deltas:
        for R in radii:
            params = VoidParameters(delta=float(d), R_mpc=float(R))
            web = generate_synthetic_void(
                params, n_points=1500, box_mpc=max(2.5 * R, 800.0), rng_seed=abs(int(1000 * d + R)) + 1
            )
            # alpha=0 gives kinematic-only; alpha=1 reveals the raw f_topo contribution
            h0 = predict_from_cosmic_web(
                web=web, params=params, alpha=0.0, k=8, stalk_dim=8, k_spec=16
            )
            h1 = predict_from_cosmic_web(
                web=web, params=params, alpha=1.0, k=8, stalk_dim=8, k_spec=16
            )
            f_topo_val = h1.topological_term   # alpha=1 => f_topo_val is raw f_topo
            scan.append(dict(
                delta=float(d),
                R=float(R),
                kinematic=h0.kinematic_term,
                f_topo=float(f_topo_val),
                ref=float(reference_delta_H0(d, R)),
            ))

    # Closed-form least-squares fit of alpha:
    #   residual = alpha * f_topo - (ref - kinematic)
    #   alpha* = <f_topo, (ref - kinematic)> / <f_topo, f_topo>
    f = np.array([s["f_topo"] for s in scan])
    y = np.array([s["ref"] - s["kinematic"] for s in scan])
    alpha_star = float((f @ y) / (f @ f))

    loss = loss_for_alpha(alpha_star, scan)

    out = dict(
        alpha_star=alpha_star,
        loss=loss,
        scan=scan,
        reference_form="(H0/3) * delta * exp(-((R-300)/200)^2)  [literature-grounded stand-in]",
    )
    (OUTPUT / "sim_calibration.json").write_text(json.dumps(out, indent=2))

    pred = np.array([s["kinematic"] + alpha_star * s["f_topo"] for s in scan])
    ref = np.array([s["ref"] for s in scan])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ref, pred, s=25)
    lim = [min(ref.min(), pred.min()) - 0.5, max(ref.max(), pred.max()) + 0.5]
    ax.plot(lim, lim, "--", alpha=0.6)
    ax.set_xlabel("reference ΔH₀ [km/s/Mpc]")
    ax.set_ylabel("predicted ΔH₀ [km/s/Mpc]")
    ax.set_title(f"Sim calibration: α* = {alpha_star:.4g}, loss = {loss:.3g}")
    fig.tight_layout()
    fig.savefig(OUTPUT / "sim_calibration.png", dpi=120)


if __name__ == "__main__":
    main()
