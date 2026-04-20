"""Sim calibration: fit alpha by matching predicted delta_H0 against the LTB reference curve.

Procedure (non-circular, spec 6.3):
  For each (delta, R) in the scan:
    1. Compute delta_H0_LTB(delta, R) from ltb_reference (independent of functional's ansatz).
    2. Compute c1*delta (kinematic term with corrected sign).
    3. Residual y = delta_H0_LTB - c1*delta (this is the NONLINEAR LTB correction).
    4. Compute f_topo(delta, R) by running the functional pipeline at alpha=1.0 and reading off
       the topological term.
  Least-squares fit:  alpha* = argmin sum (alpha * f_topo - y)^2  = <f_topo, y> / <f_topo, f_topo>.

Output: results/sim_calibration.json, results/sim_calibration.png.

Parallel execution (perf pass, 2026-04-20):
  The (delta, R) configurations are independent — no shared state. We dispatch
  through multiprocessing.Pool.imap_unordered with a module-level _scan_one
  worker (Windows spawn requires module-level picklable callable). Results are
  sorted by (delta, R) before the LSQ fit so the dot-product is deterministic.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problems.hubble_tension_web.functional import C1, H0_GLOBAL, predict_from_cosmic_web
from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
from problems.hubble_tension_web.synthetic import generate_synthetic_void
from problems.hubble_tension_web.types import VoidParameters

OUTPUT = Path(__file__).parent.parent / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

# If the scan has fewer than this many configs, skip the pool (spawn cost
# dominates). The full production scan has 30; mini-scans in tests may have 4.
_POOL_MIN_CONFIGS = 6

# Production point count per config; tests may override via the 3-tuple form.
_PRODUCTION_N_POINTS = 1500


def _scan_one(delta_R_n: tuple) -> dict:
    """Compute one (delta, R) sim-calibration row. Module-level for pool spawn.

    Accepts either a 2-tuple (delta, R) — uses _PRODUCTION_N_POINTS — or a
    3-tuple (delta, R, n_points) for tests.

    Returns a dict with keys: delta, R, ltb_full, kinematic, y, f_topo.
    """
    if len(delta_R_n) == 3:
        d, R, n_points = delta_R_n
    else:
        d, R = delta_R_n
        n_points = _PRODUCTION_N_POINTS

    params = VoidParameters(delta=float(d), R_mpc=float(R))
    box = max(2.5 * R, 800.0)
    web = generate_synthetic_void(
        params, n_points=int(n_points), box_mpc=box,
        rng_seed=abs(int(1000 * d + R)) + 1,
    )
    h1 = predict_from_cosmic_web(
        web=web, params=params, alpha=1.0, k=8, stalk_dim=8, k_spec=16,
    )
    f_topo_val = h1.topological_term
    ltb_full = delta_H0_ltb(delta=float(d), R_mpc=float(R))
    kin = C1 * float(d)
    y = ltb_full - kin
    return dict(
        delta=float(d), R=float(R),
        ltb_full=float(ltb_full),
        kinematic=float(kin),
        y=float(y),
        f_topo=float(f_topo_val),
    )


def _run_scan(configs: list) -> list:
    """Dispatch the scan. Pool if we have enough configs to amortize spawn cost."""
    if len(configs) < _POOL_MIN_CONFIGS:
        return [_scan_one(c) for c in configs]

    n_workers = min(os.cpu_count() or 1, len(configs))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(_scan_one, configs))
    return results


def main() -> None:
    deltas = np.array([-0.05, -0.10, -0.15, -0.20, -0.25, -0.30])
    radii = np.array([150.0, 250.0, 300.0, 400.0, 500.0])
    configs = [(float(d), float(R)) for d in deltas for R in radii]

    raw = _run_scan(configs)
    # Sort by (delta, R) — determinism for the LSQ fit regardless of worker
    # scheduling.
    scan = sorted(raw, key=lambda r: (r["delta"], r["R"]))

    f = np.array([s["f_topo"] for s in scan])
    y = np.array([s["y"] for s in scan])
    denom = float(f @ f)
    if denom < 1e-24:
        alpha_star = 0.0
        note = "f_topo identically zero across scan; alpha undetermined, set to 0."
    else:
        alpha_star = float((f @ y) / denom)
        note = "least-squares fit against LTB residual."

    residuals = alpha_star * f - y
    mse = float((residuals ** 2).mean())
    r_squared = float(1.0 - (residuals @ residuals) / max((y @ y), 1e-24))

    out = dict(
        alpha_star=alpha_star,
        alpha_units="km/s",
        mse=mse,
        r_squared=r_squared,
        note=note,
        reference_source="ltb_reference.delta_H0_ltb (Gaussian profile, delta^3 series + finite-R)",
        scan=scan,
    )
    (OUTPUT / "sim_calibration.json").write_text(json.dumps(out, indent=2))

    pred = np.array([s["kinematic"] + alpha_star * s["f_topo"] for s in scan])
    ref = np.array([s["ltb_full"] for s in scan])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ref, pred, s=25)
    lim = [min(ref.min(), pred.min()) - 0.5, max(ref.max(), pred.max()) + 0.5]
    ax.plot(lim, lim, "--", alpha=0.6, label="y = x")
    ax.set_xlabel("LTB reference delta_H0 [km/s/Mpc]")
    ax.set_ylabel("predicted delta_H0 [km/s/Mpc]")
    ax.set_title(f"Sim calibration (non-circular): alpha* = {alpha_star:.4g} km/s, R^2 = {r_squared:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "sim_calibration.png", dpi=120)


if __name__ == "__main__":
    main()
