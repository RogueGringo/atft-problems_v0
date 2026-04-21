"""Real-void alpha recalibration experiment.

Reads results/nbody_kbc.json (the directory override is via env var
ATFT_HUBBLE_RESULTS_DIR, defaulting to the canonical results dir), runs the
spec §"Math contract" procedure on the K per-void records, and writes
results/nbody_calibration.json.

Math (restated from spec, with RSS_0 simplification):
  Given arrays f (f_topo_at_alpha_1) and y (y_residual) of length K:
    alpha_star = (f @ y) / (f @ f)             if f @ f > 1e-24 else None
    bootstrap B samples of (alpha) with replacement-resampled indices
    alpha_16, alpha_50, alpha_84 = np.percentile(alpha_bs_valid, [16, 50, 84])
    RSS_0 = y @ y                                # kinematic-only residual
    RSS_1 = (alpha*f - y) @ (alpha*f - y)
    F     = (RSS_0 - RSS_1) / (RSS_1 / max(K-1, 1))
    p_F   = 1 - scipy.stats.f.cdf(F, 1, K-1)

Pathology gate: |alpha*| > 1e4 or p_F > 0.5 -> reason downgraded to
"LTB heuristic may be stretched — see spec §6.2.a", but the experiment
returns successfully.

Three reason codes:
  "fit succeeded"
  "insufficient voids (K<{min})"
  "f_topo all zero"
  "LTB heuristic may be stretched — see spec §6.2.a"  (override of "fit succeeded")
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.stats


def _results_dir() -> Path:
    override = os.environ.get("ATFT_HUBBLE_RESULTS_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent.parent / "results"


def _bootstrap_alpha(f: np.ndarray, y: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """Return an array of B bootstrap alpha estimates; degenerate samples dropped."""
    K = len(f)
    out: list[float] = []
    for _ in range(B):
        idx = rng.integers(0, K, size=K)
        ff = f[idx]
        yy = y[idx]
        denom = float(ff @ ff)
        if denom < 1e-24:
            continue  # skip the degenerate sample; don't bias toward zero
        out.append(float((ff @ yy) / denom))
    return np.array(out, dtype=np.float64)


def main() -> None:
    results_dir = _results_dir()
    kbc_path = results_dir / "nbody_kbc.json"
    if not kbc_path.exists():
        print(
            f"nbody_calibration: {kbc_path} not found; nothing to calibrate. Skipping.",
            file=sys.stderr,
        )
        sys.exit(0)

    kbc = json.loads(kbc_path.read_text())
    voids = kbc.get("voids", [])
    K = len(voids)
    min_voids = int(os.environ.get("ATFT_NBODY_CAL_MIN_VOIDS", "3"))
    B = int(os.environ.get("ATFT_NBODY_CAL_BOOTSTRAP_B", "2000"))

    per_void_out = [
        {
            "delta": float(v["delta_eff"]),
            "R_mpc": float(v["R_eff_mpc"]),
            "f_topo": float(v["f_topo_at_alpha_1"]),
            "y_ltb_residual": float(v["y_residual"]),
            "beta1": int(v["beta1_persistent"]),
        }
        for v in voids
    ]

    out: dict = {
        "alpha_star": None,
        "alpha_units": "km/s",
        "alpha_ci_68": None,
        "alpha_bootstrap_median": None,
        "K": K,
        "bootstrap_B": B,
        "chi2_reduced": 0.0,
        "pearson_r_f_y": 0.0,
        "p_F_alpha_vs_zero": 1.0,
        "reason": "fit succeeded",
        "per_void": per_void_out,
        "ltb_reference_source": "ltb_reference.delta_H0_ltb (Gaussian profile heuristic)",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    if K < min_voids:
        out["reason"] = f"insufficient voids (K<{min_voids})"
        (results_dir / "nbody_calibration.json").write_text(json.dumps(out, indent=2))
        return

    f = np.array([pv["f_topo"] for pv in per_void_out], dtype=np.float64)
    y = np.array([pv["y_ltb_residual"] for pv in per_void_out], dtype=np.float64)

    denom = float(f @ f)
    if denom < 1e-24:
        out["reason"] = "f_topo all zero"
        (results_dir / "nbody_calibration.json").write_text(json.dumps(out, indent=2))
        return

    alpha_star = float((f @ y) / denom)

    # Bootstrap 68% CI.
    rng = np.random.default_rng(0)
    alpha_bs = _bootstrap_alpha(f, y, B=B, rng=rng)
    if alpha_bs.size == 0:
        alpha_ci_68 = None
        alpha_median = None
    else:
        lo, med, hi = np.percentile(alpha_bs, [16.0, 50.0, 84.0])
        alpha_ci_68 = [float(lo), float(hi)]
        alpha_median = float(med)

    # F-test nested: M0 = kinematic only, M1 = kinematic + alpha*f_topo.
    # Because y = delta_H0_LTB - C1*delta, RSS_0 = y @ y exactly.
    rss_0 = float(y @ y)
    residuals = alpha_star * f - y
    rss_1 = float(residuals @ residuals)
    dof = max(K - 1, 1)
    F_stat = (rss_0 - rss_1) / (rss_1 / dof) if rss_1 > 0 else float("inf")
    p_F = float(1.0 - scipy.stats.f.cdf(F_stat, dfn=1, dfd=dof)) if F_stat != float("inf") else 0.0

    chi2_reduced = float(rss_1 / dof) if dof > 0 else 0.0
    # Pearson r on the (f, y) pair. Guard against degenerate variance.
    if np.std(f) < 1e-24 or np.std(y) < 1e-24:
        pearson_r = 0.0
    else:
        pearson_r = float(scipy.stats.pearsonr(f, y).statistic)

    out["alpha_star"] = alpha_star
    out["alpha_ci_68"] = alpha_ci_68
    out["alpha_bootstrap_median"] = alpha_median
    out["chi2_reduced"] = chi2_reduced
    out["pearson_r_f_y"] = pearson_r
    out["p_F_alpha_vs_zero"] = p_F

    # Pathology gate (spec §"No external LTB integrator in v1").
    if abs(alpha_star) > 1e4 or p_F > 0.5:
        out["reason"] = "LTB heuristic may be stretched — see spec §6.2.a"

    (results_dir / "nbody_calibration.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
