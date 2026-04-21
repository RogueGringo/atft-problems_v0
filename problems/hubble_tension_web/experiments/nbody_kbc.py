"""N-body KBC experiment: run the full pipeline on MDPL2-style halos.

Reads a Parquet halo cache, classifies the volume with the tidal-tensor
T-web, finds K void candidates, runs predict_from_cosmic_web on each,
and writes a JSON summary + an optional matplotlib histogram of the
beta1_persistent distribution.

Configured via environment variables so run_all.py can drive it:
  ATFT_NBODY_CACHE_FILE   - Parquet file path (required)
  ATFT_NBODY_GRID         - T-web grid size (default 128; fixture uses 32)
  ATFT_NBODY_LAMBDA_TH    - tidal eigenvalue threshold (default 0.0)
  ATFT_NBODY_K_VOIDS      - number of candidate voids (default 5)
  ATFT_NBODY_MASS_CUT     - halo mass cut in M_sun (default 0.0 = none)
  ATFT_NBODY_OUTPUT_JSON  - output JSON path
                             (default problems/.../results/nbody_kbc.json)

The experiment does NOT recalibrate alpha. It reuses sim_calibration.json
if present; otherwise alpha=0 (per spec v1: headline is beta1, not delta_H0).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problems.hubble_tension_web.functional import predict_from_cosmic_web
from problems.hubble_tension_web.nbody import (
    DEFAULT_LAMBDA_TH,
    DEFAULT_MASS_CUT,
    DEFAULT_N_GRID,
    NBodyDataNotAvailable,
)
from problems.hubble_tension_web.nbody.cosmic_web_from_halos import assemble
from problems.hubble_tension_web.nbody.mdpl2_fetch import load_halo_catalog
from problems.hubble_tension_web.nbody.tidal_tensor import classify, cic_deposit
from problems.hubble_tension_web.nbody.void_finder import find_voids


DEFAULT_K: int = 5
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent / "results" / "nbody_kbc.json"
)


def _alpha_from_sim_calibration() -> float:
    """Reuse alpha* from sim_calibration.json if present; otherwise 0.0."""
    sc = Path(__file__).parent.parent / "results" / "sim_calibration.json"
    if not sc.exists():
        return 0.0
    try:
        return float(json.loads(sc.read_text())["alpha_star"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def _run() -> dict:
    cache_file = os.environ.get("ATFT_NBODY_CACHE_FILE")
    if not cache_file:
        raise NBodyDataNotAvailable(
            "ATFT_NBODY_CACHE_FILE not set and no default cache located. "
            "See problems/hubble_tension_web/nbody/README.md."
        )
    cache_path = Path(cache_file)
    n_grid = int(os.environ.get("ATFT_NBODY_GRID", str(DEFAULT_N_GRID)))
    lambda_th = float(os.environ.get("ATFT_NBODY_LAMBDA_TH", str(DEFAULT_LAMBDA_TH)))
    k_voids = int(os.environ.get("ATFT_NBODY_K_VOIDS", str(DEFAULT_K)))
    mass_cut = float(os.environ.get("ATFT_NBODY_MASS_CUT", "0.0"))

    halos = load_halo_catalog(cache_path, mass_cut=mass_cut)
    env_grid, meta = classify(
        positions=halos.positions, masses=halos.masses,
        n_grid=n_grid, box_mpc=halos.box_mpc, lambda_th=lambda_th,
    )
    rho = cic_deposit(halos.positions, halos.masses,
                      n_grid=n_grid, box_mpc=halos.box_mpc)

    smoothing_mpc = min(10.0, halos.box_mpc / 5.0)
    voids = find_voids(
        rho=rho, box_mpc=halos.box_mpc,
        smoothing_mpc=smoothing_mpc,
        delta_threshold=-0.1,
        max_radius_mpc=halos.box_mpc / 2.0,
        k_top=k_voids,
    )

    alpha = _alpha_from_sim_calibration()

    per_void: list[dict] = []
    for idx, cand in enumerate(voids):
        try:
            web, params = assemble(halos=halos, env_grid=env_grid, candidate=cand)
        except ValueError:
            continue
        if web.positions.shape[0] < 4:
            continue
        k_nn = min(8, web.positions.shape[0] - 1)
        h = predict_from_cosmic_web(
            web=web, params=params, alpha=alpha,
            k=k_nn, stalk_dim=8, k_spec=min(16, web.positions.shape[0] - 1),
            rng_seed=0,
        )
        from problems.hubble_tension_web.graph import build_typed_graph
        from problems.hubble_tension_web.laplacian import typed_sheaf_laplacian
        from problems.hubble_tension_web.spectrum import summarize_spectrum
        n, edges = build_typed_graph(web, k=k_nn)
        L = typed_sheaf_laplacian(
            positions=web.positions, n=n, edges=edges, stalk_dim=8,
            rng_seed=0, environments=web.environments,
        )
        summary = summarize_spectrum(
            L=L, n_nodes=n, edges=edges, positions=web.positions,
            k_spec=min(16, web.positions.shape[0] - 1),
        )
        per_void.append(dict(
            idx=idx,
            center_mpc=list(cand.center_mpc),
            N_halos=int(web.positions.shape[0]),
            delta_eff=float(cand.delta_eff),
            R_eff_mpc=float(cand.radius_mpc),
            beta0=int(summary.beta0),
            beta1_persistent=int(summary.beta1),
            lambda_min=float(summary.lambda_min),
            delta_H0_total=float(h.delta_H0),
            kinematic_term=float(h.kinematic_term),
            topological_term=float(h.topological_term),
        ))

    beta1s = [v["beta1_persistent"] for v in per_void]
    dist = dict(
        count_nonzero=int(sum(1 for b in beta1s if b > 0)),
        count_total=int(len(beta1s)),
        median=float(statistics.median(beta1s)) if beta1s else 0.0,
        max=int(max(beta1s)) if beta1s else 0,
    )

    return dict(
        cache_source=str(cache_path.resolve()),
        grid_N=int(n_grid),
        lambda_th=float(lambda_th),
        K=int(k_voids),
        alpha_used=float(alpha),
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        voids=per_void,
        beta1_distribution=dist,
    )


def main() -> None:
    try:
        out = _run()
    except NBodyDataNotAvailable as e:
        print(f"nbody_kbc: cache not available; skipping. ({e})", file=sys.stderr)
        sys.exit(0)

    json_path = Path(
        os.environ.get("ATFT_NBODY_OUTPUT_JSON", str(DEFAULT_OUTPUT))
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2))

    if out["voids"]:
        png_path = Path(str(json_path).replace(".json", ".png"))
        betas = np.array([v["beta1_persistent"] for v in out["voids"]])
        fig, ax = plt.subplots(figsize=(6, 4))
        if betas.max() > betas.min():
            ax.hist(betas, bins=np.arange(betas.min(), betas.max() + 2) - 0.5)
        else:
            ax.hist(betas, bins=[betas.min() - 0.5, betas.min() + 0.5])
        ax.set_xlabel("beta_1_persistent")
        ax.set_ylabel("count")
        ax.set_title(
            f"N-body KBC: beta_1 across {len(betas)} candidate voids"
        )
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)

    print(json.dumps(out["beta1_distribution"], indent=2))


if __name__ == "__main__":
    main()
