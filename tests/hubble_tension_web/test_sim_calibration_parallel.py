"""Parallel vs sequential equivalence for sim_calibration.

Contract (spec 2026-04-20 §Step 3):
  Running the scan with multiprocessing.Pool must produce the same alpha_star,
  mse, r_squared, and scan list as a sequential run, up to floating-point
  reorder error (we use rtol=1e-12 which leaves ~3 orders of headroom over
  typical matmul roundoff).

This test uses a mini-scan (4 configs) so it runs in <30s.
"""
from __future__ import annotations

import numpy as np


def _mini_scan_sequential(deltas, radii):
    """Copy of the Task-3 scan body run sequentially. Ground-truth reference."""
    from problems.hubble_tension_web.functional import C1, predict_from_cosmic_web
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.types import VoidParameters

    scan = []
    for d in deltas:
        for R in radii:
            params = VoidParameters(delta=float(d), R_mpc=float(R))
            box = max(2.5 * R, 800.0)
            web = generate_synthetic_void(
                params, n_points=300, box_mpc=box,
                rng_seed=abs(int(1000 * d + R)) + 1,
            )
            h1 = predict_from_cosmic_web(
                web=web, params=params, alpha=1.0, k=8, stalk_dim=8, k_spec=16,
            )
            ltb_full = delta_H0_ltb(delta=float(d), R_mpc=float(R))
            kin = C1 * float(d)
            y = ltb_full - kin
            scan.append(dict(
                delta=float(d), R=float(R),
                ltb_full=float(ltb_full), kinematic=float(kin),
                y=float(y), f_topo=float(h1.topological_term),
            ))
    return scan


def test_parallel_scan_matches_sequential() -> None:
    from problems.hubble_tension_web.experiments.sim_calibration import _scan_one

    deltas = [-0.1, -0.2]
    radii = [200.0, 300.0]
    # Mini-configs include n_points=300 for speed.
    configs = [(float(d), float(R), 300) for d in deltas for R in radii]

    import multiprocessing as mp
    with mp.get_context("spawn").Pool(processes=2) as pool:
        par_results = list(pool.imap_unordered(_scan_one, configs))

    par_sorted = sorted(par_results, key=lambda r: (r["delta"], r["R"]))

    seq_results = _mini_scan_sequential(deltas, radii)
    seq_sorted = sorted(seq_results, key=lambda r: (r["delta"], r["R"]))

    assert len(par_sorted) == len(seq_sorted) == 4

    for p, s in zip(par_sorted, seq_sorted):
        assert p["delta"] == s["delta"]
        assert p["R"] == s["R"]
        for key in ("ltb_full", "kinematic", "y", "f_topo"):
            np.testing.assert_allclose(
                p[key], s[key], rtol=1e-12, atol=1e-14,
                err_msg=f"mismatch on field {key} at (delta={p['delta']}, R={p['R']})",
            )
