"""Aggregate the three experiment outputs into REPORT.md (REWORK version)."""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "results"


def main() -> None:
    analytical = json.loads((OUTPUT / "analytical_reduction.json").read_text())
    calib = json.loads((OUTPUT / "sim_calibration.json").read_text())
    kbc = json.loads((OUTPUT / "kbc_crosscheck.json").read_text())

    lines: list[str] = [
        "# Hubble-Tension-Web: Results Report (REWORK, v2)",
        "",
        "**Sign convention:** c1 = -H0/3. delta_H0 > 0 for delta < 0 (voids).",
        "**alpha units:** km/s.",
        "**beta1:** persistent (VR filtration, lifetime > tau*ell_bar, tau=1.5).",
        "",
        "v1 results (no-op Laplacian, single-scale beta1, wrong sign) archived in `v1_superseded/`.",
        "",
        "## Leg 1: Analytical Reduction (consistency check)",
        "",
        analytical["primary_assertion"],
        "",
        analytical["secondary_assertion"],
        "",
        analytical["tertiary_assertion"],
        "",
        "| delta | kinematic | topological (alpha=1) | kin tautology residual |",
        "|---|---|---|---|",
    ]
    for r in analytical["records"]:
        lines.append(
            f"| {r['delta']:.4f} | {r['kinematic_term']:.4g} | {r['topological_term']:.4g} | "
            f"{r['kin_tautology_residual']:.2e} |"
        )

    lines.extend([
        "",
        "## Leg 2: Sim Calibration (non-circular)",
        "",
        f"- Reference source: {calib['reference_source']}",
        f"- Fitted alpha\\* = **{calib['alpha_star']:.4g} {calib['alpha_units']}**",
        f"- MSE = {calib['mse']:.4g}, R^2 = {calib['r_squared']:.3f}",
        f"- Scan size = {len(calib['scan'])} (delta, R) combinations",
        "",
        f"Note: {calib['note']}",
        "",
        "See `sim_calibration.png` for predicted-vs-reference scatter.",
        "",
        "## Leg 3: KBC Cross-Check (signed band)",
        "",
        f"- delta = {kbc['delta']}, R = {kbc['R_mpc']} Mpc",
        f"- Kinematic term: **{kbc['kinematic_term']:.3f} km/s/Mpc**",
        f"- Topological term (alpha\\* = {kbc['alpha_star']:.4g}): **{kbc['topological_term']:.3f} km/s/Mpc**",
        f"- Total delta_H0: **{kbc['delta_H0']:.3f} km/s/Mpc**",
        f"- Literature band (signed): {kbc['literature_band_signed']} km/s/Mpc",
        f"- Verdict: **{kbc['verdict']}**",
        "",
        "## Interpretation",
        "",
        "The functional reduces to LTB in the analytical leg (tautology guard against the v1 sign bug).",
        "Sim calibration fits alpha against the LTB-reference RESIDUAL (not leading term), making the fit",
        "non-circular. KBC cross-check is the first external signed test of the calibrated functional.",
        "",
        "A WITHIN-band result is a successful reproduction of the perturbative KBC estimate by a",
        "topological route. An ABOVE-band result indicates that the sheaf Laplacian picks up structure",
        "perturbation theory omits. A SIGN ERROR result is a regression bug and must block release.",
        "",
        "## Scope limits",
        "",
        "- All voids are LTB-family synthetic. Real N-body snapshot ingestion remains deferred.",
        "- The LTB reference uses closed-form delta^3 + finite-R Gaussian weights; if algebra does not",
        "  match Garcia-Bellido 2008 at pivot points within 5%, fall back to numerical LTB integration.",
        "- c1 is ASSERTED from LTB linear theory, not DERIVED from spec(L_F). See REWORK spec 5.",
    ])

    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
