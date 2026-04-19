"""Aggregate the three experiment outputs into REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "results"


def main() -> None:
    analytical = json.loads((OUTPUT / "analytical_reduction.json").read_text())
    calib = json.loads((OUTPUT / "sim_calibration.json").read_text())
    kbc = json.loads((OUTPUT / "kbc_crosscheck.json").read_text())

    lines: list[str] = [
        "# Hubble-Tension-Web: Results Report",
        "",
        "## Leg 1: Analytical Reduction",
        "",
        "Kinematic term vs published LTB coefficient H0/3 across delta in [0, -0.3] at R = 300 Mpc.",
        "",
        "| delta | kinematic term | expected LTB | topological term (alpha=1) |",
        "|---|---|---|---|",
    ]
    for r in analytical["records"]:
        lines.append(
            f"| {r['delta']:.3f} | {r['kinematic_term']:.4g} | {r['expected_LTB']:.4g} | {r['topological_term']:.4g} |"
        )

    lines.extend([
        "",
        "Pass criterion: kinematic term matches LTB exactly (c1 = H0/3 by construction).",
        "",
        "## Leg 2: Sim Calibration",
        "",
        f"- Fitted alpha\\* = **{calib['alpha_star']:.4g}**",
        f"- Residual loss = {calib['loss']:.4g}",
        f"- Reference curve = {calib['reference_form']}",
        f"- Scan size = {len(calib['scan'])} (delta, R) combinations",
        "",
        "See `sim_calibration.png` for predicted-vs-reference scatter.",
        "",
        "## Leg 3: KBC Cross-Check",
        "",
        f"- delta = {kbc['delta']}, R = {kbc['R_mpc']} Mpc",
        f"- Kinematic term: **{kbc['kinematic_term']:.3f} km/s/Mpc**",
        f"- Topological term (alpha\\* = {kbc['alpha_star']:.4g}): **{kbc['topological_term']:.3f} km/s/Mpc**",
        f"- Total ΔH0: **{kbc['delta_H0']:.3f} km/s/Mpc**",
        f"- Literature band (magnitude): {kbc['literature_band']} km/s/Mpc",
        f"- Verdict: **{kbc['verdict']}**",
        "",
        "## Interpretation",
        "",
        "The functional K reduces exactly to LTB in the analytical leg (kinematic coefficient is",
        "structurally c1 = H0/3, no free parameter). The sim-calibration leg fits a single coefficient",
        "alpha against a literature-grounded reference. The KBC cross-check is the first external test of",
        "the calibrated functional. A WITHIN-band result is a successful reproduction of the perturbative",
        "estimate by a topological route; an ABOVE-band result would indicate that multi-scale structure",
        "captured by the sheaf Laplacian contributes beyond perturbation theory and merits attention.",
        "",
        "## Scope limits",
        "",
        "- All voids are LTB-family synthetic. Real N-body snapshot ingestion (IllustrisTNG / MDPL2) is",
        "  the sequel task.",
        "- beta1 here is the graph-level cycle-space count, not a full persistent-homology beta1 from a",
        "  filtration. Multi-scale persistence is a future refinement.",
    ])

    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
