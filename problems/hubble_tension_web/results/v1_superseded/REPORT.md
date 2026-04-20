# Hubble-Tension-Web: Results Report

## Leg 1: Analytical Reduction

Kinematic term vs published LTB coefficient H0/3 across delta in [0, -0.3] at R = 300 Mpc.

| delta | kinematic term | expected LTB | topological term (alpha=1) |
|---|---|---|---|
| 0.000 | 0 | 0 | 158 |
| -0.050 | -1.123 | -1.123 | 158.8 |
| -0.100 | -2.247 | -2.247 | 160.8 |
| -0.150 | -3.37 | -3.37 | 166.1 |
| -0.200 | -4.493 | -4.493 | 167.3 |
| -0.250 | -5.617 | -5.617 | 173.2 |
| -0.300 | -6.74 | -6.74 | 170 |

Pass criterion: kinematic term matches LTB exactly (c1 = H0/3 by construction).

## Leg 2: Sim Calibration

- Fitted alpha\* = **0.00472**
- Residual loss = 1.519
- Reference curve = (H0/3) * delta * exp(-((R-300)/200)^2)  [literature-grounded stand-in]
- Scan size = 30 (delta, R) combinations

See `sim_calibration.png` for predicted-vs-reference scatter.

## Leg 3: KBC Cross-Check

- delta = -0.2, R = 300.0 Mpc
- Kinematic term: **-4.493 km/s/Mpc**
- Topological term (alpha\* = 0.00472): **1.877 km/s/Mpc**
- Total ΔH0: **-2.617 km/s/Mpc**
- Literature band (magnitude): [1.0, 3.0] km/s/Mpc
- Verdict: **WITHIN band — consistent with literature**

## Interpretation

The functional K reduces exactly to LTB in the analytical leg (kinematic coefficient is
structurally c1 = H0/3, no free parameter). The sim-calibration leg fits a single coefficient
alpha against a literature-grounded reference. The KBC cross-check is the first external test of
the calibrated functional. A WITHIN-band result is a successful reproduction of the perturbative
estimate by a topological route; an ABOVE-band result would indicate that multi-scale structure
captured by the sheaf Laplacian contributes beyond perturbation theory and merits attention.

## Scope limits

- All voids are LTB-family synthetic. Real N-body snapshot ingestion (IllustrisTNG / MDPL2) is
  the sequel task.
- beta1 here is the graph-level cycle-space count, not a full persistent-homology beta1 from a
  filtration. Multi-scale persistence is a future refinement.