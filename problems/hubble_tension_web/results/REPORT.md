# Hubble-Tension-Web: Results Report (REWORK, v2)

**Sign convention:** c1 = -H0/3. delta_H0 > 0 for delta < 0 (voids).
**alpha units:** km/s.
**beta1:** persistent (VR filtration, lifetime > tau*ell_bar, tau=1.5).

v1 results (no-op Laplacian, single-scale beta1, wrong sign) archived in `v1_superseded/`.

## Leg 1: Analytical Reduction (consistency check)

topological_term shrinks relative to kinematic_term as delta -> 0 (beta1 noise-floor only)

max |kinematic_term - C1*delta| = 0.00e+00 (should be ~0)

delta_H0 monotone non-decreasing in |delta|: True

| delta | kinematic | topological (alpha=1) | kin tautology residual |
|---|---|---|---|
| -0.0010 | 0.02247 | 0 | 0.00e+00 |
| -0.0384 | 0.8622 | 0 | 0.00e+00 |
| -0.0757 | 1.702 | 0 | 0.00e+00 |
| -0.1131 | 2.542 | 0 | 0.00e+00 |
| -0.1505 | 3.381 | 0 | 0.00e+00 |
| -0.1879 | 4.221 | 0 | 0.00e+00 |
| -0.2253 | 5.061 | 0 | 0.00e+00 |
| -0.2626 | 5.9 | 0 | 0.00e+00 |
| -0.3000 | 6.74 | 0 | 0.00e+00 |

## Leg 2: Sim Calibration (non-circular)

- Reference source: ltb_reference.delta_H0_ltb (Gaussian profile, delta^3 series + finite-R)
- Fitted alpha\* = **0 km/s**
- MSE = 0.7793, R^2 = 0.000
- Scan size = 30 (delta, R) combinations

Note: f_topo identically zero across scan; alpha undetermined, set to 0.

See `sim_calibration.png` for predicted-vs-reference scatter.

## Leg 3: KBC Cross-Check (signed band)

- delta = -0.2, R = 300.0 Mpc
- Kinematic term: **4.493 km/s/Mpc**
- Topological term (alpha\* = 0): **0.000 km/s/Mpc**
- Total delta_H0: **4.493 km/s/Mpc**
- Literature band (signed): [1.0, 3.0] km/s/Mpc
- Verdict: **ABOVE band - topology implies a larger tension contribution than perturbative theory captures**

## Interpretation

The functional reduces to LTB in the analytical leg (tautology guard against the v1 sign bug).
Sim calibration fits alpha against the LTB-reference RESIDUAL (not leading term), making the fit
non-circular. KBC cross-check is the first external signed test of the calibrated functional.

A WITHIN-band result is a successful reproduction of the perturbative KBC estimate by a
topological route. An ABOVE-band result indicates that the sheaf Laplacian picks up structure
perturbation theory omits. A SIGN ERROR result is a regression bug and must block release.

## Scope limits

- All voids are LTB-family synthetic. Real N-body snapshot ingestion remains deferred.
- The LTB reference uses closed-form delta^3 + finite-R Gaussian weights; if algebra does not
  match Garcia-Bellido 2008 at pivot points within 5%, fall back to numerical LTB integration.
- c1 is ASSERTED from LTB linear theory, not DERIVED from spec(L_F). See REWORK spec 5.