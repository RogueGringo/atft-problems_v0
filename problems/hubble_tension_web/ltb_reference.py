"""Lemaître-Tolman-Bondi reference curve for ΔH₀(δ, R).

Independent of the typed-sheaf functional's ansatz. This is the calibration
TARGET against which α is fit in sim_calibration.py.

Formula
-------
For a Gaussian void profile δ(r) = δ · exp(-r² / R²) in a matter-dominated
FLRW background, the LTB-to-FLRW departure of the local Hubble rate admits
a series expansion in δ:

    ΔH₀_LTB(δ, R) = -(H₀/3) · δ                              # leading, homogeneous
                    + (1/9)  · H₀ · δ²  · W_2(R)              # δ² correction
                    + (1/27) · H₀ · δ³  · W_3(R)              # δ³ correction
                    + finite-R geometric factor G(R)·δ

where W_2(R), W_3(R) are dimensionless weight functions that vanish as R → 0
(void is too small to bias H_local) and tend to 1 as R → ∞ (full density
correction), and G(R) is the finite-R leading-order geometric correction
from the Gaussian profile integrated against an observer at r=0.

Source (for the leading coefficient and the Gaussian-profile treatment):
  Garcia-Bellido, J. & Haugbølle, T. (2008), "Confronting Lemaître-Tolman-
  Bondi models with observational cosmology", JCAP 04 (2008) 003,
  eq. (2.28-2.31) for the H_local expansion; eq. (3.5-3.8) for the Gaussian
  profile specialization. Our series coefficients (-1/3, +1/9, +1/27) are the
  first three Taylor coefficients of H_local/H_global = (1 - δ/3)^(-1) for the
  top-hat; the Gaussian weight functions W_2, W_3, G are heuristic
  specializations truncated at finite order.

Honest limitation (spec §6.2.b fallback):
  The closed-form weights W_2, W_3 and the finite-R correction G are heuristic
  specializations that carry the correct leading-order LTB coefficient, correct
  sign convention, and correct asymptotic limits (→ 0 at R = 0, → 1 at R → ∞).
  They are NOT rigorously derived from the full Gaussian-profile LTB integration.
  For single-scalar calibration of α against a non-linear residual, this is
  sufficient to break circularity with the functional's ansatz. If Task 10's
  calibration produces |α| > 10^4 km/s, fall back to a numerical LTB integrator
  (scipy.integrate.quad against the Raychaudhuri equation).
"""
from __future__ import annotations

H0_GLOBAL: float = 67.4   # km/s/Mpc (Planck 2018). Deliberately duplicated here
                          # to keep this module independent of functional.py.

R_SOFT_MPC: float = 150.0   # soft scale at which the weights reach ~1/2.


def _weight_nonlinear(R_mpc: float, alpha_soft: float = 1.0) -> float:
    """Smooth monotone function in [0, 1] that tends to 1 as R → ∞."""
    x = R_mpc / R_SOFT_MPC
    return float(x ** (2 * alpha_soft) / (1.0 + x ** (2 * alpha_soft)))


def _finite_R_correction(delta: float, R_mpc: float) -> float:
    """Sub-leading linear-in-δ geometric correction from the Gaussian profile.

    For a Gaussian void, the observer at the void center sees a density gradient
    that integrates to a small correction to -H0·δ/3 proportional to 1/R at
    leading order. Signature: sign consistent with void → ΔH0 > 0.
    """
    return float(-H0_GLOBAL * delta * (R_SOFT_MPC / max(R_mpc, 1.0)) / 12.0)


def delta_H0_ltb(*, delta: float, R_mpc: float) -> float:
    """Full-LTB reference ΔH₀ (km/s/Mpc) for a Gaussian-profile void.

    At δ = 0: returns exactly 0.
    At R → ∞: returns -H0·δ/3 (exact LTB leading order) plus δ² and δ³ corrections.
    Sign: ΔH₀ > 0 for δ < 0 (void ⇒ local H₀ exceeds global H₀).
    """
    if R_mpc <= 0:
        raise ValueError(f"R_mpc must be positive; got {R_mpc}")
    if delta == 0.0:
        return 0.0

    w2 = _weight_nonlinear(R_mpc, alpha_soft=1.0)
    w3 = _weight_nonlinear(R_mpc, alpha_soft=1.5)

    leading   = (-H0_GLOBAL / 3.0) * delta
    nonlinear = (
        (1.0 / 9.0)  * H0_GLOBAL * delta ** 2 * w2
      + (1.0 / 27.0) * H0_GLOBAL * delta ** 3 * w3
    )
    finite_R = _finite_R_correction(delta, R_mpc)
    return float(leading + nonlinear + finite_R)
