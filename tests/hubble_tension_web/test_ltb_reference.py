"""Tests for the LTB reference module.

The target curve is constructed to SATISFY the LTB leading-order expansion
(ΔH₀ = -H0·δ/3 at leading order), carry a nonlinear correction that is NOT
drawn from the functional's ansatz, and respect physical limits at δ=0 and R→∞.
"""
import numpy as np
import pytest


def test_ltb_reference_leading_order_matches_minus_third_H0_delta():
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb, H0_GLOBAL
    # At δ small and R large (wide profile), reference should ≈ -H0·δ/3.
    y = delta_H0_ltb(delta=-1e-4, R_mpc=1e6)
    expected = (-H0_GLOBAL / 3.0) * (-1e-4)
    # Tolerance: finite-R correction goes to zero as R→∞. At R=1e6 Mpc it's ~8e-8, well under rel=1e-3.
    assert y == pytest.approx(expected, rel=1e-3)


def test_ltb_reference_sign_matches_convention():
    """Void (δ < 0) implies ΔH0_LTB > 0."""
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
    assert delta_H0_ltb(delta=-0.2, R_mpc=300.0) > 0
    assert delta_H0_ltb(delta=-0.05, R_mpc=200.0) > 0


def test_ltb_reference_vanishes_at_zero_delta():
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
    assert delta_H0_ltb(delta=0.0, R_mpc=300.0) == pytest.approx(0.0, abs=1e-12)


def test_ltb_reference_has_nonlinear_correction():
    """The δ³ correction must make the curve NON-proportional to δ for |δ| not small."""
    from problems.hubble_tension_web.ltb_reference import delta_H0_ltb
    y1 = delta_H0_ltb(delta=-0.10, R_mpc=300.0)
    y2 = delta_H0_ltb(delta=-0.20, R_mpc=300.0)
    ratio = y2 / y1
    # If purely linear, ratio would be exactly 2.0. With δ³ correction, ratio != 2.0.
    assert abs(ratio - 2.0) > 1e-3, (
        f"ratio = {ratio} is too close to linear; δ³ correction term appears absent."
    )


def test_ltb_reference_is_functional_independent():
    """Static check: the module must not import the functional or ansatz helpers.

    This guards non-circularity at the module level. If sim_calibration.py accidentally
    routes the ansatz through ltb_reference, this test catches the import.
    """
    import problems.hubble_tension_web.ltb_reference as ltb_mod
    src = open(ltb_mod.__file__).read()
    for forbidden in [
        "from problems.hubble_tension_web.functional",
        "import problems.hubble_tension_web.functional",
        "from problems.hubble_tension_web.laplacian",
        "from problems.hubble_tension_web.spectrum",
    ]:
        assert forbidden not in src, (
            f"circularity guard: ltb_reference.py contains '{forbidden}'"
        )
