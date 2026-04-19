import numpy as np
import pytest


def test_delta_H0_vanishes_with_zero_delta_and_zero_beta1():
    from problems.hubble_tension_web.functional import delta_H0
    h = delta_H0(beta0=1, beta1=0, delta=0.0, R=300.0, lambda_min=0.1, alpha=1.0)
    assert abs(h.delta_H0) < 1e-9
    assert h.kinematic_term == pytest.approx(0.0)
    assert h.topological_term == pytest.approx(0.0)


def test_delta_H0_kinematic_matches_LTB_coefficient():
    from problems.hubble_tension_web.functional import delta_H0, H0_GLOBAL
    h = delta_H0(beta0=1, beta1=0, delta=-0.2, R=300.0, lambda_min=0.1, alpha=0.0)
    assert h.topological_term == pytest.approx(0.0)
    # phase-1: c1 = -H0/3, so kinematic = (-H0/3) * (-0.2) > 0 for a void
    assert h.kinematic_term == pytest.approx((-H0_GLOBAL / 3.0) * (-0.2))


def test_predict_from_cosmic_web_returns_hubble_shift():
    from problems.hubble_tension_web.types import VoidParameters
    from problems.hubble_tension_web.synthetic import generate_synthetic_void
    from problems.hubble_tension_web.functional import predict_from_cosmic_web
    params = VoidParameters(delta=-0.2, R_mpc=300.0)
    web = generate_synthetic_void(params, n_points=500, box_mpc=800.0, rng_seed=7)
    h = predict_from_cosmic_web(web=web, params=params, alpha=1.0, k=6, stalk_dim=4, k_spec=12)
    assert isinstance(h.delta_H0, float)
    # phase-1 sign: void (delta<0) with c1=-H0/3 means kinematic term > 0
    # (local universe in a void expands faster than global — matches observed tension direction)
    assert h.kinematic_term > 0


def test_sign_convention_delta_negative_implies_kinematic_positive():
    """Regression guard: for a void (δ < 0), the kinematic term of ΔH₀ must be POSITIVE.

    LTB leading order: H_local = H₀·(1 − δ/3), so ΔH₀ := H_local − H_global = −H₀·δ/3.
    For δ < 0, ΔH₀ > 0 (local H₀ exceeds global H₀). This matches the observed tension direction.
    """
    from problems.hubble_tension_web.functional import delta_H0
    h = delta_H0(beta0=1, beta1=0, delta=-0.2, R=300.0, lambda_min=0.1, alpha=0.0)
    assert h.topological_term == 0.0
    # Kinematic contribution of a void must be POSITIVE under the corrected sign convention.
    assert h.kinematic_term > 0, (
        f"void (δ=-0.2) must have kinematic_term > 0; got {h.kinematic_term}. "
        f"This is the v1 sign-bug regression guard."
    )
    assert h.delta_H0 > 0
