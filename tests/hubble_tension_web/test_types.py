import numpy as np
import pytest


def test_local_cosmic_web_requires_matching_shapes():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    positions = np.zeros((5, 3))
    envs = [Environment.VOID, Environment.WALL, Environment.FILAMENT]  # wrong length
    with pytest.raises(ValueError):
        LocalCosmicWeb(positions=positions, environments=envs)


def test_void_parameters_rejects_positive_delta():
    from problems.hubble_tension_web.types import VoidParameters
    with pytest.raises(ValueError):
        VoidParameters(delta=0.2, R_mpc=300.0)


def test_spectral_summary_holds_spectrum_and_bettis():
    from problems.hubble_tension_web.types import SpectralSummary
    s = SpectralSummary(spectrum=np.array([0.0, 0.1, 0.3]), beta0=1, beta1=2, lambda_min=0.1)
    assert s.beta0 == 1 and s.beta1 == 2 and s.lambda_min == pytest.approx(0.1)


def test_hubble_shift_carries_value_and_units():
    from problems.hubble_tension_web.types import HubbleShift
    h = HubbleShift(delta_H0=5.2, kinematic_term=2.5, topological_term=2.7)
    assert h.delta_H0 == pytest.approx(5.2)
    assert h.kinematic_term + h.topological_term == pytest.approx(h.delta_H0)
