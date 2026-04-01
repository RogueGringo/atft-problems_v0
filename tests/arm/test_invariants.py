import numpy as np
import pytest

def test_gini_uniform():
    from arm.prime.invariants import gini
    bars = np.array([1.0, 1.0, 1.0, 1.0])
    assert abs(gini(bars)) < 1e-6

def test_gini_one_dominant():
    from arm.prime.invariants import gini
    bars = np.array([100.0, 0.01, 0.01, 0.01])
    assert gini(bars) > 0.7

def test_gini_empty():
    from arm.prime.invariants import gini
    assert gini(np.array([])) == 0.0

def test_onset_scale():
    from arm.prime.invariants import onset_scale
    from arm.void.formats import PersistenceDiagram
    h0 = np.array([[0.0, 0.5], [0.0, 1.2], [0.0, float('inf')]])
    pd = PersistenceDiagram(h0=h0, h1=np.empty((0, 2)), filtration_range=(0, 5))
    assert onset_scale(pd) == 0.5

def test_effective_rank():
    from arm.prime.invariants import effective_rank
    data = np.eye(10) + np.random.RandomState(42).randn(10, 10) * 0.01
    er = effective_rank(data)
    assert 8.0 < er < 10.5

def test_effective_rank_rank1():
    from arm.prime.invariants import effective_rank
    v = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
    data = v.T @ v
    er = effective_rank(data)
    assert er < 1.5

def test_crystal_from_topology():
    from arm.prime.invariants import crystal_from_persistence
    from arm.void.formats import PersistenceDiagram
    h0 = np.array([
        [0, 0.1], [0, 0.1], [0, 0.2],
        [0, 1.0], [0, 1.1], [0, 1.2], [0, 1.3],
        [0, 5.0], [0, 6.0],
    ])
    pd = PersistenceDiagram(h0=h0, h1=np.empty((0, 2)), filtration_range=(0, 10))
    crystal = crystal_from_persistence(pd)
    crystal.validate()
    assert crystal.void_ratio + crystal.identity_ratio + crystal.prime_ratio == pytest.approx(1.0)
    assert crystal.void_ratio > 0
    assert crystal.prime_ratio > 0

def test_spectral_gap():
    from arm.prime.invariants import spectral_gap
    eigenvalues = np.array([10.0, 2.0, 1.0, 0.5])
    assert spectral_gap(eigenvalues) == pytest.approx(5.0)
