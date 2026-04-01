import numpy as np
import pytest

def test_crystal_distance_identical():
    from arm.prime.compare import crystal_distance
    from arm.void.formats import Crystal
    c1 = Crystal(0.22, 0.42, 0.36, 59.0, "a")
    c2 = Crystal(0.22, 0.42, 0.36, 59.0, "b")
    assert crystal_distance(c1, c2) == pytest.approx(0.0)

def test_crystal_distance_different():
    from arm.prime.compare import crystal_distance
    from arm.void.formats import Crystal
    c1 = Crystal(0.22, 0.42, 0.36, 59.0, "text")
    c2 = Crystal(0.05, 0.95, 0.00, 5.0, "simple")
    d = crystal_distance(c1, c2)
    assert d > 0.5

def test_universality_pass():
    from arm.prime.compare import universality_test
    from arm.void.formats import Crystal
    crystals = [
        Crystal(0.22, 0.42, 0.36, 59.0, "a"),
        Crystal(0.23, 0.41, 0.36, 60.0, "b"),
        Crystal(0.22, 0.43, 0.35, 58.0, "c"),
    ]
    result = universality_test(crystals, threshold=0.05)
    assert result["universal"] is True

def test_universality_fail():
    from arm.prime.compare import universality_test
    from arm.void.formats import Crystal
    crystals = [
        Crystal(0.22, 0.42, 0.36, 59.0, "complex"),
        Crystal(0.05, 0.95, 0.00, 5.0, "simple"),
    ]
    result = universality_test(crystals, threshold=0.05)
    assert result["universal"] is False

def test_barcode_distance_identical():
    from arm.prime.compare import barcode_distance
    bars = np.array([[0, 1.0], [0, 2.0], [0, 5.0]])
    assert barcode_distance(bars, bars) == pytest.approx(0.0)

def test_barcode_distance_different():
    from arm.prime.compare import barcode_distance
    bars1 = np.array([[0, 1.0], [0, 2.0]])
    bars2 = np.array([[0, 3.0], [0, 4.0]])
    d = barcode_distance(bars1, bars2)
    assert d > 0
