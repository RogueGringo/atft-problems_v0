import numpy as np
import pytest

def test_h0_single_cluster():
    from arm.identity.persistence import compute_h0
    from arm.void.formats import PointCloud
    pts = np.array([[0, 0], [0.1, 0], [0, 0.1]], dtype=np.float64)
    pc = PointCloud.from_array(pts, source="test")
    diagram = compute_h0(pc, eps_max=2.0, n_steps=100)
    assert diagram.h0.shape[0] == 3
    inf_bars = diagram.h0[diagram.h0[:, 1] == float('inf')]
    assert len(inf_bars) == 1

def test_h0_two_clusters():
    from arm.identity.persistence import compute_h0
    from arm.void.formats import PointCloud
    cluster1 = np.array([[0, 0], [0.1, 0], [0, 0.1]], dtype=np.float64)
    cluster2 = np.array([[10, 10], [10.1, 10], [10, 10.1]], dtype=np.float64)
    pts = np.vstack([cluster1, cluster2])
    pc = PointCloud.from_array(pts, source="test")
    diagram = compute_h0(pc, eps_max=20.0, n_steps=200)
    long_bars = diagram.h0[(diagram.h0[:, 1] - diagram.h0[:, 0]) > 1.0]
    assert len(long_bars) >= 2

def test_h0_barcode_sorted():
    from arm.identity.persistence import compute_h0
    from arm.void.formats import PointCloud
    pts = np.random.RandomState(42).randn(20, 3).astype(np.float64)
    pc = PointCloud.from_array(pts, source="test")
    diagram = compute_h0(pc, eps_max=5.0, n_steps=100)
    deaths = diagram.h0[:, 1]
    finite = deaths[np.isfinite(deaths)]
    # Finite deaths should be sorted (bars sorted by death time)
    assert np.all(finite[:-1] <= finite[1:]) or len(finite) <= 1

def test_h0_all_born_at_zero():
    from arm.identity.persistence import compute_h0
    from arm.void.formats import PointCloud
    pts = np.array([[0, 0], [5, 5], [10, 10]], dtype=np.float64)
    pc = PointCloud.from_array(pts, source="test")
    diagram = compute_h0(pc, eps_max=20.0, n_steps=100)
    assert np.all(diagram.h0[:, 0] == 0.0)

def test_filtration_sweep_annotations():
    from arm.identity.persistence import compute_h0
    from arm.void.formats import PointCloud
    pts = np.array([[0, 0], [1, 1]], dtype=np.float64)
    pc = PointCloud.from_array(pts, source="test")
    diagram = compute_h0(pc, eps_max=3.0, n_steps=50)
    assert diagram.filtration_range == (0.0, 3.0)
