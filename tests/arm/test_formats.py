import numpy as np
import json
import pytest

def test_point_cloud_creation():
    from arm.void.formats import PointCloud
    data = np.array([[0, 1, 3, 0], [1, 1, 0, 3]], dtype=np.int8)
    pc = PointCloud(data=data, source="test", hash="abc123")
    assert pc.data.shape == (2, 4)
    assert pc.source == "test"

def test_point_cloud_hash_computed():
    from arm.void.formats import PointCloud
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    pc = PointCloud.from_array(data, source="test")
    assert len(pc.hash) == 64

def test_crystal_validation_pass():
    from arm.void.formats import Crystal
    c = Crystal(void_ratio=0.222, identity_ratio=0.417, prime_ratio=0.361,
                eff_rank=59.0, source="test")
    c.validate()

def test_crystal_validation_fail():
    from arm.void.formats import Crystal
    c = Crystal(void_ratio=0.5, identity_ratio=0.5, prime_ratio=0.5,
                eff_rank=1.0, source="bad")
    with pytest.raises(AssertionError):
        c.validate()

def test_persistence_diagram_creation():
    from arm.void.formats import PersistenceDiagram
    h0 = np.array([[0.0, 0.5], [0.1, 1.2], [0.0, float('inf')]])
    h1 = np.empty((0, 2))
    pd = PersistenceDiagram(h0=h0, h1=h1, filtration_range=(0.0, 5.0))
    assert pd.h0.shape == (3, 2)
    assert pd.h1.shape == (0, 2)

def test_experiment_record_to_json():
    from arm.void.formats import ExperimentRecord
    rec = ExperimentRecord(
        id="ARM-001", run=1, series="test", timestamp="2026-04-01T00:00:00Z",
        hypothesis="test hyp", protocol="test proto", input_hash="abc123",
        annotations=[], result={"gini": 0.5}, comparison={},
        verdict="PASS", notes="test note"
    )
    j = rec.to_json()
    loaded = json.loads(j)
    assert loaded["id"] == "ARM-001"
    assert loaded["run"] == 1
    assert loaded["verdict"] == "PASS"

def test_experiment_record_from_json():
    from arm.void.formats import ExperimentRecord
    rec = ExperimentRecord(
        id="ARM-001", run=1, series="test", timestamp="2026-04-01T00:00:00Z",
        hypothesis="h", protocol="p", input_hash="x",
        annotations=[], result={}, comparison={}, verdict="PASS", notes=""
    )
    j = rec.to_json()
    rec2 = ExperimentRecord.from_json(j)
    assert rec2.id == rec.id
    assert rec2.run == rec.run
