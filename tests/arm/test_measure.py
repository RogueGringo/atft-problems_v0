import tempfile
import pytest

def test_measure_text_topology():
    from arm.measure import measure
    with tempfile.TemporaryDirectory() as td:
        record = measure(
            source="The topology doesn't care what the structure is.",
            source_type="text", mode="topology", results_dir=td,
        )
        assert record.verdict == "PASS"
        assert "crystal" in record.result
        assert record.result["crystal"]["void"] + \
               record.result["crystal"]["identity"] + \
               record.result["crystal"]["prime"] == pytest.approx(1.0)

def test_measure_csv_topology():
    from arm.measure import measure
    csv = "1,2,3\n4,5,6\n7,8,9\n10,11,12\n13,14,15"
    with tempfile.TemporaryDirectory() as td:
        record = measure(source=csv, source_type="csv", mode="topology", results_dir=td)
        assert record.verdict == "PASS"

def test_measure_auto_detects_text():
    from arm.measure import measure
    with tempfile.TemporaryDirectory() as td:
        record = measure(source="Hello world", source_type="auto", mode="topology", results_dir=td)
        assert record.verdict == "PASS"
