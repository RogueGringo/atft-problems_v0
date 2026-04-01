import numpy as np
import json
import os
import tempfile
import pytest

def test_topology_mode_produces_record():
    from arm.identity.pipeline import run_experiment
    with tempfile.TemporaryDirectory() as td:
        record = run_experiment(
            experiment_id="ARM-TEST", series="test",
            source="Hello World, this is a test of the measurement arm.",
            source_type="text", mode="topology", results_dir=td,
        )
        assert record.id == "ARM-TEST"
        assert record.verdict in ("PASS", "FAIL", "PARTIAL")
        assert "gini" in record.result
        assert "onset_scale" in record.result
        assert "crystal" in record.result
        files = os.listdir(td)
        assert any(f.startswith("ARM-TEST") for f in files)

def test_experiment_record_saved_as_json():
    from arm.identity.pipeline import run_experiment
    with tempfile.TemporaryDirectory() as td:
        record = run_experiment(
            experiment_id="ARM-JSON", series="test",
            source="Test text for JSON output.", source_type="text",
            mode="topology", results_dir=td,
        )
        files = [f for f in os.listdir(td) if f.endswith(".json")]
        assert len(files) >= 1
        with open(os.path.join(td, files[0])) as f:
            loaded = json.load(f)
        assert loaded["id"] == "ARM-JSON"

def test_run_counter_increments():
    from arm.identity.pipeline import run_experiment
    with tempfile.TemporaryDirectory() as td:
        r1 = run_experiment("ARM-INC", "test", "text one", "text", "topology", td)
        r2 = run_experiment("ARM-INC", "test", "text two", "text", "topology", td)
        assert r2.run == r1.run + 1

def test_csv_source_type():
    from arm.identity.pipeline import run_experiment
    with tempfile.TemporaryDirectory() as td:
        csv_data = "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n10.0,11.0,12.0"
        record = run_experiment("ARM-CSV", "test", csv_data, "csv", "topology", td)
        assert record.verdict in ("PASS", "FAIL", "PARTIAL")
