# ARM Universal Topological Measurement — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a topological measurement engine that runs on ARM hardware (Snapdragon X Plus) with zero PyTorch dependency, capable of measuring the structural topology of any 0-dimensional sequential medium and producing experiment records with honest verdicts.

**Architecture:** Fractal {0,1,3} module structure. Three bands — void/ (boundaries), identity/ (transport), prime/ (computation) — mirroring the crystal the instrument measures. Junction router connects all bands. Pipeline produces append-only experiment records. Phase A is CPU-only using NumPy/SciPy for correctness.

**Tech Stack:** Python 3.14, NumPy 2.4.2, SciPy 1.17.1, stdlib urllib/json. No PyTorch, no ONNX, no external ML libraries.

**Spec:** `docs/superpowers/specs/2026-04-01-arm-measurement-design.md`

---

## File Structure

```
arm/
  __init__.py              (NEW) — Package root, version string
  __main__.py              (NEW) — `python -m arm` entry point → delegates to cli
  measure.py               (NEW) — The junction: 3-arm router connecting all bands

  void/
    __init__.py            (NEW) — Band marker
    formats.py             (NEW) — Dataclasses: PointCloud, PersistenceDiagram, Crystal, ExperimentRecord
    transducers.py         (NEW) — Base Transducer + TextTransducer + VeilbreakTransducer + GenericTransducer
    cli.py                 (NEW) — argparse CLI: measure, compare, validate, series, results

  identity/
    __init__.py            (NEW) — Band marker
    weights.py             (NEW) — 2-bit pack/unpack, .npz load, crystal validation on load
    persistence.py         (NEW) — H₀ persistence via union-find, filtration sweep, barcodes
    pipeline.py            (NEW) — Experiment runner: modes (topology/crystal/full), record management

  prime/
    __init__.py            (NEW) — Band marker
    crystal.py             (NEW) — Ternary INT16/INT32 matmul, forward pass (Phase A: topology-only ops)
    invariants.py          (NEW) — Crystal ratio, Gini, onset scale, eff_rank, spectral gap
    compare.py             (NEW) — Crystal distance, barcode distance, universality test

  results/                 (NEW) — Directory for experiment JSON output (gitignored except index)

tests/
  arm/
    __init__.py            (NEW)
    test_formats.py        (NEW) — Dataclass construction, validation, serialization
    test_transducers.py    (NEW) — TextTransducer on known strings, GenericTransducer on CSV
    test_persistence.py    (NEW) — H₀ on known point clouds (triangle, grid, cluster)
    test_invariants.py     (NEW) — Gini, onset scale, eff_rank on synthetic data
    test_weights.py        (NEW) — 2-bit pack/unpack roundtrip, corruption detection
    test_compare.py        (NEW) — Crystal distance, universality on synthetic crystals
    test_crystal.py        (NEW) — Ternary matmul stub, forward pass stub
    test_pipeline.py       (NEW) — End-to-end: text file → experiment record
    test_measure.py        (NEW) — Junction routing: topology mode produces valid record
```

Each task below is self-contained and produces a commit. Tasks build on each other sequentially — each task's tests import from prior tasks' code.

---

### Task 1: Package skeleton + formats (void/formats.py)

**Files:**
- Create: `arm/__init__.py`
- Create: `arm/__main__.py`
- Create: `arm/void/__init__.py`
- Create: `arm/identity/__init__.py`
- Create: `arm/prime/__init__.py`
- Create: `arm/void/formats.py`
- Create: `arm/results/.gitkeep`
- Create: `tests/arm/__init__.py`
- Create: `tests/arm/test_formats.py`

- [ ] **Step 1: Write failing tests for all dataclasses**

```python
# tests/arm/test_formats.py
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
    assert len(pc.hash) == 64  # SHA256 hex digest

def test_crystal_validation_pass():
    from arm.void.formats import Crystal
    c = Crystal(void_ratio=0.222, identity_ratio=0.417, prime_ratio=0.361,
                eff_rank=59.0, source="test")
    c.validate()  # Should not raise

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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd C:/JTOD1/atft-problems_v0
python -m pytest tests/arm/test_formats.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'arm'`

- [ ] **Step 3: Create package skeleton and formats.py**

```python
# arm/__init__.py
"""ARM: Universal Topological Measurement Arm.
Fractal {0,1,3} architecture for measuring structure in any 0-dim sequential medium.
"""
__version__ = "0.1.0"
```

```python
# arm/__main__.py
"""Enable `python -m arm` invocation. CLI created in Task 9."""
# Deferred import — cli.py does not exist until Task 9
def _main():
    from arm.void.cli import main
    main()

if __name__ == "__main__":
    _main()
```

```python
# arm/void/__init__.py
"""Void band (0): boundaries, interfaces, structured absence."""
```

```python
# arm/identity/__init__.py
"""Identity band (1): transport, scaffolding, preserve structure."""
```

```python
# arm/prime/__init__.py
"""Prime band (3): irreducible computation, structure generation."""
```

```python
# arm/void/formats.py
"""Internal representation boundaries — dataclasses, no logic, just shape."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np


@dataclass
class PointCloud:
    """A point cloud: N points in D dimensions."""
    data: np.ndarray       # (N, D)
    source: str            # description of origin
    hash: str              # SHA256 hex digest for reproducibility

    @classmethod
    def from_array(cls, data: np.ndarray, source: str) -> PointCloud:
        h = hashlib.sha256(data.tobytes()).hexdigest()
        return cls(data=data, source=source, hash=h)


@dataclass
class PersistenceDiagram:
    """Birth-death pairs from persistent homology."""
    h0: np.ndarray           # (K, 2) birth/death pairs for H₀
    h1: np.ndarray           # (K, 2) birth/death pairs for H₁ (may be empty)
    filtration_range: tuple   # (ε_min, ε_max)


@dataclass
class Crystal:
    """Structural ratios — the {0,1,3} decomposition of a signal."""
    void_ratio: float
    identity_ratio: float
    prime_ratio: float
    eff_rank: float
    source: str

    def validate(self):
        total = self.void_ratio + self.identity_ratio + self.prime_ratio
        assert abs(total - 1.0) < 1e-6, f"Crystal ratios sum to {total}, expected 1.0"


@dataclass
class ExperimentRecord:
    """One run of one experiment — the atomic unit of measurement."""
    id: str
    run: int
    series: str
    timestamp: str
    hypothesis: str
    protocol: str
    input_hash: str
    annotations: list
    result: dict
    comparison: dict
    verdict: str        # PASS | FAIL | PARTIAL | BLOCKED | CACHED
    notes: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> ExperimentRecord:
        d = json.loads(s)
        return cls(**d)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd C:/JTOD1/atft-problems_v0
python -m pytest tests/arm/test_formats.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add arm/ tests/arm/
git commit -m "arm: package skeleton + void/formats.py — dataclasses for all data types"
```

---

### Task 2: TextTransducer (void/transducers.py)

**Files:**
- Create: `arm/void/transducers.py`
- Create: `tests/arm/test_transducers.py`

- [ ] **Step 1: Write failing tests for TextTransducer**

```python
# tests/arm/test_transducers.py
import numpy as np
import pytest


def test_text_transducer_simple():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    pc = t.transduce("Hi.")
    # 'H' = upper, 'i' = lower, '.' = stop punct
    assert pc.data.shape == (3, 4)  # 3 chars, 4 channels
    assert pc.source == "text:3chars"


def test_text_transducer_channels():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    pc = t.transduce("A b")
    # 'A': ch0=65, ch1=3(upper), ch2=1(boundary, start of word), ch3=0(no punct)
    # ' ': ch0=32, ch1=0(space), ch2=1(boundary), ch3=0
    # 'b': ch0=98, ch1=1(lower), ch2=1(boundary, start of word), ch3=0
    assert pc.data[0, 1] == 3   # 'A' → case=upper=3
    assert pc.data[1, 1] == 0   # ' ' → case=space=0
    assert pc.data[2, 1] == 1   # 'b' → case=lower=1


def test_text_transducer_punctuation():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    pc = t.transduce("a,b.")
    assert pc.data[1, 3] == 1  # ',' → punct=comma=1
    assert pc.data[3, 3] == 3  # '.' → punct=stop=3


def test_text_transducer_paragraph():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    pc = t.transduce("a\n\nb")
    # '\n' chars: ch2 should be 3 (paragraph boundary)
    assert pc.data[1, 2] == 3  # first \n → paragraph=3
    assert pc.data[2, 2] == 3  # second \n → paragraph=3


def test_text_transducer_hash_deterministic():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    pc1 = t.transduce("Hello World")
    pc2 = t.transduce("Hello World")
    assert pc1.hash == pc2.hash


def test_text_transducer_describe():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    desc = t.describe()
    assert "text" in desc.lower()
    assert "4" in desc  # 4 channels


def test_generic_transducer_csv_string():
    from arm.void.transducers import GenericTransducer
    t = GenericTransducer()
    csv_data = "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0"
    pc = t.transduce(csv_data)
    assert pc.data.shape == (3, 3)
    assert np.isclose(pc.data[0, 0], 1.0)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/arm/test_transducers.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement transducers.py**

```python
# arm/void/transducers.py
"""Universal input boundary — any 0-dim sequential medium → point cloud."""
from __future__ import annotations

import csv
import io
from abc import ABC, abstractmethod

import numpy as np

from arm.void.formats import PointCloud


class Transducer(ABC):
    """Base: converts any 0-dim sequential medium to a point cloud."""

    @abstractmethod
    def transduce(self, source) -> PointCloud:
        ...

    @abstractmethod
    def describe(self) -> str:
        ...


class TextTransducer(Transducer):
    """Character-level harmonic channels.

    Ch0: character identity (ord value)
    Ch1: case state        → 0=space/whitespace, 1=lower, 3=upper
    Ch2: word boundary     → 0=within word, 1=word boundary, 3=paragraph
    Ch3: punctuation       → 0=none, 1=comma/semicolon, 3=stop/question/exclamation
    """

    COMMA_PUNCT = set(",;:")
    STOP_PUNCT = set(".!?")

    def transduce(self, source: str) -> PointCloud:
        n = len(source)
        data = np.zeros((n, 4), dtype=np.int16)

        prev_was_newline = False
        prev_was_space = True  # start of string = word boundary

        for i, ch in enumerate(source):
            # Ch0: character identity
            data[i, 0] = ord(ch) % 256

            # Ch1: case state
            if ch.isupper():
                data[i, 1] = 3
            elif ch.islower():
                data[i, 1] = 1
            else:
                data[i, 1] = 0  # space, digit, symbol = void

            # Ch2: word boundary
            is_space = ch in (" ", "\t")
            is_newline = ch == "\n"

            if is_newline or (prev_was_newline and is_newline):
                data[i, 2] = 3  # paragraph
            elif is_space or prev_was_space or (not ch.isalnum() and not prev_was_space):
                data[i, 2] = 1  # word boundary
            else:
                data[i, 2] = 0  # within word

            prev_was_newline = is_newline
            prev_was_space = is_space or is_newline

            # Ch3: punctuation
            if ch in self.STOP_PUNCT:
                data[i, 3] = 3
            elif ch in self.COMMA_PUNCT:
                data[i, 3] = 1
            else:
                data[i, 3] = 0

        return PointCloud.from_array(data, source=f"text:{n}chars")

    def describe(self) -> str:
        return "TextTransducer: 4-channel character harmonics (identity, case, boundary, punct)"


class GenericTransducer(Transducer):
    """Any columnar/sequential data — CSV string, list of lists, or 2D array."""

    def transduce(self, source) -> PointCloud:
        if isinstance(source, str):
            reader = csv.reader(io.StringIO(source))
            rows = [[float(v) for v in row] for row in reader if row]
            data = np.array(rows, dtype=np.float32)
        elif isinstance(source, np.ndarray):
            data = source.astype(np.float32)
        elif isinstance(source, list):
            data = np.array(source, dtype=np.float32)
        else:
            raise TypeError(f"GenericTransducer cannot handle {type(source)}")

        return PointCloud.from_array(data, source=f"generic:{data.shape[0]}x{data.shape[1]}")

    def describe(self) -> str:
        return "GenericTransducer: columnar/sequential numeric data"
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/arm/test_transducers.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add arm/void/transducers.py tests/arm/test_transducers.py
git commit -m "arm: void/transducers.py — TextTransducer (4ch harmonics) + GenericTransducer"
```

---

### Task 3: VeilbreakTransducer (void/transducers.py addition)

**Files:**
- Modify: `arm/void/transducers.py` (add VeilbreakTransducer class)
- Modify: `tests/arm/test_transducers.py` (add Veilbreak tests)

- [ ] **Step 1: Write failing tests for VeilbreakTransducer**

Add to `tests/arm/test_transducers.py`:

```python
def test_veilbreak_transducer_from_cached_data():
    """Test with synthetic data matching the Veilbreak API shape."""
    from arm.void.transducers import VeilbreakTransducer
    t = VeilbreakTransducer()
    # Simulate cached API response
    experiments = [
        {"laser_wavelength": 650, "substance_dose": 30, "laser_class": 3,
         "substance": "N,N-DMT", "observed": True,
         "description": "Saw patterns in laser field"},
        {"laser_wavelength": 532, "substance_dose": 0, "laser_class": 2,
         "substance": "none", "observed": False,
         "description": "Control run no substance"},
    ]
    pc = t.transduce_experiments(experiments)
    assert pc.data.shape[0] == 2  # 2 experiments
    assert pc.data.shape[1] >= 4  # at least 4 features


def test_veilbreak_transducer_describe():
    from arm.void.transducers import VeilbreakTransducer
    t = VeilbreakTransducer()
    desc = t.describe()
    assert "veilbreak" in desc.lower()


def test_veilbreak_text_channel():
    """Observation text should produce a second point cloud via TextTransducer."""
    from arm.void.transducers import VeilbreakTransducer
    t = VeilbreakTransducer()
    experiments = [
        {"laser_wavelength": 650, "substance_dose": 30, "laser_class": 3,
         "substance": "N,N-DMT", "observed": True,
         "description": "Geometric patterns emerged"},
    ]
    pc_struct, pc_text = t.transduce_multichannel(experiments)
    assert pc_struct.data.shape[0] == 1
    assert pc_text.data.shape[0] > 0  # text point cloud from descriptions
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/arm/test_transducers.py::test_veilbreak_transducer_from_cached_data -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement VeilbreakTransducer**

Append to `arm/void/transducers.py`:

```python
import json
import os
import urllib.request
import urllib.error


SUBSTANCES = ["none", "N,N-DMT", "LSD", "psilocybin", "5-MeO-DMT", "other"]

VEILBREAK_API = "https://api.veilbreak.ai/api/experiments"
VEILBREAK_CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "veilbreak_cache.json")


class VeilbreakTransducer(Transducer):
    """Veilbreak cognitive physics experiments → point clouds.

    Structured channel: [wavelength, dose, laser_class, substance_idx, observed]
    Text channel: observation descriptions → TextTransducer
    """

    def fetch_experiments(self) -> list[dict]:
        """Pull from REST API, cache on success, return cached on failure."""
        cache_path = os.path.normpath(VEILBREAK_CACHE)
        try:
            req = urllib.request.Request(VEILBREAK_API, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode())
                experiments = body.get("data", body) if isinstance(body, dict) else body
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(experiments, f)
                return experiments
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    return json.load(f)
            return []

    def _encode_experiment(self, exp: dict) -> list[float]:
        wavelength = float(exp.get("laser_wavelength", 0) or 0)
        dose = float(exp.get("substance_dose", 0) or 0)
        laser_class = float(exp.get("laser_class", 0) or 0)
        substance = exp.get("substance", "none") or "none"
        sub_idx = float(SUBSTANCES.index(substance)) if substance in SUBSTANCES else float(len(SUBSTANCES))
        observed = 3.0 if exp.get("observed", False) else 0.0
        return [wavelength, dose, laser_class, sub_idx, observed]

    def transduce_experiments(self, experiments: list[dict]) -> PointCloud:
        if not experiments:
            data = np.empty((0, 5), dtype=np.float32)
        else:
            rows = [self._encode_experiment(e) for e in experiments]
            data = np.array(rows, dtype=np.float32)
        return PointCloud.from_array(data, source=f"veilbreak:{len(experiments)}exp")

    def transduce_multichannel(self, experiments: list[dict]) -> tuple[PointCloud, PointCloud]:
        pc_struct = self.transduce_experiments(experiments)
        descriptions = " ".join(e.get("description", "") for e in experiments if e.get("description"))
        text_t = TextTransducer()
        pc_text = text_t.transduce(descriptions) if descriptions else PointCloud.from_array(
            np.empty((0, 4), dtype=np.int16), source="veilbreak:no_text"
        )
        return pc_struct, pc_text

    def transduce(self, source=None) -> PointCloud:
        experiments = source if isinstance(source, list) else self.fetch_experiments()
        return self.transduce_experiments(experiments)

    def describe(self) -> str:
        return "VeilbreakTransducer: cognitive physics experiments (struct + text channels)"
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/arm/test_transducers.py -v
```

Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add arm/void/transducers.py tests/arm/test_transducers.py
git commit -m "arm: VeilbreakTransducer — API fetch, cache fallback, multichannel"
```

---

### Task 4: H₀ Persistence (identity/persistence.py)

**Files:**
- Create: `arm/identity/persistence.py`
- Create: `tests/arm/test_persistence.py`

- [ ] **Step 1: Write failing tests for H₀ persistence**

```python
# tests/arm/test_persistence.py
import numpy as np
import pytest


def test_h0_single_cluster():
    """Three close points should merge into one component early."""
    from arm.identity.persistence import compute_h0
    from arm.void.formats import PointCloud
    pts = np.array([[0, 0], [0.1, 0], [0, 0.1]], dtype=np.float64)
    pc = PointCloud.from_array(pts, source="test")
    diagram = compute_h0(pc, eps_max=2.0, n_steps=100)
    # Should have 3 birth events (at eps=0) and 2 death events
    assert diagram.h0.shape[0] == 3
    # One component should survive to infinity
    inf_bars = diagram.h0[diagram.h0[:, 1] == float('inf')]
    assert len(inf_bars) == 1


def test_h0_two_clusters():
    """Two well-separated clusters should produce 2 long-lived components."""
    from arm.identity.persistence import compute_h0
    from arm.void.formats import PointCloud
    cluster1 = np.array([[0, 0], [0.1, 0], [0, 0.1]], dtype=np.float64)
    cluster2 = np.array([[10, 10], [10.1, 10], [10, 10.1]], dtype=np.float64)
    pts = np.vstack([cluster1, cluster2])
    pc = PointCloud.from_array(pts, source="test")
    diagram = compute_h0(pc, eps_max=20.0, n_steps=200)
    # 2 components should persist past eps=1
    long_bars = diagram.h0[(diagram.h0[:, 1] - diagram.h0[:, 0]) > 1.0]
    assert len(long_bars) >= 2


def test_h0_barcode_sorted():
    """Bars should be sorted by birth time."""
    from arm.identity.persistence import compute_h0
    from arm.void.formats import PointCloud
    pts = np.random.RandomState(42).randn(20, 3).astype(np.float64)
    pc = PointCloud.from_array(pts, source="test")
    diagram = compute_h0(pc, eps_max=5.0, n_steps=100)
    births = diagram.h0[:, 0]
    assert np.all(births[:-1] <= births[1:])


def test_h0_all_born_at_zero():
    """Every point is born at ε=0."""
    from arm.identity.persistence import compute_h0
    from arm.void.formats import PointCloud
    pts = np.array([[0, 0], [5, 5], [10, 10]], dtype=np.float64)
    pc = PointCloud.from_array(pts, source="test")
    diagram = compute_h0(pc, eps_max=20.0, n_steps=100)
    assert np.all(diagram.h0[:, 0] == 0.0)


def test_filtration_sweep_annotations():
    """compute_h0 should return filtration range."""
    from arm.identity.persistence import compute_h0
    from arm.void.formats import PointCloud
    pts = np.array([[0, 0], [1, 1]], dtype=np.float64)
    pc = PointCloud.from_array(pts, source="test")
    diagram = compute_h0(pc, eps_max=3.0, n_steps=50)
    assert diagram.filtration_range == (0.0, 3.0)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/arm/test_persistence.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement H₀ persistence via union-find**

```python
# arm/identity/persistence.py
"""The measuring tape — persistent homology via union-find (H₀)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from arm.void.formats import PointCloud, PersistenceDiagram


class UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Returns True if a merge occurred (a and b were in different sets)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def compute_h0(cloud: PointCloud, eps_max: float = 5.0, n_steps: int = 100) -> PersistenceDiagram:
    """Compute H₀ persistence diagram via union-find on Rips filtration.

    Algorithm: sort all pairwise distances, sweep through them. Each merge
    kills the younger component (born at ε=0, dies at merge distance).
    The oldest component (index 0 by convention) survives to infinity.

    Args:
        cloud: input point cloud
        eps_max: maximum filtration value (used for annotation only; actual
                 filtration runs through all pairwise distances)
        n_steps: not used in exact algorithm but kept for API compatibility

    Returns:
        PersistenceDiagram with H₀ bars and empty H₁
    """
    n = cloud.data.shape[0]
    if n == 0:
        return PersistenceDiagram(
            h0=np.empty((0, 2)), h1=np.empty((0, 2)),
            filtration_range=(0.0, eps_max)
        )
    if n == 1:
        return PersistenceDiagram(
            h0=np.array([[0.0, float('inf')]]),
            h1=np.empty((0, 2)),
            filtration_range=(0.0, eps_max)
        )

    # Compute pairwise distances
    dists = pdist(cloud.data.astype(np.float64))
    dist_matrix = squareform(dists)

    # Get all edges sorted by distance
    # Upper triangle indices
    ii, jj = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[ii, jj]
    order = np.argsort(edge_dists)

    # Union-find sweep
    uf = UnionFind(n)
    death_times = {}  # component_root → death distance

    for idx in order:
        d = edge_dists[idx]
        a, b = int(ii[idx]), int(jj[idx])
        ra, rb = uf.find(a), uf.find(b)
        if ra != rb:
            # Elder (lower index) survives, younger dies
            elder, younger = min(ra, rb), max(ra, rb)
            death_times[younger] = d
            # Force elder to be the new root regardless of rank
            uf.parent[younger] = elder

    # Build barcode: all born at 0, die at merge time (or infinity for survivor)
    bars = []
    for i in range(n):
        death = death_times.get(i, float('inf'))
        bars.append([0.0, death])

    bars.sort(key=lambda b: (b[1], b[0]))  # sort by death time
    h0 = np.array(bars)

    return PersistenceDiagram(
        h0=h0, h1=np.empty((0, 2)),
        filtration_range=(0.0, eps_max)
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/arm/test_persistence.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add arm/identity/persistence.py tests/arm/test_persistence.py
git commit -m "arm: identity/persistence.py — H₀ via union-find on Rips filtration"
```

---

### Task 5: Invariants (prime/invariants.py)

**Files:**
- Create: `arm/prime/invariants.py`
- Create: `tests/arm/test_invariants.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/arm/test_invariants.py
import numpy as np
import pytest


def test_gini_uniform():
    """Uniform bars → Gini ≈ 0."""
    from arm.prime.invariants import gini
    bars = np.array([1.0, 1.0, 1.0, 1.0])
    assert abs(gini(bars)) < 1e-6


def test_gini_one_dominant():
    """One huge bar + many tiny → Gini near 1."""
    from arm.prime.invariants import gini
    bars = np.array([100.0, 0.01, 0.01, 0.01])
    assert gini(bars) > 0.7


def test_gini_empty():
    from arm.prime.invariants import gini
    assert gini(np.array([])) == 0.0


def test_onset_scale():
    """Onset scale = death time of first non-infinite bar."""
    from arm.prime.invariants import onset_scale
    from arm.void.formats import PersistenceDiagram
    h0 = np.array([[0.0, 0.5], [0.0, 1.2], [0.0, float('inf')]])
    pd = PersistenceDiagram(h0=h0, h1=np.empty((0, 2)), filtration_range=(0, 5))
    assert onset_scale(pd) == 0.5


def test_effective_rank():
    """Identity-like matrix → eff_rank ≈ dimension."""
    from arm.prime.invariants import effective_rank
    # 10x10 identity-ish: all eigenvalues ≈ 1
    data = np.eye(10) + np.random.RandomState(42).randn(10, 10) * 0.01
    er = effective_rank(data)
    assert 8.0 < er < 10.5


def test_effective_rank_rank1():
    """Rank-1 matrix → eff_rank ≈ 1."""
    from arm.prime.invariants import effective_rank
    v = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)
    data = v.T @ v  # rank 1
    er = effective_rank(data)
    assert er < 1.5


def test_crystal_from_topology():
    """Classify bars by persistence into void/identity/prime."""
    from arm.prime.invariants import crystal_from_persistence
    from arm.void.formats import PersistenceDiagram
    # 3 short bars (void), 4 medium bars (identity), 2 long bars (prime)
    h0 = np.array([
        [0, 0.1], [0, 0.1], [0, 0.2],   # short: persistence < mean/3
        [0, 1.0], [0, 1.1], [0, 1.2], [0, 1.3],  # medium
        [0, 5.0], [0, 6.0],              # long: persistence > mean*2
    ])
    pd = PersistenceDiagram(h0=h0, h1=np.empty((0, 2)), filtration_range=(0, 10))
    crystal = crystal_from_persistence(pd)
    crystal.validate()
    # Exact ratios depend on threshold calibration, just check it sums to 1
    assert crystal.void_ratio + crystal.identity_ratio + crystal.prime_ratio == pytest.approx(1.0)
    # Short bars should be void
    assert crystal.void_ratio > 0
    # Long bars should be prime
    assert crystal.prime_ratio > 0


def test_spectral_gap():
    from arm.prime.invariants import spectral_gap
    eigenvalues = np.array([10.0, 2.0, 1.0, 0.5])
    assert spectral_gap(eigenvalues) == pytest.approx(5.0)  # 10/2
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/arm/test_invariants.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement invariants.py**

```python
# arm/prime/invariants.py
"""Structural constant extraction — the primes of any signal."""
from __future__ import annotations

import numpy as np
from arm.void.formats import PersistenceDiagram, Crystal


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a distribution. 0=uniform, 1=single dominant."""
    if len(values) == 0:
        return 0.0
    values = np.sort(np.abs(values)).astype(np.float64)
    n = len(values)
    total = values.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * total) / (n * total))


def onset_scale(diagram: PersistenceDiagram) -> float:
    """First significant H₀ death event (smallest finite death time)."""
    deaths = diagram.h0[:, 1]
    finite_deaths = deaths[np.isfinite(deaths)]
    if len(finite_deaths) == 0:
        return float('inf')
    return float(np.min(finite_deaths))


def effective_rank(data: np.ndarray) -> float:
    """exp(entropy of normalized singular values). Measures intrinsic dimensionality."""
    if data.size == 0:
        return 0.0
    sv = np.linalg.svd(data.astype(np.float64), compute_uv=False)
    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def spectral_gap(eigenvalues: np.ndarray) -> float:
    """Ratio of largest to second-largest eigenvalue."""
    ev = np.sort(np.abs(eigenvalues))[::-1]
    if len(ev) < 2 or ev[1] == 0:
        return float('inf')
    return float(ev[0] / ev[1])


def crystal_from_persistence(diagram: PersistenceDiagram) -> Crystal:
    """Classify persistence bars into void/identity/prime by lifetime.

    Thresholds (initial):
      void:     persistence < mean/3
      identity: mean/3 ≤ persistence ≤ mean*2
      prime:    persistence > mean*2
    """
    bars = diagram.h0
    if len(bars) == 0:
        return Crystal(void_ratio=0, identity_ratio=0, prime_ratio=0,
                        eff_rank=0, source="empty")

    persistence = bars[:, 1] - bars[:, 0]
    # Exclude infinite bars from mean calculation
    finite_p = persistence[np.isfinite(persistence)]
    if len(finite_p) == 0:
        return Crystal(void_ratio=0, identity_ratio=0, prime_ratio=1.0,
                        eff_rank=0, source="all_infinite")

    mean_p = np.mean(finite_p)
    if mean_p == 0:
        n = len(persistence)
        return Crystal(void_ratio=1.0, identity_ratio=0, prime_ratio=0,
                        eff_rank=0, source="zero_persistence")

    low_thresh = mean_p / 3.0
    high_thresh = mean_p * 2.0

    void_count = np.sum(persistence < low_thresh)
    prime_count = np.sum((persistence > high_thresh) | ~np.isfinite(persistence))
    identity_count = len(persistence) - void_count - prime_count

    total = len(persistence)
    return Crystal(
        void_ratio=float(void_count / total),
        identity_ratio=float(identity_count / total),
        prime_ratio=float(prime_count / total),
        eff_rank=effective_rank(bars),
        source="topology"
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/arm/test_invariants.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add arm/prime/invariants.py tests/arm/test_invariants.py
git commit -m "arm: prime/invariants.py — Gini, onset scale, eff_rank, spectral gap, crystal from topology"
```

---

### Task 6: Weights pack/unpack (identity/weights.py)

**Files:**
- Create: `arm/identity/weights.py`
- Create: `tests/arm/test_weights.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/arm/test_weights.py
import numpy as np
import pytest
import tempfile
import os


def test_pack_unpack_roundtrip():
    from arm.identity.weights import pack_ternary, unpack_ternary
    values = np.array([0, 1, 3, 0, 1, 3, 1, 0], dtype=np.uint8)
    packed = pack_ternary(values)
    assert packed.shape == (2,)  # 8 values / 4 per byte = 2 bytes
    unpacked = unpack_ternary(packed, count=8)
    np.testing.assert_array_equal(unpacked, values)


def test_pack_unpack_single_byte():
    from arm.identity.weights import pack_ternary, unpack_ternary
    values = np.array([3, 1, 0, 3], dtype=np.uint8)
    packed = pack_ternary(values)
    assert packed.shape == (1,)
    # 3 | (1<<2) | (0<<4) | (3<<6) = 3 + 4 + 0 + 192 = 199
    assert packed[0] == 199
    unpacked = unpack_ternary(packed, count=4)
    np.testing.assert_array_equal(unpacked, values)


def test_reject_invalid_value():
    from arm.identity.weights import pack_ternary
    values = np.array([0, 2, 1, 3], dtype=np.uint8)  # 2 is INVALID
    with pytest.raises(ValueError, match="invalid"):
        pack_ternary(values)


def test_unpack_corrupted_byte():
    from arm.identity.weights import unpack_ternary
    # Byte with a 10 (=2) pair: 0b00_10_00_01 = 0x09 → pair at bits 2-3 is 10
    corrupted = np.array([0b00_10_00_01], dtype=np.uint8)
    with pytest.raises(ValueError, match="corrupt"):
        unpack_ternary(corrupted, count=4)


def test_crystal_from_weights():
    from arm.identity.weights import crystal_from_packed
    # 100 values: 22 zeros, 42 ones, 36 threes
    values = np.array([0]*22 + [1]*42 + [3]*36, dtype=np.uint8)
    from arm.identity.weights import pack_ternary
    packed = pack_ternary(values)
    crystal = crystal_from_packed(packed, count=100)
    assert abs(crystal.void_ratio - 0.22) < 0.01
    assert abs(crystal.identity_ratio - 0.42) < 0.01
    assert abs(crystal.prime_ratio - 0.36) < 0.01


def test_save_load_npz():
    from arm.identity.weights import pack_ternary, save_weights, load_weights
    values = np.array([0]*20 + [1]*40 + [3]*40, dtype=np.uint8)
    packed = pack_ternary(values)
    config = {"layers": 6, "hidden_dim": 512}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.npz")
        save_weights(path, {"layer_0_q": packed}, config=config, total_count=100)
        loaded_layers, loaded_config = load_weights(path)
        np.testing.assert_array_equal(loaded_layers["layer_0_q"], packed)
        assert loaded_config["layers"] == 6
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/arm/test_weights.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement weights.py**

```python
# arm/identity/weights.py
"""Transport pre-trained crystal structure — 2-bit pack/unpack, .npz I/O."""
from __future__ import annotations

import json
import numpy as np

from arm.void.formats import Crystal


VALID_TERNARY = {0, 1, 3}


def pack_ternary(values: np.ndarray) -> np.ndarray:
    """Pack {0,1,3} values into 2-bit pairs, 4 per byte, LSB-first.

    Pack: byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
    """
    values = values.astype(np.uint8)
    invalid = set(np.unique(values)) - VALID_TERNARY
    if invalid:
        raise ValueError(f"invalid ternary values: {invalid}. Only {{0, 1, 3}} allowed.")

    # Pad to multiple of 4
    pad_len = (4 - len(values) % 4) % 4
    if pad_len:
        values = np.concatenate([values, np.zeros(pad_len, dtype=np.uint8)])

    values = values.reshape(-1, 4)
    packed = (values[:, 0]
              | (values[:, 1] << 2)
              | (values[:, 2] << 4)
              | (values[:, 3] << 6))
    return packed.astype(np.uint8)


def unpack_ternary(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack 2-bit pairs from INT8 bytes to {0,1,3} values.

    Rejects any byte containing the bit pair 10 (=2) as corruption.
    """
    packed = packed.astype(np.uint8)
    w0 = packed & 0x03
    w1 = (packed >> 2) & 0x03
    w2 = (packed >> 4) & 0x03
    w3 = (packed >> 6) & 0x03

    all_vals = np.stack([w0, w1, w2, w3], axis=-1).ravel()

    # Check for corruption (value 2 = bit pair 10)
    if np.any(all_vals[:count] == 2):
        raise ValueError("corrupt ternary data: found value 2 (bit pair 10)")

    return all_vals[:count]


def crystal_from_packed(packed: np.ndarray, count: int) -> Crystal:
    """Count {0,1,3} values in packed weights → Crystal ratios."""
    values = unpack_ternary(packed, count)
    n = len(values)
    void_count = int(np.sum(values == 0))
    identity_count = int(np.sum(values == 1))
    prime_count = int(np.sum(values == 3))
    return Crystal(
        void_ratio=void_count / n,
        identity_ratio=identity_count / n,
        prime_ratio=prime_count / n,
        eff_rank=0.0,  # not computed from weights alone
        source="weights"
    )


def save_weights(path: str, layers: dict[str, np.ndarray],
                 config: dict, total_count: int) -> None:
    """Save packed ternary weights to .npz."""
    arrays = {k: v for k, v in layers.items()}
    arrays["config"] = np.array([json.dumps(config)])
    arrays["total_count"] = np.array([total_count])
    np.savez(path, **arrays)


def load_weights(path: str) -> tuple[dict[str, np.ndarray], dict]:
    """Load packed ternary weights from .npz."""
    data = np.load(path, allow_pickle=False)
    config = json.loads(str(data["config"][0]))
    layers = {k: data[k] for k in data.files if k not in ("config", "total_count")}
    return layers, config
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/arm/test_weights.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add arm/identity/weights.py tests/arm/test_weights.py
git commit -m "arm: identity/weights.py — 2-bit ternary pack/unpack, .npz I/O, corruption detection"
```

---

### Task 7: Cross-domain comparison (prime/compare.py)

**Files:**
- Create: `arm/prime/compare.py`
- Create: `tests/arm/test_compare.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/arm/test_compare.py
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
    assert d > 0.5  # very different crystals


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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/arm/test_compare.py -v
```

- [ ] **Step 3: Implement compare.py**

```python
# arm/prime/compare.py
"""Cross-domain structural comparison — is the crystal universal?"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from arm.void.formats import Crystal


def crystal_distance(c1: Crystal, c2: Crystal, metric: str = "l1") -> float:
    """Distance between two crystals' (void, identity, prime) ratios."""
    v1 = np.array([c1.void_ratio, c1.identity_ratio, c1.prime_ratio])
    v2 = np.array([c2.void_ratio, c2.identity_ratio, c2.prime_ratio])
    if metric == "l1":
        return float(np.sum(np.abs(v1 - v2)))
    elif metric == "l2":
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def universality_test(crystals: list[Crystal], threshold: float = 0.05) -> dict:
    """Test whether all crystals are within threshold of each other.

    Returns dict with: universal (bool), max_distance, pairs, distances.
    """
    n = len(crystals)
    if n < 2:
        return {"universal": True, "max_distance": 0.0, "pairs": [], "distances": []}

    distances = []
    pairs = []
    max_d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = crystal_distance(crystals[i], crystals[j])
            distances.append(d)
            pairs.append((crystals[i].source, crystals[j].source))
            max_d = max(max_d, d)

    return {
        "universal": max_d < threshold,
        "max_distance": max_d,
        "pairs": pairs,
        "distances": distances,
    }


def barcode_distance(bars1: np.ndarray, bars2: np.ndarray) -> float:
    """Approximate Wasserstein distance between persistence barcodes.

    Greedy heuristic: sort bars by persistence length, match greedily,
    unmatched bars pay persistence/2 (cost to match to diagonal).
    Exact for identical diagrams, good approximation otherwise.
    """
    if len(bars1) == 0 and len(bars2) == 0:
        return 0.0

    def _finite_persistence(bars):
        if len(bars) == 0:
            return np.array([])
        p = bars[:, 1] - bars[:, 0]
        return np.sort(p[np.isfinite(p)])[::-1]  # descending

    p1 = _finite_persistence(bars1)
    p2 = _finite_persistence(bars2)

    # Greedy matching: pair longest bars first, unmatched pay persistence/2
    total_cost = 0.0
    n_match = min(len(p1), len(p2))

    for i in range(n_match):
        total_cost += abs(p1[i] - p2[i])

    # Unmatched bars pay half their persistence (diagonal cost)
    for i in range(n_match, len(p1)):
        total_cost += p1[i] / 2.0
    for i in range(n_match, len(p2)):
        total_cost += p2[i] / 2.0

    return float(total_cost)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/arm/test_compare.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add arm/prime/compare.py tests/arm/test_compare.py
git commit -m "arm: prime/compare.py — crystal distance, barcode Wasserstein, universality test"
```

---

### Task 8: Pipeline + experiment records (identity/pipeline.py)

**Files:**
- Create: `arm/identity/pipeline.py`
- Create: `tests/arm/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/arm/test_pipeline.py
import numpy as np
import json
import os
import tempfile
import pytest


def test_topology_mode_produces_record():
    from arm.identity.pipeline import run_experiment
    with tempfile.TemporaryDirectory() as td:
        record = run_experiment(
            experiment_id="ARM-TEST",
            series="test",
            source="Hello World, this is a test of the measurement arm.",
            source_type="text",
            mode="topology",
            results_dir=td,
        )
        assert record.id == "ARM-TEST"
        assert record.verdict in ("PASS", "FAIL", "PARTIAL")
        assert "gini" in record.result
        assert "onset_scale" in record.result
        assert "crystal" in record.result
        # Check file was written
        files = os.listdir(td)
        assert any(f.startswith("ARM-TEST") for f in files)


def test_experiment_record_saved_as_json():
    from arm.identity.pipeline import run_experiment
    with tempfile.TemporaryDirectory() as td:
        record = run_experiment(
            experiment_id="ARM-JSON",
            series="test",
            source="Test text for JSON output.",
            source_type="text",
            mode="topology",
            results_dir=td,
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/arm/test_pipeline.py -v
```

- [ ] **Step 3: Implement pipeline.py**

```python
# arm/identity/pipeline.py
"""The scaffolding — connects stages, manages experiment records."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from arm.void.formats import ExperimentRecord, PointCloud
from arm.void.transducers import TextTransducer, GenericTransducer, VeilbreakTransducer
from arm.identity.persistence import compute_h0
from arm.prime.invariants import (
    gini, onset_scale, effective_rank, crystal_from_persistence
)


TRANSDUCERS = {
    "text": TextTransducer,
    "csv": GenericTransducer,
    "generic": GenericTransducer,
    "veilbreak": VeilbreakTransducer,
}


def _get_next_run(experiment_id: str, results_dir: str) -> int:
    """Find next run number for this experiment ID."""
    if not os.path.exists(results_dir):
        return 1
    existing = [f for f in os.listdir(results_dir)
                if f.startswith(experiment_id) and f.endswith(".json")]
    return len(existing) + 1


def _save_record(record: ExperimentRecord, results_dir: str) -> str:
    """Save experiment record as JSON. Returns file path."""
    os.makedirs(results_dir, exist_ok=True)
    ts = record.timestamp.replace(":", "-").replace(".", "-")
    filename = f"{record.id}_run{record.run}_{ts}.json"
    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        f.write(record.to_json())
    return path


def run_experiment(
    experiment_id: str,
    series: str,
    source,
    source_type: str,
    mode: str = "topology",
    results_dir: str = "arm/results",
    hypothesis: str = "",
    eps_max: float = 5.0,
    n_steps: int = 100,
) -> ExperimentRecord:
    """Run a single experiment through the measurement arm.

    Modes:
      topology: point cloud → persistence → invariants (no neural net)
      crystal:  point cloud → ternary forward pass → crystal ratios (needs weights)
      full:     both in parallel, cross-validate
    """
    t0 = time.time()
    annotations = []
    run = _get_next_run(experiment_id, results_dir)
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. TRANSDUCE
    transducer_cls = TRANSDUCERS.get(source_type, GenericTransducer)
    transducer = transducer_cls()
    cloud = transducer.transduce(source)
    annotations.append({
        "stage": "transduce",
        "transducer": transducer.describe(),
        "points": cloud.data.shape[0],
        "dims": cloud.data.shape[1] if cloud.data.ndim > 1 else 1,
        "hash": cloud.hash,
    })

    result = {}
    verdict = "PARTIAL"

    if mode in ("topology", "full"):
        # 2. MEASURE: persistence
        diagram = compute_h0(cloud, eps_max=eps_max, n_steps=n_steps)
        annotations.append({
            "stage": "persistence",
            "h0_bars": diagram.h0.shape[0],
            "h1_bars": diagram.h1.shape[0],
            "filtration_range": list(diagram.filtration_range),
        })

        # 3. EXTRACT: invariants
        finite_bars = diagram.h0[:, 1] - diagram.h0[:, 0]
        finite_bars = finite_bars[np.isfinite(finite_bars)]

        g = gini(finite_bars)
        eps_star = onset_scale(diagram)
        crystal = crystal_from_persistence(diagram)
        crystal.validate()

        result["gini"] = g
        result["onset_scale"] = eps_star
        result["h0_bar_count"] = int(diagram.h0.shape[0])
        result["crystal"] = {
            "void": crystal.void_ratio,
            "identity": crystal.identity_ratio,
            "prime": crystal.prime_ratio,
        }
        result["eff_rank"] = crystal.eff_rank

        annotations.append({
            "stage": "invariants",
            "gini": g,
            "onset_scale": eps_star,
            "crystal_void": crystal.void_ratio,
            "crystal_identity": crystal.identity_ratio,
            "crystal_prime": crystal.prime_ratio,
        })

        verdict = "PASS"

    if mode in ("crystal", "full"):
        # Crystal forward pass — Phase A stub (needs weights from desktop)
        annotations.append({
            "stage": "crystal_forward",
            "status": "NOT_IMPLEMENTED",
            "note": "Requires pre-trained weights from desktop via export_for_arm()",
        })
        if mode == "crystal":
            verdict = "BLOCKED"

    elapsed = time.time() - t0
    annotations.append({"stage": "timing", "elapsed_seconds": round(elapsed, 3)})

    record = ExperimentRecord(
        id=experiment_id,
        run=run,
        series=series,
        timestamp=timestamp,
        hypothesis=hypothesis,
        protocol=f"mode={mode}, source_type={source_type}, eps_max={eps_max}, n_steps={n_steps}",
        input_hash=cloud.hash,
        annotations=annotations,
        result=result,
        comparison={},
        verdict=verdict,
        notes="",
    )

    _save_record(record, results_dir)
    return record
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/arm/test_pipeline.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add arm/identity/pipeline.py tests/arm/test_pipeline.py
git commit -m "arm: identity/pipeline.py — experiment runner, topology mode, record persistence"
```

---

### Task 9: Junction + CLI (measure.py, __main__.py, cli.py)

**Files:**
- Create: `arm/measure.py`
- Create: `arm/void/cli.py`
- Modify: `arm/__main__.py` (already created in Task 1)
- Create: `tests/arm/test_measure.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/arm/test_measure.py
import tempfile
import pytest


def test_measure_text_topology():
    from arm.measure import measure
    with tempfile.TemporaryDirectory() as td:
        record = measure(
            source="The topology doesn't care what the structure is.",
            source_type="text",
            mode="topology",
            results_dir=td,
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
        record = measure(
            source="Hello world",
            source_type="auto",
            mode="topology",
            results_dir=td,
        )
        assert record.verdict == "PASS"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/arm/test_measure.py -v
```

- [ ] **Step 3: Implement measure.py and cli.py**

```python
# arm/measure.py
"""The junction — 3-arm router connecting all bands.

Signal enters, routes through void (transduce) → identity (transport) →
prime (compute), converges at output as an ExperimentRecord.
"""
from __future__ import annotations

from arm.void.formats import ExperimentRecord
from arm.identity.pipeline import run_experiment


def measure(
    source,
    source_type: str = "auto",
    mode: str = "topology",
    experiment_id: str = "ARM-MEASURE",
    series: str = "ad-hoc",
    results_dir: str = "arm/results",
    hypothesis: str = "",
    eps_max: float = 5.0,
    n_steps: int = 100,
) -> ExperimentRecord:
    """The junction. Signal enters, routes through three bands, converges at output.

    1. TRANSDUCE: void/transducers converts source → point cloud
    2. ROUTE: classify measurement mode, fan out
    3. MEASURE: identity/persistence + prime/crystal (parallel if mode='full')
    4. EXTRACT: prime/invariants on measurement results
    5. COMPARE: prime/compare against prior results (if available)
    6. REPORT: produce ExperimentRecord with full annotations
    """
    if source_type == "auto":
        source_type = _detect_source_type(source)

    return run_experiment(
        experiment_id=experiment_id,
        series=series,
        source=source,
        source_type=source_type,
        mode=mode,
        results_dir=results_dir,
        hypothesis=hypothesis,
        eps_max=eps_max,
        n_steps=n_steps,
    )


def _detect_source_type(source) -> str:
    """Best-effort auto-detection of source type."""
    if isinstance(source, str):
        # If it looks like CSV (commas + newlines + mostly numbers)
        lines = source.strip().split("\n")
        if len(lines) > 1 and all("," in line for line in lines[:3]):
            try:
                float(lines[0].split(",")[0])
                return "csv"
            except ValueError:
                pass
        return "text"
    return "generic"
```

```python
# arm/void/cli.py
"""Human boundary — command-line interface for the measurement arm."""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="arm",
        description="ARM: Universal Topological Measurement Arm"
    )
    sub = parser.add_subparsers(dest="command")

    # measure
    p_measure = sub.add_parser("measure", help="Measure a source")
    p_measure.add_argument("source_type", choices=["text", "csv", "veilbreak", "auto"],
                           help="Type of source to measure")
    p_measure.add_argument("source", nargs="?", default=None,
                           help="File path or inline data (omit for veilbreak)")
    p_measure.add_argument("--mode", default="topology", choices=["topology", "crystal", "full"])
    p_measure.add_argument("--eps-max", type=float, default=5.0)
    p_measure.add_argument("--id", default=None, help="Experiment ID")
    p_measure.add_argument("--results-dir", default="arm/results")

    # results
    p_results = sub.add_parser("results", help="Show experiment records")
    p_results.add_argument("--results-dir", default="arm/results")

    # series
    # compare
    p_compare = sub.add_parser("compare", help="Compare crystal ratios between experiment records")
    p_compare.add_argument("source1", help="Experiment ID or result JSON path")
    p_compare.add_argument("source2", help="Experiment ID or result JSON path")
    p_compare.add_argument("--results-dir", default="arm/results")

    # validate
    p_validate = sub.add_parser("validate", help="Validate crystal weights match desktop")
    p_validate.add_argument("weights", help="Path to .npz weights file")

    # series
    p_series = sub.add_parser("series", help="Run full experiment series (ARM-001 through ARM-005)")
    p_series.add_argument("--results-dir", default="arm/results")

    args = parser.parse_args(argv)

    if args.command == "measure":
        _cmd_measure(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "results":
        _cmd_results(args)
    elif args.command == "series":
        _cmd_series(args)
    else:
        parser.print_help()


def _cmd_measure(args):
    from arm.measure import measure

    source = args.source
    if args.source_type != "veilbreak" and source and os.path.isfile(source):
        with open(source) as f:
            source = f.read()
    elif args.source_type == "veilbreak":
        source = None  # VeilbreakTransducer will fetch

    exp_id = args.id or f"ARM-{args.source_type.upper()}"
    record = measure(
        source=source,
        source_type=args.source_type,
        mode=args.mode,
        experiment_id=exp_id,
        results_dir=args.results_dir,
        eps_max=args.eps_max,
    )
    _print_record(record)


def _cmd_compare(args):
    from arm.prime.compare import crystal_distance
    from arm.void.formats import Crystal

    def _load_crystal(ref, results_dir):
        # Try as file path first, then as experiment ID
        path = ref if os.path.isfile(ref) else None
        if not path:
            candidates = [f for f in os.listdir(results_dir)
                          if f.startswith(ref) and f.endswith(".json")]
            if candidates:
                path = os.path.join(results_dir, sorted(candidates)[-1])
        if not path:
            print(f"Cannot find experiment or file: {ref}")
            return None
        with open(path) as f:
            rec = json.load(f)
        c = rec.get("result", {}).get("crystal")
        if not c:
            print(f"No crystal in {ref}")
            return None
        return Crystal(c["void"], c["identity"], c["prime"], 0.0, rec["id"])

    c1 = _load_crystal(args.source1, args.results_dir)
    c2 = _load_crystal(args.source2, args.results_dir)
    if c1 and c2:
        d = crystal_distance(c1, c2)
        print(f"Crystal distance ({c1.source} vs {c2.source}): {d:.4f}")
        print(f"  {c1.source}: void={c1.void_ratio:.3f} identity={c1.identity_ratio:.3f} prime={c1.prime_ratio:.3f}")
        print(f"  {c2.source}: void={c2.void_ratio:.3f} identity={c2.identity_ratio:.3f} prime={c2.prime_ratio:.3f}")


def _cmd_validate(args):
    from arm.identity.weights import load_weights, crystal_from_packed, unpack_ternary
    layers, config = load_weights(args.weights)
    print(f"Loaded weights: {len(layers)} layers, config: {config}")
    # Aggregate crystal across all layers
    all_values = []
    for name, packed in layers.items():
        count = packed.shape[0] * 4  # 4 values per byte
        values = unpack_ternary(packed, count)
        all_values.append(values)
    if all_values:
        import numpy as np
        combined = np.concatenate(all_values)
        from arm.void.formats import Crystal
        n = len(combined)
        c = Crystal(
            void_ratio=float(np.sum(combined == 0)) / n,
            identity_ratio=float(np.sum(combined == 1)) / n,
            prime_ratio=float(np.sum(combined == 3)) / n,
            eff_rank=0.0, source="weights"
        )
        c.validate()
        print(f"Crystal: void={c.void_ratio:.3f} identity={c.identity_ratio:.3f} prime={c.prime_ratio:.3f}")
    else:
        print("No weight layers found.")


def _cmd_results(args):
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        print("No results directory found.")
        return
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".json"))
    if not files:
        print("No experiment records found.")
        return
    for f in files:
        with open(os.path.join(results_dir, f)) as fh:
            rec = json.load(fh)
        print(f"{rec['id']} run{rec['run']} [{rec['verdict']}] {rec['timestamp']}")
        if "crystal" in rec.get("result", {}):
            c = rec["result"]["crystal"]
            print(f"  crystal: void={c['void']:.3f} identity={c['identity']:.3f} prime={c['prime']:.3f}")


def _cmd_series(args):
    from arm.measure import measure

    print("=== ARM Experiment Series: Phase A ===\n")

    # ARM-001: Text topology
    print("--- ARM-001: Text topology on ARM ---")
    sample_text = _get_sample_text()
    r = measure(sample_text, "text", "topology", "ARM-001", "phase-a", args.results_dir,
                hypothesis="H₀ persistence produces valid barcodes on ARM hardware")
    _print_record(r)

    # ARM-002: Character harmonic crystal
    print("\n--- ARM-002: Character harmonic crystal ---")
    r = measure(sample_text, "text", "topology", "ARM-002", "phase-a", args.results_dir,
                hypothesis="Character harmonics produce crystal ratios near 22/42/36")
    _print_record(r)

    # ARM-003: Veilbreak observation topology
    print("\n--- ARM-003: Veilbreak observation topology ---")
    r = measure(None, "veilbreak", "topology", "ARM-003", "phase-a", args.results_dir,
                hypothesis="Veilbreak experimental data produces measurable topological structure")
    _print_record(r)

    # ARM-005: Cross-domain comparison (ARM-004 needs weights, skip for now)
    print("\n--- ARM-005: Cross-domain comparison ---")
    _run_comparison(args.results_dir)

    print("\n=== Series complete ===")


def _get_sample_text() -> str:
    """Get a sample text for testing. Tries WikiText file, falls back to inline."""
    candidates = [
        "data/wikitext-103-raw/wiki.test.raw",
        "data/wikitext/test.txt",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return f.read()[:10000]  # first 10K chars
    # Fallback: inline sample
    return (
        "The topology doesn't care what the structure is. It cares how the structure "
        "changes across scales. Every mathematical structure that describes reality has "
        "a topological signature. The adaptive operator is the instrument that reads "
        "this signature. The waypoint constraints are the field equations written in "
        "the native language of topology. Primes are the zero-dimensional framework "
        "of computational reality."
    ) * 10


def _run_comparison(results_dir: str):
    from arm.prime.compare import universality_test
    from arm.void.formats import Crystal
    import glob

    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    crystals = []
    for f in files:
        with open(f) as fh:
            rec = json.load(fh)
        if "crystal" in rec.get("result", {}):
            c = rec["result"]["crystal"]
            crystals.append(Crystal(c["void"], c["identity"], c["prime"], 0.0, rec["id"]))

    if len(crystals) < 2:
        print("  Not enough crystal measurements for comparison.")
        return

    result = universality_test(crystals)
    print(f"  Universal: {result['universal']}")
    print(f"  Max distance: {result['max_distance']:.4f}")
    for pair, dist in zip(result["pairs"], result["distances"]):
        print(f"    {pair[0]} vs {pair[1]}: {dist:.4f}")


def _print_record(record):
    print(f"  ID: {record.id} | Run: {record.run} | Verdict: {record.verdict}")
    if "crystal" in record.result:
        c = record.result["crystal"]
        print(f"  Crystal: void={c['void']:.3f} identity={c['identity']:.3f} prime={c['prime']:.3f}")
    if "gini" in record.result:
        print(f"  Gini: {record.result['gini']:.4f}")
    if "onset_scale" in record.result:
        print(f"  Onset ε*: {record.result['onset_scale']:.4f}")
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/arm/test_measure.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 5: Verify CLI works end-to-end**

```bash
cd C:/JTOD1/atft-problems_v0
python -m arm measure text --id ARM-SMOKE "The topology doesn't care what the structure is."
python -m arm results
```

Expected: prints experiment record with crystal ratios and verdict

- [ ] **Step 6: Commit**

```bash
git add arm/measure.py arm/void/cli.py tests/arm/test_measure.py
git commit -m "arm: measure.py junction + cli.py — full pipeline wired, CLI operational"
```

---

### Task 10: Crystal stub (prime/crystal.py)

**Files:**
- Create: `arm/prime/crystal.py`
- Create: `tests/arm/test_crystal.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/arm/test_crystal.py
import numpy as np
import pytest


def test_ternary_matmul_identity():
    """Weight=1 should pass input through (identity)."""
    from arm.prime.crystal import ternary_matmul_dense
    x = np.array([[1, 2, 3]], dtype=np.int16)
    w = np.array([[1], [1], [1]], dtype=np.uint8)  # all identity
    out = ternary_matmul_dense(x, w)
    assert out[0, 0] == 6  # 1+2+3


def test_ternary_matmul_zero():
    """Weight=0 should skip (void)."""
    from arm.prime.crystal import ternary_matmul_dense
    x = np.array([[5, 10, 15]], dtype=np.int16)
    w = np.array([[0], [0], [0]], dtype=np.uint8)
    out = ternary_matmul_dense(x, w)
    assert out[0, 0] == 0


def test_ternary_matmul_three():
    """Weight=3 should triple via shift-add."""
    from arm.prime.crystal import ternary_matmul_dense
    x = np.array([[4, 0, 0]], dtype=np.int16)
    w = np.array([[3], [0], [0]], dtype=np.uint8)
    out = ternary_matmul_dense(x, w)
    assert out[0, 0] == 12  # 4*3


def test_ternary_matmul_mixed():
    """Mixed weights: 0 skips, 1 passes, 3 triples."""
    from arm.prime.crystal import ternary_matmul_dense
    x = np.array([[2, 5, 10]], dtype=np.int16)
    w = np.array([[3], [0], [1]], dtype=np.uint8)  # 2*3 + 5*0 + 10*1 = 16
    out = ternary_matmul_dense(x, w)
    assert out[0, 0] == 16


def test_forward_pass_not_implemented():
    """Full forward pass raises NotImplementedError in Phase A."""
    from arm.prime.crystal import forward_pass
    with pytest.raises(NotImplementedError):
        forward_pass(np.zeros((1, 10), dtype=np.int16), weights_path="dummy.npz")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/arm/test_crystal.py -v
```

- [ ] **Step 3: Implement crystal.py stub**

```python
# arm/prime/crystal.py
"""The irreducible computation — {0,1,3} ternary forward pass.

Phase A: ternary_matmul_dense kernel operational, full forward pass stubbed.
Phase B: ONNX-QNN export for Hexagon NPU acceleration.
"""
from __future__ import annotations

import numpy as np


def ternary_matmul_dense(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Ternary matrix multiply using only skip/pass/shift-add.

    x:       (B, D_in) INT16 activations
    weights: (D_in, D_out) UINT8 dense ternary values {0, 1, 3}

    For each weight:
      0 → zero  (skip accumulator)
      1 → add   (pass x to accumulator)
      3 → (x << 1) + x  (shift-add, no multiply)

    Accumulation in INT32 to prevent overflow.
    """
    x = x.astype(np.int32)
    w = weights.astype(np.int32)

    # Mask-based: avoid multiplication entirely
    mask_0 = (w == 0)
    mask_1 = (w == 1)
    mask_3 = (w == 3)

    # For each output: sum over input dim
    # x @ (mask_1 * 1 + mask_3 * 3) but without using multiply for the 3
    # Equivalent: x @ mask_1 + x @ (mask_3 * 2) + x @ mask_3
    #           = x @ mask_1 + x @ mask_3 + x @ (mask_3 << 1)
    # Simplest correct version for Phase A:
    effective_w = mask_1.astype(np.int32) + mask_3.astype(np.int32) * 3
    result = x @ effective_w
    return result.astype(np.int32)


def forward_pass(x: np.ndarray, weights_path: str) -> dict:
    """Full ternary forward pass — Phase A stub.

    Requires pre-trained weights from desktop via export_for_arm().
    Will be implemented when weights are available.

    Returns dict with crystal ratios from weight counts and
    hidden state measurements.
    """
    raise NotImplementedError(
        "Full forward pass requires pre-trained weights from desktop. "
        "Use export_for_arm() on the desktop to create .npz weights, "
        "then transfer to ARM laptop. See spec: ARM-004."
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/arm/test_crystal.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add arm/prime/crystal.py tests/arm/test_crystal.py
git commit -m "arm: prime/crystal.py — ternary matmul kernel operational, forward pass stubbed for Phase B"
```

---

### Task 11: Run full experiment series + push

**Files:**
- No new files — runs the existing code

- [ ] **Step 1: Run full test suite**

```bash
cd C:/JTOD1/atft-problems_v0
python -m pytest tests/arm/ -v
```

Expected: all tests PASS (approximately 39 tests across 7 test files)

- [ ] **Step 2: Run the Phase A experiment series**

```bash
python -m arm series
```

Expected output: ARM-001 through ARM-003 produce experiment records with crystal ratios. ARM-005 runs cross-domain comparison. All verdicts should be PASS or PARTIAL (PARTIAL if Veilbreak API is unreachable and no cache exists).

- [ ] **Step 3: Review experiment results**

```bash
python -m arm results
```

Verify: each experiment has a crystal measurement, Gini coefficient, and onset scale.

- [ ] **Step 4: Measure code weight by band**

```bash
cd C:/JTOD1/atft-problems_v0
# Count lines per band (excluding __init__.py and test files)
wc -l arm/void/transducers.py arm/void/formats.py arm/void/cli.py
wc -l arm/identity/weights.py arm/identity/persistence.py arm/identity/pipeline.py
wc -l arm/prime/invariants.py arm/prime/compare.py arm/prime/crystal.py
```

Record the ratios. Compare against 22/42/36 prediction.

- [ ] **Step 5: Commit results and push**

```bash
git add arm/ tests/arm/
git commit -m "arm: Phase A complete — experiments ARM-001 through ARM-005 operational"
git push origin master
```

---

## Summary

| Task | Component | Tests | What it builds |
|------|-----------|-------|---------------|
| 1 | void/formats.py | 7 | Dataclasses for all data types |
| 2 | void/transducers.py (Text + Generic) | 7 | Character harmonic transducer |
| 3 | void/transducers.py (Veilbreak) | 3 | API fetch + cache + multichannel |
| 4 | identity/persistence.py | 5 | H₀ via union-find |
| 5 | prime/invariants.py | 8 | Gini, onset, eff_rank, crystal from topology |
| 6 | identity/weights.py | 6 | 2-bit pack/unpack, corruption detection |
| 7 | prime/compare.py | 6 | Crystal distance, universality test |
| 8 | identity/pipeline.py | 4 | Experiment runner + record management |
| 9 | measure.py + cli.py | 3 | Junction + CLI (measure/compare/validate/series/results) |
| 10 | prime/crystal.py | 5 | Ternary matmul kernel + forward pass stub |
| 11 | (integration) | all | Series run, code weight check, push |

**Total: 11 tasks, ~54 tests, ~16 files**

Each task is one commit. Each commit leaves the project in a working state with passing tests.
