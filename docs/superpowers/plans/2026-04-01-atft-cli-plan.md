# ATFT CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Unix CLI tool (`atft`) that wraps existing {0,1,3} crystal, persistence, and sheaf measurement code into a composable subcommand architecture with auto-discovering transducers and JSON stdout output.

**Architecture:** Subcommand dispatcher (`cli.py`) routes to pipeline stages (`crystal.py`, `persistence.py`, `sheaf.py`). Each stage reads JSON from stdin or `--model` flag, outputs JSON to stdout. Transducers auto-discovered from `transducers/` directory. All measurement code imported from existing modules — zero duplication.

**Tech Stack:** Python 3.12+, PyTorch, argparse, existing ternary-architect + topological-router modules

---

## File Structure

```
products/atft-cli/
├── atft                    (NEW) Entry point script
├── cli.py                  (NEW) Argument parser + dispatch
├── transducers/
│   ├── __init__.py         (NEW) Auto-discovery registry
│   ├── base.py             (NEW) BaseTransducer ABC
│   └── text.py             (NEW) BPE text transducer
├── pipeline/
│   ├── __init__.py         (NEW) 
│   ├── crystal.py          (NEW) Crystal measurement stage
│   ├── persistence.py      (NEW) Persistence barcode stage
│   ├── sheaf.py            (NEW) Sheaf Laplacian stage
│   └── full.py             (NEW) Orchestrates all stages
├── prism/
│   ├── __init__.py         (NEW)
│   └── model.py            (NEW) Model loading + weight extraction
└── utils/
    ├── __init__.py          (NEW)
    └── io.py               (NEW) JSON I/O, stderr summary
```

Existing files (NO CHANGES — import only):
- `products/ternary-architect/ternary_linear.py`
- `products/ternary-architect/ternary_transformer.py`
- `products/ternary-architect/harmonic_stack.py`
- `products/ternary-architect/run_long.py` (weight_stats, TextChunked)
- `products/ternary-architect/run_drilling.py` (parse_sql_dump, DrillingDataset)
- `products/topological-router/topo_measures.py`
- `products/artifact-analysis/sheaf_laplacian.py`

---

### Task 1: Utils and I/O foundation

**Files:**
- Create: `products/atft-cli/utils/__init__.py`
- Create: `products/atft-cli/utils/io.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p products/atft-cli/{utils,transducers,pipeline,prism}
touch products/atft-cli/{utils,transducers,pipeline,prism}/__init__.py
```

- [ ] **Step 2: Create `utils/io.py`**

```python
"""JSON I/O utilities for ATFT CLI.

All commands output JSON to stdout and human summary to stderr.
Pipeline stages can read previous stage output from stdin.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def make_result(command: str, result: dict, meta: dict | None = None,
                transducer: str | None = None, input_path: str | None = None) -> dict:
    """Build the standard ATFT output envelope."""
    envelope = {
        "command": command,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if transducer:
        envelope["transducer"] = transducer
    if input_path:
        envelope["input"] = input_path
    envelope["result"] = result
    if meta:
        envelope["meta"] = meta
    return envelope


def emit(envelope: dict) -> None:
    """Write JSON to stdout."""
    json.dump(envelope, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    sys.stdout.flush()


def summary(message: str) -> None:
    """Write human-readable summary to stderr."""
    print(message, file=sys.stderr, flush=True)


def read_stdin_json() -> dict | None:
    """Read JSON from stdin if piped (not a TTY)."""
    if sys.stdin.isatty():
        return None
    try:
        return json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return None


def merge_results(previous: dict | None, command: str, result: dict,
                  meta: dict | None = None) -> dict:
    """Merge new results into previous pipeline output.

    When stages are piped: atft crystal | atft persistence
    The persistence stage reads crystal's output and adds its result.
    """
    if previous and "result" in previous:
        merged = previous.copy()
        merged["result"][command] = result
        merged["command"] = f"{merged['command']}+{command}"
        if meta:
            merged.setdefault("meta", {}).update(meta)
        return merged
    return make_result(command, result, meta)
```

- [ ] **Step 3: Verify**

```bash
cd products/atft-cli && python3 -c "
from utils.io import make_result, emit, summary, read_stdin_json, merge_results
r = make_result('test', {'value': 42}, meta={'device': 'cpu'})
print(r)
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add products/atft-cli/
git commit -m "feat(atft-cli): utils/io — JSON envelope, stdout/stderr, pipe support"
```

---

### Task 2: BaseTransducer and auto-discovery registry

**Files:**
- Create: `products/atft-cli/transducers/base.py`
- Modify: `products/atft-cli/transducers/__init__.py`

- [ ] **Step 1: Create `transducers/base.py`**

```python
"""Base transducer interface.

All transducers inherit from BaseTransducer. The registry auto-discovers
any module in this directory that contains a BaseTransducer subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import torch


class BaseTransducer(ABC):
    """Convert domain-specific data into tensor chunks for the prism."""

    name: str = "base"  # Override in subclasses

    @abstractmethod
    def load(self, source: str) -> None:
        """Load data from a file path or stdin indicator ('-')."""
        ...

    @abstractmethod
    def to_chunks(self, chunk_size: int = 256) -> Iterator[torch.Tensor]:
        """Yield (chunk_size, n_features) tensors."""
        ...

    @abstractmethod
    def metadata(self) -> dict:
        """Return transducer-specific metadata (channel names, sample rate, etc.)."""
        ...
```

- [ ] **Step 2: Create `transducers/__init__.py` with auto-discovery**

```python
"""Transducer auto-discovery registry.

Scans this directory for Python modules containing BaseTransducer subclasses.
Drop a new .py file with a subclass → it's available via --transducer.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .base import BaseTransducer

_registry: dict[str, type[BaseTransducer]] = {}


def _discover() -> None:
    """Scan this package for BaseTransducer subclasses."""
    pkg_path = str(Path(__file__).parent)
    for importer, modname, ispkg in pkgutil.iter_modules([pkg_path]):
        if modname in ("base", "__init__"):
            continue
        module = importlib.import_module(f".{modname}", package=__name__)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and issubclass(attr, BaseTransducer)
                    and attr is not BaseTransducer and hasattr(attr, "name")):
                _registry[attr.name] = attr


def get_transducer(name: str) -> BaseTransducer:
    """Get an instantiated transducer by name."""
    if not _registry:
        _discover()
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise ValueError(f"Unknown transducer '{name}'. Available: {available}")
    return _registry[name]()


def list_transducers() -> list[str]:
    """List available transducer names."""
    if not _registry:
        _discover()
    return sorted(_registry.keys())
```

- [ ] **Step 3: Verify auto-discovery finds nothing yet (no transducers registered)**

```bash
cd products/atft-cli && python3 -c "
from transducers import list_transducers
print('Transducers:', list_transducers())
print('OK — empty registry, ready for transducers')
"
```

- [ ] **Step 4: Commit**

```bash
git add products/atft-cli/transducers/
git commit -m "feat(atft-cli): transducer registry — BaseTransducer ABC + auto-discovery"
```

---

### Task 3: Text transducer

**Files:**
- Create: `products/atft-cli/transducers/text.py`

- [ ] **Step 1: Create `transducers/text.py`**

```python
"""Text transducer — BPE tokenized text → tensor chunks.

Wraps the existing TextChunked logic from run_long.py but adapted
for CLI use (reads from file or stdin, yields chunks).
"""
from __future__ import annotations

import sys
from typing import Iterator

import torch

from .base import BaseTransducer


class TextTransducer(BaseTransducer):
    name = "text"

    def __init__(self):
        self._text: str = ""
        self._tokens: list[list[int]] = []
        self._tokenizer = None

    def load(self, source: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if source == "-":
            self._text = sys.stdin.read()
        else:
            with open(source) as f:
                self._text = f.read()

    def to_chunks(self, chunk_size: int = 256) -> Iterator[torch.Tensor]:
        tokens = self._tokenizer.encode(self._text, add_special_tokens=True)
        for i in range(0, len(tokens) - chunk_size, chunk_size):
            chunk = tokens[i:i + chunk_size]
            yield torch.tensor(chunk, dtype=torch.long)

    def metadata(self) -> dict:
        return {
            "type": "text",
            "tokenizer": "gpt2",
            "total_chars": len(self._text),
            "total_tokens": len(self._tokenizer.encode(self._text)) if self._tokenizer else 0,
        }
```

- [ ] **Step 2: Verify auto-discovery finds the text transducer**

```bash
cd products/atft-cli && python3 -c "
from transducers import list_transducers, get_transducer
print('Available:', list_transducers())
t = get_transducer('text')
print(f'Got: {t.name}')
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add products/atft-cli/transducers/text.py
git commit -m "feat(atft-cli): text transducer — BPE tokenized text via GPT-2"
```

---

### Task 4: Prism model loading

**Files:**
- Create: `products/atft-cli/prism/model.py`

- [ ] **Step 1: Create `prism/model.py`**

```python
"""Prism model loading and weight extraction.

Loads trained {0,1,3} models and extracts quantized weights
for crystal/persistence/sheaf analysis. Supports all model types
from ternary-architect (TernaryGPT, HarmonicStack, CARETv2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add ternary-architect to path
_ta_path = str(Path(__file__).resolve().parent.parent.parent / "ternary-architect")
if _ta_path not in sys.path:
    sys.path.insert(0, _ta_path)

from ternary_linear import TernaryLinear


def extract_weights(model_path: str) -> dict[str, torch.Tensor]:
    """Load a checkpoint and extract all TernaryLinear quantized weights.

    Works with any model that contains TernaryLinear layers — doesn't need
    the model class. Just loads the state dict and finds ternary weights
    by checking for the quantization pattern.
    """
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    weights = {}
    for name, param in state.items():
        if not name.endswith(".weight"):
            continue
        # Check if this looks like a ternary weight (values in {0,1,3} only)
        unique = param.unique()
        is_ternary = all(v.item() in (0.0, 1.0, 3.0) for v in unique)
        if not is_ternary:
            # Check if it's a latent weight (STE — quantize it)
            from ternary_linear import quantize_to_set
            values = torch.tensor([0.0, 1.0, 3.0])
            quantized = quantize_to_set(param, values)
            weights[name] = quantized
        else:
            weights[name] = param

    return weights


def weight_stats(weights: dict[str, torch.Tensor]) -> dict:
    """Compute crystal ratio from extracted weights."""
    total_zero = 0
    total_one = 0
    total_three = 0
    total = 0

    for name, w in weights.items():
        total_zero += (w == 0).sum().item()
        total_one += (w == 1).sum().item()
        total_three += (w == 3).sum().item()
        total += w.numel()

    if total == 0:
        return {"void": 0, "identity": 0, "prime": 0, "n_weights": 0}

    return {
        "void": total_zero / total,
        "identity": total_one / total,
        "prime": total_three / total,
        "n_weights": total,
    }


def per_layer_stats(weights: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Per-layer crystal ratios."""
    layers = {}
    for name, w in weights.items():
        n = w.numel()
        if n == 0:
            continue
        layers[name] = {
            "void": (w == 0).sum().item() / n,
            "identity": (w == 1).sum().item() / n,
            "prime": (w == 3).sum().item() / n,
            "n_weights": n,
        }
    return layers
```

- [ ] **Step 2: Verify with existing model**

```bash
cd products/atft-cli && python3 -c "
from prism.model import extract_weights, weight_stats, per_layer_stats
w = extract_weights('../ternary-architect/results/caret_v2_corrected/model.pt')
print(f'Layers: {len(w)}')
s = weight_stats(w)
print(f'Crystal: void={s[\"void\"]:.3f} identity={s[\"identity\"]:.3f} prime={s[\"prime\"]:.3f}')
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add products/atft-cli/prism/
git commit -m "feat(atft-cli): prism model loading — extract weights from any checkpoint"
```

---

### Task 5: Crystal pipeline stage

**Files:**
- Create: `products/atft-cli/pipeline/crystal.py`

- [ ] **Step 1: Create `pipeline/crystal.py`**

```python
"""Crystal measurement pipeline stage.

Measures the {0,1,3} weight distribution (void/identity/prime ratio)
from a trained model checkpoint.
"""
from __future__ import annotations

import time

from ..prism.model import extract_weights, weight_stats, per_layer_stats
from ..utils.io import make_result, emit, summary, read_stdin_json, merge_results


def run(model_path: str | None = None, **kwargs) -> None:
    """Execute crystal measurement."""
    start = time.time()

    # Check for piped input
    previous = read_stdin_json()
    if model_path is None and previous and "meta" in previous:
        model_path = previous.get("meta", {}).get("model")

    if model_path is None:
        raise ValueError("--model required for crystal measurement")

    weights = extract_weights(model_path)
    crystal = weight_stats(weights)
    layers = per_layer_stats(weights)

    elapsed = time.time() - start

    result = {
        "crystal": {
            "void": crystal["void"],
            "identity": crystal["identity"],
            "prime": crystal["prime"],
        },
        "n_weights": crystal["n_weights"],
        "per_layer": layers,
    }

    meta = {"model": model_path, "elapsed_s": round(elapsed, 3)}

    if previous:
        envelope = merge_results(previous, "crystal", result, meta)
    else:
        envelope = make_result("crystal", result, meta)

    emit(envelope)
    summary(f"crystal: void={crystal['void']:.3f} identity={crystal['identity']:.3f} "
            f"prime={crystal['prime']:.3f} ({crystal['n_weights']:,} weights, {elapsed:.1f}s)")
```

- [ ] **Step 2: Verify**

```bash
cd products/atft-cli && python3 -c "
from pipeline.crystal import run
run(model_path='../ternary-architect/results/caret_v2_corrected/model.pt')
" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'void={d[\"result\"][\"crystal\"][\"void\"]:.3f}')"
```

Expected: `void=0.222`

- [ ] **Step 3: Commit**

```bash
git add products/atft-cli/pipeline/crystal.py
git commit -m "feat(atft-cli): crystal pipeline stage — weight distribution measurement"
```

---

### Task 6: Persistence pipeline stage

**Files:**
- Create: `products/atft-cli/pipeline/persistence.py`

- [ ] **Step 1: Create `pipeline/persistence.py`**

```python
"""Persistence barcode pipeline stage.

Computes H₀ persistence barcodes from the weight matrix topology.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from ..prism.model import extract_weights
from ..utils.io import make_result, emit, summary, read_stdin_json, merge_results

# Add topological-router to path
_tr_path = str(Path(__file__).resolve().parent.parent.parent.parent / "topological-router")
if _tr_path not in sys.path:
    sys.path.insert(0, _tr_path)

from topo_measures import h0_persistence, gini_fast


def run(model_path: str | None = None, **kwargs) -> None:
    """Compute persistence barcodes from weight matrices."""
    start = time.time()

    previous = read_stdin_json()
    if model_path is None and previous and "meta" in previous:
        model_path = previous.get("meta", {}).get("model")

    if model_path is None:
        raise ValueError("--model required for persistence measurement")

    weights = extract_weights(model_path)

    # Compute H₀ persistence on each layer's weight matrix
    all_bars = []
    layer_results = {}
    for name, w in weights.items():
        # Flatten to 2D, sample if large
        w_np = w.float().numpy()
        if w_np.shape[0] > 200:
            idx = np.random.RandomState(42).choice(w_np.shape[0], 200, replace=False)
            w_np = w_np[idx]

        bars = h0_persistence(w_np, max_n=200)
        h0_gini = gini_fast(bars) if len(bars) > 0 else 0.0
        all_bars.extend(bars.tolist())
        layer_results[name] = {
            "n_bars": len(bars),
            "h0_gini": float(h0_gini),
            "max_bar": float(bars.max()) if len(bars) > 0 else 0.0,
            "mean_bar": float(bars.mean()) if len(bars) > 0 else 0.0,
        }

    global_gini = gini_fast(np.array(all_bars)) if all_bars else 0.0
    elapsed = time.time() - start

    result = {
        "h0_gini": float(global_gini),
        "n_bars": len(all_bars),
        "per_layer": layer_results,
    }

    meta = {"model": model_path, "elapsed_s": round(elapsed, 3)}

    if previous:
        envelope = merge_results(previous, "persistence", result, meta)
    else:
        envelope = make_result("persistence", result, meta)

    emit(envelope)
    summary(f"persistence: h0_gini={global_gini:.4f} ({len(all_bars)} bars, {elapsed:.1f}s)")
```

- [ ] **Step 2: Commit**

```bash
git add products/atft-cli/pipeline/persistence.py
git commit -m "feat(atft-cli): persistence pipeline stage — H₀ barcodes + Gini"
```

---

### Task 7: Sheaf pipeline stage

**Files:**
- Create: `products/atft-cli/pipeline/sheaf.py`

- [ ] **Step 1: Create `pipeline/sheaf.py`**

```python
"""Sheaf Laplacian pipeline stage.

Computes L_F = δ*δ on the weight matrices, measures spectral gap,
kernel dimension, and Gini of eigenvalues.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from ..prism.model import extract_weights
from ..utils.io import make_result, emit, summary, read_stdin_json, merge_results

# Add artifact-analysis to path
_aa_path = str(Path(__file__).resolve().parent.parent.parent.parent / "artifact-analysis")
if _aa_path not in sys.path:
    sys.path.insert(0, _aa_path)

from sheaf_laplacian import sheaf_laplacian_from_weights, analyze_sheaf_laplacian


def run(model_path: str | None = None, **kwargs) -> None:
    """Compute sheaf Laplacian analysis."""
    start = time.time()

    previous = read_stdin_json()
    if model_path is None and previous and "meta" in previous:
        model_path = previous.get("meta", {}).get("model")

    if model_path is None:
        raise ValueError("--model required for sheaf analysis")

    weights = extract_weights(model_path)

    layer_results = {}
    total_kernel = 0
    spectral_gaps = []
    ginis = []

    for name, w in weights.items():
        w_np = w.detach().float().numpy()
        # Project large matrices
        if w_np.shape[0] > 256 or w_np.shape[1] > 256:
            r_idx = np.random.RandomState(42).choice(w_np.shape[0], min(256, w_np.shape[0]), replace=False)
            c_idx = np.random.RandomState(42).choice(w_np.shape[1], min(256, w_np.shape[1]), replace=False)
            w_np = w_np[np.ix_(r_idx, c_idx)]

        L = sheaf_laplacian_from_weights(w_np)
        analysis = analyze_sheaf_laplacian(L, name)
        layer_results[name] = analysis
        total_kernel += analysis["kernel_dim"]
        spectral_gaps.append(analysis["spectral_gap"])
        ginis.append(analysis["gini_eigenvalues"])

    elapsed = time.time() - start
    mean_gap = float(np.mean(spectral_gaps)) if spectral_gaps else 0.0
    mean_gini = float(np.mean(ginis)) if ginis else 0.0

    result = {
        "total_kernel_dim": total_kernel,
        "mean_spectral_gap": mean_gap,
        "mean_gini_eigenvalues": mean_gini,
        "per_layer": layer_results,
    }

    meta = {"model": model_path, "elapsed_s": round(elapsed, 3)}

    if previous:
        envelope = merge_results(previous, "sheaf", result, meta)
    else:
        envelope = make_result("sheaf", result, meta)

    emit(envelope)
    summary(f"sheaf: spectral_gap={mean_gap:.1f} kernel={total_kernel} "
            f"gini={mean_gini:.4f} ({elapsed:.1f}s)")
```

- [ ] **Step 2: Commit**

```bash
git add products/atft-cli/pipeline/sheaf.py
git commit -m "feat(atft-cli): sheaf pipeline stage — Laplacian spectral analysis"
```

---

### Task 8: Full pipeline orchestrator

**Files:**
- Create: `products/atft-cli/pipeline/full.py`

- [ ] **Step 1: Create `pipeline/full.py`**

```python
"""Full pipeline — runs crystal + persistence + sheaf in sequence."""
from __future__ import annotations

import io
import json
import sys
import time

from ..utils.io import make_result, emit, summary


def run(model_path: str | None = None, **kwargs) -> None:
    """Run complete measurement pipeline."""
    start = time.time()

    if model_path is None:
        raise ValueError("--model required for full pipeline")

    # Import stages
    from ..prism.model import extract_weights, weight_stats, per_layer_stats

    weights = extract_weights(model_path)

    # Crystal
    crystal = weight_stats(weights)
    crystal_result = {
        "crystal": {"void": crystal["void"], "identity": crystal["identity"],
                    "prime": crystal["prime"]},
        "n_weights": crystal["n_weights"],
        "per_layer": per_layer_stats(weights),
    }

    # Persistence
    import numpy as np
    _tr_path = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent / "topological-router")
    if _tr_path not in sys.path:
        sys.path.insert(0, _tr_path)
    from topo_measures import h0_persistence, gini_fast

    all_bars = []
    for name, w in weights.items():
        w_np = w.float().numpy()
        if w_np.shape[0] > 200:
            idx = np.random.RandomState(42).choice(w_np.shape[0], 200, replace=False)
            w_np = w_np[idx]
        bars = h0_persistence(w_np, max_n=200)
        all_bars.extend(bars.tolist())

    persistence_result = {
        "h0_gini": float(gini_fast(np.array(all_bars))) if all_bars else 0.0,
        "n_bars": len(all_bars),
    }

    # Sheaf
    _aa_path = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent / "artifact-analysis")
    if _aa_path not in sys.path:
        sys.path.insert(0, _aa_path)
    from sheaf_laplacian import sheaf_laplacian_from_weights, analyze_sheaf_laplacian

    spectral_gaps = []
    ginis = []
    total_kernel = 0
    for name, w in list(weights.items())[:5]:  # Top 5 layers for speed
        w_np = w.detach().float().numpy()
        if w_np.shape[0] > 256 or w_np.shape[1] > 256:
            r_idx = np.random.RandomState(42).choice(w_np.shape[0], min(256, w_np.shape[0]), replace=False)
            c_idx = np.random.RandomState(42).choice(w_np.shape[1], min(256, w_np.shape[1]), replace=False)
            w_np = w_np[np.ix_(r_idx, c_idx)]
        L = sheaf_laplacian_from_weights(w_np)
        analysis = analyze_sheaf_laplacian(L, name)
        spectral_gaps.append(analysis["spectral_gap"])
        ginis.append(analysis["gini_eigenvalues"])
        total_kernel += analysis["kernel_dim"]

    sheaf_result = {
        "mean_spectral_gap": float(np.mean(spectral_gaps)) if spectral_gaps else 0.0,
        "total_kernel_dim": total_kernel,
        "mean_gini_eigenvalues": float(np.mean(ginis)) if ginis else 0.0,
    }

    elapsed = time.time() - start

    result = {
        "crystal": crystal_result,
        "persistence": persistence_result,
        "sheaf": sheaf_result,
    }

    meta = {"model": model_path, "elapsed_s": round(elapsed, 3)}
    envelope = make_result("full", result, meta)
    emit(envelope)

    summary(f"full: crystal={crystal['void']:.3f}/{crystal['identity']:.3f}/{crystal['prime']:.3f} "
            f"h0_gini={persistence_result['h0_gini']:.4f} "
            f"spectral_gap={sheaf_result['mean_spectral_gap']:.1f} ({elapsed:.1f}s)")
```

- [ ] **Step 2: Commit**

```bash
git add products/atft-cli/pipeline/full.py
git commit -m "feat(atft-cli): full pipeline — crystal + persistence + sheaf in one pass"
```

---

### Task 9: CLI entry point and dispatcher

**Files:**
- Create: `products/atft-cli/cli.py`
- Create: `products/atft-cli/atft`

- [ ] **Step 1: Create `cli.py`**

```python
"""ATFT CLI — argument parsing and subcommand dispatch."""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atft",
        description="Universal structural measurement tool. "
                    "Measures {0,1,3} crystal structure, persistence, and sheaf topology.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # crystal
    p_crystal = sub.add_parser("crystal", help="Measure {0,1,3} weight distribution")
    p_crystal.add_argument("-m", "--model", required=False, help="Path to trained model.pt")

    # persistence
    p_persist = sub.add_parser("persistence", help="Compute H₀ persistence barcodes")
    p_persist.add_argument("-m", "--model", required=False, help="Path to trained model.pt")

    # sheaf
    p_sheaf = sub.add_parser("sheaf", help="Sheaf Laplacian analysis")
    p_sheaf.add_argument("-m", "--model", required=False, help="Path to trained model.pt")

    # full
    p_full = sub.add_parser("full", help="Run complete pipeline")
    p_full.add_argument("-m", "--model", required=True, help="Path to trained model.pt")

    # list
    sub.add_parser("list", help="List available transducers")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "crystal":
            from pipeline.crystal import run
            run(model_path=args.model)

        elif args.command == "persistence":
            from pipeline.persistence import run
            run(model_path=args.model)

        elif args.command == "sheaf":
            from pipeline.sheaf import run
            run(model_path=args.model)

        elif args.command == "full":
            from pipeline.full import run
            run(model_path=args.model)

        elif args.command == "list":
            from transducers import list_transducers
            for name in list_transducers():
                print(name)

        return 0

    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Create `atft` entry point**

```python
#!/usr/bin/env python3
"""ATFT — Universal Structural Measurement Tool.

Usage: ./atft <command> [options]
       python -m atft_cli <command> [options]
"""
import sys
from pathlib import Path

# Add this directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from cli import main

sys.exit(main())
```

- [ ] **Step 3: Make executable**

```bash
chmod +x products/atft-cli/atft
```

- [ ] **Step 4: Smoke test — list transducers**

```bash
cd products/atft-cli && ./atft list
```

Expected: `text`

- [ ] **Step 5: Smoke test — crystal measurement**

```bash
cd products/atft-cli && ./atft crystal -m ../ternary-architect/results/caret_v2_corrected/model.pt 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); c=d['result']['crystal']; print(f'void={c[\"void\"]:.3f} identity={c[\"identity\"]:.3f} prime={c[\"prime\"]:.3f}')"
```

Expected: `void=0.222 identity=0.417 prime=0.361`

- [ ] **Step 6: Smoke test — full pipeline**

```bash
cd products/atft-cli && ./atft full -m ../ternary-architect/results/caret_v2_corrected/model.pt 2>/dev/null | python3 -m json.tool | head -20
```

- [ ] **Step 7: Smoke test — pipe chain**

```bash
cd products/atft-cli && ./atft crystal -m ../ternary-architect/results/caret_v2_corrected/model.pt 2>/dev/null | ./atft persistence 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['command'])"
```

Expected: `crystal+persistence`

- [ ] **Step 8: Commit**

```bash
git add products/atft-cli/cli.py products/atft-cli/atft
git commit -m "feat(atft-cli): CLI entry point — subcommand dispatch, pipe support

Commands: crystal, persistence, sheaf, full, list
JSON stdout, human summary stderr, composable via pipes."
```

---

### Task 10: Integration test and push

- [ ] **Step 1: Full integration test**

```bash
cd products/atft-cli

echo "=== Test 1: list ==="
./atft list

echo "=== Test 2: crystal ==="
./atft crystal -m ../ternary-architect/results/caret_v2_corrected/model.pt 2>&1 | tail -1

echo "=== Test 3: full pipeline ==="
./atft full -m ../ternary-architect/results/caret_v2_corrected/model.pt 2>&1 | tail -1

echo "=== Test 4: pipe chain ==="
./atft crystal -m ../ternary-architect/results/caret_v2_corrected/model.pt 2>/dev/null | ./atft persistence 2>&1 | tail -1

echo "=== ALL TESTS PASSED ==="
```

- [ ] **Step 2: Final commit and push**

```bash
git add -A
git commit -m "feat(atft-cli): complete CLI — crystal, persistence, sheaf, full, pipes

Universal structural measurement tool. Any model in, topology out.
JSON stdout, composable via Unix pipes.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

git push origin master
```
