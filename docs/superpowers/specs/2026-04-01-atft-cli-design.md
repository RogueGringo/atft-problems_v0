# ATFT CLI — Universal Structural Measurement Tool

**Date:** 2026-04-01
**Status:** Approved for implementation
**Author:** Aaron Jones + Claude

---

## Purpose

A single CLI command (`atft`) that accepts any structured data through
domain-specific transducers, runs it through the {0,1,3} crystallizer +
persistence + sheaf Laplacian + Gini analysis pipeline, and outputs
structural topology reports as JSON to stdout.

Unix philosophy: one tool, composable via pipes, JSON in/out.

---

## Command Interface

```
atft <command> [--transducer <name>] [--input <path>] [options]

Commands:
  crystal       Measure {0,1,3} weight distribution
  persistence   Compute H₀/H₁ persistence barcodes
  sheaf         Sheaf Laplacian analysis (spectral gap, kernel, Gini)
  full          Run complete pipeline (crystal + persistence + sheaf)
  train         Train a prism on input data, save model
  list          List available transducers

Transducers (--transducer):
  text          BPE tokenized text (default)
  drilling      Multi-channel time series from SQL/CSV
  timeseries    Generic numeric CSV time series
  graph         Adjacency list / edge list → graph encoding

I/O:
  --input/-i    File path (if not using stdin)
  --output/-o   File path (default: stdout as JSON)
  --model/-m    Path to trained prism model
  --format      json (default), compact, barcode
```

### Usage Examples

```bash
# Train a prism on text
atft train --transducer text --input korean.txt --steps 20000 --decay 0.0

# Measure crystal
atft crystal --model results/model.pt

# Full pipeline on drilling data
atft full --transducer drilling --input well_data.sql

# Pipe text through crystal
cat kant.txt | atft crystal --transducer text

# Chain stages
atft crystal --model results/model.pt | atft persistence | atft sheaf
```

---

## Architecture

```
products/atft-cli/
├── atft                          # Entry point (chmod +x, #!/usr/bin/env python3)
├── cli.py                        # Argument parser, subcommand dispatch
├── transducers/
│   ├── __init__.py               # Auto-discovery registry
│   ├── base.py                   # BaseTransducer ABC
│   ├── text.py                   # BPE text → tensor chunks
│   ├── drilling.py               # SQL/CSV → normalized multi-channel windows
│   ├── timeseries.py             # Generic CSV → windowed tensor
│   └── graph.py                  # Edge list → adjacency tensor
├── pipeline/
│   ├── __init__.py
│   ├── crystal.py                # Weight distribution measurement
│   ├── persistence.py            # H₀/H₁ barcode computation
│   ├── sheaf.py                  # Sheaf Laplacian, spectral gap, kernel
│   └── full.py                   # Orchestrates crystal→persistence→sheaf
├── prism/
│   ├── __init__.py
│   ├── model.py                  # Imports TernaryLinear + Block from ternary-architect
│   └── train.py                  # Training loop (from run_harmonic.py patterns)
└── utils/
    ├── io.py                     # JSON formatting, stdin reading
    └── measures.py               # Wraps topo_measures.py
```

### Key Design Decisions

1. **Transducer auto-discovery.** `transducers/__init__.py` scans the directory
   for Python modules containing a class that inherits `BaseTransducer`. Drop
   a new file → available via `--transducer`. Zero registration.

2. **BaseTransducer interface:**
   ```python
   class BaseTransducer(ABC):
       name: str
       def load(self, source) -> None
       def to_chunks(self, chunk_size=256) -> Iterator[Tensor]
       def metadata(self) -> dict
   ```

3. **Pipeline stages compose via pipes.** Each reads JSON from stdin OR takes
   `--model` path. Produces JSON to stdout. Stages accumulate results.

4. **No code duplication.** Imports directly from existing modules:
   - `ternary_linear.py` → TernaryLinear, make_linear
   - `ternary_transformer.py` → GPTConfig, Block
   - `topo_measures.py` → effective_rank, spectral_gap, gini_fast, h0_persistence
   - `sheaf_laplacian.py` → sheaf_laplacian_from_weights

5. **GPU optional.** Crystal/sheaf on CPU. Training uses GPU if available.

---

## Output Format

JSON to stdout (pipeable). Human summary to stderr.

```json
{
  "command": "crystal",
  "timestamp": "2026-04-01T19:30:00Z",
  "transducer": "text",
  "input": "korean.txt",
  "result": {
    "crystal": {"void": 0.222, "identity": 0.417, "prime": 0.361},
    "n_weights": 50341888,
    "per_layer": { ... }
  },
  "meta": {
    "model": "results/model.pt",
    "device": "cuda",
    "elapsed_s": 0.3
  }
}
```

Stderr: `crystal: void=0.222 identity=0.417 prime=0.361 (50.3M weights, 0.3s)`

Full pipeline nests all stages:
```json
{
  "command": "full",
  "result": {
    "crystal": { ... },
    "persistence": { "h0_bars": [...], "h0_gini": 0.40 },
    "sheaf": { "spectral_gap": 548.3, "kernel_dim": 0, "gini_eigenvalues": 0.04 }
  }
}
```

Piped stages accumulate: `atft crystal | atft persistence | atft sheaf`
produces final JSON with all three result sections.

---

## Existing Code Reuse

| CLI Component | Source Module | What it provides |
|---|---|---|
| Prism model | `ternary_linear.py` | TernaryLinear, make_linear, STEQuantize |
| Transformer blocks | `ternary_transformer.py` | GPTConfig, Block, CausalSelfAttention |
| Harmonic architecture | `harmonic_stack.py` | HarmonicStack, SpectralRouter |
| Topology measures | `topo_measures.py` | effective_rank, spectral_gap, gini_fast, h0_persistence |
| Weight measurement | `measure.py` | zero_mask_topology, iterative_inference |
| Sheaf analysis | `sheaf_laplacian.py` | sheaf_laplacian_from_weights, analyze_sheaf_laplacian |
| Text dataset | `run_long.py` | TextChunked (BPE tokenization) |
| Drilling dataset | `run_drilling.py` | parse_sql_dump, DrillingDataset |
| Training loop | `run_harmonic.py` | Training patterns, checkpoint logging |
| Weight stats | `run_long.py` | weight_stats function |

No new measurement code. The CLI is a thin dispatch layer over existing functions.

---

## Constraints

- Python 3.12+
- PyTorch (GPU optional)
- No web framework, no daemon, no database
- Single entry point: `./atft` or `python -m atft_cli`
- All output to stdout as JSON
- Human-readable summary to stderr only
- Exit codes: 0=success, 1=error, 2=invalid args
