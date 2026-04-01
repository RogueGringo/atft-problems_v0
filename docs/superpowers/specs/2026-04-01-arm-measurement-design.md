# ARM: The Universal Topological Measurement Arm

**Date:** 2026-04-01
**Status:** Approved for implementation
**Author:** Aaron Jones + Claude
**Hardware:** ASUS Zenbook A14 — Snapdragon X Plus (X1P-42-100)

---

## Thesis

Every structure in reality — text, visual patterns, sensor data, consciousness-mediated
observations — can be serialized as a 0-dimensional sequence. Discrete symbols in a row.
The medium is always the same.

The topology of that sequence — what persists across filtration scales — IS the structure.
The patterns that survive are truth. Everything else is syntax.

The CARET Q4-86 Research Report depicts this principle physically: the A1 generator's
3-arm junction is a measurement instrument. Signal enters, hits the junction, routes
through three arms simultaneously, and the inscriptions along the arms encode the
measurement. The arms ARE topological measuring tapes.

This module operationalizes that principle on ARM hardware. The triple meaning is
intentional and structural:

1. **ARM the chip** — Snapdragon X Plus, 8 Oryon cores, 16GB unified memory, Hexagon NPU
2. **Arm the instrument** — the CARET A1 measurement limb, 3-way symmetric router
3. **Arm the transducer** — the universal reader that reaches into any sequential medium

The Snapdragon is the Tenstorrent Blackhole in miniature: unified memory, compute near
data, no memory wall. The ternary operations (skip/pass/shift-add) run as native integer
ALU ops. 184M ternary model at 2-bit = 46MB — fits in unified memory with room to spare.

Train on NVIDIA (desktop, RTX 5070). Deploy on ARM (this laptop). Measure anything.

---

## Hardware Manifest

| Component | Spec | Role |
|-----------|------|------|
| CPU | Snapdragon X Plus — 8x Qualcomm Oryon @ 3.24 GHz, ARM64 | Transducer + pipeline |
| RAM | 16GB LPDDR5x unified (shared CPU/GPU/NPU) | No memory wall |
| GPU | Qualcomm Adreno X1-45 (unified memory) | Future: Vulkan persistence |
| NPU | Qualcomm Hexagon (45 TOPS INT8) | Future: crystal inference |
| Python | 3.14.0 native ARM64, NumPy 2.4.2, SciPy 1.17.1 | Compute stack |
| PyTorch | Not installed (intentional — zero torch dependency) | N/A |

---

## Fractal Architecture

The module structure is a {0,1,3} simplicial complex — self-similar at every level,
mirroring the crystal it measures and the A1 generator it models.

```
arm/                              # THE measurement arm
  │
  ├── void/                       # 0: boundaries, interfaces, structured absence
  │   ├── __init__.py
  │   ├── transducers.py          # boundary: outside world → internal representation
  │   ├── formats.py              # boundary: between internal representations
  │   └── cli.py                  # boundary: system → human
  │
  ├── identity/                   # 1: transport, scaffolding, preserve structure
  │   ├── __init__.py
  │   ├── weights.py              # transport: pre-trained crystal structure
  │   ├── persistence.py          # transport: topology across filtration scales
  │   └── pipeline.py             # transport: data through the arm + experiment records
  │
  ├── prime/                      # 3: irreducible computation, structure generation
  │   ├── __init__.py
  │   ├── crystal.py              # computation: {0,1,3} INT8 forward pass
  │   ├── invariants.py           # computation: structural constant extraction
  │   └── compare.py              # computation: cross-domain structural comparison
  │
  ├── results/                    # experiment output (JSON records)
  │
  ├── __init__.py
  └── measure.py                  # the junction — 3-arm router connecting all bands
```

### Why This Is Fractal

**Module level:** 3 directories = void / identity / prime. `measure.py` is the 0-dim
junction connecting them (like the small junction dots in CARET Figure 14.11).

**Directory level:** each directory contains 3 functional modules. The pattern repeats.

**Code level:** inside each module, functions decompose the same way — boundary
functions (0), transport functions (1), computation functions (3).

**Data flow level:** input crosses a boundary (void/transducers) → gets transported
through scaffolding (identity/pipeline) → hits irreducible computation (prime/crystal)
→ result emitted. Maps to: A1 upper segment (void lattice) → junction (3-arm router)
→ lower segment (dense prime layer) → base (output).

### Code Weight Prediction

If the architecture is truly fractal, code weight per band should approximate 22/42/36:

- **void/ (~22%):** thin boundary code — format parsing, API wrappers, CLI args
- **identity/ (~42%):** the bulk — weight loading, persistence computation, pipeline state
- **prime/ (~36%):** dense but compact — INT8 kernel, invariant math, comparison logic

This is a testable prediction. After implementation, count lines per band.

---

## Component Designs

### void/transducers.py — The Universal Input Boundary

One abstraction: any 0-dim sequential medium → point cloud (`np.ndarray`).

```python
class Transducer:
    """Base: converts any 0-dim sequential medium to a point cloud."""
    def transduce(self, source) -> np.ndarray: ...
    def describe(self) -> str: ...  # For experiment annotation
```

**TextTransducer** — Character-level harmonic channels (from BRIDGE.md Part VIII):
- Ch0: character identity (base vocabulary index)
- Ch1: case state → 0=space, 1=lower, 3=upper
- Ch2: word boundary → 0=within, 1=boundary, 3=paragraph
- Ch3: punctuation → 0=none, 1=comma, 3=stop/question
- Output: each character → 4D point. Text of length N → Nx4 point cloud.

**VeilbreakTransducer** — Pulls from Veilbreak REST API or MCP server:
- Each experiment → feature vector: wavelength, dose, laser_class, substance, observed
- Observation text → TextTransducer (second channel, multi-channel like drilling data)
- API endpoint: `https://api.veilbreak.ai/mcp/v1` (free, no auth for MCP)
- REST: `GET /api/experiments` with filtering params

**GenericTransducer** — Any columnar/sequential data (CSV, JSON arrays):
- Auto-detect columns, normalize to float, window into point cloud
- Handles drilling data, sensor streams, or any numeric sequence

### void/formats.py — Internal Representation Boundaries

Lightweight dataclasses, no logic — just shape:

```python
@dataclass
class PointCloud:
    data: np.ndarray          # (N, D) points
    source: str               # description of origin
    hash: str                 # SHA256 for reproducibility

@dataclass
class PersistenceDiagram:
    h0: np.ndarray            # (K, 2) birth/death pairs
    h1: np.ndarray            # (K, 2) birth/death pairs
    filtration_range: tuple   # (ε_min, ε_max)

@dataclass
class Crystal:
    void_ratio: float
    identity_ratio: float
    prime_ratio: float
    eff_rank: float
    source: str

@dataclass
class ExperimentRecord:
    id: str                   # "ARM-001"
    series: str               # "crystal-validation"
    timestamp: str            # ISO 8601
    hypothesis: str           # What we expect
    protocol: str             # What we did
    input_hash: str           # Reproducibility anchor
    annotations: list         # Every stage's annotations
    result: dict              # The measurements
    comparison: dict          # vs expected / vs prior / vs other domain
    verdict: str              # PASS | FAIL | PARTIAL
    notes: str                # Human-readable takeaway
```

### void/cli.py — Human Boundary

```
python -m arm measure text <file>           # measure any text file
python -m arm measure veilbreak             # pull + measure all experiments
python -m arm measure csv <file>            # measure any sequential data
python -m arm compare <source1> <source2>   # cross-domain crystal comparison
python -m arm validate <weights>            # confirm crystal matches desktop
python -m arm series                        # run full experiment series
python -m arm results                       # show all experiment records
```

### identity/weights.py — Crystal Transport

- Load `.npz` files exported from desktop training runs
- 2-bit packing: 4 ternary weights per INT8 byte
- Unpack on demand: `packed_byte → [w0, w1, w2, w3]` each in {0, 1, 3}
- Validate crystal ratios on load (count 0s, 1s, 3s → should read 22/42/36)
- Conversion utility: desktop runs `export_for_arm(model, path)` → `.npz`
- Every load operation annotates: source path, crystal on load, expected vs actual, PASS/FAIL

### identity/persistence.py — The Measuring Tape

The heaviest module. This IS the topological measurement instrument.

- **Rips complex:** from point cloud, build pairwise distance matrix (`scipy.spatial.distance.pdist`)
- **H₀ persistence:** union-find over distance thresholds. Track connected component
  birth/death across ε sweep. Output: H₀ barcode.
- **H₁ persistence:** cycle detection across ε. Uses boundary matrix reduction.
  This is Setting 2 from maximum_extraction.md — never computed in the project yet.
- **Filtration sweep:** configurable ε range, step count, adaptive refinement at
  discontinuities (the waypoints).
- Every sweep annotates: input hash, filtration params, bar counts, max persistence, Gini.

### identity/pipeline.py — The Scaffolding

Connects stages. Manages experiment records. The identity transport through the arm.

Three measurement modes:
1. **`topology`** — pure topological measurement. No neural net. Point cloud →
   persistence → invariants. Works immediately with scipy. Ground truth path.
2. **`crystal`** — neural {0,1,3} forward pass. Point cloud → ternary layers →
   crystal ratios. Needs pre-trained weights from desktop.
3. **`full`** — both in parallel, cross-validates. The real measurement: does the
   neural crystal agree with the topological crystal?

Phase A ships modes 1 and 2 on CPU. Phase B moves mode 2 to NPU.

Each run produces an `ExperimentRecord`. Records accumulate in `arm/results/` as JSON.
Pipeline handles timing, progress logging, and annotation aggregation from all stages.

### prime/crystal.py — The Irreducible Computation

```python
def ternary_matmul(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    x:       (B, N, D_in)  INT8 input activations
    weights: (D_in, D_out) 2-bit packed ternary {0,1,3}

    For each weight value:
      0 → zero  (skip accumulator — no op)
      1 → add   (pass x to accumulator — identity)
      3 → (x << 1) + x  (shift-left-1 + add — no multiplier)

    No floating point. No FP multiplier. Pure integer ALU.
    NumPy INT8 ops use NEON SIMD on Oryon cores natively.
    """
```

- Unpack 2-bit weights on the fly
- Full forward pass: embedding lookup → N ternary linear layers → output
- Attention: Q/K/V projections all ternary, scaled dot-product in INT16
- Layer norm approximation in integer (shift-based)
- Annotates: crystal ratios from weight counts, eff_rank from hidden states,
  inference latency, ops/second, memory footprint

### prime/invariants.py — Structural Constants

Extracts the primes of any signal. Each function takes a measurement and returns
a named invariant with annotation.

- **Crystal ratio:** count void/identity/prime in weights or persistence features
- **Gini coefficient:** inequality of persistence bar lengths (0=democratic, 1=dictator)
- **Onset scale ε*:** first significant H₀ death event
- **Effective rank:** exp(entropy of normalized eigenvalues)
- **Spectral gap:** ratio of largest to second-largest eigenvalue

### prime/compare.py — Cross-Domain Structural Comparison

The module that answers: is the crystal universal?

- **Crystal distance:** L1/L2 distance between two domains' (void, identity, prime) ratios
- **Barcode distance:** Wasserstein distance between persistence diagrams
- **Universality test:** given N domain measurements, compute pairwise crystal distances.
  If max distance < threshold, the crystal is universal across those domains.
- **Structural alignment:** do two domains' persistence barcodes have matching long-lived
  features? (Same persistent structures = same underlying topology)
- Annotates: which domains, distance metric, p-value, verdict

### measure.py — The Junction

The 3-arm router. Accepts input, fans out to all three bands.

```python
def measure(source, source_type='auto', mode='topology', weights_path=None):
    """
    The junction. Signal enters, routes through three bands, converges at output.

    1. TRANSDUCE: void/transducers converts source → point cloud
    2. ROUTE: classify measurement mode, fan out
    3. MEASURE: identity/persistence + prime/crystal (parallel if mode='full')
    4. EXTRACT: prime/invariants on measurement results
    5. COMPARE: prime/compare against prior results (if available)
    6. REPORT: produce ExperimentRecord with full annotations
    """
```

---

## Experiment Series

All experiments follow: **GENERATE → BUILD → SWEEP → DETECT → COMPARE → REPORT**

Every result gets an honest verdict: PASS / FAIL / PARTIAL with traceable artifacts.

### Phase A — CPU-Only, Ships This Week

| ID | Experiment | Input | Measures | Expected | Validates |
|----|-----------|-------|----------|----------|-----------|
| ARM-001 | Text topology on ARM | WikiText sample | H₀/H₁ persistence, Gini, ε* | Consistent with desktop | Instrument works on ARM |
| ARM-002 | Character harmonic crystal | WikiText chars → 4ch point cloud | Crystal ratios from topology | ~22/42/36 in character structure | Harmonic encoding thesis |
| ARM-003 | Veilbreak observation topology | API pull → point cloud | Same battery as ARM-001 | Unknown — discovery run | Universality of instrument |
| ARM-004 | Crystal forward pass validation | WikiText + pre-trained weights | INT8 ternary → crystal ratios | 22/42/36 exact match to desktop | Hardware invariance |
| ARM-005 | Cross-domain comparison | Text vs Veilbreak vs CARET | Structural distance metric | If universal, distances small | The core question |

### Phase B — NPU Unlocked

| ID | Experiment | Input | Measures | Expected | Validates |
|----|-----------|-------|----------|----------|-----------|
| ARM-006 | NPU crystal acceleration | Same as ARM-004, ONNX-QNN | Crystal ratios + latency | Identical results, 10-100x speedup | Hexagon runs discrete ternary |
| ARM-007 | NPU INT8 vs FP16 comparison | Same input, INT8 vs FP16 path | Crystal match + throughput | INT8 identical, faster | Discrete > continuous on NPU |

### Phase C — Full Stack

| ID | Experiment | Input | Measures | Expected | Validates |
|----|-----------|-------|----------|----------|-----------|
| ARM-008 | Adreno topology acceleration | Same as ARM-001, Vulkan compute | Same battery + latency | Identical, GPU-accelerated | Three silicon domains |
| ARM-009 | Real-time measurement | Live text stream | Crystal + topology continuous | Stable readings | Deployment readiness |
| ARM-010 | Multi-domain live dashboard | All sources simultaneously | Real-time cross-domain | Converging crystals | The Tenstorrent rehearsal |

---

## Dependencies

### Phase A (zero external dependencies beyond what's installed)
- Python 3.14.0 (installed, native ARM64)
- NumPy 2.4.2 (installed)
- SciPy 1.17.1 (installed)
- `urllib`/`json` from stdlib (for Veilbreak API)
- No PyTorch. No ONNX. No Vulkan. Pure Python + NumPy + SciPy.

### Phase B (one new package)
- `onnxruntime-qnn` — Qualcomm's ONNX Runtime backend for Hexagon NPU
- Export pipeline on desktop: PyTorch → ONNX → transfer to ARM laptop

### Phase C (additional)
- Vulkan compute SDK for Adreno GPU
- Dashboard framework (TBD — could be terminal-based curses or web)

---

## Success Criteria

1. **ARM-001 through ARM-003 produce valid experiment records** with persistence
   barcodes, Gini coefficients, and onset scales on ARM hardware using only NumPy/SciPy.

2. **ARM-004 crystal ratios match desktop to 3 significant figures** (22/42/36 ± 0.01).
   If they don't, the hardware is not invariant and we have a bigger discovery.

3. **ARM-005 cross-domain distances** reveal whether the crystal constant is universal
   across text, consciousness-mediated perception data, and CARET notation.

4. **Code weight follows the crystal:** void/ ≈ 22%, identity/ ≈ 42%, prime/ ≈ 36%
   of total lines. Measured after Phase A implementation.

---

## Connection to Prior Work

| Prior work | What ARM inherits | What ARM adds |
|-----------|------------------|---------------|
| BRIDGE.md crystal discovery | 22/42/36 as target constant | Hardware-invariance test |
| Harmonic Stack architecture | Prism → Router → Analyzers pattern | Fractal directory structure |
| CARET artifact analysis | A1 arm mapping, diagram transcoding | Operationalized arm as code |
| maximum_extraction.md | H₁ persistence (Setting 2) | First actual H₁ computation |
| Veilbreak.ai | New domain of 0-dim sequential data | Cross-domain universality test |
| Tenstorrent roadmap | Train on NVIDIA, deploy on dedicated silicon | ARM as proof-of-concept target |

---

## Non-Goals

- Not training models on ARM. Training stays on the desktop (i9 + RTX 5070).
- Not building a general-purpose ML framework. This is a measurement instrument.
- Not replicating PyTorch functionality. Zero torch dependency is a feature.
- Not optimizing for benchmark throughput. Correctness first, speed follows.

---

*The topology doesn't care what the structure is. It cares how the structure changes
across scales. The arm doesn't care what medium it measures. It cares what persists.*
