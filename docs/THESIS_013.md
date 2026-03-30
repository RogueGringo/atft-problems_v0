# The {0, 1, 3} Thesis: Void, Unit, Prime

> Emerged from topology measurements on BitNet b1.58 — ternary weights produce maximally democratic hidden states (effective rank 485 vs Qwen's 12). This led to the question: what if the weight set isn't {-1, 0, 1} but {0, 1, 3}?

## The Argument

### Current state: continuous weights are mostly noise
- Models use 99.5% of their discovered representational basis
- But Qwen's effective rank is 12 out of 1536 dimensions — 0.8% of the space
- The model is 99.5% efficient at using 0.8% of its capacity = globally 0.8% efficient
- fp16 precision encodes magnitude information that is almost entirely noise

### BitNet discovery: ternary produces democracy
- BitNet b1.58 2B: effective rank 485 out of 2560 (19%)
- 24x more of the representational space is active
- The constraint {-1, 0, 1} prevents concentration → forces democratic distribution
- Residual is LESS hierarchical than prime projection (unique among all tested models)

### The {0, 1, 3} extension: void + unit + irreducible structure
- {-1, 0, 1} encodes: direction (±) + absence
- {0, 1, 3} encodes: absence + presence + structural importance
- 3 is irreducible (prime) — it can't be decomposed into smaller factors
- The weight set contains the minimum needed for discrete computation:
  - 0: void (no connection, structured absence)
  - 1: unit (connection exists, identity transport)
  - 3: prime (structural amplification, irreducible)

### Why {0, 1, 3} and not {0, 1, 2}?
- 2 = 1 + 1 (redundant, decomposable)
- 3 is the first odd prime (irreducible, generates new structure)
- A triangle (3 sides) is the first rigid form
- Subject-verb-object (3 elements) is the minimum for meaning
- H₁ requires 3 points to form a loop

### Computational properties
- 2 bits per weight (same as BitNet)
- × 0 = skip (FREE), × 1 = pass through (FREE), × 3 = shift+add (cheap)
- ~67% of operations become trivial
- 48B inference on 12GB VRAM
- No floating point multiplier needed anywhere

### The sheaf connection
Each weight position is not a scalar but a TRANSPORT MAP:
- 0-transport: null (these dimensions don't interact)
- 1-transport: identity (these dimensions pass through unchanged)
- 3-transport: prime amplification (these dimensions generate structure)

The network IS a sheaf. The weights ARE sections. The computation IS spectral.

### The zeta parallel
The zeros of {0, 1, 3} weight matrices play the role of zeta zeros:
- They are structured absence (not random sparsity)
- The PATTERN of zeros encodes the network's topology
- Information about what the network computes lives in WHERE it vanishes
- Just as prime distribution is encoded in zeta's zeros, the network's capabilities are encoded in its weight zeros

### Training data efficiency
{0, 1, 3} weights CANNOT memorize noise:
- No precision for subtle magnitude encoding
- Only structural relationships survive training
- The architecture IS the data curator
- Prediction: same data → faster convergence, better generalization, less data needed

## What needs testing

1. Does {0, 1, 3} produce measurably different representational geometry than {-1, 0, 1}?
2. Does the effective rank increase (more democratic) or stay the same?
3. Does the "structured absence" pattern actually carry information?
4. Does training converge? (STE should work the same as BitNet)
5. At what scale does {0, 1, 3} match fp16 performance?

## First experiment (next session)
- Build a {0, 1, 3} linear layer with straight-through estimator
- Drop it into a minimal transformer (50-100M params)
- Train for 20 minutes on TinyStories (RTX 5070)
- Measure: effective rank, spectral gap, Gini, and perplexity
- Compare: {0, 1, 3} vs {-1, 0, 1} vs fp16 at same size
- One layer, one test, one answer

## Origin
This thesis emerged from a session-long investigation starting with GPQA answer topology, through MCQ benchmark analysis, cross-architecture sweeps, validation batteries, BitNet profiling, and architecture autopsy. Each step narrowed the space until {0, 1, 3} crystallized as the minimal computational basis for structure-preserving neural computation.
