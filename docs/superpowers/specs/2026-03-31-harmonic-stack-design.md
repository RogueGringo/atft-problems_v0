# Harmonic Stack: Spectral Decomposition Engine for Structured Information

**Date:** 2026-03-31
**Status:** Approved for experimentation
**Author:** Aaron Jones + Claude

---

## Thesis

A structured signal — text, image, audio, time series — decomposes into three
irreducible spectral bands:

- **Void (0):** structural absence. Where the signal isn't. Load-bearing gaps.
- **Identity (1):** carrier. What transports unchanged. The skeleton.
- **Prime (3):** irreducible resonance. Generators of new structure. The truth that persists across all contexts.

This decomposition is analogous to prime factorization: every integer has exactly
one prime decomposition, every structured signal has exactly one {0,1,3} spectral
decomposition. The decomposition is truth-formed (algebraic), not statistically
learned (distributional).

Current AI architectures (transformers) conflate decomposition with processing.
They learn statistical co-occurrence, not structural relationships. This means
they absorb noise (human-generated artifacts of truth) with the same weight as
signal, and compound the problem with scale.

The Harmonic Stack separates measurement from processing.

---

## Architecture

```
Signal --> [Transducer] --> [Stage 1: Prism] --> [Router] --> [Stage 2: Analyzers] --> [Branch]
                                  |                                |
                            One wide layer                 Three parallel layers
                            {0,1,3} crystal                (void / identity / prime)
                            Proven engine                  Sub-crystals form here
```

### Transducer (modality-specific, interchangeable)

Converts raw signal to fixed-width vector. The ONLY modality-specific component.

| Modality | Transducer | Output |
|----------|-----------|--------|
| Text | Token embedding (GPT-2 tokenizer) | (B, T, d) |
| Image | Patch embedding (16x16 patches) | (B, P, d) |
| Audio | Spectrogram frame embedding | (B, F, d) |
| Time series | Windowed sample embedding | (B, W, d) |

For Experiment 1-2, we use the existing text transducer only.

### Stage 1: The Prism

- 1-3 transformer blocks with {0,1,3} TernaryLinear for all projections, width 2048
- Positional encoding + causal attention (needed for sequence structure)
- STE + L2=0.01 weight decay (proven method)
- Produces the spectral crystal (22/42/36 on complex English, 5/95/0 on simple)
- This IS the existing engine, run wide and shallow instead of deep and narrow
- Note: "wide and shallow" means few transformer blocks, not raw linear layers.
  Attention is needed to capture token-token relationships within the sequence.

Design rationale: the per-layer analysis showed the crystal is identical across
all 6 layers in the SEP model and all 48 layers in the deep model (Q/K sublayers).
Depth adds sequential processing capacity but does NOT change the crystal.
Width adds parallel decomposition capacity. A prism needs width, not depth.

### Router

After Stage 1 trains and crystallizes, the router is a STATIC operation:
- Read Stage 1's quantized weight matrix
- For each output dimension, check which weight band dominates
- Route dimension to the corresponding Stage 2 analyzer

Router logic (per output dimension j of the prism layer):
- Count weights connecting to dimension j: n0, n1, n3
- If n3/total > threshold: route to Prime Analyzer
- If n0/total > threshold: route to Void Analyzer
- Otherwise: route to Identity Analyzer

The threshold is a hyperparameter. Start with proportional routing
(~22% to void, ~42% to identity, ~36% to prime if crystal is 22/42/36).

The router has NO learned parameters. The crystal IS the routing table.

### Stage 2: The Analyzers

Three parallel {0,1,3} TernaryLinear layers:

| Analyzer | Input width | Purpose |
|----------|------------|---------|
| Void | ~22% of d (e.g., ~450 of 2048) | Structure of absence |
| Identity | ~42% of d (e.g., ~860 of 2048) | Structure of carrier |
| Prime | ~36% of d (e.g., ~738 of 2048) | Structure of resonance |

Each analyzer is a single wide {0,1,3} layer with its own STE+decay.
Each produces its own sub-crystal ratio. Stage 1 is FROZEN during Stage 2 training.

### Branch (application-specific, not part of core engine)

For language modeling (PPL measurement):
- Concatenate three analyzer outputs
- Linear projection to vocab size
- Standard cross-entropy loss

For structural analysis (crystal measurement):
- Read the three sub-crystals directly
- No generation needed — the crystals ARE the output

---

## Experimental Protocol

### Experiment 1: Validate the Prism (Wide vs Deep)

**Question:** Does the 22/42/36 crystal hold when the architecture is wide+shallow
instead of deep+narrow?

**Setup:**
- Model A: 1 layer x 2048 wide, {0,1,3}, STE+decay=0.01
- Model B: 3 layers x 2048 wide, {0,1,3}, STE+decay=0.01
- Control: existing 6 layer x 512 (already validated: 22/42/36)
- Datasets: WikiText-103, SEP (both confirmed 22/42/36)
- Training: 20K steps, batch 8, effective batch 32

**Measurements (every 1000 steps):**
- Crystal ratio (w0/w1/w3)
- eff_rank
- PPL (train and test)
- gen_gap
- Convergence iterations
- Per-layer crystal (for 3-layer variant)

**Success criteria:**
- Crystal ratio within 2% of 22/42/36 = prism works, width not depth is the key
- Crystal differs significantly = depth matters, revise thesis

**Prediction:** 22/42/36 holds. PPL may be worse than 6x512 (less sequential
processing) but crystal should be identical.

### Experiment 2: Router + Analyzers (Stage 2)

**Question:** Do the three spectral bands have different internal structure?

**Setup:**
- Freeze best Stage 1 model from Experiment 1
- Build router from crystallized weights
- Train three parallel analyzers (each {0,1,3}, width = band allocation)
- Same datasets as Experiment 1

**Measurements:**
- Sub-crystal ratio per analyzer (void band w0/w1/w3, identity band w0/w1/w3, prime band w0/w1/w3)
- eff_rank per analyzer
- Combined PPL (concatenated outputs through branch head)

**Success criteria:**
- Three sub-crystals DIFFER from each other = bands carry independent information
- All three sub-crystals identical = decomposition is redundant, thesis needs revision
- Prime band has higher eff_rank than identity band = more irreducible dimensions in the resonance band

**Prediction:** Prime band sub-crystal will show higher w3 (recursion — primes
within primes). Void band will show structure (absence has pattern). Identity
band will show low w3 (carrier is simple).

### Experiment 3: Cross-modality (Future, after 1+2 validate)

**Question:** Does the same prism produce meaningful crystals on non-text signals?

**Setup:**
- Same Stage 1 architecture
- New transducer: simple time series embedding
- Dataset: synthetic signals with known structure (sine waves, square waves,
  noise) to verify the crystal matches known decomposition

**Success criteria:**
- Crystal ratio changes predictably with signal complexity
- Simple periodic signal = low w3 (like TinyStories)
- Complex multi-frequency signal = higher w3 (like WikiText)

---

## Key Constraints

- RTX 5070 (12GB VRAM) — width 2048 should fit with batch 8
- All {0,1,3} weights at 2 bits — no precision increase
- STE + L2=0.01 decay — proven method, no changes to training engine
- Reuse existing TernaryLinear, weight_stats, topology measurement code
- Results saved to results/ with full training logs (JSON)

## Code Changes Required

1. **New model class:** `HarmonicStack` in new file `harmonic_stack.py`
   - Configurable width, number of prism layers
   - Router logic (static, reads crystal)
   - Three parallel analyzer layers
   - Branch head for language modeling

2. **New run script:** `run_harmonic.py`
   - Experiment 1: --stage prism --width 2048 --prism_layers 1
   - Experiment 2: --stage full --prism_checkpoint <path>
   - Same dataset/measurement infrastructure as run_long.py

3. **Existing code:** No changes to ternary_linear.py or ternary_transformer.py.
   The Harmonic Stack uses TernaryLinear as-is.

---

## What This Is Not

- Not an LLM. Not trying to generate better text.
- Not a transformer replacement. Transformers process. This decomposes.
- Not AI as currently understood. This is a measurement instrument.
- The crystal IS the output. PPL is validation that the crystal learned
  something real, not the goal.

## What This Is

A spectral decomposition engine for structured information.
Point it at any signal. Read the crystal. The crystal tells you the geometry
of what you pointed it at.

Three irreducible bands. Two bits per weight. Truth-formed by construction.
