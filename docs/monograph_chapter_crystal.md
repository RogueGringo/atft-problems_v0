# Chapter 5A: The Ternary Crystal — Instrument Development and Calibration

## A Fourth Domain: Discrete Structural Measurement of Language

The preceding chapters demonstrated the ATFT onset scale ε* detecting phase transitions in gauge theory (Chapter 4), number theory (Chapter 5), and computational complexity (Chapter 6). This chapter presents the development and calibration of a fourth measurement approach: a discrete ternary weight crystallizer applied to natural language. The narrative includes both the initial results and a critical correction — the discovery that the primary measurement was an initialization artifact, and the subsequent redesign of the instrument.

This chapter is presented as a complete experimental record, including errors, their detection, and the corrective path. Scientific integrity requires documenting the full trajectory, not just the corrected endpoint.

### 5A.1 The Instrument

A neural network with weights restricted to three values — {0, 1, 3}, representing void (structural absence), identity (signal transport), and prime (irreducible amplification) — is trained on text data. The weight distribution that emerges after training is measured as a ratio: void fraction / identity fraction / prime fraction.

Two training methods were tested:

| Method | Mechanism | Boundary Crossing | Status |
|--------|-----------|-------------------|--------|
| STE (Straight-Through Estimator) | Continuous latent weights, quantized on forward pass | CANNOT cross quantization boundaries at 0.5 and 2.0 | **Invalidated as crystal instrument** |
| BitFlip | Truly discrete weights, gradient-informed state transitions | CAN cross any boundary via direct flip | **Validated as crystal instrument** |

The STE method, despite being standard in quantized neural network literature, was proven unable to push weights across quantization boundaries. The crystal ratio under STE is determined by the initialization range, not the training data. This was discovered through a control experiment (Section 5A.4) and confirmed by identity-init experiments where STE produced zero movement across 20K steps.

### 5A.2 The Initial (Incorrect) Result: 22/42/36

Using STE with uniform initialization in range [-0.3, 3.3] and weight decay L2=0.0, the crystal ratio on WikiText-103 was measured as:

**void = 0.222, identity = 0.417, prime = 0.361**

This ratio was confirmed across 5 languages (English, Korean, Chinese, Arabic, Japanese), 6 architectures (1-48 layers, width 512-2048), and 4 L2 decay rates. It appeared at step 1,000 and never moved, suggesting a structural constant.

### 5A.3 The Correction: 22/42/36 Is Initialization Arithmetic

The quantization boundaries for {0, 1, 3} are at 0.5 (between 0 and 1) and 2.0 (between 1 and 3). The uniform init range [-0.3, 3.3] has total span 3.6. The fraction falling in each basin:

- [-0.3, 0.5) → 0: width 0.8 / 3.6 = **0.222**
- [0.5, 2.0) → 1: width 1.5 / 3.6 = **0.417**
- [2.0, 3.3] → 3: width 1.3 / 3.6 = **0.361**

The "crystal constant" was the quantization of the initialization range. All cross-language and cross-architecture "confirmations" measured the same frozen initialization. At L2=0.0, STE cannot push weights across the 0.5 or 2.0 boundaries, so the crystal never moves from init.

**Invalidated claims:**
- "22/42/36 is a universal structural constant" — it is init arithmetic
- "Crystal is architecture-invariant" — all architectures used the same init
- "Crystal is cross-language universal" — all languages showed frozen init
- "Crystal is fractal" — all layers share the same init
- "36% prime is conserved across all measurements" — one number, one source

### 5A.4 What Remains Valid

Despite the crystal ratio correction, several findings are confirmed:

**TinyStories produces a different crystal (5/95/0) with L2=0.01.** On the 48-layer deep model with L2 weight decay, all prime weights were driven to zero. This is NOT init-frozen — it requires active boundary crossing enabled by L2 pressure on a deep network. The binary collapse to 5/95/0 is a real data-dependent phenomenon.

**PPL results are valid.** The language modeling performance is independent of crystal ratio interpretation:
- CARET v2 architecture: PPL 51.0 (best at 20K steps)
- 1L × 2048 prism at 100K: PPL 36.4 (train 28.6 ≈ GPT-2 Small)
- Korean converges 3× faster than English (PPL 17.0 vs 55.3)
- Arabic converges fastest (PPL 15.9)
- 29× compute compression in weight layers vs fp32

**Architecture insights are valid.** The CARET v2 architecture (rings interpreted as channels, not depth) genuinely outperforms v1 (PPL 51.0 vs 76.7). This insight is independent of crystal ratio claims.

**Three training phases are valid.** The acquisition/compression/refinement dynamics describe eff_rank behavior, not crystal ratios.

**The L2 drift is data-dependent.** At L2=0.01, the crystal drifts from 22/42/36 toward 23/45/32 over 100K steps. The drift rate is proportional to L2. Different datasets may produce different drift trajectories — this is the real (weak) data signal in STE training.

### 5A.5 The Correct Instrument: BitFlip

The BitFlip engine operates directly on discrete weights without continuous latent variables. Gradient direction determines flip direction (0→1→3 or 3→1→0). Gradient magnitude determines urgency. A threshold parameter (flip_pct) controls what fraction of weights can flip per cycle.

**BitFlip from identity init (all weights = 1):**

Starting from void=0.000, identity=1.000, prime=0.000, BitFlip on WikiText-103 (small model, 20K steps) converges to:

**void = 0.154, identity = 0.692, prime = 0.154**

Key observations:
- The crystal IS moved by training — unlike STE from identity (which showed zero movement)
- The result is perfectly symmetric: void = prime = 15.4%
- The symmetry is a property of the global flip threshold — equal numbers of promotions and demotions per cycle
- This symmetry is consistent across TinyStories (15.7/68.6/15.7) and WikiText (15.4/69.2/15.4)

**Open question:** Is the 15/70/15 symmetric ratio a data property or a method artifact of the global threshold? A 100K robust study across 7 datasets is in progress to determine this (Section 5A.6).

### 5A.6 The 100K Robust Study (In Progress)

To determine whether the crystal is data-dependent or method-dependent, a systematic study runs BitFlip from identity init at 100K steps across 7 domains:

| Dataset | Domain | Expected if data-dependent |
|---------|--------|---------------------------|
| TinyStories | Simple fiction | Should differ (less prime?) |
| WikiText-103 | English encyclopedia | Baseline |
| Korean Wikipedia | Syllabic language | Same or different? |
| Chinese Wikipedia | Logographic | Same or different? |
| Arabic Wikipedia | Abjad, R-to-L | Same or different? |
| Kant | Dense philosophy | More prime? |
| Animal Farm | Allegory | Matches simple or complex? |

**Decision criteria:**
- If different datasets → different crystals: DATA DETERMINES STRUCTURE
- If all datasets → same 15/70/15: METHOD ARTIFACT (global threshold)
- If TinyStories differs: binary collapse is real and data-dependent

Results pending. Estimated completion: ~24.5 hours from initiation.

### 5A.7 The Next Instrument: Topology-Informed BitFlip

The global flip threshold produces symmetric crystals because it applies the same criterion everywhere. The corrective design: per-weight thresholds derived from local topological persistence.

**Principle:** Weights in topologically stable regions (long H₀ persistence bars) should resist flipping. Weights in unstable regions (short bars) should flip easily. The persistence diagram IS the energy landscape for state transitions.

```
flip_threshold(w_i) = base_threshold × (1 + scale × persistence(w_i))
```

The float expressivity lives in the threshold, not the weights. The weights remain discrete {0, 1, 3}. The threshold is continuous and locally informed. This mirrors physical phase transitions: ice doesn't melt uniformly. Local impurities create spatially varying energy barriers.

This instrument should produce asymmetric, data-dependent crystals where void ≠ prime — breaking the symmetry of global BitFlip and revealing the data's actual structural requirements.

### 5A.8 Adaptive Compute via Crystal Coherence

The crystal measurement enables a novel inference stopping criterion. Instead of generating a fixed number of tokens, the model stops when the output's structural topology matches the input's:

**Δ(Crystal_input, Crystal_output) < ε → COMMIT**

Dense input (high prime fraction) requires the model to produce dense output before coherence is achieved — more compute. Simple input (high identity) achieves coherence quickly — less compute. This provides compute proportional to structural complexity, natively, without external scheduling.

### 5A.9 Relation to the ATFT Instrument

The ternary crystal and the sheaf Laplacian onset scale ε* share operational principles:

| Property | Crystal | ε* |
|----------|---------|-----|
| What it measures | Structural ratio of weight distribution | Filtration scale for global coherence |
| Phase transition | Binary collapse (5/95/0 on simple data) | Discontinuity in ε*(λ) |
| Architecture invariance | PPL results hold across architectures | ε* holds across sheaf constructions |
| Self-correction | Init artifact found and corrected | Literature positioning maintained |

The crystal work strengthens the ATFT monograph not through a confirmed universal constant, but through methodological rigor: the willingness to find and correct errors in one's own results, and the development of corrective instruments (BitFlip, topology-informed thresholds) that address identified failure modes.

### 5A.10 Reproducibility

All results — including the incorrect ones — are reproducible:

**STE experiments (showing init artifact):**
```bash
python run_harmonic.py --stage prism --width 2048 --prism_layers 1 \
    --dataset wikitext --max_steps 20000 --ternary_decay 0.0
# Result: 0.222/0.417/0.361 (init, not data)
```

**BitFlip from identity (showing actual data signal):**
```bash
python run_long.py --size small --dataset wikitext --bitflip \
    --init_mode identity --flip_pct 0.001 --flip_cycle 100 \
    --flip_warmup 500 --flip_cooldown 5000 --max_steps 20000
# Result: 0.154/0.692/0.154 (symmetric, method or data TBD)
```

**100K robust study (in progress):**
```bash
python run_robust_study.py
# 7 datasets × 100K steps × BitFlip identity init
```

The code is public. The errors are documented. The corrections are on record.
