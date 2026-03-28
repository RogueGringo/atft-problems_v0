# Tokenization Thesis: Adaptive Basis Discovery & Multi-Channel Architecture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Discover the natural representational basis of language models using ATFT topology, quantify how much current tokenization destroys, and design a multi-channel replacement grounded in empirical measurement.

**Core Thesis:** Current BPE tokenization flattens a simplicial complex of meaning into a 1D byte-pair sequence. The model then spends most of its capacity rediscovering the structure that was destroyed. We are at 1-10% optimization. The semantic primes (59 basis vectors of all human meaning) provide the seed for discovering the full natural basis — but the basis must be found adaptively, not assumed.

**Key Insight:** What is not directly encoded but true will be latent dimensions of the same truth topology structure. The residual (what the model encodes beyond the primes) should share topological properties with the prime subspace if it represents genuine knowledge. If it doesn't, it's noise or format exploitation.

---

## Stage A: Adaptive Basis Discovery (this spec)

### Architecture

Four modules, executed in sequence per model, swept across architectures.

### Module 1: Prime Basis Seed (`prime_basis.py`)

**Input:** A pretrained model (any HuggingFace causal LM).

**Prime aggregation rule:** Many semantic primes are multi-word ("a long time," "because of," "kind of"). For each prime phrase, run it through the model as a complete input, extract the hidden state of the LAST TOKEN at the target layer (layer n//3, consistent with Module 2's extraction point). This gives 59 vectors in hidden_dim space with the model's contextual representation, not raw embeddings. This is critical: the prime vectors must live in the same representational space as Module 2's hidden states.

**Process:**
1. For each of the 59 semantic primes, run through model, extract last-token hidden state at target layer → 59 vectors of shape (hidden_dim,).
2. Mean-center the 59 vectors. Compute SVD → orthonormal basis P (59 × hidden_dim).
3. For every token in the vocabulary, get its hidden-layer representation (run single-token inputs through the model to the same target layer). Compute:
   - `prime_projection = P @ P.T @ embedding` (component in prime subspace)
   - `residual = embedding - prime_projection` (component orthogonal to primes)
   - `prime_ratio = ||prime_projection|| / ||embedding||`
4. Record vocabulary-level statistics: mean/std/distribution of prime_ratio.

**Note on expected prime_ratio:** The prime subspace is at most 59-dimensional in a ~2048-4096 dimensional hidden space. A mean prime_ratio of 0.05-0.15 is geometric reality, not failure. The RELATIVE variation across tokens and the CORRELATION with model performance are the signals, not the absolute magnitude.

**Output:** `prime_basis` (59 × hidden_dim matrix), `residual_basis` (top-k singular vectors of residual space), per-token `prime_ratios`.

### Module 2: Adaptive Explorer (`adaptive_explorer.py`)

**Input:** A model + a diverse prompt set (100 prompts across 5 cognitive modes).

**Prompt Diversity Set:**
- 20 MMLU-Physics questions (technical/factual)
- 20 narrative prompts (creative/contextual)
- 20 logic puzzles (reasoning/structural)
- 20 shuffled English sentences (same tokens as narrative set, randomly reordered — controls for token distribution while destroying syntax)
- 20 multilingual parallel sentences (same meaning, different languages)

**Process — the adaptive loop:**
1. Initialize basis B = prime_basis (59 vectors).
2. For each iteration:
   a. Select next prompt batch via policy: first iteration uses one prompt from each cognitive mode (5 prompts). Subsequent iterations sample from the mode with the highest mean residual in the previous iteration (bandit-style exploration).
   b. Run model on each prompt, extract hidden states at target layer (layer n//3) for ALL tokens. Stack across all prompts in the batch → matrix of shape (total_tokens, hidden_dim). Mean-center before SVD.
   c. SVD of stacked hidden states → model's natural basis for these prompts (top-k singular vectors by magnitude).
   d. Compute alignment: cosine similarity between natural basis vectors and current B.
   e. Identify largest residual components (directions in hidden space not covered by B).
   f. If residual magnitude > threshold: add top-k residual vectors to B. Orthogonalize.
   g. Compute convergence metric: Gini of the singular value spectrum of B.
   h. If Gini change < epsilon for 3 consecutive iterations: STOP.
3. The exploration policy for step (a): rank unused prompts by expected residual magnitude, estimated from prompt features (length, vocabulary overlap with current basis, cognitive mode diversity).

**Output (saved as PyTorch `.pt` files with metadata dict):**
- `adaptive_basis` (n_discovered × hidden_dim tensor)
- `convergence_trajectory` (Gini per iteration, list of floats)
- `residual_history` (per-iteration residual magnitudes)
- `basis_growth_log` (which vectors were added at which iteration, from which cognitive mode, with residual magnitude)
- `metadata`: model name, hidden_dim, n_iterations, convergence_value, timestamp

**Convergence criterion:** Gini change < 0.005 for 3 consecutive iterations. Record the convergence value as the model's **representational fixed point** — a new measurement, not to be compared to the number-theoretic fixed points (0.997, 0.945, 0.837) which were computed on mathematical sequences, not neural representations. If the convergence value happens to land near a known fixed point, that is a finding to report, not a precondition for success. The primary comparison is ACROSS ARCHITECTURES: do different model families converge to the same value?

### Module 3: Topology Comparison (`topology_comparison.py`)

**Input:** Prime subspace, residual subspace, full embedding space, hidden states from correct/incorrect answers (from clean bench results).

**Four metrics:**

**Metric 1 — Topological Isomorphism:**
- Compute H₀ persistence → Gini in the prime subspace.
- Compute H₀ persistence → Gini in the residual subspace.
- If Gini(primes) ≈ Gini(residual): same manifold structure, different coordinates.
- If Gini(primes) ≠ Gini(residual): different structures.

**Metric 2 — Cross-Subspace Coherence:**
- Construct a Rips complex on token positions.
- Fiber at each position = prime-space projection.
- Transport between positions computed via residual-space correlation.
- Compute sheaf Laplacian spectral sum S.
- Small S → prime and residual subspaces are coupled (residual extends primes coherently).
- Large S → decoupled (residual is independent).

**Metric 3 — Persistence Under Projection:**
- Take hidden states from correctly-answered questions (likelihood mode from clean bench).
- Project onto: prime subspace, residual subspace, full space.
- Compute H₀ persistence in each projection.
- If truth signal survives in both projections: both subspaces carry truth.
- If it only survives in full space: truth requires both (entangled dimensions).

**Metric 4 — Adaptive Basis Convergence:**
- Track Gini of the adaptive basis across iterations.
- Record convergence value as the model's representational fixed point.
- Compare across architectures.

**Outcome Table:**

| Prime Gini ≈ Residual Gini | Cross-Coherence | Interpretation | Next Action |
|---|---|---|---|
| Yes | High | Residual = latent truth dimensions. Complete basis found. | Build multi-channel tokenizer on this basis. |
| Yes | Low | Two independent truth manifolds. Model sees what primes don't. | Investigate residual semantics — new primitives? |
| No | High | Residual is a transformation of prime structure. | Characterize the transformation (rotation? scaling?). |
| No | Low | Residual is noise/format artifacts. Primes are sufficient. | Multi-channel tokenizer needs only prime dimensions. |

### Module 4: Cross-Architecture Sweep (`sweep_runner.py`)

**Models (exact HuggingFace IDs, verified loadable):**
- `HuggingFaceTB/SmolLM2-360M-Instruct` (0.36B, SmolLM2)
- `Qwen/Qwen2.5-0.5B` (0.5B, Qwen2.5 — base model, not Instruct)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B, TinyLlama)
- `Qwen/Qwen2.5-1.5B-Instruct` (1.5B, Qwen2.5)
- `Qwen/Qwen2.5-3B-Instruct` (3.0B, Qwen2.5)
- `Qwen/Qwen2.5-7B-Instruct-AWQ` (7.0B, Qwen2.5 — 4-bit via autoawq, ~5.6GB VRAM)

Models are loaded sequentially, one at a time, with explicit cleanup (del model, gc.collect, torch.cuda.empty_cache) between each.

**Prerequisite:** Run clean bench (likelihood mode) on each model BEFORE Module 3, to partition questions into correct/incorrect sets for Metric 3. Models that already have clean bench results (Qwen 3B) reuse them.

**Per model:** Run Modules 1-3. Record:
- Prime ratio distribution (how prime-aligned are the embeddings?)
- Adaptive basis dimensionality (how many vectors needed for convergence?)
- Convergence Gini (the representational fixed point)
- Outcome table cell (which quadrant?)
- Cross-architecture: which families are most prime-aligned? Which have richest residuals?

**Compute estimates (RTX 5070, 12GB):**
- Module 1: ~5 min per model (59 forward passes + vocabulary projection)
- Module 2: ~15 min for ≤3B models, ~30 min for 7B-AWQ (100 forward passes + iterative SVD)
- Module 3 Metrics 1,3,4: ~5 min per model (persistence on subspace projections)
- Module 3 Metric 2 (sheaf Laplacian): ~10 min per model. Scope: subsample to N=200 token positions, k-nearest-neighbors graph with k=10 (not full Rips), use existing matfree Lanczos eigensolver from JTopo. If compute exceeds 15 min on any model, flag as stretch goal and report Metrics 1,3,4 only.
- Full sweep: ~3-4 hours total.

---

## Stage B: Sheaf-Valued Attention (future spec, contingent on Stage A)

If Stage A shows prime and residual subspaces share topology (top-left outcome), Stage B replaces standard attention:

- **Fiber** at each position = (prime_coords, latent_truth_coords) from discovered basis.
- **Transport map** between positions = learned matrix preserving topological structure.
- **Sheaf Laplacian regularization** = penalize incoherent transport during training.
- Attention becomes: "transform meaning from position j to position i via transport T_ij, where T must be topologically coherent."

Stage B gets its own spec after Stage A results are in.

---

## Success Criteria

1. The adaptive basis converges for at least 4/6 models (representational fixed point exists).
2. The outcome table has a dominant cell: at least 4 of 6 models land in the same quadrant (the pattern is universal, not model-specific).
3. At least one metric shows statistically significant correlation between basis properties and genuine knowledge (likelihood accuracy from clean bench).
4. The cross-architecture comparison reveals measurable differences in prime alignment between model families.

## Risks

1. **Basis doesn't converge.** The hidden state space may be too high-dimensional for 100 prompts to discover a stable basis. Mitigation: increase prompt set, add more iterations.
2. **Primes are too coarse.** The 59 primes were discovered for cross-linguistic communication, not for physics/math. They may not span technical domains. Mitigation: the adaptive expansion handles this — if primes are insufficient, the residual analysis finds what's missing.
3. **Topology measures are noisy at embedding scale.** H₀ persistence on 59 points has limited resolution. Mitigation: use effective rank and spectral gap as complementary measures (validated in prior experiments).
4. **Results are model-family-specific.** Qwen may show one pattern, SmolLM2 another. Mitigation: this IS informative — it tells us which architectures are more "linguistically honest."

## Dependencies

- Existing clean bench infrastructure (likelihood + MCQ evaluation)
- Existing topology measurement tools (spectral gap, effective rank, H₀ Gini)
- Semantic prime list (from `docs/semantic_deep_topology.json`)
- Pre-cached model weights for all 6 models
- MMLU-Physics dataset (cached locally)

## File Structure

```
products/topological-router/
  prime_basis.py           # Module 1: seed exploration
  adaptive_explorer.py     # Module 2: discovery loop
  topology_comparison.py   # Module 3: four metrics
  sweep_runner.py          # Module 4: cross-architecture execution
  prompts/
    physics.json           # 20 MMLU-Physics prompts
    narrative.json         # 20 narrative prompts
    logic.json             # 20 logic puzzles
    nonsense.json          # 20 random strings
    multilingual.json      # 20 parallel sentences
  results/
    basis_discovery/       # Per-model adaptive basis results
    topology_comparison/   # Per-model metric outputs
    sweep_summary.json     # Cross-architecture summary
```
