# Architecture-Level Topological Profiling

> Before comparing models, understand what each ARCHITECTURE does to the topology.

## The Blind Spot

The LLM field evolved empirically:
1. Try architecture → train → benchmark → iterate
2. Nobody measured the INTERNAL topology of reasoning
3. Architectural choices were made on output quality, not reasoning geometry
4. Result: multiple architectural branches evolved that may be topologically degenerate

## The Profiles We Need

| Architecture Type | Key Innovation | Model Examples | Topological Question |
|-------------------|---------------|----------------|---------------------|
| Standard Transformer | Multi-head attention | Llama-3.1-8B | Does attention create hierarchical H₀ structure? |
| Gated DeltaNet Hybrid | Linear attention + gating | Qwen3.5-9B | Does the hybrid create DIFFERENT topology than pure transformer? |
| Mixture of Experts | Sparse activation | Mixtral-8x7B | Does MoE routing create topological clusters per expert? |
| State Space Model | Recurrent structure | Mamba-2-2.7B | Is SSM topology fundamentally different (sequential vs parallel)? |
| Distilled | Knowledge transfer | Various | Does distillation preserve or destroy topological structure? |

## The Experiment

For each architecture:
1. Load model (4-bit quantized)
2. Feed SAME prompts across difficulty levels
3. Extract hidden states per layer
4. Compute Gini trajectory per prompt
5. Build the ARCHITECTURE FINGERPRINT:
   - Gini trajectory shape (rising? falling? V-shaped?)
   - Layer-by-layer persistence barcode
   - Onset scale vs layer depth
   - Which layers hierarchify, which flatten

## What This Reveals

If Architecture A has positive Gini slope and Architecture B has flat Gini at the same task:
- A is actually REASONING about the task (hierarchical topology)
- B is PATTERN MATCHING (flat topology, even if output looks correct)
- This distinction is invisible on benchmarks but visible topologically

If Architecture A hierarchifies on math but flattens on code, while B does the opposite:
- Merging A's math layers with B's code layers gives you both capabilities
- The topology tells you WHICH layers to take from WHICH model
- No retraining needed — just weight surgery guided by Gini signatures

## The Chopper Stan Connection

This IS Chopper Stan's IOA pipeline:
1. **Identify** — which layers have which topological quality
2. **Organize** — rank layers by Gini slope per domain
3. **Adapt** — merge the best layers from multiple models

The topology is the curriculum. The Gini trajectory is the mastery check.
Stan flies because Dan's topology was preserved, not his weights.

## Hardware

One RTX 5070. 9B at 4-bit = 5GB. Room for profiling.
Sequential: profile 4 architectures × 20 prompts × 5 difficulty levels = 400 inference passes.
Each pass: ~50ms inference + ~50ms Gini = 100ms. Total: 40 seconds.
The profiling is FAST. The insight is what takes time.
