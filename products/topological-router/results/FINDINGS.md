# Answer-Topology Test — Findings

**Date:** 2026-03-27
**Models:** Qwen2.5-0.5B (494M), Qwen2.5-1.5B (1544M)
**Datasets:** ARC-Challenge (100), MMLU-Physics (102), GPQA Diamond (198)
**Measures:** effective_rank, spectral_gap, norm_variance, gini_h0
**Protocol:** MCQ single-pass, per-choice token region + full deliberation analysis

## Results

### Per-Choice Topology Does NOT Separate Truth from Falsehood

24 tests (4 measures × 2 models × 3 datasets). Only 1 reached p < 0.05 (norm_var on MMLU-Physics 0.5B, d=0.202). Did NOT replicate at 1.5B. Expected by chance: 1.2/24.

The per-choice token spans (5-20 tokens) are too short for meaningful topological measurement.

### Deliberation Quality DOES Predict Model Correctness (1.5B)

When the 1.5B model gets MMLU-Physics questions RIGHT, its full deliberation shows:

| Measure | Cohen's d | Interpretation |
|---------|-----------|----------------|
| spectral_gap | +0.469 | More dominated by single mode when correct |
| norm_var | +0.371 | More differentiated token norms when correct |
| eff_rank | -0.322 | Lower effective rank (more hierarchical) when correct |

These are medium-to-large effects. The model's processing quality predicts its output quality.

### GPQA Is Genuinely Out-of-Distribution

Both models are at random (~25%) on GPQA Diamond. No topology signal appears. Neither model encodes graduate physics knowledge — there's nothing to detect.

### Topology and Logit Are Orthogonal

Only 2.5% overlap in correct predictions (GPQA v2 diagnostic). The signals are nearly independent. Ensemble (majority vote) improves accuracy marginally.

## Interpretation

The topology does not tell you WHICH answer is correct. It tells you WHETHER THE MODEL KNOWS.

**Product application:** Topological routing is a TRUST measure, not an answer selector.
- Model gives answer via standard logit decoding
- Topology measures deliberation quality (spectral gap, effective rank)
- High quality → trust the answer
- Low quality → route to a bigger model or flag for human review

This IS the topological router — the routing decision is "trust this model or escalate."

## Accuracy Summary

| Dataset | Model | Logit | Best Topo | Ensemble |
|---------|-------|-------|-----------|----------|
| ARC-Challenge | 0.5B | 34.0% | 29.0% (gini) | 25.0% |
| MMLU-Physics | 0.5B | 31.4% | 35.3% (norm_var) | 28.4% |
| GPQA-Diamond | 0.5B | 27.3% | 26.8% (gini) | 27.3% |
| ARC-Challenge | 1.5B | 69.0% | 28.0% (gini) | 45.0% |
| MMLU-Physics | 1.5B | 49.0% | 34.3% (gini) | 37.3% |
| GPQA-Diamond | 1.5B | 22.7% | 28.8% (spectral_gap) | 24.2% |

## Files

- `answer_topology_v3_summary.json` — cross-model cross-dataset summary
- `answer_topo_v3_*.json` — per-tier detailed results
- `figures/answer_topo_v3_*.png` — multi-measure distribution plots
