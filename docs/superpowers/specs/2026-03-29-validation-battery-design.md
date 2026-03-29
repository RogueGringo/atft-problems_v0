# Validation Battery: Stress-Test the Tokenization Thesis

**Goal:** Determine if "residual is richer than baseline" is a real phenomenon or a methodology artifact. Four tests, binary pass/fail, on Qwen2.5-1.5B-Instruct. If all pass, expand to more models.

**Core claim under test:** When a language model's hidden states are decomposed into a prime subspace (from 59 semantic primes) and a residual subspace (adaptively discovered), the residual has higher topological hierarchy (Gini) than the prime subspace. This was observed in 6/6 models with ratios of 1.3-4.7x.

**Threshold:** 3 existential controls must pass + bootstrap CIs must exclude zero.

---

## Test 1: Random Null

**Question:** Does random data show the same "residual richer than prime" pattern?

**Protocol:**
1. Load Qwen2.5-1.5B-Instruct's prime basis (58 vectors, 1536-dim) from existing results.
2. Generate a random Gaussian matrix of the same shape as the model's hidden states extracted during the sweep (same n_tokens × hidden_dim).
3. Run the full pipeline on this random matrix: project onto prime subspace, compute residual, measure SVD Gini of both.
4. Repeat 10 times with different random seeds. Record mean and std of (residual_gini - prime_gini).

**Pass criterion:** Mean(residual_gini - prime_gini) for random data is ≤ 0 or statistically indistinguishable from 0 (p > 0.05 by t-test against zero).

**Fail criterion:** Random data consistently shows residual_gini > prime_gini → the finding is an artifact of the projection geometry.

---

## Test 2: Dimensional Parity

**Question:** Is the Gini difference just because the residual subspace has more dimensions?

**Protocol:**
1. Load Qwen2.5-1.5B adaptive basis (129 vectors). The first 58 are prime-aligned, the rest are residual.
2. Take exactly 58 residual vectors (vectors 59-116).
3. Load model, extract hidden states from the same 50 prompts used in the sweep.
4. Project hidden states onto the 58 prime vectors → compute SVD Gini.
5. Project hidden states onto the 58 residual vectors → compute SVD Gini.
6. Both projections are into 58-dimensional subspaces. The only difference is WHICH 58 directions.

**Pass criterion:** Residual-58 Gini > Prime-58 Gini (the hierarchy difference persists at equal dimensionality).

**Fail criterion:** Residual-58 Gini ≈ Prime-58 Gini → the original finding was a dimensionality artifact.

---

## Test 3: Permutation

**Question:** Does the signal come from sequential structure or just from activation statistics?

**Protocol:**
1. Load model, extract hidden states at target layer for 50 prompts → (n_tokens, hidden_dim).
2. Randomly permute the token dimension (shuffle rows). This preserves each token's activation vector but destroys their sequential order.
3. Run the full Gini comparison on the shuffled matrix.
4. Compare: (residual_gini - prime_gini) on real data vs shuffled data.
5. Repeat shuffling 10 times.

**Pass criterion:** Shuffled (residual - prime) gap is significantly smaller than real gap OR the gap's sign reversal rate > 30% on shuffled data.

**Fail criterion:** Shuffled data shows the same gap as real data → the signal is in the marginal token statistics, not in the sequential structure. (Note: this is a partial fail — the finding is real but the mechanism is different than assumed.)

---

## Test 4: Bootstrap Confidence Intervals

**Question:** Is the observed Gini difference statistically reliable?

**Protocol:**
1. Load model, extract hidden states for all 100 prompts (5 modes × 20).
2. For each of 200 bootstrap iterations:
   a. Resample 100 prompts with replacement.
   b. Extract hidden states for the resampled set.
   c. Project onto prime subspace and residual subspace.
   d. Compute (residual_gini - prime_gini).
3. Compute 95% confidence interval of the Gini difference.

**Optimization:** To avoid 200 model forward passes per bootstrap, pre-compute hidden states for all 100 prompts once. Then resample by selecting pre-computed hidden state rows.

**Pass criterion:** 95% CI of (residual_gini - prime_gini) does not include zero. Both bounds are positive.

**Fail criterion:** CI includes zero → the difference could be noise.

---

## Implementation

**Single file:** `products/topological-router/validation_battery.py`

**Inputs (already exist):**
- `results/basis_discovery/prime_basis_Qwen_Qwen2.5-1.5B-Instruct.pt`
- `results/basis_discovery/adaptive_basis_Qwen_Qwen2.5-1.5B-Instruct.pt`
- `prompts/loader.py` (100 prompts)

**Outputs:**
- `results/validation/battery_results.json` — all metrics, pass/fail, CIs
- Console: formatted pass/fail table + verdict

**Compute:** Pre-compute hidden states once (~60s), then Tests 1-3 are matrix operations (~seconds each). Test 4 resamples pre-computed data (minutes). Total: ~5-10 minutes.

**Model loading:** Load once, extract all hidden states, unload. Tests operate on cached numpy arrays.

---

## Verdicts

| Result | Interpretation |
|--------|---------------|
| 4/4 PASS | CONFIRMED — the finding is real, expand to more models |
| 3/4 PASS (Test 3 fails) | REAL BUT DISTRIBUTIONAL — the hierarchy is in activation statistics, not sequence structure. Still meaningful but mechanism differs. |
| Test 1 FAIL | KILLED — methodology artifact. The projection geometry creates fake signal. |
| Test 2 FAIL | KILLED — dimensionality artifact. More dimensions = higher Gini by construction. |
| Test 4 FAIL | UNRELIABLE — signal may exist but is too noisy to confirm at n=100 prompts. |

---

## Success Criteria

The validation battery passes if:
1. Test 1 (Random Null): PASS
2. Test 2 (Dimensional Parity): PASS
3. Test 3 (Permutation): PASS or PARTIAL (distributional signal is still real)
4. Test 4 (Bootstrap): 95% CI excludes zero
