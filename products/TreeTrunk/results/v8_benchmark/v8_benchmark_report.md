# v8 Universal Slice — Benchmark Report

Generated: 2026-04-04 11:23:11
Total benchmark time: 115s

## System Configuration
- Index: 2,338,208 chunks, FAISS IVFFlat in 128-dim warped space
- Projection: v7 trained TextFeatureMap (384->128, cross-domain NQ+GSM8K)
- Sheaf: DifferentiableSheafLaplacian (stalk_dim=8, k=4, trained)
- Parameters: 132,258 (feature map + sheaf + router)

## Natural Questions (Factual Retrieval)

| Metric | Value |
|--------|-------|
| Test queries | 1000 |
| Recall@50 (FAISS haystack) | 31.7% |
| Sheaf Top-1 accuracy | 4.2% |
| Sheaf Top-1 given haystack | 13.2% |
| Mean spectral gap delta (N-P) | -0.002033 |
| Median spectral gap delta | -0.000881 |
| Delta positive rate | 25.6% |
| Avg latency | 81ms |

### Top Sheaf Saves (FAISS ranked low, sheaf promoted to #1)
- **Promoted from cosine rank 4**: lambda1_truth=0.001240, lambda1_impostor=0.001339
  Q: where were the disciples going when they saw jesus walking on water

- **Promoted from cosine rank 4**: lambda1_truth=0.000600, lambda1_impostor=0.000810
  Q: what is the role of the president's chief of staff

- **Promoted from cosine rank 4**: lambda1_truth=0.000391, lambda1_impostor=0.000797
  Q: what is the current version of ie 11

- **Promoted from cosine rank 4**: lambda1_truth=0.000107, lambda1_impostor=0.000135
  Q: why was virginia capital moved from williamsburg to richmond

- **Promoted from cosine rank 4**: lambda1_truth=0.000578, lambda1_impostor=0.000621
  Q: where is the sa node located in the heart


## GSM8K (Mathematical Reasoning)

| Metric | Value |
|--------|-------|
| Test queries | 500 |
| Sheaf discrimination accuracy | 57.0% |
| Mean spectral gap delta (N-P) | +0.002628 |
| Median spectral gap delta | +0.003523 |
| Delta positive rate | 57.0% |
| Avg latency | 52ms |

## Interpretation

The spectral gap delta measures the mathematical distance between factual
truth and the most convincing impostor. Positive delta means the trained
sheaf Laplacian correctly identifies truth as more topologically coherent
than fiction. Delta positive rate is the percentage of queries where this
holds.
