#!/usr/bin/env python3
"""v8 Universal Benchmark — The receipts.

Four dimensions of measurement across 1,500 unseen queries:
  1. Recall@50:  Does FAISS put the truth in the haystack?
  2. Sheaf Top-1: Does the trained sheaf rank truth as #1?
  3. Δ(N-P):     Spectral gap between truth and best impostor
  4. Latency:    End-to-end ms per query

Test sets:
  NQ:    1,000 unseen answerable queries (indices 5000+ of qa_pairs)
  GSM8K: 500 test-split queries (sheaf discrimination only)

Outputs:
  results/v8_benchmark/benchmark_results.json
  results/v8_benchmark/v8_benchmark_report.md
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import faiss

warnings.filterwarnings("ignore")

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
V8_DIR = Path(__file__).parent / "results" / "v8_faiss_prep"
CHECKPOINT_DIR = Path(__file__).parent / "results" / "v7_checkpoints"
OUTPUT_DIR = Path(__file__).parent / "results" / "v8_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from v6_topological_trainer import TextFeatureMap, DifferentiableSheafLaplacian


# ══════════════════════════════════════════════════════════════════════════════
# Engine setup
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkEngine:
    """Loads all components once for fast batch evaluation."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading benchmark engine...")

        # Encoder
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # FAISS index (128-dim, pre-transformed vectors)
        self.index = faiss.read_index(str(V8_DIR / "v8_topological_ivf.faiss"))
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 64

        # Projection for query transform (linear approx for FAISS search)
        self.W = np.load(V8_DIR / "v7_projection_weight.npy")   # (384, 128)
        self.b = np.load(V8_DIR / "v7_projection_bias.npy")     # (128,)

        # Trained sheaf (full PyTorch path for evaluation)
        self.feat_map = TextFeatureMap(out_dim=128, vocab_size=1, backbone_dim=384).to(self.device)
        self.sheaf_lap = DifferentiableSheafLaplacian(d_in=128, stalk_dim=8, k=4).to(self.device)

        ckpt = torch.load(CHECKPOINT_DIR / "v7_trainer_final.pt",
                          map_location=self.device, weights_only=False)
        feat_state = {k.replace("feature_map.", ""): v
                      for k, v in ckpt.items() if k.startswith("feature_map.")}
        sheaf_state = {k.replace("sheaf_loss_fn.sheaf.", ""): v
                       for k, v in ckpt.items() if k.startswith("sheaf_loss_fn.sheaf.")}
        self.feat_map.load_state_dict(feat_state, strict=False)
        self.sheaf_lap.load_state_dict(sheaf_state, strict=False)
        self.feat_map.eval()
        self.sheaf_lap.eval()

        # Chunk metadata
        with open(BASE_MAP_DIR / "chunk_texts.json") as f:
            self.chunks = json.load(f)
        with open(BASE_MAP_DIR / "chunk_meta.json") as f:
            self.chunk_meta = json.load(f)
        with open(BASE_MAP_DIR / "articles.json") as f:
            self.article_titles = json.load(f)

        # Article → chunk lookup
        self.title_to_idx = {t: i for i, t in enumerate(self.article_titles)}

        print(f"  {self.index.ntotal:,} vectors in v8 index")
        print(f"  Device: {self.device}")
        print(f"  Ready.")

    def transform_query(self, q_384: np.ndarray) -> np.ndarray:
        """Apply linear projection + normalize for FAISS search."""
        q_128 = q_384 @ self.W + self.b
        faiss.normalize_L2(q_128)
        return q_128

    def embed_sliding(self, text: str, window=5, stride=1) -> np.ndarray:
        words = text.split()
        if len(words) <= window:
            chunks = [text]
        else:
            chunks = [" ".join(words[i:i+window])
                      for i in range(0, len(words) - window + 1, stride)]
        return self.encoder.encode(chunks, convert_to_numpy=True,
                                   show_progress_bar=False)

    def sheaf_gap(self, cloud_384: np.ndarray) -> float:
        """Run full PyTorch sheaf (with GELU) on a point cloud."""
        with torch.no_grad():
            cloud_t = torch.tensor(cloud_384, dtype=torch.float32, device=self.device)
            projected = self.feat_map(cloud_t)
            eigs = self.sheaf_lap(projected)
            nonzero = eigs[eigs > 1e-6]
            return float(nonzero[0].item()) if len(nonzero) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# NQ Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_nq(engine: BenchmarkEngine, n_test: int = 1000, k_faiss: int = 50, k_sheaf: int = 5):
    """End-to-end NQ retrieval + sheaf ranking benchmark."""
    print(f"\n{'='*70}")
    print(f"  NQ BENCHMARK — {n_test} unseen queries")
    print(f"{'='*70}")

    with open(BASE_MAP_DIR / "qa_pairs.json") as f:
        all_qa = json.load(f)

    answerable = [q for q in all_qa if q["has_answer"]]
    # Use indices 5000+ (unseen by training — hard_negatives used first 5000)
    unseen = answerable[5000:]
    rng = np.random.default_rng(99)
    test_indices = rng.choice(len(unseen), min(n_test, len(unseen)), replace=False)
    test_qa = [unseen[i] for i in test_indices]

    print(f"  {len(test_qa)} test queries (from index 5000+, unseen by training)")

    results = []
    recall_50 = 0
    sheaf_top1 = 0
    sheaf_top1_when_in_haystack = 0
    in_haystack_count = 0
    deltas = []
    latencies = []
    saves = []  # cases where sheaf rescued a low cosine rank

    for i, qa in enumerate(test_qa):
        t0 = time.time()
        question = qa["question"]
        answer = qa["answer"]
        expected_title = qa["title"]
        expected_art = engine.title_to_idx.get(expected_title, -1)

        # ── FAISS search ──────────────────────────────────────────────
        q_384 = engine.encoder.encode([question], convert_to_numpy=True)
        q_128 = engine.transform_query(q_384)
        scores, indices = engine.index.search(q_128, k_faiss)
        scores, indices = scores[0], indices[0]

        # Check recall@50
        top50_arts = set()
        answer_in_chunk = False
        for idx in indices:
            if idx < 0: continue
            art = engine.chunk_meta[int(idx)]["article_idx"]
            top50_arts.add(art)
            if answer.lower() in engine.chunks[int(idx)].lower():
                answer_in_chunk = True

        in_haystack = expected_art in top50_arts
        if in_haystack:
            recall_50 += 1
            in_haystack_count += 1

        # ── Sheaf evaluation on top-k ─────────────────────────────────
        q_cloud = engine.embed_sliding(question, window=5, stride=3)
        sheaf_candidates = []

        for rank in range(min(k_sheaf, len(indices))):
            idx = int(indices[rank])
            if idx < 0: continue

            chunk_text = engine.chunks[idx]
            art = engine.chunk_meta[idx]["article_idx"]
            title = engine.article_titles[art]
            is_correct = (art == expected_art)

            # Sheaf: combine query + candidate clouds
            a_cloud = engine.embed_sliding(chunk_text, window=5, stride=3)
            full_cloud = np.vstack([q_cloud, a_cloud])
            gap = engine.sheaf_gap(full_cloud)

            sheaf_candidates.append({
                "cosine_rank": rank,
                "lambda1": gap,
                "title": title,
                "is_correct": is_correct,
                "has_answer": answer.lower() in chunk_text.lower() if answer else False,
            })

        sheaf_candidates.sort(key=lambda c: c["lambda1"])

        # Metrics
        sheaf_rank_of_truth = None
        best_correct_gap = None
        best_incorrect_gap = None

        for j, c in enumerate(sheaf_candidates):
            if c["is_correct"] and sheaf_rank_of_truth is None:
                sheaf_rank_of_truth = j
                best_correct_gap = c["lambda1"]
            if not c["is_correct"] and best_incorrect_gap is None:
                best_incorrect_gap = c["lambda1"]

        if sheaf_rank_of_truth == 0:
            sheaf_top1 += 1
            if in_haystack:
                sheaf_top1_when_in_haystack += 1

        # Δ(N-P): gap between truth and best impostor
        if best_correct_gap is not None and best_incorrect_gap is not None:
            delta = best_incorrect_gap - best_correct_gap
            deltas.append(delta)

        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        # Track sheaf "saves"
        if sheaf_rank_of_truth == 0 and sheaf_candidates:
            orig_cosine_rank = None
            for c in sheaf_candidates:
                if c["is_correct"]:
                    orig_cosine_rank = c["cosine_rank"]
                    break
            if orig_cosine_rank is not None and orig_cosine_rank > 0:
                saves.append({
                    "question": question[:80],
                    "sheaf_promoted_from": orig_cosine_rank,
                    "lambda1_truth": best_correct_gap,
                    "lambda1_impostor": best_incorrect_gap,
                })

        result = {
            "question": question[:100],
            "answer": answer[:60],
            "expected_title": expected_title[:50],
            "in_haystack": in_haystack,
            "answer_in_chunk": answer_in_chunk,
            "sheaf_rank_of_truth": sheaf_rank_of_truth,
            "best_correct_gap": best_correct_gap,
            "best_incorrect_gap": best_incorrect_gap,
            "delta": delta if best_correct_gap is not None and best_incorrect_gap is not None else None,
            "latency_ms": latency,
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test_qa)}  "
                  f"R@50={recall_50/(i+1)*100:.0f}%  "
                  f"Sheaf@1={sheaf_top1/(i+1)*100:.0f}%  "
                  f"Δ={np.mean(deltas) if deltas else 0:+.4f}  "
                  f"{np.mean(latencies):.0f}ms/q")

    n = len(test_qa)
    avg_delta = float(np.mean(deltas)) if deltas else 0.0
    avg_latency = float(np.mean(latencies))

    saves.sort(key=lambda s: s["sheaf_promoted_from"], reverse=True)

    metrics = {
        "domain": "nq",
        "n_test": n,
        "recall_at_50": recall_50,
        "recall_at_50_pct": round(recall_50 / n * 100, 1),
        "sheaf_top1": sheaf_top1,
        "sheaf_top1_pct": round(sheaf_top1 / n * 100, 1),
        "sheaf_top1_when_in_haystack": sheaf_top1_when_in_haystack,
        "sheaf_top1_given_haystack_pct": round(
            sheaf_top1_when_in_haystack / max(in_haystack_count, 1) * 100, 1),
        "mean_delta": round(avg_delta, 6),
        "median_delta": round(float(np.median(deltas)) if deltas else 0, 6),
        "delta_positive_pct": round(
            sum(1 for d in deltas if d > 0) / max(len(deltas), 1) * 100, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "top_saves": saves[:5],
    }

    print(f"\n  NQ Results:")
    print(f"    Recall@50:             {metrics['recall_at_50_pct']}%")
    print(f"    Sheaf Top-1:           {metrics['sheaf_top1_pct']}%")
    print(f"    Sheaf Top-1 | haystack: {metrics['sheaf_top1_given_haystack_pct']}%")
    print(f"    Mean Δ(N-P):           {metrics['mean_delta']:+.6f}")
    print(f"    Δ positive:            {metrics['delta_positive_pct']}%")
    print(f"    Avg latency:           {metrics['avg_latency_ms']:.0f}ms")

    return metrics, results


# ══════════════════════════════════════════════════════════════════════════════
# GSM8K Benchmark (Sheaf discrimination — no FAISS retrieval)
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_gsm8k(engine: BenchmarkEngine, n_test: int = 500):
    """Test sheaf's ability to distinguish correct vs wrong math proofs."""
    from datasets import load_dataset

    print(f"\n{'='*70}")
    print(f"  GSM8K BENCHMARK — {n_test} test queries")
    print(f"{'='*70}")

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = list(ds)
    rng = np.random.default_rng(77)
    test_idx = rng.choice(len(problems), min(n_test, len(problems)), replace=False)

    correct_wins = 0
    deltas = []
    latencies = []

    for i, idx in enumerate(test_idx):
        t0 = time.time()
        prob = problems[idx]
        question = prob["question"]
        answer = prob["answer"]
        steps = answer.split("####")[0].strip() if "####" in answer else answer

        # Pick a wrong solution (from a different problem)
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = rng.integers(len(problems))
        wrong_steps = problems[neg_idx]["answer"].split("####")[0].strip()

        # Embed
        q_cloud = engine.embed_sliding(question, window=5, stride=3)
        pos_cloud = engine.embed_sliding(steps, window=5, stride=3)
        neg_cloud = engine.embed_sliding(wrong_steps, window=5, stride=3)

        # Sheaf evaluation
        pos_full = np.vstack([q_cloud, pos_cloud])
        neg_full = np.vstack([q_cloud, neg_cloud])

        gap_pos = engine.sheaf_gap(pos_full)
        gap_neg = engine.sheaf_gap(neg_full)

        correct = gap_pos < gap_neg
        if correct:
            correct_wins += 1

        delta = gap_neg - gap_pos
        deltas.append(delta)
        latencies.append((time.time() - t0) * 1000)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_test}  "
                  f"Accuracy={correct_wins/(i+1)*100:.0f}%  "
                  f"Δ={np.mean(deltas):+.4f}")

    n = len(test_idx)
    metrics = {
        "domain": "gsm8k",
        "n_test": n,
        "sheaf_accuracy": correct_wins,
        "sheaf_accuracy_pct": round(correct_wins / n * 100, 1),
        "mean_delta": round(float(np.mean(deltas)), 6),
        "median_delta": round(float(np.median(deltas)), 6),
        "delta_positive_pct": round(sum(1 for d in deltas if d > 0) / n * 100, 1),
        "avg_latency_ms": round(float(np.mean(latencies)), 1),
    }

    print(f"\n  GSM8K Results:")
    print(f"    Sheaf accuracy:  {metrics['sheaf_accuracy_pct']}%")
    print(f"    Mean Δ(N-P):     {metrics['mean_delta']:+.6f}")
    print(f"    Δ positive:      {metrics['delta_positive_pct']}%")
    print(f"    Avg latency:     {metrics['avg_latency_ms']:.0f}ms")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(nq_metrics, gsm8k_metrics, elapsed):
    """Generate markdown benchmark report."""
    report = f"""# v8 Universal Slice — Benchmark Report

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Total benchmark time: {elapsed:.0f}s

## System Configuration
- Index: 2,338,208 chunks, FAISS IVFFlat in 128-dim warped space
- Projection: v7 trained TextFeatureMap (384->128, cross-domain NQ+GSM8K)
- Sheaf: DifferentiableSheafLaplacian (stalk_dim=8, k=4, trained)
- Parameters: 132,258 (feature map + sheaf + router)

## Natural Questions (Factual Retrieval)

| Metric | Value |
|--------|-------|
| Test queries | {nq_metrics['n_test']} |
| Recall@50 (FAISS haystack) | {nq_metrics['recall_at_50_pct']}% |
| Sheaf Top-1 accuracy | {nq_metrics['sheaf_top1_pct']}% |
| Sheaf Top-1 given haystack | {nq_metrics['sheaf_top1_given_haystack_pct']}% |
| Mean spectral gap delta (N-P) | {nq_metrics['mean_delta']:+.6f} |
| Median spectral gap delta | {nq_metrics['median_delta']:+.6f} |
| Delta positive rate | {nq_metrics['delta_positive_pct']}% |
| Avg latency | {nq_metrics['avg_latency_ms']:.0f}ms |

### Top Sheaf Saves (FAISS ranked low, sheaf promoted to #1)
"""
    for save in nq_metrics.get("top_saves", [])[:5]:
        report += f"- **Promoted from cosine rank {save['sheaf_promoted_from']}**: "
        report += f"lambda1_truth={save['lambda1_truth']:.6f}, "
        report += f"lambda1_impostor={save['lambda1_impostor']:.6f}\n"
        report += f"  Q: {save['question']}\n\n"

    report += f"""
## GSM8K (Mathematical Reasoning)

| Metric | Value |
|--------|-------|
| Test queries | {gsm8k_metrics['n_test']} |
| Sheaf discrimination accuracy | {gsm8k_metrics['sheaf_accuracy_pct']}% |
| Mean spectral gap delta (N-P) | {gsm8k_metrics['mean_delta']:+.6f} |
| Median spectral gap delta | {gsm8k_metrics['median_delta']:+.6f} |
| Delta positive rate | {gsm8k_metrics['delta_positive_pct']}% |
| Avg latency | {gsm8k_metrics['avg_latency_ms']:.0f}ms |

## Interpretation

The spectral gap delta measures the mathematical distance between factual
truth and the most convincing impostor. Positive delta means the trained
sheaf Laplacian correctly identifies truth as more topologically coherent
than fiction. Delta positive rate is the percentage of queries where this
holds.
"""

    with open(OUTPUT_DIR / "v8_benchmark_report.md", "w") as f:
        f.write(report)
    print(f"\n  Report: {OUTPUT_DIR / 'v8_benchmark_report.md'}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(n_nq=1000, n_gsm8k=500):
    start = time.time()

    engine = BenchmarkEngine()

    nq_metrics, nq_results = benchmark_nq(engine, n_test=n_nq)
    gsm8k_metrics = benchmark_gsm8k(engine, n_test=n_gsm8k)

    elapsed = time.time() - start

    # Save full results
    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "nq": nq_metrics,
        "gsm8k": gsm8k_metrics,
    }
    with open(OUTPUT_DIR / "benchmark_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    generate_report(nq_metrics, gsm8k_metrics, elapsed)

    print(f"\n{'='*70}")
    print(f"  v8 UNIVERSAL BENCHMARK COMPLETE — {elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"  NQ:    R@50={nq_metrics['recall_at_50_pct']}%  "
          f"Sheaf@1={nq_metrics['sheaf_top1_pct']}%  "
          f"Δ={nq_metrics['mean_delta']:+.6f}  "
          f"{nq_metrics['avg_latency_ms']:.0f}ms/q")
    print(f"  GSM8K: Accuracy={gsm8k_metrics['sheaf_accuracy_pct']}%  "
          f"Δ={gsm8k_metrics['mean_delta']:+.6f}  "
          f"{gsm8k_metrics['avg_latency_ms']:.0f}ms/q")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nq", type=int, default=1000)
    parser.add_argument("--n_gsm8k", type=int, default=500)
    args = parser.parse_args()
    run_benchmark(args.n_nq, args.n_gsm8k)
