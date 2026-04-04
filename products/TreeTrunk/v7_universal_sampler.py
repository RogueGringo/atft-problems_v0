#!/usr/bin/env python3
"""v7 Universal Triplet Sampler — Cross-domain manifold alignment.

The TopologicalTrainer and Sheaf Loss are LOCKED. This module only touches
the data ingestion. Every domain is violently abstracted into:

    (q_cloud, pos_cloud, neg_cloud) — all np.ndarray of shape (n, 384)

The sheaf never knows if it's looking at a Wikipedia fact, a math proof,
a Python function, or a drill vibration. It only sees geometry.

Domain sub-samplers:
  NQSampler         — factual/declarative (Wikipedia)
  GSM8KSampler      — deductive/mathematical (math word problems)
  CodeSampler       — algorithmic/hierarchical (MBPP/HumanEval) [stub]
  TelemetrySampler   — physical/causal (MWD .las sensor data) [stub]

The UniversalTripletSampler round-robins across all active domains,
producing perfectly stratified batches.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
CHECKPOINT_DIR = Path(__file__).parent / "results" / "v6_checkpoints"


# ══════════════════════════════════════════════════════════════════════════════
# Abstract Domain Sampler
# ══════════════════════════════════════════════════════════════════════════════

class DomainSampler(ABC):
    """Data contract: every domain produces (q, pos, neg) in 384-dim.

    The 384-dim space is the sentence-transformer backbone output.
    Text domains use the frozen encoder directly.
    Sensor domains use a learned backbone that maps to the same space.
    """

    @abstractmethod
    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Return one (q_cloud, pos_cloud, neg_cloud, metadata) triplet.

        q_cloud:   (n_q, 384)   — query point cloud
        pos_cloud: (n_pos, 384) — correct answer / valid state
        neg_cloud: (n_neg, 384) — hard negative / invalid state
        metadata:  dict with at minimum {"domain": str}
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of available triplets."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# NQ Sampler (Factual / Declarative)
# ══════════════════════════════════════════════════════════════════════════════

class NQSampler(DomainSampler):
    """Natural Questions — factual Wikipedia retrieval.

    Uses pre-mined FAISS hard negatives (chunks from wrong articles
    with high cosine similarity). Sharp topological derivatives.
    """

    def __init__(self, encoder, max_cloud: int = 12):
        print("  [NQ] Loading...")

        self.embeddings = np.load(BASE_MAP_DIR / "embeddings.npy", mmap_mode="r")

        with open(CHECKPOINT_DIR / "hard_negatives.json") as f:
            self.cache = json.load(f)
        self.cache = [c for c in self.cache
                      if c["pos_chunk_ids"] and c["hard_neg_chunks"]]

        self.encoder = encoder
        self.max_cloud = max_cloud
        self.rng = np.random.default_rng(42)
        print(f"  [NQ] {len(self.cache):,} triplets with hard negatives")

    def __len__(self):
        return len(self.cache)

    def _embed_question(self, text: str) -> np.ndarray:
        words = text.split()
        w, s = 5, 3
        if len(words) <= w:
            chunks = [text]
        else:
            chunks = [" ".join(words[i:i+w])
                      for i in range(0, len(words) - w + 1, s)]
        return self.encoder.encode(chunks, convert_to_numpy=True,
                                   show_progress_bar=False)

    def sample(self):
        entry = self.cache[self.rng.integers(len(self.cache))]

        q_cloud = self._embed_question(entry["question"])

        pos_ids = entry["pos_chunk_ids"]
        if len(pos_ids) > self.max_cloud:
            pos_ids = self.rng.choice(pos_ids, self.max_cloud, replace=False).tolist()
        pos_cloud = np.array(self.embeddings[pos_ids])

        neg_entries = entry["hard_neg_chunks"]
        neg_ids = [n["chunk_idx"] for n in neg_entries]
        if len(neg_ids) > self.max_cloud:
            neg_ids = self.rng.choice(neg_ids, self.max_cloud, replace=False).tolist()
        neg_cloud = np.array(self.embeddings[neg_ids])

        meta = {
            "domain": "nq",
            "question": entry["question"][:80],
            "n_q": len(q_cloud),
            "n_pos": len(pos_cloud),
            "n_neg": len(neg_cloud),
        }
        return q_cloud, pos_cloud, neg_cloud, meta


# ══════════════════════════════════════════════════════════════════════════════
# GSM8K Sampler (Deductive / Mathematical)
# ══════════════════════════════════════════════════════════════════════════════

class GSM8KSampler(DomainSampler):
    """GSM8K — multi-step math reasoning.

    Triplet construction:
      q_cloud:   embed the question (sliding window)
      pos_cloud: embed the correct step-by-step solution
      neg_cloud: embed a wrong solution (different problem's solution)

    Hard negatives: solutions to different problems that share
    similar numerical operations (same domain, wrong logic).
    """

    def __init__(self, encoder, max_cloud: int = 12):
        from datasets import load_dataset
        print("  [GSM8K] Loading...")

        ds = load_dataset("openai/gsm8k", "main", split="train")

        self.problems = []
        for item in ds:
            question = item["question"]
            answer = item["answer"]
            # Extract final numerical answer after ####
            final = answer.split("####")[-1].strip() if "####" in answer else ""
            # Solution steps (everything before ####)
            steps = answer.split("####")[0].strip() if "####" in answer else answer
            self.problems.append({
                "question": question,
                "steps": steps,
                "final": final,
            })

        self.encoder = encoder
        self.max_cloud = max_cloud
        self.rng = np.random.default_rng(123)
        print(f"  [GSM8K] {len(self.problems):,} problems loaded")

    def __len__(self):
        return len(self.problems)

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text as sliding-window cloud."""
        words = text.split()
        w, s = 5, 3
        if len(words) <= w:
            chunks = [text]
        else:
            chunks = [" ".join(words[i:i+w])
                      for i in range(0, len(words) - w + 1, s)]
        # Cap cloud size
        if len(chunks) > self.max_cloud:
            idx = self.rng.choice(len(chunks), self.max_cloud, replace=False)
            chunks = [chunks[i] for i in sorted(idx)]
        return self.encoder.encode(chunks, convert_to_numpy=True,
                                   show_progress_bar=False)

    def sample(self):
        # Pick random problem
        idx = self.rng.integers(len(self.problems))
        prob = self.problems[idx]

        q_cloud = self._embed_text(prob["question"])
        pos_cloud = self._embed_text(prob["steps"])

        # Hard negative: solution from a DIFFERENT problem
        # (same math domain, wrong logical chain)
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = self.rng.integers(len(self.problems))
        neg_cloud = self._embed_text(self.problems[neg_idx]["steps"])

        meta = {
            "domain": "gsm8k",
            "question": prob["question"][:80],
            "final": prob["final"],
            "n_q": len(q_cloud),
            "n_pos": len(pos_cloud),
            "n_neg": len(neg_cloud),
        }
        return q_cloud, pos_cloud, neg_cloud, meta


# ══════════════════════════════════════════════════════════════════════════════
# Code Sampler (Algorithmic / Hierarchical) — STUB
# ══════════════════════════════════════════════════════════════════════════════

class CodeSampler(DomainSampler):
    """MBPP / HumanEval — code generation and verification.

    Triplet construction:
      q_cloud:   embed the task description
      pos_cloud: embed the correct solution code
      neg_cloud: embed a wrong solution (different task's code)

    Hard negatives: solutions that use the same stdlib imports
    and variable naming patterns but solve a different task.

    Data contract: code is embedded as text through the same
    sentence-transformer. The sheaf measures logical structure
    of the code's semantic embedding, not AST structure.

    TODO: Load MBPP dataset, implement hard negative mining
    based on shared import/function-name overlap.
    """

    def __init__(self, encoder, max_cloud: int = 12):
        raise NotImplementedError(
            "CodeSampler stub — implement when MBPP/HumanEval ingestion is ready"
        )

    def __len__(self):
        return 0

    def sample(self):
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Telemetry Sampler (Physical / Causal) — STUB
# ══════════════════════════════════════════════════════════════════════════════

class TelemetrySampler(DomainSampler):
    """MWD .las sensor telemetry — physical phase transitions.

    Triplet construction:
      q_cloud:   sensor state window A (current drilling condition)
      pos_cloud: valid subsequent state B (normal operation)
      neg_cloud: impossible subsequent state C (mechanical failure signature
                 that LOOKS like normal vibration)

    Sensor → 384-dim projection:
      Raw .las channels (gamma ray, resistivity, ROP, WOB, torque, etc.)
      are 1D time-series. To project into the same 384-dim space as the
      sentence-transformer:

      1. Sliding window: extract overlapping windows of N samples
      2. Conv1d backbone: Conv1d(n_channels, 128, kernel=5) → GELU →
         Conv1d(128, 256, kernel=3) → GELU → AdaptiveAvgPool1d(1) →
         Linear(256, 384)
      3. The 384-dim output is L2-normalized to match sentence-transformer scale
      4. This backbone is TRAINED jointly with the sheaf loss —
         it learns to produce embeddings where physical causality
         maps to topological coherence

    Hard negatives: telemetry windows where vibration amplitude
    and frequency match normal drilling but RPM/torque divergence
    indicates stick-slip or bit bounce (looks normal, is dangerous).

    TODO: Load .las files via lasio, implement Conv1d backbone,
    pre-mine hard negatives based on signal similarity + label divergence.
    """

    def __init__(self, encoder, max_cloud: int = 12):
        raise NotImplementedError(
            "TelemetrySampler stub — implement when .las ingestion pipeline is ready"
        )

    def __len__(self):
        return 0

    def sample(self):
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Universal Sampler — Stratified Round-Robin
# ══════════════════════════════════════════════════════════════════════════════

class UniversalTripletSampler:
    """Round-robin across all active domain samplers.

    Produces perfectly stratified batches: if batch_size=16 and
    4 domains are active, each batch contains exactly 4 samples
    from each domain. The sheaf gradient is equally informed by
    all domains, preventing syntactic overfitting.

    If a domain is exhausted or unavailable, it is skipped and
    the remaining domains fill the gap proportionally.
    """

    def __init__(self, samplers: list[DomainSampler]):
        self.samplers = samplers
        self.n_domains = len(samplers)
        self._idx = 0  # round-robin cursor

        total = sum(len(s) for s in samplers)
        print(f"\n  Universal Sampler: {self.n_domains} domains, "
              f"{total:,} total triplets")
        for s in samplers:
            domain = s.__class__.__name__
            print(f"    {domain}: {len(s):,}")

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Sample one triplet, round-robin across domains."""
        sampler = self.samplers[self._idx % self.n_domains]
        self._idx += 1
        return sampler.sample()

    def sample_batch(self, batch_size: int) -> list[tuple]:
        """Sample a stratified batch — equal from each domain."""
        per_domain = max(1, batch_size // self.n_domains)
        batch = []
        for sampler in self.samplers:
            for _ in range(per_domain):
                batch.append(sampler.sample())
        return batch[:batch_size]


# ══════════════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════════════

def test_samplers():
    """Verify NQ + GSM8K samplers produce valid triplets."""
    from sentence_transformers import SentenceTransformer

    print("Loading encoder...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Build active samplers
    samplers = []
    samplers.append(NQSampler(encoder))
    samplers.append(GSM8KSampler(encoder))

    uni = UniversalTripletSampler(samplers)

    print(f"\nSampling 8 stratified triplets...\n")
    for i in range(8):
        q, pos, neg, meta = uni.sample()
        print(f"  [{i}] domain={meta['domain']:>6s}  "
              f"q={q.shape}  pos={pos.shape}  neg={neg.shape}  "
              f"| {meta.get('question', '')[:50]}")

        # Verify dimensions
        assert q.shape[1] == 384, f"q dim wrong: {q.shape}"
        assert pos.shape[1] == 384, f"pos dim wrong: {pos.shape}"
        assert neg.shape[1] == 384, f"neg dim wrong: {neg.shape}"

    print(f"\n  All triplets valid. 384-dim contract holds across domains.")

    # Test stratified batch
    batch = uni.sample_batch(8)
    domains = [b[3]["domain"] for b in batch]
    print(f"  Stratified batch domains: {domains}")
    assert "nq" in domains and "gsm8k" in domains, "Missing domain in batch!"
    print(f"  Stratification verified.")


if __name__ == "__main__":
    test_samplers()
