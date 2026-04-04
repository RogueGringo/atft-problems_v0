#!/usr/bin/env python3
"""v9 Offline Graph Compiler — Pre-compile dependency graphs for training.

spaCy and AST parsing are CPU-bound. Running them inside the PyTorch
training loop would bottleneck the GPU. This script pre-compiles ALL
training data into static graph structures that the GPU can consume
at full speed.

Pipeline:
  1. Load NQ hard negatives (5K queries × 20 pos chunks × 20 neg chunks)
  2. Load GSM8K train (7.5K problems with step-by-step solutions)
  3. Parse everything through domain-specific graphers:
     - TextCausalGrapher (spaCy) for NQ text
     - MathCausalGrapher for GSM8K equations
  4. Build global EdgeRegistry (edge_type_name ↔ integer_id)
  5. Save compiled graphs as torch tensors for fast GPU loading

Output:
  results/v9_compiled_graphs/
  ├── nq_graphs.pt            — compiled NQ triplet graphs
  ├── gsm8k_graphs.pt         — compiled GSM8K triplet graphs
  ├── v9_edge_registry.json   — global edge type mapping
  └── compile_log.json        — statistics
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from dataclasses import dataclass, field

import torch

BASE_MAP_DIR = Path(__file__).parent / "results" / "v5_nq_base_map"
HARD_NEG_PATH = Path(__file__).parent / "results" / "v6_checkpoints" / "hard_negatives.json"
OUTPUT_DIR = Path(__file__).parent / "results" / "v9_compiled_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Edge Registry — global bidirectional mapping
# ══════════════════════════════════════════════════════════════════════════════

class EdgeRegistry:
    """Bidirectional mapping: edge_type_name ↔ integer_id.

    Auto-assigns new IDs as new edge types are encountered.
    Shared across all domains — the sheaf's nn.ModuleDict uses these IDs.
    """

    def __init__(self):
        self.name_to_id: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {}
        self._next_id = 0

    def get_id(self, name: str) -> int:
        if name not in self.name_to_id:
            self.name_to_id[name] = self._next_id
            self.id_to_name[self._next_id] = name
            self._next_id += 1
        return self.name_to_id[name]

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump({
                "name_to_id": self.name_to_id,
                "id_to_name": {str(k): v for k, v in self.id_to_name.items()},
                "total_types": self._next_id,
            }, f, indent=2)

    def __len__(self):
        return self._next_id

    def __repr__(self):
        return f"EdgeRegistry({self._next_id} types)"


# ══════════════════════════════════════════════════════════════════════════════
# Compiled Graph — the output format
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompiledGraph:
    """Static graph structure ready for GPU consumption."""
    node_texts: list[str]
    edge_index: list[tuple[int, int]]   # (src, dst) pairs
    edge_type_ids: list[int]            # integer type per edge

    def to_tensors(self) -> dict:
        """Convert to torch tensors for saving."""
        n_edges = len(self.edge_index)
        if n_edges > 0:
            ei = torch.tensor(self.edge_index, dtype=torch.long)  # (n_edges, 2)
            et = torch.tensor(self.edge_type_ids, dtype=torch.long)  # (n_edges,)
        else:
            ei = torch.zeros(0, 2, dtype=torch.long)
            et = torch.zeros(0, dtype=torch.long)
        return {
            "node_texts": self.node_texts,
            "edge_index": ei,
            "edge_type_ids": et,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Text Causal Grapher — spaCy batch pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TextCausalGrapher:
    """spaCy dependency parser → CompiledGraph with integer edge types."""

    def __init__(self, registry: EdgeRegistry, model: str = "en_core_web_sm"):
        import spacy
        self.nlp = spacy.load(model, disable=["ner", "lemmatizer"])
        self.registry = registry

    def parse(self, text: str) -> CompiledGraph:
        doc = self.nlp(text)
        node_texts = [token.text for token in doc]
        edges = []
        edge_types = []

        for token in doc:
            if token.head != token:
                edges.append((token.head.i, token.i))
                edge_types.append(self.registry.get_id(token.dep_))

        return CompiledGraph(node_texts, edges, edge_types)

    def parse_batch(self, texts: list[str], batch_size: int = 256) -> list[CompiledGraph]:
        """Batch parse using spaCy's pipe() for efficiency."""
        results = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            node_texts = [token.text for token in doc]
            edges = []
            edge_types = []
            for token in doc:
                if token.head != token:
                    edges.append((token.head.i, token.i))
                    edge_types.append(self.registry.get_id(token.dep_))
            results.append(CompiledGraph(node_texts, edges, edge_types))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Math Causal Grapher — GSM8K equation + step parser
# ══════════════════════════════════════════════════════════════════════════════

class MathCausalGrapher:
    """GSM8K step-by-step solution → operator tree graph.

    Parses:
      - Sequential steps (step_i → step_{i+1})
      - Equations: detects "A op B = C" patterns
      - Operator types: add, sub, mul, div, equals
      - Narrative connectives between steps
    """

    # Regex for simple equations: number op number = number
    EQ_PATTERN = re.compile(
        r'([\d,.]+)\s*([+\-×*/]|<<)\s*([\d,.]+)\s*=\s*([\d,.]+)'
    )

    OP_MAP = {
        "+": "op_add", "-": "op_sub", "×": "op_mul", "*": "op_mul",
        "/": "op_div", "<<": "op_compute",
    }

    def __init__(self, registry: EdgeRegistry):
        self.registry = registry

    def parse(self, text: str) -> CompiledGraph:
        # Split into steps (sentences or lines)
        steps = [s.strip() for s in re.split(r'[.\n]', text) if s.strip()]
        if not steps:
            return CompiledGraph(["[empty]"], [], [])

        node_texts = list(steps)
        edges = []
        edge_types = []

        # Sequential step connections
        for i in range(len(steps) - 1):
            edges.append((i, i + 1))
            edge_types.append(self.registry.get_id("step_next"))

        # Detect equations within steps
        for i, step in enumerate(steps):
            for match in self.EQ_PATTERN.finditer(step):
                op = match.group(2)
                op_type = self.OP_MAP.get(op, "op_compute")

                # Add operand and result as sub-nodes
                left_val = match.group(1)
                right_val = match.group(3)
                result_val = match.group(4)

                left_idx = len(node_texts)
                node_texts.append(left_val)
                right_idx = len(node_texts)
                node_texts.append(right_val)
                result_idx = len(node_texts)
                node_texts.append(result_val)

                # Operand edges
                edges.append((i, left_idx))
                edge_types.append(self.registry.get_id("operand_left"))
                edges.append((i, right_idx))
                edge_types.append(self.registry.get_id("operand_right"))

                # Operation edge
                edges.append((left_idx, result_idx))
                edge_types.append(self.registry.get_id(op_type))
                edges.append((right_idx, result_idx))
                edge_types.append(self.registry.get_id(op_type))

                # Result links back to step
                edges.append((result_idx, i))
                edge_types.append(self.registry.get_id("computes"))

        return CompiledGraph(node_texts, edges, edge_types)


# ══════════════════════════════════════════════════════════════════════════════
# Batch Compiler
# ══════════════════════════════════════════════════════════════════════════════

def compile_nq(registry: EdgeRegistry, max_entries: int = 5000) -> list[dict]:
    """Compile NQ hard negative triplets into graph structures."""
    print("\n[NQ] Compiling graphs...")

    with open(HARD_NEG_PATH) as f:
        hard_negs = json.load(f)
    with open(BASE_MAP_DIR / "chunk_texts.json") as f:
        all_chunks = json.load(f)

    hard_negs = hard_negs[:max_entries]
    grapher = TextCausalGrapher(registry)

    # Collect all unique texts that need parsing
    texts_to_parse = {}  # text → index
    text_list = []

    for entry in hard_negs:
        q = entry["question"]
        if q not in texts_to_parse:
            texts_to_parse[q] = len(text_list)
            text_list.append(q)

        for cid in entry["pos_chunk_ids"][:5]:  # cap at 5 per entry
            t = all_chunks[cid]
            if t not in texts_to_parse:
                texts_to_parse[t] = len(text_list)
                text_list.append(t)

        for neg in entry["hard_neg_chunks"][:5]:
            t = all_chunks[neg["chunk_idx"]]
            if t not in texts_to_parse:
                texts_to_parse[t] = len(text_list)
                text_list.append(t)

    print(f"  {len(text_list):,} unique texts to parse")

    # Batch parse with spaCy
    t0 = time.time()
    parsed = grapher.parse_batch(text_list, batch_size=512)
    parse_time = time.time() - t0
    print(f"  Parsed in {parse_time:.1f}s ({len(text_list)/parse_time:.0f} texts/s)")

    # Build lookup
    text_to_graph = {text_list[i]: parsed[i] for i in range(len(text_list))}

    # Compile triplets
    compiled = []
    for entry in hard_negs:
        q_graph = text_to_graph[entry["question"]]

        pos_graphs = []
        for cid in entry["pos_chunk_ids"][:5]:
            t = all_chunks[cid]
            pos_graphs.append(text_to_graph[t])

        neg_graphs = []
        for neg in entry["hard_neg_chunks"][:5]:
            t = all_chunks[neg["chunk_idx"]]
            neg_graphs.append(text_to_graph[t])

        if pos_graphs and neg_graphs:
            compiled.append({
                "domain": "nq",
                "question": entry["question"][:100],
                "q_graph": q_graph.to_tensors(),
                "pos_graphs": [g.to_tensors() for g in pos_graphs],
                "neg_graphs": [g.to_tensors() for g in neg_graphs],
            })

    print(f"  {len(compiled):,} triplets compiled")
    return compiled


def compile_gsm8k(registry: EdgeRegistry, max_problems: int = 5000) -> list[dict]:
    """Compile GSM8K problems into graph structures."""
    from datasets import load_dataset
    print("\n[GSM8K] Compiling graphs...")

    ds = load_dataset("openai/gsm8k", "main", split="train")
    problems = list(ds)[:max_problems]

    text_grapher = TextCausalGrapher(registry)
    math_grapher = MathCausalGrapher(registry)

    # Batch parse questions with spaCy
    questions = [p["question"] for p in problems]
    t0 = time.time()
    q_graphs = text_grapher.parse_batch(questions, batch_size=512)
    print(f"  Questions parsed in {time.time()-t0:.1f}s")

    # Parse solutions with math grapher (sequential, fast enough)
    compiled = []
    import numpy as np
    rng = np.random.default_rng(42)

    for i, prob in enumerate(problems):
        answer = prob["answer"]
        steps = answer.split("####")[0].strip() if "####" in answer else answer

        pos_graph = math_grapher.parse(steps)

        # Hard negative: solution from different problem
        neg_idx = i
        while neg_idx == i:
            neg_idx = rng.integers(len(problems))
        neg_answer = problems[neg_idx]["answer"]
        neg_steps = neg_answer.split("####")[0].strip() if "####" in neg_answer else neg_answer
        neg_graph = math_grapher.parse(neg_steps)

        compiled.append({
            "domain": "gsm8k",
            "question": prob["question"][:100],
            "q_graph": q_graphs[i].to_tensors(),
            "pos_graphs": [pos_graph.to_tensors()],
            "neg_graphs": [neg_graph.to_tensors()],
        })

    print(f"  {len(compiled):,} triplets compiled")
    return compiled


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_compiler(max_nq: int = 5000, max_gsm8k: int = 5000):
    start = time.time()
    registry = EdgeRegistry()

    nq_compiled = compile_nq(registry, max_entries=max_nq)
    gsm8k_compiled = compile_gsm8k(registry, max_problems=max_gsm8k)

    # Save compiled graphs
    print(f"\nSaving compiled graphs...")
    torch.save(nq_compiled, OUTPUT_DIR / "nq_graphs.pt")
    torch.save(gsm8k_compiled, OUTPUT_DIR / "gsm8k_graphs.pt")
    registry.save(OUTPUT_DIR / "v9_edge_registry.json")

    elapsed = time.time() - start

    # Stats
    nq_size = (OUTPUT_DIR / "nq_graphs.pt").stat().st_size
    gsm8k_size = (OUTPUT_DIR / "gsm8k_graphs.pt").stat().st_size

    print(f"\n{'='*60}")
    print(f"  v9 GRAPH COMPILER COMPLETE — {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"  NQ:    {len(nq_compiled):,} triplets → {nq_size/1e6:.1f} MB")
    print(f"  GSM8K: {len(gsm8k_compiled):,} triplets → {gsm8k_size/1e6:.1f} MB")
    print(f"  Edge types: {len(registry)}")
    print(f"  Registry: {OUTPUT_DIR / 'v9_edge_registry.json'}")

    # Show edge type distribution
    print(f"\n  Edge types by domain:")
    text_types = [n for n in registry.name_to_id
                  if not n.startswith("op_") and not n.startswith("step_")
                  and n not in ("computes", "operand_left", "operand_right")]
    math_types = [n for n in registry.name_to_id if n not in text_types]
    print(f"    Text (spaCy): {len(text_types)} — {sorted(text_types)[:10]}...")
    print(f"    Math:         {len(math_types)} — {sorted(math_types)}")
    print(f"{'='*60}")

    log = {
        "experiment": "v9_compile_graphs",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "nq_triplets": len(nq_compiled),
        "gsm8k_triplets": len(gsm8k_compiled),
        "total_edge_types": len(registry),
        "nq_size_mb": round(nq_size / 1e6, 1),
        "gsm8k_size_mb": round(gsm8k_size / 1e6, 1),
    }
    with open(OUTPUT_DIR / "compile_log.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_nq", type=int, default=5000)
    parser.add_argument("--max_gsm8k", type=int, default=5000)
    args = parser.parse_args()
    run_compiler(args.max_nq, args.max_gsm8k)
