#!/usr/bin/env python3
"""v9 Compiler Extension — Code (MBPP) + Telemetry (MWD) graphers.

Completes the Universal Machine Code: 4 domains, one graph format.

Domain 3: Code
  CodeCausalGrapher uses Python's native ast module.
  Edge types: func_def, arg, returns, call, assign, loop_body,
              if_branch, binop_left, binop_right, compare_op, etc.

Domain 4: Physical Telemetry
  TelemetryCausalGrapher parses multi-channel time-series.
  Edge types: t_next, state_transition, causal_correlation, anomaly.

All 4 domains produce identical (node_texts, edge_index, edge_type_ids)
tensors — the sheaf never knows the source.
"""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import numpy as np
import torch

# Import the base infrastructure from the original compiler
from v9_compile_graphs import (
    EdgeRegistry, CompiledGraph, TextCausalGrapher
)

OUTPUT_DIR = Path(__file__).parent / "results" / "v9_compiled_graphs"


# ══════════════════════════════════════════════════════════════════════════════
# Code Causal Grapher — Python AST → typed graph
# ══════════════════════════════════════════════════════════════════════════════

class CodeCausalGrapher:
    """Python ast module → CompiledGraph with algorithmic edge types.

    The AST tree IS the graph. Each Python construct becomes a node,
    and the syntactic parent-child relation becomes a typed edge whose
    type is the role (body, test, orelse, args, etc.).
    """

    def __init__(self, registry: EdgeRegistry):
        self.registry = registry

    def _node_text(self, node: ast.AST) -> str:
        """Human-readable text for an AST node."""
        if isinstance(node, ast.FunctionDef):
            return f"def {node.name}"
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return f"const:{repr(node.value)[:20]}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return f"call:{node.func.id}"
            return "call"
        if isinstance(node, ast.BinOp):
            return f"binop:{type(node.op).__name__}"
        if isinstance(node, ast.Compare):
            return f"cmp:{type(node.ops[0]).__name__}" if node.ops else "cmp"
        if isinstance(node, ast.If):
            return "if"
        if isinstance(node, ast.For):
            return "for"
        if isinstance(node, ast.While):
            return "while"
        if isinstance(node, ast.Return):
            return "return"
        if isinstance(node, ast.Assign):
            return "assign"
        if isinstance(node, ast.arguments):
            return "args"
        if isinstance(node, ast.arg):
            return f"arg:{node.arg}"
        return type(node).__name__.lower()

    def parse(self, code: str) -> CompiledGraph:
        """Parse Python code into a typed causal graph."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return CompiledGraph(["[syntax_error]"], [], [])

        node_texts = []
        edges = []
        edge_types = []
        node_ids = {}  # id(ast_node) → graph node index

        def add_node(ast_node: ast.AST) -> int:
            nid = len(node_texts)
            node_texts.append(self._node_text(ast_node))
            node_ids[id(ast_node)] = nid
            return nid

        def walk_with_types(parent_node: ast.AST, parent_id: int):
            """Walk AST, creating typed edges based on field names."""
            for field_name, value in ast.iter_fields(parent_node):
                if isinstance(value, list):
                    for child in value:
                        if isinstance(child, ast.AST):
                            child_id = add_node(child)
                            edge_type = self._edge_type_for(parent_node, field_name, child)
                            edges.append((parent_id, child_id))
                            edge_types.append(self.registry.get_id(edge_type))
                            walk_with_types(child, child_id)
                elif isinstance(value, ast.AST):
                    child_id = add_node(value)
                    edge_type = self._edge_type_for(parent_node, field_name, value)
                    edges.append((parent_id, child_id))
                    edge_types.append(self.registry.get_id(edge_type))
                    walk_with_types(value, child_id)

        # Start with Module root
        root_id = add_node(tree)
        walk_with_types(tree, root_id)

        return CompiledGraph(node_texts, edges, edge_types)

    def _edge_type_for(self, parent: ast.AST, field: str, child: ast.AST) -> str:
        """Map (parent_type, field, child_type) → edge type string."""
        # The field name carries the structural role
        # Combine with parent type for specificity
        ptype = type(parent).__name__
        return f"code_{ptype}_{field}"


# ══════════════════════════════════════════════════════════════════════════════
# Telemetry Causal Grapher — time series → typed graph
# ══════════════════════════════════════════════════════════════════════════════

class TelemetryCausalGrapher:
    """Multi-channel time series → typed causal graph.

    Nodes: sliding windows across time
    Edges: temporal (t_next), state_transition (large change),
           causal (channel correlation), anomaly (outlier)
    """

    def __init__(self, registry: EdgeRegistry, window_size: int = 10):
        self.registry = registry
        self.window_size = window_size

    def parse(self, signal: np.ndarray, channel_names: list[str] | None = None) -> CompiledGraph:
        """Parse multi-channel telemetry into typed graph.

        Args:
            signal: (n_samples, n_channels) numpy array
            channel_names: optional list of channel names
        """
        if signal.ndim == 1:
            signal = signal[:, None]
        n_samples, n_channels = signal.shape

        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(n_channels)]

        # Create sliding windows
        w = self.window_size
        if n_samples < w:
            return CompiledGraph(["[signal_too_short]"], [], [])

        n_windows = n_samples - w + 1
        node_texts = []
        edges = []
        edge_types = []

        # Each window = one node, per channel
        for ch in range(n_channels):
            for t in range(n_windows):
                window = signal[t:t+w, ch]
                # Compact node text: channel + stats
                mean = window.mean()
                std = window.std()
                node_texts.append(
                    f"{channel_names[ch]}:t{t}:m{mean:.2f}s{std:.2f}"
                )

        # Temporal edges: t_next within each channel
        for ch in range(n_channels):
            ch_offset = ch * n_windows
            for t in range(n_windows - 1):
                src = ch_offset + t
                dst = ch_offset + t + 1
                edges.append((src, dst))
                edge_types.append(self.registry.get_id("t_next"))

        # State transition: detect large jumps within channel
        for ch in range(n_channels):
            ch_offset = ch * n_windows
            for t in range(n_windows - 1):
                w1 = signal[t:t+w, ch]
                w2 = signal[t+1:t+w+1, ch]
                diff = abs(w2.mean() - w1.mean())
                # Threshold: 2x std of window
                if diff > 2 * w1.std() and w1.std() > 1e-6:
                    edges.append((ch_offset + t, ch_offset + t + 1))
                    edge_types.append(self.registry.get_id("state_transition"))

        # Causal correlation: cross-channel at same time
        if n_channels > 1:
            for t in range(n_windows):
                for c1 in range(n_channels):
                    for c2 in range(c1 + 1, n_channels):
                        w1 = signal[t:t+w, c1]
                        w2 = signal[t:t+w, c2]
                        # Pearson correlation
                        if w1.std() > 1e-6 and w2.std() > 1e-6:
                            corr = np.corrcoef(w1, w2)[0, 1]
                            if abs(corr) > 0.8:  # strong correlation
                                src = c1 * n_windows + t
                                dst = c2 * n_windows + t
                                edges.append((src, dst))
                                edge_types.append(
                                    self.registry.get_id("causal_correlation")
                                )

        return CompiledGraph(node_texts, edges, edge_types)


# ══════════════════════════════════════════════════════════════════════════════
# MBPP compiler
# ══════════════════════════════════════════════════════════════════════════════

def compile_mbpp(registry: EdgeRegistry, max_problems: int = 500) -> list[dict]:
    """Compile MBPP problems: text description + Python code triplets."""
    print("\n[MBPP] Compiling code graphs...")

    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        problems = list(ds)[:max_problems]
    except Exception as e:
        print(f"  MBPP load failed ({e}), using synthetic code samples")
        problems = _synthetic_code_problems(max_problems)

    text_grapher = TextCausalGrapher(registry)
    code_grapher = CodeCausalGrapher(registry)

    # Batch parse text (descriptions)
    texts = [p.get("prompt", p.get("text", "")) for p in problems]
    t0 = time.time()
    text_graphs = text_grapher.parse_batch(texts, batch_size=128)
    print(f"  {len(problems)} descriptions parsed in {time.time()-t0:.1f}s")

    # Parse code with AST
    compiled = []
    rng = np.random.default_rng(42)
    t0 = time.time()

    for i, prob in enumerate(problems):
        code = prob.get("code", "")
        pos_graph = code_grapher.parse(code)

        # Hard negative: code from a different problem
        neg_idx = i
        while neg_idx == i:
            neg_idx = rng.integers(len(problems))
        neg_code = problems[neg_idx].get("code", "")
        neg_graph = code_grapher.parse(neg_code)

        if pos_graph.edge_index and neg_graph.edge_index:
            compiled.append({
                "domain": "mbpp",
                "question": texts[i][:100],
                "q_graph": text_graphs[i].to_tensors(),
                "pos_graphs": [pos_graph.to_tensors()],
                "neg_graphs": [neg_graph.to_tensors()],
            })

    print(f"  {len(compiled):,} code triplets compiled in {time.time()-t0:.1f}s")
    return compiled


def _synthetic_code_problems(n: int) -> list[dict]:
    """Generate synthetic Python problems as fallback."""
    templates = [
        ("sum a list of numbers",
         "def solve(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total"),
        ("find max value",
         "def solve(lst):\n    best = lst[0]\n    for x in lst:\n        if x > best:\n            best = x\n    return best"),
        ("check if number is prime",
         "def solve(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True"),
        ("reverse a string",
         "def solve(s):\n    return s[::-1]"),
        ("count vowels",
         "def solve(s):\n    return sum(1 for c in s if c in 'aeiou')"),
    ]
    problems = []
    for i in range(n):
        text, code = templates[i % len(templates)]
        problems.append({"prompt": text, "code": code})
    return problems


# ══════════════════════════════════════════════════════════════════════════════
# Telemetry compiler
# ══════════════════════════════════════════════════════════════════════════════

def compile_telemetry(registry: EdgeRegistry, n_sequences: int = 500) -> list[dict]:
    """Compile synthetic MWD telemetry with causal patterns."""
    print("\n[Telemetry] Compiling physics graphs...")

    text_grapher = TextCausalGrapher(registry)
    tel_grapher = TelemetryCausalGrapher(registry, window_size=10)

    compiled = []
    rng = np.random.default_rng(7)

    # Patterns: normal, stick-slip, bit-bounce, washout, drilling-break
    pattern_types = {
        "normal": lambda: _gen_normal(rng, 100),
        "stick_slip": lambda: _gen_stick_slip(rng, 100),
        "bit_bounce": lambda: _gen_bit_bounce(rng, 100),
        "washout": lambda: _gen_washout(rng, 100),
    }
    descriptions = {
        "normal": "normal drilling operation with steady ROP and torque",
        "stick_slip": "rotational speed oscillates while bit torque spikes",
        "bit_bounce": "axial vibration amplitude exceeds safe threshold",
        "washout": "mud pressure drops indicating hole washout",
    }

    pattern_list = list(pattern_types.keys())

    t0 = time.time()
    for i in range(n_sequences):
        pattern = pattern_list[i % len(pattern_list)]
        query_text = descriptions[pattern]
        pos_signal = pattern_types[pattern]()
        # Negative: different pattern
        neg_pattern = pattern
        while neg_pattern == pattern:
            neg_pattern = pattern_list[rng.integers(len(pattern_list))]
        neg_signal = pattern_types[neg_pattern]()

        q_graph = text_grapher.parse(query_text)
        pos_graph = tel_grapher.parse(
            pos_signal, channel_names=["rpm", "torque", "wob", "pressure"]
        )
        neg_graph = tel_grapher.parse(
            neg_signal, channel_names=["rpm", "torque", "wob", "pressure"]
        )

        if pos_graph.edge_index and neg_graph.edge_index:
            compiled.append({
                "domain": "telemetry",
                "question": query_text[:100],
                "pattern": pattern,
                "q_graph": q_graph.to_tensors(),
                "pos_graphs": [pos_graph.to_tensors()],
                "neg_graphs": [neg_graph.to_tensors()],
            })

    print(f"  {len(compiled):,} telemetry triplets compiled in {time.time()-t0:.1f}s")
    return compiled


def _gen_normal(rng, n) -> np.ndarray:
    """Normal drilling: steady values with small noise."""
    t = np.arange(n)
    rpm = 100 + 2 * np.sin(t * 0.1) + rng.normal(0, 1, n)
    torque = 50 + 0.3 * rpm + rng.normal(0, 2, n)
    wob = 20000 + rng.normal(0, 200, n)
    pressure = 3000 + rng.normal(0, 50, n)
    return np.stack([rpm, torque, wob, pressure], axis=1)


def _gen_stick_slip(rng, n) -> np.ndarray:
    """Stick-slip: RPM oscillates violently, torque inversely."""
    t = np.arange(n)
    rpm = 100 + 40 * np.sin(t * 0.3) + rng.normal(0, 3, n)
    torque = 100 - 0.5 * rpm + rng.normal(0, 5, n)
    wob = 20000 + rng.normal(0, 300, n)
    pressure = 3000 + rng.normal(0, 50, n)
    return np.stack([rpm, torque, wob, pressure], axis=1)


def _gen_bit_bounce(rng, n) -> np.ndarray:
    """Bit bounce: axial (WOB) vibration spikes."""
    t = np.arange(n)
    rpm = 100 + rng.normal(0, 2, n)
    torque = 50 + rng.normal(0, 3, n)
    wob = 20000 + 5000 * np.sin(t * 0.5) + rng.normal(0, 1000, n)
    pressure = 3000 + rng.normal(0, 50, n)
    return np.stack([rpm, torque, wob, pressure], axis=1)


def _gen_washout(rng, n) -> np.ndarray:
    """Washout: mud pressure drops progressively."""
    t = np.arange(n)
    rpm = 100 + rng.normal(0, 2, n)
    torque = 50 + rng.normal(0, 3, n)
    wob = 20000 + rng.normal(0, 200, n)
    pressure = 3000 - 15 * t + rng.normal(0, 30, n)
    return np.stack([rpm, torque, wob, pressure], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Main: extend existing compiled graphs with new domains
# ══════════════════════════════════════════════════════════════════════════════

def run_extension(max_mbpp: int = 500, n_telemetry: int = 500):
    start = time.time()

    # Load existing registry
    registry_path = OUTPUT_DIR / "v9_edge_registry.json"
    registry = EdgeRegistry()
    if registry_path.exists():
        with open(registry_path) as f:
            data = json.load(f)
        for name, id_ in data["name_to_id"].items():
            registry.name_to_id[name] = id_
            registry.id_to_name[id_] = name
        registry._next_id = data["total_types"]
        print(f"Loaded existing registry: {len(registry)} edge types")
    else:
        print("Starting fresh registry")

    # Compile new domains
    mbpp_graphs = compile_mbpp(registry, max_problems=max_mbpp)
    tel_graphs = compile_telemetry(registry, n_sequences=n_telemetry)

    # Save
    print(f"\nSaving extended graphs...")
    torch.save(mbpp_graphs, OUTPUT_DIR / "mbpp_graphs.pt")
    torch.save(tel_graphs, OUTPUT_DIR / "telemetry_graphs.pt")
    registry.save(OUTPUT_DIR / "v9_edge_registry.json")

    elapsed = time.time() - start

    # Stats
    print(f"\n{'='*65}")
    print(f"  v9 TOTAL SOLUTION COMPILER — {elapsed:.0f}s")
    print(f"{'='*65}")
    print(f"  MBPP (code):       {len(mbpp_graphs):,} triplets")
    print(f"  Telemetry (phys):  {len(tel_graphs):,} triplets")
    print(f"  Total edge types:  {len(registry)}")

    # Categorize edge types
    text_types = [n for n in registry.name_to_id
                  if not any(n.startswith(p) for p in
                             ["op_", "step_", "code_", "t_", "state_", "causal_", "anomaly"])
                  and n not in ("computes", "operand_left", "operand_right", "bridge")]
    math_types = [n for n in registry.name_to_id
                  if n.startswith(("op_", "step_")) or n in ("computes", "operand_left", "operand_right")]
    code_types = [n for n in registry.name_to_id if n.startswith("code_")]
    phys_types = [n for n in registry.name_to_id
                  if n.startswith(("t_", "state_", "causal_")) or n == "anomaly"]

    print(f"\n  Domain breakdown:")
    print(f"    Text (syntax):   {len(text_types):>3d} types")
    print(f"    Math:             {len(math_types):>3d} types")
    print(f"    Code (AST):       {len(code_types):>3d} types")
    print(f"    Physics (causal): {len(phys_types):>3d} types")
    print(f"{'='*65}")

    log = {
        "experiment": "v9_compile_extension",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mbpp_triplets": len(mbpp_graphs),
        "telemetry_triplets": len(tel_graphs),
        "total_edge_types": len(registry),
        "breakdown": {
            "text": len(text_types),
            "math": len(math_types),
            "code": len(code_types),
            "physics": len(phys_types),
        },
    }
    with open(OUTPUT_DIR / "compile_ext_log.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_mbpp", type=int, default=500)
    parser.add_argument("--n_telemetry", type=int, default=500)
    args = parser.parse_args()
    run_extension(args.max_mbpp, args.n_telemetry)
