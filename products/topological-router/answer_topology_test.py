#!/usr/bin/env python3
"""Answer-Topology Test — Does hidden state topology separate truth from falsehood?

PROTOCOL (v2 — single-pass MCQ):
    Present the model with a standard MCQ prompt (question + ALL choices visible).
    Single forward pass. The model deliberates over all options simultaneously.

    We measure THREE things:
    1. DELIBERATION TOPOLOGY — Gini trajectory at the decision point
       (does the model "know" the answer internally, even if it picks wrong?)
    2. PER-CHOICE REGION TOPOLOGY — Gini of hidden states in each choice's token span
       (does the correct choice's region have different topology than wrong choices?)
    3. DECISION-POINT TOPOLOGY — hidden state at the final "Answer:" token
       (the moment of selection — is the topology richer when the model gets it right?)

THE C/I/V DECOMPOSITION:
    Context (c): Question + all choices — the full MCQ prompt (same for all measurements)
    Intent (i): Model's topological state at deliberation — Gini trajectory
    Value (v): Ground truth correctness — the label

WHY SINGLE-PASS:
    Embedding one answer at a time treats the answer as context, not as something
    being evaluated. In causal attention, question tokens don't see the answer.
    With MCQ format, ALL choices are visible — the model's hidden state at the
    decision point integrates information about ALL options simultaneously.

DATASETS:
    Tier 1: ARC-Challenge (1172 science questions, public)
    Tier 2: MMLU college_physics (102 graduate physics, public)
    Tier 3: GPQA Diamond (198 expert physics, local CSV)
"""
from __future__ import annotations

import csv
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu, ttest_ind
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "results"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GPQA_PATH = Path("/home/wb1/Desktop/Dev/gpqa/gpqa_diamond.csv")


# ── Core topology ───────────────────────────────────────────────────────────

def gini_fast(values: np.ndarray) -> float:
    """Gini coefficient — scalar measure of topological hierarchy."""
    n = len(values)
    if n <= 1 or np.sum(values) == 0:
        return 0.0
    s = np.sort(values)
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * np.sum(i * s)) / (n * np.sum(s)) - (n + 1) / n)


def h0_persistence(points: np.ndarray, max_n: int = 150) -> np.ndarray:
    """H₀ persistence bars via GPU pairwise distance + CPU union-find."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    if n > max_n:
        idx = np.random.default_rng(42).choice(n, max_n, replace=False)
        points = points[idx]
        n = max_n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts = torch.tensor(points, dtype=torch.float32, device=device)
    dists = torch.cdist(pts, pts)
    mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    flat = dists[mask]
    sorted_d, sorted_idx = torch.sort(flat)

    rows, cols = torch.where(mask)
    si = rows[sorted_idx].cpu().numpy()
    sj = cols[sorted_idx].cpu().numpy()
    sd = sorted_d.cpu().numpy()

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bars = []
    for k in range(len(sd)):
        ri, rj = find(int(si[k])), find(int(sj[k]))
        if ri != rj:
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1
            bars.append(float(sd[k]))

    return np.array(bars) if bars else np.array([0.0])


def hidden_state_gini(hidden_state: torch.Tensor, pca_dim: int = 30) -> float:
    """Gini of H₀ persistence on a hidden state tensor."""
    h = hidden_state.cpu().float().numpy()
    if h.shape[0] < 3:
        return 0.0

    d = min(pca_dim, h.shape[0] - 1, h.shape[1])
    h_c = h - h.mean(0)
    try:
        _, _, Vt = np.linalg.svd(h_c, full_matrices=False)
        h_r = h_c @ Vt[:d].T
    except np.linalg.LinAlgError:
        h_r = h[:, :d]

    bars = h0_persistence(h_r)
    return gini_fast(bars)


# ── Data loading ────────────────────────────────────────────────────────────

@dataclass
class MCQuestion:
    """Normalized multiple-choice question."""
    question: str
    choices: list[str]
    correct_idx: int
    source: str
    domain: str


def load_arc_challenge(n_max: int = 0) -> list[MCQuestion]:
    """Load ARC-Challenge (public, 1172 hard science questions)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    questions = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    for row in ds:
        choices = row["choices"]["text"]
        key = row["answerKey"]
        if key not in label_map:
            continue
        correct_idx = label_map[key]
        if correct_idx >= len(choices):
            continue
        questions.append(MCQuestion(
            question=row["question"], choices=choices,
            correct_idx=correct_idx, source="ARC-Challenge", domain="science",
        ))
    if n_max > 0:
        questions = questions[:n_max]
    return questions


def load_mmlu_physics(n_max: int = 0) -> list[MCQuestion]:
    """Load MMLU college physics (public, 102 questions)."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "college_physics", split="test")
    questions = []
    for row in ds:
        questions.append(MCQuestion(
            question=row["question"], choices=row["choices"],
            correct_idx=int(row["answer"]), source="MMLU-Physics", domain="physics",
        ))
    if n_max > 0:
        questions = questions[:n_max]
    return questions


def load_gpqa_diamond(n_max: int = 0) -> list[MCQuestion]:
    """Load GPQA Diamond from local CSV."""
    if not GPQA_PATH.exists():
        raise FileNotFoundError(f"GPQA not found at {GPQA_PATH}")

    questions = []
    with open(GPQA_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["Question"]
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            # Skip if any choice is empty
            if not all(c.strip() for c in choices):
                continue
            questions.append(MCQuestion(
                question=q, choices=choices, correct_idx=0,
                source="GPQA-Diamond",
                domain=row.get("High-level domain", "physics"),
            ))

    # Shuffle choices so correct isn't always idx 0
    rng = np.random.default_rng(42)
    for q in questions:
        perm = rng.permutation(len(q.choices))
        q.choices = [q.choices[int(i)] for i in perm]
        q.correct_idx = int(np.where(perm == 0)[0][0])

    if n_max > 0:
        questions = questions[:n_max]
    return questions


# ── MCQ formatting ──────────────────────────────────────────────────────────

LABELS = ["A", "B", "C", "D", "E", "F"]


def format_mcq(q: MCQuestion) -> str:
    """Format as standard MCQ prompt."""
    lines = [f"Question: {q.question}", ""]
    for i, choice in enumerate(q.choices):
        lines.append(f"{LABELS[i]}) {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def find_choice_spans(tokenizer, full_text: str, choices: list[str]) -> list[tuple[int, int]]:
    """Find the token spans of each choice in the tokenized text.

    Returns list of (start_token, end_token) for each choice.
    """
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    spans = []

    for i, choice in enumerate(choices):
        # Find the choice text in the full prompt
        marker = f"{LABELS[i]}) {choice}"
        char_start = full_text.find(marker)
        if char_start < 0:
            spans.append((0, 0))
            continue

        # Tokenize prefix up to the choice and prefix+choice
        prefix = full_text[:char_start]
        prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"][0]
        prefix_choice = full_text[:char_start + len(marker)]
        prefix_choice_ids = tokenizer(prefix_choice, return_tensors="pt")["input_ids"][0]

        start_tok = len(prefix_ids)
        end_tok = len(prefix_choice_ids)
        spans.append((start_tok, end_tok))

    return spans


# ── Model wrapper ───────────────────────────────────────────────────────────

class TopologyProbe:
    """Probes hidden state topology for MCQ deliberation."""

    def __init__(self, model_name: str, device: str = "auto", pca_dim: int = 30):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pca_dim = pca_dim
        self.model_name = model_name

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            dtype=torch.float16, device_map=device,
            output_hidden_states=True,
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        n_layers = self.model.config.num_hidden_layers
        print(f"  {n_params:.0f}M params, {n_layers} layers, device={device}")

    @torch.no_grad()
    def measure_question(self, q: MCQuestion) -> dict:
        """Measure topology for an MCQ question (single forward pass).

        Protocol:
        1. Format as standard MCQ (question + all choices + "Answer:")
        2. Single forward pass
        3. Extract:
           a. Full deliberation Gini trajectory (all tokens, all layers)
           b. Per-choice region Gini (each choice's token span)
           c. Decision-point Gini (last token's hidden state context)
           d. Model's logit-based selection
        """
        prompt = format_mcq(q)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states)
        seq_len = inputs["input_ids"].shape[1]

        # Find token spans for each choice
        choice_spans = find_choice_spans(self.tokenizer, prompt, q.choices)

        # ── Channel 1: Full deliberation Gini trajectory ──
        full_gini_trajectory = []
        for hs in hidden_states:
            g = hidden_state_gini(hs[0], pca_dim=self.pca_dim)
            full_gini_trajectory.append(g)

        # ── Channel 2: Per-choice region Gini ──
        # For each choice, compute Gini of hidden states in that choice's token span
        choice_ginis = []  # list of trajectories, one per choice
        choice_terminal_ginis = []  # terminal layer Gini per choice
        for start_tok, end_tok in choice_spans:
            if end_tok - start_tok >= 3:
                traj = []
                for hs in hidden_states:
                    g = hidden_state_gini(hs[0, start_tok:end_tok, :],
                                          pca_dim=self.pca_dim)
                    traj.append(g)
                choice_ginis.append(traj)
                # Use second-to-last layer as terminal
                choice_terminal_ginis.append(traj[-2] if len(traj) > 2 else traj[-1])
            else:
                # Choice span too short — use last 5 tokens ending at span
                traj = []
                s = max(0, end_tok - 5)
                for hs in hidden_states:
                    g = hidden_state_gini(hs[0, s:end_tok, :],
                                          pca_dim=self.pca_dim)
                    traj.append(g)
                choice_ginis.append(traj)
                choice_terminal_ginis.append(traj[-2] if len(traj) > 2 else traj[-1])

        # ── Channel 3: Decision-point topology ──
        # Hidden state at the last token ("Answer:" prompt) — the deliberation point
        decision_gini_trajectory = []
        # Use a window around the decision point (last 10 tokens)
        decision_start = max(0, seq_len - 10)
        for hs in hidden_states:
            if seq_len - decision_start >= 3:
                g = hidden_state_gini(hs[0, decision_start:, :],
                                      pca_dim=self.pca_dim)
            else:
                g = 0.0
            decision_gini_trajectory.append(g)

        # ── Model's logit-based answer selection ──
        # Check logits for A, B, C, D tokens at the last position
        last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
        choice_logits = []
        for label in LABELS[:len(q.choices)]:
            token_id = self.tokenizer.encode(label, add_special_tokens=False)
            if token_id:
                choice_logits.append(float(last_logits[token_id[0]].item()))
            else:
                choice_logits.append(-1e9)

        logit_prediction = int(np.argmax(choice_logits))

        # ── Topology-based answer selection ──
        # Which choice has highest terminal Gini?
        if choice_terminal_ginis:
            topo_prediction = int(np.argmax(choice_terminal_ginis))
        else:
            topo_prediction = 0

        # ── Summary stats ──
        fgt = np.array(full_gini_trajectory)
        full_terminal = float(fgt[-2]) if n_layers > 2 else float(fgt[-1])

        if n_layers > 3:
            x = np.arange(1, n_layers)
            full_slope = float(np.polyfit(x, fgt[1:], 1)[0])
        else:
            full_slope = 0.0

        # Correct choice Gini vs mean wrong Gini
        correct_choice_gini = choice_terminal_ginis[q.correct_idx] if choice_terminal_ginis else 0.0
        wrong_choice_ginis = [g for i, g in enumerate(choice_terminal_ginis) if i != q.correct_idx]

        return {
            "question": q.question[:200],
            "source": q.source,
            "domain": q.domain,
            "n_choices": len(q.choices),
            "correct_idx": q.correct_idx,
            "seq_len": seq_len,
            # Channel 1: Full deliberation
            "full_gini_trajectory": full_gini_trajectory,
            "full_terminal_gini": full_terminal,
            "full_slope": full_slope,
            # Channel 2: Per-choice region
            "choice_terminal_ginis": choice_terminal_ginis,
            "correct_choice_gini": correct_choice_gini,
            "wrong_choice_ginis": wrong_choice_ginis,
            # Channel 3: Decision point
            "decision_gini_trajectory": decision_gini_trajectory,
            # Predictions
            "topo_prediction": topo_prediction,
            "topo_correct": topo_prediction == q.correct_idx,
            "logit_prediction": logit_prediction,
            "logit_correct": logit_prediction == q.correct_idx,
            "choice_logits": choice_logits,
            # Metadata
            "n_layers": n_layers,
        }

    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


# ── Statistical analysis ────────────────────────────────────────────────────

def analyze_results(results: list[dict], model_name: str) -> dict:
    """Full statistical analysis."""
    correct_ginis = []
    wrong_ginis = []
    topo_correct_count = 0
    logit_correct_count = 0
    n_total = len(results)

    # Separate results by whether model got it right (logit-based)
    right_full_ginis = []   # full deliberation Gini when model is correct
    wrong_full_ginis = []   # full deliberation Gini when model is incorrect

    for r in results:
        correct_ginis.append(r["correct_choice_gini"])
        wrong_ginis.extend(r["wrong_choice_ginis"])
        if r["topo_correct"]:
            topo_correct_count += 1
        if r["logit_correct"]:
            logit_correct_count += 1
            right_full_ginis.append(r["full_terminal_gini"])
        else:
            wrong_full_ginis.append(r["full_terminal_gini"])

    cg = np.array(correct_ginis)
    wg = np.array(wrong_ginis)

    # Test 1: Does correct choice have higher topology than wrong choices?
    if len(cg) > 5 and len(wg) > 5:
        u_stat, u_p = mannwhitneyu(cg, wg, alternative="greater")
        t_stat, t_p = ttest_ind(cg, wg, alternative="greater")
    else:
        u_stat, u_p, t_stat, t_p = 0, 1, 0, 1

    pooled_std = np.sqrt((np.var(cg) + np.var(wg)) / 2)
    cohens_d = (np.mean(cg) - np.mean(wg)) / pooled_std if pooled_std > 0 else 0.0

    # Test 2: Does deliberation topology predict model correctness?
    rfg = np.array(right_full_ginis)
    wfg = np.array(wrong_full_ginis)
    if len(rfg) > 5 and len(wfg) > 5:
        u_delib, p_delib = mannwhitneyu(rfg, wfg, alternative="greater")
        cohens_d_delib = ((np.mean(rfg) - np.mean(wfg)) /
                          np.sqrt((np.var(rfg) + np.var(wfg)) / 2))
    else:
        u_delib, p_delib = 0, 1
        cohens_d_delib = 0.0

    n_choices = results[0]["n_choices"] if results else 4
    random_baseline = 1.0 / n_choices

    analysis = {
        "model": model_name,
        "n_questions": n_total,
        "n_choices": n_choices,

        # Test 1: Per-choice topology separation
        "correct_gini_mean": float(np.mean(cg)),
        "correct_gini_std": float(np.std(cg)),
        "wrong_gini_mean": float(np.mean(wg)),
        "wrong_gini_std": float(np.std(wg)),
        "choice_gini_diff": float(np.mean(cg) - np.mean(wg)),
        "choice_cohens_d": float(cohens_d),
        "choice_mann_whitney_p": float(u_p),
        "choice_ttest_p": float(t_p),

        # Test 2: Deliberation topology predicts correctness
        "right_delib_gini_mean": float(np.mean(rfg)) if len(rfg) > 0 else 0,
        "wrong_delib_gini_mean": float(np.mean(wfg)) if len(wfg) > 0 else 0,
        "delib_cohens_d": float(cohens_d_delib),
        "delib_mann_whitney_p": float(p_delib),

        # Accuracy
        "topo_accuracy": topo_correct_count / n_total if n_total > 0 else 0,
        "logit_accuracy": logit_correct_count / n_total if n_total > 0 else 0,
        "random_baseline": random_baseline,
        "topo_vs_random": (topo_correct_count / n_total - random_baseline) if n_total > 0 else 0,
        "logit_vs_random": (logit_correct_count / n_total - random_baseline) if n_total > 0 else 0,
    }

    return analysis


def plot_results(results: list[dict], analysis: dict, dataset_name: str):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"gold": "#c5a03f", "teal": "#45a8b0", "red": "#e94560",
              "bg": "#0f0d08", "text": "#d6d0be", "muted": "#817a66",
              "green": "#4caf50"}

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.facecolor": COLORS["bg"], "figure.facecolor": COLORS["bg"],
        "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
        "axes.edgecolor": COLORS["muted"], "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.dpi": 200,
    })

    model_short = analysis["model"].split("/")[-1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ── Row 1, Col 1: Correct vs Wrong choice-region Gini ──
    ax = axes[0, 0]
    correct_ginis = [r["correct_choice_gini"] for r in results]
    wrong_ginis = []
    for r in results:
        wrong_ginis.extend(r["wrong_choice_ginis"])
    ax.hist(correct_ginis, bins=25, alpha=0.7, color=COLORS["teal"],
            label=f'Correct (n={len(correct_ginis)})', density=True)
    ax.hist(wrong_ginis, bins=25, alpha=0.7, color=COLORS["red"],
            label=f'Wrong (n={len(wrong_ginis)})', density=True)
    ax.axvline(np.mean(correct_ginis), color=COLORS["teal"], ls="--", lw=2)
    ax.axvline(np.mean(wrong_ginis), color=COLORS["red"], ls="--", lw=2)
    ax.set_xlabel("Choice-Region Gini")
    ax.set_ylabel("Density")
    ax.set_title(f"Per-Choice Topology\n(d={analysis['choice_cohens_d']:.3f}, p={analysis['choice_mann_whitney_p']:.2e})",
                 color=COLORS["gold"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # ── Row 1, Col 2: Per-question Gini separation ──
    ax = axes[0, 1]
    for i, r in enumerate(results):
        cg = r["correct_choice_gini"]
        wg_mean = np.mean(r["wrong_choice_ginis"]) if r["wrong_choice_ginis"] else 0
        color = COLORS["teal"] if cg > wg_mean else COLORS["red"]
        ax.scatter(i, cg - wg_mean, c=color, s=12, alpha=0.6)
    ax.axhline(0, color=COLORS["muted"], ls="--", alpha=0.5)
    n_positive = sum(1 for r in results
                     if r["correct_choice_gini"] > np.mean(r["wrong_choice_ginis"]))
    ax.set_xlabel("Question Index")
    ax.set_ylabel("Gini(correct) - mean(Gini(wrong))")
    ax.set_title(f"Per-Question Separation ({n_positive}/{len(results)} positive)",
                 color=COLORS["gold"])
    ax.grid(True, alpha=0.15)

    # ── Row 1, Col 3: Accuracy comparison ──
    ax = axes[0, 2]
    methods = ["Topology\n(choice Gini)", "Logit\n(standard)", "Random\nbaseline"]
    accs = [analysis["topo_accuracy"], analysis["logit_accuracy"], analysis["random_baseline"]]
    colors_bar = [COLORS["teal"], COLORS["gold"], COLORS["muted"]]
    bars = ax.bar(methods, accs, color=colors_bar, alpha=0.8, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=11, color=COLORS["text"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Answer Selection: Topology vs Logit vs Random", color=COLORS["gold"])
    ax.set_ylim(0, max(accs) * 1.4)
    ax.grid(True, alpha=0.15, axis="y")

    # ── Row 2, Col 1: Deliberation topology when right vs wrong ──
    ax = axes[1, 0]
    right_ginis = [r["full_terminal_gini"] for r in results if r["logit_correct"]]
    wrong_ginis_delib = [r["full_terminal_gini"] for r in results if not r["logit_correct"]]
    if right_ginis:
        ax.hist(right_ginis, bins=20, alpha=0.7, color=COLORS["green"],
                label=f'Model correct (n={len(right_ginis)})', density=True)
    if wrong_ginis_delib:
        ax.hist(wrong_ginis_delib, bins=20, alpha=0.7, color=COLORS["red"],
                label=f'Model wrong (n={len(wrong_ginis_delib)})', density=True)
    ax.set_xlabel("Full Deliberation Gini")
    ax.set_ylabel("Density")
    ax.set_title(f"Deliberation Topology vs Correctness\n(d={analysis['delib_cohens_d']:.3f}, p={analysis['delib_mann_whitney_p']:.2e})",
                 color=COLORS["gold"])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # ── Row 2, Col 2: Example Gini trajectories ──
    ax = axes[1, 1]
    n_show = min(6, len(results))
    for i in range(n_show):
        r = results[i]
        for ci, cg_traj in enumerate(zip(*[
            [(j, r["choice_terminal_ginis"][j]) for j in range(r["n_choices"])]
        ])):
            pass  # Skip complex trajectory plot, do simple version below

    # Simple: plot full deliberation trajectory for first 6 questions
    for i in range(n_show):
        r = results[i]
        traj = r["full_gini_trajectory"]
        color = COLORS["teal"] if r["logit_correct"] else COLORS["red"]
        alpha = 0.8 if r["logit_correct"] else 0.4
        ax.plot(traj, color=color, alpha=alpha, lw=1.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Full Deliberation Gini")
    ax.set_title("Gini Trajectories (teal=correct, red=wrong)", color=COLORS["gold"])
    ax.grid(True, alpha=0.15)

    # ── Row 2, Col 3: Choice Gini by position ──
    ax = axes[1, 2]
    n_choices = results[0]["n_choices"] if results else 4
    choice_ginis_by_pos = [[] for _ in range(n_choices)]
    correct_gini_vals = []
    for r in results:
        for ci, g in enumerate(r["choice_terminal_ginis"]):
            choice_ginis_by_pos[ci].append(g)
            if ci == r["correct_idx"]:
                correct_gini_vals.append(g)

    bp = ax.boxplot([cg for cg in choice_ginis_by_pos if cg],
                    labels=[LABELS[i] for i in range(len(choice_ginis_by_pos)) if choice_ginis_by_pos[i]],
                    patch_artist=True,
                    boxprops=dict(facecolor=COLORS["muted"], alpha=0.3),
                    medianprops=dict(color=COLORS["gold"]))
    ax.set_xlabel("Choice Position")
    ax.set_ylabel("Choice-Region Gini")
    ax.set_title("Topology by Choice Position", color=COLORS["gold"])
    ax.grid(True, alpha=0.15)

    fig.suptitle(f"Answer Topology Test v2 (MCQ) -- {dataset_name} x {model_short}",
                 color=COLORS["text"], fontsize=14)
    fig.tight_layout()

    fname = f"answer_topo_v2_{dataset_name.lower().replace('-','_')}_{model_short}.png"
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
    print(f"  Figure: {FIG_DIR / fname}")


# ── Main pipeline ───────────────────────────────────────────────────────────

def run_tier(probe: TopologyProbe, questions: list[MCQuestion],
             dataset_name: str, model_name: str) -> dict:
    """Run answer-topology test on one dataset tier."""
    print(f"\n{'='*70}")
    print(f"  TIER: {dataset_name} x {model_name.split('/')[-1]}")
    print(f"  Questions: {len(questions)}")
    print(f"{'='*70}")

    results = []
    t0 = time.time()

    for i, q in enumerate(questions):
        r = probe.measure_question(q)
        results.append(r)

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            elapsed = time.time() - t0
            topo_acc = sum(1 for r in results if r["topo_correct"]) / len(results)
            logit_acc = sum(1 for r in results if r["logit_correct"]) / len(results)
            print(f"  [{i+1}/{len(questions)}] "
                  f"topo={topo_acc:.1%} logit={logit_acc:.1%} "
                  f"elapsed={elapsed:.0f}s")

    total_time = time.time() - t0

    analysis = analyze_results(results, model_name)
    analysis["dataset"] = dataset_name
    analysis["total_time_s"] = round(total_time, 1)

    # Print
    print(f"\n  {'='*50}")
    print(f"  RESULTS: {dataset_name} x {model_name.split('/')[-1]}")
    print(f"  {'='*50}")
    print(f"  TEST 1 — Per-choice topology separation:")
    print(f"    Correct choice Gini: {analysis['correct_gini_mean']:.4f} +/- {analysis['correct_gini_std']:.4f}")
    print(f"    Wrong choice Gini:   {analysis['wrong_gini_mean']:.4f} +/- {analysis['wrong_gini_std']:.4f}")
    print(f"    Difference:          {analysis['choice_gini_diff']:+.4f}")
    print(f"    Cohen's d:           {analysis['choice_cohens_d']:.3f}")
    print(f"    Mann-Whitney p:      {analysis['choice_mann_whitney_p']:.2e}")
    print(f"  TEST 2 — Deliberation topology predicts correctness:")
    print(f"    Right answers Gini:  {analysis['right_delib_gini_mean']:.4f}")
    print(f"    Wrong answers Gini:  {analysis['wrong_delib_gini_mean']:.4f}")
    print(f"    Cohen's d:           {analysis['delib_cohens_d']:.3f}")
    print(f"    Mann-Whitney p:      {analysis['delib_mann_whitney_p']:.2e}")
    print(f"  ACCURACY:")
    print(f"    Topology:  {analysis['topo_accuracy']:.1%} ({analysis['topo_vs_random']:+.1%} vs random)")
    print(f"    Logit:     {analysis['logit_accuracy']:.1%} ({analysis['logit_vs_random']:+.1%} vs random)")
    print(f"    Random:    {analysis['random_baseline']:.1%}")
    print(f"  Time: {total_time:.0f}s ({total_time/len(questions):.1f}s/q)")

    # Verdict
    if analysis["choice_mann_whitney_p"] < 0.01 and analysis["choice_cohens_d"] > 0.2:
        verdict = "TOPOLOGY SEPARATES TRUTH (per-choice)"
    elif analysis["delib_mann_whitney_p"] < 0.01 and analysis["delib_cohens_d"] > 0.2:
        verdict = "TOPOLOGY PREDICTS CORRECTNESS (deliberation)"
    elif analysis["choice_mann_whitney_p"] < 0.05 or analysis["delib_mann_whitney_p"] < 0.05:
        verdict = "WEAK SIGNAL — marginally significant"
    elif analysis["topo_accuracy"] > analysis["random_baseline"] * 1.3:
        verdict = "ACCURACY SIGNAL — topology beats random"
    else:
        verdict = "NO SEPARATION"

    analysis["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    plot_results(results, analysis, dataset_name)

    fname = f"answer_topo_v2_{dataset_name.lower().replace('-','_')}_{model_name.split('/')[-1]}.json"
    with open(OUTPUT_DIR / fname, "w") as f:
        json.dump({"analysis": analysis, "results": results}, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR / fname}")

    return analysis


def main():
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  ANSWER-TOPOLOGY TEST v2 (MCQ single-pass)")
    print("  Does hidden state topology separate truth from falsehood?")
    print(f"  {timestamp}")
    print("=" * 70)

    models = [
        "Qwen/Qwen2.5-0.5B",
    ]

    # Load datasets — outside in
    print("\n  Loading datasets...")
    tiers = []

    # Tier 1: ARC-Challenge (100 for initial run)
    arc = load_arc_challenge(n_max=100)
    tiers.append(("ARC-Challenge", arc))
    print(f"  ARC-Challenge: {len(arc)} questions")

    # Tier 2: MMLU college physics
    mmlu = load_mmlu_physics()
    tiers.append(("MMLU-Physics", mmlu))
    print(f"  MMLU-Physics: {len(mmlu)} questions")

    # Tier 3: GPQA Diamond (local)
    try:
        gpqa = load_gpqa_diamond()
        tiers.append(("GPQA-Diamond", gpqa))
        print(f"  GPQA-Diamond: {len(gpqa)} questions")
    except FileNotFoundError as e:
        print(f"  GPQA-Diamond: {e}")

    all_analyses = []

    for model_name in models:
        probe = TopologyProbe(model_name)

        for dataset_name, questions in tiers:
            analysis = run_tier(probe, questions, dataset_name, model_name)
            all_analyses.append(analysis)

        probe.cleanup()

    # Cross-tier summary
    print(f"\n{'='*70}")
    print("  CROSS-TIER SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Dataset':<18} {'Model':<16} {'d_choice':>9} {'d_delib':>8} {'Topo%':>7} {'Logit%':>7} {'Verdict'}")
    print(f"  {'-'*85}")
    for a in all_analyses:
        print(f"  {a['dataset']:<18} {a['model'].split('/')[-1]:<16} "
              f"{a['choice_cohens_d']:>9.3f} {a['delib_cohens_d']:>8.3f} "
              f"{a['topo_accuracy']:>6.1%} {a['logit_accuracy']:>6.1%} "
              f"{a['verdict'][:28]}")

    with open(OUTPUT_DIR / "answer_topology_v2_summary.json", "w") as f:
        json.dump({"timestamp": timestamp, "analyses": all_analyses}, f, indent=2, default=str)

    print(f"\n  Summary: {OUTPUT_DIR / 'answer_topology_v2_summary.json'}")


if __name__ == "__main__":
    main()
