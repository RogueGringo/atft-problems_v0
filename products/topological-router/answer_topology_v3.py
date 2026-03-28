#!/usr/bin/env python3
"""Answer-Topology Test v3 — Effective rank + multi-measure + 1.5B

Key changes from v2:
1. EFFECTIVE RANK — exp(entropy of normalized singular values). Works well at
   small token counts. Lower effective rank = more hierarchical = more confident.
2. MULTIPLE MEASURES — effective rank, spectral gap, norm variance, AND Gini
   to find which signal separates truth from falsehood.
3. 1.5B MODEL — more capacity, more likely to show the signal.
4. QUESTION-ONLY BASELINE — measure topology BEFORE showing choices, compare
   to after. The DELTA is the model's response to the answer bank.

Protocol:
    For each MCQ question:
    1. Forward pass on full MCQ prompt (question + all choices + "Answer:")
    2. At each layer, for each choice's token span, compute:
       - Effective rank (spectral hierarchy)
       - Spectral gap (dominant mode strength)
       - Hidden state norm variance (activation pattern)
       - H₀ Gini (topological hierarchy)
    3. Also compute these for the question-only region (control)
    4. Test: do any measures separate correct from wrong?
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
from scipy.stats import mannwhitneyu
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "results"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GPQA_PATH = Path("/home/wb1/Desktop/Dev/gpqa/gpqa_diamond.csv")
LABELS = ["A", "B", "C", "D"]


# ── Topology measures ───────────────────────────────────────────────────────

def effective_rank(hidden_state: torch.Tensor) -> float:
    """Effective rank: exp(entropy of normalized singular values).

    Lower = more hierarchical (first few modes dominate).
    Higher = more democratic (all modes contribute equally).
    Range: [1, min(seq_len, hidden_dim)].
    """
    h = hidden_state.cpu().float()
    if h.shape[0] < 2:
        return 1.0
    # Center
    h = h - h.mean(0)
    try:
        s = torch.linalg.svdvals(h)
    except Exception:
        return 1.0
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    # Normalize to probability distribution
    p = s / s.sum()
    # Shannon entropy
    entropy = -torch.sum(p * torch.log(p)).item()
    return float(np.exp(entropy))


def spectral_gap(hidden_state: torch.Tensor) -> float:
    """Ratio of first to second singular value.

    Higher = more dominated by a single direction = more hierarchical.
    """
    h = hidden_state.cpu().float()
    if h.shape[0] < 2:
        return 0.0
    h = h - h.mean(0)
    try:
        s = torch.linalg.svdvals(h)
    except Exception:
        return 0.0
    if len(s) < 2 or s[1] < 1e-10:
        return float(s[0]) if len(s) > 0 else 0.0
    return float(s[0] / s[1])


def norm_variance(hidden_state: torch.Tensor) -> float:
    """Variance of L2 norms across tokens.

    Higher variance = more differentiated token representations.
    """
    h = hidden_state.cpu().float()
    if h.shape[0] < 2:
        return 0.0
    norms = torch.norm(h, dim=1)
    return float(torch.var(norms).item())


def gini_h0(hidden_state: torch.Tensor, pca_dim: int = 30) -> float:
    """Gini of H₀ persistence (original measure)."""
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

    n = len(h_r)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts = torch.tensor(h_r, dtype=torch.float32, device=device)
    dists = torch.cdist(pts, pts)
    mask_t = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    flat = dists[mask_t]
    sorted_d, sorted_idx = torch.sort(flat)
    rows, cols = torch.where(mask_t)
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
            if rank[ri] < rank[rj]: ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]: rank[ri] += 1
            bars.append(float(sd[k]))

    if not bars:
        return 0.0
    bars = np.array(bars)
    nn = len(bars)
    if nn <= 1 or np.sum(bars) == 0:
        return 0.0
    sv = np.sort(bars)
    idx_arr = np.arange(1, nn + 1, dtype=np.float64)
    return float((2 * np.sum(idx_arr * sv)) / (nn * np.sum(sv)) - (nn + 1) / nn)


MEASURES = {
    "eff_rank": effective_rank,
    "spectral_gap": spectral_gap,
    "norm_var": norm_variance,
    "gini_h0": gini_h0,
}


# ── Data loading ────────────────────────────────────────────────────────────

@dataclass
class MCQuestion:
    question: str
    choices: list[str]
    correct_idx: int
    source: str
    domain: str


def load_arc_challenge(n_max: int = 0) -> list[MCQuestion]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    questions = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
    for row in ds:
        choices = row["choices"]["text"]
        key = row["answerKey"]
        if key not in label_map: continue
        idx = label_map[key]
        if idx >= len(choices): continue
        questions.append(MCQuestion(row["question"], choices, idx, "ARC-Challenge", "science"))
    return questions[:n_max] if n_max > 0 else questions


def load_mmlu_physics(n_max: int = 0) -> list[MCQuestion]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "college_physics", split="test")
    questions = []
    for row in ds:
        questions.append(MCQuestion(row["question"], row["choices"],
                                    int(row["answer"]), "MMLU-Physics", "physics"))
    return questions[:n_max] if n_max > 0 else questions


def load_gpqa_diamond(n_max: int = 0) -> list[MCQuestion]:
    if not GPQA_PATH.exists():
        raise FileNotFoundError(f"GPQA not found at {GPQA_PATH}")
    questions = []
    with open(GPQA_PATH, "r") as f:
        for row in csv.DictReader(f):
            choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                       row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            if not all(c.strip() for c in choices): continue
            questions.append(MCQuestion(
                row["Question"], choices, 0, "GPQA-Diamond",
                row.get("High-level domain", "physics")))
    rng = np.random.default_rng(42)
    for q in questions:
        perm = rng.permutation(len(q.choices))
        q.choices = [q.choices[int(i)] for i in perm]
        q.correct_idx = int(np.where(perm == 0)[0][0])
    return questions[:n_max] if n_max > 0 else questions


def format_mcq(q: MCQuestion) -> str:
    lines = [f"Question: {q.question}", ""]
    for i, c in enumerate(q.choices):
        lines.append(f"{LABELS[i]}) {c}")
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def find_choice_spans(tokenizer, text: str, choices: list[str]) -> list[tuple[int, int]]:
    spans = []
    for i, c in enumerate(choices):
        marker = f"{LABELS[i]}) {c}"
        pos = text.find(marker)
        if pos < 0:
            spans.append((0, 0))
            continue
        pre = tokenizer(text[:pos], return_tensors="pt")["input_ids"][0]
        pre_c = tokenizer(text[:pos + len(marker)], return_tensors="pt")["input_ids"][0]
        spans.append((len(pre), len(pre_c)))
    return spans


# ── Probe ───────────────────────────────────────────────────────────────────

class TopologyProbe:
    def __init__(self, model_name: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
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
        n_p = sum(p.numel() for p in self.model.parameters()) / 1e6
        n_l = self.model.config.num_hidden_layers
        print(f"  {n_p:.0f}M params, {n_l} layers, {device}")

    @torch.no_grad()
    def measure_question(self, q: MCQuestion) -> dict:
        prompt = format_mcq(q)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=1024).to(self.device)
        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states)
        seq_len = inputs["input_ids"].shape[1]

        choice_spans = find_choice_spans(self.tokenizer, prompt, q.choices)

        # ── Per-choice multi-measure at each layer ──
        # Use mid-to-late layers (where reasoning happens): layer n//3 to n-2
        layer_start = max(1, n_layers // 3)
        layer_end = n_layers - 1  # skip output embedding layer
        target_layers = list(range(layer_start, layer_end))

        choice_measures = {}  # measure_name -> list of per-choice values (aggregated over layers)
        for mname in MEASURES:
            choice_measures[mname] = []

        for ci, (s, e) in enumerate(choice_spans):
            if e - s < 2:
                for mname in MEASURES:
                    choice_measures[mname].append(0.0)
                continue

            # Aggregate each measure across target layers
            per_measure_vals = {m: [] for m in MEASURES}
            for li in target_layers:
                hs_region = hidden_states[li][0, s:e, :]
                for mname, mfn in MEASURES.items():
                    per_measure_vals[mname].append(mfn(hs_region))

            for mname in MEASURES:
                # Use mean across target layers as the aggregate
                choice_measures[mname].append(float(np.mean(per_measure_vals[mname])))

        # ── Full deliberation measures (all tokens, target layers) ──
        full_measures = {}
        for mname, mfn in MEASURES.items():
            vals = [mfn(hidden_states[li][0]) for li in target_layers]
            full_measures[f"full_{mname}"] = float(np.mean(vals))

        # ── Decision-point measures (last 10 tokens) ──
        dp_start = max(0, seq_len - 10)
        decision_measures = {}
        for mname, mfn in MEASURES.items():
            vals = [mfn(hidden_states[li][0, dp_start:, :]) for li in target_layers]
            decision_measures[f"decision_{mname}"] = float(np.mean(vals))

        # ── Logit-based answer ──
        last_logits = outputs.logits[0, -1, :]
        choice_logits = []
        for label in LABELS[:len(q.choices)]:
            tids = self.tokenizer.encode(label, add_special_tokens=False)
            choice_logits.append(float(last_logits[tids[0]].item()) if tids else -1e9)

        logit_pred = int(np.argmax(choice_logits))

        # ── Topology-based predictions (one per measure) ──
        topo_preds = {}
        for mname in MEASURES:
            vals = choice_measures[mname]
            if mname == "eff_rank":
                # Lower effective rank = more hierarchical = more confident
                topo_preds[mname] = int(np.argmin(vals))
            else:
                # Higher = more hierarchical
                topo_preds[mname] = int(np.argmax(vals))

        result = {
            "question": q.question[:200],
            "source": q.source,
            "domain": q.domain,
            "n_choices": len(q.choices),
            "correct_idx": q.correct_idx,
            "seq_len": seq_len,
            "n_layers": n_layers,
            "target_layers": f"{layer_start}-{layer_end}",
            "choice_logits": choice_logits,
            "logit_pred": logit_pred,
            "logit_correct": logit_pred == q.correct_idx,
        }

        # Per-choice measures
        for mname in MEASURES:
            result[f"choice_{mname}"] = choice_measures[mname]
            result[f"correct_{mname}"] = choice_measures[mname][q.correct_idx]
            wrong_vals = [v for i, v in enumerate(choice_measures[mname]) if i != q.correct_idx]
            result[f"wrong_mean_{mname}"] = float(np.mean(wrong_vals)) if wrong_vals else 0
            result[f"topo_pred_{mname}"] = topo_preds[mname]
            result[f"topo_correct_{mname}"] = topo_preds[mname] == q.correct_idx

        # Full + decision
        result.update(full_measures)
        result.update(decision_measures)

        return result

    def cleanup(self):
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze(results: list[dict], model_name: str) -> dict:
    n = len(results)
    n_choices = results[0]["n_choices"] if results else 4
    random_base = 1.0 / n_choices

    analysis = {"model": model_name, "n": n, "n_choices": n_choices,
                "random_baseline": random_base}

    # Logit accuracy
    logit_acc = sum(1 for r in results if r["logit_correct"]) / n
    analysis["logit_accuracy"] = logit_acc

    # Per-measure analysis
    for mname in MEASURES:
        # Accuracy
        acc = sum(1 for r in results if r[f"topo_correct_{mname}"]) / n
        analysis[f"{mname}_accuracy"] = acc

        # Effect size: correct choice value vs wrong choice values
        correct_vals = [r[f"correct_{mname}"] for r in results]
        wrong_vals = []
        for r in results:
            wv = [v for i, v in enumerate(r[f"choice_{mname}"]) if i != r["correct_idx"]]
            wrong_vals.extend(wv)

        cv = np.array(correct_vals)
        wv = np.array(wrong_vals)

        if len(cv) > 5 and len(wv) > 5:
            # For eff_rank, correct should be LOWER (more hierarchical)
            if mname == "eff_rank":
                _, p = mannwhitneyu(cv, wv, alternative="less")
            else:
                _, p = mannwhitneyu(cv, wv, alternative="greater")

            pooled = np.sqrt((np.var(cv) + np.var(wv)) / 2)
            d = (np.mean(cv) - np.mean(wv)) / pooled if pooled > 0 else 0
        else:
            p, d = 1.0, 0.0

        analysis[f"{mname}_correct_mean"] = float(np.mean(cv))
        analysis[f"{mname}_wrong_mean"] = float(np.mean(wv))
        analysis[f"{mname}_cohens_d"] = float(d)
        analysis[f"{mname}_p"] = float(p)

    # Deliberation quality: does full topology predict correctness?
    for mname in MEASURES:
        key = f"full_{mname}"
        right_vals = [r[key] for r in results if r["logit_correct"]]
        wrong_vals = [r[key] for r in results if not r["logit_correct"]]
        if len(right_vals) > 3 and len(wrong_vals) > 3:
            rv, wv2 = np.array(right_vals), np.array(wrong_vals)
            pooled = np.sqrt((np.var(rv) + np.var(wv2)) / 2)
            d2 = (np.mean(rv) - np.mean(wv2)) / pooled if pooled > 0 else 0
            analysis[f"delib_{mname}_d"] = float(d2)
        else:
            analysis[f"delib_{mname}_d"] = 0.0

    # Ensemble: vote across all measures + logit
    ensemble_correct = 0
    for r in results:
        votes = [r["logit_pred"]]
        for mname in MEASURES:
            votes.append(r[f"topo_pred_{mname}"])
        # Majority vote
        from collections import Counter
        pred = Counter(votes).most_common(1)[0][0]
        if pred == r["correct_idx"]:
            ensemble_correct += 1
    analysis["ensemble_accuracy"] = ensemble_correct / n

    return analysis


def print_analysis(analysis: dict, dataset: str):
    print(f"\n  {'='*60}")
    print(f"  {dataset} x {analysis['model'].split('/')[-1]}")
    print(f"  {'='*60}")

    print(f"\n  ACCURACY (random = {analysis['random_baseline']:.1%}):")
    print(f"    {'Logit':<16} {analysis['logit_accuracy']:>6.1%}")
    for mname in MEASURES:
        acc = analysis[f"{mname}_accuracy"]
        vs = acc - analysis["random_baseline"]
        print(f"    {mname:<16} {acc:>6.1%} ({vs:+.1%})")
    print(f"    {'ENSEMBLE':<16} {analysis['ensemble_accuracy']:>6.1%}")

    print(f"\n  PER-CHOICE SEPARATION (correct vs wrong):")
    print(f"    {'Measure':<16} {'Correct':>10} {'Wrong':>10} {'d':>8} {'p':>12}")
    for mname in MEASURES:
        c = analysis[f"{mname}_correct_mean"]
        w = analysis[f"{mname}_wrong_mean"]
        d = analysis[f"{mname}_cohens_d"]
        p = analysis[f"{mname}_p"]
        marker = " ***" if p < 0.01 else " **" if p < 0.05 else " *" if p < 0.1 else ""
        print(f"    {mname:<16} {c:>10.4f} {w:>10.4f} {d:>8.3f} {p:>12.2e}{marker}")

    print(f"\n  DELIBERATION QUALITY (model right vs wrong):")
    for mname in MEASURES:
        d = analysis[f"delib_{mname}_d"]
        print(f"    {mname:<16} d = {d:>8.3f}")

    # Best measure
    best_acc = max(analysis[f"{m}_accuracy"] for m in MEASURES)
    best_m = [m for m in MEASURES if analysis[f"{m}_accuracy"] == best_acc][0]
    best_d = max(abs(analysis[f"{m}_cohens_d"]) for m in MEASURES)
    best_md = [m for m in MEASURES if abs(analysis[f"{m}_cohens_d"]) == best_d][0]
    print(f"\n  BEST ACCURACY: {best_m} ({best_acc:.1%})")
    print(f"  BEST EFFECT:   {best_md} (|d|={best_d:.3f})")


def plot_multi_measure(results: list[dict], analysis: dict, dataset: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"gold": "#c5a03f", "teal": "#45a8b0", "red": "#e94560",
              "bg": "#0f0d08", "text": "#d6d0be", "muted": "#817a66", "green": "#4caf50"}
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.facecolor": COLORS["bg"], "figure.facecolor": COLORS["bg"],
        "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
        "axes.edgecolor": COLORS["muted"], "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.dpi": 200,
    })

    model_short = analysis["model"].split("/")[-1]
    measures = list(MEASURES.keys())
    n_m = len(measures)

    fig, axes = plt.subplots(2, n_m, figsize=(5 * n_m, 10))

    for mi, mname in enumerate(measures):
        # Row 1: Correct vs Wrong distributions
        ax = axes[0, mi]
        correct_vals = [r[f"correct_{mname}"] for r in results]
        wrong_vals = []
        for r in results:
            wrong_vals.extend([v for i, v in enumerate(r[f"choice_{mname}"])
                               if i != r["correct_idx"]])

        ax.hist(correct_vals, bins=25, alpha=0.7, color=COLORS["teal"],
                label="Correct", density=True)
        ax.hist(wrong_vals, bins=25, alpha=0.7, color=COLORS["red"],
                label="Wrong", density=True)
        ax.axvline(np.mean(correct_vals), color=COLORS["teal"], ls="--", lw=2)
        ax.axvline(np.mean(wrong_vals), color=COLORS["red"], ls="--", lw=2)
        d = analysis[f"{mname}_cohens_d"]
        p = analysis[f"{mname}_p"]
        ax.set_title(f"{mname}\n(d={d:.3f}, p={p:.2e})", color=COLORS["gold"])
        ax.set_xlabel(mname)
        if mi == 0:
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

        # Row 2: Per-question separation
        ax = axes[1, mi]
        for i, r in enumerate(results):
            cv = r[f"correct_{mname}"]
            wm = r[f"wrong_mean_{mname}"]
            diff = cv - wm
            color = COLORS["teal"] if (diff > 0 if mname != "eff_rank" else diff < 0) else COLORS["red"]
            ax.scatter(i, diff, c=color, s=8, alpha=0.5)
        ax.axhline(0, color=COLORS["muted"], ls="--", alpha=0.5)
        acc = analysis[f"{mname}_accuracy"]
        ax.set_title(f"Per-Q separation (acc={acc:.1%})", color=COLORS["gold"])
        ax.set_xlabel("Question")
        if mi == 0:
            ax.set_ylabel("Correct - Wrong")
        ax.grid(True, alpha=0.15)

    fig.suptitle(f"Multi-Measure Answer Topology -- {dataset} x {model_short}",
                 color=COLORS["text"], fontsize=14)
    fig.tight_layout()
    fname = f"answer_topo_v3_{dataset.lower().replace('-','_')}_{model_short}.png"
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
    print(f"  Figure: {FIG_DIR / fname}")


# ── Pipeline ────────────────────────────────────────────────────────────────

def run_tier(probe, questions, dataset, model_name):
    print(f"\n{'='*60}")
    print(f"  {dataset} x {model_name.split('/')[-1]} ({len(questions)} questions)")
    print(f"{'='*60}")

    results = []
    t0 = time.time()
    for i, q in enumerate(questions):
        r = probe.measure_question(q)
        results.append(r)
        if (i + 1) % 20 == 0 or (i + 1) == len(questions):
            elapsed = time.time() - t0
            logit_acc = sum(1 for r in results if r["logit_correct"]) / len(results)
            best_topo = max(
                sum(1 for r in results if r[f"topo_correct_{m}"]) / len(results)
                for m in MEASURES
            )
            print(f"  [{i+1}/{len(questions)}] logit={logit_acc:.1%} "
                  f"best_topo={best_topo:.1%} elapsed={elapsed:.0f}s")

    total = time.time() - t0
    analysis = analyze(results, model_name)
    analysis["dataset"] = dataset
    analysis["time_s"] = round(total, 1)
    print_analysis(analysis, dataset)

    plot_multi_measure(results, analysis, dataset)

    fname = f"answer_topo_v3_{dataset.lower().replace('-','_')}_{model_name.split('/')[-1]}.json"
    with open(OUTPUT_DIR / fname, "w") as f:
        json.dump({"analysis": analysis, "results": results}, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR / fname}")

    return analysis


def main():
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 60)
    print("  ANSWER-TOPOLOGY TEST v3")
    print("  Multi-measure: eff_rank, spectral_gap, norm_var, gini_h0")
    print(f"  {ts}")
    print("=" * 60)

    # Models to test
    models = [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
    ]

    # Load datasets
    print("\n  Loading datasets...")
    tiers = []
    arc = load_arc_challenge(n_max=100)
    tiers.append(("ARC-Challenge", arc))
    print(f"  ARC-Challenge: {len(arc)}")

    mmlu = load_mmlu_physics()
    tiers.append(("MMLU-Physics", mmlu))
    print(f"  MMLU-Physics: {len(mmlu)}")

    try:
        gpqa = load_gpqa_diamond()
        tiers.append(("GPQA-Diamond", gpqa))
        print(f"  GPQA-Diamond: {len(gpqa)}")
    except FileNotFoundError as e:
        print(f"  GPQA: {e}")

    all_analyses = []
    for model_name in models:
        probe = TopologyProbe(model_name)
        for dname, qs in tiers:
            a = run_tier(probe, qs, dname, model_name)
            all_analyses.append(a)
        probe.cleanup()

    # Final summary
    print(f"\n{'='*80}")
    print("  FINAL CROSS-TIER CROSS-MODEL SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Dataset':<16} {'Model':<16} {'Logit':>6} ", end="")
    for m in MEASURES:
        print(f"  {m:>10}", end="")
    print(f"  {'Ensemble':>8}")
    print(f"  {'-'*90}")
    for a in all_analyses:
        print(f"  {a['dataset']:<16} {a['model'].split('/')[-1]:<16} "
              f"{a['logit_accuracy']:>5.1%} ", end="")
        for m in MEASURES:
            print(f"  {a[f'{m}_accuracy']:>9.1%}", end="")
        print(f"  {a['ensemble_accuracy']:>7.1%}")

    print(f"\n  EFFECT SIZES (Cohen's d, per-choice correct vs wrong):")
    print(f"  {'Dataset':<16} {'Model':<16}", end="")
    for m in MEASURES:
        print(f"  {m:>10}", end="")
    print()
    print(f"  {'-'*80}")
    for a in all_analyses:
        print(f"  {a['dataset']:<16} {a['model'].split('/')[-1]:<16}", end="")
        for m in MEASURES:
            d = a[f"{m}_cohens_d"]
            print(f"  {d:>10.3f}", end="")
        print()

    with open(OUTPUT_DIR / "answer_topology_v3_summary.json", "w") as f:
        json.dump({"timestamp": ts, "analyses": all_analyses}, f, indent=2, default=str)
    print(f"\n  Summary: {OUTPUT_DIR / 'answer_topology_v3_summary.json'}")


if __name__ == "__main__":
    main()
