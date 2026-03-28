#!/usr/bin/env python3
"""Topology Test Bench — Position-controlled, confound-free MCQ evaluation.

Controls:
  1. POSITION ROTATION — Each question tested with correct answer at each
     of A/B/C/D. Results averaged over rotations. Eliminates position bias.
  2. NULL BASELINE — Scrambled questions (wrong answers from different questions).
     If topology still shows signal on scrambled, the measurement is broken.
  3. ANSWER LENGTH — Track token count per choice. Verify topology doesn't
     correlate with length.
  4. PROMPT FORMAT — Test multiple MCQ formats to verify format-independence.
  5. CONSISTENCY — Same question across rotations should have similar
     deliberation topology (question hasn't changed, only letter assignments).

Metrics reported:
  - Position-averaged logit accuracy (eliminates position bias)
  - Position-averaged deliberation topology (eliminates position effects)
  - Rotation consistency (how stable is deliberation across rotations?)
  - Null baseline (scrambled control)
  - Answer length correlation (confound check)
"""
from __future__ import annotations

import csv
import gc
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu, pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "results"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GPQA_PATH = Path("/home/wb1/Desktop/Dev/gpqa/gpqa_diamond.csv")
LABELS = ["A", "B", "C", "D"]


# ── Topology measures ───────────────────────────────────────────────────────

def effective_rank(hs: torch.Tensor) -> float:
    h = hs.cpu().float()
    if h.shape[0] < 2: return 1.0
    h = h - h.mean(0)
    try: s = torch.linalg.svdvals(h)
    except Exception: return 1.0
    s = s[s > 1e-10]
    if len(s) == 0: return 1.0
    p = s / s.sum()
    return float(np.exp(-torch.sum(p * torch.log(p)).item()))

def spectral_gap(hs: torch.Tensor) -> float:
    h = hs.cpu().float()
    if h.shape[0] < 2: return 0.0
    h = h - h.mean(0)
    try: s = torch.linalg.svdvals(h)
    except Exception: return 0.0
    if len(s) < 2 or s[1] < 1e-10: return float(s[0]) if len(s) > 0 else 0.0
    return float(s[0] / s[1])

def norm_variance(hs: torch.Tensor) -> float:
    h = hs.cpu().float()
    if h.shape[0] < 2: return 0.0
    return float(torch.var(torch.norm(h, dim=1)).item())

MEASURES = {"eff_rank": effective_rank, "spectral_gap": spectral_gap, "norm_var": norm_variance}


# ── Data ────────────────────────────────────────────────────────────────────

@dataclass
class MCQ:
    question: str
    choices: list[str]       # original order, choices[0] is always correct
    source: str
    domain: str


def load_gpqa(n_max=0) -> list[MCQ]:
    qs = []
    with open(GPQA_PATH) as f:
        for r in csv.DictReader(f):
            ch = [r["Correct Answer"], r["Incorrect Answer 1"],
                  r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            if not all(c.strip() for c in ch): continue
            qs.append(MCQ(r["Question"], ch, "GPQA-Diamond",
                         r.get("High-level domain", "physics")))
    return qs[:n_max] if n_max > 0 else qs


def load_mmlu(n_max=0) -> list[MCQ]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "college_physics", split="test")
    qs = []
    for r in ds:
        # Rearrange so choices[0] is always correct
        correct_idx = int(r["answer"])
        choices = list(r["choices"])
        correct = choices[correct_idx]
        others = [c for i, c in enumerate(choices) if i != correct_idx]
        qs.append(MCQ(r["question"], [correct] + others, "MMLU-Physics", "physics"))
    return qs[:n_max] if n_max > 0 else qs


# ── Prompt formats ──────────────────────────────────────────────────────────

PROMPT_FORMATS = {
    "standard": lambda q, labels, choices: (
        f"Question: {q}\n\n" +
        "\n".join(f"{l}) {c}" for l, c in zip(labels, choices)) +
        "\n\nAnswer:"
    ),
    "parens": lambda q, labels, choices: (
        f"Question: {q}\n\n" +
        "\n".join(f"({l}) {c}" for l, c in zip(labels, choices)) +
        "\n\nAnswer:"
    ),
    "numbered": lambda q, labels, choices: (
        f"Question: {q}\n\n" +
        "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices)) +
        "\n\nAnswer:"
    ),
}


# ── Rotation generator ──────────────────────────────────────────────────────

def generate_rotations(q: MCQ, n_rotations: int = 4) -> list[dict]:
    """Generate rotated versions of a question.

    Each rotation places the correct answer at a different position.
    Returns list of dicts with {choices, correct_idx, rotation_id}.
    """
    correct = q.choices[0]
    wrong = q.choices[1:]

    rotations = []
    for rot_id in range(n_rotations):
        # Place correct answer at position rot_id
        shuffled = list(wrong)
        # Deterministic shuffle of wrong answers for this rotation
        rng = np.random.default_rng(hash(q.question) + rot_id)
        rng.shuffle(shuffled)
        # Insert correct at position rot_id
        choices = shuffled[:rot_id] + [correct] + shuffled[rot_id:]
        rotations.append({
            "choices": choices[:4],  # ensure exactly 4
            "correct_idx": rot_id,
            "rotation_id": rot_id,
        })
    return rotations


def generate_scrambled(questions: list[MCQ], seed: int = 99) -> list[MCQ]:
    """Create null baseline: pair each question with answers from DIFFERENT questions."""
    rng = np.random.default_rng(seed)
    n = len(questions)
    scrambled = []
    for i in range(n):
        # Pick answers from a different question
        j = (i + rng.integers(1, n)) % n
        scrambled.append(MCQ(
            question=questions[i].question,
            choices=questions[j].choices,  # wrong question's answers
            source=questions[i].source + "-SCRAMBLED",
            domain=questions[i].domain,
        ))
    return scrambled


# ── Probe ───────────────────────────────────────────────────────────────────

class Probe:
    def __init__(self, model_name, device="auto", use_awq=False):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        print(f"Loading {model_name}...")
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if use_awq or "AWQ" in model_name:
            from awq import AutoAWQForCausalLM
            awq = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False)
            self.model = awq.model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                dtype=torch.float16, device_map=device,
                output_hidden_states=True)

        self.model.eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        np_ = sum(p.numel() for p in self.model.parameters()) / 1e6
        nl = self.model.config.num_hidden_layers
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  {np_:.0f}M params, {nl} layers, VRAM={vram:.1f}GB")

    @torch.no_grad()
    def measure_likelihood(self, question_text: str, choices: list[str],
                           correct_idx: int) -> dict:
        """Score each answer by conditional log-probability. NO LETTERS.

        For each candidate answer:
          text = "Q: {question}\nA: {answer}"
          score = mean log-prob of answer tokens given the question

        The answer with highest score is the model's selection.
        No A/B/C/D. No position. No format. Pure content evaluation.
        """
        q_prefix = f"Q: {question_text}\nA:"
        q_ids = self.tok(q_prefix, return_tensors="pt",
                         truncation=True, max_length=768)["input_ids"]
        q_len = q_ids.shape[1]

        choice_scores = []
        choice_measures = []
        choice_lengths = []

        for ci, answer in enumerate(choices):
            text = f"Q: {question_text}\nA: {answer}"
            inp = self.tok(text, return_tensors="pt",
                           truncation=True, max_length=1024).to(self.device)
            out = self.model(input_ids=inp["input_ids"], output_hidden_states=True)
            hs = out.hidden_states
            nl = len(hs)
            sl = inp["input_ids"].shape[1]
            answer_len = sl - q_len

            # Conditional log-prob of answer tokens
            if answer_len > 0:
                logprob = -torch.nn.functional.cross_entropy(
                    out.logits[0, q_len - 1:-1, :],
                    inp["input_ids"][0, q_len:],
                    reduction="mean"
                ).item()
            else:
                logprob = -1e9

            # Topology of full completion
            ls, le = max(1, nl // 3), nl - 1
            target = list(range(ls, le))
            meas = {}
            for mn, mf in MEASURES.items():
                meas[mn] = float(np.mean([mf(hs[li][0]) for li in target]))

            choice_scores.append(logprob)
            choice_measures.append(meas)
            choice_lengths.append(answer_len)

        # Selection: highest conditional log-prob
        likelihood_pred = int(np.argmax(choice_scores))

        return {
            "correct_idx": correct_idx,
            "likelihood_pred": likelihood_pred,
            "likelihood_correct": likelihood_pred == correct_idx,
            "choice_scores": choice_scores,
            "choice_measures": choice_measures,
            "choice_lengths": choice_lengths,
            "correct_score": choice_scores[correct_idx],
            "wrong_scores": [s for i, s in enumerate(choice_scores) if i != correct_idx],
            "correct_measures": choice_measures[correct_idx],
        }

    @torch.no_grad()
    def measure_mcq(self, question_text: str, choices: list[str],
                    correct_idx: int, prompt_format: str = "standard") -> dict:
        """Score via MCQ letter selection (for comparison)."""
        fmt_fn = PROMPT_FORMATS[prompt_format]

        if prompt_format == "numbered":
            labels = ["1", "2", "3", "4"]
        else:
            labels = LABELS[:len(choices)]

        prompt = fmt_fn(question_text, labels, choices)
        inp = self.tok(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(self.device)
        out = self.model(input_ids=inp["input_ids"], output_hidden_states=True)
        hs = out.hidden_states
        nl = len(hs)
        sl = inp["input_ids"].shape[1]

        ls, le = max(1, nl // 3), nl - 1
        target = list(range(ls, le))

        full_m = {}
        for mn, mf in MEASURES.items():
            full_m[mn] = float(np.mean([mf(hs[li][0]) for li in target]))

        last_logits = out.logits[0, -1, :]
        choice_logits = []
        for lb in labels[:len(choices)]:
            tids = self.tok.encode(lb, add_special_tokens=False)
            choice_logits.append(float(last_logits[tids[0]].item()) if tids else -1e9)

        logit_pred = int(np.argmax(choice_logits))
        answer_lengths = [len(self.tok.encode(c, add_special_tokens=False)) for c in choices]

        return {
            "correct_idx": correct_idx,
            "logit_pred": logit_pred,
            "logit_correct": logit_pred == correct_idx,
            "choice_logits": choice_logits,
            "full_measures": full_m,
            "seq_len": sl,
            "answer_lengths": answer_lengths,
            "prompt_format": prompt_format,
        }

    def cleanup(self):
        del self.model, self.tok
        gc.collect()
        torch.cuda.empty_cache()


# ── Test bench ──────────────────────────────────────────────────────────────

def run_test_bench(probe: Probe, questions: list[MCQ], dataset_name: str,
                   n_rotations: int = 4, test_formats: bool = True) -> dict:
    """Full position-controlled test bench.

    For each question:
    1. Generate n_rotations rotated versions (correct at each position)
    2. Measure each rotation
    3. Compute position-averaged metrics
    4. Check rotation consistency (deliberation should be stable across rotations)
    """
    print(f"\n{'='*65}")
    print(f"  TEST BENCH: {dataset_name} x {probe.model_name.split('/')[-1]}")
    print(f"  {len(questions)} questions x {n_rotations} rotations = {len(questions)*n_rotations} measurements")
    print(f"{'='*65}")

    results = []
    t0 = time.time()

    for qi, q in enumerate(questions):
        rotations = generate_rotations(q, n_rotations)
        q_results = []

        for rot in rotations:
            r = probe.measure(q.question, rot["choices"], rot["correct_idx"])
            r["question_idx"] = qi
            r["rotation_id"] = rot["rotation_id"]
            r["domain"] = q.domain
            q_results.append(r)

        # Position-averaged accuracy: correct if majority of rotations correct
        n_correct = sum(1 for r in q_results if r["logit_correct"])
        avg_correct = n_correct / n_rotations

        # Deliberation consistency: std of spectral_gap across rotations
        sg_values = [r["full_measures"]["spectral_gap"] for r in q_results]
        sg_mean = np.mean(sg_values)
        sg_std = np.std(sg_values)

        results.append({
            "question_idx": qi,
            "question": q.question[:200],
            "domain": q.domain,
            "rotation_results": q_results,
            "position_avg_correct": avg_correct,
            "n_rotations_correct": n_correct,
            "sg_mean": sg_mean,
            "sg_std": sg_std,
            "sg_cv": sg_std / sg_mean if sg_mean > 0 else 0,  # coefficient of variation
        })

        if (qi + 1) % 20 == 0 or (qi + 1) == len(questions):
            elapsed = time.time() - t0
            avg_acc = np.mean([r["position_avg_correct"] for r in results])
            avg_cv = np.mean([r["sg_cv"] for r in results])
            print(f"  [{qi+1}/{len(questions)}] pos_avg_acc={avg_acc:.1%} "
                  f"sg_consistency(cv)={avg_cv:.3f} elapsed={elapsed:.0f}s")

    total_time = time.time() - t0

    # ── Null baseline: scrambled questions ──
    print(f"\n  Running null baseline (scrambled answers)...")
    scrambled = generate_scrambled(questions)
    null_results = []
    for qi, sq in enumerate(scrambled[:min(50, len(scrambled))]):
        # Single rotation for null (no need to rotate scrambled)
        r = probe.measure(sq.question, sq.choices, 0)  # correct_idx=0 is arbitrary
        null_results.append(r)
    null_sg = [r["full_measures"]["spectral_gap"] for r in null_results]

    # ── Format test (if enabled) ──
    format_results = {}
    if test_formats and len(questions) >= 20:
        print(f"  Testing prompt formats on first 20 questions...")
        for fmt_name in PROMPT_FORMATS:
            fmt_accs = []
            fmt_sgs = []
            for qi in range(min(20, len(questions))):
                q = questions[qi]
                # Use a fixed rotation (correct at position 0)
                r = probe.measure(q.question, q.choices, 0, prompt_format=fmt_name)
                fmt_accs.append(r["logit_correct"])
                fmt_sgs.append(r["full_measures"]["spectral_gap"])
            format_results[fmt_name] = {
                "accuracy": np.mean(fmt_accs),
                "sg_mean": np.mean(fmt_sgs),
            }

    # ── Analysis ──
    analysis = compute_analysis(results, null_results, format_results,
                                probe.model_name, dataset_name, total_time)

    print_bench_results(analysis)
    plot_bench(results, null_results, analysis, dataset_name)

    return analysis, results, null_results


def compute_analysis(results, null_results, format_results,
                     model_name, dataset, total_time) -> dict:
    """Compute position-controlled analysis."""
    n = len(results)

    # Position-averaged accuracy
    pos_avg_acc = np.mean([r["position_avg_correct"] for r in results])

    # Strict accuracy: correct on ALL rotations
    strict_acc = np.mean([1 if r["n_rotations_correct"] == 4 else 0 for r in results])

    # Majority accuracy: correct on majority of rotations
    majority_acc = np.mean([1 if r["n_rotations_correct"] >= 3 else 0 for r in results])

    # Rotation consistency
    cvs = [r["sg_cv"] for r in results]
    mean_cv = np.mean(cvs)

    # Deliberation signal (position-averaged)
    # Split by whether model gets it right on majority of rotations
    right_qs = [r for r in results if r["n_rotations_correct"] >= 3]
    wrong_qs = [r for r in results if r["n_rotations_correct"] <= 1]
    mixed_qs = [r for r in results if r["n_rotations_correct"] == 2]

    delib_analysis = {}
    for mn in MEASURES:
        right_vals = [r["rotation_results"][0]["full_measures"][mn] for r in right_qs]
        # Use mean across rotations for more stable estimate
        right_vals = [np.mean([rot["full_measures"][mn] for rot in r["rotation_results"]]) for r in right_qs]
        wrong_vals = [np.mean([rot["full_measures"][mn] for rot in r["rotation_results"]]) for r in wrong_qs]

        if len(right_vals) > 3 and len(wrong_vals) > 3:
            rv, wv = np.array(right_vals), np.array(wrong_vals)
            ps = np.sqrt((np.var(rv) + np.var(wv)) / 2)
            d = (np.mean(rv) - np.mean(wv)) / ps if ps > 0 else 0
            alt = "less" if mn == "eff_rank" else "greater"
            _, p = mannwhitneyu(rv, wv, alternative=alt)
            delib_analysis[mn] = {"d": float(d), "p": float(p),
                                  "right_mean": float(np.mean(rv)),
                                  "wrong_mean": float(np.mean(wv)),
                                  "n_right": len(right_vals), "n_wrong": len(wrong_vals)}
        else:
            delib_analysis[mn] = {"d": 0, "p": 1, "n_right": len(right_vals),
                                  "n_wrong": len(wrong_vals)}

    # Answer length confound check
    all_rotations = [rot for r in results for rot in r["rotation_results"]]
    correct_lengths = [rot["answer_lengths"][rot["correct_idx"]] for rot in all_rotations]
    correct_sgs = [rot["full_measures"]["spectral_gap"] for rot in all_rotations]
    if len(correct_lengths) > 10:
        len_corr, len_p = pearsonr(correct_lengths, correct_sgs)
    else:
        len_corr, len_p = 0, 1

    # Null baseline
    real_sgs = [r["sg_mean"] for r in results]
    null_sgs = [r["full_measures"]["spectral_gap"] for r in null_results]

    a = {
        "model": model_name,
        "dataset": dataset,
        "n_questions": n,
        "n_rotations": 4,
        "total_measurements": n * 4,
        "time_s": round(total_time, 1),

        # Position-averaged accuracy
        "pos_avg_accuracy": float(pos_avg_acc),
        "strict_accuracy": float(strict_acc),
        "majority_accuracy": float(majority_acc),
        "random_baseline": 0.25,

        # Rotation consistency
        "mean_sg_cv": float(mean_cv),
        "median_sg_cv": float(np.median(cvs)),

        # Deliberation signal (position-controlled)
        "deliberation": delib_analysis,
        "n_right_qs": len(right_qs),
        "n_wrong_qs": len(wrong_qs),
        "n_mixed_qs": len(mixed_qs),

        # Answer length confound
        "answer_length_correlation": float(len_corr),
        "answer_length_p": float(len_p),

        # Null baseline
        "real_sg_mean": float(np.mean(real_sgs)),
        "null_sg_mean": float(np.mean(null_sgs)),
        "null_sg_std": float(np.std(null_sgs)),

        # Format sensitivity
        "format_results": format_results,
    }
    return a


def print_bench_results(a: dict):
    print(f"\n{'='*65}")
    print(f"  TEST BENCH RESULTS: {a['dataset']} x {a['model'].split('/')[-1]}")
    print(f"  {a['n_questions']} questions x {a['n_rotations']} rotations = {a['total_measurements']} measurements")
    print(f"{'='*65}")

    print(f"\n  POSITION-AVERAGED ACCURACY (eliminates position bias):")
    print(f"    Position-averaged: {a['pos_avg_accuracy']:.1%}")
    print(f"    Majority (>=3/4):  {a['majority_accuracy']:.1%}")
    print(f"    Strict (4/4):      {a['strict_accuracy']:.1%}")
    print(f"    Random baseline:   {a['random_baseline']:.1%}")

    print(f"\n  ROTATION CONSISTENCY (should be low = stable):")
    print(f"    Mean CV of spectral_gap: {a['mean_sg_cv']:.3f}")
    print(f"    Median CV:               {a['median_sg_cv']:.3f}")
    print(f"    (CV < 0.1 = very consistent, CV > 0.3 = unstable)")

    print(f"\n  DELIBERATION SIGNAL (position-controlled, majority-split):")
    print(f"    Right questions: {a['n_right_qs']}, Wrong: {a['n_wrong_qs']}, Mixed: {a['n_mixed_qs']}")
    print(f"    {'Measure':<14} {'d':>8} {'p':>12} {'Right':>10} {'Wrong':>10}")
    for mn in MEASURES:
        da = a["deliberation"][mn]
        sig = " ***" if da["p"] < 0.01 else " **" if da["p"] < 0.05 else " *" if da["p"] < 0.1 else ""
        print(f"    {mn:<14} {da['d']:>8.3f} {da['p']:>12.2e} "
              f"{da.get('right_mean',0):>10.2f} {da.get('wrong_mean',0):>10.2f}{sig}")

    print(f"\n  CONFOUND CHECKS:")
    print(f"    Answer length vs topology: r={a['answer_length_correlation']:.3f} (p={a['answer_length_p']:.3e})")
    if abs(a["answer_length_correlation"]) > 0.2 and a["answer_length_p"] < 0.05:
        print(f"    WARNING: Answer length confound detected!")
    else:
        print(f"    Clean — answer length does not confound topology.")

    print(f"    Null baseline (scrambled): sg={a['null_sg_mean']:.2f} +/- {a['null_sg_std']:.2f}")
    print(f"    Real data:                sg={a['real_sg_mean']:.2f}")

    if a["format_results"]:
        print(f"\n  FORMAT SENSITIVITY (first 20 questions):")
        for fmt, fr in a["format_results"].items():
            print(f"    {fmt:12s}: acc={fr['accuracy']:.1%}  sg={fr['sg_mean']:.2f}")


def plot_bench(results, null_results, analysis, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"gold": "#c5a03f", "teal": "#45a8b0", "red": "#e94560",
         "bg": "#0f0d08", "text": "#d6d0be", "muted": "#817a66", "green": "#4caf50"}
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.facecolor": C["bg"], "figure.facecolor": C["bg"],
        "text.color": C["text"], "axes.labelcolor": C["text"],
        "xtick.color": C["muted"], "ytick.color": C["muted"],
        "axes.edgecolor": C["muted"], "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.dpi": 200,
    })

    ms = analysis["model"].split("/")[-1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Position-averaged accuracy by rotation
    ax = axes[0, 0]
    for rot_id in range(4):
        accs = []
        for r in results:
            rr = r["rotation_results"][rot_id]
            accs.append(rr["logit_correct"])
        ax.bar(rot_id, np.mean(accs), color=C["teal"], alpha=0.7)
        ax.text(rot_id, np.mean(accs) + 0.01, f"{np.mean(accs):.1%}",
                ha="center", fontsize=9, color=C["text"])
    ax.axhline(0.25, color=C["red"], ls="--", alpha=0.5, label="Random")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Correct@A", "Correct@B", "Correct@C", "Correct@D"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Correct Answer Position", color=C["gold"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15, axis="y")

    # 2. Rotation consistency
    ax = axes[0, 1]
    cvs = [r["sg_cv"] for r in results]
    ax.hist(cvs, bins=25, color=C["teal"], alpha=0.7)
    ax.axvline(np.mean(cvs), color=C["gold"], ls="--", lw=2,
               label=f"Mean CV={np.mean(cvs):.3f}")
    ax.set_xlabel("Coefficient of Variation (spectral_gap across rotations)")
    ax.set_ylabel("Count")
    ax.set_title("Rotation Consistency", color=C["gold"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # 3. Deliberation: right vs wrong vs null
    ax = axes[0, 2]
    right = [r["sg_mean"] for r in results if r["n_rotations_correct"] >= 3]
    wrong = [r["sg_mean"] for r in results if r["n_rotations_correct"] <= 1]
    null = [r["full_measures"]["spectral_gap"] for r in null_results]
    if right: ax.hist(right, bins=20, alpha=0.6, color=C["green"], label=f"Right (n={len(right)})", density=True)
    if wrong: ax.hist(wrong, bins=20, alpha=0.6, color=C["red"], label=f"Wrong (n={len(wrong)})", density=True)
    if null: ax.hist(null, bins=20, alpha=0.4, color=C["muted"], label=f"Null (n={len(null)})", density=True)
    ax.set_xlabel("Mean Spectral Gap (across rotations)")
    ax.set_title("Deliberation: Right vs Wrong vs Null", color=C["gold"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # 4. Per-question position-averaged accuracy distribution
    ax = axes[1, 0]
    avg_accs = [r["position_avg_correct"] for r in results]
    counts = {0: 0, 0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0}
    for a_ in avg_accs:
        counts[round(a_ * 4) / 4] = counts.get(round(a_ * 4) / 4, 0) + 1
    ax.bar(range(5), [counts.get(k, 0) for k in [0, 0.25, 0.5, 0.75, 1.0]],
           color=C["teal"], alpha=0.7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["0/4", "1/4", "2/4", "3/4", "4/4"])
    ax.set_xlabel("Correct rotations out of 4")
    ax.set_ylabel("Questions")
    ax.set_title("Position-Averaged Accuracy Distribution", color=C["gold"])
    ax.grid(True, alpha=0.15, axis="y")

    # 5. Answer length vs topology
    ax = axes[1, 1]
    all_rots = [rot for r in results for rot in r["rotation_results"]]
    lengths = [rot["answer_lengths"][rot["correct_idx"]] for rot in all_rots]
    sgs = [rot["full_measures"]["spectral_gap"] for rot in all_rots]
    ax.scatter(lengths, sgs, s=5, alpha=0.3, color=C["teal"])
    r_val = analysis["answer_length_correlation"]
    ax.set_xlabel("Correct Answer Token Length")
    ax.set_ylabel("Spectral Gap")
    ax.set_title(f"Length Confound Check (r={r_val:.3f})", color=C["gold"])
    ax.grid(True, alpha=0.15)

    # 6. Deliberation consistency across rotations (first 20 questions)
    ax = axes[1, 2]
    for qi in range(min(20, len(results))):
        r = results[qi]
        sgs_q = [rot["full_measures"]["spectral_gap"] for rot in r["rotation_results"]]
        color = C["green"] if r["n_rotations_correct"] >= 3 else C["red"] if r["n_rotations_correct"] <= 1 else C["muted"]
        ax.plot(range(4), sgs_q, color=color, alpha=0.5, marker="o", ms=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["@A", "@B", "@C", "@D"])
    ax.set_xlabel("Correct Answer Position")
    ax.set_ylabel("Spectral Gap")
    ax.set_title("Deliberation Stability Across Rotations", color=C["gold"])
    ax.grid(True, alpha=0.15)

    fig.suptitle(f"Topology Test Bench -- {dataset} x {ms}\n"
                 f"pos_avg_acc={analysis['pos_avg_accuracy']:.1%}, "
                 f"delib_sg_d={analysis['deliberation']['spectral_gap']['d']:.3f}",
                 color=C["text"], fontsize=13)
    fig.tight_layout()
    fname = f"test_bench_{dataset.lower().replace('-','_')}_{ms}.png"
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
    print(f"  Figure: {FIG_DIR / fname}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 65)
    print("  TOPOLOGY TEST BENCH — Position-Controlled Evaluation")
    print(f"  {ts}")
    print("=" * 65)

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    probe = Probe(model_name)

    all_results = {}

    # MMLU-Physics (where signal is strongest)
    mmlu = load_mmlu()
    print(f"\n  MMLU-Physics: {len(mmlu)} questions")
    a_m, res_m, null_m = run_test_bench(probe, mmlu, "MMLU-Physics")
    all_results["mmlu"] = {"analysis": a_m, "results_count": len(res_m)}

    # GPQA Diamond
    gpqa = load_gpqa()
    print(f"\n  GPQA-Diamond: {len(gpqa)} questions")
    a_g, res_g, null_g = run_test_bench(probe, gpqa, "GPQA-Diamond")
    all_results["gpqa"] = {"analysis": a_g, "results_count": len(res_g)}

    probe.cleanup()

    # Save
    output = {
        "timestamp": ts,
        "model": model_name,
        "analyses": [a_m, a_g],
    }
    with open(OUTPUT_DIR / "test_bench_3b.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR / 'test_bench_3b.json'}")


if __name__ == "__main__":
    main()
