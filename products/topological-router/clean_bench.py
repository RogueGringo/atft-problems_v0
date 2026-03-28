#!/usr/bin/env python3
"""Clean Test Bench — Likelihood (no letters) vs MCQ (rotated letters).

TWO EVALUATION MODES, SIDE BY SIDE:

MODE 1: LIKELIHOOD (no letters, no position, no format)
  For each answer candidate:
    text = "Q: {question}\nA: {answer_text}"
    score = mean log-prob of answer tokens given the question
  Selection = argmax(score)
  Topology = hidden state measures during scoring

MODE 2: MCQ with rotation (letters, position-controlled)
  Each question tested 4 times with correct answer at A, B, C, D.
  Selection = argmax(logit over letter tokens)
  Position-averaged accuracy eliminates bias.

CONTROLS:
  - Null baseline: scrambled Q/A pairs
  - Answer length tracking
  - Deliberation quality: right vs wrong comparison

If Mode 1 and Mode 2 agree on which questions the model "knows":
  the signal is about KNOWLEDGE, not FORMAT.
If they disagree:
  the signal is about FORMAT INTERACTION, not knowledge.
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
from scipy.stats import mannwhitneyu, pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "results"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GPQA_PATH = Path("/home/wb1/Desktop/Dev/gpqa/gpqa_diamond.csv")
LABELS = ["A", "B", "C", "D"]


# ── Measures ────────────────────────────────────────────────────────────────

def effective_rank(hs):
    h = hs.cpu().float()
    if h.shape[0] < 2: return 1.0
    h = h - h.mean(0)
    try: s = torch.linalg.svdvals(h)
    except Exception: return 1.0
    s = s[s > 1e-10]
    if len(s) == 0: return 1.0
    p = s / s.sum()
    return float(np.exp(-torch.sum(p * torch.log(p)).item()))

def spectral_gap(hs):
    h = hs.cpu().float()
    if h.shape[0] < 2: return 0.0
    h = h - h.mean(0)
    try: s = torch.linalg.svdvals(h)
    except Exception: return 0.0
    if len(s) < 2 or s[1] < 1e-10: return float(s[0]) if len(s) > 0 else 0.0
    return float(s[0] / s[1])

def norm_variance(hs):
    h = hs.cpu().float()
    if h.shape[0] < 2: return 0.0
    return float(torch.var(torch.norm(h, dim=1)).item())

MEASURES = {"eff_rank": effective_rank, "spectral_gap": spectral_gap, "norm_var": norm_variance}


# ── Data ────────────────────────────────────────────────────────────────────

@dataclass
class MCQ:
    question: str
    choices: list[str]  # [0] = correct, [1:] = wrong
    source: str
    domain: str

def load_gpqa(n_max=0):
    qs = []
    with open(GPQA_PATH) as f:
        for r in csv.DictReader(f):
            ch = [r["Correct Answer"], r["Incorrect Answer 1"],
                  r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            if not all(c.strip() for c in ch): continue
            qs.append(MCQ(r["Question"], ch, "GPQA-Diamond",
                         r.get("High-level domain", "physics")))
    return qs[:n_max] if n_max > 0 else qs

def load_mmlu(n_max=0):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "college_physics", split="test")
    qs = []
    for r in ds:
        ci = int(r["answer"])
        ch = list(r["choices"])
        correct = ch[ci]
        others = [c for i, c in enumerate(ch) if i != ci]
        qs.append(MCQ(r["question"], [correct] + others, "MMLU-Physics", "physics"))
    return qs[:n_max] if n_max > 0 else qs


# ── Probe ───────────────────────────────────────────────────────────────────

class Probe:
    def __init__(self, model_name, device="auto"):
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        print(f"Loading {model_name}...")
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            dtype=torch.float16, device_map=device, output_hidden_states=True)
        self.model.eval()
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        np_ = sum(p.numel() for p in self.model.parameters()) / 1e6
        nl = self.model.config.num_hidden_layers
        print(f"  {np_:.0f}M params, {nl} layers, VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    def _target_layers(self, nl):
        return list(range(max(1, nl // 3), nl - 1))

    def _compute_measures(self, hs, target):
        return {mn: float(np.mean([mf(hs[li][0]) for li in target]))
                for mn, mf in MEASURES.items()}

    @torch.no_grad()
    def score_likelihood(self, question: str, answer: str) -> dict:
        """Score one (question, answer) pair by conditional log-prob. NO LETTERS."""
        prefix = f"Q: {question}\nA:"
        prefix_len = self.tok(prefix, return_tensors="pt")["input_ids"].shape[1]

        text = f"Q: {question}\nA: {answer}"
        inp = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=1024).to(self.device)
        out = self.model(input_ids=inp["input_ids"], output_hidden_states=True)

        total_len = inp["input_ids"].shape[1]
        answer_len = total_len - prefix_len

        # Conditional log-prob of answer tokens
        if answer_len > 0:
            score = -torch.nn.functional.cross_entropy(
                out.logits[0, prefix_len - 1:-1, :],
                inp["input_ids"][0, prefix_len:],
                reduction="mean"
            ).item()
        else:
            score = -1e9

        # Deliberation measures
        hs = out.hidden_states
        target = self._target_layers(len(hs))
        measures = self._compute_measures(hs, target)

        return {"score": score, "measures": measures, "answer_len": answer_len}

    @torch.no_grad()
    def score_mcq(self, question: str, choices: list[str], correct_pos: int) -> dict:
        """Score via MCQ letter selection. With specific answer positions."""
        prompt = f"Question: {question}\n\n"
        for i, c in enumerate(choices):
            prompt += f"{LABELS[i]}) {c}\n"
        prompt += "\nAnswer:"

        inp = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(self.device)
        out = self.model(input_ids=inp["input_ids"], output_hidden_states=True)

        # Letter logits
        last_logits = out.logits[0, -1, :]
        letter_scores = []
        for lb in LABELS[:len(choices)]:
            tids = self.tok.encode(lb, add_special_tokens=False)
            letter_scores.append(float(last_logits[tids[0]].item()) if tids else -1e9)

        pred = int(np.argmax(letter_scores))

        hs = out.hidden_states
        target = self._target_layers(len(hs))
        measures = self._compute_measures(hs, target)

        return {"pred": pred, "correct": pred == correct_pos,
                "letter_scores": letter_scores, "measures": measures}

    def cleanup(self):
        del self.model, self.tok; gc.collect(); torch.cuda.empty_cache()


# ── Test bench ──────────────────────────────────────────────────────────────

def run_bench(probe, questions, dataset):
    print(f"\n{'='*65}")
    print(f"  CLEAN BENCH: {dataset} x {probe.model_name.split('/')[-1]}")
    print(f"  {len(questions)} questions")
    print(f"{'='*65}")

    results = []
    t0 = time.time()

    for qi, q in enumerate(questions):
        correct_answer = q.choices[0]
        wrong_answers = q.choices[1:4]

        # ── MODE 1: Likelihood (no letters) ──
        lk_correct = probe.score_likelihood(q.question, correct_answer)
        lk_wrongs = [probe.score_likelihood(q.question, w) for w in wrong_answers]

        all_scores = [lk_correct["score"]] + [w["score"] for w in lk_wrongs]
        lk_pred = int(np.argmax(all_scores))  # 0 = correct
        lk_is_correct = lk_pred == 0

        # ── MODE 2: MCQ with 4 rotations ──
        mcq_results = []
        for rot in range(4):
            # Place correct answer at position `rot`
            rng = np.random.default_rng(abs(hash(q.question)) + rot)
            shuffled_wrong = list(wrong_answers)
            rng.shuffle(shuffled_wrong)
            rotated = shuffled_wrong[:rot] + [correct_answer] + shuffled_wrong[rot:]
            rotated = rotated[:4]

            mcq_r = probe.score_mcq(q.question, rotated, rot)
            mcq_results.append(mcq_r)

        mcq_n_correct = sum(1 for r in mcq_results if r["correct"])
        mcq_avg_acc = mcq_n_correct / 4

        # Deliberation measures (averaged over rotations for MCQ)
        mcq_sg = np.mean([r["measures"]["spectral_gap"] for r in mcq_results])

        results.append({
            "qi": qi,
            "question": q.question[:200],
            "domain": q.domain,
            # Likelihood mode
            "lk_correct": lk_is_correct,
            "lk_scores": all_scores,
            "lk_correct_score": lk_correct["score"],
            "lk_wrong_scores": [w["score"] for w in lk_wrongs],
            "lk_measures": lk_correct["measures"],
            "lk_answer_len": lk_correct["answer_len"],
            # MCQ mode (position-averaged)
            "mcq_avg_acc": mcq_avg_acc,
            "mcq_n_correct": mcq_n_correct,
            "mcq_sg_mean": float(mcq_sg),
            "mcq_sg_cv": float(np.std([r["measures"]["spectral_gap"]
                                        for r in mcq_results]) / mcq_sg) if mcq_sg > 0 else 0,
        })

        if (qi + 1) % 20 == 0 or (qi + 1) == len(questions):
            elapsed = time.time() - t0
            lk_acc = np.mean([r["lk_correct"] for r in results])
            mcq_acc = np.mean([r["mcq_avg_acc"] for r in results])
            print(f"  [{qi+1}/{len(questions)}] "
                  f"likelihood={lk_acc:.1%} mcq_pos_avg={mcq_acc:.1%} "
                  f"elapsed={elapsed:.0f}s")

    total_time = time.time() - t0

    # ── Null baseline ──
    print(f"  Running null baseline...")
    rng = np.random.default_rng(99)
    null_scores = []
    for i in range(min(50, len(questions))):
        j = (i + rng.integers(1, len(questions))) % len(questions)
        # Score wrong question's answer against this question
        s = probe.score_likelihood(questions[i].question, questions[j].choices[0])
        null_scores.append(s["measures"]["spectral_gap"])

    # ── Analysis ──
    analysis = analyze_bench(results, null_scores, probe.model_name, dataset, total_time)
    print_bench(analysis)
    plot_bench_fig(results, null_scores, analysis, dataset)

    # Save
    fname = f"clean_bench_{dataset.lower().replace('-','_')}_{probe.model_name.split('/')[-1]}.json"
    with open(OUTPUT_DIR / fname, "w") as f:
        json.dump({"analysis": analysis, "results": results}, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR / fname}")

    return analysis


def analyze_bench(results, null_sgs, model_name, dataset, total_time):
    n = len(results)

    lk_acc = np.mean([r["lk_correct"] for r in results])
    mcq_acc = np.mean([r["mcq_avg_acc"] for r in results])
    mcq_strict = np.mean([1 if r["mcq_n_correct"] == 4 else 0 for r in results])

    # Agreement: do both modes agree on which questions the model knows?
    both_right = sum(1 for r in results if r["lk_correct"] and r["mcq_n_correct"] >= 3)
    both_wrong = sum(1 for r in results if not r["lk_correct"] and r["mcq_n_correct"] <= 1)
    lk_only = sum(1 for r in results if r["lk_correct"] and r["mcq_n_correct"] <= 1)
    mcq_only = sum(1 for r in results if not r["lk_correct"] and r["mcq_n_correct"] >= 3)
    agreement = (both_right + both_wrong) / n

    # Deliberation signal — LIKELIHOOD MODE (no position confound at all)
    lk_right = [r for r in results if r["lk_correct"]]
    lk_wrong = [r for r in results if not r["lk_correct"]]

    delib = {}
    for mn in MEASURES:
        rv = np.array([r["lk_measures"][mn] for r in lk_right])
        wv = np.array([r["lk_measures"][mn] for r in lk_wrong])
        if len(rv) > 3 and len(wv) > 3:
            ps = np.sqrt((np.var(rv) + np.var(wv)) / 2)
            d = (np.mean(rv) - np.mean(wv)) / ps if ps > 0 else 0
            alt = "less" if mn == "eff_rank" else "greater"
            _, p = mannwhitneyu(rv, wv, alternative=alt)
            delib[mn] = {"d": float(d), "p": float(p),
                         "right_mean": float(np.mean(rv)),
                         "wrong_mean": float(np.mean(wv))}
        else:
            delib[mn] = {"d": 0, "p": 1}

    # MCQ deliberation (position-averaged, for comparison)
    mcq_right = [r for r in results if r["mcq_n_correct"] >= 3]
    mcq_wrong = [r for r in results if r["mcq_n_correct"] <= 1]
    delib_mcq = {}
    for mn in ["spectral_gap"]:
        rv = np.array([r["mcq_sg_mean"] for r in mcq_right])
        wv = np.array([r["mcq_sg_mean"] for r in mcq_wrong])
        if len(rv) > 3 and len(wv) > 3:
            ps = np.sqrt((np.var(rv) + np.var(wv)) / 2)
            d = (np.mean(rv) - np.mean(wv)) / ps if ps > 0 else 0
            _, p = mannwhitneyu(rv, wv, alternative="greater")
            delib_mcq[mn] = {"d": float(d), "p": float(p)}
        else:
            delib_mcq[mn] = {"d": 0, "p": 1}

    # Answer length confound
    lengths = [r["lk_answer_len"] for r in results]
    correct_flags = [1 if r["lk_correct"] else 0 for r in results]
    if len(lengths) > 10:
        len_r, len_p = pearsonr(lengths, correct_flags)
    else:
        len_r, len_p = 0, 1

    # MCQ rotation consistency
    cvs = [r["mcq_sg_cv"] for r in results]

    return {
        "model": model_name, "dataset": dataset, "n": n,
        "time_s": round(total_time, 1),
        # Accuracy
        "lk_accuracy": float(lk_acc),
        "mcq_pos_avg_accuracy": float(mcq_acc),
        "mcq_strict_accuracy": float(mcq_strict),
        "random_baseline": 0.25,
        # Agreement
        "both_right": both_right, "both_wrong": both_wrong,
        "lk_only": lk_only, "mcq_only": mcq_only,
        "agreement": float(agreement),
        # Deliberation (likelihood — no position confound)
        "lk_delib": delib,
        "n_lk_right": len(lk_right), "n_lk_wrong": len(lk_wrong),
        # Deliberation (MCQ, position-averaged)
        "mcq_delib": delib_mcq,
        "n_mcq_right": len(mcq_right), "n_mcq_wrong": len(mcq_wrong),
        # Confound checks
        "answer_length_r": float(len_r), "answer_length_p": float(len_p),
        "mcq_rotation_cv": float(np.mean(cvs)),
        # Null
        "null_sg_mean": float(np.mean(null_sgs)),
        "real_sg_mean": float(np.mean([r["lk_measures"]["spectral_gap"] for r in results])),
    }


def print_bench(a):
    print(f"\n{'='*65}")
    print(f"  {a['dataset']} x {a['model'].split('/')[-1]}")
    print(f"  n={a['n']}, {a['time_s']}s")
    print(f"{'='*65}")

    print(f"\n  ACCURACY:")
    print(f"    Likelihood (no letters):     {a['lk_accuracy']:.1%}")
    print(f"    MCQ (position-averaged):     {a['mcq_pos_avg_accuracy']:.1%}")
    print(f"    MCQ (strict, 4/4 rotations): {a['mcq_strict_accuracy']:.1%}")
    print(f"    Random baseline:             {a['random_baseline']:.1%}")

    print(f"\n  MODE AGREEMENT:")
    print(f"    Both right:        {a['both_right']:3d} ({a['both_right']/a['n']:.1%})")
    print(f"    Both wrong:        {a['both_wrong']:3d} ({a['both_wrong']/a['n']:.1%})")
    print(f"    Likelihood only:   {a['lk_only']:3d} ({a['lk_only']/a['n']:.1%})")
    print(f"    MCQ only:          {a['mcq_only']:3d} ({a['mcq_only']/a['n']:.1%})")
    print(f"    Agreement rate:    {a['agreement']:.1%}")

    print(f"\n  DELIBERATION (likelihood mode — NO position confound):")
    print(f"    n_right={a['n_lk_right']}, n_wrong={a['n_lk_wrong']}")
    print(f"    {'Measure':<14} {'d':>8} {'p':>12}")
    for mn in MEASURES:
        da = a["lk_delib"][mn]
        sig = " ***" if da["p"] < 0.01 else " **" if da["p"] < 0.05 else " *" if da["p"] < 0.1 else ""
        print(f"    {mn:<14} {da['d']:>8.3f} {da['p']:>12.2e}{sig}")

    print(f"\n  DELIBERATION (MCQ, position-averaged, for comparison):")
    for mn, da in a["mcq_delib"].items():
        sig = " ***" if da["p"] < 0.01 else " **" if da["p"] < 0.05 else " *" if da["p"] < 0.1 else ""
        print(f"    {mn:<14} {da['d']:>8.3f} {da['p']:>12.2e}{sig}")

    print(f"\n  CONFOUND CHECKS:")
    print(f"    Answer length vs correctness: r={a['answer_length_r']:.3f} (p={a['answer_length_p']:.3e})")
    print(f"    MCQ rotation consistency (CV): {a['mcq_rotation_cv']:.3f}")
    print(f"    Null baseline sg: {a['null_sg_mean']:.2f} vs real: {a['real_sg_mean']:.2f}")


def plot_bench_fig(results, null_sgs, analysis, dataset):
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Accuracy comparison
    ax = axes[0, 0]
    methods = ["Likelihood\n(no letters)", "MCQ\n(pos-avg)", "MCQ\n(strict)", "Random"]
    accs = [analysis["lk_accuracy"], analysis["mcq_pos_avg_accuracy"],
            analysis["mcq_strict_accuracy"], 0.25]
    colors = [C["teal"], C["gold"], C["gold"], C["muted"]]
    ax.bar(range(4), accs, color=colors, alpha=0.8)
    for i, a_ in enumerate(accs):
        ax.text(i, a_ + 0.01, f"{a_:.1%}", ha="center", fontsize=10, color=C["text"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Accuracy")
    ax.set_title("No-Letter vs Letter-Based Evaluation", color=C["gold"])
    ax.grid(True, alpha=0.15, axis="y")

    # 2. Mode agreement
    ax = axes[0, 1]
    cats = ["Both\nright", "Both\nwrong", "LK\nonly", "MCQ\nonly"]
    vals = [analysis["both_right"], analysis["both_wrong"],
            analysis["lk_only"], analysis["mcq_only"]]
    colors2 = [C["green"], C["red"], C["teal"], C["gold"]]
    ax.bar(range(4), vals, color=colors2, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=10, color=C["text"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(cats)
    ax.set_ylabel("Questions")
    ax.set_title(f"Mode Agreement ({analysis['agreement']:.0%})", color=C["gold"])
    ax.grid(True, alpha=0.15, axis="y")

    # 3. Deliberation: right vs wrong (likelihood mode)
    ax = axes[1, 0]
    right_sg = [r["lk_measures"]["spectral_gap"] for r in results if r["lk_correct"]]
    wrong_sg = [r["lk_measures"]["spectral_gap"] for r in results if not r["lk_correct"]]
    if right_sg:
        ax.hist(right_sg, bins=20, alpha=0.6, color=C["green"],
                label=f"Right (n={len(right_sg)})", density=True)
    if wrong_sg:
        ax.hist(wrong_sg, bins=20, alpha=0.6, color=C["red"],
                label=f"Wrong (n={len(wrong_sg)})", density=True)
    if null_sgs:
        ax.hist(null_sgs, bins=15, alpha=0.3, color=C["muted"],
                label=f"Null (n={len(null_sgs)})", density=True)
    da = analysis["lk_delib"]["spectral_gap"]
    ax.set_title(f"Likelihood Deliberation\n(d={da['d']:.3f}, p={da['p']:.2e})", color=C["gold"])
    ax.set_xlabel("Spectral Gap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    # 4. Answer length check
    ax = axes[1, 1]
    lengths = [r["lk_answer_len"] for r in results]
    sgs = [r["lk_measures"]["spectral_gap"] for r in results]
    colors3 = [C["green"] if r["lk_correct"] else C["red"] for r in results]
    ax.scatter(lengths, sgs, c=colors3, s=15, alpha=0.5)
    ax.set_xlabel("Correct Answer Token Length")
    ax.set_ylabel("Spectral Gap")
    r_val = analysis["answer_length_r"]
    ax.set_title(f"Length Confound (r={r_val:.3f})", color=C["gold"])
    ax.grid(True, alpha=0.15)

    fig.suptitle(f"Clean Test Bench -- {dataset} x {ms}",
                 color=C["text"], fontsize=13)
    fig.tight_layout()
    fname = f"clean_bench_{dataset.lower().replace('-','_')}_{ms}.png"
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
    print(f"  Figure: {FIG_DIR / fname}")


def main():
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 65)
    print("  CLEAN TEST BENCH")
    print("  Likelihood (no letters) vs MCQ (rotated letters)")
    print(f"  {ts}")
    print("=" * 65)

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    probe = Probe(model_name)

    mmlu = load_mmlu()
    gpqa = load_gpqa()
    print(f"  MMLU-Physics: {len(mmlu)}")
    print(f"  GPQA-Diamond: {len(gpqa)}")

    a_m = run_bench(probe, mmlu, "MMLU-Physics")
    a_g = run_bench(probe, gpqa, "GPQA-Diamond")

    probe.cleanup()

    with open(OUTPUT_DIR / "clean_bench_summary.json", "w") as f:
        json.dump({"timestamp": ts, "model": model_name,
                   "analyses": [a_m, a_g]}, f, indent=2, default=str)
    print(f"\n  Summary: {OUTPUT_DIR / 'clean_bench_summary.json'}")


if __name__ == "__main__":
    main()
