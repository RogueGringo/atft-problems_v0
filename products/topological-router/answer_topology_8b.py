#!/usr/bin/env python3
"""Answer-Topology Test — 8B model on GPQA Diamond.

Hypothesis: 8B is the threshold where GPQA knowledge begins to appear.
Partial knowledge is exactly the regime where deliberation topology
should show the strongest signal.

Uses Qwen2.5-7B-Instruct-AWQ (4-bit, ~5GB VRAM).
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
    h = hidden_state.cpu().float()
    if h.shape[0] < 2:
        return 1.0
    h = h - h.mean(0)
    try:
        s = torch.linalg.svdvals(h)
    except Exception:
        return 1.0
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    entropy = -torch.sum(p * torch.log(p)).item()
    return float(np.exp(entropy))


def spectral_gap(hidden_state: torch.Tensor) -> float:
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
    h = hidden_state.cpu().float()
    if h.shape[0] < 2:
        return 0.0
    norms = torch.norm(h, dim=1)
    return float(torch.var(norms).item())


MEASURES = {
    "eff_rank": effective_rank,
    "spectral_gap": spectral_gap,
    "norm_var": norm_variance,
}


# ── Data loading ────────────────────────────────────────────────────────────

@dataclass
class MCQuestion:
    question: str
    choices: list[str]
    correct_idx: int
    source: str
    domain: str


def load_gpqa_diamond(n_max: int = 0) -> list[MCQuestion]:
    if not GPQA_PATH.exists():
        raise FileNotFoundError(f"GPQA not found at {GPQA_PATH}")
    questions = []
    with open(GPQA_PATH, "r") as f:
        for row in csv.DictReader(f):
            choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                       row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            if not all(c.strip() for c in choices):
                continue
            questions.append(MCQuestion(
                row["Question"], choices, 0, "GPQA-Diamond",
                row.get("High-level domain", "physics")))
    rng = np.random.default_rng(42)
    for q in questions:
        perm = rng.permutation(len(q.choices))
        q.choices = [q.choices[int(i)] for i in perm]
        q.correct_idx = int(np.where(perm == 0)[0][0])
    return questions[:n_max] if n_max > 0 else questions


def load_mmlu_physics(n_max: int = 0) -> list[MCQuestion]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "college_physics", split="test")
    questions = []
    for row in ds:
        questions.append(MCQuestion(row["question"], row["choices"],
                                    int(row["answer"]), "MMLU-Physics", "physics"))
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
    def __init__(self, model_name: str, device: str = "auto", use_awq: bool = False):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if use_awq or "AWQ" in model_name:
            from awq import AutoAWQForCausalLM
            awq_model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False)
            self.model = awq_model.model  # underlying HF model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                device_map=device,
                output_hidden_states=True,
            )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_p = sum(p.numel() for p in self.model.parameters()) / 1e6
        n_l = self.model.config.num_hidden_layers
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"  {n_p:.0f}M params, {n_l} layers, {device}, VRAM={vram:.1f}GB")

    @torch.no_grad()
    def measure_question(self, q: MCQuestion) -> dict:
        prompt = format_mcq(q)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=1024).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states)
        seq_len = inputs["input_ids"].shape[1]

        choice_spans = find_choice_spans(self.tokenizer, prompt, q.choices)

        # Target layers: mid-to-late (where reasoning happens)
        layer_start = max(1, n_layers // 3)
        layer_end = n_layers - 1
        target_layers = list(range(layer_start, layer_end))

        # ── Per-choice measures ──
        choice_measures = {m: [] for m in MEASURES}
        for ci, (s, e) in enumerate(choice_spans):
            if e - s < 2:
                for m in MEASURES:
                    choice_measures[m].append(0.0)
                continue
            per_m = {m: [] for m in MEASURES}
            for li in target_layers:
                hs = hidden_states[li][0, s:e, :]
                for mname, mfn in MEASURES.items():
                    per_m[mname].append(mfn(hs))
            for m in MEASURES:
                choice_measures[m].append(float(np.mean(per_m[m])))

        # ── Full deliberation measures ──
        full_measures = {}
        for mname, mfn in MEASURES.items():
            vals = [mfn(hidden_states[li][0]) for li in target_layers]
            full_measures[f"full_{mname}"] = float(np.mean(vals))

        # ── Logit-based answer ──
        last_logits = outputs.logits[0, -1, :]
        choice_logits = []
        for label in LABELS[:len(q.choices)]:
            tids = self.tokenizer.encode(label, add_special_tokens=False)
            choice_logits.append(float(last_logits[tids[0]].item()) if tids else -1e9)

        logit_pred = int(np.argmax(choice_logits))

        # Topology predictions
        topo_preds = {}
        for mname in MEASURES:
            vals = choice_measures[mname]
            if mname == "eff_rank":
                topo_preds[mname] = int(np.argmin(vals))
            else:
                topo_preds[mname] = int(np.argmax(vals))

        result = {
            "question": q.question[:200],
            "source": q.source,
            "domain": q.domain,
            "n_choices": len(q.choices),
            "correct_idx": q.correct_idx,
            "seq_len": seq_len,
            "n_layers": n_layers,
            "choice_logits": choice_logits,
            "logit_pred": logit_pred,
            "logit_correct": logit_pred == q.correct_idx,
        }

        for mname in MEASURES:
            result[f"choice_{mname}"] = choice_measures[mname]
            result[f"correct_{mname}"] = choice_measures[mname][q.correct_idx]
            wrong_vals = [v for i, v in enumerate(choice_measures[mname]) if i != q.correct_idx]
            result[f"wrong_mean_{mname}"] = float(np.mean(wrong_vals)) if wrong_vals else 0
            result[f"topo_pred_{mname}"] = topo_preds[mname]
            result[f"topo_correct_{mname}"] = topo_preds[mname] == q.correct_idx

        result.update(full_measures)
        return result

    def cleanup(self):
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze(results: list[dict], model_name: str, dataset: str) -> dict:
    n = len(results)
    n_choices = results[0]["n_choices"] if results else 4
    random_base = 1.0 / n_choices

    logit_acc = sum(1 for r in results if r["logit_correct"]) / n

    analysis = {"model": model_name, "dataset": dataset, "n": n,
                "random_baseline": random_base, "logit_accuracy": logit_acc}

    for mname in MEASURES:
        acc = sum(1 for r in results if r[f"topo_correct_{mname}"]) / n
        analysis[f"{mname}_accuracy"] = acc

        correct_vals = [r[f"correct_{mname}"] for r in results]
        wrong_vals = []
        for r in results:
            wrong_vals.extend([v for i, v in enumerate(r[f"choice_{mname}"])
                               if i != r["correct_idx"]])
        cv, wv = np.array(correct_vals), np.array(wrong_vals)
        if len(cv) > 5 and len(wv) > 5:
            alt = "less" if mname == "eff_rank" else "greater"
            _, p = mannwhitneyu(cv, wv, alternative=alt)
            pooled = np.sqrt((np.var(cv) + np.var(wv)) / 2)
            d = (np.mean(cv) - np.mean(wv)) / pooled if pooled > 0 else 0
        else:
            p, d = 1.0, 0.0
        analysis[f"{mname}_correct_mean"] = float(np.mean(cv))
        analysis[f"{mname}_wrong_mean"] = float(np.mean(wv))
        analysis[f"{mname}_d"] = float(d)
        analysis[f"{mname}_p"] = float(p)

    # Deliberation quality
    for mname in MEASURES:
        key = f"full_{mname}"
        right = [r[key] for r in results if r["logit_correct"]]
        wrong = [r[key] for r in results if not r["logit_correct"]]
        if len(right) > 3 and len(wrong) > 3:
            rv, wv2 = np.array(right), np.array(wrong)
            pooled = np.sqrt((np.var(rv) + np.var(wv2)) / 2)
            d2 = (np.mean(rv) - np.mean(wv2)) / pooled if pooled > 0 else 0
            _, p2 = mannwhitneyu(rv, wv2, alternative="greater" if mname != "eff_rank" else "less")
            analysis[f"delib_{mname}_d"] = float(d2)
            analysis[f"delib_{mname}_p"] = float(p2)
            analysis[f"delib_{mname}_right_mean"] = float(np.mean(rv))
            analysis[f"delib_{mname}_wrong_mean"] = float(np.mean(wv2))
        else:
            analysis[f"delib_{mname}_d"] = 0.0
            analysis[f"delib_{mname}_p"] = 1.0

    # Domain breakdown
    domains = {}
    for r in results:
        d = r["domain"]
        if d not in domains:
            domains[d] = {"total": 0, "logit_correct": 0}
        domains[d]["total"] += 1
        if r["logit_correct"]:
            domains[d]["logit_correct"] += 1
    analysis["domain_accuracy"] = {
        d: v["logit_correct"] / v["total"] for d, v in domains.items()
    }

    return analysis


def print_results(analysis: dict):
    print(f"\n{'='*65}")
    print(f"  {analysis['dataset']} x {analysis['model'].split('/')[-1]}")
    print(f"  n={analysis['n']}, random={analysis['random_baseline']:.1%}")
    print(f"{'='*65}")

    print(f"\n  LOGIT ACCURACY: {analysis['logit_accuracy']:.1%}")
    if "domain_accuracy" in analysis:
        for d, acc in sorted(analysis["domain_accuracy"].items()):
            print(f"    {d:15s}: {acc:.1%}")

    print(f"\n  PER-CHOICE TOPOLOGY (correct vs wrong):")
    print(f"    {'Measure':<14} {'Correct':>10} {'Wrong':>10} {'d':>8} {'p':>12} {'Acc':>7}")
    for m in MEASURES:
        c = analysis[f"{m}_correct_mean"]
        w = analysis[f"{m}_wrong_mean"]
        d = analysis[f"{m}_d"]
        p = analysis[f"{m}_p"]
        acc = analysis[f"{m}_accuracy"]
        sig = " ***" if p < 0.01 else " **" if p < 0.05 else " *" if p < 0.1 else ""
        print(f"    {m:<14} {c:>10.4f} {w:>10.4f} {d:>8.3f} {p:>12.2e} {acc:>6.1%}{sig}")

    print(f"\n  DELIBERATION QUALITY (model right vs wrong):")
    print(f"    {'Measure':<14} {'Right':>10} {'Wrong':>10} {'d':>8} {'p':>12}")
    for m in MEASURES:
        d = analysis[f"delib_{m}_d"]
        p = analysis[f"delib_{m}_p"]
        rm = analysis.get(f"delib_{m}_right_mean", 0)
        wm = analysis.get(f"delib_{m}_wrong_mean", 0)
        sig = " ***" if p < 0.01 else " **" if p < 0.05 else " *" if p < 0.1 else ""
        print(f"    {m:<14} {rm:>10.4f} {wm:>10.4f} {d:>8.3f} {p:>12.2e}{sig}")


def plot_8b(results: list[dict], analysis: dict, dataset: str):
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

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: Per-choice separation for each measure
    for mi, m in enumerate(measures):
        ax = axes[0, mi]
        correct_vals = [r[f"correct_{m}"] for r in results]
        wrong_vals = []
        for r in results:
            wrong_vals.extend([v for i, v in enumerate(r[f"choice_{m}"])
                               if i != r["correct_idx"]])
        ax.hist(correct_vals, bins=25, alpha=0.7, color=COLORS["teal"],
                label="Correct", density=True)
        ax.hist(wrong_vals, bins=25, alpha=0.7, color=COLORS["red"],
                label="Wrong", density=True)
        d = analysis[f"{m}_d"]
        p = analysis[f"{m}_p"]
        ax.set_title(f"Per-Choice {m}\n(d={d:.3f}, p={p:.2e})", color=COLORS["gold"])
        if mi == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    # Row 2: Deliberation quality
    for mi, m in enumerate(measures):
        ax = axes[1, mi]
        right = [r[f"full_{m}"] for r in results if r["logit_correct"]]
        wrong = [r[f"full_{m}"] for r in results if not r["logit_correct"]]
        if right:
            ax.hist(right, bins=20, alpha=0.7, color=COLORS["green"],
                    label=f"Correct (n={len(right)})", density=True)
        if wrong:
            ax.hist(wrong, bins=20, alpha=0.7, color=COLORS["red"],
                    label=f"Wrong (n={len(wrong)})", density=True)
        d = analysis[f"delib_{m}_d"]
        p = analysis[f"delib_{m}_p"]
        ax.set_title(f"Deliberation {m}\n(d={d:.3f}, p={p:.2e})", color=COLORS["gold"])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    fig.suptitle(f"8B Answer Topology -- {dataset} x {model_short}\n"
                 f"Logit acc: {analysis['logit_accuracy']:.1%}",
                 color=COLORS["text"], fontsize=14)
    fig.tight_layout()
    fname = f"answer_topo_8b_{dataset.lower().replace('-','_')}_{model_short}.png"
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
    print(f"  Figure: {FIG_DIR / fname}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 65)
    print("  ANSWER-TOPOLOGY TEST — 8B on GPQA Diamond")
    print("  Hypothesis: 8B is threshold for GPQA partial knowledge")
    print(f"  {ts}")
    print("=" * 65)

    model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"

    # Load datasets
    print("\n  Loading datasets...")
    gpqa = load_gpqa_diamond()
    print(f"  GPQA-Diamond: {len(gpqa)} questions")
    mmlu = load_mmlu_physics()
    print(f"  MMLU-Physics: {len(mmlu)} questions (comparison)")

    probe = TopologyProbe(model_name)

    all_analyses = []

    # Run MMLU first (comparison to 0.5B/1.5B results)
    print(f"\n{'='*65}")
    print(f"  MMLU-Physics x 7B-AWQ ({len(mmlu)} questions)")
    print(f"{'='*65}")
    results_mmlu = []
    t0 = time.time()
    for i, q in enumerate(mmlu):
        r = probe.measure_question(q)
        results_mmlu.append(r)
        if (i + 1) % 20 == 0 or (i + 1) == len(mmlu):
            elapsed = time.time() - t0
            logit_acc = sum(1 for r in results_mmlu if r["logit_correct"]) / len(results_mmlu)
            print(f"  [{i+1}/{len(mmlu)}] logit={logit_acc:.1%} elapsed={elapsed:.0f}s")

    a_mmlu = analyze(results_mmlu, model_name, "MMLU-Physics")
    print_results(a_mmlu)
    plot_8b(results_mmlu, a_mmlu, "MMLU-Physics")
    all_analyses.append(a_mmlu)

    # Run GPQA Diamond (the main event)
    print(f"\n{'='*65}")
    print(f"  GPQA-Diamond x 7B-AWQ ({len(gpqa)} questions)")
    print(f"{'='*65}")
    results_gpqa = []
    t0 = time.time()
    for i, q in enumerate(gpqa):
        r = probe.measure_question(q)
        results_gpqa.append(r)
        if (i + 1) % 20 == 0 or (i + 1) == len(gpqa):
            elapsed = time.time() - t0
            logit_acc = sum(1 for r in results_gpqa if r["logit_correct"]) / len(results_gpqa)
            print(f"  [{i+1}/{len(gpqa)}] logit={logit_acc:.1%} elapsed={elapsed:.0f}s")

    a_gpqa = analyze(results_gpqa, model_name, "GPQA-Diamond")
    print_results(a_gpqa)
    plot_8b(results_gpqa, a_gpqa, "GPQA-Diamond")
    all_analyses.append(a_gpqa)

    probe.cleanup()

    # Save
    output = {
        "timestamp": ts,
        "model": model_name,
        "analyses": all_analyses,
        "results_mmlu": results_mmlu,
        "results_gpqa": results_gpqa,
    }
    with open(OUTPUT_DIR / "answer_topology_8b.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Cross-scale comparison
    print(f"\n{'='*65}")
    print("  CROSS-SCALE COMPARISON: GPQA-Diamond")
    print(f"{'='*65}")
    print(f"  {'Model':<25} {'Logit':>7} {'delib_sg_d':>11} {'delib_nv_d':>11} {'delib_er_d':>11}")
    print(f"  {'-'*70}")
    prev = [
        ("Qwen2.5-0.5B", 0.273, 0.101, 0.080, -0.109),
        ("Qwen2.5-1.5B", 0.227, 0.057, -0.000, -0.082),
    ]
    for name, lacc, sg, nv, er in prev:
        print(f"  {name:<25} {lacc:>6.1%} {sg:>11.3f} {nv:>11.3f} {er:>11.3f}")

    a = a_gpqa
    print(f"  {model_name.split('/')[-1]:<25} {a['logit_accuracy']:>6.1%} "
          f"{a['delib_spectral_gap_d']:>11.3f} {a['delib_norm_var_d']:>11.3f} "
          f"{a['delib_eff_rank_d']:>11.3f}")

    print(f"\n  Summary: {OUTPUT_DIR / 'answer_topology_8b.json'}")


if __name__ == "__main__":
    main()
