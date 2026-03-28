#!/usr/bin/env python3
"""Cross-Architecture Sweep — Map format exploitation vs real knowledge across model families.

Models (all fit in 12GB VRAM):
  Qwen2.5:   0.5B, 1.5B, 3B          (dense, fp16)
  Llama-3.2: 1B, 3B                   (dense, fp16)
  Phi-3.5:   mini-instruct (3.8B)     (dense, fp16)
  Gemma-2:   2B-it                    (dense, fp16)

For each model, on MMLU-Physics:
  1. Likelihood accuracy (no letters — real knowledge)
  2. MCQ strict accuracy (4/4 rotations — position-invariant)
  3. MCQ inflation ratio (pos-avg / strict — format exploitation)
  4. Deliberation signal in likelihood mode (genuine topology signal)
  5. Rotation consistency (CV of deliberation across positions)
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

LABELS = ["A", "B", "C", "D"]

# Models to sweep — ordered by size
MODELS = [
    {"name": "HuggingFaceTB/SmolLM2-360M-Instruct", "family": "SmolLM2", "size": 0.36},
    {"name": "Qwen/Qwen2.5-0.5B", "family": "Qwen2.5", "size": 0.5},
    {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "TinyLlama", "size": 1.1},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct", "family": "Qwen2.5", "size": 1.5},
    {"name": "stabilityai/stablelm-2-1_6b-chat", "family": "StableLM", "size": 1.6},
    {"name": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "family": "SmolLM2", "size": 1.7},
    {"name": "Qwen/Qwen2.5-3B-Instruct", "family": "Qwen2.5", "size": 3.0},
]


# ── Measures ────────────────────────────────────────────────────────────────

def spectral_gap(hs):
    h = hs.cpu().float()
    if h.shape[0] < 2: return 0.0
    h = h - h.mean(0)
    try: s = torch.linalg.svdvals(h)
    except Exception: return 0.0
    if len(s) < 2 or s[1] < 1e-10: return float(s[0]) if len(s) > 0 else 0.0
    return float(s[0] / s[1])

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


# ── Data ────────────────────────────────────────────────────────────────────

@dataclass
class MCQ:
    question: str
    choices: list[str]  # [0] = correct
    source: str; domain: str

def load_mmlu():
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "college_physics", split="test")
    qs = []
    for r in ds:
        ci = int(r["answer"]); ch = list(r["choices"])
        correct = ch[ci]; others = [c for i, c in enumerate(ch) if i != ci]
        qs.append(MCQ(r["question"], [correct] + others, "MMLU-Physics", "physics"))
    return qs


# ── Probe ───────────────────────────────────────────────────────────────────

class Probe:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        print(f"\n  Loading {model_name}...")
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            dtype=torch.float16, device_map=self.device,
            output_hidden_states=True)
        self.model.eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        np_ = sum(p.numel() for p in self.model.parameters()) / 1e6
        nl = self.model.config.num_hidden_layers
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  {np_:.0f}M params, {nl}L, {vram:.1f}GB VRAM")

    def _targets(self, nl):
        return list(range(max(1, nl // 3), nl - 1))

    @torch.no_grad()
    def score_likelihood(self, question, answer):
        prefix = f"Q: {question}\nA:"
        plen = self.tok(prefix, return_tensors="pt")["input_ids"].shape[1]
        text = f"Q: {question}\nA: {answer}"
        inp = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=1024).to(self.device)
        out = self.model(input_ids=inp["input_ids"], output_hidden_states=True)
        alen = inp["input_ids"].shape[1] - plen
        if alen > 0:
            score = -torch.nn.functional.cross_entropy(
                out.logits[0, plen-1:-1, :], inp["input_ids"][0, plen:],
                reduction="mean").item()
        else:
            score = -1e9
        hs = out.hidden_states
        tgt = self._targets(len(hs))
        sg = float(np.mean([spectral_gap(hs[li][0]) for li in tgt]))
        er = float(np.mean([effective_rank(hs[li][0]) for li in tgt]))
        return {"score": score, "sg": sg, "er": er}

    @torch.no_grad()
    def score_mcq(self, question, choices, correct_pos):
        prompt = f"Question: {question}\n\n"
        for i, c in enumerate(choices):
            prompt += f"{LABELS[i]}) {c}\n"
        prompt += "\nAnswer:"
        inp = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(self.device)
        out = self.model(input_ids=inp["input_ids"], output_hidden_states=True)
        ll = out.logits[0, -1, :]
        scores = []
        for lb in LABELS[:len(choices)]:
            t = self.tok.encode(lb, add_special_tokens=False)
            scores.append(float(ll[t[0]].item()) if t else -1e9)
        pred = int(np.argmax(scores))
        hs = out.hidden_states
        tgt = self._targets(len(hs))
        sg = float(np.mean([spectral_gap(hs[li][0]) for li in tgt]))
        return {"pred": pred, "correct": pred == correct_pos, "sg": sg}

    def cleanup(self):
        del self.model, self.tok; gc.collect(); torch.cuda.empty_cache()


# ── Sweep ───────────────────────────────────────────────────────────────────

def sweep_model(probe, questions):
    results = []
    for qi, q in enumerate(questions):
        ca = q.choices[0]; wa = q.choices[1:4]

        # Likelihood
        lk_c = probe.score_likelihood(q.question, ca)
        lk_w = [probe.score_likelihood(q.question, w) for w in wa]
        all_s = [lk_c["score"]] + [w["score"] for w in lk_w]
        lk_pred = int(np.argmax(all_s))

        # MCQ × 4 rotations
        mcq_correct = 0
        mcq_sgs = []
        for rot in range(4):
            rng = np.random.default_rng(abs(hash(q.question)) + rot)
            sw = list(wa); rng.shuffle(sw)
            rotated = (sw[:rot] + [ca] + sw[rot:])[:4]
            mr = probe.score_mcq(q.question, rotated, rot)
            if mr["correct"]: mcq_correct += 1
            mcq_sgs.append(mr["sg"])

        results.append({
            "lk_correct": lk_pred == 0,
            "lk_sg": lk_c["sg"],
            "lk_er": lk_c["er"],
            "mcq_n_correct": mcq_correct,
            "mcq_strict": mcq_correct == 4,
            "mcq_avg": mcq_correct / 4,
            "mcq_sg_cv": float(np.std(mcq_sgs) / np.mean(mcq_sgs)) if np.mean(mcq_sgs) > 0 else 0,
        })

        if (qi + 1) % 25 == 0 or (qi + 1) == len(questions):
            la = np.mean([r["lk_correct"] for r in results])
            ma = np.mean([r["mcq_avg"] for r in results])
            ms = np.mean([r["mcq_strict"] for r in results])
            print(f"    [{qi+1}/{len(questions)}] lk={la:.1%} mcq_avg={ma:.1%} strict={ms:.1%}")

    return results


def analyze_sweep(results, model_info):
    n = len(results)
    lk_acc = np.mean([r["lk_correct"] for r in results])
    mcq_avg = np.mean([r["mcq_avg"] for r in results])
    mcq_strict = np.mean([r["mcq_strict"] for r in results])
    inflation = mcq_avg / mcq_strict if mcq_strict > 0 else float("inf")
    cv = np.mean([r["mcq_sg_cv"] for r in results])

    # Deliberation in likelihood mode
    right = [r["lk_sg"] for r in results if r["lk_correct"]]
    wrong = [r["lk_sg"] for r in results if not r["lk_correct"]]
    if len(right) > 3 and len(wrong) > 3:
        rv, wv = np.array(right), np.array(wrong)
        ps = np.sqrt((np.var(rv) + np.var(wv)) / 2)
        d = (np.mean(rv) - np.mean(wv)) / ps if ps > 0 else 0
        _, p = mannwhitneyu(rv, wv, alternative="greater")
    else:
        d, p = 0, 1

    return {
        "model": model_info["name"],
        "family": model_info["family"],
        "size_B": model_info["size"],
        "n": n,
        "lk_accuracy": float(lk_acc),
        "mcq_pos_avg": float(mcq_avg),
        "mcq_strict": float(mcq_strict),
        "mcq_inflation": float(inflation),
        "rotation_cv": float(cv),
        "delib_sg_d": float(d),
        "delib_sg_p": float(p),
    }


def main():
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 65)
    print("  CROSS-ARCHITECTURE SWEEP")
    print("  Format exploitation vs real knowledge")
    print(f"  {ts}")
    print("=" * 65)

    questions = load_mmlu()
    print(f"  MMLU-Physics: {len(questions)} questions")
    print(f"  Models: {len(MODELS)}")

    all_analyses = []

    for mi, minfo in enumerate(MODELS):
        print(f"\n{'='*65}")
        print(f"  [{mi+1}/{len(MODELS)}] {minfo['name']} ({minfo['size']}B, {minfo['family']})")
        print(f"{'='*65}")

        try:
            probe = Probe(minfo["name"])
            results = sweep_model(probe, questions)
            analysis = analyze_sweep(results, minfo)
            all_analyses.append(analysis)
            probe.cleanup()
        except Exception as e:
            print(f"  FAILED: {e}")
            all_analyses.append({
                "model": minfo["name"], "family": minfo["family"],
                "size_B": minfo["size"], "error": str(e)})
            gc.collect(); torch.cuda.empty_cache()
            continue

    # ── Summary table ──
    print(f"\n{'='*90}")
    print("  CROSS-ARCHITECTURE RESULTS: MMLU-Physics")
    print(f"{'='*90}")
    print(f"  {'Model':<35} {'Size':>5} {'LK%':>6} {'MCQ%':>6} {'Strict':>7} {'Inflate':>8} {'d(sg)':>7} {'CV':>6}")
    print(f"  {'-'*85}")
    for a in all_analyses:
        if "error" in a:
            print(f"  {a['model']:<35} {a['size_B']:>5.1f} {'FAILED':>6}")
            continue
        sig = "***" if a["delib_sg_p"] < 0.01 else "**" if a["delib_sg_p"] < 0.05 else "*" if a["delib_sg_p"] < 0.1 else ""
        print(f"  {a['model'].split('/')[-1]:<35} {a['size_B']:>5.1f} "
              f"{a['lk_accuracy']:>5.1%} {a['mcq_pos_avg']:>5.1%} "
              f"{a['mcq_strict']:>6.1%} {a['mcq_inflation']:>7.1f}x "
              f"{a['delib_sg_d']:>6.3f}{sig} {a['rotation_cv']:>5.3f}")

    # By family
    families = {}
    for a in all_analyses:
        if "error" in a: continue
        f = a["family"]
        if f not in families: families[f] = []
        families[f].append(a)

    print(f"\n  BY FAMILY — Format honesty (lower inflation = more honest):")
    for f, models in sorted(families.items()):
        avg_inflate = np.mean([m["mcq_inflation"] for m in models])
        avg_lk = np.mean([m["lk_accuracy"] for m in models])
        print(f"    {f:<12}: inflation={avg_inflate:.1f}x, mean_lk={avg_lk:.1%}")

    # Save
    with open(OUTPUT_DIR / "arch_sweep.json", "w") as f:
        json.dump({"timestamp": ts, "analyses": all_analyses}, f, indent=2)
    print(f"\n  Saved: {OUTPUT_DIR / 'arch_sweep.json'}")


if __name__ == "__main__":
    main()
