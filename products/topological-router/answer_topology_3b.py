#!/usr/bin/env python3
"""Answer-Topology Test — 3B fills the scaling gap between 1.5B and 7B.

Scaling law so far:
  0.5B: d = 0 (no signal)
  1.5B: d = 0.469 (medium)
  7B:   d = 0.654 (large)

3B should slot in between. And since it loads in fp16 (no quantization),
the signal is cleaner than the 7B-AWQ result.
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
    question: str; choices: list[str]; correct_idx: int; source: str; domain: str

def load_gpqa(n_max=0):
    qs = []
    with open(GPQA_PATH) as f:
        for r in csv.DictReader(f):
            ch = [r["Correct Answer"], r["Incorrect Answer 1"],
                  r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            if not all(c.strip() for c in ch): continue
            qs.append(MCQ(r["Question"], ch, 0, "GPQA-Diamond", r.get("High-level domain","physics")))
    rng = np.random.default_rng(42)
    for q in qs:
        p = rng.permutation(len(q.choices))
        q.choices = [q.choices[int(i)] for i in p]
        q.correct_idx = int(np.where(p == 0)[0][0])
    return qs[:n_max] if n_max > 0 else qs

def load_mmlu(n_max=0):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "college_physics", split="test")
    qs = [MCQ(r["question"], r["choices"], int(r["answer"]), "MMLU-Physics", "physics") for r in ds]
    return qs[:n_max] if n_max > 0 else qs

def format_mcq(q):
    lines = [f"Question: {q.question}", ""]
    for i, c in enumerate(q.choices): lines.append(f"{LABELS[i]}) {c}")
    lines.extend(["", "Answer:"])
    return "\n".join(lines)

def find_spans(tok, text, choices):
    spans = []
    for i, c in enumerate(choices):
        m = f"{LABELS[i]}) {c}"
        pos = text.find(m)
        if pos < 0: spans.append((0,0)); continue
        s = len(tok(text[:pos], return_tensors="pt")["input_ids"][0])
        e = len(tok(text[:pos+len(m)], return_tensors="pt")["input_ids"][0])
        spans.append((s, e))
    return spans


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
        np_ = sum(p.numel() for p in self.model.parameters())/1e6
        nl = self.model.config.num_hidden_layers
        vram = torch.cuda.memory_allocated()/1e9
        print(f"  {np_:.0f}M params, {nl} layers, VRAM={vram:.1f}GB")

    @torch.no_grad()
    def measure(self, q):
        prompt = format_mcq(q)
        inp = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        out = self.model(**inp)
        hs = out.hidden_states
        nl = len(hs); sl = inp["input_ids"].shape[1]
        spans = find_spans(self.tok, prompt, q.choices)

        ls, le = max(1, nl//3), nl-1
        target = list(range(ls, le))

        # Per-choice
        cm = {m: [] for m in MEASURES}
        for s, e in spans:
            if e-s < 2:
                for m in MEASURES: cm[m].append(0.0)
                continue
            pm = {m: [] for m in MEASURES}
            for li in target:
                for mn, mf in MEASURES.items(): pm[mn].append(mf(hs[li][0,s:e,:]))
            for m in MEASURES: cm[m].append(float(np.mean(pm[m])))

        # Full deliberation
        fm = {}
        for mn, mf in MEASURES.items():
            fm[f"full_{mn}"] = float(np.mean([mf(hs[li][0]) for li in target]))

        # Logits
        ll = out.logits[0,-1,:]
        cl = []
        for lb in LABELS[:len(q.choices)]:
            t = self.tok.encode(lb, add_special_tokens=False)
            cl.append(float(ll[t[0]].item()) if t else -1e9)
        lp = int(np.argmax(cl))

        # Topo preds
        tp = {}
        for mn in MEASURES:
            v = cm[mn]
            tp[mn] = int(np.argmin(v)) if mn == "eff_rank" else int(np.argmax(v))

        r = {"question": q.question[:200], "source": q.source, "domain": q.domain,
             "n_choices": len(q.choices), "correct_idx": q.correct_idx, "seq_len": sl,
             "n_layers": nl, "choice_logits": cl, "logit_pred": lp,
             "logit_correct": lp == q.correct_idx}
        for mn in MEASURES:
            r[f"choice_{mn}"] = cm[mn]
            r[f"correct_{mn}"] = cm[mn][q.correct_idx]
            wv = [v for i,v in enumerate(cm[mn]) if i != q.correct_idx]
            r[f"wrong_mean_{mn}"] = float(np.mean(wv)) if wv else 0
            r[f"topo_pred_{mn}"] = tp[mn]
            r[f"topo_correct_{mn}"] = tp[mn] == q.correct_idx
        r.update(fm)
        return r

    def cleanup(self):
        del self.model, self.tok; gc.collect(); torch.cuda.empty_cache()


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze(results, model_name, dataset):
    n = len(results); nc = results[0]["n_choices"]; rb = 1.0/nc
    la = sum(1 for r in results if r["logit_correct"])/n
    a = {"model": model_name, "dataset": dataset, "n": n, "random": rb, "logit_acc": la}

    for mn in MEASURES:
        a[f"{mn}_acc"] = sum(1 for r in results if r[f"topo_correct_{mn}"])/n
        cv = np.array([r[f"correct_{mn}"] for r in results])
        wv = []
        for r in results: wv.extend([v for i,v in enumerate(r[f"choice_{mn}"]) if i!=r["correct_idx"]])
        wv = np.array(wv)
        if len(cv)>5 and len(wv)>5:
            alt = "less" if mn=="eff_rank" else "greater"
            _,p = mannwhitneyu(cv,wv,alternative=alt)
            ps = np.sqrt((np.var(cv)+np.var(wv))/2)
            d = (np.mean(cv)-np.mean(wv))/ps if ps>0 else 0
        else: p,d = 1,0
        a[f"{mn}_d"] = float(d); a[f"{mn}_p"] = float(p)

    # Deliberation
    for mn in MEASURES:
        rv = [r[f"full_{mn}"] for r in results if r["logit_correct"]]
        wv = [r[f"full_{mn}"] for r in results if not r["logit_correct"]]
        if len(rv)>3 and len(wv)>3:
            rv,wv = np.array(rv),np.array(wv)
            ps = np.sqrt((np.var(rv)+np.var(wv))/2)
            d = (np.mean(rv)-np.mean(wv))/ps if ps>0 else 0
            alt = "less" if mn=="eff_rank" else "greater"
            _,p = mannwhitneyu(rv,wv,alternative=alt)
            a[f"delib_{mn}_d"] = float(d); a[f"delib_{mn}_p"] = float(p)
            a[f"delib_{mn}_right"] = float(np.mean(rv))
            a[f"delib_{mn}_wrong"] = float(np.mean(wv))
        else:
            a[f"delib_{mn}_d"] = 0; a[f"delib_{mn}_p"] = 1

    # Domain breakdown
    doms = {}
    for r in results:
        d = r["domain"]
        if d not in doms: doms[d] = {"n":0,"c":0}
        doms[d]["n"] += 1
        if r["logit_correct"]: doms[d]["c"] += 1
    a["domains"] = {d: v["c"]/v["n"] for d,v in doms.items()}
    return a


def main():
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("="*65)
    print("  ANSWER-TOPOLOGY — 3B (fills the scaling gap)")
    print(f"  {ts}")
    print("="*65)

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    probe = Probe(model_name)

    # MMLU-Physics
    mmlu = load_mmlu()
    print(f"\n  MMLU-Physics: {len(mmlu)} questions")
    res_m = []
    t0 = time.time()
    for i,q in enumerate(mmlu):
        res_m.append(probe.measure(q))
        if (i+1)%20==0 or (i+1)==len(mmlu):
            la = sum(1 for r in res_m if r["logit_correct"])/len(res_m)
            print(f"  [{i+1}/{len(mmlu)}] logit={la:.1%} elapsed={time.time()-t0:.0f}s")
    a_m = analyze(res_m, model_name, "MMLU-Physics")

    # GPQA Diamond
    gpqa = load_gpqa()
    print(f"\n  GPQA-Diamond: {len(gpqa)} questions")
    res_g = []
    t0 = time.time()
    for i,q in enumerate(gpqa):
        res_g.append(probe.measure(q))
        if (i+1)%20==0 or (i+1)==len(gpqa):
            la = sum(1 for r in res_g if r["logit_correct"])/len(res_g)
            print(f"  [{i+1}/{len(gpqa)}] logit={la:.1%} elapsed={time.time()-t0:.0f}s")
    a_g = analyze(res_g, model_name, "GPQA-Diamond")

    probe.cleanup()

    # Print results
    for a in [a_m, a_g]:
        print(f"\n{'='*65}")
        print(f"  {a['dataset']} x {a['model'].split('/')[-1]}")
        print(f"  Logit accuracy: {a['logit_acc']:.1%}")
        if "domains" in a:
            for d,acc in sorted(a["domains"].items()): print(f"    {d:15s}: {acc:.1%}")
        print(f"\n  DELIBERATION QUALITY (right vs wrong):")
        print(f"    {'Measure':<14} {'d':>8} {'p':>12}")
        for mn in MEASURES:
            d = a[f"delib_{mn}_d"]; p = a[f"delib_{mn}_p"]
            sig = " ***" if p<0.01 else " **" if p<0.05 else " *" if p<0.1 else ""
            print(f"    {mn:<14} {d:>8.3f} {p:>12.2e}{sig}")

    # Cross-scale
    print(f"\n{'='*65}")
    print("  SCALING LAW: MMLU-Physics deliberation spectral_gap")
    print(f"{'='*65}")
    prev = [("0.5B",0.314,-0.350), ("1.5B",0.490,0.469), ("3B",a_m["logit_acc"],a_m["delib_spectral_gap_d"]),
            ("7B-AWQ",0.471,0.654)]
    print(f"  {'Model':<12} {'Logit':>7} {'delib_sg_d':>11}")
    for name,la,d in prev:
        sig = " ***" if abs(d)>0.5 else " **" if abs(d)>0.3 else ""
        print(f"  {name:<12} {la:>6.1%} {d:>11.3f}{sig}")

    print(f"\n  SCALING LAW: GPQA-Diamond deliberation spectral_gap")
    prev_g = [("0.5B",0.273,0.101), ("1.5B",0.227,0.057), ("3B",a_g["logit_acc"],a_g["delib_spectral_gap_d"]),
              ("7B-AWQ",0.273,0.134)]
    print(f"  {'Model':<12} {'Logit':>7} {'delib_sg_d':>11}")
    for name,la,d in prev_g:
        print(f"  {name:<12} {la:>6.1%} {d:>11.3f}")

    # Save
    out = {"timestamp": ts, "model": model_name,
           "analyses": [a_m, a_g], "results_mmlu": res_m, "results_gpqa": res_g}
    with open(OUTPUT_DIR / "answer_topology_3b.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR / 'answer_topology_3b.json'}")


if __name__ == "__main__":
    main()
