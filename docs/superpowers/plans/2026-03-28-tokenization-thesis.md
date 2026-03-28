# Tokenization Thesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover the natural representational basis of language models by seeding with 59 semantic primes, adaptively expanding via residual analysis, and measuring topological isomorphism between prime and residual subspaces across 6 model architectures.

**Architecture:** Four modules executed sequentially per model. Module 1 seeds a prime subspace from Wierzbicka's 59 semantic primes. Module 2 adaptively expands the basis via bandit-style prompt exploration. Module 3 computes four topology metrics comparing prime and residual subspaces. Module 4 sweeps all six models and produces cross-architecture comparison.

**Tech Stack:** PyTorch, transformers, autoawq (for 7B), numpy, scipy. Reuses existing topology measures (spectral_gap, effective_rank, norm_variance) from `clean_bench.py`. RTX 5070, 12GB VRAM, models loaded one at a time.

**Spec:** `docs/superpowers/specs/2026-03-27-tokenization-thesis-design.md`

---

## File Structure

```
products/topological-router/
  semantic_primes.py         # The 59 Wierzbicka primes + categories (data, no logic)
  prime_basis.py             # Module 1: compute prime subspace and vocabulary projection
  adaptive_explorer.py       # Module 2: adaptive basis discovery loop
  topology_comparison.py     # Module 3: four topology metrics
  sweep_runner.py            # Module 4: cross-architecture execution (replaces arch_sweep.py)
  topo_measures.py           # Shared topology functions extracted from clean_bench.py
  prompts/
    physics.json             # 20 MMLU-Physics questions
    narrative.json           # 20 narrative prompts
    logic.json               # 20 logic puzzles
    shuffled.json            # 20 syntax-destroyed sentences (null control)
    multilingual.json        # 20 parallel sentences
  results/
    basis_discovery/         # Per-model: adaptive_basis.pt, convergence.json
    topology_comparison/     # Per-model: metrics.json
    sweep_summary.json       # Cross-architecture summary table
```

**Key design decisions:**
- `semantic_primes.py` is pure data — the 59 primes grouped by category. No imports, no logic. Every other module imports from it.
- `topo_measures.py` extracts the shared measure functions (spectral_gap, effective_rank, norm_variance, h0_gini) so they aren't duplicated across modules. Existing scripts (clean_bench.py, arch_sweep.py) are NOT modified — they still work standalone.
- Module 4 replaces `arch_sweep.py` for the new pipeline but does not modify the old file.

---

### Task 1: Semantic Primes Data + Shared Measures

**Files:**
- Create: `products/topological-router/semantic_primes.py`
- Create: `products/topological-router/topo_measures.py`

- [ ] **Step 1: Create semantic_primes.py with all 59 Wierzbicka primes**

```python
# products/topological-router/semantic_primes.py
"""Wierzbicka's 59 Natural Semantic Metalanguage primes, grouped by category.

Source: Wierzbicka, A. (1996). Semantics: Primes and Universals.
These are the irreducible building blocks of meaning found in ALL human languages.
"""

PRIMES = {
    "substantives": ["I", "you", "someone", "something", "people", "body"],
    "determiners": ["this", "the same", "other"],
    "quantifiers": ["one", "two", "some", "all", "many", "much"],
    "evaluators": ["good", "bad"],
    "descriptors": ["big", "small"],
    "mental": ["think", "know", "want", "don't want", "feel", "see", "hear"],
    "speech": ["say", "words", "true"],
    "actions": ["do", "happen", "move"],
    "existence": ["there is", "be (someone/something)"],
    "possession": ["have", "be (someone's)"],
    "life": ["live", "die"],
    "time": ["when", "now", "before", "after", "a long time", "a short time", "for some time"],
    "space": ["where", "here", "above", "below", "far", "near", "side", "inside", "touch"],
    "logic": ["not", "maybe", "can", "because", "if"],
    "intensifier": ["very", "more"],
    "taxonomy": ["kind of", "like", "part of"],
}

# Flat list for iteration
ALL_PRIMES = [p for category in PRIMES.values() for p in category]

# Category lookup
PRIME_TO_CATEGORY = {p: cat for cat, primes in PRIMES.items() for p in primes}

assert len(ALL_PRIMES) == 59, f"Expected 59 primes, got {len(ALL_PRIMES)}"
```

- [ ] **Step 2: Verify the count**

Run: `cd /home/wb1/Desktop/Dev/atft-problems && python3 -c "from products.topological_router.semantic_primes import ALL_PRIMES; print(f'{len(ALL_PRIMES)} primes'); print(ALL_PRIMES)"`

Note: if Python import fails due to hyphens in path, use: `python3 products/topological-router/semantic_primes.py` with a `__main__` guard, or use `importlib`.

- [ ] **Step 3: Create topo_measures.py with shared topology functions**

```python
# products/topological-router/topo_measures.py
"""Shared topology measurement functions.

Extracted from clean_bench.py so all modules use the same implementations.
"""
import numpy as np
import torch


def effective_rank(hs: torch.Tensor) -> float:
    """Effective rank: exp(entropy of normalized singular values).
    Lower = more hierarchical. Range: [1, min(seq_len, hidden_dim)].
    """
    h = hs.cpu().float()
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
    return float(np.exp(-torch.sum(p * torch.log(p)).item()))


def spectral_gap(hs: torch.Tensor) -> float:
    """Ratio of first to second singular value. Higher = more hierarchical."""
    h = hs.cpu().float()
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


def norm_variance(hs: torch.Tensor) -> float:
    """Variance of L2 norms across tokens. Higher = more differentiated."""
    h = hs.cpu().float()
    if h.shape[0] < 2:
        return 0.0
    return float(torch.var(torch.norm(h, dim=1)).item())


def gini_fast(values: np.ndarray) -> float:
    """Gini coefficient of a 1D array."""
    n = len(values)
    if n <= 1 or np.sum(values) == 0:
        return 0.0
    s = np.sort(values)
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * np.sum(i * s)) / (n * np.sum(s)) - (n + 1) / n)


def h0_persistence(points: np.ndarray, max_n: int = 200) -> np.ndarray:
    """H0 persistence bars via GPU pairwise distance + CPU union-find."""
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


def h0_gini(points: np.ndarray, max_n: int = 200) -> float:
    """Gini coefficient of H0 persistence bars."""
    bars = h0_persistence(points, max_n)
    return gini_fast(bars)


MEASURES = {
    "eff_rank": effective_rank,
    "spectral_gap": spectral_gap,
    "norm_var": norm_variance,
}
```

- [ ] **Step 4: Quick sanity test**

Run: `cd /home/wb1/Desktop/Dev/atft-problems && python3 -c "
import torch, sys; sys.path.insert(0, 'products/topological-router')
from topo_measures import spectral_gap, effective_rank, h0_gini
import numpy as np
# Test on random tensor
t = torch.randn(20, 128)
print(f'spectral_gap: {spectral_gap(t):.3f}')
print(f'effective_rank: {effective_rank(t):.3f}')
print(f'h0_gini: {h0_gini(np.random.randn(50, 10)):.3f}')
print('All measures working.')
"`

Expected: Three numeric values printed, no errors.

- [ ] **Step 5: Commit**

```bash
cd /home/wb1/Desktop/Dev/atft-problems
git add products/topological-router/semantic_primes.py products/topological-router/topo_measures.py
git commit -m "feat: semantic primes data + shared topology measures"
```

---

### Task 2: Prompt Dataset Curation

**Files:**
- Create: `products/topological-router/prompts/physics.json`
- Create: `products/topological-router/prompts/narrative.json`
- Create: `products/topological-router/prompts/logic.json`
- Create: `products/topological-router/prompts/shuffled.json`
- Create: `products/topological-router/prompts/multilingual.json`
- Create: `products/topological-router/prompts/loader.py`

- [ ] **Step 1: Create physics.json — 20 MMLU-Physics questions**

```python
# Script to extract 20 MMLU-Physics questions
import json
from datasets import load_dataset
ds = load_dataset("cais/mmlu", "college_physics", split="test")
prompts = [{"text": row["question"], "mode": "physics"} for row in list(ds)[:20]]
with open("products/topological-router/prompts/physics.json", "w") as f:
    json.dump(prompts, f, indent=2)
print(f"Saved {len(prompts)} physics prompts")
```

- [ ] **Step 2: Create narrative.json — 20 narrative prompts**

```json
[
  {"text": "The old lighthouse keeper watched the storm approach from the west, knowing that tonight would test everything he had built.", "mode": "narrative"},
  {"text": "She opened the letter carefully, not because it was fragile, but because she knew the words inside would change everything.", "mode": "narrative"},
  {"text": "The market in Marrakech was a symphony of color, sound, and the sharp sweetness of fresh mint tea.", "mode": "narrative"},
  {"text": "He had been walking for three days when the forest finally opened into a valley unlike anything the maps had shown.", "mode": "narrative"},
  {"text": "The violin had belonged to her grandmother, and each note it produced carried a century of memory.", "mode": "narrative"},
  {"text": "Rain fell on the ancient stones of the temple, pooling in carvings that had held water for a thousand years.", "mode": "narrative"},
  {"text": "The train pulled away from the station, and with it went the last connection to the life she had known.", "mode": "narrative"},
  {"text": "Between the pages of the dictionary, someone had pressed a flower that still smelled faintly of summer.", "mode": "narrative"},
  {"text": "The astronaut looked back at Earth and understood, for the first time, what the poets had been trying to say.", "mode": "narrative"},
  {"text": "Every morning the baker arrived before dawn, and every morning the bread rose as if greeting the sun.", "mode": "narrative"},
  {"text": "The chess grandmaster stared at the board, seeing not pieces but a conversation spanning forty moves.", "mode": "narrative"},
  {"text": "Wind carried seeds from the old garden across the wall into a place where nothing had grown for years.", "mode": "narrative"},
  {"text": "The translator paused at a word that existed in no other language, a word for the feeling of returning home.", "mode": "narrative"},
  {"text": "Beneath the ice of the arctic lake, life continued in forms that had never seen light.", "mode": "narrative"},
  {"text": "The architect drew a bridge not between two banks, but between two ideas of what a city could be.", "mode": "narrative"},
  {"text": "In the quiet hour before the children woke, the house itself seemed to breathe and remember.", "mode": "narrative"},
  {"text": "The musician played a note so pure that for one moment the audience forgot they were separate people.", "mode": "narrative"},
  {"text": "Years after the war, the field had healed so completely that only the poppies remembered.", "mode": "narrative"},
  {"text": "The code compiled on the first try, which is how she knew something was deeply, fundamentally wrong.", "mode": "narrative"},
  {"text": "At the edge of the known map, the cartographer wrote not 'here be dragons' but 'here be questions.'", "mode": "narrative"}
]
```

- [ ] **Step 3: Create logic.json — 20 logic puzzles**

```json
[
  {"text": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?", "mode": "logic"},
  {"text": "If it rains, the ground gets wet. The ground is wet. Did it rain?", "mode": "logic"},
  {"text": "A is taller than B. B is taller than C. Is A taller than C?", "mode": "logic"},
  {"text": "Every prime number greater than 2 is odd. 7 is a prime number greater than 2. What follows?", "mode": "logic"},
  {"text": "If all birds can fly, and penguins are birds, can penguins fly? What is wrong with this argument?", "mode": "logic"},
  {"text": "Three boxes contain apples, oranges, or both. All labels are wrong. You pick one fruit from the box labeled 'both'. How do you determine the contents of all boxes?", "mode": "logic"},
  {"text": "A says B is lying. B says C is lying. C says both A and B are lying. Who is telling the truth?", "mode": "logic"},
  {"text": "You have 8 balls, one heavier. You have a balance scale and can weigh twice. How do you find the heavy ball?", "mode": "logic"},
  {"text": "If no fish are mammals, and all dolphins are mammals, what can we conclude about dolphins and fish?", "mode": "logic"},
  {"text": "A train leaves city A at 60 mph. Another leaves city B toward A at 40 mph. Cities are 200 miles apart. Where do they meet?", "mode": "logic"},
  {"text": "The barber shaves everyone who does not shave themselves. Does the barber shave himself?", "mode": "logic"},
  {"text": "If P implies Q, and Q implies R, does P imply R? What is this rule called?", "mode": "logic"},
  {"text": "You flip a fair coin 10 times and get heads every time. What is the probability of heads on the next flip?", "mode": "logic"},
  {"text": "All swans observed so far are white. Is it valid to conclude all swans are white?", "mode": "logic"},
  {"text": "If exactly one of these three statements is true, which one is it? Statement 1: Statement 2 is true. Statement 2: Statement 3 is true. Statement 3: Statements 1 and 2 are false.", "mode": "logic"},
  {"text": "A room has three light switches outside and three bulbs inside. You can enter the room only once. How do you determine which switch controls which bulb?", "mode": "logic"},
  {"text": "Is the set of all sets that do not contain themselves a member of itself?", "mode": "logic"},
  {"text": "If you have a 3-liter jug and a 5-liter jug, how do you measure exactly 4 liters?", "mode": "logic"},
  {"text": "The sum of two consecutive integers is always odd. True or false? Prove it.", "mode": "logic"},
  {"text": "If increasing the temperature increases reaction rate, and increasing pressure increases temperature, does increasing pressure increase reaction rate?", "mode": "logic"}
]
```

- [ ] **Step 4: Create shuffled.json — syntax-destroyed sentences (null control)**

Generate by shuffling the narrative prompts word by word:

```python
import json, random
random.seed(42)
with open("products/topological-router/prompts/narrative.json") as f:
    narratives = json.load(f)
shuffled = []
for n in narratives:
    words = n["text"].split()
    random.shuffle(words)
    shuffled.append({"text": " ".join(words), "mode": "shuffled"})
with open("products/topological-router/prompts/shuffled.json", "w") as f:
    json.dump(shuffled, f, indent=2)
print(f"Saved {len(shuffled)} shuffled prompts")
```

- [ ] **Step 5: Create multilingual.json — 20 parallel sentences**

```json
[
  {"text": "Water boils at one hundred degrees.", "mode": "multilingual", "lang": "en"},
  {"text": "El agua hierve a cien grados.", "mode": "multilingual", "lang": "es"},
  {"text": "L'eau bout à cent degrés.", "mode": "multilingual", "lang": "fr"},
  {"text": "Wasser kocht bei hundert Grad.", "mode": "multilingual", "lang": "de"},
  {"text": "The sun rises in the east.", "mode": "multilingual", "lang": "en"},
  {"text": "El sol sale por el este.", "mode": "multilingual", "lang": "es"},
  {"text": "Le soleil se lève à l'est.", "mode": "multilingual", "lang": "fr"},
  {"text": "Die Sonne geht im Osten auf.", "mode": "multilingual", "lang": "de"},
  {"text": "All humans need water to survive.", "mode": "multilingual", "lang": "en"},
  {"text": "Todos los humanos necesitan agua para sobrevivir.", "mode": "multilingual", "lang": "es"},
  {"text": "Tous les humains ont besoin d'eau pour survivre.", "mode": "multilingual", "lang": "fr"},
  {"text": "Alle Menschen brauchen Wasser zum Überleben.", "mode": "multilingual", "lang": "de"},
  {"text": "Mathematics is the language of nature.", "mode": "multilingual", "lang": "en"},
  {"text": "Las matemáticas son el lenguaje de la naturaleza.", "mode": "multilingual", "lang": "es"},
  {"text": "Les mathématiques sont le langage de la nature.", "mode": "multilingual", "lang": "fr"},
  {"text": "Mathematik ist die Sprache der Natur.", "mode": "multilingual", "lang": "de"},
  {"text": "Time moves in one direction.", "mode": "multilingual", "lang": "en"},
  {"text": "El tiempo se mueve en una dirección.", "mode": "multilingual", "lang": "es"},
  {"text": "Le temps se déplace dans une direction.", "mode": "multilingual", "lang": "fr"},
  {"text": "Die Zeit bewegt sich in eine Richtung.", "mode": "multilingual", "lang": "de"}
]
```

- [ ] **Step 6: Create loader.py — unified prompt loader**

```python
# products/topological-router/prompts/loader.py
"""Load prompt datasets for adaptive exploration."""
import json
from pathlib import Path

PROMPT_DIR = Path(__file__).parent


def load_prompts(mode: str | None = None) -> list[dict]:
    """Load prompts, optionally filtered by mode.

    Modes: physics, narrative, logic, shuffled, multilingual
    If mode is None, load all.
    """
    all_prompts = []
    for f in PROMPT_DIR.glob("*.json"):
        if f.name == "loader.py":
            continue
        with open(f) as fh:
            prompts = json.load(fh)
            all_prompts.extend(prompts)

    if mode is not None:
        all_prompts = [p for p in all_prompts if p["mode"] == mode]

    return all_prompts


def load_by_mode() -> dict[str, list[dict]]:
    """Load all prompts grouped by mode."""
    all_p = load_prompts()
    modes = {}
    for p in all_p:
        m = p["mode"]
        if m not in modes:
            modes[m] = []
        modes[m].append(p)
    return modes
```

- [ ] **Step 7: Verify all prompts load**

Run: `cd /home/wb1/Desktop/Dev/atft-problems && python3 -c "
import sys; sys.path.insert(0, 'products/topological-router')
from prompts.loader import load_prompts, load_by_mode
all_p = load_prompts()
by_mode = load_by_mode()
print(f'Total prompts: {len(all_p)}')
for m, ps in sorted(by_mode.items()):
    print(f'  {m}: {len(ps)}')
"`

Expected: 100 total, 20 per mode.

- [ ] **Step 8: Commit**

```bash
cd /home/wb1/Desktop/Dev/atft-problems
git add products/topological-router/prompts/ products/topological-router/semantic_primes.py
git commit -m "feat: prompt datasets (5 cognitive modes) + semantic primes"
```

---

### Task 3: Module 1 — Prime Basis Seed

**Files:**
- Create: `products/topological-router/prime_basis.py`

**References:**
- Spec: Module 1, lines 19-36
- Data: `semantic_primes.py` (ALL_PRIMES)
- Measures: `topo_measures.py`

- [ ] **Step 1: Write prime_basis.py**

```python
# products/topological-router/prime_basis.py
"""Module 1: Compute prime subspace from Wierzbicka's 59 semantic primes.

For each prime phrase, extracts the model's contextual hidden state (last token
at target layer), then builds an orthonormal prime subspace via SVD.
Projects the full vocabulary to measure prime_ratio per token.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from semantic_primes import ALL_PRIMES, PRIME_TO_CATEGORY

OUTPUT_DIR = Path(__file__).parent / "results" / "basis_discovery"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_prime_basis(model, tokenizer, device="cuda") -> dict:
    """Compute the prime subspace from 59 semantic primes.

    Args:
        model: HuggingFace causal LM with output_hidden_states=True
        tokenizer: corresponding tokenizer

    Returns:
        dict with prime_basis (59 x hidden_dim), prime_vectors, target_layer
    """
    n_layers = model.config.num_hidden_layers
    target_layer = n_layers // 3  # consistent with spec

    prime_vectors = []  # 59 vectors, one per prime

    for prime in ALL_PRIMES:
        inputs = tokenizer(prime, return_tensors="pt", truncation=True,
                          max_length=64).to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"],
                          output_hidden_states=True)

        # Last token, target layer
        hs = outputs.hidden_states[target_layer]  # (1, seq_len, hidden_dim)
        vec = hs[0, -1, :].cpu().float()  # (hidden_dim,)
        prime_vectors.append(vec)

    # Stack: (59, hidden_dim)
    prime_matrix = torch.stack(prime_vectors)

    # Mean-center
    prime_centered = prime_matrix - prime_matrix.mean(dim=0)

    # SVD -> orthonormal basis
    U, S, Vt = torch.linalg.svd(prime_centered, full_matrices=False)
    # Vt: (59, hidden_dim) — the orthonormal basis vectors
    prime_basis = Vt  # rows are basis vectors

    return {
        "prime_basis": prime_basis,           # (59, hidden_dim)
        "singular_values": S,                 # (59,)
        "prime_vectors": prime_matrix,        # (59, hidden_dim) raw vectors
        "target_layer": target_layer,
        "hidden_dim": prime_matrix.shape[1],
    }


def project_vocabulary(model, tokenizer, prime_basis: torch.Tensor,
                       target_layer: int, device="cuda",
                       batch_size: int = 64) -> dict:
    """Project vocabulary tokens onto prime subspace.

    Args:
        prime_basis: (59, hidden_dim) orthonormal basis
        target_layer: which layer to extract from
        batch_size: tokens per batch for hidden state extraction

    Returns:
        dict with prime_ratios, residual norms, per-token stats
    """
    vocab_size = len(tokenizer)
    P = prime_basis.to(device)  # (59, hidden_dim)

    prime_ratios = []
    residual_norms = []

    # Process vocabulary in batches
    all_token_ids = list(range(vocab_size))

    for batch_start in range(0, vocab_size, batch_size):
        batch_ids = all_token_ids[batch_start:batch_start + batch_size]

        # Create single-token inputs
        input_ids = torch.tensor([[tid] for tid in batch_ids], device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

        # Extract hidden states at target layer: (batch, 1, hidden_dim)
        hs = outputs.hidden_states[target_layer][:, 0, :].float()  # (batch, hidden_dim)

        # Project onto prime subspace: projection = P^T @ P @ embedding
        proj = hs @ P.T @ P  # (batch, hidden_dim)
        residual = hs - proj

        # Compute ratios
        hs_norm = torch.norm(hs, dim=1)
        proj_norm = torch.norm(proj, dim=1)
        res_norm = torch.norm(residual, dim=1)

        ratio = proj_norm / (hs_norm + 1e-10)

        prime_ratios.extend(ratio.cpu().tolist())
        residual_norms.extend(res_norm.cpu().tolist())

        if (batch_start // batch_size) % 100 == 0 and batch_start > 0:
            print(f"  Vocabulary projection: {batch_start}/{vocab_size}")

    prime_ratios = np.array(prime_ratios)
    residual_norms = np.array(residual_norms)

    return {
        "prime_ratios": prime_ratios,
        "residual_norms": residual_norms,
        "mean_prime_ratio": float(np.mean(prime_ratios)),
        "std_prime_ratio": float(np.std(prime_ratios)),
        "median_prime_ratio": float(np.median(prime_ratios)),
        "vocab_size": vocab_size,
    }


def run_module1(model_name: str, device: str = "cuda",
                use_awq: bool = False) -> dict:
    """Run Module 1 end-to-end for a single model."""
    print(f"\n  Module 1: Prime Basis — {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if use_awq or "AWQ" in model_name:
        from awq import AutoAWQForCausalLM
        awq = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False)
        model = awq.model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            dtype=torch.float16, device_map=device,
            output_hidden_states=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Compute prime basis
    basis_result = compute_prime_basis(model, tokenizer, device)
    print(f"    Prime basis: {basis_result['prime_basis'].shape}")
    print(f"    Target layer: {basis_result['target_layer']}")
    print(f"    Top-5 singular values: {basis_result['singular_values'][:5].tolist()}")

    # Step 2: Project vocabulary
    vocab_result = project_vocabulary(
        model, tokenizer, basis_result["prime_basis"],
        basis_result["target_layer"], device)
    print(f"    Vocab size: {vocab_result['vocab_size']}")
    print(f"    Mean prime_ratio: {vocab_result['mean_prime_ratio']:.4f}")
    print(f"    Std prime_ratio: {vocab_result['std_prime_ratio']:.4f}")

    elapsed = time.time() - t0
    print(f"    Time: {elapsed:.1f}s")

    # Save
    model_short = model_name.split("/")[-1]
    save_path = OUTPUT_DIR / f"prime_basis_{model_short}.pt"
    torch.save({
        "prime_basis": basis_result["prime_basis"],
        "singular_values": basis_result["singular_values"],
        "prime_vectors": basis_result["prime_vectors"],
        "target_layer": basis_result["target_layer"],
        "hidden_dim": basis_result["hidden_dim"],
        "vocab_stats": vocab_result,
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, save_path)
    print(f"    Saved: {save_path}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {**basis_result, **vocab_result, "time_s": elapsed}


if __name__ == "__main__":
    result = run_module1("Qwen/Qwen2.5-0.5B")
```

- [ ] **Step 2: Test on smallest model**

Run: `cd /home/wb1/Desktop/Dev/atft-problems && python3 products/topological-router/prime_basis.py`

Expected output:
- Prime basis shape (59, hidden_dim)
- Mean prime_ratio between 0.01 and 0.30
- Saved .pt file in results/basis_discovery/
- Completes in < 5 minutes

- [ ] **Step 3: Commit**

```bash
cd /home/wb1/Desktop/Dev/atft-problems
git add products/topological-router/prime_basis.py
git commit -m "feat: Module 1 — prime basis seed from 59 semantic primes"
```

---

### Task 4: Module 2 — Adaptive Explorer

**Files:**
- Create: `products/topological-router/adaptive_explorer.py`

**References:**
- Spec: Module 2, lines 38-69
- Inputs: `prime_basis.py` output, `prompts/loader.py`
- Measures: `topo_measures.py`

- [ ] **Step 1: Write adaptive_explorer.py**

```python
# products/topological-router/adaptive_explorer.py
"""Module 2: Adaptive basis discovery via bandit-style prompt exploration.

Starts from the prime basis (59 vectors), iteratively expands by finding
directions in hidden space not covered by the current basis. Uses bandit-style
selection: each iteration samples from the cognitive mode with the highest
residual in the previous iteration.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from topo_measures import gini_fast
from prompts.loader import load_by_mode

OUTPUT_DIR = Path(__file__).parent / "results" / "basis_discovery"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Convergence parameters
GINI_EPSILON = 0.005
CONVERGENCE_PATIENCE = 3
MAX_ITERATIONS = 25
RESIDUAL_THRESHOLD = 0.01  # minimum residual magnitude to add new vectors
NEW_VECTORS_PER_ITER = 5   # max new basis vectors per iteration


def extract_hidden_states(model, tokenizer, prompts: list[str],
                          target_layer: int, device: str = "cuda") -> torch.Tensor:
    """Extract hidden states at target layer for a batch of prompts.

    Returns: stacked tensor of shape (total_tokens, hidden_dim), mean-centered.
    """
    all_hs = []

    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=512).to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"],
                          output_hidden_states=True)

        hs = outputs.hidden_states[target_layer][0]  # (seq_len, hidden_dim)
        all_hs.append(hs.cpu().float())

    # Stack all tokens
    stacked = torch.cat(all_hs, dim=0)  # (total_tokens, hidden_dim)

    # Mean-center
    stacked = stacked - stacked.mean(dim=0)

    return stacked


def compute_residual(hidden_states: torch.Tensor,
                     basis: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Project hidden states onto current basis and compute residual.

    Args:
        hidden_states: (n_tokens, hidden_dim)
        basis: (n_basis, hidden_dim) orthonormal

    Returns:
        (residual tensor, mean residual magnitude)
    """
    # Project: proj = H @ B^T @ B
    proj = hidden_states @ basis.T @ basis
    residual = hidden_states - proj

    res_norms = torch.norm(residual, dim=1)
    mean_res = float(res_norms.mean())

    return residual, mean_res


def expand_basis(basis: torch.Tensor, residual: torch.Tensor,
                 n_new: int = NEW_VECTORS_PER_ITER) -> torch.Tensor:
    """Find top-k directions in residual space and add to basis.

    Args:
        basis: (n_basis, hidden_dim) current orthonormal basis
        residual: (n_tokens, hidden_dim) residual after projection
        n_new: max new vectors to add

    Returns:
        expanded basis (n_basis + n_added, hidden_dim), orthonormalized
    """
    # SVD of residual to find dominant unexplained directions
    U, S, Vt = torch.linalg.svd(residual, full_matrices=False)

    # Take top-n_new directions (by singular value magnitude)
    new_vectors = Vt[:n_new]

    # Filter by threshold
    sig_threshold = S[0] * RESIDUAL_THRESHOLD if len(S) > 0 else 0
    mask = S[:n_new] > sig_threshold
    new_vectors = new_vectors[mask]

    if len(new_vectors) == 0:
        return basis

    # Concatenate and re-orthogonalize
    combined = torch.cat([basis, new_vectors], dim=0)

    # QR decomposition for orthonormalization
    Q, R = torch.linalg.qr(combined.T)
    # Keep only columns with non-trivial norm (avoid numerical rank deficiency)
    norms = torch.norm(R, dim=1)
    valid = norms > 1e-8
    expanded = Q[:, valid].T  # (n_valid, hidden_dim)

    return expanded


def adaptive_explore(model, tokenizer, prime_basis: torch.Tensor,
                     target_layer: int, device: str = "cuda") -> dict:
    """Run the adaptive basis discovery loop.

    Args:
        prime_basis: (59, hidden_dim) from Module 1
        target_layer: which layer to extract from

    Returns:
        dict with adaptive_basis, convergence_trajectory, basis_growth_log
    """
    prompts_by_mode = load_by_mode()
    modes = list(prompts_by_mode.keys())

    # Track which prompts have been used per mode
    used = {m: set() for m in modes}

    basis = prime_basis.clone()

    convergence_trajectory = []
    residual_history = []
    basis_growth_log = []
    mode_residuals = {m: 0.0 for m in modes}

    prev_gini = None
    stable_count = 0

    for iteration in range(MAX_ITERATIONS):
        # ── Select prompts via bandit policy ──
        if iteration == 0:
            # First iteration: one prompt from each mode
            batch_prompts = []
            batch_modes = []
            for m in modes:
                available = [i for i in range(len(prompts_by_mode[m]))
                            if i not in used[m]]
                if available:
                    idx = available[0]
                    used[m].add(idx)
                    batch_prompts.append(prompts_by_mode[m][idx]["text"])
                    batch_modes.append(m)
        else:
            # Bandit: sample from mode with highest residual
            best_mode = max(mode_residuals, key=mode_residuals.get)
            available = [i for i in range(len(prompts_by_mode[best_mode]))
                        if i not in used[best_mode]]
            if not available:
                # Fallback: try other modes
                for m in modes:
                    available = [i for i in range(len(prompts_by_mode[m]))
                                if i not in used[m]]
                    if available:
                        best_mode = m
                        break

            if not available:
                print(f"    Iteration {iteration}: all prompts exhausted")
                break

            batch_prompts = []
            batch_modes = []
            n_batch = min(5, len(available))
            for idx in available[:n_batch]:
                used[best_mode].add(idx)
                batch_prompts.append(prompts_by_mode[best_mode][idx]["text"])
                batch_modes.append(best_mode)

        # ── Extract hidden states ──
        hs = extract_hidden_states(model, tokenizer, batch_prompts,
                                   target_layer, device)

        # ── Compute residual ──
        residual, mean_res = compute_residual(hs, basis)

        # Track per-mode residual
        for m in set(batch_modes):
            mode_residuals[m] = mean_res  # simplified: use batch mean

        # ── Expand basis if residual is significant ──
        old_size = basis.shape[0]
        basis = expand_basis(basis, residual)
        new_size = basis.shape[0]
        n_added = new_size - old_size

        # ── Convergence check ──
        # Gini of singular values of the basis when applied to recent hidden states
        proj_on_basis = hs @ basis.T  # (n_tokens, n_basis)
        sv = torch.linalg.svdvals(proj_on_basis)
        gini = gini_fast(sv.numpy())

        convergence_trajectory.append(float(gini))
        residual_history.append(float(mean_res))
        basis_growth_log.append({
            "iteration": iteration,
            "n_added": n_added,
            "basis_size": new_size,
            "gini": float(gini),
            "mean_residual": float(mean_res),
            "mode": batch_modes[0] if batch_modes else "none",
            "n_prompts": len(batch_prompts),
        })

        # Check stability
        if prev_gini is not None and abs(gini - prev_gini) < GINI_EPSILON:
            stable_count += 1
        else:
            stable_count = 0
        prev_gini = gini

        print(f"    Iter {iteration:2d}: basis={new_size:3d} (+{n_added}) "
              f"gini={gini:.4f} res={mean_res:.4f} mode={batch_modes[0] if batch_modes else '?'}"
              f"{' CONVERGED' if stable_count >= CONVERGENCE_PATIENCE else ''}")

        if stable_count >= CONVERGENCE_PATIENCE:
            print(f"    Converged at iteration {iteration}")
            break

    return {
        "adaptive_basis": basis,
        "convergence_trajectory": convergence_trajectory,
        "residual_history": residual_history,
        "basis_growth_log": basis_growth_log,
        "n_iterations": len(convergence_trajectory),
        "converged": stable_count >= CONVERGENCE_PATIENCE,
        "final_gini": convergence_trajectory[-1] if convergence_trajectory else 0,
        "final_basis_size": basis.shape[0],
    }


def run_module2(model_name: str, prime_basis_path: str | Path,
                device: str = "cuda") -> dict:
    """Run Module 2 end-to-end."""
    print(f"\n  Module 2: Adaptive Explorer — {model_name}")
    t0 = time.time()

    # Load prime basis from Module 1
    data = torch.load(prime_basis_path, weights_only=False)
    prime_basis = data["prime_basis"]
    target_layer = data["target_layer"]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if "AWQ" in model_name:
        from awq import AutoAWQForCausalLM
        awq = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False)
        model = awq.model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            dtype=torch.float16, device_map=device,
            output_hidden_states=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    result = adaptive_explore(model, tokenizer, prime_basis, target_layer, device)

    elapsed = time.time() - t0
    print(f"    Converged: {result['converged']}")
    print(f"    Final basis: {result['final_basis_size']} vectors")
    print(f"    Final Gini: {result['final_gini']:.4f}")
    print(f"    Time: {elapsed:.1f}s")

    # Save
    model_short = model_name.split("/")[-1]
    save_path = OUTPUT_DIR / f"adaptive_basis_{model_short}.pt"
    torch.save({
        "adaptive_basis": result["adaptive_basis"],
        "convergence_trajectory": result["convergence_trajectory"],
        "residual_history": result["residual_history"],
        "basis_growth_log": result["basis_growth_log"],
        "metadata": {
            "model_name": model_name,
            "hidden_dim": result["adaptive_basis"].shape[1],
            "n_iterations": result["n_iterations"],
            "convergence_value": result["final_gini"],
            "converged": result["converged"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }, save_path)
    print(f"    Saved: {save_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {**result, "time_s": elapsed}


if __name__ == "__main__":
    # Requires Module 1 output
    basis_path = OUTPUT_DIR / "prime_basis_Qwen2.5-0.5B.pt"
    if basis_path.exists():
        run_module2("Qwen/Qwen2.5-0.5B", basis_path)
    else:
        print(f"Run Module 1 first: python3 prime_basis.py")
```

- [ ] **Step 2: Test on smallest model (requires Task 3 output)**

Run:
```bash
cd /home/wb1/Desktop/Dev/atft-problems
python3 products/topological-router/prime_basis.py  # Module 1 first
python3 products/topological-router/adaptive_explorer.py  # Then Module 2
```

Expected:
- Iterative output showing basis growth
- Convergence within 25 iterations (or exhausts prompts)
- Saved adaptive_basis .pt file
- Completes in < 15 minutes

- [ ] **Step 3: Commit**

```bash
cd /home/wb1/Desktop/Dev/atft-problems
git add products/topological-router/adaptive_explorer.py
git commit -m "feat: Module 2 — adaptive basis discovery with bandit exploration"
```

---

### Task 5: Module 3 — Topology Comparison

**Files:**
- Create: `products/topological-router/topology_comparison.py`

**References:**
- Spec: Module 3, lines 71-110
- Inputs: Module 1 + Module 2 outputs, clean bench results
- Measures: `topo_measures.py`

- [ ] **Step 1: Write topology_comparison.py**

```python
# products/topological-router/topology_comparison.py
"""Module 3: Four topology metrics comparing prime and residual subspaces.

Metric 1: Topological Isomorphism (H0 Gini comparison)
Metric 2: Cross-Subspace Coherence (sheaf Laplacian spectral sum)
Metric 3: Persistence Under Projection (truth signal in subspaces)
Metric 4: Adaptive Basis Convergence (representational fixed point)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).parent))
from topo_measures import h0_gini, gini_fast, spectral_gap, effective_rank

OUTPUT_DIR = Path(__file__).parent / "results" / "topology_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def metric1_topological_isomorphism(prime_basis: torch.Tensor,
                                      adaptive_basis: torch.Tensor,
                                      hidden_states: torch.Tensor) -> dict:
    """Metric 1: Compare H0 Gini in prime subspace vs residual subspace.

    Args:
        prime_basis: (59, hidden_dim)
        adaptive_basis: (n_basis, hidden_dim) — includes primes + discovered
        hidden_states: (n_tokens, hidden_dim) from diverse prompts
    """
    # Project onto prime subspace
    prime_proj = (hidden_states @ prime_basis.T).numpy()  # (n_tokens, 59)

    # Residual subspace = adaptive basis vectors NOT in the prime basis
    # Approximate: take the difference in basis size
    n_prime = prime_basis.shape[0]
    if adaptive_basis.shape[0] > n_prime:
        residual_basis = adaptive_basis[n_prime:]
        residual_proj = (hidden_states @ residual_basis.T).numpy()
    else:
        residual_proj = np.zeros((hidden_states.shape[0], 1))

    # H0 Gini on each projection
    gini_prime = h0_gini(prime_proj, max_n=200)
    gini_residual = h0_gini(residual_proj, max_n=200) if residual_proj.shape[1] > 1 else 0.0

    # Are they similar? (within 20% relative)
    if gini_prime > 0 and gini_residual > 0:
        relative_diff = abs(gini_prime - gini_residual) / max(gini_prime, gini_residual)
        isomorphic = relative_diff < 0.20
    else:
        isomorphic = False
        relative_diff = 1.0

    return {
        "gini_prime": float(gini_prime),
        "gini_residual": float(gini_residual),
        "relative_diff": float(relative_diff),
        "isomorphic": isomorphic,
    }


def metric2_cross_subspace_coherence(prime_basis: torch.Tensor,
                                       adaptive_basis: torch.Tensor,
                                       hidden_states: torch.Tensor,
                                       n_positions: int = 200,
                                       k_neighbors: int = 10) -> dict:
    """Metric 2: Sheaf Laplacian spectral sum measuring prime-residual coupling.

    Lightweight implementation: kNN graph on token positions, fiber = prime
    projection, transport = residual correlation. Spectral sum via direct
    eigenvalue computation (small graph).
    """
    n_tokens = hidden_states.shape[0]
    n_prime = prime_basis.shape[0]

    # Subsample
    if n_tokens > n_positions:
        idx = np.random.default_rng(42).choice(n_tokens, n_positions, replace=False)
        hs = hidden_states[idx]
    else:
        hs = hidden_states
        n_positions = n_tokens

    # Prime-space projections (fibers)
    fibers = (hs @ prime_basis.T).numpy()  # (n_pos, 59)

    # kNN graph based on full hidden state distances
    dists = torch.cdist(hs, hs).numpy()

    # For each node, find k nearest neighbors
    # Build adjacency and sheaf Laplacian directly
    # L_F = sum over edges (e_ij): (f_i - T_ij f_j)(f_i - T_ij f_j)^T
    # where T_ij is the transport from j to i

    # Simple transport: correlation of residual projections between neighbors
    if adaptive_basis.shape[0] > n_prime:
        res_basis = adaptive_basis[n_prime:]
        res_proj = (hs @ res_basis.T).numpy()  # (n_pos, n_res)
    else:
        # No residual basis — coherence is trivially 0
        return {"spectral_sum": 0.0, "high_coherence": True,
                "n_positions": n_positions, "note": "no residual basis"}

    # Compute spectral sum as mean squared fiber difference across kNN edges
    # (simplified sheaf Laplacian trace)
    total_diff = 0.0
    n_edges = 0

    for i in range(n_positions):
        # k nearest neighbors of i
        neighbors = np.argsort(dists[i])[1:k_neighbors + 1]
        for j in neighbors:
            # Transport: scale fiber_j by residual correlation
            res_corr = np.dot(res_proj[i], res_proj[j]) / (
                np.linalg.norm(res_proj[i]) * np.linalg.norm(res_proj[j]) + 1e-10)
            transported = fibers[j] * res_corr
            diff = fibers[i] - transported
            total_diff += np.dot(diff, diff)
            n_edges += 1

    spectral_sum = total_diff / n_edges if n_edges > 0 else 0.0

    # Normalize by fiber variance for interpretability
    fiber_var = np.var(fibers)
    normalized = spectral_sum / (fiber_var + 1e-10)

    return {
        "spectral_sum": float(spectral_sum),
        "normalized_spectral_sum": float(normalized),
        "high_coherence": normalized < 2.0,  # threshold: coupling is strong
        "n_positions": n_positions,
        "n_edges": n_edges,
    }


def metric3_persistence_under_projection(prime_basis: torch.Tensor,
                                           adaptive_basis: torch.Tensor,
                                           correct_hs: torch.Tensor,
                                           wrong_hs: torch.Tensor) -> dict:
    """Metric 3: Does the truth signal survive projection?

    Args:
        correct_hs: hidden states from correctly-answered questions
        wrong_hs: hidden states from incorrectly-answered questions
    """
    n_prime = prime_basis.shape[0]

    results = {}

    for name, basis in [("prime", prime_basis),
                        ("full", adaptive_basis)]:
        # Project
        correct_proj = (correct_hs @ basis.T).numpy()
        wrong_proj = (wrong_hs @ basis.T).numpy()

        # Compute spectral gap in each projection
        sg_correct = spectral_gap(torch.tensor(correct_proj))
        sg_wrong = spectral_gap(torch.tensor(wrong_proj))

        # H0 Gini
        gini_correct = h0_gini(correct_proj, max_n=150)
        gini_wrong = h0_gini(wrong_proj, max_n=150)

        results[name] = {
            "sg_correct": float(sg_correct),
            "sg_wrong": float(sg_wrong),
            "sg_diff": float(sg_correct - sg_wrong),
            "gini_correct": float(gini_correct),
            "gini_wrong": float(gini_wrong),
            "gini_diff": float(gini_correct - gini_wrong),
            "signal_survives": abs(sg_correct - sg_wrong) > 0.1,
        }

    # Residual projection
    if adaptive_basis.shape[0] > n_prime:
        res_basis = adaptive_basis[n_prime:]
        correct_res = (correct_hs @ res_basis.T).numpy()
        wrong_res = (wrong_hs @ res_basis.T).numpy()
        sg_c = spectral_gap(torch.tensor(correct_res))
        sg_w = spectral_gap(torch.tensor(wrong_res))
        gini_c = h0_gini(correct_res, max_n=150)
        gini_w = h0_gini(wrong_res, max_n=150)
        results["residual"] = {
            "sg_correct": float(sg_c), "sg_wrong": float(sg_w),
            "sg_diff": float(sg_c - sg_w),
            "gini_correct": float(gini_c), "gini_wrong": float(gini_w),
            "gini_diff": float(gini_c - gini_w),
            "signal_survives": abs(sg_c - sg_w) > 0.1,
        }
    else:
        results["residual"] = {"signal_survives": False, "note": "no residual basis"}

    return results


def metric4_convergence(convergence_trajectory: list[float]) -> dict:
    """Metric 4: Adaptive basis convergence analysis."""
    if not convergence_trajectory:
        return {"converged": False, "final_gini": 0}

    traj = np.array(convergence_trajectory)
    diffs = np.abs(np.diff(traj))

    return {
        "converged": len(diffs) >= 3 and all(diffs[-3:] < 0.005),
        "final_gini": float(traj[-1]),
        "n_iterations": len(traj),
        "trajectory": convergence_trajectory,
        "mean_change_last_3": float(np.mean(diffs[-3:])) if len(diffs) >= 3 else float("inf"),
    }


def classify_outcome(m1: dict, m2: dict) -> dict:
    """Classify into the 2x2 outcome table."""
    gini_similar = m1["isomorphic"]
    high_coherence = m2["high_coherence"]

    if gini_similar and high_coherence:
        cell = "latent_truth"
        interpretation = "Residual = latent truth dimensions. Complete basis found."
        action = "Build multi-channel tokenizer on this basis."
    elif gini_similar and not high_coherence:
        cell = "independent_manifolds"
        interpretation = "Two independent truth manifolds. Model sees what primes don't."
        action = "Investigate residual semantics — new primitives?"
    elif not gini_similar and high_coherence:
        cell = "transformation"
        interpretation = "Residual is a transformation of prime structure."
        action = "Characterize the transformation (rotation? scaling?)."
    else:
        cell = "noise"
        interpretation = "Residual is noise/format artifacts. Primes are sufficient."
        action = "Multi-channel tokenizer needs only prime dimensions."

    return {"cell": cell, "interpretation": interpretation, "action": action}


def run_module3(model_name: str, device: str = "cuda") -> dict:
    """Run Module 3 end-to-end. Requires Modules 1 and 2 output."""
    print(f"\n  Module 3: Topology Comparison — {model_name}")
    t0 = time.time()

    model_short = model_name.split("/")[-1]
    basis_dir = Path(__file__).parent / "results" / "basis_discovery"

    # Load Module 1 output
    m1_path = basis_dir / f"prime_basis_{model_short}.pt"
    m1_data = torch.load(m1_path, weights_only=False)
    prime_basis = m1_data["prime_basis"]
    target_layer = m1_data["target_layer"]

    # Load Module 2 output
    m2_path = basis_dir / f"adaptive_basis_{model_short}.pt"
    m2_data = torch.load(m2_path, weights_only=False)
    adaptive_basis = m2_data["adaptive_basis"]
    convergence_traj = m2_data["convergence_trajectory"]

    # Load model to extract hidden states for metrics
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if "AWQ" in model_name:
        from awq import AutoAWQForCausalLM
        awq = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False)
        model = awq.model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            dtype=torch.float16, device_map=device,
            output_hidden_states=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get diverse hidden states for Metrics 1 and 2
    from prompts.loader import load_prompts
    all_prompts = load_prompts()
    prompt_texts = [p["text"] for p in all_prompts[:50]]  # use 50 for speed

    from adaptive_explorer import extract_hidden_states
    hs = extract_hidden_states(model, tokenizer, prompt_texts, target_layer, device)

    # Metric 1
    print("    Computing Metric 1: Topological Isomorphism...")
    m1_result = metric1_topological_isomorphism(prime_basis, adaptive_basis, hs)
    print(f"      Gini(prime)={m1_result['gini_prime']:.4f}, "
          f"Gini(residual)={m1_result['gini_residual']:.4f}, "
          f"isomorphic={m1_result['isomorphic']}")

    # Metric 2
    print("    Computing Metric 2: Cross-Subspace Coherence...")
    m2_result = metric2_cross_subspace_coherence(prime_basis, adaptive_basis, hs)
    print(f"      Normalized spectral sum: {m2_result['normalized_spectral_sum']:.4f}, "
          f"coherent={m2_result['high_coherence']}")

    # Metric 3 — needs correct/incorrect hidden states
    # Try to load from clean bench results
    bench_dir = Path(__file__).parent / "results"
    bench_file = bench_dir / f"clean_bench_mmlu_physics_{model_short}.json"

    if bench_file.exists():
        print("    Computing Metric 3: Persistence Under Projection...")
        with open(bench_file) as f:
            bench_data = json.load(f)

        # Extract hidden states for correct vs incorrect questions
        correct_texts = []
        wrong_texts = []
        for r in bench_data.get("results", []):
            q = r.get("question", "")
            if r.get("lk_correct", False):
                correct_texts.append(f"Q: {q}")
            else:
                wrong_texts.append(f"Q: {q}")

        if len(correct_texts) >= 5 and len(wrong_texts) >= 5:
            correct_hs = extract_hidden_states(
                model, tokenizer, correct_texts[:30], target_layer, device)
            wrong_hs = extract_hidden_states(
                model, tokenizer, wrong_texts[:30], target_layer, device)
            m3_result = metric3_persistence_under_projection(
                prime_basis, adaptive_basis, correct_hs, wrong_hs)
        else:
            m3_result = {"note": "insufficient correct/incorrect split"}
    else:
        print(f"    Metric 3: skipped (no clean bench results for {model_short})")
        m3_result = {"note": f"no clean bench results at {bench_file}"}

    # Metric 4
    print("    Computing Metric 4: Convergence...")
    m4_result = metric4_convergence(convergence_traj)
    print(f"      Converged: {m4_result['converged']}, "
          f"final Gini: {m4_result['final_gini']:.4f}")

    # Classify outcome
    outcome = classify_outcome(m1_result, m2_result)
    print(f"    Outcome: {outcome['cell']} — {outcome['interpretation']}")

    elapsed = time.time() - t0

    # Save
    save_data = {
        "model": model_name,
        "metric1_isomorphism": m1_result,
        "metric2_coherence": m2_result,
        "metric3_persistence": m3_result,
        "metric4_convergence": m4_result,
        "outcome": outcome,
        "time_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_path = OUTPUT_DIR / f"metrics_{model_short}.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"    Saved: {save_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return save_data


if __name__ == "__main__":
    run_module3("Qwen/Qwen2.5-0.5B")
```

- [ ] **Step 2: Test on smallest model (requires Tasks 3+4 output)**

Run:
```bash
cd /home/wb1/Desktop/Dev/atft-problems
python3 products/topological-router/topology_comparison.py
```

Expected: Four metrics computed, outcome classified, JSON saved.

- [ ] **Step 3: Commit**

```bash
cd /home/wb1/Desktop/Dev/atft-problems
git add products/topological-router/topology_comparison.py
git commit -m "feat: Module 3 — four topology metrics for prime vs residual subspaces"
```

---

### Task 6: Module 4 — Sweep Runner + Integration

**Files:**
- Create: `products/topological-router/sweep_runner.py`

- [ ] **Step 1: Write sweep_runner.py**

```python
# products/topological-router/sweep_runner.py
"""Module 4: Cross-architecture sweep.

Runs Modules 1-3 on all 6 models, produces cross-architecture comparison.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from prime_basis import run_module1
from adaptive_explorer import run_module2
from topology_comparison import run_module3

OUTPUT_DIR = Path(__file__).parent / "results"
BASIS_DIR = OUTPUT_DIR / "basis_discovery"

MODELS = [
    {"name": "HuggingFaceTB/SmolLM2-360M-Instruct", "family": "SmolLM2", "size": 0.36},
    {"name": "Qwen/Qwen2.5-0.5B", "family": "Qwen2.5", "size": 0.5},
    {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "TinyLlama", "size": 1.1},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct", "family": "Qwen2.5", "size": 1.5},
    {"name": "Qwen/Qwen2.5-3B-Instruct", "family": "Qwen2.5", "size": 3.0},
    {"name": "Qwen/Qwen2.5-7B-Instruct-AWQ", "family": "Qwen2.5", "size": 7.0},
]


def run_sweep():
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 65)
    print("  TOKENIZATION THESIS — CROSS-ARCHITECTURE SWEEP")
    print(f"  {ts}")
    print("=" * 65)

    all_results = []

    for mi, minfo in enumerate(MODELS):
        model_name = minfo["name"]
        model_short = model_name.split("/")[-1]
        print(f"\n{'='*65}")
        print(f"  [{mi+1}/{len(MODELS)}] {model_name} ({minfo['size']}B)")
        print(f"{'='*65}")

        try:
            # Module 1
            m1 = run_module1(model_name)

            # Module 2
            basis_path = BASIS_DIR / f"prime_basis_{model_short}.pt"
            m2 = run_module2(model_name, basis_path)

            # Module 3
            m3 = run_module3(model_name)

            all_results.append({
                "model": model_name,
                "family": minfo["family"],
                "size_B": minfo["size"],
                "prime_ratio_mean": m1.get("mean_prime_ratio", 0),
                "adaptive_basis_size": m2.get("final_basis_size", 0),
                "converged": m2.get("converged", False),
                "convergence_gini": m2.get("final_gini", 0),
                "outcome_cell": m3.get("outcome", {}).get("cell", "unknown"),
                "gini_prime": m3.get("metric1_isomorphism", {}).get("gini_prime", 0),
                "gini_residual": m3.get("metric1_isomorphism", {}).get("gini_residual", 0),
                "coherence": m3.get("metric2_coherence", {}).get("normalized_spectral_sum", 0),
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({
                "model": model_name, "family": minfo["family"],
                "size_B": minfo["size"], "error": str(e),
            })

    # Summary
    print(f"\n{'='*90}")
    print("  CROSS-ARCHITECTURE SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Model':<30} {'Size':>5} {'PR%':>6} {'Basis':>6} {'Conv':>5} "
          f"{'Gini':>6} {'Cell':<20}")
    print(f"  {'-'*85}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['model'].split('/')[-1]:<30} {r['size_B']:>5.1f} FAILED: {r['error'][:30]}")
        else:
            print(f"  {r['model'].split('/')[-1]:<30} {r['size_B']:>5.1f} "
                  f"{r['prime_ratio_mean']:>5.3f} {r['adaptive_basis_size']:>6d} "
                  f"{'Y' if r['converged'] else 'N':>5} "
                  f"{r['convergence_gini']:>5.3f} {r['outcome_cell']:<20}")

    # Save
    summary = {"timestamp": ts, "results": all_results}
    with open(OUTPUT_DIR / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIR / 'sweep_summary.json'}")


if __name__ == "__main__":
    run_sweep()
```

- [ ] **Step 2: Smoke test on one model**

Run: `cd /home/wb1/Desktop/Dev/atft-problems && python3 -c "
import sys; sys.path.insert(0, 'products/topological-router')
from sweep_runner import MODELS
print(f'{len(MODELS)} models configured')
for m in MODELS:
    print(f'  {m[\"name\"]} ({m[\"size\"]}B, {m[\"family\"]})')
"`

- [ ] **Step 3: Commit**

```bash
cd /home/wb1/Desktop/Dev/atft-problems
git add products/topological-router/sweep_runner.py
git commit -m "feat: Module 4 — cross-architecture sweep runner"
```

- [ ] **Step 4: Run full pipeline on Qwen2.5-0.5B as integration test**

```bash
cd /home/wb1/Desktop/Dev/atft-problems
python3 products/topological-router/prime_basis.py        # Module 1
python3 products/topological-router/adaptive_explorer.py  # Module 2
python3 products/topological-router/topology_comparison.py # Module 3
```

Verify:
- `results/basis_discovery/prime_basis_Qwen2.5-0.5B.pt` exists
- `results/basis_discovery/adaptive_basis_Qwen2.5-0.5B.pt` exists
- `results/topology_comparison/metrics_Qwen2.5-0.5B.json` exists with all 4 metrics
- Outcome classification is one of: latent_truth, independent_manifolds, transformation, noise

- [ ] **Step 5: Commit integration test results**

```bash
cd /home/wb1/Desktop/Dev/atft-problems
git add products/topological-router/results/
git commit -m "experiment: integration test — Modules 1-3 on Qwen2.5-0.5B"
```

- [ ] **Step 6: Run full 6-model sweep**

```bash
cd /home/wb1/Desktop/Dev/atft-problems
python3 products/topological-router/sweep_runner.py
```

Expected: ~3-4 hours total. Results in `results/sweep_summary.json`.

- [ ] **Step 7: Commit sweep results**

```bash
cd /home/wb1/Desktop/Dev/atft-problems
git add products/topological-router/results/
git commit -m "experiment: tokenization thesis — 6-model cross-architecture sweep"
```
