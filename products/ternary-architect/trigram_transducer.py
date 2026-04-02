#!/usr/bin/env python3
"""
Trigram Structural Transducer — 27-state universal tokenization.

Each character → {0, 1, 3} classification.
Each trigram → 3-element vector in {0,1,3}³ = 27 states.
No learned vocabulary. No BPE. Structure IS the encoding.

Classification:
  0 (void):     whitespace, punctuation, boundaries
  1 (identity): lowercase letters, digits, standard flow
  3 (prime):    uppercase letters, emphasis markers

27 states = 3³. Three values × three positions.
Universal across languages (with per-domain classification overrides).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset


def classify_char(c: str) -> int:
    """Classify a character into {0, 1, 3}.

    English baseline:
      0 = void: whitespace, punctuation, structural markers
      1 = identity: lowercase, digits, standard content
      3 = prime: uppercase, emphasis
    """
    if c in ' \t\n\r':
        return 0
    if c in '.,;:!?-–—()[]{}"\'/\\@#$%^&*~`|<>+=_':
        return 0
    if c.isupper():
        return 3
    return 1  # lowercase, digits, everything else


def text_to_structural_classes(text: str) -> list[int]:
    """Convert raw text to sequence of {0, 1, 3} structural classes."""
    return [classify_char(c) for c in text]


def structural_trigram_id(a: int, b: int, c: int) -> int:
    """Map a trigram of {0,1,3} classes to a unique ID in [0, 26].

    Encoding: map {0,1,3} → {0,1,2} then base-3.
    0→0, 1→1, 3→2.
    ID = a_idx * 9 + b_idx * 3 + c_idx
    """
    MAP = {0: 0, 1: 1, 3: 2}
    return MAP[a] * 9 + MAP[b] * 3 + MAP[c]


def trigram_to_classes(trigram_id: int) -> tuple[int, int, int]:
    """Reverse: ID → (a, b, c) structural classes."""
    RMAP = {0: 0, 1: 1, 2: 3}
    c_idx = trigram_id % 3
    b_idx = (trigram_id // 3) % 3
    a_idx = trigram_id // 9
    return RMAP[a_idx], RMAP[b_idx], RMAP[c_idx]


def encode_text(text: str) -> list[int]:
    """Encode text as sequence of 27-state structural trigram IDs.

    Returns list of trigram IDs, length = len(text) - 2.
    """
    classes = text_to_structural_classes(text)
    if len(classes) < 3:
        return []
    trigrams = []
    for i in range(len(classes) - 2):
        tid = structural_trigram_id(classes[i], classes[i+1], classes[i+2])
        trigrams.append(tid)
    return trigrams


class TrigramDataset(Dataset):
    """Dataset of structural trigram sequences from raw text.

    Chunks text into fixed-length trigram sequences.
    Target: predict next trigram (autoregressive).
    """

    def __init__(self, text: str, chunk_size: int = 256):
        trigrams = encode_text(text)
        self.chunks = []
        for i in range(0, len(trigrams) - chunk_size, chunk_size):
            self.chunks.append(trigrams[i:i + chunk_size])
        print(f"  TrigramDataset: {len(text)} chars → {len(trigrams)} trigrams → {len(self.chunks)} chunks of {chunk_size}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        toks = torch.tensor(self.chunks[idx], dtype=torch.long)
        return toks[:-1], toks[1:]  # input, target


def load_dataset_text(dataset_name: str, n_samples: int = 200000) -> str:
    """Load raw text from a named dataset."""
    from datasets import load_dataset as hf_load

    if dataset_name == "wikitext":
        ds = hf_load("wikitext", "wikitext-103-raw-v1", split="train", streaming=False)
        ds = ds.select(range(min(n_samples, len(ds))))
        return "\n".join(item["text"] for item in ds if item["text"].strip())

    elif dataset_name == "tinystories":
        ds = hf_load("roneneldan/TinyStories", split="train", streaming=False)
        ds = ds.select(range(min(n_samples, len(ds))))
        return "\n".join(item["text"] for item in ds if item.get("text", "").strip())

    elif dataset_name == "kant":
        kant_path = Path(__file__).parent / "data" / "kant_critique.txt"
        return kant_path.read_text()

    elif dataset_name == "animalfarm":
        af_path = Path(__file__).parent / "data" / "animal_farm.txt"
        return af_path.read_text()

    elif dataset_name == "korean":
        ds = hf_load("wikimedia/wikipedia", "20231101.ko", split="train", streaming=False)
        ds = ds.select(range(min(n_samples, len(ds))))
        return "\n".join(item["text"] for item in ds if item["text"].strip())

    elif dataset_name == "chinese":
        ds = hf_load("wikimedia/wikipedia", "20231101.zh", split="train", streaming=False)
        ds = ds.select(range(min(n_samples, len(ds))))
        return "\n".join(item["text"] for item in ds if item["text"].strip())

    elif dataset_name == "arabic":
        ds = hf_load("wikimedia/wikipedia", "20231101.ar", split="train", streaming=False)
        ds = ds.select(range(min(n_samples, len(ds))))
        return "\n".join(item["text"] for item in ds if item["text"].strip())

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ── Trigram distribution analysis ──

def trigram_distribution(text: str) -> dict:
    """Compute the structural trigram distribution of a text.

    Returns: frequency of each of the 27 trigram states,
    plus the derived crystal (aggregated void/identity/prime).
    """
    trigrams = encode_text(text)
    n = len(trigrams)
    if n == 0:
        return {}

    # Count each of 27 states
    counts = [0] * 27
    for t in trigrams:
        counts[t] += 1

    freq = {i: counts[i] / n for i in range(27)}

    # Aggregate crystal: average the {0,1,3} values across all trigrams
    total_void = 0
    total_ident = 0
    total_prime = 0
    for tid, f in freq.items():
        a, b, c = trigram_to_classes(tid)
        for val in (a, b, c):
            if val == 0:
                total_void += f
            elif val == 1:
                total_ident += f
            else:
                total_prime += f

    total = total_void + total_ident + total_prime
    crystal = {
        "void": total_void / total if total > 0 else 0,
        "identity": total_ident / total if total > 0 else 0,
        "prime": total_prime / total if total > 0 else 0,
    }

    return {"trigram_freq": freq, "crystal": crystal, "n_trigrams": n}


if __name__ == "__main__":
    # Test the 27-state encoding
    print("=" * 60)
    print("TRIGRAM STRUCTURAL TRANSDUCER — 27-state test")
    print("=" * 60)

    # Show all 27 states
    print("\n27 structural states:")
    for i in range(27):
        a, b, c = trigram_to_classes(i)
        label = f"({'void' if a==0 else 'ident' if a==1 else 'prime'},"
        label += f"{'void' if b==0 else 'ident' if b==1 else 'prime'},"
        label += f"{'void' if c==0 else 'ident' if c==1 else 'prime'})"
        print(f"  ID {i:>2d}: ({a},{b},{c}) = {label}")

    # Test encoding
    test = "The cat sat. A Dog!"
    classes = text_to_structural_classes(test)
    trigrams = encode_text(test)
    print(f"\nText: '{test}'")
    print(f"Classes: {classes}")
    print(f"Trigrams: {trigrams}")

    # Distribution
    dist = trigram_distribution(test)
    print(f"\nCrystal of test text:")
    print(f"  void={dist['crystal']['void']:.3f}")
    print(f"  identity={dist['crystal']['identity']:.3f}")
    print(f"  prime={dist['crystal']['prime']:.3f}")

    # Compare to character-level stats
    v = sum(1 for c in test if classify_char(c) == 0) / len(test)
    i = sum(1 for c in test if classify_char(c) == 1) / len(test)
    p = sum(1 for c in test if classify_char(c) == 3) / len(test)
    print(f"\nCharacter-level (for comparison):")
    print(f"  void={v:.3f} identity={i:.3f} prime={p:.3f}")

    print("\nOK — transducer works")
