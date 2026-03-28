"""Load prompt datasets for adaptive exploration."""
import json
from pathlib import Path

PROMPT_DIR = Path(__file__).parent

def load_prompts(mode=None):
    all_prompts = []
    for f in sorted(PROMPT_DIR.glob("*.json")):
        with open(f) as fh:
            all_prompts.extend(json.load(fh))
    if mode is not None:
        all_prompts = [p for p in all_prompts if p["mode"] == mode]
    return all_prompts

def load_by_mode():
    all_p = load_prompts()
    modes = {}
    for p in all_p:
        m = p["mode"]
        if m not in modes:
            modes[m] = []
        modes[m].append(p)
    return modes
