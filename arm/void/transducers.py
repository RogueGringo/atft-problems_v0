"""Universal input boundary — any 0-dim sequential medium → point cloud."""
from __future__ import annotations
import csv
import io
import json
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
import numpy as np
from arm.void.formats import PointCloud

class Transducer(ABC):
    @abstractmethod
    def transduce(self, source) -> PointCloud: ...
    @abstractmethod
    def describe(self) -> str: ...

class TextTransducer(Transducer):
    """Character-level harmonic channels.
    Ch0: character identity (ord value)
    Ch1: case state → 0=space/whitespace, 1=lower, 3=upper
    Ch2: word boundary → 0=within word, 1=word boundary, 3=paragraph
    Ch3: punctuation → 0=none, 1=comma/semicolon, 3=stop/question/exclamation
    """
    COMMA_PUNCT = set(",;:")
    STOP_PUNCT = set(".!?")

    def transduce(self, source: str) -> PointCloud:
        n = len(source)
        data = np.zeros((n, 4), dtype=np.int16)
        prev_was_newline = False
        prev_was_space = True

        for i, ch in enumerate(source):
            data[i, 0] = ord(ch) % 256
            if ch.isupper():
                data[i, 1] = 3
            elif ch.islower():
                data[i, 1] = 1
            else:
                data[i, 1] = 0

            is_space = ch in (" ", "\t")
            is_newline = ch == "\n"
            if is_newline or (prev_was_newline and is_newline):
                data[i, 2] = 3
            elif is_space or prev_was_space or (not ch.isalnum() and not prev_was_space):
                data[i, 2] = 1
            else:
                data[i, 2] = 0
            prev_was_newline = is_newline
            prev_was_space = is_space or is_newline

            if ch in self.STOP_PUNCT:
                data[i, 3] = 3
            elif ch in self.COMMA_PUNCT:
                data[i, 3] = 1
            else:
                data[i, 3] = 0

        return PointCloud.from_array(data, source=f"text:{n}chars")

    def describe(self) -> str:
        return "TextTransducer: 4-channel character harmonics (identity, case, boundary, punct)"

class GenericTransducer(Transducer):
    def transduce(self, source) -> PointCloud:
        if isinstance(source, str):
            reader = csv.reader(io.StringIO(source))
            rows = [[float(v) for v in row] for row in reader if row]
            data = np.array(rows, dtype=np.float32)
        elif isinstance(source, np.ndarray):
            data = source.astype(np.float32)
        elif isinstance(source, list):
            data = np.array(source, dtype=np.float32)
        else:
            raise TypeError(f"GenericTransducer cannot handle {type(source)}")
        return PointCloud.from_array(data, source=f"generic:{data.shape[0]}x{data.shape[1]}")

    def describe(self) -> str:
        return "GenericTransducer: columnar/sequential numeric data"


SUBSTANCES = ["none", "N,N-DMT", "LSD", "psilocybin", "5-MeO-DMT", "other"]
VEILBREAK_API = "https://api.veilbreak.ai/api/experiments"
VEILBREAK_CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "veilbreak_cache.json")

class VeilbreakTransducer(Transducer):
    """Veilbreak cognitive physics experiments → point clouds."""
    COMMA_PUNCT = set()  # not used but keeps ABC happy if needed

    def fetch_experiments(self) -> list[dict]:
        cache_path = os.path.normpath(VEILBREAK_CACHE)
        try:
            req = urllib.request.Request(VEILBREAK_API, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode())
                experiments = body.get("data", body) if isinstance(body, dict) else body
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(experiments, f)
                return experiments
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    return json.load(f)
            return []

    def _encode_experiment(self, exp: dict) -> list[float]:
        wavelength = float(exp.get("laser_wavelength", 0) or 0)
        dose = float(exp.get("substance_dose", 0) or 0)
        laser_class = float(exp.get("laser_class", 0) or 0)
        substance = exp.get("substance", "none") or "none"
        sub_idx = float(SUBSTANCES.index(substance)) if substance in SUBSTANCES else float(len(SUBSTANCES))
        observed = 3.0 if exp.get("observed", False) else 0.0
        return [wavelength, dose, laser_class, sub_idx, observed]

    def transduce_experiments(self, experiments: list[dict]) -> PointCloud:
        if not experiments:
            data = np.empty((0, 5), dtype=np.float32)
        else:
            rows = [self._encode_experiment(e) for e in experiments]
            data = np.array(rows, dtype=np.float32)
        return PointCloud.from_array(data, source=f"veilbreak:{len(experiments)}exp")

    def transduce_multichannel(self, experiments: list[dict]) -> tuple[PointCloud, PointCloud]:
        pc_struct = self.transduce_experiments(experiments)
        descriptions = " ".join(e.get("description", "") for e in experiments if e.get("description"))
        text_t = TextTransducer()
        pc_text = text_t.transduce(descriptions) if descriptions else PointCloud.from_array(
            np.empty((0, 4), dtype=np.int16), source="veilbreak:no_text"
        )
        return pc_struct, pc_text

    def transduce(self, source=None) -> PointCloud:
        experiments = source if isinstance(source, list) else self.fetch_experiments()
        return self.transduce_experiments(experiments)

    def describe(self) -> str:
        return "VeilbreakTransducer: cognitive physics experiments (struct + text channels)"
