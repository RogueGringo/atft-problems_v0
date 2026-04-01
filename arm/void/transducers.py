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

    def _encode_char(self, ch: str, prev_was_newline: bool, prev_was_space: bool):
        """Encode a single character into 4 harmonic channels."""
        ch0 = ord(ch) % 256
        ch1 = 3 if ch.isupper() else (1 if ch.islower() else 0)
        is_space = ch in (" ", "\t")
        is_newline = ch == "\n"
        if is_newline:
            ch2 = 3
        elif is_space or prev_was_space or (not ch.isalnum() and not prev_was_space):
            ch2 = 1
        else:
            ch2 = 0
        ch3 = 3 if ch in self.STOP_PUNCT else (1 if ch in self.COMMA_PUNCT else 0)
        return (ch0, ch1, ch2, ch3), is_newline, (is_space or is_newline)

    def transduce(self, source: str, window: int = 0, stride: int = 0, max_points: int = 0) -> PointCloud:
        """Transduce text into a point cloud.

        Args:
            source: input text
            window: if >0, use sliding window n-grams (each point = window chars x 4 channels).
                    Creates (window*4)-dimensional points with much richer topology.
                    Recommended: window=16 or 32 for meaningful persistence.
            stride: step between windows (default: window//2 for 50% overlap)
            max_points: if >0, subsample to this many points (uniform random)
        """
        n = len(source)
        # First encode all characters
        encoded = np.zeros((n, 4), dtype=np.int16)
        prev_was_newline = False
        prev_was_space = True
        for i, ch in enumerate(source):
            channels, prev_was_newline, prev_was_space = self._encode_char(
                ch, prev_was_newline, prev_was_space)
            encoded[i] = channels

        if window > 0:
            # Windowed mode: each point is a flattened window of characters
            if stride <= 0:
                stride = max(1, window // 2)
            starts = range(0, n - window + 1, stride)
            data = np.zeros((len(starts), window * 4), dtype=np.int16)
            for j, s in enumerate(starts):
                data[j] = encoded[s:s + window].ravel()
            label = f"text:{n}chars:w{window}s{stride}:{data.shape[0]}pts"
        else:
            data = encoded
            label = f"text:{n}chars"

        if max_points > 0 and data.shape[0] > max_points:
            rng = np.random.RandomState(42)
            idx = rng.choice(data.shape[0], max_points, replace=False)
            idx.sort()
            data = data[idx]
            label += f":sub{max_points}"

        return PointCloud.from_array(data, source=label)

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
