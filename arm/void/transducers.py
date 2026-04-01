"""Universal input boundary — any 0-dim sequential medium → point cloud."""
from __future__ import annotations
import csv
import io
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
