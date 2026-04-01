"""
transducers/text.py — Text transducer for the atft CLI.

Loads text from a file or stdin, tokenizes with GPT-2 BPE, and yields
fixed-size token chunks as LongTensors.
"""

import sys
from typing import Iterator

import torch
from transformers import AutoTokenizer

from .base import BaseTransducer


class TextTransducer(BaseTransducer):
    """Transducer that reads plain text and tokenizes with GPT-2 BPE."""

    name = "text"

    _TOKENIZER_NAME = "gpt2"

    def __init__(self) -> None:
        self._text: str | None = None
        self._source: str | None = None
        self._tokenizer = None
        self._token_ids: list[int] | None = None

    def load(self, source: str) -> None:
        """Load text from *source*.

        Parameters
        ----------
        source : str
            File path to read from, or ``"-"`` to read from stdin.
        """
        if source == "-":
            self._text = sys.stdin.read()
        else:
            with open(source, "r", encoding="utf-8") as fh:
                self._text = fh.read()

        self._source = source
        self._tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_NAME)
        encoded = self._tokenizer.encode(self._text, add_special_tokens=False)
        self._token_ids = encoded

    def to_chunks(self, chunk_size: int = 256) -> Iterator[torch.Tensor]:
        """Yield successive (chunk_size,) LongTensors of token ids.

        The final chunk is yielded even if shorter than *chunk_size*.
        """
        if self._token_ids is None:
            raise RuntimeError("Call load() before to_chunks().")

        ids = self._token_ids
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            yield torch.tensor(chunk, dtype=torch.long)

    def metadata(self) -> dict:
        """Return metadata about the loaded text."""
        if self._text is None or self._token_ids is None:
            raise RuntimeError("Call load() before metadata().")
        return {
            "type": "text",
            "tokenizer": self._TOKENIZER_NAME,
            "source": self._source,
            "total_chars": len(self._text),
            "total_tokens": len(self._token_ids),
        }
