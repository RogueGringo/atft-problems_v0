"""
transducers/base.py — Abstract base class for all atft transducers.

A transducer adapts a data source (file, stream, corpus) into a sequence of
torch.Tensor chunks that the analysis pipeline can consume.
"""

from abc import ABC, abstractmethod
from typing import Iterator

import torch


class BaseTransducer(ABC):
    """
    Abstract base class for transducers.

    Subclasses must set `name` and implement `load`, `to_chunks`,
    and `metadata`.
    """

    name: str = "base"

    @abstractmethod
    def load(self, source: str) -> None:
        """
        Load data from *source* (file path, URL, or other identifier).

        Must be called before `to_chunks` or `metadata`.
        """

    @abstractmethod
    def to_chunks(self, chunk_size: int = 256) -> Iterator[torch.Tensor]:
        """
        Yield successive tensor chunks of length *chunk_size*.

        Each chunk is a 1-D LongTensor of token ids (or equivalent integers).
        The final chunk may be shorter than *chunk_size*.
        """

    @abstractmethod
    def metadata(self) -> dict:
        """
        Return a dict of metadata about the loaded source.

        Typical keys: source, n_tokens, vocab_size, encoding, …
        """
