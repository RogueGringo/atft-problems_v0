#!/usr/bin/env python3
"""Crystal coherence stopping criterion for {0,1,3} inference.

The model stops generating when the output's structural topology matches the
input's structural topology — not at a fixed token count.

Dense input (high prime) → model runs longer
Simple input (high identity) → model resolves fast
Sparse input (high void) → nearly instant
"""
from __future__ import annotations

from collections import deque


class CrystalCoherenceMonitor:
    """Monitors structural coherence between input and output during inference.

    Measures the {0,1,3} crystal of input tokens and tracks the evolving
    crystal of output tokens. Signals COMMIT when Δ < ε.

    Dense input (high prime) → model runs longer
    Simple input (high identity) → model resolves fast
    Sparse input (high void) → nearly instant
    """

    def __init__(self, epsilon: float = 0.05, min_tokens: int = 10,
                 window_size: int = 50):
        """
        epsilon: coherence threshold (L1 distance between crystals)
        min_tokens: minimum output tokens before checking coherence
        window_size: rolling window for output crystal (recent tokens matter more)
        """
        self.epsilon = epsilon
        self.min_tokens = min_tokens
        self.window_size = window_size

        # Input crystal (set by set_input)
        self.input_crystal: dict = {"void": 0.0, "identity": 0.0, "prime": 0.0}

        # Rolling window of per-character classifications for output
        # Each entry is a tuple (n_void, n_identity, n_prime) for a token
        self._window: deque = deque()
        self._window_void = 0
        self._window_identity = 0
        self._window_prime = 0

        # Total output token count
        self._n_tokens = 0

        # Cache last computed state
        self._last_delta: float = float("inf")
        self._last_output_crystal: dict = {"void": 0.0, "identity": 0.0, "prime": 0.0}

    # ── Classification ────────────────────────────────────────────────────

    @staticmethod
    def classify_char(c: str) -> int:
        """Classify a single character: 0=void, 1=identity, 3=prime."""
        if c in ' \t\n\r':
            return 0  # void — whitespace / boundary
        if c in '.,;:!?-()[]{}"\'/':
            return 0  # void — punctuation (structural markers)
        if c.isupper():
            return 3  # prime — capitals (amplifiers)
        return 1      # identity — lowercase, digits (carrier)

    @classmethod
    def _crystal_from_text(cls, text: str) -> dict:
        """Compute void/identity/prime fractions for a text string."""
        n_void = n_identity = n_prime = 0
        for ch in text:
            v = cls.classify_char(ch)
            if v == 0:
                n_void += 1
            elif v == 1:
                n_identity += 1
            else:
                n_prime += 1

        total = n_void + n_identity + n_prime
        if total == 0:
            return {"void": 0.0, "identity": 0.0, "prime": 0.0}

        return {
            "void":     n_void     / total,
            "identity": n_identity / total,
            "prime":    n_prime    / total,
        }

    # ── Public interface ──────────────────────────────────────────────────

    def set_input(self, text: str) -> dict:
        """Measure input crystal. Returns {'void': x, 'identity': y, 'prime': z}.

        Also resets the output tracking state so this instance can be reused
        for a fresh generation pass.
        """
        self.input_crystal = self._crystal_from_text(text)

        # Reset output state
        self._window.clear()
        self._window_void = 0
        self._window_identity = 0
        self._window_prime = 0
        self._n_tokens = 0
        self._last_delta = float("inf")
        self._last_output_crystal = {"void": 0.0, "identity": 0.0, "prime": 0.0}

        return dict(self.input_crystal)

    def update(self, new_token_text: str) -> bool:
        """Add a generated token. Returns True if COMMIT (coherence reached).

        Maintains a rolling window of width `window_size` over generated tokens.
        The output crystal is computed from this window so that recent structure
        dominates — a long preamble won't prevent the monitor from recognising
        that the last 50 tokens have settled.
        """
        # Classify all characters in this token
        n_void = n_identity = n_prime = 0
        for ch in new_token_text:
            v = self.classify_char(ch)
            if v == 0:
                n_void += 1
            elif v == 1:
                n_identity += 1
            else:
                n_prime += 1

        # Push onto rolling window
        self._window.append((n_void, n_identity, n_prime))
        self._window_void     += n_void
        self._window_identity += n_identity
        self._window_prime    += n_prime

        # Evict oldest entry when window is full
        if len(self._window) > self.window_size:
            old_v, old_i, old_p = self._window.popleft()
            self._window_void     -= old_v
            self._window_identity -= old_i
            self._window_prime    -= old_p

        self._n_tokens += 1

        # Don't commit before minimum output length
        if self._n_tokens < self.min_tokens:
            self._last_output_crystal = {"void": 0.0, "identity": 0.0, "prime": 0.0}
            self._last_delta = float("inf")
            return False

        # Compute current output crystal from window totals
        total = self._window_void + self._window_identity + self._window_prime
        if total == 0:
            self._last_output_crystal = {"void": 0.0, "identity": 0.0, "prime": 0.0}
            self._last_delta = float("inf")
            return False

        out_void     = self._window_void     / total
        out_identity = self._window_identity / total
        out_prime    = self._window_prime    / total

        self._last_output_crystal = {
            "void":     out_void,
            "identity": out_identity,
            "prime":    out_prime,
        }

        # L1 distance between crystals
        delta = (
            abs(self.input_crystal["void"]     - out_void)
            + abs(self.input_crystal["identity"] - out_identity)
            + abs(self.input_crystal["prime"]    - out_prime)
        )
        self._last_delta = delta

        return delta < self.epsilon

    def get_state(self) -> dict:
        """Return current state: input_crystal, output_crystal, delta, n_tokens."""
        return {
            "input_crystal":  dict(self.input_crystal),
            "output_crystal": dict(self._last_output_crystal),
            "delta":          self._last_delta,
            "n_tokens":       self._n_tokens,
        }


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    monitor = CrystalCoherenceMonitor(epsilon=0.05, min_tokens=10)

    # Test 1: Simple input → should commit fast
    print("=== Test 1: simple input ===")
    monitor.set_input("the cat sat on the mat")
    for word in "a small dog ran to the big park and played".split():
        done = monitor.update(word + " ")
        state = monitor.get_state()
        flag = "  COMMIT" if done else ""
        print(f"  token='{word}' Δ={state['delta']:.4f}{flag}")

    # Test 2: Complex input → should take longer
    print("\n=== Test 2: complex input ===")
    monitor.set_input("The Transcendental Unity of Apperception presupposes...")
    for word in "This fundamental concept requires careful examination of the conditions under which".split():
        done = monitor.update(word + " ")
        state = monitor.get_state()
        flag = "  COMMIT" if done else ""
        print(f"  token='{word}' Δ={state['delta']:.4f}{flag}")

    # Crystal comparison
    print("\n=== Crystal comparison ===")
    print("Simple input crystal: ",
          monitor.set_input("the cat sat on the mat"))
    print("Complex input crystal:",
          monitor.set_input("The Transcendental Unity of Apperception"))
