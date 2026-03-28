"""Wierzbicka's 59 Natural Semantic Metalanguage primes, grouped by category.

Source: Wierzbicka, A. (1996). Semantics: Primes and Universals.
These are the irreducible building blocks of meaning found in ALL human languages.
"""

PRIMES = {
    "substantives": ["I", "you", "someone", "something", "people"],
    "determiners": ["this", "the same", "other"],
    "quantifiers": ["one", "two", "some", "all", "many"],
    "evaluators": ["good", "bad"],
    "descriptors": ["big", "small"],
    "mental": ["think", "know", "want", "don't want", "feel", "see", "hear"],
    "speech": ["say", "true"],
    "actions": ["do", "happen", "move"],
    "existence": ["there is", "be (someone/something)"],
    "possession": ["have", "be (someone's)"],
    "life": ["live", "die"],
    "time": ["when", "now", "before", "after", "a long time", "a short time", "for some time"],
    "space": ["where", "here", "above", "below", "far", "near", "side", "inside"],
    "logic": ["not", "maybe", "can", "because", "if"],
    "intensifier": ["very", "more"],
    "taxonomy": ["kind of", "like"],
}

ALL_PRIMES = [p for category in PRIMES.values() for p in category]
PRIME_TO_CATEGORY = {p: cat for cat, primes in PRIMES.items() for p in primes}

assert len(ALL_PRIMES) == 59, f"Expected 59 primes, got {len(ALL_PRIMES)}"
