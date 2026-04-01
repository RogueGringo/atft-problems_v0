"""Transport pre-trained crystal structure — 2-bit pack/unpack, .npz I/O."""
from __future__ import annotations
import json
import numpy as np
from arm.void.formats import Crystal

VALID_TERNARY = {0, 1, 3}

def pack_ternary(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint8)
    invalid = set(np.unique(values)) - VALID_TERNARY
    if invalid:
        raise ValueError(f"invalid ternary values: {invalid}. Only {{0, 1, 3}} allowed.")
    pad_len = (4 - len(values) % 4) % 4
    if pad_len:
        values = np.concatenate([values, np.zeros(pad_len, dtype=np.uint8)])
    values = values.reshape(-1, 4)
    packed = (values[:, 0] | (values[:, 1] << 2) | (values[:, 2] << 4) | (values[:, 3] << 6))
    return packed.astype(np.uint8)

def unpack_ternary(packed: np.ndarray, count: int) -> np.ndarray:
    packed = packed.astype(np.uint8)
    w0 = packed & 0x03
    w1 = (packed >> 2) & 0x03
    w2 = (packed >> 4) & 0x03
    w3 = (packed >> 6) & 0x03
    all_vals = np.stack([w0, w1, w2, w3], axis=-1).ravel()
    if np.any(all_vals[:count] == 2):
        raise ValueError("corrupt ternary data: found value 2 (bit pair 10)")
    return all_vals[:count]

def crystal_from_packed(packed: np.ndarray, count: int) -> Crystal:
    values = unpack_ternary(packed, count)
    n = len(values)
    return Crystal(
        void_ratio=int(np.sum(values == 0)) / n,
        identity_ratio=int(np.sum(values == 1)) / n,
        prime_ratio=int(np.sum(values == 3)) / n,
        eff_rank=0.0, source="weights"
    )

def save_weights(path: str, layers: dict[str, np.ndarray], config: dict, total_count: int) -> None:
    arrays = {k: v for k, v in layers.items()}
    arrays["config"] = np.array([json.dumps(config)])
    arrays["total_count"] = np.array([total_count])
    np.savez(path, **arrays)

def load_weights(path: str) -> tuple[dict[str, np.ndarray], dict]:
    data = np.load(path, allow_pickle=False)
    config = json.loads(str(data["config"][0]))
    layers = {k: data[k] for k in data.files if k not in ("config", "total_count")}
    return layers, config
