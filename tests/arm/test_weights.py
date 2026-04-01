import numpy as np
import pytest
import tempfile
import os

def test_pack_unpack_roundtrip():
    from arm.identity.weights import pack_ternary, unpack_ternary
    values = np.array([0, 1, 3, 0, 1, 3, 1, 0], dtype=np.uint8)
    packed = pack_ternary(values)
    assert packed.shape == (2,)
    unpacked = unpack_ternary(packed, count=8)
    np.testing.assert_array_equal(unpacked, values)

def test_pack_unpack_single_byte():
    from arm.identity.weights import pack_ternary, unpack_ternary
    values = np.array([3, 1, 0, 3], dtype=np.uint8)
    packed = pack_ternary(values)
    assert packed.shape == (1,)
    assert packed[0] == 199  # 3 | (1<<2) | (0<<4) | (3<<6) = 3+4+0+192
    unpacked = unpack_ternary(packed, count=4)
    np.testing.assert_array_equal(unpacked, values)

def test_reject_invalid_value():
    from arm.identity.weights import pack_ternary
    values = np.array([0, 2, 1, 3], dtype=np.uint8)
    with pytest.raises(ValueError, match="invalid"):
        pack_ternary(values)

def test_unpack_corrupted_byte():
    from arm.identity.weights import unpack_ternary
    corrupted = np.array([0b00_10_00_01], dtype=np.uint8)
    with pytest.raises(ValueError, match="corrupt"):
        unpack_ternary(corrupted, count=4)

def test_crystal_from_weights():
    from arm.identity.weights import crystal_from_packed, pack_ternary
    values = np.array([0]*22 + [1]*42 + [3]*36, dtype=np.uint8)
    packed = pack_ternary(values)
    crystal = crystal_from_packed(packed, count=100)
    assert abs(crystal.void_ratio - 0.22) < 0.01
    assert abs(crystal.identity_ratio - 0.42) < 0.01
    assert abs(crystal.prime_ratio - 0.36) < 0.01

def test_save_load_npz():
    from arm.identity.weights import pack_ternary, save_weights, load_weights
    values = np.array([0]*20 + [1]*40 + [3]*40, dtype=np.uint8)
    packed = pack_ternary(values)
    config = {"layers": 6, "hidden_dim": 512}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.npz")
        save_weights(path, {"layer_0_q": packed}, config=config, total_count=100)
        loaded_layers, loaded_config = load_weights(path)
        np.testing.assert_array_equal(loaded_layers["layer_0_q"], packed)
        assert loaded_config["layers"] == 6
