import numpy as np
import pytest

def test_ternary_matmul_identity():
    from arm.prime.crystal import ternary_matmul_dense
    x = np.array([[1, 2, 3]], dtype=np.int16)
    w = np.array([[1], [1], [1]], dtype=np.uint8)
    out = ternary_matmul_dense(x, w)
    assert out[0, 0] == 6

def test_ternary_matmul_zero():
    from arm.prime.crystal import ternary_matmul_dense
    x = np.array([[5, 10, 15]], dtype=np.int16)
    w = np.array([[0], [0], [0]], dtype=np.uint8)
    out = ternary_matmul_dense(x, w)
    assert out[0, 0] == 0

def test_ternary_matmul_three():
    from arm.prime.crystal import ternary_matmul_dense
    x = np.array([[4, 0, 0]], dtype=np.int16)
    w = np.array([[3], [0], [0]], dtype=np.uint8)
    out = ternary_matmul_dense(x, w)
    assert out[0, 0] == 12

def test_ternary_matmul_mixed():
    from arm.prime.crystal import ternary_matmul_dense
    x = np.array([[2, 5, 10]], dtype=np.int16)
    w = np.array([[3], [0], [1]], dtype=np.uint8)
    out = ternary_matmul_dense(x, w)
    assert out[0, 0] == 16

def test_forward_pass_not_implemented():
    from arm.prime.crystal import forward_pass
    with pytest.raises(NotImplementedError):
        forward_pass(np.zeros((1, 10), dtype=np.int16), weights_path="dummy.npz")
