"""Tests for the batched feed-forward brain (16 -> 12 -> 8 -> 8)."""
import numpy as np
import pytest

from evolve.brain import LAYER_SIZES, N_PARAMS, forward, random_weights, split
from evolve.rng import Rng


def test_layout_constants():
    assert LAYER_SIZES == (16, 12, 8, 8)
    assert N_PARAMS == 16 * 12 + 12 + 12 * 8 + 8 + 8 * 8 + 8 == 380


def test_output_shape_and_range():
    rng = Rng(0)
    w = random_weights(7, rng)
    x = rng.random((7, 16)) * 2.0 - 1.0
    out = forward(x, w)
    assert out.shape == (7, 8)
    # channels 0-1 are tanh (signed), 2-7 are sigmoid (gated)
    assert np.all(out[:, :2] >= -1.0) and np.all(out[:, :2] <= 1.0)
    assert np.all(out[:, 2:] >= 0.0) and np.all(out[:, 2:] <= 1.0)


def test_deterministic():
    rng = Rng(1)
    w = random_weights(5, rng)
    x = rng.random((5, 16))
    assert np.array_equal(forward(x, w), forward(x, w))


def test_different_weights_differ():
    rng = Rng(2)
    w1 = random_weights(4, rng)
    w2 = random_weights(4, rng)
    x = rng.random((4, 16))
    assert not np.allclose(forward(x, w1), forward(x, w2))


def test_zero_weights_give_bias_only_pass():
    rng = Rng(3)
    w = random_weights(3, rng)
    W1, b1, W2, b2, W3, b3 = split(w)
    zero_w = w.copy()
    zero_w[:, :192] = 0.0
    zero_w[:, 204:300] = 0.0
    zero_w[:, 308:372] = 0.0
    x = rng.random((3, 16))
    out = forward(x, zero_w)
    assert np.allclose(out[:, :2], np.tanh(b3[:, :2]))
    assert np.allclose(out[:, 2:], 1.0 / (1.0 + np.exp(-b3[:, 2:])))


def test_batch_equals_single():
    rng = Rng(4)
    w = random_weights(6, rng)
    x = rng.random((6, 16)) * 2 - 1
    batch = forward(x, w)
    for i in range(6):
        single = forward(x[i : i + 1], w[i : i + 1])
        assert np.allclose(batch[i], single[0]), i


def test_split_shapes():
    rng = Rng(5)
    w = random_weights(2, rng)
    W1, b1, W2, b2, W3, b3 = split(w)
    assert W1.shape == (2, 16, 12) and b1.shape == (2, 12)
    assert W2.shape == (2, 12, 8) and b2.shape == (2, 8)
    assert W3.shape == (2, 8, 8) and b3.shape == (2, 8)
    assert W1.size + b1.size + W2.size + b2.size + W3.size + b3.size == 2 * N_PARAMS
