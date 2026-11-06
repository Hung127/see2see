"""Batched feed-forward controller: 16 -> 12 -> 8 -> 8 (DESIGN §5.1).

Every organism runs the *same* network shape with its own 380 parameters, drawn
from the neural block of its genome. The batch is a single set of ``einsum``
calls, so 5,000 organisms cost three matrix products per tick.

Output activations follow DESIGN §5.3: channels 0 and 1 (forward, turn) are
signed and use ``tanh``; channels 2–7 are gated and use ``sigmoid``.

Parameter layout (row-major, shared with :mod:`evolve.genome`)::

    [0:192)   W1 (16x12)   [192:204)  b1 (12)
    [204:300) W2 (12x8)    [300:308)  b2 (8)
    [308:372) W3 (8x8)     [372:380)  b3 (8)
"""
from __future__ import annotations

import numpy as np

from .rng import Rng

LAYER_SIZES = (16, 12, 8, 8)
N_PARAMS = sum(
    LAYER_SIZES[i] * LAYER_SIZES[i + 1] + LAYER_SIZES[i + 1]
    for i in range(len(LAYER_SIZES) - 1)
)  # 380

_W1 = slice(0, 192)
_B1 = slice(192, 204)
_W2 = slice(204, 300)
_B2 = slice(300, 308)
_W3 = slice(308, 372)
_B3 = slice(372, 380)


def split(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]:
    """Split a parameter batch into ``(W1, b1, W2, b2, W3, b3)`` views."""
    w = np.atleast_2d(np.asarray(w, dtype=np.float64))
    n = w.shape[0]
    return (
        w[:, _W1].reshape(n, 16, 12),
        w[:, _B1],
        w[:, _W2].reshape(n, 12, 8),
        w[:, _B2],
        w[:, _W3].reshape(n, 8, 8),
        w[:, _B3],
    )


def forward(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """One batched forward pass; ``x`` is ``(n, 16)``, ``w`` is ``(n, 380)``."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    w = np.atleast_2d(np.asarray(w, dtype=np.float64))
    W1, b1, W2, b2, W3, b3 = split(w)
    h1 = np.tanh(np.einsum("ni,nij->nj", x, W1) + b1)
    h2 = np.tanh(np.einsum("ni,nij->nj", h1, W2) + b2)
    z = np.einsum("ni,nij->nj", h2, W3) + b3
    out = np.empty_like(z)
    out[:, :2] = np.tanh(z[:, :2])
    out[:, 2:] = 0.5 * (1.0 + np.tanh(0.5 * z[:, 2:]))  # sigmoid, numerically stable
    return out


def random_weights(n: int, rng: Rng) -> np.ndarray:
    """Small zero-mean parameters: N(0, 0.35) weights, N(0, 0.05) biases."""
    return np.concatenate(
        [rng.normal(0.0, 0.35, size=(n, 360)), rng.normal(0.0, 0.05, size=(n, 20))],
        axis=1,
    )
