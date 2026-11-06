"""Action decoding: network outputs -> physical attempts (DESIGN §5.3).

This is the *only* decision point in the codebase that is not the network
itself: fixed threshold comparisons that turn a continuous output into a
discrete physical attempt. They encode no preference over targets or
situations. ``escape`` scales sprint speed and movement cost but can never
steer; ``rest`` scales throttle down and blocks eating and attacking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Actions:
    throttle: np.ndarray   # |o0| * o2 * (1 - o7), units of speed_max
    turn: np.ndarray       # o1 * turn_rate_max, rad/tick
    eat: np.ndarray        # bool
    attack: np.ndarray     # bool
    sprint: np.ndarray     # 1 + 1.5 * o5
    reproduce: np.ndarray  # bool
    rest: np.ndarray       # bool


def decode(outputs: np.ndarray, phen: np.ndarray) -> Actions:
    """Decode one batch of outputs ``(n, 8)`` given phenotypes ``(n, 12)``."""
    outputs = np.atleast_2d(np.asarray(outputs, dtype=np.float64))
    phen = np.atleast_2d(np.asarray(phen, dtype=np.float64))
    o0, o1, o2 = outputs[:, 0], outputs[:, 1], outputs[:, 2]
    o3, o4, o5, o6, o7 = (outputs[:, i] for i in range(3, 8))

    rest = o7 > 0.5
    turn_rate_max = 3.0 / np.maximum(phen[:, 0], 1e-6) ** 0.33
    return Actions(
        throttle=np.abs(o0) * o2 * (1.0 - o7),
        turn=o1 * turn_rate_max,
        eat=(o3 > 0.5) & ~rest,
        attack=(o4 > 0.5) & ~rest,
        sprint=1.0 + 1.5 * o5,
        reproduce=o6 > 0.5,
        rest=rest,
    )
