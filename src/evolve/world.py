"""World geometry: boundaries, periodic displacement and bearings."""
from __future__ import annotations

import numpy as np

from .config import WorldConfig


class World:
    """Continuous 2-D world with configurable boundaries."""

    def __init__(self, cfg: WorldConfig) -> None:
        self.cfg = cfg
        self.width = float(cfg.width)
        self.height = float(cfg.height)
        self.mode = cfg.boundaries

    # -- boundary handling ---------------------------------------------------
    def _reflect_axis(self, coord: np.ndarray, size: float) -> np.ndarray:
        m = np.mod(coord, 2.0 * size)
        return np.where(m > size, 2.0 * size - m, m)

    def apply(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map coordinates back into the world according to the boundary mode."""
        if self.mode == "wrap":
            return np.mod(x, self.width), np.mod(y, self.height)
        if self.mode == "reflect":
            return self._reflect_axis(x, self.width), self._reflect_axis(y, self.height)
        return (np.clip(x, 0.0, self.width), np.clip(y, 0.0, self.height))

    # -- displacements -------------------------------------------------------
    #
    # These are the hottest functions in the tick: sensing computes a distance
    # and a bearing for every candidate pair, and the pair count grows with
    # local density. The naive form ``(d + w/2) % w - w/2`` allocates four
    # temporaries per axis; the in-place form below allocates one and performs
    # the same float operations in the same order, so results are bit-identical
    # (checked by ``tests/test_optimization_equivalence.py``).
    def dx(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        d = np.subtract(x1, x0, dtype=np.float64)
        if self.mode == "wrap":
            d += 0.5 * self.width
            np.mod(d, self.width, out=d)
            d -= 0.5 * self.width
        return d

    def dy(self, y0: np.ndarray, y1: np.ndarray) -> np.ndarray:
        d = np.subtract(y1, y0, dtype=np.float64)
        if self.mode == "wrap":
            d += 0.5 * self.height
            np.mod(d, self.height, out=d)
            d -= 0.5 * self.height
        return d

    def delta(self, x0: np.ndarray, y0: np.ndarray,
              x1: np.ndarray, y1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Periodic displacement ``(dx, dy)`` from ``(x0, y0)`` to ``(x1, y1)``.

        ``distance`` and ``bearing`` both need it, and calling them separately
        computed the wrap arithmetic twice.
        """
        return self.dx(x0, x1), self.dy(y0, y1)

    def distance(self, x0: np.ndarray, y0: np.ndarray,
                 x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
        return np.hypot(self.dx(x0, x1), self.dy(y0, y1))

    def bearing(self, from_x: np.ndarray, from_y: np.ndarray,
                to_x: np.ndarray, to_y: np.ndarray,
                heading: np.ndarray) -> np.ndarray:
        """Signed relative bearing in ``[-pi, pi)`` (0 = straight ahead)."""
        ang = np.arctan2(self.dy(from_y, to_y), self.dx(from_x, to_x))
        rel = ang - heading
        return (rel + np.pi) % (2.0 * np.pi) - np.pi

    def distance_and_bearing(self, from_x: np.ndarray, from_y: np.ndarray,
                             to_x: np.ndarray, to_y: np.ndarray,
                             heading: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
        """Both quantities from one displacement computation.

        Identical values to calling :meth:`distance` and :meth:`bearing` — the
        arithmetic is the same, only the duplicated wrap is removed.
        """
        dx, dy = self.delta(from_x, from_y, to_x, to_y)
        d = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)
        rel = ang - heading
        return d, (rel + np.pi) % (2.0 * np.pi) - np.pi
