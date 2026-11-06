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
    def dx(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        d = x1 - x0
        if self.mode == "wrap":
            d = (d + 0.5 * self.width) % self.width - 0.5 * self.width
        return d

    def dy(self, y0: np.ndarray, y1: np.ndarray) -> np.ndarray:
        d = y1 - y0
        if self.mode == "wrap":
            d = (d + 0.5 * self.height) % self.height - 0.5 * self.height
        return d

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
