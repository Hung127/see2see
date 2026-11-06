"""Coarse environmental fields: climate, soil, terrain, vegetation cover, pheromone.

The field grid is a *summary* of the world used by sensors and by plant growth. It is
not an independent source of truth for plants or organisms: ``plant_density`` and
``organism_density`` are written by the engine each tick from the actual entities.
"""
from __future__ import annotations

import numpy as np

from .config import EnvironmentConfig, WorldConfig
from .rng import Rng

_EVENT_KEYS = ("temperature", "moisture", "plant_growth", "damage")


class Fields:
    def __init__(self, world: WorldConfig, env: EnvironmentConfig, rng: Rng) -> None:
        self.cfg = env
        self.n = int(world.n_field_cells)
        self.cell_size = float(world.width) / self.n
        self.rng = rng
        shape = (self.n, self.n)
        self.terrain = self._terrain_from_noise(shape)
        # Soil is richer in low, wet ground; height raises carrying capacity but not nutrients.
        self.soil = 1.0 + 5.0 * (1.0 - self.terrain)
        self.temperature = np.full(shape, env.t_mean, dtype=np.float64)
        self.moisture = np.full(shape, env.m_mean, dtype=np.float64)
        self.pheromone = np.zeros(shape, dtype=np.float64)
        self.plant_density = np.zeros(shape, dtype=np.float64)
        self.organism_density = np.zeros(shape, dtype=np.float64)
        # Per-cell additive modifiers written by events; reset each tick by apply_events.
        self._t_offset = np.zeros(shape, dtype=np.float64)
        self._m_offset = np.zeros(shape, dtype=np.float64)
        self._growth_factor = np.ones(shape, dtype=np.float64)

    # -- terrain -------------------------------------------------------------
    def _terrain_from_noise(self, shape: tuple[int, int]) -> np.ndarray:
        field = self.rng.random(shape)
        for _ in range(10):  # periodic box blur -> smooth, seamless landscape
            field = (
                4.0 * field
                + 2.0 * (np.roll(field, 1, 0) + np.roll(field, -1, 0)
                         + np.roll(field, 1, 1) + np.roll(field, -1, 1))
                + (np.roll(np.roll(field, 1, 0), 1, 1) + np.roll(np.roll(field, 1, 0), -1, 1)
                   + np.roll(np.roll(field, -1, 0), 1, 1) + np.roll(np.roll(field, -1, 0), -1, 1))
            ) / 16.0
        lo, hi = float(field.min()), float(field.max())
        if hi - lo < 1e-12:
            return np.full(shape, 0.5)
        return (field - lo) / (hi - lo)

    # -- per-tick forcing ----------------------------------------------------
    def step(self, tick: int) -> None:
        period = max(1, int(self.cfg.season_period))
        phase = 2.0 * np.pi * tick / period
        temp = self.cfg.t_mean + self.cfg.t_amp * np.sin(phase)
        moisture = self.cfg.m_mean + self.cfg.m_amp * np.sin(phase - 0.5 * np.pi)

        if self.cfg.rain_rate > 0.0:
            rain = (self.rng.random((self.n, self.n)) < self.cfg.rain_rate).astype(np.float64)
            moisture = moisture + 0.02 * rain

        self.temperature = np.clip(temp + self._t_offset, -30.0, 60.0)
        self.moisture = np.clip(moisture + self._m_offset, 0.0, 1.0)

        # Pheromone: 5-point diffusion then exponential decay.
        p = self.pheromone
        self.pheromone = 0.9 * (
            0.5 * p
            + 0.125 * (np.roll(p, 1, 0) + np.roll(p, -1, 0)
                       + np.roll(p, 1, 1) + np.roll(p, -1, 1))
        )
        self.soil = np.maximum(self.soil, 0.0)
        # Event modifiers last exactly one tick; events re-apply them every tick they run.
        self._t_offset.fill(0.0)
        self._m_offset.fill(0.0)
        self._growth_factor.fill(1.0)

    # -- sampling ------------------------------------------------------------
    def cell_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor(np.asarray(x) / self.cell_size).astype(np.int64) % self.n
        iy = np.floor(np.asarray(y) / self.cell_size).astype(np.int64) % self.n
        return ix, iy

    def sample(self, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        ix, iy = self.cell_index(x, y)
        return {
            "temperature": self.temperature[ix, iy],
            "moisture": self.moisture[ix, iy],
            "soil": self.soil[ix, iy],
            "terrain": self.terrain[ix, iy],
            "plant_density": self.plant_density[ix, iy],
            "organism_density": self.organism_density[ix, iy],
            "pheromone": self.pheromone[ix, iy],
        }

    def add_pheromone(self, x: np.ndarray, y: np.ndarray, amount: np.ndarray | float) -> None:
        ix, iy = self.cell_index(x, y)
        np.add.at(self.pheromone, (ix, iy), amount)
        np.clip(self.pheromone, 0.0, 1.0, out=self.pheromone)

    def growth_factor(self) -> np.ndarray:
        """Per-cell multiplier applied to plant growth this tick."""
        return self._growth_factor
