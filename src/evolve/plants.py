"""Discrete plant pool with nutrient-limited logistic growth (DESIGN §6.7).

Plants are entities, not a field: the per-cell ``plant_density`` used by sensors is
derived from them each tick. Growth is paid from the cell's inorganic nutrient pool
and from light (photosynthesis stores light as chemical energy 1:1), so the two
channels that must be booked are ``soil`` and ``ledger.light_in``.

Mass bookkeeping contract:

* ``grow`` returns the total biomass created; the caller can use it for statistics.
  Soil decreases by exactly the same amount and ``light_in`` increases by it.
* ``senesce`` returns the biomass that left the plants; the engine adds it to the
  detritus pool (organic matter keeps its energy).
* ``bite`` returns the biomass taken; the engine hands it to the eater.
"""
from __future__ import annotations

import numpy as np

from .config import PlantConfig, WorldConfig
from .energy import Ledger
from .fields import Fields
from .rng import Rng

_DEAD_EPS = 1e-12


class Plants:
    """Structure-of-arrays plant pool with deterministic slot reuse."""

    def __init__(self, cfg: PlantConfig, world: WorldConfig) -> None:
        self.cfg = cfg
        self.width = float(world.width)
        self.height = float(world.height)
        self.capacity = int(cfg.max)
        cap = self.capacity
        self.id = np.arange(cap, dtype=np.int64)
        self._next_id = 0
        self.x = np.zeros(cap, dtype=np.float64)
        self.y = np.zeros(cap, dtype=np.float64)
        self.biomass = np.zeros(cap, dtype=np.float64)
        self.age = np.zeros(cap, dtype=np.int64)
        self.alive = np.zeros(cap, dtype=bool)

    # -- population management ----------------------------------------------
    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def add(self, x: float, y: float, biomass: float = 1.0) -> int:
        dead = np.flatnonzero(~self.alive)
        if dead.size == 0:
            return -1
        slot = int(dead[0])
        self.alive[slot] = True
        self.id[slot] = self._next_id
        self._next_id += 1
        self.x[slot] = float(x)
        self.y[slot] = float(y)
        self.biomass[slot] = float(biomass)
        self.age[slot] = 0
        return slot

    def seed_initial(self, n: int, rng: Rng) -> int:
        """Scatter ``n`` seedlings uniformly over the world."""
        n = min(int(n), self.capacity)
        xs = rng.random(n) * self.width
        ys = rng.random(n) * self.height
        made = 0
        for i in range(n):
            if self.add(float(xs[i]), float(ys[i]), 1.0) < 0:
                break
            made += 1
        return made

    def remove_dead(self) -> None:
        keep = np.flatnonzero(self.alive)
        if keep.size == self.alive.size:
            return
        for name in ("id", "x", "y", "biomass", "age", "alive"):
            setattr(self, name, getattr(self, name)[keep])
        pad = self.capacity - self.alive.size
        if pad > 0:
            for name, dtype in (("id", np.int64), ("x", np.float64), ("y", np.float64),
                                ("biomass", np.float64), ("age", np.int64),
                                ("alive", bool)):
                arr = getattr(self, name)
                setattr(self, name, np.concatenate([arr, np.zeros(pad, dtype=dtype)]))

    # -- growth ---------------------------------------------------------------
    def grow(self, fields: Fields, ledger: Ledger) -> float:
        """Logistic growth limited by soil nutrient, temperature and moisture."""
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return 0.0
        cfg = self.cfg
        n = fields.n
        ix, iy = fields.cell_index(self.x[idx], self.y[idx])
        cell = iy * n + ix
        p = self.biomass[idx]
        height = fields.terrain[ix, iy]
        k_local = cfg.k0 * (0.6 + 0.8 * height)
        temp = fields.temperature[ix, iy]
        moisture = fields.moisture[ix, iy]
        f_t = np.exp(-((temp - 22.0) / 8.0) ** 2)
        f_m = 0.3 + 0.7 * moisture
        demand = cfg.r * p * np.maximum(0.0, 1.0 - p / k_local) * f_t * f_m

        soil = fields.soil.reshape(-1)
        total_demand = np.bincount(cell, weights=demand, minlength=n * n)
        avail = np.maximum(soil, 0.0)
        limited = total_demand > avail
        scale = np.ones_like(demand)
        if np.any(limited):
            limited_per_plant = limited[cell]
            scale[limited_per_plant] = (
                avail[cell[limited_per_plant]] / total_demand[cell[limited_per_plant]]
            )
        growth = demand * scale

        # Exact floating-point remainder goes to the lowest-ID plant in each limited
        # cell, so soil and plant stocks close to rtol 1e-9 regardless of summation order.
        alloc = np.bincount(cell, weights=growth, minlength=n * n)
        residual = np.where(limited, avail - alloc, 0.0)
        order = np.lexsort((self.id[idx], cell))
        sorted_cells = cell[order]
        first = np.ones(sorted_cells.size, dtype=bool)
        first[1:] = sorted_cells[1:] != sorted_cells[:-1]
        lowest = order[first]
        growth[lowest] += residual[cell[lowest]]

        self.biomass[idx] += growth
        self.age[idx] += 1
        taken = np.bincount(cell, weights=growth, minlength=n * n)
        soil -= taken
        led_light = float(growth.sum())
        ledger.light_in += led_light
        ledger.mass_from_soil += led_light
        return led_light

    # -- germination ----------------------------------------------------------
    def germinate(self, rng: Rng, fields: Fields, ledger: Ledger) -> int:
        """Seed up to ``germination_per_tick`` plants, biomass-weighted by cell."""
        room = self.capacity - self.n_alive
        if room <= 0:
            return 0
        n_new = min(int(self.cfg.germination_per_tick), room)
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return 0
        n = fields.n
        ix, iy = fields.cell_index(self.x[idx], self.y[idx])
        flat_cell = iy * n + ix
        weights = np.bincount(flat_cell, weights=self.biomass[idx], minlength=n * n)
        total = float(weights.sum())
        if total <= 0.0:
            return 0
        probs = weights / total
        picks = rng.choice(n * n, size=n_new, replace=True, p=probs)
        soil = fields.soil.reshape(-1)
        made = 0
        for raw in picks:
            c = int(raw)
            if soil[c] < self.cfg.seed_cost:
                continue
            cy, cx = divmod(c, n)
            x = (cx + rng.random()) * fields.cell_size
            y = (cy + rng.random()) * fields.cell_size
            if self.add(float(x), float(y), self.cfg.seed_cost) < 0:
                break
            soil[c] -= self.cfg.seed_cost
            ledger.light_in += self.cfg.seed_cost
            ledger.mass_from_soil += self.cfg.seed_cost
            made += 1
        return made

    # -- losses ---------------------------------------------------------------
    def senesce(self) -> float:
        """Ageing loss ``0.002·P`` per tick; returns the mass that left the pool."""
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return 0.0
        loss = self.cfg.senescence * self.biomass[idx]
        self.biomass[idx] -= loss
        dead = self.biomass[idx] <= _DEAD_EPS
        if np.any(dead):
            self.biomass[idx[dead]] = 0.0
            self.alive[idx[dead]] = False
        return float(loss.sum())

    def bite(self, i: int, amount: float) -> float:
        """Remove up to ``amount`` biomass from plant ``i``; returns what was taken."""
        if not self.alive[i]:
            return 0.0
        taken = float(min(amount, self.biomass[i]))
        self.biomass[i] -= taken
        if self.biomass[i] <= _DEAD_EPS:
            self.biomass[i] = 0.0
            self.alive[i] = False
        return taken

    # -- derived views --------------------------------------------------------
    def biomass_field(self, fields: Fields) -> np.ndarray:
        idx = np.flatnonzero(self.alive)
        out = np.zeros((fields.n, fields.n), dtype=np.float64)
        if idx.size == 0:
            return out
        ix, iy = fields.cell_index(self.x[idx], self.y[idx])
        np.add.at(out, (ix, iy), self.biomass[idx])
        return out

    def total_biomass(self) -> float:
        return float(self.biomass[self.alive].sum())
