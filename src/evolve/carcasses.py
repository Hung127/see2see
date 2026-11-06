"""Carcass entities: the organic pool left by dead animals (DESIGN §6.6).

A carcass is one indivisible lump of organic matter. It carries both mass and
chemical energy (1:1, like living tissue), so scavenging transfers both and
microbial mineralisation books both: ``δ`` leaves the carcass, ``δ`` enters the
inorganic soil pool as mass, and exactly ``δ`` leaves as heat.

This is the same mass/energy bookkeeping as the scalar detritus pool used for
senescent plant matter and undigested food, but carcasses are spatially explicit
so scavengers can find and compete for them.
"""
from __future__ import annotations

import numpy as np

from .energy import Ledger

_DEAD_EPS = 1e-12
_INITIAL_CAPACITY = 64


class Carcasses:
    """Structure-of-arrays carcass pool with deterministic slot reuse."""

    def __init__(self, capacity: int = _INITIAL_CAPACITY) -> None:
        self.capacity = int(capacity)
        cap = self.capacity
        self.id = np.arange(cap, dtype=np.int64)
        self._next_id = 0
        self.x = np.zeros(cap, dtype=np.float64)
        self.y = np.zeros(cap, dtype=np.float64)
        self.biomass = np.zeros(cap, dtype=np.float64)
        self.age = np.zeros(cap, dtype=np.int64)
        self.alive = np.zeros(cap, dtype=bool)

    # -- population management ------------------------------------------------
    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def _grow(self) -> None:
        old = self.alive.size
        new = max(old * 2, _INITIAL_CAPACITY)
        for name, dtype in (("id", np.int64), ("x", np.float64), ("y", np.float64),
                            ("biomass", np.float64), ("age", np.int64),
                            ("alive", bool)):
            arr = getattr(self, name)
            setattr(self, name, np.concatenate([arr, np.zeros(new - old, dtype=dtype)]))
        self.capacity = new

    def add(self, x: float, y: float, biomass: float) -> int:
        """Place a carcass; returns its slot index, or -1 for non-positive mass."""
        if biomass <= 0.0:
            return -1
        dead = np.flatnonzero(~self.alive)
        if dead.size:
            slot = int(dead[0])
        else:
            slot = self.alive.size
            self._grow()
        self.alive[slot] = True
        self.id[slot] = self._next_id
        self._next_id += 1
        self.x[slot] = float(x)
        self.y[slot] = float(y)
        self.biomass[slot] = float(biomass)
        self.age[slot] = 0
        return slot

    def remove_dead(self) -> None:
        """Compact dead slots; live order is preserved."""
        keep = np.flatnonzero(self.alive)
        if keep.size == self.alive.size:
            return
        for name in ("id", "x", "y", "biomass", "age", "alive"):
            setattr(self, name, getattr(self, name)[keep])

    # -- processes ------------------------------------------------------------
    def decay(self, ledger: Ledger, rate: float) -> float:
        """Mineralise ``rate·C`` per living carcass; returns total mass decayed."""
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return 0.0
        amount = rate * self.biomass[idx]
        self.biomass[idx] -= amount
        total = float(amount.sum())
        ledger.mass_to_soil += total
        ledger.heat_out += total
        dead = self.biomass[idx] <= _DEAD_EPS
        if np.any(dead):
            self.biomass[idx[dead]] = 0.0
            self.alive[idx[dead]] = False
        return total

    def bite(self, i: int, amount: float) -> float:
        """Remove up to ``amount`` biomass from carcass ``i``; returns the take."""
        if not self.alive[i]:
            return 0.0
        taken = float(min(amount, self.biomass[i]))
        self.biomass[i] -= taken
        if self.biomass[i] <= _DEAD_EPS:
            self.biomass[i] = 0.0
            self.alive[i] = False
        return taken

    # -- derived views --------------------------------------------------------
    def total_biomass(self) -> float:
        return float(self.biomass[self.alive].sum())

    def age_all(self) -> None:
        self.age[np.flatnonzero(self.alive)] += 1
