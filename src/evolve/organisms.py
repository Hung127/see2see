"""Organism state as structure-of-arrays.

The arrays are the authoritative state; :class:`OrganismView` is a frozen snapshot for
inspection, statistics and persistence. Slots are recycled deterministically (first
dead slot wins) and ``compact()`` removes dead slots with a stable remap so organism
IDs stay monotonic and never change meaning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OrganismConfig
from .genome import N_GENES, phenotype


@dataclass(frozen=True)
class OrganismView:
    """Read-only snapshot of one organism (for the inspector and statistics)."""

    index: int
    id: int
    x: float
    y: float
    heading: float
    energy: float
    health: float
    age: int
    alive: bool
    generation: int
    parent_a: int | None
    parent_b: int | None
    genome: np.ndarray
    phenotypes: np.ndarray
    counters: dict[str, float]


_COUNTERS = (
    "dist_moved", "bites", "hunts", "escapes", "repro_attempts", "repro_success",
    "rest_ticks", "plant_intake", "meat_intake", "damage_taken",
)


class Population:
    """SoA of organisms with deterministic slot management."""

    def __init__(self, cfg: OrganismConfig, capacity: int | None = None) -> None:
        self.cfg = cfg
        self.capacity = int(capacity if capacity is not None else cfg.max)
        n = self.capacity
        self.id = np.arange(n, dtype=np.int64)
        self._next_id = 0
        self.genome = np.zeros((n, N_GENES), dtype=np.float64)
        self.x = np.zeros(n, dtype=np.float64)
        self.y = np.zeros(n, dtype=np.float64)
        self.heading = np.zeros(n, dtype=np.float64)
        self.energy = np.zeros(n, dtype=np.float64)
        self.health = np.full(n, cfg.health_max, dtype=np.float64)
        self.age = np.zeros(n, dtype=np.int64)
        self.alive = np.zeros(n, dtype=bool)
        self.generation = np.zeros(n, dtype=np.int64)
        self.parent_a = np.full(n, -1, dtype=np.int64)
        self.parent_b = np.full(n, -1, dtype=np.int64)
        self.death_cause = np.full(n, "", dtype=object)
        for name in _COUNTERS:
            setattr(self, name, np.zeros(n, dtype=np.float64))

    # -- bookkeeping ---------------------------------------------------------
    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def add(self, genome: np.ndarray, x: float, y: float, energy: float,
            generation: int = 0, parents: tuple[int | None, int | None] = (None, None)) -> int:
        """Insert an organism; returns its slot index, or -1 when the cap is reached."""
        dead = np.flatnonzero(~self.alive)
        if dead.size == 0:
            if self.alive.size >= self.capacity:
                return -1
            self._grow()
            slot = self.alive.size - 1
        else:
            slot = int(dead[0])
        self.alive[slot] = True
        self.id[slot] = self._next_id
        self._next_id += 1
        self.genome[slot] = genome
        self.x[slot] = x
        self.y[slot] = y
        self.heading[slot] = 0.0
        self.energy[slot] = energy
        self.health[slot] = self.cfg.health_max
        self.age[slot] = 0
        self.generation[slot] = generation
        self.parent_a[slot] = -1 if parents[0] is None else parents[0]
        self.parent_b[slot] = -1 if parents[1] is None else parents[1]
        self.death_cause[slot] = ""
        for name in _COUNTERS:
            getattr(self, name)[slot] = 0.0
        return slot

    def kill(self, i: int, cause: str) -> None:
        self.alive[i] = False
        self.death_cause[i] = cause

    def compact(self) -> None:
        """Drop dead slots (stable remap; IDs are never renumbered)."""
        keep = np.flatnonzero(self.alive)
        dropped = self.alive.size - keep.size
        if dropped == 0:
            return
        for name in ("id", "genome", "x", "y", "heading", "energy", "health", "age",
                     "alive", "generation", "parent_a", "parent_b", "death_cause",
                     *_COUNTERS):
            setattr(self, name, getattr(self, name)[keep])
        self._grow_to(self.capacity)

    def _grow(self) -> None:
        self._grow_to(min(self.capacity, max(self.alive.size * 2, 64)))

    def _grow_to(self, n: int) -> None:
        if n <= self.alive.size:
            return
        old = self.alive.size
        for name in ("id", "x", "y", "heading", "energy", "health", "age", "alive",
                     "generation", "parent_a", "parent_b", "death_cause", *_COUNTERS):
            arr = getattr(self, name)
            fill = np.zeros(n - old, dtype=arr.dtype) if arr.dtype != object else np.full(
                n - old, "", dtype=object)
            setattr(self, name, np.concatenate([arr, fill]))
        genome = np.zeros((n, N_GENES), dtype=np.float64)
        genome[:old] = self.genome
        self.genome = genome

    # -- access --------------------------------------------------------------
    def phenotypes(self) -> np.ndarray:
        """Phenotypes of living organisms, in slot order (compacted)."""
        return phenotype(self.genome[self.alive])

    def view(self, i: int) -> OrganismView:
        return OrganismView(
            index=int(i),
            id=int(self.id[i]),
            x=float(self.x[i]),
            y=float(self.y[i]),
            heading=float(self.heading[i]),
            energy=float(self.energy[i]),
            health=float(self.health[i]),
            age=int(self.age[i]),
            alive=bool(self.alive[i]),
            generation=int(self.generation[i]),
            parent_a=int(self.parent_a[i]) if self.parent_a[i] >= 0 else None,
            parent_b=int(self.parent_b[i]) if self.parent_b[i] >= 0 else None,
            genome=self.genome[i].copy(),
            phenotypes=phenotype(self.genome[i : i + 1])[0],
            counters={name: float(getattr(self, name)[i]) for name in _COUNTERS},
        )
