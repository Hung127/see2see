"""Seeded random number generation.

EVOLVE uses exactly one generator per simulation. Sub-streams (experiments, assay
replicates) are spawned from ``SeedSequence`` so that a run is reproducible from its
seed alone. The draw *order* is part of the contract; it is documented in
``docs/ARCHITECTURE.md`` and asserted by determinism tests.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np


class Rng:
    """Thin wrapper around ``numpy.random.Generator(PCG64(seed))``."""

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._g = np.random.Generator(np.random.PCG64(self._seed))

    # -- draws ---------------------------------------------------------------
    def random(self, size: int | tuple[int, ...] | None = None) -> np.ndarray | float:
        return self._g.random(size)

    def normal(self, loc: float = 0.0, scale: float = 1.0,
               size: int | tuple[int, ...] | None = None) -> np.ndarray | float:
        return self._g.normal(loc, scale, size)

    def integers(self, low: int, high: int | None = None,
                 size: int | tuple[int, ...] | None = None) -> np.ndarray | int:
        return self._g.integers(low, high, size)

    def choice(self, a: Sequence[Any] | int, size: int | tuple[int, ...] | None = None,
               replace: bool = True, p: Sequence[float] | None = None) -> np.ndarray:
        return self._g.choice(a, size=size, replace=replace, p=p)

    def beta(self, a: float, b: float,
             size: int | tuple[int, ...] | None = None) -> np.ndarray | float:
        return self._g.beta(a, b, size)

    def poisson(self, lam: float, size: int | tuple[int, ...] | None = None) -> np.ndarray | int:
        return self._g.poisson(lam, size)

    def permutation(self, n: int) -> np.ndarray:
        return self._g.permutation(n)

    # -- state ---------------------------------------------------------------
    def state(self) -> bytes:
        """Serialise the bit-generator state (part of every snapshot)."""
        return json.dumps(self._g.bit_generator.state, sort_keys=True).encode("utf-8")

    def set_state(self, blob: bytes) -> None:
        self._g.bit_generator.state = json.loads(blob.decode("utf-8"))

    # -- streams -------------------------------------------------------------
    def spawn(self, n: int) -> list["Rng"]:
        """Independent child streams, deterministic for a given seed."""
        children = np.random.SeedSequence(self._seed).spawn(n)
        return [Rng(int(child.generate_state(1)[0])) for child in children]

    @property
    def generator(self) -> np.random.Generator:
        return self._g
