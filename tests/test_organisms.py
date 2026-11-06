"""Tests for the organism structure-of-arrays and read-only views."""
import dataclasses

import numpy as np
import pytest

from evolve.config import OrganismConfig
from evolve.genome import founders
from evolve.organisms import Population
from evolve.rng import Rng


def make_pop(n=3, capacity=10):
    pop = Population(OrganismConfig(), capacity=capacity)
    genes = founders(n, Rng(1))
    for i in range(n):
        pop.add(genes[i], 10.0 * i, 5.0, energy=5.0, generation=i, parents=(None, None))
    return pop


def test_add_assigns_monotonic_ids():
    pop = make_pop()
    assert pop.n_alive == 3
    assert list(pop.id[:3]) == [0, 1, 2]
    assert list(pop.generation[:3]) == [0, 1, 2]


def test_add_kill_compact_preserves_ids():
    pop = make_pop()
    ids = pop.id[:pop.n_alive].copy()
    genomes = pop.genome[:pop.n_alive].copy()
    pop.kill(1, "starvation")
    pop.compact()
    assert pop.n_alive == 2
    assert np.array_equal(pop.id[:2], ids[[0, 2]])
    assert np.allclose(pop.genome[:2], genomes[[0, 2]])


def test_capacity_respected():
    pop = make_pop(capacity=3)
    assert pop.add(founders(1, Rng(2))[0], 0.0, 0.0, 1.0, 0, (None, None)) == -1


def test_view_readonly():
    pop = make_pop()
    v = pop.view(0)
    assert v.energy == 5.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.energy = 9.0


def test_counters_start_zero():
    pop = make_pop()
    for name in ("bites", "hunts", "escapes", "repro_attempts", "repro_success",
                 "rest_ticks", "plant_intake", "meat_intake", "dist_moved"):
        assert np.all(getattr(pop, name)[:pop.n_alive] == 0.0), name


def test_phenotypes_shape():
    pop = make_pop()
    phen = pop.phenotypes()
    assert phen.shape == (3, 12)
