"""Tests for the per-organism inspection payload."""
import numpy as np

from evolve.engine import Engine
from evolve.genome import N_TRAITS
from tests.test_engine_m1 import small_config


def test_inspect_roundtrip():
    eng = Engine(small_config(organisms=5, seed=2))
    eng.step()
    d = eng.inspect(0)
    pop = eng.population
    assert d["id"] == int(pop.id[0])
    assert d["age"] == int(pop.age[0])
    assert d["energy"] == float(pop.energy[0])
    assert d["body"] == float(pop.body[0])
    assert d["health"] == float(pop.health[0])
    assert d["generation"] == int(pop.generation[0])
    assert d["position"] == (float(pop.x[0]), float(pop.y[0]))
    assert d["heading"] == float(pop.heading[0])
    assert d["phenotypes"].shape == (12,)
    assert np.allclose(d["phenotypes"], pop.phenotypes()[0])
    w = d["weights"]
    assert w["W1"].shape == (16, 12) and w["b1"].shape == (12,)
    assert w["W2"].shape == (12, 8) and w["b2"].shape == (8,)
    assert w["W3"].shape == (8, 8) and w["b3"].shape == (8,)
    # weights come from the genome's neural block
    assert np.allclose(w["W1"].ravel(), pop.genome[0, N_TRAITS:N_TRAITS + 192])
    assert d["sensors"].shape == (16,)
    assert d["outputs"].shape == (8,)
    assert np.allclose(d["outputs"], eng.last_outputs[0])
    assert d["counters"]["bites"] == float(pop.bites[0])


def test_inspect_parents():
    eng = Engine(small_config(organisms=5, seed=3))
    d = eng.inspect(0)
    assert d["parents"] == (None, None)  # founders
