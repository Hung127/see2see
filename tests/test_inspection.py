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


def test_inspect_matches_vectors_by_id_not_slot():
    """Rows of ``last_sensors`` are in think-time order, slots are re-sorted.

    If any organism dies after thinking, slot ``i`` no longer corresponds to
    row ``i``; the payload must still return the vectors belonging to the
    organism at that slot. The death is forced here because a 30-organism
    scene can run for hundreds of ticks without one.
    """
    eng = Engine(small_config(organisms=30, seed=1))
    eng.step()
    pop = eng.population
    assert eng.last_sensor_ids.size == pop.n_alive
    # Kill the two lowest slots, then re-densify exactly as ``step`` does.
    pop.alive[0] = False
    pop.alive[1] = False
    pop.compact()
    pop.sort_by_id()
    shifted = False
    for i in np.flatnonzero(pop.alive):
        d = eng.inspect(int(i))
        row = int(np.flatnonzero(eng.last_sensor_ids == d["id"])[0])
        shifted |= row != int(i)
        assert d["has_last_vectors"] is True
        assert np.array_equal(d["sensors"], eng.last_sensors[row])
        assert np.array_equal(d["outputs"], eng.last_outputs[row])
    assert shifted, "slot order did not shift; the regression is untested"


def test_inspect_reports_no_vectors_instead_of_someone_elses():
    """A newborn has no think-time row, so it must report zeros, not a neighbour's."""
    eng = Engine(small_config(organisms=20, seed=1))
    eng.step()
    # Give an organism a fresh id that appears in no think-time row.
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    i = int(idx[0])
    real_id = int(pop.id[i])
    pop.id[i] = 10_000
    d = eng.inspect(i)
    assert d["id"] == 10_000
    assert d["has_last_vectors"] is False
    assert not np.any(d["sensors"]) and not np.any(d["outputs"])
    pop.id[i] = real_id

