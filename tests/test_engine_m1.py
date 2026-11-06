"""Tests for the M1 tick pipeline: determinism, stability, cap, conservation."""
import numpy as np

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine


def small_config(organisms=100, plants=200, width=500.0, max_pop=150, seed=1,
                 ticks=100, rain=0.0, maturity=50):
    return Config(
        seed=seed,
        ticks=ticks,
        world=WorldConfig(width=width, height=width, n_field_cells=10,
                          cell_size=width / 10.0),
        plants=PlantConfig(initial=plants, max=1000),
        organisms=OrganismConfig(initial=organisms, max=max_pop, maturity_age=maturity),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=rain),
        stats=StatsConfig(),
    )


def test_initial_state_is_seeded():
    eng = Engine(small_config(organisms=30, plants=50))
    assert eng.population.n_alive == 30
    assert eng.plants.n_alive == 50
    assert np.all(eng.population.energy[eng.population.alive] > 0)
    assert np.all(eng.population.body[eng.population.alive] > 0)


def test_seed_reproducibility():
    """Same seed -> identical state at every checkpoint; different seed -> not."""
    a = Engine(small_config(seed=7))
    b = Engine(small_config(seed=7))
    for target in (1, 10, 100):
        while a.tick < target:
            a.step()
            b.step()
        assert a.checksum() == b.checksum(), target


def test_different_seeds_differ():
    a = Engine(small_config(seed=1))
    b = Engine(small_config(seed=2))
    a.step()
    b.step()
    assert a.checksum() != b.checksum()


def test_no_nan_after_2000_ticks():
    eng = Engine(small_config(organisms=100, plants=300, seed=3))
    for _ in range(2000):
        eng.step()
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    for name in ("x", "y", "heading", "energy", "body", "health"):
        assert np.all(np.isfinite(getattr(pop, name)[idx])), name
    assert np.all(np.isfinite(eng.plants.biomass[eng.plants.alive]))
    assert np.isfinite(eng.detritus)
    assert eng.detritus >= 0.0


def test_population_capped():
    eng = Engine(small_config(organisms=30, plants=20, max_pop=10))
    assert eng.population.n_alive == 10
    for _ in range(20):
        eng.step()
    assert eng.population.n_alive <= 10


def test_mass_and_energy_conserved_m1():
    eng = Engine(small_config(organisms=80, plants=300, seed=5))
    m0, e0 = eng.initial_mass, eng.initial_energy
    for _ in range(200):
        eng.step()
    eng.ledger.assert_closed(m0, eng.mass_stock(), e0, eng.energy_stock())


def test_ledger_counters_move():
    eng = Engine(small_config(organisms=50, plants=100, seed=6, rain=0.01))
    for _ in range(20):
        eng.step()
    assert eng.ledger.light_in > 0.0  # plants grew / germinated
    assert eng.ledger.mass_to_soil > 0.0  # something respired or decayed


def test_checksum_changes_when_state_changes():
    eng = Engine(small_config(organisms=20, plants=20, seed=8))
    before = eng.checksum()
    eng.step()
    assert eng.checksum() != before


def test_organism_ids_sorted_after_tick():
    eng = Engine(small_config(organisms=30, plants=30, seed=9))
    for _ in range(5):
        eng.step()
    ids = eng.population.id[eng.population.alive]
    assert np.all(np.diff(ids) > 0)
