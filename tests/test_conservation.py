"""Ledger closure across every channel, and roles measured from intake."""
import numpy as np
import pytest

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, EventConfig, GeneticsConfig,
    OrganismConfig, PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine
from evolve.stats import classify_roles


def small_config(ticks=2000, seed=5):
    return Config(
        seed=seed, ticks=ticks,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=10,
                          cell_size=50.0),
        plants=PlantConfig(initial=200, max=800),
        organisms=OrganismConfig(initial=100, max=250),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(
            rain_rate=0.0,
            events=(
                EventConfig(kind="plant_blight", start_tick=300, duration=1,
                            magnitude=0.5),
                EventConfig(kind="nutrient_pulse", start_tick=700, duration=1,
                            magnitude=2.0),
                EventConfig(kind="wildfire", start_tick=1200, duration=1,
                            magnitude=1.0),
            ),
        ),
        stats=StatsConfig(),
    )


def test_mass_and_energy_conservation_all_channels():
    eng = Engine(small_config())
    m0, e0 = eng.initial_mass, eng.initial_energy
    total0 = eng.initial_total
    for _ in range(2000):
        eng.step()
    eng.ledger.assert_closed(m0, eng.mass_stock(), e0, eng.energy_stock())
    eng.ledger.assert_full_closed(total0, eng.total_mass())
    # every channel actually fired during the run
    assert eng.ledger.light_in > 0.0
    assert eng.ledger.heat_out > 0.0
    assert eng.ledger.mass_to_soil > 0.0
    assert eng.ledger.mass_from_soil > 0.0
    assert eng.ledger.mass_in > 0.0  # nutrient pulse


def test_roles_measured_not_declared():
    labels = classify_roles(np.array([10.0, 0.0, 5.0, 0.4, 3.0]),
                            np.array([0.0, 10.0, 4.0, 0.4, 0.0]))
    assert labels[0] == "herbivore"   # only plants
    assert labels[1] == "carnivore"   # only meat
    assert labels[2] == "omnivore"    # mixed
    assert labels[3] == "unclassified"  # below 1 bu lifetime intake
    assert labels[4] == "herbivore"   # meat == 0


def test_engine_debug_asserts_closure():
    eng = Engine(small_config(ticks=100, seed=9), debug=True)
    for _ in range(100):
        eng.step()  # would raise if any channel opened the ledger


def test_roles_vectorised_shape():
    n = 7
    labels = classify_roles(np.full(n, 2.0), np.full(n, 0.5))
    assert len(labels) == n
    assert all(label in ("herbivore", "carnivore", "omnivore", "unclassified")
               for label in labels)
