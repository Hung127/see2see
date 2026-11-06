"""Tests for environmental events (DESIGN §3.3): physical effects only."""
import numpy as np
import pytest

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, EventConfig, GeneticsConfig,
    OrganismConfig, PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine
from evolve.fields import Fields
from evolve.plants import Plants
from evolve.rng import Rng


def make_config(events=(), random_events=(), seed=1, n=10):
    return Config(
        seed=seed, ticks=100,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=n,
                          cell_size=500.0 / n),
        plants=PlantConfig(initial=100, max=300),
        organisms=OrganismConfig(initial=40, max=100),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.0, events=events,
                                      random_events=random_events),
        stats=StatsConfig(),
    )


def test_drought_reduces_moisture_and_growth():
    eng = Engine(make_config(events=(EventConfig(kind="drought", start_tick=3,
                                                 duration=10, magnitude=0.8),)))
    for _ in range(2):
        eng.step()
    base_moisture = eng.fields.moisture.copy()
    eng.events.update(3, eng)
    assert np.all(eng.fields.moisture < base_moisture)
    assert np.all(eng.fields.growth_factor() < 1.0)

    # and plants.grow respects the factor
    from evolve.energy import Ledger
    from evolve.plants import Plants

    world = eng.cfg.world
    fields = eng.fields
    fields.soil[:] = 10.0
    fields._growth_factor.fill(1.0)
    normal = Plants(PlantConfig(), world)
    normal.add(100.0, 100.0, 3.0)
    normal.grow(fields, Ledger())
    grew_normal = float(normal.biomass[0] - 3.0)
    fields.soil[:] = 10.0
    fields._growth_factor.fill(0.5)
    stressed = Plants(PlantConfig(), world)
    stressed.add(100.0, 100.0, 3.0)
    stressed.grow(fields, Ledger())
    grew_stressed = float(stressed.biomass[0] - 3.0)
    assert 0.0 < grew_stressed < grew_normal


def test_wildfire_books_soil_and_heat():
    eng = Engine(make_config(events=(EventConfig(kind="wildfire", start_tick=3,
                                                 duration=1, magnitude=1.0),)))
    for _ in range(2):
        eng.step()
    p0 = eng.plants.total_biomass()
    assert p0 > 0.0
    soil0, heat0 = eng.ledger.mass_to_soil, eng.ledger.heat_out
    eng.events.update(3, eng)  # fire the scheduled wildfire directly
    assert eng.plants.total_biomass() == pytest.approx(0.0, abs=1e-9)
    assert eng.ledger.mass_to_soil - soil0 == pytest.approx(p0, rel=1e-9)
    assert eng.ledger.heat_out - heat0 == pytest.approx(p0, rel=1e-9)


def test_nutrient_pulse_books_mass_in():
    eng = Engine(make_config(events=(EventConfig(kind="nutrient_pulse", start_tick=3,
                                                 duration=1, magnitude=10.0),)))
    for _ in range(2):
        eng.step()
    soil0 = float(eng.fields.soil.sum())
    mass_in0 = eng.ledger.mass_in
    eng.events.update(3, eng)
    assert float(eng.fields.soil.sum()) == pytest.approx(soil0 + 10.0 * 100, rel=1e-9)
    assert eng.ledger.mass_in - mass_in0 == pytest.approx(10.0 * 100, rel=1e-9)


def test_plant_blight_moves_to_detritus():
    eng = Engine(make_config(events=(EventConfig(kind="plant_blight", start_tick=3,
                                                 duration=1, magnitude=0.5),)))
    for _ in range(2):
        eng.step()
    p0 = eng.plants.total_biomass()
    det0 = eng.detritus
    soil0, heat0 = eng.ledger.mass_to_soil, eng.ledger.heat_out
    eng.events.update(3, eng)
    lost = p0 - eng.plants.total_biomass()
    assert lost > 0.0
    assert eng.detritus == pytest.approx(det0 + lost, rel=1e-9)
    assert eng.ledger.mass_to_soil == soil0  # killed, not burned
    assert eng.ledger.heat_out == heat0


def test_events_never_touch_genomes():
    eng = Engine(make_config(events=(
        EventConfig(kind="wildfire", start_tick=5, duration=1, magnitude=1.0),
        EventConfig(kind="drought", start_tick=5, duration=10, magnitude=0.9),
    )))
    for _ in range(4):
        eng.step()
    before = {int(i): g.tobytes() for i, g in
              zip(eng.population.id, eng.population.genome)}
    for _ in range(10):
        eng.step()
    after = {int(i): g.tobytes() for i, g in
             zip(eng.population.id, eng.population.genome)}
    for oid, genome in before.items():
        assert after.get(oid) == genome, f"genome {oid} changed by an event"


def test_scheduled_events_fire_on_time():
    cfg = make_config(events=(EventConfig(kind="drought", start_tick=10,
                                          duration=5, magnitude=0.8),))
    plain = make_config()
    eng = Engine(cfg)
    twin = Engine(plain)
    for _ in range(9):
        eng.step()
        twin.step()
    assert np.allclose(eng.fields.moisture, twin.fields.moisture)
    for _ in range(5):  # ticks 10..14: drought active
        eng.step()
        twin.step()
        assert np.all(eng.fields.moisture < twin.fields.moisture - 1e-6)
    eng.step()  # tick 15: over
    twin.step()
    assert np.allclose(eng.fields.moisture, twin.fields.moisture)


def test_random_events_spawn_by_poisson():
    cfg = make_config(random_events=(("heatwave", 0.5),), seed=11)
    eng = Engine(cfg)
    fired = 0
    for _ in range(20):
        n_before = len(eng.events.log)
        eng.step()
        fired += len(eng.events.log) - n_before
    assert fired > 0  # with rate 0.5/tick over 20 ticks, zero is essentially impossible
    for ev in eng.events.log:
        assert ev.kind == "heatwave"
        assert ev.start_tick >= 1


def test_event_region_disc_limits_effects():
    eng = Engine(make_config(events=(EventConfig(kind="nutrient_pulse", start_tick=3,
                                                 duration=1, magnitude=10.0,
                                                 region=(250.0, 250.0, 100.0)),)))
    for _ in range(2):
        eng.step()
    soil0 = eng.fields.soil.copy()
    eng.events.update(3, eng)
    changed = eng.fields.soil != soil0
    assert np.any(changed)
    assert not np.all(changed)  # only the disc
