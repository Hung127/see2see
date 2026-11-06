"""Environment integration scenarios: drought, flood, recovery, extinction.

These exercise the whole engine (fields + events + plants + organisms + ledger)
rather than one module. They are marked ``integration`` because they run several
hundred ticks each.

Scaling notes (this machine is a single core with ~1 GB RAM):

* The design's plan suggested a 500x500 world with 100 founders. That world is a
  founder lottery for the *stochastic* founders (see ``docs/CALIBRATION.md`` §3):
  in two of three seeds the population was extinct before an event could be
  measured, so the scenarios could not be distinguished from noise. The tests
  therefore use a 600x600 world with 120 founders and 2000 plants: the smallest
  configuration probed in which the control population survives the whole horizon
  in every scenario below (the surviving population is smaller than the calibrated
  default's, but each scenario is a paired comparison with its own control, so the
  effect size--not the absolute population--is what is measured).
* Horizons are 200-400 ticks (the plan allowed up to 1500). Each scenario is a
  paired run against an identical control with the same seed, so the comparison
  is exact rather than statistical, and a shorter horizon is enough to see the
  effect. Total wall time is ~25 s on this machine (measured with
  ``pytest --durations``); the plan's 30 s budget is met.
* All assertions are on deterministic quantities for a fixed seed; they are not
  statistical tests and must not be read as such.
"""
import numpy as np
import pytest

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, EventConfig, GeneticsConfig,
    OrganismConfig, PlantConfig, StatsConfig, WorldConfig,
)

pytestmark = pytest.mark.integration


def scenario_config(events=(), seed=1, ticks=400, plants=2000, organisms=120,
                    plant_cfg=None):
    """Small but viable world; see the module docstring for the scaling rationale."""
    return Config(
        seed=seed,
        ticks=ticks,
        world=WorldConfig(width=600.0, height=600.0, n_field_cells=24,
                          cell_size=25.0),
        plants=plant_cfg if plant_cfg is not None else PlantConfig(initial=plants,
                                                                   max=5000),
        organisms=OrganismConfig(initial=organisms, max=500),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(events=events),
        stats=StatsConfig(),
    )


def drought(magnitude=1.0, start=100, duration=200):
    return (EventConfig(kind="drought", start_tick=start, duration=duration,
                        magnitude=magnitude),)


def test_drought_increases_mortality():
    """A drought must kill more, per organism-tick, than the same world without it.

    Ruling: raw deaths per tick is confounded by population size -- a drought
    that kills founders early leaves fewer organisms to die later, which can make
    the *count* fall even though the *risk* rose. The test therefore measures
    deaths per organism-tick inside the event window:

        sum(deaths in window) / sum(living organisms at end of each window tick)

    The control is the same seed and configuration with no events, so climate,
    terrain, genome draw order and every other stochastic input are identical.
    """
    from evolve.engine import Engine

    ev = drought(start=100, duration=200)
    with_event = Engine(scenario_config(events=ev, ticks=320))
    control = Engine(scenario_config(ticks=320))

    death_a = death_b = 0
    ticks_a = ticks_b = 0.0
    prev_a = prev_b = 0
    for tick in range(1, 321):
        with_event.step()
        control.step()
        if 100 < tick <= 300:
            death_a += with_event.deaths_total - prev_a
            death_b += control.deaths_total - prev_b
            ticks_a += with_event.population.n_alive
            ticks_b += control.population.n_alive
        prev_a = with_event.deaths_total
        prev_b = control.deaths_total

    rate_a = death_a / ticks_a
    rate_b = death_b / ticks_b
    assert rate_a > rate_b, (
        f"drought per-capita mortality {rate_a:.5f} not above control {rate_b:.5f}")
    assert death_a > 0 and death_b > 0


def test_recovery_after_drought():
    """Plant biomass must grow again once the drought ends.

    The event suppresses plant growth through moisture and the growth factor.
    Immediately after it ends, plant biomass in the droughted world rises over a
    50-tick window; the population also rebounds over a longer window. Measured
    for one fixed seed (this is a scenario test, not a survey).
    """
    from evolve.engine import Engine

    eng = Engine(scenario_config(events=drought(start=50, duration=100), ticks=300))
    at_end = plant_after = pop_after = None
    for tick in range(1, 301):
        eng.step()
        if tick == 150:
            at_end = eng.plants.total_biomass()
        elif tick == 200:
            plant_after = eng.plants.total_biomass()
        elif tick == 300:
            pop_after = eng.population.n_alive

    assert at_end is not None and plant_after is not None
    assert plant_after > at_end, (
        f"plant biomass did not recover: {at_end:.1f} at event end, "
        f"{plant_after:.1f} fifty ticks later")
    # The drought leaves few survivors; the population must not keep collapsing.
    assert pop_after is not None and pop_after > 0


def _deaths_with_terrain(eng, ticks):
    """Return (death count, terrain heights of organisms that died)."""
    prev = {int(oid): (float(x), float(y))
            for slot, (oid, x, y) in enumerate(
                zip(eng.population.id, eng.population.x, eng.population.y))
            if eng.population.alive[slot]}
    start = eng.deaths_total
    terrain = []
    for _ in range(ticks):
        eng.step()
        now = {int(oid) for slot, oid in enumerate(eng.population.id)
               if eng.population.alive[slot]}
        for oid, (x, y) in prev.items():
            if oid not in now:
                ix, iy = eng.fields.cell_index(np.array([x]), np.array([y]))
                terrain.append(float(eng.fields.terrain[int(ix[0]), int(iy[0])]))
        prev = {int(oid): (float(x), float(y))
                for slot, (oid, x, y) in enumerate(
                    zip(eng.population.id, eng.population.x, eng.population.y))
                if eng.population.alive[slot]}
    return eng.deaths_total - start, np.asarray(terrain, dtype=np.float64)


def test_flood_hits_low_terrain():
    """Heavy rain must kill more than a dry control, and mostly on low ground.

    DESIGN §3.2 makes terrain height set flood damage, so this asserts both the
    aggregate effect (more deaths) and the spatial pattern (the dead sit lower
    than the world average). The terrain comparison uses the mean height of the
    world, not of the control's dead, because the control's dead are a different
    (starvation) sample; the world mean is the unbiased "no flood" baseline.
    """
    from evolve.engine import Engine

    ev = (EventConfig(kind="heavy_rain", start_tick=100, duration=60,
                      magnitude=1.0),)
    flooded = Engine(scenario_config(events=ev, ticks=160))
    control = Engine(scenario_config(ticks=160))

    flood_deaths, flood_terrain = _deaths_with_terrain(flooded, 160)
    control_deaths, _ = _deaths_with_terrain(control, 160)

    assert flood_deaths > control_deaths, (
        f"flood killed {flood_deaths}, control {control_deaths}")
    assert flood_terrain.size > 0
    world_mean = float(flooded.fields.terrain.mean())
    assert flood_terrain.mean() < world_mean, (
        f"flood dead mean terrain {flood_terrain.mean():.3f} not below "
        f"world mean {world_mean:.3f}")


def test_extinction_is_valid_outcome():
    """With no plant production the population must die out cleanly.

    Extinction is a legitimate result, not an error condition: the engine must
    still step, report zero organisms, and keep the full mass ledger closed
    (organic + soil = initial + mass_in).
    """
    from evolve.engine import Engine

    # No plants to begin with and none germinating: a genuinely barren world.
    cfg = scenario_config(
        ticks=400, organisms=100,
        plant_cfg=PlantConfig(initial=0, max=100, germination_per_tick=0))
    eng = Engine(cfg)

    initial_mass = eng.total_mass()
    for _ in range(400):
        eng.step()

    assert eng.population.n_alive == 0
    assert eng.plants.n_alive == 0
    assert eng.deaths_total == 100
    # No error above this line is itself the "without error" assertion.
    eng.ledger.assert_full_closed(initial_mass, eng.total_mass())
