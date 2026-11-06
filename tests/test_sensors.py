"""Tests for the 16-channel phenotype-based sensor vector."""
import numpy as np
import pytest

from evolve.config import EnvironmentConfig, OrganismConfig, PlantConfig, WorldConfig
from evolve.fields import Fields
from evolve.genome import N_GENES
from evolve.organisms import Population
from evolve.plants import Plants
from evolve.rng import Rng
from evolve.sensors import collect
from evolve.world import World


def make_scene(width=500.0, n=10):
    wc = WorldConfig(width=width, height=width, n_field_cells=n, cell_size=width / n)
    env = EnvironmentConfig(rain_rate=0.0, m_amp=0.0, t_amp=0.0, t_mean=22.0, m_mean=0.7)
    fields = Fields(wc, env, Rng(0))
    fields.step(0)
    fields.soil[:] = 10.0
    return wc, World(wc), fields


def organism(pop, x, y, heading=0.0, body=0.428, vision=1.0, angle=1.0):
    """Add an organism and pin the perceptual genes that matter for these tests."""
    g = np.full((1, N_GENES), 0.5)
    g[0, 0] = body           # body_size
    g[0, 2] = vision         # vision_range (1.0 -> 160)
    g[0, 3] = angle          # vision_angle (1.0 -> 360)
    slot = pop.add(g[0], x, y, energy=1.0)
    pop.heading[slot] = heading
    return slot


def test_nothing_detected_is_minus_one():
    wc, world, fields = make_scene()
    pop = Population(OrganismConfig(), capacity=10)
    organism(pop, 250.0, 250.0)
    plants = Plants(PlantConfig(), wc)
    s = collect(pop, plants, fields, world, OrganismConfig())
    assert s.shape == (1, 16)
    assert np.all(s[0, [0, 1, 2, 3, 4, 5, 6]] == -1.0)


def test_cone_masks_behind_and_sees_ahead():
    wc, world, fields = make_scene()
    pop = Population(OrganismConfig(), capacity=10)
    organism(pop, 250.0, 250.0, heading=0.0, angle=(90.0 - 30.0) / 330.0)  # 90 deg cone
    plants = Plants(PlantConfig(), wc)
    plants.add(300.0, 250.0, 1.0)   # dead ahead, 50 units
    plants.add(200.0, 250.0, 1.0)   # directly behind
    s = collect(pop, plants, fields, world, OrganismConfig())
    assert s[0, 0] > 0.0                       # food seen ahead
    assert abs(s[0, 1]) < 0.05                 # bearing ~ 0


def test_range_masks():
    wc, world, fields = make_scene()
    pop = Population(OrganismConfig(), capacity=10)
    organism(pop, 250.0, 250.0, heading=0.0, vision=0.0)  # 40 units
    plants = Plants(PlantConfig(), wc)
    plants.add(300.0, 250.0, 1.0)  # 50 units > 40
    s = collect(pop, plants, fields, world, OrganismConfig())
    assert s[0, 0] == -1.0


def test_nearest_food_wins():
    wc, world, fields = make_scene()
    pop = Population(OrganismConfig(), capacity=10)
    organism(pop, 250.0, 250.0, heading=0.0)
    plants = Plants(PlantConfig(), wc)
    plants.add(290.0, 250.0, 1.0)   # 40 units
    plants.add(320.0, 250.0, 1.0)   # 70 units
    s = collect(pop, plants, fields, world, OrganismConfig())
    assert s[0, 0] == pytest.approx(1.0 - 40.0 / 160.0, rel=1e-6)


def test_energy_and_environment_channels():
    wc, world, fields = make_scene()
    pop = Population(OrganismConfig(), capacity=10)
    organism(pop, 100.0, 100.0)
    pop.energy[0] = 5.0
    pop.health[0] = 50.0
    pop.age[0] = 100
    plants = Plants(PlantConfig(), wc)
    s = collect(pop, plants, fields, world, OrganismConfig())
    phen = pop.phenotypes()[0]
    emax = 10.0 * phen[0] ** 0.9
    assert s[0, 7] == pytest.approx(5.0 / emax)
    assert s[0, 8] == pytest.approx(0.5)
    assert s[0, 9] == pytest.approx(min(100.0 / phen[10], 1.0))
    assert s[0, 10] == pytest.approx((22.0 - 20.0) / 15.0, abs=1e-9)
    assert s[0, 11] == pytest.approx(2 * 0.7 - 1.0, abs=1e-9)


def test_normalisation_bounds():
    wc, world, fields = make_scene()
    pop = Population(OrganismConfig(), capacity=60)
    rng = Rng(7)
    from evolve.genome import founders
    genes = founders(50, rng)
    xs = rng.random(50) * wc.width
    ys = rng.random(50) * wc.height
    for i in range(50):
        pop.add(genes[i], float(xs[i]), float(ys[i]), energy=1.0)
        pop.heading[i] = float(rng.random() * 2 * np.pi - np.pi)
        pop.energy[i] = float(rng.random() * 10.0)
    plants = Plants(PlantConfig(), wc)
    px = rng.random(100) * wc.width
    py = rng.random(100) * wc.height
    for i in range(100):
        plants.add(float(px[i]), float(py[i]), 1.0)
    s = collect(pop, plants, fields, world, OrganismConfig())
    assert s.shape == (50, 16)
    assert np.all(np.isfinite(s))
    assert s.min() >= -1.0 - 1e-12 and s.max() <= 1.0 + 1e-12


def test_sensors_do_not_import_actions():
    import inspect

    import evolve.sensors as sensors
    src = inspect.getsource(sensors)
    imports = [ln for ln in src.splitlines()
               if ln.strip().startswith(("import ", "from "))]
    assert not any("actions" in ln for ln in imports)
