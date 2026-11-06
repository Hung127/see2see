"""Tests for the plant pool (logistic growth, contention, germination, grazing)."""
import numpy as np
import pytest

from evolve.config import EnvironmentConfig, PlantConfig, WorldConfig
from evolve.energy import Ledger
from evolve.fields import Fields
from evolve.plants import Plants
from evolve.rng import Rng


def make_world(width=500.0, n=10):
    wc = WorldConfig(width=width, height=width, n_field_cells=n, cell_size=width / n)
    env = EnvironmentConfig(rain_rate=0.0, m_amp=0.0, t_amp=0.0, t_mean=22.0, m_mean=0.7)
    fields = Fields(wc, env, Rng(0))
    fields.step(0)
    return wc, fields


def make_plants(wc, **kw):
    return Plants(PlantConfig(**kw), wc)


def cell_of(fields, x, y):
    ix, iy = fields.cell_index(np.array([x]), np.array([y]))
    return int(ix[0]), int(iy[0])


def test_seed_initial_inside_world():
    wc, fields = make_world()
    plants = make_plants(wc, initial=50)
    n = plants.seed_initial(50, Rng(1))
    assert n == 50
    assert np.all((plants.x[:50] >= 0) & (plants.x[:50] < wc.width))
    assert np.all((plants.y[:50] >= 0) & (plants.y[:50] < wc.height))
    assert np.all(plants.biomass[:50] > 0)


def test_growth_nutrient_limited():
    wc, fields = make_world()
    plants = make_plants(wc)
    plants.add(100.0, 100.0, 3.0)
    ix, iy = cell_of(fields, 100.0, 100.0)
    fields.soil[ix, iy] = 0.0
    led = Ledger()
    plants.grow(fields, led)
    assert plants.biomass[0] == pytest.approx(3.0)
    assert led.light_in == 0.0


def test_growth_paid_from_soil_and_light():
    wc, fields = make_world()
    plants = make_plants(wc)
    plants.add(100.0, 100.0, 3.0)
    ix, iy = cell_of(fields, 100.0, 100.0)
    fields.soil[ix, iy] = 10.0
    led = Ledger()
    before_soil = float(fields.soil[ix, iy])
    before_bio = float(plants.biomass[0])
    plants.grow(fields, led)
    grew = float(plants.biomass[0]) - before_bio
    assert grew > 0.0
    assert float(fields.soil[ix, iy]) == pytest.approx(before_soil - grew, rel=1e-12)
    assert led.light_in == pytest.approx(grew, rel=1e-12)
    assert led.mass_from_soil == pytest.approx(grew, rel=1e-12)


def test_contention_proportional():
    wc, fields = make_world()
    plants = make_plants(wc)
    plants.add(100.0, 100.0, 3.0)
    plants.add(105.0, 105.0, 3.0)  # same 50-unit cell
    ix, iy = cell_of(fields, 100.0, 100.0)
    fields.soil[ix, iy] = 0.01
    before = plants.biomass[:2].copy()
    led = Ledger()
    plants.grow(fields, led)
    growth = plants.biomass[:2] - before
    assert growth.sum() == pytest.approx(0.01, rel=1e-12)
    assert growth[0] == pytest.approx(growth[1], rel=1e-6)
    assert float(fields.soil[ix, iy]) == pytest.approx(0.0, abs=1e-12)


def test_germination_costs_soil_and_light():
    wc, fields = make_world()
    plants = make_plants(wc)
    plants.add(100.0, 100.0, 3.0)
    ix, iy = cell_of(fields, 100.0, 100.0)
    fields.soil[ix, iy] = 1.0
    led = Ledger()
    made = plants.germinate(Rng(1), fields, led)
    assert made == 5  # 1.0 bu / 0.2 bu per seed
    assert plants.n_alive == 1 + made
    assert float(fields.soil[ix, iy]) == pytest.approx(0.0, abs=1e-12)
    assert led.light_in == pytest.approx(0.2 * made)
    assert led.mass_from_soil == pytest.approx(0.2 * made)


def test_germination_respects_capacity():
    wc, fields = make_world()
    plants = make_plants(wc, max=2, initial=2, germination_per_tick=20)
    plants.add(100.0, 100.0, 3.0)
    plants.add(110.0, 110.0, 3.0)
    fields.soil[:] = 10.0
    made = plants.germinate(Rng(2), fields, Ledger())
    assert made == 0
    assert plants.n_alive == 2


def test_bite_removes_at_zero():
    wc, fields = make_world()
    plants = make_plants(wc)
    plants.add(100.0, 100.0, 2.0)
    assert plants.bite(0, 1.5) == pytest.approx(1.5)
    assert plants.biomass[0] == pytest.approx(0.5)
    assert plants.alive[0]
    assert plants.bite(0, 1.0) == pytest.approx(0.5)
    assert plants.biomass[0] == 0.0
    assert not plants.alive[0]


def test_senescence_moves_mass_out():
    wc, fields = make_world()
    plants = make_plants(wc, senescence=0.002)
    plants.add(100.0, 100.0, 100.0)
    lost = plants.senesce()
    assert lost == pytest.approx(0.2)
    assert plants.biomass[0] == pytest.approx(99.8)


def test_biomass_field_sums():
    wc, fields = make_world()
    plants = make_plants(wc)
    plants.add(100.0, 100.0, 2.0)
    plants.add(110.0, 110.0, 3.0)
    field = plants.biomass_field(fields)
    assert field.shape == (10, 10)
    assert field.sum() == pytest.approx(5.0)


def test_remove_dead_compacts():
    wc, fields = make_world()
    plants = make_plants(wc)
    plants.add(100.0, 100.0, 2.0)
    plants.add(110.0, 110.0, 3.0)
    plants.bite(0, 5.0)
    plants.remove_dead()
    assert plants.n_alive == 1
    assert plants.biomass[0] == pytest.approx(3.0)
