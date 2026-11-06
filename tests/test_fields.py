"""Tests for the coarse environmental field grid."""
import numpy as np

from evolve.config import EnvironmentConfig, WorldConfig
from evolve.fields import Fields
from evolve.rng import Rng


def make_fields(width=500.0, n=10, **env):
    wc = WorldConfig(width=width, height=width, n_field_cells=n, cell_size=width / n)
    return Fields(wc, EnvironmentConfig(**env), Rng(1))


def test_terrain_smooth_and_bounded():
    f = make_fields()
    assert f.terrain.shape == (10, 10)
    assert f.terrain.min() >= 0.0 and f.terrain.max() <= 1.0
    diffs = np.abs(np.diff(f.terrain, axis=0)).max()
    assert diffs < 0.3


def test_season_period():
    f = make_fields(season_period=200, rain_rate=0.0)
    f.step(0)
    t0 = f.temperature.copy()
    f.step(200)
    assert np.allclose(f.temperature, t0)


def test_rain_raises_moisture():
    f = make_fields(rain_rate=0.05, m_mean=0.5, m_amp=0.0)
    f.moisture[:] = 0.5
    f.step(1)
    assert f.moisture.mean() > 0.5


def test_pheromone_decays():
    f = make_fields()
    f.add_pheromone(np.array([10.0]), np.array([10.0]), 1.0)
    before = f.pheromone.sum()
    assert before > 0
    f.step(1)
    assert f.pheromone.sum() < before


def test_sample_returns_cell_values():
    f = make_fields()
    vals = f.sample(np.array([5.0]), np.array([5.0]))
    assert set(vals) == {"temperature", "moisture", "soil", "terrain", "plant_density",
                         "organism_density", "pheromone", "water_stress"}
    assert vals["temperature"].shape == (1,)
