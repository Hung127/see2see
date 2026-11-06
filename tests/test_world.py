"""Tests for world geometry (torus, reflection, bearings)."""
import numpy as np

from evolve.config import WorldConfig
from evolve.world import World


def test_wrap_distance_shortest():
    w = World(WorldConfig(width=100.0, height=100.0))
    assert np.isclose(abs(w.dx(np.array([2.0]), np.array([98.0]))[0]), 4.0)
    assert np.isclose(abs(w.dy(np.array([98.0]), np.array([2.0]))[0]), 4.0)


def test_reflect_keeps_inside():
    w = World(WorldConfig(width=100.0, height=100.0, boundaries="reflect"))
    x, y = w.apply(np.array([-3.0, 130.0]), np.array([250.0, -1.0]))
    assert np.all((x >= 0) & (x < 100.0))
    assert np.all((y >= 0) & (y < 100.0))
    assert np.isclose(x[0], 3.0)


def test_hard_clips():
    w = World(WorldConfig(width=100.0, height=100.0, boundaries="hard"))
    x, y = w.apply(np.array([-3.0, 130.0]), np.array([-1.0, 200.0]))
    assert np.all((x >= 0) & (x <= 100.0))
    assert np.all((y >= 0) & (y <= 100.0))


def test_bearing_is_relative_and_bounded():
    w = World(WorldConfig(width=100.0, height=100.0))
    # target directly ahead (east) of a heading-east organism -> 0
    b = w.bearing(np.array([10.0]), np.array([10.0]), np.array([20.0]), np.array([10.0]),
                  np.array([0.0]))
    assert np.isclose(b[0], 0.0)
    # target behind -> +/- pi
    b2 = w.bearing(np.array([10.0]), np.array([10.0]), np.array([0.0]), np.array([10.0]),
                   np.array([0.0]))
    assert abs(abs(b2[0]) - np.pi) < 1e-9
