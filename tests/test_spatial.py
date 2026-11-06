"""Tests for the uniform spatial grid (pair expansion, nearest queries)."""
import numpy as np

from evolve.config import WorldConfig
from evolve.spatial import SpatialGrid
from evolve.world import World


def test_pairs_match_bruteforce():
    rng = np.random.default_rng(0)
    n = 200
    x, y = rng.random(n) * 1000.0, rng.random(n) * 1000.0
    w = World(WorldConfig(width=1000.0, height=1000.0))
    g = SpatialGrid(w, cell=80.0)
    g.build(x, y)
    i, j = g.pairs(radius_cells=2)
    got = set(zip(i.tolist(), j.tolist()))
    want = set()
    for a in range(n):
        for b in range(a + 1, n):
            if w.distance(x[a:a+1], y[a:a+1], x[b:b+1], y[b:b+1])[0] <= 160.0:
                want.add((a, b))
    assert got == want


def test_nearest_tiebreak_lowest_index():
    w = World(WorldConfig(width=1000.0, height=1000.0))
    g = SpatialGrid(w, cell=80.0)
    tx = np.array([500.0, 500.0])
    ty = np.array([500.0, 500.0])
    g.build(tx, ty)
    idx, dist = g.nearest(np.array([490.0]), np.array([500.0]), tx, ty)
    assert idx[0] == 0  # equidistant -> lowest target index wins
    assert np.isclose(dist[0], 10.0)


def test_wrap_pairs():
    w = World(WorldConfig(width=1000.0, height=1000.0))
    g = SpatialGrid(w, cell=80.0)
    x = np.array([10.0, 990.0])
    y = np.array([500.0, 500.0])
    g.build(x, y)
    i, j = g.pairs(radius_cells=2)
    assert (0, 1) in set(zip(i.tolist(), j.tolist()))


def test_nearest_respects_mask():
    w = World(WorldConfig(width=1000.0, height=1000.0))
    g = SpatialGrid(w, cell=80.0)
    tx = np.array([500.0, 520.0])
    ty = np.array([500.0, 500.0])
    g.build(tx, ty)
    mask = np.array([False, True])
    idx, dist = g.nearest(np.array([490.0]), np.array([500.0]), tx, ty, mask=mask)
    assert idx[0] == 1
    assert np.isclose(dist[0], 30.0)
