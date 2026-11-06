"""Tests for the seeded RNG wrapper."""
import numpy as np
import pytest

from evolve.rng import Rng


def test_rng_reproducible():
    a, b = Rng(7), Rng(7)
    assert np.array_equal(a.random(10), b.random(10))
    assert np.array_equal(a.normal(0, 1, 5), b.normal(0, 1, 5))


def test_rng_state_roundtrip():
    r = Rng(3)
    r.random(100)
    state = r.state()
    first = r.random(100)
    r2 = Rng(999)
    r2.set_state(state)
    assert np.array_equal(r2.random(100), first)


def test_spawn_independent_streams():
    r = Rng(11)
    kids = r.spawn(3)
    seqs = [k.random(4) for k in kids]
    assert len(set(map(tuple, seqs))) == 3  # all different
    again = [Rng(11).spawn(3)[i].random(4) for i in range(3)]
    assert all(np.array_equal(s, a) for s, a in zip(seqs, again))


def test_choice_and_integers():
    r = Rng(5)
    idx = r.integers(0, 10, size=100)
    assert idx.min() >= 0 and idx.max() < 10
    pick = r.choice(np.arange(10), size=5, replace=False)
    assert len(set(pick.tolist())) == 5
