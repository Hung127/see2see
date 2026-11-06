"""Tests for mutation and crossover operators (DESIGN §4)."""
import numpy as np

from evolve.config import GeneticsConfig
from evolve.genome import N_GENES, N_TRAITS, NN_HI, NN_LO, founders, phenotype
from evolve.genome import crossover, mutate
from evolve.rng import Rng

CFG = GeneticsConfig()


def test_zero_rate_is_copy():
    genes = founders(5, Rng(1))
    child, n_mut = mutate(genes, np.zeros(5), Rng(2), CFG)
    assert np.array_equal(child, genes)
    assert np.array_equal(n_mut, np.zeros(5, dtype=n_mut.dtype))


def test_full_rate_modifies_almost_every_locus():
    genes = founders(4, Rng(3))
    child, n_mut = mutate(genes, np.ones(4), Rng(4), CFG)
    changed = (child != genes)
    assert changed.mean() > 0.9
    assert np.all(n_mut == N_GENES)


def test_bounds_hold_after_mutation():
    genes = founders(200, Rng(5))
    for seed in range(10):
        genes, _ = mutate(genes, np.ones(200), Rng(seed), CFG)
    assert np.all(genes[:, :N_TRAITS] >= 0.0)
    assert np.all(genes[:, :N_TRAITS] <= 1.0)
    assert np.all(genes[:, N_TRAITS:] >= NN_LO)
    assert np.all(genes[:, N_TRAITS:] <= NN_HI)


def test_mutation_rate_locus_respects_floor_ceiling():
    genes = founders(500, Rng(6))
    for seed in range(5):
        genes, _ = mutate(genes, np.ones(500), Rng(100 + seed), CFG)
    rates = phenotype(genes)[:, 11]
    assert np.all(rates >= CFG.mutation_floor - 1e-9)
    assert np.all(rates <= CFG.mutation_ceil + 1e-9)


def test_large_effects_are_rare():
    """At equal per-locus rate, the large-effect multiplier must be a tail, not the norm."""
    genes = np.full((1, N_GENES), 0.5)
    genes[:, N_TRAITS:] = 0.0
    child, _ = mutate(genes, np.ones(1), Rng(7), CFG)
    delta = np.abs(child[0, N_TRAITS:] - genes[0, N_TRAITS:])
    # sigma_w = 0.05; a 5x large effect is > 0.15 for >3 sigma of the base distribution
    assert np.mean(delta > 0.15) < 0.01


def test_crossover_uses_both_parents():
    a = founders(1, Rng(8))[0]
    b = founders(1, Rng(9))[0]
    child = crossover(a, b, Rng(10))
    assert child.shape == (N_GENES,)
    assert np.all((child == a) | (child == b))
    from_a = np.mean(child == a)
    assert 0.2 < from_a < 0.8  # per-locus uniform, not one-parent


def test_crossover_batch():
    a = founders(3, Rng(11))
    b = founders(3, Rng(12))
    child = crossover(a, b, Rng(13))
    assert child.shape == (3, N_GENES)
    for i in range(3):
        assert np.all((child[i] == a[i]) | (child[i] == b[i]))


def test_mutation_is_deterministic():
    genes = founders(2, Rng(14))
    c1, _ = mutate(genes, np.full(2, 0.1), Rng(15), CFG)
    c2, _ = mutate(genes, np.full(2, 0.1), Rng(15), CFG)
    assert np.array_equal(c1, c2)
