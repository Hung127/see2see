"""Tests for gene layout, phenotype maps and the founder population."""
import numpy as np
import pytest

from evolve.genome import (
    GENE_SPECS, N_GENES, N_TRAITS, TRAIT_NAMES, founders, hash_genome, phenotype,
    validate_bounds,
)
from evolve.rng import Rng


def test_layout_and_names():
    assert N_TRAITS == 12 and N_GENES == 392
    assert TRAIT_NAMES == (
        "body_size", "speed_max", "vision_range", "vision_angle", "metabolic_rate",
        "attack", "defense", "repro_threshold", "repro_investment", "diet_preference",
        "lifespan", "mutation_rate",
    )
    assert len(GENE_SPECS) == N_TRAITS


def test_phenotype_ranges():
    rng = Rng(2)
    genes = founders(10000, rng)
    phen = phenotype(genes)
    for i, spec in enumerate(GENE_SPECS):
        assert phen[:, i].min() >= spec.lo - 1e-9, spec.name
        assert phen[:, i].max() <= spec.hi + 1e-9, spec.name


def test_phenotype_monotone():
    g = np.zeros((2, N_GENES))
    g[:, 0] = [0.0, 1.0]
    sizes = phenotype(g)[:, 0]
    assert sizes[0] == pytest.approx(0.3) and sizes[1] == pytest.approx(5.0)


def test_no_sterile_founders():
    """Every founder must be able to pay for a child's body at full reserves.

    Guards the regression where absolute thresholds drawn independently of body size
    made ~77% of founders mathematically sterile (DESIGN §4.1).
    """
    genes = founders(5000, Rng(3))
    phen = phenotype(genes)
    size = phen[:, 0]
    emax = 10.0 * size ** 0.9
    invest = phen[:, 8] * emax
    threshold = phen[:, 7] * emax
    assert np.all(invest > size)  # birth possible at full reserves
    assert np.all(threshold < emax)  # gate reachable: threshold below reserve capacity
    assert np.all(invest < emax)  # investment affordable at all
    assert np.mean(threshold > invest) > 0.95  # threshold gate is reachable


def test_founders_midbiased():
    genes = founders(5000, Rng(4))
    mean_body = genes[:, 0].mean()
    assert 0.35 <= mean_body <= 0.65


def test_validate_bounds():
    genes = founders(4, Rng(5))
    validate_bounds(genes)
    bad = genes.copy()
    bad[0, 0] = 1.5
    with pytest.raises(ValueError):
        validate_bounds(bad)
    bad2 = genes.copy()
    bad2[0, 20] = 9.0
    with pytest.raises(ValueError):
        validate_bounds(bad2)


def test_hash_genome_stable():
    g = founders(3, Rng(6))
    assert hash_genome(g[0]) == hash_genome(g[0].copy())
    assert hash_genome(g[0]) != hash_genome(g[1])
