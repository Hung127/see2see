"""Tests for the scientific metrics added in the upgrade.

The point of these tests is not that the numbers are "good" — they are whatever
the simulation produces — but that they are *correct*: that cumulative counters
reconcile with the engine, that per-capita rates use the elapsed interval, and
that the survival curve is built from real deaths rather than invented.
"""
from __future__ import annotations

import numpy as np
import pytest

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import DEATH_AGE_BIN, DEATH_AGE_BINS, DEATH_CAUSES, Engine
from evolve.stats import COLUMNS, NEW_COLUMNS, StatsRecorder, diversity


def make_config(seed=1, n=40, ticks=150, interval=10):
    return Config(
        seed=seed, ticks=ticks,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=16,
                          cell_size=31.25),
        plants=PlantConfig(initial=1200, max=3000),
        organisms=OrganismConfig(initial=n, max=200),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.002),
        stats=StatsConfig(interval=interval),
    )


def test_new_columns_are_appended_and_unique():
    """The upgrade may only append; persistence and old CSVs depend on order."""
    original = COLUMNS[:COLUMNS.index("birth_rate")]
    assert original[-1] == "mean_moisture"
    assert set(NEW_COLUMNS) == {
        "birth_rate", "death_rate", "mean_generation", "max_generation",
        "trait_variance", "nn_variance", "mean_organism_biomass",
        "deaths_starvation", "deaths_senescence", "deaths_health",
        "deaths_predation", "survival_median", "population_cv",
        # second additive block, appended after the first (IMPROVE §3.3)
        "environmental_stress", "behavioural_diversity",
    }
    assert len(set(COLUMNS)) == len(COLUMNS)
    # The pre-upgrade prefix is untouched and still a contiguous block.
    assert COLUMNS[:len(original)] == original


def test_death_causes_reconcile_with_deaths_total():
    eng = Engine(make_config(seed=2, n=60))
    for _ in range(200):
        eng.step()
    assert sum(eng.death_counts.values()) == eng.deaths_total
    assert set(eng.death_counts) == set(DEATH_CAUSES)
    assert int(eng.death_age_hist.sum()) == eng.deaths_total
    assert eng.death_age_hist.size == DEATH_AGE_BINS


def test_death_histogram_is_bounded_regardless_of_horizon():
    eng = Engine(make_config(seed=4, n=40))
    for _ in range(300):
        eng.step()
    # Fixed-size array: this is the memory guarantee, not an accident.
    assert eng.death_age_hist.shape == (DEATH_AGE_BINS,)
    assert eng.death_age_hist.dtype == np.int64


def test_per_capita_rates_use_elapsed_interval():
    eng = Engine(make_config(seed=5, n=50, interval=10))
    rec = StatsRecorder()
    for _ in range(60):
        eng.step()
        if eng.tick % 10 == 0:
            rec.record(eng)
    row = rec.rows[-1]
    births = row["births"]
    deaths = row["deaths"]
    # interval = 10, population = the previous sample (10 ticks earlier).
    assert row["birth_rate"] == pytest.approx(births / (10 * rec.rows[-2]["population"]))
    assert row["death_rate"] == pytest.approx(deaths / (10 * rec.rows[-2]["population"]))
    assert row["birth_rate"] >= 0.0 and row["death_rate"] >= 0.0


def test_trait_and_nn_variance_are_separate_scales():
    rng = np.random.default_rng(0)
    genes = np.concatenate([
        rng.random((50, 12)),                # trait loci in [0, 1]
        rng.normal(0, 0.3, size=(50, 380)),  # network weights
    ], axis=1)
    d = diversity(genes)
    assert d["trait_variance"] > 0.0
    assert d["nn_variance"] > 0.0
    # Different blocks, different scales: they must not be the same number.
    assert d["trait_variance"] != d["nn_variance"]


def test_survival_median_is_zero_before_any_death():
    eng = Engine(make_config(seed=6, n=5))
    rec = StatsRecorder()
    row = rec.record(eng)
    assert eng.deaths_total == 0
    assert row["survival_median"] == 0.0


def test_survival_median_is_a_bin_midpoint():
    from evolve.stats import _survival_median

    class Fake:
        death_age_hist = np.zeros(DEATH_AGE_BINS, dtype=np.int64)

    fake = Fake()
    fake.death_age_hist[4] = 10          # all deaths in bin 4
    assert _survival_median(fake) == (4 + 0.5) * DEATH_AGE_BIN
    fake.death_age_hist[:] = 0
    fake.death_age_hist[0] = 1
    fake.death_age_hist[10] = 1          # half-way point lands in bin 0
    assert _survival_median(fake) == 0.5 * DEATH_AGE_BIN


def test_rest_fraction_is_a_lifetime_mean_not_a_tick_average():
    """Σ rest_ticks / Σ age over living animals; not Σ rest_ticks / (n · tick).

    Founder ages equal the current tick, so the two formulas agree unless the
    ages are set explicitly — which is what this test does. Anchoring on ages
    keeps the metric honest for a mixed-age population after births.
    """
    eng = Engine(make_config(seed=9, n=4))
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    assert idx.size == 4
    pop.rest_ticks[idx] = [5.0, 0.0, 0.0, 0.0]
    pop.age[idx] = [20, 100, 100, 100]

    row = StatsRecorder().record(eng)

    assert row["rest_fraction"] == pytest.approx(5.0 / 320.0)
    assert row["rest_fraction"] != pytest.approx(5.0 / 4 / max(1, eng.tick))


def test_population_cv_zero_for_constant_population():
    rec = StatsRecorder()
    for pop in (10, 10, 10, 10):
        rec.rows.append({"tick": len(rec.rows), "population": pop})
    assert rec._population_cv(10) == pytest.approx(0.0)


def test_population_cv_positive_when_population_moves():
    rec = StatsRecorder()
    for pop in (10, 20, 10, 20):
        rec.rows.append({"tick": len(rec.rows), "population": pop})
    assert rec._population_cv(20) > 0.0


def test_generation_columns_track_turnover():
    eng = Engine(make_config(seed=7, n=30))
    rec = StatsRecorder()
    for _ in range(120):
        eng.step()
    row = rec.record(eng)
    if eng.population.n_alive:
        living = eng.population.generation[np.flatnonzero(eng.population.alive)]
        assert row["max_generation"] == int(living.max())
        assert row["mean_generation"] == pytest.approx(float(living.mean()))


# -- IMPROVE §3.3: environmental stress and behavioural diversity ------------
def test_behavioural_diversity_is_zero_for_identical_profiles():
    """The metric must measure behaviour, not the presence of a genome.

    Two animals with the same lifetime activity rates behave identically, so
    their profile distance is zero even though this function never looks at a
    genome at all.
    """
    from evolve.stats import behavioural_diversity

    eng = Engine(make_config(seed=3, n=6))
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    pop.age[idx] = 50
    for name in ("dist_moved", "bites", "hunts", "repro_attempts", "rest_ticks"):
        getattr(pop, name)[idx] = 7.0
    assert behavioural_diversity(pop, idx) == pytest.approx(0.0)


def test_behavioural_diversity_is_positive_when_activity_differs():
    from evolve.stats import behavioural_diversity

    eng = Engine(make_config(seed=3, n=6))
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    pop.age[idx] = 50
    pop.bites[idx] = np.arange(idx.size, dtype=np.float64) * 10.0
    assert behavioural_diversity(pop, idx) > 0.0
    # A single animal has no pairwise distance to average.
    assert behavioural_diversity(pop, idx[:1]) == pytest.approx(0.0)
    assert behavioural_diversity(pop, np.zeros(0, dtype=np.int64)) == 0.0


def test_behavioural_diversity_ignores_genes_but_responds_to_counters():
    """Guard against the forbidden conflation of genetic and behaviour spread."""
    from evolve.stats import behavioural_diversity

    eng = Engine(make_config(seed=5, n=8))
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    pop.age[idx] = 40
    same = np.repeat(pop.genome[idx][:1], idx.size, axis=0)
    pop.genome[idx] = same
    pop.hunts[idx] = 0.0
    flat = behavioural_diversity(pop, idx)
    pop.hunts[idx] = np.arange(idx.size, dtype=np.float64)
    changed = behavioural_diversity(pop, idx)
    assert flat == pytest.approx(0.0)
    assert changed > flat


def test_behaviour_profiles_are_per_tick_rates_not_lifetime_totals():
    """A 10-tick-old and a 100-tick-old animal with the same rate match."""
    from evolve.stats import behaviour_profiles

    eng = Engine(make_config(seed=3, n=2))
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    pop.age[idx] = [10, 100]
    pop.bites[idx] = [2.0, 20.0]       # 0.2 bites/tick in both cases
    pop.hunts[idx] = [1.0, 10.0]
    pop.dist_moved[idx] = [30.0, 300.0]
    prof = behaviour_profiles(pop, idx)
    assert prof.shape == (2, 5)
    assert np.allclose(prof[0], prof[1])
    assert prof[0, 1] == pytest.approx(0.2)   # bites per tick, unscaled


def test_environmental_stress_samples_where_the_animals_are():
    """The field mean is wrong if the population is not spread over the world."""
    eng = Engine(make_config(seed=4, n=20))
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    # Put every animal in one cell and stress only that cell.
    pop.x[idx] = 5.0
    pop.y[idx] = 5.0
    eng.fields.water_stress[:] = 0.0
    ix, iy = eng.fields.cell_index(pop.x[idx], pop.y[idx])
    eng.fields.water_stress[ix[0], iy[0]] = 0.5

    row = StatsRecorder().record(eng)

    assert row["environmental_stress"] == pytest.approx(0.5)
    assert eng.fields.water_stress.mean() < 0.5


def test_environmental_stress_is_zero_outside_a_drought():
    eng = Engine(make_config(seed=4, n=20))
    assert not eng.fields.water_stress.any()
    assert StatsRecorder().record(eng)["environmental_stress"] == 0.0

