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
        # third additive block, appended after the second (IMPROVE §5)
        "population_drop", "diversity_drop", "bottleneck",
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



# -- IMPROVE §4.2: cumulative run totals for the evaluation endpoints --------
def test_run_totals_are_statistics_not_state():
    """Touching a run total must not change the checksum.

    These totals answer "what has this population done"; nothing in the tick
    reads them, so the contract is the same as ``death_counts``: snapshotted for
    reporting continuity, excluded from ``checksum()``.
    """
    from evolve.engine import RUN_TOTALS

    eng = Engine(make_config(seed=11, n=30))
    eng.run(40)
    before = eng.checksum()
    assert set(eng.run_totals) == set(RUN_TOTALS)
    for name in RUN_TOTALS:
        eng.run_totals[name] += 12345.0
    assert eng.checksum() == before


def test_run_totals_reconcile_with_the_tick():
    """The accumulated totals must be consistent with the engine's own events.

    Not exact bit-equality with per-organism counters (those only cover living
    animals, while the run totals also include the dead), but the relationships
    that must hold for any honest accumulation do hold.
    """
    from dataclasses import replace

    from evolve.config import EventConfig
    from evolve.engine import RUN_TOTALS

    cfg = make_config(seed=12, n=60, ticks=200)
    cfg = replace(cfg, environment=replace(
        cfg.environment,
        events=(EventConfig(kind="drought", start_tick=0, duration=200,
                            magnitude=0.3),)))
    eng = Engine(cfg)
    eng.run(150)
    totals = eng.run_totals
    assert set(totals) == set(RUN_TOTALS)
    assert all(v >= 0.0 for v in totals.values())
    # Assimilated energy can never exceed the food that was taken.
    assert (totals["energy_gain"]
            <= totals["plant_intake"] + totals["meat_intake"] + 1e-9)
    # Successes are subsets of attempts.
    assert totals["feed_successes"] <= totals["feed_attempts"]
    assert totals["strikes_hit"] <= totals["strikes"]
    assert totals["repro_successes"] <= totals["repro_attempts"]
    # Death ages are summed once per death and the mean must be an age.
    assert totals["death_age_sum"] >= float(eng.deaths_total)


def test_run_totals_keep_counting_after_death():
    """Unlike the per-organism counters, a run total survives its organism."""
    eng = Engine(make_config(seed=13, n=8, ticks=200))
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    # Drive one animal's activity, then kill it and continue.
    pop.bites[idx[0]] = 5.0
    pop.energy[idx[0]] = 0.0
    pop.body[idx[0]] = 0.0
    before_distance = eng.run_totals["distance"]
    eng.step()
    # The run total of the dead animal's earlier ticks is not rolled back.
    assert eng.run_totals["distance"] >= before_distance
    # Slots compact, so the population cannot be larger than it was.
    assert eng.population.n_alive <= idx.size


def test_run_totals_carry_a_mean_lifespan_denominator():
    """``death_age_sum`` is the numerator of an exact mean lifespan."""
    eng = Engine(make_config(seed=14, n=20, ticks=120))
    eng.run(100)
    if eng.deaths_total:
        mean_age = eng.run_totals["death_age_sum"] / eng.deaths_total
        assert 0.0 <= mean_age <= float(eng.tick) + 1.0


# -- IMPROVE §5: contraction / bottleneck indicator ---------------------------
def test_bottleneck_needs_an_earlier_sample_before_it_says_anything():
    """No history means "not measurable", which must not read as "no bottleneck"."""
    from evolve.stats import bottleneck

    out = bottleneck([], 100, 1.5)
    assert out == {"population_drop": 0.0, "diversity_drop": 0.0,
                   "bottleneck": 0.0}


def test_bottleneck_flags_when_both_population_and_diversity_collapse():
    from evolve.stats import BOTTLENECK_DROP, bottleneck

    rows = [
        {"population": 100, "mean_pairwise": 2.0},
        {"population": 90, "mean_pairwise": 1.8},
    ]
    out = bottleneck(rows, 20, 0.2)
    assert out["population_drop"] == pytest.approx(0.80)
    assert out["diversity_drop"] == pytest.approx(0.90)
    assert out["bottleneck"] == 1.0
    assert BOTTLENECK_DROP == 0.5


def test_bottleneck_does_not_flag_a_mild_decline():
    from evolve.stats import bottleneck

    rows = [{"population": 100, "mean_pairwise": 2.0}]
    out = bottleneck(rows, 70, 1.4)          # 30% down on both: not a squeeze
    assert out["bottleneck"] == 0.0
    assert out["population_drop"] == pytest.approx(0.30)


def test_bottleneck_requires_both_series_to_fall():
    """A population crash with stable diversity is not (yet) a genetic squeeze."""
    from evolve.stats import bottleneck

    rows = [{"population": 100, "mean_pairwise": 2.0}]
    out = bottleneck(rows, 10, 2.0)
    assert out["population_drop"] == pytest.approx(0.90)
    assert out["diversity_drop"] == pytest.approx(0.0)
    assert out["bottleneck"] == 0.0


def test_bottleneck_handles_a_zero_peak_diversity():
    """A window that never had diversity cannot report a fractional drop."""
    from evolve.stats import bottleneck

    rows = [{"population": 10, "mean_pairwise": 0.0}]
    out = bottleneck(rows, 2, 0.0)
    assert out["diversity_drop"] == 0.0
    assert out["bottleneck"] == 0.0


def test_recorder_reports_bottleneck_columns_without_touching_the_checksum():
    from evolve.stats import BOTTLENECK_DROP, bottleneck  # noqa: F401

    eng = Engine(make_config(seed=15, n=30, ticks=120))
    recorder = StatsRecorder()
    before = eng.checksum()
    row = recorder.record(eng)
    assert {"population_drop", "diversity_drop", "bottleneck"} <= set(row)
    assert row["population_drop"] == 0.0      # first sample: no history yet
    assert row["bottleneck"] in (0.0, 1.0)
    assert eng.checksum() == before


def test_bottleneck_is_sampled_not_state():
    """The indicator is an observation; recording it must not alter the run."""
    eng = Engine(make_config(seed=16, n=30, ticks=120))
    a = StatsRecorder()
    b = StatsRecorder()
    while eng.tick < 60:
        eng.step()
        if eng.tick % 10 == 0:
            a.record(eng)
    mid = eng.checksum()
    # Re-record the same tick many times: the checksum cannot move.
    for _ in range(5):
        b.record(eng)
    assert eng.checksum() == mid

