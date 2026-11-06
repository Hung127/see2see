"""Tests for the statistics recorder (DESIGN §8)."""
import numpy as np

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine
from evolve.genome import founders
from evolve.rng import Rng
from evolve.stats import COLUMNS, StatsRecorder, diversity


def make_config(seed=1, n=30):
    return Config(
        seed=seed, ticks=100,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=10, cell_size=50.0),
        plants=PlantConfig(initial=100, max=300),
        organisms=OrganismConfig(initial=n, max=100),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.0),
        stats=StatsConfig(),
    )


def test_stats_schema():
    assert isinstance(COLUMNS, tuple)
    a = Engine(make_config(seed=1))
    b = Engine(make_config(seed=1))
    rec_a = StatsRecorder()
    rec_b = StatsRecorder()
    row_a = rec_a.record(a)
    row_b = rec_b.record(b)
    assert tuple(row_a) == COLUMNS
    assert tuple(row_b) == COLUMNS
    assert row_a == row_b
    assert len(set(COLUMNS)) == len(COLUMNS)


def test_diversity_senses_bottleneck():
    rng = Rng(3)
    base = founders(1, rng)[0]
    clones = np.repeat(base[None, :], 40, axis=0)
    mixed = founders(40, rng)
    low = diversity(clones)
    high = diversity(mixed)
    assert low["mean_pairwise"] < 1e-12
    assert low["unique_genomes"] == 1
    assert high["mean_pairwise"] > low["mean_pairwise"]
    assert high["unique_genomes"] > 1


def test_diversity_is_bounded_and_deterministic():
    rng = Rng(4)
    genes = founders(400, rng)
    d1 = diversity(genes)
    d2 = diversity(genes)
    assert d1 == d2  # deterministic striding
    assert 0.0 <= d1["diet_entropy"] <= 1.0
    assert d1["mean_variance"] >= 0.0
    assert 0 < d1["unique_genomes"] <= 400


def test_role_counts_sum_to_population():
    eng = Engine(make_config(seed=2))
    rec = StatsRecorder()
    for _ in range(20):
        eng.step()
        row = rec.record(eng)
        roles = row["herbivores"] + row["carnivores"] + row["omnivores"] \
            + row["unclassified"]
        assert roles == row["population"]
        assert row["population"] == eng.population.n_alive


def test_recorder_tracks_births_deaths_and_ring_buffer():
    eng = Engine(make_config(seed=5))
    rec = StatsRecorder(window=5)
    for _ in range(12):
        eng.step()
        rec.record(eng)
    assert len(rec.rows) == 5  # ring buffer bound
    assert rec.rows[-1]["tick"] == eng.tick


def test_mutation_stats_present_after_births():
    eng = Engine(make_config(seed=6, n=10))
    rec = StatsRecorder()
    for _ in range(60):
        eng.step()
    row = rec.record(eng)
    assert row["mean_mutation_rate"] >= 5e-4
    assert row["mutations_per_birth"] >= 0.0


def test_extinction_events_logged_when_hash_disappears():
    from evolve.stats import extinction_events

    rec = StatsRecorder()
    hashes_a = [b"aaa", b"bbb"]
    hashes_b = [b"aaa"]
    assert rec.track_extinctions(hashes_a, tick=1) == []
    gone = rec.track_extinctions(hashes_b, tick=2)
    assert gone == [b"bbb"]
    assert rec.extinctions[-1]["tick"] == 2
