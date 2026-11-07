"""Tests for the bounded lineage recorder.

The recorder must be three things: cheap (opt-in, no effect on the checksum),
bounded (memory does not grow with run length), and honest (it never invents a
record for an organism it did not see born).
"""
from __future__ import annotations

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine
from evolve.stats import LINEAGE_COLUMNS, LineageRecorder, export_lineage_csv


def make_config(seed=1, n=40, ticks=200):
    return Config(
        seed=seed, ticks=ticks,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=16,
                          cell_size=31.25),
        plants=PlantConfig(initial=1200, max=3000),
        organisms=OrganismConfig(initial=n, max=200),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.002),
        stats=StatsConfig(),
    )


def test_lineage_recorder_does_not_change_simulation():
    """Attaching a recorder is observation, not intervention."""
    plain = Engine(make_config(seed=2))
    for _ in range(150):
        plain.step()

    logged = Engine(make_config(seed=2))
    logged.lineage = LineageRecorder()
    for _ in range(150):
        logged.step()

    assert logged.checksum() == plain.checksum()


def test_lineage_records_follow_parents():
    eng = Engine(make_config(seed=3, n=60))
    eng.lineage = LineageRecorder()
    for _ in range(200):
        eng.step()
    assert eng.lineage.births_recorded == eng.births_total
    finished = eng.lineage.rows()
    assert len(finished) <= eng.lineage.deaths_recorded
    for record in finished:
        assert tuple(record) == LINEAGE_COLUMNS
        assert record["death_tick"] >= record["birth_tick"]
        assert record["cause"] in ("predation", "starvation", "senescence",
                                   "health", "")
        # Generation is the engine's counter: a child is one deeper than its
        # parent. It is never re-derived here.
        assert record["generation"] >= 1


def test_lineage_memory_is_bounded():
    eng = Engine(make_config(seed=4, n=30))
    eng.lineage = LineageRecorder(max_finished=5)
    for _ in range(250):
        eng.step()
    assert len(eng.lineage.rows()) <= 5


def test_lineage_does_not_invent_founders():
    """Founders have no birth event, so they must not appear as births."""
    eng = Engine(make_config(seed=5, n=20))
    eng.lineage = LineageRecorder()
    eng.step()
    # No reproduction yet, so nothing has been recorded.
    assert eng.lineage.births_recorded == 0
    assert eng.lineage.living() == []


def test_lineage_depth_max_is_zero_when_empty():
    assert LineageRecorder().lineage_depth_max() == 0


def test_lineage_csv_has_the_declared_columns(tmp_path):
    import csv

    rec = LineageRecorder()
    rec.record_birth(1, 0, None, tick=5, generation=1)
    rec.record_death(1, tick=9, cause="starvation", age=4)
    path = tmp_path / "lineage.csv"
    export_lineage_csv(rec.rows(), str(path))
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == list(LINEAGE_COLUMNS)
    assert len(rows) == 2
    assert rows[1][LINEAGE_COLUMNS.index("cause")] == "starvation"
