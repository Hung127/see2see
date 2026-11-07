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


def test_record_looks_up_living_and_finished_only():
    """A lookup returns a real record or ``None``; nothing is synthesised."""
    rec = LineageRecorder()
    rec.record_birth(10, 3, None, tick=2, generation=1)
    assert rec.record(10)["org_id"] == 10
    assert rec.record(99) is None
    rec.record_death(10, tick=8, cause="starvation", age=6)
    dead = rec.record(10)
    assert dead is not None
    assert dead["death_tick"] == 8
    assert dead["cause"] == "starvation"


def test_ancestry_follows_recorded_parents_to_a_founder():
    """The chain is parent -> child, oldest first, and stops at a founder."""
    rec = LineageRecorder()
    rec.record_birth(1, None, None, tick=0, generation=0)   # founder
    rec.record_birth(2, 1, None, tick=5, generation=1)
    rec.record_birth(3, 2, None, tick=9, generation=2)
    chain = rec.ancestry(3, depth=4)
    assert [r["org_id"] for r in chain] == [1, 2, 3]
    assert chain[-1]["org_id"] == 3


def test_ancestry_reports_an_evicted_link_as_unknown():
    """No ancestor is ever inferred when its record has been dropped."""
    rec = LineageRecorder(max_finished=1)
    rec.record_birth(1, None, None, tick=0, generation=0)
    rec.record_birth(2, 1, None, tick=5, generation=1)
    rec.record_death(1, tick=6, cause="senescence", age=6)
    rec.record_death(2, tick=7, cause="senescence", age=2)   # evicts record 1
    # 2 is still known; its parent 1 has been evicted, so the chain starts with
    # an explicit ``None`` rather than a guessed ancestor.
    assert rec.record(1) is None
    assert rec.record(2)["org_id"] == 2
    chain = rec.ancestry(2)
    assert chain[0] is None
    assert chain[-1]["org_id"] == 2


def test_ancestry_respects_the_requested_depth():
    rec = LineageRecorder()
    for i in range(1, 8):
        rec.record_birth(i, i - 1 if i > 1 else None, None, tick=i,
                         generation=i - 1)
    assert len(rec.ancestry(7, depth=3)) == 3


def test_ancestry_of_an_unknown_organism_is_a_single_none():
    rec = LineageRecorder()
    assert rec.ancestry(42) == [None]
