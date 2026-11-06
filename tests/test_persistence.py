"""Tests for SQLite experiment storage (DESIGN §9)."""
import sqlite3

import pytest

from evolve.config import Config, OrganismConfig, StatsConfig
from evolve.engine import Engine
from evolve.persistence import ExperimentDB, full_logging_allowed
from evolve.stats import COLUMNS, INTERACTION_COLUMNS, LINEAGE_COLUMNS, StatsRecorder


def make_row(tick, population=10):
    """A plausible stats row: every COLUMNS key present, a few non-zero."""
    row = {c: 0.0 for c in COLUMNS}
    row.update(tick=tick, population=population, births=1, deaths=0,
               herbivores=population, unique_genomes=population,
               plant_biomass=100.0 + tick, mean_energy=5.0)
    return row


def test_schema_and_roundtrip(tmp_path):
    path = str(tmp_path / "exp.sqlite")
    db = ExperimentDB(path)
    try:
        cfg = Config()
        exp_id = db.create_experiment("smoke", cfg, notes="unit test")
        run_id = db.start_run(exp_id)

        rows = [make_row(t) for t in (50, 100, 150)]
        assert db.flush_stats(run_id, rows) == 3
        back = db.read_stats(run_id)
        assert len(back) == 3
        for got, want in zip(back, rows):
            for col in COLUMNS:
                assert got[col] == want[col], (col, got[col], want[col])

        # lineage / interactions / events round-trip through the same schema
        lineage = [{"org_id": 1, "parent_a": None, "parent_b": None,
                    "birth_tick": 0, "death_tick": 42, "generation": 0,
                    "cause": "starvation"}]
        assert db.append_lineage(run_id, lineage) == 1
        inter = [{"tick": 3, "kind": "kill", "actor_id": 1, "target_id": 2,
                  "amount": 0.5}]
        assert db.append_interaction(run_id, inter) == 1
        assert db.record_event(run_id, 7, "drought", {"magnitude": 1.0}) == 1
        db.finish_run(run_id, end_tick=150, status="completed")

        con = sqlite3.connect(path)
        try:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            assert {"experiments", "runs", "tick_stats", "snapshots", "events",
                    "lineage", "interactions"} <= tables
            assert con.execute(
                "SELECT status, end_tick FROM runs WHERE id=?",
                (run_id,)).fetchone() == ("completed", 150)
            assert con.execute("SELECT COUNT(*) FROM lineage").fetchone()[0] == 1
            assert con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0] == 1
            assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        finally:
            con.close()

        # the stored config round-trips exactly
        assert db.experiment_config(exp_id) == cfg
    finally:
        db.close()


def test_sampled_logging_budget(tmp_path):
    """sampled keeps kills/matings and a deterministic 1-in-20 of feeds;
    full is refused for a run too large to afford it (DESIGN §9)."""
    db = ExperimentDB(str(tmp_path / "exp.sqlite"))
    try:
        big = Config(ticks=30000)  # default 300 founders: ticks alone exceed the guard
        assert not full_logging_allowed(big)
        exp = db.create_experiment("big", big)
        run = db.start_run(exp)

        # 20 feed rows (ticks 1..20, actor 0) + 2 kills + 1 mating
        rows = [{"tick": t, "kind": "feed", "actor_id": 0, "target_id": 100 + t,
                 "amount": 0.1} for t in range(1, 21)]
        rows += [{"tick": 5, "kind": "kill", "actor_id": 1, "target_id": 2,
                  "amount": 1.0},
                 {"tick": 6, "kind": "mate", "actor_id": 3, "target_id": 4,
                  "amount": 0.0}]
        kept = db.append_interaction(run, rows)  # mode from config: sampled
        assert kept == 3  # 1 of the 20 feeds + both non-feed rows
        con = sqlite3.connect(db.path)
        try:
            ticks = [r[0] for r in con.execute(
                "SELECT tick FROM interactions WHERE run_id=? ORDER BY tick",
                (run,))]
        finally:
            con.close()
        assert ticks == [5, 6, 20]

        with pytest.raises(ValueError):
            db.append_interaction(run, rows, mode="full")

        small = Config(ticks=100, organisms=OrganismConfig(initial=10, max=50))
        assert full_logging_allowed(small)
        exp2 = db.create_experiment("small", small)
        run2 = db.start_run(exp2)
        assert db.append_interaction(run2, rows, mode="full") == len(rows)
    finally:
        db.close()


def test_no_per_tick_writes(tmp_path):
    """Statistics land once per stats.interval, never per tick."""
    db = ExperimentDB(str(tmp_path / "exp.sqlite"))
    try:
        cfg = Config(ticks=200, stats=StatsConfig(interval=50))
        exp = db.create_experiment("run", cfg)
        run = db.start_run(exp)

        eng = Engine(cfg)
        recorder = StatsRecorder()
        rows = []
        for _ in range(200):
            eng.step()
            if eng.tick % cfg.stats.interval == 0:
                rows.append(recorder.record(eng))
        db.flush_stats(run, rows)

        back = db.read_stats(run)
        assert len(back) == 200 // cfg.stats.interval == 4
        assert [r["tick"] for r in back] == [50, 100, 150, 200]
    finally:
        db.close()
