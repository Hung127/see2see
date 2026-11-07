"""Additive-schema migration for ``tick_stats``.

A database written before the upgrade has no ``birth_rate`` column (and no
others from the appended block). Opening it must add the missing columns
without touching existing values or column order, so old runs stay readable and
new runs can be written into the same file.
"""
from __future__ import annotations

import sqlite3

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine
from evolve.persistence import ExperimentDB
from evolve.stats import COLUMNS, NEW_COLUMNS, StatsRecorder


def make_config(seed=1):
    return Config(
        seed=seed, ticks=10,
        world=WorldConfig(width=300.0, height=300.0, n_field_cells=10,
                          cell_size=30.0),
        plants=PlantConfig(initial=200, max=400),
        organisms=OrganismConfig(initial=20, max=60),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.0),
        stats=StatsConfig(),
    )


def _legacy_db(path):
    """Create a minimal pre-upgrade ``tick_stats`` schema."""
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE experiments (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
        seed INTEGER NOT NULL, config_yaml TEXT NOT NULL, spec_version TEXT NOT NULL,
        git_commit TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '');
    CREATE TABLE runs (
        id INTEGER PRIMARY KEY, experiment_id INTEGER NOT NULL,
        started_at TEXT NOT NULL, ended_at TEXT, start_tick INTEGER NOT NULL DEFAULT 0,
        end_tick INTEGER, status TEXT NOT NULL DEFAULT 'running', end_checksum TEXT);
    CREATE TABLE tick_stats (
        run_id INTEGER NOT NULL, tick INTEGER NOT NULL,
        population REAL NOT NULL DEFAULT 0, PRIMARY KEY (run_id, tick));
    INSERT INTO tick_stats (run_id, tick, population) VALUES (1, 5, 42);
    """)
    conn.commit()
    conn.close()


def test_migration_adds_only_missing_columns(tmp_path):
    path = str(tmp_path / "legacy.db")
    _legacy_db(path)
    db = ExperimentDB(path)
    try:
        cols = [row[1] for row in
                db.conn.execute("PRAGMA table_info(tick_stats)")]
        assert set(COLUMNS) <= set(cols)
        for col in NEW_COLUMNS:
            assert col in cols
        # Original columns keep their position; appended ones go after them.
        assert cols[0:2] == ["run_id", "tick"]
        assert cols.index("population") < cols.index("birth_rate")
        # Existing data still reads back with its original value.
        assert db.conn.execute(
            "SELECT population FROM tick_stats WHERE tick=5").fetchone()[0] == 42
    finally:
        db.close()


def test_migrated_database_accepts_new_rows(tmp_path):
    path = str(tmp_path / "legacy.db")
    _legacy_db(path)
    cfg = make_config()
    db = ExperimentDB(path)
    try:
        eng = Engine(cfg)
        eng.step()
        row = StatsRecorder().record(eng)
        exp = db.create_experiment("after-migration", cfg)
        run = db.start_run(exp)
        db.flush_stats(run, [row])
        read = db.read_stats(run)[0]
        assert read["tick"] == eng.tick
        assert read["birth_rate"] == row["birth_rate"]
        assert "survival_median" in read
        # The §5 flag is an INTEGER column, so it reads back as an int.
        assert isinstance(read["bottleneck"], int)
        assert read["population_drop"] == row["population_drop"]
    finally:
        db.close()


def test_migration_is_idempotent(tmp_path):
    path = str(tmp_path / "db.sqlite")
    ExperimentDB(path).close()
    db = ExperimentDB(path)
    try:
        cols = [row[1] for row in
                db.conn.execute("PRAGMA table_info(tick_stats)")]
    finally:
        db.close()
    # Opening twice must not duplicate columns or fail.
    assert len(cols) == len(set(cols))
    assert set(COLUMNS) <= set(cols)
