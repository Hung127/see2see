"""SQLite experiment storage (DESIGN §9).

One database holds many experiments; one experiment is a ``(name, config, seed,
spec_version, git_commit)`` tuple, and one *run* is one execution of it. Rows are
batched: statistics are written once per ``stats.interval`` ticks, never per
tick, because the design budget (5,000 organisms x 30,000 ticks) would otherwise
produce tens of millions of rows.

Interaction edges follow ``config.stats.interactions``:

* ``off``     — nothing is written;
* ``sampled`` — every kill and every mating, plus a deterministic 1-in-20 sample
  of feeding events. The sample rule is ``(tick + actor_id) % 20 == 0``: it
  depends only on the event, not on arrival order, so it is reproducible even if
  the caller groups rows differently;
* ``full``    — every event, but refused for runs larger than 500 founders or
  2,000 ticks (the small-run guard from DESIGN §9), because the row count would
  exceed the logging budget.

Per-tick *aggregates* (diet totals, role counts) live in ``tick_stats`` and are
always exact, independent of the interaction mode.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from typing import Any, Iterable, Mapping

import yaml

from .config import SPEC_VERSION, Config, config_from_dict
from .stats import COLUMNS

#: ``tick_stats`` columns stored as SQLite INTEGER; everything else is REAL.
_INT_COLUMNS = frozenset({
    "tick", "population", "births", "deaths", "herbivores", "carnivores",
    "omnivores", "unclassified", "unique_genomes",
})

#: The 1-in-20 sampling rule for feeding edges (deterministic, order-free).
_FEED_SAMPLE_MODULUS = 20

_FULL_LOGGING_MAX_ORGANISMS = 500
_FULL_LOGGING_MAX_TICKS = 2000


def full_logging_allowed(config: Config) -> bool:
    """Whether ``verbose`` interaction logging fits the DESIGN §9 budget."""
    return (config.organisms.initial <= _FULL_LOGGING_MAX_ORGANISMS
            and config.ticks <= _FULL_LOGGING_MAX_TICKS)


def _tick_stats_ddl() -> str:
    """Build the tick_stats DDL from ``stats.COLUMNS`` so it cannot drift."""
    lines = ["    run_id INTEGER NOT NULL REFERENCES runs(id)",
             "    tick INTEGER NOT NULL"]
    for col in COLUMNS:
        if col == "tick":
            continue
        kind = "INTEGER" if col in _INT_COLUMNS else "REAL"
        lines.append(f"    {col} {kind} NOT NULL DEFAULT 0")
    lines.append("    PRIMARY KEY (run_id, tick)")
    return "CREATE TABLE IF NOT EXISTS tick_stats (\n" + ",\n".join(lines) + "\n);"


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    seed INTEGER NOT NULL,
    config_yaml TEXT NOT NULL,
    spec_version TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    start_tick INTEGER NOT NULL DEFAULT 0,
    end_tick INTEGER,
    status TEXT NOT NULL DEFAULT 'running'
);
{_tick_stats_ddl()}
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    tick INTEGER NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    rng_state BLOB,
    manifest_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    tick INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{{}}'
);
CREATE TABLE IF NOT EXISTS lineage (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    org_id INTEGER NOT NULL,
    parent_a INTEGER,
    parent_b INTEGER,
    birth_tick INTEGER NOT NULL,
    death_tick INTEGER,
    generation INTEGER NOT NULL DEFAULT 0,
    cause TEXT
);
CREATE TABLE IF NOT EXISTS interactions (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    tick INTEGER NOT NULL,
    kind TEXT NOT NULL,
    actor_id INTEGER NOT NULL,
    target_id INTEGER,
    amount REAL
);
CREATE INDEX IF NOT EXISTS idx_tick_stats_run ON tick_stats(run_id, tick);
CREATE INDEX IF NOT EXISTS idx_snapshots_run ON snapshots(run_id, tick);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, tick);
CREATE INDEX IF NOT EXISTS idx_lineage_run ON lineage(run_id, org_id);
CREATE INDEX IF NOT EXISTS idx_interactions_run ON interactions(run_id, tick);
"""


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_commit() -> str:
    """Short HEAD hash, or ``"unknown"`` outside a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip()[:12] or "unknown"


class ExperimentDB:
    """A single SQLite database of experiments, runs and their measurements."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ExperimentDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- experiments and runs ------------------------------------------------
    def create_experiment(self, name: str, config: Config, seed: int | None = None,
                          notes: str = "") -> int:
        seed = config.seed if seed is None else int(seed)
        payload = yaml.safe_dump(config.to_dict(), sort_keys=True)
        cur = self.conn.execute(
            "INSERT INTO experiments (name, created_at, seed, config_yaml, "
            "spec_version, git_commit, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, _utcnow(), seed, payload, SPEC_VERSION, _git_commit(), notes))
        self.conn.commit()
        return int(cur.lastrowid)

    def experiment_config(self, experiment_id: int) -> Config:
        row = self.conn.execute(
            "SELECT config_yaml FROM experiments WHERE id=?",
            (int(experiment_id),)).fetchone()
        if row is None:
            raise KeyError(f"no experiment with id {experiment_id}")
        return config_from_dict(yaml.safe_load(row[0]) or {})

    def start_run(self, experiment_id: int, start_tick: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (experiment_id, started_at, start_tick, status) "
            "VALUES (?, ?, ?, 'running')",
            (int(experiment_id), _utcnow(), int(start_tick)))
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, end_tick: int,
                   status: str = "completed") -> None:
        self.conn.execute(
            "UPDATE runs SET ended_at=?, end_tick=?, status=? WHERE id=?",
            (_utcnow(), int(end_tick), status, int(run_id)))
        self.conn.commit()

    def _run_config(self, run_id: int) -> Config:
        row = self.conn.execute(
            "SELECT e.config_yaml FROM runs r JOIN experiments e "
            "ON e.id = r.experiment_id WHERE r.id=?", (int(run_id),)).fetchone()
        if row is None:
            raise KeyError(f"no run with id {run_id}")
        return config_from_dict(yaml.safe_load(row[0]) or {})

    # -- measurements --------------------------------------------------------
    def flush_stats(self, run_id: int, rows: Iterable[Mapping[str, Any]]) -> int:
        """Insert one batch of per-interval rows; re-flushing a tick replaces it."""
        rows = list(rows)
        if not rows:
            return 0
        cols = list(COLUMNS)
        placeholders = ", ".join("?" for _ in range(len(cols) + 1))
        sql = (f"INSERT OR REPLACE INTO tick_stats (run_id, {', '.join(cols)}) "
               f"VALUES ({placeholders})")
        payload = [(int(run_id), *(row[c] for c in cols)) for row in rows]
        self.conn.executemany(sql, payload)
        self.conn.commit()
        return len(payload)

    def read_stats(self, run_id: int) -> list[dict[str, Any]]:
        cols = list(COLUMNS)
        sql = (f"SELECT {', '.join(cols)} FROM tick_stats WHERE run_id=? "
               f"ORDER BY tick")
        out = []
        for values in self.conn.execute(sql, (int(run_id),)):
            row = dict(zip(cols, values))
            for col in _INT_COLUMNS:
                row[col] = int(row[col])
            out.append(row)
        return out

    def append_lineage(self, run_id: int,
                       rows: Iterable[Mapping[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        sql = ("INSERT INTO lineage (run_id, org_id, parent_a, parent_b, "
               "birth_tick, death_tick, generation, cause) "
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
        payload = [(int(run_id), int(r["org_id"]), r["parent_a"], r["parent_b"],
                    int(r["birth_tick"]),
                    None if r["death_tick"] is None else int(r["death_tick"]),
                    int(r["generation"]), r.get("cause"))
                   for r in rows]
        self.conn.executemany(sql, payload)
        self.conn.commit()
        return len(payload)

    def append_interaction(self, run_id: int, rows: Iterable[Mapping[str, Any]],
                           mode: str | None = None) -> int:
        """Write interaction edges under the configured logging mode.

        ``mode=None`` uses the experiment's ``stats.interactions``.
        """
        rows = list(rows)
        if mode is None:
            mode = self._run_config(run_id).stats.interactions
        if mode == "off":
            return 0
        if mode == "full":
            if not full_logging_allowed(self._run_config(run_id)):
                raise ValueError(
                    "full interaction logging exceeds the DESIGN §9 budget for "
                    f"this run (limit: {_FULL_LOGGING_MAX_ORGANISMS} founders, "
                    f"{_FULL_LOGGING_MAX_TICKS} ticks); use 'sampled'")
        elif mode == "sampled":
            rows = [r for r in rows
                    if r["kind"] != "feed"
                    or (int(r["tick"]) + int(r["actor_id"])) % _FEED_SAMPLE_MODULUS == 0]
        else:
            raise ValueError(f"unknown interaction logging mode {mode!r}")
        if not rows:
            return 0
        sql = ("INSERT INTO interactions (run_id, tick, kind, actor_id, "
               "target_id, amount) VALUES (?, ?, ?, ?, ?, ?)")
        payload = [(int(run_id), int(r["tick"]), str(r["kind"]),
                    int(r["actor_id"]),
                    None if r["target_id"] is None else int(r["target_id"]),
                    None if r["amount"] is None else float(r["amount"]))
                   for r in rows]
        self.conn.executemany(sql, payload)
        self.conn.commit()
        return len(payload)

    def record_event(self, run_id: int, tick: int, kind: str,
                     payload: Mapping[str, Any] | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO events (run_id, tick, kind, payload_json) "
            "VALUES (?, ?, ?, ?)",
            (int(run_id), int(tick), str(kind),
             json.dumps(dict(payload or {}), sort_keys=True)))
        self.conn.commit()
        return int(cur.lastrowid)
