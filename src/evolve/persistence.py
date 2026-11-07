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

import hashlib
import json
import sqlite3
import subprocess
import time
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .config import SPEC_VERSION, Config, config_from_dict, config_hash
from .engine import Engine
from .events import Event
from .organisms import _STATE_ARRAYS as _POP_ARRAYS
from .stats import COLUMNS

#: ``tick_stats`` columns stored as SQLite INTEGER; everything else is REAL.
_INT_COLUMNS = frozenset({
    "tick", "population", "births", "deaths", "herbivores", "carnivores",
    "omnivores", "unclassified", "unique_genomes",
    # appended in the upgrade
    "max_generation", "deaths_starvation", "deaths_senescence",
    "deaths_health", "deaths_predation",
    # IMPROVE §5: a 0/1 flag, stored as an integer like the other flags
    "bottleneck",
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
    status TEXT NOT NULL DEFAULT 'running',
    end_checksum TEXT
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
        self._migrate_tick_stats()
        self.conn.commit()

    def _migrate_tick_stats(self) -> None:
        """Add statistics columns introduced after the database was created.

        ``CREATE TABLE IF NOT EXISTS`` leaves an older ``tick_stats`` alone, so a
        database written before the upgrade would fail on the new columns.
        Adding the missing ones is the additive, backward-compatible migration:
        old rows get the column's default, no existing value changes, and the
        original column order is preserved.
        """
        present = {row[1] for row in
                   self.conn.execute("PRAGMA table_info(tick_stats)")}
        for col in COLUMNS:
            if col == "tick" or col in present:
                continue
            kind = "INTEGER" if col in _INT_COLUMNS else "REAL"
            self.conn.execute(
                f"ALTER TABLE tick_stats ADD COLUMN {col} {kind} "
                f"NOT NULL DEFAULT 0")

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
                   status: str = "completed",
                   checksum: str | None = None) -> None:
        self.conn.execute(
            "UPDATE runs SET ended_at=?, end_tick=?, status=?, end_checksum=? "
            "WHERE id=?",
            (_utcnow(), int(end_tick), status, checksum, int(run_id)))
        self.conn.commit()

    def run_record(self, run_id: int) -> dict[str, Any]:
        """Stored config, seed and final checksum of one run."""
        row = self.conn.execute(
            "SELECT e.config_yaml, e.seed, r.end_tick, r.status, r.end_checksum "
            "FROM runs r JOIN experiments e ON e.id = r.experiment_id "
            "WHERE r.id=?", (int(run_id),)).fetchone()
        if row is None:
            raise KeyError(f"no run with id {run_id}")
        return {
            "config": config_from_dict(yaml.safe_load(row[0]) or {}),
            "seed": int(row[1]),
            "end_tick": row[2],
            "status": row[3],
            "end_checksum": row[4],
        }

    def add_snapshot(self, run_id: int, tick: int, path: str, sha256: str,
                     format_version: int, rng_state: bytes | None = None,
                     manifest: Mapping[str, Any] | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO snapshots (run_id, tick, path, sha256, format_version, "
            "rng_state, manifest_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (int(run_id), int(tick), str(path), str(sha256),
             int(format_version), rng_state,
             json.dumps(dict(manifest or {}), sort_keys=True)))
        self.conn.commit()
        return int(cur.lastrowid)

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

    def read_lineage(self, run_id: int) -> list[dict[str, Any]]:
        """Recorded ancestry rows for a run, ordered by birth then organism ID."""
        sql = ("SELECT org_id, parent_a, parent_b, birth_tick, death_tick, "
               "generation, cause FROM lineage WHERE run_id=? "
               "ORDER BY birth_tick, org_id")
        return [dict(zip(("org_id", "parent_a", "parent_b", "birth_tick",
                          "death_tick", "generation", "cause"), values))
                for values in self.conn.execute(sql, (int(run_id),))]

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


# ---------------------------------------------------------------------------
# Snapshots (DESIGN §9): compressed .npz of every state array + RNG state +
# counters + config hash. ``format_version`` is checked on load; unknown
# versions are refused with a clear error.
# ---------------------------------------------------------------------------
SNAPSHOT_FORMAT_VERSION = 1


class SnapshotVersionError(ValueError):
    """The snapshot's format version is not readable by this build."""


_ENTITY_ARRAYS = ("id", "x", "y", "biomass", "age", "alive")
_FIELD_ARRAYS = ("terrain", "soil", "temperature", "moisture", "pheromone",
                 "plant_density", "organism_density", "_growth_factor",
                 "water_stress")
_LEDGER_FIELDS = ("light_in", "heat_out", "mass_to_soil", "mass_from_soil",
                  "mass_in")
_ENGINE_INTS = ("tick", "births_total", "deaths_total", "mutations_total")
_ENGINE_FLOATS = ("detritus", "initial_mass", "initial_energy", "initial_soil",
                  "initial_total")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    """Content hash over the snapshot arrays in canonical (sorted-key) order.

    The .npz container itself is not byte-stable (zip entries carry timestamps),
    so the manifest hash is computed from the arrays instead.
    """
    digest = hashlib.sha256()
    for key in sorted(payload):
        arr = np.asarray(payload[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def _damage_credit_payload(engine) -> np.ndarray:
    """Damage-credit map as a (victim, attacker, amount) int/float table."""
    rows = [(victim, attacker, amount)
            for victim, credits in sorted(engine._damage_credit.items())
            for attacker, amount in credits]
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def save_snapshot(engine, path: str) -> dict[str, Any]:
    """Write the full engine state to ``path`` (.npz) and return a manifest."""
    payload: dict[str, Any] = {}

    pop = engine.population
    for name in _POP_ARRAYS:
        arr = getattr(pop, name)
        payload[f"pop_{name}"] = (np.asarray(arr, dtype="U32")
                                  if arr.dtype == object else np.asarray(arr))
    payload["pop_genome"] = np.asarray(pop.genome)
    payload["pop__next_id"] = np.int64(pop._next_id)
    payload["pop_capacity"] = np.int64(pop.capacity)

    for prefix, obj in (("plants", engine.plants), ("carc", engine.carcasses)):
        for name in _ENTITY_ARRAYS:
            payload[f"{prefix}_{name}"] = np.asarray(getattr(obj, name))
        payload[f"{prefix}__next_id"] = np.int64(obj._next_id)
        payload[f"{prefix}_capacity"] = np.int64(obj.capacity)

    for name in _FIELD_ARRAYS:
        payload[f"fields_{name}"] = np.asarray(getattr(engine.fields, name))

    events = engine.events._events
    payload["ev_kind"] = np.asarray([e.kind for e in events], dtype="U32")
    payload["ev_start"] = np.asarray([e.start_tick for e in events],
                                     dtype=np.int64)
    payload["ev_duration"] = np.asarray([e.duration for e in events],
                                        dtype=np.int64)
    payload["ev_magnitude"] = np.asarray([e.magnitude for e in events],
                                         dtype=np.float64)
    payload["ev_region"] = np.asarray([e.region for e in events],
                                      dtype=np.float64).reshape(-1, 3)
    payload["ev_fired"] = np.asarray(sorted(engine.events._fired),
                                     dtype=np.int64)

    payload["rng_state"] = np.frombuffer(engine.rng.state(), dtype=np.uint8)
    for name in _ENGINE_INTS:
        payload[name] = np.int64(getattr(engine, name))
    for name in _ENGINE_FLOATS:
        payload[name] = np.float64(getattr(engine, name))
    for name in _LEDGER_FIELDS:
        payload[f"ledger_{name}"] = np.float64(getattr(engine.ledger, name))
    payload["damage_credit"] = _damage_credit_payload(engine)
    payload["death_causes"] = np.asarray(sorted(engine.death_counts),
                                        dtype="U32")
    payload["death_counts"] = np.asarray(
        [engine.death_counts[c] for c in sorted(engine.death_counts)],
        dtype=np.int64)
    payload["death_age_hist"] = np.asarray(engine.death_age_hist, dtype=np.int64)
    # Cumulative run totals (IMPROVE §4.2). Statistics-only, so they are stored
    # for reporting continuity but excluded from ``checksum()``. Names and order
    # come from ``engine.RUN_TOTALS`` so the two cannot drift.
    from .engine import RUN_TOTALS
    payload["run_total_names"] = np.asarray(list(RUN_TOTALS), dtype="U32")
    payload["run_total_values"] = np.asarray(
        [float(engine.run_totals.get(name, 0.0)) for name in RUN_TOTALS],
        dtype=np.float64)
    payload["config_hash"] = np.asarray(config_hash(engine.cfg))
    payload["spec_version"] = np.asarray(SPEC_VERSION)
    payload["format_version"] = np.int64(SNAPSHOT_FORMAT_VERSION)

    np.savez_compressed(path, **payload)
    return {
        "path": str(path),
        "tick": int(engine.tick),
        "population": int(engine.population.n_alive),
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "spec_version": SPEC_VERSION,
        "config_hash": config_hash(engine.cfg),
        "sha256": _canonical_sha256(payload),
    }


def load_snapshot(path: str, config: Config):
    """Rebuild an :class:`~evolve.engine.Engine` from a snapshot file."""
    with np.load(path, allow_pickle=False) as z:
        version = int(z["format_version"])
        if version != SNAPSHOT_FORMAT_VERSION:
            raise SnapshotVersionError(
                f"unknown snapshot format version {version}; this build reads "
                f"version {SNAPSHOT_FORMAT_VERSION} ({path})")
        stored_hash = str(z["config_hash"])
        if stored_hash != config_hash(config):
            raise ValueError(
                "snapshot was written with a different configuration "
                f"(stored {stored_hash[:12]}..., given "
                f"{config_hash(config)[:12]}...); refusing to load")

        engine = Engine(config)

        pop = engine.population
        for name in _POP_ARRAYS:
            arr = z[f"pop_{name}"]
            if name == "death_cause":
                arr = np.array([str(s) for s in arr], dtype=object)
            setattr(pop, name, arr.copy())
        pop.genome = z["pop_genome"].copy()
        pop._next_id = int(z["pop__next_id"])
        pop.capacity = int(z["pop_capacity"])

        for prefix, obj in (("plants", engine.plants),
                            ("carc", engine.carcasses)):
            for name in _ENTITY_ARRAYS:
                setattr(obj, name, z[f"{prefix}_{name}"].copy())
            obj._next_id = int(z[f"{prefix}__next_id"])
            obj.capacity = int(z[f"{prefix}_capacity"])

        for name in _FIELD_ARRAYS:
            setattr(engine.fields, name, z[f"fields_{name}"].copy())

        kinds = [str(k) for k in z["ev_kind"]]
        starts = z["ev_start"]
        durations = z["ev_duration"]
        magnitudes = z["ev_magnitude"]
        regions = z["ev_region"]
        events = [Event(kinds[i], int(starts[i]), int(durations[i]),
                        float(magnitudes[i]), tuple(float(v) for v in regions[i]))
                  for i in range(len(kinds))]
        engine.events._events = events
        engine.events.log = list(events)
        engine.events._fired = {int(i) for i in z["ev_fired"]}

        engine.rng.set_state(bytes(z["rng_state"]))
        for name in _ENGINE_INTS:
            setattr(engine, name, int(z[name]))
        for name in _ENGINE_FLOATS:
            setattr(engine, name, float(z[name]))
        for name in _LEDGER_FIELDS:
            setattr(engine.ledger, name, float(z[f"ledger_{name}"]))
        credit: dict[int, list[tuple[int, float]]] = {}
        for victim, attacker, amount in z["damage_credit"]:
            credit.setdefault(int(victim), []).append((int(attacker),
                                                       float(amount)))
        engine._damage_credit = credit
        # Death counters are statistics, not dynamics, but a resumed run should
        # keep reporting them, so they are stored when present. Snapshots from
        # before this array existed simply start the counters at zero.
        if "death_causes" in z.files:
            engine.death_counts = {
                str(cause): int(count) for cause, count
                in zip(z["death_causes"], z["death_counts"])}
            engine.death_age_hist = np.asarray(z["death_age_hist"],
                                               dtype=np.int64).copy()
        # Cumulative run totals (IMPROVE §4.2). Statistics, not dynamics, so a
        # build that predates them simply starts its totals at zero; the names
        # are read back rather than assumed, so an older snapshot with a
        # different (shorter) name list still loads.
        if "run_total_names" in z.files and "run_total_values" in z.files:
            names = [str(name) for name in z["run_total_names"]]
            values = np.asarray(z["run_total_values"], dtype=np.float64)
            for name, value in zip(names, values):
                engine.run_totals[name] = float(value)

    engine._refresh_phen()
    return engine
