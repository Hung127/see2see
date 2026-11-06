"""Command-line interface tests (exit codes, headless run, reproduce)."""
import os

import numpy as np
import pytest
import yaml

from evolve.cli import main
from evolve.config import load_config
from evolve.engine import Engine
from evolve.persistence import (
    SNAPSHOT_FORMAT_VERSION, ExperimentDB, save_snapshot,
)


def write_config(tmp_path, **overrides):
    raw = {
        "seed": 5,
        "ticks": 200,
        "world": {"width": 400.0, "height": 400.0, "n_field_cells": 16,
                  "cell_size": 25.0},
        "plants": {"initial": 400, "max": 2000},
        "organisms": {"initial": 40, "max": 200},
        "stats": {"interval": 50},
    }
    raw.update(overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_run_headless_smoke(tmp_path):
    cfg_path = write_config(tmp_path)
    db_path = str(tmp_path / "run.db")

    rc = main(["run", "--config", cfg_path, "--ticks", "200", "--headless",
               "--db", db_path, "--seed", "5"])
    assert rc == 0

    db = ExperimentDB(db_path)
    try:
        status, end_tick, checksum = db.conn.execute(
            "SELECT status, end_tick, end_checksum FROM runs").fetchone()
        assert (status, end_tick) == ("completed", 200)
        assert checksum and len(checksum) == 64
        # one row per stats.interval, never per tick
        assert len(db.read_stats(1)) == 200 // 50
    finally:
        db.close()


def test_reproduce_matches(tmp_path):
    cfg_path = write_config(tmp_path)
    checksums = []
    for i in (1, 2):
        db_path = str(tmp_path / f"run{i}.db")
        assert main(["run", "--config", cfg_path, "--db", db_path,
                     "--seed", "7"]) == 0
        db = ExperimentDB(db_path)
        try:
            checksums.append(db.conn.execute(
                "SELECT end_checksum FROM runs").fetchone()[0])
        finally:
            db.close()
    assert checksums[0] == checksums[1]

    # `reproduce` re-executes from the stored config and must match the
    # stored final checksum.
    assert main(["reproduce", "--db", str(tmp_path / "run1.db"),
                 "--run-id", "1"]) == 0


def test_bad_config_exit_code(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("world:\n  boundaries: diagonal\n", encoding="utf-8")
    assert main(["run", "--config", str(p)]) == 2

    missing = tmp_path / "nope.yaml"
    assert main(["run", "--config", str(missing)]) == 2


def test_benchmark_smoke():
    assert main(["benchmark", "--population", "20", "--ticks", "5"]) == 0


def test_experiments_subcommand_smoke(tmp_path):
    out = str(tmp_path / "exp")
    assert main(["experiments", "--id", "c", "--out", out, "--scale", "0.2",
                 "--ticks", "10"]) == 0
    assert os.path.exists(os.path.join(out, "summary.md"))
    assert os.path.exists(os.path.join(out, "stats.csv"))


def test_snapshot_version_exit_code(tmp_path):
    cfg_path = write_config(tmp_path)
    cfg = load_config(cfg_path)
    eng = Engine(cfg)
    eng.run(10)
    snap = str(tmp_path / "snap.npz")
    save_snapshot(eng, snap)

    with np.load(snap) as z:
        data = {k: z[k] for k in z.files}
    data["format_version"] = np.int64(SNAPSHOT_FORMAT_VERSION + 1)
    tampered = str(tmp_path / "tampered.npz")
    np.savez_compressed(tampered, **data)

    assert main(["run", "--config", cfg_path, "--resume", tampered]) == 3
