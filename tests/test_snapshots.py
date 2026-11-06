"""Snapshot save/load and exact resume (DESIGN §9)."""
import numpy as np
import pytest

from evolve.config import (
    Config, EnvironmentConfig, EventConfig, OrganismConfig, PlantConfig,
    StatsConfig, WorldConfig,
)
from evolve.engine import Engine
from evolve.persistence import SNAPSHOT_FORMAT_VERSION, load_snapshot, save_snapshot


def small_config(ticks=400):
    """Small world; the drought is active when the snapshot is taken and random
    events add scheduler state that must survive the round-trip.

    Scaling (this machine is single-core, ~1 GB): the plan's 400 + 600 ticks on a
    500x500 world cost 61 s for the file; this 400-tick world with a 200/200
    split costs ~30 s and still covers a mid-event snapshot, spawned random
    events, live births and live predation.
    """
    return Config(
        seed=3,
        ticks=ticks,
        world=WorldConfig(width=400.0, height=400.0, n_field_cells=16,
                          cell_size=25.0),
        plants=PlantConfig(initial=1200, max=3000),
        organisms=OrganismConfig(initial=120, max=400),
        environment=EnvironmentConfig(
            events=(EventConfig(kind="drought", start_tick=150, duration=100,
                                magnitude=0.8),),
            random_events=(("drought", 0.02), ("heavy_rain", 0.02))),
        stats=StatsConfig(),
    )


def test_resume_equals_continuation(tmp_path):
    """Running 200, snapshotting, then 200 in a fresh engine must equal an
    uninterrupted 400-tick run, bit for bit (rounded to the checksum's 1e-9).

    The snapshot is taken at tick 200, in the middle of the scheduled drought
    (150-250), so continuing correctly requires the event state, the RNG stream
    and any spawned random events to be restored exactly.
    """
    cfg = small_config()
    a = Engine(cfg)
    a.run(200)
    assert a.tick == 200
    manifest = save_snapshot(a, str(tmp_path / "snap.npz"))
    assert manifest["tick"] == 200
    assert manifest["format_version"] == SNAPSHOT_FORMAT_VERSION
    assert a.events.active(200), "snapshot should be taken mid-event"

    resumed = load_snapshot(str(tmp_path / "snap.npz"), cfg)
    assert resumed.tick == 200
    resumed.run(200)

    direct = Engine(cfg)
    direct.run(400)

    assert resumed.checksum() == direct.checksum()
    assert resumed.population.n_alive == direct.population.n_alive
    assert resumed.births_total == direct.births_total
    assert resumed.deaths_total == direct.deaths_total


def test_unknown_version_refused(tmp_path):
    path = str(tmp_path / "snap.npz")
    eng = Engine(small_config(ticks=10))
    eng.run(10)
    save_snapshot(eng, path)

    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data["format_version"] = np.int64(SNAPSHOT_FORMAT_VERSION + 1)
    tampered = str(tmp_path / "tampered.npz")
    np.savez_compressed(tampered, **data)

    with pytest.raises(ValueError, match="version"):
        load_snapshot(tampered, small_config(ticks=10))


def test_snapshot_deterministic(tmp_path):
    """Two saves of the same state produce the same content hash.

    The hash covers the canonical array contents (sorted keys, dtype, shape,
    raw bytes), not the .npz container bytes: zip entries carry timestamps, so
    the container is not byte-stable and the manifest hash is the integrity
    anchor instead.
    """
    cfg = small_config(ticks=150)
    e1 = Engine(cfg)
    e1.run(150)
    e2 = Engine(cfg)
    e2.run(150)
    p1 = str(tmp_path / "a.npz")
    p2 = str(tmp_path / "b.npz")
    man1 = save_snapshot(e1, p1)
    man2 = save_snapshot(e2, p2)
    assert man1["sha256"] == man2["sha256"]
    assert man1["config_hash"] == man2["config_hash"]
    # and the saved file loads back to the same checksum
    back = load_snapshot(p1, cfg)
    assert back.checksum() == e1.checksum()
