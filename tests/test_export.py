"""Tests for CSV and matplotlib export (DESIGN §8 outputs)."""
import csv

import matplotlib

matplotlib.use("Agg")

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine
from evolve.stats import COLUMNS, StatsRecorder, export_csv, make_plots


def make_config(seed=1):
    return Config(
        seed=seed, ticks=100,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=10, cell_size=50.0),
        plants=PlantConfig(initial=100, max=300),
        organisms=OrganismConfig(initial=30, max=100),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.0),
        stats=StatsConfig(),
    )


def collect_rows(ticks=30):
    eng = Engine(make_config())
    rec = StatsRecorder()
    for _ in range(ticks):
        eng.step()
        rec.record(eng)
    return list(rec.rows)


def test_csv_roundtrip(tmp_path):
    rows = collect_rows()
    path = tmp_path / "stats.csv"
    export_csv(rows, str(path))
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        body = list(reader)
    assert header == list(COLUMNS)
    assert len(body) == len(rows)
    assert len(body[0]) == len(COLUMNS)
    # spot-check a numeric roundtrip
    tick_col = COLUMNS.index("tick")
    assert float(body[0][tick_col]) == rows[0]["tick"]


def test_export_empty_rows(tmp_path):
    path = tmp_path / "empty.csv"
    export_csv([], str(path))
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        assert next(reader) == list(COLUMNS)
        assert list(reader) == []


def test_plots_written(tmp_path):
    from evolve.stats import _PLOT_NAMES

    rows = collect_rows(60)
    out = tmp_path / "plots"
    make_plots(rows, str(out))
    names = sorted(p.name for p in out.iterdir())
    # Every declared plot must be produced; checking against _PLOT_NAMES means a
    # newly added plot is covered automatically instead of silently unasserted.
    assert names == sorted(f"{name}.png" for name in _PLOT_NAMES)
    for name in names:
        assert name.endswith(".png")
        assert (out / name).stat().st_size > 0


def test_lineage_and_interactions_csv(tmp_path):
    from evolve.stats import (INTERACTION_COLUMNS, LINEAGE_COLUMNS,
                              export_interactions_csv, export_lineage_csv)

    lineage = [{"org_id": 3, "parent_a": 1, "parent_b": None, "birth_tick": 10,
                "death_tick": 40, "generation": 2, "cause": "predation"}]
    path = tmp_path / "lineage.csv"
    export_lineage_csv(lineage, str(path))
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == list(LINEAGE_COLUMNS)
    assert rows[1][0] == "3" and rows[1][-1] == "predation"

    edges = [{"tick": 5, "kind": "bite", "actor_id": 1, "target_id": 9,
              "amount": 0.25}]
    epath = tmp_path / "interactions.csv"
    export_interactions_csv(edges, str(epath))
    with open(epath, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == list(INTERACTION_COLUMNS)
    assert float(rows[1][-1]) == 0.25


def test_viz_never_imports_export():
    import inspect

    import evolve.viz.app as app
    src = inspect.getsource(app)
    assert "export_csv" not in src and "make_plots" not in src
