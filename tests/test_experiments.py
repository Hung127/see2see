"""Experiment suite A-H and the pre-registered assay (DESIGN §13).

The suite itself is *scaled down* on this machine; the tests below therefore run
miniature versions (a few ticks, tiny worlds) that exercise the machinery --
config loading, the scenario hooks, mirrored arena pairing, plant-layout
identity, the pre-registration ordering -- not the scientific result. The
scientific runs and their honest outcomes live in ``docs/RESULTS.md``.
"""
import os

import pytest

from evolve.config import load_config
from evolve.experiments import (
    ASSAY_CONFIG, CONFIGS_DIR, SCENARIOS, run_assay, run_scenario,
)


def test_all_configs_load():
    files = sorted(f for f in os.listdir(CONFIGS_DIR) if f.endswith(".yaml"))
    assert "assay.yaml" in files
    for name in files:
        cfg = load_config(os.path.join(CONFIGS_DIR, name))
        assert cfg.ticks > 0
        assert cfg.organisms.initial > 0
        assert cfg.plants.initial > 0
        # every config must already obey the model invariants
        assert cfg.world.n_field_cells > 0

    # G is the assay, driven by run_assay, not a single-scenario entry.
    assert set(SCENARIOS) == set("abcdefh")
    for scenario in SCENARIOS.values():
        assert scenario.config_file in files
        assert scenario.name and scenario.manipulation
        assert scenario.horizon > 0
    assert ASSAY_CONFIG in files


def test_run_scenario_writes_outputs(tmp_path):
    """A scenario run writes CSV, the seven plots and a summary."""
    summary = run_scenario("d", str(tmp_path), scale=0.25, ticks=15)

    assert (tmp_path / "stats.csv").exists()
    for name in ("population", "births_deaths", "roles", "traits",
                 "diversity", "mutation", "predator_prey"):
        assert (tmp_path / f"{name}.png").exists(), name
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "# Experiment D" in text
    assert "extinct" in text.lower()
    assert summary["scenario"] == "d"
    assert summary["ticks"] >= 1


def test_assay_paired_design(tmp_path):
    """Evolution and random genomes share each arena, mirrored and seed-locked.

    Positions must be mirror images about the arena centre (so neither side gets
    a systematic home-field advantage), and every arena in a replicate must use
    the identical plant layout (same arena seed), so intake differences are
    attributable to the genomes and not to the plants.
    """
    result = run_assay(str(tmp_path), scale=0.2, replicates=1, samples=2,
                       ticks=10, arena_ticks=5)

    arenas = result["arenas"]
    assert len(arenas) == 2
    width, height = result["arena_world"]
    first_layout = arenas[0]["plant_layout_hash"]
    for arena in arenas:
        ex, ey = arena["evolved_pos"]
        rx, ry = arena["random_pos"]
        assert (ex + rx) % width == pytest.approx(0.0, abs=1e-9)
        assert (ey + ry) % height == pytest.approx(0.0, abs=1e-9)
        assert arena["plant_layout_hash"] == first_layout
        assert arena["evolved_intake"] >= 0.0
        assert arena["random_intake"] >= 0.0


def test_pre_registration_written_first(tmp_path):
    """The protocol document must exist before any engine is constructed."""
    class Refused(RuntimeError):
        pass

    def failing_factory(config):
        raise Refused("engine construction refused")

    with pytest.raises(Refused):
        run_assay(str(tmp_path), scale=0.2, replicates=1, samples=1,
                  ticks=5, arena_ticks=1, engine_factory=failing_factory)

    text = (tmp_path / "pre_registration.md").read_text(encoding="utf-8")
    assert "pre-registration" in text.lower()
    assert "wilcoxon" in text.lower()
    # and nothing was written after the refusal
    assert not (tmp_path / "assay.csv").exists()
