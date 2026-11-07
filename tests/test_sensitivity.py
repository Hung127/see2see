"""Tests for the one-factor sensitivity harness (IMPROVE §3.4 / §9).

The harness must be an *observation* tool: it copies the base config, never
mutates a default, pairs runs by seed, and reports the measured outcome rather
than an optimised one. These tests pin those properties on a small, fast sweep.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.genome import TRAIT_NAMES
from evolve.sensitivity import (
    SWEEPS, Sweep, _set_parameter, _trait_locus, run_sweep,
)


def make_config(seed=1, ticks=30):
    return Config(
        seed=seed, ticks=ticks,
        world=WorldConfig(width=300.0, height=300.0, n_field_cells=10,
                          cell_size=30.0),
        plants=PlantConfig(initial=300, max=600),
        organisms=OrganismConfig(initial=20, max=60),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.0),
        stats=StatsConfig(interval=10),
    )


def test_set_parameter_copies_and_never_mutates_the_base():
    base = make_config()
    changed = _set_parameter(base, "energy.bite", 0.9)
    assert changed.energy.bite == pytest.approx(0.9)
    assert base.energy.bite == pytest.approx(0.5)   # unchanged
    assert changed is not base


def test_set_parameter_preserves_declared_scalar_types():
    base = make_config()
    changed = _set_parameter(base, "organisms.maturity_age", 75.7)
    assert changed.organisms.maturity_age == 76   # nearest int, field stays int
    assert isinstance(changed.organisms.maturity_age, int)
    # A float field stays a float even when an int is supplied.
    as_float = _set_parameter(base, "energy.bite", 1)
    assert isinstance(as_float.energy.bite, float)
    assert as_float.energy.bite == pytest.approx(1.0)


def test_set_parameter_rejects_unknown_sections_and_fields():
    base = make_config()
    with pytest.raises(ValueError, match="section"):
        _set_parameter(base, "nope.field", 1)
    with pytest.raises(ValueError, match="field"):
        _set_parameter(base, "energy.nope", 1)
    with pytest.raises(ValueError, match="section"):
        _set_parameter(base, "bite", 1)


def test_sweep_requires_a_baseline_member_and_two_values():
    with pytest.raises(ValueError, match="baseline"):
        Sweep("x", "energy.bite", (0.1, 0.2), 0.5, "h", "r")
    with pytest.raises(ValueError, match="two values"):
        Sweep("x", "energy.bite", (0.5,), 0.5, "h", "r")


def test_declared_sweeps_reference_real_config_fields():
    """A typo in a sweep declaration must fail here, not silently at run time."""
    base = make_config()
    for key, decl in SWEEPS.items():
        if decl.is_trait:
            # A trait sweep names a real locus and never touches the config.
            assert decl.trait in TRAIT_NAMES, key
            assert decl.parameter == f"trait.{decl.trait}", key
            locus = _trait_locus(decl.trait, decl.values[0])
            assert 0.0 <= locus <= 1.0, key
            continue
        changed = _set_parameter(base, decl.parameter, decl.values[0])
        assert changed is not base, key


def test_declared_sweeps_are_honest_about_their_design():
    for key, decl in SWEEPS.items():
        assert decl.hypothesis, f"{key} has no pre-registered hypothesis"
        assert decl.rationale, f"{key} has no rationale"
        assert len(decl.values) >= 2
        assert decl.baseline in decl.values


def test_run_sweep_writes_csv_and_summary(tmp_path):
    report = run_sweep(
        str(tmp_path), sweep="bite", seeds=(11,), ticks=20, scale=0.15)
    assert (tmp_path / "sensitivity.csv").exists()
    assert (tmp_path / "summary.md").exists()
    with open(tmp_path / "sensitivity.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(SWEEPS["bite"].values)      # one seed, all values
    for row in rows:
        assert row["sweep"] == "bite"
        assert row["parameter"] == "energy.bite"
    text = (tmp_path / "summary.md").read_text()
    assert "Hypothesis (pre-registered)" in text
    assert "Limitations" in text
    assert "underpowered" in text


def test_run_sweep_is_reproducible(tmp_path):
    a = run_sweep(str(tmp_path / "a"), sweep="eta_p", seeds=(11, 12),
                  ticks=20, scale=0.15)
    b = run_sweep(str(tmp_path / "b"), sweep="eta_p", seeds=(11, 12),
                  ticks=20, scale=0.15)
    assert [r["checksum"] for r in a["rows"]] == [r["checksum"] for r in b["rows"]]
    for row_a, row_b in zip(a["sweeps"]["eta_p"]["table"],
                            b["sweeps"]["eta_p"]["table"]):
        assert row_a["final_population_diff"] == pytest.approx(
            row_b["final_population_diff"])
        assert row_a["final_population_p"] == pytest.approx(
            row_b["final_population_p"])


def test_run_sweep_pairs_runs_by_seed(tmp_path):
    """Every parameter value must run exactly the same seeds."""
    report = run_sweep(str(tmp_path), sweep="decay", seeds=(11, 12, 13),
                       ticks=15, scale=0.12)
    seen = {}
    for row in report["rows"]:
        seen.setdefault(row["parameter_value"], set()).add(row["seed"])
    for seeds in seen.values():
        assert seeds == {11, 12, 13}
    assert len(seen) == len(SWEEPS["decay"].values)


def test_run_sweep_rejects_unknown_and_empty_seeds(tmp_path):
    with pytest.raises(ValueError, match="unknown sweep"):
        run_sweep(str(tmp_path), sweep="nope", seeds=(11,), ticks=5)
    with pytest.raises(ValueError, match="at least one seed"):
        run_sweep(str(tmp_path), sweep="bite", seeds=(), ticks=5)


def test_sweep_reports_extinction_rather_than_hiding_it(tmp_path):
    """A sweep that kills the population must say so in the CSV."""
    report = run_sweep(str(tmp_path), sweep="bite", seeds=(11,), ticks=20,
                       scale=0.1)
    assert any("extinct" in row for row in report["rows"])
    for row in report["rows"]:
        assert isinstance(row["extinct"], bool)


# -- IMPROVE §3.4: trait-level sweeps (founder genome loci) -------------------
def test_trait_locus_inverts_the_gene_map():
    """A physical value must map to the locus that reproduces it exactly."""
    from evolve.genome import GENE_SPECS, phenotype

    for spec in GENE_SPECS:
        # Sample the declared span; for exp genes use positive interior points.
        if spec.kind == "exp":
            values = (spec.lo, spec.lo * 4.0, spec.hi)
        else:
            values = (spec.lo, 0.5 * (spec.lo + spec.hi), spec.hi)
        for value in values:
            locus = _trait_locus(spec.name, value)
            assert 0.0 <= locus <= 1.0
            genes = np.zeros((1, len(GENE_SPECS)))
            genes[0, TRAIT_NAMES.index(spec.name)] = locus
            got = phenotype(genes)[0, TRAIT_NAMES.index(spec.name)]
            assert got == pytest.approx(value, rel=1e-9, abs=1e-9)


def test_trait_locus_rejects_out_of_range_and_unknown():
    with pytest.raises(ValueError, match="outside"):
        _trait_locus("vision_range", 10_000.0)
    # ``lifespan`` is log-mapped, so a non-positive value has no locus.
    with pytest.raises(ValueError, match="positive"):
        _trait_locus("lifespan", 0.0)
    with pytest.raises(ValueError, match="unknown trait"):
        _trait_locus("nope", 1.0)


def test_sweep_validates_trait_declarations():
    with pytest.raises(ValueError, match="unknown trait"):
        Sweep("x", "trait.nope", (1.0, 2.0), 1.0, "h", "r", trait="nope")
    with pytest.raises(ValueError, match="must name"):
        Sweep("x", "energy.bite", (1.0, 2.0), 1.0, "h", "r",
              trait="vision_range")
    ok = Sweep("x", "trait.vision_range", (60.0, 100.0), 100.0, "h", "r",
               trait="vision_range")
    assert ok.is_trait
    assert not SWEEPS["bite"].is_trait


def test_trait_sweep_leaves_the_base_config_alone_and_is_reproducible(tmp_path):
    from dataclasses import asdict

    from evolve.config import load_config
    from evolve.sensitivity import CONFIGS_DIR, DEFAULT_CONFIG

    import os

    base = load_config(os.path.join(CONFIGS_DIR, DEFAULT_CONFIG))
    snapshot = asdict(base)
    a = run_sweep(str(tmp_path / "a"), sweep="vision_range", seeds=(11,),
                  ticks=20, scale=0.15)
    b = run_sweep(str(tmp_path / "b"), sweep="vision_range", seeds=(11,),
                  ticks=20, scale=0.15)
    assert asdict(load_config(os.path.join(CONFIGS_DIR, DEFAULT_CONFIG))) == snapshot
    assert [r["checksum"] for r in a["rows"]] == [r["checksum"] for r in b["rows"]]
    # Different values must actually change the engine's state.
    assert len({r["checksum"] for r in a["rows"]}) > 1


def test_trait_sweep_refuses_body_size():
    from evolve.sensitivity import _run_one

    with pytest.raises(ValueError, match="cannot be swept"):
        _run_one(make_config(), 11, trait="body_size", trait_value=1.0)


def test_trait_sweep_summary_names_the_variable_kind(tmp_path):
    run_sweep(str(tmp_path), sweep="metabolic_rate", seeds=(11,), ticks=10,
              scale=0.12)
    text = (tmp_path / "summary.md").read_text()
    assert "founder trait locus" in text
    assert "config coefficient" not in text.split("## `metabolic_rate`")[1] \
        .split("## Limitations")[0]

