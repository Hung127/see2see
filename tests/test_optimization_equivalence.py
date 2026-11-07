"""Determinism guard for the optimization pass.

Every engine change in this upgrade had to leave `Engine.checksum()` identical
to the pre-change baseline. These three scenarios were captured *before* the
first optimization (commit ``a66cc0c``) and are the contract:

* ``plain``  — the default configuration, 60 ticks;
* ``sexual`` — the same world with ``genetics.mode = "sexual"`` (exercises
  crossover and mate search);
* ``nopred`` — the same world with the predation stage disabled (exercises the
  alternate branch and the carcass path).

If an optimization changes one of these hashes it changed simulation semantics,
which the upgrade forbids: a faster tick that produces different organisms is a
different experiment. When a future change *intentionally* alters semantics it
must update this file and say so in the commit message and SPEC_VERSION.

The scenarios are deliberately small: the point is to pin the arithmetic, not to
cover every code path (the rest of the suite does that). Total runtime is a few
seconds, so the guard runs in the normal suite.
"""
from __future__ import annotations

import pytest

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine

#: Hashes measured at commit a66cc0c, before the optimization pass began.
BASELINE = {
    "plain": "0ce2ac79196f4358fe6c116d106b6001d2d36c554c3b30d2646857a3626d26f1",
    "sexual": "f2aa1771733035ea239f77d1a11767974a21ebf57910ec9529940d60f7b19209",
    "nopred": "c28bfab38b053797d19973bc620dcf9f2a3d8ca287d9eebea3f23f8cfab29ceb",
}

TICKS = 60


def _config(**overrides) -> Config:
    base = Config(
        seed=11,
        ticks=TICKS,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=16,
                          cell_size=31.25),
        plants=PlantConfig(initial=1200, max=3000),
        organisms=OrganismConfig(initial=90, max=200),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.003),
        stats=StatsConfig(),
    )
    return type(base)(**{**base.__dict__, **overrides}) if overrides else base


def _run(config: Config, **step_kwargs) -> Engine:
    engine = Engine(config)
    for _ in range(TICKS):
        engine.step(**step_kwargs)
    return engine


@pytest.mark.parametrize("name,config,step_kwargs", [
    ("plain", _config(), {}),
    ("sexual", _config(genetics=GeneticsConfig(mode="sexual")), {}),
    ("nopred", _config(), {"predation": False}),
])
def test_optimised_engine_matches_pre_optimisation_checksum(
        name, config, step_kwargs):
    engine = _run(config, **step_kwargs)
    assert engine.checksum() == BASELINE[name], (
        f"{name}: simulation semantics changed during optimisation; "
        f"got {engine.checksum()}, baseline {BASELINE[name]}"
    )


def test_baseline_states_still_close_their_ledgers():
    """The pinned runs must also still satisfy conservation exactly."""
    for config, kwargs in ((_config(), {}), (_config(), {"predation": False})):
        engine = _run(config, **kwargs)
        engine.ledger.assert_closed(engine.initial_mass, engine.mass_stock(),
                                    engine.initial_energy, engine.energy_stock())
        engine.ledger.assert_full_closed(engine.initial_total,
                                         engine.total_mass())
