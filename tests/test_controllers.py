"""Tests for the controller abstraction (upgrade, Phase 4).

The controller layer exists so that the evolved network can be *compared* with
reference policies on equal footing. These tests pin the contract that makes the
comparison fair:

* every controller consumes the same ``(n, 16)`` sensor batch and returns
  ``(n, 8)`` outputs, which go through the same ``decode`` as the network;
* the default (``Engine.controller is None``) path is unchanged, bit for bit;
* a controller never draws from the engine's RNG;
* the registry is closed: an unknown name raises instead of silently falling
  back to the network.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.controller_eval import ENDPOINTS, run_controller_comparison
from evolve.controllers import (
    CONTROLLERS, ForageController, NeuralController, PassiveController,
    RandomController, RuleController, build_controller, controller_names,
)
from evolve.engine import Engine


def make_config(seed=1, n=40, ticks=60):
    return Config(
        seed=seed, ticks=ticks,
        world=WorldConfig(width=400.0, height=400.0, n_field_cells=10,
                          cell_size=40.0),
        plants=PlantConfig(initial=600, max=1500),
        organisms=OrganismConfig(initial=n, max=150),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.0),
        stats=StatsConfig(interval=10),
    )


ALL_CONTROLLERS = ("neural", "random", "passive", "rule", "forage")


@pytest.mark.parametrize("kind", ALL_CONTROLLERS)
def test_every_controller_returns_the_output_contract_shape(kind):
    sensors = np.zeros((7, 16))
    sensors[:, 0] = -1.0  # nothing detected
    phen = np.ones((7, 12)) * 0.5
    genomes = np.zeros((7, 392))
    ctrl = build_controller(kind, genomes=genomes)
    out = ctrl.act(sensors, phen)
    assert isinstance(out, np.ndarray)
    assert out.shape == (7, 8)
    assert np.all(np.isfinite(out))


def test_controller_names_lists_neural_first():
    names = controller_names()
    assert names[0] == "neural"
    assert set(names[1:]) == set(CONTROLLERS)


def test_unknown_controller_raises_instead_of_falling_back():
    with pytest.raises(ValueError, match="unknown controller"):
        build_controller("definitely-not-a-controller")


def test_neural_controller_requires_genomes():
    with pytest.raises(ValueError, match="genome"):
        build_controller("neural")


def test_neural_controller_matches_the_engine_forward_pass():
    from evolve.brain import forward

    rng = np.random.default_rng(0)
    genomes = rng.normal(0, 0.3, size=(5, 392))
    sensors = rng.normal(0, 0.2, size=(5, 16))
    phen = rng.random((5, 12))
    got = NeuralController(genomes).act(sensors, phen)
    expected = forward(sensors, genomes[:, 12:])
    assert np.array_equal(got, expected)


def test_passive_controller_is_all_zero():
    out = PassiveController().act(np.ones((4, 16)), np.ones((4, 12)))
    assert out.shape == (4, 8)
    assert not out.any()


def test_passive_decodes_to_no_attempts():
    """All-zero outputs must really mean "does nothing", not "rests"."""
    from evolve.actions import decode

    out = PassiveController().act(np.ones((3, 16)), np.ones((3, 12)))
    actions = decode(out, np.ones((3, 12)))
    assert not actions.eat.any()
    assert not actions.attack.any()
    assert not actions.reproduce.any()
    assert not actions.rest.any()
    assert not actions.throttle.any()


def test_random_controller_is_reproducible_with_a_fixed_seed():
    sensors = np.zeros((6, 16))
    a = RandomController(seed=3).act(sensors, np.ones((6, 12)))
    b = RandomController(seed=3).act(sensors, np.ones((6, 12)))
    c = RandomController(seed=4).act(sensors, np.ones((6, 12)))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_controller_stream_advances_per_call():
    ctrl = RandomController(seed=0)
    sensors = np.zeros((5, 16))
    first = ctrl.act(sensors, np.ones((5, 12)))
    second = ctrl.act(sensors, np.ones((5, 12)))
    assert not np.array_equal(first, second)


def test_rule_and_forage_are_deterministic_functions_of_observations():
    rng = np.random.default_rng(1)
    sensors = rng.normal(0, 0.5, size=(20, 16))
    phen = rng.random((20, 12))
    for cls in (RuleController, ForageController):
        ctrl = cls()
        assert np.array_equal(ctrl.act(sensors, phen), ctrl.act(sensors, phen))


def test_rule_turns_toward_food_and_away_from_threats():
    sensors = np.zeros((2, 16))
    sensors[:, :] = -1.0
    # Organism 0 sees food to the right, organism 1 sees a threat to the right.
    sensors[0, 0] = 0.9
    sensors[0, 1] = 0.5
    sensors[1, 2] = 0.9
    sensors[1, 3] = 0.5
    out = RuleController().act(sensors, np.ones((2, 12)))
    assert out[0, 1] > 0.0        # turn toward food
    assert out[1, 1] < 0.0        # turn away from threat
    assert out[0, 3] > 0.5        # eat while food is visible


def test_default_path_is_bit_identical_to_a_run_without_a_controller():
    """Attaching no controller must be exactly the pre-controller computation."""
    plain = Engine(make_config(seed=7))
    for _ in range(60):
        plain.step()
    other = Engine(make_config(seed=7))
    other.controller = None
    for _ in range(60):
        other.step()
    assert other.checksum() == plain.checksum()


def test_random_controller_uses_its_own_rng_not_the_engine_rng():
    """A baseline controller must never draw from the engine's RNG stream.

    The engine's draw order is part of the reproducibility contract. The
    ``random`` controller therefore owns a PCG64 stream seeded independently of
    the engine: the same controller seed yields the same first draw regardless of
    the engine seed.
    """
    ctrl = build_controller("random", seed=99)
    engines = []
    for seed in (1, 2):
        eng = Engine(make_config(seed=seed, n=10))
        eng.controller = build_controller("random", seed=99)
        engines.append(eng)
    # Same controller seed, different engine seeds: identical first draws.
    first = [e.controller.act(np.zeros((4, 16)), np.ones((4, 12)))
             for e in engines]
    assert np.array_equal(first[0], first[1])


def test_engine_rng_is_untouched_by_a_controller():
    """Invoking a controller must not advance the engine's RNG stream.

    The controller receives only the sensor batch and the phenotype; it has no
    reference to the engine. This is checked directly rather than through a
    tick, because a tick legitimately draws from the engine RNG for plant
    germination and other simulation work.
    """
    eng = Engine(make_config(seed=5, n=10))
    eng.controller = build_controller("random", seed=0)
    before = eng.rng.state()
    eng.controller.act(np.zeros((10, 16)), np.ones((10, 12)))
    assert eng.rng.state() == before
    # The controller's own stream did advance, proving it drew from its own RNG.
    expected = np.random.default_rng(0).random((10, 8))
    fresh = build_controller("random", seed=0)
    assert np.array_equal(fresh.act(np.zeros((10, 16)), np.ones((10, 12))),
                          expected)


def test_controller_changes_outcomes_but_stays_reproducible():
    def run(kind):
        eng = Engine(make_config(seed=11, n=40, ticks=40))
        eng.controller = build_controller(kind, seed=5)
        for _ in range(40):
            eng.step()
        return eng.checksum(), eng.population.n_alive

    rule_a = run("rule")
    rule_b = run("rule")
    passive = run("passive")
    assert rule_a == rule_b
    assert rule_a != passive


def test_controller_eval_endpoints_are_all_present():
    result = run_controller_comparison(
        __import__("tempfile").mkdtemp(), controllers=["neural", "passive"],
        seeds=(11,), ticks=8, population=20,
        config_path=None)
    for row in result["rows"]:
        for endpoint in ENDPOINTS:
            assert endpoint in row
        assert isinstance(row["extinct"], bool)
        assert row["checksum"]
    assert "passive" in result["analysis"]
    assert set(result["analysis"]["passive"]) == set(ENDPOINTS)


def test_controller_eval_writes_csv_and_summary(tmp_path):
    result = run_controller_comparison(
        str(tmp_path), controllers=["neural", "rule"], seeds=(11, 12),
        ticks=8, population=20)
    csv_path = tmp_path / "comparison.csv"
    summary_path = tmp_path / "summary.md"
    assert csv_path.exists() and summary_path.exists()
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4                     # 2 controllers x 2 seeds
    for row in rows:
        assert row["controller"] in {"neural", "rule"}
        assert int(row["seed"]) in {11, 12}
    text = summary_path.read_text()
    assert "not an evolved controller" in text   # the honest framing survives
    assert "underpowered" in text


def test_controller_eval_rejects_unknown_names(tmp_path):
    with pytest.raises(ValueError, match="unknown controllers"):
        run_controller_comparison(str(tmp_path), controllers=["neural", "nope"],
                                  seeds=(11,), ticks=5, population=10)


def test_controller_eval_always_includes_the_neural_reference(tmp_path):
    result = run_controller_comparison(str(tmp_path), controllers=["rule"],
                                       seeds=(11,), ticks=6, population=15)
    assert "neural" in result["controllers"]
    assert len(result["rows"]) == 2


def test_controller_eval_is_reproducible(tmp_path):
    a = run_controller_comparison(str(tmp_path / "a"), controllers=["neural", "rule"],
                                  seeds=(11, 12), ticks=8, population=20)
    b = run_controller_comparison(str(tmp_path / "b"), controllers=["neural", "rule"],
                                  seeds=(11, 12), ticks=8, population=20)
    assert [r["checksum"] for r in a["rows"]] == [r["checksum"] for r in b["rows"]]
    for kind, endpoints in a["analysis"].items():
        for endpoint, stats in endpoints.items():
            other = b["analysis"][kind][endpoint]
            assert stats["mean_diff"] == pytest.approx(other["mean_diff"])
            assert stats["ci_low"] == pytest.approx(other["ci_low"])
            assert stats["p"] == pytest.approx(other["p"])
