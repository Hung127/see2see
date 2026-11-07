"""Pluggable controllers: observation -> controller -> action.

Every controller consumes the **same** 16-channel sensor vector and the same
phenotype row, and returns the same 8 continuous outputs that
:func:`evolve.actions.decode` turns into physical attempts. That is the whole
contract, and it is what makes comparisons fair: a baseline cannot see anything
the evolved network cannot see, and it cannot bypass the action rules (thresholds,
rest blocking, sprint cost).

The evolved MLP is the default and is *not* replaced. ``Engine.controller`` is
``None`` by default, which calls ``brain.forward`` exactly as before; setting a
controller opts in to a different decision function for every organism in the
run.

Baseline controllers receive their own RNG stream, never the engine's. The
engine's draw order is part of the reproducibility contract, so a controller that
needs randomness must not consume from it.

None of these controllers is scripted behaviour added to the model: they exist
only to answer "is the evolved controller better than this reference?" in a
separate evaluation run. They are never enabled implicitly.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .brain import forward


class Controller(Protocol):
    """The minimal interface every controller implements."""

    name: str

    def act(self, sensors: np.ndarray, phen: np.ndarray) -> np.ndarray:
        """Return ``(n, 8)`` outputs for ``sensors`` ``(n, 16)``."""
        ...


class NeuralController:
    """The existing evolved MLP, unchanged (the default when none is set)."""

    name = "neural"

    def __init__(self, genomes: np.ndarray) -> None:
        self.genomes = np.asarray(genomes, dtype=np.float64)

    def act(self, sensors: np.ndarray, phen: np.ndarray) -> np.ndarray:
        # The network weights come from the genome, exactly as in the engine.
        weights = self.genomes[:, 12:]
        return forward(sensors, weights)


class RandomController:
    """Uniform outputs in ``[0, 1]`` — the "no information" reference.

    Uses its own PCG64 stream so it cannot perturb the engine's RNG. A fixed
    seed makes the comparison reproducible.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(int(seed))

    def act(self, sensors: np.ndarray, phen: np.ndarray) -> np.ndarray:
        return self.rng.random((np.atleast_2d(sensors).shape[0], 8))


class PassiveController:
    """All zero outputs: no movement, no feeding, no reproduction.

    This is the "does anything at all help?" floor. Zero output means throttle
    zero, every threshold false, and ``o7 = 0`` so the organism is not resting
    either — it simply does nothing and starves.
    """

    name = "passive"

    def act(self, sensors: np.ndarray, phen: np.ndarray) -> np.ndarray:
        return np.zeros((np.atleast_2d(sensors).shape[0], 8))


class RuleController:
    """Deterministic reflex controller driven only by the sensor vector.

    Rules, in priority order (all thresholds are on *observations*):

    * if a threat is in the cone (sensors 2/3 set), turn away from it and sprint;
    * else if food is in the cone (0/1 set), turn toward it, move and eat;
    * else if a mate is in the cone and energy is high, reproduce;
    * otherwise wander forward slowly.

    It is deliberately simple and readable: it is the "does a hand-written
    reflex beat evolution?" reference. It still goes through ``decode``, so it
    pays the same movement and feeding costs as the neural controller.
    """

    name = "rule"

    _THREAT = 2
    _FOOD = 0
    _MATE = 6
    _ENERGY = 7

    def act(self, sensors: np.ndarray, phen: np.ndarray) -> np.ndarray:
        sensors = np.atleast_2d(np.asarray(sensors, dtype=np.float64))
        n = sensors.shape[0]
        out = np.zeros((n, 8), dtype=np.float64)

        food_seen = sensors[:, self._FOOD] >= 0.0
        threat_seen = sensors[:, self._THREAT] >= 0.0
        mate_seen = sensors[:, self._MATE] >= 0.0
        energy = sensors[:, self._ENERGY]

        # Bearing channel is the neighbour of the detection channel: 1 for food,
        # 3 for threat. Positive bearing means the target is to the right.
        food_bearing = sensors[:, 1]
        threat_bearing = sensors[:, 3]

        # Wander forward by default.
        out[:, 0] = 0.6      # |forward| magnitude
        out[:, 2] = 1.0      # always-on movement gate

        # Turn toward food, away from threats. Turn is o1 and is signed.
        out[food_seen, 1] = np.clip(2.0 * food_bearing[food_seen], -1.0, 1.0)
        out[threat_seen, 1] = np.clip(-2.0 * threat_bearing[threat_seen], -1.0, 1.0)

        # Eat anything in reach when it is in the cone.
        out[food_seen, 3] = 1.0

        # Flee faster when a threat is close.
        out[threat_seen, 5] = 1.0

        # Reproduce only when there is a mate and reserves are healthy.
        out[mate_seen & (energy > 0.5), 6] = 1.0

        # Stop moving and resting is never useful for this controller.
        return out


class ForageController:
    """A hungrier, faster heuristic: sprint toward food, eat, breed when full.

    Differs from :class:`RuleController` in that movement effort scales with
    hunger (energy sensor) and it reproduces at a lower reserve threshold — a
    plausible "optimised" hand-written policy, used to check that the evolved
    controller is not merely beating a strawman reflex.
    """

    name = "forage"

    def act(self, sensors: np.ndarray, phen: np.ndarray) -> np.ndarray:
        sensors = np.atleast_2d(np.asarray(sensors, dtype=np.float64))
        n = sensors.shape[0]
        out = np.zeros((n, 8), dtype=np.float64)

        food_seen = sensors[:, 0] >= 0.0
        threat_seen = sensors[:, 2] >= 0.0
        mate_seen = sensors[:, 6] >= 0.0
        energy = sensors[:, 7]

        # Hungrier organisms try harder (energy sensor is in [-1, 1]).
        hunger = np.clip(1.0 - energy, 0.0, 1.0)
        out[:, 0] = 0.5 + 0.5 * hunger
        out[:, 2] = 1.0

        out[food_seen, 1] = np.clip(2.5 * sensors[food_seen, 1], -1.0, 1.0)
        out[food_seen, 3] = 1.0
        # Sprint toward food when hungry.
        out[food_seen & (hunger > 0.5), 5] = 1.0

        out[threat_seen, 1] = np.clip(-2.5 * sensors[threat_seen, 3], -1.0, 1.0)
        out[threat_seen, 5] = 1.0

        out[mate_seen & (energy > 0.3), 6] = 1.0
        return out


#: Built-in controllers selectable by name. This is a closed registry — no
#: arbitrary external code is ever imported or executed inside the simulation.
CONTROLLERS = {
    "random": RandomController,
    "passive": PassiveController,
    "rule": RuleController,
    "forage": ForageController,
}


def build_controller(kind: str, genomes: np.ndarray | None = None,
                     seed: int = 0) -> Controller:
    """Construct a controller by name.

    ``neural`` needs the genome batch (it *is* the evolved controller); the
    baselines ignore it. Unknown names raise ``ValueError`` rather than falling
    back silently to the neural network.
    """
    kind = str(kind).lower()
    if kind == "neural":
        if genomes is None:
            raise ValueError("the neural controller needs a genome batch")
        return NeuralController(genomes)
    if kind in CONTROLLERS:
        factory = CONTROLLERS[kind]
        return factory(seed) if factory is RandomController else factory()
    raise ValueError(
        f"unknown controller {kind!r}; choose from "
        f"{['neural', *sorted(CONTROLLERS)]}")


def controller_names() -> list[str]:
    """All selectable controller names, neural first."""
    return ["neural", *sorted(CONTROLLERS)]
