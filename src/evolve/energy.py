"""Metabolic formulas and the dual conservation ledger (DESIGN §6.1–6.4).

Two invariants hold simultaneously:

* **Mass** — organic matter is never created or destroyed, only moved between
  organisms, plants, detritus and the inorganic soil pool::

      final = initial + mass_from_soil - mass_to_soil + mass_in

* **Energy** — light is the only input, heat the only sink::

      final = initial + light_in - heat_out

Detritus carries *both* mass and energy (it is organic matter); soil carries mass
only (it is inorganic nutrient). Every channel that sends organic mass to soil
must release exactly that much chemical energy as heat, which is what makes the
two invariants closable at the same time.
"""
from __future__ import annotations

import numpy as np

from .config import EnergyConfig
from .genome import N_NN

RTOL = 1e-9

DEFAULT_CFG = EnergyConfig()


class Ledger:
    """Counters for the two conservation invariants.

    The engine increments these as channels move matter or energy; tests and
    debug assertions compare stocks against them.
    """

    __slots__ = ("light_in", "heat_out", "mass_to_soil", "mass_from_soil", "mass_in")

    def __init__(self) -> None:
        self.light_in = 0.0
        self.heat_out = 0.0
        self.mass_to_soil = 0.0
        self.mass_from_soil = 0.0
        self.mass_in = 0.0

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__slots__}

    def assert_closed(self, initial_mass: float, final_mass: float,
                      initial_energy: float, final_energy: float) -> None:
        """Assert both invariants to ``rtol = 1e-9`` (DESIGN §6.1)."""
        expected_mass = initial_mass + self.mass_from_soil - self.mass_to_soil + self.mass_in
        expected_energy = initial_energy + self.light_in - self.heat_out
        assert np.isclose(final_mass, expected_mass, rtol=RTOL, atol=0.0), (
            f"mass ledger open: stock {final_mass!r} != expected {expected_mass!r} "
            f"(to_soil={self.mass_to_soil!r}, from_soil={self.mass_from_soil!r}, "
            f"in={self.mass_in!r})"
        )
        assert np.isclose(final_energy, expected_energy, rtol=RTOL, atol=0.0), (
            f"energy ledger open: stock {final_energy!r} != expected {expected_energy!r} "
            f"(light_in={self.light_in!r}, heat_out={self.heat_out!r})"
        )


# ---------------------------------------------------------------------------
# Metabolism
# ---------------------------------------------------------------------------
def metabolic_cost(phen: np.ndarray, speed: np.ndarray, temp: np.ndarray,
                   cfg: EnergyConfig = DEFAULT_CFG) -> np.ndarray:
    """Per-tick metabolic cost, summing every term of DESIGN §6.2.

    ``phen`` columns follow DESIGN §4.1: 0 body size, 2 vision range, 3 vision
    angle, 4 metabolic rate, 5 attack, 6 defence, 9 diet preference,
    10 lifespan. ``speed`` is the *actual* speed this tick (the movement term is
    charged on distance moved, not on the genetic maximum).
    """
    phen = np.atleast_2d(np.asarray(phen, dtype=np.float64))
    speed = np.asarray(speed, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    mass = phen[:, 0]
    rate = phen[:, 4]
    lifespan = phen[:, 10]
    diet = phen[:, 9]

    basal = cfg.k_b * mass ** 0.75 * rate
    repair = cfg.k_r * mass ** 0.75 * np.sqrt(lifespan / 1000.0)
    brain = cfg.k_n * N_NN
    move = cfg.k_m * mass * (speed / 10.0) ** 2
    vision = cfg.k_v * (phen[:, 2] / 100.0) ** 2 * (phen[:, 3] / 360.0)
    attack = cfg.k_a * phen[:, 5]
    defense = cfg.k_d * phen[:, 6]
    enzyme = cfg.k_e * 4.0 * diet * (1.0 - diet)
    thermal = basal * 0.5 * ((temp - 20.0) / 10.0) ** 2
    return basal + repair + brain + move + vision + attack + defense + enzyme + thermal


def basal_cost(phen: np.ndarray, cfg: EnergyConfig = DEFAULT_CFG) -> np.ndarray:
    """``C_basal = k_b · M^0.75 · metabolic_rate`` (the regen gate uses 2·C_basal)."""
    phen = np.atleast_2d(np.asarray(phen, dtype=np.float64))
    return cfg.k_b * phen[:, 0] ** 0.75 * phen[:, 4]


def assimilation(eta_base: float, diet: np.ndarray, rate: np.ndarray,
                 *, plant: bool) -> np.ndarray:
    """Digestive efficiency on one food item (DESIGN §6.3).

    ``diet`` is the organism's ``diet_preference`` (0 = herbivore, 1 = carnivore).
    Plant matter is gated by ``(1 - d) ** 0.5``, meat by ``d ** 0.5``, and both
    scale with ``metabolic_rate ** 0.5`` (metabolic scope). A pure herbivore
    therefore digests meat at exactly zero efficiency, and vice versa.
    """
    diet = np.asarray(diet, dtype=np.float64)
    gate = (1.0 - diet) if plant else diet
    return eta_base * np.sqrt(np.clip(gate, 0.0, 1.0)) * np.sqrt(np.clip(rate, 0.0, None))


# ---------------------------------------------------------------------------
# Matter and energy channels
# ---------------------------------------------------------------------------
def feed(demand: np.ndarray, available: np.ndarray,
         eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One bite from a resource (plant or carrion) at efficiency ``eta``.

    Returns ``(taken, gain, detritus)``: ``taken`` leaves the resource, ``gain``
    becomes reserves, and the undigested ``detritus`` is organic matter — it
    keeps both its mass and its energy, so it must be added back to the detritus
    pool rather than vanished.
    """
    demand = np.asarray(demand, dtype=np.float64)
    available = np.asarray(available, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    taken = np.minimum(demand, available)
    gain = eta * taken
    detritus = taken - gain
    return taken, gain, detritus


def respiration(ledger: Ledger, amount: np.ndarray | float) -> tuple[float, float]:
    """Oxidise ``amount`` bu of reserves/body: mass to soil, equal energy to heat."""
    amount = float(np.sum(amount))
    ledger.mass_to_soil += amount
    ledger.heat_out += amount
    return amount, amount


def strike_damage(attack: np.ndarray, size: np.ndarray,
                  defense: np.ndarray, def_size: np.ndarray,
                  cfg: EnergyConfig = DEFAULT_CFG) -> np.ndarray:
    """Health lost to one strike: ``max(0, D - 0.8·R)`` (DESIGN §6.4)."""
    dmg = cfg.k_dmg * np.asarray(attack) * np.asarray(size) ** 0.4
    mit = cfg.k_mit * np.asarray(defense) * np.asarray(def_size) ** 0.4
    return np.maximum(0.0, dmg - cfg.damage_mit_factor * mit)


def decay_amount(mass: np.ndarray, cfg: EnergyConfig = DEFAULT_CFG) -> np.ndarray:
    """Microbial mineralisation per tick: ``δ = decay_rate · C`` (DESIGN §6.6)."""
    return cfg.decay * np.asarray(mass, dtype=np.float64)
