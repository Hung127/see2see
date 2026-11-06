"""Environmental events (DESIGN §3.3).

An event is ``{kind, start_tick, duration, magnitude, region}`` with region a
disc ``(cx, cy, radius)`` — a huge radius means the whole world. Effects are
*physical only*: they change fields, plants, soil, health or the ledger, and
never a genome (asserted by ``test_events_never_touch_genomes``).

Continuous events (active for ``duration`` ticks, re-applied every tick):

* ``drought``      — moisture ``×(1−m)``, plant growth ``×(1−0.8·m)``, animal
  water stress ``m`` (extra metabolic cost
  ``water_stress_cost·C_basal·m``, paid by the engine's metabolism stage);
* ``heavy_rain``   — moisture ``+m`` (clamped to [0, 1]); organisms in the
  region stand in water of depth ``m·(1−terrain)`` and lose
  ``flood_damage·depth`` health per tick (DESIGN §3.2: terrain height sets
  flood damage, so low ground is hit hardest);
* ``heatwave``     — temperature ``+10·m`` °C;
* ``cold_snap``    — temperature ``−10·m`` °C.

Instant events (fire once, on their start tick):

* ``wildfire``        — plant biomass in region → soil (mass) with an equal
  energy release to heat; organisms in region lose ``20·m`` health;
* ``plant_blight``    — fraction ``m`` of plant biomass in region → detritus
  (killed, not burned: no soil/heat channel);
* ``nutrient_pulse``  — ``m`` bu of nutrient per cell in region added to soil,
  booked as ``ledger.mass_in`` (the one external mass input).

Random events are spawned per tick by a Poisson draw per configured kind, in
config order. Draw order per spawn: centre x, centre y, radius, duration,
magnitude — part of the reproducibility contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnvironmentConfig
from .fields import Fields
from .rng import Rng

KINDS = ("drought", "heavy_rain", "heatwave", "cold_snap", "wildfire",
         "nutrient_pulse", "plant_blight")
CONTINUOUS = ("drought", "heavy_rain", "heatwave", "cold_snap")
INSTANT = ("wildfire", "nutrient_pulse", "plant_blight")

_EPS = 1e-12


@dataclass(frozen=True)
class Event:
    """One environmental event (DESIGN §3.3)."""

    kind: str
    start_tick: int
    duration: int
    magnitude: float
    region: tuple[float, float, float] = (0.0, 0.0, 1.0e9)

    def active_at(self, tick: int) -> bool:
        return self.start_tick <= tick < self.start_tick + max(1, int(self.duration))


class Events:
    """Scheduler and applier for scheduled and random events."""

    def __init__(self, cfg: EnvironmentConfig, fields: Fields, rng: Rng) -> None:
        self.cfg = cfg
        self.fields = fields
        self.rng = rng
        self._events: list[Event] = [
            Event(e.kind, int(e.start_tick), int(e.duration), float(e.magnitude),
                  tuple(e.region))
            for e in cfg.events
        ]
        self._fired: set[int] = set()
        self.log: list[Event] = list(self._events)
        self.random_rates = tuple(cfg.random_events)

    # -- scheduling -----------------------------------------------------------
    def _spawn_random(self, tick: int) -> None:
        for kind, rate in self.random_rates:
            n = int(self.rng.poisson(rate))
            for _ in range(n):
                cx = float(self.rng.random()) * self.fields.cell_size * self.fields.n
                cy = float(self.rng.random()) * self.fields.cell_size * self.fields.n
                radius = 50.0 + float(self.rng.random()) * 200.0
                duration = int(self.rng.integers(1, 51))
                magnitude = float(self.rng.random())
                ev = Event(kind, tick, duration, magnitude, (cx, cy, radius))
                self._events.append(ev)
                self.log.append(ev)

    def active(self, tick: int) -> list[Event]:
        return [ev for ev in self._events if ev.active_at(tick)]

    # -- application ----------------------------------------------------------
    def update(self, tick: int, engine) -> None:
        """Spawn random events, apply continuous modifiers, fire instant ones."""
        self._spawn_random(tick)
        for i, ev in enumerate(self._events):
            if not ev.active_at(tick):
                continue
            if ev.kind in CONTINUOUS:
                self._apply_continuous(ev, engine)
            elif ev.kind in INSTANT and i not in self._fired:
                self._fired.add(i)
                self._apply_instant(ev, engine)

    def _mask(self, region: tuple[float, float, float]) -> np.ndarray:
        cx, cy, r = region
        n = self.fields.n
        if r >= 1e8:
            return np.ones((n, n), dtype=bool)
        xs = (np.arange(n) + 0.5) * self.fields.cell_size
        x = np.broadcast_to(xs[:, None], (n, n))  # axis 0 = ix (x index)
        y = np.broadcast_to(xs[None, :], (n, n))  # axis 1 = iy (y index)
        return ((x - cx) ** 2 + (y - cy) ** 2) <= r * r

    def _apply_continuous(self, ev: Event, engine) -> None:
        mask = self._mask(ev.region)
        m = float(ev.magnitude)
        f = self.fields
        if ev.kind == "drought":
            f.moisture[mask] *= 1.0 - m
            f._growth_factor[mask] *= 1.0 - 0.8 * m
            f.water_stress[mask] += m
        elif ev.kind == "heavy_rain":
            f.moisture[mask] = np.clip(f.moisture[mask] + m, 0.0, 1.0)
            # Standing water: rain depth grows toward low ground (DESIGN §3.2
            # ties flood damage to terrain height), damaging organisms there.
            pop = engine.population
            pidx = np.flatnonzero(pop.alive)
            if pidx.size:
                pix, piy = f.cell_index(pop.x[pidx], pop.y[pidx])
                hit = mask[pix, piy]
                depth = m * (1.0 - f.terrain[pix, piy])
                harm = engine.cfg.environment.flood_damage * depth
                pop.health[pidx[hit]] = np.maximum(
                    pop.health[pidx[hit]] - harm[hit], 0.0)
        elif ev.kind == "heatwave":
            f.temperature[mask] += 10.0 * m
        elif ev.kind == "cold_snap":
            f.temperature[mask] -= 10.0 * m

    def _apply_instant(self, ev: Event, engine) -> None:
        mask = self._mask(ev.region)
        m = float(ev.magnitude)
        f = self.fields
        if ev.kind == "wildfire":
            plants = engine.plants
            idx = np.flatnonzero(plants.alive)
            if idx.size:
                ix, iy = f.cell_index(plants.x[idx], plants.y[idx])
                burning = mask[ix, iy]
                burned = plants.biomass[idx[burning]].copy()
                if burned.size and burned.sum() > 0.0:
                    np.add.at(f.soil, (ix[burning], iy[burning]), burned)
                    plants.biomass[idx[burning]] = 0.0
                    plants.alive[idx[burning]] = False
                    engine.ledger.mass_to_soil += float(burned.sum())
                    engine.ledger.heat_out += float(burned.sum())
            pop = engine.population
            pidx = np.flatnonzero(pop.alive)
            if pidx.size:
                pix, piy = f.cell_index(pop.x[pidx], pop.y[pidx])
                hit = mask[pix, piy]
                pop.health[pidx[hit]] = np.maximum(pop.health[pidx[hit]] - 20.0 * m, 0.0)
        elif ev.kind == "plant_blight":
            plants = engine.plants
            idx = np.flatnonzero(plants.alive)
            if idx.size:
                ix, iy = f.cell_index(plants.x[idx], plants.y[idx])
                hit = mask[ix, iy]
                lost = m * plants.biomass[idx[hit]]
                plants.biomass[idx[hit]] -= lost
                engine.detritus += float(lost.sum())
                dead = plants.biomass <= _EPS
                if np.any(dead):
                    plants.biomass[dead] = 0.0
                    plants.alive[dead] = False
        elif ev.kind == "nutrient_pulse":
            f.soil[mask] += m
            engine.ledger.mass_in += m * int(mask.sum())
