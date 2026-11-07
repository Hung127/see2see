"""The staged tick pipeline (DESIGN §7).

Stages that only read state (sensing, brains, action decoding) write proposals;
stages that write (movement, feeding, metabolism, mortality, decomposition) apply
them. No organism observes another's post-move position.

Deterministic draw order (part of the reproducibility contract):

1. construction — terrain noise, founder genomes, founder positions/headings,
   plant positions (in that order);
2. each tick — rainfall, germination picks/positions, then senescence survival
   draws (only when some organism is past its lifespan);
3. births add mutation draws in ascending parent-ID order (Task 11).

Mass/energy accounting follows :class:`evolve.energy.Ledger`: organic stocks are
``Σ(body + E) + plants + detritus``; soil is the inorganic pool outside that sum.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .actions import Actions, decode
from .brain import N_PARAMS, forward
from .carcasses import Carcasses
from .config import Config
from .energy import Ledger, assimilation, basal_cost, metabolic_cost, strike_damage
from .events import Events
from .fields import Fields
from .genome import N_TRAITS, crossover, founders, mutate, phenotype
from .organisms import Population
from .plants import Plants
from .rng import Rng
from .sensors import collect
from .spatial import SpatialGrid, best_per_query as _best_per_query
from .world import World

_EPS = 1e-12

#: The only ways an organism can die. Used for cumulative per-cause counters.
DEATH_CAUSES = ("predation", "starvation", "senescence", "health")

#: Age-at-death histogram: fixed bins, so the survival curve costs O(1) memory.
#: Bin ``i`` counts deaths at ages ``[i*width, (i+1)*width)``; the final bin also
#: collects everything older than the range.
DEATH_AGE_BIN = 25
DEATH_AGE_BINS = 40


class Engine:
    """One deterministic simulation instance."""

    def __init__(self, config: Config, db_path: str | None = None,
                 debug: bool = False) -> None:
        self.cfg = config
        self.db_path = db_path
        self.debug = bool(debug)
        self.rng = Rng(config.seed)
        self.world = World(config.world)
        self.fields = Fields(config.world, config.environment, self.rng)
        self.fields.step(0)
        self.events = Events(config.environment, self.fields, self.rng)
        self.population = Population(config.organisms)
        self.plants = Plants(config.plants, config.world)
        self.carcasses = Carcasses()
        # victim organism id -> [(attacker id, credited damage), ...] (max 8)
        self._damage_credit: dict[int, list[tuple[int, float]]] = {}
        self.ledger = Ledger()
        self.detritus = 0.0
        self.births_total = 0
        self.deaths_total = 0
        self.mutations_total = 0
        # Deaths by cause, cumulative. This is a *derived statistic*, not state:
        # nothing in the tick reads it, so it is snapshotted (so statistics
        # survive a resume) but deliberately excluded from ``checksum()``, whose
        # contract is "arrays that can influence future dynamics".
        self.death_counts: dict[str, int] = {cause: 0 for cause in DEATH_CAUSES}
        # Age-at-death histogram (bounded memory; see DEATH_AGE_BIN). Statistics
        # only — excluded from checksum for the same reason as death_counts.
        self.death_age_hist = np.zeros(DEATH_AGE_BINS, dtype=np.int64)
        self.tick = 0
        self.stats = None  # wired to StatsRecorder in Task 18
        # Optional bounded ancestry log (see stats.LineageRecorder). ``None`` by
        # default so the tick pays nothing unless a caller asks for lineage.
        self.lineage = None
        # Optional non-neural controller (see controllers.py). ``None`` means the
        # evolved MLP decides, which is the default and the only mode the model
        # itself uses; baselines exist for evaluation runs only.
        self.controller = None
        self.last_sensors = np.zeros((0, 16))
        self.last_outputs = np.zeros((0, 8))
        self.last_sensor_ids = np.zeros(0, dtype=np.int64)
        self._phen = np.zeros((0, 12))
        self._speed = np.zeros(0)
        self._seed()
        self.initial_mass = self.mass_stock()
        self.initial_energy = self.energy_stock()
        self.initial_soil = self.soil_stock()
        self.initial_total = self.initial_mass + self.initial_soil
        self._update_density_fields()

    # -- construction ---------------------------------------------------------
    def _seed(self) -> None:
        n = int(self.cfg.organisms.initial)
        genes = founders(n, self.rng)
        xs = self.rng.random(n) * self.world.width
        ys = self.rng.random(n) * self.world.height
        headings = self.rng.random(n) * 2.0 * np.pi - np.pi
        phen = phenotype(genes)
        emax = 10.0 * phen[:, 0] ** 0.9
        for i in range(n):
            slot = self.population.add(genes[i], float(xs[i]), float(ys[i]),
                                       energy=0.5 * float(emax[i]))
            if slot < 0:
                break
            self.population.heading[slot] = float(headings[i])
        self.plants.seed_initial(int(self.cfg.plants.initial), self.rng)

    # -- stocks ---------------------------------------------------------------
    def mass_stock(self) -> float:
        """Organic mass: organisms (body + reserves), plants, carcasses, detritus."""
        pop = self.population
        idx = np.flatnonzero(pop.alive)
        return float(pop.body[idx].sum() + pop.energy[idx].sum()
                     + self.plants.total_biomass() + self.carcasses.total_biomass()
                     + self.detritus)

    def energy_stock(self) -> float:
        """Chemical energy: living tissue and reserves carry 1:1, detritus too."""
        pop = self.population
        idx = np.flatnonzero(pop.alive)
        return float(pop.body[idx].sum() + pop.energy[idx].sum()
                     + self.plants.total_biomass() + self.carcasses.total_biomass()
                     + self.detritus)

    def soil_stock(self) -> float:
        """Inorganic nutrient pool (mass only, no chemical energy)."""
        return float(self.fields.soil.sum())

    def total_mass(self) -> float:
        """Full-system mass: organic stock + soil (DESIGN §6.1)."""
        return self.mass_stock() + self.soil_stock()

    # -- tick -----------------------------------------------------------------
    def step(self, *, reproduce: bool = True, predation: bool = True) -> None:
        """One tick.

        ``reproduce`` and ``predation`` exist only as experiment controls (the
        assay arena removes both processes); their defaults are the full model.
        They are not behavioural hooks: no organism decision depends on them.
        """
        self.tick += 1
        self.fields.step(self.tick)               # 2 environment
        self.events.update(self.tick, self)       # 2b events (physical effects only)
        self._plants_update()                     # 3 plants
        self.population.compact()                 # 4 dense slots
        self.population.sort_by_id()              # 4 deterministic tie order
        actions = self._think()                   # 4-7 sense / brain / decode
        if actions is not None and self.population.n_alive:
            self._move(actions)                   # 8
            self._feed(actions)                   # 9
            if predation:
                self._predation(actions)          # 10
            self._metabolise()                    # 11
            if reproduce:
                self._reproduce(actions)          # 12
            self._mortality()                     # 13
        self._decompose()                         # 14
        self.population.compact()
        self.plants.remove_dead()
        self.carcasses.remove_dead()
        self._update_density_fields()             # 16 (density fields for viz)
        if self.debug and self.tick % max(1, int(self.cfg.stats.interval)) == 0:
            self.ledger.assert_closed(self.initial_mass, self.mass_stock(),
                                      self.initial_energy, self.energy_stock())
            self.ledger.assert_full_closed(self.initial_total, self.total_mass())

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    def checksum(self) -> str:
        """sha256 over the full state, floats rounded to 1e-9 (DESIGN §16.6).

        Covers every array that can influence future dynamics: organisms,
        plants, carcasses, fields, the ledger counters, the scalar counters,
        the damage-credit map and the RNG state. Two engines with equal
        checksums are exchangeable: resuming one and continuing the other
        produce identical futures.
        """
        digest = hashlib.sha256()

        def feed(name: str, value) -> None:
            arr = np.asarray(value)
            if arr.dtype == np.float64:
                arr = np.round(arr, 9)
            arr = np.ascontiguousarray(arr)
            digest.update(name.encode("utf-8"))
            digest.update(str(arr.dtype).encode("utf-8"))
            digest.update(str(arr.shape).encode("utf-8"))
            digest.update(arr.tobytes())

        pop = self.population
        for name in ("id", "x", "y", "heading", "energy", "body", "health",
                     "age", "alive", "generation", "genome"):
            feed(f"pop_{name}", getattr(pop, name))
        feed("pop__next_id", np.int64(pop._next_id))
        for prefix, obj in (("plants", self.plants), ("carc", self.carcasses)):
            for name in ("id", "x", "y", "biomass", "age", "alive"):
                feed(f"{prefix}_{name}", getattr(obj, name))
            feed(f"{prefix}__next_id", np.int64(obj._next_id))
        for name in ("terrain", "soil", "temperature", "moisture", "pheromone",
                     "plant_density", "organism_density", "_growth_factor",
                     "water_stress"):
            feed(f"fields_{name}", getattr(self.fields, name))
        for name in ("tick", "births_total", "deaths_total", "mutations_total"):
            feed(name, np.int64(getattr(self, name)))
        for name in ("detritus", "initial_mass", "initial_energy",
                     "initial_soil", "initial_total"):
            feed(name, np.float64(getattr(self, name)))
        for name in ("light_in", "heat_out", "mass_to_soil", "mass_from_soil",
                     "mass_in"):
            feed(f"ledger_{name}", np.float64(getattr(self.ledger, name)))
        credit = [(victim, attacker, amount)
                  for victim, hits in sorted(self._damage_credit.items())
                  for attacker, amount in hits]
        feed("damage_credit", np.asarray(credit, dtype=np.float64)
             if credit else np.zeros((0, 3)))
        feed("rng_state", self.rng.state())
        return digest.hexdigest()

    # -- stage 3: plants ------------------------------------------------------
    def _plants_update(self) -> None:
        self.plants.grow(self.fields, self.ledger)
        self.plants.germinate(self.rng, self.fields, self.ledger)
        self.detritus += self.plants.senesce()

    # -- stages 4-7: sense, think, decode -------------------------------------
    def _refresh_phen(self) -> np.ndarray:
        """Phenotype cache for the current dense slot set.

        The pipeline refreshes it in ``_think``; direct stage calls (tests,
        scripted experiments) may arrive with a stale cache after births.
        """
        if self._phen.shape[0] != self.population.alive.size:
            self._phen = phenotype(self.population.genome)
        return self._phen

    def _think(self) -> Actions | None:
        if self.population.n_alive == 0:
            return None
        self.last_sensors = collect(self.population, self.plants, self.fields,
                                    self.world, self.cfg.organisms,
                                    carcasses=self.carcasses, grid=None)
        # Row identity: slots are compacted and reordered *after* thinking, so
        # keeping the ids lets the inspector map a vector to the right organism.
        self.last_sensor_ids = self.population.id[self.population.alive].copy()
        self._phen = phenotype(self.population.genome)
        if self.controller is None:
            # Default path: the evolved MLP. Kept as the direct call so a default
            # run is the same computation it was before controllers existed.
            weights = self.population.genome[:, N_TRAITS:]
            self.last_outputs = forward(self.last_sensors, weights)
        else:
            self.last_outputs = np.asarray(
                self.controller.act(self.last_sensors, self._phen),
                dtype=np.float64)
        return decode(self.last_outputs, self._phen)

    # -- stage 8: movement ----------------------------------------------------
    def _move(self, actions: Actions) -> None:
        pop = self.population
        heading = (pop.heading + actions.turn + np.pi) % (2.0 * np.pi) - np.pi
        speed = self._phen[:, 1] * actions.throttle * actions.sprint
        x, y = self.world.apply(pop.x + speed * np.cos(heading),
                                pop.y + speed * np.sin(heading))
        pop.x[:] = x
        pop.y[:] = y
        pop.heading[:] = heading
        pop.dist_moved += speed
        pop.rest_ticks += actions.rest
        self._speed = speed

    # -- stage 9: feeding -----------------------------------------------------
    def _feed(self, actions: Actions) -> None:
        """One bite per eater at the nearest plant *or* carcass within reach.

        Contact, not vision, decides what can be bitten (DESIGN §6.3). Both
        resources compete for the same reach, so a carcass can block a plant and
        vice versa; the nearest wins, ties broken by resource kind then ID.
        """
        eaters = np.flatnonzero(actions.eat)
        if eaters.size == 0:
            return
        pop = self.population
        phen = self._refresh_phen()
        mass = phen[:, 0]
        emax = 10.0 * mass ** 0.9
        bite_rate = self.cfg.energy.bite * mass ** 0.75
        reach = 1.5 * mass ** (1.0 / 3.0) + 2.0

        p_idx = np.flatnonzero(self.plants.alive)
        c_idx = np.flatnonzero(self.carcasses.alive)
        if p_idx.size + c_idx.size == 0:
            return
        rx = np.concatenate([self.plants.x[p_idx], self.carcasses.x[c_idx]])
        ry = np.concatenate([self.plants.y[p_idx], self.carcasses.y[c_idx]])
        # 0 = plant, 1 = carcass; keeps the two ID spaces apart in tie-breaks.
        kind = np.concatenate([np.zeros(p_idx.size, dtype=np.int8),
                               np.ones(c_idx.size, dtype=np.int8)])
        rid = np.concatenate([self.plants.id[p_idx], self.carcasses.id[c_idx]])
        supply = np.concatenate([self.plants.biomass[p_idx],
                                 self.carcasses.biomass[c_idx]])

        grid = SpatialGrid(self.world, cell=80.0)
        grid.build(rx, ry)
        # Query only the eaters. Building candidates for the whole population and
        # then discarding ~most of them cost the same as the sensing grid, and
        # the eater list is ascending, so the per-row order (and therefore the
        # tie-break below) is identical to the previous full-population query.
        rep, tgt, dist = grid.candidates(pop.x[eaters], pop.y[eaters],
                                         radius_cells=2)
        rep = eaters[rep]
        keep = dist <= reach[rep]
        rep, tgt, dist = rep[keep], tgt[keep], dist[keep]
        if rep.size == 0:
            return

        # Nearest resource per eater; ties by (kind, resource ID). Same winners
        # as a full lexsort, without sorting every candidate (see
        # ``spatial.best_per_query``).
        rows, pick = _best_per_query(rep, dist, (kind[tgt], rid[tgt]),
                                     pop.alive.size)
        target = tgt[pick]

        demand = np.minimum.reduce([
            bite_rate[rows],
            supply[target],
            np.maximum(emax[rows] - pop.energy[rows], 0.0),
        ])
        active = demand > 0.0
        rows, target, demand = rows[active], target[active], demand[active]
        if rows.size == 0:
            return

        # Contested resources: proportional shares, exact remainder to the
        # lowest eater ID, so the resource stock closes to rtol 1e-9.
        total = np.bincount(target, weights=demand, minlength=rx.size)
        limited = total > supply
        scale = np.ones(rows.size)
        limited_rows = limited[target]
        if np.any(limited_rows):
            scale[limited_rows] = supply[target[limited_rows]] / total[target[limited_rows]]
        alloc = demand * scale
        alloc_by_res = np.bincount(target, weights=alloc, minlength=rx.size)
        residual = np.where(limited, supply - alloc_by_res, 0.0)
        if np.any(limited):
            res_order = np.argsort(target, kind="stable")
            sorted_target = target[res_order]
            first_in_res = np.ones(sorted_target.size, dtype=bool)
            first_in_res[1:] = sorted_target[1:] != sorted_target[:-1]
            lowest = res_order[first_in_res]
            alloc[lowest] += residual[target[lowest]]

        is_plant = kind[target] == 0
        eta = np.empty(rows.size)
        eta[is_plant] = assimilation(self.cfg.energy.eta_p, phen[rows[is_plant], 9],
                                     phen[rows[is_plant], 4], plant=True)
        eta[~is_plant] = assimilation(self.cfg.energy.eta_m, phen[rows[~is_plant], 9],
                                      phen[rows[~is_plant], 4], plant=False)
        gain = eta * alloc
        undigested = alloc - gain
        pop.energy[rows] += gain
        self.detritus += float(undigested.sum())

        # `target` may repeat (several eaters on one resource), so accumulate
        # the take per resource, then subtract each resource exactly once.
        taken = np.bincount(target, weights=alloc, minlength=rx.size)
        n_plant_res = p_idx.size
        if n_plant_res:
            self.plants.biomass[p_idx] -= taken[:n_plant_res]
            pop.plant_intake[rows[is_plant]] += alloc[is_plant]
        if c_idx.size:
            self.carcasses.biomass[c_idx] -= taken[n_plant_res:]
            pop.meat_intake[rows[~is_plant]] += alloc[~is_plant]
        pop.bites[rows] += 1.0

        dead = self.plants.biomass <= _EPS
        if np.any(dead):
            self.plants.biomass[dead] = 0.0
            self.plants.alive[dead] = False
        dead_c = self.carcasses.biomass <= _EPS
        if np.any(dead_c):
            self.carcasses.biomass[dead_c] = 0.0
            self.carcasses.alive[dead_c] = False

    def _record_deaths(self, idx: np.ndarray,
                       causes: "np.ndarray | str | None" = None) -> None:
        """Record deaths for statistics (histogram + optional lineage log).

        Called from the two stages that kill organisms, *after* the cause is
        known but before the slot is recycled. ``idx`` are slot indices whose
        ``age`` is still valid (``compact()`` runs later in the tick).
        ``causes`` is either a per-index array of strings or one string for all.
        """
        if idx.size == 0:
            return
        pop = self.population
        bins = np.minimum(pop.age[idx] // DEATH_AGE_BIN,
                          DEATH_AGE_BINS - 1)
        self.death_age_hist += np.bincount(bins, minlength=DEATH_AGE_BINS)
        if self.lineage is not None:
            if causes is None:
                cause_list = [str(pop.death_cause[int(i)]) for i in idx]
            elif isinstance(causes, str):
                cause_list = [causes] * idx.size
            else:
                cause_list = [str(c) for c in causes]
            for i, cause in zip(idx, cause_list):
                i = int(i)
                self.lineage.record_death(int(pop.id[i]), self.tick,
                                          cause, int(pop.age[i]))

    # -- stage 10: predation --------------------------------------------------
    def _predation(self, actions: Actions) -> None:
        """Strike the nearest organism in reach; kill credit is damage-weighted.

        All strikes in a tick are resolved simultaneously against the health
        values at the start of the stage: a victim killed by one attacker can
        still be hit by another, so the meat split is order-independent.
        """
        pop = self.population
        attackers = np.flatnonzero(actions.attack & pop.alive)
        if attackers.size == 0:
            return
        phen = self._refresh_phen()
        mass = phen[:, 0]
        reach = 0.5 * mass ** (1.0 / 3.0) + 2.0

        grid = SpatialGrid(self.world, cell=80.0)
        grid.build(pop.x, pop.y)
        # Query only the attackers, as in ``_feed``: the discarded rows were the
        # majority of the population, and ``attackers`` is ascending so each
        # attacker's candidate order — hence the ID tie-break below — is unchanged.
        rep, tgt, dist = grid.candidates(pop.x[attackers], pop.y[attackers],
                                         radius_cells=2)
        rep = attackers[rep]
        keep = (tgt != rep) & (dist <= reach[rep])
        rep, tgt, dist = rep[keep], tgt[keep], dist[keep]
        if rep.size == 0:
            return

        # Nearest target per attacker; ties by lowest target ID. Single pass, no
        # full sort — ``best_per_query`` is the same winner rule as the lexsort.
        rows, pick = _best_per_query(rep, dist, (pop.id[tgt],), pop.alive.size)
        target = tgt[pick]

        # Strike cost: 0.05 * attack per actual strike (no target -> no attempt).
        cost = np.minimum(self.cfg.energy.strike_cost * phen[rows, 5],
                          pop.energy[rows])
        pop.energy[rows] -= cost
        aix, aiy = self.fields.cell_index(pop.x[rows], pop.y[rows])
        np.add.at(self.fields.soil, (aix, aiy), cost)
        paid = float(cost.sum())
        self.ledger.mass_to_soil += paid
        self.ledger.heat_out += paid
        pop.hunts[rows] += 1.0
        self.fields.add_pheromone(pop.x[rows], pop.y[rows], 0.05)

        damage = strike_damage(phen[rows, 5], mass[rows],
                               phen[target, 6], mass[target], self.cfg.energy)
        np.add.at(pop.health, target, -damage)
        pop.health[pop.health < 0.0] = 0.0
        np.add.at(pop.damage_taken, target, damage)  # victims may be hit twice

        # Credit is cumulative per victim and capped at 8 attackers.
        for a, v, d in zip(rows, target, damage):
            if d <= 0.0:
                continue
            vid = int(pop.id[v])
            aid = int(pop.id[a])
            entries = self._damage_credit.setdefault(vid, [])
            for k, (existing, _) in enumerate(entries):
                if existing == aid:
                    entries[k] = (aid, entries[k][1] + float(d))
                    break
            else:
                entries.append((aid, float(d)))
                if len(entries) > 8:
                    entries.pop(0)  # oldest entry dropped (truncated credit)

        dead = np.flatnonzero(pop.alive & (pop.health <= 0.0))
        if dead.size == 0:
            return
        self._record_deaths(dead, "predation")
        killed = {int(pop.id[v]) for v in dead}
        slot_by_id = {int(pop.id[i]): i for i in np.flatnonzero(pop.alive)}
        for v in dead:
            vid = int(pop.id[v])
            total = float(pop.body[v] + pop.energy[v])
            entries = self._damage_credit.pop(vid, [])
            credit_total = sum(d for _, d in entries)
            gained = 0.0
            if total > 0.0 and entries and credit_total > 0.0:
                for aid, dmg in entries:
                    a = slot_by_id.get(aid)
                    if a is None or not pop.alive[a]:
                        continue  # attacker died this tick: its share is carrion
                    eta = assimilation(self.cfg.energy.eta_m, phen[a, 9], phen[a, 4],
                                       plant=False)
                    raw = (dmg / credit_total) * total  # meat apportioned by credit
                    emax = 10.0 * mass[a] ** 0.9
                    gain = min(float(eta) * raw,
                               max(emax - float(pop.energy[a]), 0.0))
                    if gain <= 0.0:
                        continue
                    pop.energy[a] += gain
                    pop.meat_intake[a] += gain
                    gained += gain
            leftover = total - gained
            if leftover > _EPS:
                self.carcasses.add(float(pop.x[v]), float(pop.y[v]), leftover)
            pop.body[v] = 0.0
            pop.energy[v] = 0.0
            pop.alive[v] = False
            pop.death_cause[v] = "predation"
            self.deaths_total += 1
            self.death_counts["predation"] += 1

    # -- stage 11: metabolism -------------------------------------------------
    def _metabolise(self) -> None:
        pop = self.population
        phen = self._refresh_phen()
        env = self.fields.sample(pop.x, pop.y)
        temp = env["temperature"]
        basal = basal_cost(phen, self.cfg.energy)
        # Drought water stress (DESIGN §3.3): extra cost = k * C_basal * m.
        cost = metabolic_cost(phen, self._speed, temp, self.cfg.energy) \
            + basal * self.cfg.energy.water_stress_cost * env["water_stress"]

        paid = np.minimum(cost, pop.energy)
        pop.energy -= paid
        # Respired matter returns to the soil pool at the organism's cell: the
        # ledger counter and the physical pool must move together (DESIGN §6.1).
        cix, ciy = self.fields.cell_index(pop.x, pop.y)
        np.add.at(self.fields.soil, (cix, ciy), paid)
        total_paid = float(paid.sum())
        self.ledger.mass_to_soil += total_paid
        self.ledger.heat_out += total_paid

        deficit = cost - paid
        starving = deficit > 0.0
        if np.any(starving):
            burn = np.minimum(self.cfg.energy.catabolism, pop.body[starving])
            pop.body[starving] -= burn
            np.add.at(self.fields.soil, (cix[starving], ciy[starving]), burn)
            burned = float(burn.sum())
            self.ledger.mass_to_soil += burned
            self.ledger.heat_out += burned
            pop.health[starving] -= self.cfg.energy.starvation_health_loss
            pop.body[pop.body < _EPS] = 0.0

        regen = (pop.energy > 2.0 * basal) & ~starving
        pop.health = np.minimum(pop.health + regen * self.cfg.organisms.health_regen,
                                self.cfg.organisms.health_max)
        pop.health = np.maximum(pop.health, 0.0)
        pop.age += 1

    # -- stage 12: reproduction ----------------------------------------------
    def _reproduce(self, actions: Actions) -> None:
        pop = self.population
        if pop.n_alive == 0:
            return
        phen = self._phen
        if phen.shape[0] != pop.alive.size:
            phen = phenotype(pop.genome)
            self._phen = phen
        emax = 10.0 * phen[:, 0] ** 0.9
        threshold = phen[:, 7] * emax
        invest = phen[:, 8] * emax
        mature = pop.age >= self.cfg.organisms.maturity_age
        want = np.flatnonzero(actions.reproduce & mature & (pop.energy >= threshold))
        if want.size == 0:
            return
        mode = self.cfg.genetics.mode
        used: set[int] = set()

        for i in want:  # slot order == ascending ID (see sort_by_id)
            if pop.n_alive >= pop.capacity:
                break
            if i in used:
                continue
            pop.repro_attempts[i] += 1.0
            used.add(i)
            mate = (-1 if mode == "asexual"
                    else self._find_mate(i, want, used, emax, threshold))
            if mate >= 0:
                used.add(mate)
                pop.repro_attempts[mate] += 1.0
                child_genome = crossover(pop.genome[i], pop.genome[mate], self.rng)
                child_genome, n_mut = mutate(
                    child_genome[None, :],
                    0.5 * (phen[i, 11] + phen[mate, 11]),
                    self.rng, self.cfg.genetics,
                )
                self.mutations_total += int(n_mut.sum())
                child_genome = child_genome[0]
                generation = max(int(pop.generation[i]), int(pop.generation[mate])) + 1
                parents = (int(pop.id[i]), int(pop.id[mate]))
                total = invest[i]
                share = 0.5 * total
                ok = self._birth(i, child_genome, total, share, generation, parents)
                if ok:
                    pop.energy[mate] -= share
                    pop.repro_success[mate] += 1.0
                continue
            if mode == "sexual":
                continue  # no mate: the attempt fails with nobody paying
            # asexual, or 'both' without an available mate
            child_genome, n_mut = mutate(
                pop.genome[i][None, :], phen[i, 11], self.rng, self.cfg.genetics
            )
            self.mutations_total += int(n_mut.sum())
            self._birth(i, child_genome[0], invest[i], invest[i],
                        int(pop.generation[i]) + 1, (int(pop.id[i]), None))
        # Children changed the row count; refresh the phenotype cache.
        self._phen = phenotype(pop.genome)

    def _find_mate(self, i: int, want: np.ndarray, used: set[int],
                   emax: np.ndarray, threshold: np.ndarray) -> int:
        pop = self.population
        cand = np.array([j for j in want if j != i and j not in used
                         and pop.energy[j] >= 0.5 * threshold[j]], dtype=np.int64)
        if cand.size == 0:
            return -1
        reach = 3.0 * pop.body[i] ** (1.0 / 3.0)
        d = self.world.distance(pop.x[i], pop.y[i], pop.x[cand], pop.y[cand])
        ok = d <= reach
        if not np.any(ok):
            return -1
        good = cand[ok]
        dist = d[ok]
        return int(good[int(np.argmin(dist))])  # ties -> first (lowest ID)

    def _birth(self, parent: int, genome: np.ndarray, total_investment: float,
               payer_share: float, generation: int,
               parents: tuple[int, int | None]) -> bool:
        """Attempt one birth. Gates fail with *nobody paying* (DESIGN §6.5)."""
        pop = self.population
        child_phen = phenotype(genome[None, :])[0]
        body = float(child_phen[0])
        if total_investment <= self.cfg.energy.build_cost * body:
            return False  # cannot build a body of this size
        if pop.energy[parent] < payer_share:
            return False  # cannot afford the share
        reserves = 0.9 * (total_investment - body)
        waste = 0.1 * (total_investment - body)
        slot = pop.add(genome, float(pop.x[parent]), float(pop.y[parent]),
                       energy=reserves, generation=generation, parents=parents)
        if slot < 0:
            return False  # cap reached: parent pays nothing
        pop.energy[parent] -= payer_share
        pop.repro_success[parent] += 1.0
        self.births_total += 1
        if self.lineage is not None:
            # Only new ids are recorded; the founders were never born in a tick,
            # so they have no record and are honestly absent rather than invented.
            self.lineage.record_birth(int(pop.id[slot]), parents[0], parents[1],
                                      self.tick, generation)
        bix, biy = self.fields.cell_index(np.array([pop.x[parent]]),
                                          np.array([pop.y[parent]]))
        np.add.at(self.fields.soil, (bix, biy), waste)
        self.ledger.mass_to_soil += waste
        self.ledger.heat_out += waste
        return True

    # -- stage 13: mortality --------------------------------------------------
    def _mortality(self) -> None:
        pop = self.population
        n = pop.alive.size
        if n == 0:
            return
        phen = self._phen
        health_dead = pop.alive & (pop.health <= 0.0)
        starved = pop.alive & (pop.energy <= _EPS) & (pop.body <= _EPS)

        past = pop.alive & (pop.age > phen[:, 10])
        old_age = np.zeros(n, dtype=bool)
        if np.any(past):
            p = 1.0 - np.exp(-(pop.age[past] - phen[past, 10]) / 200.0)
            draw = self.rng.random(int(past.sum()))
            old_age[past] = draw < p

        dead = health_dead | starved | old_age
        if not np.any(dead):
            return
        cause = np.where(starved[dead], "starvation",
                         np.where(old_age[dead], "senescence", "health"))
        self._record_deaths(np.flatnonzero(dead), cause)
        # Animal deaths already became carcasses in _predation; what is left here
        # (starvation, senescence, residual health loss) is diffuse litter.
        self.detritus += float(pop.body[dead].sum() + pop.energy[dead].sum())
        pop.body[dead] = 0.0
        pop.energy[dead] = 0.0
        pop.alive[dead] = False
        self.deaths_total += int(dead.sum())
        pop.death_cause[dead] = cause
        for name, count in zip(("starvation", "senescence", "health"),
                               (starved[dead].sum(), old_age[dead].sum(),
                                (~starved[dead] & ~old_age[dead]).sum())):
            self.death_counts[name] += int(count)

    # -- stage 14: decomposition ---------------------------------------------
    def _decompose(self) -> None:
        amount = self.cfg.energy.decay * self.detritus
        if amount > 0.0:
            self.detritus -= amount
            # The scalar detritus pool has no position: mineralised mass returns
            # evenly across the grid (coarse-summary ruling, documented).
            self.fields.soil += amount / self.fields.soil.size
            self.ledger.mass_to_soil += amount
            self.ledger.heat_out += amount
        self.carcasses.decay(self.ledger, self.cfg.energy.decay, self.fields)

    # -- derived density fields ----------------------------------------------
    def _update_density_fields(self) -> None:
        n = self.fields.n
        self.fields.plant_density[:] = self.plants.biomass_field(self.fields)
        idx = np.flatnonzero(self.population.alive)
        if idx.size:
            ix, iy = self.fields.cell_index(self.population.x[idx], self.population.y[idx])
            counts = np.bincount(iy * n + ix, minlength=n * n)
        else:
            counts = np.zeros(n * n, dtype=np.int64)
        self.fields.organism_density[:] = counts.reshape(n, n)

    # -- reproducibility ------------------------------------------------------
    def inspect(self, i: int) -> dict:
        """Full behavioural payload for one organism (slot index).

        ``sensors``/``outputs`` are the last vectors computed for *this* id. The
        live slots are compacted and re-sorted after thinking, so rows are
        matched by id, never by position; an organism born or dead since the
        last think has no vector and reports zeros with ``has_last_vectors``
        false rather than another animal's numbers.
        """
        from .brain import split as split_weights

        pop = self.population
        genome = pop.genome[i]
        W1, b1, W2, b2, W3, b3 = split_weights(genome[None, N_TRAITS:])
        parents = (
            int(pop.parent_a[i]) if pop.parent_a[i] >= 0 else None,
            int(pop.parent_b[i]) if pop.parent_b[i] >= 0 else None,
        )
        oid = int(pop.id[i])
        row = None
        if self.last_sensor_ids.size:
            hits = np.flatnonzero(self.last_sensor_ids == oid)
            if hits.size:
                row = int(hits[0])
        return {
            "id": oid,
            "age": int(pop.age[i]),
            "energy": float(pop.energy[i]),
            "body": float(pop.body[i]),
            "health": float(pop.health[i]),
            "generation": int(pop.generation[i]),
            "parents": parents,
            "position": (float(pop.x[i]), float(pop.y[i])),
            "heading": float(pop.heading[i]),
            "phenotypes": phenotype(genome[None, :])[0],
            "weights": {"W1": W1[0], "b1": b1[0], "W2": W2[0], "b2": b2[0],
                        "W3": W3[0], "b3": b3[0]},
            "has_last_vectors": row is not None,
            "sensors": self.last_sensors[row].copy() if row is not None
                       else np.zeros(16),
            "outputs": self.last_outputs[row].copy() if row is not None
                       else np.zeros(8),
            "counters": {name: float(getattr(pop, name)[i])
                         for name in ("dist_moved", "bites", "hunts", "escapes",
                                      "repro_attempts", "repro_success",
                                      "rest_ticks", "plant_intake", "meat_intake",
                                      "damage_taken")},
        }

    def checksum(self) -> str:
        """SHA-256 over rounded state; independent instances with equal state agree."""
        h = hashlib.sha256()
        pop = self.population
        idx = np.flatnonzero(pop.alive)
        pidx = np.flatnonzero(self.plants.alive)
        cidx = np.flatnonzero(self.carcasses.alive)
        parts = [
            np.array([self.tick], dtype=np.int64),
            pop.id[idx], pop.genome[idx],
            np.round(pop.x[idx], 9), np.round(pop.y[idx], 9),
            np.round(pop.heading[idx], 9),
            np.round(pop.energy[idx], 9), np.round(pop.body[idx], 9),
            np.round(pop.health[idx], 9), pop.age[idx], pop.generation[idx],
            self.plants.id[pidx], np.round(self.plants.x[pidx], 9),
            np.round(self.plants.y[pidx], 9), np.round(self.plants.biomass[pidx], 9),
            self.carcasses.id[cidx], np.round(self.carcasses.x[cidx], 9),
            np.round(self.carcasses.y[cidx], 9),
            np.round(self.carcasses.biomass[cidx], 9),
            np.array([self.detritus, self.ledger.light_in, self.ledger.heat_out,
                      self.ledger.mass_to_soil, self.ledger.mass_from_soil,
                      self.ledger.mass_in], dtype=np.float64),
        ]
        for part in parts:
            h.update(np.ascontiguousarray(part).tobytes())
        h.update(self.rng.state())
        return h.hexdigest()
