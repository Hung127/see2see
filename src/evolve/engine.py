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
from .config import Config
from .energy import Ledger, assimilation, basal_cost, metabolic_cost
from .fields import Fields
from .genome import N_TRAITS, crossover, founders, mutate, phenotype
from .organisms import Population
from .plants import Plants
from .rng import Rng
from .sensors import collect
from .spatial import SpatialGrid
from .world import World

_EPS = 1e-12


class Engine:
    """One deterministic simulation instance."""

    def __init__(self, config: Config, db_path: str | None = None) -> None:
        self.cfg = config
        self.db_path = db_path
        self.rng = Rng(config.seed)
        self.world = World(config.world)
        self.fields = Fields(config.world, config.environment, self.rng)
        self.fields.step(0)
        self.population = Population(config.organisms)
        self.plants = Plants(config.plants, config.world)
        self.ledger = Ledger()
        self.detritus = 0.0
        self.tick = 0
        self.stats = None  # wired to StatsRecorder in Task 18
        self.last_sensors = np.zeros((0, 16))
        self.last_outputs = np.zeros((0, 8))
        self._phen = np.zeros((0, 12))
        self._speed = np.zeros(0)
        self._seed()
        self.initial_mass = self.mass_stock()
        self.initial_energy = self.energy_stock()
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
        """Organic mass: organisms (body + reserves), plants, detritus."""
        pop = self.population
        idx = np.flatnonzero(pop.alive)
        return float(pop.body[idx].sum() + pop.energy[idx].sum()
                     + self.plants.total_biomass() + self.detritus)

    def energy_stock(self) -> float:
        """Chemical energy: living tissue and reserves carry 1:1, detritus too."""
        pop = self.population
        idx = np.flatnonzero(pop.alive)
        return float(pop.body[idx].sum() + pop.energy[idx].sum()
                     + self.plants.total_biomass() + self.detritus)

    # -- tick -----------------------------------------------------------------
    def step(self) -> None:
        self.tick += 1
        self.fields.step(self.tick)               # 2 environment
        self._plants_update()                     # 3 plants
        self.population.compact()                 # 4 dense slots
        self.population.sort_by_id()              # 4 deterministic tie order
        actions = self._think()                   # 4-7 sense / brain / decode
        if actions is not None and self.population.n_alive:
            self._move(actions)                   # 8
            self._feed(actions)                   # 9
            self._metabolise()                    # 11
            self._reproduce(actions)              # 12
            self._mortality()                     # 13
        self._decompose()                         # 14
        self.population.compact()
        self.plants.remove_dead()
        self._update_density_fields()             # 16 (density fields for viz)

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    # -- stage 3: plants ------------------------------------------------------
    def _plants_update(self) -> None:
        self.plants.grow(self.fields, self.ledger)
        self.plants.germinate(self.rng, self.fields, self.ledger)
        self.detritus += self.plants.senesce()

    # -- stages 4-7: sense, think, decode -------------------------------------
    def _think(self) -> Actions | None:
        if self.population.n_alive == 0:
            return None
        self.last_sensors = collect(self.population, self.plants, self.fields,
                                    self.world, self.cfg.organisms, grid=None)
        weights = self.population.genome[:, N_TRAITS:]
        self.last_outputs = forward(self.last_sensors, weights)
        self._phen = phenotype(self.population.genome)
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
        eaters = np.flatnonzero(actions.eat)
        p_idx = np.flatnonzero(self.plants.alive)
        if eaters.size == 0 or p_idx.size == 0:
            return
        pop = self.population
        phen = self._phen
        mass = phen[:, 0]
        emax = 10.0 * mass ** 0.9
        bite_rate = self.cfg.energy.bite * mass ** 0.75
        reach = 1.5 * mass ** (1.0 / 3.0) + 2.0

        grid = SpatialGrid(self.world, cell=80.0)
        grid.build(self.plants.x[p_idx], self.plants.y[p_idx])
        rep, tgt, dist = grid.candidates(pop.x, pop.y, radius_cells=2)
        keep = actions.eat[rep] & (dist <= reach[rep])
        rep, tgt, dist = rep[keep], tgt[keep], dist[keep]
        if rep.size == 0:
            return

        # Nearest plant per eater, ties by lowest plant ID.
        plant_slot = p_idx[tgt]
        plant_id = self.plants.id[plant_slot]
        order = np.lexsort((plant_id, dist, rep))
        rep, plant_slot = rep[order], plant_slot[order]
        first = np.ones(rep.size, dtype=bool)
        first[1:] = rep[1:] != rep[:-1]
        rows = rep[first]
        target = plant_slot[first]

        demand = np.minimum.reduce([
            bite_rate[rows],
            self.plants.biomass[target],
            np.maximum(emax[rows] - pop.energy[rows], 0.0),
        ])
        active = demand > 0.0
        rows, target, demand = rows[active], target[active], demand[active]
        if rows.size == 0:
            return

        # Contested plants: proportional shares, exact remainder to lowest eater ID.
        total = np.bincount(target, weights=demand, minlength=self.plants.alive.size)
        supply = self.plants.biomass
        limited = total > supply
        scale = np.ones(rows.size)
        limited_rows = limited[target]
        if np.any(limited_rows):
            scale[limited_rows] = supply[target[limited_rows]] / total[target[limited_rows]]
        alloc = demand * scale
        alloc_by_plant = np.bincount(target, weights=alloc,
                                     minlength=self.plants.alive.size)
        residual = np.where(limited, supply - alloc_by_plant, 0.0)
        if np.any(limited):
            plant_order = np.argsort(target, kind="stable")
            sorted_target = target[plant_order]
            first_in_plant = np.ones(sorted_target.size, dtype=bool)
            first_in_plant[1:] = sorted_target[1:] != sorted_target[:-1]
            lowest = plant_order[first_in_plant]
            alloc[lowest] += residual[target[lowest]]

        eta = assimilation(self.cfg.energy.eta_p, phen[rows, 9], phen[rows, 4], plant=True)
        gain = eta * alloc
        undigested = alloc - gain
        pop.energy[rows] += gain
        # `target` may repeat (several eaters on one plant), so accumulate with
        # bincount: fancy-index `-=` would apply only the last write per plant.
        taken_by_plant = np.bincount(target, weights=alloc,
                                     minlength=self.plants.alive.size)
        self.plants.biomass -= taken_by_plant
        self.detritus += float(undigested.sum())
        pop.plant_intake[rows] += alloc
        pop.bites[rows] += 1.0
        dead = self.plants.biomass <= _EPS
        if np.any(dead):
            self.plants.biomass[dead] = 0.0
            self.plants.alive[dead] = False

    # -- stage 11: metabolism -------------------------------------------------
    def _metabolise(self) -> None:
        pop = self.population
        phen = self._phen
        temp = self.fields.sample(pop.x, pop.y)["temperature"]
        basal = basal_cost(phen, self.cfg.energy)
        cost = metabolic_cost(phen, self._speed, temp, self.cfg.energy)

        paid = np.minimum(cost, pop.energy)
        pop.energy -= paid
        total_paid = float(paid.sum())
        self.ledger.mass_to_soil += total_paid
        self.ledger.heat_out += total_paid

        deficit = cost - paid
        starving = deficit > 0.0
        if np.any(starving):
            burn = np.minimum(self.cfg.energy.catabolism, pop.body[starving])
            pop.body[starving] -= burn
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
        sexual = self.cfg.genetics.mode == "sexual"
        used: set[int] = set()

        for i in want:  # slot order == ascending ID (see sort_by_id)
            if pop.n_alive >= pop.capacity:
                break
            if i in used:
                continue
            pop.repro_attempts[i] += 1.0
            used.add(i)
            if sexual:
                mate = self._find_mate(i, want, used, emax, threshold)
                if mate < 0:
                    continue
                used.add(mate)
                pop.repro_attempts[mate] += 1.0
                child_genome = crossover(pop.genome[i], pop.genome[mate], self.rng)
                child_genome, _ = mutate(
                    child_genome[None, :],
                    0.5 * (phen[i, 11] + phen[mate, 11]),
                    self.rng, self.cfg.genetics,
                )
                child_genome = child_genome[0]
                generation = max(int(pop.generation[i]), int(pop.generation[mate])) + 1
                parents = (int(pop.id[i]), int(pop.id[mate]))
                total = invest[i]
                share = 0.5 * total
                ok = self._birth(i, child_genome, total, share, generation, parents)
                if ok:
                    pop.energy[mate] -= share
                    pop.repro_success[mate] += 1.0
            else:
                child_genome, _ = mutate(
                    pop.genome[i][None, :], phen[i, 11], self.rng, self.cfg.genetics
                )
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
        health_dead = pop.health <= 0.0
        starved = (pop.energy <= _EPS) & (pop.body <= _EPS)

        past = pop.age > phen[:, 10]
        old_age = np.zeros(n, dtype=bool)
        if np.any(past):
            p = 1.0 - np.exp(-(pop.age[past] - phen[past, 10]) / 200.0)
            draw = self.rng.random(int(past.sum()))
            old_age[past] = draw < p

        dead = health_dead | starved | old_age
        if not np.any(dead):
            return
        self.detritus += float(pop.body[dead].sum() + pop.energy[dead].sum())
        pop.body[dead] = 0.0
        pop.energy[dead] = 0.0
        pop.alive[dead] = False
        cause = np.where(starved[dead], "starvation",
                         np.where(old_age[dead], "senescence", "health"))
        pop.death_cause[dead] = cause

    # -- stage 14: decomposition ---------------------------------------------
    def _decompose(self) -> None:
        amount = self.cfg.energy.decay * self.detritus
        if amount <= 0.0:
            return
        self.detritus -= amount
        self.ledger.mass_to_soil += amount
        self.ledger.heat_out += amount

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

        ``sensors``/``outputs`` are the last computed vectors; rows align with
        slot order at think time (arrays are compacted before sensing).
        """
        from .brain import split as split_weights

        pop = self.population
        genome = pop.genome[i]
        W1, b1, W2, b2, W3, b3 = split_weights(genome[None, N_TRAITS:])
        parents = (
            int(pop.parent_a[i]) if pop.parent_a[i] >= 0 else None,
            int(pop.parent_b[i]) if pop.parent_b[i] >= 0 else None,
        )
        row = i if i < self.last_sensors.shape[0] else None
        return {
            "id": int(pop.id[i]),
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
        parts = [
            np.array([self.tick], dtype=np.int64),
            pop.id[idx], pop.genome[idx],
            np.round(pop.x[idx], 9), np.round(pop.y[idx], 9),
            np.round(pop.heading[idx], 9),
            np.round(pop.energy[idx], 9), np.round(pop.body[idx], 9),
            np.round(pop.health[idx], 9), pop.age[idx], pop.generation[idx],
            self.plants.id[pidx], np.round(self.plants.x[pidx], 9),
            np.round(self.plants.y[pidx], 9), np.round(self.plants.biomass[pidx], 9),
            np.array([self.detritus, self.ledger.light_in, self.ledger.heat_out,
                      self.ledger.mass_to_soil, self.ledger.mass_from_soil,
                      self.ledger.mass_in], dtype=np.float64),
        ]
        for part in parts:
            h.update(np.ascontiguousarray(part).tobytes())
        h.update(self.rng.state())
        return h.hexdigest()
