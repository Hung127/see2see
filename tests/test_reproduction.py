"""Tests for energy-gated reproduction (DESIGN §6.5)."""
import numpy as np

from evolve.actions import Actions
from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine
from evolve.genome import phenotype


def make_engine(mode="asexual", n=4, seed=1, max_pop=50, maturity=50):
    cfg = Config(
        seed=seed, ticks=100,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=10, cell_size=50.0),
        plants=PlantConfig(initial=50, max=200),
        organisms=OrganismConfig(initial=n, max=max_pop, maturity_age=maturity),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(mode=mode),
        environment=EnvironmentConfig(rain_rate=0.0),
        stats=StatsConfig(),
    )
    return Engine(cfg)


def force_ready(eng, frac=0.9):
    pop = eng.population
    pop.compact()  # stages run on dense slots, as in the real pipeline
    phen = phenotype(pop.genome)
    emax = 10.0 * phen[:, 0] ** 0.9
    pop.energy[:] = frac * emax
    pop.age[:] = 100
    return phen, emax


def all_reproduce(n):
    z = np.zeros(n)
    return Actions(z, z, np.zeros(n, bool), np.zeros(n, bool), np.ones(n),
                   np.ones(n, bool), z)


def test_asexual_birth_inherits_mutated_genome():
    eng = make_engine()
    pop = eng.population
    phen, emax = force_ready(eng)
    parent_genome = pop.genome[0].copy()
    parent_id = int(pop.id[0])
    parent_gen = int(pop.generation[0])
    n0 = pop.n_alive
    eng._reproduce(all_reproduce(pop.n_alive))
    assert pop.n_alive == 2 * n0  # every founder birthed once
    child = n0  # first child slot
    child_genome = pop.genome[child]
    diff = np.mean(child_genome != parent_genome)
    assert 0.0 < diff < 0.2  # mutated, but overwhelmingly inherited
    assert int(pop.parent_a[child]) == parent_id
    assert int(pop.parent_b[child]) == -1
    assert int(pop.generation[child]) == parent_gen + 1
    assert np.allclose(pop.x[child], pop.x[0]) and np.allclose(pop.y[child], pop.y[0])
    # parent paid its investment
    invest = phen[0, 8] * emax[0]
    assert pop.energy[0] == np.float64(0.9 * emax[0]) - invest
    # child got 90% of the surplus over its body cost
    body = float(phenotype(child_genome[None, :])[0, 0])
    assert pop.body[child] == np.float64(body)
    assert pop.energy[child] == np.float64(0.9 * (invest - body))


def test_cap_birth_fails_free():
    eng = make_engine(n=4, max_pop=4)
    pop = eng.population
    force_ready(eng)
    before = pop.energy.copy()
    n0 = pop.n_alive
    eng._reproduce(all_reproduce(pop.n_alive))
    assert pop.n_alive == n0
    assert np.array_equal(pop.energy[:n0], before)  # parent pays nothing


def test_sexual_pays_half_each():
    eng = make_engine(mode="sexual", n=2)
    pop = eng.population
    phen, emax = force_ready(eng)
    pop.x[:] = [100.0, 101.0]
    pop.y[:] = [250.0, 250.0]
    n0 = pop.n_alive
    eng._reproduce(all_reproduce(pop.n_alive))
    assert pop.n_alive == n0 + 1
    child = n0
    assert int(pop.parent_a[child]) == int(pop.id[0])
    assert int(pop.parent_b[child]) == int(pop.id[1])
    assert int(pop.generation[child]) == 1
    invest = phen[0, 8] * emax[0]
    assert pop.energy[0] == np.float64(0.9 * emax[0]) - invest / 2.0
    assert pop.energy[1] == np.float64(0.9 * emax[1]) - invest / 2.0


def test_sexual_requires_mate_in_reach():
    eng = make_engine(mode="sexual", n=2)
    pop = eng.population
    force_ready(eng)
    pop.x[:] = [100.0, 400.0]  # far apart
    pop.y[:] = [250.0, 250.0]
    n0 = pop.n_alive
    eng._reproduce(all_reproduce(pop.n_alive))
    assert pop.n_alive == n0  # no mate in reach -> no birth


def test_offspring_counters_zero():
    eng = make_engine()
    pop = eng.population
    force_ready(eng)
    # dirty the parents' counters first
    pop.bites[:] = 5.0
    pop.dist_moved[:] = 100.0
    eng._reproduce(all_reproduce(pop.n_alive))
    child = 4
    for name in ("bites", "hunts", "escapes", "repro_attempts", "repro_success",
                 "rest_ticks", "plant_intake", "meat_intake", "damage_taken",
                 "dist_moved"):
        assert float(getattr(pop, name)[child]) == 0.0, name


def test_energy_gate_blocks_poor_parents():
    eng = make_engine()
    pop = eng.population
    phen, emax = force_ready(eng, frac=0.2)  # below every threshold (>= 0.4)
    n0 = pop.n_alive
    eng._reproduce(all_reproduce(pop.n_alive))
    assert pop.n_alive == n0


def test_mass_ledger_still_closed_after_births():
    eng = make_engine(n=6, seed=3)
    pop = eng.population
    force_ready(eng)
    m0, e0 = eng.mass_stock(), eng.energy_stock()  # measured after the energy boost
    eng._reproduce(all_reproduce(pop.n_alive))
    eng.ledger.assert_closed(m0, eng.mass_stock(), e0, eng.energy_stock())


def test_both_mode_falls_back_to_asexual_without_mate():
    """genetics.mode='both': sexual when a mate is in reach, asexual otherwise."""
    eng = make_engine(mode="both", n=2)
    pop = eng.population
    phen, emax = force_ready(eng)
    pop.x[:] = [100.0, 400.0]  # far apart: no mate in reach
    pop.y[:] = [250.0, 250.0]
    n0 = pop.n_alive
    eng._reproduce(all_reproduce(pop.n_alive))
    assert pop.n_alive == n0 + 2  # both parents fall back to asexual birth
    for child in (n0, n0 + 1):
        assert int(pop.parent_b[child]) == -1  # asexual: no second parent

    # and with a mate in reach it is sexual: both parents pay half
    eng2 = make_engine(mode="both", n=2)
    pop2 = eng2.population
    phen2, emax2 = force_ready(eng2)
    pop2.x[:] = [100.0, 101.0]
    pop2.y[:] = [250.0, 250.0]
    n02 = pop2.n_alive
    eng2._reproduce(all_reproduce(pop2.n_alive))
    assert pop2.n_alive == n02 + 1
    child2 = n02
    assert int(pop2.parent_b[child2]) == int(pop2.id[1])
    invest = phen2[0, 8] * emax2[0]
    assert pop2.energy[0] == np.float64(0.9 * emax2[0]) - invest / 2.0
