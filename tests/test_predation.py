"""Tests for predation, carcass decay and scavenging (DESIGN §6.4, §6.6)."""
import numpy as np
import pytest

from evolve.actions import Actions
from evolve.carcasses import Carcasses
from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.energy import Ledger
from evolve.engine import Engine
from evolve.genome import phenotype


def make_engine(n=2, seed=1, max_pop=50):
    cfg = Config(
        seed=seed, ticks=100,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=10, cell_size=50.0),
        plants=PlantConfig(initial=10, max=100),
        organisms=OrganismConfig(initial=n, max=max_pop, maturity_age=50),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.0),
        stats=StatsConfig(),
    )
    return Engine(cfg)


def set_traits(pop, i, attack=0.0, defense=0.0, diet=0.5, body=None, rate=None):
    g = pop.genome[i]
    g[5] = attack / 5.0
    g[6] = defense / 5.0
    g[9] = diet
    if rate is not None:
        g[4] = rate
    if body is not None:
        g[0] = (np.log(body) - np.log(0.3)) / np.log(16.67)


def actions_for(n, attack=(), eat=()):
    z = np.zeros(n)
    atk = np.zeros(n, bool)
    eat_mask = np.zeros(n, bool)
    for i in attack:
        atk[i] = True
    for i in eat:
        eat_mask[i] = True
    return Actions(z, z, eat_mask, atk, np.ones(n), np.zeros(n, bool), z)


def scene(n=2):
    eng = make_engine(n=n)
    eng.population.compact()
    return eng, eng.population


def test_kill_transfers_biomass():
    eng, pop = scene(2)
    pop.x[:] = [100.0, 101.0]
    pop.y[:] = [250.0, 250.0]
    set_traits(pop, 0, attack=5.0, diet=1.0)
    set_traits(pop, 1, defense=0.0, diet=0.5)
    phen = phenotype(pop.genome)
    pop.body[:] = phen[:, 0]
    pop.health[1] = 2.0
    pop.energy[1] = 1.0
    pop.energy[0] = 0.5 * 10.0 * phen[0, 0] ** 0.9

    total_prey = float(pop.body[1] + pop.energy[1])
    e_before = float(pop.energy[0])
    cost = 0.05 * phen[0, 5]
    emax_att = 10.0 * phen[0, 0] ** 0.9
    eta = 0.85 * np.sqrt(phen[0, 9]) * np.sqrt(phen[0, 4])
    m0, e0 = eng.mass_stock(), eng.energy_stock()

    eng._predation(actions_for(2, attack=[0]))
    assert not pop.alive[1]
    gain = float(pop.energy[0]) - (e_before - cost)
    expected_gain = min(eta * total_prey, emax_att - (e_before - cost))
    assert gain == pytest.approx(expected_gain, rel=1e-9)
    assert eng.carcasses.total_biomass() == pytest.approx(total_prey - gain, rel=1e-9)
    eng.ledger.assert_closed(m0, eng.mass_stock(), e0, eng.energy_stock())


def test_damage_split_proportional():
    eng, pop = scene(3)
    pop.x[:] = [100.0, 101.0, 99.0]
    pop.y[:] = [250.0, 250.0, 250.0]
    set_traits(pop, 0, defense=0.0, diet=0.5)
    set_traits(pop, 1, attack=3.5, diet=1.0, rate=0.5)
    set_traits(pop, 2, attack=1.5, diet=1.0, rate=0.5)
    phen = phenotype(pop.genome)
    pop.body[:] = phen[:, 0]
    pop.health[0] = 1.0
    pop.energy[0] = 1.0
    pop.energy[1] = 0.2 * 10.0 * phen[1, 0] ** 0.9
    pop.energy[2] = 0.2 * 10.0 * phen[2, 0] ** 0.9
    m0, e0 = eng.mass_stock(), eng.energy_stock()
    e1, e2 = float(pop.energy[1]), float(pop.energy[2])
    cost1 = 0.05 * phen[1, 5]
    cost2 = 0.05 * phen[2, 5]
    eng._predation(actions_for(3, attack=[1, 2]))
    assert not pop.alive[0]
    g1 = float(pop.energy[1]) - (e1 - cost1)
    g2 = float(pop.energy[2]) - (e2 - cost2)
    # Meat is apportioned by credited damage; damage scales with attacker mass.
    d1 = 3.0 * phen[1, 5] * phen[1, 0] ** 0.4
    d2 = 3.0 * phen[2, 5] * phen[2, 0] ** 0.4
    assert g1 / (g1 + g2) == pytest.approx(d1 / (d1 + d2), abs=1e-6)
    eng.ledger.assert_closed(m0, eng.mass_stock(), e0, eng.energy_stock())


def test_carcass_decay_closes_ledger():
    led = Ledger()
    carc = Carcasses()
    carc.add(100.0, 100.0, 100.0)
    m0 = carc.total_biomass()
    e0 = carc.total_biomass()
    for _ in range(100):
        carc.decay(led, 0.015)
    carc.remove_dead()
    assert led.mass_to_soil > 0.0
    assert led.heat_out == pytest.approx(led.mass_to_soil, rel=1e-12)
    led.assert_closed(m0, carc.total_biomass(), e0, carc.total_biomass())


def test_scavenger_eats_carcass():
    eng, pop = scene(1)
    pop.x[:] = [100.0, ]
    pop.y[:] = [250.0, ]
    set_traits(pop, 0, diet=1.0)
    phen = phenotype(pop.genome)
    pop.body[:] = phen[:, 0]
    pop.energy[0] = 0.2 * 10.0 * phen[0, 0] ** 0.9
    eng.carcasses.add(100.0, 250.0, 5.0)
    m0, e0 = eng.mass_stock(), eng.energy_stock()
    e_before = float(pop.energy[0])
    eng._feed(actions_for(1, eat=[0]))
    taken = float(5.0 - eng.carcasses.biomass[0])
    assert taken > 0.0
    eta = 0.85 * np.sqrt(1.0) * np.sqrt(phen[0, 4])
    assert float(pop.energy[0]) - e_before == pytest.approx(eta * taken, rel=1e-9)
    assert eng.detritus == pytest.approx((1.0 - eta) * taken, rel=1e-9)
    eng.ledger.assert_closed(m0, eng.mass_stock(), e0, eng.energy_stock())
