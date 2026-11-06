"""Tests for metabolic costs and the dual conservation ledger."""
import numpy as np
import pytest

from evolve.config import EnergyConfig
from evolve.energy import (
    Ledger, assimilation, feed, metabolic_cost, respiration, strike_damage,
)

CFG = EnergyConfig()


def phen_row(*, size=1.0, speed_max=20.0, vision=120.0, angle=180.0, rate=1.0,
             attack=0.0, defense=0.0, diet=0.0, lifespan=1000.0):
    row = np.zeros((1, 12))
    row[0, 0] = size
    row[0, 1] = speed_max
    row[0, 2] = vision
    row[0, 3] = angle
    row[0, 4] = rate
    row[0, 5] = attack
    row[0, 6] = defense
    row[0, 9] = diet
    row[0, 10] = lifespan
    return row


def test_metabolic_cost_terms():
    p = phen_row()
    base = metabolic_cost(p, np.array([0.0]), np.array([20.0]))[0]
    # basal + repair + brain + vision; no movement, no thermal, no attack/defense/enzyme
    expected = (CFG.k_b * 1.0 * 1.0 + CFG.k_r * 1.0 * 1.0 + CFG.k_n * 380
                + CFG.k_v * (1.2 ** 2) * 0.5)
    assert base == pytest.approx(expected)
    moving = metabolic_cost(p, np.array([10.0]), np.array([20.0]))[0]
    assert moving - base == pytest.approx(CFG.k_m * 1.0 * 1.0)
    warm = metabolic_cost(p, np.array([0.0]), np.array([30.0]))[0]
    assert warm > base


def test_attack_defense_and_enzyme_costs():
    p = phen_row(attack=2.0, defense=3.0, diet=0.5)
    c = metabolic_cost(p, np.array([0.0]), np.array([20.0]))[0]
    p0 = phen_row()
    c0 = metabolic_cost(p0, np.array([0.0]), np.array([20.0]))[0]
    assert c - c0 == pytest.approx(
        CFG.k_a * 2.0 + CFG.k_d * 3.0 + CFG.k_e * 4 * 0.5 * 0.5)


def test_respiration_moves_mass_to_soil_and_energy_to_heat():
    led = Ledger()
    mass_to_soil, heat = respiration(led, 2.5)
    assert mass_to_soil == 2.5 and heat == 2.5
    assert led.mass_to_soil == 2.5 and led.heat_out == 2.5


def test_undigested_goes_to_detritus():
    taken, gain, detritus = feed(demand=0.4, available=1.0, eta=0.6)
    assert taken == pytest.approx(0.4)
    assert gain == pytest.approx(0.24)
    assert detritus == pytest.approx(0.16)
    assert gain + detritus == pytest.approx(taken)


def test_feed_limited_by_available():
    taken, gain, detritus = feed(demand=1.0, available=0.3, eta=0.6)
    assert taken == pytest.approx(0.3)
    assert gain + detritus == pytest.approx(0.3)


def test_strike_damage_mitigation():
    dmg = strike_damage(attack=2.0, size=2.0, defense=0.0, def_size=1.0)
    assert dmg > 0
    assert strike_damage(2.0, 2.0, 5.0, 5.0) == 0.0


def test_ledger_closes():
    led = Ledger()
    led.light_in = 10.0
    led.heat_out = 4.0
    mass = 100.0 + 0.0  # mass unchanged (light adds energy only)
    led.assert_closed(initial_mass=100.0, final_mass=mass,
                      initial_energy=50.0, final_energy=56.0)
    with pytest.raises(AssertionError):
        led.assert_closed(100.0, 101.0, 50.0, 56.0)


def test_assimilation_specialists_beat_generalist():
    herb = assimilation(CFG.eta_p, diet=0.0, rate=1.0, plant=True)
    carn = assimilation(CFG.eta_m, diet=1.0, rate=1.0, plant=False)
    mixed_p = assimilation(CFG.eta_p, diet=0.5, rate=1.0, plant=True)
    mixed_m = assimilation(CFG.eta_m, diet=0.5, rate=1.0, plant=False)
    assert herb > mixed_p and carn > mixed_m
    assert assimilation(CFG.eta_m, diet=0.0, rate=1.0, plant=False) == 0.0
    assert assimilation(CFG.eta_p, diet=1.0, rate=1.0, plant=True) == 0.0
