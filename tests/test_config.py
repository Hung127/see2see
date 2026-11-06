"""Tests for the configuration tree (exact defaults from docs/DESIGN.md)."""
import pytest

from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig, config_hash, load_config,
)


def test_defaults_match_design():
    cfg = Config()
    assert cfg.seed == 42 and cfg.ticks == 10000
    assert cfg.world == WorldConfig(width=1000.0, height=1000.0, boundaries="wrap",
                                    n_field_cells=40, cell_size=25.0)
    assert cfg.plants.initial == 6000 and cfg.plants.max == 20000
    assert cfg.plants.k0 == 5.0 and cfg.plants.r == 0.06
    assert cfg.plants.seed_cost == 0.2 and cfg.plants.germination_per_tick == 20
    assert cfg.plants.senescence == 0.002
    assert cfg.organisms.initial == 300 and cfg.organisms.max == 2000
    assert cfg.organisms.maturity_age == 50
    assert cfg.organisms.health_max == 100.0 and cfg.organisms.health_regen == 0.05
    e = cfg.energy
    assert e.k_b == 0.015 and e.k_r == 0.002 and e.k_n == 2e-6 and e.k_m == 0.0008
    assert e.k_v == 0.002 and e.k_a == 0.004 and e.k_d == 0.003 and e.k_e == 0.004
    # Calibrated values (docs/CALIBRATION.md): the design's original 0.65/0.10
    # failed the founder-viability criterion in every tested seed.
    assert e.eta_p == 0.75 and e.eta_m == 0.85 and e.bite == 0.50
    assert e.strike_cost == 0.05 and e.k_dmg == 3.0 and e.k_mit == 1.5
    assert e.damage_mit_factor == 0.8 and e.build_cost == 1.0
    assert e.repro_eff == 0.9 and e.decay == 0.015 and e.catabolism == 0.5
    g = cfg.genetics
    assert g.mode == "asexual" and g.sigma_trait == 0.08 and g.sigma_w == 0.05
    assert g.large_effect_p == 0.002 and g.large_effect_mult == 5.0
    assert g.mutation_floor == 5e-4 and g.mutation_ceil == 5e-2 and g.crossover_p == 0.5
    env = cfg.environment
    assert env.t_mean == 20.0 and env.t_amp == 6.0 and env.m_mean == 0.55
    assert env.m_amp == 0.15 and env.season_period == 2000 and env.rain_rate == 0.002
    s = cfg.stats
    assert s.interval == 50 and s.snapshot_interval == 5000 and s.interactions == "sampled"


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("seed: 1\nworld:\n  width: 100\n  bogus: 3\n")
    with pytest.raises(ValueError, match="bogus"):
        load_config(str(p))


def test_load_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\nticks: 500\nworld:\n  width: 500\n  height: 500\n")
    cfg = load_config(str(p))
    assert cfg.seed == 7 and cfg.ticks == 500
    assert cfg.world.width == 500.0 and cfg.world.height == 500.0
    assert cfg.plants.initial == 6000  # untouched section keeps defaults


def test_config_hash_stable(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\n")
    assert config_hash(load_config(str(p))) == config_hash(load_config(str(p)))
    q = tmp_path / "d.yaml"
    q.write_text("seed: 8\n")
    assert config_hash(load_config(str(p))) != config_hash(load_config(str(q)))
