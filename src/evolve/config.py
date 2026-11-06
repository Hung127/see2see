"""Configuration tree for EVOLVE.

Every value here is documented in ``docs/DESIGN.md``; the defaults are the ones the
design argues for, not arbitrary placeholders. Configuration is immutable: a run is
identified by ``(config, seed, version)`` and nothing mutates a live config.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import yaml

BOUNDARY_MODES = ("wrap", "reflect", "hard")
REPRO_MODES = ("asexual", "sexual", "both")
INTERACTION_MODES = ("off", "sampled", "full")


@dataclass(frozen=True)
class WorldConfig:
    width: float = 2000.0
    height: float = 2000.0
    boundaries: str = "wrap"
    n_field_cells: int = 40
    cell_size: float = 50.0

    def __post_init__(self) -> None:
        if self.boundaries not in BOUNDARY_MODES:
            raise ValueError(f"boundaries must be one of {BOUNDARY_MODES}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("world dimensions must be positive")
        if self.n_field_cells <= 0 or self.cell_size <= 0:
            raise ValueError("field grid must be positive")


@dataclass(frozen=True)
class PlantConfig:
    initial: int = 1500
    max: int = 5000
    k0: float = 5.0
    r: float = 0.06
    seed_cost: float = 0.2
    germination_per_tick: int = 20
    senescence: float = 0.002


@dataclass(frozen=True)
class OrganismConfig:
    initial: int = 300
    max: int = 5000
    maturity_age: int = 50
    health_max: float = 100.0
    health_regen: float = 0.05


@dataclass(frozen=True)
class EnergyConfig:
    """Coefficients of the metabolic / feeding / predation model (DESIGN §6)."""

    k_b: float = 0.015
    k_r: float = 0.002
    k_n: float = 2e-6
    k_m: float = 0.0008
    k_v: float = 0.002
    k_a: float = 0.004
    k_d: float = 0.003
    k_e: float = 0.004
    eta_p: float = 0.65
    eta_m: float = 0.85
    bite: float = 0.10
    strike_cost: float = 0.05
    k_dmg: float = 3.0
    k_mit: float = 1.5
    damage_mit_factor: float = 0.8
    build_cost: float = 1.0
    repro_eff: float = 0.9
    decay: float = 0.015
    catabolism: float = 0.5


@dataclass(frozen=True)
class GeneticsConfig:
    mode: str = "asexual"
    sigma_trait: float = 0.08
    sigma_w: float = 0.05
    large_effect_p: float = 0.002
    large_effect_mult: float = 5.0
    mutation_floor: float = 5e-4
    mutation_ceil: float = 5e-2
    crossover_p: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in REPRO_MODES:
            raise ValueError(f"genetics.mode must be one of {REPRO_MODES}")


@dataclass(frozen=True)
class EventConfig:
    """A scheduled environmental event (DESIGN §3.3)."""

    kind: str = "drought"
    start_tick: int = 0
    duration: int = 1000
    magnitude: float = 0.5
    region: tuple[float, float, float] = (0.0, 0.0, 1.0e9)


@dataclass(frozen=True)
class EnvironmentConfig:
    t_mean: float = 20.0
    t_amp: float = 6.0
    m_mean: float = 0.55
    m_amp: float = 0.15
    season_period: int = 2000
    rain_rate: float = 0.002
    events: tuple[EventConfig, ...] = ()


@dataclass(frozen=True)
class StatsConfig:
    interval: int = 50
    snapshot_interval: int = 5000
    interactions: str = "sampled"

    def __post_init__(self) -> None:
        if self.interactions not in INTERACTION_MODES:
            raise ValueError(f"stats.interactions must be one of {INTERACTION_MODES}")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    ticks: int = 10000
    world: WorldConfig = field(default_factory=WorldConfig)
    plants: PlantConfig = field(default_factory=PlantConfig)
    organisms: OrganismConfig = field(default_factory=OrganismConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    genetics: GeneticsConfig = field(default_factory=GeneticsConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS: dict[str, type] = {
    "world": WorldConfig,
    "plants": PlantConfig,
    "organisms": OrganismConfig,
    "energy": EnergyConfig,
    "genetics": GeneticsConfig,
    "environment": EnvironmentConfig,
    "stats": StatsConfig,
}


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a config dataclass, rejecting unknown keys loudly."""
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown configuration key(s) for {cls.__name__}: {sorted(unknown)}")
    if cls is EnvironmentConfig and "events" in data:
        data = dict(data)
        data["events"] = tuple(
            EventConfig(**e) if isinstance(e, dict) else e for e in data["events"]
        )
    return cls(**data)


def config_from_dict(raw: dict[str, Any]) -> Config:
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    top_allowed = {f.name for f in fields(Config)}
    unknown = set(raw) - top_allowed
    if unknown:
        raise ValueError(f"unknown top-level configuration key(s): {sorted(unknown)}")
    kwargs: dict[str, Any] = {k: v for k, v in raw.items() if k not in _SECTIONS}
    for name, cls in _SECTIONS.items():
        if name in raw:
            section = raw[name]
            if not isinstance(section, dict):
                raise ValueError(f"configuration section {name!r} must be a mapping")
            kwargs[name] = _build(cls, section)
    return Config(**kwargs)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return config_from_dict(raw)


def config_hash(cfg: Config) -> str:
    payload = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
