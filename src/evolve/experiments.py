"""Experiments A-H and the pre-registered assay G (DESIGN §13).

Each scenario is a YAML configuration plus, where the design calls for a
manipulation the model has no configuration key for, a *setup* or *intervention*
hook applied to the live engine:

===========  ===========================================================
Scenario     Hook
===========  ===========================================================
A            none (calibrated default)
B            ``carnivore_founders``: 200 extra founders with diet 0.85
C            none (drought event is in the YAML; magnitude 0.6)
D            ``clumped_plants``: 60 % fewer plants, clustered
E            ``move_nutrients`` at t = 15,000 (heatwave is in the YAML)
F            ``no_predators``: attack loci clipped to 0, predation stage off
G            assay runner (evolution phase -> sampled genomes -> arenas)
H            sexual arm and an asexual control at the same density
===========  ===========================================================

Hooks are *initial conditions* or *environmental manipulations*: they never read
or write an organism's decisions. Setup hooks that change mass (B adds
organisms) **rebaseline** the engine's ledger, because they are part of
constructing the initial state, not a process inside the run.

Scaling
-------
This machine is single-core with ~1 GB RAM; DESIGN-scale runs are ~7.5 h of
wall clock (docs/BENCHMARKS.md), so every scenario is run at ``scale`` (the CLI
default is :data:`DEFAULT_SCALE`):

* lengths and world coordinates × ``scale`` (field cells and cell size keep the
  25-unit resolution);
* plant and founder *counts* × ``scale²`` so density is preserved;
* the horizon and every scheduled event time × ``scale``.

A scaled run is therefore *smaller and shorter* than the design: it is a
feasibility demonstration, not an equivalent experiment. Every summary says so,
and docs/RESULTS.md reports the scale actually used.
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

import numpy as np

from .config import Config, EventConfig, load_config
from .engine import Engine
from .genome import N_TRAITS, TRAIT_NAMES, founders, phenotype
from .rng import Rng
from .stats import COLUMNS, StatsRecorder, export_csv, make_plots

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs")

ASSAY_CONFIG = "assay.yaml"

#: Default area/horizon multiplier for experiment runs on this machine.
DEFAULT_SCALE = 0.5

#: Arena geometry for assay G: a small world at the calibrated plant density.
ARENA_SIZE = 200.0
ARENA_PLANT_DENSITY = 6.0e-3  # plants per unit^2, the calibrated default

#: Fraction of a paired arena's ticks the design specifies (500).
DESIGN_ARENA_TICKS = 500

TRAIT_INDEX = {name: i for i, name in enumerate(TRAIT_NAMES)}


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
def scaled_config(config: Config, scale: float = DEFAULT_SCALE,
                  ticks: int | None = None) -> Config:
    """Scale a design-scale configuration down (area, counts, time).

    ``ticks`` overrides the scaled horizon (used by the tests and by
    ``--ticks`` on the CLI). See the module docstring for exactly what scales.
    """
    s = float(scale)
    if s <= 0:
        raise ValueError("scale must be positive")
    if s == 1.0 and ticks is None:
        return config

    world = replace(
        config.world,
        width=config.world.width * s,
        height=config.world.height * s,
        n_field_cells=max(4, int(round(config.world.n_field_cells * s))),
    )
    plants = replace(
        config.plants,
        initial=max(10, int(round(config.plants.initial * s * s))),
        max=max(20, int(round(config.plants.max * s * s))),
    )
    organisms = replace(
        config.organisms,
        initial=max(2, int(round(config.organisms.initial * s * s))),
        max=max(4, int(round(config.organisms.max * s * s))),
    )
    events = tuple(
        replace(ev, start_tick=int(round(ev.start_tick * s)),
                duration=max(1, int(round(ev.duration * s))))
        for ev in config.environment.events
    )
    environment = replace(config.environment, events=events)
    horizon = int(ticks) if ticks is not None else max(1, int(round(config.ticks * s)))
    return replace(config, ticks=horizon, world=world, plants=plants,
                   organisms=organisms, environment=environment)


# ---------------------------------------------------------------------------
# Setup and intervention hooks
# ---------------------------------------------------------------------------
def rebaseline(engine: Engine) -> None:
    """Treat the current state as the initial state for the mass ledger.

    Setup hooks are part of *constructing* the initial condition. If a hook adds
    mass (extra founders), the ledger's reference points must move with it, or
    every later conservation check would report the setup as a leak.
    """
    engine.initial_mass = engine.mass_stock()
    engine.initial_energy = engine.energy_stock()
    engine.initial_soil = engine.soil_stock()
    engine.initial_total = engine.initial_mass + engine.initial_soil


def carnivore_founders(engine: Engine, scale: float = DEFAULT_SCALE,
                       count: int = 200, diet: float = 0.85) -> int:
    """Experiment B: add ``count`` founders biased to a carnivorous diet.

    The extra genomes are drawn from the same founder distribution as everyone
    else (``Beta(2, 2)`` traits, ``N(0, 0.35)`` weights); only the diet locus is
    set to ``diet``. Nothing else is touched, so any effect is a diet effect.
    """
    n = max(1, int(round(count * scale * scale)))
    rng = Rng(engine.cfg.seed + 7919)
    genes = np.asarray(founders(n, rng), dtype=np.float64)
    genes[:, TRAIT_INDEX["diet_preference"]] = float(diet)
    xs = rng.random(n) * engine.world.width
    ys = rng.random(n) * engine.world.height
    phen = phenotype(genes)
    emax = 10.0 * phen[:, 0] ** 0.9
    added = 0
    for i in range(n):
        slot = engine.population.add(genes[i], float(xs[i]), float(ys[i]),
                                     energy=0.5 * float(emax[i]))
        if slot < 0:
            break
        engine.population.heading[slot] = float(rng.random() * 2.0 * math.pi
                                                - math.pi)
        added += 1
    engine._refresh_phen()
    rebaseline(engine)
    return added


def no_predators(engine: Engine, scale: float = DEFAULT_SCALE) -> None:
    """Experiment F: clip every founder's attack locus to 0.

    The predation *stage* is also switched off by the runner. Clipping the gene
    as well keeps the genotype consistent with a world that has no predators:
    there is nothing to attack, and nothing that attacks (DESIGN §13 F).
    """
    engine.population.genome[:, TRAIT_INDEX["attack"]] = 0.0
    engine._refresh_phen()


def clumped_plants(engine: Engine, scale: float = DEFAULT_SCALE) -> None:
    """Experiment D / assay G: move living plants into tight clusters.

    Positions change, biomass does not: the same number of plants with the same
    total biomass, concentrated so that foraging requires travel between
    patches. Draws come from a private ``Rng`` so the engine's documented tick
    stream is untouched.
    """
    plants = engine.plants
    idx = np.flatnonzero(plants.alive)
    if idx.size == 0:
        return
    rng = Rng(engine.cfg.seed + 104729)
    n_clusters = max(2, int(round(40 * scale * scale)))
    centers = np.column_stack([
        rng.random(n_clusters) * engine.world.width,
        rng.random(n_clusters) * engine.world.height,
    ])
    assign = rng.integers(0, n_clusters, size=idx.size)
    spread = 0.06 * min(engine.world.width, engine.world.height)
    jitter = rng.normal(0.0, spread, size=(idx.size, 2))
    pos = centers[assign] + jitter
    plants.x[idx] = np.mod(pos[:, 0], engine.world.width)
    plants.y[idx] = np.mod(pos[:, 1], engine.world.height)


def move_nutrients(engine: Engine, scale: float = DEFAULT_SCALE) -> None:
    """Experiment E: move all soil from the right half onto the left half.

    Element-wise and therefore mass-preserving: each source column is added to
    its mirror column on the left and then zeroed, so the soil total is unchanged
    to floating point and the mass ledger stays closed.
    """
    soil = engine.fields.soil
    n = soil.shape[1]
    half = n // 2
    source = soil[:, half:].copy()
    soil[:, half:] = 0.0
    targets = n - 1 - np.arange(half, n)
    # targets are distinct and all < half, so a plain fancy-index add is exact.
    soil[:, targets] += source


HOOKS: dict[str, Callable[..., Any]] = {
    "carnivore_founders": carnivore_founders,
    "no_predators": no_predators,
    "clumped_plants": clumped_plants,
    "move_nutrients": move_nutrients,
}


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """One experiment: a config file plus the manipulations of DESIGN §13."""

    id: str
    name: str
    manipulation: str
    config_file: str
    horizon: int
    setup: str | None = None
    intervention: str | None = None
    intervention_tick: int | None = None
    step_kwargs: tuple[tuple[str, bool], ...] = ()
    note: str = ""

    def step_options(self) -> dict[str, bool]:
        return dict(self.step_kwargs)


SCENARIOS: dict[str, Scenario] = {
    "a": Scenario(
        "a", "Stable ecosystem", "calibrated default, 30,000 ticks",
        "experiment_a.yaml", 30000),
    "b": Scenario(
        "b", "Predator pressure",
        "+200 founders with diet_preference 0.85 on the default baseline",
        "experiment_b.yaml", 30000, setup="carnivore_founders"),
    "c": Scenario(
        "c", "Severe drought",
        "drought magnitude 0.6 for 600 ticks starting at tick 500",
        "experiment_c.yaml", 3000,
        note=("The implemented drought gives moisture -60 % and plant growth "
              "-48 % (1-0.8*0.6), not the design's -70 % growth. The gap is "
              "reported rather than forced by changing the drought model.")),
    "d": Scenario(
        "d", "Resource competition",
        "plants reduced to 40 % and clustered, 1,000 ticks",
        "experiment_d.yaml", 1000, setup="clumped_plants"),
    "e": Scenario(
        "e", "Environmental shift",
        "+8 C from tick 15,000 and all right-half soil moved to the left half",
        "experiment_e.yaml", 30000, intervention="move_nutrients",
        intervention_tick=15000),
    "f": Scenario(
        "f", "No predators",
        "attack loci clipped to 0 at founding and the predation stage disabled",
        "experiment_f.yaml", 30000, setup="no_predators",
        step_kwargs=(("predation", False),)),
    "h": Scenario(
        "h", "Sexual reproduction",
        "mode=sexual with 1,000 founders, against an asexual control at the "
        "same density and seed",
        "experiment_h.yaml", 30000),
}


# ---------------------------------------------------------------------------
# Running one scenario
# ---------------------------------------------------------------------------
def _load_scenario_config(scenario: Scenario, scale: float,
                          ticks: int | None, seed: int | None) -> Config:
    config = load_config(os.path.join(CONFIGS_DIR, scenario.config_file))
    config = scaled_config(config, scale, ticks)
    if seed is not None:
        config = replace(config, seed=int(seed))
    return config


def _prepare_engine(config: Config, scenario: Scenario, scale: float,
                    engine_factory: Callable[[Config], Engine]) -> Engine:
    engine = engine_factory(config)
    if scenario.setup:
        HOOKS[scenario.setup](engine, scale)
        rebaseline(engine)
    return engine


def _run_engine(engine: Engine, scenario: Scenario, scale: float,
                recorder: StatsRecorder) -> list[dict[str, Any]]:
    """Step to the configured horizon, sample stats, run the intervention."""
    horizon = int(engine.cfg.ticks)
    interval = max(1, int(engine.cfg.stats.interval))
    intervention_tick = (int(round(scenario.intervention_tick * scale))
                         if scenario.intervention and scenario.intervention_tick
                         else None)
    options = scenario.step_options()
    rows: list[dict[str, Any]] = []
    while engine.tick < horizon:
        if (intervention_tick is not None and engine.tick == intervention_tick - 1):
            HOOKS[scenario.intervention](engine, scale)
        engine.step(**options)
        if engine.tick % interval == 0:
            rows.append(recorder.record(engine))
    if not rows or rows[-1]["tick"] != engine.tick:
        rows.append(recorder.record(engine))
    return rows


def _trait_drift(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """First-to-last population mean for each reported trait column."""
    if len(rows) < 2:
        return {}
    out = {}
    for col in ("mean_diet", "mean_attack", "mean_defense", "mean_speed",
                "mean_body", "mean_vision", "mean_lifespan"):
        out[col] = (float(rows[0][col]), float(rows[-1][col]))
    return out


def _summary_markdown(scenario: Scenario, config: Config, scale: float,
                      rows: list[dict[str, Any]], engine: Engine,
                      extra_note: str = "") -> str:
    last = rows[-1] if rows else {}
    tail = rows[-max(1, len(rows) // 5):] if rows else []
    mean_pop = (sum(r["population"] for r in tail) / len(tail)) if tail else 0.0
    extinct = engine.population.n_alive == 0
    notes = [scenario.note, extra_note]
    notes = [n for n in notes if n]
    lines = [
        f"# Experiment {scenario.id.upper()} — {scenario.name}",
        "",
        f"**Manipulation.** {scenario.manipulation}",
        "",
        "## Run",
        "",
        f"- seed: `{config.seed}`",
        f"- scale: `{scale}` (design scale would be `1.0`)",
        f"- simulated ticks: `{engine.tick}` (design horizon: `{scenario.horizon}`)",
        f"- world: `{config.world.width:.0f} x {config.world.height:.0f}` units, "
        f"`{config.world.n_field_cells}` field cells",
        f"- founders: `{config.organisms.initial}`, plants: "
        f"`{config.plants.initial}`",
        f"- reproduction: `{config.genetics.mode}`",
        "",
        "## Outcome",
        "",
        f"- final population: **{engine.population.n_alive}**",
        f"- extinct: **{'yes' if extinct else 'no'}**",
        f"- mean population over the last fifth of the run: **{mean_pop:.1f}**",
        f"- births / deaths / mutations: `{engine.births_total}` / "
        f"`{engine.deaths_total}` / `{engine.mutations_total}`",
        f"- final plant biomass: `{engine.plants.total_biomass():.1f}` bu",
        f"- final soil nutrients: `{engine.soil_stock():.1f}`",
        "",
    ]
    if last:
        roles = ", ".join(
            f"{role} {last.get(role, 0)}"
            for role in ("herbivores", "carnivores", "omnivores",
                         "unclassified"))
        lines += [
            "## Readouts at the final sample",
            "",
            f"- role mix (lifetime intake): {roles}",
            f"- mean diet preference: `{last['mean_diet']:.3f}`, "
            f"attack: `{last['mean_attack']:.3f}`, "
            f"defence: `{last['mean_defense']:.3f}`",
        ]
        drift = _trait_drift(rows)
        if drift:
            lines.append("- trait drift (first → last sample): " + "; ".join(
                f"{col} {a:.3f} → {b:.3f}" for col, (a, b) in drift.items()))
        lines.append("")
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    lines += [
        "## Interpretation limits",
        "",
        "This is a *scaled* run (see the module docstring of "
        "`evolve.experiments`): smaller world, fewer organisms and a shorter",
        "horizon than DESIGN §13. It demonstrates that the mechanism runs and "
        "what it did here; it is not evidence at design scale.",
        "",
    ]
    return "\n".join(lines)


def run_scenario(scenario_id: str, out_dir: str, *,
                 scale: float = DEFAULT_SCALE, ticks: int | None = None,
                 seed: int | None = None,
                 engine_factory: Callable[[Config], Engine] | None = None
                 ) -> dict[str, Any]:
    """Run one scenario (A-F, H) and write CSV, plots and ``summary.md``."""
    key = str(scenario_id).lower()
    if key not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario_id!r}; "
                       f"known: {sorted(SCENARIOS)}")
    scenario = SCENARIOS[key]
    factory = engine_factory or (lambda config: Engine(config))
    config = _load_scenario_config(scenario, scale, ticks, seed)
    os.makedirs(out_dir, exist_ok=True)

    if key == "h":
        return _run_h(scenario, config, out_dir, scale, factory)

    engine = _prepare_engine(config, scenario, scale, factory)
    recorder = StatsRecorder()
    rows = _run_engine(engine, scenario, scale, recorder)
    export_csv(rows, os.path.join(out_dir, "stats.csv"))
    make_plots(rows, out_dir)
    summary = {
        "scenario": key,
        "name": scenario.name,
        "seed": config.seed,
        "scale": float(scale),
        "ticks": int(engine.tick),
        "final_population": int(engine.population.n_alive),
        "extinct": engine.population.n_alive == 0,
        "births": int(engine.births_total),
        "deaths": int(engine.deaths_total),
        "final_plant_biomass": float(engine.plants.total_biomass()),
    }
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(_summary_markdown(scenario, config, scale, rows, engine))
    return summary


def _run_h(scenario: Scenario, config: Config, out_dir: str, scale: float,
           factory: Callable[[Config], Engine]) -> dict[str, Any]:
    """Experiment H: the sexual arm and an asexual control, same seed/density."""
    arms = {}
    for label, mode in (("sexual", "sexual"), ("asexual", "asexual")):
        arm_cfg = replace(config, genetics=replace(config.genetics, mode=mode))
        arm_dir = os.path.join(out_dir, label)
        os.makedirs(arm_dir, exist_ok=True)
        engine = factory(arm_cfg)
        recorder = StatsRecorder()
        rows = _run_engine(engine, scenario, scale, recorder)
        export_csv(rows, os.path.join(arm_dir, "stats.csv"))
        make_plots(rows, arm_dir)
        last = rows[-1] if rows else {}
        arms[label] = {
            "config": arm_cfg,
            "engine": engine,
            "rows": rows,
            "summary": {
                "final_population": int(engine.population.n_alive),
                "births": int(engine.births_total),
                "deaths": int(engine.deaths_total),
                "extinct": engine.population.n_alive == 0,
                "mean_pairwise": float(last.get("mean_pairwise", 0.0)),
                "unique_genomes": int(last.get("unique_genomes", 0)),
                "repro_attempts": float(last.get("repro_attempts", 0.0)),
                "repro_successes": float(last.get("repro_successes", 0.0)),
            },
        }
        with open(os.path.join(arm_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write(_summary_markdown(
                replace(scenario, name=f"{scenario.name} ({label} arm)"),
                arm_cfg, scale, rows, engine,
                extra_note=f"Arm of experiment H: genetics.mode = {mode}."))

    lines = [
        "# Experiment H — Sexual reproduction",
        "",
        f"**Manipulation.** {scenario.manipulation}",
        "",
        "Both arms use the same seed, world, plant layout and founder genomes; "
        "they differ only in `genetics.mode`.",
        "",
        f"- scale: `{scale}`, simulated ticks: `{arms['sexual']['engine'].tick}`",
        "",
        "| Arm | final pop | births | deaths | extinct | mean pairwise "
        "distance | unique genomes | repro attempts | repro successes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label in ("sexual", "asexual"):
        s = arms[label]["summary"]
        lines.append(
            f"| {label} | {s['final_population']} | {s['births']} | "
            f"{s['deaths']} | {'yes' if s['extinct'] else 'no'} | "
            f"{s['mean_pairwise']:.4f} | {s['unique_genomes']} | "
            f"{s['repro_attempts']:.2f} | {s['repro_successes']:.2f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Birth rate, mate encounters and offspring viability are read from "
        "the same simulated time in both arms; the sexual arm must find mates, "
        "the asexual arm does not.",
        "- A scaled run is shorter than the design horizon, so the sexual arm "
        "gets fewer opportunities to encounter a mate than DESIGN §13 assumes.",
        "",
        "## Interpretation limits",
        "",
        "Scaled run (see `evolve.experiments`): not evidence at design scale.",
        "",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return {
        "scenario": "h",
        "name": scenario.name,
        "seed": config.seed,
        "scale": float(scale),
        "ticks": int(arms["sexual"]["engine"].tick),
        "arms": {k: v["summary"] for k, v in arms.items()},
    }


# ---------------------------------------------------------------------------
# Assay G: pre-registered evolved-vs-random comparison
# ---------------------------------------------------------------------------
def _wilcoxon_ranks(sorted_abs: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) for a sorted array, ties get the mean rank."""
    n = sorted_abs.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs: Iterable[float]) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test (normal approximation, tie-corrected).

    Implemented in-house because scipy is not a dependency. Zero differences
    are dropped (Wilcoxon's standard convention), which is also why ``n`` is
    reported next to ``p``.
    """
    d = np.asarray(list(diffs), dtype=np.float64)
    d = d[d != 0.0]
    n = int(d.size)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "z": 0.0, "p": 1.0}
    absd = np.abs(d)
    order = np.argsort(absd, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = _wilcoxon_ranks(absd[order])
    w_plus = float(ranks[d > 0.0].sum())
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(absd, return_counts=True)
    tie_term = float(((counts.astype(np.float64) ** 3 - counts) / 2.0).sum())
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    if var <= 0.0:
        return {"n": n, "w_plus": w_plus, "z": 0.0, "p": 1.0}
    z = (w_plus - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))  # two-sided, normal approximation
    return {"n": n, "w_plus": w_plus, "z": float(z), "p": float(min(1.0, p))}


def bootstrap_ci(diffs: Iterable[float], rng: Rng,
                 iterations: int = 10_000,
                 level: float = 0.95) -> tuple[float, float, float]:
    """Mean difference with a percentile bootstrap CI."""
    d = np.asarray(list(diffs), dtype=np.float64)
    n = d.size
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(int(iterations), n))
    means = d[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(d.mean()), float(lo), float(hi)


def _plant_layout_hash(engine: Engine) -> str:
    plants = engine.plants
    idx = np.flatnonzero(plants.alive)
    digest = hashlib.sha256()
    for name in ("x", "y", "biomass"):
        digest.update(np.round(getattr(plants, name)[idx], 9).tobytes())
    return digest.hexdigest()[:16]


def _write_pre_registration(path: str, *, scale: float, replicates: int,
                            samples: int, evolution_ticks: int,
                            arena_ticks: int, base_seed: int) -> None:
    text = f"""# Assay G — pre-registration

Written before any engine is constructed. This file is the protocol; the
results in `summary.md` and `assay.csv` are reported against it whatever they
say, including "no effect" and "random genomes did better".

Status: **pre-registered** (DESIGN §13, protocol fixed before execution).

## Parameters used in this run

| Parameter | Value |
|---|---|
| scale (area, counts, time) | `{scale}` |
| evolution replicates | `{replicates}` |
| genomes sampled per replicate | `{samples}` |
| evolution horizon (ticks) | `{evolution_ticks}` |
| arena horizon (ticks) | `{arena_ticks}` |
| base seed | `{base_seed}` |

`scale = 1.0` is the design; values below 1 are a documented down-scale for a
single-core machine (see `evolve.experiments`).

## 1. Evolution phase

Run `configs/assay.yaml` (calibrated baseline, asexual, **no predators**,
**clumped plants**) for the evolution horizon, one run per replicate seed
`base_seed + i`. Predation is switched off at the stage level; no genome is
edited.

## 2. Sampling

From each replicate's **living** population at the final tick, draw `{samples}`
genomes **uniformly at random, with replacement if the population is smaller**.
No ranking by energy, intake, age or any other quantity: ranking would confound
the assay with the selection that already happened.

Control genomes: `{samples}` genomes per replicate drawn from the same founder
distribution as everyone else (`Beta(2, 2)` trait loci, `N(0, 0.35)` neural
weights, `N(0, 0.05)` biases).

## 3. Arena

For each evolved/control pair, one arena contains **both** individuals:

* identical plant layout in every arena of a replicate (one arena seed; the
  layout is shared, so neither genome can be paired with a luckier patch),
* mirrored starting positions about the arena centre,
* equal starting reserves as a fraction of each individual's own `E_max`,
* no reproduction, no predation, no mutation; the arena horizon above.

The primary endpoint is a within-arena difference, so plant luck cancels.

## 4. Endpoints

* **Primary:** total food intake in biomass units (plant + meat; meat is zero
  without predation) over the arena horizon.
* **Secondary:** final energy, survival (alive at the final tick).

## 5. Analysis

Paired differences `intake(evolved) - intake(random)` per arena. Two-sided
**Wilcoxon signed-rank** test (normal approximation, tie-corrected, zero
differences dropped) and the **mean difference with a percentile bootstrap
95 % CI** (10,000 resamples), reported per replicate and pooled over all
arenas. No alternative endpoint, subgroup or exclusion rule will be added
after seeing the data.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _arena_config(base: Config, arena_ticks: int) -> Config:
    """Fixed 200 x 200 arena at the calibrated plant density."""
    n_cells = max(4, int(round(ARENA_SIZE / 25.0)))
    plants = max(10, int(round(ARENA_PLANT_DENSITY * ARENA_SIZE * ARENA_SIZE)))
    return replace(
        base,
        ticks=int(arena_ticks),
        world=replace(base.world, width=ARENA_SIZE, height=ARENA_SIZE,
                      n_field_cells=n_cells),
        plants=replace(base.plants, initial=plants, max=plants * 2,
                       germination_per_tick=0),
        organisms=replace(base.organisms, initial=2, max=2),
        environment=replace(base.environment, events=()),
    )


def _run_arena(factory: Callable[[Config], Engine], config: Config,
               evolved: np.ndarray, control: np.ndarray,
               position_rng: Rng) -> dict[str, Any]:
    """Place one evolved and one control genome in a shared, mirrored arena."""
    engine = factory(config)
    pop = engine.population
    if pop.n_alive != 2:
        raise RuntimeError("arena must contain exactly two organisms")

    width, height = engine.world.width, engine.world.height
    x = float(position_rng.random()) * width
    y = float(position_rng.random()) * height
    evolved_pos = (x, y)
    control_pos = ((width - x) % width, (height - y) % height)

    pop.genome[0] = evolved
    pop.genome[1] = control
    pop.x[0], pop.y[0] = evolved_pos
    pop.x[1], pop.y[1] = control_pos
    pop.heading[0] = pop.heading[1] = 0.0
    phen = phenotype(pop.genome[:2])
    emax = 10.0 * phen[:, 0] ** 0.9
    pop.body[0], pop.body[1] = phen[0, 0], phen[1, 0]
    pop.energy[0], pop.energy[1] = 0.5 * emax[0], 0.5 * emax[1]
    engine._refresh_phen()
    rebaseline(engine)

    evolved_id, control_id = int(pop.id[0]), int(pop.id[1])
    layout = _plant_layout_hash(engine)

    for _ in range(int(config.ticks)):
        engine.step(reproduce=False, predation=False)

    def endpoint(org_id: int) -> tuple[float, float, bool]:
        slots = np.flatnonzero(pop.id == org_id)
        if slots.size == 0:
            return 0.0, 0.0, False
        slot = int(slots[0])
        intake = float(pop.plant_intake[slot] + pop.meat_intake[slot])
        energy = float(pop.energy[slot]) if pop.alive[slot] else 0.0
        return intake, energy, bool(pop.alive[slot])

    e_intake, e_energy, e_alive = endpoint(evolved_id)
    c_intake, c_energy, c_alive = endpoint(control_id)
    return {
        "evolved_pos": evolved_pos,
        "random_pos": control_pos,
        "evolved_intake": e_intake,
        "random_intake": c_intake,
        "evolved_energy": e_energy,
        "random_energy": c_energy,
        "evolved_alive": e_alive,
        "random_alive": c_alive,
        "plant_layout_hash": layout,
    }


def run_assay(out_dir: str, *, scale: float = DEFAULT_SCALE,
              replicates: int = 6, samples: int = 10,
              ticks: int | None = None, arena_ticks: int | None = None,
              seed: int | None = None,
              engine_factory: Callable[[Config], Engine] | None = None
              ) -> dict[str, Any]:
    """Execute assay G: pre-register, evolve, sample uniformly, run arenas."""
    factory = engine_factory or (lambda config: Engine(config))
    os.makedirs(out_dir, exist_ok=True)

    config = scaled_config(load_config(os.path.join(CONFIGS_DIR, ASSAY_CONFIG)),
                           scale, ticks)
    base_seed = int(config.seed if seed is None else seed)
    arena_ticks = (max(1, int(round(DESIGN_ARENA_TICKS * scale)))
                   if arena_ticks is None else int(arena_ticks))

    # 1. Pre-registration is written before any engine exists.
    _write_pre_registration(
        os.path.join(out_dir, "pre_registration.md"),
        scale=float(scale), replicates=int(replicates), samples=int(samples),
        evolution_ticks=int(config.ticks), arena_ticks=int(arena_ticks),
        base_seed=base_seed)

    arena_cfg = _arena_config(config, arena_ticks)
    arenas: list[dict[str, Any]] = []
    evolution: list[dict[str, Any]] = []

    for rep in range(int(replicates)):
        rep_seed = base_seed + rep
        rep_cfg = replace(config, seed=rep_seed)
        engine = factory(rep_cfg)
        clumped_plants(engine, scale)
        rebaseline(engine)
        for _ in range(int(rep_cfg.ticks)):
            engine.step(predation=False)

        pop = engine.population
        survivors = np.flatnonzero(pop.alive)
        if survivors.size == 0:
            evolution.append({"replicate": rep, "seed": rep_seed,
                              "survivors": 0, "sampled": 0})
            continue

        # 2. Uniform sampling from survivors, no ranking.
        sample_rng = Rng(rep_seed + 17)
        picks = sample_rng.choice(survivors, size=int(samples), replace=True)
        evolved_genomes = pop.genome[picks].copy()
        # Control genomes from the same founder distribution.
        control_rng = Rng(rep_seed + 5000)
        control_genomes = np.asarray(founders(int(samples), control_rng),
                                     dtype=np.float64)

        evolution.append({"replicate": rep, "seed": rep_seed,
                          "survivors": int(survivors.size),
                          "sampled": int(samples)})

        # 3. Paired arenas: one evolved vs one control per arena.
        for pair in range(int(samples)):
            position_rng = Rng(rep_seed * 1000 + pair + 1)
            arena = _run_arena(factory, arena_cfg, evolved_genomes[pair],
                               control_genomes[pair], position_rng)
            arena.update({"replicate": rep, "pair": pair,
                          "diff": arena["evolved_intake"]
                          - arena["random_intake"]})
            arenas.append(arena)

    # 4. Analysis: per replicate and pooled.
    diff = np.asarray([a["diff"] for a in arenas], dtype=np.float64)
    analysis_rng = Rng(base_seed + 777)
    pooled = _analyse(diff, analysis_rng, prefix="pooled")
    per_replicate = []
    for rep in range(int(replicates)):
        d = np.asarray([a["diff"] for a in arenas if a["replicate"] == rep],
                       dtype=np.float64)
        # A replicate whose evolution phase went extinct contributes no arenas;
        # _analyse reports n = 0 rather than inventing a result.
        stats = _analyse(d, Rng(base_seed + 1000 + rep), prefix="")
        stats["replicate"] = rep
        per_replicate.append(stats)

    csv_path = os.path.join(out_dir, "assay.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["replicate", "pair", "evolved_pos_x", "evolved_pos_y",
                         "random_pos_x", "random_pos_y", "evolved_intake",
                         "random_intake", "diff", "evolved_energy",
                         "random_energy", "evolved_alive", "random_alive",
                         "plant_layout_hash"])
        for a in arenas:
            writer.writerow([
                a["replicate"], a["pair"],
                f"{a['evolved_pos'][0]:.6f}", f"{a['evolved_pos'][1]:.6f}",
                f"{a['random_pos'][0]:.6f}", f"{a['random_pos'][1]:.6f}",
                f"{a['evolved_intake']:.9f}", f"{a['random_intake']:.9f}",
                f"{a['diff']:.9f}", f"{a['evolved_energy']:.9f}",
                f"{a['random_energy']:.9f}", int(a["evolved_alive"]),
                int(a["random_alive"]), a["plant_layout_hash"]])

    summary = _assay_markdown(out_dir, scale=scale, replicates=replicates,
                              samples=samples, evolution=evolution,
                              arena_ticks=arena_ticks, config=config,
                              arenas=arenas, pooled=pooled,
                              per_replicate=per_replicate)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)

    return {
        "arenas": arenas,
        "evolution": evolution,
        "pooled": pooled,
        "per_replicate": per_replicate,
        "arena_world": (ARENA_SIZE, ARENA_SIZE),
        "scale": float(scale),
        "replicates": int(replicates),
        "samples": int(samples),
    }


def _analyse(diffs: np.ndarray, rng: Rng, prefix: str) -> dict[str, Any]:
    test = wilcoxon_signed_rank(diffs)
    mean, lo, hi = bootstrap_ci(diffs, rng)
    out = {
        "n": int(diffs.size),
        "mean": float(mean),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "median": float(np.median(diffs)) if diffs.size else 0.0,
        "w_plus": test["w_plus"],
        "z": test["z"],
        "p": test["p"],
    }
    if prefix:
        out["prefix"] = prefix
    return out


def _assay_markdown(out_dir: str, *, scale: float, replicates: int,
                    samples: int, evolution: list[dict[str, Any]],
                    arena_ticks: int, config: Config,
                    arenas: list[dict[str, Any]], pooled: dict[str, Any],
                    per_replicate: list[dict[str, Any]]) -> str:
    run_label = ("this scaled run" if scale < 1.0
                 else "this design-scale run")
    sign = "higher" if pooled["mean"] > 0 else "lower"
    if pooled["n"] == 0:
        conclusion = ("no arenas were measured (every replicate's evolution "
                      "phase went extinct before sampling), so the assay "
                      "produced no data")
    elif pooled["ci_low"] <= 0.0 <= pooled["ci_high"]:
        conclusion = (f"the 95 % bootstrap CI includes zero, so {run_label} "
                      "does not establish a difference")
    else:
        conclusion = (f"the 95 % bootstrap CI excludes zero; evolved genomes "
                      f"had {sign} intake in {run_label}")
    lines = [
        "# Assay G — evolved vs random (paired arenas)",
        "",
        "Protocol: `pre_registration.md` in this directory, written before any "
        "engine was constructed.",
        "",
        "## Run",
        "",
        f"- scale: `{scale}` (design scale is `1.0`)",
        f"- evolution: {replicates} replicates of {config.ticks} ticks "
        f"(assay config, asexual, no predators, clumped plants)",
        f"- sampled genomes: {samples} per replicate from living survivors, "
        "uniformly at random, no ranking",
        f"- arenas: {len(arenas)} paired arenas, {arena_ticks} ticks each, "
        f"{ARENA_SIZE:.0f} x {ARENA_SIZE:.0f} world at the calibrated plant "
        "density, mirrored positions, identical plant layout",
        "",
        "## Evolution phase",
        "",
        "| Replicate | Seed | Living survivors at final tick | Sampled |",
        "|---|---|---|---|",
    ]
    for ev in evolution:
        lines.append(f"| {ev['replicate']} | {ev['seed']} | "
                     f"{ev['survivors']} | {ev['sampled']} |")
    lines += [
        "",
        "## Primary endpoint: food intake (bu)",
        "",
        "| Set | n pairs | mean diff (evolved - random) | 95 % CI | "
        "median diff | W+ | z | p (Wilcoxon) |",
        "|---|---|---|---|---|---|---|---|",
        f"| pooled | {pooled['n']} | {pooled['mean']:+.4f} | "
        f"[{pooled['ci_low']:+.4f}, {pooled['ci_high']:+.4f}] | "
        f"{pooled['median']:+.4f} | {pooled['w_plus']:.1f} | "
        f"{pooled['z']:+.3f} | {pooled['p']:.4f} |",
    ]
    for stats in per_replicate:
        lines.append(
            f"| replicate {stats['replicate']} | {stats['n']} | "
            f"{stats['mean']:+.4f} | [{stats['ci_low']:+.4f}, "
            f"{stats['ci_high']:+.4f}] | {stats['median']:+.4f} | "
            f"{stats['w_plus']:.1f} | {stats['z']:+.3f} | "
            f"{stats['p']:.4f} |")
    alive_e = sum(1 for a in arenas if a["evolved_alive"])
    alive_r = sum(1 for a in arenas if a["random_alive"])
    lines += [
        "",
        "## Secondary endpoints",
        "",
        f"- survival at the final arena tick: evolved {alive_e}/{len(arenas)}, "
        f"random {alive_r}/{len(arenas)}",
        f"- mean final energy: evolved "
        f"{np.mean([a['evolved_energy'] for a in arenas]):.4f}, random "
        f"{np.mean([a['random_energy'] for a in arenas]):.4f}",
        "",
        "## Conclusion (as pre-registered)",
        "",
        f"Mean paired difference in intake: **{pooled['mean']:+.4f} bu** "
        f"(95 % CI [{pooled['ci_low']:+.4f}, {pooled['ci_high']:+.4f}]), "
        f"Wilcoxon p = {pooled['p']:.4f} on {pooled['n']} non-zero pairs. "
        f"Under the pre-registered reading, {conclusion}.",
        "",
        ("This is a scaled-down run: smaller world and shorter horizons than "
         "DESIGN §13, so it is a feasibility demonstration, not evidence at "
         "design scale. " if scale < 1.0 else
         "This is a design-scale run (horizon, arena length and plant density "
         "as specified). "),
        "The result is reported as measured, including any null result.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Whole programme
# ---------------------------------------------------------------------------
def run_all(out_dir: str, *, scale: float = DEFAULT_SCALE,
            ticks: int | None = None,
            engine_factory: Callable[[Config], Engine] | None = None,
            assay_replicates: int = 6, assay_samples: int = 10,
            assay_arena_ticks: int | None = None) -> dict[str, Any]:
    """Run A-F, H and the assay G, writing one directory per experiment."""
    os.makedirs(out_dir, exist_ok=True)
    results: dict[str, Any] = {}
    for key in ("a", "b", "c", "d", "e", "f"):
        results[key] = run_scenario(key, os.path.join(out_dir, key),
                                    scale=scale, ticks=ticks,
                                    engine_factory=engine_factory)
    results["g"] = run_assay(os.path.join(out_dir, "g"), scale=scale,
                             replicates=assay_replicates,
                             samples=assay_samples, ticks=ticks,
                             arena_ticks=assay_arena_ticks,
                             engine_factory=engine_factory)
    results["h"] = run_scenario("h", os.path.join(out_dir, "h"),
                                scale=scale, ticks=ticks,
                                engine_factory=engine_factory)

    lines = [
        "# Experiment programme — index",
        "",
        f"Scale `{scale}` for every run (design scale is `1.0`). "
        "See each directory's `summary.md` and `docs/RESULTS.md`.",
        "",
        "| ID | Name | Exit | Final population | Births | Extinct |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("a", "b", "c", "d", "f"):
        r = results[key]
        lines.append(f"| {key.upper()} | {r['name']} | ticks {r['ticks']} | "
                     f"{r['final_population']} | {r['births']} | "
                     f"{'yes' if r['extinct'] else 'no'} |")
    h = results["h"]["arms"]
    lines.append(f"| H | Sexual reproduction | ticks {results['h']['ticks']} | "
                 f"sexual {h['sexual']['final_population']} / asexual "
                 f"{h['asexual']['final_population']} | "
                 f"sexual {h['sexual']['births']} / asexual "
                 f"{h['asexual']['births']} | "
                 f"sexual {'yes' if h['sexual']['extinct'] else 'no'} / "
                 f"asexual {'yes' if h['asexual']['extinct'] else 'no'} |")
    g = results["g"]["pooled"]
    lines.append(f"| G | Assay (evolved vs random) | {g['n']} paired arenas | "
                 f"diff {g['mean']:+.4f} bu | p = {g['p']:.4f} | — |")
    lines.append("")
    with open(os.path.join(out_dir, "index.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return results
