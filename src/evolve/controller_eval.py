"""Controller comparison experiment (upgrade, Phase 4).

Question: on the *same* initial world and the *same* founder genomes, how do the
shipped controllers compare on survival and reproduction?

Design, chosen so the comparison is fair:

* **Paired seeds.** Every controller is run on the same base seed, so the world,
  plant layout, founder genomes and every environmental draw are identical. The
  only thing that differs between arms is the decision function, which is the
  variable under test. Seeds are independent replicates, never cherry-picked.
* **Same observation and action contract.** All controllers see the 16-channel
  sensor vector and all outputs go through ``actions.decode``, so a baseline
  pays the same movement, feeding and reproduction costs as the network. A
  controller cannot gain an advantage by bypassing physics.
* **No hidden fitness.** Nothing scores a controller during the run. The
  endpoints are measured *outcomes* (who is alive, who reproduced), not rewards.
* **Honest baselines.** ``random`` is uniform output (information-free),
  ``passive`` is all-zero output (does nothing), ``rule`` and ``forage`` are
  hand-written reflex policies that use only the sensor vector.

Endpoints, all measured:

* ``final_population`` — organisms alive at the horizon (primary);
* ``extinct`` — whether the population reached zero;
* ``births`` / ``deaths`` — cumulative counters;
* ``mean_energy``, ``mean_organism_biomass`` — final condition of survivors;
* ``survival_median`` — median age at death from the bounded histogram;
* ``foraging_efficiency`` — intake per unit distance moved.

Because the founders are random genomes, the ``neural`` arm of the default
comparison is *not* a trained controller: it is the same random MLP the model
ships with. Comparing it with the baselines therefore measures "random network
vs hand-written reflex", not "evolution vs a reflex". That is stated in the
report, and it is why the harness also supports an optional **warm-up phase**
(``warmup_ticks``):

* the warm-up runs the ordinary model — the evolved MLP, no controller — for
  ``warmup_ticks`` on each seed, so selection acts on the population first;
* the resulting state is written to a snapshot and *every* arm is resumed from
  that same snapshot, so the arms still differ only in the decision function;
* statistics are reset at the resume point, so endpoints describe the
  evaluation window, not the warm-up (``births``/``deaths`` are reported as
  deltas from the resume).

This is a separate, more expensive condition, and it is never folded in
silently: with ``warmup_ticks = 0`` (the default) the comparison is exactly the
random-founder experiment, and the report says which condition was run.

Statistics: paired differences against the ``neural`` arm on the same seed,
with the same in-house Wilcoxon signed-rank test and percentile bootstrap CI
used by assay G. With few seeds the test has low power; the report says so and
does not convert a non-significant difference into a claim of equivalence.
"""
from __future__ import annotations

import csv
import os
from dataclasses import replace
from typing import Any, Callable, Iterable

import numpy as np

from .config import Config, load_config
from .controllers import build_controller, controller_names
from .engine import Engine
from .experiments import bootstrap_ci, wilcoxon_signed_rank
from .rng import Rng
from .stats import StatsRecorder

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs")
DEFAULT_CONFIG = "default.yaml"

#: Endpoints written to ``comparison.csv`` and analysed pairwise.
#:
#: The first seven were the original set; the rest were appended (IMPROVE §4.2)
#: when the engine gained cumulative run totals. Order is append-only so an old
#: CSV's columns keep their meaning.
ENDPOINTS = (
    "final_population", "births", "deaths", "mean_energy",
    "mean_organism_biomass", "survival_median", "foraging_efficiency",
    # -- appended for IMPROVE §4.2 -------------------------------------------
    "survival_rate",          # fraction of the cohort present at the start
    "mean_lifespan",          # mean age at death, evaluation window only
    "reproduction_success",   # births / reproduction attempts
    "energy_acquired",        # bu actually added to reserves (post-assimilation)
    "energy_efficiency",      # energy acquired per unit respired
    "movement_efficiency",    # energy acquired per world unit moved
    "feeding_success",        # bites that connected / feeding attempts
    "predation_success",      # damaging strikes / strikes
    "behavioural_diversity",  # spread of activity profiles (final sample)
    "environmental_stress",   # mean drought stress where the animals were
)


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or ``0.0`` when nothing was measured.

    A zero denominator means "no attempt was made", not "zero success rate", but
    the CSV is numeric; the report's endpoint table names each ratio so a 0.0
    from an empty window is not read as a measured failure.
    """
    return float(numerator) / float(denominator) if denominator else 0.0


def _make_engine(factory: Callable[[Config], Engine], cfg: Config,
                 scenario: Any, scale: float) -> Engine:
    """Construct one arm's starting engine, applying a scenario's setup hook.

    Setup hooks are *initial conditions*, so they run once at construction —
    including for the warm-up engine, which otherwise would evolve a world the
    evaluated arms never see. ``rebaseline`` is applied for the same reason as
    in ``experiments.run_scenario``: a hook that adds mass changes the ledger's
    reference point.
    """
    engine = factory(cfg)
    if scenario is not None and scenario.setup:
        from .experiments import HOOKS, rebaseline
        HOOKS[scenario.setup](engine, scale)
        rebaseline(engine)
    return engine


def _run_one(config: Config, kind: str, seed: int, *,
             warmup_snapshot: str | None = None,
             engine_factory: Callable[[Config], Engine] | None = None,
             scenario: Any = None,
             scale: float = 1.0
             ) -> dict[str, Any]:
    """Run one controller on one seed and return its measured endpoints.

    ``warmup_snapshot`` resumes a state produced by the no-controller warm-up,
    so every arm starts from the *same* evolved population and only the decision
    function differs. Statistics are primed from the resume point, so
    ``births``/``deaths`` and every run total count the evaluation window, not
    the warm-up.

    ``scenario`` (an ``experiments.Scenario``) supplies the declared
    manipulation of that scenario: its setup hook at construction, its step
    options every tick, and its intervention at the scheduled tick. ``None``
    means the plain configuration and is the default path.
    """
    factory = engine_factory or (lambda config: Engine(config))
    cfg = replace(config, seed=int(seed))
    if warmup_snapshot is None:
        engine = _make_engine(factory, cfg, scenario, scale)
        baseline_births = baseline_deaths = 0
        baseline_totals = {name: 0.0 for name in engine.run_totals}
    else:
        from .persistence import load_snapshot
        engine = load_snapshot(warmup_snapshot, cfg)
        baseline_births = int(engine.births_total)
        baseline_deaths = int(engine.deaths_total)
        baseline_totals = {name: float(value)
                           for name, value in engine.run_totals.items()}
    start_population = int(engine.population.n_alive)
    # ``neural`` means "whatever the model ships with", so an evaluation arm
    # always clears any controller a previous caller may have attached.
    engine.controller = (None if kind == "neural"
                         else build_controller(kind, seed=seed))
    recorder = StatsRecorder()
    recorder.prime(engine)
    interval = max(1, int(cfg.stats.interval))
    options = scenario.step_options() if scenario is not None else {}
    intervention = getattr(scenario, "intervention", None) if scenario else None
    intervention_tick = (
        int(round(scenario.intervention_tick * scale))
        if intervention and scenario.intervention_tick else None)
    while engine.tick < int(cfg.ticks):
        if intervention_tick is not None and engine.tick == intervention_tick - 1:
            from .experiments import HOOKS
            HOOKS[intervention](engine, scale)
        engine.step(**options)
        if engine.tick % interval == 0:
            recorder.record(engine)
    # Always sample the final state, even when the horizon is not a multiple of
    # the interval, so every arm is measured at the same tick.
    row = recorder.record(engine)

    births = int(engine.births_total) - baseline_births
    deaths = int(engine.deaths_total) - baseline_deaths
    total = {name: float(value) - baseline_totals.get(name, 0.0)
             for name, value in engine.run_totals.items()}

    # Survivorship over the evaluation window: animals alive at the horizon as a
    # fraction of every animal that existed in the window (starters + births).
    existed = start_population + births
    survival_rate = _ratio(int(engine.population.n_alive), existed)
    # Mean age at death. The age is the animal's own age, so with a warm-up this
    # is age-since-birth, not time spent in the window; the report says so.
    mean_lifespan = _ratio(total["death_age_sum"], deaths)

    return {
        "controller": kind,
        "seed": int(seed),
        "final_population": int(engine.population.n_alive),
        "extinct": bool(engine.population.n_alive == 0),
        "births": births,
        "deaths": deaths,
        "mean_energy": float(row["mean_energy"]),
        "mean_organism_biomass": float(row["mean_organism_biomass"]),
        "survival_median": float(row["survival_median"]),
        "foraging_efficiency": float(row["foraging_efficiency"]),
        # -- IMPROVE §4.2 endpoints, all from cumulative run totals -----------
        "survival_rate": survival_rate,
        "mean_lifespan": mean_lifespan,
        "reproduction_success": _ratio(total["repro_successes"],
                                       total["repro_attempts"]),
        "energy_acquired": total["energy_gain"],
        "energy_efficiency": _ratio(total["energy_gain"],
                                    total["metabolic_cost"]),
        "movement_efficiency": _ratio(total["energy_gain"], total["distance"]),
        "feeding_success": _ratio(total["feed_successes"],
                                  total["feed_attempts"]),
        "predation_success": _ratio(total["strikes_hit"], total["strikes"]),
        "behavioural_diversity": float(row["behavioural_diversity"]),
        "environmental_stress": float(row["environmental_stress"]),
        "checksum": engine.checksum(),
    }


def _pair_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed for a bootstrap stream.

    ``hash()`` on strings is salted per process, so it cannot be used here: the
    comparison report has to be byte-reproducible across invocations. A small
    FNV-1a-style fold of the ``repr`` is stable and dependency-free.
    """
    value = 0x811C9DC5
    for part in parts:
        for byte in repr(part).encode("utf-8"):
            value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value & 0x7FFFFFFF


def _analyse(rows: list[dict[str, Any]], controllers: Iterable[str],
             seeds: Iterable[int], scenario: str | None = None
             ) -> dict[str, Any]:
    """Paired comparison of each controller against ``neural`` on the same seed.

    ``scenario`` filters the rows to one declared experiment; ``None`` means the
    plain configuration, so a single-configuration run keeps the flat shape.
    """
    if scenario is not None:
        rows = [r for r in rows if r.get("scenario", "") == scenario]
    by_key = {(r["controller"], r["seed"]): r for r in rows}
    seeds = list(seeds)
    reference = "neural"
    out: dict[str, Any] = {}
    for kind in controllers:
        if kind == reference:
            continue
        per_endpoint: dict[str, Any] = {}
        for endpoint in ENDPOINTS:
            pairs = [(kind, s) for s in seeds
                     if (kind, s) in by_key and (reference, s) in by_key]
            diffs = [by_key[(kind, s)][endpoint] - by_key[(reference, s)][endpoint]
                     for kind, s in pairs]
            arr = np.asarray(diffs, dtype=np.float64)
            mean, lo, hi = bootstrap_ci(
                arr, Rng(_pair_seed(scenario or "", kind, endpoint)))
            test = wilcoxon_signed_rank(arr)
            per_endpoint[endpoint] = {
                "n": int(arr.size),
                "mean_diff": mean,
                "ci_low": lo,
                "ci_high": hi,
                "p": test["p"],
            }
        out[kind] = per_endpoint
    return out


def _robustness(rows: list[dict[str, Any]], controllers: Iterable[str],
                scenarios: Iterable[str]) -> dict[str, Any]:
    """Per-controller summary *across* scenarios: IMPROVE §4.2 robustness.

    Three declared numbers, each with its own meaning; none of them is a hidden
    fitness score:

    * ``persistence_rate`` — fraction of scenario × seed runs in which the
      population was still non-zero at the horizon;
    * ``mean_survival_rate`` — mean of the per-run survival rates;
    * ``worst_survival_rate`` — the minimum, i.e. the scenario that hurt that
      controller most.

    A controller can lead on the mean and still be the worst in one environment;
    reporting the minimum is what keeps that visible.
    """
    keys = [str(s).lower() for s in scenarios]
    out: dict[str, Any] = {}
    for kind in controllers:
        cells = [r for r in rows if r["controller"] == kind
                 and r.get("scenario", "") in keys]
        rates = [float(r["survival_rate"]) for r in cells]
        persisted = [0.0 if r["extinct"] else 1.0 for r in cells]
        repro = [float(r["reproduction_success"]) for r in cells]
        out[kind] = {
            "scenarios": len(keys),
            "runs": len(cells),
            "persistence_rate": (float(np.mean(persisted)) if cells else 0.0),
            "mean_survival_rate": (float(np.mean(rates)) if rates else 0.0),
            "worst_survival_rate": (float(np.min(rates)) if rates else 0.0),
            "mean_reproduction_success": (float(np.mean(repro)) if repro else 0.0),
        }
    return out


def run_controller_comparison(out_dir: str, *, controllers: Iterable[str] | None = None,
                              seeds: Iterable[int] = (11, 12, 13),
                              ticks: int | None = None,
                              config_path: str | None = None,
                              population: int | None = None,
                              warmup_ticks: int = 0,
                              scenarios: Iterable[str] | None = None,
                              scale: float = 1.0,
                              engine_factory: Callable[[Config], Engine] | None = None
                              ) -> dict[str, Any]:
    """Run every controller on every seed and write the comparison report.

    ``warmup_ticks`` > 0 runs the ordinary evolved model for that many ticks on
    each seed first, then resumes every arm from that one snapshot. That is the
    "evolution happened, now evaluate the controllers" condition; it is off by
    default so the cheap comparison stays cheap and honest.

    ``scenarios`` names declared experiments (``experiments.SCENARIOS``); each
    controller then runs on *every* scenario × seed, so a controller that only
    wins in the benign default is visible as such. ``None`` (the default) runs
    the single plain configuration and keeps the report and CSV shape it has
    always had. ``scale`` only applies to scenarios, which are design-scale
    configs and must be scaled like every other experiment run.
    """
    controllers = list(controllers) if controllers is not None else controller_names()
    unknown = [c for c in controllers if c not in controller_names()]
    if unknown:
        raise ValueError(f"unknown controllers {unknown}; known: {controller_names()}")
    if "neural" not in controllers:
        controllers.insert(0, "neural")

    factory = engine_factory or (lambda config: Engine(config))
    path = config_path or os.path.join(CONFIGS_DIR, DEFAULT_CONFIG)
    base = load_config(path)
    scenario_keys = ([str(s).lower() for s in scenarios]
                     if scenarios is not None else [])
    if scenario_keys:
        from .experiments import SCENARIOS, scenario_config

        unknown_scenarios = [s for s in scenario_keys if s not in SCENARIOS
                             and s != "default"]
        if unknown_scenarios:
            raise ValueError(
                f"unknown scenarios {unknown_scenarios}; known: "
                f"{sorted(SCENARIOS) + ['default']}")

    overrides: dict[str, Any] = {}
    if ticks is not None:
        overrides["ticks"] = int(ticks)
    if population is not None:
        overrides["organisms"] = replace(base.organisms, initial=int(population),
                                         max=int(population))
    if overrides:
        base = replace(base, **overrides)

    # Build the run list: one (label, config, scenario) cell per environment.
    # A scenario's own YAML and scaling win; the CLI overrides (ticks,
    # population) are applied on top so every cell is measured comparably.
    cells: list[tuple[str, Config, Any]] = []
    for key in (scenario_keys or ["default"]):
        if key == "default":
            cells.append(("default", base, None))
            continue
        cfg, scenario = scenario_config(key, scale=scale, ticks=ticks)
        cfg = replace(cfg, seed=base.seed)
        if population is not None:
            cfg = replace(cfg, organisms=replace(
                cfg.organisms, initial=int(population),
                max=int(population)))
        cells.append((key, cfg, scenario))

    warmup = max(0, int(warmup_ticks or 0))
    if warmup:
        shortest = min(int(cfg.ticks) for _, cfg, _ in cells)
        if warmup >= shortest:
            raise ValueError(
                f"warmup_ticks ({warmup}) must be shorter than the shortest "
                f"horizon ({shortest}); the evaluation window would otherwise "
                "be empty")

    os.makedirs(out_dir, exist_ok=True)
    seeds = [int(s) for s in seeds]
    rows: list[dict[str, Any]] = []
    warmups: dict[tuple[str, int], str] = {}
    if warmup:
        # The warm-up is the *model*, not a controller: no baseline is allowed
        # to shape the population it will later be judged on.
        from .persistence import save_snapshot
        for label, cell_cfg, cell_scenario in cells:
            for seed in seeds:
                cfg = replace(cell_cfg, seed=int(seed))
                pre = _make_engine(factory, cfg, cell_scenario, scale)
                pre.controller = None
                pre.run(warmup)
                # The plain configuration keeps its historical filename; a
                # scenario run namespaces the snapshot by scenario so cells
                # cannot overwrite each other.
                name = (f"warmup_seed{seed}.npz" if label == "default"
                        else f"warmup_{label}_seed{seed}.npz")
                snap = os.path.join(out_dir, name)
                save_snapshot(pre, snap)
                warmups[(label, seed)] = snap
    for label, cell_cfg, cell_scenario in cells:
        for kind in controllers:
            for seed in seeds:
                row = _run_one(cell_cfg, kind, seed,
                               warmup_snapshot=warmups.get((label, seed)),
                               engine_factory=factory,
                               scenario=cell_scenario, scale=scale)
                row["scenario"] = label
                rows.append(row)

    multi = len(cells) > 1
    # Per-run table. The original column order is preserved; ``scenario`` is
    # only inserted when there is more than one, so old readers keep working.
    columns = (("scenario",) if multi else ()) + (
        "controller", "seed", *ENDPOINTS, "extinct", "checksum")
    csv_path = os.path.join(out_dir, "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r[c] for c in columns])

    if multi:
        analysis = {label: _analyse(rows, controllers, seeds, scenario=label)
                    for label, _, _ in cells}
        robustness = _robustness(rows, controllers, [c[0] for c in cells])
    else:
        analysis = _analyse(rows, controllers, seeds)
        robustness = None
    summary = {
        "config": path,
        "ticks": int(base.ticks),
        "seeds": seeds,
        "controllers": controllers,
        "warmup_ticks": warmup,
        "scenarios": [c[0] for c in cells],
        "scale": float(scale) if scenario_keys else 1.0,
        "rows": rows,
        "analysis": analysis,
        "robustness": robustness,
        "note": (
            ("Founders use random genomes, so the neural arm is the shipped "
             "random MLP, not an evolved controller. Differences measure "
             "controller policies on an equal footing; they are not evidence "
             "that evolution produced a better controller.")
            if not warmup else
            (f"Every arm was resumed from the same {warmup}-tick evolved "
             "population on its seed, so the neural arm is the model's own "
             "MLP *after* selection, not a random founder network. Births and "
             "deaths are counted from the resume point (the evaluation "
             "window).")
        ),
    }
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(_markdown(summary))
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    seeds = summary["seeds"]
    rows = summary["rows"]
    warmup = int(summary.get("warmup_ticks", 0) or 0)
    condition = ("random founders" if not warmup
                 else f"{warmup}-tick evolved population, then evaluation")
    scenarios = list(summary.get("scenarios") or ["default"])
    multi = len(scenarios) > 1
    scenario_column = ("scenario",) if multi else ()
    lines = [
        "# Controller comparison",
        "",
        f"Configuration: `{os.path.basename(summary['config'])}`, "
        f"{summary['ticks']} ticks, seeds {seeds}.",
        f"Condition: {condition}.",
        f"Scenarios: {', '.join(f'`{s}`' for s in scenarios)}"
        + (f" at scale {summary['scale']:g}" if multi else "") + ".",
        "",
        "All controllers receive the same 16-channel observation and their "
        "outputs go through the same action decoder, so movement, feeding and "
        "reproduction costs are identical. Runs are paired by seed; when there "
        "is more than one scenario, runs are paired by scenario and seed.",
        "",
        "## Measured endpoints",
        "",
        ("| Scenario | " if multi else "| ")
        + "Controller | Seed | Final pop | Births | Deaths | Mean energy | "
        "Mean biomass | Median age at death | Foraging efficiency | Extinct |",
        "|" + "---|" * (len(scenario_column) + 10),
    ]
    for r in rows:
        prefix = f"| {r['scenario']} " if multi else "| "
        lines.append(
            prefix
            + f"| {r['controller']} | {r['seed']} | {r['final_population']} | "
            f"{r['births']} | {r['deaths']} | {r['mean_energy']:.3f} | "
            f"{r['mean_organism_biomass']:.3f} | {r['survival_median']:.1f} | "
            f"{r['foraging_efficiency']:.4f} | {'yes' if r['extinct'] else 'no'} |")

    lines += [
        "",
        "### Survival, reproduction and efficiency (IMPROVE §4.2)",
        "",
        "All rates are computed from cumulative run totals over the evaluation "
        "window, so they include animals that have since died. `0.000` means "
        "the denominator was zero in this window (no attempt, no strike, no "
        "movement) — not a measured zero.",
        "",
        ("| Scenario | " if multi else "| ")
        + "Controller | Seed | Survival rate | Mean lifespan | Repro success | "
        "Energy acquired | Energy efficiency | Movement efficiency | "
        "Feeding success | Predation success | Behavioural diversity | "
        "Env. stress |",
        "|" + "---|" * (len(scenario_column) + 12),
    ]
    for r in rows:
        prefix = f"| {r['scenario']} " if multi else "| "
        lines.append(
            prefix
            + f"| {r['controller']} | {r['seed']} | {r['survival_rate']:.3f} | "
            f"{r['mean_lifespan']:.1f} | {r['reproduction_success']:.3f} | "
            f"{r['energy_acquired']:.2f} | {r['energy_efficiency']:.4f} | "
            f"{r['movement_efficiency']:.5f} | {r['feeding_success']:.3f} | "
            f"{r['predation_success']:.3f} | "
            f"{r['behavioural_diversity']:.4f} | "
            f"{r['environmental_stress']:.4f} |")

    lines += [
        "",
        "Definitions: survival rate = animals alive at the horizon / (starters "
        "+ births); mean lifespan = mean age at death; reproduction success = "
        "births / reproduction attempts; energy acquired = bu added to reserves "
        "after assimilation; energy efficiency = energy acquired / energy "
        "respired; movement efficiency = energy acquired / world units moved; "
        "feeding success = bites in reach / feeding attempts; predation success "
        "= damaging strikes / strikes; behavioural diversity and environmental "
        "stress are the final sampled statistics (definitions in `stats.py`).",
        "",
    ]

    if multi:
        lines += [
            "## Robustness across scenarios",
            "",
            "Three declared numbers per controller, none of them a hidden "
            "fitness score: `persistence_rate` is the fraction of "
            "scenario × seed runs still non-zero at the horizon, "
            "`mean_survival_rate` averages the per-run survival rates, and "
            "`worst_survival_rate` is the minimum (the environment that hurt "
            "that controller most).",
            "",
            "| Controller | Scenarios | Runs | Persistence rate | "
            "Mean survival rate | Worst survival rate | "
            "Mean reproduction success |",
            "|---|---|---|---|---|---|---|",
        ]
        for kind, s in (summary.get("robustness") or {}).items():
            lines.append(
                f"| {kind} | {s['scenarios']} | {s['runs']} | "
                f"{s['persistence_rate']:.3f} | {s['mean_survival_rate']:.3f} | "
                f"{s['worst_survival_rate']:.3f} | "
                f"{s['mean_reproduction_success']:.3f} |")
        lines.append("")

    if multi:
        lines += [
            "## Paired differences vs `neural` (same scenario and seed)",
            "",
        ]
        analysis_blocks = [(label, summary["analysis"][label])
                           for label in scenarios]
    else:
        lines += [
            "## Paired differences vs `neural` (same seed)",
            "",
        ]
        analysis_blocks = [(None, summary["analysis"])]

    lines += [
        "Mean difference, percentile bootstrap 95 % CI, and two-sided Wilcoxon "
        "signed-rank p. With few seeds the test has low power: a large p is "
        "*not* evidence that two controllers are equivalent.",
        "",
    ]
    for label, analysis in analysis_blocks:
        if label is not None:
            lines += [f"### Scenario `{label}`", ""]
        lines += [
            "| Controller | Endpoint | n | Mean diff | 95 % CI | p |",
            "|---|---|---|---|---|---|",
        ]
        for kind, endpoints in analysis.items():
            for endpoint, s in endpoints.items():
                lines.append(
                    f"| {kind} | {endpoint} | {s['n']} | {s['mean_diff']:+.4f} | "
                    f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] | {s['p']:.4f} |")
        lines.append("")

    lines += [
        "## Interpretation limits",
        "",
        f"- {summary['note']}",
        "- The horizon is short relative to a full evolutionary run; these are "
        "short-horizon survival and reproduction outcomes, not fitness in the "
        "design-scale experiment.",
        "- Seeds are few; intervals are wide and the tests are underpowered. "
        "Null results are reported as null, not as proof of equivalence.",
        "- A controller that survives longer but reproduces less is not "
        "'better' by definition: population and births are reported separately "
        "rather than combined into a single score.",
    ]
    if multi:
        lines.append(
            "- Scenarios are declared manipulations, not a representative "
            "sample of environments. The robustness table describes these "
            "scenarios only; it is not a generalisation to unseen worlds.")
    lines.append("")
    return "\n".join(lines)
