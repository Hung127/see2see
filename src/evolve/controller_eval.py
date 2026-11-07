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

Because the founders are random genomes, the ``neural`` arm is *not* a trained
controller: it is the same random MLP the model ships with. Comparing it with
the baselines therefore measures "random network vs hand-written reflex", not
"evolution vs a reflex". That is stated in the report and is the reason the
experiment also supports a warm-up phase (``evolve_ticks``) that is documented
as a separate, more expensive condition rather than being silently folded in.

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
ENDPOINTS = (
    "final_population", "births", "deaths", "mean_energy",
    "mean_organism_biomass", "survival_median", "foraging_efficiency",
)


def _run_one(config: Config, kind: str, seed: int) -> dict[str, Any]:
    """Run one controller on one seed and return its measured endpoints."""
    cfg = replace(config, seed=int(seed))
    engine = Engine(cfg)
    if kind != "neural":
        engine.controller = build_controller(kind, seed=seed)
    recorder = StatsRecorder()
    interval = max(1, int(cfg.stats.interval))
    while engine.tick < int(cfg.ticks):
        engine.step()
        if engine.tick % interval == 0:
            recorder.record(engine)
    # Always sample the final state, even when the horizon is not a multiple of
    # the interval, so every arm is measured at the same tick.
    row = recorder.record(engine)

    return {
        "controller": kind,
        "seed": int(seed),
        "final_population": int(engine.population.n_alive),
        "extinct": bool(engine.population.n_alive == 0),
        "births": int(engine.births_total),
        "deaths": int(engine.deaths_total),
        "mean_energy": float(row["mean_energy"]),
        "mean_organism_biomass": float(row["mean_organism_biomass"]),
        "survival_median": float(row["survival_median"]),
        "foraging_efficiency": float(row["foraging_efficiency"]),
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
             seeds: Iterable[int]) -> dict[str, Any]:
    """Paired comparison of each controller against ``neural`` on the same seed."""
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
            mean, lo, hi = bootstrap_ci(arr, Rng(_pair_seed(kind, endpoint)))
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


def run_controller_comparison(out_dir: str, *, controllers: Iterable[str] | None = None,
                              seeds: Iterable[int] = (11, 12, 13),
                              ticks: int | None = None,
                              config_path: str | None = None,
                              population: int | None = None,
                              engine_factory: Callable[[Config], Engine] | None = None
                              ) -> dict[str, Any]:
    """Run every controller on every seed and write the comparison report."""
    controllers = list(controllers) if controllers is not None else controller_names()
    unknown = [c for c in controllers if c not in controller_names()]
    if unknown:
        raise ValueError(f"unknown controllers {unknown}; known: {controller_names()}")
    if "neural" not in controllers:
        controllers.insert(0, "neural")

    factory = engine_factory or (lambda config: Engine(config))
    path = config_path or os.path.join(CONFIGS_DIR, DEFAULT_CONFIG)
    base = load_config(path)
    overrides: dict[str, Any] = {}
    if ticks is not None:
        overrides["ticks"] = int(ticks)
    if population is not None:
        overrides["organisms"] = replace(base.organisms, initial=int(population),
                                         max=int(population))
    if overrides:
        base = replace(base, **overrides)

    os.makedirs(out_dir, exist_ok=True)
    seeds = [int(s) for s in seeds]
    rows: list[dict[str, Any]] = []
    for kind in controllers:
        for seed in seeds:
            rows.append(_run_one(base, kind, seed))

    # Per-run table.
    columns = ("controller", "seed", *ENDPOINTS, "extinct", "checksum")
    csv_path = os.path.join(out_dir, "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r[c] for c in columns])

    analysis = _analyse(rows, controllers, seeds)
    summary = {
        "config": path,
        "ticks": int(base.ticks),
        "seeds": seeds,
        "controllers": controllers,
        "rows": rows,
        "analysis": analysis,
        "note": (
            "Founders use random genomes, so the neural arm is the shipped "
            "random MLP, not an evolved controller. Differences measure "
            "controller policies on an equal footing; they are not evidence "
            "that evolution produced a better controller."
        ),
    }
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(_markdown(summary))
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    seeds = summary["seeds"]
    rows = summary["rows"]
    lines = [
        "# Controller comparison",
        "",
        f"Configuration: `{os.path.basename(summary['config'])}`, "
        f"{summary['ticks']} ticks, seeds {seeds}.",
        "",
        "All controllers receive the same 16-channel observation and their "
        "outputs go through the same action decoder, so movement, feeding and "
        "reproduction costs are identical. Runs are paired by seed.",
        "",
        "## Measured endpoints",
        "",
        "| Controller | Seed | Final pop | Births | Deaths | Mean energy | "
        "Mean biomass | Median age at death | Foraging efficiency | Extinct |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['controller']} | {r['seed']} | {r['final_population']} | "
            f"{r['births']} | {r['deaths']} | {r['mean_energy']:.3f} | "
            f"{r['mean_organism_biomass']:.3f} | {r['survival_median']:.1f} | "
            f"{r['foraging_efficiency']:.4f} | {'yes' if r['extinct'] else 'no'} |")

    lines += [
        "",
        "## Paired differences vs `neural` (same seed)",
        "",
        "Mean difference, percentile bootstrap 95 % CI, and two-sided Wilcoxon "
        "signed-rank p. With few seeds the test has low power: a large p is "
        "*not* evidence that two controllers are equivalent.",
        "",
        "| Controller | Endpoint | n | Mean diff | 95 % CI | p |",
        "|---|---|---|---|---|---|",
    ]
    for kind, endpoints in summary["analysis"].items():
        for endpoint, s in endpoints.items():
            lines.append(
                f"| {kind} | {endpoint} | {s['n']} | {s['mean_diff']:+.4f} | "
                f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] | {s['p']:.4f} |")

    lines += [
        "",
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
        "",
    ]
    return "\n".join(lines)
