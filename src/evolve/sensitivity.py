"""One-factor-at-a-time sensitivity sweeps (upgrade, IMPROVE §3.4 / §9).

A sweep answers one question: *how much do the measured outcomes move when this
one parameter moves, with everything else held fixed?* It is deliberately not an
optimiser and not a calibration tool — nothing here writes back a default. The
calibrated values remain those in ``config.py``.

Each sweep declares, as the plan requires:

* **hypothesis** — the directional expectation, written before the run;
* **independent variable** — the single dotted config field and its values;
* **controlled variables** — the base config is copied, never mutated; every
  seed sees the same base, so runs are paired across values;
* **dependent variables** — :data:`ENDPOINTS` (measured outcomes, not rewards);
* **seeds / duration** — recorded in the report;
* **limitations** — written into the report, including that a short scaled run
  is a sensitivity probe, not the design-scale experiment.

Statistical handling follows the rest of the project: per-value means and
standard deviations across seeds, paired differences against the sweep's
baseline value, percentile bootstrap CI and the in-house Wilcoxon signed-rank
test. With few seeds the test is underpowered and the report says so. A null is
reported as a null; it is never converted into "no effect".
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import replace
from typing import Any, Iterable, Mapping

import numpy as np

from .config import Config, load_config
from .engine import Engine
from .experiments import (
    DEFAULT_SCALE, bootstrap_ci, scaled_config, wilcoxon_signed_rank,
)
from .genome import GENE_SPECS, TRAIT_NAMES, phenotype
from .rng import Rng
from .stats import StatsRecorder, bottleneck

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs")
DEFAULT_CONFIG = "default.yaml"

#: Measured outcomes compared across parameter values. Append-only, so a CSV
#: written by an earlier version still parses by name.
ENDPOINTS = (
    "final_population", "extinct", "births", "deaths",
    "mean_energy", "mean_organism_biomass", "survival_median",
    "mean_pairwise", "trait_variance", "population_cv",
    # IMPROVE §5: the contraction indicator, so a sweep can show whether a
    # setting drives the population into a bottleneck.
    "population_drop", "diversity_drop", "bottleneck",
)


class Sweep:
    """One declared one-factor sweep.

    The independent variable is either a config field (``parameter`` is a
    dotted ``section.field`` name) or a **genetic locus** (``trait`` names one of
    :data:`evolve.genome.TRAIT_NAMES`, and ``parameter`` is ``trait.<name>`` for
    reporting). A trait sweep sets the locus in every founder genome before the
    first tick, which is why the physical value is converted back to a gene
    value: the sweep varies what the animal is born with, not a global
    coefficient.
    """

    __slots__ = ("id", "parameter", "values", "baseline", "hypothesis",
                 "rationale", "expectation", "trait")

    def __init__(self, id: str, parameter: str, values: Iterable[float],
                 baseline: float, hypothesis: str, rationale: str,
                 expectation: str = "", trait: str | None = None) -> None:
        self.id = id
        self.parameter = parameter
        self.values = tuple(values)
        self.baseline = baseline
        self.hypothesis = hypothesis
        self.rationale = rationale
        self.expectation = expectation
        self.trait = trait
        if self.baseline not in self.values:
            raise ValueError(f"sweep {id!r}: baseline {baseline!r} not in values")
        if len(self.values) < 2:
            raise ValueError(f"sweep {id!r}: needs at least two values")
        if trait is not None and trait not in TRAIT_NAMES:
            raise ValueError(f"sweep {id!r}: unknown trait {trait!r}; "
                             f"known: {TRAIT_NAMES}")
        if trait is not None and parameter != f"trait.{trait}":
            raise ValueError(f"sweep {id!r}: trait sweeps must name "
                             f"'trait.{trait}', got {parameter!r}")

    @property
    def is_trait(self) -> bool:
        return self.trait is not None


#: Traits that may be varied at founding. ``body_size`` is excluded on purpose:
#: the founder reserve is derived from body mass at construction, so changing
#: mass afterwards would leave every animal's starting energy inconsistent with
#: its own body. The rest do not feed the founder reserve, so overriding them
#: before tick 1 is a clean initial-condition change.
TRAIT_SWEEP_ALLOWED = ("vision_range", "vision_angle", "metabolic_rate",
                       "attack", "defense", "repro_threshold",
                       "repro_investment", "diet_preference", "lifespan",
                       "mutation_rate", "speed_max")


def _trait_locus(trait: str, physical: float) -> float:
    """Invert the monotone gene→phenotype map for one trait.

    ``GENE_SPECS`` maps a locus ``g`` in [0, 1] to a physical value with a
    linear or log-linear curve. A sweep declares physical values (units a reader
    can reason about: units, bu/tick, mutations per locus per birth), so the
    value is mapped back to the locus that produces it. The result is clipped to
    [0, 1], and a value outside the gene's physical span raises rather than
    silently saturating: a sweep that cannot be expressed must fail loudly.
    """
    spec = next((s for s in GENE_SPECS if s.name == trait), None)
    if spec is None:
        raise ValueError(f"unknown trait {trait!r}")
    if spec.kind == "exp":
        if physical <= 0.0:
            raise ValueError(f"{trait}: a log-mapped trait needs a positive "
                             f"value, got {physical!r}")
        g = ((math.log(physical) - math.log(spec.lo))
             / (math.log(spec.hi) - math.log(spec.lo)))
    else:
        g = (float(physical) - spec.lo) / (spec.hi - spec.lo)
    if g < 0.0 or g > 1.0:
        raise ValueError(f"{trait}: {physical!r} lies outside the gene's "
                         f"physical span [{spec.lo}, {spec.hi}]")
    return float(g)


#: The declared sweeps. Values bracket the calibrated default; they are chosen
#: to be visibly different, not to be "best".
SWEEPS: dict[str, Sweep] = {
    "bite": Sweep(
        "bite", "energy.bite", (0.25, 0.50, 0.75, 1.00), 0.50,
        hypothesis=("A larger bite rate raises per-contact energy gain, so "
                    "population and mean energy should increase."),
        rationale=("bite is the only coefficient directly scaling plant intake, "
                   "so it is the cleanest test that the feeding channel is "
                   "behaving monotonically.")),
    "eta_p": Sweep(
        "eta_p", "energy.eta_p", (0.50, 0.75, 1.00), 0.75,
        hypothesis=("Higher plant assimilation efficiency lowers the intake "
                    "needed to break even, so population should rise."),
        rationale=("eta_p is the plant-side counterpart of eta_m; it tests "
                   "whether the energy ledger's efficiency term is wired.")),
    "soil_diffusion": Sweep(
        "soil_diffusion", "environment.soil_diffusion", (0.02, 0.10, 0.30), 0.10,
        hypothesis=("More soil diffusion should reduce stranded nutrients and "
                    "raise plant biomass, but too much may flatten the "
                    "resource gradient foragers rely on."),
        rationale=("CALIBRATION.md identifies nutrient transport as the "
                   "mechanism that keeps plant growth sustainable.")),
    "sigma_trait": Sweep(
        "sigma_trait", "genetics.sigma_trait", (0.02, 0.08, 0.20), 0.08,
        hypothesis=("Higher trait mutation rate raises trait variance but may "
                    "cost population if it breaks adapted combinations."),
        rationale=("Separates the diversity response from the population "
                   "response, which are often conflated.")),
    "maturity_age": Sweep(
        "maturity_age", "organisms.maturity_age", (25, 50, 100), 50,
        hypothesis=("Later maturity delays first reproduction, so births and "
                    "population should fall."),
        rationale=("A direct test of the reproduction schedule: the effect "
                   "should appear in births before it appears in population.")),
    "decay": Sweep(
        "decay", "energy.decay", (0.005, 0.015, 0.045), 0.015,
        hypothesis=("Faster decomposition returns nutrients to soil sooner, "
                    "which should raise plant biomass."),
        rationale=("Tests the detritus -> soil -> plant pathway end to end.")),
    # -- trait-level sweeps (IMPROVE §3.4): the independent variable is a
    # founder genome locus, not a config coefficient. Values are physical units
    # from GENE_SPECS, converted back to a locus before tick 1.
    "vision_range": Sweep(
        "vision_range", "trait.vision_range", (40.0, 100.0, 160.0), 100.0,
        hypothesis=("Wider vision should let animals find plants and threats "
                    "sooner, but it is not free: the vision term of the "
                    "metabolic cost grows with range squared, so the net "
                    "population effect may be flat or negative."),
        rationale=("Directly tests the sense-before-act premise: if vision does "
                   "nothing, the entire sensor channel is decorative."),
        expectation=("A rise in intake and a rise in metabolic cost together; "
                     "population alone is not the test, because the two "
                     "effects oppose."),
        trait="vision_range"),
    "metabolic_rate": Sweep(
        "metabolic_rate", "trait.metabolic_rate", (0.60, 1.00, 1.60), 1.00,
        hypothesis=("A faster metabolism raises the basal bill and digestion "
                    "scope at once; at the low end animals should starve more "
                    "slowly with a low ceiling, so the population response is "
                    "expected to be hump-shaped rather than monotone."),
        rationale=("The metabolic trade-off is the model's central energetic "
                   "constraint; sweeping the locus tests whether that "
                   "trade-off is actually load-bearing."),
        expectation=("Both extremes worse than the middle if the trade-off is "
                     "real; a monotone response would mean one side dominates."),
        trait="metabolic_rate"),
    "mutation_rate": Sweep(
        "mutation_rate", "trait.mutation_rate", (0.001, 0.010, 0.050), 0.010,
        hypothesis=("A higher mutation rate raises trait and network variance "
                    "but risks breaking adapted combinations, so diversity "
                    "should rise while population may fall."),
        rationale=("Complements `sigma_trait` (a config-level mutation scale) by "
                   "varying the per-organism locus that evolution itself can "
                   "adjust, which is the quantity under selection."),
        expectation=("Higher trait_variance and mean_pairwise at the top value; "
                     "no claim about population direction at this scale."),
        trait="mutation_rate"),
}


def _set_parameter(config: Config, dotted: str, value: Any) -> Config:
    """Return a copy of ``config`` with one dotted field replaced.

    The config tree is frozen, so this is a copy and the caller's object is
    never mutated. Unknown sections or fields raise rather than being ignored.
    """
    if "." not in dotted:
        raise ValueError(f"parameter {dotted!r} must be '<section>.<field>'")
    section, field_name = dotted.split(".", 1)
    current = getattr(config, section, None)
    if current is None:
        raise ValueError(f"unknown configuration section {section!r}")
    if not hasattr(current, field_name):
        raise ValueError(f"unknown field {field_name!r} in section {section!r}")
    # Preserve the declared type of the field (int fields must stay ints).
    existing = getattr(current, field_name)
    if isinstance(existing, bool):
        coerced: Any = bool(value)
    elif isinstance(existing, int):
        coerced = int(round(float(value)))
    else:
        coerced = float(value)
    return replace(config, **{section: replace(current, **{field_name: coerced})})


def _run_one(config: Config, seed: int, trait: str | None = None,
             trait_value: float | None = None) -> dict[str, Any]:
    """Run one engine to the horizon and return measured endpoints.

    When ``trait`` is given, every founder's locus for that trait is set to the
    gene value that maps to ``trait_value`` **before the first tick**, and the
    phenotype cache is refreshed. This is an initial-condition change, not a
    per-tick intervention: no animal is modified after tick 0, so the engine's
    RNG draw order is untouched and two runs remain paired by seed.

    ``body_size`` is not accepted here (see :data:`TRAIT_SWEEP_ALLOWED`): founder
    reserves are derived from body mass at construction, so overriding mass
    afterwards would make starting energy inconsistent with body.
    """
    cfg = replace(config, seed=int(seed))
    engine = Engine(cfg)
    if trait is not None:
        if trait_value is None:
            raise ValueError("trait_value is required when trait is set")
        if trait not in TRAIT_SWEEP_ALLOWED:
            raise ValueError(
                f"trait {trait!r} cannot be swept at founding; allowed: "
                f"{TRAIT_SWEEP_ALLOWED}")
        locus = _trait_locus(trait, float(trait_value))
        index = TRAIT_NAMES.index(trait)
        engine.population.genome[:, index] = locus
        engine._refresh_phen()
    recorder = StatsRecorder()
    interval = max(1, int(cfg.stats.interval))
    while engine.tick < int(cfg.ticks):
        engine.step()
        if engine.tick % interval == 0:
            recorder.record(engine)
    row = recorder.record(engine)
    if row is None:  # pragma: no cover - record always returns a row
        row = {}
    return {
        "seed": int(seed),
        "final_population": int(engine.population.n_alive),
        "extinct": bool(engine.population.n_alive == 0),
        "births": int(engine.births_total),
        "deaths": int(engine.deaths_total),
        "mean_energy": float(row.get("mean_energy", 0.0)),
        "mean_organism_biomass": float(row.get("mean_organism_biomass", 0.0)),
        "survival_median": float(row.get("survival_median", 0.0)),
        "mean_pairwise": float(row.get("mean_pairwise", 0.0)),
        "trait_variance": float(row.get("trait_variance", 0.0)),
        "population_cv": float(row.get("population_cv", 0.0)),
        # Bottleneck indicator sampled at the horizon (IMPROVE §5).
        "population_drop": float(row.get("population_drop", 0.0)),
        "diversity_drop": float(row.get("diversity_drop", 0.0)),
        "bottleneck": float(row.get("bottleneck", 0.0)),
        "checksum": engine.checksum(),
    }


def _stable_seed(*parts: Any) -> int:
    """Process-independent seed for a bootstrap stream (``hash()`` is salted)."""
    value = 0x811C9DC5
    for part in parts:
        for byte in repr(part).encode("utf-8"):
            value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value & 0x7FFFFFFF


def _summarise(sweep: Sweep, rows: list[dict[str, Any]],
               seeds: list[int]) -> list[dict[str, Any]]:
    """Per-value means, SDs and paired differences against the baseline value."""
    by_key = {(r["parameter_value"], r["seed"]): r for r in rows}
    baseline_rows = [by_key[(sweep.baseline, s)] for s in seeds
                     if (sweep.baseline, s) in by_key]
    out: list[dict[str, Any]] = []
    for value in sweep.values:
        members = [by_key[(value, s)] for s in seeds if (value, s) in by_key]
        entry: dict[str, Any] = {"value": value, "n": len(members)}
        for endpoint in ENDPOINTS:
            series = np.asarray([float(m[endpoint]) for m in members],
                                dtype=np.float64)
            entry[f"{endpoint}_mean"] = (float(series.mean())
                                         if series.size else 0.0)
            entry[f"{endpoint}_sd"] = (float(series.std(ddof=1))
                                       if series.size > 1 else 0.0)
        # Paired differences vs the baseline value, same seed.
        for endpoint in ENDPOINTS:
            diffs = []
            for member in members:
                base = by_key.get((sweep.baseline, member["seed"]))
                if base is not None:
                    diffs.append(float(member[endpoint]) - float(base[endpoint]))
            arr = np.asarray(diffs, dtype=np.float64)
            mean, lo, hi = bootstrap_ci(arr, Rng(_stable_seed(sweep.id, value,
                                                              endpoint)))
            entry[f"{endpoint}_diff"] = mean
            entry[f"{endpoint}_ci_low"] = lo
            entry[f"{endpoint}_ci_high"] = hi
            entry[f"{endpoint}_p"] = wilcoxon_signed_rank(arr)["p"]
        out.append(entry)
    return out


def run_sweep(out_dir: str, *, sweep: str = "all",
              seeds: Iterable[int] = (11, 12, 13),
              ticks: int | None = 250,
              scale: float = DEFAULT_SCALE,
              config_path: str | None = None,
              engine_factory=None) -> dict[str, Any]:
    """Run one declared sweep (or all of them) and write CSV + ``summary.md``."""
    selected = list(SWEEPS) if str(sweep).lower() == "all" else [str(sweep).lower()]
    unknown = [s for s in selected if s not in SWEEPS]
    if unknown:
        raise ValueError(f"unknown sweep(s) {unknown}; known: {sorted(SWEEPS)}")

    base = load_config(config_path or os.path.join(CONFIGS_DIR, DEFAULT_CONFIG))
    base = scaled_config(base, scale, ticks)
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("at least one seed is required")
    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, Any]]] = {}
    for key in selected:
        decl = SWEEPS[key]
        for value in decl.values:
            # A trait sweep varies a founder genome locus (converted from the
            # physical value); a parameter sweep copies the config. Neither
            # mutates ``base`` and both keep the engine's RNG stream untouched.
            cfg = (base if decl.is_trait
                   else _set_parameter(base, decl.parameter, value))
            for seed in seeds:
                result = _run_one(cfg, seed, trait=decl.trait,
                                  trait_value=(value if decl.is_trait else None))
                result.update({"sweep": key, "parameter": decl.parameter,
                               "parameter_value": value})
                rows.append(result)
        summaries[key] = _summarise(
            decl, [r for r in rows if r["sweep"] == key], seeds)

    columns = ("sweep", "parameter", "parameter_value", "seed", *ENDPOINTS,
               "checksum")
    with open(os.path.join(out_dir, "sensitivity.csv"), "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[c] for c in columns])

    report = {
        "config": config_path or DEFAULT_CONFIG,
        "scale": float(scale),
        "ticks": int(base.ticks),
        "seeds": seeds,
        "sweeps": {k: {
            "parameter": SWEEPS[k].parameter,
            "values": list(SWEEPS[k].values),
            "baseline": SWEEPS[k].baseline,
            "hypothesis": SWEEPS[k].hypothesis,
            "rationale": SWEEPS[k].rationale,
            "expectation": SWEEPS[k].expectation,
            "trait": SWEEPS[k].trait,
            "is_trait": SWEEPS[k].is_trait,
            "table": summaries[k],
        } for k in selected},
        "rows": rows,
    }
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(_markdown(report))
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Sensitivity analysis",
        "",
        f"Configuration: `{os.path.basename(str(report['config']))}`, "
        f"scale {report['scale']}, {report['ticks']} ticks, "
        f"seeds {list(report['seeds'])}.",
        "",
        "One factor is varied at a time; the base configuration is copied for "
        "every run and never mutated. Runs are paired by seed, so the same "
        "world, plant layout and founder genomes are used at every value.",
        "",
    ]
    for key, block in report["sweeps"].items():
        kind = ("founder trait locus" if block.get("is_trait")
                else "config coefficient")
        lines += [
            f"## `{key}` — {block['parameter']}",
            "",
            f"**Independent variable.** A {kind}; values are the physical "
            f"quantities a reader can reason about.",
            "",
            f"**Hypothesis (pre-registered).** {block['hypothesis']}",
            "",
            f"**Why this parameter.** {block['rationale']}",
            "",
            f"**Values.** {block['values']} (baseline "
            f"`{block['baseline']}`).",
            "",
            "| Value | n | Final pop (mean ± SD) | Births | Deaths | "
            "Mean energy | Trait variance | Bottleneck rate | "
            "Δpop vs baseline [95 % CI] p |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in block["table"]:
            lines.append(
                f"| {row['value']} | {row['n']} | "
                f"{row['final_population_mean']:.1f} ± "
                f"{row['final_population_sd']:.1f} | "
                f"{row['births_mean']:.1f} | {row['deaths_mean']:.1f} | "
                f"{row['mean_energy_mean']:.3f} | "
                f"{row['trait_variance_mean']:.5f} | "
                f"{row['bottleneck_mean']:.2f} | "
                f"{row['final_population_diff']:+.1f} "
                f"[{row['final_population_ci_low']:+.1f}, "
                f"{row['final_population_ci_high']:+.1f}] "
                f"p={row['final_population_p']:.3f} |")
        if block.get("expectation"):
            lines += ["", f"**Expected interpretation.** {block['expectation']}"]
        lines.append("")
    lines += [
        "## Limitations",
        "",
        "- Each sweep is a short, scaled-down probe, not the design-scale "
        "experiment: the horizon and world are reduced to fit this machine.",
        "- One factor at a time cannot detect interactions between parameters; "
        "a flat response here means \"no detectable main effect at this scale\", "
        "not \"no effect\".",
        "- With few seeds the signed-rank test is underpowered. Wide intervals "
        "and large p-values are reported as such and are not evidence of "
        "equivalence.",
        "- Extinction, when it happens, is reported in the CSV and is not "
        "smoothed away by averaging.",
        "- \"Bottleneck rate\" is the fraction of runs whose final sample "
        "flagged a contraction: both population and genetic spread had fallen "
        "by at least half from their own peak in the retained window. It is an "
        "indicator of a demographic/genetic squeeze, **not** an effective "
        "population size (that would need a neutral-locus model this "
        "simulation does not satisfy) and not a species count.",
        "- A trait sweep varies the *founder* genome locus before tick 1, so it "
        "measures the effect of starting condition, not of a policy the "
        "population can evolve away from; mutation then acts on that locus as "
        "usual. `body_size` is excluded because founder reserves are derived "
        "from body mass at construction.",
        "- No calibrated default is changed by this analysis; the sweep only "
        "reports what the model does.",
        "",
    ]
    return "\n".join(lines)
