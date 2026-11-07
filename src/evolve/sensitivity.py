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
import os
from dataclasses import replace
from typing import Any, Iterable, Mapping

import numpy as np

from .config import Config, load_config
from .engine import Engine
from .experiments import (
    DEFAULT_SCALE, bootstrap_ci, scaled_config, wilcoxon_signed_rank,
)
from .rng import Rng
from .stats import StatsRecorder

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "configs")
DEFAULT_CONFIG = "default.yaml"

#: Measured outcomes compared across parameter values.
ENDPOINTS = (
    "final_population", "extinct", "births", "deaths",
    "mean_energy", "mean_organism_biomass", "survival_median",
    "mean_pairwise", "trait_variance", "population_cv",
)


class Sweep:
    """One declared one-factor sweep."""

    __slots__ = ("id", "parameter", "values", "baseline", "hypothesis",
                 "rationale", "expectation")

    def __init__(self, id: str, parameter: str, values: Iterable[float],
                 baseline: float, hypothesis: str, rationale: str,
                 expectation: str = "") -> None:
        self.id = id
        self.parameter = parameter
        self.values = tuple(values)
        self.baseline = baseline
        self.hypothesis = hypothesis
        self.rationale = rationale
        self.expectation = expectation
        if self.baseline not in self.values:
            raise ValueError(f"sweep {id!r}: baseline {baseline!r} not in values")
        if len(self.values) < 2:
            raise ValueError(f"sweep {id!r}: needs at least two values")


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


def _run_one(config: Config, seed: int) -> dict[str, Any]:
    """Run one engine to the horizon and return measured endpoints."""
    cfg = replace(config, seed=int(seed))
    engine = Engine(cfg)
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
            cfg = _set_parameter(base, decl.parameter, value)
            for seed in seeds:
                result = _run_one(cfg, seed)
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
        lines += [
            f"## `{key}` — {block['parameter']}",
            "",
            f"**Hypothesis (pre-registered).** {block['hypothesis']}",
            "",
            f"**Why this parameter.** {block['rationale']}",
            "",
            f"**Values.** {block['values']} (baseline "
            f"`{block['baseline']}`).",
            "",
            "| Value | n | Final pop (mean ± SD) | Births | Deaths | "
            "Mean energy | Trait variance | Δpop vs baseline [95 % CI] p |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in block["table"]:
            lines.append(
                f"| {row['value']} | {row['n']} | "
                f"{row['final_population_mean']:.1f} ± "
                f"{row['final_population_sd']:.1f} | "
                f"{row['births_mean']:.1f} | {row['deaths_mean']:.1f} | "
                f"{row['mean_energy_mean']:.3f} | "
                f"{row['trait_variance_mean']:.5f} | "
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
        "- No calibrated default is changed by this analysis; the sweep only "
        "reports what the model does.",
        "",
    ]
    return "\n".join(lines)
