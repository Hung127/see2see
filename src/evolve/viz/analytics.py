"""Live analytics view (IMPROVE §6.4).

The report path writes matplotlib PNGs from *stored* rows after a run. This is
the live counterpart: a bounded, sampled history of measured quantities drawn
with :mod:`evolve.viz.charts` while the simulation runs.

Two rules keep it honest:

* sampling is on simulated time (every ``interval`` ticks), not on frames, so
  the x axis means ticks even when the reader changes speed or pauses;
* every chart prints its unit and its numeric range, and a series with no
  samples says so instead of drawing a straight line through nothing.

Nothing here feeds back into the engine. The history is a presentation-side
ring buffer, and the analytics view never advances or mutates the simulation.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pygame

from .charts import bar_chart, histogram, line_chart
from .palette import ACCENT, BAD, GOOD, TEXT, TEXT_DIM, TEXT_FAINT, WARN

#: Simulated ticks between samples. Rendering runs at frame rate; sampling at a
#: fixed tick interval keeps the charts comparable across speeds and bounded in
#: memory (``HISTORY_CAP`` samples, ~600 ticks at the default).
SAMPLE_INTERVAL = 20
HISTORY_CAP = 600

#: Live population histograms (IMPROVE §6.4 items 3 and 4). A *distribution*
#: is a snapshot of the living population, not a history series: storing a
#: histogram per sample would cost ~2,400 floats per sample for no reader
#: benefit, so these are computed from the current engine state and cached per
#: tick. Body mass and reserve are both in bu but span very different ranges
#: (a body is roughly 0.5–5 bu; a reserve can reach E_max ≈ 10·body^0.9), so
#: they get explicit separate ranges rather than a shared axis.
BODY_HIST_RANGE = (0.0, 6.0)
BODY_HIST_BINS = 24
ENERGY_HIST_RANGE = (0.0, 20.0)
ENERGY_HIST_BINS = 24

CHART_H = 132
CHART_GAP = 10
HEADER_H = 46

#: Quantities kept per sample. Each is measured from the engine at sample time;
#: derived rates are computed from the difference between consecutive totals.
FIELDS = (
    "tick", "population", "births", "deaths", "population_cv",
    "plant_biomass", "carcass_biomass", "detritus", "organism_biomass",
    "mean_energy", "mean_body", "mean_speed", "mean_vision", "mean_attack",
    "mean_diet", "mean_generation", "trait_variance", "nn_variance",
    "mean_pairwise", "mean_temperature", "mean_moisture", "survival_median",
    "bites", "hunts", "repro_successes",
)


class AnalyticsHistory:
    """Bounded, tick-sampled history of measured quantities."""

    def __init__(self, interval: int = SAMPLE_INTERVAL,
                 cap: int = HISTORY_CAP) -> None:
        self.interval = max(1, int(interval))
        self.cap = max(2, int(cap))
        self._rows: deque[dict[str, float]] = deque(maxlen=self.cap)
        self._last_tick: int | None = None
        self._last_births = 0
        self._last_deaths = 0

    @property
    def samples(self) -> int:
        return len(self._rows)

    @property
    def last_tick(self) -> int | None:
        return self._last_tick

    def reset(self) -> None:
        self._rows.clear()
        self._last_tick = None
        self._last_births = 0
        self._last_deaths = 0

    def maybe_sample(self, engine, force: bool = False) -> bool:
        """Append a sample if ``interval`` ticks have passed; return whether it did.

        ``force`` records even when the interval has not elapsed (used for the
        first frame, so the panel is not empty after one tick).
        """
        if not force and self._last_tick is not None:
            if engine.tick - self._last_tick < self.interval:
                return False
        if self._last_tick is not None and engine.tick == self._last_tick:
            return False
        self._rows.append(self._measure(engine))
        self._last_tick = int(engine.tick)
        self._last_births = int(engine.births_total)
        self._last_deaths = int(engine.deaths_total)
        return True

    def _measure(self, engine) -> dict[str, float]:
        from .. import stats as stats_mod

        pop = engine.population
        idx = np.flatnonzero(pop.alive)
        n = idx.size
        genomes = pop.genome[idx]
        if n:
            from ..genome import phenotype

            phen = phenotype(genomes)
            div = stats_mod.diversity(genomes)
            means = {
                "mean_energy": float(pop.energy[idx].mean()),
                "mean_body": float(pop.body[idx].mean()),
                "mean_speed": float(phen[:, 1].mean()),
                "mean_vision": float(phen[:, 2].mean()),
                "mean_attack": float(phen[:, 5].mean()),
                "mean_diet": float(phen[:, 9].mean()),
                "mean_generation": float(pop.generation[idx].mean()),
                "trait_variance": div["trait_variance"],
                "nn_variance": div["nn_variance"],
                "mean_pairwise": div["mean_pairwise"],
                "organism_biomass": float(pop.body[idx].sum()
                                          + pop.energy[idx].sum()),
            }
        else:
            means = {name: 0.0 for name in (
                "mean_energy", "mean_body", "mean_speed", "mean_vision",
                "mean_attack", "mean_diet", "mean_generation",
                "trait_variance", "nn_variance", "mean_pairwise",
                "organism_biomass")}
        # Population CV over the samples retained so far: a stability measure
        # computed only from observed values, never a forecast.
        values = [row["population"] for row in self._rows]
        values.append(float(n))
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean()) if arr.size else 0.0
        cv = float(arr.std() / mean) if mean > 0.0 else 0.0
        births = int(engine.births_total) - self._last_births
        deaths = int(engine.deaths_total) - self._last_deaths
        # Lifetime counters summed over the living population. They are
        # cumulative per organism (reset only by death), so the charts below
        # describe the standing population, not a per-interval count.
        def total(name: str) -> float:
            return float(getattr(pop, name)[idx].sum()) if n else 0.0
        row: dict[str, float] = {
            "tick": float(engine.tick),
            "population": float(n),
            "births": float(births),
            "deaths": float(deaths),
            "population_cv": cv,
            "plant_biomass": float(engine.plants.total_biomass()),
            "carcass_biomass": float(engine.carcasses.total_biomass()),
            "detritus": float(engine.detritus),
            "mean_temperature": float(engine.fields.temperature.mean()),
            "mean_moisture": float(engine.fields.moisture.mean()),
            "survival_median": stats_mod._survival_median(engine),
            "bites": total("bites"),
            "hunts": total("hunts"),
            "repro_successes": total("repro_success"),
            **means,
        }
        return row

    def arrays(self) -> dict[str, np.ndarray]:
        """All series as float arrays, aligned by sample (oldest first)."""
        if not self._rows:
            return {name: np.zeros(0) for name in FIELDS}
        return {name: np.asarray([row[name] for row in self._rows],
                                 dtype=np.float64) for name in FIELDS}


def _live_histogram(engine, attr: str, span: tuple[float, float],
                    bins: int) -> np.ndarray:
    """Histogram of one per-organism quantity over the *living* population.

    This is a snapshot of who is alive now, so it is computed from engine state
    at draw time rather than stored per sample: a histogram per sample would
    cost thousands of floats in a ring buffer for a chart that only ever shows
    the current population. The bins are fixed by the caller so two frames are
    comparable even as the population changes.
    """
    pop = engine.population
    idx = np.flatnonzero(pop.alive)
    low, high = span
    if idx.size == 0:
        return np.zeros(bins, dtype=np.float64)
    values = getattr(pop, attr)[idx]
    counts, _ = np.histogram(np.clip(values, low, high), bins=bins,
                             range=(low, high))
    return counts.astype(np.float64)


def _blocks(data: dict[str, np.ndarray], engine
            ) -> list[dict]:
    """Chart descriptors ``{kind, key, title, unit}``, in reading order.

    Each line chart carries **one unit**. Mixing quantities of different
    magnitude on a shared axis (vision ≈100 u beside a diet gene in [0, 1], or a
    pairwise distance ≈1,400 beside a gene variance ≈0.05) flattens the small
    series into the axis and makes the unit label meaningless, so those are
    split instead of overlaid. The only shared-axis charts combine quantities
    in the *same* unit (biomass pools in bu, births/deaths in individuals).
    """
    return [
        {"kind": "line", "key": "population", "title": "population",
         "unit": "organisms"},
        {"kind": "line", "key": "births", "title": "births and deaths per sample",
         "unit": "individuals"},
        {"kind": "line", "key": "biomass", "title": "biomass pools", "unit": "bu"},
        {"kind": "line", "key": "reserve", "title": "mean reserve and body",
         "unit": "bu / organism"},
        {"kind": "line", "key": "speed", "title": "mean max speed",
         "unit": "world units / tick"},
        {"kind": "line", "key": "vision", "title": "mean vision range",
         "unit": "world units"},
        {"kind": "line", "key": "genes", "title": "mean attack and diet genes",
         "unit": "gene value 0–1 or 0–5"},
        {"kind": "line", "key": "gene_variance", "title": "genetic variance",
         "unit": "variance of gene values"},
        {"kind": "line", "key": "pairwise", "title": "mean pairwise distance",
         "unit": "world units"},
        {"kind": "line", "key": "temperature", "title": "mean temperature",
         "unit": "°C"},
        {"kind": "line", "key": "moisture", "title": "mean moisture",
         "unit": "fraction 0–1"},
        {"kind": "line", "key": "generation", "title": "mean generation depth",
         "unit": "generations"},
        {"kind": "line", "key": "survival", "title": "median age at death",
         "unit": "ticks"},
        {"kind": "bar", "key": "deaths", "title": "deaths by cause (cumulative)",
         "unit": "deaths"},
        {"kind": "hist", "key": "age_at_death", "title": "age at death",
         "unit": "deaths"},
        # IMPROVE §6.4 items 3 and 4: the *distribution* of body mass and of
        # reserve across the living population, not a mean over time. Both are
        # in bu but over different spans, so each declares its own range.
        {"kind": "live_hist", "key": "body_size",
         "title": "body mass distribution (living)", "unit": "organisms"},
        {"kind": "live_hist", "key": "energy",
         "title": "reserve distribution (living)", "unit": "organisms"},
        {"kind": "line", "key": "cv", "title": "population variability",
         "unit": "coefficient of variation"},
        {"kind": "line", "key": "behaviour", "title": "behaviour totals (living)",
         "unit": "events"},
    ]


#: ``(population attribute, axis span, bin count, x label, caption)`` for the
#: live distribution charts. Kept beside the block list so the descriptor and
#: the data it draws cannot drift apart.
LIVE_HISTOGRAMS: dict[str, tuple[str, tuple[float, float], int, str, str]] = {
    "body_size": ("body", BODY_HIST_RANGE, BODY_HIST_BINS, "bu",
                  "living organisms only; clipped to the stated span"),
    "energy": ("energy", ENERGY_HIST_RANGE, ENERGY_HIST_BINS, "bu",
               "living organisms only; reserves above the span are clipped"),
}


def _draw_block(screen, rect, fonts, cache, block: dict,
                data: dict[str, np.ndarray], engine) -> None:
    kind, title, unit = block["kind"], block["title"], block["unit"]
    d = data
    if kind == "bar":
        counts = getattr(engine, "death_counts", {}) or {}
        order = ("predation", "starvation", "senescence", "health")
        palette = {"predation": BAD, "starvation": WARN,
                   "senescence": TEXT_DIM, "health": TEXT_FAINT}
        bars = [(name, palette[name], float(counts.get(name, 0)))
                for name in order]
        bar_chart(screen, rect, fonts, cache, title=title, y_unit=unit,
                  bars=bars,
                  caption="since the run began; zero means none observed")
        return
    if kind == "hist":
        from ..engine import DEATH_AGE_BIN

        histogram(screen, rect, fonts, cache, title=title, y_unit=unit,
                  counts=getattr(engine, "death_age_hist",
                                 np.zeros(0, dtype=np.int64)),
                  bin_width=float(DEATH_AGE_BIN),
                  caption=f"age in ticks, {DEATH_AGE_BIN}-tick bins")
        return
    if kind == "live_hist":
        attr, span, bins, x_label, caption = LIVE_HISTOGRAMS[block["key"]]
        counts = _live_histogram(engine, attr, span, bins)
        edge = (span[1] - span[0]) / bins
        histogram(screen, rect, fonts, cache, title=title, y_unit=unit,
                  counts=counts, bin_width=edge,
                  caption=f"{caption} · x axis in {x_label}",
                  empty_text="no living organisms to measure",
                  x_span=span)
        return
    x = ("tick 0", f"tick {int(d['tick'][-1]) if d['tick'].size else 0:,}")
    key = block["key"]
    caption = None
    if key == "population":
        series = [("population", ACCENT, d["population"])]
    elif key == "births":
        series = [("births", GOOD, d["births"]), ("deaths", BAD, d["deaths"])]
        caption = "counts since the previous sample, not per tick"
    elif key == "biomass":
        series = [
            ("plants", (104, 200, 112), d["plant_biomass"]),
            ("carcasses", (156, 124, 92), d["carcass_biomass"]),
            ("detritus", (120, 120, 96), d["detritus"]),
            ("animals", (232, 196, 120), d["organism_biomass"]),
        ]
    elif key == "reserve":
        series = [("reserve", GOOD, d["mean_energy"]),
                  ("body", ACCENT, d["mean_body"])]
    elif key == "speed":
        series = [("speed_max", (120, 190, 240), d["mean_speed"])]
        caption = "gene-derived maximum, not the distance actually moved"
    elif key == "vision":
        series = [("vision_range", (168, 220, 140), d["mean_vision"])]
    elif key == "genes":
        series = [("attack (0–5)", (236, 150, 120), d["mean_attack"]),
                  ("diet (0–1)", (226, 196, 96), d["mean_diet"])]
        caption = "a preference, not observed intake"
    elif key == "gene_variance":
        series = [("trait loci", GOOD, d["trait_variance"]),
                  ("network weights", ACCENT, d["nn_variance"])]
        caption = "variance of gene values, not phenotype spread"
    elif key == "pairwise":
        series = [("mean pairwise distance", WARN, d["mean_pairwise"])]
        caption = "phenotypic distance between sampled genomes"
    elif key == "temperature":
        series = [("temperature", (236, 150, 96), d["mean_temperature"])]
    elif key == "moisture":
        series = [("moisture", (120, 180, 240), d["mean_moisture"])]
    elif key == "generation":
        series = [("mean generation", ACCENT, d["mean_generation"])]
    elif key == "survival":
        series = [("median age at death", WARN, d["survival_median"])]
        caption = "0 until the first death is recorded; a 25-tick bin midpoint"
    elif key == "cv":
        series = [("CV in window", WARN, d["population_cv"])]
        caption = "coefficient of variation over the retained samples"
    else:  # behaviour
        series = [("bites", GOOD, d["bites"]),
                  ("attacks", BAD, d["hunts"]),
                  ("reproductions", ACCENT, d["repro_successes"])]
        caption = ("lifetime totals summed over the living population; "
                   "they reset when an organism dies")
    line_chart(screen, rect, fonts, cache, title=title, y_unit=unit,
               series=series, x_labels=x, caption=caption)


def _layout(rect: pygame.Rect, columns: int) -> tuple[int, int]:
    """Column width for a two-column grid, and the chart height."""
    gap = CHART_GAP
    return max(120, (rect.width - gap * (columns - 1)) // columns), CHART_H


def draw_analytics(screen: pygame.Surface, rect: pygame.Rect,
                   fonts: dict[str, pygame.font.Font], cache,
                   history: AnalyticsHistory, engine,
                   scroll: int = 0) -> tuple[int, int]:
    """Draw the analytics grid; return ``(content_height, view_height)``.

    The caller keeps the scroll offset so the view is a normal clipped,
    scrollable panel rather than a modal. Both the world and the simulation
    keep running behind it.
    """
    screen.fill((14, 16, 20), rect)
    inner = pygame.Rect(rect.x + 12, rect.y + 8,
                        max(80, rect.width - 24), max(40, rect.height - 16))
    columns = 2 if inner.width >= 760 else 1
    col_w, chart_h = _layout(inner, columns)
    data = history.arrays()
    blocks = _blocks(data, engine)

    screen.set_clip(inner)
    y = inner.y - int(scroll)
    # Header: what this view is, and how densely it samples.
    if data["tick"].size:
        first = int(data["tick"][0])
        last = int(data["tick"][-1])
        span = (f"{history.samples} samples · ticks {first:,}–{last:,} · "
                f"every {history.interval} ticks")
    else:
        span = "no samples yet — the simulation has not advanced"
    screen.blit(cache.render(fonts["title"], "analytics", TEXT),
                (inner.x, y))
    y += fonts["title"].get_linesize() + 2
    screen.blit(cache.render(fonts["small"], span, TEXT_DIM),
                (inner.x, y))
    y += fonts["small"].get_linesize() + 2
    screen.blit(cache.render(
        fonts["small"],
        "measured from the running simulation; charts are sampled, not "
        "per-tick records", TEXT_FAINT), (inner.x, y))
    y += HEADER_H - fonts["small"].get_linesize()

    for i, block in enumerate(blocks):
        col = i % columns
        x = inner.x + col * (col_w + CHART_GAP)
        rect_block = pygame.Rect(x, y, col_w, chart_h)
        _draw_block(screen, rect_block, fonts, cache, block, data, engine)
        if col == columns - 1 or i == len(blocks) - 1:
            y += chart_h + CHART_GAP
    screen.set_clip(None)
    content = max(0, y - (inner.y - int(scroll)))
    return content, inner.height
