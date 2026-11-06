"""Throughput benchmarks and profiling (DESIGN §11).

The benchmark world keeps the calibrated default's *density* rather than its size:
the default is 6,000 plants / 300 founders = 20 plants per organism over
1,000 x 1,000 units (6e-3 plants per unit^2). A point with ``N`` organisms
therefore gets ``20 N`` plants and a world whose area is ``20 N / 6e-3``, so
sensor work per organism (which depends on neighbour density) stays comparable
to the real configuration. Field cells stay ~25 units.

These numbers are measurements for *this* machine (1 core, ~1 GB). They are
never targets restated as met: ``docs/BENCHMARKS.md`` records what was measured,
including misses.
"""
from __future__ import annotations

import cProfile
import io
import math
import pstats
import time

import numpy as np

from .config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from .engine import Engine

PLANTS_PER_ORGANISM = 20.0
PLANT_DENSITY = 6.0e-3  # plants per unit^2 in the calibrated default world
FIELD_CELL_SIZE = 25.0


def benchmark_config(population: int, ticks: int, seed: int = 1) -> Config:
    """A world at the default density holding ``population`` adult organisms."""
    population = max(1, int(population))
    plants = max(10, int(round(PLANTS_PER_ORGANISM * population)))
    width = float(math.sqrt(plants / PLANT_DENSITY))
    n_cells = max(8, int(round(width / FIELD_CELL_SIZE)))
    return Config(
        seed=int(seed),
        ticks=int(ticks),
        world=WorldConfig(width=width, height=width, n_field_cells=n_cells,
                          cell_size=width / n_cells),
        plants=PlantConfig(initial=plants, max=plants * 2),
        organisms=OrganismConfig(initial=population, max=population),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.002),
        stats=StatsConfig(),
    )


def run_point(population: int, ticks: int, seed: int = 1,
              sample_every: int = 10) -> dict:
    """Run one benchmark point; returns throughput and population statistics.

    The population starts at the requested size and is capped there. Because the
    model is closed (organisms die of starvation), the *mean* population over the
    run is reported alongside the nominal one; a throughput measured while the
    population collapsed is not a throughput at the nominal size.
    """
    cfg = benchmark_config(population, ticks, seed=seed)
    eng = Engine(cfg)
    samples = []
    started = time.perf_counter()
    for _ in range(int(ticks)):
        eng.step()
        if eng.tick % max(1, int(sample_every)) == 0:
            samples.append(eng.population.n_alive)
    elapsed = time.perf_counter() - started
    mean_pop = float(np.mean(samples)) if samples else float(eng.population.n_alive)
    return {
        "population": int(population),
        "plants": int(cfg.plants.initial),
        "world": round(float(cfg.world.width), 1),
        "cells": int(cfg.world.n_field_cells),
        "ticks": int(ticks),
        "seconds": round(elapsed, 2),
        "ticks_per_s": round(float(ticks) / elapsed, 2) if elapsed > 0 else 0.0,
        "mean_population": round(mean_pop, 1),
        "end_population": int(eng.population.n_alive),
        "births": int(eng.births_total),
        "deaths": int(eng.deaths_total),
    }


def profile_point(population: int, ticks: int, seed: int = 1,
                  top: int = 10) -> str:
    """``cProfile`` the hot loop and return the top-``top`` table as text."""
    cfg = benchmark_config(population, ticks, seed=seed)
    eng = Engine(cfg)
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(int(ticks)):
        eng.step()
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(top)
    return stream.getvalue()


#: Assumed tick counts and populations for the DESIGN §13 experiments, used only
#: for wall-clock *estimates*. Where the design states a number (C: 3k ticks,
#: D: 1k, E: shift at 15k, G: 20k x 6 + 60 arenas, H: 1,000 founders) it is used;
#: unstated horizons are assumed to be A's 30k. These assumptions are also
#: printed in docs/BENCHMARKS.md.
EXPERIMENT_ESTIMATES = (
    ("A stable ecosystem", 300, 30_000, 1),
    ("B predator pressure", 500, 30_000, 1),
    ("C severe drought", 300, 3_000, 1),
    ("D resource competition", 300, 1_000, 1),
    ("E environmental shift", 300, 30_000, 1),
    ("F no predators", 300, 30_000, 1),
    ("G assay (evolution)", 300, 20_000, 6),
    ("G assay (arenas)", 2, 500, 60),
    ("H sexual reproduction", 1_000, 30_000, 1),
)


def estimate_wall_clock(ticks_per_s_at) -> list[dict]:
    """Estimate experiment runtimes from a ``ticks_per_s_at(population)`` callable."""
    out = []
    for name, population, ticks, replicates in EXPERIMENT_ESTIMATES:
        rate = max(1e-9, ticks_per_s_at(population))
        seconds = ticks * replicates / rate
        out.append({"experiment": name, "population": population,
                    "ticks": ticks, "replicates": replicates,
                    "wall_clock_s": round(seconds, 1)})
    return out
