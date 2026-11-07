# Benchmarks and profile

Status: **measured** (Task 25). Every number below was measured on this machine;
nothing is extrapolated except the experiment wall-clock *estimates*, which are
labelled as such.

## Machine

* 1 CPU core, ~1 GB RAM, Linux
* Python 3.14.4, NumPy 2.5.4
* Headless (no display; the engine never opens a window)

## Method

`evolve.bench.run_point` builds a world that keeps the calibrated default's
*density*, not its size: the default is 6,000 plants / 300 founders = 20 plants
per organism over 1,000 × 1,000 units (6e-3 plants/unit²). A point with `N`
organisms gets `20·N` plants and a world of area `20·N / 6e-3`, so per-organism
sensor work (which depends on neighbour density) stays comparable to the real
configuration. Field cells stay ≈25 units. The population starts at `N` and is
capped at `N`.

The *mean* population over the run is reported next to the nominal one, because
a throughput measured while the population collapsed is not a throughput at the
nominal size. Here the populations stayed at or near nominal (deaths were few
over these short horizons), so the numbers are valid at the stated sizes.

## Measured throughput (DESIGN §11 targets)

Numbers below are the **pre-upgrade** baseline (commit `a66cc0c`). The
optimization pass section further down reports the post-upgrade figures and the
speedup; this table is kept unchanged so the before/after comparison is honest.

| Population | Plants | World | Ticks run | Mean pop | ticks/s | Target | Result |
|---|---|---|---|---|---|---|---|
| 300 | 6,000 | 1000 | 100 | 291.6 | **14.0** | — | reference point |
| 500 | 10,000 | 1291 | 100 | 480.5 | **9.03** | ≥ 25 | **MISSED** (36 % of target) |
| 1,000 | 20,000 | 1826 | 50 | 1000.0 | **4.49** | ≥ 12 | **MISSED** (37 % of target) |
| 5,000 | 100,000 | 4082 | 20 | 4999.5 | **0.78** | ≥ 5 | **MISSED** (16 % of target) |

All three §11 targets are missed on this machine, both before and after the
optimization pass. Throughput scales almost
exactly linearly in population: the product `population × ticks/s` is
4,200–4,900 organism-ticks/s across the whole range, i.e. the cost is
per-organism work, not per-tick overhead. (For context, the calibrated default
world with its live population oscillating around ~70 organisms runs at
67.7 ticks/s — consistent with the same product.)

**Post-upgrade the same targets are still missed** — 8.36 ticks/s at 1,000
organisms against a target of 12. The pass bought ~1.8×, not the 2.7× that
target would need. This is stated rather than glossed: opt-in closed below what
the design asked for, and the remaining cost is the periodic wrap arithmetic
that cannot be changed without altering results.

## Profile (1,000 organisms, 20 ticks, cProfile top-10 by cumulative time)

```
         78828 function calls in 4.200 seconds

   Ordered by: cumulative time
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       20    0.002    0.000    4.200    0.210 engine.py:121(step)
       20    0.025    0.001    2.792    0.140 engine.py:217(_think)
       20    0.180    0.009    2.739    0.137 sensors.py:51(collect)
       60    0.606    0.010    2.262    0.038 spatial.py:115(candidates)
     1540    0.506    0.000    1.607    0.001 world.py:44(distance)
       20    0.723    0.036    1.279    0.064 sensors.py:25(_nearest_visible)
       20    0.087    0.004    1.202    0.060 engine.py:243(_feed)
     1580    0.717    0.000    0.717    0.000 world.py:32(dx)
     1580    0.640    0.000    0.640    0.000 world.py:38(dy)
       40    0.333    0.008    0.589    0.015 world.py:48(bearing)
```

**Bottleneck: sensing.** `collect` is 65 % of the tick (2.74 s of 4.2 s), and
inside it the neighbour-query path (`spatial.candidates` → `world.distance` →
`dx`/`dy`) dominates. `world.dx`/`dy` are called 1,580 times in 20 ticks —
roughly 79 small-array calls per tick — which is per-call Python overhead on
arrays of a few hundred elements. The obvious optimisation (batch the distance
computation once per sensor family instead of per call site) is recorded here as
future work; it was **not** applied in this task, so the numbers above remain the
honest baseline.

## Optimization pass (upgrade, after the audit)

Baseline commit `a66cc0c`. Every change below left `Engine.checksum()`
bit-identical on the three pinned scenarios (plain / sexual / no-predation);
`tests/test_optimization_equivalence.py` enforces that in CI.

| # | Change | Effect |
|---|---|---|
| 1 | `world.dx`/`dy` computed in place (`np.subtract(..., dtype=np.float64)`, `np.mod(out=)`); new `World.delta` and `World.distance_and_bearing` so a pair's wrap is computed once, not twice | one temporary per axis instead of four; sensing no longer wraps twice per pair |
| 2 | `sensors._nearest_visible` filters by vision radius *before* `arctan2` | vision cone rejects ~80 % of stencil candidates before the transcendental |
| 3 | `_feed` and `_predation` query the grid with the eater/attacker positions only, then remap rows | stops building candidate rows for organisms that cannot act |
| 4 | `sensors.collect` no longer builds the full 40×40 plant-biomass field; it bins plants into a flat index and gathers the organism cells | removes a 20,000-element `np.add.at` + 40×40 allocation per tick |
| 5 | New `spatial.best_per_query`: per-query minimum first, then tie-break only the survivors, replacing three full `np.lexsort` selections | selection over ~1–4 million candidate rows drops from hundreds of ms to tens |

Throughput, same method and machine as above (30 ticks, warm):

| Population | Before (ticks/s) | After (ticks/s) | Speedup |
|---|---|---|---|
| 300 | 14.30 | **24.98** | 1.75× |
| 1,000 | 4.53 | **8.36** | 1.85× |
| 5,000 | 0.80 | **1.43** | 1.79× |

Profile after the pass (1,000 organisms, 20 ticks):

```
         81298 function calls in 2.407 seconds

   Ordered by: cumulative time
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       20    0.001    0.000    2.407    0.120 engine.py:122(step)
       20    0.001    0.000    1.870    0.093 engine.py:226(_think)
       20    0.198    0.010    1.838    0.092 sensors.py:64(collect)
       60    0.303    0.005    1.338    0.022 spatial.py:154(candidates)
     1520    0.335    0.000    1.338    0.001 world.py:64(distance)
       20    0.158    0.008    0.441    0.022 sensors.py:25(_nearest_visible)
     1560    0.391    0.000    0.391    0.000 world.py:47(dy)
     1560    0.381    0.000    0.381    0.000 world.py:39(dx)
```

Tick time is down 4.542 → 2.407 s (−47 %); `_feed` dropped 1.239 → 0.352 s.
What is left is the periodic wrap itself (`np.mod` on every candidate pair).
The cheaper conditional form (`where d > w/2: d -= w`) is **not** bit-identical
— it differs by up to 2.3e-13 — so it was rejected: a faster tick that produces
different organisms is a different experiment. That is the honest stopping point
for optimization that preserves semantics.

Memory at 1,000 organisms: traced steady state **6.5 MB**, traced peak
**56.4 MB**, process peak RSS **113 MB** (was 57.4 MB / 122 MB). Still far
inside the < 650 MB target and lower than the baseline.



Estimates only. Rate model: measured organism-ticks/s ≈ 4,800 (linear in
population); the 2-organism assay arenas are dominated by fixed per-tick
overhead and are estimated at 200 ticks/s. Where DESIGN §13 states a horizon it
is used (C: 3k ticks, D: 1k, E: shift at 15k, G: 20k × 6 replicates + 60 × 500
arena ticks, H: 1,000 founders); unstated horizons are assumed to be A's 30k.

| Experiment | Population | Ticks | Reps | Estimate |
|---|---|---|---|---|
| A stable ecosystem | 300 | 30,000 | 1 | 31 min |
| B predator pressure | 500 | 30,000 | 1 | 52 min |
| C severe drought | 300 | 3,000 | 1 | 3 min |
| D resource competition | 300 | 1,000 | 1 | 1 min |
| E environmental shift | 300 | 30,000 | 1 | 31 min |
| F no predators | 300 | 30,000 | 1 | 31 min |
| G assay, evolution phase | 300 | 20,000 | 6 | 125 min |
| G assay, arenas | 2 | 500 | 60 | 3 min |
| H sexual reproduction | 1,000 | 30,000 | 1 | 104 min |
| **Total at design scale** | | | | **≈ 7.5 h** |

At design scale the full programme does not fit a reasonable session on this
machine. Task 26 therefore runs *scaled-down* versions of each experiment
(smaller worlds and shorter horizons, documented per experiment in
`docs/EXPERIMENTS.md`), with the design-scale estimate recorded here so the
difference is explicit rather than silent.

## Reproduce

```bash
PYTHONPATH=src python3 -m evolve.cli benchmark --population 1000 --ticks 50
PYTHONPATH=src python3 -m evolve.cli benchmark --population 1000 --ticks 20 --profile
```
