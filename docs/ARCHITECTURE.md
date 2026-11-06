# Architecture

EVOLVE is a headless, deterministic simulation core with optional presentation
layers on top. The core (`evolve.engine` and everything it imports) never
imports pygame or matplotlib; a test enforces that. The pygame inspector and the
matplotlib plots are separate, lazy imports.

## Package layout

```
src/evolve/
  config.py        frozen dataclass tree, YAML load, validation, config hash
  rng.py           one seeded PCG64 generator per simulation, state capture
  world.py         torus/reflect/hard boundaries, distance, bearing
  fields.py        coarse environmental grid: terrain, soil, season, rain,
                   pheromone, per-cell event modifiers
  plants.py        plant SoA: growth, germination, senescence, biomass field
  organisms.py     organism SoA, OrganismView, counters, birth/death bookkeeping
  genome.py        trait/neural gene layout, founders, mutation, crossover
  brain.py         batched 16 → 12 → 8 → 8 MLP
  sensors.py       16-channel sensor assembly
  actions.py       network outputs → physical proposals (the one decision point)
  energy.py        metabolic formulas + the conservation ledger
  spatial.py       uniform grid, pair expansion, segmented reductions
  carcasses.py     spatially explicit carrion pool
  events.py        scheduled and random environmental events (physical only)
  engine.py        the tick pipeline, checksums, ledger assertions
  stats.py         per-interval aggregations, diversity, CSV and plots
  persistence.py   SQLite schema, batched writers, snapshots
  experiments.py   scenarios A–H, hooks, scaling, pre-registered assay
  bench.py         throughput measurement and profiling
  cli.py           run / reproduce / report / viz / benchmark / experiments
  viz/             app.py, render.py, camera.py, panels.py (pygame, optional)
```

## The tick pipeline

Each `Engine.step()` runs the stages in this fixed order (DESIGN §7). Stages that
only read state (sensing, brains, decoding) write proposals; stages that write
state apply them. No organism observes another organism's post-move position.

1. **Environment** — `fields.step(tick)`: season phase, optional rain,
   pheromone diffusion/decay, mass-conserving soil percolation, and reset of the
   one-tick event modifiers.
2. **Events** — `events.update(tick)` re-applies every active continuous event
   and fires instant events. Effect is physical only: fields, health, plant
   biomass, soil, ledger. Never a genome.
3. **Plants** — growth limited by soil nutrient, temperature and moisture, then
   germination, then senescence (and event damage).
4. **Compaction and ordering** — dead slots removed, organisms sorted by ID.
   From here on, array row order is ascending ID, so every tie-break is defined.
5. **Sensing → brain → decoding** — one batched sensor pass, one batched forward
   pass, one deterministic threshold decode (see `NEURAL_NETWORKS.md`).
6. **Movement** — heading integrates the turn output; position integrates
   speed = `speed_max · throttle · sprint`; boundaries wrap/reflect/block.
7. **Feeding** — one bite per eater from the nearest plant *or* carcass within
   reach; contested resources are shared proportionally with an exact remainder
   to the lowest-ID eater, so stocks close to 1e-9.
8. **Predation** — one strike per attacker at the nearest organism in reach;
   all strikes resolve simultaneously against start-of-stage health; kill credit
   is damage-weighted; meat is apportioned by credit and the remainder becomes a
   carcass.
9. **Metabolism** — the full cost model is charged; reserves pay first, then
   body is catabolised; respired mass goes to the local soil cell and equal
   energy to heat. Health regenerates only above twice basal cost.
10. **Reproduction** — mature organisms above their energy threshold invest a
    fraction of `E_max`; gates fail with *nobody paying*; the child is built
    from the genome and the parent's investment is consumed (part reserve, part
    body, 10 % waste to soil).
11. **Mortality** — health ≤ 0, starvation (energy and body exhausted), and
    probabilistic senescence past the lifespan locus
    (`p = 1 − exp(−(age − lifespan)/200)` per tick). Non-predation deaths enter
    the diffuse detritus pool (predation kills already left carcasses).
12. **Decomposition** — detritus and carcasses mineralise at `decay · C` per
    tick: mass to soil, equal energy to heat.
13. **Compaction and density fields** — dead removal and the derived density
    fields used for sensing/visualisation. Statistics are *not* recorded inside
    `step()`; the runner samples them every `stats.interval`. (Code comments
    carry the DESIGN §7 stage numbers — environment 2, plants 3, sort 4,
    move 8 … density fields 16 — and the list above groups them for reading.)

## Determinism contract

One generator, `numpy.random.Generator(PCG64(seed))`, float64, for the whole
simulation. The draw order is part of the interface:

1. **Construction** — terrain noise, founder genomes, founder positions and
   headings, plant positions (in that order).
2. **Every tick** — rainfall, germination picks and positions, senescence
   survival draws (only when some organism is past its lifespan).
3. **Births** — mutation draws in ascending parent-ID order (mutation itself
   draws mask, deltas, large-effect flags, in that order).
4. **Random events** — one Poisson draw per configured kind per tick, in config
   order, then draws for centre, radius, duration and magnitude.

Ties are broken by ascending organism/resource ID, never by array position.
`Engine.checksum()` hashes the full state (rounded to 1e-9) so two engines with
equal checksums are exchangeable. `tests/test_determinism.py` asserts identical
state hashes for equal seeds and different hashes for different seeds, and
`tests/test_snapshots.py` asserts resume-equals-continuation.

## Persistence

SQLite (WAL) with the DESIGN §9 schema: `experiments`, `runs`, `tick_stats`,
`snapshots`, `events`, `lineage`, `interactions`. Statistics are written in
batches once per `stats.interval`, never per tick. Snapshots are compressed
`.npz` files holding every state array, the event scheduler, the damage-credit
map, counters and the RNG state; `format_version` is checked and unknown
versions are refused (`SnapshotVersionError`).

A run is identified by `(config YAML, seed, spec_version, git_commit)`;
`evolve reproduce --run-id` re-executes and compares the stored final checksum.

## Scaling and performance

The engine is NumPy structure-of-arrays with one batched forward pass per tick.
Sensing dominates the cost (65 % of a tick at 1,000 organisms; see
`BENCHMARKS.md`). Throughput is ≈4,800 organism-ticks/s on this single core, so
large populations run slower than DESIGN §11 targets. This is a measured
property, not a hidden degradation: every report uses the measured rate.
