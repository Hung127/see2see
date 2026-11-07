# Experiments

This document describes what each experiment does, how it is implemented, and
how to reproduce it. Measured outcomes live in [`RESULTS.md`](RESULTS.md); the
design rationale lives in [`DESIGN.md`](DESIGN.md) §13.

## Running

```bash
# Scaled-down programme (default: scale 0.5, suitable for one core / ~1 GB)
PYTHONPATH=src python3 -m evolve.cli experiments --all --out experiments_out/run

# One experiment
PYTHONPATH=src python3 -m evolve.cli experiments --id c --out experiments_out/c

# The assay alone, with its pre-registration
PYTHONPATH=src python3 -m evolve.cli experiments --id g --out experiments_out/g

# Design scale (documents the ~7.5 h estimate of docs/BENCHMARKS.md)
PYTHONPATH=src python3 -m evolve.cli experiments --all --full --out experiments_out/full
```

Each experiment writes `<out>/<id>/` containing `stats.csv` (the
[`stats.COLUMNS`](../src/evolve/stats.py) wide table — 58 columns: the original
40, then 18 appended in the upgrade (13 in the first additive block, 2 in the
second, 3 bottleneck indicators in the third), never reordered), seven PNGs, and
`summary.md`. `--all` additionally writes `index.md`. The assay also writes
`pre_registration.md` and `assay.csv`.

## Configurations and hooks

The YAML files in `configs/` hold the **design-scale** parameters. The runner
applies a *setup* hook — an initial-condition manipulation — and, for E, one
mid-run *intervention*. No hook reads or writes an organism's decisions; the
neural controller is the only thing that chooses actions.

| ID | Config | Setup / intervention | Design horizon |
|----|--------|----------------------|----------------|
| A | `experiment_a.yaml` | none | 30,000 |
| B | `experiment_b.yaml` | `carnivore_founders`: +200 founders, diet locus 0.85 | 30,000 |
| C | `experiment_c.yaml` | drought event, magnitude 0.6, ticks 500–1100 | 3,000 |
| D | `experiment_d.yaml` | `clumped_plants`; config has 40 % of the A plant count | 1,000 |
| E | `experiment_e.yaml` | heatwave event +8 °C at t = 15,000; `move_nutrients` at the same tick | 30,000 |
| F | `experiment_f.yaml` | `no_predators`: attack loci → 0, predation stage disabled | 30,000 |
| G | `assay.yaml` | assay runner (below) | 20,000 per replicate |
| H | `experiment_h.yaml` | sexual arm + asexual control, same seed and density | 30,000 |

Notes on the manipulations:

* **B** draws the extra genomes from the ordinary founder distribution
  (`Beta(2,2)` traits, `N(0,0.35)` weights) and changes *only* the diet locus.
  Adding organisms adds mass, so the engine's ledger reference values are
  re-baselined — the setup is initial-condition construction, not a process
  inside the run.
* **D** reduces the initial plant count to 40 % of A (the config), and the hook
  clusters the living plants into patches. Plant density is therefore genuinely
  lower than A, and patches are separated by empty ground.
* **E** warms the whole world for the remaining 15,000 ticks. The nutrient move
  is element-wise mass-conserving: each soil column in the right half is added
  to its mirror column on the left and then zeroed, so the soil total is
  unchanged to floating point.
* **F** disables the predation *stage* as well as clipping the attack gene.
  Clipping alone would leave a predatory organism able to strike (for zero
  damage), which would still pay strike costs and still hit the target's
  defences; the design says "no predators", so the whole stage is off.
* **H** runs the sexual configuration and an asexual configuration with the
  same seed, world, plant layout and founder genomes. The two arms therefore
  differ only in `genetics.mode`.

## Assay G

The protocol is pre-registered: `run_assay` writes `pre_registration.md`
**before it constructs any engine**, so the sampling rule, endpoints and
analysis cannot be adjusted after seeing data. The tests assert this ordering by
injecting an engine factory that raises.

1. **Evolution phase.** `assay.yaml` (calibrated baseline, asexual, predation
   disabled, clumped plants) for the horizon, one run per replicate seed
   `base_seed + i`.
2. **Sampling.** At the final tick, draw genomes *uniformly at random* from the
   living population (with replacement if the population is smaller than the
   sample). No ranking by energy, intake or anything else: ranking would
   confound the assay with the selection that already happened. Control genomes
   come from the same founder distribution.
3. **Arena.** Each evolved/control pair shares one 200 × 200 arena at the
   calibrated plant density, with mirrored starting positions, identical plant
   layout across all arenas of a replicate, equal starting reserves as a
   fraction of each individual's own `E_max`, no reproduction, no predation, no
   mutation (then no mutation can occur). One paired difference per arena.
4. **Endpoints.** Primary: total food intake (bu; meat is zero without
   predation). Secondary: final energy and survival.
5. **Analysis.** Two-sided Wilcoxon signed-rank (in-house normal approximation,
   tie-corrected, zero differences dropped — scipy is not a dependency) and the
   mean paired difference with a percentile bootstrap 95 % CI (10,000
   resamples). Reported per replicate and pooled. The result is reported
   whatever it is.

**What can go wrong, and is reported rather than hidden:** the evolution phase
can go extinct before sampling (then the replicate contributes no arenas), the
sample can be smaller than requested (with replacement is used), and a genome
can die inside an arena (intake up to death counts, survival is recorded).
Because the arena removes reproduction, an organism that dies early has fewer
opportunities to feed; that is part of the endpoint, not a reason to drop the
pair.

## Controller comparison across scenarios

`evolve.cli controllers` evaluates the controller baselines (IMPROVE §4.2). By
default it measures the plain calibrated configuration. With `--scenarios` it
runs every controller in every **declared** experiment and every seed:

```bash
# plain configuration (the original single-environment report)
PYTHONPATH=src python3 -m evolve.cli controllers --out out/ctl \
    --seeds 11 12 --ticks 120

# the same controllers in three declared environments
PYTHONPATH=src python3 -m evolve.cli controllers --out out/ctl_multi \
    --seeds 11 12 --ticks 120 --scale 0.25 --scenarios default d f
```

Each scenario contributes its configuration, its setup hook (applied once at
construction, with `rebaseline`), its supervised `step` options and its
scheduled intervention. The `scenario` column appears in `comparison.csv` only
when more than one environment runs, so a single-environment file keeps its
original shape. `summary.md` gains a **Robustness across scenarios** table with
three declared numbers per controller: `persistence_rate` (runs still non-zero
at the horizon), `mean_survival_rate` and `worst_survival_rate` (the minimum,
i.e. the environment that hurt that controller most), plus
`mean_reproduction_success`. None of them is a fitness score, and the report
says the scenario list is a declared set, not a sample of environments.

`--scale` applies to the scenarios only (their YAMLs are design scale); the
plain configuration is used as written. A warm-up (`--warmup N`) is validated
against the *shortest* horizon across the selected cells, so it can never
consume the whole evaluation window of a short scenario.

## Sensitivity sweeps

`evolve.cli sensitivity` varies **one factor at a time** and writes a CSV plus
`summary.md` (IMPROVE §3.4 / §9). Each declared sweep has a hypothesis written
before the run, a rationale, and values bracketing the calibrated default. No
default is written back.

```bash
# one config sweep
PYTHONPATH=src python3 -m evolve.cli sensitivity --sweep bite \
    --seeds 11 12 --ticks 120 --scale 0.25 --out out/sens

# a founder-trait sweep (a genome locus, not a config field)
PYTHONPATH=src python3 -m evolve.cli sensitivity --sweep vision_range \
    --seeds 11 12 --ticks 120 --scale 0.25 --out out/sens

# every declared sweep
PYTHONPATH=src python3 -m evolve.cli sensitivity --sweep all --out out/sens_all
```

Six sweeps vary a **config coefficient** (`energy.bite`, `energy.eta_p`,
`environment.soil_diffusion`, `genetics.sigma_trait`,
`organisms.maturity_age`, `energy.decay`). Three vary a **founder genome
locus**: `vision_range`, `metabolic_rate` and `mutation_rate`. A trait sweep
takes a physical value (world units, × basal rate, mutations per locus per
birth), maps it back to the gene value through `GENE_SPECS`, and sets that locus
in every founder **before tick 1**; the phenotype cache is refreshed immediately.
After tick 1 the locus mutates like any other gene, so the sweep measures the
effect of the *starting condition*, not a policy the population cannot escape.

`body_size` is deliberately excluded from trait sweeps: founder reserve is
derived from body mass at construction, so overriding mass after construction
would leave every animal's starting energy inconsistent with its own body. The
declaration raises rather than silently saturating, and a value outside the
gene's physical span is an error.

### Measured smoke results (scale 0.25, 120 ticks, seeds 11/12, n = 2)

These are *probes*, not design-scale findings; n = 2 is underpowered and no
p-value here is evidence of anything.

| Sweep | Value | Final population (mean ± SD) |
|---|---|---|
| `vision_range` | 40 / 100 / 160 world units | 16.0 ± 1.4 / 16.5 ± 2.1 / 15.5 ± 0.7 |
| `metabolic_rate` | 0.6 / 1.0 / 1.6 × basal | 18.5 ± 0.7 / 17.5 ± 2.1 / 14.5 ± 2.1 |
| `mutation_rate` | 0.001 / 0.010 / 0.050 per locus | 17.0 ± 1.4 / 17.5 ± 0.7 / 17.5 ± 0.7 |

No direction is claimed from these numbers. The metabolic sweep is the only one
that moves outside its own seed-to-seed spread at this scale, and even that is
two seeds.

## Bottleneck indicator

Every `tick_stats` row carries `population_drop`, `diversity_drop` and
`bottleneck` (IMPROVE §5). The indicator compares the current sample with the
peaks of population and `mean_pairwise` **within the retained window**: when
both have fallen by at least `BOTTLENECK_DROP` (0.5), `bottleneck` is 1.
`population_drop` and `diversity_drop` are always reported so a one-sided
decline is visible.

This is a **contraction indicator**, not an effective population size: an
`Ne` estimate needs a neutral-locus coalescent model that this simulation does
not satisfy. It is also not a species count and not a fitness score. A window
with fewer than two samples reports zeros — "not yet measurable", not "no
bottleneck". The live analytics view draws the two components as a chart from
the same retained window, so the panel and the CSV describe the same quantity.

## Scaling

`--scale s` multiplies world lengths and the horizon by `s`, and plant and
founder counts by `s²`, preserving density. Field resolution stays 25 units per
cell; the number of cells scales with area. Scheduled events scale in time.

A scaled run is **smaller and shorter** than the design, so the results are a
feasibility demonstration and a record of what happened, not evidence at design
scale. Every `summary.md` states the scale used. The default (`0.5`) and the
scale actually used for the published results are recorded in
[`RESULTS.md`](RESULTS.md).

## Determinism

Every run is reproducible from `(config YAML, seed, spec_version, git_commit)`.
Setup hooks use private `Rng` streams seeded from the config seed, so they do
not perturb the engine's documented tick stream. Assay sampling uses
`base_seed + 17` per replicate; control genomes use `base_seed + 5000`; arena
positions use `rep_seed * 1000 + pair + 1`. The analysis bootstrap uses
`base_seed + 777`.
