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

Each experiment writes `<out>/<id>/` containing `stats.csv` (the 40-column
statistics table), seven PNGs, and `summary.md`. `--all` additionally writes
`index.md`. The assay also writes `pre_registration.md` and `assay.csv`.

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
* **D** keeps the same number of plants per unit area in the config's
  *initial count*? No: the config reduces the initial count to 40 % of A and the
  hook clusters the survivors. Plant density is therefore genuinely lower than
  A, and patches are separated by empty ground.
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
