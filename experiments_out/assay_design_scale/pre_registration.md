# Assay G — pre-registration

Written before any engine is constructed. This file is the protocol; the
results in `summary.md` and `assay.csv` are reported against it whatever they
say, including "no effect" and "random genomes did better".

Status: **pre-registered** (DESIGN §13, protocol fixed before execution).

## Parameters used in this run

| Parameter | Value |
|---|---|
| scale (area, counts, time) | `1.0` |
| evolution replicates | `6` |
| genomes sampled per replicate | `10` |
| evolution horizon (ticks) | `20000` |
| arena horizon (ticks) | `500` |
| base seed | `100` |

`scale = 1.0` is the design; values below 1 are a documented down-scale for a
single-core machine (see `evolve.experiments`).

## 1. Evolution phase

Run `configs/assay.yaml` (calibrated baseline, asexual, **no predators**,
**clumped plants**) for the evolution horizon, one run per replicate seed
`base_seed + i`. Predation is switched off at the stage level; no genome is
edited.

## 2. Sampling

From each replicate's **living** population at the final tick, draw `10`
genomes **uniformly at random, with replacement if the population is smaller**.
No ranking by energy, intake, age or any other quantity: ranking would confound
the assay with the selection that already happened.

Control genomes: `10` genomes per replicate drawn from the same founder
distribution as everyone else (`Beta(2, 2)` trait loci, `N(0, 0.35)` neural
weights, `N(0, 0.05)` biases).

## 3. Arena

For each evolved/control pair, one arena contains **both** individuals:

* identical plant layout in every arena of a replicate (one arena seed; the
  layout is shared, so neither genome can be paired with a luckier patch),
* mirrored starting positions about the arena centre,
* equal starting reserves as a fraction of each individual's own `E_max`,
* no reproduction, no predation, no mutation; the arena horizon above.

The primary endpoint is a within-arena difference, so plant luck cancels.

## 4. Endpoints

* **Primary:** total food intake in biomass units (plant + meat; meat is zero
  without predation) over the arena horizon.
* **Secondary:** final energy, survival (alive at the final tick).

## 5. Analysis

Paired differences `intake(evolved) - intake(random)` per arena. Two-sided
**Wilcoxon signed-rank** test (normal approximation, tie-corrected, zero
differences dropped) and the **mean difference with a percentile bootstrap
95 % CI** (10,000 resamples), reported per replicate and pooled over all
arenas. No alternative endpoint, subgroup or exclusion rule will be added
after seeing the data.
