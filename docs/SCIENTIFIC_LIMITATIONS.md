# Scientific limitations

What this model is, and — more importantly — what it is not. These are
limitations of the *implemented* system, not aspirations to fix later.

## Scale and time

* The design-scale programme (30,000-tick runs, 1,000 founders) needs ~7.5 h of
  wall clock on this single-core machine (`BENCHMARKS.md`); the published
  results are scaled-down runs (smaller worlds, shorter horizons). A scaled run
  is a feasibility demonstration, not an equivalent experiment: less simulated
  time means less room for adaptation, and smaller populations mean stronger
  drift.
* Sensing is 65 % of a tick. Optimising it would change what is affordable, but
  the measured numbers here are for the unoptimised engine.

## Model structure

* **No speciation.** Lineages are parentage chains, not species. Any statement
  about "species" in the results is informal.
* **No learning.** The controller is fixed at birth. Within-lifetime
  behavioural change is impossible by construction.
* **No culture, no communication beyond a pheromone field** that organisms
  deposit when they strike and that decays in ~10 ticks. Nothing in the model
  can represent information transfer between generations except genomes.
* **Coarse environment.** The world is a 25-unit grid for fields; organisms are
  continuous but their environment is not. Sub-cell structure (a plant at the
  edge of a dry cell) is invisible to the model.
* **Scalar detritus has no position.** Non-predation deaths and undigested food
  enter a global detritus pool whose mineralisation returns nutrients evenly
  across the grid. This is a documented simplification; it slightly smooths
  nutrient hotspots that a spatial detritus pool would create.
* **Damage credit is truncated** at 8 attackers per victim (oldest dropped).
  With more than 8 attackers the meat split is approximate; the mass ledger
  still closes exactly.
* **`escapes` is never counted.** Sprinting is a continuous modifier, so there
  is no discrete escape event; the statistics column exists and is always zero.
* **`genetics.mode = both`** falls back to asexual reproduction when no mate is
  available; it is not exercised by any published experiment.
* **The drought model differs from the design text.** DESIGN §13 C says
  "plant growth −70 %"; the implemented drought at magnitude 0.6 gives
  moisture −60 % and growth −48 % (growth factor `1 − 0.8·m`). The gap is
  reported, not papered over.
* **Vision is capped at 160 units** and the food sensor only reports the
  *nearest* plant or carcass. Dense patches are indistinguishable from one big
  plant at the sensor level.

## Statistical limitations

* Every published comparison is a small number of runs (often one per arm).
  Nothing here is a significance statement about nature; the assay's Wilcoxon
  and bootstrap CI describe the paired arenas that were actually run, at the
  scale that was actually run.
* Founder effects are real: with 48–75 founders, run-to-run variance is large.
  Where a result is fragile to the seed, the results say so.
* Extinction is common at small scales and is reported as an outcome, not
  treated as a failure of the machinery.

## What would be needed to claim more

* Design-scale runs of A–H (the estimates and the machine constraint are in
  `BENCHMARKS.md`).
* Multiple seeds per arm with the same horizon, and a pre-registered analysis
  plan for each experiment (only the assay has one).
* A spatial detritus pool and a finer field grid, if nutrient hotspots matter
  for the question being asked.
* Profiling and optimisation of sensing before scaling populations further.

## Long-horizon viability (measured after calibration)

The calibrated ecosystem's long-horizon fate is **seed-dependent**. At design
scale (300 founders, 1,000 × 1,000 world):

* seed 1 reproduces for ~6,000 ticks, then declines to extinction by ~11,950
  while plants sit at carrying capacity and survivors hold full reserves. The
  proximate cause is the neural reproduction output drifting below the fixed
  0.5 decode threshold, not starvation (`RESULTS.md`, "Diagnostic");
* seed 2 is still alive at tick 12,000 (76 organisms, births continuing).

Consequences:

* at 48 founders (the scale-0.4 programme) the founder lottery is usually lost:
  4 of 6 long runs went extinct, one before its own manipulation even started;
* design-horizon readouts (30,000 ticks) describe a transient regime whose end
  state depends on the founder draw, not a guaranteed equilibrium;
* the calibration criterion "non-extinct equilibrium" was met at its 6,000-tick
  test horizon; persistence beyond it is not guaranteed (`CALIBRATION.md` §8).

This is reported as measured. The constants were not retuned after calibration:
DESIGN §18 allows one calibration window, and it is closed.
