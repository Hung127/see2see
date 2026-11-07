# EVOLVE — upgrade report

This is the completion report for `docs/my-plans/IMPROVE.md`. It answers that
document's §13 deliverables in order and says, for every item, whether it is
**fully implemented**, **partial**, or **deferred**.

Two reading rules apply throughout, and they are the same rules the upgrade was
built under:

* **Every performance claim is a measurement** from this machine (1 vCPU,
  ~1 GB RAM, CPython 3.14, NumPy only), reproducible with the commands shown.
  Nothing is extrapolated except where it says "estimate".
* **Every scientific claim names the experiment behind it and its
  limitations.** An underpowered result is reported as underpowered, not
  quietly rounded up to a finding.

Baseline for all comparisons: commit `a66cc0c` (the last commit before the
upgrade). Work was done directly on `main`; nothing was pushed to a remote.

---

## 1. Repository audit — **DONE**

[`docs/AUDIT.md`](AUDIT.md) is the Phase 1 deliverable, written from reading the
code and profiling a real tick, before any change. It records:

* what each module actually does (config through CLI);
* the architectural strengths that had to be preserved (one seeded RNG, stable
  IDs, deterministic slot reuse, `checksum()` coverage, the dual mass/energy
  ledger, a `sim`/`viz` split with no pygame in the engine);
* eight measured or read-from-code weaknesses, written as diagnoses rather than
  guesses;
* a ten-item change list with a risk column and a determinism column.

The two weaknesses that most shaped the rest of the work were real and are worth
repeating: sensing was **68 % of a tick** (2.79 s of 4.20 s at 1,000
organisms), and there was **no way to record a per-organism history** —
`death_cause` was overwritten when a slot was reused, so long-run mortality
structure was lost.

Baseline suite before the upgrade: **196 tests**, ~87 s, all green.

---

## 2. Architecture changes — **DONE**

Nothing was rebuilt. Six seams were added, each additive and each off by
default or presentation-only:

| Seam | Module | What it is | Default |
|---|---|---|---|
| Controller protocol | `controllers.py` | `Observation → Controller → Action`; closed registry (`neural`, `random`, `passive`, `rule`, `forage`) | `Engine.controller = None` → the exact pre-upgrade MLP path |
| Controller evaluation | `controller_eval.py` | paired-seed comparison with Wilcoxon + bootstrap CI | opt-in CLI command |
| Sensitivity sweeps | `sensitivity.py` | nine declared one-factor sweeps (six config coefficients + three founder trait loci) | opt-in CLI command |
| Derived statistics | `stats.py` | 18 appended columns; death-cause counters; age-at-death histogram; `LineageRecorder` | stats always on, lineage recorder opt-in |
| Analytics history | `viz/analytics.py`, `viz/charts.py` | bounded, tick-sampled ring buffer + dependency-free chart primitives | presentation only |
| Ancestry view | `viz/ancestry.py` | tree built from recorded links every frame | presentation only |

Invariants that were preserved and are now enforced by tests rather than by
memory:

* **`checksum()` identity.** Every engine optimization leaves `Engine.checksum()`
  bit-identical on three pinned scenarios (plain / sexual / no-predation,
  60 ticks, seed 11). `tests/test_optimization_equivalence.py` holds the three
  baseline SHA-256 values from `a66cc0c` and fails if any moves.
* **RNG draw order.** No optimization changed when or how many draws happen.
* **Stable IDs and tie-breaks.** `spatial.best_per_query` picks the same winners
  as the three `lexsort` passes it replaced, including distance and ID ties.
* **Snapshot/resume equivalence.** New snapshot fields (death counters, the
  histogram) are restored, and they are *excluded* from `checksum()` because
  nothing in the tick reads them — the contract is "arrays that can influence
  future dynamics".
* **The engine never imports pygame or matplotlib**, and the viewer never
  advances or mutates the simulation.

`SPEC_VERSION` was **not** bumped: no change in this upgrade altered simulation
semantics, so runs remain reproducible under the existing version. That is a
claim the equivalence suite checks, not an assumption.

---

## 3. Engine optimization report — **DONE**

Hot path first, measured with `cProfile` at 1,000 organisms (see
[`BENCHMARKS.md`](BENCHMARKS.md) for the raw tables). Five changes, all
bit-identical:

1. **`world.dx`/`dy` in place.** `(d + w/2) % w − w/2` allocated four
   temporaries per axis; it now writes into a reused array (`np.subtract(...,
   out=)` + `np.mod(out=)`). New `World.delta` and `World.distance_and_bearing`
   compute a pair's periodic wrap **once**, where the old code wrapped it twice
   (once for distance, again for bearing). That is ~6 pair-length allocations
   removed per `distance()` call.
2. **Distance before `arctan2` in `_nearest_visible`.** The 5×5 stencil hands
   back ~144 candidate rows per query; the vision cone keeps a handful. The
   radius test is now applied before the transcendental, rejecting ~80 % of
   candidates first.
3. **`_feed` / `_predation` query only actors.** Both used to build candidates
   for the whole population and then discard every row belonging to an organism
   that was not eating or attacking. They now query with the eater/attacker
   positions and remap rows afterwards — and because `attackers` is ascending,
   each actor's candidate order (hence the ID tie-break) is unchanged.
4. **No full plant-biomass field in `collect`.** Sensing rebuilt a 40×40 field
   (`np.add.at` over ~20,000 plants) every tick to sample the ~1,000 cells
   organisms actually stood in. It now bins plants into a flat index and gathers
   only those cells.
5. **`spatial.best_per_query`.** Per-query minimum first, then tie-break only
   the survivors, replacing three full `np.lexsort` passes over ~1–4 million
   candidate rows.

One optimization was **measured and rejected**: the conditional wrap
(`np.subtract(d, w, out=d, where=d > w/2)`) is not bit-identical to `np.mod` —
it differs by up to **2.3e-13**. A faster tick that produces different
organisms is a different experiment, so `np.mod` stays. This is recorded as a
comment in `world.py` and in the benchmark document so it is not re-attempted.

---

## 4. Before/after benchmarks — **DONE**

Same method and machine. `evolve.bench` keeps the calibrated default's
*density* (20 plants per organism, ≈6e-3 plants/unit²), so per-organism sensor
work stays comparable, and reports the mean population next to the nominal one.
Short horizons kept populations near nominal, so the numbers are valid at the
stated sizes.

| Population | Before (ticks/s) | After (ticks/s) | Speedup |
|---|---|---|---|
| 300 | 14.30 | **24.98** | 1.75× |
| 1,000 | 4.53 | **8.36** | 1.85× |
| 5,000 | 0.80 | **1.43** | 1.79× |

Tick time at 1,000 organisms fell **4.542 s → 2.407 s (−47 %)**; `_feed` fell
1.239 s → 0.352 s. The remaining cost is the periodic wrap itself (`np.mod` on
every candidate pair), which cannot be made cheaper without changing results.

**The DESIGN §11 targets are still missed** and this is stated rather than
glossed: 8.36 ticks/s at 1,000 organisms against a target of 12. The pass bought
~1.8×, not the 2.7× that target would need. Throughput is almost exactly linear
in population (4,200–4,900 organism-ticks/s across the whole range), so the cost
is per-organism work, not per-tick overhead.

---

## 5. Peak and steady-state memory — **DONE**

Measured with `tracemalloc` and process RSS at 1,000 organisms:

| | Before | After |
|---|---|---|
| Traced steady state | 6.5 MB | **6.5 MB** |
| Traced peak | 56.4 MB | **56.4 MB** |
| Process peak RSS | 122 MB | **113 MB** |

The optimization pass did not raise memory. In the viewer, with the 5,000
organism / 100,000 plant stress point included, the process high-water mark was
**339.7 MB** for the whole measurement run and **88.0 MB** at the end with the
viewer and analytics history live. Everything is inside the **< 650 MB** target.

Bounded by construction, not by hope:

* `StatsRecorder.rows` is a ring buffer;
* `AnalyticsHistory` is `HISTORY_CAP = 600` samples at `SAMPLE_INTERVAL = 20`
  ticks;
* the age-at-death histogram is a fixed 40 × 25-tick array (O(1) memory for a
  survival curve);
* `LineageRecorder` holds at most one record per living organism plus a deque of
  `max_finished`;
* the ancestry view is rebuilt every frame and stores nothing;
* charts draw at most `MAX_POINTS = 180` points per series.

---

## 6. New scientific metrics — **DONE**

Eighteen columns were **appended** to `stats.COLUMNS` (never reordered, so old
CSVs still parse by name), in three additive blocks — each new block exists only
because the previous one was already committed, and the schema rule is "append,
never reorder":

| Column | What it is | Observed or derived |
|---|---|---|
| `birth_rate`, `death_rate` | per organism per tick, over the sample interval | derived from counter differences |
| `mean_generation`, `max_generation` | generation turnover | observed (engine's counter) |
| `trait_variance` | mean variance over the 12 trait loci | observed |
| `nn_variance` | mean variance over the 380 network weights | observed |
| `mean_organism_biomass` | Σ(body + E)/n over living organisms | observed |
| `deaths_starvation`, `deaths_senescence`, `deaths_health`, `deaths_predation` | causes, reconciled against `deaths_total` | observed |
| `survival_median` | age at which half of all deaths fall | derived (histogram bin midpoint — stated as such) |
| `population_cv` | coefficient of variation over the retained window | derived |
| `environmental_stress` | mean drought stress **sampled at the living organisms' cells**, not world-wide | observed |
| `behavioural_diversity` | mean pairwise distance between per-tick lifetime activity profiles (distance moved, bites, hunts, reproduction attempts, rest), on a ≤200-organism stride | derived from measured counters |
| `population_drop`, `diversity_drop` | fall from each series' own peak (population, `mean_pairwise`) within the retained window | derived |
| `bottleneck` | `1` when **both** drops reach `BOTTLENECK_DROP` (0.5), else `0` | derived indicator, **not** an effective population size |

`behavioural_diversity` is deliberately **not** genetic diversity: it never reads
a genome. Two clones can behave differently and two genomes can behave
identically, which is exactly why the brief lists it as a separate axis. The
five activity dimensions are divided by each animal's own age and then by fixed
*declared* reference rates (3 u/tick for movement, 1 event/tick for the rest) —
normalising by the sample's own spread would give every run the same score.

`environmental_stress` samples `fields.water_stress` at each animal's position,
the same lookup the metabolism stage uses to charge the drought cost. Reporting
the field mean instead would understate a population sheltering in dry cells.

`bottleneck` is the honest form of the §5 "bottleneck detection" request. What
can be measured without a neutral-locus model is a **contraction**: the current
population and the current genetic spread are compared with their own peaks in
the retained window. When both have fallen by at least `BOTTLENECK_DROP` (0.5),
the sample is flagged. `population_drop` and `diversity_drop` are always
reported so a reader can see a one-sided decline rather than only the flag. This
is deliberately **not** an effective population size (that needs a coalescent
model the simulation does not satisfy), not a species count, and not a fitness
score. A window with fewer than two samples reports all zeros — "not yet
measurable", not "no bottleneck".

Supporting engine state, all statistics-only and excluded from `checksum()`:

* `death_counts` — a cumulative counter per cause, reconciled against
  `deaths_total` on every death;
* `death_age_hist` — 40 bins × 25 ticks, giving a survival curve without
  storing per-organism histories;
* `LineageRecorder` — a bounded parent/child log, opt-in, with real birth and
  death ticks and causes.

**The distinction the brief asks for is kept.** "Observed" columns are measured
from entities; "derived" columns are arithmetic on counters and are labelled so
in the code. Diversity proxies are named proxies: `mean_pairwise`,
`mean_variance`, `diet_entropy`, `unique_genomes`, `trait_variance`,
`nn_variance`. **None is called species diversity**, and `trait_variance` and
`nn_variance` are separate columns on purpose — they answer different questions
("is the phenotype pool narrowing?" vs "are the controllers diverging?").

### Coverage against the IMPROVE §3.3 checklist

The brief lists candidate statistics rather than a fixed schema. Where an item
is served by an existing column the report names it instead of adding a
duplicate column measuring the same quantity twice:

| §3.3 group | Item | Where it lives | Kind |
|---|---|---|---|
| Population dynamics | birth rate, death rate, growth rate | `birth_rate`, `death_rate`, `growth_rate` | derived |
| | extinction events | `StatsRecorder.extinctions` / `extinction_events()` | observed |
| | survival duration | `survival_median` + `death_age_hist` | derived |
| | population stability | `population_cv` | derived |
| | generation turnover | `mean_generation`, `max_generation` | observed |
| Evolution | trait distributions | `_blocks` histograms (body, reserve) + per-trait means | observed |
| | trait variance / genetic diversity / NN diversity | `trait_variance`, `mean_pairwise`, `nn_variance` | observed |
| | mutation statistics | `mean_mutation_rate`, `mutations_per_birth` | observed / derived |
| | lineage depth | `LineageRecorder` generation + ancestry view | observed |
| | generation-wise trait changes | `experiments._trait_drift` (first vs last), experiment summaries | derived |
| Ecology | plant / organism / carcass biomass | `plant_biomass`, `mean_organism_biomass`, `carcass_biomass` | observed |
| | resource consumption | `plant_intake`, `meat_intake` (run totals) | observed |
| | predator/prey relationships | `strikes`, `strikes_hit`, `deaths_predation` | observed |
| | trophic distribution | `herbivores`/`carnivores`/`omnivores`/`unclassified` (intake evidence, not diet gene) | derived |
| | energy efficiency | `foraging_efficiency` (stats column); `energy_efficiency` (controller-eval endpoint) | derived |
| | environmental stress | `environmental_stress` | observed |
| | population density | not a stored column — computed from `population` and world area | derived |
| Behavioural | movement, rest, feeding, attack, reproduction | per-organism counters + `mean_*` columns + `rest_fraction` | observed / derived |
| | behavioural diversity | `behavioural_diversity` | derived |

Two deliberate omissions: **population density** is not stored because it is a
constant-area division of a column that is stored (and the arena size is in the
config), and a stored duplicate would be a second definition to keep in sync;
**trophic class** is derived from observed intake rather than the diet gene, so
`diet_preference` stays a preference and `herbivores` stays evidence. Extinction
events are kept as a compact list of distinct genome hashes, not per-organism
history.

### Honest gaps in this area

Two gaps found by the audit were **fixed** after the report was first written.
Both are statistics-only, so they change no state transition, no RNG draw and
no checksum (`SPEC_VERSION` stays `"1"`); both are covered by regression tests.

* `escapes` is now incremented in `Engine._predation`: a **damaging strike whose
  victim is still alive after the tick's strikes resolve** counts as one
  survived attack. A victim hit by two attackers and still standing is credited
  twice — the counter answers "how many attacks has this animal survived", not
  "how many fights". The letter of the contract is in DESIGN §8. Note that the
  count is not proof of evasion: it measures survival of a strike, which may
  also be defence or sheer body mass.
* `rest_fraction` is now a **lifetime mean per living organism**,
  `Σ rest_ticks / Σ age`, instead of `Σ rest_ticks / n / tick`. The old formula
  was only correct for a cohort born at tick 0; the new one is the mean
  fraction of its own life that an average living animal has spent resting.

Neither counter is read by the tick, so neither is part of `checksum()`; both
are snapshotted with the other per-organism counters.

---

## 7. New experiment descriptions — **DONE (harnesses) / PARTIAL (runs)**

Two harnesses were added. Both are observations: they copy the base config per
run, never mutate it, and never write back a calibration parameter.

### Controller comparison (`evolve.cli controllers`)

Paired seeds: every controller runs on the same base seed, so the world, plant
layout, founder genomes and every environmental draw are identical. The only
variable is the decision function. All controllers see the same 16-channel
observation and all outputs go through `actions.decode`, so a baseline pays
identical movement, feeding and reproduction costs.

Endpoints (append-only order, so old CSVs keep their column meaning):

| Group | Endpoints |
|---|---|
| original | `final_population`, `births`, `deaths`, `mean_energy`, `mean_organism_biomass`, `survival_median`, `foraging_efficiency` |
| IMPROVE §4.2 | `survival_rate`, `mean_lifespan`, `reproduction_success`, `energy_acquired`, `energy_efficiency`, `movement_efficiency`, `feeding_success`, `predation_success`, `behavioural_diversity`, `environmental_stress` |

The §4.2 endpoints come from a population-level cumulative `Engine.run_totals`
dict (13 online accumulators: distance, rest ticks, feeding attempts/successes,
plant and meat intake, assimilated energy, strikes/hits, reproduction
attempts/successes, metabolic cost, summed age at death). They are *statistics*:
snapshotted so a resumed report does not lose them, but excluded from
`checksum()` because nothing in the tick reads them. `energy_acquired` is energy
actually added to reserves *after* assimilation, so a poor diet is not credited
with food it could not use.

Definitions the report states verbatim: `survival_rate` = alive at horizon /
(starters + births); `mean_lifespan` = Σ age-at-death / deaths; ratios return
`0.0` on a zero denominator and the report notes that `0.000` means **no attempt
in this window**, not a measured zero.

**Multiple scenarios.** `controllers --scenarios a b f` (or `--scenarios default
f`) evaluates every controller in every declared experiment × seed, applying
that scenario's setup hook at construction, its supervised step options and its
scheduled intervention. The CSV gains a `scenario` column *only* when more than
one environment runs; a single environment — including `--scenarios default` —
keeps the original shape. Cross-scenario results are summarised with three
declared numbers per controller (`persistence_rate`, `mean_survival_rate`,
`worst_survival_rate`, plus `mean_reproduction_success`). They are explicitly
**not** a hidden fitness score, and the report says the scenario set is a
declared list, not a sample of environments.

Statistics: paired differences against `neural` on the same scenario and seed,
percentile bootstrap CI and the in-house Wilcoxon signed-rank test already used
by assay G.

### Sensitivity sweeps (`evolve.cli sensitivity`)

Nine declared sweeps, each with a **pre-registered hypothesis**, a rationale, and
values bracketing the calibrated default. Six vary a **config coefficient**;
three vary a **founder trait locus** (the genome value every founder is born
with, converted from a physical quantity through `GENE_SPECS` before tick 1).

| Sweep | Parameter | Kind | Values | Baseline |
|---|---|---|---|---|
| `bite` | `energy.bite` | config | 0.25 / 0.50 / 0.75 / 1.00 | 0.50 |
| `eta_p` | `energy.eta_p` | config | 0.50 / 0.75 / 1.00 | 0.75 |
| `soil_diffusion` | `environment.soil_diffusion` | config | 0.02 / 0.10 / 0.30 | 0.10 |
| `sigma_trait` | `genetics.sigma_trait` | config | 0.02 / 0.08 / 0.20 | 0.08 |
| `maturity_age` | `organisms.maturity_age` | config | 25 / 50 / 100 | 50 |
| `decay` | `energy.decay` | config | 0.005 / 0.015 / 0.045 | 0.015 |
| `vision_range` | `trait.vision_range` | founder locus (world units) | 40 / 100 / 160 | 100 |
| `metabolic_rate` | `trait.metabolic_rate` | founder locus (× basal) | 0.60 / 1.00 / 1.60 | 1.00 |
| `mutation_rate` | `trait.mutation_rate` | founder locus (per locus per birth) | 0.001 / 0.010 / 0.050 | 0.010 |

Every run is paired by seed. The report gives per-value mean/SD, the paired
difference against the baseline value with a bootstrap CI and a Wilcoxon
p-value, a bottleneck rate, and states that extinction is reported, not averaged
away. The trait sweeps vary **what an animal is born with**, not a global
coefficient: `body_size` is excluded because founder reserve is derived from body
mass at construction, so overriding mass afterwards would make starting energy
inconsistent with body. After tick 1 the locus mutates like any other gene, so a
trait sweep measures the effect of the starting condition, not a policy.

**Both are implemented and tested; only short smoke runs were executed here**
(see §8). Design-scale runs are left to the experiment programme, which already
documents its own scaled-vs-full distinction.

The three trait sweeps map directly onto the §3.4 experiment list: `vision_range`
is the vision-range sensitivity experiment, `metabolic_rate` is the metabolic
trade-off experiment and `mutation_rate` is the mutation-rate sensitivity
experiment. A short probe (scale 0.25, 120 ticks, seeds 11/12) gave final
populations of 16.0 ± 1.4 / 16.5 ± 2.1 / 15.5 ± 0.7 for vision 40/100/160,
18.5 ± 0.7 / 17.5 ± 2.1 / 14.5 ± 2.1 for metabolic rate 0.6/1.0/1.6 and
17.0 ± 1.4 / 17.5 ± 0.7 / 17.5 ± 0.7 for mutation rate 0.001/0.010/0.050. With
two seeds no direction is claimed; the numbers are recorded as a feasibility
check of the harness, not as findings.

---

## 8. Controller comparison results — **DONE (measured), NULL at this power**

Run: `default.yaml`, **120 ticks**, paired seeds **[11, 12]**. 120 ticks is far
too short for an evolutionary question, so this is a *smoke measurement of the
harness*, not a result about controllers. It is reported because the brief asks
for the numbers, and framed because it is underpowered.

| Controller | Seed 11 pop | Seed 12 pop | Mean energy (11/12) |
|---|---|---|---|
| `rule` | 60 | 60 | 12.02 / 12.87 |
| `forage` | 60 | 60 | 11.12 / 10.62 |
| `random` | 50 | 48 | 4.82 / 4.91 |
| `neural` | 47 | 44 | 3.69 / 3.70 |
| `passive` | 40 | 41 | 3.28 / 3.23 |

Paired against `neural` on the same seed, every controller's
`final_population` difference carries **p = 0.1797**. With n = 2 that p is the
smallest value the test can reach, so the result is **null: the comparison is
underpowered and says nothing about whether the reflex controllers are better**.
The report says exactly that in its own "Interpretation limits" section.

Three caveats that matter more than the table:

* **The neural arm is the shipped *random* MLP.** Founders use random genomes,
  so this table compares a random network to hand-written reflexes — it is
  **not** evidence about evolution. The harness now also implements the
  warm-up condition (`controllers --warmup N`, `warmup_ticks`): the model's own
  MLP evolves the population for N ticks, that state is snapshotted, and every
  arm is resumed from the *same* snapshot so only the decision function
  differs. Births/deaths are then counted from the resume point. It is off by
  default and the report states which condition ran; no warm-up run is claimed
  here, because none was executed at design scale.
* **A controller that survives longer but reproduces less is not "better" by
  definition**, so population and births are reported separately rather than
  combined into one score.
* No calibration parameter was touched to make any arm look better.

Reproduce: `PYTHONPATH=src python3 -m evolve.cli controllers --out <dir>
--seeds 11 12 --ticks 120`.

### Multi-scenario run — **DONE (measured), still n = 2**

The same harness was run across three environments, at term `--scale 0.25
--ticks 150 --scenarios default d f --seeds 11 12` (30 runs). This is a
different question from the table above: not "who is best" but "does the
ranking survive a harder world".

| Controller | Mean survival | Worst survival | Persistence | Mean repro success |
|---|---|---|---|---|
| `rule` | 0.823 | 0.522 | 1.000 | 0.910 |
| `forage` | 0.786 | 0.565 | 1.000 | 0.872 |
| `random` | 0.718 | 0.526 | 1.000 | 0.415 |
| `neural` | 0.576 | 0.286 | 1.000 | 0.833 |
| `passive` | 0.502 | 0.316 | 1.000 | 0.000 |

What separates is **conditional on the scenario**: in `default` the reflex
policies reach 3–4× the neural arm's population (rule 527/576 vs neural
175/180), but in the clumped-resource scenario `d` every arm collapses to
single digits (rule 15/12, neural 8/6). That collapse is the honest result of a
world one quarter the size with 40 % of the plants; it is **not** an extinction
(no arm reaches zero) and it is reported, not averaged away.

The paired tests still stop at **p = 0.1797** on n = 2 — the smallest value
reachable — so even with this separation the correct statement is that the
*effect is large and consistent across both seeds*, while *statistical support
is absent at this sample size*. No arm is declared better. Two extra seeds
(`--seeds 11 12 13 14`) would be the cheapest way to change that; it was not
run here, so it is not claimed.

A sensitivity smoke run (`bite`, scale 0.25, 60 ticks, seeds 11/12) gave final
population 19.0 / 19.5 / 19.5 / 19.5 across `bite` 0.25 / 0.5 / 0.75 / 1.0 with
p = 0.317 / 1.000 / 1.000 — also **underpowered and reported as null**, not as
"bite has no effect". A longer run at scale 0.5 (300 ticks) did separate the low
end (`bite` 0.25 → 5.0 ± 1.4 final population vs 16.0 ± 1.4 at baseline,
unpaired p = 1.000 with n = 2), which is a direction consistent with the
pre-registered hypothesis but still far short of evidence.

---

## 9. Visualization changes — **DONE**

The viewer is a separate presentation layer throughout: it reads engine state,
never writes it, and the engine never imports pygame.

**World view.** Organism size encodes body mass; colour encodes the inherited
`diet_preference` gene (green → yellow → red), with the role *measured from
lifetime intake* shown beside it because preference and diet can disagree.
Terrain, plants, carcasses and event outlines are drawn viewport-culled.
Local events get a coloured ring; worldwide events (radius ≥ 1e8) draw none and
say "worldwide".

**Overlays (`v`).** Nine layers: climate, moisture, soil, temperature, terrain
— plus the four added here: **plants**, **organisms**, **pheromone**,
**vision**. Every single-quantity layer prints its numeric span and unit in the
badge; the vision overlay is *derived at draw time* from the living population
(no engine vision field, so snapshots and checksums are untouched) and draws no
marker for an empty cell — **no-data is visually distinct from short-sighted**.

**Perception overlay (`p`).** Vision cone plus the targets the network actually
detected this tick, reconstructed **from the sensor vector itself**
(`closeness = 1 − d/r`, `bearing = θ/π`). A `−1` "nothing detected" channel draws
no marker, so nothing is drawn that the organism did not see.

**Inspector.** A proper panel, not text over the map: id, generation, age,
reserve/capacity, health, body mass, diet gene *and* measured role,
**controller**, speed/vision/cone, attack/defence, parent IDs, lifetime intake,
reproduction attempts vs successes, lifetime actions, distance moved.

**On-demand network inspection (`w`, §4.3).** The brief asks for neural
inspection but explicitly says *not* to render all weights continuously, so the
inspector shows a per-layer **summary** for the selected animal only: parameter
count, mean |weight| and value range for `W1/b1/W2/b2/W3/b3`, the total
parameter count and the single largest-magnitude connection with its index.
`W` toggles it; it is off by default and is built from the same split of the
genome's neural block that `brain.forward` multiplies, so it describes the
network the animal actually ran. The engine's per-organism genome, physical
traits, neural outputs, decoded actions and lifetime counters are all in the
same panel; only the 380-number dump is withheld.

**Analytics (`a`).** 22 chart blocks, each declaring **one unit** — population;
births and deaths; biomass pools; reserve and body; speed; vision; genes;
genetic variance; pairwise distance; temperature; moisture; generation depth;
median age at death; deaths by cause; an age-at-death histogram; **live body-mass
and reserve distributions** (computed at draw time, not stored per sample); CV;
**drought stress where the animals are**; **behavioural diversity**;
**contraction from the window's peak** (IMPROVE §5); behaviour
totals. Mixed-magnitude series were **split** rather than overlaid, so
no axis has a meaningless unit. Charts print unit and range, `nan` breaks the
line instead of becoming zero, and a series with no samples says so instead of
drawing through nothing. Sampling is on **simulated ticks** (every 20), so the
x axis means ticks at any speed, and the chart history is capped.

**Ancestry (`g`).** Full-screen tree from recorded links only: the selected
animal's recorded ancestor chain, then recorded descendants with generation,
alive/dead (cause and tick of death), and the phenotype recorded at birth. A
trait line prints only where it differs from the parent's — in one 400-tick
fixture 544 of 579 recorded children were genuinely trait-identical, so
printing "identical" under every node buried the branches that did change. A
**founder** ("no parent recorded — founder or born before logging") and an
**evicted/never-logged parent id** ("unknown") are worded differently because
they are different epistemic states. Default root is the living organism with the
most recorded descendants. Node/depth caps are reported in the header, never
applied silently. `ASCII` indentation only: the bundled default font renders
box-drawing glyphs as identical tofu boxes (verified glyph-by-glyph).

**Controls (`§6.6`).** Start/pause/resume (`space`), step (`s`), four speeds
(`1`–`4`), pan (arrows), zoom (`+`/`-`, `home`), selection (click, `x`, `f`),
overlay cycling (`v`), panel visibility (`i`), help (`h`). Displayed: tick,
speed, population, plant count, the selected organism, and **the loaded run**
(the config file's basename, passed from the CLI — omitted when the caller does
not supply one, rather than invented). The simulation runs behind every view and
works headlessly.

**Rendering cost.** `draw_world` culling was vectorised in NumPy (drawing still
goes through `pygame.draw`/`fill`, and the frames are pixel-identical to the old
scalar path). Plant drawing fell 27 ms → 3.2 ms at default density and 27 ms →
9.8 ms at 20,000 plants; the whole world layer is ~20 ms default and ~28 ms at
1,000 organisms / 20,000 plants. A batched terrain blit was measured and
**rejected**: 187k–321k differing pixels at common zooms, and slower once zoomed
in. The viewer drops frames rather than the simulation dropping steps. When the
largest scene cannot hit 60 FPS, that is stated, not fixed with a cheaper
approximation that changes what is on screen.

---

## 10. Test results — **DONE**

`python3 -m pytest tests/ -o addopts="" -q` → **389 passed in 144.35 s**.
(Earlier stages of the upgrade: 343 passed in 266.55 s, then 346 in 259.34 s,
then 357 in 176.12 s, then 365 in 145.99 s, then 374 in 200.38 s, then 387 in
147.49 s; the runtime moves with the machine's load, not with correctness.)
Suite grew from 196 to 389 tests. No existing assertion was weakened or
deleted; several were strengthened. The notable ones:

* `tests/test_optimization_equivalence.py` pins the three pre-upgrade checksums.
* `tests/test_controllers.py` (38) covers the output contract, reproducibility,
  the default path being bit-identical to a run with no controller, and the
  controller using its **own** RNG so the engine stream is untouched. It also
  pins the §4.2 endpoint set: every declared endpoint is present and bounded,
  the endpoints reproduce across runs, the summary names them, a warm-up resume
  differences the run totals from the resume point, and multi-scenario mode
  runs every controller in every scenario × seed with a nested analysis and a
  bounded robustness block (and rejects unknown scenario names).
* `tests/test_sensitivity.py` (17) covers config copy-not-mutate, type
  preservation, real config fields, and extinction being reported. The six new
  tests pin the trait-level sweeps: `_trait_locus` inverts the gene map for
  every declared trait (linear and log arms) and reproduces the physical value
  through `phenotype`, out-of-range and non-positive values raise rather than
  saturate, `body_size` is refused at founding, a trait sweep leaves the base
  config untouched and is reproducible with different checksums per value, and
  the summary names the independent variable's kind.
* `tests/test_metrics.py` (28) covers the appended columns and that the original
  block is a prefix of `COLUMNS`, plus the two second-block metrics: identical
  profiles score zero, differing activity moves the score, the metric ignores
  genomes, the profile is a per-tick rate (10-tick-old and 100-tick-old animals
  with the same rate agree), and stress is sampled at the animals' cells rather
  than world-wide. Four pin the cumulative run totals: they are excluded
  from `checksum()`, they reconcile with the tick (successes ⊆ attempts,
  assimilated energy ≤ intake, summed ages ≥ deaths), they survive their
  organism's death, and `death_age_sum` is the exact mean-lifespan denominator.
  Seven pin the §5 indicator: no earlier sample reports zeros, a joint collapse
  flags, a mild decline does not, a one-sided drop does not, a zero peak
  diversity cannot yield a fractional drop, and the recorder's three new columns
  leave the checksum unchanged and are sampled rather than state.
* `tests/test_snapshots.py` (3) now also asserts the run totals survive a
  snapshot round-trip bit-for-bit and stay out of the checksum.
* `tests/test_cli.py` (8) includes the new multi-scenario smoke run and the
  unknown-scenario error.
* `tests/test_lineage.py` (15) covers the bounded recorder, real birth traits,
  no fabrication, and that attaching it does not change the checksum.
* `tests/test_viz_ancestry.py` (15), `test_viz_charts.py` (35),
  `test_viz_panels.py` (46), `test_viz_render_lod.py` (7) and
  `test_viz_smoke.py` (6) cover the viewer headlessly. Two chart tests also pin
  the §5 contraction chart: it is declared with one unit, its three series are
  bounded fractions in the history, and the history's value equals the
  `bottleneck()` result recomputed from its own earlier samples. The four new
  panel tests
  pin the on-demand network summary: off by default, `W` toggles it, the rows
  describe the real weights (and a zero network reports zero rather than
  hiding), a missing payload does not crash the inspector, and opening the
  summary leaves the run's checksum untouched.

Two real bugs were found and fixed by new tests rather than papered over:

1. **`LineageRecorder` records did not match `LINEAGE_COLUMNS`.** `record_birth`
   created an `age_at_death` key that the schema, the SQLite table,
   `append_lineage` and `read_lineage` all omit. The pre-existing assertion
   passed only because its fixture recorded **zero** deaths and iterated an empty
   list. The record now keeps exactly the declared schema (age at death is
   `death_tick − birth_tick`), and the fixture was strengthened to produce
   finished records.
2. **The viewer printed a false claim.** The ancestry view said lineage logging
   was on, but `launch()` never attached a recorder. The *code* was fixed (a new
   `make_app()` attaches the bounded recorder) rather than the sentence
   weakened, and a test now asserts the viewer's run is bit-identical to a plain
   run.

A third, smaller fix: the ancestry header derived its generation span from the
ancestor chain alone and printed "generation 0" above a tree spanning 0–4. It
now accumulates the range over every drawn node.

Tests were **scaled down** for this machine (smaller worlds, fewer organisms,
shorter horizons) and each such fixture says so. That is a deliberate
compromise, documented here: they prove behaviour, not design-scale outcomes.

---

## 11. Known limitations — **DONE**

Reported, not hidden:

* **DESIGN §11 throughput targets are missed** — 8.36 vs 12 ticks/s at 1,000
  organisms. The remaining cost is semantics-preserving-unavoidable.
* **`escapes` counts survived strikes, not successful evasions.** A strike that
  does no damage is not an escape; a strike the victim survives because of
  armour or size still is. The column is honest about what it measures, but it
  is not a measure of agility.
* **`rest_fraction` is a lifetime mean** over living organisms, not a per-tick
  population rate, and animals that died are not in the mean (their lifetime
  resting is not reconstructible from the live counters).
* **`behavioural_diversity` is a five-dimension summary**, not a complete
  behavioural repertoire. It measures the spread of per-tick rates of movement,
  biting, attacking, reproduction attempts and resting over the *living*
  population. Two animals can have the same summary and still differ in, say,
  which food they chose or when they ran. It is reported separately from genetic
  diversity precisely because it is not the same thing.
* **`environmental_stress` is exposure, not harm.** It is the mean drought
  modifier at the living animals' cells; the cost it implies depends on each
  animal's basal metabolism, so two runs with the same column value need not
  have paid the same price.
* **Cross-scenario robustness describes the declared scenarios only.** The
  `--scenarios` table reports persistence and survival over experiments A–F/H as
  listed; it is not a sample of environments and must not be read as a
  generalisation to unseen worlds. The worst case is the minimum over those runs,
  so adding a harder scenario would lower it.
* **Controller and sensitivity results here are underpowered nulls** (n = 2,
  short horizons), not findings. The neural arm is the random shipped MLP.
* **The viewer's frame rate is unmeasured on a desktop GPU or under a large
  population**; only headless correctness and world-layer draw cost are
  measured.
* **Truncated damage credit** (max 8 attackers) remains a documented
  approximation.
* **The ancestry view depends on the recorder being attached.** In a run where
  logging started late or a record was evicted, links are shown as unknown —
  correctly, not filled in.
* **Lineage is a descent chain, never a species.** No metric here is called
  species diversity.
* **`bottleneck` is a contraction indicator, not `Ne`.** It flags a joint fall
  from the window's own peak in population and genetic spread. It is not an
  effective population size, it says nothing about *which* lineages survived,
  and a long slow decline that never had a peak inside the retained window is
  not flagged. The two component drops are always reported so a one-sided
  decline is visible.

---

## 12. Deferred improvements — **DEFERRED**

### IMPROVE §5 short-term items — status

| §5 item | Status |
|---|---|
| Better genetic diversity metrics | **DONE** — `trait_variance`, `nn_variance`, `mean_pairwise`, `diet_entropy`, `unique_genomes`, kept as separate proxies |
| Lineage analysis | **DONE** — bounded `LineageRecorder` + ancestry view; descent chain, never species |
| Mutation-rate analysis | **DONE** — `mutation_rate` founder-locus sweep |
| Reproductive success tracking | **DONE** — per-organism `repro_attempts`/`repro_success` and population `run_totals["repro_*"]` |
| Evolutionary bottleneck detection | **DONE as an indicator** — `population_drop`/`diversity_drop`/`bottleneck`, explicitly not `Ne` |

Medium- and long-term §5 items remain deferred below.

Each is deferred with a reason, not forgotten:

| Item | Why deferred |
|---|---|
| **NEAT / topology evolution, innovation numbering, structural crossover, speciation (M8)** | Requires a *valid comparison experiment* against the fixed 16→12→8→8 MLP before it may replace it. That experiment does not exist yet, and the brief forbids replacing the MLP first. |
| **Recurrent controllers** | Same reason: no comparison experiment, and recurrence changes the determinism contract (hidden state must be snapshotted). |
| **Controller-complexity analysis** | Needs a complexity measure agreed before the comparison is meaningful; premature without the NEAT experiment. |
| **Behavioural-specialisation analysis** | The metric it needs now exists (`behavioural_diversity`, deliberately distinct from genetic diversity), but *classifying* animals into specialisations would need a declared clustering rule and a test that the clusters are stable; not done, and calling them species is forbidden. |
| **Design-scale runs of every experiment** | ~7.5 h on this machine (state in `BENCHMARKS.md`); scaled runs are what the suite executes. |
| **60 FPS at the largest scene** | The measured terrain fill (~15.5 ms) is the floor; the only faster option measured was not pixel-equivalent, so it was rejected rather than quietly accepted. |

---

## 13. Updated documentation — **DONE**

| Document | Change |
|---|---|
| [`AUDIT.md`](AUDIT.md) | Phase 1 audit (unchanged since writing it; it is a before-picture) |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Optimization pass, before/after throughput, profile, memory, viewer/analytics cost |
| [`ODD.md`](ODD.md) | Sensitivity and controller paragraphs in §2.6/§2.7 and §5; trait sweeps and the bottleneck indicator |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Statistics table is now 58 columns (40 + 13 + 2 + 3 appended), never reordered; trait sweeps and the bottleneck indicator documented |
| [`README.md`](README.md) | Ancestry section and key, updated status, link to this report |
| **`UPGRADE.md`** | This report |

Documents that already described the pre-upgrade model—`DESIGN.md`,
`ARCHITECTURE.md`, `ECOSYSTEM.md`, `ENERGY_MODEL.md`, `EVOLUTION.md`,
`GENETICS.md`, `NEURAL_NETWORKS.md`, `RESULTS.md`, `SCIENTIFIC_LIMITATIONS.md`—were
left alone where the model did not change. Every engine optimization preserved
semantics, so those descriptions remain accurate; this report is the record of
what was added around them.

---

## Summary table

| Deliverable | Status |
|---|---|
| 1. Repository audit | **DONE** (`AUDIT.md`) |
| 2. Architecture changes | **DONE** (six additive seams, invariants tested) |
| 3. Engine optimization report | **DONE** (5 changes, 1 rejected with measurement) |
| 4. Before/after benchmarks | **DONE** (1.8×, targets still missed) |
| 5. Peak/steady-state memory | **DONE** (< 650 MB, no growth with run length) |
| 6. New scientific metrics | **DONE** (18 columns in three additive blocks, observed vs derived labelled) |
| 7. New experiment descriptions | **DONE** (harnesses) / **PARTIAL** (design-scale runs deferred) |
| 8. Controller comparison results | **DONE** (measured; **underpowered null**) |
| 9. Visualization changes | **DONE** (overlays, inspector, analytics, ancestry) |
| 10. Test results | **DONE** (389 passed; 2 real bugs found by tests) |
| 11. Known limitations | **DONE** (listed, not hidden) |
| 12. Deferred improvements | **DEFERRED** (each with a reason) |
| 13. Updated documentation | **DONE** |
