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
| Sensitivity sweeps | `sensitivity.py` | six declared one-factor sweeps | opt-in CLI command |
| Derived statistics | `stats.py` | 13 appended columns; death-cause counters; age-at-death histogram; `LineageRecorder` | stats always on, lineage recorder opt-in |
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

Fifteen columns were **appended** to `stats.COLUMNS` (never reordered, so old
CSVs still parse by name), in two additive blocks — the second exists only
because the first was already committed, and the schema rule is "append, never
reorder":

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

`behavioural_diversity` is deliberately **not** genetic diversity: it never reads
a genome. Two clones can behave differently and two genomes can behave
identically, which is exactly why the brief lists it as a separate axis. The
five activity dimensions are divided by each animal's own age and then by fixed
*declared* reference rates (3 u/tick for movement, 1 event/tick for the rest) —
normalising by the sample's own spread would give every run the same score.

`environmental_stress` samples `fields.water_stress` at each animal's position,
the same lookup the metabolism stage uses to charge the drought cost. Reporting
the field mean instead would understate a population sheltering in dry cells.

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
identical movement, feeding and reproduction costs. Endpoints:
`final_population`, `births`, `deaths`, `mean_energy`,
`mean_organism_biomass`, `survival_median`, `foraging_efficiency`. Statistics:
paired differences against `neural` on the same seed, percentile bootstrap CI
and the in-house Wilcoxon signed-rank test already used by assay G.

### Sensitivity sweeps (`evolve.cli sensitivity`)

Six declared sweeps, each with a **pre-registered hypothesis**, a rationale, and
values bracketing the calibrated default:

| Sweep | Parameter | Values | Baseline |
|---|---|---|---|
| `bite` | `energy.bite` | 0.25 / 0.50 / 0.75 / 1.00 | 0.50 |
| `eta_p` | `energy.eta_p` | 0.50 / 0.75 / 1.00 | 0.75 |
| `soil_diffusion` | `environment.soil_diffusion` | 0.02 / 0.10 / 0.30 | 0.10 |
| `sigma_trait` | `genetics.sigma_trait` | 0.02 / 0.08 / 0.20 | 0.08 |
| `maturity_age` | `organisms.maturity_age` | 25 / 50 / 100 | 50 |
| `decay` | `energy.decay` | 0.005 / 0.015 / 0.045 | 0.015 |

Every run is paired by seed. The report gives per-value mean/SD, the paired
difference against the baseline value with a bootstrap CI and a Wilcoxon
p-value, and states that extinction is reported, not averaged away.

**Both are implemented and tested; only short smoke runs were executed here**
(see §8). Design-scale runs are left to the experiment programme, which already
documents its own scaled-vs-full distinction.

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

**Analytics (`a`).** 21 chart blocks, each declaring **one unit** — population;
births and deaths; biomass pools; reserve and body; speed; vision; genes;
genetic variance; pairwise distance; temperature; moisture; generation depth;
median age at death; deaths by cause; an age-at-death histogram; **live body-mass
and reserve distributions** (computed at draw time, not stored per sample); CV;
**drought stress where the animals are**; **behavioural diversity**; behaviour
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

`python3 -m pytest tests/ -o addopts="" -q` → **357 passed in 176.12 s**.
(Earlier stages of the upgrade: 343 passed in 266.55 s, then 346 in 259.34 s;
the runtime moves with the machine's load, not with correctness.) Suite grew
from 196 to 357 tests. No existing assertion was weakened or deleted; several
were strengthened. The notable ones:

* `tests/test_optimization_equivalence.py` pins the three pre-upgrade checksums.
* `tests/test_controllers.py` (24) covers the output contract, reproducibility,
  the default path being bit-identical to a run with no controller, and the
  controller using its **own** RNG so the engine stream is untouched.
* `tests/test_sensitivity.py` (11) covers config copy-not-mutate, type
  preservation, real config fields, and extinction being reported.
* `tests/test_metrics.py` (17) covers the appended columns and that the original
  block is a prefix of `COLUMNS`, plus the two second-block metrics: identical
  profiles score zero, differing activity moves the score, the metric ignores
  genomes, the profile is a per-tick rate (10-tick-old and 100-tick-old animals
  with the same rate agree), and stress is sampled at the animals' cells rather
  than world-wide.
* `tests/test_lineage.py` (15) covers the bounded recorder, real birth traits,
  no fabrication, and that attaching it does not change the checksum.
* `tests/test_viz_ancestry.py` (15), `test_viz_charts.py` (33),
  `test_viz_panels.py` (46), `test_viz_render_lod.py` (7) and
  `test_viz_smoke.py` (6) cover the viewer headlessly. The four new panel tests
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

---

## 12. Deferred improvements — **DEFERRED**

Each is deferred with a reason, not forgotten:

| Item | Why deferred |
|---|---|
| **NEAT / topology evolution, innovation numbering, structural crossover, speciation (M8)** | Requires a *valid comparison experiment* against the fixed 16→12→8→8 MLP before it may replace it. That experiment does not exist yet, and the brief forbids replacing the MLP first. |
| **Recurrent controllers** | Same reason: no comparison experiment, and recurrence changes the determinism contract (hidden state must be snapshotted). |
| **Controller-complexity analysis** | Needs a complexity measure agreed before the comparison is meaningful; premature without the NEAT experiment. |
| **Behavioural-specialisation analysis** | Would need a behavioural-diversity metric distinct from genetic diversity; conflating them is explicitly forbidden. |
| **Design-scale runs of every experiment** | ~7.5 h on this machine (state in `BENCHMARKS.md`); scaled runs are what the suite executes. |
| **60 FPS at the largest scene** | The measured terrain fill (~15.5 ms) is the floor; the only faster option measured was not pixel-equivalent, so it was rejected rather than quietly accepted. |

---

## 13. Updated documentation — **DONE**

| Document | Change |
|---|---|
| [`AUDIT.md`](AUDIT.md) | Phase 1 audit (unchanged since writing it; it is a before-picture) |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Optimization pass, before/after throughput, profile, memory, viewer/analytics cost |
| [`ODD.md`](ODD.md) | Sensitivity and controller paragraphs in §2.6/§2.7 and §5 |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Statistics table is now 53 columns (40 + 13 appended), never reordered |
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
| 6. New scientific metrics | **DONE** (13 columns, observed vs derived labelled) |
| 7. New experiment descriptions | **DONE** (harnesses) / **PARTIAL** (design-scale runs deferred) |
| 8. Controller comparison results | **DONE** (measured; **underpowered null**) |
| 9. Visualization changes | **DONE** (overlays, inspector, analytics, ancestry) |
| 10. Test results | **DONE** (346 passed; 2 real bugs found by tests) |
| 11. Known limitations | **DONE** (listed, not hidden) |
| 12. Deferred improvements | **DEFERRED** (each with a reason) |
| 13. Updated documentation | **DONE** |
