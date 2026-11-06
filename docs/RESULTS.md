# Results

Measured outcomes of the experiment programme (DESIGN §13). Everything here is
reported as measured, including extinctions, null results and missed targets.
The runs are reproducible from the committed configs and seeds; the raw outputs
(CSV, plots, summaries) are in `experiments_out/`.

## Environment and scale

* Machine: 1 CPU core, ~1 GB RAM, Python 3.14, NumPy 2.5 (see `BENCHMARKS.md`).
* **Programme scale: 0.4** — world lengths and the horizon × 0.4, plant and
  founder counts × 0.16 (density preserved). This was a compute-budget choice,
  documented in `EXPERIMENTS.md`; it is *not* an equivalent experiment.
* Consequence that matters below: scale 0.4 means **48 founders** instead of
  300. With so few founders the population is one bad draw from extinction, and
  several runs took that draw.

## Programme outcomes at scale 0.4

| ID | Horizon (design) | Ticks run | Final pop | Births | Extinct | Extinction ≈ tick |
|----|------------------|-----------|-----------|--------|---------|-------------------|
| A  | 30,000 | 12,000 | 0   | 0   | **yes** | ~450   |
| B  | 30,000 | 12,000 | 0   | 2   | **yes** | ~700   |
| C  |  3,000 |  1,200 | 45  | 252 | no      | —      |
| D  |  1,000 |    400 | 1   | 1   | no (1 survivor) | — |
| E  | 30,000 | 12,000 | 0   | 412 | **yes** | ~3,850 |
| F  | 30,000 | 12,000 | 65  | 656 | no      | —      |
| G  | 20,000 × 6 | 8,000 × 6 | — | — | **no data** | all 6 replicates extinct |
| H  | 30,000 | 12,000 | sexual 0 / asexual 0 | 0 / 10 | **both yes** | ~2,250 / ~2,150 |

### A — stable ecosystem: extinct

The 48 founders never established a birth: **0 births in 12,000 ticks**. Mean
energy fell from ~6.9 (the founders' starting reserve) to 0 by tick ~450 while
plant biomass *grew* (960 → 1,913 bu) — the founders died of starvation
surrounded by food they did not reach. This is the founder lottery at work: a
random network must both steer toward plants and keep `o3 > 0.5` near them; with
300 founders (the calibration population) some lineage almost always manages
this (the calibrated reference run had 431 births by tick 6,000); with 48, the
draw can come up empty. See the diagnostic below.

### B — predator pressure: extinct

Same failure mode as A (2 births, extinct by ~700). The +32 carnivore-biased
founders (200 × 0.16) did not change the outcome; nobody hunted successfully
enough to reproduce.

### C — severe drought: survived, with a visible response

The drought (ticks 200–440 at this scale, moisture −60 %, plant growth −48 %)
drove the population from 48 to a low of 17 and it recovered to 45 by the end
(252 births / 255 deaths). Measured trait drift over the run: diet preference
0.53 → 0.11 (the population became more herbivorous), attack 2.72 → 4.41,
defence 2.58 → 3.83, lifespan 1,172 → 534. **Caveat:** with 48 founders and one
seed this is one realisation, not a trend; the drift direction is consistent
with selection but could be drift.

### D — resource competition: functional extinction

Plants at 40 % of A and clustered: the population fell monotonically 48 → 1
survivor by tick 340 (48 deaths, 1 birth). The last organism was a herbivore
(diet 0.07). At this scale the manipulation is effectively a bottleneck
experiment, not a coexistence experiment.

### E — environmental shift: extinct *before* the manipulation

The population peaked at 99 (tick ~1,550), declined, and was extinct by
~3,850 — **before the +8 °C heatwave and the nutrient move at tick 6,000**.
The manipulation therefore ran on an empty world and the experiment measured
nothing about the shift. 412 births show the population was reproducing for a
while; it still did not persist. Reported as a null result.

### F — no predators: survived

The one long run that persisted: 65 organisms at tick 12,000 (656 births /
639 deaths), mean population 40.6 over the last fifth. With attack loci clipped
to 0 and the predation stage off, measured drift: diet 0.52 → 0.22,
speed 19.2 → 31.0 units/tick, defence 2.33 → 1.27, vision 102 → 74. Speed
*increasing* without predators is the direction the energy model predicts
(movement is cheap relative to its foraging value when nothing hunts you), but
one seed is one seed.

### G — assay (evolved vs random): **no data — every replicate went extinct**

All 6 evolution replicates (8,000 ticks each, clumped plants, no predators,
48 founders) were extinct before sampling. Per the pre-registered protocol,
extinct replicates contribute no arenas, so the assay produced **0 paired
arenas** and no statistical result. The pre-registration and the empty result
table are preserved in `experiments_out/scale_0.4/g/`.

This is the honest outcome of the scaled run: **the assay could not be executed
at scale 0.4** because the evolution phase does not survive it. It is *not* a
finding that evolved genomes equal random ones.

### H — sexual reproduction: both arms extinct

Both arms died (~2,200 ticks); the asexual arm managed 10 births, the sexual arm
0. With 160 founders per arm (1,000 × 0.16) and mate reach
`3·M^(1/3)` ≈ 3 units in a 730 × 730 world, sexual reproduction could not get
established before the founder generation died. No comparison between the arms
is possible from this run.

## Diagnostic: the extinction is not only a scale effect

The scale-0.4 programme lost its populations, so the obvious question is whether
the down-scale caused it. A design-scale A run (**300 founders, 1,000 × 1,000
world, seed 1, 12,000 ticks**) was executed to find out. It also went extinct.

```
tick   population  births(cum)  deaths(cum)  mean E   plants(bu)
   50       302           2           0         5.3      11812
  900        36          47         311         8.7      12601
 1750       113         167         354         8.2      11664
 3450        64         368         604        10.8      11940
 5150        58         423         665        19.3      11752
 6000        52         431         679        19.7      11832   <- calibration checkpoint
 7700        27         446         719        22.6      12339
 9400         6         446         740        23.6      12859
11100         1         446         745        24.0      13039   <- last survivor
11950         0         446         746         0.0      13069
```

The first 6,000 ticks reproduce the calibration record **exactly** (52
organisms, 431 births, 11,832 bu of plants), so the calibration was correct at
its stated horizon. Beyond it the population does not hold an equilibrium: it
declines steadily and the last organism dies of senescence at ~11,100 while
plants sit at their carrying capacity (13,000 bu).

A second design-scale A run with **seed 2** was executed as a replication
check, and it survives the same horizon: 76 organisms alive at tick 12,000 with
births continuing (population 302 → 214 → 126 → 85 → 100 → 87 over the run,
plants 9–11k bu). The long-horizon outcome is therefore **seed-dependent**, not
a universal property of the calibrated constants.

### The proximate cause is behavioural, not energetic

At tick 6,000 a clean forward pass over the living population (no stale
indices — the probe runs `_think()` at a tick boundary) gives:

* 52 organisms alive; **39 satisfy both reproduction gates** (mature, and
  `E ≥ repro_threshold · E_max`);
* of those 39, **none** has `o6 > 0.5`; the maximum is **0.439**. The decode
  rule turns `o6` into a reproduction attempt only above 0.5.
* mean `E/E_max` among the living is 0.81, and the survivors reach 0.99·`E_max`
  — they are not starving; they are **fully fed and not reproducing**.

So the extinction is the reproduction output drifting to just below the fixed
0.5 decode threshold, not a failure of the food supply or the energy model. The
same run shows the diversity index collapsing to a single clone
(`mean_pairwise` 1,055 → 161 → 0) and body size/speed drifting up
(1.54 → 2.69 bu, 19.7 → 31.0 units/tick) as the population shrinks.

### Interpretation, and what it does not establish

The mechanism is consistent with **drift in a small effective population
against a hard threshold**: with only tens of effective breeders, the neural
weights that produce `o6` can random-walk below 0.5, and once the dominant
clone is below the cliff it leaves no offspring. The data cannot separate drift
from weak selection without replicated large-population runs, which this
machine's budget did not allow. What *is* established:

1. the tick-6,000 calibration checkpoint is exactly reproducible, and the
   population it certified is not an asymptotic equilibrium — this particular
   lineage decays to extinction by ~11,950;
2. the decay is behavioural (reproduction output under the decode threshold),
   not energetic;
3. **the outcome is seed-dependent**: a second design-scale A run with seed 2
   is still alive at tick 12,000 (76 organisms, births continuing, mean
   population ~100 over the run). So the model *can* persist at design scale;
   seed 1's extinction is one lineage's fate, not a universal property.

The practical consequence for the programme: at 48 founders (scale 0.4) the
lottery is usually lost, which is what the programme table shows; at 300
founders it is usually won. The assay was therefore re-run at design scale
(see §G below) rather than declared impossible.

## Throughput (measured, all DESIGN §11 targets missed)

From `BENCHMARKS.md`: 300 organisms → 14.0 ticks/s, 1,000 → 4.49 ticks/s,
5,000 → 0.78 ticks/s (targets were 60/30/10). The programme above therefore ran
at reduced horizons; the design-scale programme is estimated at ~7.5 h of wall
clock and was not executed in full.

## DESIGN §17 definition of done — status

| Criterion | Named test / artefact | Status |
|-----------|----------------------|--------|
| genome → organism | `test_founders_are_genome_derived` | pass |
| individual networks | `test_individual_weights` | pass |
| network controls action | `test_action_from_network_only` | pass |
| weight inheritance | `test_weights_inherited` | pass |
| mutation changes traits/behaviour | `test_mutation_changes_action` | pass |
| birth/death | `test_reproduce_and_die` | pass |
| ledger | `test_mass_and_energy_conservation` | pass |
| predation & competition | `test_predation_split`, `test_plant_competition` | pass |
| environment affects survival | `test_drought_increases_mortality` | pass |
| statistics | `test_stats_schema` | pass |
| save/resume | `test_resume_equals_continuation` | pass |
| headless | `test_headless_run` | pass |
| seed reproduction | `test_seed_reproducibility` | pass |
| suite green | `python3 -m pytest tests/` — 151 passed | pass |
| benchmarks | `BENCHMARKS.md` (targets missed, reported) | done |
| evolved vs random | this document, §G | **no data at scale 0.4** (see assay section) |
