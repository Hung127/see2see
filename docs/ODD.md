# EVOLVE — ODD protocol description

A description of the implemented model following the ODD protocol (Overview,
Design concepts, Details) of Grimm et al. (2006, 2010, 2020). This document
describes **what the code does**, not what it aspires to do; where a rule is
approximate or a mechanism is missing, that is said here rather than in a
footnote.

Every statement below is traceable to a module: the pipeline is
`src/evolve/engine.py`, the environment `fields.py` and `events.py`, organisms
`organisms.py`, plants `plants.py`, carrion `carcasses.py`, the decision path
`sensors.py` → `brain.py` → `actions.py`, the accounting `energy.py`, and the
observations `stats.py`.

Model version: `SPEC_VERSION = "1"` (`config.py`). A run is identified by
`(config, seed, spec_version, git_commit)`.

---

## 1. Overview

### 1.1 Purpose

EVOLVE is a deterministic individual-based model of a closed-mass ecosystem used
to ask: **what foraging, fleeing and reproductive strategies evolve when a small
fixed-topology neural network is the only thing deciding an organism's
actions, and matter and energy are conserved?**

It is built for:

* controlled artificial-life experiments on one CPU core with no GPU and no
  external services;
* reproducible comparison of alternative controllers against the *same*
  observation and action contract;
* teaching and inspection: every decision and every mass transfer is
  observable.

It is **not** a model of a real species assemblage. There are no species, no
taxonomy, no genetics in the molecular sense (a "genome" here is a vector of
12 trait values plus 380 network parameters) and no claim of biological realism
beyond the conservation accounting. Lineages are lineages, not species; see
`docs/SCIENTIFIC_LIMITATIONS.md`.

### 1.2 Entities and state variables

| Entity | State | Where |
|---|---|---|
| **Organism** | id, position, heading, reserve `E`, body `B`, health `H`, age, generation, alive, parent ids, genome (12 trait loci + 380 network parameters), lifetime counters (distance moved, bites, hunts, escapes, reproduction attempts/successes, rest ticks, plant/meat intake, damage taken), death cause | `organisms.py` |
| **Plant** | id, position, biomass, age, alive | `plants.py` |
| **Carcass** | id, position, biomass, age, alive | `carcasses.py` |
| **Environment** | 40×40 grid of terrain height, soil nutrient, temperature, moisture, pheromone; derived plant-biomass and organism-count fields; per-tick growth modifier and water-stress field | `fields.py` |
| **Detritus** | one scalar pool of dead organic matter with no position | `engine.detritus` |
| **Events** | active/scheduled environmental events (kind, start, duration, magnitude, region) | `events.py` |
| **Ledger** | cumulative mass/energy channel counters (`light_in`, `heat_out`, `mass_to_soil`, `mass_from_soil`, `mass_in`) | `energy.py` |
| **Global counters** | tick, births, deaths, mutations; per-cause death counts; age-at-death histogram | `engine.py` |

### 1.3 Scales

* **Space.** A torus of `1000 × 1000` world units (default). Organism bodies
  are 0.3–5 units, so the world is roughly 200–3,000 body lengths across. Fields
  are a coarse 40×40 summary (25-unit cells), not a physical lattice.
* **Time.** One tick is one decision + one physical update. No wall-clock
  mapping is asserted. The default horizon is 10,000 ticks; a season is 2,000
  ticks; maturity is 50 ticks; lifespan genes span 200–5,000 ticks, so an
  organism lives for a fraction of one season to a few seasons.
* **Population.** Default 300 founders, cap 2,000 organisms and 20,000 plants.
  Calibration used 6,000 initial plants (20 plants per organism), which is the
  density at which foragers break even (`docs/CALIBRATION.md`).

### 1.4 Process overview and scheduling

One tick, in fixed order (`Engine.step`):

1. `t += 1`.
2. Environment: season and rain update the fields; events apply their physical
   modifiers; pheromone diffuses and decays; soil percolates.
3. Plants grow, germinate and senesce.
4. `compact()` drops dead organism rows; `sort_by_id()` imposes ascending-ID row
   order (the tie-break contract).
5. Sense: one batched 16-channel sensor vector per living organism.
6. Think: one batched forward pass of each organism's own 16→12→8→8 network
   (or, only when explicitly opted in, a non-neural controller).
7. Decode: fixed thresholds turn the 8 outputs into throttle, turn, eat,
   attack, sprint, reproduce and rest attempts.
8. Move: positions integrate, the boundary mode applies, movement distance is
   accumulated.
9. Feed: one bite per eater at the nearest plant or carcass within reach, with
   proportional sharing of contested resources.
10. Predation (unless disabled for an assay): simultaneous strikes, damage
    credited to attackers, deaths spawn carcasses.
11. Metabolise: pay the metabolic bill, burn body if reserves run out, apply
    water stress, regenerate health, age by one tick.
12. Reproduce (unless disabled for an assay): in ascending parent-ID order.
13. Mortality: health, starvation and age-related deaths; non-predation deaths
    become diffuse detritus.
14. Decompose: detritus and carcasses mineralise into soil, releasing equal
    heat.
15. Refresh derived density fields.

All decisions use the state *before* movement; no organism observes another's
post-move position, so there is no first-mover advantage. Each stage is
vectorised over the live population with NumPy; the Python loops are over
organisms only where order matters (reproduction, kill credit).

---

## 2. Design concepts

### 2.1 Basic principles

* **One decision point.** Behaviour is the network output passed through
  `actions.decode`. The decode step encodes no preference about *what* to eat,
  *whom* to attack or *when* to flee; it only turns continuous scores into
  physical attempts with fixed thresholds (`o > 0.5`; `rest` blocks eating and
  attacking; `throttle = |o0|·o2·(1−o7)`).
* **Conserved matter and energy.** Organic mass moves between organisms, plants,
  carcasses and detritus and is exchanged with the inorganic soil pool; energy
  enters only as light and leaves only as heat. Both are checked to
  `rtol = 1e-9` when `Engine.debug` is on.
* **Determinism.** One PCG64 stream, fixed draw order, stable IDs, ascending-ID
  tie-breaks, and the RNG state is part of the state checksum and of every
  snapshot.

### 2.2 Emergence

Population cycles, diet specialisation, body-size and speed trends, and
predator–prey relationships are **not coded**. They arise from selection on
heritable traits interacting with the energy budget. The model does not label
anything a "species", and trait drift in one seed is not evidence of adaptation
(see `docs/RESULTS.md`, which reports null results and extinctions).

### 2.3 Adaptation and fitness

There is no explicit fitness function and no reward shaping. Differential
survival and reproduction *are* the fitness mechanism: an organism reproduces
only if it accumulates enough reserve and its network emits `o6 > 0.5`, and it
dies if it starves, is killed, or ages out. Nothing in the engine ranks
genomes.

### 2.4 Interaction

* **Organism–plant/carcass:** contact-based feeding. Reach is
  `1.5·M^(1/3) + 2` world units; the bite is limited by bite rate
  (`bite·M^0.75`), available biomass and remaining reserve capacity. A pure
  herbivore digests meat at zero efficiency and vice versa.
* **Organism–organism:** vision (range/angle genes) supplies threat, prey and
  mate channels; attacks land only within strike reach
  (`0.5·M^(1/3) + 2`) and deal `max(0, D − 0.8·R)` health damage. Fleeing only
  changes speed and cost; it cannot steer.
* **Organism–environment:** temperature affects metabolic cost quadratically
  around 20 °C; drought adds a basal-cost multiplier; floods damage animals in
  low ground; wildfire converts plant biomass to soil; nutrient pulses add soil
  mass (`ledger.mass_in`).

### 2.5 Stochasticity

Randomness comes from:

* founder genomes, positions and headings;
* mutation (per-locus Bernoulli mask, Gaussian deltas, rare large-effect
  mutations);
* germination cell choice and seedling jitter;
* rain cells, random-event spawning (Poisson per kind) and event placement;
* the age-related mortality draw past the lifespan gene.

All draws go through the single engine generator in a fixed, documented order.
`np.random.Generator(PCG64(seed))` gives the same stream on every platform that
NumPy supports, which is what makes cross-platform reproduction possible.

### 2.6 Observation

Statistics are computed from real entities at a fixed tick interval
(`stats.interval`, default 50), not per frame. The live viewer samples on
simulated ticks (`viz/analytics.py`, every 20 ticks) and never writes to the
engine. Derived metrics are labelled as derived: per-capita rates come from
counter differences, and the median survival is a histogram bin midpoint.

### 2.7 Prediction

There is no prediction or learning in the model. Networks are fixed at birth
and change only by inheritance and mutation. Baselines (`random`, `passive`,
`rule`, `forage`) exist for *evaluation* and are never enabled by default.

---

## 3. Details

### 3.1 Initialization

* **Founders:** `organisms.initial` (300) genomes drawn as
  `traits ~ Beta(2, 2)`, network weights `~ N(0, 0.35)`, biases `~ N(0, 0.05)`.
  Positions and headings uniform on the world; reserve `0.5·E_max` where
  `E_max = 10·M^0.9`. Founders are generation 0 and have no recorded parents.
* **Plants:** `plants.initial` (6,000) seedlings of 1 bu scattered uniformly.
* **Fields:** terrain is a 10-pass periodic box blur of uniform noise,
  normalised to [0, 1]; soil starts richer in low ground
  (`1 + 5·(1 − terrain)`); temperature and moisture start at their seasonal
  means; pheromone starts at zero.
* **Ledger:** the initial organic mass, energy and soil stock are recorded so
  the invariants can be checked against them for the whole run.

### 3.2 Inputs

The model has no external input files at run time. The only external forcing is
what the configuration declares: the seasonal sine (`t_mean ± t_amp`,
`m_mean ± m_amp` with a quarter-cycle phase offset), rain (`rain_rate`), and any
scheduled or random events. There is no weather reanalysis, no observed data
and no user interaction that changes dynamics — the viewer only reads.

### 3.3 Submodels

#### 3.3.1 Genome and phenotype

12 trait loci in [0, 1] are mapped to physical values by `GENE_SPECS`
(`genome.py`): body size (0.3–5, log), max speed (0–40), vision range (40–160),
vision angle (30–360°), metabolic rate (0.6–1.6), attack (0–5), defence (0–5),
reproduction threshold (0.40–0.95 of `E_max`), reproduction investment
(0.15–0.60 of `E_max`), diet preference (0–1), lifespan (200–5,000, log), and
mutation rate (5e-4–5e-2, log). The remaining 380 values parameterise the
network; they are clipped to [−8, 8].

Mutation draws a per-locus mask at the organism's own rate, Gaussian deltas
(`sigma_trait` for traits, `sigma_w·(1+|w|/4)` for weights), and a rare
`large_effect` multiplier. Crossover is uniform per locus. Trait loci clip to
[0, 1].

#### 3.3.2 Sensors (16 channels, all roughly [−1, 1])

| # | Channel | Meaning |
|---|---|---|
| 0, 1 | food closeness, bearing | nearest plant or carcass in the vision cone |
| 2, 3 | threat closeness, bearing | nearest visible organism that is >15 % larger **and** has attack > 1 |
| 4, 5 | prey closeness, bearing | nearest visible organism <85 % of own body |
| 6 | mate closeness | nearest visible mature organism within 0.7–1.43× own body |
| 7 | energy | reserve / `E_max` |
| 8 | health | health / `health_max` |
| 9 | age | age / lifespan gene |
| 10 | temperature | `(T − 20)/15` |
| 11 | moisture | `2·m − 1` |
| 12 | plant biomass in own cell | `2·B/10 − 1` |
| 13 | organisms in own cell | `2·n/20 − 1` |
| 14 | threat pheromone | deposited by attackers each strike, diffused and decayed |
| 15 | reproductive drive | `(E − threshold)/(E_max − threshold)` |

`−1` means "nothing detected" for the four detection channels — not "at
distance zero". Threat and prey are *perceptual* categories derived from
observable body size and attack, never species labels. Sensors only report;
they never imply an action.

#### 3.3.3 Controller

Default: the evolved network. Each organism's 380 parameters are shaped into
`16→12→8→8`; hidden layers use `tanh`, outputs 0–1 use `tanh`, outputs 2–7 use
a numerically stable sigmoid. The whole population is one batched `einsum`
pass per tick.

Optional (evaluation only, `Engine.controller is not None`): `random`
(uniform outputs, own PCG64 stream), `passive` (zeros), `rule` (a documented
reflex: flee threats, approach food, reproduce when fed), `forage` (hunger-
scaled approach). Every controller sees the same 16-vector and the same
phenotype and returns the same 8 outputs; baselines cannot bypass the decode
thresholds or the movement/feeding costs.

#### 3.3.4 Action decoding

`throttle = |o0|·o2·(1−o7)` in units of `speed_max`; `turn = o1·(3/M^(1/3))`
radians; `sprint = 1 + 1.5·o5`; `eat = o3 > 0.5` unless resting;
`attack = o4 > 0.5` unless resting; `reproduce = o6 > 0.5`; `rest = o7 > 0.5`,
which blocks eating and attacking and zeroes throttle.

#### 3.3.5 Movement and boundaries

Positions advance by `speed·(cos θ, sin θ)` with the sprint multiplier; the
boundary mode (`wrap` default, or `reflect`, or `hard`) maps them back into the
world. Distance moved is accumulated for the statistics.

#### 3.3.6 Energy and mass

Per tick, an organism of body `M`, speed `v`, vision `r`/`a`, temperature `T`:

```
basal   = k_b · M^0.75 · rate
repair  = k_r · M^0.75 · sqrt(lifespan/1000)
brain   = k_n · 380
move    = k_m · M · (v/10)^2
vision  = k_v · (r/100)^2 · (a/360)
attack  = k_a · attack_gene
defence = k_d · defence_gene
enzyme  = k_e · 4·d·(1−d)            (d = diet preference)
thermal = basal · 0.5 · ((T − 20)/10)^2
water   = water_stress_cost · basal · water_stress
cost    = sum of the above
```

Reserve pays `min(cost, E)`; any remainder burns body at up to `catabolism` per
tick and costs 1 health per tick. Respired and burned matter goes to the soil
at the organism's cell and releases equal heat. Health regenerates by
`health_regen` when reserves exceed twice the basal cost.

Feeding uses assimilation efficiency
`eta_base · sqrt(gate) · sqrt(metabolic_rate)`, where `gate = (1−d)` for plants
and `d` for meat. Undigested mass becomes detritus (it keeps both mass and
energy).

#### 3.3.7 Reproduction

Mature organisms with `E ≥ threshold·E_max` that emit `o6 > 0.5` attempt to
reproduce, processed in ascending ID. Asexual reproduction mutates a copy; in
sexual mode the nearest mature mate within `3·M^(1/3)` that can pay half the
investment is used, with uniform crossover. A birth needs
`investment > build_cost·child_body` and a free slot; if it succeeds the child
gets `0.9·(investment − body)` reserve and the remaining 10 % returns to soil
as waste (with equal heat). Failed gates are free — no parent pays.

Generation = max(parent generations) + 1. Parents are recorded by ID on the
child row (and in the optional lineage log) — never inferred later.

#### 3.3.8 Predation

Attackers (`o4 > 0.5`, not resting) strike the nearest organism within
`0.5·M^(1/3) + 2`. A strike costs `0.05·attack` reserve (paid to soil as heat)
and deposits pheromone. Damage is `max(0, k_dmg·attack·M^0.4 −
damage_mit_factor·k_mit·defence·M_v^0.4)`, applied simultaneously against
start-of-stage health. Damage credit is accumulated per victim (capped at eight
attackers); when the victim dies, its total body+reserve is split among living
attackers by damage share, limited by each attacker's reserve space and
digestive efficiency. The remainder becomes a carcass. If no attacker survives
to claim it, the whole body becomes carrion.

#### 3.3.9 Mortality and decomposition

A living organism dies if health ≤ 0 (cause `health`), if reserve and body are
both exhausted (`starvation`), or if past its lifespan gene, with per-tick
probability `1 − exp(−(age − lifespan)/200)` (`senescence`). Predation deaths
are recorded in the predation stage. Non-predation bodies become a scalar
detritus pool. Each tick, detritus loses `decay·detritus` and every carcass
loses `decay·biomass`; that mass enters the soil at the carcass's cell (or
evenly across the grid for the positionless detritus pool) and releases equal
heat.

Deaths are tallied by cause and binned into a 40-bin histogram of 25-tick age
bins. These are *derived statistics*: they are snapshotted but excluded from
the state checksum, because nothing downstream reads them.

#### 3.3.10 Plants

Each plant grows logistically toward a local capacity set by terrain
(`k0·(0.6 + 0.8·height)`), scaled by a temperature optimum at 22 °C, a moisture
factor `0.3 + 0.7·m` and any event modifier, and limited by the soil nutrient
available in its cell (shared proportionally if demand exceeds supply). Growth
draws the same mass from soil (`ledger.mass_from_soil`); senescence loses
`0.002·biomass` per tick to detritus. Germination adds up to
`germination_per_tick` seedlings, with cells chosen in proportion to existing
plant biomass, each costing `seed_cost` soil mass (`ledger.light_in`). Soil mass
therefore becomes organic mass through two channels: growth
(`ledger.mass_from_soil`) and the seed cost (`ledger.light_in`), and both are
booked so the full-system invariant closes.

#### 3.3.11 Environment and events

Temperature and moisture follow the seasonal sine (`season_period`, default
2,000 ticks, moisture offset a quarter cycle). Rain adds moisture to random
cells. Pheromone diffuses with a 5-point kernel and decays by 10 % per tick.
Soil percolates with a symmetric 5-point, exactly mass-conserving kernel
(`soil_diffusion`).

Events (`events.py`) are physical only — they never touch a genome:

* continuous: `drought` (moisture ×(1−m), plant growth ×(1−0.8m), water
  stress m), `heavy_rain` (moisture +m, flood damage in low ground),
  `heatwave` (+10m °C), `cold_snap` (−10m °C);
* instant: `wildfire` (plant biomass → soil + heat, animals lose health),
  `plant_blight` (fraction m of plant biomass → detritus), `nutrient_pulse`
  (m bu/cell added to soil, booked as `mass_in`).

Random events spawn per kind by a Poisson draw in configuration order; each
spawn draws centre x, centre y, radius, duration, magnitude in that order — part
of the reproducibility contract.

#### 3.3.12 Statistics and persistence

`StatsRecorder` samples a row every `stats.interval` ticks into a bounded ring
buffer (2,000 rows): counts, per-capita birth and death rates, generation
turnover, trait means and variances, genetic-diversity proxies, biomass pools,
environment means, per-cause deaths, median survival and population CV.
Diversity measures are proxies and are named as such (genetic variance is gene
value variance, not phenotypic spread; pairwise distance is capped at 200
genomes).

SQLite stores runs, tick statistics, events, snapshots, lineage and sampled
interactions. Snapshots contain every array and counter that can influence
future dynamics, including the RNG state, and are verified by re-executing the
run and comparing checksums (`cli reproduce`).

---

## 4. What is deliberately not in the model

These are limitations, not omissions to be papered over:

* No species concept, no speciation, no taxonomy. Lineage depth is a count of
  recorded generations, nothing more.
* No learning: networks do not train within a lifetime; only inheritance and
  mutation change them. Baselines are evaluation references, not part of the
  model.
* No explicit space for three-dimensional structure, no hydrodynamics, no
  disease, no age structure beyond the lifespan gene.
* Events are stylised physical perturbations, not calibrated climate scenarios.
* Deaths cause a *positionless* detritus pool, so the exact cell where a
  starved animal's mass returns is not represented (the mass returns evenly).
* Collisions are not resolved; several organisms may occupy one point.

## 5. Verification and validation

* **Verification** (does the code implement these rules?): the tests assert
  determinism and snapshot equivalence, dual ledger closure, feeding and
  predation accounting, reproduction gates, mutation bounds, sensor index
  mapping, controller contract, statistics correctness and headless rendering.
* **Validation** (are the rules appropriate for the question?): only partly.
  Calibration established a non-extinct plant–herbivore equilibrium at the
  default settings (`docs/CALIBRATION.md`), which shows the energy budget is
  tuned to the model's own units, not that it matches any real ecosystem.
  Multi-seed results and their failures are reported in `docs/RESULTS.md`.
* **Sensitivity:** declared one-factor sweeps (`cli sensitivity`, module
  `sensitivity.py`) report paired-seed outcomes per parameter value; small
  sample sizes are stated, and no sweep result is presented as a trend when
  the p-value does not support it. Six sweeps vary a config coefficient and
  three vary a founder trait locus (a genome value set before tick 1, not a
  config field). The smoke run in `docs/UPGRADE.md` is reported as
  underpowered, not as a finding.
* **Bottleneck indicator:** `stats.bottleneck` flags a sample when both the
  population and the genetic spread have fallen by at least half from their own
  peak in the retained window. It is a contraction indicator, not an effective
  population size (that needs a neutral-locus coalescent model this model does
  not satisfy) and not a species count.

## References

1. Grimm, V. et al. (2006). A standard protocol for describing individual-based
   and agent-based models. *Ecological Modelling* 198, 115–126.
   https://doi.org/10.1016/j.ecolmodel.2006.04.023
2. Grimm, V. et al. (2010). The ODD protocol: A review and first update.
   *Ecological Modelling* 221, 2760–2768.
   https://doi.org/10.1016/j.ecolmodel.2010.08.019
3. Grimm, V. et al. (2020). The ODD protocol: A second update to improve
   clarity, replication, and structural realism. *JASSS* 23(2), 7.
   https://doi.org/10.18564/jasss.4259
4. Grimm, V. et al. (2005). Pattern-oriented modeling of agent-based complex
   systems. *Science* 310, 987–991.
   https://doi.org/10.1126/science.1116681
5. Thiele, J. C., Kurth, W., Grimm, V. (2014). Facilitating parameter
   estimation and sensitivity analysis of agent-based models. *JASSS* 17(3), 11.
   https://doi.org/10.18564/jasss.2503
