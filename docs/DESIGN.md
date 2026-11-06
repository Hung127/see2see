# EVOLVE — Design

Date: 2026-11-06
Status: awaiting approval
Companion documents: `PROJECT_SPEC.md` (requirements), `research_notes.md` (M0 research)

---

## 1. Intent

Build an artificial-life simulator in which behaviour, ecological roles and (possibly)
specialisation arise from inherited variation, physical constraints and differential
reproduction — not from rules a programmer wrote. The scientific value is in what can be
*measured and reproduced*, not in what the screen looks like.

Success, concretely:
1. Every action an organism takes is a pure function of (sensors, genome-encoded network weights).
2. Every joule and every unit of biomass can be traced through a documented ledger.
3. The same seed + config + code version produces a bit-identical trajectory, and a snapshot can
   be resumed to continue that trajectory exactly.
4. 5,000 organisms run headless on one CPU core at a usable rate, and 500 run interactively.
5. The claimed evolution experiments either show an effect or are reported as showing none.

### Goals
- Deterministic, testable, modular engine with no dependency on Pygame.
- Genomes → phenotype → behaviour → energy → survival → reproduction, with no shortcuts.
- Complete scientific observability: population, genetics, behaviour, lineage, interactions.
- Six reproducible scenario experiments + one pre-registered "evolved vs random" assay.

### Non-goals (v1)
- Photorealism, animation quality, sound, game feel.
- Biological realism beyond the stated equations. No development, no learning, no disease,
  no diploid genetics, no species concept beyond measured clusters.
- Open-ended complexity, multicellularity, culture, tool use.

### Anti-goals (would make the build a failure even if it "works")
- Hiding an if-statement that steers an organism toward food behind a neural network.
- Rescuing a dying population by silently raising energy inputs.
- Presenting one lucky run as evidence of adaptation.
- Tuning the model until a chosen species appears, then claiming it emerged.
- Reporting benchmark numbers that were not measured on this machine.

---

## 2. Approaches considered

### 2.1 Engine core

| | Approach | Trade-off |
|---|---|---|
| A | **Object-per-organism** Python dataclasses, loops over individuals | Readable, trivially inspectable; ~1–5 ticks/s at 5,000 on one core; fails the perf target and makes 20k-tick experiments take hours |
| B | **Pure NumPy structure-of-arrays** | Fast and compact; inspection, persistence and line-by-line reasoning become awkward; every ad-hoc query has to be re-expressed as a vector op |
| C | **Hybrid: SoA authoritative + read-only views** | Slightly more code; hot loop stays vectorised, inspector/stats/save-load get a clean per-organism API |

**Chosen: C.** A is a dead end at the required scale; B alone fights the "scientifically
interpretable" requirement. Arrays are the source of truth; `OrganismView` is a thin, cached
read-only facade used by the UI, statistics detail views and snapshot export.

### 2.2 Genome representation
- *(a) Dict of named floats* — readable, slow, memory-heavy, awkward to batch.
- *(b) Fixed-length float vector + `GeneSpec` table* — chosen. One contiguous `(max_pop, n_genes)`
  array; each gene's name, map, range and cost lives in a spec table used by docs, the inspector
  and validation. Batched mutation = fancy indexing.
- *(c) Variable-length gene list (NEAT-ready)* — premature; deferred with forward-compatible
  gene IDs.

### 2.3 Controller
- *(a) Hand-written rules* — explicitly forbidden.
- *(b) Fixed-topology MLP, batched matmul* — chosen. 16→12→8→8, tanh hidden, per-organism weights
  as `(max_pop, 380)`; one `einsum` per layer for the whole population.
- *(c) Recurrent/FCM (Gras et al.)* — richer (memory, interpretable concept graph), ~10× slower and
  unnecessary for the acceptance criteria; deferred to M8.

### 2.4 Energy accounting
- *(a) Energy-only ledger* — simple, but silently creates and destroys mass when organisms are
  built and decomposed.
- *(b) Dual ledger: closed mass + energy with one external source and one sink* — chosen. Mass is
  exactly conserved (organisms, plants, carcasses, soil); energy has one documented input
  (light captured by photosynthesis) and one sink (heat from oxidation). Both are asserted in tests.

### 2.5 Decisions taken (mine, flagged so they can be overruled)
- **Asexual reproduction is the default.** Mate encounter probability in the default 2000×2000
  world with 300 founders is too low for sexual reproduction to be viable from generation 0.
  `reproduction.mode` ∈ {asexual, sexual, both}; sexual gets its own experiment.
- **Plant biomass is spatially clumped**, not uniform. With uniform plants the foraging gradient is
  nearly flat and selection on movement has nothing to grip. Patchiness is ecologically defensible
  and is generated from the same terrain field used by the environment.
- **Torus world by default**, with `wrap | reflect | hard` configurable. Reflection creates edge
  accumulation that distorts density statistics.
- **Roles are measured, never assigned.** A "herbivore" is an organism whose lifetime plant intake
  exceeds twice its meat intake.
- **`escape` output scales sprint speed only.** It cannot steer. Otherwise "run away" is handed to
  the organism as a rule.
- **Random initial weights** (zero-mean, small σ) and mid-range traits: an unbiased founder
  population. Early mortality is expected and reported.

---

## 3. World, time and environment

### 3.1 Space and time
- Continuous 2-D torus, default 2000×2000 units.
- One tick = one discrete update. No wall-clock meaning is claimed.
- Organism motion: position `p ← p + v·Δt̂` with `Δt̂ = 1`, `v = speed_max · throttle · sprint`,
  heading integrated from the turn output.
- Simulation is a fixed-step function `Engine.step() -> None`; all timing/decoupling lives in the
  renderer. Headless runs call `step()` in a loop.

### 3.2 Environmental fields (coarse grid)
`n_fields = 40×40` cells of 50×50 units, periodic. Per cell:
`temperature (°C)`, `moisture [0,1]`, `soil_nutrient (bu)`, `terrain_height [0,1]`,
`plant_density (bu/cell, derived from plant entities)`, `organism_density (count/cell, derived)`,
`threat_pheromone [0,1]` (diffusing, decaying).

Soil nutrient percolates between neighbouring cells (symmetric 5-point mixing,
`soil_diffusion = 0.10` per tick, mass-conserving). Without transport, nutrients
respired in one cell strand away from plant roots and plants starve on a global
surplus; this was found by the calibration harness (docs/CALIBRATION.md §4).

Terrain is generated once from smoothed value noise (seeded) and then fixed. Height sets each cell's
initial soil nutrient, its plant carrying-capacity multiplier (§6.7) and flood damage; it is *not* a
per-tick force. Floods therefore hit low ground repeatedly, which is a persistent environmental
gradient rather than a decorative map.

Seasonal forcing:
`T(t) = T_mean + A_T · sin(2π t / P_season) + T_event(t)`,
`M(t) = clamp(M_mean + A_M · sin(2π (t − P_season/4)/P_season) + rain(t) + M_event(t), 0, 1)`
with defaults `T_mean = 20 °C`, `A_T = 6`, `M_mean = 0.55`, `A_M = 0.15`, `P_season = 2000` ticks.
Rainfall is a stochastic per-cell increment; drought/flood events are scheduled modifiers with
ramp-up and duration.

### 3.3 Events
Event = `{kind, start_tick, duration, magnitude, region}`; region = whole world or disc.
Kinds: `drought`, `heavy_rain`, `heatwave`, `cold_snap`, `wildfire`, `nutrient_pulse`,
`plant_blight`. Effects are physical only:
- `drought`: moisture ×(1−m), plant growth ×(1−0.8m), animal water stress → extra metabolic cost.
- `heatwave`/`cold_snap`: temperature offset; plants hit their thermal response curve; animals pay
  the thermal penalty (see §6.3).
- `wildfire`: plants in region → 0, biomass → `S_soil` with an equal energy release to `heat_out`;
  organisms in region take damage.
- `plant_blight`: moves a fraction of plant biomass in region to detritus (killed, not burned).
- `nutrient_pulse`: adds nutrient to `S_soil` in region — an external mass input, logged as
  `mass_in` so the mass invariant becomes `initial + mass_in`.

**Never** does an event touch a genome.

---

## 4. Genome and phenotype

### 4.1 Layout
A genome is `float64[n_genes]`, `n_genes = 12 traits + 380 network parameters = 392`.
Each trait locus is stored in `[0,1]` (a "genetic value"); phenotype mapping is monotone and
documented. Network parameters are stored raw (unbounded, soft-clipped at ±8 on expression).

| # | Gene | Map (g∈[0,1]) | Phenotype range | Cost channel |
|---|------|---------------|-----------------|--------------|
| 0 | `body_size` | `exp(ln 0.3 + g·ln(16.67))` | 0.3 – 5.0 bu | basal ∝ M^0.75; build cost ∝ M; easier to hit, harder to hide |
| 1 | `speed_max` | linear | 0 – 40 units/tick | movement ∝ M·v² |
| 2 | `vision_range` | linear | 40 – 160 units | vision ∝ r²·(angle/2π) |
| 3 | `vision_angle` | linear | 30° – 360° | vision ∝ r²·(angle/2π) |
| 4 | `metabolic_rate` | linear | 0.6 – 1.6 | basal ∝ rate; max intake & assimilation ∝ rate^0.5 (metabolic scope) |
| 5 | `attack` | linear | 0 – 5 | upkeep ∝ attack; strike cost ∝ attack |
| 6 | `defense` | linear | 0 – 5 | upkeep ∝ defense; reduces damage taken |
| 7 | `repro_threshold` | fraction of `E_max` | `0.40 + 0.55g` → 40–95 % of reserve capacity | higher threshold = later, larger births |
| 8 | `repro_investment` | fraction of `E_max` | `0.15 + 0.45g` → 15–60 % of reserve capacity | must exceed offspring build cost `1.0·M'` |
| 9 | `diet_preference` | linear | 0 (herbivore) – 1 (carnivore) | generalist enzyme cost ∝ 4d(1−d); assimilation concave (§6.2) |
| 10 | `lifespan` | `exp` | 200 – 5000 ticks | repair cost ∝ √(lifespan/1000) |
| 11 | `mutation_rate` | log | 5e-4 – 5e-2 per locus per birth | no direct cost; indirect mutation load |

Derived, not genes: `turn_rate_max = 3.0 / M^0.33` rad/tick (small bodies turn faster);
`E_max = 10·M^0.9` (reserve capacity); `maturity_age = 50` ticks; `bite_rate = 0.10·M^0.75` bu/tick.

Note on thresholds: reproduction threshold and investment are **fractions of `E_max`**, not absolute
amounts. Absolute thresholds drawn independently of body size would make ~77 % of the founder
population mathematically sterile (`threshold > E_max`), silently biasing founders toward large
bodies. Expressing both as state-relative fractions removes the artefact without favouring any size,
and matches the standard state-dependent threshold used in energy-budget models.

Note on vision: the range is capped at 160 units for a reason that is both physical and
computational — sensor work per organism grows as `r²`, so a 300-unit vision would make the
neighbour pass roughly an order of magnitude more expensive at 5,000 organisms. The cap is a
documented modelling constraint (§11), not a hidden tuning knob.

A note on "unlimited positive traits": nothing is capped by fiat. Every axis has a cost that grows
at least as fast as its benefit in the relevant regime (v² for speed, r² for vision, M^0.75 for
basal, linear upkeep for attack/defense). Whether an optimum exists is an empirical question the
simulation answers per environment — that is the point.

### 4.2 Founder population
Traits sampled `Beta(2,2)` over the unit interval (mild central bias, full support); network
weights `N(0, 0.35)`, biases `N(0, 0.05)`. No trait is pre-adapted to any diet or habitat.

### 4.3 Mutation
Per birth, for each locus with probability `mutation_rate`:
- trait locus: `g ← clip(g + N(0, σ_trait), 0, 1)`, `σ_trait = 0.08`;
- network parameter: `w ← clip(w + N(0, σ_w), −8, 8)`, `σ_w = 0.05·(1 + |w|/4)` (multiplicative
  drift, so large weights move proportionally);
- rare "large-effect" mutation (p = 0.002): `σ × 5`.
Mutation is symmetric and unbiased: it cannot know whether a change helps. `mutation_rate` itself
mutates, with a configured floor/ceiling, so mutation-rate evolution is observable.

### 4.4 Crossover (sexual mode)
Uniform crossover on trait loci; per-parameter crossover with probability 0.5 on the network block;
each child gets one parent's value per locus. No blending (keeps loci interpretable). In asexual
mode, crossover is not applied.

### 4.5 Inheritance and identity
Offspring receive genome + a fresh ID + `generation = max(parent.generation) + 1` +
`parent_ids`. Lifetime counters (intake, distance, etc.) start at zero. No learned state is
inherited because there is none — the controller is stateless.

---

## 5. Neural controller

### 5.1 Topology
Feed-forward, fully connected: `16 → 12 → 8 → 8`. Hidden activation `tanh`; output activation
`tanh` for signed channels (`forward`, `turn`) and `sigmoid` for gated channels. Parameters:
`16·12+12 + 12·8+8 + 8·8+8 = 380`.

### 5.2 Inputs (all normalised to roughly [−1,1]; `−1` = "nothing detected")
| # | Input | Definition |
|---|-------|-----------|
| 0 | food distance | `1 − d/r_vision` if a plant or carcass is inside the vision cone, else −1 |
| 1 | food bearing | signed relative angle / π |
| 2 | threat distance | nearest neighbour with `M_n > 1.15·M` and `attack_n > 1.0`, inside cone |
| 3 | threat bearing | signed relative angle / π |
| 4 | prey distance | nearest neighbour with `M_n < 0.85·M`, inside cone |
| 5 | prey bearing | signed relative angle / π |
| 6 | mate distance | nearest reproductively mature neighbour of similar size, inside cone |
| 7 | energy | `E / E_max` |
| 8 | health | `H / 100` |
| 9 | age | `min(age / lifespan, 1)` |
| 10 | temperature | `(T − 20)/15`, clipped |
| 11 | moisture | `2·M − 1` |
| 12 | plant density | `clip(2·cell_biomass/10 − 1, −1, 1)` (vegetation cover index) |
| 13 | organism density | `clip(2·count/20 − 1, −1, 1)` |
| 14 | threat signal | threat pheromone at the organism's cell |
| 15 | reproduction drive | `clip((E − repro_threshold)/(E_max − repro_threshold), −1, 1)` |

Sensors only report. "Threat"/"prey"/"mate" are *perceptual* categories derived from observable
phenotype (size, attack, maturity), not from species labels, and they never imply an action.

### 5.3 Outputs
| # | Output | Activation | Physical effect |
|---|--------|-----------|-----------------|
| 0 | forward | tanh | signed throttle contribution |
| 1 | turn | tanh | heading change `ω = o₁·ω_max` rad/tick |
| 2 | movement intensity | sigmoid | `throttle = |o₀| · o₂` |
| 3 | eat attempt | sigmoid | `o₃ > 0.5` → attempt to bite plants/carrion in reach |
| 4 | attack attempt | sigmoid | `o₄ > 0.5` → strike nearest organism in reach |
| 5 | escape | sigmoid | `sprint = 1 + 1.5·o₅` multiplies speed **and** movement cost; does not steer |
| 6 | reproduce attempt | sigmoid | `o₆ > 0.5` → attempt reproduction (subject to energy/age/cap) |
| 7 | rest | sigmoid | `o₇ > 0.5` → throttle × (1 − o₇); lowers movement cost, blocks eating/attacking |

Threshold comparisons are *action decoding* (turning a continuous output into a discrete physical
attempt), documented as such; they encode no preference over targets or situations.

### 5.4 Forbidden-rule audit
Things the code must never contain, and how we know:
- No function named/behaving like `flee_from`, `seek_food`, `hunt`, `avoid`.
- `sensors.py` has no dependency on `actions.py` or on genome traits other than perceptual ones.
- `test_behaviour_purity.py`: for a fixed sensor vector, action decoding is a pure function of the
  weight vector; permuting weights changes outputs; zero weights produce a stationary organism.
- A short `docs/NEURAL_NETWORKS.md` "audit" section lists every place a decision is made (exactly
  one: the thresholds in §5.3) so a reviewer can check by reading one page.

---

## 6. Energy and mass model

All quantities in biomass-energy units (`bu`). `1 bu` of living tissue carries `1 bu` of chemical
energy. Reserves `E` are fat (biomass + energy). Body `B = M` is structural biomass.

### 6.1 Two ledgers
There are two conserved quantities with different stock sets.

**Mass** (closed except for logged external inputs):
`Σ(M + E)_organisms + P_plants + D_detritus + S_soil = initial + mass_in − mass_out`
where `D_detritus` is organic waste (undigested food, carcasses, blight-killed biomass) and
`S_soil` is inorganic nutrient. `mass_in` is nonzero only for `nutrient_pulse` events, which are
logged per event. Whenever biomass is oxidised, its *mass* returns to `S_soil` and its *chemical
energy* leaves as heat. (Modelling fiction: mineralisation is instantaneous and complete. Stated in
`docs/ENERGY_MODEL.md`; it makes the invariant exactly testable.)

**Energy** (one input, one sink):
`Σ(M + E)_organisms + P_plants + D_detritus = initial + L_light − Q_heat`.
`S_soil` carries **no** chemical energy — it is inorganic nutrient. `L_light` is the only input
(photosynthesis stores light as chemical energy 1:1); `Q_heat` is the only sink. Every channel that
moves biomass to `S_soil` must release exactly that much chemical energy as heat, and every channel
that moves biomass to `D_detritus` must preserve both mass and energy. This is what makes the two
invariants simultaneously closable.

Every tick the engine increments four counters: `light_in`, `heat_out`, `mass_to_soil`,
`mass_from_soil`. Tests assert both invariants to `rtol = 1e-9`.

### 6.2 Metabolic costs (per tick, organism of body M, speed v, vision r/angle a)
```
C_basal   = k_b · M^0.75 · metabolic_rate                     k_b = 0.015
C_repair  = k_r · M^0.75 · sqrt(lifespan/1000)                k_r = 0.002
C_brain   = k_n · n_params                                    k_n = 2e-6
C_move    = k_m · M · (v/10)^2                                k_m = 0.0008
C_vision  = k_v · (r/100)^2 · (a/360)                         k_v = 0.002
C_attack  = k_a · attack                                      k_a = 0.004
C_defense = k_d · defense                                     k_d = 0.003
C_enzyme  = k_e · 4·d·(1−d)                                   k_e = 0.004
C_thermal = C_basal · 0.5·((T−20)/10)^2
C_total   = sum of the above
```
If `C_total > intake` the deficit is drawn from `E`; if `E = 0`, body mass is catabolised at
`0.5 bu/tick` (mass → soil, heat released) and health declines.

### 6.3 Feeding
Reach is physical contact, not vision: `reach_eat = 1.5·M^{1/3} + 2` units. An organism may bite a
resource it is touching even if it never sensed it — contact is itself a sense; vision only informs
steering. Target selection is the **nearest** resource within reach (ties by lowest resource ID);
this is the one non-network choice in the action layer, justified physically (you can only bite
what is in your mouth) and listed in the audit section of `docs/NEURAL_NETWORKS.md`.
- Plant bite: `b = min(bite_rate, plant.biomass, E_max − E)`, `ΔP = −b`,
  `E += η_p · b`, `η_p = 0.65·(1−d)^0.5 · metabolic_rate^0.5`; the undigested remainder
  `(1−η_p)·b` moves to **detritus** (mass and energy preserved — it is organic matter, not ash).
- Carrion bite: same with `η_m = 0.85·d^0.5 · metabolic_rate^0.5`, remainder → detritus.
- One bite per tick per organism. Intake is capped by `E_max`; the surplus stays in the resource.
- A pure herbivore (`d=0`) has `η_m = 0`: it cannot digest meat even if its network makes it bite.
  A pure carnivore likewise cannot digest plants. Omnivores pay `C_enzyme` and are mediocre at both.

### 6.4 Predation
- Strike cost `0.05·attack` bu per attempt; requires target within `0.5·M^{1/3} + 2` units.
- Damage `D = 3.0·attack·M^0.4`; mitigation `R = 1.5·defense·M_def^0.4`; health loss
  `max(0, D − 0.8·R)`.
- On death: attacker gains `η_m · (M_def + E_def)`, capped by `E_max`; the remainder becomes a
  carcass. Damage credit is **cumulative since the victim's first damage in its current life** and is
  kept in a per-victim ledger of at most 8 attackers (oldest entry dropped beyond that, logged as
  truncated). Kill credit and meat are split **proportionally to credited damage**, so the split is
  order-independent and includes an early hit that was never followed up.
- Attacking emits threat pheromone in the local cell (sensor-only consequence).
- Health regenerates at `0.05/tick` if `E > 2·C_basal`, up to 100.

### 6.5 Reproduction
Requires: `E ≥ repro_threshold`, `age ≥ maturity_age`, `reproduce` output > 0.5, a free population
slot, and `repro_investment > build_cost(M')` where `build_cost = 1.0·M'`.
Parent pays `I = repro_investment`; child body `M' = M` (plus small mutation of the size gene);
child reserves `0.9·(I − M')`; `0.1·(I − M')` → soil (mass) and heat (energy).
Sexual mode: both parents pay `I/2`, child genome from crossover; a compatible mate must be within
`3·M^{1/3}` units and also attempting reproduction.
If the population cap is reached the birth simply fails and the parent pays nothing — resource
limitation, not a fitness-based queue.

### 6.6 Death and decomposition
Death causes: `E = 0` and body catabolised to 0 (starvation), `health ≤ 0` (predation/damage),
`age > lifespan` (senescence probability `1 − exp(−(age−lifespan)/200)` per tick), thermal shock
(health), and disaster damage.
On death, `M + E` becomes one **carcass** entity in the detritus pool. Each tick a carcass loses
`δ = 0.015·C`; that mass moves to `S_soil` and exactly `δ` of chemical energy leaves as heat
(microbial mineralisation, 1:1). Scavengers eat a carcass directly (§6.3) and therefore compete with
decay for the same biomass. No fraction is "lost" unaccounted: `mass_to_soil` and `heat_out` both
increase by `δ`.

### 6.7 Soil and plant growth

**Plant data model.** Plants are discrete entities in their own SoA (`id, x, y, biomass, age`);
the per-cell `plant_density` field used by sensors is *derived* from them each tick (there is no
independent plant field). `K` is per plant but modulated by terrain: `K_local = K_0·(0.6 + 0.8·h)`
with `K_0 = 5 bu`, so low, wet cells support more standing biomass.

Plant biomass grows logistically with nutrient, thermal and moisture limitation:
`dP = r_p · P · (1 − P/K_local) · f_T(T) · f_M(moisture)`
with defaults `r_p = 0.06`, `f_T` Gaussian (optimum 22 °C, width 8 °C), `f_M = 0.3 + 0.7·moisture`.
Growth is paid for from the cell's inorganic nutrient pool and from light:
`S_cell −= dP`, `P += dP`, `light_in += dP`.

**Nutrient contention.** Total demand `ΣdP` in a cell may exceed `S_cell`. Each plant then receives
`dPᵢ · S_cell/ΣdP` (proportional, order-free), with the exact floating-point remainder assigned to
the lowest plant ID so mass closes to `rtol = 1e-9`.

**Fates.** Grazed to zero → the plant row is removed (its mass was already transferred to the eater
and detritus). Senescence: `0.002·P` per tick moves to detritus. Blight: the affected fraction moves
to detritus (biomass killed, not burned). Wildfire: all biomass moves to `S_soil` and an equal
amount of energy to `heat_out`, the plant row is removed.

**Germination.** Each tick, up to `g_max = 20` new plants are seeded: a candidate cell is drawn with
probability proportional to local plant biomass (so vegetation spreads from existing patches), a
position is drawn uniformly inside it, and the seed costs `0.2 bu` from that cell's `S_soil`
(`S_cell -= 0.2`, `P_new = 0.2`, `light_in += 0.2` — the seed's chemical energy is stored light,
exactly as in growth). Germination fails if `S_cell < 0.2` or `n_plants = max_plants`. This keeps
patches persistent and lets them spread, instead of leaving a static list of 1,500 plants.

**Terrain coupling.** `f_M` uses cell moisture, and terrain height sets both `K_local` and the cell's
base nutrient, so vegetation patches are anchored to the landscape for the whole run rather than
homogenising. This is what gives selection on movement a persistent gradient to exploit (§2.5).

---

## 7. Tick pipeline and determinism

```
step():
  1  t += 1
  2  environment.update()          fields, season, scheduled events, disasters
  3  plants.grow()                 nutrient-limited logistic growth, germination, decay
  4  index.rebuild()               organism + plant uniform grids (cell 100 units)
  5  sensors.collect()             per-organism neighbour aggregation (batched)
  6  brains.forward()              one batched pass, 16→12→8→8
  7  actions.decode()              outputs → proposed throttle/turn/eat/attack/repro/rest
  8  movement.resolve()            integrate positions, wrap, charge movement cost
  9  feeding.resolve()             plant/carrion bites, proportional allocation
 10  predation.resolve()           aggregated damage, proportional kill credit, deaths
 11  metabolism.apply()            basal/repair/vision/thermal costs, starvation
 12  reproduction.resolve()        births in deterministic parent-ID order
 13  mortality.resolve()           senescence, thermal, disaster deaths → carcasses
 14  decomposition.apply()         carcass → soil + heat
 15  pheromone.update()            diffusion + decay
 16  stats.record()                per-tick aggregates, counters, lineage append
 17  persist.maybe_flush()         batched SQLite writes, periodic snapshot
```

**Staging.** Steps 5–7 only read state and write proposals into scratch buffers. Steps 8–14 apply.
No organism observes the post-move position of another, so there is no first-mover advantage.

**Determinism rules.**
- Single `np.random.Generator(PCG64(seed))`; all draws happen in a fixed order (documented in
  `docs/ARCHITECTURE.md` §RNG). No `python random`, no `set` iteration, no dict-order dependence.
- Float64 everywhere; reductions via `np.bincount`/`np.add.at` on sorted, stable index arrays.
- Ties broken by ascending organism ID only, never by hash or memory order.
- Cap-reached births fail rather than being ranked by energy.
- The RNG state is part of every snapshot and every determinism hash.

**Conflict resolution.**
- Plant contested by several eaters: total demand `Σbᵢ`; if `Σbᵢ > P`, each gets `bᵢ·P/Σbᵢ`
  (proportional, order-free). Exact remainder is assigned to the lowest-ID eater so mass closes.
- Prey killed by several attackers: damage accumulates, kill credit and meat split by damage share.
- Two organisms occupying one point: allowed (no collision resolution) — collisions would require a
  physics solver and add no ecological value; documented limitation.
- Reproduction into the same free slot: parents processed in ascending ID; later births see the
  updated count. Deterministic, no advantage because the cap is the only queue.

---

## 8. Statistics and evolution observability

Per tick (vectorised, stored in a ring buffer) and per generation-window:
- **Population**: total, births, deaths, by measured role, mean/median age, mean energy, mean health,
  growth rate `(N_t − N_{t−k})/k`.
- **Genetics**: per-trait mean/std/quantiles of phenotype (not raw gene), genetic diversity as
  (i) mean pairwise Euclidean distance over up to 200 deterministically strided individuals,
  (ii) per-locus variance of genetic values, (iii) Shannon entropy of discretised `diet_preference`
  (20 bins), (iv) count of unique genome hashes. All are labelled proxies; none is "species".
- **Mutation**: mutations per birth, per-locus mutation frequency, mean `mutation_rate` gene.
- **Behaviour**: mean distance moved, bites, successful hunts, escapes (attacks survived),
  reproduction attempts vs successes, rest fraction, intake per unit moved (foraging efficiency).
- **Interactions**: `(attacker, victim, tick, damage)` and `(eater, resource, tick, amount)` edges →
  food web degree distribution, diet overlap, realised trophic level.
- **History**: births/deaths/genomes of a lineage table, extinction events (last individual of a
  genome-hash cluster), environment event log.

Outputs: `stats.csv` (wide, one row per sample interval), `lineage.csv`, `interactions.csv`,
`trait_distributions/*.npz`, plus matplotlib PNGs: population, roles, traits vs generation,
diversity, mutation rate, birth/death rates, predator–prey phase plot. Live charts in the UI are
drawn with Pygame primitives (fast); matplotlib is for offline reporting only.

---

## 9. Persistence and reproducibility

SQLite (WAL) tables:
```
experiments(id, name, created_at, seed, config_yaml, spec_version, git_commit, notes)
runs(id, experiment_id, started_at, ended_at, start_tick, end_tick, status)
tick_stats(run_id, tick, <35 columns>)            -- batched, every stats_interval ticks
snapshots(id, run_id, tick, path, sha256, format_version, rng_state BLOB, manifest_json)
events(id, run_id, tick, kind, payload_json)
lineage(run_id, org_id, parent_a, parent_b, birth_tick, death_tick, generation, cause)
interactions(run_id, tick, kind, actor_id, target_id, amount)
```
Snapshots are `.npz` (compressed) holding every SoA array + plant/carcass/field arrays + RNG state +
counters + config hash. `format_version` is checked on load; unknown majors are refused with a clear
error. A run is reproduced by `(config_yaml, seed, spec_version, git_commit)`; `evolve reproduce
--run-id` re-executes and compares per-tick checksums.

Never write the full population per tick. Statistics flush every `stats_interval` (default 50),
snapshots every `snapshot_interval` (default 5000).

Interaction edges follow `logging.interactions` ∈ {off, sampled, full}. Default `sampled` writes
every kill, every mating, and a deterministic 1-in-20 sample of feeding events, while per-tick
*aggregates* (diet totals, role counts, interaction counts by trait class) are always exact. `full`
exists for short diagnostic runs. Budget at 5,000 organisms × 30,000 ticks: sampled logging stays
under ~3 M rows, full logging would reach tens of millions and is refused unless the run has fewer
than 500 organisms and 2,000 ticks.

---

## 10. Visualization (Pygame, headless-testable)

- Camera with pan/zoom; world drawn as filled cells for the environment (temperature/moisture/
  nutrient layer, toggleable) + circles for organisms sized by `M`.
- Colour modes: measured role; diet preference; lineage hash; age; energy. Vision-cone overlay and
  interaction flashes (attack/eat) toggleable.
- Controls: start/pause (`space`), single step (`s`), reset (`r`), speed 1×/2×/5×/20× (`1-4`),
  seed entry, save (`F5`), load (`F9`), toggle overlays (`v`, `i`, `h`), quit (`esc`).
- Inspector on click: id, age, energy, health, generation, parents, all 12 phenotypes, the 380
  weights (grouped by layer, scrollable), last-tick sensor vector, last-tick outputs, lifetime
  counters.
- Stats panel: sparklines for population, roles, mean size/speed/vision, diversity, births/deaths,
  plant biomass.
- Loop: `events → fixed-step sim (k steps per frame per speed) → render`; simulation never blocks
  on rendering. Verified headlessly via `SDL_VIDEODRIVER=dummy` with screenshot capture at fixed
  ticks (documented as a limitation: no human eye has watched it on this machine).

---

## 11. Performance strategy

1. Vectorise everything on the SoA; zero per-organism Python loops in the hot path.
2. Uniform grid, cell 80 units, rebuilt by counting sort (`np.argsort` on cell ids). A 5×5 stencil
   covers 160 units, exactly the maximum vision range (§4.1); the vision cap exists because sensor
   work grows as `r²` and a larger cap would need a wider stencil or a second index tier.
3. Neighbour aggregation by pair expansion over the 5×5 stencil, then segmented reductions
   (`np.minimum.at`, `np.bincount`, `np.add.at`) — one pass per sensor family.
4. Batched MLP via `einsum`; weights gathered once into a contiguous `(N, 380)` buffer.
5. Dead individuals are compacted by swap-with-last over a free list, with a stable remap so IDs
   stay monotonic and statistics stay ordered.
6. Profile with `cProfile` + `--profile` flag; publish the top-10 table in `BENCHMARKS.md`.

Measured targets on this machine (1 core, ~1 GB) — to be verified, not assumed:
| Population | Plants | Target |
|---|---|---|
| 500 | 1,500 | ≥ 25 ticks/s headless (interactive at 60 FPS render) |
| 1,000 | 2,500 | ≥ 12 ticks/s headless |
| 5,000 | 5,000 | ≥ 5 ticks/s headless |

If a target is missed, `BENCHMARKS.md` reports the measured number and the bottleneck, and the
target is not restated as met.

---

## 12. Testing strategy

Unit: gene maps round-trip; bounds; mutation distribution; crossover provenance; weight inheritance;
MLP shape/determinism; sensor normalisation and cone masking; action decoding; energy formulas;
grid query correctness vs brute force on random points; RNG stream ordering.

Property-based: mass conservation over 2,000 ticks with feeding/predation/decomposition forced on;
energy ledger closure; population never exceeds cap; no negative biomass; no organism outside the
world; every live organism has a genome row.

Determinism: same seed twice → identical state hashes at ticks 100/500/1000; snapshot at 400 +
resume for 600 ticks ≡ uninterrupted 1,000-tick run; different seed → different hash.

Evolution: a synthetic selection test (two genome variants, one given an energy advantage through a
trait, not through special-casing) shifts trait frequency; mutation alone without selection does not
systematically shift the mean beyond drift bounds; diversity metrics respond to a bottleneck.

Integration (bounded runtime, small world overrides): 2,000-tick runs for growth, decline, drought,
predation, extinction, and post-disaster recovery. Assertions are about *mechanisms* (e.g. "plant
biomass falls under drought"), never "the population survives".

A `balance` harness prints per-mille extinction times and equilibrium populations over a parameter
grid so calibration is reproducible rather than vibes.

---

## 13. Experiments

| ID | Name | Manipulation | Primary readouts |
|----|------|--------------|------------------|
| A | Stable ecosystem | default config, 30k ticks | population stability, role mix, trait drift |
| B | Predator pressure | +200 carnivore-biased founders, d=0.85 | prey population, prey size/speed/defense, hunt success |
| C | Severe drought | moisture −60 %, plant growth −70 %, 3k ticks | survival, biomass crash, recovery, trait shift |
| D | Resource competition | plants −60 %, patchier, 1000 ticks regrowth | diet breadth, foraging efficiency, coexistence |
| E | Environmental shift | at t=15k, T +8 °C and nutrients moved to one half | migration, trait shift, extinction risk |
| F | No predators | all `attack` genes clipped to 0 at founding, predation disabled | defense, speed, size, energy allocation drift |
| G | Evolved vs random (assay) | see protocol below | paired food intake over 500 ticks |
| H | Sexual reproduction | `mode = sexual`, 1,000 founders (mate encounters viable), asexual control at same density | birth rate, mate encounters, offspring viability, diversity vs control |

Each experiment is a YAML file plus a runner; `evolve experiments --all` writes a results directory
with CSV, plots and a markdown summary.

**Assay G protocol (pre-registered before the run).**
1. *Evolution phase:* run the A configuration (asexual, no predators, clumped plants) for 20,000
   ticks, 6 replicate seeds.
2. *Sampling:* from each replicate's **living** population at the final tick, draw 10 genomes
   uniformly at random (no ranking by energy or intake — ranking would confound the assay with the
   selection that already happened). From the same founder distribution (`Beta(2,2)` traits,
   `N(0, 0.35)` weights) draw 10 random genomes per replicate.
3. *Arena:* for each of 6 × 10 pairs, one arena with one evolved and one random individual placed at
   mirrored positions, identical plant layout, no reproduction, no predators, 500 ticks, no
   mutation. 60 paired arenas.
4. *Primary endpoint:* food intake (bu) over 500 ticks. Secondary: final energy, survival.
5. *Analysis:* paired Wilcoxon signed-rank and mean difference with bootstrap 95 % CI, per replicate
   and pooled. The result is reported whatever it is, including "no effect" and "random genomes
   did better"; the pre-registration text is stored in the results directory before execution.

---

## 14. Package layout

```
src/evolve/
  config.py        dataclass config tree, YAML load/validate, config hash
  rng.py           seeded streams, state capture/restore
  genome.py        GeneSpec table, founders, mutation, crossover, phenotype mapping
  brain.py         batched MLP, weight pack/unpack
  world.py         bounds, torus/reflect, terrain
  fields.py        coarse environmental grid, season, events, pheromone
  plants.py        plant SoA, growth, germination, blight, decay
  organisms.py     organism SoA, OrganismView, counters, death/birth bookkeeping
  spatial.py       uniform grid, pair expansion, segmented reductions
  sensors.py       16-input sensor assembly
  actions.py       output → physical proposals
  energy.py        cost/feeding/predation formulas, ledger counters
  engine.py        tick pipeline, staging buffers, checksums
  stats.py         aggregators, diversity metrics, CSV/plot export
  persistence.py   SQLite schema, batched writers, snapshot save/load
  experiments.py   six scenarios + assay runner
  cli.py           run / viz / benchmark / experiments / reproduce / report
  viz/             app.py (loop, controls), render.py, panels.py, camera.py
configs/           default.yaml, experiment_a..f.yaml, assay.yaml
tests/             unit, property, determinism, evolution, integration
docs/              ARCHITECTURE, GENETICS, NEURAL_NETWORKS, ECOSYSTEM, ENERGY_MODEL,
                   EVOLUTION, EXPERIMENTS, SCIENTIFIC_LIMITATIONS, BENCHMARKS
research_notes.md  README.md  requirements.txt  pyproject.toml
```

Engine imports never touch `viz`. `cli viz` is the only entry point that imports Pygame.

---

## 15. Milestones and gates

| M | Deliverable | Gate |
|---|-------------|------|
| M1 | World, plant/resource fields, organism SoA, movement, metabolism, feeding, death, minimal headless run + Pygame view | energy+mass tests pass; 500 organisms ≥ 25 ticks/s; a 5k-tick run completes without NaN |
| M2 | Genome, phenotype costs, mutation, crossover, asexual/sexual reproduction, parentage | genetics tests pass; trait means stay in range over 20k births; lineage table complete |
| M3 | 16→12→8→8 batched MLP, sensors, action decoding, inspector | purity test passes; identical inputs+weights → identical actions; weights inherited and mutate |
| M4 | Predation, carcasses, scavenging, decomposition, nutrient loop, emergent roles | predation biomass split test passes; ledger closes with all channels active |
| M5 | Stats, diversity, trait distributions, ancestry, CSV + plots, experiment comparison | determinism test on stats; plots render; CSV schema stable |
| M6 | Seasons, weather, six event types, scheduled events | drought/flood/wildfire change survival measurably; no gene is touched by an event |
| M7 | SQLite, snapshots, resume, headless CLI, benchmark, experiments A–F + assay | resume ≡ continuation; reproduce-by-seed; benchmarks documented; all tests green |
| M8 (deferred) | NEAT topology, speciation, richer sexual reproduction | out of scope for this build; design notes only |

After each milestone: run the suite, run the app headless, fix what breaks, update docs, and report
what works and what does not — including anything that failed.

---

## 16. Principal risks (and how they will be reported)

1. **Founder extinction.** Random controllers may starve before reproducing. Mitigation: calibration
   harness, generous but finite plants, and a documented viability report. If experiment A goes
   extinct under defaults, that is published, not tuned away silently.
2. **Weak selection on behaviour.** Few generations in feasible runtimes may mean no measurable
   adaptation. Mitigation: longer asexual runs, the pre-registered assay, and reporting null results.
3. **Parameter sensitivity.** Predator–prey models are notoriously sensitive (R1 documents this).
   Mitigation: seed replication, parameter grid in `BENCHMARKS.md`, and no claims from single runs.
4. **Single-core budget.** 5,000 organisms may fall short of 5 ticks/s. Mitigation: vectorised
   design, profiler-driven fixes; publish measured throughput either way.
5. **Pygame verification.** No display here. Mitigation: dummy-driver smoke test + screenshots;
   state the limitation in `SCIENTIFIC_LIMITATIONS.md`.
6. **Numerical drift.** Float summation order is fixed, but cross-platform libm differences can
   change last bits. Mitigation: hashes computed on rounded state (1e-9), documented.

---

## 17. Definition of done (maps to spec §17)

Each acceptance criterion becomes a named test or a named section in a results document:
genome→organism (`test_founders_are_genome_derived`), individual networks (`test_individual_weights`),
network controls action (`test_action_from_network_only`), weight inheritance
(`test_weights_inherited`), mutation changes traits and behaviour (`test_mutation_changes_action`),
birth/death (`test_reproduce_and_die`), ledger (`test_mass_and_energy_conservation`), predation and
competition (`test_predation_split`, `test_plant_competition`), environment affects survival
(`test_drought_increases_mortality`), statistics (`test_stats_schema`), save/resume
(`test_resume_equals_continuation`), headless (`test_headless_run`), seed reproduction
(`test_seed_reproducibility`), suite green (CI script), benchmarks (`BENCHMARKS.md`), evolved vs
random (`docs/RESULTS.md` §G, honest either way).

---

## 18. Calibration and compute policy

**Calibration window.** Global constants (the `k_*` coefficients, `r_p`, `K_0`, founder counts,
event magnitudes) may be adjusted **once**, during M1–M2, using the `balance` harness, and only to
satisfy mechanistic criteria: the ledgers close, a non-trivial fraction of founders reaches
maturity, and the plant–herbivore system has a non-extinct equilibrium in at least one
parameterisation. After the freeze, any constant change requires re-running every experiment, and
the change is recorded in the config history shipped with the results.

**Calibration outcome (recorded after M6).** The window above was not actually
closed during M1–M2: the balance harness had never been run, and the originally
shipped constants failed the founder-viability criterion. It was run late, in full,
and the result is `docs/CALIBRATION.md`. Adopted defaults: world 1000×1000, plants
6000 (max 20000), organisms 300 (max 2000), `bite = 0.50`, `eta_p = 0.75`, plus
mass-conserving soil percolation (DESIGN §3.2). The original constants are reported
there as a negative result; they are not silently deleted.

**What calibration may never do.** Change a parameter in response to a *specific experiment's*
outcome to make that experiment "work"; raise energy inputs to rescue a dying population; add a
behavioural rule; or touch genomes. A null or negative result after the freeze is a result.

**Compute budget.** This machine is one core with ~1 GB RAM. The suite is therefore run at
documented, scaled-down sizes: A 10k ticks, B 10k, C 6k, D 6k, E 8k, F 10k, H 6k, all at ≤ 1,000
organisms; G as specified (20k + 60 arenas). Estimated wall-clock at the measured throughput is
published in `BENCHMARKS.md` before the runs start, and one longer 30k-tick A run is attempted if
the budget allows. Scaling is stated in every results document; no claim is made that a scaled-down
run reproduces full-scale dynamics.
