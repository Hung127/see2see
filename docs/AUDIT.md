# EVOLVE — implementation audit

Written before any change in this upgrade, from reading the code and measuring it.
Baseline commit: `a66cc0c`. Every number here is from `evolve.bench` on this
machine (1 vCPU, ~1 GB RAM, CPython 3.14, NumPy only).

This document is the Phase 1 deliverable of `docs/my-plans/IMPROVE.md`. It
records what exists, what is weak, and what will be changed. It is not a wish
list: items marked **DONE** later in `docs/UPGRADE.md` were implemented, items
marked **DEFERRED** were not, and each says why.

---

## 1. What the repository actually contains

| Layer | Module | Role |
|---|---|---|
| Config | `config.py` | Frozen dataclasses; YAML loader rejects unknown keys; `config_hash` |
| RNG | `rng.py` | One `Generator(PCG64)`; every draw goes through it |
| World | `world.py` | Torus geometry, periodic displacement, bearing |
| Fields | `fields.py` | 40×40 climate/soil/terrain/pheromone summary grid |
| Plants | `plants.py` | Discrete SoA pool, nutrient-limited logistic growth |
| Organisms | `organisms.py` | SoA pool, deterministic slot reuse, `compact()`, `sort_by_id()` |
| Spatial | `spatial.py` | Periodic uniform grid, vectorised pair/candidate expansion |
| Genome | `genome.py` | 392 floats: 12 trait loci + 380 network parameters |
| Brain | `brain.py` | Batched 16→12→8→8 MLP via `einsum` |
| Sensors | `sensors.py` | 16-channel observation vector |
| Actions | `actions.py` | The only non-network decision point (thresholds) |
| Energy | `energy.py` | Metabolic formulas + dual closure ledger |
| Engine | `engine.py` | 14-stage tick pipeline |
| Events | `events.py` | Scheduled/random physical events |
| Stats | `stats.py` | Per-interval row recorder, diversity proxies, CSV/plots |
| Persistence | `persistence.py` | SQLite runs, snapshots, exact resume |
| Experiments | `experiments.py` | Scenarios A–H, assay G, Wilcoxon, bootstrap |
| Viz | `viz/*` | Pygame viewer, camera, palette, panels, renderer |
| CLI | `cli.py` | `run`, `reproduce`, `report`, `viz`, `benchmark`, `experiments` |

Test suite: **196 tests**, ~87 s, all green before the upgrade.

## 2. Architectural strengths (keep these)

1. **One RNG, documented draw order.** Reproducibility is a tested contract, not
   an aspiration. `checksum()` covers every array that can influence the future.
2. **Dual conservation ledger.** Mass and energy closed to `rtol 1e-9` with a
   documented absolute floor; `assert_full_closed` covers organic + soil.
3. **No behavioural rules.** A single decision point (`actions.decode`) turns
   network outputs into physical attempts; the §17 audit test enforces it.
4. **Structure-of-arrays with deterministic slot reuse** and stable ID ordering.
5. **Focus/fade inspector**: shows the *exact* vectors from the last think step,
   matched by ID, never by position.
6. **Engine and viz are separable.** `test_viz_never_imports_export` and the
   headless CLI tests keep the simulation independent of pygame/matplotlib.
7. **Snapshot/resume equivalence** is tested against a directly-run engine.

## 3. Measured bottlenecks (baseline)

`cProfile`, 1000 organisms at calibrated density (20 plants each), 20 ticks:

| Function | cumtime (s) | share of tick |
|---|---|---|
| `Engine.step` | 4.542 | 100 % |
| `Engine._think` (sense+brain+decode) | 3.094 | **68 %** |
| `sensors.collect` | 3.060 | 67 % |
| `SpatialGrid.candidates` | 2.264 | 50 % |
| `world.distance` | 1.548 | 34 % |
| `_nearest_visible` | 1.543 | 34 % |
| `Engine._feed` | 1.239 | 27 % |
| `world.dx` + `world.dy` | 1.381 | 30 % |

Throughput (`evolve.bench`, 30 ticks, mean population held at the nominal size):

| Nominal population | ticks/s |
|---|---|
| 300 | 14.30 |
| 1000 | 4.53 |
| 5000 | 0.80 |

Memory, 1000 organisms: traced steady state **6.5 MB**, traced peak **57.4 MB**,
process peak RSS **122 MB**. Well inside the 650 MB target; memory is not the
constraint, CPU is.

### Diagnoses (not guesses)

* **`world.dx`/`dy` allocate three temporaries each** for `(d + w/2) % w − w/2`.
  Called twice per `distance()`, so one `distance()` allocates ~6 arrays of the
  pair count. The pair count is quadratic in local density.
* **`candidates()` builds the same ragged expansion 12 times per call**, and is
  called three times per tick — once for food sensing (all organisms), once in
  `_feed` (all organisms, then filtered to the eaters), once in `_predation`
  (all organisms, then filtered to the attackers). The last two compute
  candidates for organisms that will never use them.
* **`_nearest_visible` computes `arctan2` for every candidate in the stencil**
  (a 5×5 cell stencil, ~144 neighbours) and *then* discards those outside the
  vision radius/cone. The distance filter is already available and is cheaper.
* **`sensors.collect` rebuilds a full 40×40 plant-biomass field every tick**
  (`plants.biomass_field`) via `np.add.at`, only to sample the ~1000 cells the
  organisms stand in. It also computes a `bincount` for organism counts that
  `_update_density_fields` computes again in the same tick.
* `_feed` re-concatenates every living plant and carcass into new arrays each
  tick; at 20,000 plants that is 3 × 20,000-element allocations per tick.

## 4. Scientific weaknesses

1. **No controller abstraction.** The MLP is the only controller, so there is no
   way to ask "is this behaviour better than random?" inside the simulator.
   Assay G answers it offline with arena pairs, which is good but expensive.
2. **No per-organism survival/lineage record.** `death_cause` is overwritten
   when a slot is reused, so long-run mortality structure is lost.
3. **Statistics are one row per interval** with no derived rates; birth/death
   *rates* and per-capita rates must be recomputed by the reader.
4. **No sensitivity analysis.** Parameters change only by editing YAML; there is
   no supported way to sweep one parameter and report the response.
5. **No ODD document.** The protocol is described across DESIGN/ARCHITECTURE but
   not in the standard Overview–Design–Details form.
6. **`rest_fraction` divides a lifetime counter by the current tick**, which is
   correct for a cohort that all started at tick 0 but is mislabelled: it is the
   mean fraction of a lifetime spent resting, not a population rate.
7. **`escapes` is never incremented** (already documented as a known gap).
8. **Truncated damage credit** (max 8 attackers) is a real approximation and is
   documented, not hidden.

## 5. Visualization weaknesses

1. No overlays besides the five background layers: plant/organism density, soil
   richness, pheromone and vision are already fields but not viewable with a
   proper colour ramp and legend.
2. No analytics view: the sidebar has a 44 px sparkline and nothing else. There
   is no way to see births/deaths, trait drift or diversity without exporting.
3. No lineage view. `parent_a`/`parent_b` exist on every organism, so ancestry
   is available for the living population and must not be fabricated for the
   dead.
4. Charts redraw from full history every frame (the sparkline) — acceptable at
   44 px, not acceptable for a multi-panel analytics view.
5. Nothing shows a per-organism perception overlay, even though the sensor
   vector already contains what was seen.

## 6. What this upgrade will change

Ordered, with the risk of each step stated up front.

| # | Change | Risk | Determinism |
|---|---|---|---|
| 1 | Allocation-free `world.dx/dy/distance` | low | must be bit-identical |
| 2 | Distance-first filtering in `_nearest_visible` | low | must be bit-identical |
| 3 | Query only actors in `_feed`/`_predation` | low | must be bit-identical |
| 4 | Reuse grid objects; avoid full-field rebuilds in `collect` | medium | must be bit-identical |
| 5 | Controller abstraction + baselines | medium | opt-in only, default unchanged |
| 6 | New statistics: per-capita rates, trait variance, lineage depth | low | additive columns only at the end of `COLUMNS` |
| 7 | Sensitivity sweep harness | low | additive |
| 8 | ODD protocol document | none | none |
| 9 | Analytics view, overlays, lineage panel, perception overlay | none | none (viz only) |
| 10 | Benchmark + memory report | none | none |

Rule applied throughout: **every engine optimization must leave `checksum()`
identical to the pre-change baseline** on three fixed scenarios (plain, sexual,
no-predation). If a change cannot be made determinism-preserving, it is not
made. The baseline hashes are recorded in `tests/test_optimization_equivalence.py`
so the guarantee is enforced by CI, not by memory.
