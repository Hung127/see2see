# Ecosystem

The world is a coarse environmental grid (40 × 40 cells of 25 units in the
default 1,000 × 1,000 world) with terrain, soil, temperature, moisture and
pheromone. Plants and organisms are continuous-position agents on the same
world. All of it is inside the two conservation ledgers (see
`ENERGY_MODEL.md`).

## World and boundaries

`wrap` (default), `reflect` or `hard`. Distance and bearing are computed on the
chosen topology; the wrap world has no edges to exploit. Terrain is seeded
smooth noise normalised to `[0, 1]`; height raises local plant carrying capacity
(`K_local = k0·(0.6 + 0.8·h)`) and lowers flood depth (`depth = m·(1 − h)`).

## Environment forcing

Each tick (`fields.step`):

* **Season**: `T = t_mean + t_amp·sin(2π·t/period)`,
  `m = m_mean + m_amp·sin(2π·t/period − π/2)`, clipped to `[−30, 60] °C` and
  `[0, 1]` moisture. Defaults: 20 ± 6 °C, 0.55 ± 0.15 moisture, period 2,000
  ticks.
* **Rain**: each cell gets `+0.02` moisture with probability `rain_rate` per
  tick (default 0.002), drawn from the single simulation generator.
* **Pheromone**: 5-point diffusion (weights 0.5 centre, 0.125 neighbours) then
  ×0.9 decay. Organisms deposit pheromone when they strike.
* **Soil percolation**: symmetric 5-point mixing with wrap-around, weights
  summing to 1, so the soil total is exactly conserved. This exists because
  respiration deposits nutrients where organisms die, which may be far from
  roots; without percolation, plants starve on a global surplus
  (`CALIBRATION.md`).
* **Event modifiers** reset to neutral each tick and are re-applied by the
  event system (below).

## Plants

* **Growth**: logistic, limited by local soil, temperature and moisture
  (`ENERGY_MODEL.md`). Growth is the only light input and the only
  soil → organic channel.
* **Germination**: up to `germination_per_tick` (20) seedlings per tick,
  biomass-weighted by cell, each costing `seed_cost = 0.2` bu of local soil.
  No soil, no seedling.
* **Senescence**: `0.002·P` per tick leaves the plant and enters detritus.
* **Consumption**: organisms bite the nearest plant or carcass in reach; a
  plant reduced to zero dies.

Plants are not individuals with a physiology beyond biomass: age is tracked for
statistics only.

## Carrion and detritus

Carcasses are spatially explicit lumps left by predation; scavengers can find
them with the food sensor and compete for them exactly as for plants.
Non-predation deaths (starvation, senescence, residual health loss) become
diffuse detritus at the death site's mass total but without a position.
Detritus and carcasses mineralise at `0.015·C` per tick: mass to soil, equal
energy to heat. Nothing is deleted; the organic pool always has a destination.

## Predation

A strike is possible when the network says `attack` (`o4 > 0.5`), the organism
is not resting, and another organism is within
`reach = 0.5·M^(1/3) + 2`. Damage is `max(0, 3·attack·M^0.4 − 0.8·1.5·defense·M_def^0.4)`
health points. Killing is not a special action: it is what happens when health
reaches zero. Meat is apportioned among credited attackers, the rest becomes a
carcass. There is no "predator species"; a herbivore with a high attack locus
can strike, it just digests meat badly (`η_meat = 0.85·sqrt(d)`).

## Environmental events

Event effects are **physical only** — they change fields, plants, soil, health
or the ledger, and never a genome (tested). Scheduled events live in the config;
random events are Poisson draws per configured kind per tick.

| Kind | Type | Effect |
|------|------|--------|
| `drought` | continuous | moisture × (1 − m); plant growth × (1 − 0.8m); animal water stress m → extra cost `2·C_basal·m` |
| `heavy_rain` | continuous | moisture + m (clipped); organisms lose `10·m·(1 − terrain)` health per tick — low ground floods |
| `heatwave` | continuous | temperature + 10m °C |
| `cold_snap` | continuous | temperature − 10m °C |
| `wildfire` | instant | plant biomass in the region → soil, equal energy → heat; organisms in the region lose `20·m` health |
| `plant_blight` | instant | fraction `m` of plant biomass in the region → detritus (killed, not burned: no soil/heat channel) |
| `nutrient_pulse` | instant | `m` bu of nutrient per cell in the region enters soil (`ledger.mass_in`: the only external mass input) |

Continuous events are re-applied every active tick after `fields.step`, so they
win over the seasonal baseline for their duration. Regions are discs
`(cx, cy, radius)`; a huge radius means the whole world. Random events draw
their centre, radius, duration and magnitude from the simulation generator in
config order.

## What is emergent

* **Carrying capacity** — no animal cap is imposed beyond the array capacity;
  the equilibrium is whatever plant growth and metabolism allow.
* **Trophic structure** — "herbivore/carnivore/omnivore" are *measured* labels
  from lifetime intake (plant > 2× meat, meat > 2× plant, else omnivore;
  unclassified below 1 bu), not assigned roles.
* **Patch use, clustering, local extinction** — consequences of movement and
  feeding, not rules.
