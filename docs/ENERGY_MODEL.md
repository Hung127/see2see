# Energy model

All formulas below are the implemented ones (`src/evolve/energy.py`,
`src/evolve/engine.py`); each is verified against the code. Units are biomass
units (bu); mass and chemical energy are 1:1 in living tissue, reserves,
detritus and carcasses.

## The two invariants

Two ledgers are tracked simultaneously and asserted to `rtol = 1e-9` (with a
documented `atol = 1e-9` floor for near-zero stocks):

* **Organic mass**: `final = initial + mass_from_soil − mass_to_soil`.
  Organic matter = organisms (`body + E`) + plants + detritus + carcasses.
  `mass_in` is deliberately *not* part of this invariant: it is an input to the
  *soil* pool (rain-borne nutrients), and a nutrient pulse that cycles
  soil → plant → soil would otherwise be double-counted.
* **Chemical energy**: `final = initial + light_in − heat_out`. Light is the
  only input (plant growth); heat is the only sink. Every channel that sends
  organic mass to soil releases exactly that much energy as heat, which is what
  lets both invariants close at once.
* **Full-system mass** (checked separately): `organic + soil = initial +
  mass_in`.

## Metabolic cost (per tick)

```
C_basal   = k_b · M^0.75 · metabolic_rate
C_repair  = k_r · M^0.75 · sqrt(lifespan / 1000)
C_brain   = k_n · 380                      (380 network parameters)
C_move    = k_m · M · (v / 10)^2           (v = actual speed this tick)
C_vision  = k_v · (r_vis / 100)^2 · (angle / 360)
C_attack  = k_a · attack
C_defense = k_d · defense
C_enzyme  = k_e · 4 · d · (1 − d)          (d = diet_preference; 0 for specialists)
C_thermal = C_basal · 0.5 · ((T − 20)/10)^2
C_total   = Σ of the above + water_stress_cost · C_basal · water_stress
```

Defaults: `k_b = 0.015, k_r = 0.002, k_n = 2e-6, k_m = 0.0008, k_v = 0.002,
k_a = 0.004, k_d = 0.003, k_e = 0.004, water_stress_cost = 2.0`.
The movement term is charged on the *actual* distance moved, not the genetic
maximum, so resting is cheap and sprinting is expensive.

Payment order: reserves pay first (`paid = min(C_total, E)`); any deficit is
catabolised from the body at `catabolism = 0.5` bu/tick, and each catabolism
tick also costs `starvation_health_loss = 1.0` health. Health regenerates
(`health_regen = 0.05`) only when `E > 2·C_basal` and the organism is not
starving.

Every paid unit is respired: mass to the organism's soil cell, equal energy to
heat (`ledger.mass_to_soil += paid`, `ledger.heat_out += paid`).

## Feeding

One bite per eater per tick at the nearest plant *or* carcass within reach
`reach = 1.5·M^(1/3) + 2`. Demand is
`min(bite_rate, supply, E_max − E)` with `bite_rate = bite · M^0.75`
(`bite = 0.5`). Contested resources are shared proportionally; the exact
floating-point remainder goes to the lowest-ID eater so the stock closes.

Digestive efficiency (DESIGN §6.3):

```
η_plant = eta_p · sqrt(1 − d) · sqrt(metabolic_rate)
η_meat  = eta_m · sqrt(d)     · sqrt(metabolic_rate)
```

`eta_p = 0.75`, `eta_m = 0.85`. A pure herbivore (`d = 0`) digests meat at
exactly zero efficiency and vice versa; omnivores pay the enzyme cost and are
mediocre at both. What is eaten but not digested (`taken − gain`) becomes
detritus — it keeps both its mass and its energy.

## Predation

Strike damage (DESIGN §6.4):

```
damage = max(0, k_dmg · attack · M_att^0.4 − 0.8 · k_mit · defense · M_def^0.4)
```

`k_dmg = 3.0`, `k_mit = 1.5`. The strike cost `0.05 · attack` is charged only
on an actual strike (a target in reach), never on an empty swing. All strikes
in a tick resolve simultaneously against start-of-stage health. A kill's meat
(`body + E` of the victim) is apportioned by credited damage (each attacker
digests at its own `η_m`); whatever no attacker took becomes a spatially
explicit carcass. Damage credit is cumulative per victim and capped at 8
attackers (oldest dropped).

## Reproduction

`E_max = 10 · M^0.9`. An organism reproduces when the network wants to, it is
mature (`maturity_age = 50`), and `E ≥ repro_threshold · E_max`. It invests
`repro_investment · E_max` (sexual: both parents pay half). The investment
builds the child: `body` from the child's genome, `0.9` of the remainder as the
child's reserves, `0.1` as waste to soil (mass to soil, equal energy to heat).
Gates fail with nobody paying.

## Plant growth and the light input

Per tick, each plant grows by logistic demand limited by local soil:

```
demand = r · P · max(0, 1 − P/K_local) · f_T · f_M · f_event
K_local = k0 · (0.6 + 0.8 · terrain)
f_T = exp(−((T − 22)/8)^2)
f_M = 0.3 + 0.7 · moisture
```

`r = 0.06`, `k0 = 5.0`. When a cell's total demand exceeds its soil, growth is
scaled down proportionally and the exact remainder goes to the lowest-ID plant
in the cell. Growth is the *only* `light_in` and `mass_from_soil` channel:
`ledger.light_in += growth`, `ledger.mass_from_soil += growth`. Germination
costs `seed_cost = 0.2` bu of soil per seedling and books the same amount as
light input.

## Losses to soil

* **Senescence**: plants lose `0.002 · P` per tick (mass leaves the plant pool
  as detritus, not directly to soil).
* **Decomposition**: detritus and carcasses mineralise at `decay · C` per tick
  (`decay = 0.015`): `δ` leaves the organic pool, `δ` enters the soil pool as
  mass, and exactly `δ` leaves as heat. The scalar detritus pool has no
  position, so its mineralised mass returns evenly across the grid (documented
  coarse-summary ruling); carcasses mineralise at their own position.
* **Undigested food** becomes detritus (above), so it re-enters the decay path
  rather than vanishing.

## What is *not* in the model

No trophic multipliers, no fixed energy quotas, no carrying-capacity fudge for
animals: the animal carrying capacity is an *emergent* consequence of plant
growth, bite rates and metabolism. There is no way to add energy to the system
except plant growth, and no way to remove it except heat.
