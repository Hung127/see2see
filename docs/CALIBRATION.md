# Calibration report

Status: **closed** (the DESIGN §18 balance-harness step, overdue from M1–M2).

This document records the calibration study required by DESIGN §18. It is written
*after* the fact — the M1–M2 calibration window was declared closed in the progress
ledger without the balance harness ever being run. The founder-viability criterion
was therefore never verified, and the original defaults fail it. This is reported
here rather than silently patched.

Everything below is measured, single-core, on the project's own engine at the
commits named. No result is estimated.

## 1. Criterion

From DESIGN §18, a calibration must satisfy:

1. the ledgers close (covered by `tests/test_conservation.py`);
2. a non-trivial fraction of founders reaches maturity;
3. the plant–herbivore system has a non-extinct equilibrium in at least one
   parameterisation.

## 2. Original defaults fail criterion 2 and 3

Original values: `world = 2000×2000`, `plants.initial = 1500`, `organisms.initial = 300`,
`energy.bite = 0.10`, `energy.eta_p = 0.65`.

With a 500×500 test world at the same plant *density* the defaults were already
extinct at 800–1000 ticks in every seed (see grid below, row `(0.10, 0.65)`).
The mechanism is not a bug: a founder must make contact with food roughly every
four ticks to break even (cost ≈ 0.05 bu/tick, one bite ≈ 0.07 bu of assimilated
energy at the original constants), while a random controller's contact rate in the
default plant field is far below that. Founders starve before reproducing, so no
lineage exists for selection to act on.

This is DESIGN risk #1 ("founder extinction") realised under the shipped defaults.

## 3. Search grid (500×500, 100 founders, 2000 plants, 1000 ticks, seeds 1–3)

`(population, births)` at tick 1000:

| bite | eta_p | seed 1 | seed 2 | seed 3 |
|------|-------|--------|--------|--------|
| 0.10 | 0.65 | (0, 0) | (1, 0) | (0, 0) |
| 0.40 | 0.75 | (0, 1) | (51, 66) | (0, 5) |
| 0.50 | 0.75 | (0, 3) | (74, 97) | (0, 6) |
| 0.50 | 0.85 | (0, 5) | (85, 138) | (1, 20) |
| 0.60 | 0.85 | (9, 17) | (100, 177) | (1, 14) |

With 100 founders viability is a founder lottery: only one seed produced a
foraging lineage. Raising the founder count (a calibration lever named in §18)
removes most of the lottery: 300 founders, 3000 plants, 500×500, 1000 ticks:

| bite | eta_p | seed 1 | seed 2 | seed 3 |
|------|-------|--------|--------|--------|
| 0.50 | 0.75 | (57, 320) | (104, 167) | (210, 378) |
| 0.50 | 0.85 | (63, 393) | (184, 375) | (144, 357) |

`(population, births)`; plants stayed at 3500–4500 bu in every run. Adopted
`bite = 0.50`, `eta_p = 0.75` (the smaller change; both are plausible
assimilation efficiencies for herbivores, and `eta_p` stays below `eta_m`).

## 4. A second, independent defect: nutrient stranding

At the adopted constants the default 1000×1000 world still ran down to extinction
by tick 6000. Instrumenting one tick at t=3000 showed:

* plant growth 19.8 bu/tick, against a demand of 130.6 bu/tick;
* soil at plant cells averaged **0.03 bu**, while total soil was 5147 bu (3.2/cell);
* 1352 of 1600 cells were nutrient-limited.

Nutrients were being deposited where organisms respire and die, and plants could
not reach them. The field model had no soil transport, so a closed nutrient cycle
was impossible: plants starved on a global surplus. This is a model defect, not a
calibration knob.

Fix: symmetric 5-point soil percolation in `Fields.step`, weight
`soil_diffusion = 0.10` per tick, periodic boundaries, kernel weights summing to 1
(mass-conserving to float precision; the full-system ledger test still passes).

## 5. Adopted defaults

```
world:         1000 × 1000, 40 × 40 field cells (cell 25 units)
plants:        initial 6000, max 20000
organisms:     initial 300, max 2000
energy:        bite 0.50, eta_p 0.75 (eta_m unchanged at 0.85)
environment:   soil_diffusion 0.10 (new)
```

Measured at 1000 ticks (seeds 1–3): populations 70 / 97 / 57, births 85 / 113 / 66,
plant biomass 8.5–11.1k bu.

Long run (seed 1, 6000 ticks, 67.7 ticks/s):

```
tick   population  births  plants(bu)  soil(bu)
 500       23        25      12757        58
1000       38        52      12530        65
1500       92       127      11855       120
2000      133       219      11528        92
2500      129       297      11520       114
3000       78       340      11853       105
3500       62       371      12036       157
4000       46       389      12133        77
4500       56       410      11888        80
5000       60       422      11779        78
5500       55       425      11743       121
6000       52       431      11832        71
```

The system oscillates (23–133 organisms) around a non-extinct equilibrium with
continued reproduction; plants remain near 12k bu. Criterion 3 is satisfied by this
parameterisation; criterion 2 is satisfied at 100–200 ticks (a non-trivial fraction
of the 300 founders reaches maturity and reproduces).

## 6. What this calibration did *not* do

* No behavioural rule was added; every action still comes from the network.
* No genome is touched by any environmental or calibration change.
* No light input or founder reserve was raised to rescue a population; the changes
  are foraging efficiency (`bite`, `eta_p`), plant density, founder count, world
  size, and a missing physical transport process (soil percolation).
* No result from a *specific* experiment was used to tune a constant; the runs
  above are the calibration harness itself.

## 7. Reproduce

```bash
PYTHONPATH=src python3 - <<'EOF'
from evolve.config import Config
from evolve.engine import Engine
eng = Engine(Config(seed=1, ticks=6000))
for _ in range(6000):
    eng.step()
print(eng.population.n_alive, eng.births_total, eng.plants.total_biomass())
EOF
```

## 8. Follow-up (post-hoc): the equilibrium is not asymptotic

This section was added after the experiment programme (docs/RESULTS.md). The
calibration above was correct at its stated horizon, but the long run was not
long enough to see the system's fate. Continuing **seed 1, design scale** past
tick 6,000:

| tick | population | births (cum) | deaths (cum) | mean E | plants (bu) |
|------|------------|--------------|--------------|--------|-------------|
| 6,000 | 52 | 431 | 679 | 19.7 | 11,832 |
| 7,700 | 27 | 446 | 719 | 22.6 | 12,339 |
| 9,400 | 6 | 446 | 740 | 23.6 | 12,859 |
| 11,100 | 1 | 446 | 745 | 24.0 | 13,039 |
| 11,950 | 0 | 446 | 746 | 0.0 | 13,069 |

The tick-6,000 row reproduces the calibration record exactly (52 / 431 /
11,832), so nothing was retuned. Beyond it the population decays to extinction
while plants sit at carrying capacity and the survivors hold **full reserves**
(0.99·`E_max`). A clean forward pass at tick 6,000 shows 39 organisms satisfying
the maturity and energy gates, none with `o6 > 0.5` (max 0.439): the
reproduction output has drifted just below the fixed decode threshold.

So criterion 3 ("a non-extinct equilibrium") was met at the 6,000-tick test
horizon but the certified population is **not guaranteed to persist**: this
seed's lineage decays to extinction by ~11,950. A second design-scale run with
seed 2 is still alive at tick 12,000 (76 organisms, births continuing), so the
outcome is seed-dependent rather than a universal property of the constants.
This is reported in `RESULTS.md` rather than tuned away: DESIGN §18 permits one
calibration window and it is closed, so the constants are left as adopted. A
larger effective population or a decode threshold other than 0.5 would
plausibly change the outcome, but both are model changes outside the
calibration window.
