# Experiment A — Stable ecosystem

**Manipulation.** calibrated default, 30,000 ticks

## Run

- seed: `2`
- scale: `1.0` (design scale would be `1.0`)
- simulated ticks: `12000` (design horizon: `30000`)
- world: `1000 x 1000` units, `40` field cells
- founders: `300`, plants: `6000`
- reproduction: `asexual`

## Outcome

- final population: **76**
- extinct: **no**
- mean population over the last fifth of the run: **85.4**
- births / deaths / mutations: `854` / `1078` / `266`
- final plant biomass: `11230.4` bu
- final soil nutrients: `185.2`

## Readouts at the final sample

- role mix (lifetime intake): herbivores 76, carnivores 0, omnivores 0, unclassified 0
- mean diet preference: `0.365`, attack: `2.635`, defence: `3.853`
- trait drift (first → last sample): mean_diet 0.484 → 0.365; mean_attack 2.460 → 2.635; mean_defense 2.649 → 3.853; mean_speed 20.455 → 31.172; mean_body 1.481 → 2.658; mean_vision 98.386 → 67.269; mean_lifespan 1246.786 → 2730.429

## Interpretation limits

This is a *scaled* run (see the module docstring of `evolve.experiments`): smaller world, fewer organisms and a shorter
horizon than DESIGN §13. It demonstrates that the mechanism runs and what it did here; it is not evidence at design scale.
