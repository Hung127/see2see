# Experiment H — Sexual reproduction (sexual arm)

**Manipulation.** mode=sexual with 1,000 founders, against an asexual control at the same density and seed

## Run

- seed: `8`
- scale: `0.4` (design scale would be `1.0`)
- simulated ticks: `12000` (design horizon: `30000`)
- world: `730 x 730` units, `29` field cells
- founders: `160`, plants: `3200`
- reproduction: `sexual`

## Outcome

- final population: **0**
- extinct: **yes**
- mean population over the last fifth of the run: **0.0**
- births / deaths / mutations: `0` / `160` / `0`
- final plant biomass: `6446.0` bu
- final soil nutrients: `21.2`

## Readouts at the final sample

- role mix (lifetime intake): herbivores 0, carnivores 0, omnivores 0, unclassified 0
- mean diet preference: `0.000`, attack: `0.000`, defence: `0.000`
- trait drift (first → last sample): mean_diet 0.486 → 0.000; mean_attack 2.494 → 0.000; mean_defense 2.584 → 0.000; mean_speed 19.445 → 0.000; mean_body 1.374 → 0.000; mean_vision 98.816 → 0.000; mean_lifespan 1379.801 → 0.000

## Notes

- Arm of experiment H: genetics.mode = sexual.

## Interpretation limits

This is a *scaled* run (see the module docstring of `evolve.experiments`): smaller world, fewer organisms and a shorter
horizon than DESIGN §13. It demonstrates that the mechanism runs and what it did here; it is not evidence at design scale.
