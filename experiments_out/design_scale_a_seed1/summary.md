# Experiment A — Stable ecosystem

**Manipulation.** calibrated default, 30,000 ticks

## Run

- seed: `1`
- scale: `1.0` (design scale would be `1.0`)
- simulated ticks: `12000` (design horizon: `30000`)
- world: `1000 x 1000` units, `40` field cells
- founders: `300`, plants: `6000`
- reproduction: `asexual`

## Outcome

- final population: **0**
- extinct: **yes**
- mean population over the last fifth of the run: **0.8**
- births / deaths / mutations: `446` / `746` / `2690`
- final plant biomass: `13068.7` bu
- final soil nutrients: `52.5`

## Readouts at the final sample

- role mix (lifetime intake): herbivores 0, carnivores 0, omnivores 0, unclassified 0
- mean diet preference: `0.000`, attack: `0.000`, defence: `0.000`
- trait drift (first → last sample): mean_diet 0.481 → 0.000; mean_attack 2.584 → 0.000; mean_defense 2.471 → 0.000; mean_speed 19.706 → 0.000; mean_body 1.539 → 0.000; mean_vision 101.354 → 0.000; mean_lifespan 1290.884 → 0.000

## Interpretation limits

This is a *scaled* run (see the module docstring of `evolve.experiments`): smaller world, fewer organisms and a shorter
horizon than DESIGN §13. It demonstrates that the mechanism runs and what it did here; it is not evidence at design scale.
