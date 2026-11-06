# Experiment E — Environmental shift

**Manipulation.** +8 C from tick 15,000 and all right-half soil moved to the left half

## Run

- seed: `5`
- scale: `0.4` (design scale would be `1.0`)
- simulated ticks: `12000` (design horizon: `30000`)
- world: `400 x 400` units, `16` field cells
- founders: `48`, plants: `960`
- reproduction: `asexual`

## Outcome

- final population: **0**
- extinct: **yes**
- mean population over the last fifth of the run: **0.0**
- births / deaths / mutations: `412` / `460` / `1557`
- final plant biomass: `1933.9` bu
- final soil nutrients: `92.2`

## Readouts at the final sample

- role mix (lifetime intake): herbivores 0, carnivores 0, omnivores 0, unclassified 0
- mean diet preference: `0.000`, attack: `0.000`, defence: `0.000`
- trait drift (first → last sample): mean_diet 0.511 → 0.000; mean_attack 2.571 → 0.000; mean_defense 2.510 → 0.000; mean_speed 21.246 → 0.000; mean_body 1.301 → 0.000; mean_vision 94.468 → 0.000; mean_lifespan 1211.162 → 0.000

## Interpretation limits

This is a *scaled* run (see the module docstring of `evolve.experiments`): smaller world, fewer organisms and a shorter
horizon than DESIGN §13. It demonstrates that the mechanism runs and what it did here; it is not evidence at design scale.
