# Experiment F — No predators

**Manipulation.** attack loci clipped to 0 at founding and the predation stage disabled

## Run

- seed: `6`
- scale: `0.4` (design scale would be `1.0`)
- simulated ticks: `12000` (design horizon: `30000`)
- world: `400 x 400` units, `16` field cells
- founders: `48`, plants: `960`
- reproduction: `asexual`

## Outcome

- final population: **65**
- extinct: **no**
- mean population over the last fifth of the run: **40.6**
- births / deaths / mutations: `656` / `639` / `1097`
- final plant biomass: `1441.8` bu
- final soil nutrients: `15.8`

## Readouts at the final sample

- role mix (lifetime intake): herbivores 63, carnivores 0, omnivores 0, unclassified 2
- mean diet preference: `0.222`, attack: `0.000`, defence: `1.272`
- trait drift (first → last sample): mean_diet 0.517 → 0.222; mean_attack 0.000 → 0.000; mean_defense 2.329 → 1.272; mean_speed 19.162 → 31.045; mean_body 1.476 → 1.406; mean_vision 101.605 → 74.074; mean_lifespan 1246.313 → 820.446

## Interpretation limits

This is a *scaled* run (see the module docstring of `evolve.experiments`): smaller world, fewer organisms and a shorter
horizon than DESIGN §13. It demonstrates that the mechanism runs and what it did here; it is not evidence at design scale.
