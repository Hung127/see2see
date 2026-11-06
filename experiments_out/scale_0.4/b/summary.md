# Experiment B — Predator pressure

**Manipulation.** +200 founders with diet_preference 0.85 on the default baseline

## Run

- seed: `2`
- scale: `0.4` (design scale would be `1.0`)
- simulated ticks: `12000` (design horizon: `30000`)
- world: `400 x 400` units, `16` field cells
- founders: `48`, plants: `960`
- reproduction: `asexual`

## Outcome

- final population: **0**
- extinct: **yes**
- mean population over the last fifth of the run: **0.0**
- births / deaths / mutations: `2` / `82` / `6`
- final plant biomass: `2262.8` bu
- final soil nutrients: `8.1`

## Readouts at the final sample

- role mix (lifetime intake): herbivores 0, carnivores 0, omnivores 0, unclassified 0
- mean diet preference: `0.000`, attack: `0.000`, defence: `0.000`
- trait drift (first → last sample): mean_diet 0.628 → 0.000; mean_attack 2.452 → 0.000; mean_defense 2.350 → 0.000; mean_speed 20.108 → 0.000; mean_body 1.670 → 0.000; mean_vision 97.121 → 0.000; mean_lifespan 1461.595 → 0.000

## Interpretation limits

This is a *scaled* run (see the module docstring of `evolve.experiments`): smaller world, fewer organisms and a shorter
horizon than DESIGN §13. It demonstrates that the mechanism runs and what it did here; it is not evidence at design scale.
