# Experiment A — Stable ecosystem

**Manipulation.** calibrated default, 30,000 ticks

## Run

- seed: `1`
- scale: `0.4` (design scale would be `1.0`)
- simulated ticks: `12000` (design horizon: `30000`)
- world: `400 x 400` units, `16` field cells
- founders: `48`, plants: `960`
- reproduction: `asexual`

## Outcome

- final population: **0**
- extinct: **yes**
- mean population over the last fifth of the run: **0.0**
- births / deaths / mutations: `0` / `48` / `0`
- final plant biomass: `1912.9` bu
- final soil nutrients: `5.3`

## Readouts at the final sample

- role mix (lifetime intake): herbivores 0, carnivores 0, omnivores 0, unclassified 0
- mean diet preference: `0.000`, attack: `0.000`, defence: `0.000`
- trait drift (first → last sample): mean_diet 0.482 → 0.000; mean_attack 2.717 → 0.000; mean_defense 2.233 → 0.000; mean_speed 20.775 → 0.000; mean_body 1.417 → 0.000; mean_vision 90.992 → 0.000; mean_lifespan 1382.681 → 0.000

## Interpretation limits

This is a *scaled* run (see the module docstring of `evolve.experiments`): smaller world, fewer organisms and a shorter
horizon than DESIGN §13. It demonstrates that the mechanism runs and what it did here; it is not evidence at design scale.
