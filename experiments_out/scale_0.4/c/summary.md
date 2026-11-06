# Experiment C — Severe drought

**Manipulation.** drought magnitude 0.6 for 600 ticks starting at tick 500

## Run

- seed: `3`
- scale: `0.4` (design scale would be `1.0`)
- simulated ticks: `1200` (design horizon: `3000`)
- world: `400 x 400` units, `16` field cells
- founders: `48`, plants: `960`
- reproduction: `asexual`

## Outcome

- final population: **45**
- extinct: **no**
- mean population over the last fifth of the run: **47.6**
- births / deaths / mutations: `252` / `255` / `166`
- final plant biomass: `1560.1` bu
- final soil nutrients: `57.9`

## Readouts at the final sample

- role mix (lifetime intake): herbivores 41, carnivores 0, omnivores 0, unclassified 4
- mean diet preference: `0.109`, attack: `4.414`, defence: `3.830`
- trait drift (first → last sample): mean_diet 0.529 → 0.109; mean_attack 2.720 → 4.414; mean_defense 2.577 → 3.830; mean_speed 20.364 → 22.941; mean_body 1.395 → 1.276; mean_vision 97.199 → 81.031; mean_lifespan 1172.219 → 533.696

## Notes

- The implemented drought gives moisture -60 % and plant growth -48 % (1-0.8*0.6), not the design's -70 % growth. The gap is reported rather than forced by changing the drought model.

## Interpretation limits

This is a *scaled* run (see the module docstring of `evolve.experiments`): smaller world, fewer organisms and a shorter
horizon than DESIGN §13. It demonstrates that the mechanism runs and what it did here; it is not evidence at design scale.
