# Experiment D — Resource competition

**Manipulation.** plants reduced to 40 % and clustered, 1,000 ticks

## Run

- seed: `4`
- scale: `0.4` (design scale would be `1.0`)
- simulated ticks: `400` (design horizon: `1000`)
- world: `400 x 400` units, `16` field cells
- founders: `48`, plants: `384`
- reproduction: `asexual`

## Outcome

- final population: **1**
- extinct: **no**
- mean population over the last fifth of the run: **1.0**
- births / deaths / mutations: `1` / `48` / `4`
- final plant biomass: `1230.1` bu
- final soil nutrients: `257.9`

## Readouts at the final sample

- role mix (lifetime intake): herbivores 1, carnivores 0, omnivores 0, unclassified 0
- mean diet preference: `0.067`, attack: `0.294`, defence: `1.834`
- trait drift (first → last sample): mean_diet 0.524 → 0.067; mean_attack 2.553 → 0.294; mean_defense 2.868 → 1.834; mean_speed 17.934 → 23.868; mean_body 1.629 → 0.992; mean_vision 104.262 → 63.164; mean_lifespan 1264.611 → 1403.246

## Interpretation limits

This is a *scaled* run (see the module docstring of `evolve.experiments`): smaller world, fewer organisms and a shorter
horizon than DESIGN §13. It demonstrates that the mechanism runs and what it did here; it is not evidence at design scale.
