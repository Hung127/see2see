# Experiment H — Sexual reproduction

**Manipulation.** mode=sexual with 1,000 founders, against an asexual control at the same density and seed

Both arms use the same seed, world, plant layout and founder genomes; they differ only in `genetics.mode`.

- scale: `0.4`, simulated ticks: `12000`

| Arm | final pop | births | deaths | extinct | mean pairwise distance | unique genomes | repro attempts | repro successes |
|---|---|---|---|---|---|---|---|---|
| sexual | 0 | 0 | 160 | yes | 0.0000 | 0 | 0.00 | 0.00 |
| asexual | 0 | 10 | 170 | yes | 0.0000 | 0 | 0.00 | 0.00 |

## Notes

- Birth rate, mate encounters and offspring viability are read from the same simulated time in both arms; the sexual arm must find mates, the asexual arm does not.
- A scaled run is shorter than the design horizon, so the sexual arm gets fewer opportunities to encounter a mate than DESIGN §13 assumes.

## Interpretation limits

Scaled run (see `evolve.experiments`): not evidence at design scale.
