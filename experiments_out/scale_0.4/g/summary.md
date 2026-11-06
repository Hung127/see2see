# Assay G — evolved vs random (paired arenas)

Protocol: `pre_registration.md` in this directory, written before any engine was constructed.

## Run

- scale: `0.4` (design scale is `1.0`)
- evolution: 6 replicates of 8000 ticks (assay config, asexual, no predators, clumped plants)
- sampled genomes: 10 per replicate from living survivors, uniformly at random, no ranking
- arenas: 0 paired arenas, 200 ticks each, 200 x 200 world at the calibrated plant density, mirrored positions, identical plant layout

## Evolution phase

| Replicate | Seed | Living survivors at final tick | Sampled |
|---|---|---|---|
| 0 | 100 | 0 | 0 |
| 1 | 101 | 0 | 0 |
| 2 | 102 | 0 | 0 |
| 3 | 103 | 0 | 0 |
| 4 | 104 | 0 | 0 |
| 5 | 105 | 0 | 0 |

## Primary endpoint: food intake (bu)

| Set | n pairs | mean diff (evolved - random) | 95 % CI | median diff | W+ | z | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|
| pooled | 0 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0.0 | +0.000 | 1.0000 |
| replicate 0 | 0 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0.0 | +0.000 | 1.0000 |
| replicate 1 | 0 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0.0 | +0.000 | 1.0000 |
| replicate 2 | 0 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0.0 | +0.000 | 1.0000 |
| replicate 3 | 0 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0.0 | +0.000 | 1.0000 |
| replicate 4 | 0 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0.0 | +0.000 | 1.0000 |
| replicate 5 | 0 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0.0 | +0.000 | 1.0000 |

## Secondary endpoints

- survival at the final arena tick: evolved 0/0, random 0/0
- mean final energy: evolved nan, random nan

## Conclusion (as pre-registered)

Mean paired difference in intake: **+0.0000 bu** (95 % CI [+0.0000, +0.0000]), Wilcoxon p = 1.0000 on 0 non-zero pairs. Under the pre-registered reading, the 95 % bootstrap CI includes zero, so this scaled run does not establish a difference.

This is a scaled-down run: fewer, shorter replicates than DESIGN §13. The result is reported as measured, including any null result.
