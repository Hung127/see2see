# Assay G — evolved vs random (paired arenas)

Protocol: `pre_registration.md` in this directory, written before any engine was constructed.

## Run

- scale: `1.0` (design scale is `1.0`)
- evolution: 6 replicates of 20000 ticks (assay config, asexual, no predators, clumped plants)
- sampled genomes: 10 per replicate from living survivors, uniformly at random, no ranking
- arenas: 30 paired arenas, 500 ticks each, 200 x 200 world at the calibrated plant density, mirrored positions, identical plant layout

## Evolution phase

| Replicate | Seed | Living survivors at final tick | Sampled |
|---|---|---|---|
| 0 | 100 | 0 | 0 |
| 1 | 101 | 0 | 0 |
| 2 | 102 | 131 | 10 |
| 3 | 103 | 705 | 10 |
| 4 | 104 | 181 | 10 |
| 5 | 105 | 0 | 0 |

## Primary endpoint: food intake (bu)

| Set | n pairs | mean diff (evolved - random) | 95 % CI | median diff | W+ | z | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|
| pooled | 30 | +49.9319 | [+42.1076, +57.7217] | +54.4405 | 465.0 | +4.782 | 0.0000 |
| replicate 0 | 0 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0.0 | +0.000 | 1.0000 |
| replicate 1 | 0 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0.0 | +0.000 | 1.0000 |
| replicate 2 | 10 | +73.3682 | [+66.8662, +78.6874] | +76.5081 | 55.0 | +2.803 | 0.0051 |
| replicate 3 | 10 | +27.0936 | [+26.4587, +27.5768] | +27.3316 | 55.0 | +2.803 | 0.0051 |
| replicate 4 | 10 | +49.3337 | [+38.7732, +55.6256] | +54.8213 | 55.0 | +2.803 | 0.0051 |
| replicate 5 | 0 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 0.0 | +0.000 | 1.0000 |

## Secondary endpoints

- survival at the final arena tick: evolved 30/30, random 1/30
- mean final energy: evolved 17.7657, random 0.4487

## Conclusion (as pre-registered)

Mean paired difference in intake: **+49.9319 bu** (95 % CI [+42.1076, +57.7217]), Wilcoxon p = 0.0000 on 30 non-zero pairs. Under the pre-registered reading, the 95 % bootstrap CI excludes zero; evolved genomes had higher intake in this design-scale run.

This is a design-scale run (horizon, arena length and plant density as specified). 
The result is reported as measured, including any null result.
