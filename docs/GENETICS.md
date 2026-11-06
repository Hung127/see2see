# Genetics

## Gene layout

One genome is a 392-element float64 vector: 12 trait loci followed by 380 neural
parameters. Trait loci live in `[0, 1]` and are mapped to physical phenotypes by
a per-locus specification; neural loci live in `[-8, 8]` and are used directly as
network weights (see `NEURAL_NETWORKS.md`).

| # | Locus | Mapping | Range | Meaning |
|---|-------|---------|-------|---------|
| 0 | `body_size` | exponential | 0.3 – 5.0 | structural mass `M` (bu) |
| 1 | `speed_max` | linear | 0 – 40 | maximum speed (units/tick) |
| 2 | `vision_range` | linear | 40 – 160 | sensor radius (units) |
| 3 | `vision_angle` | linear | 30 – 360 | sensor cone (degrees) |
| 4 | `metabolic_rate` | linear | 0.6 – 1.6 | metabolic scope multiplier |
| 5 | `attack` | linear | 0 – 5 | strike strength |
| 6 | `defense` | linear | 0 – 5 | damage mitigation |
| 7 | `repro_threshold` | fraction | 0.40 – 0.95 | reproduction energy gate (× `E_max`) |
| 8 | `repro_investment` | fraction | 0.15 – 0.60 | investment (× `E_max`) |
| 9 | `diet_preference` | linear | 0 – 1 | 0 = herbivore, 1 = carnivore |
| 10 | `lifespan` | exponential | 200 – 5000 | senescence age (ticks) |
| 11 | `mutation_rate` | exponential | 5e-4 – 5e-2 | per-locus mutation probability |

Exponential mapping (`exp(lo + g·(log hi − log lo))`) is used where a trait
spans orders of magnitude; linear where the range is already multiplicative in
effect. The mapping is `phenotype()` in `genome.py`; the table above is the
single source of truth (`GENE_SPECS`).

## Founders

`founders(n, rng)` draws trait loci from `Beta(2, 2)` (mid-biased, no
corner solutions) and neural parameters from `N(0, 0.35)` weights and
`N(0, 0.05)` biases. The founder population is deliberately unbiased: no
"starter instincts", no species labels, no diet prior. What a founder does is
whatever its random network does.

## Mutation

Per-locus Bernoulli mutation (DESIGN §4.2). For each child, each locus mutates
with probability `mutation_rate` (locus 11, mapped). A mutated locus moves by
`N(0, σ)`:

* trait loci: `σ = sigma_trait` (0.08);
* neural loci: `σ = sigma_w · (1 + |w|/4)` (0.05 scaled by weight magnitude, so
  large weights drift faster);
* a small fraction (`large_effect_p` = 0.002) of mutations is
  `large_effect_mult` = 5× larger — occasional jumps, so adaptation is not
  limited to a random walk's pace.

Draw order is mask, deltas, large-effect flags. Trait loci are clipped to
`[0, 1]`, neural loci to `[-8, 8]`. `mutate()` returns the child and the number
of mutated loci (counted in `mutations_total`).

## Crossover

Sexual reproduction uses uniform per-locus crossover with probability 0.5 and no
blending: each locus is taken whole from one parent. Blending would smooth
trait distributions and hide recombination's effect on linkage; whole-locus
inheritance keeps the trait/neural block structure visible.

## Reproduction modes

`genetics.mode` ∈ {`asexual`, `sexual`, `both`}:

* **asexual** — one parent invests its full `repro_investment · E_max`; the child
  is a mutated copy.
* **sexual** — two willing, mature, sufficiently rich parents within mate reach
  (`3 · M^(1/3)`) each pay half the investment; the child is a crossover of both
  genomes, mutated at the mean of the parents' mutation rates.
* **both** — a parent reproduces sexually when a mate is available and asexually
  otherwise (implemented in `_reproduce`'s mode dispatch).

Reproduction gates fail with *nobody paying*: if the investment cannot build the
child's body, or the parent cannot afford its share, or the population cap is
reached, nothing is deducted. Failed attempts are counted
(`repro_attempts`/`repro_success`).

## What genetics does *not* do

* No species, no speciation, no fitness function, no elitism. Selection is
  purely demographic: organisms that acquire energy reproduce more.
* No directed mutation: deltas are symmetric and independent of the trait's
  current value (except the neural magnitude scaling above).
* No gene flow, no migration, no kin recognition.
