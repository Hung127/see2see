# Evolution

## What selection is here

There is no fitness function. Selection is demographic: an organism reproduces
when its network says so, it is mature, and its reserves exceed the genome's own
threshold. Genomes that acquire energy faster — by foraging, digesting,
defending, or simply costing less — leave more children. Everything else is
physics.

This is the model's central claim: adaptation is an *outcome* of the closed
energy economy, not an objective the code optimises.

## Heritable variation

* 12 trait loci (see `GENETICS.md`) with per-locus mutation at the genome's own
  `mutation_rate` locus, `σ = 0.08` per mutated trait locus, occasional 5×
  large-effect jumps.
* 380 neural loci with magnitude-scaled σ (`0.05·(1 + |w|/4)`), so behaviour
  drifts faster where weights are already large.
* Sexual reproduction adds uniform whole-locus crossover.

Mutation is symmetric and undirected; there is no back-mutation bias, no
mutational pressure toward any trait value, and no hidden tweak of any trait.

## What is measured

Per statistics interval (`stats.interval`, default 50 ticks):

* **Population and flows**: population, births, deaths, growth rate (population
  delta per interval), mutations per birth.
* **Roles** (lifetime intake): herbivore if plant intake > 2× meat, carnivore if
  meat > 2× plant, omnivore otherwise; unclassified below 1 bu total. Roles are
  measured, never assigned.
* **Trait means**: body size, speed, vision, attack, defence, diet preference,
  lifespan, mutation rate.
* **Diversity**: mean pairwise genome distance over the 12 trait loci, mean
  per-locus variance, diet entropy (normalised by `log2(20)` bins), and the
  count of unique genome hashes (sha256 of the genome rounded to 1e-6).
* **Behaviour**: distance moved, bites, hunts, rest fraction, foraging
  efficiency, reproduction attempts and successes. (`mean_escapes` is present in
  the statistics schema but always zero: "escape" in this model is the continuous
  sprint modifier, not a discrete attempt, so there is no event to count. The
  column is kept for schema stability and is documented as a known gap rather
  than filled with a proxy.)

## Expected dynamics (and what would falsify the model)

* **Drift without selection**: with mutation alone and no differential
  reproduction, trait means should random-walk, not trend. A systematic trend
  in a closed, selection-free run would indicate a hidden bias.
* **Selection under pressure**: a trait that changes energy intake (e.g. diet
  preference matching available food) should shift in the direction that
  increases intake, at a rate limited by mutation supply and generation time.
* **Diversity collapse**: a bottleneck (few founders, or a crash) should reduce
  unique genomes and mean pairwise distance; recovery requires new mutation.
* **Role composition** should track the food environment: clumped, scarce
  plants (experiment D) should shift the measured role mix toward whatever
  strategy actually acquires more energy — including "neither", i.e. death.

## Sexual vs asexual

Sexual reproduction costs mate-finding time and splits the investment, but
recombines loci. The model does not decide which is better; experiment H runs
both arms at the same density and seed and reports what happened (see
`RESULTS.md`). `genetics.mode = both` falls back to asexual reproduction when no
mate is available.

## What evolution cannot do here

* No speciation or species labels: lineages are tracked by parentage
  (`lineage` table), not by clusters.
* No learning: the network is fixed at birth; nothing updates weights during
  life.
* No Lamarckian inheritance: what an organism does in life never edits its
  genome.
* No mutation of the mutation machinery beyond the `mutation_rate` locus
  itself, which mutates like any other trait.
