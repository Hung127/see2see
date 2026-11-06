# EVOLVE — Research Notes

Status: research pass complete (M0). Sources read 2026-11-06.

Sources:

| # | Source | Kind |
|---|--------|------|
| R1 | Wilensky, U. (1997). *NetLogo Wolf Sheep Predation* model. CCL, Northwestern. https://ccl.northwestern.edu/netlogo/models/WolfSheepPredation | Agent-based teaching model |
| R2 | *Avida* entry, Encyclopedia of Artificial Life (ISAL), ed. Vostinar/Dolson/Lalejini. https://alife.org/encyclopedia/digital-evolution/avida/ | Platform overview |
| R3 | Stanley, K. O. & Miikkulainen, R. (2002). *Evolving Neural Networks through Augmenting Topologies.* Evolutionary Computation 10(2):99–127. PMID 12180173 | Neuroevolution method |
| R4 | Fortuna, M. A., Zaman, L., Wagner, A. P., Ofria, C. (2013). *Evolving Digital Ecological Networks.* PLoS Comput Biol 9(3):e1002928 | Review / topic page |
| R5 | Gras, R., Devaurs, D., Wozniak, A., Aspinall, A. (2009). *An individual-based evolving predator-prey ecosystem simulation using a fuzzy cognitive map as the behavior model.* Artificial Life 15(4):423–63. PMID 19463060 | Individual-based ALife model |

---

## 1. What each source actually contributes

### R1 — Wolf Sheep Predation (NetLogo)

**Concepts.** Two variants. In *sheep-wolves*, grass is infinite and implicit; wolves hunt sheep,
every step costs one energy unit, and each agent has a fixed per-tick probability of reproducing.
In *sheep-wolves-grass*, grass is an explicit patch variable that regrows after a fixed delay, and
sheep must eat it. The grass variant is reported as generally stable and closer to Lotka–Volterra
oscillation, while the two-species variant is generally unstable. Small populations underestimate
real-valued LV dynamics because integer agents go extinct.

**Worth taking.**
- Energy as the single universal currency: distance moved is paid for in energy.
- Resources as an explicit, regenerating field rather than an implicit constant.
- Explicit acknowledgement that *stability is a parameter-dependent outcome*, not a design goal;
  the model documents that some settings kill the system.
- BehaviourSpace-style batch experiments with a seed (`random-seed (474 + run-number)`), i.e.
  reproducibility is a first-class feature of the experiment, not of the program.

**Limits.** Breeds (wolves/sheep) are hardcoded with hardcoded rules ("wolves look for sheep").
Reproduction is a fixed probability per tick, decoupled from energy, so there is no heritable
variation and no evolution — it is a population-dynamics model, not an evolutionary one. Grass
regrowth is a timer, not a resource-limited process.

**Verdict for EVOLVE.** Adopt the energy/step economy, the regenerating resource field, and the
"reproduce only when energy allows" correction (the model itself lists energy-based reproduction
as an extension). Reject fixed breeds, fixed behavioural rules, and probabilistic reproduction.

### R2 — Avida

**Concepts.** Self-replicating machine-code organisms on a lattice of cells; offspring are placed in
a neighbouring cell, displacing the occupant. Mutation happens during copy. Reward for performing
logical tasks arrives as extra CPU cycles, i.e. faster self-replication. Because the genome is
executed, the genotype→phenotype map is indirect and the system produces a *perfect fossil record*:
every birth, death, mutation and lineage can be logged exactly. The platform is configured almost
entirely from config files, and the literature built on it includes replicate-and-replay studies
("rewinding the tape of life"), extinction/recovery experiments, and evolution of mutation rates.

**Worth taking.**
- Complete audit trail as a first-class output: parentage, birth/death tick, genome diff.
- Experiment = configuration + seed, stored so the run can be replayed and compared.
- Spatial/local reproduction and resource competition as the source of ecological structure.
- Phenotype matching (not species labels) as the mechanism that defines interactions.

**Limits.** Instruction-level genomes are far too slow for 5,000 organisms with neural controllers,
and the "task reward" fitness channel is an externally supplied objective. EVOLVE must not have an
external objective function; survival and reproduction are the only currency.

**Verdict.** Adopt the audit trail, config-driven experiments, and phenotype-based interaction.
Reject the virtual-CPU genome and task-reward fitness.

### R3 — NEAT

**Concepts.** Evolve topology *and* weights. Three mechanisms: (1) historical marking / innovation
numbers so that genes in different topologies can be aligned for crossover; (2) speciation, which
protects new structural innovations from being outcompeted before they are optimised; (3)
incremental growth from a minimal network. Ablations show all three are needed for the reported
speedup.

**Worth taking now.** Not the algorithm — the *bookkeeping discipline*. If genome layout and save
format are designed with stable per-gene identifiers from day one, topology evolution can be added
later without invalidating snapshots. Crossover of differently-structured networks is the hard part;
we defer it, but we do not paint ourselves into a corner.

**Limits for EVOLVE v1.** Speciation in NEAT is a *reproductive* mechanism tied to a fitness
function evaluated per generation. EVOLVE has overlapping generations, no fitness function, and a
population controlled by an energy economy. Porting NEAT's speciation directly would mean inventing
a fitness proxy, which the brief forbids. Topology evolution is therefore M8+, and any speciation
in v1 is *descriptive statistics* (clustering), never reproductive isolation.

**Verdict.** Defer NEAT. Keep gene IDs and save format forward-compatible.

### R4 — Evolving digital ecological networks (Fortuna et al.)

**Concepts.** Digital organisms that perform logic tasks form interaction networks by phenotype
matching: host–parasite links where task sets overlap, mutualism where tasks overlap for mutual
benefit, and predator–prey links where *behaviour* (an evolved attack instruction) is the mechanism.
Key claims relevant to us: predator–prey experiments are typically run in **spatially structured**
populations because the interaction depends on movement and information; network structure is a
joint product of ecology (abundance) and coevolution (specificity); historical contingency matters,
so replicate runs with different seeds are scientifically necessary; digital systems give a perfect
fossil record and let you *manipulate* interaction types.

**Worth taking.**
- Spatial structure is not decoration — it is what makes behavioural predator–prey evolution work.
- Log *interactions* (who ate whom, who attacked whom, who mated with whom), not just populations.
  That yields a food web and lets us report realised roles.
- Replication across seeds is part of the experimental protocol, reported with spread, not a single
  anecdote.
- "Do not translate digital results to biology without caveats" — echoed in our limitations doc.

**Limits.** Task-based phenotypes have no analogue in a continuous 2-D world with neural controllers.
We borrow the *network* view (interaction graph, realised roles) but not the task machinery.

**Verdict.** Adopt spatial structure, interaction logging, seed replication, and the food-web view.

### R5 — Gras et al.: evolving predator–prey with fuzzy cognitive maps

**Concepts.** Each agent's behaviour is a fuzzy cognitive map (FCM): a small recurrent graph of
concepts (distance to predator, distance to prey, distance to food, distance to mate, energy, fear,
hunger, curiosity, …) with evolvable causal edge weights. Agents choose among evade, eat, breed.
Each individual's FCM is unique and products of evolution; **species emerge** by clustering the
evolving population rather than being declared up front. The reported results include
macroevolutionary patterns and behaviour–speciation links.

**Worth taking.**
- Nearly the same sensor/action vocabulary we need (this is the closest prior art to EVOLVE).
- Evolvable per-individual controller, inherited, mutated — confirms the architecture.
- Ecological roles and "species" as *derived clusters*, not hardcoded types.
- Internal state variables (fear/hunger) are legitimate network inputs; they are computed from
  physical state, not scripted decisions.

**Limits.** FCMs are recurrent and interpretable but expensive to evaluate per individual in Python;
the published model runs at far smaller population sizes than our 5,000-organism target. Their
"species" are statistical clusters, not reproductively isolated units — an important honesty point
we will repeat in our own docs.

**Verdict.** Adopt the sensor/action framing, derived roles via clustering, and internal-state
inputs. Use a batched feed-forward MLP instead of FCM for v1 (throughput); note recurrent/FCM
controllers as a deferred alternative (M8).

---

## 2. Algorithms and mechanisms evaluated

| Mechanism | Decision | Reason |
|---|---|---|
| Energy as universal currency; no external fitness | **Use** | Required for natural selection without a designer objective. |
| Regenerating resource field (patch model) | **Use**, but continuous biomass with logistic growth | Timer-based regrowth (R1) cannot produce resource depletion/competition. |
| Hardcoded species/breeds | **Reject** | Roles must emerge from traits and realised diet. |
| Per-individual neural controller, inherited weights | **Use** | Core requirement; supported by R3/R5. |
| Fixed topology + weight/bias mutation + crossover | **Use (v1)** | Cheap to batch; adequate for the acceptance criteria. |
| NEAT topology growth, innovation numbers, speciation | **Defer (M8)** | Needs structural crossover; must not fake a fitness function. |
| FCM / recurrent controller (R5) | **Defer** | Better interpretability and memory, ~order-of-magnitude slower per individual. |
| Spatial uniform grid for neighbour queries | **Use** | Required; O(pairs) instead of O(N²). |
| Staged (read → propose → resolve → apply) tick | **Use** | Removes processing-order advantage; enables deterministic conflict rules. |
| Perfect lineage/fossil record (R2) | **Use** | Cheap at these scales, high scientific value. |
| Interaction graph / realised food web (R4) | **Use** | Makes predator–prey claims checkable. |
| Phenotype clustering as descriptive "species" | **Use (statistics only)** | Honest alternative to declaring species. |
| Lotka–Volterra comparison of predator/prey series | **Use (analysis only)** | Diagnostic, not a target the sim is tuned to hit. |
| Replicate runs over seeds (R4) | **Use** | Single-run anecdotes are not evidence. |
| Task-reward fitness (R2), fixed reproduction probability (R1) | **Reject** | External objective / non-heritable reproduction. |

---

## 3. Design consequences worth stating explicitly

1. **Sparsity of the default world.** 300 organisms over 2000×2000 gives a mean nearest-neighbour
   spacing of ≈58 units (density 7.5×10⁻⁵ /unit²). 1,500 plants give ≈26 units. With a 120-unit
   vision radius an organism sees ~4–5 plants and ~1 conspecific on average. Consequence: foraging
   is learnable, *mate finding is not* — so asexual reproduction is the default, sexual reproduction
   is a configurable mode tested in its own experiment, and plant biomass must be spatially clumped
   rather than uniform, or foraging gradients are too weak for selection to act on.
2. **Random initial brains are the honest starting point.** Any hand-set bias toward food or away
   from threats is a behavioural rule in disguise. Initial weights are therefore small zero-mean
   Gaussian noise, giving a random walk. A meaningful fraction of the founding population is
   expected to die without reproducing; this is a reported outcome, not a bug to paper over.
3. **`escape` must not steer.** If the escape output rotated the organism away from a threat, the
   simulator would be doing the deciding. Escape may only scale sprint speed (and its metabolic
   cost); the network has to learn to turn.
4. **Roles are measurements.** "Herbivore/carnivore/omnivore" are computed from lifetime intake
   ratios, and reported alongside a caveat that they are behavioural labels, not taxa.
5. **`mutation_rate` is itself a gene** (with a configurable floor/ceiling) so mutation-rate
   evolution is observable; it has no direct fitness term — it pays off only through its effects.
6. **No biological-realism claims.** The model is a caricature: no development, no learning within
   a lifetime, no disease, no spatial genetics beyond dispersal, and a feed-forward controller with
   no memory. Documents will say so.

---

## 4. Deferred to future versions

- NEAT-style topology evolution, innovation tracking, structural crossover, speciation (M8).
- Recurrent controllers / FCMs for memory and internal state (M8).
- Parasitism, mutualism, cooperation, kin recognition (needs interaction machinery beyond v1).
- Explicit genetics of recombination (linkage, chromosomes, diploidy) — v1 is haploid, one locus
  per trait.
- Terrain height, water bodies, migration barriers, metapopulation structure.
- GPU/multiprocessing execution; v1 is single-process NumPy.
- Comparative phylogenetics (tree reconstruction, ancestral state inference).
