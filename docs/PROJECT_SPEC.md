# PROJECT EVOLVE — Artificial Life, Evolution & Ecosystem Simulation

> Stored copy of the client brief. This is the authoritative requirements document for the
> build. It will be moved to `docs/PROJECT_SPEC.md` in the repository once implementation
> begins.

You are a senior scientific software engineer, computational biologist, evolutionary
algorithm researcher, and simulation architect.

Your mission is to design and implement EVOLVE: a scientifically inspired artificial-life
simulator where digital organisms live, interact, reproduce, mutate, compete, adapt, and
evolve without direct human intervention.

This is NOT a conventional survival game.

The primary goal is to create an emergent artificial ecosystem in which complex behaviors
arise from genetic variation, environmental pressure, natural selection, and interactions
between organisms.

The simulation must be deterministic when given the same random seed, scientifically
interpretable, modular, testable, and capable of running thousands of organisms.

Graphics are secondary. Prioritize correctness, simulation quality, evolutionary mechanics,
performance, and scientific observability.

## 1. RESEARCH BEFORE IMPLEMENTATION

Before writing code, investigate the following projects and scientific concepts.

Required references
- NetLogo Wolf Sheep Predation: https://ccl.northwestern.edu/netlogo/models/WolfSheepPredation
- Avida Digital Evolution: https://alife.org/encyclopedia/digital-evolution/avida/
- NEAT: NeuroEvolution of Augmenting Topologies: https://pubmed.ncbi.nlm.nih.gov/12180173/
- Evolving Digital Ecological Networks: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002928
- Evolving Predator-Prey Behavior: https://pubmed.ncbi.nlm.nih.gov/19463060/

Study their concepts and extract useful architectural ideas.

Do not copy an entire existing implementation.

Create a research_notes.md documenting:
- Relevant concepts.
- Algorithms worth implementing.
- Advantages and limitations.
- Which approaches will be used in EVOLVE.
- Which approaches are deferred to future versions.

Do not claim biological realism beyond what the mathematical model supports.

## 2. TECHNOLOGY STACK

Use: Python 3.12+, NumPy, Pygame, Matplotlib, SQLite, PyYAML, Pytest.

Keep the simulation engine independent of the visualization layer.

The simulation must be capable of running in headless mode without opening a graphical window.

Use clear type hints, dataclasses where appropriate, modular components, and comprehensive
documentation.

Avoid unnecessary frameworks and dependencies.

## 3. WORLD SIMULATION

Create a two-dimensional ecosystem.

Default world configuration:
- Width: 2000 world units.
- Height: 2000 world units.
- Initial organisms: 300.
- Initial plants: 1500.
- Maximum organism population: 5000.
- Initial simulation seed: configurable.
- World boundaries: configurable.
- Time model: discrete simulation ticks.

The world should contain:
- Plants.
- Herbivores.
- Carnivores.
- Omnivores.
- Environmental resources.
- Terrain and environmental conditions.
- Dead organisms and decomposing biomass.

Do not require all animal roles to be hardcoded species.

Organisms should have genetically encoded characteristics that influence their feeding
capabilities and behavior.

Ecological roles should emerge from their traits and interactions wherever practical.

Implement a spatial partitioning system such as a uniform spatial grid or spatial hash.

Avoid checking every organism against every other organism on every simulation tick.

## 4. ORGANISM MODEL

Every organism must have:
- Unique ID.
- Genome.
- Position.
- Velocity.
- Direction.
- Energy.
- Health.
- Age.
- Generation.
- Parent IDs.
- Reproduction state.
- Internal neural network state if needed.

Physical genetics

Implement genes for:
- Body size.
- Movement speed.
- Vision range.
- Vision angle.
- Metabolic rate.
- Attack strength.
- Defense.
- Reproduction threshold.
- Reproduction investment.
- Diet preference.
- Maximum lifespan.
- Mutation rate.

Each trait must have: a defined range, an inheritance mechanism, a mutation mechanism, an
observable phenotype, and a biological or ecological cost where appropriate.

Avoid unlimited positive traits.

Implement realistic trade-offs, e.g.:
- Large bodies consume more energy.
- High speed increases movement costs.
- High attack strength requires additional energy investment.
- High reproduction investment reduces energy available for survival.

## 5. EVOLVABLE INTELLIGENCE

THIS IS THE MOST IMPORTANT REQUIREMENT.

Do not implement creatures using manually programmed behavioral rules.

Forbidden as direct behavioral decision rules:
- If food is nearby, move toward food.
- If a predator is nearby, run away.
- If energy is low, search for food.
- If prey is nearby, attack.

Instead, implement an artificial neural network for each organism.

Initial neural architecture: feed-forward, 16 inputs, 12 first hidden-layer neurons,
8 second hidden-layer neurons, 8 outputs. Weights and biases encoded in the genome. tanh or
sigmoid activations as appropriate.

Neural inputs (numeric sensor information):
- Food distance.
- Food direction.
- Predator distance.
- Predator direction.
- Prey distance.
- Prey direction.
- Potential mate distance.
- Energy level.
- Health level.
- Age.
- Temperature.
- Moisture.
- Nearby plant density.
- Nearby organism density.
- Recent threat signal.
- Reproduction drive.

Normalize all inputs consistently. Sensors report information only; they must not decide
what an organism should do.

Neural outputs control:
- Forward movement.
- Turning direction.
- Movement intensity.
- Eating attempt.
- Attack attempt.
- Escape movement intensity.
- Reproduction attempt.
- Resting intensity.

Translate continuous network outputs into valid physical actions. An organism should not have
access to information outside its sensory range.

Evolution of intelligence: fixed topology with evolvable weights and biases in v1. Implement
weight mutation, bias mutation, genetic crossover, parent genome inheritance. Later (M8):
NEAT-style node addition, connection addition, connection disabling, innovation tracking,
speciation, crossover between different structures.

## 6. NATURAL SELECTION AND REPRODUCTION

Reproduction requires sufficient energy, satisfied reproduction conditions, and population
limits permitting.

Two reproduction modes:
A. Asexual: one parent, genome copying, random mutation.
B. Sexual: two compatible parents, crossover, mutation, energy investment from both.

Make reproduction mode configurable.

Offspring inherit genomes, not lifetime experience or learned memories.

Implement point mutations, trait mutations, neural weight mutations, crossover, mutation rate
configuration, optional adaptive mutation rates.

Mutations must not automatically improve organisms. Most mutations neutral or harmful, some
beneficial in a particular environment. Natural selection emerges from differential survival
and reproduction. Never manually increase fitness for behavior that looks intelligent.

## 7. FOOD CHAIN AND ENERGY SYSTEM

Energy-based ecological system.

Plants grow over time, consume environmental resources, produce biomass, compete for space and
resources, can be consumed by herbivores and omnivores.

Herbivores consume plants and convert a fraction of consumed biomass into energy.

Carnivores attempt to hunt organisms within their physical capabilities. Predation depends on
distance, movement, relative speed, attack strength, defense, body size, energy cost.

Omnivores consume different resource categories according to physical traits and evolved
feeding behavior.

Every energy transfer must be accounted for. Energy consumed through basal metabolism,
movement, neural processing if modeled, reproduction, physical interactions. Gained through
feeding, consumption of prey, environmental biomass production.

On death, remaining biomass becomes a carcass; scavengers may consume it; decomposition
returns nutrients to the environment.

Documented energy and biomass accounting model. Avoid unexplained energy creation or
destruction.

## 8. ENVIRONMENTAL SIMULATION

Environment independent of organisms. Required: temperature, moisture, seasons, rainfall,
drought, plant regeneration, resource availability, environmental disasters.

Changes over simulation time: long-term drought, sudden temperature changes, flooding,
resource depletion, changes in vegetation distribution.

Events influence survival and reproduction through physical consequences. Never directly
modify organism genes. The environment creates selection pressure, not evolution.

Support configurable event schedules and random events.

## 9. EVOLUTIONARY STATISTICS

Population statistics: total population, population by ecological role, births, deaths,
extinctions, average lifespan, average energy, population growth rate.

Genetic statistics: average body size, speed, vision range, metabolism, attack, defense,
genetic diversity, mutation frequency, genome similarity, trait distribution.

Behavioral statistics: average movement distance, feeding frequency, hunting success, escape
success, reproduction success, average time resting, resource acquisition efficiency.

Evolutionary history: generation number, ancestry, parent IDs, genome changes, population
lineages, extinction events, major environmental changes.

Do not assume synchronized generations; use parentage and generation depth.

Plots: population over time; population by ecological role; average trait values across
generations; genetic diversity; mutation frequency; birth and death rates; predator-prey
population relationships.

Export statistics as CSV.

## 10. VISUALIZATION

Simple, functional Pygame 2D visualization. Green plants; different colors for organisms or
lineages; organism size reflects physical size; optional direction indicators, vision-radius
overlay, sensory/interaction visualization.

Controls: start, pause, reset, single-step, speed controls, time acceleration, random seed
selection, save, load.

Inspection panel on click: ID, age, energy, health, generation, genome traits, parent IDs,
neural network information, recent behavioral statistics.

Statistics panel: live charts for population, trait evolution, genetic diversity, births and
deaths, resource levels.

Interface must remain responsive; separate rendering from simulation updates.

## 11. PERSISTENCE AND EXPERIMENTS

SQLite-based experiment storage. Users must be able to create an experiment, configure initial
conditions, start a simulation, save state, resume, compare experiments, export statistics,
reproduce results with the same seed.

Store sufficient information to reproduce and inspect an experiment. Do not write the entire
population to SQLite every tick. Use periodic snapshots and batched statistics persistence.
Implement versioned save formats.

## 12. SIMULATION ENGINE DESIGN

Clear tick pipeline. Suggested order:
1. Update environmental conditions.
2. Regenerate plants and resources.
3. Build or update spatial indexes.
4. Collect organism sensory information.
5. Evaluate neural networks.
6. Resolve movement.
7. Resolve feeding and predation.
8. Apply energy consumption and transfer.
9. Process reproduction.
10. Process aging and deaths.
11. Process decomposition.
12. Update statistics.
13. Periodically save snapshots.

Deterministic update ordering; no advantage to organisms processed earlier. Staged updates:
read current state, generate proposed actions, resolve interactions, apply state changes.
Define how simultaneous conflicts are resolved. Seeded RNG; no uncontrolled randomness.

## 13. PERFORMANCE

Targets: 500 organisms smooth interactive; 1,000 practical; 5,000 headless.

Implement spatial partitioning. Profile actual bottlenecks. Do not prematurely introduce
multiprocessing or GPU computing. Make headless accelerated runs possible. Provide performance
benchmarks.

## 14. TESTING

Genetics: mutation modifies genomes; crossover combines parental information; inherited traits
stay in bounds; offspring have correct parent references; mutation rates behave as configured.

Neural network: valid outputs; identical weights and inputs produce identical outputs;
different weights can produce different actions; weights are inherited; mutation can modify
behavior.

Ecosystem: plants regenerate; valid feeding gains energy; metabolism costs energy; predation
transfers biomass correctly; death on failed survival conditions; decomposition returns
nutrients.

Evolution: reproduction creates offspring; mutation introduces heritable variation;
differential survival can change trait frequencies; diversity is measurable; environmental
change influences survival.

Determinism: identical seed + configuration + version produces identical results, subject to
documented platform-specific numerical limitations.

Integration: thousands of ticks; population growth and decline; resource depletion;
predator-prey interactions; drought; recovery after disruption; extinction scenarios. Do not
assert every simulation reaches equilibrium; extinction and instability are valid outcomes.

## 15. EXPERIMENTAL SCENARIOS

At least six predefined experiments:
- A: Stable Ecosystem — moderate plant regeneration, balanced population, stable conditions.
- B: Predator Pressure — additional predators; observe prey population and trait changes.
- C: Severe Drought — reduced moisture and plant regeneration; observe survival and adaptation.
- D: Resource Competition — limited resources; observe feeding strategy differences.
- E: Environmental Shift — change temperature and resource distribution halfway; observe
  adaptation, migration, or extinction.
- F: Evolution Without Predators — remove predation pressure; observe defense, movement, and
  energy allocation.

Provide scripts to run these experiments automatically.

## 16. DEVELOPMENT MILESTONES

M1 Basic simulation: world grid, plants, organisms, movement, energy, basic feeding, death,
simple 2D visualization.
M2 Genetics: genome representation, trait inheritance, mutation, reproduction, parentage.
M3 Evolving intelligence: neural network, sensors, action outputs, evolvable weights,
behavioral inspection.
M4 Ecosystem: predation, multiple diets, competition, decomposition, environmental resources.
M5 Evolution analysis: population statistics, genetic diversity, trait distributions, ancestry,
experiment comparison.
M6 Environmental adaptation: climate, seasons, drought, resource fluctuations, disasters.
M7 Persistence and performance: SQLite, save/load, deterministic experiments, spatial
optimization, headless execution, profiling.
M8 Advanced evolution: NEAT-style topology evolution, speciation, sexual reproduction
improvements, additional experiments.

Do not implement the entire project in one attempt.

## 17. ACCEPTANCE CRITERIA

The project is not complete merely because the interface looks good. It must demonstrate:
- Organisms are generated from genomes.
- Organisms have individual neural networks.
- Their neural networks control their actions.
- Their neural weights are inherited.
- Mutations can change physical traits and neural behavior.
- Organisms reproduce and die.
- Energy and resources are conserved according to the documented model.
- Predation and resource competition work.
- Environmental changes influence survival.
- Population and genetic statistics are collected.
- Simulations can be saved and resumed.
- Headless simulations work.
- Experiments can be reproduced with fixed seeds.
- The test suite passes.
- Performance benchmarks are documented.

Additionally, run experiments demonstrating that evolved behavior can outperform random
behavior under specific controlled conditions. Do not claim that intelligence, specialization,
or a particular species will necessarily evolve. Report observed results honestly.

## 18. DOCUMENTATION

Create: README.md, docs/ARCHITECTURE.md, docs/GENETICS.md, docs/NEURAL_NETWORKS.md,
docs/ECOSYSTEM.md, docs/ENERGY_MODEL.md, docs/EVOLUTION.md, docs/EXPERIMENTS.md,
docs/SCIENTIFIC_LIMITATIONS.md.

Explain mathematical models, algorithms, assumptions, limitations. Include diagrams where
useful.

## 19. FINAL INSTRUCTIONS

Work autonomously. Start by inspecting the environment and studying the references. Then
produce: research notes, architecture proposal, detailed implementation plan, initial
configuration, first milestone implementation.

After every milestone: run automated tests, run the application, fix discovered bugs, update
documentation, report what works and what remains incomplete.

Do not create placeholder implementations and call them finished. Do not simulate evolution by
manually changing traits according to generation number. Do not hardcode intelligent
behaviors. Do not force populations to survive. Do not hide failures.

The central objective is a functioning artificial ecosystem in which evolutionary dynamics
emerge from the interaction between inherited variation, environmental constraints, and
differential reproduction. Build a real scientific simulation, not just an attractive
animation.
