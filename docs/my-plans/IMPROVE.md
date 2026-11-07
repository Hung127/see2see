# EVOLVE — Scientific Simulation, AI Agent Testing & Visualization Upgrade

## Mission

You are the lead simulation engineer, computational scientist, performance engineer, and UI/UX designer responsible for improving EVOLVE.

Your task is to evolve the existing project into a polished, scientifically credible artificial-life simulation and lightweight autonomous-agent testing environment.

Do not rebuild the project from scratch.

Preserve the existing architecture, scientific foundations, deterministic behavior, conservation guarantees, and working functionality unless a demonstrably better implementation is justified.

You have access to the existing repository. Inspect the actual source code before making decisions. Treat the source implementation and tests as authoritative; the project report is a description, not a substitute for code inspection.

Your job is to IMPLEMENT improvements, not merely recommend them.

---

# 1. HARDWARE CONSTRAINTS — NON-NEGOTIABLE

The application runs inside a severely resource-constrained Docker sandbox:

- CPU: 1 vCPU
- RAM: 1 GB
- No GPU
- No multiprocessing
- No distributed computing
- No external LLM services
- No heavyweight AI inference
- No large model downloads

Design for this environment first.

### Resource budget

Target:

- Application steady-state RSS: preferably below 650 MB.
- Leave sufficient memory for the operating system and other processes.
- Avoid unbounded memory growth.
- Avoid storing complete simulation histories in RAM.
- Avoid unnecessary copies of large NumPy arrays.
- Keep rendering and simulation workloads controlled.

These are engineering targets, not excuses to compromise correctness.

### Forbidden approaches

Do not introduce:

- LangChain, LangGraph, CrewAI or similar agent frameworks.
- LLM-based organism decision-making.
- Cloud AI APIs.
- Transformers or large pretrained models.
- PyTorch/TensorFlow merely to execute the existing small MLP.
- Redis, Kafka, Elasticsearch or unnecessary infrastructure.
- Kubernetes or distributed simulation infrastructure.
- Multiple databases when SQLite already satisfies the requirements.
- Excessive dependencies.

Prefer the existing Python, NumPy, SQLite, Pygame and Matplotlib stack.

Only introduce dependencies when they solve a demonstrated problem.

---

# 2. FIRST PHASE — REPOSITORY AUDIT

Before changing anything, inspect:

1. Project structure.
2. docs/DESIGN.md.
3. Configuration and simulation defaults.
4. Engine tick pipeline.
5. Sensing implementation.
6. Spatial indexing.
7. Neural controller.
8. Genetics and reproduction.
9. Energy and conservation accounting.
10. Experiment definitions.
11. Statistics and persistence.
12. Pygame visualizer.
13. Existing tests and benchmark scripts.

Understand the actual data flow.

Identify:

- Existing architectural strengths.
- Technical debt.
- Scientific limitations.
- Performance bottlenecks.
- Memory allocation hotspots.
- Missing experiments.
- Missing visualization capabilities.
- Inconsistent or misleading metrics.
- Features described in documentation but not implemented.

Produce a concise audit document.

Do not immediately rewrite modules or introduce new architecture.

---

# 3. ENGINE IMPROVEMENTS

## 3.1 Performance optimization

The current report identifies sensing as approximately 65% of tick execution.

Investigate this first.

Profile:

- Sensor calculations.
- Distance and bearing calculations.
- Spatial pair expansion.
- Neighbor aggregation.
- Neural network forward passes.
- Movement.
- Feeding.
- Predation.
- Reproduction.
- Statistics.
- Persistence.

Use actual measurements rather than assumptions.

Potential improvements:

- Reuse NumPy scratch buffers.
- Avoid repeated array allocations.
- Avoid redundant distance calculations.
- Reduce unnecessary temporary arrays.
- Improve spatial candidate filtering.
- Avoid repeated sorting when ordering is already guaranteed.
- Improve memory locality.
- Reduce Python-level loops where vectorization is appropriate.
- Cache values only when their invalidation rules are clear.

Do not blindly vectorize everything. A large temporary matrix can consume more memory than a carefully designed loop.

Use cProfile, tracemalloc and existing benchmark infrastructure where appropriate.

Establish before/after measurements.

## 3.2 Preserve deterministic simulation

Determinism is a core feature.

Any optimization must preserve:

- RNG draw ordering.
- Stable organism IDs.
- Stable tie-breaking.
- Deterministic sorting.
- Reproducible snapshots.
- Resume equivalence.
- Conservation invariants.

If a scientific rule or RNG sequence changes intentionally:

1. Document the change.
2. Update the specification version where necessary.
3. Add regression tests.
4. Explain snapshot compatibility implications.

Never silently change simulation semantics for performance.

## 3.3 Improve ecological analysis

Introduce additional measurable statistics where they can be computed efficiently.

Consider:

### Population dynamics
- Birth rate.
- Death rate.
- Population growth rate.
- Extinction events.
- Survival duration.
- Population stability.
- Generation turnover.

### Evolution
- Trait distributions.
- Trait variance.
- Genetic diversity.
- Neural parameter diversity.
- Mutation statistics.
- Lineage depth.
- Generation-wise trait changes.

### Ecology
- Plant biomass.
- Organism biomass.
- Carcass biomass.
- Resource consumption.
- Predator/prey relationships.
- Trophic distribution.
- Energy efficiency.
- Environmental stress.
- Population density.

### Behavioral analysis
- Movement distance.
- Rest frequency.
- Feeding frequency.
- Attack frequency.
- Reproduction attempts.
- Successful reproduction.
- Behavioral diversity.

Avoid collecting enormous per-organism histories.

Prefer online aggregation, periodic sampling, and compact records.

Clearly distinguish observed metrics from inferred metrics.

## 3.4 Improve experiment infrastructure

Build on experiments A-H instead of replacing them.

Consider adding:

- Replicated seed experiments.
- Population stability experiments.
- Mutation-rate sensitivity experiments.
- Vision-range sensitivity experiments.
- Metabolic trade-off experiments.
- Environmental resilience experiments.
- Controller comparison experiments.
- Long-term evolutionary experiments.

Each experiment must specify:

- Hypothesis.
- Independent variables.
- Controlled variables.
- Dependent variables.
- Seeds.
- Duration.
- Evaluation metrics.
- Expected interpretation.
- Limitations.

Support confidence intervals and appropriate statistical comparisons when sample sizes justify them.

Never present a single successful seed as proof of general evolutionary success.

Do not modify calibration parameters just to make experiments look successful.

---

# 4. AI AGENT TESTING ENVIRONMENT

This is an important strategic improvement.

EVOLVE should support testing multiple autonomous controllers without requiring LLMs.

## 4.1 Controller abstraction

Investigate a lightweight controller interface.

Conceptually:

Observation -> Controller -> Action

All controllers must use the same observation and action contracts.

Preserve the existing MLP as the default implementation.

Potential controller types:

1. Existing evolved neural controller.
2. Random baseline.
3. Passive baseline.
4. Rule-based controller.
5. Heuristic foraging controller.
6. External custom controller.

Do not let baseline controllers access information unavailable to other controllers unless the experiment explicitly defines a different information condition.

Do not introduce arbitrary external code execution inside the simulation process.

## 4.2 Evaluation framework

Compare controllers using:

- Survival rate.
- Average lifespan.
- Reproduction success.
- Energy acquisition.
- Energy efficiency.
- Movement efficiency.
- Feeding success.
- Predation success.
- Behavioral diversity.
- Environmental robustness.

Evaluate across multiple scenarios and independent seeds.

Use paired seeds when comparing controllers under identical initial conditions.

Separate training/evolution and evaluation conditions where applicable.

Avoid hidden fitness rewards that undermine EVOLVE's survival-and-reproduction-based evolutionary model.

## 4.3 Controller inspection

Provide tools to inspect:

- Controller type.
- Genome.
- Physical traits.
- Neural outputs.
- Actions taken.
- Lifetime statistics.

Do not render all neural weights continuously.

Provide on-demand inspection instead.

---

# 5. EVOLUTIONARY IMPROVEMENTS

Explore improvements without destabilizing the existing model.

Potential research directions:

### Short-term
- Better genetic diversity metrics.
- Lineage analysis.
- Mutation-rate analysis.
- Reproductive success tracking.
- Evolutionary bottleneck detection.

### Medium-term
- Optional recurrent neural controllers.
- Controller complexity analysis.
- Behavioral specialization analysis.
- More informative genetic visualization.

### Long-term experimental milestone
- NEAT-style topology evolution.
- Innovation numbering.
- Structural crossover.
- Speciation.
- Species-level statistics.

NEAT must remain an optional experimental controller architecture.

Do not replace the fixed 16 -> 12 -> 8 -> 8 MLP before implementing a valid comparison experiment.

Do not claim that genetic diversity, behavioral diversity, and species diversity are interchangeable.

---

# 6. VISUALIZATION REDESIGN

The current visualizer should become a scientific artificial-life observatory.

Design direction:

**Scientific Observatory × Artificial Life × Modern Scientific Visualization**

Avoid:
- Generic SaaS dashboards.
- AI-generated-looking interfaces.
- Excessive gradients.
- Excessive rounded cards.
- Neon cyberpunk aesthetics.
- Unnecessary decorative elements.
- Overloaded HUDs.
- Unreadable tiny charts.

Aim for an established scientific research tool.

## 6.1 Main simulation view

The ecosystem should be the visual centerpiece.

Improve:

- Organism rendering.
- Plant representation.
- Terrain appearance.
- Environmental field visualization.
- Organism movement.
- Predator/prey visual distinction.
- Visual hierarchy.
- Zoom and pan.
- Camera controls.
- Selection and inspection.

Organisms should have meaningful visual representations.

Potential visual encoding:

- Body size -> organism size.
- Diet preference -> color.
- Health -> visual condition.
- Energy -> optional secondary indicator.
- Selected organism -> clear highlight.
- Threats -> optional perception overlay.

Keep visual encoding consistent and documented.

Avoid rendering every visual effect at all times.

## 6.2 Organism inspector

Allow users to select an organism and inspect:

- Organism ID.
- Age.
- Health.
- Energy.
- Body mass.
- Physical traits.
- Diet preference.
- Generation.
- Parent IDs.
- Controller type.
- Recent behavioral statistics.
- Reproductive history.
- Death status, where historical data exists.

Use a clean inspector panel rather than placing text directly over the simulation.

## 6.3 Scientific overlays

Provide optional overlays:

- Plant density.
- Organism density.
- Soil nutrients.
- Temperature.
- Moisture.
- Threat pheromone.
- Vision range.
- Perception targets.

Use appropriate sequential or diverging color maps.

Include legends and units.

Do not confuse missing data with zero values.

## 6.4 Analytics interface

Create a dedicated analytics view or organized side panels.

Important charts:

1. Population over time.
2. Births and deaths.
3. Biomass distribution.
4. Energy distribution.
5. Trait evolution.
6. Genetic diversity.
7. Plant and organism biomass.
8. Environmental conditions.
9. Predation statistics.
10. Survival curves.

Use meaningful axes, units, legends and readable typography.

Avoid misleading scales and decorative charts.

Charts should update at controlled intervals rather than requiring expensive full-history redraws every simulation tick.

## 6.5 Lineage visualization

Introduce an interactive ancestry visualization.

Show:

- Parent-child relationships.
- Generation depth.
- Lineage survival.
- Trait changes.
- Branching patterns.

Do not call lineages species.

Do not fabricate ancestry for missing historical data.

## 6.6 Simulation controls

Provide clear controls for:

- Start.
- Pause.
- Resume.
- Step one tick.
- Speed adjustment.
- Camera zoom.
- Camera pan.
- Organism selection.
- Overlay selection.
- Statistics visibility.

Display:

- Current tick.
- Population.
- Plant count.
- Simulation speed.
- Selected organism.
- Current experiment.

Keep simulation execution independent of rendering.

The simulation must continue to work headlessly.

---

# 7. VISUAL DESIGN SYSTEM

Use a consistent design language.

Suggested direction:

- Background: charcoal / near-black.
- Main surface: slightly lighter charcoal.
- Text: warm white.
- Secondary text: muted gray.
- Plant colors: natural greens.
- Herbivore colors: restrained yellow/green.
- Predator colors: restrained orange/red.
- Environmental stress: appropriate amber/red.
- Neutral information: desaturated blue.

These are semantic suggestions, not mandatory fixed colors.

Typography:

- Clear hierarchy.
- Readable labels.
- Consistent numerical formatting.
- Good spacing.
- Minimal unnecessary decoration.

Use restrained borders and consistent alignment.

Prioritize legibility and scientific clarity.

The interface should feel like a real scientific instrument.

---

# 8. PERFORMANCE-AWARE RENDERING

Rendering must not compromise simulation performance.

Implement where appropriate:

- Separate simulation and rendering update rates.
- Controlled statistics refresh intervals.
- Camera-based culling.
- Efficient sprite or primitive rendering.
- Level of detail for dense populations.
- Cached static terrain.
- Reusable drawing resources.
- Limited chart history.
- On-demand detailed inspection.

Do not render every neural network or every organism's complete statistics continuously.

Measure rendering performance independently from engine performance.

Ensure that disabling the visualizer introduces no simulation dependency.

---

# 9. SCIENTIFIC VALIDATION

Scientific correctness takes priority over attractive results.

Follow established agent-based modeling practices.

Use the ODD protocol:

- Overview.
- Design concepts.
- Details.

Document:

- Purpose.
- State variables.
- Spatial and temporal scales.
- Process scheduling.
- Design concepts.
- Initialization.
- Inputs.
- Submodels.

Improve verification and validation:

### Verification
Does the code implement the intended mathematical and computational rules?

### Validation
Are the model's assumptions and outputs appropriate for its intended research questions?

### Sensitivity analysis
How strongly do results depend on parameter choices?

### Reproducibility
Can another person reproduce the experiment from its configuration, seed, and version information?

Do not claim biological realism without empirical evidence.

Do not interpret emergent behavior as proof of intelligence.

Clearly label experimental findings and limitations.

---

# 10. TESTING REQUIREMENTS

Every meaningful engine change must include tests.

Preserve and extend existing acceptance tests.

Test:

- Deterministic execution.
- Snapshot/resume equivalence.
- RNG state restoration.
- Conservation.
- Feeding.
- Predation.
- Reproduction.
- Mutation.
- Controller interfaces.
- Baseline controller behavior.
- Experiment reproducibility.
- Statistics correctness.
- Headless execution.
- Visualization smoke tests.

Add performance regression measurements where appropriate.

Do not weaken existing assertions to make tests pass.

Do not remove tests simply because an optimization changes internal implementation.

---

# 11. IMPLEMENTATION STRATEGY

Work incrementally.

### Phase 1 — Audit and baseline
- Inspect the repository.
- Understand the implementation.
- Run existing tests.
- Establish performance and memory baselines.
- Document issues.

### Phase 2 — Engine optimization
- Profile the actual bottlenecks.
- Optimize sensing first.
- Reduce unnecessary allocations.
- Improve spatial calculations.
- Validate determinism and conservation.

### Phase 3 — Scientific analysis
- Add useful evolutionary and ecological metrics.
- Improve multi-seed experiment analysis.
- Add sensitivity experiments.
- Improve reporting.

### Phase 4 — Controller experimentation
- Introduce controller abstraction.
- Implement lightweight baselines.
- Build fair evaluation experiments.
- Preserve the existing MLP.

### Phase 5 — Visualization redesign
- Improve ecosystem rendering.
- Add organism inspection.
- Add overlays.
- Improve analytics.
- Add lineage visualization.
- Improve controls and visual consistency.

### Phase 6 — Integration and verification
- Run the complete test suite.
- Test headless execution.
- Test the visualizer.
- Measure CPU and memory.
- Compare benchmarks.
- Verify documentation.
- Produce a final implementation report.

You may adjust this sequence when repository dependencies require it, but preserve the separation between risky engine changes and presentation work.

---

# 12. ENGINEERING RULES

1. Follow existing architecture unless there is a measurable reason to change it.
2. Prefer small, understandable modules.
3. Avoid unnecessary abstractions.
4. Avoid duplicated logic.
5. Keep configuration backward-compatible where possible.
6. Document new configuration parameters.
7. Keep visualization separate from simulation logic.
8. Keep analytics separate from core simulation state transitions.
9. Preserve headless operation.
10. Never introduce hidden randomness.
11. Never introduce scripted organism decisions into the existing neural controller.
12. Never fake experimental results.
13. Never hide extinctions or unsuccessful experiments.
14. Never silently change conservation accounting.
15. Never optimize based solely on theoretical assumptions.

If an improvement is too expensive for the hardware, implement a lightweight alternative or document it as deferred.

---

# 13. DELIVERABLES

At completion, provide:

1. Repository audit.
2. Architecture changes.
3. Engine optimization report.
4. Before/after benchmark results.
5. Peak and steady-state memory measurements.
6. New scientific metrics.
7. New experiment descriptions.
8. Controller comparison results.
9. Visualization changes.
10. Test results.
11. Known limitations.
12. Deferred improvements.
13. Updated documentation.

For every performance claim, include actual measurements.

For every scientific claim, identify the supporting experiment and its limitations.

For every feature, distinguish fully implemented, partially implemented, and deferred.

---

# 14. SCIENTIFIC REFERENCES

Use these references as conceptual foundations. Consult the actual papers where relevant; do not blindly copy their implementations.

1. Grimm et al. (2006), A standard protocol for describing individual-based and agent-based models.
   https://doi.org/10.1016/j.ecolmodel.2006.04.023

2. Grimm et al. (2010), The ODD protocol: A review and first update.
   https://doi.org/10.1016/j.ecolmodel.2010.08.019

3. Grimm et al. (2005), Pattern-oriented modeling of agent-based complex systems.
   https://doi.org/10.1126/science.1116681

4. Stanley & Miikkulainen (2002), Evolving Neural Networks through Augmenting Topologies.
   https://doi.org/10.1162/106365602320169811

5. Thiele, Kurth & Grimm (2014), Facilitating Parameter Estimation and Sensitivity Analysis of Agent-Based Models.
   https://doi.org/10.18564/jasss.2503

6. Grimm et al. (2020), The ODD protocol: A second update to improve clarity, replication, and structural realism.
   https://doi.org/10.18564/jasss.4259

7. Principles of Effective Data Visualization.
   https://doi.org/10.1016/j.patter.2020.100141

---

# FINAL DIRECTIVE

Build a better EVOLVE, not a bigger EVOLVE.

The objective is to make it:

- More scientifically defensible.
- More computationally efficient.
- More experimentally useful.
- More visually understandable.
- More extensible for autonomous-agent research.
- More reliable on extremely limited hardware.

Prioritize measurable improvements over feature quantity.

Keep the simulation lightweight, deterministic, explainable, and scientifically honest.

Inspect first. Plan second. Implement incrementally. Test continuously. Measure everything.

Do not stop after producing a plan. Continue through implementation and verification within the available resources.
