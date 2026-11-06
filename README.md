# EVOLVE

A deterministic artificial-life ecosystem: organisms with genome-derived traits
run batched evolvable neural controllers in a closed mass/energy world with
seasons, events, plants, carrion and predation. Nothing about an organism's
behaviour is scripted — the neural network is the only thing that decides.

```bash
# install (Python >= 3.12)
pip install --user -e .[viz,report,dev]

# run a simulation headless and store it in an experiment database
PYTHONPATH=src python3 -m evolve.cli run --config configs/default.yaml --db run.db

# re-execute a stored run and compare the final checksum
PYTHONPATH=src python3 -m evolve.cli reproduce --db run.db --run-id 1

# CSV + plots for a stored run
PYTHONPATH=src python3 -m evolve.cli report --db run.db --run-id 1 --out report/

# interactive pygame application
PYTHONPATH=src python3 -m evolve.cli viz --config configs/default.yaml

# throughput measurement
PYTHONPATH=src python3 -m evolve.cli benchmark --population 1000 --ticks 50

# the DESIGN §13 experiment programme (scaled down by default)
PYTHONPATH=src python3 -m evolve.cli experiments --all --out experiments_out/run
```

Exit codes: `0` success, `2` configuration error, `3` snapshot version error.

## What is in the box

| Area | Document |
|------|----------|
| Architecture, tick pipeline, determinism contract | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Genome, traits, mutation, crossover | [`docs/GENETICS.md`](docs/GENETICS.md) |
| Sensors, network, action decoding, decision-point audit | [`docs/NEURAL_NETWORKS.md`](docs/NEURAL_NETWORKS.md) |
| Plants, soil, seasons, events, predation, decomposition | [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md) |
| Metabolic formulas and the conservation ledger | [`docs/ENERGY_MODEL.md`](docs/ENERGY_MODEL.md) |
| Selection, drift, diversity, reproduction modes | [`docs/EVOLUTION.md`](docs/EVOLUTION.md) |
| Experiments A–H and the pre-registered assay | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| Measured outcomes, including nulls and misses | [`docs/RESULTS.md`](docs/RESULTS.md) |
| What this model cannot tell you | [`docs/SCIENTIFIC_LIMITATIONS.md`](docs/SCIENTIFIC_LIMITATIONS.md) |
| Measured throughput and profile | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| How the defaults were chosen | [`docs/CALIBRATION.md`](docs/CALIBRATION.md) |
| The approved design | [`docs/DESIGN.md`](docs/DESIGN.md) |

## Guarantees

* **Mass is conserved exactly** (rtol 1e-9, with a documented 1e-9 absolute
  floor for near-zero stocks): organic matter moves between organisms, plants,
  detritus, carcasses and the inorganic soil pool, and never appears or
  vanishes. Energy has one input (light → plant growth) and one sink (heat).
* **Determinism**: one seeded `np.random.Generator(PCG64)` per simulation, a
  documented draw order, ties broken by ascending organism ID, float64
  throughout. Same seed → bit-identical futures; snapshots resume exactly.
* **No behavioural scripting**: sensors report, the network decides, and
  `actions.decode` is the single non-network decision point (threshold
  comparisons only). `escape` scales sprint speed and cost; it can never steer.
* **Extinction is a valid outcome.** Nothing forces a population to survive.

## Status

Milestones M1–M7 are implemented and tested (151 tests). Measured throughput on
a single-core machine misses the design's §11 targets — see
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the honest numbers. NEAT-style
speciation (M8) is deferred.
