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

## Using the viewer

`viz` opens a resizable window: the world on the left, a live information
panel on the right, a control strip along the bottom.

* **What you see.** Green specks are plants, brown lumps are carcasses and
  dots are organisms. Organism colour is the **inherited `diet_preference`
  gene** (green = plant-leaning, yellow = mixed, red = meat-leaning); it is a
  gene, not an observed behaviour. The panel also reports roles **measured
  from lifetime intake**, which can differ from the gene.
* **Background layers.** `v` cycles climate → moisture → soil → temperature →
  terrain. The badge in the top-left corner names the layer and spells out what
  its colours mean (for example "moisture — brown is dry, blue is wet").
* **First launch opens a guide.** `viz` starts with a modal five-minute guide
  (also on `h`) that explains what the dots, colours and numbers mean. While it
  is open only close (`h`/`esc`) and scroll act, so reading it cannot pause,
  step or reset the run. The window is resizable; the text reflows and the
  footer drops its legend line before it would ever run off the edge.
* **Click to inspect.** Click any organism (a ring plus a heading arrow marks
  it) to switch the panel to that individual: genes, reserve/health, parents,
  lifetime intake and actions, then the exact 16 sensor values the network saw
  and the 8 outputs it produced on its last think step. Sensors read `—` when
  nothing was detected (the encoding is −1). `f` follows the selected animal,
  `x` clears it. If it dies, a note says so and the panel returns to the
  population view rather than silently showing a different animal.
* **Reading the outputs.** Each row shows both the raw score and the decision:
  a score above 0.5 means yes. `resulting speed` is what `actions.decode`
  actually produced, so a resting organism shows no movement; `rest` also
  blocks eating and attacking, and those rows say so.
* **Statistics.** Population, births/deaths/mutations, role counts, the
  conserved matter pools (plant biomass, carcasses, detritus, organism tissue,
  soil nutrients) **plus the total and the value it started from**, averages,
  the season phase and any active or scheduled environmental event. The total
  is the conservation claim made checkable: if it ever drifted, you would see
  it there.
* **Scrolling.** When the panel is taller than the window it fades at the edge,
  shows a scrollbar and says "more below"; the wheel and page up/down scroll it.
* **Extinction** is a normal outcome: the screen says `EXTINCT`, the ledger
  still balances, and `r` restarts the same seed.

| Key | Action |
|-----|--------|
| `space` | pause / resume |
| `s` | advance one tick |
| `1` `2` `3` `4` | speed 1× / 2× / 5× / 20× |
| arrows | pan · `+` / `-` zoom · `home` fit world |
| click | select · `x` clear · `f` follow |
| `v` | cycle environment layer |
| `i` | show/hide the side panel |
| `h` | full help overlay (modal; `esc` closes it) |
| mouse wheel / `pgup` `pgdn` | scroll the panel or the help overlay |
| `r` | restart from the seed |
| `q` / `esc` | quit |

The viewer is verified headlessly (`SDL_VIDEODRIVER=dummy` + screenshots) on a
single-core machine; it has not been run on a desktop GPU or at 60 FPS under a
large population, so treat frame-rate claims as unmeasured.

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

Milestones M1–M7 are implemented and tested (192 tests). Measured throughput on
a single-core machine misses the design's §11 targets — see
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the honest numbers. NEAT-style
speciation (M8) is deferred.
