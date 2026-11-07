"""Statistics aggregators (DESIGN §8).

Roles are *measured from lifetime intake*, never declared from the diet gene:
the gene is a preference, the counters are the evidence. The row schema
(:data:`COLUMNS`) is a frozen tuple — persistence and plots depend on it, so it
may only grow at the end, never reorder.

Diversity proxies (all labelled proxies; none is "species"):
* mean pairwise Euclidean distance over ≤200 deterministically strided
  individuals, computed on the 12 trait loci (network weights would swamp the
  trait signal with their dimension count);
* per-locus variance of genetic values (mean over all 392 loci);
* Shannon entropy of discretised ``diet_preference`` over 20 bins, normalised
  by ``log2(20)`` so 1.0 means maximally spread;
* count of unique genome hashes.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .genome import N_TRAITS, hash_genome, phenotype

ROLE_HERBIVORE = "herbivore"
ROLE_CARNIVORE = "carnivore"
ROLE_OMNIVORE = "omnivore"
ROLE_UNCLASSIFIED = "unclassified"

_MIN_LIFETIME_INTAKE = 1.0  # bu; below this there is not enough evidence
_RING_BUFFER = 2000
_DIVERSITY_CAP = 200
_DIET_BINS = 20


def classify_roles(plant_intake: np.ndarray, meat_intake: np.ndarray) -> list[str]:
    """Classify organisms by what they actually ate (DESIGN §8).

    ``plant > 2·meat`` → herbivore, ``meat > 2·plant`` → carnivore, otherwise
    omnivore; below 1 bu lifetime intake the label is ``unclassified``.
    """
    plant = np.asarray(plant_intake, dtype=np.float64)
    meat = np.asarray(meat_intake, dtype=np.float64)
    total = plant + meat
    labels = [ROLE_UNCLASSIFIED] * plant.size
    evidence = total >= _MIN_LIFETIME_INTAKE
    herb = evidence & (plant > 2.0 * meat)
    carn = evidence & (meat > 2.0 * plant)
    omni = evidence & ~herb & ~carn
    for i in np.flatnonzero(herb):
        labels[i] = ROLE_HERBIVORE
    for i in np.flatnonzero(carn):
        labels[i] = ROLE_CARNIVORE
    for i in np.flatnonzero(omni):
        labels[i] = ROLE_OMNIVORE
    return labels


def diversity(genomes: np.ndarray) -> dict:
    """Genetic diversity proxies over a population (DESIGN §8)."""
    genomes = np.atleast_2d(np.asarray(genomes, dtype=np.float64))
    n = genomes.shape[0]
    if n == 0:
        return {"mean_pairwise": 0.0, "mean_variance": 0.0, "diet_entropy": 0.0,
                "unique_genomes": 0, "trait_variance": 0.0,
                "nn_variance": 0.0}
    stride = max(1, (n + _DIVERSITY_CAP - 1) // _DIVERSITY_CAP)
    sample = genomes[::stride][: _DIVERSITY_CAP]
    traits = phenotype(sample)[:, :N_TRAITS]
    if sample.shape[0] > 1:
        diffs = traits[:, None, :] - traits[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(-1))
        mean_pairwise = float(dists[np.triu_indices(sample.shape[0], 1)].mean())
    else:
        mean_pairwise = 0.0
    mean_variance = float(genomes.var(axis=0).mean())
    # Separate trait-locus and network-weight variance: they answer different
    # questions ("is the phenotype pool narrowing?" vs "are the controllers
    # diverging?"). Both are variances of *genetic values*, not phenotypes.
    trait_variance = float(genomes[:, :N_TRAITS].var(axis=0).mean())
    nn_variance = float(genomes[:, N_TRAITS:].var(axis=0).mean())
    diet = phenotype(sample)[:, 9]
    counts = np.bincount(np.clip((diet * _DIET_BINS).astype(np.int64), 0, _DIET_BINS - 1),
                         minlength=_DIET_BINS)
    p = counts[counts > 0] / counts.sum()
    diet_entropy = float(-(p * np.log2(p)).sum() / np.log2(_DIET_BINS))
    unique = len({hash_genome(g) for g in genomes})
    return {"mean_pairwise": mean_pairwise, "mean_variance": mean_variance,
            "diet_entropy": diet_entropy, "unique_genomes": unique,
            "trait_variance": trait_variance, "nn_variance": nn_variance}


def _survival_median(engine) -> float:
    """Median age at death from the engine's bounded histogram.

    Returns ``0.0`` until at least one death has been observed, so callers can
    tell "not yet measured" from a real value. The histogram is cumulative and
    coarse by design (25-tick bins); the value returned is the midpoint of the
    bin containing the median death, which the column documentation states.
    """
    hist = getattr(engine, "death_age_hist", None)
    if hist is None:
        return 0.0
    total = int(hist.sum())
    if total == 0:
        return 0.0
    from .engine import DEATH_AGE_BIN

    cumulative = np.cumsum(hist)
    bin_index = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    bin_index = min(bin_index, hist.size - 1)
    return float((bin_index + 0.5) * DEATH_AGE_BIN)


COLUMNS = (
    "tick", "population", "births", "deaths", "growth_rate",
    "herbivores", "carnivores", "omnivores", "unclassified",
    "mean_age", "median_age", "mean_energy", "mean_health", "mean_body",
    "mean_diet", "mean_speed", "mean_vision", "mean_attack", "mean_defense",
    "mean_lifespan", "mean_mutation_rate",
    "mean_pairwise", "mean_variance", "diet_entropy", "unique_genomes",
    "mutations_per_birth",
    "mean_distance_moved", "mean_bites", "mean_hunts", "mean_escapes",
    "repro_attempts", "repro_successes", "rest_fraction", "foraging_efficiency",
    "plant_biomass", "carcass_biomass", "detritus", "soil",
    "mean_temperature", "mean_moisture",
    # -- appended in the upgrade; never reorder the block above ---------------
    "birth_rate", "death_rate",          # per organism per tick, this interval
    "mean_generation", "max_generation",  # generation turnover
    "trait_variance",                     # mean over the 12 trait loci
    "nn_variance",                        # mean over the 380 network weights
    "mean_organism_biomass",              # Σ(body+E)/n, living organisms
    "deaths_starvation", "deaths_senescence", "deaths_health", "deaths_predation",
    "survival_median",                    # age at which half of all deaths fall
    "population_cv",                      # coefficient of variation in window
)

#: Columns added in the upgrade. Kept separate so persistence/tests can tell the
#: additive block from the original schema without re-listing it.
NEW_COLUMNS = COLUMNS[COLUMNS.index("birth_rate"):]


class StatsRecorder:
    """Row-per-interval statistics with a ring buffer for live charts."""

    def __init__(self, window: int = _RING_BUFFER) -> None:
        self.rows: deque[dict] = deque(maxlen=int(window))
        self.extinctions: list[dict] = []
        self._last_births = 0
        self._last_deaths = 0
        self._last_mutations = 0
        self._last_population = 0
        self._last_tick = 0
        self._seen_hashes: set[bytes] = set()

    def record(self, engine) -> dict:
        """Sample one row from the engine and append it to the ring buffer."""
        pop = engine.population
        idx = np.flatnonzero(pop.alive)
        n = idx.size
        phen = phenotype(pop.genome[idx]) if n else np.zeros((0, 12))
        births = engine.births_total - self._last_births
        deaths = engine.deaths_total - self._last_deaths
        mutations = engine.mutations_total - self._last_mutations
        self._last_births = engine.births_total
        self._last_deaths = engine.deaths_total
        self._last_mutations = engine.mutations_total
        growth = n - self._last_population  # per interval; k = interval length
        self._last_population = n
        interval = max(1, engine.tick - self._last_tick)
        self._last_tick = engine.tick

        roles = classify_roles(pop.plant_intake[idx], pop.meat_intake[idx]) \
            if n else []
        role_count = {r: roles.count(r) for r in
                      (ROLE_HERBIVORE, ROLE_CARNIVORE, ROLE_OMNIVORE,
                       ROLE_UNCLASSIFIED)}

        def mean(col: str) -> float:
            return float(getattr(pop, col)[idx].mean()) if n else 0.0

        moved = mean("dist_moved")
        intake = float(pop.plant_intake[idx].sum() + pop.meat_intake[idx].sum())
        div = diversity(pop.genome[idx]) if n else {
            "mean_pairwise": 0.0, "mean_variance": 0.0,
            "diet_entropy": 0.0, "unique_genomes": 0,
            "trait_variance": 0.0, "nn_variance": 0.0,
        }
        # Per-capita rates over the elapsed interval. Dividing by population and
        # by elapsed ticks makes rows comparable across sampling intervals; the
        # raw ``births``/``deaths`` counters stay for exact reconciliation.
        per_capita = float(interval) * max(1, self._last_population)
        organism_biomass = (float(pop.body[idx].sum() + pop.energy[idx].sum())
                            if n else 0.0)
        cause = getattr(engine, "death_counts", {})
        survival_median = _survival_median(engine)
        row = {
            "tick": engine.tick,
            "population": n,
            "births": births,
            "deaths": deaths,
            "growth_rate": float(growth),
            "herbivores": role_count[ROLE_HERBIVORE],
            "carnivores": role_count[ROLE_CARNIVORE],
            "omnivores": role_count[ROLE_OMNIVORE],
            "unclassified": role_count[ROLE_UNCLASSIFIED],
            "mean_age": float(pop.age[idx].mean()) if n else 0.0,
            "median_age": float(np.median(pop.age[idx])) if n else 0.0,
            "mean_energy": mean("energy"),
            "mean_health": mean("health"),
            "mean_body": mean("body"),
            "mean_diet": float(phen[:, 9].mean()) if n else 0.0,
            "mean_speed": float(phen[:, 1].mean()) if n else 0.0,
            "mean_vision": float(phen[:, 2].mean()) if n else 0.0,
            "mean_attack": float(phen[:, 5].mean()) if n else 0.0,
            "mean_defense": float(phen[:, 6].mean()) if n else 0.0,
            "mean_lifespan": float(phen[:, 10].mean()) if n else 0.0,
            "mean_mutation_rate": float(phen[:, 11].mean()) if n else 0.0,
            "mean_pairwise": div["mean_pairwise"],
            "mean_variance": div["mean_variance"],
            "diet_entropy": div["diet_entropy"],
            "unique_genomes": div["unique_genomes"],
            "mutations_per_birth": (mutations / births) if births > 0 else 0.0,
            "mean_distance_moved": moved,
            "mean_bites": mean("bites"),
            "mean_hunts": mean("hunts"),
            "mean_escapes": mean("escapes"),
            "repro_attempts": float(pop.repro_attempts[idx].sum()) if n else 0.0,
            "repro_successes": float(pop.repro_success[idx].sum()) if n else 0.0,
            "rest_fraction": (float(pop.rest_ticks[idx].sum()) / n
                              / max(1, engine.tick)) if n else 0.0,
            "foraging_efficiency": (intake / moved) if moved > 0.0 else 0.0,
            "plant_biomass": engine.plants.total_biomass(),
            "carcass_biomass": engine.carcasses.total_biomass(),
            "detritus": engine.detritus,
            "soil": engine.soil_stock(),
            "mean_temperature": float(engine.fields.temperature.mean()),
            "mean_moisture": float(engine.fields.moisture.mean()),
            "birth_rate": births / per_capita if per_capita else 0.0,
            "death_rate": deaths / per_capita if per_capita else 0.0,
            "mean_generation": mean("generation"),
            "max_generation": int(pop.generation[idx].max()) if n else 0,
            "trait_variance": div["trait_variance"],
            "nn_variance": div["nn_variance"],
            "mean_organism_biomass": organism_biomass / n if n else 0.0,
            "deaths_starvation": int(cause.get("starvation", 0)),
            "deaths_senescence": int(cause.get("senescence", 0)),
            "deaths_health": int(cause.get("health", 0)),
            "deaths_predation": int(cause.get("predation", 0)),
            "survival_median": survival_median,
            "population_cv": self._population_cv(n),
        }
        self.rows.append(row)
        return row

    def _population_cv(self, current: int) -> float:
        """Coefficient of variation of the recorded population in the window.

        A stability measure, not a forecast: it is computed over the rows still
        held by the ring buffer, so it reflects only observed samples.
        """
        values = [float(r["population"]) for r in self.rows]
        values.append(float(current))
        if len(values) < 2:
            return 0.0
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        return float(arr.std() / mean) if mean > 0.0 else 0.0

    def _last_tick(self) -> int:
        return self.rows[-1]["tick"] if self.rows else 0

    def track_extinctions(self, hashes: list[bytes], tick: int) -> list[bytes]:
        """Log genome-hash clusters that vanished since the previous call."""
        current = set(hashes)
        gone = sorted(self._seen_hashes - current)
        for h in gone:
            self.extinctions.append({"tick": tick, "genome_hash": h.hex()})
        self._seen_hashes = current
        return gone


def extinction_events(recorder: StatsRecorder) -> list[dict]:
    """Extinction log of a recorder (convenience accessor)."""
    return recorder.extinctions


class LineageRecorder:
    """Bounded parent/child record for the living population and recent deaths.

    The engine calls :meth:`record_birth` and :meth:`record_death` only when a
    recorder is attached, so the default run pays nothing. Memory is bounded on
    both sides:

    * the living map holds at most one record per living organism (its size is
      capped by ``Population.capacity``);
    * finished records go to a deque of at most ``max_finished`` entries, so a
      long run stores recent history, not all of it.

    Ancestry is never invented. A parent whose record has already been evicted is
    reported as ``parent_unknown`` rather than being guessed, and ``generation``
    is the engine's own parent-generation counter — the lineage depth — not a
    number computed here.
    """

    def __init__(self, max_finished: int = 5000) -> None:
        self._living: dict[int, dict] = {}
        self.finished: deque[dict] = deque(maxlen=int(max_finished))
        self.births_recorded = 0
        self.deaths_recorded = 0

    def record_birth(self, org_id: int, parent_a: int | None,
                     parent_b: int | None, tick: int, generation: int) -> None:
        self._living[int(org_id)] = {
            "org_id": int(org_id),
            "parent_a": parent_a,
            "parent_b": parent_b,
            "birth_tick": int(tick),
            "death_tick": None,
            "generation": int(generation),
            "cause": None,
            "age_at_death": None,
        }
        self.births_recorded += 1

    def record_death(self, org_id: int, tick: int, cause: str, age: int) -> None:
        record = self._living.pop(int(org_id), None)
        if record is None:
            return  # born before the recorder was attached: no fabricated row
        record["death_tick"] = int(tick)
        record["cause"] = str(cause)
        record["age_at_death"] = int(age)
        self.finished.append(record)
        self.deaths_recorded += 1

    def living(self) -> list[dict]:
        """Records for organisms still alive, ascending by ID."""
        return [self._living[k] for k in sorted(self._living)]

    def rows(self) -> list[dict]:
        """Finished records (oldest first), suitable for ``lineage.csv``."""
        return list(self.finished)

    def lineage_depth_max(self) -> int:
        """Deepest generation among living organisms (0 when none)."""
        return max((r["generation"] for r in self._living.values()), default=0)


# ---------------------------------------------------------------------------
# Export (DESIGN §8 outputs). Matplotlib is imported lazily so headless runs and
# the engine never pay for it; ``viz/`` must not import this module.
# ---------------------------------------------------------------------------
LINEAGE_COLUMNS = ("org_id", "parent_a", "parent_b", "birth_tick", "death_tick",
                   "generation", "cause")
INTERACTION_COLUMNS = ("tick", "kind", "actor_id", "target_id", "amount")


def _write_csv(columns: tuple[str, ...], rows: list[dict], path: str) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(c, "") for c in columns])


def export_csv(rows: list[dict], path: str) -> None:
    """Wide stats CSV: one row per sample interval, :data:`COLUMNS` order."""
    _write_csv(COLUMNS, rows, path)


def export_lineage_csv(records: list[dict], path: str) -> None:
    """Lineage table: one row per recorded organism."""
    _write_csv(LINEAGE_COLUMNS, records, path)


def export_interactions_csv(records: list[dict], path: str) -> None:
    """Interaction edges: ``(tick, kind, actor_id, target_id, amount)``."""
    _write_csv(INTERACTION_COLUMNS, records, path)


_PLOT_NAMES = (
    "population", "roles", "traits", "diversity", "mutation",
    "births_deaths", "predator_prey",
    # added in the upgrade
    "rates", "biomass", "environment", "survival",
)


def _series(rows: list[dict], key: str):
    return [float(r.get(key, 0.0)) for r in rows]


def make_plots(rows: list[dict], out_dir: str) -> None:
    """Write the seven reporting PNGs into ``out_dir`` (Agg backend)."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    ticks = _series(rows, "tick")

    def save(name: str) -> None:
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=100)
        plt.close(fig)

    def note_if_empty(ax) -> None:
        if not rows:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)

    fig, ax = plt.subplots()
    ax.plot(ticks, _series(rows, "population"), label="population")
    ax.set_title("Population"); ax.set_xlabel("tick"); ax.set_ylabel("count")
    note_if_empty(ax); save("population")

    fig, ax = plt.subplots()
    if rows:
        ax.stackplot(ticks,
                     _series(rows, "herbivores"), _series(rows, "omnivores"),
                     _series(rows, "carnivores"), _series(rows, "unclassified"),
                     labels=("herbivores", "omnivores", "carnivores", "unclassified"))
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Measured roles"); ax.set_xlabel("tick"); ax.set_ylabel("count")
    note_if_empty(ax); save("roles")

    fig, ax = plt.subplots()
    for key in ("mean_diet", "mean_body", "mean_speed", "mean_attack"):
        ax.plot(ticks, _series(rows, key), label=key)
    ax.set_title("Mean traits"); ax.set_xlabel("tick"); ax.legend(fontsize=8)
    note_if_empty(ax); save("traits")

    fig, ax = plt.subplots()
    ax.plot(ticks, _series(rows, "mean_pairwise"), label="mean pairwise distance")
    ax.plot(ticks, _series(rows, "diet_entropy"), label="diet entropy (norm.)")
    ax.set_title("Diversity"); ax.set_xlabel("tick"); ax.legend(fontsize=8)
    note_if_empty(ax); save("diversity")

    fig, ax = plt.subplots()
    ax.plot(ticks, _series(rows, "mean_mutation_rate"), label="mean mutation rate")
    ax.plot(ticks, _series(rows, "mutations_per_birth"), label="mutations / birth")
    ax.set_title("Mutation"); ax.set_xlabel("tick"); ax.legend(fontsize=8)
    note_if_empty(ax); save("mutation")

    fig, ax = plt.subplots()
    ax.plot(ticks, _series(rows, "births"), label="births")
    ax.plot(ticks, _series(rows, "deaths"), label="deaths")
    ax.set_title("Births and deaths"); ax.set_xlabel("tick"); ax.legend(fontsize=8)
    note_if_empty(ax); save("births_deaths")

    fig, ax = plt.subplots()
    if rows:
        ax.plot(_series(rows, "herbivores"), _series(rows, "carnivores"))
    ax.set_title("Predator–prey phase"); ax.set_xlabel("herbivores")
    ax.set_ylabel("carnivores")
    note_if_empty(ax); save("predator_prey")

    # -- upgrade additions ---------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(ticks, _series(rows, "birth_rate"), label="births / (organism·tick)")
    ax.plot(ticks, _series(rows, "death_rate"), label="deaths / (organism·tick)")
    ax.set_title("Per-capita rates"); ax.set_xlabel("tick")
    ax.set_ylabel("per organism per tick"); ax.legend(fontsize=8)
    note_if_empty(ax); save("rates")

    fig, ax = plt.subplots()
    ax.plot(ticks, _series(rows, "mean_organism_biomass"),
            label="organism biomass (mean)")
    ax.plot(ticks, _series(rows, "plant_biomass"), label="plant biomass")
    ax.plot(ticks, _series(rows, "carcass_biomass"), label="carcass biomass")
    ax.plot(ticks, _series(rows, "detritus"), label="detritus")
    ax.set_title("Biomass stocks"); ax.set_xlabel("tick")
    ax.set_ylabel("biomass units"); ax.legend(fontsize=8)
    note_if_empty(ax); save("biomass")

    fig, ax = plt.subplots()
    ax.plot(ticks, _series(rows, "mean_temperature"), label="temperature (°C)")
    ax.plot(ticks, _series(rows, "mean_moisture"), label="moisture (0–1)")
    ax.set_title("Environmental conditions"); ax.set_xlabel("tick")
    ax.legend(fontsize=8)
    note_if_empty(ax); save("environment")

    fig, ax = plt.subplots()
    if rows:
        ax.plot(ticks, _series(rows, "survival_median"),
                label="median age at death")
    ax.set_title("Survival"); ax.set_xlabel("tick"); ax.set_ylabel("age (ticks)")
    ax.legend(fontsize=8)
    note_if_empty(ax); save("survival")


