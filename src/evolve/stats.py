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
                "unique_genomes": 0}
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
    diet = phenotype(sample)[:, 9]
    counts = np.bincount(np.clip((diet * _DIET_BINS).astype(np.int64), 0, _DIET_BINS - 1),
                         minlength=_DIET_BINS)
    p = counts[counts > 0] / counts.sum()
    diet_entropy = float(-(p * np.log2(p)).sum() / np.log2(_DIET_BINS))
    unique = len({hash_genome(g) for g in genomes})
    return {"mean_pairwise": mean_pairwise, "mean_variance": mean_variance,
            "diet_entropy": diet_entropy, "unique_genomes": unique}


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
)


class StatsRecorder:
    """Row-per-interval statistics with a ring buffer for live charts."""

    def __init__(self, window: int = _RING_BUFFER) -> None:
        self.rows: deque[dict] = deque(maxlen=int(window))
        self.extinctions: list[dict] = []
        self._last_births = 0
        self._last_deaths = 0
        self._last_mutations = 0
        self._last_population = 0
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
        }
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
        }
        self.rows.append(row)
        return row

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


