"""Genome layout, phenotype mapping and founder sampling (DESIGN §4).

A genome is ``float64[392]``: 12 trait loci in ``[0, 1]`` followed by 380 neural
parameters. Trait loci are *genetic values*; ``phenotype`` maps them monotonically to
the physical ranges in the gene table. The network block layout is fixed and shared
with :mod:`evolve.brain`:

    [0:192)   W1 (16x12, row-major)   [192:204)  b1 (12)
    [204:300) W2 (12x8,  row-major)   [300:308)  b2 (8)
    [308:372) W3 (8x8,   row-major)   [372:380)  b3 (8)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .rng import Rng

N_TRAITS = 12
N_NN = 380
N_GENES = N_TRAITS + N_NN

TRAIT_NAMES = (
    "body_size", "speed_max", "vision_range", "vision_angle", "metabolic_rate",
    "attack", "defense", "repro_threshold", "repro_investment", "diet_preference",
    "lifespan", "mutation_rate",
)


@dataclass(frozen=True)
class GeneSpec:
    name: str
    kind: str  # linear | exp | frac
    lo: float
    hi: float


GENE_SPECS: tuple[GeneSpec, ...] = (
    GeneSpec("body_size", "exp", 0.3, 5.0),
    GeneSpec("speed_max", "linear", 0.0, 40.0),
    GeneSpec("vision_range", "linear", 40.0, 160.0),
    GeneSpec("vision_angle", "linear", 30.0, 360.0),
    GeneSpec("metabolic_rate", "linear", 0.6, 1.6),
    GeneSpec("attack", "linear", 0.0, 5.0),
    GeneSpec("defense", "linear", 0.0, 5.0),
    GeneSpec("repro_threshold", "frac", 0.40, 0.95),
    GeneSpec("repro_investment", "frac", 0.15, 0.60),
    GeneSpec("diet_preference", "linear", 0.0, 1.0),
    GeneSpec("lifespan", "exp", 200.0, 5000.0),
    GeneSpec("mutation_rate", "exp", 5e-4, 5e-2),
)

NN_LO, NN_HI = -8.0, 8.0


def _map(g: np.ndarray, spec: GeneSpec) -> np.ndarray:
    if spec.kind == "linear" or spec.kind == "frac":
        return spec.lo + np.clip(g, 0.0, 1.0) * (spec.hi - spec.lo)
    return np.exp(np.log(spec.lo) + np.clip(g, 0.0, 1.0) * (np.log(spec.hi) - np.log(spec.lo)))


def phenotype(genes: np.ndarray) -> np.ndarray:
    """Map trait loci ``(n, 12)`` to physical phenotypes ``(n, 12)``.

    ``repro_threshold`` and ``repro_investment`` are returned as fractions of the
    organism's reserve capacity ``E_max`` (state-relative by design, DESIGN §4.1).
    """
    genes = np.atleast_2d(np.asarray(genes, dtype=np.float64))
    out = np.empty((genes.shape[0], N_TRAITS), dtype=np.float64)
    for i, spec in enumerate(GENE_SPECS):
        out[:, i] = _map(genes[:, i], spec)
    return out


def founders(n: int, rng: Rng) -> np.ndarray:
    """An unbiased founder population: mid-biased traits, small random weights."""
    traits = rng.beta(2.0, 2.0, size=(n, N_TRAITS))
    weights = rng.normal(0.0, 0.35, size=(n, N_NN - 20))
    biases = rng.normal(0.0, 0.05, size=(n, 20))
    return np.concatenate([traits, weights, biases], axis=1)


def validate_bounds(genes: np.ndarray) -> None:
    genes = np.atleast_2d(np.asarray(genes, dtype=np.float64))
    if genes.shape[1] != N_GENES:
        raise ValueError(f"genome must have {N_GENES} columns, got {genes.shape[1]}")
    if not np.all((genes[:, :N_TRAITS] >= 0.0) & (genes[:, :N_TRAITS] <= 1.0)):
        raise ValueError("trait loci must lie in [0, 1]")
    if not np.all((genes[:, N_TRAITS:] >= NN_LO) & (genes[:, N_TRAITS:] <= NN_HI)):
        raise ValueError("neural parameters must lie in [-8, 8]")


def hash_genome(genes: np.ndarray) -> bytes:
    """Stable per-genome identity used for diversity statistics."""
    return hashlib.sha256(np.round(np.asarray(genes, dtype=np.float64), 6).tobytes()).digest()
