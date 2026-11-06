"""Statistics aggregators (DESIGN §8).

Task 17 lands the role classifier; Task 18 adds the recorder, diversity and
mutation statistics. Roles are *measured from lifetime intake*, never declared
from the diet gene: the gene is a preference, the counters are the evidence.
"""
from __future__ import annotations

import numpy as np

ROLE_HERBIVORE = "herbivore"
ROLE_CARNIVORE = "carnivore"
ROLE_OMNIVORE = "omnivore"
ROLE_UNCLASSIFIED = "unclassified"

_MIN_LIFETIME_INTAKE = 1.0  # bu; below this there is not enough evidence


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
