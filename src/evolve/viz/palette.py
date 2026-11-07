"""Colours shared by the world renderer and the information panels.

The encodings are *visual summaries of measured state*: terrain shading comes
from the field grid, plant colour is constant, and an organism's colour is its
inherited ``diet_preference`` gene (a gene, not an observed behaviour). The
sidebar additionally reports measured roles from lifetime intake, and says so.
"""
from __future__ import annotations

Color = tuple[int, int, int]

BG: Color = (14, 16, 20)
PANEL_BG: Color = (22, 25, 31)
PANEL_EDGE: Color = (52, 58, 70)
TEXT: Color = (226, 230, 238)
TEXT_DIM: Color = (148, 156, 170)
TEXT_FAINT: Color = (104, 112, 126)
ACCENT: Color = (96, 178, 245)
GOOD: Color = (104, 216, 140)
WARN: Color = (245, 194, 92)
BAD: Color = (236, 96, 88)
PLANT: Color = (64, 168, 78)
CARCASS: Color = (156, 124, 92)
SELECTION: Color = (255, 255, 255)
WORLD_EDGE: Color = (74, 82, 96)

LAYERS = ("climate", "moisture", "soil", "temperature", "terrain")

LAYER_LABELS = {
    "climate": "climate — moisture, warmth and soil richness",
    "moisture": "moisture — brown is dry, blue is wet",
    "soil": "soil nutrients — dark is poor, pale is rich",
    "temperature": "temperature — blue is cold, red is hot",
    "terrain": "terrain height — dark is low, pale is high",
}


def next_layer(layer: str) -> str:
    """Cycle through :data:`LAYERS`; unknown names restart at the first layer."""
    try:
        i = LAYERS.index(layer)
    except ValueError:
        return LAYERS[0]
    return LAYERS[(i + 1) % len(LAYERS)]


def _clamp01(v: float) -> float:
    return 0.0 if v != v else (1.0 if v > 1.0 else (0.0 if v < 0.0 else v))


def layer_color(layer: str, temperature: float, moisture: float, soil: float,
                terrain: float) -> Color:
    """Cell colour for one field layer.

    ``climate`` is the original combined view (moisture green, warmth red,
    soil richness yellow-green); the other layers show a single quantity so a
    beginner can tell which part of the environment changed.
    """
    m = _clamp01(moisture)
    if layer == "moisture":
        return (int(96 - 56 * m), int(76 + 46 * m), int(48 + 104 * m))
    if layer == "soil":
        s = _clamp01(soil / 6.0)
        return (int(50 + 100 * s), int(46 + 74 * s), int(40 + 30 * s))
    if layer == "temperature":
        u = _clamp01((temperature + 30.0) / 90.0)
        return (int(40 + 200 * u), int(80 + 40 * u), int(240 - 200 * u))
    if layer == "terrain":
        h = _clamp01(terrain)
        return (int(34 + 126 * h), int(48 + 112 * h), int(66 + 84 * h))
    base = 40 + int(60 * m)
    warm = int(40 * _clamp01((temperature - 20.0) / 15.0))
    rich = int(30 * _clamp01(soil / 6.0))
    return (base + warm, base + rich, base)


def diet_color(d: float) -> Color:
    """Herbivore green -> omnivore yellow -> carnivore red (gene locus 9)."""
    d = _clamp01(d)
    return (int(40 + 200 * d), int(200 - 120 * d), 60)


def diet_word(d: float) -> str:
    """Plain-language label for a diet *preference* gene value."""
    if d < 0.33:
        return "herbivore-leaning"
    if d > 0.66:
        return "carnivore-leaning"
    return "mixed"
