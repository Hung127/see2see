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
PLANT: Color = (52, 118, 62)
CARCASS: Color = (156, 124, 92)
SELECTION: Color = (255, 255, 255)
WORLD_EDGE: Color = (74, 82, 96)

LAYERS = (
    "climate", "moisture", "soil", "temperature", "terrain",
    # added in the upgrade — scientific overlays
    "plants", "organisms", "pheromone",
)

# One colour per event kind, used to outline the affected area in the world.
# These are outlines only: they mark where the physical effect is applied, not
# a property of the organisms, so they must not reuse a diet or plant colour.
EVENT_COLORS: dict[str, Color] = {
    "drought": (232, 176, 96),
    "heavy_rain": (104, 156, 236),
    "heatwave": (240, 120, 84),
    "cold_snap": (128, 196, 236),
    "wildfire": (255, 138, 66),
    "nutrient_pulse": (176, 212, 110),
    "plant_blight": (188, 148, 216),
}
EVENT_DEFAULT: Color = (226, 230, 238)

# Plain name of each outline colour, so the panel can point at the right ring
# when two events are active at once.
EVENT_COLOUR_WORDS: dict[str, str] = {
    "drought": "sandy",
    "heavy_rain": "blue",
    "heatwave": "red",
    "cold_snap": "pale blue",
    "wildfire": "orange",
    "nutrient_pulse": "green",
    "plant_blight": "purple",
}

LAYER_LABELS = {
    "climate": "climate — moisture, warmth and soil richness",
    "moisture": "moisture — brown is dry, blue is wet",
    "soil": "soil nutrients — dark is poor, pale is rich",
    "temperature": "temperature — blue is cold, red is hot",
    "terrain": "terrain height — dark is low, pale is high",
    # scientific overlays (upgrade)
    "plants": "plants — biomass per cell; dark is bare, green is lush",
    "organisms": "organisms — animals per cell; dark is empty, amber is crowded",
    "pheromone": "pheromone — threat signal per cell; dark is none, magenta is strong",
}

#: Unit legend drawn beside the badge for each overlay layer. ``None`` means the
#: layer has no single numeric unit worth quoting (climate is a composite).
LAYER_UNITS: dict[str, str | None] = {
    "climate": None,
    "moisture": "0 – 1 (fraction of saturation)",
    "soil": "biomass units (bu) per cell",
    "temperature": "degrees Celsius",
    "terrain": "0 – 1 (relative height)",
    "plants": "biomass units (bu) per cell",
    "organisms": "animals per cell",
    "pheromone": "0 – 1 (signal strength)",
}

#: The numeric range each sequential overlay maps to, for the legend. Values
#: outside the range are clamped, and the legend says so rather than hiding it.
LAYER_RANGES: dict[str, tuple[float, float] | None] = {
    "climate": None,
    "moisture": (0.0, 1.0),
    "soil": (0.0, 6.0),
    "temperature": (-30.0, 60.0),
    "terrain": (0.0, 1.0),
    "plants": (0.0, 10.0),
    "organisms": (0.0, 10.0),
    "pheromone": (0.0, 1.0),
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


def _ramp(dark: Color, light: Color, u: float) -> Color:
    """Linear interpolation between two colours, ``u`` clamped to [0, 1]."""
    t = _clamp01(u)
    return (int(dark[0] + (light[0] - dark[0]) * t),
            int(dark[1] + (light[1] - dark[1]) * t),
            int(dark[2] + (light[2] - dark[2]) * t))


#: Sequential colour ramps for the single-quantity overlays. Chosen to be
#: distinguishable and to avoid reusing the diet red/green so an overlay can
#: never be mistaken for an organism attribute.
_RAMPS: dict[str, tuple[Color, Color]] = {
    "plants": ((18, 34, 24), (126, 226, 136)),
    "organisms": ((34, 28, 18), (250, 204, 108)),
    "pheromone": ((28, 18, 34), (232, 108, 220)),
}


def layer_color(layer: str, temperature: float = 20.0, moisture: float = 0.5,
                soil: float = 3.0, terrain: float = 0.5,
                plant_density: float | None = None,
                organism_density: float | None = None,
                pheromone: float | None = None) -> Color:
    """Cell colour for one field layer.

    ``climate`` is the original combined view (moisture green, warmth red,
    soil richness yellow-green); the other layers show a single quantity so a
    beginner can tell which part of the environment changed. The optional
    density/signal arguments are only read by the overlays that need them; a
    missing value (``None``) is drawn as the ramp's dark end, which is *not* the
    same claim as "measured zero" — the panel reports the numeric value.
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
    if layer in _RAMPS:
        lo, hi = _RAMPS[layer]
        value = {"plants": plant_density,
                 "organisms": organism_density,
                 "pheromone": pheromone}[layer]
        span = LAYER_RANGES[layer]
        if value is None or span is None:
            return lo
        low, high = span
        return _ramp(lo, hi, (float(value) - low) / max(1e-9, high - low))
    base = 40 + int(60 * m)
    warm = int(40 * _clamp01((temperature - 20.0) / 15.0))
    rich = int(30 * _clamp01(soil / 6.0))
    return (base + warm, base + rich, base)


def layer_ramp(layer: str, steps: int = 9) -> list[Color] | None:
    """Colours across a layer's declared display range, dark end first.

    ``None`` when the layer has no single numeric quantity to sweep (the
    combined climate view). The range and the ramp are the same values the
    renderer uses, so the legend cannot describe a different mapping.
    """
    span = LAYER_RANGES.get(layer)
    if span is None:
        return None
    steps = max(2, int(steps))
    low, high = span
    out: list[Color] = []
    for i in range(steps):
        value = low + (high - low) * i / (steps - 1)
        if layer in _RAMPS:
            out.append(layer_color(layer, plant_density=value,
                                   organism_density=value, pheromone=value))
        elif layer == "soil":
            out.append(layer_color(layer, soil=value))
        elif layer == "temperature":
            out.append(layer_color(layer, temperature=value))
        elif layer == "terrain":
            out.append(layer_color(layer, terrain=value))
        else:
            out.append(layer_color(layer, moisture=value))
    return out


def layer_scale_text(layer: str) -> str | None:
    """``"0 – 1 (fraction of saturation)"`` style label for the layer legend."""
    span = LAYER_RANGES.get(layer)
    unit = LAYER_UNITS.get(layer)
    if span is None or unit is None:
        return None
    return f"{span[0]:g} – {span[1]:g} {unit}"


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
