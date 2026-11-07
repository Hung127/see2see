"""Beginner-facing sidebar, footer and inspector text (DESIGN §10).

Everything here is a *readout of measured state*. Nothing is inferred: role
counts come from lifetime intake (``stats.classify_roles``), diet colour comes
from the inherited preference gene, and the sensor/output rows are the exact
last-tick vectors the network saw and produced. "Nothing detected" is shown as
``—`` rather than a made-up value.
"""
from __future__ import annotations

import numpy as np
import pygame

from .. import stats as stats_mod
from .palette import (
    ACCENT, CARCASS, EVENT_COLOUR_WORDS, GOOD, LAYER_LABELS, PANEL_BG,
    PANEL_EDGE, PLANT, SELECTION, TEXT, TEXT_DIM, TEXT_FAINT, WARN, diet_color,
)

PANEL_W = 468
FOOTER_H = 58
MIN_WORLD_W = 320
SCROLL_STEP = 48

# ---------------------------------------------------------------------------
# Sensor / output vocabulary (indices: src/evolve/sensors.py, actions.py)
# ---------------------------------------------------------------------------
SENSOR_LABELS = (
    "food closeness", "food bearing",
    "threat closeness", "threat bearing",
    "prey closeness", "prey bearing",
    "mate closeness",
    "energy", "health", "age",
    "temperature", "moisture",
    "plant biomass", "organisms in cell",
    "threat signal", "repro drive",
)

OUTPUT_LABELS = (
    "forward (|o0|)", "turn", "throttle gate", "eat",
    "attack", "sprint", "reproduce", "rest",
)

KIND_INFO = {
    "drought": "Drought — air and soil dry out; animals pay water stress.",
    "heavy_rain": "Heavy rain — low ground floods and injures animals.",
    "heatwave": "Heatwave — air temperature rises.",
    "cold_snap": "Cold snap — air temperature falls.",
    "wildfire": "Wildfire — plants burn to soil; animals in the area are hurt.",
    "nutrient_pulse": "Nutrient pulse — soil is enriched with new mass.",
    "plant_blight": "Plant blight — plants in the area die back to detritus.",
}

def _clamp01(v: float) -> float:
    return 0.0 if v != v else (1.0 if v > 1.0 else (0.0 if v < 0.0 else v))


def amount(v: float) -> str:
    """Compact human-readable number (no misleading precision)."""
    if v != v:
        return "—"
    if abs(v) >= 10_000.0:
        return f"{v:,.0f}"
    if abs(v) >= 100.0:
        return f"{v:,.0f}"
    if abs(v) >= 10.0:
        return f"{v:.1f}"
    return f"{v:.2f}"


def percent(v: float) -> str:
    return f"{100.0 * _clamp01(v):.0f}%"


def season_text(engine) -> str:
    """Position in the sinusoidal season, as a fraction of the cycle.

    The environment is ``t_mean + t_amp·sin(2πt/period)``; describing the phase
    and whether it is currently warming or cooling is factual, unlike naming
    arbitrary seasons.
    """
    period = max(1, int(engine.cfg.environment.season_period))
    phase = (engine.tick % period) / period
    warming = np.cos(2.0 * np.pi * phase) > 0.0
    return f"{phase * 100:.0f}% of cycle · {'warming' if warming else 'cooling'}"


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
def legend_items() -> list[tuple[str, tuple[int, int, int], str]]:
    """``(label, colour, marker)`` rows; marker is circle/square/organism/ring.

    Organism swatches go through :func:`palette.diet_color`, so the legend and
    the world cannot drift apart.
    """
    return [
        ("plant", PLANT, "circle"),
        ("carcass (meat)", CARCASS, "square"),
        ("herbivore-leaning", diet_color(0.05), "organism"),
        ("mixed diet gene", diet_color(0.5), "organism"),
        ("carnivore-leaning", diet_color(0.95), "organism"),
        ("selected organism", SELECTION, "ring"),
    ]


# ---------------------------------------------------------------------------
# Overview text
# ---------------------------------------------------------------------------
def population_text(engine) -> str:
    return f"{engine.population.n_alive}"


def births_deaths_text(engine) -> str:
    return (f"{engine.births_total:,} born · {engine.deaths_total:,} died · "
            f"{engine.mutations_total:,} mutations")


def role_counts(engine) -> dict[str, int]:
    """Role counts measured from lifetime intake (not the diet gene)."""
    pop = engine.population
    idx = np.flatnonzero(pop.alive)
    names = (stats_mod.ROLE_HERBIVORE, stats_mod.ROLE_OMNIVORE,
             stats_mod.ROLE_CARNIVORE, stats_mod.ROLE_UNCLASSIFIED)
    if idx.size == 0:
        return {name: 0 for name in names}
    labels = stats_mod.classify_roles(pop.plant_intake[idx], pop.meat_intake[idx])
    return {name: labels.count(name) for name in names}


def role_text(engine) -> str:
    """Compact role breakdown; zero categories are omitted to stay one line."""
    counts = role_counts(engine)
    if sum(counts.values()) == 0:
        return "no living organisms"
    parts = []
    for key, word in ((stats_mod.ROLE_HERBIVORE, "herbivore"),
                      (stats_mod.ROLE_OMNIVORE, "omnivore"),
                      (stats_mod.ROLE_CARNIVORE, "carnivore"),
                      (stats_mod.ROLE_UNCLASSIFIED, "unclassified")):
        if counts[key]:
            parts.append(f"{counts[key]} {word}")
    return " · ".join(parts)


def role_note(engine) -> str | None:
    """Explain "unclassified" only when it is actually on screen."""
    counts = role_counts(engine)
    if not counts.get(stats_mod.ROLE_UNCLASSIFIED):
        return None
    return ("unclassified means it has not eaten 1 bu yet, so there is not "
            "enough evidence to call it anything")


def matter_rows(engine) -> list[tuple[str, str]]:
    """The matter pools, then the total that conservation holds fixed.

    The total is the point: it is what makes "plants and animals are made of
    soil" a measurable claim rather than a slogan, so it is shown next to the
    value it started from.
    """
    pop = engine.population
    idx = np.flatnonzero(pop.alive)
    reserve = float(pop.energy[idx].sum()) if idx.size else 0.0
    body = float(pop.body[idx].sum()) if idx.size else 0.0
    return [
        ("plant biomass", f"{amount(engine.plants.total_biomass())} bu "
                          f"({engine.plants.n_alive:,} plants)"),
        ("carcasses", f"{amount(engine.carcasses.total_biomass())} bu"),
        ("detritus", f"{amount(engine.detritus)} bu"),
        ("organism tissue", f"{amount(body + reserve)} bu"),
        ("soil nutrients", f"{amount(engine.soil_stock())} bu"),
        ("total (organic + soil)", f"{amount(engine.total_mass())} bu "
                                   f"· started at "
                                   f"{amount(engine.initial_total)} bu"),
    ]


def average_rows(engine) -> list[tuple[str, str]]:
    pop = engine.population
    idx = np.flatnonzero(pop.alive)
    if idx.size == 0:
        return [("living organisms", "none — the population is extinct")]
    gen_mean = float(pop.generation[idx].mean())
    gen_max = int(pop.generation[idx].max())
    return [
        ("mean reserve", f"{amount(float(pop.energy[idx].mean()))} bu"),
        ("mean body", f"{amount(float(pop.body[idx].mean()))} bu"),
        ("mean health", f"{amount(float(pop.health[idx].mean()))} / 100"),
        ("mean age", f"{amount(float(pop.age[idx].mean()))} ticks"),
        ("generation", f"mean {gen_mean:.1f} · newest {gen_max}"),
    ]


def environment_rows(engine) -> list[tuple[str, str]]:
    f = engine.fields
    return [
        ("temperature", f"{float(f.temperature.mean()):.1f} °C"),
        ("moisture", percent(float(f.moisture.mean()))),
        ("season", season_text(engine)),
    ]


def event_lines(engine, max_lines: int = 3) -> list[str]:
    """Active events first, then the next scheduled start (if any)."""
    active = engine.events.active(engine.tick)
    lines: list[str] = []
    for ev in active[:max_lines]:
        # "local area" alone left the reader hunting the map for the affected
        # patch, so the region gets an outline in the same colour (render.py)
        # and the text says what that outline means.
        if ev.region[2] >= 1e8:
            where = "worldwide"
        else:
            where = (f"inside the "
                     f"{EVENT_COLOUR_WORDS.get(ev.kind, 'white')} ring on "
                     f"the map")
        left = max(0, ev.start_tick + max(1, int(ev.duration)) - engine.tick)
        lines.append(f"{KIND_INFO.get(ev.kind, ev.kind)} "
                     f"[{where}, {left} ticks left]")
    upcoming = [ev for ev in engine.events.log if ev.start_tick > engine.tick]
    if upcoming and len(lines) < max_lines:
        ev = min(upcoming, key=lambda e: e.start_tick)
        lines.append(f"Next scheduled: {ev.kind} at tick {ev.start_tick} "
                     f"(in {ev.start_tick - engine.tick}).")
    if not lines:
        lines = ["No environmental event right now; random ones may start."]
    return lines[:max_lines]


# ---------------------------------------------------------------------------
# Inspector text
# ---------------------------------------------------------------------------
def _bearing_text(value: float, closeness: float) -> str:
    """Bearing channels are meaningless when the paired sight channel is empty.

    World +y is drawn downward, so a positive relative bearing is clockwise on
    screen, i.e. on the organism's right when it faces right. (Verified against
    ``world.bearing``: a target at larger y gives a positive bearing.)
    """
    if closeness <= -0.999:
        return "—"
    side = "right" if value > 0 else "left"
    return f"{value * 180.0:+.0f}° {side}"


def _inverse_sensor(index: int, value: float) -> str:
    """Undo the documented sensor normalisation for the human readout."""
    if index in (0, 2, 4, 6):          # closeness = 1 − d/r_vision
        return "— (nothing in view)" if value <= -0.999 else f"{percent(value)} close"
    if index == 7:                       # E / E_max
        return percent(value)
    if index == 8:                       # health / health_max
        return percent(value)
    if index == 9:                       # age / lifespan
        return percent(value) + " of lifespan"
    if index == 10:                      # (T − 20)/15
        return f"{20.0 + 15.0 * value:.1f} °C"
    if index == 11:                      # 2m − 1
        return percent(0.5 * (value + 1.0))
    if index == 12:                      # 2B/10 − 1
        return f"{max(0.0, 5.0 * (value + 1.0)):.1f} bu"
    if index == 13:                      # 2n/20 − 1, counts self
        return f"{max(0.0, 10.0 * (value + 1.0)):.0f} here (incl. self)"
    if index == 14:                      # clipped pheromone at the cell
        return f"{value:+.2f} (pheromone)"
    if index == 15:                      # threshold-relative drive
        return f"{value:+.2f} (0 = at threshold)"
    return f"{value:+.2f}"


def sensor_rows(payload: dict) -> list[tuple[str, str]]:
    sensors = np.asarray(payload.get("sensors", np.zeros(16)), dtype=np.float64)

    def at(i: int) -> float:
        return float(sensors[i]) if i < sensors.size else -1.0

    rows: list[tuple[str, str]] = []
    for i, label in enumerate(SENSOR_LABELS):
        if i in (1, 3, 5):
            text = _bearing_text(at(i), at(i - 1))
        else:
            text = _inverse_sensor(i, at(i))
        rows.append((label, text))
    return rows


def output_rows(payload: dict) -> list[tuple[str, str, bool]]:
    """``(label, text, is_active)`` — raw value plus the decoded physical attempt.

    The decoded effect is shown too: ``rest`` suppresses eating and attacking,
    so those rows are reported as blocked rather than as attempts.
    """
    o = np.asarray(payload.get("outputs", np.zeros(8)), dtype=np.float64)
    phen = np.asarray(payload.get("phenotypes", np.zeros(12)), dtype=np.float64)
    speed_max = float(phen[1]) if phen.size > 1 else 0.0
    resting = float(o[7]) > 0.5
    # Mirrors ``actions.decode`` exactly: rest scales throttle by (1 - o7),
    # and also blocks eating/attacking.
    throttle = abs(float(o[0])) * float(o[2]) * (1.0 - float(o[7]))
    sprint = 1.0 + 1.5 * float(o[5])
    turn = float(o[1])
    if turn > 0.05:
        turn_word = "right (clockwise on screen)"
    elif turn < -0.05:
        turn_word = "left (anticlockwise)"
    else:
        turn_word = "straight"
    speed = speed_max * throttle * sprint

    def attempt(value: float, label: str) -> tuple[str, bool]:
        """The row label names the action; the value shows the score and result."""
        if value <= 0.5:
            return f"{value:.2f} · no", False
        if resting:
            return f"{value:.2f} · yes, but blocked by rest", False
        return f"{value:.2f} · yes", True

    eat_text, eat_on = attempt(float(o[3]), "eat")
    attack_text, attack_on = attempt(float(o[4]), "attack")
    repro = float(o[6]) > 0.5
    rows = [
        ("resulting speed", f"{speed:.2f} u/tick", speed > 1e-6),
        (OUTPUT_LABELS[0], f"{abs(float(o[0])):.2f} (sign ignored)", False),
        (OUTPUT_LABELS[1], f"{turn:+.2f} · {turn_word}", abs(turn) > 0.05),
        (OUTPUT_LABELS[2], f"{float(o[2]):.2f}", False),
        (OUTPUT_LABELS[3], eat_text, eat_on),
        (OUTPUT_LABELS[4], attack_text, attack_on),
        (OUTPUT_LABELS[5], f"×{sprint:.2f} speed", float(o[5]) > 0.5),
        (OUTPUT_LABELS[6],
         "TRY (gated by maturity + reserve)" if repro else "not now", repro),
        (OUTPUT_LABELS[7], "RESTING" if resting else "active", resting),
    ]
    return rows


def organism_rows(payload: dict) -> list[tuple[str, str]]:
    from .palette import diet_word

    phen = np.asarray(payload.get("phenotypes", np.zeros(12)), dtype=np.float64)
    body = float(payload.get("body", 0.0))
    emax = 10.0 * max(body, 1e-9) ** 0.9
    energy = float(payload.get("energy", 0.0))
    counters = payload.get("counters", {})
    parents = payload.get("parents", (None, None))
    parent_text = "none (founder)" if parents[0] is None and parents[1] is None \
        else f"{parents[0]} + {parents[1]}"
    diet = float(phen[9]) if phen.size > 9 else 0.0
    # The population section classifies roles from lifetime intake; the
    # inspector shows the same classification for this one animal, from the
    # same function, so the two readouts cannot disagree. The gene above is a
    # preference; this row is the evidence.
    role = stats_mod.classify_roles(
        np.array([float(counters.get("plant_intake", 0.0))]),
        np.array([float(counters.get("meat_intake", 0.0))]))[0]
    return [
        ("id / generation", f"{payload.get('id')} / {payload.get('generation')}"),
        ("age", f"{payload.get('age')} ticks"),
        ("reserve / capacity", f"{amount(energy)} / {amount(emax)} bu"),
        ("health", f"{amount(float(payload.get('health', 0.0)))} / 100"),
        ("body mass", f"{amount(body)} bu"),
        ("diet gene", f"{diet:.2f} · {diet_word(diet)}"),
        ("role (from intake)", role),
        ("speed max", f"{phen[1]:.1f} u/tick"),
        ("vision", f"{phen[2]:.0f} u · {phen[3]:.0f}° cone"),
        ("attack / defense", f"{phen[5]:.2f} / {phen[6]:.2f}"),
        ("parents", parent_text),
        ("lifetime intake",
         f"{amount(float(counters.get('plant_intake', 0.0)))} plant · "
         f"{amount(float(counters.get('meat_intake', 0.0)))} meat bu"),
        ("lifetime actions",
         f"{counters.get('bites', 0):.0f} bites · "
         f"{counters.get('hunts', 0):.0f} attacks · "
         f"{counters.get('repro_success', 0):.0f} children"),
        ("distance moved", f"{amount(float(counters.get('dist_moved', 0.0)))} u"),
    ]


def lineage_rows(engine, org_id: int, depth: int = 4) -> list[tuple[str, str]]:
    """Ancestry rows for one organism, from recorded data only.

    Two sources, both real: the population arrays carry the parent IDs recorded
    at birth, and (when a :class:`~evolve.stats.LineageRecorder` is attached) the
    recorder can follow the chain further back. A founder has no recorded
    parents, and a record that has been evicted is reported as unknown — neither
    is ever invented.
    """
    pop = engine.population
    idx = np.flatnonzero(pop.alive)
    rows: list[tuple[str, str]] = []
    if idx.size == 0:
        return [("lineage", "no living organisms")]
    hits = np.flatnonzero(pop.id[idx] == int(org_id))
    if hits.size == 0:
        return [("lineage", "organism is no longer alive")]
    i = int(idx[int(hits[0])])
    pa = int(pop.parent_a[i])
    pb = int(pop.parent_b[i])
    if pa < 0 and pb < 0:
        rows.append(("parents", "none recorded — founder of this run"))
    elif pb < 0:
        rows.append(("parent", f"{pa} (asexual)"))
    else:
        rows.append(("parents", f"{pa} and {pb}"))

    recorder = getattr(engine, "lineage", None)
    if recorder is None:
        rows.append(("ancestry depth", "not recorded in this session"))
        return rows
    chain = recorder.ancestry(int(org_id), depth=depth)
    if not chain or chain[-1] is None:
        rows.append(("ancestry depth", "not recorded (born before logging)"))
        return rows
    # Oldest first: each generation's id, with the unknown link marked as such.
    parts = []
    for record in chain:
        if record is None:
            parts.append("unknown")
        else:
            parts.append(f"{record['org_id']} (gen {record['generation']})")
    rows.append(("ancestry (oldest first)", " → ".join(parts)))
    deepest = max((r["generation"] for r in chain if r is not None), default=0)
    rows.append(("recorded depth", f"{len(chain)} generations · gen {deepest}"))
    finished = len(getattr(recorder, "finished", ()))
    rows.append(("records kept", f"{finished} finished · "
                                 f"{len(recorder.living())} living"))
    return rows


def decision_summary(payload: dict) -> str:
    o = np.asarray(payload.get("outputs", np.zeros(8)), dtype=np.float64)
    bits = []
    if o[7] > 0.5:
        bits.append("resting (cannot eat or attack)")
    else:
        if o[3] > 0.5:
            bits.append("trying to eat")
        if o[4] > 0.5:
            bits.append("trying to attack")
    if o[6] > 0.5:
        bits.append("trying to reproduce")
    return "; ".join(bits) if bits else "moving without eating, attacking or reproducing"


# ---------------------------------------------------------------------------
# Pygame text helpers
# ---------------------------------------------------------------------------
class TextCache:
    """Per-app glyph cache: rendering text every frame is the expensive part."""

    def __init__(self, limit: int = 800) -> None:
        self._cache: dict[tuple, pygame.Surface] = {}
        self._limit = int(limit)

    def render(self, font: pygame.font.Font, text: str,
               color: tuple[int, int, int]) -> pygame.Surface:
        key = (id(font), text, color)
        surf = self._cache.get(key)
        if surf is None:
            surf = font.render(text, True, color)
            if len(self._cache) >= self._limit:
                self._cache.clear()
            self._cache[key] = surf
        return surf

    def fit(self, font: pygame.font.Font, text: str,
            color: tuple[int, int, int], width: int) -> pygame.Surface:
        """Render ``text`` cut to ``width`` pixels with an ellipsis."""
        if font.size(text)[0] <= width:
            return self.render(font, text, color)
        cut = text
        while cut and font.size(cut + "…")[0] > width:
            cut = cut[:-1]
        return self.render(font, (cut + "…") if cut else "…", color)


class Panel:
    """A clipped vertical cursor of text rows inside ``rect``.

    The cursor starts at ``rect.y - scroll`` so callers can scroll content; the
    surface clip hides everything outside the panel.
    """

    def __init__(self, screen: pygame.Surface, rect: pygame.Rect,
                 fonts: dict[str, pygame.font.Font], cache: TextCache,
                 scroll: int = 0) -> None:
        self.screen = screen
        self.rect = rect
        self.fonts = fonts
        self.cache = cache
        self.top = rect.y - int(scroll)
        self.y = self.top
        self._pad = 12

    def content_height(self) -> int:
        return max(0, self.y - self.top + 8)

    def _line(self, font: pygame.font.Font, text: str, color, x: int = 0) -> None:
        surf = self.cache.render(font, text, color)
        self.screen.blit(surf, (self.rect.x + x, self.y))

    def gap(self, px: int) -> None:
        self.y += int(px)

    def title(self, text: str) -> None:
        self._line(self.fonts["title"], text, TEXT)
        self.y += self.fonts["title"].get_linesize() + 2

    def section(self, text: str) -> None:
        self.gap(10)
        self._line(self.fonts["small"], text.upper(), ACCENT)
        self.y += self.fonts["small"].get_linesize() + 1
        pygame.draw.line(self.screen, PANEL_EDGE,
                         (self.rect.x, self.y),
                         (self.rect.right, self.y), 1)
        self.y += 5

    def text(self, text: str, color=TEXT, font: str = "body") -> None:
        f = self.fonts[font]
        surf = self.cache.fit(f, text, color, self.rect.width - 2 * self._pad)
        self.screen.blit(surf, (self.rect.x + self._pad, self.y))
        self.y += f.get_linesize()

    def wrapped(self, text: str, color=TEXT, font: str = "small",
                indent: int = 0) -> None:
        """Word-wrap into the panel width."""
        f = self.fonts[font]
        width = self.rect.width - 2 * self._pad - indent
        words, line = text.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if f.size(trial)[0] > width and line:
                self.screen.blit(self.cache.render(f, line, color),
                                 (self.rect.x + self._pad + indent, self.y))
                self.y += f.get_linesize()
                line = word
            else:
                line = trial
        if line:
            self.screen.blit(self.cache.render(f, line, color),
                             (self.rect.x + self._pad + indent, self.y))
            self.y += f.get_linesize()

    def row(self, label: str, value: str, color=TEXT) -> None:
        """One label left, one value right; the label is ellipsised if needed."""
        f = self.fonts["small"]
        value_surf = self.cache.render(f, value, color)
        label_w = max(40, self.rect.width - value_surf.get_width()
                      - 3 * self._pad)
        label_surf = self.cache.fit(f, label, TEXT_DIM, label_w)
        self.screen.blit(label_surf, (self.rect.x + self._pad, self.y))
        self.screen.blit(value_surf, (self.rect.right - self._pad
                                      - value_surf.get_width(), self.y))
        self.y += f.get_linesize()

    def footer_hint(self, text: str) -> None:
        self.gap(8)
        self.wrapped(text, TEXT_FAINT, "small")


def draw_sidebar(screen: pygame.Surface, rect: pygame.Rect,
                 fonts: dict[str, pygame.font.Font], cache: TextCache,
                 engine, state) -> None:
    """Draw the right-hand column: guide, legend and live statistics.

    ``state`` carries presentation flags: ``paused``, ``speed``, ``layer``,
    ``show_inspector`` and ``selected`` (payload dict or ``None``).
    """
    pygame.draw.rect(screen, PANEL_BG, rect)
    pygame.draw.line(screen, PANEL_EDGE, (rect.x, rect.y),
                     (rect.x, rect.bottom), 1)
    # Clamp with the previous frame's measured height; the value only shrinks
    # when the population dies or the selection is cleared.
    max_scroll = max(0, int(getattr(state, "content_height", 0))
                     - int(getattr(state, "view_height", 0)))
    state.scroll = max(0, min(int(state.scroll), max_scroll))
    # Leave a gutter on the right for the scrollbar, so text never runs under it.
    inner = pygame.Rect(rect.x + 14, rect.y + 8,
                        rect.width - 14 - 26, rect.height - 30)
    state.view_height = inner.height
    # Clip to the text box, not the whole panel: otherwise the half-visible
    # next line spills into the bottom gutter below the fade.
    screen.set_clip(inner)
    p = Panel(screen, inner, fonts, cache, scroll=state.scroll)

    p.title("EVOLVE")
    status = "PAUSED — SPACE to run" if state.paused else f"RUNNING  {state.speed}×"
    status_color = WARN if state.paused else GOOD
    p.row(f"tick {engine.tick:,}", status, status_color)
    if getattr(state, "follow", False) and state.selected_id is not None:
        # Without a marker the camera appears to drift on its own, which reads
        # as a bug rather than as "you pressed F".
        p.wrapped(f"camera following organism {state.selected_id} "
                  "(F to stop)", ACCENT, "small")
    if getattr(state, "selected_gone", False):
        p.wrapped("The organism you were inspecting died. The panel is back "
                  "to the population view; click another dot to inspect it.",
                  WARN, "small")

    selected = state.selected
    if selected is not None:
        # Keep the world context visible while inspecting; otherwise the whole
        # panel is about one animal and the user loses the population reading.
        p.wrapped(f"{engine.population.n_alive:,} alive · "
                  f"{engine.births_total:,} born · "
                  f"{engine.deaths_total:,} died", TEXT_DIM, "small")
        p.section("selected organism")
        for label, value in organism_rows(selected):
            p.row(label, value)
        p.section("ancestry (recorded only)")
        for label, value in lineage_rows(engine, selected["id"]):
            p.row(label, value)
        p.footer_hint("Ancestry comes from parents recorded at birth. Founders "
                      "have none, and evicted records are shown as unknown, "
                      "never guessed.")
        if selected.get("has_last_vectors", True):
            p.section("it senses (16 inputs)")
            for label, value in sensor_rows(selected):
                p.row(label, value)
            p.section("its network decided (8 outputs)")
            for label, value, active in output_rows(selected):
                p.row(label, value, GOOD if active else TEXT)
            p.wrapped(decision_summary(selected), TEXT_DIM, "small")
        else:
            p.section("it senses / decided")
            p.wrapped("No vector yet: this organism was born after the last "
                      "think step, so there is nothing to show until the "
                      "simulation advances.", TEXT_DIM, "small")
        p.footer_hint("Click empty space or X to clear. F follows it. "
                      "Mouse wheel scrolls this panel.")
        state.content_height = p.content_height()
        screen.set_clip(None)
        _panel_fades(screen, inner, state.scroll, state.content_height)
        _more_hint(screen, inner, state.scroll, state.content_height,
                   fonts, cache)
        _draw_scrollbar(screen, rect, state)
        return

    p.section("legend")
    _draw_legend(p)

    p.section("life")
    p.text(population_text(engine), TEXT, "big")
    p.text("organisms alive", TEXT_DIM, "small")
    p.wrapped(births_deaths_text(engine), TEXT_DIM, "small")
    p.wrapped("roles by lifetime intake: " + role_text(engine), TEXT_DIM, "small")
    note = role_note(engine)
    if note:
        p.wrapped(note, TEXT_FAINT, "small")
    p.gap(4)
    p.text("population, recent frames", TEXT_FAINT, "small")
    _draw_sparkline(p, state.history, (96, 178, 245),
                    engine.population.n_alive)

    p.section("matter (mass is conserved)")
    for label, value in matter_rows(engine):
        p.row(label, value)
    p.section("averages")
    for label, value in average_rows(engine):
        p.row(label, value)

    p.section("environment")
    for label, value in environment_rows(engine):
        p.row(label, value)
    p.footer_hint("V cycles the background layer.")

    p.section("events")
    active = engine.events.active(engine.tick)
    for line in event_lines(engine, max_lines=3):
        p.wrapped(line, WARN if active else TEXT_DIM, "small")

    p.section("controls")
    for line in CONTROL_LINES:
        p.wrapped(line, TEXT_DIM, "small")

    state.content_height = p.content_height()
    screen.set_clip(None)
    _panel_fades(screen, inner, state.scroll, state.content_height)
    _more_hint(screen, inner, state.scroll, state.content_height, fonts, cache)
    _draw_scrollbar(screen, rect, state)


CONTROL_LINES = (
    "SPACE pause · S step one tick · 1/2/3/4 speed 1/2/5/20×",
    "Click an organism to inspect it · X clear · F follow it · P perception",
    "A analytics charts · arrows pan · + / − zoom · HOME reset view",
    "V environment layer · I show/hide this panel · H big help · R restart · Q quit",
)


_FADE_CACHE: dict[tuple, pygame.Surface] = {}


def _fade_edge(screen: pygame.Surface, rect: pygame.Rect, at_bottom: bool,
               colour: tuple[int, int, int], height: int = 26) -> None:
    """Soften a panel edge so scrolled text does not end mid-glyph.

    Without this the last visible line is sliced in half at the panel border,
    which reads as a rendering fault rather than as "there is more below".
    The gradient strip is cached: building a per-pixel surface every frame is
    wasted work on the single-core target machine.
    """
    height = min(height, max(1, rect.height // 3))
    key = (rect.width, height, bool(at_bottom), colour)
    strip = _FADE_CACHE.get(key)
    if strip is None:
        if len(_FADE_CACHE) > 24:
            _FADE_CACHE.clear()
        strip = pygame.Surface((rect.width, height), pygame.SRCALPHA)
        for row in range(height):
            frac = (row + 1) / height
            # 0 at the outer edge of the strip, 1 at the panel border. The
            # square root ramps up fast: a linear ramp left a half-legible
            # slice of the next line right at the border, and that slice is
            # exactly the artefact this is meant to hide.
            t = frac if at_bottom else 1.0 - frac
            strip.fill((*colour, int(255 * float(t) ** 0.5)),
                       pygame.Rect(0, row, rect.width, 1))
        _FADE_CACHE[key] = strip
    y = rect.bottom - height if at_bottom else rect.y
    screen.blit(strip, (rect.x, y))


def _draw_legend(p: Panel) -> None:
    font = p.fonts["small"]
    for label, color, marker in legend_items():
        cy = p.y + font.get_linesize() // 2
        cx = p.rect.x + p._pad + 5
        if marker == "circle":
            pygame.draw.circle(p.screen, color, (cx, cy), 3)
        elif marker == "square":
            pygame.draw.rect(p.screen, color, (cx - 4, cy - 4, 9, 9))
        elif marker == "organism":
            # Same rim-plus-fill the world uses, so the swatch is the sprite.
            rim = (int(color[0] * 0.35), int(color[1] * 0.35),
                   int(color[2] * 0.35))
            pygame.draw.circle(p.screen, rim, (cx, cy), 4)
            pygame.draw.circle(p.screen, color, (cx, cy), 3)
        else:
            pygame.draw.circle(p.screen, color, (cx, cy), 6, 1)
        p.screen.blit(p.cache.render(font, label, TEXT_DIM),
                      (p.rect.x + p._pad + 18, p.y))
        p.y += font.get_linesize()
    p.gap(2)
    # One sentence under the swatches: the colour is a gene, not behaviour, and
    # the background is a layer the badge names. Both are easy to misread.
    p.wrapped("organism colour is its inherited diet gene, not what it ate; "
              "the background is the environmental layer named in the corner.",
              TEXT_FAINT, "small")


def _draw_sparkline(p: Panel, history: list[tuple[int, int]],
                    color: tuple[int, int, int],
                    alive: int | None = None) -> None:
    """Population line over the recent frames; labelled in words, no axes.

    With one or two samples a line means nothing, so the box is skipped and the
    current count is shown instead: the reader gets a fact, not an empty chart.
    ``alive`` is the engine's live count, used when the history is empty (a
    paused viewer on its first frame drew "0 right now" beside "17 organisms
    alive", which reads as a contradiction).
    """
    height = 44
    width = p.rect.width - 2 * p._pad
    x0 = p.rect.x + p._pad
    y0 = p.y
    values = [v for _, v in history]
    if len(values) < 3:
        now = values[-1] if values else (0 if alive is None else int(alive))
        p.text(f"{now} right now", TEXT_DIM, "small")
        p.text("a line appears after a few frames", TEXT_FAINT, "small")
        return
    first, last = values[0], values[-1]
    peak = max(1, max(values))
    n = len(values)
    step = width / max(1, n - 1)
    # The caption is a line of its own: drawn inside the box it collided with
    # the plotted line and with the peak label.
    small = p.fonts["small"]
    change = last - first
    word = "up" if change > 0 else ("down" if change < 0 else "flat")
    caption = p.cache.render(
        small, f"was {first}, now {last} ({word}) · peak {peak}", TEXT_FAINT)
    p.screen.blit(caption, (x0, y0))
    top = y0 + caption.get_height() + 2
    points = [(int(x0 + i * step),
               int(top + height - 3 - (height - 12) * v / peak))
              for i, v in enumerate(values)]
    pygame.draw.rect(p.screen, (30, 34, 41), (x0, top, width, height))
    if len(points) > 1:
        pygame.draw.lines(p.screen, color, False, points, 2)
    p.y = top + height + 5


def _scrollbar(screen: pygame.Surface, rect: pygame.Rect, content: int,
               scroll: int, view: int | None = None) -> None:
    """Thin scroll indicator for a panel whose content exceeds ``view``."""
    view = rect.height if view is None else max(1, int(view))
    if content <= view:
        return
    track = pygame.Rect(rect.right - 8, rect.y + 6, 5, rect.height - 12)
    pygame.draw.rect(screen, (40, 44, 52), track)
    frac = view / max(1, content)
    h = max(28, int(track.height * frac))
    max_scroll = content - view
    frac_pos = min(1.0, max(0.0, scroll / max_scroll)) if max_scroll > 0 else 0.0
    y = track.y + int((track.height - h) * frac_pos)
    pygame.draw.rect(screen, (110, 120, 136), (track.x, y, track.w, h))


def _panel_fades(screen: pygame.Surface, inner: pygame.Rect, scroll: int,
                 content_height: int,
                 colour: tuple[int, int, int] = PANEL_BG) -> None:
    """Fade the text box edges only where content actually continues.

    A half-drawn glyph at the border looks like a rendering bug; a soft edge
    reads as "there is more". The top fade appears only after scrolling.
    """
    if content_height > inner.height:
        if scroll > 0:
            _fade_edge(screen, inner, False, colour)
        if scroll + inner.height < content_height:
            _fade_edge(screen, inner, True, colour)


def _more_hint(screen: pygame.Surface, inner: pygame.Rect, scroll: int,
               content_height: int, fonts: dict, cache: TextCache) -> None:
    """Say "more below" in words, in the gutter under the text box.

    The fade and scrollbar show that something continues, but a first-time
    reader does not necessarily know that the wheel scrolls *this* panel.
    """
    if content_height <= inner.height:
        return
    below = scroll + inner.height < content_height
    above = scroll > 0
    if not (below or above):
        return
    text = ("more below — scroll" if below and not above
            else "more above" if above and not below
            else "more above and below")
    surf = cache.render(fonts["small"], text, TEXT_FAINT)
    screen.blit(surf, (inner.right - surf.get_width(),
                       inner.bottom + 3))


def _draw_scrollbar(screen: pygame.Surface, rect: pygame.Rect, state) -> None:
    view = int(getattr(state, "view_height", 0)) or rect.height
    _scrollbar(screen, rect, max(1, int(getattr(state, "content_height", 0))),
               int(getattr(state, "scroll", 0)), view)


def draw_footer(screen: pygame.Surface, rect: pygame.Rect,
                fonts: dict[str, pygame.font.Font], cache: TextCache,
                state) -> None:
    pygame.draw.rect(screen, PANEL_BG, rect)
    pygame.draw.line(screen, PANEL_EDGE, (rect.x, rect.y),
                     (rect.right, rect.y), 1)
    small = fonts["small"]
    width = rect.width - 20
    lines = []
    full_keys = ("SPACE pause · S step · 1-4 speed 1/2/5/20× · click inspect "
                 "· X clear · F follow · P perception · A charts · I panel "
                 "· V layer · H help · Q quit")
    if small.size(full_keys)[0] <= width:
        lines.append(full_keys)
    else:
        # Narrow window: keep the controls that need no explanation and let the
        # legend drop off, rather than running text under the window edge.
        compact = ("SPACE pause · S step · 1-4 speed · click inspect · X clear "
                   "· A charts · I panel · H help · Q quit")
        lines.append(compact)
    legend = ("Green = plant · brown = carcass · organism colour = inherited "
              "diet gene · background = environment layer · mass is conserved")
    if small.size(legend)[0] <= width:
        lines.append(legend)
    y = rect.y + 5
    for line in lines:
        if y + small.get_linesize() > rect.bottom:
            break
        screen.blit(cache.fit(small, line, TEXT_DIM, width), (10, y))
        y += small.get_linesize()


HELP_BLOCKS = (
    ("Start here", (
        "Left: the world. Right: live readings. Bottom: the keys you can press. "
        "Nothing here is a video — every dot is an individual simulated animal, "
        "and every number on the right is measured from the simulation, not "
        "estimated for show.",
        "The simulation runs while you read this. H or Escape closes the guide; "
        "SPACE then pauses or resumes the run at any time.",
    )),
    ("What am I looking at?", (
        "Green specks are plants, brown lumps are carcasses, and each coloured "
        "dot is an animal. The shaded background is one environmental layer, and "
        "the top-left badge names it and explains its colours; V cycles through "
        "climate, moisture, soil, temperature, terrain, plant biomass, animal "
        "counts and pheromone. The badge prints the numeric range of whatever "
        "layer is showing.",
        "Mass is conserved: plants and animals are built from soil nutrients, and "
        "everything that dies decays back into soil. The side panel lists the "
        "pools so you can watch matter move between them.",
    )),
    ("What do the numbers mean?", (
        "bu is biomass: the same unit for a plant, a carcass, an animal's body "
        "and the soil it decays into, which is what makes the matter ledger "
        "balance. u/tick is world distance per tick. °C and % are as labelled.",
        "'Vision 81 u · 325° cone' means the animal can see 81 world units in a "
        "325-degree arc. 'Attack / defence 0.29 / 1.68' is how hard it hits and "
        "how hard it is to hurt. 'Speed max' is the top speed its body allows; "
        "the resulting speed row shows what it actually asked for this tick.",
    )),
    ("Why do the animals move?", (
        "Each animal carries a genome. Part of it builds a small neural network "
        "(16 senses into 12, then 8, then 8 outputs) and the rest sets physical "
        "traits such as size, speed, vision and diet preference. Every tick the "
        "network reads the senses and decides: forward force, turn, throttle, "
        "eat, attack, sprint, reproduce, rest.",
        "Nothing is scripted and no rule steers an animal toward food or away "
        "from danger. If a lineage survives, it is because its inherited network "
        "happened to work in this world.",
    )),
    ("How do I read one animal?", (
        "Click any dot; a circle and a heading arrow mark it. The panel switches "
        "to that individual: its genes, reserve, health, parents, lifetime intake "
        "and actions, then the exact 16 sensor values its network saw and the 8 "
        "outputs it produced on the last think step. A dash (—) means 'nothing "
        "detected' (the sensor encoding is −1).",
        "The output rows show both the raw score and the decision: a value above "
        "0.5 means yes. Rest overrides eating and attacking, and the rows say so "
        "when that happens. F locks the camera onto the animal, X clears it.",
        "Two rows look similar and answer different questions. 'diet gene' is "
        "the inherited preference that colours the sprite; 'role (from intake)' "
        "is what this animal has actually eaten, by the same rule that produces "
        "the herbivore/omnivore/carnivore counts for the whole population. They "
        "can disagree, and that difference is real.",
    )),
    ("Why did the panel change?", (
        "If the inspected animal dies, the panel returns to the population view "
        "and a note says the animal died — it never silently shows a different "
        "one under the same heading.",
        "The mouse wheel scrolls the side panel when it is longer than the "
        "window; page up and page down do the same from the keyboard.",
    )),
    ("What happens when everything dies?", (
        "Extinction is a normal outcome, not a crash. The simulation keeps its "
        "state and the ledger still balances; the screen shows EXTINCT and R "
        "restarts from the same seed. Different seeds give different fates, and "
        "the experiment documents report both survivors and extinctions.",
    )),
    ("Colours, events and controls", (
        "Organism colour is the inherited diet gene (green = plant-leaning, "
        "yellow = mixed, red = meat-leaning). The panel also counts roles "
        "measured from what each animal actually ate, which can differ from the "
        "gene — that difference is the point of showing both.",
        "Environmental events (drought, rain, heatwave, cold snap, wildfire, "
        "nutrient pulse, blight) are physical: they change fields, health, plants "
        "or soil, never a genome. Active and next-scheduled events are listed in "
        "the panel. A local event is also outlined on the map in its own colour, "
        "and the panel names that colour so you can find the affected patch.",
        "Keys: SPACE pause · S one tick · 1/2/3/4 speed 1/2/5/20× · arrows pan · "
        "+/− zoom · HOME reset view · click inspect · X clear · F follow · "
        "P perception · A analytics · I panel · V layer · H this guide · R "
        "restart · Q quit.",
    )),
    ("Charts, perception and ancestry", (
        "A opens the analytics view: population, births and deaths, biomass "
        "pools, mean traits, genetic diversity, environment, generation depth, "
        "deaths by cause and the age-at-death histogram, all measured from the "
        "running simulation and sampled every few ticks. Every chart prints its "
        "unit and its numeric scale. Mouse wheel scrolls the grid; A returns to "
        "the world. Charts with no samples say so rather than drawing a flat line.",
        "P toggles the perception overlay for the selected animal: the vision "
        "cone and a marker on each target its network actually reported this "
        "tick, reconstructed from the sensor vector itself. A class reported as "
        "'nothing detected' draws no marker, so the overlay cannot show food "
        "where the network saw none.",
        "The inspector lists ancestry from parents recorded at birth and, when "
        "the run logs lineage, the recorded chain further back. Founders have no "
        "recorded parents and evicted records are shown as unknown — ancestry is "
        "never guessed from genome similarity or ID order.",
    )),
)


def draw_help(screen: pygame.Surface, rect: pygame.Rect,
              fonts: dict[str, pygame.font.Font], cache: TextCache,
              scroll: int = 0, content_height: int = 0) -> int:
    """Full-window help modal; returns its content height for scrolling.

    ``rect`` is the whole window, not just the world viewport: the guide is
    modal, so the live panels are shaded out behind it rather than competing
    for attention.
    """
    shade = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    shade.fill((10, 12, 16, 255))   # opaque: the guide is modal
    screen.blit(shade, (rect.x, rect.y))
    # Margins, then a centred text column: long lines are hard to track, and the
    # gutter to the right of the column holds the scrollbar. ``inflate`` already
    # centres, so no extra offset here.
    inner = rect.inflate(-120, -140)
    # ~80 characters per line: at 940 px the guide read as a wall of text and
    # the eye lost its place between lines.
    width = min(inner.w - 60, 760)
    inner = pygame.Rect(inner.x + (inner.w - width) // 2, inner.y,
                        width, inner.h)
    # Clamp with the real text-viewport height, not an approximation, or the
    # last screenful can never be reached.
    max_scroll = max(0, int(content_height) - inner.height)
    scroll = max(0, min(int(scroll), max_scroll))
    # Clip to the text column: without this the next paragraph's first line
    # bleeds past the fade into the bottom margin.
    screen.set_clip(inner)
    panel = Panel(screen, inner, fonts, cache, scroll=scroll)
    panel.title("EVOLVE — a five-minute guide")
    panel.wrapped("H or Escape closes this. Scroll with the mouse wheel.",
                  TEXT_FAINT, "small")
    for heading, paragraphs in HELP_BLOCKS:
        panel.section(heading)
        for text in paragraphs:
            panel.wrapped(text, TEXT_DIM, "small")
            panel.gap(6)
    height = panel.content_height()
    screen.set_clip(None)
    _panel_fades(screen, inner, scroll, height, (10, 12, 16))
    _scrollbar(screen, pygame.Rect(inner.right + 10, inner.y, 12, inner.height),
               height, scroll)
    if height > inner.height:
        hint_text = "scroll for more below"
        if scroll > 0 and scroll + inner.height < height:
            hint_text = "more above and below"
        elif scroll > 0:
            hint_text = "more above"
        hint = cache.render(fonts["small"], hint_text, TEXT_FAINT)
        # Sit the hint in the bottom margin, below the text column: drawn over
        # the last line it looked like a rendering collision.
        screen.blit(hint, (inner.right - hint.get_width(),
                           inner.bottom + 8))
    return height
