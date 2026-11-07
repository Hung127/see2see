"""World rendering: terrain layer, plants, carcasses and organisms."""
from __future__ import annotations

import numpy as np
import pygame

from ..engine import Engine
from .camera import Camera
from .palette import (
    BG, CARCASS, EVENT_COLORS, EVENT_DEFAULT, PLANT, SELECTION, WORLD_EDGE,
    diet_color, layer_color,
)


def _cell_range(camera: Camera, n: int, cell: float) -> tuple[int, int, int, int]:
    """Field-cell index box ``(ix0, ix1, iy0, iy1)`` covering the viewport.

    Cells outside the viewport are skipped, so zooming in costs less, not more.
    """
    half_w = camera.screen_w / (2.0 * camera.zoom)
    half_h = camera.screen_h / (2.0 * camera.zoom)
    fx, fy = camera.focus
    ix0 = max(0, int(np.floor((fx - half_w) / cell)) - 1)
    ix1 = min(n, int(np.ceil((fx + half_w) / cell)) + 1)
    iy0 = max(0, int(np.floor((fy - half_h) / cell)) - 1)
    iy1 = min(n, int(np.ceil((fy + half_h) / cell)) + 1)
    return ix0, max(ix0 + 1, ix1), iy0, max(iy0 + 1, iy1)


def draw_world(engine: Engine, screen: pygame.Surface, camera: Camera,
               layer: str = "climate") -> None:
    """Terrain grid plus plants, carcasses and organisms (no text)."""
    screen.fill(BG)
    fields = engine.fields
    cell = fields.cell_size
    ix0, ix1, iy0, iy1 = _cell_range(camera, fields.n, cell)

    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            color = layer_color(layer, float(fields.temperature[ix, iy]),
                                float(fields.moisture[ix, iy]),
                                float(fields.soil[ix, iy]),
                                float(fields.terrain[ix, iy]),
                                plant_density=float(fields.plant_density[ix, iy]),
                                organism_density=float(fields.organism_density[ix, iy]),
                                pheromone=float(fields.pheromone[ix, iy]))
            x0, y0 = camera.to_screen(ix * cell, iy * cell)
            x1, y1 = camera.to_screen((ix + 1) * cell, (iy + 1) * cell)
            pygame.draw.rect(screen, color, (x0, y0, x1 - x0 + 1, y1 - y0 + 1))

    _draw_plants(engine, screen, camera)
    _draw_carcasses(engine, screen, camera)
    _draw_organisms(engine, screen, camera)
    # Event outlines sit above the organisms: they are an annotation about the
    # area, and hiding them behind a single animal would defeat the point.
    _draw_events(engine, screen, camera)
    # Thin outline of the simulated area, so the dark margin reads as "outside
    # the world" rather than as part of it.
    x0, y0 = camera.to_screen(0.0, 0.0)
    x1, y1 = camera.to_screen(engine.world.width, engine.world.height)
    pygame.draw.rect(screen, WORLD_EDGE, (x0, y0, x1 - x0, y1 - y0), 1)


def _on_screen(camera: Camera, sx: int, sy: int, margin: int) -> bool:
    return (-margin <= sx <= camera.screen_w + margin
            and -margin <= sy <= camera.screen_h + margin)


def _rim(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """A darker version of a body colour, used as a one-pixel outline."""
    return (int(color[0] * 0.35), int(color[1] * 0.35), int(color[2] * 0.35))


def _draw_plants(engine: Engine, screen: pygame.Surface, camera: Camera) -> None:
    plants = engine.plants
    idx = np.flatnonzero(plants.alive)
    if idx.size == 0:
        return
    radius = max(1, camera.scale(1.2))
    for i in idx:
        sx, sy = camera.to_screen(plants.x[i], plants.y[i])
        if _on_screen(camera, sx, sy, radius):
            pygame.draw.circle(screen, PLANT, (sx, sy), radius)


def _draw_carcasses(engine: Engine, screen: pygame.Surface, camera: Camera) -> None:
    carr = engine.carcasses
    idx = np.flatnonzero(carr.alive)
    for i in idx:
        sx, sy = camera.to_screen(carr.x[i], carr.y[i])
        size = camera.scale(max(2.0, 2.0 * float(carr.biomass[i]) ** 0.5))
        if _on_screen(camera, sx, sy, size):
            screen.fill(CARCASS, (sx - size // 2, sy - size // 2, size, size))


def _draw_organisms(engine: Engine, screen: pygame.Surface, camera: Camera) -> None:
    pop = engine.population
    idx = np.flatnonzero(pop.alive)
    if idx.size == 0:
        return
    phen = pop.phenotypes()
    for row, i in enumerate(idx):
        sx, sy = camera.to_screen(pop.x[i], pop.y[i])
        # At least 3 px: a 1-px dot with a rim is indistinguishable from a
        # plant speck, and telling "alive" from "plant" is the first thing a
        # beginner needs to do.
        radius = max(3, camera.scale(max(1.5, 2.0 * float(phen[row, 0]) ** 0.5)))
        if not _on_screen(camera, sx, sy, radius):
            continue
        color = diet_color(float(phen[row, 9]))
        # Dark rim: without it a herbivore-leaning animal is the same green
        # blob as a plant, and a beginner cannot tell which specks are alive.
        pygame.draw.circle(screen, _rim(color), (sx, sy), radius)
        pygame.draw.circle(screen, color, (sx, sy), radius - 1)


def _draw_events(engine: Engine, screen: pygame.Surface, camera: Camera) -> None:
    """Outline the area each active environmental event is acting on.

    The panel can name an event, but not *where* it is; without a mark the
    reader sees plants dying in one corner with no stated cause. The disc is the
    same ``region`` the scheduler applies, so the outline cannot drift from the
    effect. Worldwide events have a huge radius and no meaningful outline.
    """
    events = getattr(engine, "events", None)
    if events is None:
        return
    for ev in events.active(engine.tick):
        cx, cy, radius = ev.region
        if radius >= 1e8:
            continue                      # worldwide: no local area to outline
        sx, sy = camera.to_screen(cx, cy)
        r = camera.scale(radius)
        if r < 2 or r > 6 * max(camera.screen_w, camera.screen_h):
            continue
        colour = EVENT_COLORS.get(ev.kind, EVENT_DEFAULT)
        pygame.draw.circle(screen, colour, (sx, sy), r, 2)


def draw_selection(engine: Engine, screen: pygame.Surface, camera: Camera,
                   organism_id: int | None) -> None:
    """Ring the selected organism and show which way it faces.

    The heading tick is drawn from the same ``pop.heading`` the simulation
    steers with, so the arrow cannot disagree with the motion.
    """
    if organism_id is None:
        return
    pop = engine.population
    idx = np.flatnonzero(pop.alive)
    if idx.size == 0:
        return
    hits = np.flatnonzero(pop.id[idx] == int(organism_id))
    if hits.size == 0:
        return
    i = int(idx[int(hits[0])])
    sx, sy = camera.to_screen(pop.x[i], pop.y[i])
    radius = camera.scale(max(6.0, 4.0 * float(pop.body[i]) ** 0.5))
    pygame.draw.circle(screen, SELECTION, (sx, sy), radius, 2)
    # Heading tick, drawn from the same ``pop.heading`` used by the motion
    # update, so it cannot disagree with where the animal actually moves.
    h = float(pop.heading[i])
    ca, sa = float(np.cos(h)), float(np.sin(h))
    tip = (sx + ca * (radius + 9), sy + sa * (radius + 9))
    pygame.draw.line(screen, SELECTION, (sx + ca * (radius + 2),
                                         sy + sa * (radius + 2)), tip, 2)
    # Small arrowhead: two barbs swept back from the tip.
    for side in (-0.45, 0.45):
        back = (tip[0] - np.cos(h + side) * 6.0,
                tip[1] - np.sin(h + side) * 6.0)
        pygame.draw.line(screen, SELECTION, tip, back, 2)


def draw_perception(engine: Engine, screen: pygame.Surface, camera: Camera,
                    payload: dict | None) -> None:
    """Vision cone and the targets the network actually detected this tick.

    The geometry is *reconstructed from the sensor vector*, not re-run against
    the world: closeness ``c`` is ``1 - d/r`` and bearing ``b`` is ``θ/π``, so
    the inverse is exact for the nearest visible target of each class. That
    means the marks cannot disagree with what the network saw, and a class with
    the "nothing detected" code (−1) draws no mark at all rather than a mark at
    distance zero.

    Nothing here is a second perception model: if the sensor vector is missing
    (an organism born after the last think step) the overlay draws only the
    cone and says nothing about targets.
    """
    if payload is None:
        return
    phen = np.asarray(payload.get("phenotypes", np.zeros(12)), dtype=np.float64)
    if phen.size < 4:
        return
    x, y = payload.get("position", (0.0, 0.0))
    heading = float(payload.get("heading", 0.0))
    rvis = float(phen[2])
    half = np.deg2rad(float(phen[3])) / 2.0
    sx, sy = camera.to_screen(x, y)
    px_radius = camera.scale(rvis)
    if px_radius < 2:
        return
    # Cone as a faint polyline: three points plus the apex reads as a wedge
    # without filling the world with a translucent shape.
    colour = (120, 180, 240)
    steps = 12
    points = [(sx, sy)]
    for k in range(steps + 1):
        ang = heading - half + (2.0 * half) * k / steps
        points.append((int(sx + np.cos(ang) * px_radius),
                       int(sy + np.sin(ang) * px_radius)))
    pygame.draw.lines(screen, colour, False, points, 1)

    if not payload.get("has_last_vectors", False):
        return
    sensors = np.asarray(payload.get("sensors", np.zeros(16)), dtype=np.float64)
    # (closeness index, bearing index, label, colour). Bearing index is the
    # channel immediately after closeness except for the mate channel (6 has no
    # bearing channel of its own; 7 is own energy).
    classes = ((0, 1, "food", (126, 226, 136)),
               (2, 3, "threat", (236, 96, 88)),
               (4, 5, "prey", (245, 194, 92)),
               (6, None, "mate", (232, 128, 220)))
    for close_i, bearing_i, _label, colour in classes:
        closeness = float(sensors[close_i]) if close_i < sensors.size else -1.0
        if closeness <= -0.999:
            continue                       # "nothing detected", not "at zero"
        dist = max(0.0, (1.0 - closeness)) * rvis
        bearing = 0.0
        if bearing_i is not None and bearing_i < sensors.size:
            bearing = float(sensors[bearing_i]) * np.pi
        ang = heading + bearing
        tx = int(sx + np.cos(ang) * camera.scale(dist))
        ty = int(sy + np.sin(ang) * camera.scale(dist))
        # A dashed tick plus a hollow ring: the plant/carcass markers are
        # filled, so a detection mark cannot be confused with an entity.
        pygame.draw.circle(screen, colour, (tx, ty), 5, 1)
        pygame.draw.line(screen, colour, (tx - 7, ty), (tx - 2, ty), 1)
        pygame.draw.line(screen, colour, (tx + 2, ty), (tx + 7, ty), 1)
        pygame.draw.line(screen, colour, (tx, ty - 7), (tx, ty - 2), 1)
        pygame.draw.line(screen, colour, (tx, ty + 2), (tx, ty + 7), 1)


def draw(engine: Engine, screen: pygame.Surface, camera: Camera,
         font: pygame.font.Font | None = None, layer: str = "climate") -> None:
    """Backwards-compatible single call: world only, optional one-line HUD."""
    draw_world(engine, screen, camera, layer=layer)
    if font is not None:
        hud = (f"tick {engine.tick}   pop {engine.population.n_alive}   "
               f"plants {engine.plants.n_alive}   "
               f"detritus {engine.detritus:.1f}")
        screen.blit(font.render(hud, True, (220, 220, 220)), (8, 6))
