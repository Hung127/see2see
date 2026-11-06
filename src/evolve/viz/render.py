"""Rendering: terrain background, plants, organisms and a HUD line."""
from __future__ import annotations

import pygame

from ..engine import Engine
from .camera import Camera

_BG = (16, 18, 22)
_HUD = (220, 220, 220)


def _terrain_color(t: float, m: float, soil: float) -> tuple[int, int, int]:
    """Cheap visual encoding: moisture -> green, temperature -> red shift."""
    base = 40 + int(60 * min(max(m, 0.0), 1.0))
    warm = int(40 * min(max((t - 20.0) / 15.0, 0.0), 1.0))
    rich = int(30 * min(max(soil / 6.0, 0.0), 1.0))
    return (base + warm, base + rich, base)


def _diet_color(d: float) -> tuple[int, int, int]:
    """Herbivore green -> omnivore yellow -> carnivore red (observable phenotype)."""
    d = min(max(d, 0.0), 1.0)
    return (int(40 + 200 * d), int(200 - 120 * d), 60)


def draw(engine: Engine, screen: pygame.Surface, camera: Camera,
         font: pygame.font.Font | None = None) -> None:
    screen.fill(_BG)
    fields = engine.fields
    cell = fields.cell_size
    for iy in range(fields.n):
        for ix in range(fields.n):
            color = _terrain_color(fields.temperature[iy, ix], fields.moisture[iy, ix],
                                   fields.soil[iy, ix])
            x0, y0 = camera.to_screen(ix * cell, iy * cell)
            x1, y1 = camera.to_screen((ix + 1) * cell, (iy + 1) * cell)
            pygame.draw.rect(screen, color, (x0, y0, x1 - x0, y1 - y0))

    plants = engine.plants
    idx = plants.alive.nonzero()[0]
    for i in idx:
        sx, sy = camera.to_screen(plants.x[i], plants.y[i])
        pygame.draw.circle(screen, (60, 160, 70), (sx, sy), camera.scale(1.5))

    pop = engine.population
    oidx = pop.alive.nonzero()[0]
    phen = pop.phenotypes()
    for row, i in enumerate(oidx):
        sx, sy = camera.to_screen(pop.x[i], pop.y[i])
        radius = camera.scale(max(1.0, 2.0 * phen[row, 0] ** 0.5))
        pygame.draw.circle(screen, _diet_color(phen[row, 9]), (sx, sy), radius)

    if font is not None:
        hud = (f"tick {engine.tick}   pop {pop.n_alive}   plants {plants.n_alive}   "
               f"detritus {engine.detritus:.1f}   zoom {camera.zoom:.1f}x")
        screen.blit(font.render(hud, True, _HUD), (8, 6))
