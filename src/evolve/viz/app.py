"""Minimal interactive view (DESIGN §10).

Controls: space start/pause, ``s`` single step, ``r`` reset, ``1-4`` speed
1x/2x/5x/20x, ``q``/escape quit, arrows pan, ``+``/``-`` zoom. The simulation
never blocks on rendering: one frame pumps events, advances a fixed number of
steps, then draws.
"""
from __future__ import annotations

import pygame

from ..config import Config
from ..engine import Engine
from .camera import Camera
from .render import draw

_SPEEDS = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 5, pygame.K_4: 20}


class VizApp:
    def __init__(self, engine: Engine, speed: int = 1,
                 size: tuple[int, int] = (900, 700), title: str = "EVOLVE") -> None:
        pygame.display.init()
        pygame.font.init()
        self.engine = engine
        self.speed = int(speed)
        self.running = True
        self.paused = False
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.camera = Camera((engine.world.width, engine.world.height), size)
        self.font = pygame.font.Font(None, 20)

    # -- input ----------------------------------------------------------------
    def handle_key(self, key: int) -> None:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.running = False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_s:
            self.engine.step()
        elif key == pygame.K_r:
            self.engine = Engine(self.engine.cfg)
        elif key in _SPEEDS:
            self.speed = _SPEEDS[key]
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            self.camera.zoom_by(1.25)
        elif key == pygame.K_MINUS:
            self.camera.zoom_by(0.8)
        elif key == pygame.K_LEFT:
            self.camera.pan(-40.0, 0.0)
        elif key == pygame.K_RIGHT:
            self.camera.pan(40.0, 0.0)
        elif key == pygame.K_UP:
            self.camera.pan(0.0, -40.0)
        elif key == pygame.K_DOWN:
            self.camera.pan(0.0, 40.0)

    # -- frame loop -----------------------------------------------------------
    def run_frame(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_key(event.key)
        if not self.paused:
            for _ in range(self.speed):
                self.engine.step()
        draw(self.engine, self.screen, self.camera, self.font)
        pygame.display.flip()
        self.clock.tick(60)

    def run(self) -> None:
        while self.running:
            self.run_frame()

    # -- output ---------------------------------------------------------------
    def screenshot(self, path: str) -> None:
        pygame.image.save(self.screen, path)


def launch(config: Config) -> None:
    VizApp(Engine(config)).run()
