"""Headless smoke tests for the minimal Pygame view."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # must precede pygame import

import pygame  # noqa: E402

from evolve.config import (  # noqa: E402
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine  # noqa: E402
from evolve.viz.app import VizApp  # noqa: E402


def small_config():
    return Config(
        seed=1, ticks=100,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=10, cell_size=50.0),
        plants=PlantConfig(initial=100, max=300),
        organisms=OrganismConfig(initial=40, max=100),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.0),
        stats=StatsConfig(),
    )


def test_headless_frame(tmp_path):
    eng = Engine(small_config())
    app = VizApp(eng, speed=1)
    for _ in range(60):
        app.run_frame()
    assert eng.tick == 60
    shot = tmp_path / "shot.png"
    app.screenshot(str(shot))
    data = shot.read_bytes()
    assert len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_keys_control_speed_and_pause():
    eng = Engine(small_config())
    app = VizApp(eng, speed=1)
    before = eng.tick
    app.handle_key(pygame.K_SPACE)   # pause
    app.run_frame()
    assert eng.tick == before        # paused: no steps
    app.handle_key(pygame.K_SPACE)   # resume
    app.handle_key(pygame.K_3)       # 5x speed
    app.run_frame()
    assert eng.tick == before + 5
    app.handle_key(pygame.K_s)       # single step
    assert eng.tick == before + 6


def test_camera_maps_world_to_screen():
    from evolve.viz.camera import Camera
    cam = Camera(world_size=(500.0, 500.0), screen_size=(250, 250))
    sx, sy = cam.to_screen(250.0, 250.0)
    assert (sx, sy) == (125, 125)
    cam.zoom_by(2.0)
    sx, sy = cam.to_screen(250.0, 250.0)
    assert (sx, sy) == (125, 125)  # centre stays centred
    sx, sy = cam.to_screen(0.0, 0.0)
    assert (sx, sy) == (125 - 500, 125 - 500)  # zoomed out from centre
