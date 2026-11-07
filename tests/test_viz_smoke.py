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


def test_launched_viewer_has_lineage_logging():
    """`viz` must attach the recorder the ancestry view claims to read.

    The view's own text says logging is on in the viewer. A wiring change that
    dropped the recorder would leave that sentence untrue, so the claim is
    tested rather than assumed.
    """
    from evolve.viz.app import make_app

    app = make_app(small_config())
    assert app.engine.lineage is not None
    assert app.engine.lineage.record(1) is None   # nothing fabricated up front
    app.run_frame()
    assert app.engine.tick == 1


def test_lineage_view_does_not_change_the_checksum():
    """Attaching the viewer's recorder must leave the run bit-identical."""
    from evolve.viz.app import make_app

    plain = Engine(small_config())
    for _ in range(40):
        plain.step()

    app = make_app(small_config(), speed=1)
    for _ in range(40):
        app.run_frame()

    assert app.engine.tick == plain.tick
    assert app.engine.checksum() == plain.checksum()


def test_run_label_is_display_text_only():
    """The name shown for the loaded run must not reach the simulation.

    The sidebar names the run so a screenshot says which experiment it is. That
    value is presentation state: setting it must leave the engine untouched, so
    two viewers differing only in the label produce the same checksum.
    """
    from evolve.viz.app import make_app

    a = make_app(small_config(), speed=1)
    b = make_app(small_config(), speed=1)
    a.state.run_label = "experiment_a.yaml"
    assert b.state.run_label == ""
    for _ in range(10):
        a.run_frame()
        b.run_frame()
    assert a.engine.checksum() == b.engine.checksum()
    assert a.state.run_label == "experiment_a.yaml"
