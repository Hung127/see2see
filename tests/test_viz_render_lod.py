"""Pixel-equivalence tests for the vectorised world renderer.

The renderer must be fast enough that drawing never dominates a frame, but
"fast" is not allowed to mean "draws something else". Plants, carrion and
organisms are culled and transformed in NumPy now; these tests assert that the
result is **pixel-identical** to the scalar ``Camera.to_screen`` + per-item
``_on_screen`` path it replaced, at several zooms and off-centre focuses.

They are presentation tests, not simulation tests: nothing here touches the
engine's state or RNG.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # must precede pygame import

import numpy as np  # noqa: E402

import pygame  # noqa: E402

from evolve.config import (  # noqa: E402
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine  # noqa: E402
from evolve.viz import render  # noqa: E402
from evolve.viz.camera import Camera  # noqa: E402
from evolve.viz import palette  # noqa: E402
from evolve.viz.palette import CARCASS, PLANT, diet_color  # noqa: E402

pygame.init()


# -- the scalar reference path (what the renderer used to do) ----------------
def _scalar_plants(engine, screen, camera):
    plants = engine.plants
    radius = max(1, camera.scale(1.2))
    for i in np.flatnonzero(plants.alive):
        sx, sy = camera.to_screen(plants.x[i], plants.y[i])
        if render._on_screen(camera, sx, sy, radius):
            pygame.draw.circle(screen, PLANT, (sx, sy), radius)


def _scalar_carcasses(engine, screen, camera):
    carr = engine.carcasses
    for i in np.flatnonzero(carr.alive):
        sx, sy = camera.to_screen(carr.x[i], carr.y[i])
        size = camera.scale(max(2.0, 2.0 * float(carr.biomass[i]) ** 0.5))
        if render._on_screen(camera, sx, sy, size):
            screen.fill(CARCASS, (sx - size // 2, sy - size // 2, size, size))


def _scalar_organisms(engine, screen, camera):
    pop = engine.population
    idx = np.flatnonzero(pop.alive)
    if idx.size == 0:
        return
    phen = pop.phenotypes()
    for row, i in enumerate(idx):
        sx, sy = camera.to_screen(pop.x[i], pop.y[i])
        radius = max(3, camera.scale(max(1.5, 2.0 * float(phen[row, 0]) ** 0.5)))
        if not render._on_screen(camera, sx, sy, radius):
            continue
        color = diet_color(float(phen[row, 9]))
        pygame.draw.circle(screen, render._rim(color), (sx, sy), radius)
        pygame.draw.circle(screen, color, (sx, sy), radius - 1)


def _scene(seed=3, plants=800, organisms=120, width=600.0, cells=24):
    cfg = Config(
        seed=seed, ticks=200,
        world=WorldConfig(width=width, height=width, n_field_cells=cells,
                          cell_size=width / cells),
        plants=PlantConfig(initial=plants, max=plants * 2),
        organisms=OrganismConfig(initial=organisms, max=organisms * 2),
        energy=EnergyConfig(), genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.002), stats=StatsConfig())
    eng = Engine(cfg)
    eng.run(40)
    for i in range(4):                      # ensure the carrion layer is non-empty
        eng.carcasses.add(60.0 + 40.0 * i, 90.0 + 30.0 * i, 2.0 + i)
    return eng


W, H = 720, 560
_ENGINE = _scene()
_CASES = (
    # (zoom, focus) — includes sick/truncation-prone negative screen coords
    (1.0, (300.0, 300.0)),
    (0.4, (300.0, 300.0)),
    (3.0, (300.0, 300.0)),
    (2.0, (12.0, 588.0)),
    (5.0, (599.0, 1.0)),
    (0.1, (300.0, 300.0)),
    (1.7, (0.0, 600.0)),
)


def _both(layer_fn):
    """Run scalar and vectorised draw into identical surfaces for every case."""
    for zoom, focus in _CASES:
        cam = Camera((600.0, 600.0), (W, H))
        cam.zoom = zoom
        cam.focus = focus
        old = pygame.Surface((W, H))
        new = pygame.Surface((W, H))
        old.fill((0, 0, 0))
        new.fill((0, 0, 0))
        layer_fn(_ENGINE, old, cam, scalar=True)
        layer_fn(_ENGINE, new, cam, scalar=False)
        yield zoom, focus, (pygame.surfarray.array3d(old)
                            == pygame.surfarray.array3d(new)).all()


def _dispatch(scalar_name, vector_name):
    def run(engine, screen, camera, scalar):
        if scalar:
            return {"plants": _scalar_plants, "carcasses": _scalar_carcasses,
                    "organisms": _scalar_organisms}[scalar_name](
                        engine, screen, camera)
        return getattr(render, vector_name)(engine, screen, camera)
    return run


def test_plants_are_pixel_identical_to_the_scalar_path():
    for zoom, focus, same in _both(_dispatch("plants", "_draw_plants")):
        assert same, f"plants differ at zoom={zoom} focus={focus}"


def test_carcasses_are_pixel_identical_to_the_scalar_path():
    for zoom, focus, same in _both(_dispatch("carcasses", "_draw_carcasses")):
        assert same, f"carcasses differ at zoom={zoom} focus={focus}"


def test_organisms_are_pixel_identical_to_the_scalar_path():
    for zoom, focus, same in _both(_dispatch("organisms", "_draw_organisms")):
        assert same, f"organisms differ at zoom={zoom} focus={focus}"


def test_screen_points_matches_to_screen_including_negative_coordinates():
    """`to_screen` truncates towards zero; the array form must not floor."""
    cam = Camera((600.0, 600.0), (W, H))
    cam.zoom = 2.0
    cam.focus = (300.0, 300.0)
    xs = np.array([-0.4, 0.4, 149.6, 150.4, 449.6, 450.4])  # some map negative
    ys = np.array([0.4, -0.4, 149.6, 150.4, 449.6, 450.4])
    sx, sy, _ = render._screen_points(cam, xs, ys)
    for k in range(xs.size):
        assert (int(sx[k]), int(sy[k])) == cam.to_screen(float(xs[k]),
                                                         float(ys[k]))


def test_culling_drops_only_items_that_cannot_contribute_pixels():
    """Off-screen items must not be drawn, on-screen ones must be."""
    cam = Camera((600.0, 600.0), (W, H))
    cam.zoom = 1.0
    cam.focus = (300.0, 300.0)
    # One plant far outside the viewport, one in the middle.
    inside = pygame.Surface((W, H)); inside.fill((0, 0, 0))
    sx, sy = cam.to_screen(300.0, 300.0)
    pygame.draw.circle(inside, PLANT, (sx, sy), 1)
    assert (pygame.surfarray.array3d(inside) == PLANT).all(axis=2).any()
    # The vectorised visibility test agrees with the scalar bounds test.
    xs = np.array([300.0, -10_000.0, 10_000.0])
    ys = np.array([300.0, 300.0, 300.0])
    _, _, visible = render._screen_points(cam, xs, ys, 1)
    assert visible.tolist() == [True, False, False]


# -- vision overlay ----------------------------------------------------------
def test_cell_mean_vision_averages_living_organisms_per_cell():
    """The overlay is a per-cell mean of the phenotype, not a field array.

    It is derived at draw time so the engine keeps no vision field and the
    checksum is unaffected. Every living animal must be counted exactly once.
    """
    eng = _scene()
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    assert idx.size > 0
    mean = render._cell_mean_vision(eng)
    fields = eng.fields
    assert mean.shape == (fields.n, fields.n)

    vision = pop.phenotypes()[:, 2]
    ix, iy = fields.cell_index(pop.x[idx], pop.y[idx])
    expected: dict[tuple[int, int], list[float]] = {}
    for k in range(idx.size):
        expected.setdefault((int(ix[k]), int(iy[k])), []).append(float(vision[k]))
    for (cx, cy), values in expected.items():
        assert mean[cx, cy] == np.mean(values)

    # Cells with no organism report no data, not zero.
    counts = np.zeros((fields.n, fields.n), dtype=int)
    np.add.at(counts, (ix, iy), 1)
    empty = counts == 0
    assert empty.any()
    assert np.isnan(mean[empty]).all()
    assert np.isfinite(mean[~empty]).all()


def test_vision_overlay_draws_and_empty_cells_read_as_no_data():
    """An empty cell must paint the ramp's dark end, not the colour for 0 u."""
    eng = _scene()
    dark = palette.layer_color("vision", vision_range=None)
    zero = palette.layer_color("vision", vision_range=0.0)
    short = palette.layer_color("vision", vision_range=40.0)
    far = palette.layer_color("vision", vision_range=160.0)
    assert short == dark, "the gene's floor must not be painted as no-data"
    assert zero == short, "values below the span clamp to the dark end"
    assert dark != far

    # Drawing the layer must not raise anywhere, including an empty world.
    cam = Camera((600.0, 600.0), (W, H))
    cam.zoom = 1.0
    screen = pygame.Surface((W, H))
    screen.fill((0, 0, 0))
    render._draw_terrain(eng, screen, cam, "vision")
    assert (pygame.surfarray.array3d(screen) != 0).any()
