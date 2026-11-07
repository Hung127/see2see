"""Tests for the beginner-facing presentation layer (panels, palette, camera).

These are presentation-only checks: they assert that the readouts are derived
from the *documented* index mapping and that the view helpers behave, not that
any particular colour "looks nice".
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # must precede pygame import

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pygame  # noqa: E402

from evolve.config import (  # noqa: E402
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine  # noqa: E402
from evolve.viz import palette  # noqa: E402
from evolve.viz import panels  # noqa: E402
from evolve.viz.camera import Camera  # noqa: E402

pygame.init()
pygame.font.init()
_FONT = pygame.font.Font(None, 22)


def small_config(seed=1, organisms=40):
    return Config(
        seed=seed, ticks=100,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=10,
                          cell_size=50.0),
        plants=PlantConfig(initial=100, max=300),
        organisms=OrganismConfig(initial=organisms, max=100),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.0),
        stats=StatsConfig(),
    )


# -- palette -----------------------------------------------------------------
def test_next_layer_cycles_and_recovers():
    seen = []
    layer = palette.LAYERS[0]
    for _ in range(len(palette.LAYERS)):
        seen.append(layer)
        layer = palette.next_layer(layer)
    assert seen == list(palette.LAYERS)
    assert layer == palette.LAYERS[0]
    assert palette.next_layer("not-a-layer") == palette.LAYERS[0]


def test_layer_colors_are_bounded_and_distinct():
    for name in palette.LAYERS:
        cs = [palette.layer_color(name, t, m, s, h)
              for t, m, s, h in ((-30.0, 0.0, 0.0, 0.0),
                                 (20.0, 0.5, 3.0, 0.5),
                                 (60.0, 1.0, 6.0, 1.0))]
        for c in cs:
            assert len(c) == 3
            assert all(isinstance(v, int) and 0 <= v <= 255 for v in c), (name, c)
        assert cs[0] != cs[2], name
    # Unknown layer must still return a valid colour (falls back to climate).
    assert palette.layer_color("bogus", 20.0, 0.5, 3.0, 0.5) == \
        palette.layer_color("climate", 20.0, 0.5, 3.0, 0.5)


def test_diet_color_spans_green_to_red_and_words():
    green = palette.diet_color(0.0)
    red = palette.diet_color(1.0)
    assert green[1] > green[0] and red[0] > red[1]
    assert palette.diet_word(0.1) == "herbivore-leaning"
    assert palette.diet_word(0.5) == "mixed"
    assert palette.diet_word(0.9) == "carnivore-leaning"


def test_legend_uses_the_same_colors_as_the_world():
    items = {label: colour for label, colour, _ in panels.legend_items()}
    assert items["herbivore-leaning"] == palette.diet_color(0.05)
    assert items["mixed diet gene"] == palette.diet_color(0.5)
    assert items["carnivore-leaning"] == palette.diet_color(0.95)
    assert items["plant"] == palette.PLANT
    assert items["carcass (meat)"] == palette.CARCASS


# -- camera ------------------------------------------------------------------
def test_camera_to_world_inverts_to_screen():
    cam = Camera((800.0, 400.0), (320, 240))
    cam.zoom_by(1.7)
    cam.pan(120.0, -60.0)
    for wx, wy in ((0.0, 0.0), (123.0, 45.0), (799.0, 399.0)):
        sx, sy = cam.to_screen(wx, wy)
        bx, by = cam.to_world(sx, sy)
        # Round-tripping is exact to the pixel, i.e. within 1/zoom world units.
        assert abs(bx - wx) <= 1.0 / cam.zoom + 1e-9
        assert abs(by - wy) <= 1.0 / cam.zoom + 1e-9


def test_camera_fit_shows_the_whole_world():
    cam = Camera((1000.0, 1000.0), (400, 500))
    cam.zoom_by(6.0)
    cam.fit()
    assert cam.zoom == pytest.approx(0.99 * 400.0 / 1000.0)
    # All four corners land inside the viewport.
    for wx, wy in ((0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0)):
        sx, sy = cam.to_screen(wx, wy)
        assert -1 <= sx <= 401 and -1 <= sy <= 501


def test_camera_centre_on_pins_a_point_to_the_middle():
    cam = Camera((600.0, 600.0), (200, 200))
    cam.centre_on(123.0, 456.0)
    assert cam.to_screen(123.0, 456.0) == (100, 100)


# -- sensor / output formatting ---------------------------------------------
def test_sensor_labels_match_the_code_channel_count():
    assert len(panels.SENSOR_LABELS) == 16
    assert len(panels.OUTPUT_LABELS) == 8
    # Index 7 is own energy, not a mate bearing: the mate channel is 6 only.
    assert panels.SENSOR_LABELS[6] == "mate closeness"
    assert panels.SENSOR_LABELS[7] == "energy"


def test_sensor_rows_show_dash_for_nothing_detected():
    payload = {"sensors": np.full(16, -1.0)}
    rows = dict(panels.sensor_rows(payload))
    assert "nothing in view" in rows["food closeness"]
    assert rows["food bearing"] == "—"
    assert rows["prey bearing"] == "—"


def test_sensor_rows_decode_documented_normalisations():
    sensors = np.full(16, -1.0)
    sensors[0], sensors[1] = 0.5, 0.5        # half close, 90 deg to the right
    sensors[7] = 0.25                         # 25% of E_max
    sensors[10] = (25.0 - 20.0) / 15.0        # 25 C
    sensors[11] = 0.0                         # moisture 0.5
    sensors[13] = (2 * 4 / 20.0) - 1.0        # 4 organisms in the cell
    rows = dict(panels.sensor_rows({"sensors": sensors}))
    assert rows["food closeness"] == "50% close"
    assert rows["food bearing"].startswith("+90")
    # World +y is screen-down, so a positive bearing is clockwise = right.
    assert "right" in rows["food bearing"]
    assert rows["energy"] == "25%"
    assert rows["temperature"] == "25.0 °C"
    assert rows["moisture"] == "50%"
    assert rows["organisms in cell"].startswith("4 here")


def test_bearing_side_matches_world_geometry():
    """A target at larger y must read as 'right', matching the drawn heading."""
    from evolve.world import World
    from evolve.config import WorldConfig
    w = World(WorldConfig(width=500.0, height=500.0, n_field_cells=10,
                          cell_size=50.0))
    heading = np.array([0.0])          # facing +x (screen right)
    b = w.bearing(np.array([100.0]), np.array([100.0]),
                  np.array([150.0]), np.array([150.0]), heading)
    assert b[0] > 0                    # +y is screen-down
    rows = dict(panels.sensor_rows({"sensors": np.array(
        [0.5, b[0] / np.pi] + [-1.0] * 14)}))
    assert "right" in rows["food bearing"]
    # And the sign of the turn control is reported the same way.
    outs = np.zeros(8)
    outs[1] = 0.8
    shown = dict((label, text) for label, text, _ in panels.output_rows(
        {"outputs": outs, "phenotypes": np.zeros(12)}))
    assert "right" in shown["turn"] or "clockwise" in shown["turn"]


def test_output_rows_mark_rest_as_blocking_eat_and_attack():
    outputs = np.zeros(8)
    outputs[3] = 0.9      # wants to eat
    outputs[4] = 0.9      # wants to attack
    outputs[7] = 0.9      # but is resting
    phen = np.zeros(12)
    phen[1] = 10.0        # speed_max
    rows = dict((label, (text, active))
                for label, text, active in panels.output_rows(
                    {"outputs": outputs, "phenotypes": phen}))
    assert "blocked by rest" in rows["eat"][0]
    assert "blocked by rest" in rows["attack"][0]
    assert rows["eat"][1] is False and rows["attack"][1] is False
    assert rows["rest"] == ("RESTING", True)
    # Rest also scales throttle down to nothing, matching actions.decode.
    assert rows["resulting speed"][0].startswith("0.00")


def test_output_rows_speed_matches_action_decoding():
    from evolve.actions import decode
    rng = np.random.default_rng(0)
    for _ in range(20):
        o = rng.uniform(0.0, 1.0, size=8)
        phen = np.zeros(12)
        phen[0] = 1.0
        phen[1] = 12.0
        acts = decode(o[None, :], phen[None, :])
        shown = dict((label, (text, active))
                     for label, text, active in panels.output_rows(
                         {"outputs": o, "phenotypes": phen}))
        assert shown["resulting speed"][0] == \
            f"{float(acts.throttle[0] * acts.sprint[0] * phen[1]):.2f} u/tick"
        # Rows report the network's *intent*; the flag reports the decoded
        # attempt, so a resting organism can intend to eat without acting.
        assert shown["eat"][1] is bool(acts.eat[0])
        assert shown["attack"][1] is bool(acts.attack[0])
        assert shown["rest"][1] is bool(acts.rest[0])
        assert shown["eat"][0].endswith("· no") is bool(o[3] <= 0.5)
        assert shown["attack"][0].endswith("· no") is bool(o[4] <= 0.5)


def test_decision_summary_reports_rest_not_eating():
    outputs = np.zeros(8)
    outputs[3] = 0.9
    outputs[7] = 0.9
    text = panels.decision_summary({"outputs": outputs})
    assert "resting" in text and "trying to eat" not in text


# -- panel text --------------------------------------------------------------
def test_organism_rows_report_gene_and_measured_intake_separately():
    payload = {
        "id": 7, "generation": 2, "age": 30, "energy": 2.0, "body": 1.0,
        "health": 90.0,
        "phenotypes": np.array([1.0, 12.0, 100.0, 120.0, 1.0, 0.5, 0.5, 0.5,
                                0.4, 0.2, 500.0, 0.01]),
        "parents": (3, 4),
        "counters": {"plant_intake": 12.0, "meat_intake": 0.5, "bites": 40,
                     "hunts": 1, "repro_success": 2, "dist_moved": 900.0},
    }
    rows = dict(panels.organism_rows(payload))
    assert rows["diet gene"].startswith("0.20")      # the gene, not behaviour
    assert "herbivore-leaning" in rows["diet gene"]
    assert "12." in rows["lifetime intake"] and "plant" in rows["lifetime intake"]
    assert rows["parents"] == "3 + 4"


def test_amount_formats_without_false_precision():
    assert panels.amount(float("nan")) == "—"
    assert panels.amount(3.14159) == "3.14"
    assert panels.amount(42.4) == "42.4"
    assert panels.amount(1234.5) == "1,234"


def test_role_counts_come_from_intake_not_the_gene():
    eng = Engine(small_config(organisms=8))
    eng.step()
    counts = panels.role_counts(eng)
    assert set(counts) == {panels.stats_mod.ROLE_HERBIVORE,
                           panels.stats_mod.ROLE_OMNIVORE,
                           panels.stats_mod.ROLE_CARNIVORE,
                           panels.stats_mod.ROLE_UNCLASSIFIED}
    assert sum(counts.values()) == eng.population.n_alive
    # Nothing has eaten yet, so every organism is indistinguishable by intake.
    assert counts[panels.stats_mod.ROLE_UNCLASSIFIED] == eng.population.n_alive


# -- app behaviour -----------------------------------------------------------
def test_click_selects_the_nearest_live_organism():
    from evolve.viz.app import VizApp
    eng = Engine(small_config(organisms=6, seed=3))
    eng.step()
    app = VizApp(eng, speed=1)
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    i = int(idx[0])
    sx, sy = app.camera.to_screen(float(pop.x[i]), float(pop.y[i]))
    app.handle_click((sx, sy))
    assert app.state.selected_id == int(pop.id[i])
    assert app.state.selected is not None
    assert app.state.selected["id"] == int(pop.id[i])


def test_click_on_empty_space_clears_and_far_clicks_miss():
    from evolve.viz.app import VizApp
    eng = Engine(small_config(organisms=4, seed=5))
    eng.step()
    app = VizApp(eng, speed=1)
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    i = int(idx[0])
    sx, sy = app.camera.to_screen(float(pop.x[i]), float(pop.y[i]))
    app.handle_click((sx, sy))
    assert app.state.selected_id is not None
    app.handle_click((sx + 200, sy + 200))   # nothing within 12 px
    assert app.state.selected_id is None
    # Clicking outside the world viewport (the sidebar) must not select.
    app.handle_click((app.size[0] - 5, 5))
    assert app.state.selected_id is None


def test_toggles_set_expectations():
    from evolve.viz.app import VizApp
    eng = Engine(small_config(organisms=4))
    app = VizApp(eng, speed=1)
    layer = app.state.layer
    app.handle_key(pygame.K_v)
    assert app.state.layer == palette.next_layer(layer)
    app.handle_key(pygame.K_i)
    assert app.state.show_panel is False
    app.handle_key(pygame.K_i)
    assert app.state.show_panel is True
    app.handle_key(pygame.K_h)
    assert app.state.show_help is True
    app.handle_key(pygame.K_h)
    assert app.state.show_help is False
    app.handle_key(pygame.K_x)
    assert app.state.selected_id is None


def test_help_overlay_is_modal_and_escape_closes_it_first():
    from evolve.viz.app import VizApp
    eng = Engine(small_config(organisms=4))
    app = VizApp(eng, speed=1)
    app.handle_key(pygame.K_h)
    app.run_frame()
    assert app.state.show_help and app.running
    app.handle_key(pygame.K_ESCAPE)
    assert app.state.show_help is False
    assert app.running is True      # first Escape closes help, does not quit
    app.handle_key(pygame.K_ESCAPE)
    assert app.running is False


def test_help_swallows_every_other_control_while_open():
    """A reader must not pause, step, reset or select by accident."""
    from evolve.viz.app import VizApp
    eng = Engine(small_config(organisms=6, seed=3))
    eng.step()
    app = VizApp(eng, speed=1)
    app.handle_key(pygame.K_h)
    assert app.state.show_help
    paused, layer, engine, speed = (app.state.paused, app.state.layer,
                                    app.engine, app.state.speed)
    tick = app.engine.tick
    for key in (pygame.K_SPACE, pygame.K_s, pygame.K_r, pygame.K_v,
                pygame.K_i, pygame.K_x, pygame.K_1):
        app.handle_key(key)
    assert app.state.paused == paused
    assert app.state.layer == layer
    assert app.state.speed == speed
    assert app.engine is engine and app.engine.tick == tick
    assert app.state.show_panel is True
    # A click on the world must not change the selection either.
    pop = app.engine.population
    idx = np.flatnonzero(pop.alive)
    i = int(idx[0])
    app.handle_click(app.camera.to_screen(float(pop.x[i]), float(pop.y[i])))
    assert app.state.selected_id is None
    # H closes it (it does not reopen), and only then do keys act again.
    app.handle_key(pygame.K_h)
    assert app.state.show_help is False
    app.handle_key(pygame.K_SPACE)
    assert app.state.paused is True


def test_layer_labels_describe_every_layer_in_words():
    assert set(palette.LAYER_LABELS) == set(palette.LAYERS)
    for layer in palette.LAYERS:
        text = palette.LAYER_LABELS[layer]
        assert layer in text
        assert len(text) > len(layer) + 8      # explains, not just names
        assert "->" not in text                # the bundled font has no arrow


def test_footer_and_badge_fit_the_smallest_window():
    """Text must never spill past the window at the documented minimum size."""
    from evolve.viz.app import _MIN_SIZE, VizApp
    eng = Engine(small_config(organisms=4))
    app = VizApp(eng, size=_MIN_SIZE)
    assert app.size == _MIN_SIZE
    app.run_frame()          # draws footer + badge + panel
    arr = pygame.surfarray.array3d(app.screen)   # (width, height, 3)
    footer_top = app.size[1] - panels.FOOTER_H
    # Skip the divider line at the footer's top edge; below it the strip must be
    # plain background at the rightmost columns, i.e. no text runs off-screen.
    footer = arr[:, footer_top + 2:]
    assert (footer[-3:] == np.array(palette.PANEL_BG)).all()
    # And the badge must not overlap the sidebar. Skip the panel's own divider
    # line at its left edge.
    panel_x = app.size[0] - panels.PANEL_W
    top = arr[panel_x + 4:, :60]
    assert (top[:6] == np.array(palette.PANEL_BG)).all()
    assert not (top == np.array(palette.TEXT)).all()


def test_help_and_panel_scroll_one_step_per_notch():
    from evolve.viz.app import VizApp
    eng = Engine(small_config(organisms=6))
    app = VizApp(eng, speed=1)
    app.state.scroll = 0
    app.state.scroll += 3 * panels.SCROLL_STEP
    assert app.state.scroll == 3 * panels.SCROLL_STEP
    # Raw page keys move by a screenful-ish amount and never go negative.
    app.state.scroll = 0
    app.handle_key(pygame.K_PAGEUP)
    assert app.state.scroll == 0
    app.handle_key(pygame.K_PAGEDOWN)
    assert app.state.scroll == 8 * panels.SCROLL_STEP


def test_selection_survives_compaction_by_id():
    """The inspector must follow an id, not a slot index."""
    from evolve.viz.app import VizApp
    eng = Engine(small_config(organisms=10, seed=2))
    eng.step()
    app = VizApp(eng, speed=1)
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    i = int(idx[-1])
    chosen = int(pop.id[i])
    sx, sy = app.camera.to_screen(float(pop.x[i]), float(pop.y[i]))
    app.handle_click((sx, sy))
    assert app.state.selected_id == chosen
    for _ in range(5):
        app.run_frame()
        if app.state.selected is None:
            break       # it died; covered by the other test
        assert app.state.selected["id"] == chosen
