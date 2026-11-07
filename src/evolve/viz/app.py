"""Interactive view (DESIGN §10).

The simulation never blocks on rendering: one frame pumps events, advances a
fixed number of steps, then draws. All controls are listed in the footer and in
the ``h`` help overlay, and the sidebar explains what every mark on screen is.

Beginner affordances: a legend, plain-language statistics (births, deaths,
roles by measured intake, conserved matter pools), a named environment layer,
and a click-to-inspect readout of the exact last-tick sensor/output vectors the
network saw and produced.
"""
from __future__ import annotations

import numpy as np
import pygame

from ..config import Config
from ..engine import Engine
from .analytics import AnalyticsHistory, draw_analytics
from .camera import Camera
from .palette import (
    LAYER_LABELS, layer_ramp, layer_scale_text, next_layer,
)
from .panels import (
    FOOTER_H, MIN_WORLD_W, PANEL_W, SCROLL_STEP, TextCache, draw_footer,
    draw_help, draw_sidebar,
)
from .render import draw_perception, draw_selection, draw_world

_SPEEDS = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 5, pygame.K_4: 20}
_DEFAULT_SIZE = (1400, 860)
_MIN_SIZE = (MIN_WORLD_W + PANEL_W, 560)


class ViewState:
    """Presentation-only state; it never influences the simulation."""

    def __init__(self, speed: int = 1) -> None:
        self.speed = int(speed)
        self.paused = False
        self.layer = "climate"
        self.show_panel = True
        self.show_help = False
        self.follow = False
        self.scroll = 0
        self.content_height = 0
        self.view_height = 0
        self.help_scroll = 0
        self.help_height = 0
        self.selected_id: int | None = None
        self.selected: dict | None = None
        self.selected_gone = False
        self.inspect_tick: int = -1
        self.history: list[tuple[int, int]] = []
        # Analytics view: a separate full-width screen that reads the same
        # simulation. It is a presentation mode, not a second simulation.
        self.show_analytics = False
        self.analytics_scroll = 0
        self.analytics_height = 0
        self.analytics_view = 0
        # Draw the selected organism's vision cone and detected targets.
        self.show_perception = True


class VizApp:
    def __init__(self, engine: Engine, speed: int = 1,
                 size: tuple[int, int] = _DEFAULT_SIZE,
                 title: str = "EVOLVE — artificial life") -> None:
        pygame.display.init()
        pygame.font.init()
        self.engine = engine
        self.state = ViewState(speed)
        self.running = True
        self.size = (max(_MIN_SIZE[0], int(size[0])), max(_MIN_SIZE[1], int(size[1])))
        self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.fonts = self._make_fonts()
        self.cache = TextCache()
        self.camera = self._new_camera()
        # Live analytics history: sampled on simulated time, bounded in size.
        self.analytics = AnalyticsHistory()
        # Seed the first sample so the analytics view is never blank on the
        # very first frame, before a single tick has run.
        self.analytics.maybe_sample(engine, force=True)

    # -- layout ---------------------------------------------------------------
    def _make_fonts(self) -> dict[str, pygame.font.Font]:
        pygame.font.init()
        # ``pygame.font.Font(None, size)`` uses the bundled default face, which
        # exists on every pygame install and needs no system font search. The
        # bundled face renders cramped lowercase below ~22 px (verified by
        # screenshot on the test machine), so the smallest size used is 22.
        return {
            "title": pygame.font.Font(None, 31),
            "big": pygame.font.Font(None, 46),
            "body": pygame.font.Font(None, 25),
            "small": pygame.font.Font(None, 22),
        }

    def _panel_rect(self) -> pygame.Rect:
        if not self.state.show_panel or self.state.show_analytics:
            return pygame.Rect(self.size[0], 0, 0, self.size[1])
        return pygame.Rect(self.size[0] - PANEL_W, 0, PANEL_W,
                           self.size[1] - FOOTER_H)

    def _world_rect(self) -> pygame.Rect:
        if self.state.show_analytics:
            return pygame.Rect(0, 0, 0, 0)   # analytics replaces the world view
        w = self.size[0] - (PANEL_W if self.state.show_panel else 0)
        return pygame.Rect(0, 0, w, self.size[1] - FOOTER_H)

    def _analytics_rect(self) -> pygame.Rect:
        return pygame.Rect(0, 0, self.size[0], self.size[1] - FOOTER_H)

    def _camera_rect(self) -> pygame.Rect:
        """World viewport used for camera geometry, ignoring the analytics mode.

        The camera keeps its zoom and focus while the analytics view is open, so
        leaving that view does not silently re-fit or crop the world. Using
        ``_world_rect`` here would build a zero-sized camera when analytics was
        toggled on (and ``R`` can restart the run while it is open).
        """
        w = self.size[0] - (PANEL_W if self.state.show_panel else 0)
        return pygame.Rect(0, 0, w, self.size[1] - FOOTER_H)

    def _new_camera(self) -> Camera:
        rect = self._camera_rect()
        cam = Camera((self.engine.world.width, self.engine.world.height),
                     (rect.w, rect.h))
        cam.fit()
        return cam

    # -- compatibility shorthands --------------------------------------------
    @property
    def speed(self) -> int:
        return self.state.speed

    @speed.setter
    def speed(self, value: int) -> None:
        self.state.speed = int(value)

    @property
    def paused(self) -> bool:
        return self.state.paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self.state.paused = bool(value)

    # -- input ----------------------------------------------------------------
    def handle_key(self, key: int) -> None:
        state = self.state
        # The guide is modal: while it is open, only closing and scrolling act,
        # so a beginner cannot accidentally zoom, step or quit the simulation
        # while reading it.
        if state.show_help:
            if key in (pygame.K_ESCAPE, pygame.K_h):
                state.show_help = False
            elif key == pygame.K_q:
                self.running = False
            elif key == pygame.K_PAGEUP:
                state.help_scroll = max(0, state.help_scroll
                                        - 8 * SCROLL_STEP)
            elif key == pygame.K_PAGEDOWN:
                state.help_scroll += 8 * SCROLL_STEP
            return
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.running = False
        elif key == pygame.K_SPACE:
            state.paused = not state.paused
        elif key == pygame.K_s:
            self.engine.step()
            self._refresh_selection()
        elif key == pygame.K_r:
            self.engine = Engine(self.engine.cfg)
            state.selected_id = None
            state.selected = None
            state.selected_gone = False
            # The old samples belong to the previous run: keeping them made the
            # sparkline claim "was 24, now 120" across two different worlds.
            state.history.clear()
            self.analytics.reset()
            self.analytics.maybe_sample(self.engine, force=True)
            state.scroll = 0
            state.follow = False
            self.camera = self._new_camera()
        elif key in _SPEEDS:
            state.speed = _SPEEDS[key]
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.camera.zoom_by(1.25)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.camera.zoom_by(0.8)
        elif key == pygame.K_LEFT:
            state.follow = False
            self.camera.pan(-40.0, 0.0)
        elif key == pygame.K_RIGHT:
            state.follow = False
            self.camera.pan(40.0, 0.0)
        elif key == pygame.K_UP:
            state.follow = False
            self.camera.pan(0.0, -40.0)
        elif key == pygame.K_DOWN:
            state.follow = False
            self.camera.pan(0.0, 40.0)
        elif key == pygame.K_HOME:
            state.follow = False
            self.camera.fit()
        elif key == pygame.K_v:
            state.layer = next_layer(state.layer)
        elif key == pygame.K_a:
            state.show_analytics = not state.show_analytics
            if not state.show_analytics:
                self.camera = self._resize_camera()
        elif key == pygame.K_p:
            state.show_perception = not state.show_perception
        elif key == pygame.K_i:
            state.show_panel = not state.show_panel
            self.camera = self._resize_camera()
        elif key == pygame.K_h:
            state.show_help = True
            state.help_scroll = 0
        elif key == pygame.K_x:
            self._clear_selection()
        elif key == pygame.K_f:
            state.follow = not state.follow and state.selected_id is not None
        elif key == pygame.K_PAGEUP:
            state.scroll = max(0, state.scroll - 8 * SCROLL_STEP)
        elif key == pygame.K_PAGEDOWN:
            state.scroll += 8 * SCROLL_STEP

    def handle_click(self, pos: tuple[int, int], button: int = 1) -> None:
        """Left click selects the nearest live organism under the cursor."""
        if self.state.show_help:
            return          # the guide is modal; a click must not change state
        if button != 1 or not self._world_rect().collidepoint(pos):
            return
        wx, wy = self.camera.to_world(pos[0], pos[1])
        pop = self.engine.population
        idx = np.flatnonzero(pop.alive)
        if idx.size == 0:
            self._clear_selection()
            return
        d = self.engine.world.distance(pop.x[idx], pop.y[idx],
                                       np.array([wx]), np.array([wy]))
        best = int(np.argmin(d))
        # 12 screen pixels tolerance, converted to world units.
        if float(d[best]) * self.camera.zoom > 12.0:
            self._clear_selection()
            return
        self.state.selected_id = int(pop.id[idx[best]])
        self._refresh_selection()

    # -- selection ------------------------------------------------------------
    def _clear_selection(self) -> None:
        self.state.selected_id = None
        self.state.selected = None
        self.state.selected_gone = False
        self.state.follow = False

    def _refresh_selection(self) -> None:
        """Re-read the selected organism; mark it gone if it died.

        ``engine.inspect`` rebuilds phenotypes and splits network weights, so it
        is only repeated when the simulation has actually advanced.
        """
        sid = self.state.selected_id
        if sid is None:
            return
        pop = self.engine.population
        idx = np.flatnonzero(pop.alive)
        hits = np.flatnonzero(pop.id[idx] == sid) if idx.size else np.zeros(0, int)
        if hits.size == 0:
            self.state.selected = None
            self.state.selected_gone = True
            self.state.follow = False
            return
        self.state.selected_gone = False
        if self.state.selected is not None and self.state.inspect_tick == self.engine.tick:
            payload = self.state.selected
        else:
            i = int(idx[int(hits[0])])
            try:
                payload = self.engine.inspect(i)
            except Exception:  # an inspector must never take down the frame loop
                payload = None
            self.state.selected = payload
            self.state.inspect_tick = self.engine.tick
        if self.state.follow and payload is not None:
            x, y = payload["position"]
            self.camera.centre_on(x, y)

    # -- frame loop -----------------------------------------------------------
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_click(event.pos, event.button)
            elif event.type == pygame.MOUSEWHEEL:
                pos = pygame.mouse.get_pos()
                if self.state.show_help:
                    self.state.help_scroll = max(
                        0, self.state.help_scroll
                        - event.y * SCROLL_STEP)
                elif self.state.show_analytics:
                    self.state.analytics_scroll = max(
                        0, self.state.analytics_scroll
                        - event.y * SCROLL_STEP)
                elif self._panel_rect().collidepoint(pos):
                    self.state.scroll = max(
                        0, self.state.scroll - event.y * SCROLL_STEP)
            elif event.type == pygame.VIDEORESIZE:
                self.resize((event.w, event.h))

    def resize(self, size: tuple[int, int]) -> None:
        w = max(_MIN_SIZE[0], int(size[0]))
        h = max(_MIN_SIZE[1], int(size[1]))
        self.size = (w, h)
        self.screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
        self.camera = self._resize_camera()

    def _resize_camera(self) -> Camera:
        """Rebuild the camera for a new viewport.

        If the reader never zoomed or panned, keep the whole world fitted: the
        old magnification means something different in the new viewport, and
        carrying it over silently cropped the world when the window shrank.
        A zoom or focus the user chose is preserved.
        """
        rect = self._camera_rect()
        old = self.camera
        cam = Camera((self.engine.world.width, self.engine.world.height),
                     (rect.w, rect.h))
        old_fit = max(0.1, min(old.screen_w / old.world_w,
                               old.screen_h / old.world_h) * 0.99)
        untouched = (abs(old.zoom - old_fit) <= 1e-9
                     and abs(old.focus[0] - old.world_w / 2.0) <= 1e-9
                     and abs(old.focus[1] - old.world_h / 2.0) <= 1e-9)
        if untouched:
            cam.fit()
        else:
            cam.focus = old.focus
            cam.zoom = old.zoom
        return cam

    def run_frame(self) -> None:
        self._handle_events()
        if not self.state.paused:
            for _ in range(self.state.speed):
                self.engine.step()
            self._refresh_selection()
        # Sampled outside the pause branch so a paused reader stepping with S
        # also moves the population line: otherwise the world advanced and the
        # chart did not, which looks like the sparkline is broken.
        self._record_history()
        # Analytics is sampled on simulated time, not frames, so the x axis
        # means ticks even when the reader changes speed.
        self.analytics.maybe_sample(self.engine)
        self.render()
        pygame.display.flip()
        self.clock.tick(60)

    def _record_history(self, cap: int = 240) -> None:
        """Population sparkline samples, one per rendered frame that advanced."""
        hist = self.state.history
        tick = self.engine.tick
        if hist and hist[-1][0] == tick:
            return              # paused: nothing new happened, keep the sample
        hist.append((tick, self.engine.population.n_alive))
        if len(hist) > cap:
            del hist[:len(hist) - cap]

    def render(self) -> None:
        if self.state.show_analytics:
            self._render_analytics()
        else:
            self._render_world()
        draw_footer(self.screen, pygame.Rect(0, self.size[1] - FOOTER_H,
                                             self.size[0], FOOTER_H),
                    self.fonts, self.cache, self.state)
        if self.state.show_help:
            self.state.help_height = draw_help(
                self.screen, pygame.Rect(0, 0, self.size[0], self.size[1]),
                self.fonts, self.cache, self.state.help_scroll,
                self.state.help_height)

    def _render_world(self) -> None:
        rect = self._world_rect()
        self.screen.set_clip(rect)
        draw_world(self.engine, self.screen, self.camera, layer=self.state.layer)
        draw_selection(self.engine, self.screen, self.camera,
                       self.state.selected_id)
        # Perception overlay sits above the entity it belongs to but below the
        # sidebar: it is an annotation about one animal, not a world layer.
        if self.state.show_perception and self.state.selected is not None:
            draw_perception(self.engine, self.screen, self.camera,
                            self.state.selected)
        self.screen.set_clip(None)
        if self.state.show_panel:
            draw_sidebar(self.screen, self._panel_rect(), self.fonts, self.cache,
                         self.engine, self.state)
        self._draw_badges(rect)

    def _render_analytics(self) -> None:
        rect = self._analytics_rect()
        # Clamp the *previous* frame's known height before drawing. Clamping only
        # after drawing left one blank frame at the bottom of the grid: the
        # out-of-range offset pushed every chart above the clip, then the state
        # was corrected for the next frame. The reader saw a flash of nothing.
        max_scroll = max(0, self.state.analytics_height
                            - self.state.analytics_view)
        if max_scroll:
            self.state.analytics_scroll = min(self.state.analytics_scroll,
                                              max_scroll)
        content, view = draw_analytics(
            self.screen, rect, self.fonts, self.cache, self.analytics,
            self.engine, self.state.analytics_scroll)
        self.state.analytics_height = content
        self.state.analytics_view = view
        self.state.analytics_scroll = max(
            0, min(self.state.analytics_scroll, max(0, content - view)))

    def _draw_badges(self, rect: pygame.Rect) -> None:
        """Top-left status strip: tick, and what the background layer means.

        Fitted to the world viewport: with the panel open on a narrow window the
        strip would otherwise run under the sidebar and collide with its title.
        Single-quantity layers also get a ramp legend with the numeric range
        printed beside it, so the colour scale is stated, not implied.
        """
        small = self.fonts["small"]
        layer = self.state.layer
        avail = max(120, rect.width - 16)
        surfaces = [self.cache.fit(small, f"tick {self.engine.tick:,}  ·  "
                                          f"{layer} layer", (226, 230, 238),
                                   avail - 10),
                    self.cache.fit(small, LAYER_LABELS.get(layer, layer),
                                   (148, 156, 170), avail - 10)]
        scale = layer_scale_text(layer)
        ramp = layer_ramp(layer, steps=48)
        if scale:
            surfaces.append(self.cache.fit(small, scale, (104, 112, 126),
                                           avail - 10))
        pad = 5
        width = max(s.get_width() for s in surfaces) + 2 * pad
        if ramp:
            width = max(width, min(avail, 220) + 2 * pad)
        height = (sum(s.get_height() for s in surfaces) + 2 * pad
                  + (10 if ramp else 0))
        self.screen.fill((14, 16, 20), (6, 6, width, height))
        y = 6 + pad
        for surf in surfaces:
            self.screen.blit(surf, (6 + pad, y))
            y += surf.get_height()
        if ramp:
            bar = min(avail - 10, 200)
            y += 2
            for i, colour in enumerate(ramp):
                x0 = 6 + pad + int(bar * i / len(ramp))
                x1 = 6 + pad + int(bar * (i + 1) / len(ramp))
                pygame.draw.rect(self.screen, colour, (x0, y, max(1, x1 - x0), 8))
            y += 8
        if self.engine.population.n_alive == 0:
            self._draw_extinct(rect)
        if self.state.selected_gone and not self.state.show_panel:
            msg = self.cache.render(small, "the selected organism died",
                                    (245, 194, 92))
            self.screen.blit(msg, (10, 8 + height + 6))

    def _draw_extinct(self, rect: pygame.Rect) -> None:
        """Centre the extinction notice inside the *world* viewport.

        The notice is fitted to the world width: at the smallest window the
        world is 320 px wide, and an unfitted 46 px headline ran under the
        sidebar and off the left edge, which looked like a broken overlay.
        """
        small = self.fonts["small"]
        avail = max(120, rect.width - 24)
        headline = "EXTINCT — no organisms remain"
        for name in ("big", "body", "small"):
            font = self.fonts[name]
            if font.size(headline)[0] <= avail:
                break
        msg = self.cache.fit(font, headline, (236, 96, 88), avail)
        hint_lines = self._wrap_plain(small, "R restarts from the seed · "
                                            "the run is still valid data", avail)
        hint = [self.cache.render(small, line, (226, 230, 238))
                for line in hint_lines]
        # A backing plate keeps the notice readable over the plant field.
        pad = 18
        w = max([msg.get_width()] + [s.get_width() for s in hint]) + 2 * pad
        h = (msg.get_height() + 2 * pad + 8
             + sum(s.get_height() for s in hint))
        plate = pygame.Surface((w, h), pygame.SRCALPHA)
        plate.fill((12, 14, 18, 232))
        px = max(rect.x + 6, min(rect.centerx - w // 2, rect.right - w - 6))
        py = max(rect.y + 6, min(rect.centery - h // 2, rect.bottom - h - 6))
        self.screen.blit(plate, (px, py))
        pygame.draw.rect(self.screen, (70, 78, 92), (px, py, w, h), 1)
        self.screen.blit(msg, (px + (w - msg.get_width()) // 2, py + pad))
        y = py + pad + msg.get_height() + 8
        for surf in hint:
            self.screen.blit(surf, (px + (w - surf.get_width()) // 2, y))
            y += surf.get_height()

    @staticmethod
    def _wrap_plain(font: pygame.font.Font, text: str,
                    width: int) -> list[str]:
        """Greedy word wrap used where a ``Panel`` cursor is not available."""
        lines: list[str] = []
        line = ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if font.size(trial)[0] > width and line:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
        return lines

    def run(self) -> None:
        while self.running:
            self.run_frame()

    # -- output ---------------------------------------------------------------
    def screenshot(self, path: str) -> None:
        pygame.image.save(self.screen, path)


def launch(config: Config) -> None:
    app = VizApp(Engine(config))
    # Open the guide once, before anything moves: a first-time user has no way
    # to know what the dots mean, and the guide is modal so nothing happens
    # behind their back while they read it. H or Escape closes it.
    app.state.show_help = True
    app.run()
