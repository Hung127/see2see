"""Chart primitives for the analytics view.

Small, dependency-free line/bar/histogram plots drawn with pygame primitives.
They exist because the viewer must run headless on a single core with no
matplotlib import in the frame loop: the reporting path already writes matplotlib
PNGs from recorded rows, and this is the *live* counterpart.

Rules the helpers enforce, because a chart that breaks them misleads:

* every plot is given an explicit unit label, drawn on the same row as the title;
* the y range is reported as numbers, never implied;
* a series with too few samples draws an explanatory line instead of a
  straight segment that would look like data;
* missing values (``nan``) break the line rather than becoming zero;
* the number of drawn points is capped, with the cap stated in the caption;
* the layout reserves a row for the legend, a row for the x labels and a row
  for the caption, so nothing is drawn over anything else at any chart size.
"""
from __future__ import annotations

import numpy as np
import pygame

from .palette import ACCENT, PANEL_EDGE, TEXT, TEXT_DIM, TEXT_FAINT

#: Upper bound on points drawn per series. A live panel cannot show 10,000
#: samples legibly, and drawing them is wasted work on the target machine.
MAX_POINTS = 180


def _fmt(value: float) -> str:
    """Compact axis tick label; no false precision."""
    if value != value:
        return "—"
    a = abs(value)
    if a >= 1000:
        return f"{value / 1000.0:,.1f}k"
    if a >= 100:
        return f"{value:,.0f}"
    if a >= 10:
        return f"{value:.0f}"
    if a >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _thin(values: np.ndarray, cap: int = MAX_POINTS) -> np.ndarray:
    """Subsample to at most ``cap`` points, keeping both endpoints.

    Deterministic striding, not random sampling: the same history draws the same
    picture on every frame, which matters when comparing screenshots.
    """
    n = values.size
    if n <= cap:
        return values
    idx = np.linspace(0, n - 1, cap).astype(np.int64)
    return values[idx]


class _Frame:
    """Row layout for one chart: title+unit, legend, plot box, caption.

    Reserving every row up front is what keeps a short chart from drawing its
    caption over the x axis, and a long inline legend from running over the
    published scale on the right edge.
    """

    __slots__ = ("rect", "box", "title_y", "legend_y", "caption_y")

    def __init__(self, rect: pygame.Rect, small: pygame.font.Font,
                 caption: bool) -> None:
        self.rect = rect
        line = small.get_linesize()
        self.title_y = rect.y + 1
        self.legend_y = self.title_y + line
        top = self.legend_y + line + 1
        reserved = line * (2 if caption else 1) + 3
        height = max(20, rect.height - (top - rect.y) - reserved)
        self.box = pygame.Rect(rect.x + 46, top,
                               max(20, rect.width - 56), height)
        self.caption_y = self.box.bottom + line + 2


def _draw_header(screen: pygame.Surface, frame: _Frame, small: pygame.font.Font,
                 cache, title: str, unit: str) -> None:
    screen.blit(cache.render(small, title, TEXT), (frame.rect.x + 2,
                                                  frame.title_y))
    surf = cache.render(small, unit, TEXT_FAINT)
    screen.blit(surf, (frame.rect.right - surf.get_width() - 2,
                       frame.title_y))


def _draw_legend(screen: pygame.Surface, frame: _Frame, small: pygame.font.Font,
                 cache, series: list[tuple[str, tuple[int, int, int]]]) -> None:
    """One row of swatch+label entries, with "+N" when they do not all fit."""
    if not series:
        return
    x = frame.rect.x + 2
    limit = frame.rect.right - 2
    shown = 0
    for label, colour in series:
        surf = cache.render(small, label, TEXT_DIM)
        need = 20 + surf.get_width() + 8
        if shown and x + need > limit:
            break
        pygame.draw.line(screen, colour, (x, frame.legend_y + 7),
                         (x + 13, frame.legend_y + 7), 2)
        screen.blit(surf, (x + 16, frame.legend_y))
        x += need
        shown += 1
    if shown < len(series):
        surf = cache.render(small, f"+{len(series) - shown}", TEXT_FAINT)
        screen.blit(surf, (x, frame.legend_y))


def _draw_axes(screen: pygame.Surface, frame: _Frame, small: pygame.font.Font,
               cache, vmin: float, vmax: float,
               x_labels: tuple[str, str]) -> None:
    box = frame.box
    pygame.draw.rect(screen, (28, 31, 38), box)
    pygame.draw.line(screen, PANEL_EDGE, (box.x, box.bottom),
                     (box.right, box.bottom), 1)
    pygame.draw.line(screen, PANEL_EDGE, (box.x, box.y),
                     (box.x, box.bottom), 1)
    # Three y ticks: bottom, middle, top. The values are printed, so the reader
    # never has to infer the scale from the shape of the line. Labels are
    # clamped inside the box: centred on the top edge, the top label spilled
    # up into the legend row and collided with it.
    for frac in (0.0, 0.5, 1.0):
        value = vmin + (vmax - vmin) * frac
        y = int(box.bottom - frac * box.height)
        text = cache.render(small, _fmt(value), TEXT_FAINT)
        ly = min(max(box.y, y - text.get_height() // 2),
                 box.bottom - text.get_height())
        screen.blit(text, (box.x - text.get_width() - 4, ly))
    for text, x in ((x_labels[0], box.x), (x_labels[1], box.right)):
        surf = cache.render(small, text, TEXT_FAINT)
        if x == box.right:
            x -= surf.get_width()
        screen.blit(surf, (x, box.bottom + 2))


def _value_range(arrays: list[np.ndarray], zero_based: bool
                 ) -> tuple[float, float] | None:
    finite = [a[np.isfinite(a)] for a in arrays]
    finite = [a for a in finite if a.size]
    if not finite:
        return None
    allv = np.concatenate(finite)
    vmin, vmax = float(allv.min()), float(allv.max())
    if zero_based:
        vmin = min(0.0, vmin)
        vmax = max(0.0, vmax)
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    pad = 0.06 * (vmax - vmin)
    return vmin - pad, vmax + pad


def line_chart(screen: pygame.Surface, rect: pygame.Rect,
               fonts: dict[str, pygame.font.Font], cache, *,
               title: str, y_unit: str,
               series: list[tuple[str, tuple[int, int, int], np.ndarray]],
               x_labels: tuple[str, str] = ("oldest", "now"),
               zero_based: bool = True,
               caption: str | None = None) -> None:
    """Stacked line chart with a printed scale and a small inline legend.

    ``series`` is ``[(label, colour, values)]``. All series are drawn on one
    shared y range so their relative size is truthful; ``zero_based`` keeps zero
    in view for counts (a truncated count axis exaggerates small wobbles).
    """
    small = fonts["small"]
    arrays = [(label, colour, np.asarray(values, dtype=np.float64))
              for label, colour, values in series]
    arrays = [item for item in arrays if item[2].size]
    span = _value_range([a for _, _, a in arrays], zero_based)
    if span is None:
        screen.blit(cache.render(small, title, TEXT), (rect.x + 2, rect.y + 1))
        unit = cache.render(small, y_unit, TEXT_FAINT)
        screen.blit(unit, (rect.right - unit.get_width() - 2, rect.y + 1))
        screen.blit(cache.render(small, "no samples yet", TEXT_FAINT),
                    (rect.x + 4, rect.y + 4 + small.get_linesize()))
        return
    vmin, vmax = span
    frame = _Frame(rect, small, caption is not None)
    _draw_header(screen, frame, small, cache, title, y_unit)
    _draw_legend(screen, frame, small, cache,
                 [(label, colour) for label, colour, _ in arrays])
    _draw_axes(screen, frame, small, cache, vmin, vmax, x_labels)
    box = frame.box
    for _, colour, values in arrays:
        values = _thin(values)
        points = []
        for i, value in enumerate(values):
            if not np.isfinite(value):
                if len(points) > 1:
                    pygame.draw.lines(screen, colour, False, points, 2)
                points = []
                continue
            px = box.x + int(box.width * (i / max(1, values.size - 1)))
            py = int(box.bottom - box.height * (value - vmin) / (vmax - vmin))
            points.append((px, py))
        if len(points) > 1:
            pygame.draw.lines(screen, colour, False, points, 2)
        elif points:
            pygame.draw.circle(screen, colour, points[0], 2)
    if caption:
        # Fitted to the plot width: a long caption used to run out of the chart
        # and under the neighbouring column.
        screen.blit(cache.fit(small, caption, TEXT_FAINT, box.width),
                    (box.x, frame.caption_y))


def bar_chart(screen: pygame.Surface, rect: pygame.Rect,
              fonts: dict[str, pygame.font.Font], cache, *,
              title: str, y_unit: str,
              bars: list[tuple[str, tuple[int, int, int], float]],
              caption: str | None = None) -> None:
    """Horizontal bars: the right form when labels are words, not numbers."""
    small = fonts["small"]
    frame = _Frame(rect, small, caption is not None)
    _draw_header(screen, frame, small, cache, title, y_unit)
    if not bars:
        screen.blit(cache.render(small, "nothing to show", TEXT_FAINT),
                    (rect.x + 4, frame.box.y))
        return
    label_w = max(70, min(140, frame.box.width // 3))
    value_w = 58
    track_x = frame.box.x
    track_w = max(20, frame.box.width - label_w - value_w)
    peak = max(1e-12, max(float(v) for _, _, v in bars))
    rows_h = frame.box.height // max(1, len(bars))
    row_h = max(8, min(small.get_linesize() + 4, rows_h - 2))
    for i, (label, colour, value) in enumerate(bars):
        y = frame.box.y + i * rows_h
        screen.blit(cache.fit(small, label, TEXT_DIM, label_w - 4),
                    (track_x, y))
        track = pygame.Rect(track_x + label_w, y, track_w, row_h)
        pygame.draw.rect(screen, (30, 34, 41), track)
        w = int(track.width * max(0.0, float(value)) / peak)
        if w > 0:
            pygame.draw.rect(screen, colour, (track.x, track.y, w, track.h))
        text = cache.render(small, _fmt(float(value)), TEXT)
        screen.blit(text, (track.right + 4, y))
    if caption:
        screen.blit(cache.fit(
            small, f"{caption} · scale 0 – {_fmt(peak)} {y_unit}", TEXT_FAINT,
            frame.box.width), (frame.box.x, frame.caption_y))


def histogram(screen: pygame.Surface, rect: pygame.Rect,
              fonts: dict[str, pygame.font.Font], cache, *,
              title: str, y_unit: str,
              counts: np.ndarray, bin_width: float,
              colour: tuple[int, int, int] = ACCENT,
              caption: str | None = None) -> None:
    """Vertical bar histogram; x is a labelled continuous axis, not categories."""
    small = fonts["small"]
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.nansum(counts)) if counts.size else 0.0
    if counts.size == 0 or total <= 0:
        screen.blit(cache.render(small, title, TEXT), (rect.x + 2, rect.y + 1))
        unit = cache.render(small, y_unit, TEXT_FAINT)
        screen.blit(unit, (rect.right - unit.get_width() - 2, rect.y + 1))
        screen.blit(cache.render(small, "no deaths recorded yet", TEXT_FAINT),
                    (rect.x + 4, rect.y + 4 + small.get_linesize()))
        return
    frame = _Frame(rect, small, caption is not None)
    _draw_header(screen, frame, small, cache, title, y_unit)
    vmax = float(np.nanmax(counts))
    _draw_axes(screen, frame, small, cache, 0.0, max(1.0, vmax),
               ("0", f"{counts.size * bin_width:.0f}"))
    box = frame.box
    bw = box.width / max(1, counts.size)
    for i, value in enumerate(counts):
        h = int(box.height * float(value) / max(1.0, vmax))
        if h <= 0:
            continue
        pygame.draw.rect(screen, colour,
                         (int(box.x + i * bw), box.bottom - h,
                          max(1, int(bw) - 1), h))
    if caption:
        # Fitted to the plot width: a long caption used to run out of the chart
        # and under the neighbouring column.
        screen.blit(cache.fit(small, caption, TEXT_FAINT, box.width),
                    (box.x, frame.caption_y))
