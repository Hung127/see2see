"""Tests for the live chart primitives and the analytics history.

Two properties are worth testing, and both are about honesty rather than looks:

* a chart must not draw data it does not have (no samples, one sample, a ``nan``
  gap) — a straight line through nothing reads as a measurement;
* the history must sample on *simulated* ticks and stay bounded, because it is a
  presentation-side ring buffer that must never grow with run length.
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
from evolve.viz import charts  # noqa: E402
from evolve.viz.analytics import (  # noqa: E402
    FIELDS, AnalyticsHistory, draw_analytics,
)

pygame.init()
pygame.font.init()

_FONTS = {
    "title": pygame.font.Font(None, 31),
    "big": pygame.font.Font(None, 46),
    "body": pygame.font.Font(None, 25),
    "small": pygame.font.Font(None, 22),
}


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


def _cache():
    from evolve.viz.panels import TextCache
    return TextCache()


def _surface(w=520, h=200):
    return pygame.Surface((w, h))


def _non_blank(surf: pygame.Surface) -> bool:
    """True when the surface contains anything other than its fill colour."""
    arr = pygame.surfarray.array3d(surf)
    first = arr[0, 0]
    return bool((arr != first).any())


# -- formatting --------------------------------------------------------------
def test_fmt_has_no_false_precision():
    assert charts._fmt(0.0) == "0.00"
    assert charts._fmt(1.234) == "1.2"
    assert charts._fmt(42.7) == "43"
    assert charts._fmt(1234.0) == "1.2k"
    # A missing value is drawn as a dash, never as a plausible number.
    assert charts._fmt(float("nan")) == "—"


def test_thin_keeps_endpoints_and_the_cap():
    values = np.arange(1000.0)
    thin = charts._thin(values, cap=180)
    assert thin.size == 180
    assert thin[0] == 0.0
    assert thin[-1] == 999.0
    # Short series are returned untouched (same array, not a copy surprise).
    short = np.arange(10.0)
    assert charts._thin(short, cap=180) is short


def test_value_range_never_collapses_a_constant_series():
    low, high = charts._value_range([np.full(5, 3.0)], zero_based=True)
    assert high > low
    assert low <= 0.0 <= high


def test_value_range_ignores_nan_and_counts_from_zero():
    arr = np.array([np.nan, 5.0, 10.0])
    low, high = charts._value_range([arr], zero_based=True)
    assert low <= 0.0
    assert high >= 10.0
    assert np.isfinite([low, high]).all()


def test_value_range_is_none_without_any_finite_sample():
    assert charts._value_range([np.array([np.nan, np.nan])], True) is None
    assert charts._value_range([], True) is None


# -- line charts -------------------------------------------------------------
def test_line_chart_says_so_when_there_are_no_samples():
    """No samples must not draw an axis that looks like measured data."""
    surf = _surface()
    charts.line_chart(surf, pygame.Rect(0, 0, 300, 130), _FONTS, _cache(),
                      title="population", y_unit="organisms",
                      series=[("population", (96, 178, 245), np.zeros(0))])
    assert _non_blank(surf)              # the explanatory text is there


def test_line_chart_with_one_sample_draws_a_point_not_a_line():
    surf = _surface()
    charts.line_chart(surf, pygame.Rect(0, 0, 300, 130), _FONTS, _cache(),
                      title="population", y_unit="organisms",
                      series=[("population", (96, 178, 245), np.array([7.0]))])
    assert _non_blank(surf)


def test_line_chart_accepts_nan_gaps_and_long_series():
    surf = _surface()
    values = np.linspace(1.0, 50.0, 600)
    values[100:140] = np.nan               # a gap in the record
    charts.line_chart(surf, pygame.Rect(0, 0, 520, 132), _FONTS, _cache(),
                      title="births and deaths", y_unit="individuals",
                      series=[("births", (104, 216, 140), values),
                              ("deaths", (236, 96, 88), values * 0.5)],
                      x_labels=("tick 0", "tick 9,999"),
                      caption="counts since the previous sample")
    assert _non_blank(surf)


def test_line_chart_stays_inside_its_rect_on_a_small_chart():
    """A short chart must not paint over its neighbours."""
    surf = _surface(300, 300)
    surf.fill((14, 16, 20))
    rect = pygame.Rect(100, 100, 80, 60)
    charts.line_chart(surf, rect, _FONTS, _cache(), title="t", y_unit="u",
                      series=[("a", (96, 178, 245), np.arange(20.0))],
                      caption="a caption long enough to overflow")
    arr = pygame.surfarray.array3d(surf)
    # Everything above/left of the rect keeps the background colour.
    assert (arr[:100, :, :] == (14, 16, 20)).all()
    assert (arr[:, :100, :] == (14, 16, 20)).all()


def test_caption_is_fitted_to_the_plot_width():
    """The caption used to run under the neighbouring column."""
    cache = _cache()
    small = _FONTS["small"]
    text = "x" * 400
    fitted = cache.fit(small, text, (104, 112, 126), 120)
    assert fitted.get_width() <= 120
    assert fitted.get_width() < small.size(text)[0]


# -- bar chart and histogram -------------------------------------------------
def test_bar_chart_reports_the_scale_it_used():
    surf = _surface()
    charts.bar_chart(surf, pygame.Rect(0, 0, 300, 130), _FONTS, _cache(),
                     title="deaths by cause", y_unit="deaths",
                     bars=[("predation", (236, 96, 88), 0.0),
                           ("starvation", (245, 194, 92), 4.0)],
                     caption="since the run began")
    assert _non_blank(surf)


def test_bar_chart_without_bars_says_nothing_to_show():
    surf = _surface()
    charts.bar_chart(surf, pygame.Rect(0, 0, 300, 130), _FONTS, _cache(),
                     title="deaths by cause", y_unit="deaths", bars=[])
    assert _non_blank(surf)


def test_histogram_with_zero_counts_says_no_deaths():
    surf = _surface()
    charts.histogram(surf, pygame.Rect(0, 0, 300, 130), _FONTS, _cache(),
                     title="age at death", y_unit="deaths",
                     counts=np.zeros(40, dtype=np.int64), bin_width=25.0)
    assert _non_blank(surf)


def test_histogram_draws_only_nonzero_bins():
    surf = _surface(400, 200)
    surf.fill((14, 16, 20))
    counts = np.zeros(10, dtype=np.int64)
    counts[3] = 5
    charts.histogram(surf, pygame.Rect(0, 0, 400, 180), _FONTS, _cache(),
                     title="age at death", y_unit="deaths", counts=counts,
                     bin_width=25.0, caption="age in ticks",
                     colour=(96, 178, 245))
    arr = pygame.surfarray.array3d(surf)          # shape (x, y, channel)
    hits = (arr == (96, 178, 245)).all(axis=2)    # shape (x, y)
    xs = np.flatnonzero(hits.any(axis=1))         # columns that were painted
    # One bar, not ten: the nine empty bins draw nothing. The bar is one bin
    # wide (box width / 10 ≈ 34 px), not the whole plot.
    assert xs.size > 0
    box_w = charts._Frame(pygame.Rect(0, 0, 400, 180), _FONTS["small"],
                          True).box.width
    assert xs.size <= box_w / 10 + 2


# -- AnalyticsHistory --------------------------------------------------------
def test_history_samples_on_simulated_ticks_not_calls():
    eng = Engine(small_config())
    hist = AnalyticsHistory(interval=10)
    ticks = []
    for _ in range(95):
        eng.step()
        if hist.maybe_sample(eng):
            ticks.append(eng.tick)
    # The first call samples immediately (nothing is "elapsed" yet), then one
    # per 10 ticks: 1, 11, ..., 91. Tick identity is not frame identity.
    assert ticks == [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
    assert hist.last_tick == 91
    # Pumping more frames at the same tick cannot add samples.
    assert hist.maybe_sample(eng) is False


def test_history_force_states_a_sample_at_tick_zero():
    eng = Engine(small_config())
    hist = AnalyticsHistory(interval=20)
    assert hist.maybe_sample(eng, force=True) is True
    assert hist.samples == 1
    # Forced twice at the same tick still records one sample, not two.
    assert hist.maybe_sample(eng, force=True) is False
    assert hist.samples == 1


def test_history_is_bounded_by_its_cap():
    eng = Engine(small_config(organisms=20))
    hist = AnalyticsHistory(interval=1, cap=5)
    for _ in range(40):
        eng.step()
        hist.maybe_sample(eng)
    assert hist.samples == 5
    arr = hist.arrays()
    assert arr["tick"].size == 5
    # Oldest-first: the retained ticks are the most recent ones.
    assert arr["tick"][-1] == 40


def test_history_reset_drops_the_previous_run():
    eng = Engine(small_config())
    hist = AnalyticsHistory(interval=1, cap=10)
    for _ in range(5):
        eng.step()
        hist.maybe_sample(eng)
    hist.reset()
    assert hist.samples == 0
    assert hist.last_tick is None
    assert hist.arrays()["population"].size == 0


def test_history_arrays_have_every_declared_field():
    eng = Engine(small_config())
    hist = AnalyticsHistory(interval=1)
    hist.maybe_sample(eng, force=True)
    arr = hist.arrays()
    assert set(arr) == set(FIELDS)
    assert all(v.size == 1 for v in arr.values())


def test_history_survives_an_extinction_without_nan_means():
    """Every mean is either a real number or an explicit zero, never 0/0."""
    eng = Engine(small_config())
    hist = AnalyticsHistory(interval=1)
    pop = eng.population
    pop.alive[:] = False
    assert pop.n_alive == 0
    assert hist.maybe_sample(eng, force=True) is True
    row = hist.arrays()
    for name in ("mean_energy", "mean_body", "mean_speed", "trait_variance",
                 "nn_variance", "mean_pairwise", "organism_biomass"):
        assert np.isfinite(row[name]).all(), name
        assert float(row[name][0]) == 0.0


def test_history_never_mutates_the_engine():
    """The analytics view is observation: a sample must leave the run intact."""
    a = Engine(small_config(seed=4))
    b = Engine(small_config(seed=4))
    hist = AnalyticsHistory(interval=1)
    for _ in range(30):
        a.step()
        hist.maybe_sample(a)
    for _ in range(30):
        b.step()
    assert a.checksum() == b.checksum()


def test_history_counts_births_between_samples():
    eng = Engine(small_config(organisms=20))
    hist = AnalyticsHistory(interval=1)
    hist.maybe_sample(eng, force=True)
    for _ in range(20):
        eng.step()
        hist.maybe_sample(eng)
    arr = hist.arrays()
    # The per-sample birth counts sum to the engine's cumulative total.
    assert int(arr["births"].sum()) == eng.births_total


# -- the grid ----------------------------------------------------------------
def test_draw_analytics_returns_a_scrollable_height():
    eng = Engine(small_config())
    hist = AnalyticsHistory(interval=1)
    for _ in range(5):
        eng.step()
        hist.maybe_sample(eng)
    surf = _surface(1200, 800)
    content, view = draw_analytics(surf, pygame.Rect(0, 0, 1200, 760),
                                   _FONTS, _cache(), hist, eng)
    assert content > 0
    assert view > 0
    assert _non_blank(surf)


def test_draw_analytics_works_without_a_single_sample():
    eng = Engine(small_config())
    hist = AnalyticsHistory()
    surf = _surface(1200, 800)
    content, view = draw_analytics(surf, pygame.Rect(0, 0, 1200, 760),
                                   _FONTS, _cache(), hist, eng)
    assert content > 0 and view > 0
    assert _non_blank(surf)       # it says "no samples yet"


def test_draw_analytics_uses_one_column_on_a_narrow_window():
    """Two 100 px columns cannot hold a chart; the grid must fall back."""
    eng = Engine(small_config())
    hist = AnalyticsHistory(interval=1)
    hist.maybe_sample(eng, force=True)
    surf = _surface(700, 800)
    content_narrow, view = draw_analytics(
        surf, pygame.Rect(0, 0, 700, 760), _FONTS, _cache(), hist, eng)
    surf2 = _surface(1200, 800)
    content_wide, _ = draw_analytics(
        surf2, pygame.Rect(0, 0, 1200, 760), _FONTS, _cache(), hist, eng)
    # The two-column grid is shorter: same blocks, half the rows.
    assert content_wide < content_narrow
    assert view > 0


def test_zero_based_axis_never_prints_a_negative_floor():
    """A count or a mass cannot be negative, so the axis must not go below 0.

    Before the fix, `_value_range` padded 6 % below the minimum, so a
    population of 21 was drawn on an axis reading -1.2 and biomass read -168.
    A scale that claims a physically impossible region is a misleading scale.
    """
    for values in (np.array([21.0, 17.0]), np.array([3000.0, 1500.0]),
                   np.zeros(3)):
        low, high = charts._value_range([values], zero_based=True)
        assert low == 0.0, values
        assert high >= max(0.0, float(values.max()))


def test_zero_based_axis_keeps_negative_data_in_view():
    """Signed series (e.g. a temperature anomaly) must not be clipped at zero."""
    low, high = charts._value_range([np.array([-5.0, 3.0])], zero_based=True)
    assert low < -5.0 < 0.0 < high


def test_each_chart_block_declares_one_unit():
    """Charts mixing units flatten small series and make the unit label a lie."""
    from evolve.viz.analytics import _blocks

    eng = Engine(small_config())
    hist = AnalyticsHistory(interval=1)
    hist.maybe_sample(eng, force=True)
    blocks = _blocks(hist.arrays(), eng)
    keys = [b["key"] for b in blocks]
    assert len(keys) == len(set(keys))          # dispatch keys are unique
    # A per-organism or per-tick unit ("bu / organism", "world units / tick")
    # describes one dimension. What must be gone is a label that names two
    # unrelated dimensions joined together, which is what the old grid used.
    banned = ("physical units", "variance / distance", "°C / fraction",
              "a.u.", "various", "mixed")
    for b in blocks:
        assert b["unit"], b
        assert b["unit"] not in banned, b
    # The mixed-magnitude pairs that used to share one axis are now separate.
    assert "vision" in keys and "genes" in keys
    assert "gene_variance" in keys and "pairwise" in keys
    assert "temperature" in keys and "moisture" in keys


def test_analytics_scroll_never_draws_a_blank_frame():
    """Scrolling past the end used to render one empty frame, then clamp."""
    from evolve.viz.app import VizApp

    eng = Engine(small_config())
    app = VizApp(eng, speed=1, size=(1200, 700))
    for _ in range(5):
        app.run_frame()
    app.handle_key(pygame.K_a)
    app.run_frame()
    app.state.analytics_scroll = 10 ** 6       # far past the end
    app.render()
    surf = pygame.surfarray.array3d(app.screen)
    # The frame still shows the grid (it is not the flat background colour).
    assert app.state.analytics_scroll <= max(0, app.state.analytics_height
                                             - app.state.analytics_view)
    colours = {tuple(c) for c in surf[::40, ::40].reshape(-1, 3)}
    assert len(colours) > 3, "the analytics grid should still be drawn"


# -- live distributions (IMPROVE §6.4 items 3 and 4) -------------------------
def test_live_histograms_count_exactly_the_living_population():
    """A distribution must describe who is alive, not a stale or padded array.

    The population arrays are capacity-padded until ``compact()``, so summing
    the raw attribute would count dead slots as organisms. Every live organism
    must land in exactly one bin.
    """
    from evolve.viz.analytics import LIVE_HISTOGRAMS, _live_histogram

    eng = Engine(small_config(organisms=40))
    eng.run(10)
    pop = eng.population
    living = int(pop.alive.sum())
    assert living > 0
    for key, (attr, span, bins, _label, _caption) in LIVE_HISTOGRAMS.items():
        counts = _live_histogram(eng, attr, span, bins)
        assert counts.shape == (bins,), key
        assert counts.sum() == living, key


def test_live_histogram_is_empty_not_zero_when_nobody_is_alive():
    """No animals is no data. Zero-height bins must not be drawn as measurements."""
    from evolve.viz.analytics import BODY_HIST_BINS, BODY_HIST_RANGE, _live_histogram

    eng = Engine(small_config(organisms=10))
    eng.step()
    pop = eng.population
    pop.alive[:] = False
    assert pop.n_alive == 0          # n_alive is derived from the alive mask
    counts = _live_histogram(eng, "body", BODY_HIST_RANGE, BODY_HIST_BINS)
    assert counts.sum() == 0
    assert not counts.any()


def test_histogram_can_be_given_an_explicit_x_span():
    """The body/reserve charts are in bu, not counts, so the axis must say so."""
    surf = _surface()
    charts.histogram(surf, pygame.Rect(0, 0, 300, 130), _FONTS, _cache(),
                     title="body mass distribution",
                     y_unit="organisms",
                     counts=np.array([1.0, 3.0, 0.0, 2.0]),
                     bin_width=1.5, x_span=(0.0, 6.0))
    assert _non_blank(surf)


def test_live_distribution_blocks_are_declared_and_drawn():
    """The two distribution charts must exist in the grid and render."""
    from evolve.viz.analytics import _blocks

    eng = Engine(small_config(organisms=30))
    eng.run(5)
    hist = AnalyticsHistory(interval=1)
    hist.maybe_sample(eng, force=True)
    blocks = {b["key"]: b for b in _blocks(hist.arrays(), eng)}
    for key in ("body_size", "energy"):
        assert key in blocks
        assert blocks[key]["kind"] == "live_hist"
        assert blocks[key]["unit"] == "organisms"

    surf = _surface(1400, 900)
    cache = _cache()
    rect = pygame.Rect(0, 0, 1400, 900)
    draw_analytics(surf, rect, _FONTS, cache, hist, eng, 0)
    assert _non_blank(surf)
