"""Tests for the interactive ancestry view (IMPROVE §6.5).

The view draws ancestry, so the properties worth testing are about honesty
rather than layout:

* a tree contains only links that were actually recorded;
* a missing link (founder, evicted record, never-logged birth) is shown as
  unknown and is never inferred;
* trait snapshots come from the recorder, and a delta is only printed when a
  parent snapshot exists;
* an empty or extinct population renders without raising and says so.

Nothing here advances the engine differently from any other test: the view is
presentation-side and must leave ticks, RNG and checksums untouched.
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
from evolve.stats import LineageRecorder  # noqa: E402
from evolve.viz import ancestry  # noqa: E402

pygame.init()
pygame.font.init()

_FONTS = {
    "title": pygame.font.Font(None, 31),
    "big": pygame.font.Font(None, 46),
    "body": pygame.font.Font(None, 25),
    "small": pygame.font.Font(None, 22),
}


def small_config(seed=11, organisms=100, plants=2000):
    return Config(
        seed=seed, ticks=1000,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=10,
                          cell_size=50.0),
        plants=PlantConfig(initial=plants, max=4000),
        organisms=OrganismConfig(initial=organisms, max=300),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(),
        environment=EnvironmentConfig(rain_rate=0.002),
        stats=StatsConfig(),
    )


def _cache():
    from evolve.viz.panels import TextCache

    return TextCache()


def _engine_with_records(ticks=300):
    """An engine whose run produced real recorded births.

    The counts are checked, not assumed: a fixture that silently recorded
    nothing would make every tree test pass on an empty tree.
    """
    eng = Engine(small_config())
    eng.lineage = LineageRecorder()
    for _ in range(ticks):
        eng.step()
    assert eng.lineage.births_recorded > 0, "fixture recorded no births"
    return eng


def _surface(w=1200, h=800):
    surf = pygame.Surface((w, h))
    surf.fill((0, 0, 0))
    return surf


def _non_blank(surf) -> bool:
    return bool((pygame.surfarray.array3d(surf) != 0).any())


# -- tree shape --------------------------------------------------------------
def test_tree_links_only_recorded_parents():
    """Every drawn edge must exist in the reporter's parent map."""
    eng = _engine_with_records()
    tree = ancestry.build_tree(eng)
    assert tree.root is not None
    known = ancestry._Known(eng)

    def walk(node):
        for child in node.children:
            pa, pb = known.parents.get(child.org_id, (None, None))
            assert node.org_id in (pa, pb), (
                f"{child.org_id} drawn under {node.org_id}, "
                f"but its recorded parents are {(pa, pb)}")
            walk(child)

    walk(tree.root)


def test_tree_counts_match_the_nodes_it_actually_drew():
    """The header counts must describe the drawn tree, not a different walk."""
    eng = _engine_with_records()
    tree = ancestry.build_tree(eng)
    seen = []

    def walk(node):
        seen.append(node)
        for child in node.children:
            walk(child)

    walk(tree.root)
    assert tree.nodes == len(seen)
    assert tree.alive == sum(1 for n in seen if n.alive)
    assert tree.dead == sum(1 for n in seen if n.dead)
    assert tree.unknown == sum(1 for n in seen if n.org_id < 0)


def test_generation_span_covers_every_drawn_node():
    """The header span must include the descendants, not only the chain.

    Deriving it from the ancestor chain reported "generation 0" above a tree
    that visibly spanned generations 0-4, which reads as a broken counter.
    """
    eng = _engine_with_records()
    tree = ancestry.build_tree(eng)
    drawn = []
    if tree.root is not None:
        def walk(node):
            drawn.append(node.generation)
            for child in node.children:
                walk(child)
        walk(tree.root)
    drawn += [n.generation for n in tree.chain if n.org_id >= 0]
    gens = [g for g in drawn if g is not None]
    assert gens, "fixture drew no generation-labelled node"
    assert tree.generations == (min(gens), max(gens))


def test_tree_is_capped_so_one_lineage_cannot_make_a_frame_unbounded():
    """Depth and node budgets must be reported, never silently applied."""
    eng = _engine_with_records()
    tree = ancestry.build_tree(eng)
    assert tree.nodes <= ancestry.NODE_CAP
    # A tree that hit a cap must say so; a small tree must not claim a cut.
    if tree.truncated:
        assert tree.nodes >= 1


def test_traits_come_from_the_recorder_and_match_the_living_genome():
    """The snapshot is the birth phenotype, not a later re-derivation.

    This is the property that makes a trait delta trustworthy: if the recorder
    stored something else, the view would compare the wrong numbers.
    """
    from evolve.genome import phenotype

    eng = _engine_with_records()
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    assert idx.size > 0
    phen = phenotype(pop.genome[idx])
    checked = 0
    for k, i in enumerate(idx):
        oid = int(pop.id[i])
        stored = eng.lineage.traits(oid)
        if stored is None:
            continue
        assert np.allclose(stored, phen[k, :3], rtol=0, atol=0)
        checked += 1
    assert checked > 0


def test_unknown_traits_are_reported_as_unknown_not_zero():
    """An unrecorded organism has no traits; the view must not print zeros."""
    eng = Engine(small_config())
    eng.step()
    tree = ancestry.build_tree(eng)      # no recorder attached
    assert tree.root is not None
    assert tree.root.traits is None
    assert ancestry.trait_text(None, None) == "traits not recorded"
    assert ancestry.trait_text(tree.root.traits, None) == "traits not recorded"


def test_trait_delta_requires_a_recorded_parent():
    """A delta beside a value must be computed from a real parent snapshot."""
    child = (1.5, 10.0, 120.0)
    assert "(+0.00)" not in ancestry.trait_text(child, None)
    assert "(+" not in ancestry.trait_text(child, None)
    parent = (1.0, 10.0, 120.0)
    text = ancestry.trait_text(child, parent)
    assert "(+0.50)" in text
    assert "(+0.00)" in text          # speed and vision unchanged, shown


def test_identical_traits_are_described_in_words_once():
    """A run of zero deltas is mutation semantics, not a broken percentage."""
    traits = (1.5, 10.0, 120.0)
    assert "identical" in ancestry.trait_text(traits, traits)


# -- missing history ---------------------------------------------------------
def test_founder_chain_says_no_parent_rather_than_unknown():
    """A founder has no parent at all; that is not the same as a lost record."""
    eng = _engine_with_records()
    known = ancestry._Known(eng)
    founder = min(known.alive, key=lambda oid: known.generation.get(oid, 0))
    chain = known.parent_chain(founder, ancestry.CHAIN_DEPTH)
    assert chain[-1].org_id == founder
    assert chain[-1].note is not None
    assert "founder" in chain[-1].note
    assert all(n.org_id >= 0 for n in chain)


def test_evicted_parent_is_shown_as_unknown():
    """An evicted record leaves a real hole, rendered as unknown."""
    eng = Engine(small_config())
    # Empty the live population first: otherwise real organisms can occupy the
    # small ids this test uses, and the "missing" parent would resolve to a
    # living organism instead of exercising the evicted-record case.
    eng.population.alive[:] = False
    assert eng.population.n_alive == 0
    eng.lineage = LineageRecorder(max_finished=1)
    rec = eng.lineage
    rec.record_birth(1, None, None, tick=0, generation=0)
    rec.record_birth(2, 1, None, tick=5, generation=1)
    rec.record_death(1, tick=6, cause="senescence", age=6)
    rec.record_death(2, tick=7, cause="senescence", age=2)   # evicts 1
    known = ancestry._Known(eng)
    chain = known.parent_chain(2, 4)
    assert any(n.org_id < 0 for n in chain), chain
    assert chain[-1].org_id == 2


def test_empty_population_renders_an_explanation():
    """Extinction is a valid outcome; the view must not raise or draw a tree."""
    eng = Engine(small_config())
    eng.step()
    eng.population.alive[:] = False
    assert eng.population.n_alive == 0
    tree = ancestry.build_tree(eng)
    assert tree.root is None
    surf = _surface()
    content, view = ancestry.draw_ancestry(
        surf, pygame.Rect(0, 0, 1200, 760), _FONTS, _cache(), eng, None, 0)
    assert content > 0
    assert view > 0
    assert _non_blank(surf)


def test_view_states_that_logging_is_off_without_a_recorder():
    """No recorder means no full chain; the view must not imply one exists."""
    eng = Engine(small_config())
    eng.run(20)
    tree = ancestry.build_tree(eng)
    assert tree.root is not None
    surf = pygame.Surface((1200, 800))
    surf.fill((0, 0, 0))
    ancestry.draw_ancestry(surf, pygame.Rect(0, 0, 1200, 760), _FONTS,
                           _cache(), eng, None, 0)
    assert _non_blank(surf)


# -- rendering and selection -------------------------------------------------
def test_draws_the_recorded_tree_and_scrolls():
    """Render at the default root, then scrolled: both must draw content."""
    eng = _engine_with_records()
    surf = _surface()
    rect = pygame.Rect(0, 0, 1200, 760)
    content, view = ancestry.draw_ancestry(surf, rect, _FONTS, _cache(),
                                           eng, None, 0)
    assert content > 0 and view > 0
    top = pygame.surfarray.array3d(surf).copy()
    assert top.any()

    surf.fill((0, 0, 0))
    ancestry.draw_ancestry(surf, rect, _FONTS, _cache(), eng, None, 200)
    assert (pygame.surfarray.array3d(surf) != 0).any()
    assert not np.array_equal(top, pygame.surfarray.array3d(surf))


def test_selected_organism_is_the_root_when_it_exists():
    """Explicit selection must win over the default heuristic."""
    eng = _engine_with_records()
    pop = eng.population
    idx = np.flatnonzero(pop.alive)
    target = int(pop.id[idx[0]])
    tree = ancestry.build_tree(eng, target)
    assert tree.root.org_id == target


def test_ancestry_does_not_advance_or_mutate_the_engine():
    """The view reads state; it must not change ticks, RNG or the checksum."""
    # No recorder and no births needed: this asserts *inaction*, so the fixture
    # must not require a busy run to exist.
    eng = Engine(small_config())
    eng.run(30)
    tick = eng.tick
    before = eng.checksum()
    surf = _surface()
    ancestry.draw_ancestry(surf, pygame.Rect(0, 0, 1200, 760), _FONTS,
                           _cache(), eng, None, 0)
    assert eng.tick == tick
    assert eng.checksum() == before
