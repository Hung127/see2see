"""Interactive ancestry view (IMPROVE §6.5).

The inspector already answers "who were this animal's parents". This module
answers the wider questions the brief lists — parent/child relationships,
generation depth, lineage survival, trait change and branching — as a tree you
can scroll through while the simulation keeps running.

Two rules decide every line drawn here:

* **Recorded only.** A node exists because a birth or death was recorded, or
  because the animal is alive in the population arrays. A missing record (a
  founder, an organism born before logging was attached, or a record evicted
  from the bounded deque) is rendered as ``unknown`` and never as an inferred
  ancestor.
* **A lineage is not a species.** Nothing here groups organisms by genome
  similarity, and the view never calls a branch a species; it shows descent
  links that were actually observed.

The tree is rebuilt from engine state each frame and never stored, so it cannot
grow with run length and cannot feed back into the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pygame

from .palette import (
    BAD, GOOD, TEXT, TEXT_DIM, TEXT_FAINT, WARN,
)
from .panels import Panel, TextCache, _draw_scrollbar, _more_hint, _panel_fades

#: How many ancestor links to follow above the root.
CHAIN_DEPTH = 6
#: How deep (in generations below the root) the descendant tree goes.
TREE_DEPTH = 5
#: Children listed per node before the rest are summarised on their own line.
CHILD_CAP = 10
#: Hard cap on drawn nodes, so a prolific lineage cannot make a frame unbounded.
NODE_CAP = 300

#: Trait triple recorded at birth: ``(body bu, speed_max u/tick, vision u)``.
TRAIT_NAMES = ("body", "speed", "vision")
TRAIT_UNITS = ("bu", "u/tick", "u")


@dataclass
class Node:
    """One organism in the ancestor chain or descendant tree."""

    org_id: int
    record: dict | None
    traits: tuple[float, float, float] | None
    alive: bool
    generation: int | None
    children: list["Node"] = field(default_factory=list)
    hidden_children: int = 0
    # Why a link is not a real organism: a founder has no parent at all, while
    # an evicted record is genuinely unknown. The two must not read the same.
    note: str | None = None

    @property
    def recorded(self) -> bool:
        return self.record is not None

    @property
    def cause(self) -> str | None:
        return None if self.record is None else self.record.get("cause")

    @property
    def dead(self) -> bool:
        return bool(self.record is not None
                    and self.record.get("death_tick") is not None)

    def status(self) -> tuple[str, tuple[int, int, int]]:
        """``(text, colour)`` for the node's survival state."""
        if self.alive:
            return "alive", GOOD
        if self.record is None:
            return "unknown — no record", TEXT_FAINT
        if self.dead:
            tick = self.record.get("death_tick")
            cause = self.record.get("cause") or "cause unrecorded"
            return f"died tick {tick:,} ({cause})", BAD
        return "not living, no death recorded", WARN


@dataclass
class Tree:
    """Result of one ancestry build, plus the counts the view reports."""

    root: Node | None
    chain: list[Node]
    truncated: bool = False
    nodes: int = 0
    alive: int = 0
    dead: int = 0
    unknown: int = 0
    max_children: int = 0
    # Generation range over every node the view actually draws (chain and
    # descendant tree). Computed during the walk; ``None`` when no drawn node
    # carries a generation. Deriving this from the chain alone reported
    # "generation 0" while the tree below it spanned generations 0–4.
    gen_lo: int | None = None
    gen_hi: int | None = None

    @property
    def generations(self) -> tuple[int, int] | None:
        if self.gen_lo is None or self.gen_hi is None:
            return None
        return self.gen_lo, self.gen_hi

    def note_generation(self, value: int | None) -> None:
        if value is None:
            return
        if self.gen_lo is None or value < self.gen_lo:
            self.gen_lo = value
        if self.gen_hi is None or value > self.gen_hi:
            self.gen_hi = value


# ---------------------------------------------------------------------------
# Snapshot of what is known
# ---------------------------------------------------------------------------
class _Known:
    """One consistent read of the engine's recorded + living state."""

    def __init__(self, engine) -> None:
        recorder = getattr(engine, "lineage", None)
        self.recorder = recorder
        self.records: dict[int, dict] = {}
        if recorder is not None:
            for row in recorder.living():
                self.records[int(row["org_id"])] = row
            for row in recorder.finished:
                self.records[int(row["org_id"])] = row
        pop = engine.population
        idx = np.flatnonzero(pop.alive)
        self.alive: dict[int, int] = {int(pop.id[i]): int(i) for i in idx}
        # Parent/generation for every known organism. Records win where they
        # exist; live organisms born before logging fall back to their own
        # arrays, which is still recorded state, not inference.
        self.parents: dict[int, tuple[int | None, int | None]] = {}
        self.generation: dict[int, int] = {}
        for oid in self.alive:
            i = self.alive[oid]
            pa = int(pop.parent_a[i])
            pb = int(pop.parent_b[i])
            self.parents[oid] = (pa if pa >= 0 else None,
                                 pb if pb >= 0 else None)
            self.generation[oid] = int(pop.generation[i])
        for oid, row in self.records.items():
            self.parents[oid] = (row.get("parent_a"), row.get("parent_b"))
            self.generation[oid] = int(row["generation"])
            if row.get("death_tick") is not None:
                self.alive.pop(oid, None)
        self.ids = sorted(set(self.records) | set(self.alive))
        self._children: dict[int, list[int]] | None = None
        self._traits = (recorder.traits if recorder is not None else None)

    # -- queries ---------------------------------------------------------
    def traits(self, org_id: int) -> tuple[float, float, float] | None:
        return None if self._traits is None else self._traits(org_id)

    def node(self, org_id: int) -> Node:
        oid = int(org_id)
        return Node(org_id=oid,
                    record=self.records.get(oid),
                    traits=self.traits(oid),
                    alive=oid in self.alive,
                    generation=self.generation.get(oid))

    def children(self) -> dict[int, list[int]]:
        """Parent id -> child ids, built from recorded parent links only."""
        if self._children is not None:
            return self._children
        kids: dict[int, list[int]] = {}
        for child in self.ids:
            pa, pb = self.parents.get(child, (None, None))
            for parent in (pa, pb):
                if parent is None:
                    continue
                bucket = kids.setdefault(int(parent), [])
                if child not in bucket:
                    bucket.append(child)
        # Oldest first where a birth tick exists; unrecorded births last, then
        # by ID so the order is stable between frames.
        def order(cid: int):
            row = self.records.get(cid)
            tick = None if row is None else row.get("birth_tick")
            return (tick is None, tick if tick is not None else 0, cid)

        for bucket in kids.values():
            bucket.sort(key=order)
        self._children = kids
        return kids

    def parent_chain(self, org_id: int, depth: int) -> list[Node]:
        """Oldest-known ancestor first, requested organism last.

        Three cases are kept distinct, because collapsing them would invent
        ancestry:

        * a **founder** (or an organism born before logging was attached) has no
          parent id at all — the chain simply starts there and says so;
        * a parent **id** recorded at birth whose record was evicted, or which
          was never logged, is a real missing link — it is shown as ``unknown``;
        * the depth cap is a *display* limit, which the caller reports.
        """
        chain: list[Node] = []
        current: int | None = int(org_id)
        for _ in range(max(1, int(depth))):
            if current is None:
                break
            node = self.node(current)
            chain.append(node)
            if current not in self.ids:
                # Referenced as a parent but neither recorded nor alive.
                chain.append(Node(-1, None, None, False, None))
                break
            pa, _pb = self.parents.get(current, (None, None))
            if pa is None:
                node.note = "no parent recorded — founder or born before logging"
                break
            current = int(pa)
        chain.reverse()
        return chain

    def descendants(self, root_id: int, depth: int, child_cap: int,
                    node_cap: int) -> tuple[Node, bool, int]:
        """Descendant tree under ``root_id``; returns ``(root, cut, nodes)``.

        ``cut`` is true when the depth or node budget truncated the tree, so the
        view can say so instead of implying the family ended there.
        """
        kids = self.children()
        counter = [0]
        cut = [False]

        def build(oid: int, level: int) -> Node:
            node = self.node(oid)
            counter[0] += 1
            if level >= depth:
                if kids.get(oid):
                    cut[0] = True
                return node
            children = kids.get(oid, [])
            for cid in children[:child_cap]:
                if counter[0] >= node_cap:
                    cut[0] = True
                    break
                node.children.append(build(cid, level + 1))
            if len(children) > child_cap:
                node.hidden_children = len(children) - child_cap
            elif counter[0] >= node_cap and len(children) > len(node.children):
                node.hidden_children = len(children) - len(node.children)
            return node

        root = build(int(root_id), 0)
        return root, cut[0], counter[0]


def _descendant_counts(known: _Known, max_depth: int = 40) -> dict[int, int]:
    """Recorded descendant count per organism (0 when it has no recorded child).

    Iterative and depth-capped: a long run can build a chain hundreds of
    generations deep, and a recursive walk would hit Python's recursion limit on
    a view that must never take down the frame loop. Beyond ``max_depth`` the
    count stops growing, which only affects which default root is chosen — the
    tree itself is still built from the real links.
    """
    kids = known.children()
    cache: dict[int, int] = {}
    for oid in known.ids:
        stack: list[tuple[int, int, list[int]]] = [(oid, 0, [])]
        while stack:
            node, level, path = stack.pop()
            if node in cache or level > max_depth:
                continue
            children = kids.get(node, ())
            pending = [c for c in children if c not in cache and c not in path]
            if pending:
                stack.append((node, level, path + [node]))
                for child in pending:
                    if child not in path:
                        stack.append((child, level + 1, path + [node]))
                continue
            cache[node] = sum(1 + cache.get(c, 0) for c in children)
    return cache


def _default_root(known: _Known) -> int | None:
    """A living founder of the biggest recorded family, else any recorded one.

    Picking merely the deepest living organism usually lands on a leaf, which
    draws an empty descendant tree and hides the branching the view exists to
    show. Choosing the living organism with the most recorded descendants keeps
    the selection honest (it is still a real, currently living id) while making
    the default view informative. Ties break by generation, then by lowest id,
    so the default is stable between frames.
    """
    if known.alive:
        counts = _descendant_counts(known)
        return max(known.alive, key=lambda oid: (
            counts.get(oid, 0), known.generation.get(oid, 0), -oid))
    if known.records:
        return max(known.records, key=lambda oid: (
            known.records[oid]["generation"], -oid))
    return None


def build_tree(engine, root_id: int | None = None) -> Tree:
    """Build the ancestry view's data from recorded state only."""
    known = _Known(engine)
    if root_id is None:
        root_id = _default_root(known)
    if root_id is None:
        return Tree(root=None, chain=[])
    root_id = int(root_id)
    chain = known.parent_chain(root_id, CHAIN_DEPTH)
    root, cut, nodes = known.descendants(root_id, TREE_DEPTH, CHILD_CAP,
                                         NODE_CAP)
    tree = Tree(root=root, chain=chain, truncated=cut, nodes=nodes)

    def walk(node: Node) -> None:
        if node.org_id < 0:
            tree.unknown += 1
        elif node.alive:
            tree.alive += 1
        elif node.dead:
            tree.dead += 1
        else:
            tree.unknown += 1
        tree.note_generation(node.generation)
        tree.max_children = max(tree.max_children, len(node.children))
        for child in node.children:
            walk(child)

    walk(root)
    for node in chain:
        if node.org_id < 0:
            tree.unknown += 1
        else:
            tree.note_generation(node.generation)
    return tree


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt_trait(value: float, unit: str) -> str:
    if unit == "u/tick":
        return f"{value:.2f} {unit}"
    return f"{value:.2f} {unit}" if unit == "bu" else f"{value:.0f} {unit}"


def trait_text(child: tuple[float, float, float] | None,
               parent: tuple[float, float, float] | None) -> str:
    """Trait snapshot, with deltas only where a parent snapshot exists.

    When the parent was not recorded there is no honest delta to print, so the
    child's absolute values are shown with no delta at all. When the child is
    bit-identical to its parent the view says so in words once, instead of
    repeating ``(+0.00)`` three times: mutation is per-locus and most offspring
    inherit all three recorded traits unchanged, so a run of zero deltas is the
    normal case and must not look like a broken percentage.
    """
    if child is None:
        return "traits not recorded"
    if parent is not None and all(float(child[i]) == float(parent[i])
                                  for i in range(3)):
        return "traits identical to parent (no mutation at these loci)"
    parts = []
    for i, (name, unit) in enumerate(zip(TRAIT_NAMES, TRAIT_UNITS)):
        value = _fmt_trait(float(child[i]), unit)
        if parent is not None:
            value += f" ({float(child[i]) - float(parent[i]):+.2f})"
        parts.append(f"{name} {value}")
    return " · ".join(parts)


def _node_label(node: Node) -> tuple[str, tuple[int, int, int]]:
    if node.org_id < 0:
        return "unknown — no record for this link", TEXT_FAINT
    status, colour = node.status()
    gen = "gen ?" if node.generation is None else f"gen {node.generation}"
    return f"#{node.org_id}  {gen}  {status}", colour


def _tree_rows(node: Node, prefix: str = "", last: bool = True,
               parent_traits=None, rows: list | None = None,
               depth: int = 0, is_root: bool = False,
               ) -> list[tuple[str, str, tuple[int, int, int]]]:
    """Flatten the tree into ``(indent_prefix, text, colour, faint)`` rows.

    Indentation is plain ASCII (``|--``, ``\\`--``, ``|``). The bundled pygame
    font has no box-drawing or arrow glyphs — ``pygame.font.Font(None, 22)``
    renders them as the same tofu box as any unknown character, which was
    verified glyph-by-glyph — so a Unicode tree would draw a column of
    identical placeholders instead of a shape.

    A trait line is emitted only when it says something the parent line did not:
    the root's absolute snapshot, a genuine change, or a missing snapshot. Most
    offspring inherit all three recorded traits unchanged (mutation is
    per-locus), and printing "identical" under every node turned the tree into a
    wall of identical text that hid the few branches that did change.
    """
    if rows is None:
        rows = []
    if depth > TREE_DEPTH:
        return rows
    if depth == 0:
        marker, child_prefix = "", ""
    else:
        marker = "`-- " if last else "|-- "
        child_prefix = prefix + ("    " if last else "|   ")
    label, colour = _node_label(node)
    rows.append((prefix + marker, label, colour, False))
    if node.traits is None:
        rows.append((child_prefix + ("   " if depth else ""),
                     "traits not recorded", TEXT_FAINT, True))
    elif is_root or parent_traits is None or node.traits != parent_traits:
        body = trait_text(node.traits, None if is_root else parent_traits)
        rows.append((child_prefix + ("   " if depth else ""), body,
                     TEXT_DIM, True))
    if node.hidden_children:
        rows.append((child_prefix,
                     f"+{node.hidden_children} more recorded children "
                     f"(not drawn; child cap {CHILD_CAP})", TEXT_FAINT, True))
    for i, child in enumerate(node.children):
        _tree_rows(child, child_prefix, i == len(node.children) - 1,
                   node.traits, rows, depth + 1, False)
    return rows


def draw_ancestry(screen: pygame.Surface, rect: pygame.Rect,
                  fonts: dict[str, pygame.font.Font], cache: TextCache,
                  engine, root_id: int | None,
                  scroll: int = 0) -> tuple[int, int]:
    """Draw the ancestry view; return ``(content_height, view_height)``.

    ``root_id`` is the organism to centre the view on (normally the selected
    one). ``None`` falls back to the living organism with the most recorded
    descendants, and the header says that is what happened.
    """
    screen.fill((14, 16, 20), rect)
    inner = pygame.Rect(rect.x + 14, rect.y + 8,
                        max(80, rect.width - 28), max(40, rect.height - 16))
    screen.set_clip(inner)
    p = Panel(screen, inner, fonts, cache, scroll=scroll)

    tree = build_tree(engine, root_id)
    recorder = getattr(engine, "lineage", None)

    p.title("ancestry")
    if tree.root is None:
        p.text("No organism to trace: the population is empty and no lineage "
               "records exist for this run.", TEXT_DIM)
        content = p.content_height()
        screen.set_clip(None)
        return content, inner.height

    root = tree.root
    heading = (f"tracing #{root.org_id}" if root_id is not None
               else f"tracing #{root.org_id} — most-recorded descendants "
                    "(no organism selected)")
    p.wrapped(heading, TEXT, "body")
    gens = tree.generations
    span = ("generation unknown" if gens is None
            else (f"generation {gens[0]}" if gens[0] == gens[1]
                  else f"generations {gens[0]}–{gens[1]}"))
    p.row("recorded span", span)
    p.row("nodes drawn", f"{tree.nodes}"
                         + (" (cut at the depth/node cap)" if tree.truncated
                            else ""))
    p.row("survival in tree",
          f"{tree.alive} alive · {tree.dead} recorded dead · "
          f"{tree.unknown} unknown",
          GOOD if tree.alive else BAD)
    p.row("branching", f"max {tree.max_children} recorded children per node")
    if recorder is None:
        p.wrapped("Lineage logging is off for this run: only parent links "
                  "stored with the living population are shown. Start the "
                  "viewer with lineage logging (the viz command attaches it) "
                  "to see the full recorded chain.", WARN)
    else:
        p.wrapped(f"Recording is on: {recorder.births_recorded:,} births and "
                  f"{recorder.deaths_recorded:,} deaths logged; finished "
                  f"records are bounded to {recorder.finished.maxlen:,}.",
                  TEXT_FAINT)

    p.section("ancestors (oldest recorded first)")
    chain = tree.chain
    for i, node in enumerate(chain):
        label, colour = _node_label(node)
        step = "-> " if i else ""
        p.text(step + label, colour, "small")
        if node.note:
            p.wrapped(node.note, TEXT_FAINT, "small", indent=18)
        elif i:                     # delta versus the previous known snapshot
            prev = next((n.traits for n in reversed(chain[:i])
                         if n.traits is not None), None)
            p.wrapped(trait_text(node.traits, prev), TEXT_DIM, "small",
                      indent=18)
    if len(chain) >= CHAIN_DEPTH:
        p.wrapped(f"chain stops at the {CHAIN_DEPTH}-link display depth; "
                  f"older links are not shown, not absent.", TEXT_FAINT,
                  "small")

    p.section("descendants (recorded parent links)")
    p.wrapped("Traits are the phenotype recorded at birth. A bracketed value "
              "is the change from the recorded parent; a node is shown without "
              "values when its traits were not recorded. Trait lines appear "
              "only where the snapshot differs from the parent's, because most "
              "offspring inherit all three unchanged. A lineage is a descent "
              "chain, not a species.", TEXT_FAINT)
    rows = _tree_rows(root, is_root=True)
    for prefix, text, colour, faint in rows:
        p.text(prefix + text, colour, "small")

    p.section("how to read this")
    p.wrapped("Only observed links are drawn. Founders have no parent, and a "
              "record dropped by the bounded store is shown as unknown — the "
              "view never guesses an ancestor from genome similarity, ID order "
              "or position. The simulation keeps running behind this view.",
              TEXT_DIM, "small")

    content = p.content_height()
    screen.set_clip(None)
    _panel_fades(screen, inner, scroll, content, (14, 16, 20))
    state = _ScrollState(scroll, content, inner.height)
    _more_hint(screen, inner, scroll, content, fonts, cache)
    _draw_scrollbar(screen, rect, state)
    return content, inner.height


class _ScrollState:
    """Minimal state object for the shared scrollbar helper."""

    def __init__(self, scroll: int, content_height: int,
                 view_height: int) -> None:
        self.scroll = int(scroll)
        self.content_height = int(content_height)
        self.view_height = int(view_height)


def ancestry_rect(size: tuple[int, int], footer_h: int) -> pygame.Rect:
    """Full-window view rectangle above the footer."""
    return pygame.Rect(0, 0, size[0], size[1] - footer_h)
