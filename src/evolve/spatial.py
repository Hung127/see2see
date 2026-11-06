"""Uniform spatial grid with vectorised pair expansion.

Cells are indexed by ``cell`` world units; a query with ``radius_cells=r`` covers a
``(2r+1)^2`` stencil. Pair generation is fully vectorised: for each stencil offset the
points of cell ``c`` are expanded against the points of cell ``c+offset`` using a ragged
range trick, so the Python loop is over offsets (12), not over cells or organisms.

Distance filtering uses the *true* periodic distance, so stencil corners that exceed the
radius are dropped.
"""
from __future__ import annotations

import numpy as np

from .world import World


class SpatialGrid:
    def __init__(self, world: World, cell: float = 80.0) -> None:
        self.world = world
        self.cell = float(cell)
        # Tile the world exactly: ceil() leaves a partial final cell, which breaks the
        # periodic cell partition (points 160 units apart could land 3 cells apart).
        self.nx = max(1, int(round(world.width / self.cell)))
        self.ny = max(1, int(round(world.height / self.cell)))
        self.cell_x = world.width / self.nx
        self.cell_y = world.height / self.ny
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self._cell = np.zeros(0, dtype=np.int64)
        self._order = np.zeros(0, dtype=np.int64)
        self._starts = np.zeros(self.nx * self.ny + 1, dtype=np.int64)

    # -- build ---------------------------------------------------------------
    def build(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        ix = np.floor(self.x / self.cell_x).astype(np.int64) % self.nx
        iy = np.floor(self.y / self.cell_y).astype(np.int64) % self.ny
        cell = iy * self.nx + ix
        self._cell = cell
        self._order = np.argsort(cell, kind="stable")
        counts = np.bincount(cell, minlength=self.nx * self.ny)
        self._starts = np.concatenate([[0], np.cumsum(counts)])

    def _sorted(self) -> tuple[np.ndarray, np.ndarray]:
        return self._order, self._cell[self._order]

    def _counts(self) -> np.ndarray:
        return np.diff(self._starts)

    # -- pair expansion ------------------------------------------------------
    def _offsets(self, radius_cells: int) -> list[tuple[int, int]]:
        offs = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dy < 0 or (dy == 0 and dx < 0):
                    continue
                offs.append((dx, dy))
        return offs

    def _expand(self, off: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Pairs (a, b) with a in cell c and b in cell c+off (slot indices)."""
        dx, dy = off
        n_cells = self.nx * self.ny
        counts = self._counts()
        order, cells = self._sorted()
        # Source cell of each sorted point and the shifted target cell id.
        src_cell = cells
        tgt_cell = ((src_cell // self.nx + dy) % self.ny) * self.nx + \
                   ((src_cell % self.nx + dx) % self.nx)
        n_b = counts[tgt_cell]
        total = int(n_b.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        # For each sorted source point p (position in `order`), b ranges over
        # [start[tgt_cell], start[tgt_cell] + n_b).
        starts = self._starts
        rep_src = np.repeat(np.arange(order.size, dtype=np.int64), n_b)
        # offsets inside each target cell
        cum = np.cumsum(n_b) - n_b
        inner = np.arange(total, dtype=np.int64) - np.repeat(cum, n_b)
        b_pos = starts[tgt_cell][rep_src] + inner
        a_idx = order[rep_src]
        b_idx = order[b_pos]
        return a_idx, b_idx

    def pairs(self, radius_cells: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """All unordered pairs within ``radius_cells`` cells, as slot indices ``i < j``."""
        ai, bj = [], []
        for off in self._offsets(radius_cells):
            a, b = self._expand(off)
            if a.size:
                ai.append(a)
                bj.append(b)
        if not ai:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        a = np.concatenate(ai)
        b = np.concatenate(bj)
        keep = a != b
        a, b = a[keep], b[keep]
        d = self.world.distance(self.x[a], self.y[a], self.x[b], self.y[b])
        keep = d <= radius_cells * self.cell + 1e-9
        a, b = a[keep], b[keep]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        # Deduplicate (pairs can appear twice through different offsets) and sort.
        key = lo.astype(np.int64) * (self.x.size + 1) + hi
        _, first = np.unique(key, return_index=True)
        lo, hi = lo[first], hi[first]
        order = np.lexsort((hi, lo))
        return lo[order], hi[order]

    # -- queries -------------------------------------------------------------
    def nearest(self, qx: np.ndarray, qy: np.ndarray,
                tx: np.ndarray, ty: np.ndarray,
                mask: np.ndarray | None = None,
                radius_cells: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """Nearest target for each query point; ties resolve to the lowest target index."""
        qx = np.asarray(qx, dtype=np.float64)
        qy = np.asarray(qy, dtype=np.float64)
        if qx.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        if tx.size == 0:
            return np.full(qx.size, -1, dtype=np.int64), np.full(qx.size, np.inf)

        # Index the targets in this grid.
        self.build(tx, ty)
        nq = qx.size
        best = np.full(nq, np.inf)
        best_idx = np.full(nq, tx.size, dtype=np.int64)

        qix = np.floor(qx / self.cell_x).astype(np.int64) % self.nx
        qiy = np.floor(qy / self.cell_y).astype(np.int64) % self.ny
        counts = self._counts()
        order, _cells = self._sorted()
        starts = self._starts

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cell = ((qiy + dy) % self.ny) * self.nx + ((qix + dx) % self.nx)
                n_b = counts[cell]
                total = int(n_b.sum())
                if total == 0:
                    continue
                rep_q = np.repeat(np.arange(nq, dtype=np.int64), n_b)
                cum = np.cumsum(n_b) - n_b
                inner = np.arange(total, dtype=np.int64) - np.repeat(cum, n_b)
                t_pos = starts[cell][rep_q] + inner
                t_idx = order[t_pos]
                d = self.world.distance(qx[rep_q], qy[rep_q], tx[t_idx], ty[t_idx])
                if mask is not None:
                    d = np.where(mask[t_idx], d, np.inf)
                np.minimum.at(best, rep_q, d)
                # Track the smallest target index achieving the best distance so far.
                cand = np.where(d <= best[rep_q], t_idx, tx.size)
                np.minimum.at(best_idx, rep_q, cand)

        found = np.isfinite(best)
        out_idx = np.where(found & (best_idx < tx.size), best_idx, -1).astype(np.int64)
        return out_idx, best
