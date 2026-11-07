"""16-channel sensor vector (DESIGN §5.2).

Sensors only report. The threat/prey/mate classes are *perceptual* categories
derived from observable phenotype (body size, attack, age) — never species
labels — and they never imply an action. "Nothing detected" is encoded as −1.

This module must not import :mod:`evolve.actions`; the audit test
``test_sensors_do_not_import_actions`` enforces it.
"""
from __future__ import annotations

import numpy as np

from .config import OrganismConfig
from .fields import Fields
from .genome import phenotype
from .organisms import Population
from .plants import Plants
from .spatial import SpatialGrid, best_per_query as _best_per_query
from .world import World

N_INPUTS = 16


def _nearest_visible(rep_q: np.ndarray, t_idx: np.ndarray, d: np.ndarray,
                     qx: np.ndarray, qy: np.ndarray, heading: np.ndarray,
                     rvis: np.ndarray, half_angle: np.ndarray,
                     world: World, tx: np.ndarray, ty: np.ndarray,
                     n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest visible target per query; ties by lowest target index.

    The distance test runs *before* the bearing is computed: the stencil hands
    back roughly 144 candidates per query while a vision cone keeps a handful,
    so computing ``arctan2`` for all of them was the single largest cost in
    sensing. Filtering first changes no result — a candidate outside the radius
    was already excluded by ``(d <= rvis) & (|bearing| <= half_angle)``.
    """
    if rep_q.size == 0:
        return np.full(n, -1, dtype=np.int64), np.full(n, np.inf), np.zeros(n)
    near = d <= rvis[rep_q]
    if not np.any(near):
        return np.full(n, -1, dtype=np.int64), np.full(n, np.inf), np.zeros(n)
    rep_q, t_idx, d = rep_q[near], t_idx[near], d[near]
    bearing = world.bearing(qx[rep_q], qy[rep_q], tx[t_idx], ty[t_idx], heading[rep_q])
    vis = np.abs(bearing) <= half_angle[rep_q]
    if not np.any(vis):
        return np.full(n, -1, dtype=np.int64), np.full(n, np.inf), np.zeros(n)
    rep_q, t_idx, d, bearing = rep_q[vis], t_idx[vis], d[vis], bearing[vis]
    # Nearest visible target per query, ties by lowest target index. The
    # candidate list is large (a 5x5 stencil) so this is the tick's busiest
    # selection; ``best_per_query`` gets the same winners without sorting all
    # of it. ``rep_q`` is already grouped by query because ``candidates`` emits
    # it in query order, so row order is the final tie-break in both forms.
    winners, rows = _best_per_query(rep_q, d, (t_idx,), n)
    best_idx = np.full(n, -1, dtype=np.int64)
    best_dist = np.full(n, np.inf)
    best_bearing = np.zeros(n)
    best_idx[winners] = t_idx[rows]
    best_dist[winners] = d[rows]
    best_bearing[winners] = bearing[rows]
    return best_idx, best_dist, best_bearing


def collect(pop: Population, plants: Plants, fields: Fields, world: World,
            cfg: OrganismConfig, carcasses=None,
            grid: SpatialGrid | None = None) -> np.ndarray:
    """Sensor matrix ``(n_alive, 16)`` in slot order (slots must be compacted)."""
    idx = np.flatnonzero(pop.alive)
    n = idx.size
    out = np.full((n, N_INPUTS), -1.0, dtype=np.float64)
    if n == 0:
        return out

    phen = phenotype(pop.genome[idx])
    mass = phen[:, 0]
    rvis = phen[:, 2]
    angle = phen[:, 3]
    lifespan = phen[:, 10]
    emax = 10.0 * mass ** 0.9
    half_angle = np.deg2rad(angle) / 2.0

    qx = pop.x[idx]
    qy = pop.y[idx]
    heading = pop.heading[idx]

    # -- organism neighbours (threat / prey / mate) ---------------------------
    if grid is None:
        grid = SpatialGrid(world, cell=80.0)
    grid.build(qx, qy)
    a, b = grid.pairs(radius_cells=2)
    src = np.concatenate([a, b])
    tgt = np.concatenate([b, a])
    # One displacement for both distance and bearing: the wrap was computed
    # twice per pair before.
    d, bearing = world.distance_and_bearing(qx[src], qy[src],
                                            qx[tgt], qy[tgt], heading[src])
    near = d <= rvis[src]
    vis = near & (np.abs(bearing) <= half_angle[src])

    m_src = mass[src]
    m_tgt = mass[tgt]
    age_tgt = pop.age[idx][tgt]
    attack_tgt = phen[tgt, 5]

    threat = vis & (m_tgt > 1.15 * m_src) & (attack_tgt > 1.0)
    prey = vis & (m_tgt < 0.85 * m_src)
    mate = (vis & (age_tgt >= cfg.maturity_age)
            & (m_tgt >= 0.7 * m_src) & (m_tgt <= 1.43 * m_src))

    for cls, mask in ((2, threat), (4, prey), (6, mate)):
        if not np.any(mask):
            continue
        s, t, dd, bb = src[mask], tgt[mask], d[mask], bearing[mask]
        order = np.lexsort((t, dd, s))
        s, t, dd, bb = s[order], t[order], dd[order], bb[order]
        first = np.ones(s.size, dtype=bool)
        first[1:] = s[1:] != s[:-1]
        rows = s[first]
        out[rows, cls] = 1.0 - dd[first] / rvis[rows]
        # Mate has a distance channel only (index 6): index 7 is own energy.
        if cls != 6:
            out[rows, cls + 1] = bb[first] / np.pi

    # -- food: plants + carcasses --------------------------------------------
    food_x, food_y = [], []
    p_idx = np.flatnonzero(plants.alive)
    if p_idx.size:
        food_x.append(plants.x[p_idx])
        food_y.append(plants.y[p_idx])
    if carcasses is not None:
        c_idx = np.flatnonzero(carcasses.alive)
        if c_idx.size:
            food_x.append(carcasses.x[c_idx])
            food_y.append(carcasses.y[c_idx])
    if food_x:
        tx = np.concatenate(food_x)
        ty = np.concatenate(food_y)
        tgrid = SpatialGrid(world, cell=80.0)
        tgrid.build(tx, ty)
        rep_q, t_idx, dist = tgrid.candidates(qx, qy, radius_cells=2)
        keep = dist <= 2.0 * tgrid.cell + 1e-9
        rep_q, t_idx, dist = rep_q[keep], t_idx[keep], dist[keep]
        f_idx, f_dist, f_bearing = _nearest_visible(
            rep_q, t_idx, dist, qx, qy, heading, rvis, half_angle, world, tx, ty, n)
        seen = f_idx >= 0
        out[seen, 0] = 1.0 - f_dist[seen] / rvis[seen]
        out[seen, 1] = f_bearing[seen] / np.pi

    # -- self and environment -------------------------------------------------
    out[:, 7] = np.clip(pop.energy[idx] / emax, -1.0, 1.0)
    out[:, 8] = np.clip(pop.health[idx] / cfg.health_max, -1.0, 1.0)
    out[:, 9] = np.clip(pop.age[idx] / lifespan, -1.0, 1.0)
    env = fields.sample(qx, qy)
    out[:, 10] = np.clip((env["temperature"] - 20.0) / 15.0, -1.0, 1.0)
    out[:, 11] = np.clip(2.0 * env["moisture"] - 1.0, -1.0, 1.0)
    cix, ciy = fields.cell_index(qx, qy)
    # Only the cells the organisms stand in are read, so bin the plants into a
    # flat index once and gather, instead of building the whole n x n field.
    # ``bincount(ix*n + iy)`` accumulates in plant order, the same order as
    # ``np.add.at(out, (ix, iy), w)``, so the sums are bit-identical.
    p_idx = np.flatnonzero(plants.alive)
    if p_idx.size:
        pix, piy = fields.cell_index(plants.x[p_idx], plants.y[p_idx])
        biomass = np.bincount(pix * fields.n + piy,
                              weights=plants.biomass[p_idx],
                              minlength=fields.n ** 2)
        out[:, 12] = np.clip(
            2.0 * biomass[cix * fields.n + ciy] / 10.0 - 1.0, -1.0, 1.0)
    counts = np.bincount(ciy * fields.n + cix, minlength=fields.n ** 2)
    out[:, 13] = np.clip(2.0 * counts[ciy * fields.n + cix] / 20.0 - 1.0, -1.0, 1.0)
    out[:, 14] = np.clip(env["pheromone"], -1.0, 1.0)
    threshold = phen[:, 7] * emax
    drive = (pop.energy[idx] - threshold) / np.maximum(emax - threshold, 1e-9)
    out[:, 15] = np.clip(drive, -1.0, 1.0)
    return out
