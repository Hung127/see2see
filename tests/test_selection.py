"""Tie-breaking contract for :func:`evolve.spatial.best_per_query`.

The optimization pass replaced three full ``np.lexsort`` + "first row per
query" selections with a min-reduction first. The winner must be *exactly* the
same, including when distances tie: ties fall through to the explicit keys in
order, and the last resort is the earliest candidate row. These tests pin that
against the slow reference implementation on tie-heavy data.
"""
from __future__ import annotations

import numpy as np
import pytest

from evolve.spatial import best_per_query


def _reference(rep, d, ties):
    """The pre-optimization selection: full lexsort, first row per query."""
    keys = tuple(reversed(ties)) + (d, rep)
    order = np.lexsort(keys)
    rep_sorted = rep[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = rep_sorted[1:] != rep_sorted[:-1]
    return rep_sorted[first], order[first]


@pytest.mark.parametrize("trial", range(8))
def test_best_per_query_matches_lexsort_with_many_ties(trial):
    rng = np.random.default_rng(trial)
    n_query = 120
    n_rows = 4000
    rep = np.sort(rng.integers(0, n_query, n_rows))
    # Quantised distances create many exact ties on purpose.
    d = rng.integers(0, 15, n_rows).astype(float)
    tie_a = rng.integers(0, 40, n_rows)
    tie_b = rng.integers(0, 7, n_rows)

    expected_q, expected_row = _reference(rep, d, (tie_a, tie_b))
    got_q, got_row = best_per_query(rep, d, (tie_a, tie_b), n_query)

    assert np.array_equal(got_q, expected_q)
    assert np.array_equal(got_row, expected_row)


def test_best_per_query_empty_input():
    q, r = best_per_query(np.zeros(0, dtype=np.int64), np.zeros(0),
                          (), n_query=10)
    assert q.size == 0 and r.size == 0


def test_best_per_query_skips_queries_without_candidates():
    rep = np.array([0, 2, 2], dtype=np.int64)
    d = np.array([5.0, 9.0, 3.0])
    ties = (np.array([1, 1, 1]),)
    q, r = best_per_query(rep, d, ties, n_query=5)
    # Query 1 has no rows and must not appear.
    assert q.tolist() == [0, 2]
    assert d[r].tolist() == [5.0, 3.0]
