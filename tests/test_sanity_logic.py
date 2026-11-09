"""Unit tests for backend.sanity scoring/rank/tie logic (no network, no GPU)."""

import numpy as np
import pytest

from backend.sanity import exact_search, lexical_overlap, rank_metrics, select_queries


def test_rank_metrics_hits_and_rr():
    m = rank_metrics(["g"], ["a", "g", "b"])          # gold at rank 2
    assert m["top1hit"] is False and m["top5hit"] is True
    assert m["top10hit"] is True and m["first_gold_rank"] == 2
    assert m["reciprocal_rank@10"] == pytest.approx(0.5)

    m = rank_metrics(["g"], ["g"])
    assert m["top1hit"] and m["reciprocal_rank@10"] == 1.0
    m = rank_metrics(["x"], ["a", "b"])
    assert m["top10hit"] is False and m["reciprocal_rank@10"] == 0.0

    m = rank_metrics(["g1", "g2"], ["b", "g2", "g1"])  # multi-gold: first counts
    assert m["first_gold_rank"] == 2


def test_rank_metrics_beyond_list():
    assert rank_metrics(["g"], ["a"] * 10 + ["g"])["top10hit"] is False
    assert rank_metrics(["g"], ["g"] + ["a"] * 9)["reciprocal_rank@10"] == 1.0


def test_exact_search_order_and_scores():
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    vecs = rng.normal(size=(100, 4))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs[17] = q  # row 17 is the exact top-1
    rows, sims, extra = exact_search(q, vecs, top_k=3)
    assert rows[0] == 17
    assert sims[0] == pytest.approx(1.0, abs=1e-6)
    assert extra == []
    assert np.all(np.diff(sims) <= 1e-12)  # returned similarities descending



def test_exact_search_ties_reported_not_broken():
    q = np.array([1.0, 0.0])
    dup = np.array([
        [0.6, 0.8], [0.6, 0.8], [0.6, 0.8],  # three identical texts, sim 0.6
        [0.0, 1.0],                          # distinct text, sim 0
    ])
    rows, sims, extra = exact_search(q, dup, top_k=2)
    # stable argsort picks rows 0,1, but row 2 has the identical vector and
    # identical similarity: it must be surfaced as an extra tie, never dropped.
    assert set(rows) == {0, 1}
    assert extra == [2]
    assert len(set(sims)) == 1
    # all-inside case: no extra ties
    rows, _, extra = exact_search(q, dup, top_k=3)
    assert extra == []
    # and the full tie set equals the true cosine argmax set
    manual = np.flatnonzero(np.isclose(dup @ q, (dup @ q).max()))
    assert set(rows) | set(extra) == set(manual.tolist())


def test_select_queries_deterministic_and_performance_blind():
    queries = {str(i): f"q{i}" for i in range(100)}
    qrels = {str(i): [f"c{i}"] for i in range(100)}
    a = select_queries(queries, qrels)
    b = select_queries(queries, qrels)
    assert a == b and len(a) == 10  # seed 42 is fully deterministic
    # permutation-invariance of id set order in the dict must not matter
    reversed_qrels = dict(reversed(list(qrels.items())))
    assert select_queries(queries, reversed_qrels) == a


def test_lexical_overlap_is_diagnostic_only():
    d = lexical_overlap("Why do thioamides resist oxidation?", "Thioamide oxidation by peracids.")
    # exact-token matching only: plural 'thioamides' does NOT match singular
    assert d["overlap_terms_sample"] == ["oxidation"]
    assert d["query_coverage"] == pytest.approx(1 / 3, abs=1e-3)
    assert 0.0 <= d["query_coverage"] <= 1.0
