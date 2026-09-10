import pytest

from irag import metrics as M


# ----------------------------------------------------------------------
# per-question primitives
# ----------------------------------------------------------------------
def test_average_precision_single_relevant():
    assert M.average_precision([1, 0, 0]) == pytest.approx(1.0)


def test_average_precision_two_relevant():
    # precisions at ranks 1 and 3: (1/1 + 2/3) / 2
    assert M.average_precision([1, 0, 1]) == pytest.approx((1.0 + 2 / 3) / 2)


def test_average_precision_no_relevant_returns_none():
    assert M.average_precision([0, 0, 0]) is None


def test_reciprocal_rank():
    assert M.reciprocal_rank([0, 1, 0]) == 0.5
    assert M.reciprocal_rank([1, 0, 0]) == 1.0
    assert M.reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert M.reciprocal_rank([0, 0, 0]) is None


def test_reciprocal_rank_at_k_marco_semantics():
    # relevant at rank 12: RR@10 = 0 (MS MARCO convention)
    labels = [0] * 11 + [1]
    assert M.reciprocal_rank_at_k(labels, 10) == 0.0
    # uncapped RR would be 1/12
    assert M.reciprocal_rank(labels) == pytest.approx(1 / 12)
    # relevant at rank 10: RR@10 = 0.1
    assert M.reciprocal_rank_at_k([0] * 9 + [1], 10) == pytest.approx(0.1)


# ----------------------------------------------------------------------
# aggregates
# ----------------------------------------------------------------------
def test_map_skips_unanswerable_questions():
    ranked = {"q1": [1, 0], "q2": [0, 0]}
    assert M.mean_average_precision(ranked) == pytest.approx(1.0)


def test_map_mixed():
    ranked = {
        "q1": [1, 0, 1],   # AP = (1 + 2/3) / 2
        "q2": [0, 1],      # AP = 1/2
        "q3": [0, 0],      # skipped
    }
    expected = ((1.0 + 2 / 3) / 2 + 0.5) / 2
    assert M.mean_average_precision(ranked) == pytest.approx(expected)


def test_mrr_skips_unanswerable():
    ranked = {"q1": [0, 1], "q2": [0, 0]}
    assert M.mean_reciprocal_rank(ranked) == pytest.approx(0.5)


def test_mrr_at_k_counts_all_queries():
    # q2 contributes 0 (MS MARCO dev protocol)
    ranked = {"q1": [0, 1], "q2": [0, 0]}
    assert M.mrr_at_k(ranked, 10) == pytest.approx(0.25)


def test_recall_at_k():
    ranked = {
        "q1": [1, 0, 0],
        "q2": [0, 0, 1],
        "q3": [0, 0, 0],
    }
    assert M.recall_at_k(ranked, 1) == pytest.approx(1 / 3)
    assert M.recall_at_k(ranked, 3) == pytest.approx(2 / 3)


# ----------------------------------------------------------------------
# significance testing
# ----------------------------------------------------------------------
def test_paired_t_test_same_mean_not_significant():
    a = {"q1": 1.0, "q2": 0.5, "q3": 0.0}
    b = {"q1": 0.0, "q2": 0.5, "q3": 1.0}  # differences sum to zero
    t, p = M.paired_t_test(a, b)
    assert p == pytest.approx(1.0)


def test_paired_t_test_different_systems():
    a = {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0}
    b = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0}
    t, p = M.paired_t_test(a, b)
    assert p < 0.05


def test_paired_t_test_uses_intersection_of_queries():
    a = {"q1": 1.0, "q2": 0.5, "q3": 0.2}
    b = {"q2": 0.0, "q3": 1.0, "q4": 0.9}
    t, p = M.paired_t_test(a, b)  # q1 and q4 excluded
    assert 0.0 <= p <= 1.0


def test_paired_t_test_requires_two_shared_queries():
    with pytest.raises(ValueError):
        M.paired_t_test({"q1": 1.0}, {"q2": 0.0})


# ----------------------------------------------------------------------
# run-file parsing
# ----------------------------------------------------------------------
def test_load_run_wikiqa_format(tmp_path):
    run = tmp_path / "run.tsv"
    run.write_text(
        "q1\tc2\t9.0\n"
        "q1\tc0\t8.0\n"
        "q2\tc1\t5.0\n",
        encoding="utf-8",
    )
    labels = {
        "q1": {"c0": 1, "c1": 0, "c2": 0},
        "q2": {"c1": 1},
    }
    ranked = M.load_run(str(run), labels)
    assert ranked["q1"] == [0, 1]  # c2 (label 0) then c0 (label 1)
    assert ranked["q2"] == [1]


def test_load_run_trec_format(tmp_path):
    run = tmp_path / "run.txt"
    run.write_text(
        "q1\tQ0\tc2\t1\t30.0\tirag\n"
        "q1\tQ0\tc0\t2\t20.0\tirag\n"
        "q2\tQ0\tc1\t1\t10.0\tirag\n",
        encoding="utf-8",
    )
    labels = {
        "q1": {"c0": 1, "c2": 0},
        "q2": {"c1": 1},
    }
    ranked = M.load_run(str(run), labels)
    assert ranked["q1"] == [0, 1]
    assert ranked["q2"] == [1]


def test_load_marco_qrels(tmp_path):
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text(
        "q1\t0\tp1\t1\n"
        "q1\t0\tp2\t0\n"
        "q2\t0\tp3\t1\n",
        encoding="utf-8",
    )
    q = M.load_marco_qrels(str(qrels))
    assert q == {"q1": {"p1": 1}, "q2": {"p3": 1}}
