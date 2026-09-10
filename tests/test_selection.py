import torch

from irag.selection import select_working_set


def test_k_none_returns_all():
    scores = torch.tensor([0.1, 0.9, 0.5])
    sel = select_working_set(scores, None, None)
    assert sel.tolist() == [0, 1, 2]


def test_k_zero_returns_all():
    scores = torch.tensor([0.1, 0.9, 0.5])
    sel = select_working_set(scores, None, 0)
    assert sel.tolist() == [0, 1, 2]


def test_k_ge_n_returns_all():
    scores = torch.tensor([0.1, 0.9, 0.5])
    sel = select_working_set(scores, None, 10)
    assert sel.tolist() == [0, 1, 2]


def test_top_k_selects_highest_scores():
    scores = torch.tensor([0.1, 0.9, 0.5, 0.3])
    sel = select_working_set(scores, None, 2, "top_k")
    assert sel.tolist() == [1, 2]  # ascending candidate order


def test_top_k_breaks_ties_by_candidate_index():
    scores = torch.tensor([0.5, 0.9, 0.5, 0.9])
    sel = select_working_set(scores, None, 2, "top_k")
    assert sel.tolist() == [1, 3]  # earlier index wins among ties


def test_top_k_without_labels_is_default_for_label_aware():
    scores = torch.tensor([0.1, 0.9, 0.5])
    sel = select_working_set(scores, None, 2, "label_aware")
    assert sel.tolist() == [1, 2]  # falls back to top_k (inference condition)


def test_label_aware_keeps_all_positives():
    scores = torch.tensor([0.05, 0.9, 0.2, 0.8])
    labels = torch.tensor([1, 0, 0, 0])
    sel = select_working_set(scores, labels, 2, "label_aware")
    # the positive (index 0) plus the highest-scoring negative (index 1, 0.9)
    assert sel.tolist() == [0, 1]


def test_label_aware_more_positives_than_k_keeps_highest_scored():
    scores = torch.tensor([0.1, 0.9, 0.5, 0.3])
    labels = torch.tensor([1, 1, 1, 0])
    sel = select_working_set(scores, labels, 2, "label_aware")
    # k highest-scoring positives: 0.9 (index 1) and 0.5 (index 2)
    assert sel.tolist() == [1, 2]


def test_label_aware_fills_remaining_slots_with_negatives():
    scores = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.05])
    labels = torch.tensor([1, 0, 1, 0, 0, 0])
    sel = select_working_set(scores, labels, 4, "label_aware")
    # positives {0, 2} + two highest negatives {4 (0.7), 1 or 3 (0.2/0.1)} -> 4 and 3
    assert sel.tolist() == [0, 2, 3, 4]


def test_label_aware_fewer_than_k_candidates():
    scores = torch.tensor([0.3, 0.1])
    labels = torch.tensor([0, 0])
    sel = select_working_set(scores, labels, 8, "label_aware")
    assert sel.tolist() == [0, 1]


def test_selection_is_discrete_no_gradient():
    scores = torch.tensor([0.1, 0.9, 0.5], requires_grad=True)
    sel = select_working_set(scores.detach(), None, 2, "top_k")
    assert not sel.requires_grad


def test_unknown_policy_raises():
    scores = torch.tensor([0.1, 0.9])
    try:
        select_working_set(scores, None, 1, "random")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown policy")
