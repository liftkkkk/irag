"""Stage 2 of IRAG: bounded working-set construction (CPL).

Two procedures are defined, matching the paper's Algorithms 1 and 2:

* ``top_k`` (inference-time selection, label-free, Algorithm 1): keep the
  k candidates with the highest coarse scores, ties broken by candidate
  index. Implemented exactly by a stable descending argsort, which is
  mathematically identical to streaming candidates through a size-k
  min-priority queue with index tie-breaking.

* ``label_aware`` (training-time construction, Algorithm 2): all positive
  passages are included and the remaining slots are filled with the
  highest-scoring negatives (online hard-negative selection). If a query
  has more than k positives, only the k highest-scoring positives are kept.

Selection is discrete: no gradient flows through the admit/discard decision.
Callers pass detached scores; gradients flow only through the representations
of the passages that end up inside the working set.
"""

import torch

SELECTION_POLICIES = ("label_aware", "top_k", "annealed")


def _top_k_by_score(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k highest scores, ties broken by candidate index.

    A stable descending sort keeps the original (index) order among ties,
    which is exactly the tie-breaking rule of Algorithm 1.
    """
    n = scores.size(0)
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    if k >= n:
        return torch.arange(n, device=scores.device)
    order = torch.argsort(scores, descending=True, stable=True)
    selected = torch.sort(order[:k]).values
    return selected


def select_working_set(
    scores: torch.Tensor,
    labels: torch.Tensor | None,
    k: int | None,
    policy: str = "label_aware",
) -> torch.Tensor:
    """Return the working-set indices, sorted by candidate index.

    Args:
        scores: 1-D tensor of coarse scores (higher = more relevant).
        labels: 1-D tensor of binary relevance labels, or ``None`` at
            inference time (forces ``top_k``).
        k: working-set size; ``None`` or ``<= 0`` means no selection
            (the working set is the full candidate list, k = infinity).
        policy: ``label_aware`` (training) or ``top_k`` (inference /
            score-only training).

    Returns:
        LongTensor of selected candidate indices in ascending order.
    """
    if policy not in SELECTION_POLICIES:
        raise ValueError(f"unknown selection policy {policy!r}")
    n = scores.size(0)
    if k is None or k <= 0 or k >= n:
        return torch.arange(n, device=scores.device)

    if labels is None or policy == "top_k":
        return _top_k_by_score(scores, k)

    # ---- label-aware construction (Algorithm 2) ----
    labels = labels.to(scores.device)
    pos = torch.nonzero(labels > 0, as_tuple=False).squeeze(-1)
    neg = torch.nonzero(labels <= 0, as_tuple=False).squeeze(-1)

    if pos.numel() > k:
        # keep only the k highest-scoring positives (ties by index);
        # _top_k_by_score returns positions within `pos`, map back to global
        pos = pos[_top_k_by_score(scores[pos], k)]

    remaining = max(k - pos.numel(), 0)
    if neg.numel() > remaining:
        neg = neg[_top_k_by_score(scores[neg], remaining)]

    selected = torch.cat([pos, neg]).sort().values
    return selected
