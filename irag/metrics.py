"""Evaluation metrics.

* WIKIQA: MAP and MRR over test questions. Following the standard WIKIQA
  evaluation protocol (and the original 2019 code), questions with no
  positively-labelled candidate are excluded from MAP/MRR.
* MS MARCO: MRR@10 with the semantics of the official
  ``msmarco_eval.py``: for every query present in both the run and the
  qrels, take the rank of the highest-ranked relevant passage; RR = 1/rank
  if the rank is <= 10, else 0.
* Significance: two-sided paired t-test over per-question average
  precision (WIKIQA) or per-query RR@10 (MS MARCO).
"""

from scipy import stats


# ----------------------------------------------------------------------
# per-question primitives
# ----------------------------------------------------------------------
def average_precision(ranked_labels):
    """AP for one question. ``ranked_labels`` is relevance in rank order.

    AP = (1/R) * sum_{i: rel_i} precision@i, R = number of relevant items.
    Returns None when the question has no relevant candidate.
    """
    hits = 0
    ap = 0.0
    for i, label in enumerate(ranked_labels, start=1):
        if label > 0:
            hits += 1
            ap += hits / i
    if hits == 0:
        return None
    return ap / hits


def reciprocal_rank(ranked_labels):
    rr = None
    for i, label in enumerate(ranked_labels, start=1):
        if label > 0:
            rr = 1.0 / i
            break
    return rr


def reciprocal_rank_at_k(ranked_labels, k=10):
    """MS MARCO semantics: RR = 0 when no relevant passage in the top k."""
    rr = reciprocal_rank(ranked_labels[:k])
    return 0.0 if rr is None else rr


# ----------------------------------------------------------------------
# aggregate metrics from (qid -> ranked candidate labels)
# ----------------------------------------------------------------------
def mean_average_precision(ranked_labels_by_qid):
    """MAP over questions that have at least one relevant candidate."""
    aps = []
    skipped = 0
    for ranked in ranked_labels_by_qid.values():
        ap = average_precision(ranked)
        if ap is None:
            skipped += 1
        else:
            aps.append(ap)
    if skipped:
        print(f"[metrics] {skipped} question(s) without relevant candidate skipped")
    return sum(aps) / len(aps) if aps else 0.0


def mean_reciprocal_rank(ranked_labels_by_qid):
    rrs = [
        reciprocal_rank(ranked)
        for ranked in ranked_labels_by_qid.values()
        if reciprocal_rank(ranked) is not None
    ]
    return sum(rrs) / len(rrs) if rrs else 0.0


def mrr_at_k(ranked_labels_by_qid, k=10):
    """MS MARCO dev MRR@10 over the given queries (all contribute)."""
    rrs = [reciprocal_rank_at_k(ranked, k) for ranked in ranked_labels_by_qid.values()]
    return sum(rrs) / len(rrs) if rrs else 0.0


def recall_at_k(ranked_labels_by_qid, k=10):
    """Fraction of queries with at least one relevant passage in the top k."""
    total = len(ranked_labels_by_qid)
    if total == 0:
        return 0.0
    hits = sum(1 for ranked in ranked_labels_by_qid.values() if any(l > 0 for l in ranked[:k]))
    return hits / total


# ----------------------------------------------------------------------
# significance testing
# ----------------------------------------------------------------------
def paired_t_test(per_item_scores_a, per_item_scores_b):
    """Two-sided paired t-test (a vs b) over aligned per-item scores."""
    import math

    keys = sorted(set(per_item_scores_a) & set(per_item_scores_b))
    if len(keys) < 2:
        raise ValueError("need at least two aligned items for a paired t-test")
    a = [per_item_scores_a[q] for q in keys]
    b = [per_item_scores_b[q] for q in keys]
    t, p = stats.ttest_rel(a, b)
    if math.isnan(t) or math.isnan(p):
        # zero-variance differences (e.g. identical runs): no difference detected
        return 0.0, 1.0
    return float(t), float(p)


# ----------------------------------------------------------------------
# run-file parsing
# ----------------------------------------------------------------------
def load_run(run_path, labels_by_qid):
    """Parse a run file into {qid: [label_of_rank1, label_of_rank2, ...]}.

    Two formats are accepted (auto-detected per file):

    * WIKIQA TSV with 3 columns: ``qid  cid  score`` (ranked by score desc)
    * TREC / MS MARCO run with >= 5 columns:
      ``qid  Q0  pid  rank  score  runid`` (ranked by the rank column)

    ``labels_by_qid`` maps qid -> {cid: label}; candidates missing from the
    gold mapping receive label 0.
    """
    entries = {}
    trec_format = None
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5 and parts[3].strip().lstrip("-").isdigit():
                trec_format = True
                qid, cid, rank = parts[0], parts[2], int(parts[3])
                entries.setdefault(qid, []).append((rank, cid))
            elif len(parts) == 3:
                qid, cid, score = parts[0], parts[1], float(parts[2])
                entries.setdefault(qid, []).append((score, cid))
            elif len(parts) == 2:
                qid, cid = parts
                entries.setdefault(qid, []).append((len(entries[qid]), cid))

    out = {}
    for qid, items in entries.items():
        if trec_format:
            items = sorted(items, key=lambda x: x[0])  # rank ascending
        else:
            items = sorted(items, key=lambda x: -x[0])  # score descending
        gold = labels_by_qid.get(qid, {})
        out[qid] = [gold.get(cid, 0) for _, cid in items]
    return out


def load_marco_qrels(qrels_path):
    """Parse an MS MARCO qrels file into {qid: {pid: 1}}."""
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, pid, label = parts[0], parts[2], int(parts[3])
            if label > 0:
                qrels.setdefault(qid, {})[pid] = 1
    return qrels
