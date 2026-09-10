#!/usr/bin/env python
"""Evaluate run files against gold labels (MAP / MRR / MRR@10 / recall@k).

Gold labels come from a prepared JSONL dataset (--gold, WIKIQA style) or an
MS MARCO qrels file (--qrels). Pass --baseline_run for a two-sided paired
t-test between the two systems on the shared queries.

Examples
--------
WIKIQA (MAP / MRR over questions with a relevant candidate)::

    python scripts/evaluate.py \
        --run runs/wikiqa_irag/test.tsv \
        --gold data/wikiqa/test.jsonl \
        --metrics map,mrr

MS MARCO (MRR@10 with the official msmarco_eval.py semantics)::

    python scripts/evaluate.py \
        --run runs/marco_irag/dev.tsv \
        --qrels data/marco/qrels.dev.small.tsv \
        --metrics mrr@10 \
        --baseline_run runs/marco_baseline/dev.tsv \
        --ttest_metric mrr@10
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irag import metrics as M
from irag.dataset import load_jsonl


def gold_from_jsonl(path):
    labels = {}
    for rec in load_jsonl(path):
        labels[rec["qid"]] = {
            c["id"]: int(c.get("label", 0)) for c in rec["candidates"]
        }
    return labels


def compute_metric(name, ranked):
    name = name.lower()
    if name == "map":
        return M.mean_average_precision(ranked)
    if name == "mrr":
        return M.mean_reciprocal_rank(ranked)
    if name.startswith("mrr@"):
        return M.mrr_at_k(ranked, int(name.split("@", 1)[1]))
    if name.startswith("recall@"):
        return M.recall_at_k(ranked, int(name.split("@", 1)[1]))
    raise SystemExit(f"unknown metric {name!r}")


def per_query_scores(name, ranked):
    """Per-query scores aligned by qid, for the paired t-test."""
    name = name.lower()
    out = {}
    if name == "map":
        for qid, labels in ranked.items():
            ap = M.average_precision(labels)
            if ap is not None:  # WIKIQA protocol: skip questions w/o relevant candidate
                out[qid] = ap
    elif name == "mrr":
        for qid, labels in ranked.items():
            rr = M.reciprocal_rank(labels)
            if rr is not None:
                out[qid] = rr
    elif name.startswith("mrr@"):
        k = int(name.split("@", 1)[1])
        out = {qid: M.reciprocal_rank_at_k(l, k) for qid, l in ranked.items()}
    elif name.startswith("recall@"):
        k = int(name.split("@", 1)[1])
        out = {qid: float(any(l > 0 for l in labels[:k])) for qid, labels in ranked.items()}
    else:
        raise SystemExit(f"unknown metric {name!r}")
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True, help="run file to evaluate")
    p.add_argument("--gold", help="prepared JSONL dataset with labels")
    p.add_argument("--qrels", help="MS MARCO qrels TSV (qid 0 pid rank)")
    p.add_argument("--metrics", default="map,mrr,mrr@10")
    p.add_argument("--baseline_run", default=None, help="second system for the t-test")
    p.add_argument(
        "--ttest_metric", default=None,
        help="metric used for the paired t-test (default: first of --metrics)",
    )
    p.add_argument("--output_json", default=None, help="also write results as JSON")
    args = p.parse_args()

    if bool(args.gold) == bool(args.qrels):
        raise SystemExit("exactly one of --gold or --qrels is required")

    labels_by_qid = (
        M.load_marco_qrels(args.qrels) if args.qrels else gold_from_jsonl(args.gold)
    )
    ranked = M.load_run(args.run, labels_by_qid)

    names = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    results = {name: compute_metric(name, ranked) for name in names}

    report = {"run": args.run, "n_queries": len(ranked), "metrics": results}

    if args.baseline_run:
        base_ranked = M.load_run(args.baseline_run, labels_by_qid)
        ttest_metric = (args.ttest_metric or names[0]).lower()
        a = per_query_scores(ttest_metric, ranked)
        b = per_query_scores(ttest_metric, base_ranked)
        t, pv = M.paired_t_test(a, b)
        report["paired_t_test"] = {
            "metric": ttest_metric,
            "baseline_run": args.baseline_run,
            "n_paired_queries": len(set(a) & set(b)),
            "t": t,
            "p_value": pv,
            "significant_at_0.05": pv < 0.05,
        }

    print(json.dumps(report, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
