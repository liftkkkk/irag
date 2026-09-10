#!/usr/bin/env python
"""Rerank a candidate list with a trained IRAG checkpoint (Algorithm 1).

Selection at inference is label-free: the k highest coarse-scoring passages
form the working set, CPA produces their final scores, and the remaining
candidates follow below in coarse-score order.

Examples
--------
WIKIQA (3-column TSV run)::

    python scripts/rerank.py \
        --checkpoint runs/wikiqa_irag/best \
        --test_file data/wikiqa/test.jsonl \
        --output runs/wikiqa_irag/test.tsv --format wikiqa

MS MARCO (TREC-style run)::

    python scripts/rerank.py \
        --checkpoint runs/marco_irag/best \
        --test_file data/marco/dev.jsonl \
        --output runs/marco_irag/dev.tsv --format marco
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from irag.dataset import RerankDataset
from irag.modeling import IRAGModel
from irag.pipeline import rerank_dataset, write_marco_run, write_wikiqa_run


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="directory saved by train.py")
    p.add_argument("--test_file", required=True, help="JSONL with candidate lists")
    p.add_argument("--output", required=True, help="run file to write")
    p.add_argument("--format", choices=["wikiqa", "marco"], default="wikiqa")
    p.add_argument(
        "--k", type=int, default=None,
        help="override the working-set size (default: checkpoint config; <=0 disables selection)",
    )
    p.add_argument("--encode_batch_size", type=int, default=32)
    p.add_argument(
        "--use_coarse_only", action="store_true",
        help="rank by coarse score only (scorer of pipeline mode)",
    )
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--run_id", default="irag")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    model = IRAGModel.from_checkpoint(args.checkpoint).to(device)
    model.eval()
    cfg = model.config

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    dataset = RerankDataset(
        args.test_file, tokenizer, cfg.max_query_length, cfg.max_passage_length
    )
    if args.max_queries and args.max_queries < len(dataset):
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(args.max_queries)))

    k = cfg.k if args.k is None else args.k
    rankings, stats = rerank_dataset(
        model,
        dataset,
        device,
        k=k,
        encode_batch_size=args.encode_batch_size,
        use_coarse_only=args.use_coarse_only,
    )

    if args.format == "wikiqa":
        write_wikiqa_run(rankings, args.output)
    else:
        write_marco_run(rankings, args.output, run_id=args.run_id)

    print(f"[rerank] {stats['n_queries']} queries -> {args.output}")
    answerable = stats["queries_with_positive"]
    if answerable:
        covered = stats["ws_has_positive"]
        print(
            f"[rerank] working set contains a gold passage for {covered}/{answerable} "
            f"answerable queries ({covered / answerable:.1%})"
        )


if __name__ == "__main__":
    main()
