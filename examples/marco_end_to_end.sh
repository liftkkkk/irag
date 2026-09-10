#!/usr/bin/env bash
# End-to-end MS MARCO pipeline (paper Table 3 configuration).
#
# Download first (see scripts/prepare_marco.py docstring for URLs):
#   collection.tar (or collectionandqueries.tar.gz)
#   triples.train.small.tar.gz
#   top1000.dev.tar.gz
#   qrels.dev.small.tsv
set -e

RAW=data/marco/raw
DATA=data/marco
RUN=runs/marco_irag

# 1. data ---------------------------------------------------------------
python scripts/prepare_marco.py --marco_dir "$RAW" --out_dir "$DATA"

# 2. IRAG (400-token passages, 2 epochs) ----------------------------------
python scripts/train.py \
    --train_file "$DATA/train.jsonl" \
    --dev_file   "$DATA/dev.jsonl" \
    --output_dir "$RUN" \
    --eval_metric mrr@10 \
    --max_passage_length 400 \
    --num_epochs 2

# 3. controlled baseline ---------------------------------------------------
python scripts/train.py \
    --train_file "$DATA/train.jsonl" \
    --dev_file   "$DATA/dev.jsonl" \
    --output_dir runs/marco_base \
    --baseline \
    --eval_metric mrr@10 \
    --max_passage_length 400 \
    --num_epochs 2

# 4. rerank the dev set -----------------------------------------------------
python scripts/rerank.py \
    --checkpoint "$RUN/best" --test_file "$DATA/dev.jsonl" \
    --output "$RUN/dev.tsv" --format marco
python scripts/rerank.py \
    --checkpoint runs/marco_base/best --test_file "$DATA/dev.jsonl" \
    --output runs/marco_base/dev.tsv --format marco

# 5. evaluate + paired t-test -------------------------------------------------
python scripts/evaluate.py \
    --run "$RUN/dev.tsv" --qrels "$DATA/qrels.dev.small.tsv" \
    --metrics mrr@10 \
    --baseline_run runs/marco_base/dev.tsv \
    --ttest_metric mrr@10
