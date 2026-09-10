#!/usr/bin/env bash
# End-to-end WIKIQA pipeline (paper Table 2 configuration).
set -e

DATA=data/wikiqa
RUN=runs/wikiqa_irag

# 1. data ---------------------------------------------------------------
python scripts/prepare_wikiqa.py --out_dir "$DATA"

# 2. IRAG (joint training, label-aware selection, k=16; 3 epochs) --------
python scripts/train.py \
    --train_file "$DATA/train.jsonl" \
    --dev_file   "$DATA/dev.jsonl" \
    --output_dir "$RUN" \
    --eval_metric map \
    --num_epochs 3

# 3. controlled baseline: plain cross-encoder ---------------------------
python scripts/train.py \
    --train_file "$DATA/train.jsonl" \
    --dev_file   "$DATA/dev.jsonl" \
    --output_dir runs/wikiqa_base \
    --baseline \
    --eval_metric map \
    --num_epochs 3

# 4. rerank the test set --------------------------------------------------
python scripts/rerank.py \
    --checkpoint "$RUN/best" --test_file "$DATA/test.jsonl" \
    --output "$RUN/test.tsv" --format wikiqa
python scripts/rerank.py \
    --checkpoint runs/wikiqa_base/best --test_file "$DATA/test.jsonl" \
    --output runs/wikiqa_base/test.tsv --format wikiqa

# 5. evaluate + paired t-test ---------------------------------------------
python scripts/evaluate.py \
    --run "$RUN/test.tsv" --gold "$DATA/test.jsonl" \
    --metrics map,mrr \
    --baseline_run runs/wikiqa_base/test.tsv \
    --ttest_metric map
