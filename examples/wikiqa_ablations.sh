#!/usr/bin/env bash
# Reproduce the paper's ablation tables (WIKIQA test set).
# Each configuration trains from scratch; every run shares the paper defaults.
set -e

DATA=data/wikiqa
COMMON=(--train_file "$DATA/train.jsonl" --dev_file "$DATA/dev.jsonl"
        --eval_metric map --num_epochs 3)

run () {  # run <name> <extra flags...>
    local name=$1; shift
    python scripts/train.py "${COMMON[@]}" \
        --output_dir "runs/wikiqa_$name" "$@"
    python scripts/rerank.py \
        --checkpoint "runs/wikiqa_$name/best" --test_file "$DATA/test.jsonl" \
        --output "runs/wikiqa_$name/test.tsv" --format wikiqa
    python scripts/evaluate.py \
        --run "runs/wikiqa_$name/test.tsv" --gold "$DATA/test.jsonl" \
        --metrics map,mrr
}

# ---- Table: CPA variants (attention ablation) ----
run cpa_full             --cpa_variant cpa_full
run cpa_pairwise_dot     --cpa_variant cpa_pairwise_dot
run cpa_global_only      --cpa_variant cpa_global_only
run cpa_additive_mlp     --cpa_variant cpa_additive_mlp
run cpa_scaled           --cpa_variant cpa_scaled
run cpa_temperature      --cpa_variant cpa_temperature

# ---- Table: full architecture ablation ----
run maxpool              --cpa_variant maxpool
run lstm                 --cpa_variant lstm
run none                 --cpa_variant none            # w/o CPA entirely
run baseline             --baseline                    # w/o Stage 3
run pipeline             --training_mode pipeline      # w/o joint training

# ---- Table: training-time selection policy ----
run sel_topk             --train_selection top_k
run sel_annealed         --train_selection annealed

# ---- Working-set size sweep (k) ----
for k in 1 2 4 8 12 16 20 24 32 48 64; do
    run "k$k" --k "$k"
done
run kinf --k 0    # no selection: CPA over all candidates
