#!/usr/bin/env python
"""Train IRAG (or the plain cross-encoder baseline) on a prepared JSONL dataset.

The dataset format is one query per line (produced by scripts/prepare_wikiqa.py
or scripts/prepare_marco.py)::

    {"qid": "...", "query": "...",
     "candidates": [{"id": "...", "text": "...", "label": 0}, ...]}

Examples
--------
WIKIQA (MAP on dev, paper Table 2)::

    python scripts/train.py \
        --train_file data/wikiqa/train.jsonl \
        --dev_file   data/wikiqa/dev.jsonl \
        --output_dir runs/wikiqa_irag \
        --eval_metric map

MS MARCO (MRR@10 on dev, paper Table 3)::

    python scripts/train.py \
        --train_file data/marco/train.jsonl \
        --dev_file   data/marco/dev.jsonl \
        --output_dir runs/marco_irag \
        --eval_metric mrr@10

Ablations (paper Tables 4-6) change a single flag::

    --cpa_variant cpa_pairwise_dot   # CPA attention variant (Table 4)
    --k 8                            # working-set size (Table 5)
    --train_selection top_k          # training-set construction (Table 6)
    --baseline                       # w/o IRAG: plain cross-encoder
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from irag.config import CPA_VARIANTS, SELECTION_POLICIES, TRAINING_MODES, IRAGConfig
from irag.dataset import RerankDataset
from irag.modeling import IRAGModel
from irag.trainer import Trainer


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ---- data ----------------------------------------------------------
    p.add_argument("--train_file", required=True, help="training JSONL")
    p.add_argument("--dev_file", default=None, help="dev JSONL (early stopping)")
    p.add_argument("--output_dir", default="runs/irag")

    # ---- architecture (IRAGConfig) --------------------------------------
    p.add_argument("--model_name_or_path", default="bert-base-uncased")
    p.add_argument(
        "--encoder_feature",
        default="cls",
        choices=["cls", "pooler"],
        help="final-layer [CLS] (paper) or tanh pooler output",
    )
    p.add_argument("--d", type=int, default=200, help="working dimension")
    p.add_argument("--cpa_variant", default="cpa_full", choices=CPA_VARIANTS)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda_fine", type=float, default=1.0, help="lambda in L_c + lambda*L_f")
    p.add_argument("--k", type=int, default=16, help="working-set size (<=0: no selection)")
    p.add_argument(
        "--use_irag",
        dest="use_irag",
        action="store_true",
        default=True,
        help="train the full IRAG model (default)",
    )
    p.add_argument(
        "--baseline",
        dest="use_irag",
        action="store_false",
        help="train the plain cross-encoder baseline instead of IRAG",
    )
    p.add_argument("--train_selection", default="label_aware", choices=SELECTION_POLICIES)
    p.add_argument("--anneal_switch_fraction", type=float, default=0.5)
    p.add_argument("--training_mode", default="joint", choices=TRAINING_MODES)
    p.add_argument("--loss_reduction", default="mean", choices=["mean", "sum"])
    p.add_argument("--max_query_length", type=int, default=40)
    p.add_argument("--max_passage_length", type=int, default=200)

    # ---- optimisation (paper defaults) ----------------------------------
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument(
        "--eval_metric", default="map", choices=["map", "mrr", "mrr@10"]
    )

    # ---- practical --------------------------------------------------------
    p.add_argument("--max_train_queries", type=int, default=None)
    p.add_argument("--eval_max_queries", type=int, default=None)
    p.add_argument("--encode_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p


def main():
    args = build_parser().parse_args()
    if not args.dev_file and args.patience and args.num_epochs > 1:
        print("[train] warning: no dev file given; early stopping disabled")

    config = IRAGConfig(
        model_name_or_path=args.model_name_or_path,
        encoder_feature=args.encoder_feature,
        d=args.d,
        cpa_variant=args.cpa_variant,
        dropout=args.dropout,
        lambda_fine=args.lambda_fine,
        k=args.k,
        use_irag=args.use_irag,
        train_selection=args.train_selection,
        anneal_switch_fraction=args.anneal_switch_fraction,
        training_mode=args.training_mode,
        loss_reduction=args.loss_reduction,
        max_query_length=args.max_query_length,
        max_passage_length=args.max_passage_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False
    )
    train_dataset = RerankDataset(
        args.train_file, tokenizer,
        config.max_query_length, config.max_passage_length,
    )
    dev_dataset = None
    if args.dev_file:
        dev_dataset = RerankDataset(
            args.dev_file, tokenizer,
            config.max_query_length, config.max_passage_length,
        )

    model = IRAGModel(config)
    print(f"[train] {sum(p.numel() for p in model.parameters()):,} parameters")
    if model.interaction is not None:
        n_int = sum(p.numel() for p in model.interaction.parameters())
        print(f"[train] interaction-stage parameters: {n_int:,}")

    trainer = Trainer(
        model,
        tokenizer,
        train_dataset,
        dev_dataset=dev_dataset,
        output_dir=args.output_dir,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        patience=args.patience,
        eval_metric=args.eval_metric,
        max_train_queries=args.max_train_queries,
        eval_max_queries=args.eval_max_queries,
        num_workers=args.num_workers,
        encode_batch_size=args.encode_batch_size,
        seed=args.seed,
        log_every=args.log_every,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    best, history = trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump({"best": best, "eval_metric": args.eval_metric, "history": history}, f, indent=2)
    if best is not None:
        print(f"[train] done; best dev {args.eval_metric} = {best:.4f}")
    else:
        print("[train] done (no dev set provided)")
    print(f"[train] checkpoints in {args.output_dir}/best and {args.output_dir}/last")


if __name__ == "__main__":
    main()
