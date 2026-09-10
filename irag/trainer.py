"""Joint and pipeline training for IRAG.

Training follows the paper's configuration: AdamW (peak LR 2e-5, linear
decay with 10% warm-up, weight decay 0.01), FP32, one optimizer step per
query (all of a query's candidates are encoded per step; ``grad_accum``
queries may be accumulated), dev evaluation per epoch with early stopping
(patience 2).

Two training modes:

* ``joint`` (default): L = L_c + lambda * L_f on the training working set;
  the shared BERT encoder receives gradient from both losses.
* ``pipeline``: phase 1 trains the scorer (encoder + projection + coarse
  head) with L_c on label-aware working sets, then freezes it; phase 2
  trains the interaction stage (CPA) alone with L_f.

Selection during training:

* ``label_aware`` (Algorithm 2, default)
* ``top_k`` (the deployment condition applied throughout training)
* ``annealed`` (label-aware, switching to top-k after
  ``anneal_switch_fraction`` of the total steps)
"""

import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from . import metrics as M
from .dataset import collate_one
from .pipeline import rerank_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _gold_labels_by_qid(dataset):
    gold = {}
    for rec in dataset.records:
        gold[rec.qid] = dict(zip(rec.ids, rec.labels))
    return gold


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        dev_dataset=None,
        output_dir="runs/irag",
        device="cuda",
        # optimisation (paper defaults)
        learning_rate=2e-5,
        num_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        grad_accum=1,
        patience=2,
        eval_metric="map",
        # data handling
        max_train_queries=None,
        eval_max_queries=None,
        num_workers=0,
        encode_batch_size=32,
        seed=42,
        log_every=50,
        gradient_checkpointing=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.output_dir = output_dir
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.grad_accum = max(1, grad_accum)
        self.patience = patience
        self.eval_metric = eval_metric
        self.max_train_queries = max_train_queries
        self.eval_max_queries = eval_max_queries
        self.num_workers = num_workers
        self.encode_batch_size = encode_batch_size
        self.seed = seed
        self.log_every = log_every
        self.gradient_checkpointing = gradient_checkpointing
        self._anneal_switch_step = None

        if eval_metric not in ("map", "mrr", "mrr@10"):
            raise ValueError("eval_metric must be one of map | mrr | mrr@10")

    # ------------------------------------------------------------------
    def _build_optimizer(self, trainable_params, num_training_steps):
        from transformers import get_linear_schedule_with_warmup

        no_decay = ("bias", "LayerNorm.weight")
        groups = [
            {
                "params": [p for n, p in trainable_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in trainable_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(groups, lr=self.learning_rate)
        warmup_steps = int(self.warmup_ratio * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _selection_for_step(self, step):
        cfg = self.model.config
        if cfg.train_selection == "top_k":
            return "top_k"
        if cfg.train_selection == "annealed":
            if self._anneal_switch_step is None:
                return "label_aware"
            return "label_aware" if step < self._anneal_switch_step else "top_k"
        return "label_aware"

    def _train_phase(
        self,
        phase_name,
        optimizer,
        scheduler,
        loss_key,
        num_epochs,
        freeze=None,
        best_metric=-math.inf,
        history=None,
        eval_use_coarse_only=False,
    ):
        """Run one training phase; returns the best dev metric."""
        model = self.model
        cfg = model.config

        if freeze is not None:
            for p in freeze:
                p.requires_grad_(False)

        train_ds = self.train_dataset
        if self.max_train_queries and self.max_train_queries < len(train_ds):
            indices = list(range(self.max_train_queries))
            train_ds = Subset(train_ds, indices)

        loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_one,
            num_workers=self.num_workers,
            drop_last=False,
        )

        steps_per_epoch = math.ceil(len(loader) / self.grad_accum)
        total_steps = steps_per_epoch * num_epochs
        if cfg.train_selection == "annealed":
            self._anneal_switch_step = int(cfg.anneal_switch_fraction * total_steps)

        best = best_metric
        epochs_no_improve = 0
        history = history if history is not None else []

        for epoch in range(num_epochs):
            model.train()
            running = {"loss": 0.0, "loss_c": 0.0, "loss_f": 0.0, "steps": 0}
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(loader, desc=f"{phase_name} epoch {epoch}")
            for it, batch in enumerate(pbar):
                labels = batch["labels"]
                batch = {
                    k: (v.to(self.device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                step_index = epoch * steps_per_epoch + it // self.grad_accum
                out = model(
                    batch,
                    k=cfg.k,
                    selection=self._selection_for_step(step_index),
                    encode_batch_size=self.encode_batch_size,
                )
                loss = out[loss_key]
                (loss / self.grad_accum).backward()

                running["loss"] += loss.item()
                running["loss_c"] += out.get("loss_c", torch.tensor(0.0)).item()
                running["loss_f"] += out.get("loss_f", torch.tensor(0.0)).item()
                running["steps"] += 1

                if (it + 1) % self.grad_accum == 0 or it + 1 == len(loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if (it + 1) % self.log_every == 0:
                    pbar.set_postfix(
                        loss=running["loss"] / running["steps"],
                        Lc=running["loss_c"] / running["steps"],
                        Lf=running["loss_f"] / running["steps"],
                    )

            msg = (
                f"[{phase_name}] epoch {epoch}: "
                f"loss={running['loss'] / max(running['steps'], 1):.4f} "
                f"Lc={running['loss_c'] / max(running['steps'], 1):.4f} "
                f"Lf={running['loss_f'] / max(running['steps'], 1):.4f}"
            )

            if self.dev_dataset is not None:
                metric = self.evaluate(use_coarse_only=eval_use_coarse_only)
                history.append({"phase": phase_name, "epoch": epoch, self.eval_metric: metric})
                msg += f" dev_{self.eval_metric}={metric:.4f}"
                if metric > best:
                    best = metric
                    epochs_no_improve = 0
                    self._save(os.path.join(self.output_dir, "best"))
                else:
                    epochs_no_improve += 1
            else:
                self._save(os.path.join(self.output_dir, "best"))
            print(msg)

            if self.dev_dataset is not None and epochs_no_improve >= self.patience:
                print(f"[{phase_name}] early stopping (patience {self.patience})")
                break

        if freeze is not None:
            for p in freeze:
                p.requires_grad_(True)
        if best == -math.inf:
            best = None
        return best, history

    # ------------------------------------------------------------------
    def evaluate(self, use_coarse_only=False):
        dev = self.dev_dataset
        if self.eval_max_queries and self.eval_max_queries < len(dev):
            dev = Subset(dev, list(range(self.eval_max_queries)))
            dev = _SubsetWithRecords(dev, dev.dataset)
        rankings, _ = rerank_dataset(
            self.model,
            dev,
            self.device,
            k=self.model.config.k,
            encode_batch_size=self.encode_batch_size,
            use_coarse_only=use_coarse_only,
            progress=False,
        )
        gold = _gold_labels_by_qid(dev)
        ranked_labels = {
            qid: [gold.get(qid, {}).get(cid, 0) for cid in cids]
            for qid, cids in rankings.items()
        }
        if self.eval_metric == "map":
            return M.mean_average_precision(ranked_labels)
        if self.eval_metric == "mrr":
            return M.mean_reciprocal_rank(ranked_labels)
        return M.mrr_at_k(ranked_labels, 10)

    def _save(self, directory):
        self.model.save(directory, tokenizer=self.tokenizer)
        print(f"[trainer] checkpoint saved to {directory}")

    # ------------------------------------------------------------------
    def train(self):
        set_seed(self.seed)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        self.model.to(self.device)

        cfg = self.model.config
        history = []

        train_len = len(self.train_dataset)
        if self.max_train_queries:
            train_len = min(train_len, self.max_train_queries)
        steps_per_epoch = math.ceil(train_len / self.grad_accum)

        if cfg.training_mode == "joint":
            num_steps = steps_per_epoch * self.num_epochs
            optimizer, scheduler = self._build_optimizer(
                [(n, p) for n, p in self.model.named_parameters() if p.requires_grad],
                num_steps,
            )
            best, history = self._train_phase(
                "joint", optimizer, scheduler, "loss", self.num_epochs, history=history
            )
        elif cfg.training_mode == "pipeline":
            # Phase 1: scorer only (encoder + projection + coarse head).
            interaction_params = []
            if self.model.interaction is not None:
                interaction_params = list(self.model.interaction.parameters())
            num_steps_1 = steps_per_epoch * self.num_epochs
            opt1, sched1 = self._build_optimizer(
                [
                    (n, p)
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and (self.model.interaction is None or not n.startswith("interaction"))
                ],
                num_steps_1,
            )
            best, history = self._train_phase(
                "pipeline/scorer",
                opt1,
                sched1,
                "loss_c",
                self.num_epochs,
                freeze=interaction_params or None,
                history=history,
                eval_use_coarse_only=True,
            )
            # Phase 2: interaction stage only, encoder frozen.
            encoder_params = [
                p
                for n, p in self.model.named_parameters()
                if not n.startswith("interaction")
            ]
            opt2, sched2 = self._build_optimizer(
                [(n, p) for n, p in self.model.named_parameters() if n.startswith("interaction")],
                steps_per_epoch * self.num_epochs,
            )
            best, history = self._train_phase(
                "pipeline/cpa",
                opt2,
                sched2,
                "loss_f",
                self.num_epochs,
                freeze=encoder_params,
                history=history,
            )
        else:
            raise ValueError(f"unknown training_mode {cfg.training_mode}")

        self._save(os.path.join(self.output_dir, "last"))
        return best, history


class _SubsetWithRecords:
    """Adapter so rerank/gold helpers work on a truncated dev set."""

    def __init__(self, subset, base):
        self._subset = subset
        self.records = [base.records[i] for i in subset.indices]

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        return self._subset[idx]
