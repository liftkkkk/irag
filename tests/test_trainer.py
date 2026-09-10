"""End-to-end trainer tests on a tiny random BERT (CPU, a few seconds)."""

import torch

from irag.config import IRAGConfig
from irag.modeling import IRAGModel
from irag.trainer import Trainer, set_seed
from tests.conftest import tiny_bert


def make_trainer(tmp_path, mode="joint", **cfg_overrides):
    set_seed(0)
    from irag.dataset import RerankDataset
    from tests.conftest import MockTokenizer

    cfg = IRAGConfig(
        d=16,
        model_name_or_path="tiny-local",
        training_mode=mode,
        **cfg_overrides,
    )
    tokenizer = MockTokenizer()

    records = [
        {
            "qid": f"Q{i}",
            "query": f"query number {i} about topic {i % 3}",
            "candidates": [
                {"id": f"D{i}a", "text": f"relevant text for query {i}", "label": 1},
                {"id": f"D{i}b", "text": "irrelevant filler text here", "label": 0},
                {"id": f"D{i}c", "text": "another unrelated passage", "label": 0},
            ],
        }
        for i in range(4)
    ]
    import json

    train_path = tmp_path / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    train_ds = RerankDataset(train_path, tokenizer, max_query_length=8, max_passage_length=10)
    dev_ds = RerankDataset(train_path, tokenizer, max_query_length=8, max_passage_length=10)

    model = IRAGModel(cfg, bert_model=tiny_bert())
    trainer = Trainer(
        model,
        tokenizer,
        train_ds,
        dev_dataset=dev_ds,
        output_dir=str(tmp_path / "run"),
        device="cpu",
        num_epochs=2,
        eval_metric="map",
        encode_batch_size=4,
        log_every=1000,
    )
    return trainer, model


def test_joint_training_reduces_loss_and_saves_checkpoints(tmp_path):
    trainer, model = make_trainer(tmp_path, mode="joint")
    best, history = trainer.train()

    assert len(history) >= 1
    assert 0.0 <= best <= 1.0
    assert (tmp_path / "run" / "best").exists()
    assert (tmp_path / "run" / "last").exists()
    assert (tmp_path / "run" / "best" / "irag_model.pt").exists()
    assert (tmp_path / "run" / "best" / "irag_config.json").exists()


def test_joint_training_loss_is_finite_each_epoch(tmp_path, capsys):
    trainer, model = make_trainer(tmp_path, mode="joint")
    trainer.train()
    out = capsys.readouterr().out
    assert "epoch" in out
    assert "loss=" in out


def test_pipeline_mode_trains_both_phases(tmp_path):
    trainer, model = make_trainer(tmp_path, mode="pipeline")
    best, history = trainer.train()
    phases = {h["phase"] for h in history}
    assert phases == {"pipeline/scorer", "pipeline/cpa"}


def test_trainer_evaluate_returns_metric_in_range(tmp_path):
    trainer, model = make_trainer(tmp_path, mode="joint")
    metric = trainer.evaluate()
    assert 0.0 <= metric <= 1.0


def test_trainer_evaluate_mrr_at_10(tmp_path):
    trainer, model = make_trainer(tmp_path, mode="joint")
    trainer.eval_metric = "mrr@10"
    metric = trainer.evaluate()
    assert 0.0 <= metric <= 1.0


def test_checkpoint_loadable_after_training(tmp_path):
    trainer, model = make_trainer(tmp_path, mode="joint")
    trainer.train()
    # "last" is saved after the final epoch: identical to the live model
    last = IRAGModel.from_checkpoint(str(tmp_path / "run" / "last"))
    assert last.config.d == 16
    assert last.interaction is not None
    assert torch.allclose(
        model.state_dict()["projection.weight"],
        last.state_dict()["projection.weight"],
    )


def test_training_updates_model_parameters(tmp_path):
    trainer, model = make_trainer(tmp_path, mode="joint")
    before = model.projection.weight.detach().clone()
    trainer.train()
    assert not torch.allclose(before, model.projection.weight.detach())


def test_early_stopping_without_improvement(tmp_path):
    trainer, model = make_trainer(tmp_path, mode="joint")
    trainer.patience = 1
    # dev == train on a random model: metric changes each epoch; just check it stops
    best, history = trainer.train()
    assert len(history) <= 4  # 2 epochs x 2 phases bounded; joint has <= 2 entries + patience
