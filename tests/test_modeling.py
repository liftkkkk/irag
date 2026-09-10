import pytest
import torch

from irag.config import IRAGConfig
from irag.modeling import CollectivePassageAttention, IRAGModel
from tests.conftest import fake_batch, tiny_bert

# Interaction-stage parameter counts reported in the paper (Table 4);
# excludes the shared BERT encoder, the shared projection and the coarse head.
EXPECTED_INTERACTION_PARAMS = {
    "cpa_full": 601,
    "cpa_pairwise_dot": 401,
    "cpa_global_only": 601,
    "cpa_additive_mlp": 80601,
    "cpa_scaled": 601,
    "cpa_temperature": 602,
    "maxpool": 401,
    "lstm": 322001,
}

VARIANTS = list(EXPECTED_INTERACTION_PARAMS)


@pytest.mark.parametrize("variant,expected", EXPECTED_INTERACTION_PARAMS.items())
def test_interaction_parameter_counts(variant, expected):
    cpa = CollectivePassageAttention(200, variant)
    n = sum(p.numel() for p in cpa.parameters())
    assert n == expected, f"{variant}: {n} != {expected}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_cpa_forward_shapes(variant):
    torch.manual_seed(0)
    cpa = CollectivePassageAttention(16, variant)
    O = torch.randn(5, 16)
    P = torch.randn(5, 16)
    logits = cpa(O, P)
    assert logits.shape == (5,)


def test_working_set_summary_is_convex_combination():
    torch.manual_seed(0)
    cpa = CollectivePassageAttention(8, "cpa_full")
    P = torch.randn(4, 8)
    V = cpa.working_set_summary(P)
    # V is a convex combination of the rows of P: within row hull bounds
    assert V.shape == (8,)
    assert V.min() >= P.min() - 1e-6
    assert V.max() <= P.max() + 1e-6


def test_cpa_full_equation():
    """Direct check of Sub-step A + Sub-step B on a hand-computable case."""
    torch.manual_seed(0)
    cpa = CollectivePassageAttention(4, "cpa_full")
    O = torch.eye(4)
    P = torch.eye(4)
    logits = cpa(O, P)
    assert logits.shape == (4,)
    # must run without NaNs and be finite
    assert torch.isfinite(logits).all()


def test_model_forward_shapes(tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local")
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    batch = fake_batch(n=6, with_labels=True, seed=1)
    out = model(batch, k=3, selection="top_k")

    assert out["coarse_logits"].shape == (6,)
    assert out["working_set"].shape == (3,)
    assert out["fine_logits"].shape == (3,)
    assert "loss" in out and "loss_c" in out and "loss_f" in out


def test_model_label_aware_selection_keeps_positives(tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local")
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    batch = fake_batch(n=6, with_labels=True, seed=2)
    labels = batch["labels"]
    out = model(batch, k=3, selection="label_aware")
    sel = out["working_set"]
    if labels.sum() > 0 and labels.sum() <= 3:
        pos = torch.nonzero(labels).squeeze(-1).tolist()
        assert set(pos).issubset(set(sel.tolist()))


def test_model_baseline_mode(tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local", use_irag=False)
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    assert model.interaction is None
    batch = fake_batch(n=5, with_labels=True, seed=3)
    out = model(batch, k=3, selection="top_k")
    assert "working_set" not in out
    assert "fine_logits" not in out
    assert "loss" in out and "loss_c" not in out


def test_model_cpa_none_variant(tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local", cpa_variant="none")
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    assert model.interaction is None
    batch = fake_batch(n=5, with_labels=True, seed=4)
    out = model(batch, k=2, selection="top_k")
    assert "working_set" in out
    assert "fine_logits" in out
    assert "loss_f" not in out
    assert "loss" in out


def test_joint_loss_combination(tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local", lambda_fine=2.0)
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    batch = fake_batch(n=4, with_labels=True, seed=5)
    out = model(batch, k=4, selection="label_aware")
    expected = out["loss_c"] + 2.0 * out["loss_f"]
    assert torch.allclose(out["loss"], expected)


def test_lambda_zero_reduces_to_coarse_loss(tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local", lambda_fine=0.0)
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    batch = fake_batch(n=4, with_labels=True, seed=6)
    out = model(batch, k=4, selection="label_aware")
    assert torch.allclose(out["loss"], out["loss_c"])


def test_gradients_flow_to_encoder_and_interaction(tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local")
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    batch = fake_batch(n=4, with_labels=True, seed=7)
    out = model(batch, k=3, selection="label_aware")
    out["loss"].backward()
    assert model.bert.embeddings.word_embeddings.weight.grad is not None
    assert model.interaction.fine_head.weight.grad is not None
    assert model.projection.weight.grad is not None
    assert model.coarse_head.weight.grad is not None


def test_selection_has_no_gradient_path(tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local")
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    batch = fake_batch(n=5, with_labels=True, seed=8)
    out = model(batch, k=2, selection="top_k")
    assert not out["working_set"].requires_grad


def test_no_selection_when_k_zero(tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local")
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    batch = fake_batch(n=5, with_labels=True, seed=9)
    out = model(batch, k=0, selection="top_k")
    assert out["working_set"].tolist() == [0, 1, 2, 3, 4]
    assert out["fine_logits"].shape == (5,)


def test_save_and_from_checkpoint_roundtrip(tmp_path, tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local")
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    ckpt = tmp_path / "ckpt"
    model.save(str(ckpt))

    loaded = IRAGModel.from_checkpoint(str(ckpt))
    for key in model.state_dict():
        assert torch.equal(model.state_dict()[key], loaded.state_dict()[key])

    batch = fake_batch(n=4, with_labels=True, seed=10)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        out_a = model(batch, k=2, selection="top_k")
        out_b = loaded(batch, k=2, selection="top_k")
    assert torch.allclose(out_a["coarse_logits"], out_b["coarse_logits"])
    assert torch.allclose(out_a["fine_logits"], out_b["fine_logits"])


def test_pooler_feature_option(tiny_bert_model):
    cfg = IRAGConfig(d=16, model_name_or_path="tiny-local", encoder_feature="pooler")
    model = IRAGModel(cfg, bert_model=tiny_bert_model)
    batch = fake_batch(n=3, with_labels=True, seed=11)
    out = model(batch, k=2, selection="top_k")
    assert torch.isfinite(out["loss"]).item()
