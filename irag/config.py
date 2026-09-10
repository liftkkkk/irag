"""Configuration for IRAG models, training and inference.

All values default to the configuration reported in the paper
(Table "Training and measurement configuration"):

  encoder            bert-base-uncased, final-layer [CLS]
  projection dim d   200
  working-set size k 16
  optimizer          AdamW, peak LR 2e-5, linear decay, 10% warm-up
  weight decay       0.01
  dropout            0.1
  loss weight lambda 1.0
  precision          FP32
"""

from dataclasses import dataclass, field, asdict


CPA_VARIANTS = (
    "cpa_full",          # Sub-step A + Sub-step B (default; 601 interaction params)
    "cpa_pairwise_dot",  # drop the global term  -> plain dot-product self-attention (401)
    "cpa_global_only",   # Sub-step A only: Z_i = V_l for every i (601)
    "cpa_additive_mlp",  # GATv2-style additive scorer e_ij = a^T LeakyReLU(W_s O_i + W_t O_j) (80,601)
    "cpa_scaled",        # full CPA with 1/sqrt(d)-scaled attention logits (601)
    "cpa_temperature",  # full CPA with a learned temperature (602)
    "maxpool",           # replace CPA with an element-wise max over the working set (401)
    "lstm",              # replace CPA with a single-layer LSTM aggregator (322,001)
    "none",              # no interaction stage; final score = coarse score (0)
)

SELECTION_POLICIES = ("label_aware", "top_k", "annealed")

TRAINING_MODES = ("joint", "pipeline")


@dataclass
class IRAGConfig:
    """Architecture and training configuration for IRAG."""

    # ---- encoder -----------------------------------------------------
    model_name_or_path: str = "bert-base-uncased"
    # Which final-layer feature feeds the shared projection:
    #   "cls"    -> final-layer [CLS] hidden state (paper default)
    #   "pooler" -> tanh pooler output over [CLS] (legacy behaviour of the
    #               original 2019 implementation)
    encoder_feature: str = "cls"

    # ---- interaction stage --------------------------------------------
    d: int = 200                       # working dimension of O^(c), P^(c)
    cpa_variant: str = "cpa_full"      # see CPA_VARIANTS
    dropout: float = 0.1
    lambda_fine: float = 1.0           # lambda in L = L_c + lambda * L_f

    # ---- working set ---------------------------------------------------
    k: int = 16                        # working-set size (k = 0 or None -> no selection)

    # ---- training ------------------------------------------------------
    # When use_irag is False the model degenerates to a plain cross-encoder
    # baseline trained on all candidates with the coarse loss only.
    use_irag: bool = True
    train_selection: str = "label_aware"  # label_aware | top_k | annealed
    anneal_switch_fraction: float = 0.5   # fraction of total steps after which
                                          # "annealed" switches label_aware -> top_k
    training_mode: str = "joint"          # joint | pipeline (scorer first, then frozen)
    loss_reduction: str = "mean"          # mean | sum (over working-set members)

    # ---- tokenization ---------------------------------------------------
    max_query_length: int = 40
    max_passage_length: int = 200

    def __post_init__(self):
        if self.cpa_variant not in CPA_VARIANTS:
            raise ValueError(
                f"unknown cpa_variant {self.cpa_variant!r}; expected one of {CPA_VARIANTS}"
            )
        if self.train_selection not in SELECTION_POLICIES:
            raise ValueError(
                f"unknown train_selection {self.train_selection!r}; expected one of {SELECTION_POLICIES}"
            )
        if self.training_mode not in TRAINING_MODES:
            raise ValueError(
                f"unknown training_mode {self.training_mode!r}; expected one of {TRAINING_MODES}"
            )
        if self.encoder_feature not in ("cls", "pooler"):
            raise ValueError("encoder_feature must be 'cls' or 'pooler'")
        if self.loss_reduction not in ("mean", "sum"):
            raise ValueError("loss_reduction must be 'mean' or 'sum'")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}  # noqa: F841
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
