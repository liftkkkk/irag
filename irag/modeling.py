"""IRAG model: BERT cross-encoder + bounded working-set selection + CPA.

Implements the architecture of Section 3 of the paper.

Stage 1 (Scoring)
    h_i^(q) = BERT([CLS] Q [SEP] O_i [SEP])      final-layer [CLS], R^768
    h_i^(p) = BERT([CLS] O_i [SEP])              final-layer [CLS], R^768
    O_i^(c) = tanh(W_p h_i^(q) + b_p) in R^d          (shared projection)
    P_i^(c) = tanh(W_p h_i^(p) + b_p) in R^d
    p_i^(c) = sigmoid(W^c O_i^(c) + b^c)              (coarse score)

Stage 2 (Selection)
    top-k by coarse score at inference (label-aware during training);
    see irag/selection.py.

Stage 3 (Interaction, CPA)
    Sub-step A:  u_i = C_l^T P_i^(c)
                 alpha_i = softmax(u_i)
                 V_l = sum_i alpha_i P_i^(c)
    Sub-step B:  w_ij = O_i^(c)^T O_j^(c) + V_l^T O_j^(c)
                 beta_ij = softmax_j(w_ij)
                 Z_i = sum_j beta_ij O_j^(c)
    Final score: p_i^(f) = sigmoid(W^(f) [P_i^(c); Z_i] + b^(f))

Joint training (Section 3.4)
    L = L_c + lambda * L_f, both BCE over the working set only.
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IRAGConfig
from .selection import select_working_set

WEIGHTS_NAME = "irag_model.pt"
IRAG_CONFIG_NAME = "irag_config.json"


def _softmax_stable(u: torch.Tensor) -> torch.Tensor:
    """Softmax with max-subtraction, as written for Sub-step A."""
    return torch.softmax(u - u.max(), dim=0)


class CollectivePassageAttention(nn.Module):
    """Stage 3: collective passage attention over the working set.

    Variants (paper Tables 4 and 5):

    ==========================  ============================================
    variant                     description
    ==========================  ============================================
    ``cpa_full``                Sub-steps A + B (default)
    ``cpa_pairwise_dot``        drop the global term V_l (plain dot-product
                                self-attention over the working set)
    ``cpa_global_only``        Sub-step A only: Z_i = V_l for every i
    ``cpa_additive_mlp``        GATv2-style additive scorer
                                e_ij = a^T LeakyReLU(W_s O_i + W_t O_j)
    ``cpa_scaled``             full CPA, logits scaled by 1/sqrt(d)
    ``cpa_temperature``         full CPA, learned temperature tau
    ``maxpool``                 element-wise max over the working set
    ``lstm``                    single-layer LSTM aggregator
    ==========================  ============================================

    Interaction parameter counts (excluding the shared BERT encoder, the
    shared projection W_p and the coarse head W^c): full 601,
    pairwise_dot 401, global_only 601, additive_mlp 80,601, scaled 601,
    temperature 602, maxpool 401, lstm 322,001.
    """

    LEAKY_SLOPE = 0.2

    def __init__(self, d: int, variant: str = "cpa_full"):
        super().__init__()
        self.d = d
        self.variant = variant

        if variant in ("cpa_full", "cpa_global_only", "cpa_scaled", "cpa_temperature"):
            self.context_vector = nn.Parameter(torch.empty(d))  # C_l

        if variant == "cpa_temperature":
            self.temperature = nn.Parameter(torch.ones(1))

        if variant == "cpa_additive_mlp":
            self.source_proj = nn.Linear(d, d, bias=False)  # W_s
            self.target_proj = nn.Linear(d, d, bias=False)  # W_t
            self.additive_vector = nn.Parameter(torch.empty(d))  # a

        if variant == "lstm":
            self.lstm = nn.LSTM(d, d, batch_first=True)

        self.fine_head = nn.Linear(2 * d, 1)  # W^(f), b^(f)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if name.endswith("context_vector") or name == "additive_vector":
                nn.init.normal_(param, std=0.02)
            elif name.startswith(("fine_head", "source_proj", "target_proj")):
                if param.dim() > 1:
                    nn.init.normal_(param, std=0.02)
                else:
                    nn.init.zeros_(param)
        # LSTM keeps its default uniform initialisation.

    def working_set_summary(self, P: torch.Tensor) -> torch.Tensor:
        """Sub-step A: V_l = sum_i alpha_i P_i^(c) (passage-only inputs)."""
        u = P @ self.context_vector
        alpha = _softmax_stable(u)
        return alpha @ P

    def forward(self, O: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Compute final scores for the working set.

        Args:
            O: (k, d) query-conditioned representations of the retained
                passages (rows follow the working-set order).
            P: (k, d) passage-only representations, same order.

        Returns:
            (k,) final score logits p_i^(f) (pre-sigmoid).
        """
        k = P.size(0)
        variant = self.variant

        if variant == "lstm":
            _, (h_n, _) = self.lstm(P.unsqueeze(0))
            Z = h_n.squeeze(0).expand(k, -1)
        elif variant == "maxpool":
            Z = P.max(dim=0).values.unsqueeze(0).expand(k, -1)
        elif variant == "cpa_global_only":
            V_l = self.working_set_summary(P)
            Z = V_l.unsqueeze(0).expand(k, -1)
        else:
            # Sub-step B: pairwise logits w_ij = O_i . O_j (+ V_l . O_j).
            if variant == "cpa_pairwise_dot":
                logits = O @ O.t()
            elif variant == "cpa_additive_mlp":
                e = F.leaky_relu(
                    self.source_proj(O).unsqueeze(1) + self.target_proj(O).unsqueeze(0),
                    self.LEAKY_SLOPE,
                )
                logits = e @ self.additive_vector
            else:
                # Sub-step A: global summary from passage-only representations.
                V_l = self.working_set_summary(P)
                logits = O @ O.t() + (V_l @ O.t()).unsqueeze(0)
                if variant == "cpa_scaled":
                    logits = logits / math.sqrt(self.d)
                elif variant == "cpa_temperature":
                    logits = logits / self.temperature
            beta = torch.softmax(logits, dim=1)
            Z = beta @ O

        return self.fine_head(torch.cat([P, Z], dim=-1)).squeeze(-1)


class IRAGModel(nn.Module):
    """The full three-stage IRAG reranker.

    Set ``config.use_irag = False`` for the plain cross-encoder baseline
    (BERT base trained on all candidates with the coarse loss only), which
    is the controlled comparison of the paper.
    """

    def __init__(
        self,
        config: IRAGConfig,
        bert_model: torch.nn.Module | None = None,
        encode_batch_size: int = 32,
    ):
        super().__init__()
        self.config = config
        if bert_model is not None:
            self.bert = bert_model
        else:
            from transformers import AutoModel

            self.bert = AutoModel.from_pretrained(config.model_name_or_path)

        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(hidden, config.d)  # shared W_p, b_p
        self.coarse_head = nn.Linear(config.d, 1)  # W^c, b^c
        self.encode_batch_size = encode_batch_size

        self.interaction = None
        if config.use_irag and config.cpa_variant != "none":
            self.interaction = CollectivePassageAttention(config.d, config.cpa_variant)

        self._init_new_parameters()

    # ------------------------------------------------------------------
    def _init_new_parameters(self):
        """Initialise the new (non-pretrained) heads, BERT-style std=0.02."""

        def init_linear(linear):
            nn.init.normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)

        init_linear(self.projection)
        init_linear(self.coarse_head)
        if self.interaction is not None:
            self.interaction.reset_parameters()

    # ------------------------------------------------------------------
    def enable_gradient_checkpointing(self):
        self.bert.gradient_checkpointing_enable()

    def _encode(self, input_ids, attention_mask, token_type_ids, batch_size):
        """Chunked BERT forward; returns (n, hidden) encoder features."""
        outputs = []
        n = input_ids.size(0)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            out = self.bert(
                input_ids[start:end],
                attention_mask=attention_mask[start:end],
                token_type_ids=token_type_ids[start:end],
            )
            if self.config.encoder_feature == "pooler":
                h = out.pooler_output
            else:  # final-layer [CLS]
                h = out.last_hidden_state[:, 0]
            outputs.append(h)
        return torch.cat(outputs, dim=0)

    def _project(self, h):
        """tanh(W_p h + b_p), shared by query-conditioned and passage-only."""
        return torch.tanh(self.projection(self.dropout(h)))

    # ------------------------------------------------------------------
    def forward(
        self,
        batch,
        k: int | None = None,
        selection: str = "label_aware",
        encode_batch_size: int | None = None,
    ):
        """Run the three stages on one query's candidate list.

        Args:
            batch: dict of tensors for a single query:
                ``pair_input_ids/pair_attention_mask/pair_token_type_ids``
                (n, L) for [CLS] Q [SEP] O_j [SEP];
                ``psg_input_ids/psg_attention_mask/psg_token_type_ids``
                (n, L) for the passage-only encoding [CLS] O_j [SEP]
                (only the selected rows are ever encoded);
                ``labels`` optional (n,) binary labels.
            k: working-set size (None / <= 0 means no selection).
            selection: ``label_aware`` or ``top_k``; ``label_aware`` falls
                back to ``top_k`` automatically when labels are absent.
            encode_batch_size: candidates per BERT forward chunk.

        Returns a dict with:
            coarse_logits (n,) and, when IRAG stages are enabled,
            working_set (k',) indices and fine_logits (k',) for the
            working set; plus ``loss`` / ``loss_c`` / ``loss_f`` when
            labels are provided.
        """
        cfg = self.config
        labels = batch.get("labels")
        bs = encode_batch_size or self.encode_batch_size

        # ---- Stage 1: scoring ----
        h_q = self._encode(
            batch["pair_input_ids"],
            batch["pair_attention_mask"],
            batch["pair_token_type_ids"],
            bs,
        )
        O = self._project(h_q)  # (n, d) query-conditioned
        coarse_logits = self.coarse_head(O).squeeze(-1)  # (n,)

        out = {"coarse_logits": coarse_logits}

        if labels is not None:
            labels = labels.to(coarse_logits.device)

        # ---- Plain cross-encoder baseline: no stages 2/3 ----
        if not cfg.use_irag:
            if labels is not None:
                out["loss"] = F.binary_cross_entropy_with_logits(
                    coarse_logits, labels.float(), reduction=cfg.loss_reduction
                )
            return out

        # ---- Stage 2: selection (discrete; no gradient through it) ----
        sel = select_working_set(coarse_logits.detach(), labels, k, selection)
        out["working_set"] = sel

        # ---- Stage 3: interaction over the working set ----
        if self.interaction is None:
            # "w/o CPA entirely": two-stage structure, coarse score only.
            out["fine_logits"] = coarse_logits[sel]
        else:
            h_p = self._encode(
                batch["psg_input_ids"][sel],
                batch["psg_attention_mask"][sel],
                batch["psg_token_type_ids"][sel],
                bs,
            )
            P = self._project(h_p)  # (k', d) passage-only
            out["fine_logits"] = self.interaction(O[sel], P)

        if labels is not None:
            y = labels[sel].float()
            loss_c = F.binary_cross_entropy_with_logits(
                coarse_logits[sel], y, reduction=cfg.loss_reduction
            )
            out["loss_c"] = loss_c
            if self.interaction is None:
                # "w/o CPA entirely": keep the two-stage structure but train
                # with the coarse loss on the working set only.
                out["loss"] = loss_c
            else:
                loss_f = F.binary_cross_entropy_with_logits(
                    out["fine_logits"], y, reduction=cfg.loss_reduction
                )
                out["loss_f"] = loss_f
                out["loss"] = loss_c + cfg.lambda_fine * loss_f
        return out

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, save_directory: str, tokenizer=None):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, WEIGHTS_NAME))
        with open(os.path.join(save_directory, IRAG_CONFIG_NAME), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        self.bert.config.save_pretrained(save_directory)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_checkpoint(cls, save_directory: str, map_location="cpu") -> "IRAGModel":
        from transformers import AutoConfig, AutoModel

        with open(os.path.join(save_directory, IRAG_CONFIG_NAME)) as f:
            cfg = IRAGConfig.from_dict(json.load(f))
        bert_config = AutoConfig.from_pretrained(save_directory)
        bert = AutoModel.from_config(bert_config)  # architecture only
        model = cls(cfg, bert_model=bert)
        state = torch.load(
            os.path.join(save_directory, WEIGHTS_NAME), map_location=map_location
        )
        model.load_state_dict(state)
        return model
