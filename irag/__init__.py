"""IRAG: list-aware passage reranking with a BERT-base cross-encoder.

Reference implementation of "IRAG: A BERT-Based List-Aware Architecture for
Efficient Passage Reranking via Multi-Stage Collective Passage Attention".

The architecture has three stages:
  Stage 1 (Scoring): a BERT cross-encoder scores every query-passage pair.
  Stage 2 (Selection, CPL): a bounded working set of the k highest-scoring
      candidates is retained, decoupling interaction cost from the
      candidate-list size n.
  Stage 3 (Interaction, CPA): collective passage attention lets each retained
      passage update its representation through attention over the working
      set, conditioned on a global working-set summary.
"""

__version__ = "1.0.0"

from .config import IRAGConfig, CPA_VARIANTS, SELECTION_POLICIES, TRAINING_MODES
from .modeling import IRAGModel, CollectivePassageAttention
from .selection import select_working_set
from .dataset import RerankDataset, load_jsonl
from .trainer import Trainer, set_seed
from . import metrics
from . import pipeline

__all__ = [
    "IRAGConfig",
    "IRAGModel",
    "CollectivePassageAttention",
    "select_working_set",
    "RerankDataset",
    "load_jsonl",
    "Trainer",
    "set_seed",
    "metrics",
    "pipeline",
    "CPA_VARIANTS",
    "SELECTION_POLICIES",
    "TRAINING_MODES",
    "__version__",
]
