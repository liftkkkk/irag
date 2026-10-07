# IRAG

**IRAG: A BERT-Based List-Aware Architecture for Efficient Passage Reranking via Multi-Stage Collective Passage Attention**

IRAG reranks a retrieved candidate list in three stages: a BERT cross-encoder scores each
query–passage pair (**Scoring**), a bounded working set of `k` candidates is kept
(**Selection**), and a lightweight *Collective Passage Attention* (CPA) module lets the
retained passages interact before the final scores are produced (**Interaction**). The
interaction stage adds only **601 parameters** on top of BERT-base and is trained jointly
with the scorer through a two-term loss `L = L_c + λ·L_f`.

```
 candidates (n)
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1 · Scoring                                            │
│   h_i^(q) = BERT([CLS] Q [SEP] O_i [SEP])  →  [CLS]          │
│   h_i^(p) = BERT([CLS] O_i [SEP])           →  [CLS]         │
│   O_i^(c) = tanh(W_p h_i^(q) + b_p) ∈ R^d     (shared W_p)   │
│   P_i^(c) = tanh(W_p h_i^(p) + b_p) ∈ R^d                    │
│   p_i^(c) = σ(W^c O_i^(c) + b^c)              (coarse score)  │
├──────────────────────────────────────────────────────────────┤
│ Stage 2 · Selection (CPL)                                    │
│   inference: top-k by coarse score (label-free, Alg. 1)      │
│   training: all positives + hardest negatives  (Alg. 2)      │
├──────────────────────────────────────────────────────────────┤
│ Stage 3 · Interaction (CPA)                                  │
│   A: V_l  = Σ_i α_i P_i^(c),  α = softmax(C_l^T P)           │
│   B: w_ij = O_i^(c)·O_j^(c) + V_l·O_j^(c)                    │
│      Z_i  = Σ_j softmax_j(w_ij) O_j^(c)                      │
│   p_i^(f) = σ(W^(f) [P_i^(c); Z_i] + b^(f))   (final score)  │
└──────────────────────────────────────────────────────────────┘
```

At inference the final ranking is: working-set members ordered by their CPA score,
followed by the remaining candidates in coarse-score order.

## Results (paper configuration)

WIKIQA test set (controlled comparison, identical data / candidate sets / scripts):

| Model | Params (interaction) | MAP | MRR |
|---|---|---|---|
| BERT base (200 tok) | 0 | 0.7831 | 0.7923 |
| BERT base + MaxPooling | 401 | 0.8119 | 0.8215 |
| **IRAG (200 tok)** | **601** | **0.8448** | **0.8605** |

MS MARCO passage reranking dev, official BM25 top-1,000 candidates:

| Model | MRR@10 (dev) | MRR@10 (test) |
|---|---|---|
| BERT base (400 tok) | 0.351 | 0.363 |
| IRAG (200 tok) | 0.347 | 0.359 |
| **IRAG (400 tok)** | **0.364** | **0.377** |

### External evidence: the official MS MARCO leaderboard

The IRAG (400 tok) numbers correspond to the anonymous entry **"BERT base + attention
ranking"** (reranking, submitted 2019/08/26) on the official
[MS MARCO Passage Ranking leaderboard](https://microsoft.github.io/MSMARCO-Passage-Ranking-Submissions/leaderboard/):
**MRR@10 = 0.364 (eval) / 0.377 (dev)**, where *eval* is the hidden leaderboard test
set scored by the organizers (6,530 queries) and *dev* is the public dev set (6,980
queries). Both scores come from the official organizer evaluation — MRR@10 =
0.3640139 (eval) and 0.3772412 (dev). The evaluation email from the MS MARCO team
(Daniel Campos, Microsoft, Aug 27, 2019) is archived
[here](docs/msmarco_eval_email.png), and a snapshot of the leaderboard entry is
[here](docs/msmarco_leaderboard.png).

Among reranking submissions dated on or before that day, only "Enriched BERT base +
AOA index" (0.368 eval, 2019/05/06) scores higher on eval; the dev MRR@10 of 0.377 is
the highest of any reranking entry submitted up to that date. BERT-base reranking
systems submitted **7 months to ~2.7 years later** still score below it on both metrics
(values from the official leaderboard CSV, retrieved 2026-10-07):

| Leaderboard entry | Team | Submitted | MRR@10 (eval) | MRR@10 (dev) |
|---|---|---|---|---|
| **BERT base + attention ranking (IRAG)** | anonymous | 2019/08/26 | **0.364** | **0.377** |
| CSEN-base (256) | T03lab, South China Normal University | 2022/04/14 | 0.360 | 0.373 |
| LTR-BERT (pointwise / CrossEntropy loss) | Weijie Liu, OpenPLM | 2022/03/30 | 0.356 | 0.366 |
| Anonymous | anonymous | 2022/02/06 | 0.356 | 0.365 |
| Bert-base trained with self-supervised manner in 128-tokens limitation | South China Normal University | 2021/10/20 | 0.355 | 0.362 |
| BERT-base + SMAP | Dengwen Lin, NUDT | 2021/07/31 | 0.355 | 0.367 |
| PCBERT-base (256 tokens) | SCNU-FOUR | 2021/03/15 | 0.355 | 0.361 |
| PCBERT-base (128 tokens) | SCNU-FOUR | 2021/03/03 | 0.349 | 0.357 |
| BERT-Base | IR group, Tsinghua University | 2020/04/08 | 0.349 | 0.358 |

The leaderboard data itself is maintained in the
[microsoft/MSMARCO-Passage-Ranking-Submissions](https://github.com/microsoft/MSMARCO-Passage-Ranking-Submissions)
repository (`leaderboard/leaderboard.csv`).

Ablations (WIKIQA test): see `--cpa_variant`, `--k`, `--train_selection` and
`--training_mode` below; every row of the paper's ablation tables maps to one flag.

## Installation

```bash
git clone https://github.com/liftkkkk/irag.git
cd irag
pip install -r requirements.txt        # or: pip install -e ".[test]"
python -m pytest tests/                # optional: verify the installation
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, transformers ≥ 4.30. The first training run
downloads `bert-base-uncased` from the HuggingFace hub.

## Data format

All scripts share one JSONL format — one query per line:

```json
{"qid": "Q1", "query": "how far is the moon",
 "candidates": [{"id": "D1", "text": "the moon is 384400 km away", "label": 1},
                {"id": "D2", "text": "cats are cute", "label": 0}]}
```

### WIKIQA

```bash
python scripts/prepare_wikiqa.py --out_dir data/wikiqa
```

Downloads the official `WikiQACorpus.zip`, parses the train/dev/test splits
(3,047 questions, 29,258 candidate sentences, 1,473 positives) and writes
`train.jsonl / dev.jsonl / test.jsonl`.

### MS MARCO 2.0

```bash
python scripts/prepare_marco.py --marco_dir data/marco/raw --out_dir data/marco
```

Requires the official files (see the script docstring for URLs): `collection.tsv`,
`triples.train.small.tar.gz` (398,792 queries) and the BM25 `top1000.dev` candidate
lists plus `qrels.dev.small.tsv` (6,980 dev queries). Training lists are built from the
official triples; the dev set reranks the official BM25 top-1,000 per query.

## Training

```bash
# IRAG on WIKIQA (paper default: joint training, label-aware selection, k=16)
python scripts/train.py \
    --train_file data/wikiqa/train.jsonl \
    --dev_file   data/wikiqa/dev.jsonl \
    --output_dir runs/wikiqa_irag \
    --eval_metric map \
    --num_epochs 3

# IRAG on MS MARCO (400-token passages, 2 epochs)
python scripts/train.py \
    --train_file data/marco/train.jsonl \
    --dev_file   data/marco/dev.jsonl \
    --output_dir runs/marco_irag \
    --eval_metric mrr@10 \
    --max_passage_length 400 \
    --num_epochs 2
```

Paper hyperparameters are the defaults (AdamW, peak LR 2e-5, linear decay with 10 %
warm-up, weight decay 0.01, dropout 0.1, λ = 1.0, FP32, one query per step with
32-passage encoding chunks, dev evaluation per epoch with early stopping patience 2,
seed 42). `runs/<name>/best` and `runs/<name>/last` receive checkpoints;
`runs/<name>/history.json` records the dev metric per epoch.

### Baselines and ablations

| Paper row | Flag |
|---|---|
| w/o Stage 3 (plain cross-encoder) | `--baseline` |
| w/o CPA entirely (two-stage) | `--cpa_variant none` |
| w/o global summary V_l | `--cpa_variant cpa_pairwise_dot` |
| w/o pairwise attention | `--cpa_variant cpa_global_only` |
| CPA → MaxPooling | `--cpa_variant maxpool` |
| CPA → LSTM | `--cpa_variant lstm` |
| 1/√d-scaled logits | `--cpa_variant cpa_scaled` |
| learned temperature | `--cpa_variant cpa_temperature` |
| w/o joint training (pipeline) | `--training_mode pipeline` |
| score-only top-k training | `--train_selection top_k` |
| annealed selection | `--train_selection annealed` |
| working-set size | `--k {1,2,4,8,12,16,20,24,32,48,64}` |

## Inference

```bash
python scripts/rerank.py \
    --checkpoint runs/wikiqa_irag/best \
    --test_file data/wikiqa/test.jsonl \
    --output runs/wikiqa_irag/test.tsv \
    --format wikiqa            # use --format marco for TREC-style output
```

Selection at inference is label-free (Algorithm 1). `--k` overrides the working-set
size without retraining; `--use_coarse_only` ranks by the coarse scorer alone
(useful for pipeline-mode phase-1 evaluation). The command prints the fraction of
answerable queries whose working set contains a gold passage (recall of Stage 2).

## Evaluation

```bash
# WIKIQA: MAP / MRR (questions without a relevant candidate are skipped)
python scripts/evaluate.py \
    --run runs/wikiqa_irag/test.tsv \
    --gold data/wikiqa/test.jsonl \
    --metrics map,mrr

# MS MARCO: MRR@10 with the official semantics + paired t-test vs a baseline
python scripts/evaluate.py \
    --run runs/marco_irag/dev.tsv \
    --qrels data/marco/qrels.dev.small.tsv \
    --metrics mrr@10 \
    --baseline_run runs/marco_baseline/dev.tsv \
    --ttest_metric mrr@10
```

Supported metrics: `map`, `mrr`, `mrr@k`, `recall@k`. `--baseline_run` adds a two-sided
paired t-test over per-question AP (WIKIQA) or per-query RR@k (MS MARCO) on the shared
queries.

## Library use

```python
from transformers import AutoTokenizer
from irag import IRAGConfig, IRAGModel, RerankDataset, Trainer, pipeline, metrics

cfg = IRAGConfig()                       # paper defaults: d=200, k=16, cpa_full
model = IRAGModel(cfg)                   # bert-base-uncased
tok = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=False)
train = RerankDataset("data/wikiqa/train.jsonl", tok)
dev = RerankDataset("data/wikiqa/dev.jsonl", tok)

Trainer(model, tok, train, dev, output_dir="runs/lib").train()

rankings, stats = pipeline.rerank_dataset(model, dev, device="cuda")
print(metrics.mean_average_precision(...))          # see irag/metrics.py
```

## Project layout

```
irag/
  config.py      IRAGConfig (all paper hyperparameters, validated)
  modeling.py    IRAGModel + CollectivePassageAttention (all CPA variants)
  selection.py   Algorithm 1 (top-k) and Algorithm 2 (label-aware) selection
  dataset.py     JSONL loading, [CLS] Q [SEP] O [SEP] tokenization, padding
  trainer.py     joint & pipeline training, early stopping, checkpointing
  pipeline.py    label-free inference (Algorithm 1), run-file writers
  metrics.py     MAP / MRR / MRR@10 / recall@k, paired t-test, run parsing
scripts/
  prepare_wikiqa.py   official WikiQACorpus.zip -> JSONL
  prepare_marco.py    MS MARCO collection/triples/top1000 -> JSONL + qrels
  train.py            train IRAG / baseline / ablations
  rerank.py           write run files from a checkpoint
  evaluate.py         score run files (+ significance tests)
tests/            86 unit & integration tests (tiny random BERT, CPU, seconds)
```

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers the selection algorithms (tie-breaking, label-aware edge cases),
every CPA variant (shapes and the paper's parameter counts: 601 / 401 / 80,601 /
322,001 ...), the inference ordering rule, all metrics with hand-computed values,
run-file parsing, checkpoint round-trips and end-to-end joint/pipeline training on a
tiny random BERT.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{irag2026,
  title   = {IRAG: A BERT-Based List-Aware Architecture for Efficient Passage
             Reranking via Multi-Stage Collective Passage Attention},
  year    = {2026}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE)).
