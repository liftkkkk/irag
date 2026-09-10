"""Tests for the inference pipeline (Algorithm 1) with a fully controlled model."""

import types

import torch

from irag.pipeline import rankings_to_rows, rerank_dataset, write_marco_run, write_wikiqa_run
from irag.selection import select_working_set


class FakeModel:
    """A model whose scores are fully controlled by the test."""

    def __init__(self, coarse_by_qid, fine_by_qid=None, use_interaction=True):
        self.config = types.SimpleNamespace(use_irag=use_interaction)
        self.interaction = object() if use_interaction else None
        self._coarse = coarse_by_qid
        self._fine = fine_by_qid or {}

    def eval(self):
        return self

    def __call__(self, batch, k=None, selection="top_k", encode_batch_size=None):
        qid = batch["qid"]
        coarse = torch.tensor(self._coarse[qid])
        out = {"coarse_logits": coarse}
        if not self.config.use_irag:
            return out
        sel = select_working_set(coarse, None, k, selection)
        out["working_set"] = sel
        if self.interaction is not None:
            out["fine_logits"] = torch.tensor(self._fine[qid])
        else:
            out["fine_logits"] = coarse[sel]
        return out


class FakeDataset:
    """Yields batches in the RerankDataset format."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        n = len(rec["candidates"])
        return {
            "qid": rec["qid"],
            "cand_ids": [c["id"] for c in rec["candidates"]],
            "pair_input_ids": torch.zeros(n, 8, dtype=torch.long),
            "pair_attention_mask": torch.ones(n, 8, dtype=torch.long),
            "pair_token_type_ids": torch.zeros(n, 8, dtype=torch.long),
            "psg_input_ids": torch.zeros(n, 8, dtype=torch.long),
            "psg_attention_mask": torch.ones(n, 8, dtype=torch.long),
            "psg_token_type_ids": torch.zeros(n, 8, dtype=torch.long),
            "labels": torch.tensor([c["label"] for c in rec["candidates"]]),
        }


RECORD = {
    "qid": "Q1",
    "candidates": [
        {"id": "D0", "label": 0},
        {"id": "D1", "label": 1},
        {"id": "D2", "label": 0},
        {"id": "D3", "label": 0},
        {"id": "D4", "label": 1},
    ],
}


def test_working_set_ranks_first_then_coarse_order():
    # coarse scores: D4 .7 < D1 .9 < D2 .5 < D3 .3 < D0 .1
    coarse = [0.1, 0.9, 0.5, 0.3, 0.7]
    # working set (top-2 by coarse) = {D1, D4}; fine scores in working-set order [D1, D4]
    fine = [0.2, 0.8]
    model = FakeModel({"Q1": coarse}, {"Q1": fine})
    rankings, _ = rerank_dataset(model, FakeDataset([RECORD]), "cpu", k=2, progress=False)
    # D4 (fine .8) first, D1 (fine .2) second, then D2, D3, D0 by coarse
    assert rankings["Q1"] == ["D4", "D1", "D2", "D3", "D0"]


def test_working_set_members_sorted_by_fine_descending():
    coarse = [0.1, 0.9, 0.5, 0.3, 0.7]
    fine = [0.9, 0.1]  # working-set order [D1, D4] -> D1 first
    model = FakeModel({"Q1": coarse}, {"Q1": fine})
    rankings, _ = rerank_dataset(model, FakeDataset([RECORD]), "cpu", k=2, progress=False)
    assert rankings["Q1"] == ["D1", "D4", "D2", "D3", "D0"]


def test_use_coarse_only_ranks_all_by_coarse():
    coarse = [0.1, 0.9, 0.5, 0.3, 0.7]
    model = FakeModel({"Q1": coarse}, {"Q1": [99.0, 99.0]})
    rankings, _ = rerank_dataset(
        model, FakeDataset([RECORD]), "cpu", k=2, use_coarse_only=True, progress=False
    )
    assert rankings["Q1"] == ["D1", "D4", "D2", "D3", "D0"]


def test_baseline_model_ranks_by_coarse():
    coarse = [0.1, 0.9, 0.5, 0.3, 0.7]
    model = FakeModel({"Q1": coarse}, use_interaction=False)
    rankings, stats = rerank_dataset(
        model, FakeDataset([RECORD]), "cpu", k=2, progress=False
    )
    assert rankings["Q1"] == ["D1", "D4", "D2", "D3", "D0"]
    assert stats["n_queries"] == 1


def test_no_selection_k_zero():
    coarse = [0.1, 0.9, 0.5, 0.3, 0.7]
    fine = [0.05, 0.1, 0.15, 0.2, 0.25]
    model = FakeModel({"Q1": coarse}, {"Q1": fine})
    rankings, _ = rerank_dataset(model, FakeDataset([RECORD]), "cpu", k=0, progress=False)
    assert rankings["Q1"] == ["D4", "D3", "D2", "D1", "D0"]


def test_stats_report_gold_in_working_set():
    coarse = [0.1, 0.9, 0.5, 0.3, 0.7]  # top-2: D1 (label 1), D4 (label 1)
    model = FakeModel({"Q1": coarse}, {"Q1": [0.5, 0.5]})
    _, stats = rerank_dataset(model, FakeDataset([RECORD]), "cpu", k=2, progress=False)
    assert stats["n_queries"] == 1
    assert stats["queries_with_positive"] == 1
    assert stats["ws_has_positive"] == 1


def test_stats_gold_missing_from_working_set():
    # positives (D1, D4) both score low: outside the top-2 working set
    coarse = [0.5, 0.05, 0.45, 0.4, 0.1]
    model = FakeModel({"Q1": coarse}, {"Q1": [0.5, 0.5]})
    _, stats = rerank_dataset(model, FakeDataset([RECORD]), "cpu", k=2, progress=False)
    assert stats["queries_with_positive"] == 1
    assert stats["ws_has_positive"] == 0


def test_ties_in_coarse_broken_by_candidate_index():
    coarse = [0.5, 0.5, 0.5]
    fine = [0.1, 0.3]  # working-set order after tie-break: [A, B]
    record = {
        "qid": "Q2",
        "candidates": [
            {"id": "A", "label": 0},
            {"id": "B", "label": 0},
            {"id": "C", "label": 0},
        ],
    }
    model = FakeModel({"Q2": coarse}, {"Q2": fine})
    rankings, _ = rerank_dataset(model, FakeDataset([record]), "cpu", k=2, progress=False)
    # working set = first two by tie-break (A, B); A fine .1, B fine .3 -> B, A, C
    assert rankings["Q2"] == ["B", "A", "C"]


def test_rankings_to_rows_scores_are_strictly_decreasing():
    rankings = {"Q1": ["D1", "D2", "D3"]}
    rows = rankings_to_rows(rankings)
    assert rows == [("Q1", "D1", 1, 2.0), ("Q1", "D2", 2, 1.0), ("Q1", "D3", 3, 0.0)]


def test_write_wikiqa_run(tmp_path):
    rankings = {"Q1": ["D1", "D2"]}
    path = tmp_path / "run.tsv"
    write_wikiqa_run(rankings, str(path))
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert lines == ["Q1\tD1\t1.000000", "Q1\tD2\t0.000000"]


def test_write_marco_run(tmp_path):
    rankings = {"Q1": ["D1", "D2"]}
    path = tmp_path / "run.txt"
    write_marco_run(rankings, str(path), run_id="irag")
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert lines == ["Q1\tQ0\tD1\t1\t1.000000\tirag", "Q1\tQ0\tD2\t2\t0.000000\tirag"]
