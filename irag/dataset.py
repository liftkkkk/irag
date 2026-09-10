"""Datasets and tokenization for IRAG.

Both benchmarks are normalised to one JSONL record per query::

    {"qid": "Q1", "query": "how far is ...",
     "candidates": [{"id": "D1", "text": "...", "label": 1}, ...]}

WIKIQA:  id = SentenceID,  text = answer sentence.
MS MARCO: id = passage PID,  text = passage.  Training lists are built
from the official triples (positives + negatives); development lists are
the official BM25 top-1000 candidates with labels from qrels.
"""

import json

import torch
from torch.utils.data import Dataset


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class QueryRecord:
    __slots__ = ("qid", "query", "ids", "texts", "labels")

    def __init__(self, qid, query, candidates):
        self.qid = qid
        self.query = query
        self.ids = [c["id"] for c in candidates]
        self.texts = [c["text"] for c in candidates]
        self.labels = [int(c.get("label", 0)) for c in candidates]

    @property
    def n(self):
        return len(self.ids)


class RerankDataset(Dataset):
    """One item = one query with its full candidate list."""

    def __init__(self, path, tokenizer, max_query_length=40, max_passage_length=200):
        self.records = [QueryRecord(r["qid"], r["query"], r["candidates"]) for r in load_jsonl(path)]
        if not self.records:
            raise ValueError(f"no records parsed from {path}")
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length

    def __len__(self):
        return len(self.records)

    def _encode_ids(self, text, max_len):
        tokens = self.tokenizer.tokenize(text)[:max_len]
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def __getitem__(self, idx):
        rec = self.records[idx]
        query_ids = self._encode_ids(rec.query, self.max_query_length)
        pair_ids, pair_mask, pair_seg = [], [], []
        psg_ids_rows, psg_mask_rows, psg_seg_rows = [], [], []

        for text in rec.texts:
            psg_ids = self._encode_ids(text, self.max_passage_length)
            # [CLS] Q [SEP] O [SEP]
            ids = (
                [self.tokenizer.cls_token_id]
                + query_ids
                + [self.tokenizer.sep_token_id]
                + psg_ids
                + [self.tokenizer.sep_token_id]
            )
            seg = [0] * (len(query_ids) + 2) + [1] * (len(psg_ids) + 1)
            pair_ids.append(ids)
            pair_mask.append([1] * len(ids))
            pair_seg.append(seg)
            # [CLS] O [SEP]
            ids_b = [self.tokenizer.cls_token_id] + psg_ids + [self.tokenizer.sep_token_id]
            psg_ids_rows.append(ids_b)
            psg_mask_rows.append([1] * len(ids_b))
            psg_seg_rows.append([0] * len(ids_b))

        # Gold labels are always attached by the dataset; the inference
        # pipeline is responsible for removing them before the forward pass
        # (inference-time selection must be label-free, Algorithm 1).
        batch = {
            "qid": rec.qid,
            "cand_ids": rec.ids,
            "pair_input_ids": _pad(pair_ids, self.tokenizer.pad_token_id),
            "pair_attention_mask": _pad(pair_mask, 0),
            "pair_token_type_ids": _pad(pair_seg, 0),
            "psg_input_ids": _pad(psg_ids_rows, self.tokenizer.pad_token_id),
            "psg_attention_mask": _pad(psg_mask_rows, 0),
            "psg_token_type_ids": _pad(psg_seg_rows, 0),
            "labels": torch.tensor(rec.labels, dtype=torch.long),
        }
        return batch


def _pad(sequences, pad_value):
    max_len = max(len(s) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(sequences):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate_one(batch_list):
    """DataLoader collate: batch size is always one query."""
    assert len(batch_list) == 1, "IRAG processes one query (candidate list) per item"
    return batch_list[0]
