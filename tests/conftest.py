"""Shared fixtures: a tiny random BERT and a deterministic mock tokenizer."""

import os

import pytest
import torch
from transformers import BertConfig, BertModel

TINY_VOCAB = 111


class MockTokenizer:
    """Deterministic whitespace tokenizer with an id per distinct word.

    Used where the tests must control exact token ids without downloading
    a real BERT vocabulary.
    """

    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self._vocab = {}
        self._next = 4  # 3 reserved for unknown

    def tokenize(self, text):
        return text.lower().split()

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for tok in tokens:
            if tok not in self._vocab:
                self._vocab[tok] = self._next
                self._next += 1
                if self._next >= TINY_VOCAB:
                    self._next = 4
            ids.append(self._vocab[tok])
        return ids

    def save_pretrained(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "mock_tokenizer.txt"), "w") as f:
            f.write("mock\n")


def tiny_bert(hidden=32, layers=2, heads=2):
    """A small random-weights BERT for fast CPU tests."""
    cfg = BertConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=4 * hidden,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    return BertModel(cfg)


def fake_batch(n=6, length=24, vocab=TINY_VOCAB, with_labels=True, seed=0):
    """A random batch of pair / passage-only encodings for one query."""
    g = torch.Generator().manual_seed(seed)
    batch = {
        "pair_input_ids": torch.randint(3, vocab, (n, length), generator=g),
        "pair_attention_mask": torch.ones(n, length, dtype=torch.long),
        "pair_token_type_ids": torch.zeros(n, length, dtype=torch.long),
        "psg_input_ids": torch.randint(3, vocab, (n, length), generator=g),
        "psg_attention_mask": torch.ones(n, length, dtype=torch.long),
        "psg_token_type_ids": torch.zeros(n, length, dtype=torch.long),
    }
    if with_labels:
        batch["labels"] = torch.randint(0, 2, (n,), generator=g)
    return batch


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def tiny_bert_model():
    return tiny_bert()


@pytest.fixture
def toy_records():
    return [
        {
            "qid": "Q1",
            "query": "how far is the moon",
            "candidates": [
                {"id": "D0", "text": "the moon is very far away", "label": 0},
                {"id": "D1", "text": "the moon is 384400 km away", "label": 1},
                {"id": "D2", "text": "cats are cute animals", "label": 0},
                {"id": "D3", "text": "the sun is a star", "label": 0},
                {"id": "D4", "text": "distance to the moon measured in km", "label": 1},
            ],
        },
        {
            "qid": "Q2",
            "query": "who wrote hamlet",
            "candidates": [
                {"id": "E0", "text": "hamlet was written by shakespeare", "label": 1},
                {"id": "E1", "text": "hamlet is a tragedy", "label": 0},
                {"id": "E2", "text": "shakespeare lived in england", "label": 0},
            ],
        },
    ]


@pytest.fixture
def toy_jsonl(tmp_path, toy_records):
    import json

    path = tmp_path / "toy.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in toy_records:
            f.write(json.dumps(rec) + "\n")
    return path
