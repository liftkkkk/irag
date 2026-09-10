import torch

from irag.dataset import RerankDataset, collate_one, load_jsonl


def test_load_jsonl_roundtrip(toy_jsonl, toy_records):
    loaded = list(load_jsonl(toy_jsonl))
    assert len(loaded) == len(toy_records)
    assert loaded[0]["qid"] == "Q1"


def test_dataset_lengths(toy_jsonl, mock_tokenizer):
    ds = RerankDataset(toy_jsonl, mock_tokenizer)
    assert len(ds) == 2
    assert ds.records[0].n == 5
    assert ds.records[1].n == 3


def test_labels_are_attached(toy_jsonl, mock_tokenizer):
    ds = RerankDataset(toy_jsonl, mock_tokenizer)
    assert ds[0]["labels"].tolist() == [0, 1, 0, 0, 1]


def test_pair_encoding_structure(toy_jsonl, mock_tokenizer):
    ds = RerankDataset(toy_jsonl, mock_tokenizer, max_query_length=4, max_passage_length=6)
    item = ds[0]
    ids = item["pair_input_ids"][0].tolist()
    mask = item["pair_attention_mask"][0].tolist()
    seg = item["pair_token_type_ids"][0].tolist()

    cls, sep, pad = mock_tokenizer.cls_token_id, mock_tokenizer.sep_token_id, mock_tokenizer.pad_token_id
    assert ids[0] == cls
    assert mask[0] == 1

    # first SEP ends the query segment; second SEP ends the passage segment
    sep_positions = [i for i, t in enumerate(ids) if t == sep and mask[i] == 1]
    assert len(sep_positions) == 2
    first_sep = sep_positions[0]
    assert all(s == 0 for s in seg[: first_sep + 1])
    assert all(s == 1 for s in seg[first_sep + 1 : sep_positions[1] + 1])

    # padding is masked out
    assert all(m == 0 for m, t in zip(mask, ids) if t == pad)


def test_passage_encoding_is_segment_zero(toy_jsonl, mock_tokenizer):
    ds = RerankDataset(toy_jsonl, mock_tokenizer)
    item = ds[0]
    assert item["psg_token_type_ids"].max().item() == 0
    assert item["psg_input_ids"][:, 0].tolist() == [mock_tokenizer.cls_token_id] * 5


def test_truncation_respects_max_lengths(toy_jsonl, mock_tokenizer):
    max_q, max_p = 3, 4
    ds = RerankDataset(toy_jsonl, mock_tokenizer, max_query_length=max_q, max_passage_length=max_p)
    item = ds[0]
    # pair = [CLS] + q + [SEP] + p + [SEP]
    assert item["pair_input_ids"].shape[1] <= max_q + max_p + 3
    # passage-only = [CLS] + p + [SEP]
    assert item["psg_input_ids"].shape[1] <= max_p + 2


def test_query_prefix_shared_across_candidates(toy_jsonl, mock_tokenizer):
    ds = RerankDataset(toy_jsonl, mock_tokenizer)
    item = ds[0]
    ids = item["pair_input_ids"].tolist()
    sep = mock_tokenizer.sep_token_id
    prefixes = [row[: row.index(sep)] for row in ids]
    assert all(p == prefixes[0] for p in prefixes)
    assert len(ids) == 5


def test_collate_one_passthrough(toy_jsonl, mock_tokenizer):
    ds = RerankDataset(toy_jsonl, mock_tokenizer)
    batch = collate_one([ds[0]])
    assert batch["qid"] == "Q1"
    assert torch.is_tensor(batch["pair_input_ids"])


def test_collate_one_rejects_multiple_queries(toy_jsonl, mock_tokenizer):
    ds = RerankDataset(toy_jsonl, mock_tokenizer)
    try:
        collate_one([ds[0], ds[1]])
    except AssertionError:
        pass
    else:
        raise AssertionError("collate_one must refuse batches larger than one query")
