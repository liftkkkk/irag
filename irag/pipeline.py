"""Inference pipeline: Algorithm 1 (label-free top-k selection + CPA).

For every query the pipeline produces a total order of the candidate list:

  1. working-set members are ranked by their final (CPA) score, and
  2. passages outside the working set follow below, retaining their
     coarse-score order (ties broken by candidate index).
"""

import torch
from tqdm import tqdm


def _move_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def rerank_dataset(
    model,
    dataset,
    device,
    k=16,
    encode_batch_size=32,
    use_coarse_only=False,
    progress=True,
):
    """Rerank every query of ``dataset``.

    Args:
        model: a trained IRAGModel (or baseline; then all candidates are
            ranked by coarse score).
        dataset: RerankDataset with gold labels attached.
        k: working-set size (None/<=0 = no selection, k = infinity).
        use_coarse_only: rank by coarse score even if the model has an
            interaction stage (used for the first phase of pipeline
            training, where only the scorer is trained).

    Returns:
        (rankings, stats) where rankings = {qid: [cid, ...] in rank order}
        and stats contains working-set diagnostics (gold labels are used
        for diagnostics only, never for selection).
    """
    from torch.utils.data import DataLoader

    from .dataset import collate_one

    model.eval()
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_one, num_workers=0
    )
    rankings = {}
    stats = {"n_queries": 0, "ws_has_positive": 0, "queries_with_positive": 0}

    iterator = tqdm(loader, desc="rerank") if progress else loader
    for batch in iterator:
        qid = batch["qid"]
        cand_ids = batch["cand_ids"]
        gold = batch.pop("labels").tolist()  # never shown to the model
        batch = _move_to_device(batch, device)

        out = model(
            batch, k=k, selection="top_k", encode_batch_size=encode_batch_size
        )
        coarse = out["coarse_logits"].detach().float().cpu()
        n = coarse.numel()

        has_interaction = (
            not use_coarse_only
            and model.config.use_irag
            and "working_set" in out
            and model.interaction is not None
        )

        if has_interaction:
            sel = out["working_set"].cpu()
            fine = out["fine_logits"].detach().float().cpu()
            in_ws = torch.zeros(n, dtype=torch.bool)
            in_ws[sel] = True
            # working set ranked by final score (ties -> candidate index)
            ws_order = torch.argsort(-fine, stable=True)
            sel_sorted = sel[ws_order]
            rest = torch.nonzero(~in_ws, as_tuple=False).squeeze(-1)
            # rest retains coarse-score order (ties -> candidate index)
            rest_order = torch.argsort(-coarse[rest], stable=True)
            rest_sorted = rest[rest_order]
            order = torch.cat([sel_sorted, rest_sorted])
        else:
            order = torch.argsort(-coarse, stable=True)

        rankings[qid] = [cand_ids[i] for i in order.tolist()]

        stats["n_queries"] += 1
        if any(g > 0 for g in gold):
            stats["queries_with_positive"] += 1
            if has_interaction and any(gold[i] > 0 for i in out["working_set"].tolist()):
                stats["ws_has_positive"] += 1

    return rankings, stats


def rankings_to_rows(rankings):
    """Attach monotone scores to a ranking for run-file output.

    Scores are strictly decreasing (n - rank), so any evaluation script
    that sorts by score reproduces the emitted order exactly.
    """
    rows = []
    for qid, cids in rankings.items():
        n = len(cids)
        for rank, cid in enumerate(cids, start=1):
            rows.append((qid, cid, rank, float(n - rank)))
    return rows


def write_wikiqa_run(rankings, path):
    """Write ``qid  cid  score`` (3-column TSV, one line per candidate)."""
    with open(path, "w", encoding="utf-8") as f:
        for qid, cid, _, score in rankings_to_rows(rankings):
            f.write(f"{qid}\t{cid}\t{score:.6f}\n")


def write_marco_run(rankings, path, run_id="irag"):
    """Write a TREC-style run: ``qid  Q0  pid  rank  score  run_id``."""
    with open(path, "w", encoding="utf-8") as f:
        for qid, cid, rank, score in rankings_to_rows(rankings):
            f.write(f"{qid}\tQ0\t{cid}\t{rank}\t{score:.6f}\t{run_id}\n")
