"""Download and prepare MS MARCO 2.0 passage reranking data.

Outputs the unified JSONL format (one line per query)::

    {"qid": "...", "query": "...",
     "candidates": [{"id": "<pid>", "text": "...", "label": 0|1}]}

* Development set: the official BM25 ``top1000.dev`` candidate lists with
  labels from ``qrels.dev.small.tsv`` (identical candidate sets for every
  system; 6,980 queries).
* Training set: built from the official training triples
  (``qidpidtriples.train.small.tsv``, 398,792 queries): all positives of a
  query plus ``--max_neg`` distinct hard negatives (first seen in the
  triples file), with passage text from ``collection.tsv``.

Note on fidelity: the original experiments scored BM25 top-400 candidate
lists per training query (not publicly distributed); this script instead
builds training lists from the official triples, which is documented in
the README. Development-set candidate lists are the official ones.

Required input files (downloaded automatically when missing):

* ``collectionandqueries.tar.gz``  (collection.tsv, queries.train.tsv,
  qrels.dev.small.tsv)
* ``qidpidtriples.train.small.tar.gz`` -> qidpidtriples.train.small.tsv
* ``top1000.dev.tar.gz`` -> top1000.dev

Usage:
    python scripts/prepare_marco.py --out_dir data/marco
    python scripts/prepare_marco.py --out_dir data/marco --skip_train
    python scripts/prepare_marco.py --out_dir data/marco --max_neg 15
"""

import argparse
import json
import os
import tarfile
import urllib.request

BASE = "https://msmarco.z22.web.core.windows.net/msmarcoranking"
URLS = {
    "collectionandqueries.tar.gz": f"{BASE}/collectionandqueries.tar.gz",
    "qidpidtriples.train.small.tar.gz": f"{BASE}/qidpidtriples.train.small.tar.gz",
    "top1000.dev.tar.gz": f"{BASE}/top1000.dev.tar.gz",
    "qrels.dev.small.tsv": f"{BASE}/qrels.dev.small.tsv",
}


def download(url, dest):
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"saved -> {dest}")


def ensure_file(data_dir, name, tarball=None):
    """Locate ``name`` in data_dir, extracting/downloading as needed."""
    path = os.path.join(data_dir, name)
    if os.path.exists(path):
        return path
    if tarball:
        tb = os.path.join(data_dir, tarball)
        if os.path.exists(tb):
            print(f"extracting {name} from {tarball}")
            with tarfile.open(tb, "r:gz") as tf:
                member = next((m for m in tf.getnames() if os.path.basename(m) == name), None)
                if member is None:
                    sys_exit_missing(name, [tarball])
                f = tf.extractfile(member)
                with open(path, "wb") as out:
                    out.write(f.read())
            return path
        if tarball in URLS:
            download(URLS[tarball], tb)
            return ensure_file(data_dir, name, tarball)
    if name in URLS:
        download(URLS[name], path)
        return path
    sys_exit_missing(name, list(URLS))


def sys_exit_missing(name, checked):
    raise SystemExit(
        f"could not locate {name!r} (checked: {', '.join(checked)}). "
        "Download it from https://microsoft.github.io/msmarco/ and place it "
        "in the data directory."
    )


# ----------------------------------------------------------------------
# collection.tsv random access via a byte-offset index
# ----------------------------------------------------------------------
def build_collection_index(collection_path):
    """One streaming pass; returns sorted (pids, offsets) numpy arrays."""
    import numpy as np

    pids = np.empty(8_900_000, dtype=np.int64)  # > 8,841,823 passages
    offsets = np.empty(8_900_000, dtype=np.int64)
    count = 0
    with open(collection_path, "rb") as f:
        offset = 0
        for line in f:
            if count >= pids.size:
                pids = np.concatenate([pids, np.empty(pids.size, dtype=np.int64)])
                offsets = np.concatenate([offsets, np.empty(offsets.size, dtype=np.int64)])
            pids[count] = int(line.split(b"\t", 1)[0])
            offsets[count] = offset
            offset += len(line)
            count += 1
    pids = pids[:count]
    offsets = offsets[:count]
    order = np.argsort(pids)
    print(f"indexed {count} passages in collection.tsv")
    return pids[order], offsets[order]


def fetch_passages(collection_path, index, wanted):
    """Fetch texts for ``wanted`` (list of int pids) using the offset index."""
    import numpy as np

    pids, offsets = index
    texts = {}
    with open(collection_path, "rb") as f:
        for pid in wanted:
            i = int(np.searchsorted(pids, pid))
            if i < len(pids) and pids[i] == pid:
                f.seek(int(offsets[i]))
                line = f.readline()
                parts = line.rstrip(b"\r\n").split(b"\t", 1)
                texts[str(pid)] = parts[1].decode("utf-8", errors="replace") if len(parts) == 2 else ""
    return texts


# ----------------------------------------------------------------------
# dev set: top1000.dev + qrels
# ----------------------------------------------------------------------
def prepare_dev(data_dir, out_path):
    qrels_path = ensure_file(data_dir, "qrels.dev.small.tsv", tarball="collectionandqueries.tar.gz")
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and int(parts[3]) > 0:
                qrels.setdefault(parts[0], set()).add(parts[2])

    top1000 = ensure_file(data_dir, "top1000.dev", tarball="top1000.dev.tar.gz")
    n_queries = 0
    current_qid, current_query, cands = None, None, []
    relevant_qids = 0

    def flush():
        nonlocal n_queries, relevant_qids
        if current_qid is None:
            return
        pos = qrels.get(current_qid, set())
        if pos:
            relevant_qids += 1
        rec = {
            "qid": current_qid,
            "query": current_query,
            "candidates": [
                {"id": pid, "text": passage, "label": 1 if pid in pos else 0}
                for pid, passage in cands
            ],
        }
        with open(out_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_queries += 1

    print(f"streaming {top1000} ...")
    with open(top1000, "r", encoding="utf-8") as f:
        for line in f:
            qid, pid, query, passage = line.rstrip("\n").split("\t")
            if qid != current_qid:
                flush()
                current_qid, current_query, cands = qid, query, []
            cands.append((pid, passage))
    flush()
    print(f"dev: {n_queries} queries ({relevant_qids} with >=1 relevant) -> {out_path}")


# ----------------------------------------------------------------------
# training set: triples + collection + queries
# ----------------------------------------------------------------------
def prepare_train(data_dir, out_path, max_neg, max_queries=None):
    triples_path = ensure_file(data_dir, "qidpidtriples.train.small.tsv", tarball="qidpidtriples.train.small.tar.gz")
    collection_path = ensure_file(data_dir, "collection.tsv", tarball="collectionandqueries.tar.gz")
    queries_path = ensure_file(data_dir, "queries.train.tsv", tarball="collectionandqueries.tar.gz")

    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t")[:2]
            queries[qid] = text

    index = build_collection_index(collection_path)

    print(f"streaming {triples_path} ...")
    n_queries = 0
    prev_qid = None
    current_qid, pos, neg = None, [], set()

    def flush():
        nonlocal n_queries
        if current_qid is None:
            return
        cand_pids = list(dict.fromkeys(pos)) + list(neg)
        texts = fetch_passages(collection_path, index, [int(p) for p in cand_pids])
        rec = {
            "qid": current_qid,
            "query": queries.get(current_qid, ""),
            "candidates": [
                {"id": p, "text": texts.get(p, ""), "label": 1 if i < len(dict.fromkeys(pos)) else 0}
                for i, p in enumerate(cand_pids)
                if p in texts
            ],
        }
        with open(out_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_queries += 1

    with open(triples_path, "r", encoding="utf-8") as f:
        for line in f:
            qid, pos_pid, neg_pid = line.split()
            if qid != current_qid:
                if prev_qid is not None and qid < prev_qid:
                    raise SystemExit(
                        "triples file is not sorted by qid; sort it first "
                        "(e.g. `sort -k1,1 -o out.tsv in.tsv`)"
                    )
                flush()
                if max_queries and n_queries >= max_queries:
                    print(f"reached --max_queries {max_queries}")
                    return
                prev_qid = current_qid
                current_qid, pos, neg = qid, [], set()
            if pos_pid not in neg and pos_pid not in pos:
                pos.append(pos_pid)
            if len(neg) < max_neg and neg_pid not in neg and neg_pid not in pos:
                neg.add(neg_pid)
    flush()
    print(f"train: {n_queries} queries -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/marco")
    ap.add_argument("--data_dir", default=None, help="directory with the official archives (default: out_dir)")
    ap.add_argument("--max_neg", type=int, default=63,
                    help="distinct hard negatives per training query (default 63; "
                         "the original paper used BM25 top-400 lists, see README)")
    ap.add_argument("--max_queries", type=int, default=None, help="cap on training queries (debug)")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_dev", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data_dir = args.data_dir or args.out_dir
    dev_path = os.path.join(args.out_dir, "dev.jsonl")
    train_path = os.path.join(args.out_dir, "train.jsonl")

    if not args.skip_dev:
        if os.path.exists(dev_path):
            os.remove(dev_path)
        prepare_dev(data_dir, dev_path)
    if not args.skip_train:
        if os.path.exists(train_path):
            os.remove(train_path)
        prepare_train(data_dir, train_path, args.max_neg, args.max_queries)


if __name__ == "__main__":
    main()
