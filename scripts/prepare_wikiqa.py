"""Download and prepare the WIKIQA corpus.

Produces one JSONL file per split (train/dev/test), each line::

    {"qid": "...", "query": "...",
     "candidates": [{"id": "<SentenceID>", "text": "...", "label": 0|1}]}

The official corpus (3,047 questions / 29,258 candidates / 1,473 positive
candidates) is downloaded from Microsoft unless a local zip is supplied.

Usage:
    python scripts/prepare_wikiqa.py --out_dir data/wikiqa
    python scripts/prepare_wikiqa.py --out_dir data/wikiqa --zip WikiQACorpus.zip
"""

import argparse
import csv
import io
import json
import os
import sys
import urllib.request
import zipfile

WIKIQA_URL = (
    "https://download.microsoft.com/download/E/5/F/"
    "E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip"
)
OFFICIAL_PAGE = "https://www.microsoft.com/en-us/download/details.aspx?id=52419"

SPLITS = ("train", "dev", "test")


def download(url, dest):
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"saved -> {dest}")


def parse_split(path):
    """Parse a WikiQA-{split}.txt into query-grouped records."""
    # Header: QuestionID  Question  DocumentID  DocumentTitle  SentenceID
    #         Sentence  Label  (some mirrors add an extra empty column).
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        header = [h.strip() for h in header]
        col = {name: i for i, name in enumerate(header)}

        # Robust column lookup (handles duplicated/shifted header names).
        def find(*names, default=None):
            for name in names:
                if name in col:
                    return col[name]
            return default

        qid_i = find("QuestionID")
        query_i = find("Question")
        sid_i = find("SentenceID")
        sent_i = find("Sentence")
        label_i = find("Label")
        if None in (qid_i, query_i, sid_i, sent_i, label_i):
            # fall back to fixed positions (standard layout, 7 columns)
            qid_i, query_i, sid_i, sent_i, label_i = 0, 1, -3, -2, -1

        groups = {}
        order = []
        for row in reader:
            if not row or len(row) <= max(qid_i, label_i):
                continue
            qid = row[qid_i].strip()
            if qid not in groups:
                groups[qid] = {"qid": qid, "query": row[query_i].strip(), "candidates": []}
                order.append(qid)
            try:
                label = int(float(row[label_i].strip()))
            except ValueError:
                label = 0
            groups[qid]["candidates"].append(
                {
                    "id": row[sid_i].strip(),
                    "text": row[sent_i].strip(),
                    "label": label,
                }
            )
    return [groups[q] for q in order]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/wikiqa")
    ap.add_argument("--zip", default=None, help="local WikiQACorpus.zip (skips download)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    zip_path = args.zip or os.path.join(args.out_dir, "WikiQACorpus.zip")
    if not os.path.exists(zip_path):
        download(WIKIQA_URL, zip_path)

    total_q, total_c, total_p = 0, 0, 0
    with zipfile.ZipFile(zip_path) as zf:
        names = {os.path.basename(n): n for n in zf.namelist()}
        for split in SPLITS:
            inner = names.get(f"WikiQA-{split}.txt")
            if inner is None:
                sys.exit(f"WikiQA-{split}.txt not found inside {zip_path}")
            with zf.open(inner) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8").read()
            tmp = os.path.join(args.out_dir, f"WikiQA-{split}.txt")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            records = parse_split(tmp)
            out = os.path.join(args.out_dir, f"{split}.jsonl")
            write_jsonl(records, out)

            n_c = sum(len(r["candidates"]) for r in records)
            n_p = sum(
                1 for r in records for c in r["candidates"] if c["label"] > 0
            )
            n_qa = sum(1 for r in records if any(c["label"] > 0 for c in r["candidates"]))
            print(
                f"{split}: {len(records)} questions ({n_qa} with answers), "
                f"{n_c} candidates, {n_p} positive -> {out}"
            )
            total_q += len(records)
            total_c += n_c
            total_p += n_p
            os.remove(tmp)

    print(f"total: {total_q} questions, {total_c} candidates, {total_p} positive")
    print("expected: 3047 questions, 29258 candidates, 1473 positive")
    if (total_q, total_c, total_p) != (3047, 29258, 1473):
        print("WARNING: counts differ from the official corpus statistics.")


if __name__ == "__main__":
    main()
