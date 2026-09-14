"""
BM25 baseline on the TOMATO test set -> predictions file in the unified schema (see score.py).
Local, no GPU. Same corpus (3,033 docs) / same cold query field (b_text) / same stratum as GFM-RAG,
so the numbers are directly comparable.

Deps: pip install rank_bm25 tqdm
Usage: python eval/bm25.py   then   python eval/score.py eval/predictions_bm25.json --name BM25
"""

import argparse
import json
import os
import re

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def tok(s: str) -> list:
    return re.findall(r"[a-z0-9]+", s.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(os.environ.get("CARGO_ROOT") or (os.environ.get("CARGO_ROOT") or os.path.expanduser("~/Desktop/CARGO")), "outputs/caches"))
    ap.add_argument("--field", default="b_text")  # cold query, matches the graph + GFM-RAG run
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "predictions_bm25.json"))
    args = ap.parse_args()

    corpus = json.load(open(f"{args.cache}/corpus.json"))
    queries = json.load(open(f"{args.cache}/queries.json"))
    doc_ids = list(corpus)
    print(f"corpus {len(doc_ids)} docs | {len(queries)} queries | field={args.field}")

    print("tokenizing corpus...")
    bm = BM25Okapi([tok(corpus[d]) for d in tqdm(doc_ids)])

    out = []
    for q in tqdm(queries, desc="BM25"):
        scores = bm.get_scores(tok(q[args.field]))
        top = sorted(range(len(doc_ids)), key=lambda i: -scores[i])[: args.topk]
        out.append({
            "id": q["query_id"], "stratum": q["stratum"],
            "supporting_documents": [q["gold_key"]],
            "predictions": {"document": [[doc_ids[i], float(scores[i])] for i in top]},
        })
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
