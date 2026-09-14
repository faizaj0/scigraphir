"""
BGE-large dense baseline on the TOMATO test set -> predictions file in the unified schema.
Local (CPU/MPS). Same corpus / cold query field (b_text) / stratum as GFM-RAG, so it's comparable.

Protocol: BAAI/bge-large-en-v1.5, L2-normalized embeddings, cosine. BGE's recommended retrieval
setup prepends a query instruction (queries only); toggle with --no_instruct to drop it.

Deps: pip install sentence-transformers tqdm numpy
Usage: python eval/bge.py   then   python eval/score.py eval/predictions_bge.json --name BGE
"""

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BGE_QUERY_INSTRUCT = "Represent this sentence for searching relevant passages: "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(os.environ.get("CARGO_ROOT") or (os.environ.get("CARGO_ROOT") or os.path.expanduser("~/Desktop/CARGO")), "outputs/caches"))
    ap.add_argument("--field", default="b_text")
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--no_instruct", action="store_true", help="drop the BGE query instruction")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "predictions_bge.json"))
    args = ap.parse_args()

    corpus = json.load(open(f"{args.cache}/corpus.json"))
    queries = json.load(open(f"{args.cache}/queries.json"))
    doc_ids = list(corpus)
    instruct = "" if args.no_instruct else BGE_QUERY_INSTRUCT
    print(f"corpus {len(doc_ids)} | {len(queries)} queries | field={args.field} | instruct={bool(instruct)}")

    m = SentenceTransformer(args.model)
    D = m.encode([corpus[d] for d in doc_ids], normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    Q = m.encode([instruct + q[args.field] for q in queries], normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    D = np.asarray(D, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)

    out = []
    for qi, q in enumerate(tqdm(queries, desc="BGE score")):
        s = D @ Q[qi]
        k = min(args.topk, len(doc_ids))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        out.append({
            "id": q["query_id"], "stratum": q["stratum"],
            "supporting_documents": [q["gold_key"]],
            "predictions": {"document": [[doc_ids[i], float(s[i])] for i in top]},
        })
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
