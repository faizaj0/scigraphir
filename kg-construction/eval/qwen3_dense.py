"""
Qwen3-Embedding-0.6B dense baseline on the TOMATO test set -> predictions in the unified schema.

This is the apples-to-apples dense ceiling for G-reasoner: G-reasoner DISTILLS this exact embedder's
question<->document similarity (sft_trainer.py: distillation_target = question_emb @ doc_emb.T), so the
gap between this baseline and G-reasoner is *what the graph adds* (or doesn't) over the raw embedder.
Same corpus / cold query (b_text) / single-gold as BM25 / BGE / GFM-RAG / G-reasoner -> directly comparable.

Protocol matches gfm_reasoner/config/text_emb_model/qwen3.yaml: Qwen/Qwen3-Embedding-0.6B, L2-normalized
(cosine), query gets the retrieval instruction, passages get none, native 1024-dim.

Deps: pip install sentence-transformers tqdm numpy  (transformers>=4.51 for the Qwen3 architecture)
Usage: python eval/qwen3_dense.py   then   python eval/score.py eval/predictions_qwen3.json --name "Qwen3-0.6B"
"""

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Exact query instruction from the gfm_reasoner qwen3 config (passages get NO instruction).
QWEN_QUERY_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(os.environ.get("CARGO_ROOT") or (os.environ.get("CARGO_ROOT") or os.path.expanduser("~/Desktop/CARGO")), "outputs/caches"))
    ap.add_argument("--field", default="b_text")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_seq_length", type=int, default=512,
                    help="cap sequence length (Qwen3 defaults to 32768 -> unusably slow on CPU/MPS)")
    ap.add_argument("--no_instruct", action="store_true", help="drop the Qwen3 query instruction")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "predictions_qwen3.json"))
    args = ap.parse_args()

    corpus = json.load(open(f"{args.cache}/corpus.json"))
    queries = json.load(open(f"{args.cache}/queries.json"))
    doc_ids = list(corpus)
    instruct = "" if args.no_instruct else QWEN_QUERY_INSTRUCT
    print(f"corpus {len(doc_ids)} docs | {len(queries)} queries | field={args.field} | "
          f"model={args.model} | instruct={bool(instruct)}")

    m = SentenceTransformer(args.model)
    m.max_seq_length = args.max_seq_length
    print(f"max_seq_length capped to {m.max_seq_length}")
    print("encoding corpus (passages, no instruction)...")
    D = m.encode([corpus[d] for d in doc_ids], normalize_embeddings=True,
                 batch_size=args.batch_size, show_progress_bar=True)
    print("encoding queries (with retrieval instruction)...")
    Q = m.encode([instruct + q[args.field] for q in queries], normalize_embeddings=True,
                 batch_size=args.batch_size, show_progress_bar=True)
    D = np.asarray(D, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    print(f"embeddings: docs {D.shape}, queries {Q.shape}")

    out = []
    for qi, q in enumerate(tqdm(queries, desc="Qwen3 score")):
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
