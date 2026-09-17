"""
baselines_sir4.py -- the retrieval-baseline table: BM25, and any HF dense encoder.

WHY NOT bge_sir4.py. That script is not just "the BGE baseline": score_sir4.py
defines the `dissimilar` slice as "plain BGE ranks this query's best gold below
100", so its output file IS the slice definition. Adding encoders and pooling
modes to it risks a baseline run silently redefining the slice every other arm
is measured against. This is a separate file and bge_sir4.py is left alone.

POOLING IS NOT A DETAIL. SentenceTransformer wraps a bare HF checkpoint with MEAN
pooling when the repo carries no ST config. SPECTER2 and SciNCL are both trained
with CLS pooling, so loading them the default way silently evaluates a different
model than the paper's and understates both. `--pooling` is therefore explicit,
and `auto` picks the documented pooling per family rather than whatever ST
guesses.

OUTPUT is byte-compatible with bge_sir4.py, so score_sir4.py consumes it
unchanged:
    [{id, stratum, supporting_documents, predictions:{document:[[doc_id, score]]}}]

Usage
-----
    python3 eval/baselines_sir4.py --dataset tomato --split test --model bm25
    python3 eval/baselines_sir4.py --dataset tomato --split test \
        --model allenai/specter2_base --pooling cls --tag specter2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, REPO_ROOT)

from scigraphir_paths import add_dataset_arg, banner, corpus_dir, set_dataset  # noqa: E402

# Documented pooling per family. Anything not listed falls back to mean, which is
# what a generic sentence-transformers checkpoint expects.
CLS_POOLED = ("specter", "scincl", "scibert")

_WORD_RE = re.compile(r"[a-zA-Z]+")
_STOP = set("""a an and are as at be by for from has have in is it of on or that the to
was were will with this these those we our using used use based via study paper""".split())


def _tokens(text: str) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _WORD_RE.finditer(text))
            if len(t) > 2 and t not in _STOP]


def run_bm25(queries, corpus, doc_ids, topk):
    """BM25Okapi over the same corpus text the dense arms encode.

    Ranks the FULL corpus per query, so its recall@100 is comparable with the
    dense arms rather than being capped by a candidate set.
    """
    from rank_bm25 import BM25Okapi
    print(f"[bm25] tokenising {len(doc_ids):,} documents ...")
    bm = BM25Okapi([_tokens(corpus[d]) for d in doc_ids])
    out = []
    for i, q in enumerate(queries):
        if i % 500 == 0:
            print(f"  [bm25] {i:,}/{len(queries):,}", flush=True)
        s = np.asarray(bm.get_scores(_tokens(q["question"])), dtype=np.float32)
        k = min(topk, len(doc_ids))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        out.append((q, [(doc_ids[j], float(s[j])) for j in top]))
    return out


def run_dense(queries, corpus, doc_ids, topk, model_name, pooling, instruct, batch, cpu,
              trust_remote_code=False, dtype=None):
    import torch
    from sentence_transformers import SentenceTransformer, models

    dev = "cpu" if cpu else ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    if pooling == "auto":
        low = model_name.lower()
        pooling = "cls" if any(k in low for k in CLS_POOLED) else "st"
    # An 8B encoder in float32 is ~32 GB of weights before a single activation, so
    # the big ones have to load in half precision or they do not load at all.
    margs = {}
    if trust_remote_code:
        margs["trust_remote_code"] = True
    if dtype:
        margs["torch_dtype"] = getattr(torch, dtype)
    if pooling == "st":
        # The checkpoint ships its own pooling/normalisation config; trust it.
        model = SentenceTransformer(model_name, device=dev,
                                    trust_remote_code=trust_remote_code,
                                    model_kwargs=margs or None)
    else:
        # Build the tower explicitly so the pooling is the documented one rather
        # than sentence-transformers' mean-pooling fallback.
        w = models.Transformer(model_name, max_seq_length=512,
                               model_args=margs or None,
                               tokenizer_args={"trust_remote_code": True}
                               if trust_remote_code else None)
        p = models.Pooling(w.get_word_embedding_dimension(),
                           pooling_mode_cls_token=(pooling == "cls"),
                           pooling_mode_mean_tokens=(pooling == "mean"))
        model = SentenceTransformer(modules=[w, p], device=dev)
    model.max_seq_length = 512
    print(f"[dense] {model_name} on {dev} | pooling={pooling} | instruct={bool(instruct)}")

    def enc(texts, tag):
        print(f"  [enc] {tag}: {len(texts):,} texts")
        v = model.encode(texts, batch_size=batch, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=True)
        return v.astype(np.float32)

    D = enc([corpus[d] for d in doc_ids], "documents")
    Q = enc([(instruct + q["question"]) if instruct else q["question"] for q in queries],
            "queries")
    out, k = [], min(topk, len(doc_ids))
    for i0 in range(0, len(queries), 256):          # chunked: the full matrix is large
        S = Q[i0:i0 + 256] @ D.T
        for r in range(S.shape[0]):
            s = S[r]
            top = np.argpartition(-s, k - 1)[:k]
            top = top[np.argsort(-s[top])]
            out.append((queries[i0 + r], [(doc_ids[j], float(s[j])) for j in top]))
        del S
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--model", required=True,
                    help="'bm25', or any HF/sentence-transformers model id or local path")
    ap.add_argument("--pooling", default="auto", choices=["auto", "st", "cls", "mean"],
                    help="auto = ST config for generic encoders, CLS for SPECTER/SciNCL. "
                         "The default ST fallback is MEAN, which is the wrong model for "
                         "both of those and understates them.")
    ap.add_argument("--instruct", default="",
                    help="prefix prepended to each QUERY only. Instruction-tuned "
                         "retrievers (Qwen3, ReasonIR) expect one; BM25 and SPECTER do not.")
    ap.add_argument("--tag", default=None, help="filename tag; defaults to a slug of --model")
    ap.add_argument("--topk", type=int, default=300,
                    help="ranked documents per query. Must exceed 100: the dissimilar "
                         "rule is 'best gold below rank 100'")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="required by ReasonIR-8B and other custom-architecture encoders")
    ap.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"],
                    help="weight dtype. An 8B encoder needs float16/bfloat16 to fit at all.")
    ap.add_argument("--out", default=None)
    add_dataset_arg(ap)
    a = ap.parse_args()
    set_dataset(a.dataset)
    print(banner())
    if a.topk < 100:
        print("--topk below 100 makes the dissimilar slice undefined", file=sys.stderr)
        return 2

    corpus = json.load(open(f"{corpus_dir(a.split)}/raw/documents.json"))
    queries = json.load(open(f"{corpus_dir(a.split)}/raw/{a.split}.json"))
    doc_ids = list(corpus)
    tag = a.tag or re.sub(r"[^a-z0-9]+", "-", a.model.lower()).strip("-")
    print(f"corpus {len(doc_ids):,} docs | {len(queries):,} queries | topk={a.topk} | tag={tag}")

    if a.model.lower() == "bm25":
        ranked = run_bm25(queries, corpus, doc_ids, a.topk)
    else:
        ranked = run_dense(queries, corpus, doc_ids, a.topk, a.model, a.pooling,
                           a.instruct, a.batch, a.cpu, a.trust_remote_code, a.dtype)

    out = [{"id": q["id"],
            "stratum": q.get("stratum"),
            "supporting_documents": q.get("supporting_documents") or [],
            "predictions": {"document": [[d, s] for d, s in top]}}
           for q, top in ranked]
    dest = a.out or f"{ROOT}/data/predictions_{tag}_{a.dataset}_{a.split}.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"))
    print(f"wrote {dest}  ({len(out):,} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
