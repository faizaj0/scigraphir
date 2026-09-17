"""
bge_sir4.py -- step 14. Plain BGE-large dense baseline on a SIR-4 split.

This is the dense baseline for every dataset, TOMATO-Star included: it shares the corpus
loader and the multi-gold record format with the rest of the pipeline.

WHY THIS MATTERS BEYOND "a baseline". `score_sir4.py` defines the `dissimilar`
slice as "plain BGE ranks this query's BEST gold below 100". Without this file
there is no similar/dissimilar split at all, which is the axis the whole
cross-domain argument rests on.

The encoder is imported from handcrafted_scorer.py rather than re-specified, so the
baseline and the handcrafted scorer's dense term are the SAME vectors, same instruction,
same cache. A baseline built with a slightly different encoder would make every
delta unattributable.

Usage
-----
    python3 eval/bge_sir4.py --dataset sir4_cs --split test
    python3 eval/bge_sir4.py --dataset sir4_cs --split test --topk 300
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ROOT)
KG = f"{REPO_ROOT}/retriever"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, KG)
from scigraphir_paths import add_dataset_arg, banner, corpus_dir, set_dataset  # noqa: E402


DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"

# Imported rather than restated. This used to be a second copy of the string,
# which meant swapping the instruction in handcrafted_scorer.py left the dense
# baseline on the old one and the two silently disagreed.
QWEN_QI = None          # resolved from handcrafted_scorer in main(); see op.QWEN_QI


def model_slug(name: str) -> str:
    """'' for the default encoder, else a filename-safe tag.

    THIS IS A CORRECTNESS FIX, NOT COSMETICS. cached_encode keys purely on the
    tag it is handed, and the tag used to be `{split}_doc` with no mention of the
    model. Running --model Qwen/... therefore loaded BGE's cached .npy and
    reported it as Qwen3: a silent contamination of exactly the kind that has
    already cost this project a day. Empty for the default keeps the existing
    cache hits (and the shared-encoder guarantee with handcrafted_scorer) intact.
    """
    if name == DEFAULT_MODEL:
        return ""
    import re
    return "_" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_handcrafted_module():
    """Import handcrafted_scorer.py for cached_encode + BGE_QI, so the baseline and
    the handcrafted scorer's dense term are literally the same encoder and cache."""
    spec = importlib.util.spec_from_file_location("op", f"{KG}/eval/handcrafted_scorer.py")
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)
    return op


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--topk", type=int, default=300,
                    help="ranked documents per query. Must exceed 100: the "
                         "dissimilar rule is 'best gold below rank 100'")
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=None)
    add_dataset_arg(ap)
    a = ap.parse_args()
    set_dataset(a.dataset)
    print(banner())
    if a.topk < 100:
        print("--topk below 100 makes the dissimilar slice undefined", file=sys.stderr)
        return 2

    op = load_handcrafted_module()
    from sentence_transformers import SentenceTransformer

    corpus = json.load(open(f"{corpus_dir(a.split)}/raw/documents.json"))
    queries = json.load(open(f"{corpus_dir(a.split)}/raw/{a.split}.json"))
    doc_ids = list(corpus)
    qkey = hashlib.md5("|".join(q["id"] for q in queries).encode()).hexdigest()[:8]
    print(f"corpus {len(doc_ids):,} docs | {len(queries):,} queries | topk={a.topk}")

    import torch
    dev = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available()
                               else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(a.model, device=dev)
    model.max_seq_length = 512

    # Same tags handcrafted_scorer.py uses, so whatever it already encoded is reused
    # rather than recomputed with a different seed of the same model.
    slug = model_slug(a.model)
    qi = op.query_instruction(a.model)      # one definition, shared with the handcrafted scorer
    print(f"encoder {a.model}  cache tag suffix {slug!r}  instruct {qi[:40]!r}...")
    D = op.cached_encode(model, [corpus[d] for d in doc_ids],
                         f"{a.split}_doc{slug}").astype(np.float32)
    Q = op.cached_encode(model, [q["question"] for q in queries],
                         f"{a.split}_query_{qkey}{slug}{op.qi_tag(qi)}",
                         instruct=qi).astype(np.float32)

    out, k = [], min(a.topk, len(doc_ids))
    CH = 256                                    # chunked: the full score matrix is
    for i0 in range(0, len(queries), CH):       # 5,445 x 20,203 on train
        S = Q[i0:i0 + CH] @ D.T
        for r in range(S.shape[0]):
            q = queries[i0 + r]
            s = S[r]
            top = np.argpartition(-s, k - 1)[:k]
            top = top[np.argsort(-s[top])]
            out.append({
                "id": q["id"],
                "stratum": q.get("stratum"),
                "supporting_documents": q.get("supporting_documents") or [],
                "predictions": {"document": [[doc_ids[i], float(s[i])] for i in top]},
            })
        del S

    # The slug is in the DEFAULT filename too. Without it a Qwen3 run would
    # overwrite predictions_bge_*.json, and score_sir4.py reads that exact file
    # to define the `dissimilar` slice -- so one Qwen3 run would silently
    # redefine the slice for every arm scored afterwards.
    dest = a.out or f"{ROOT}/data/predictions_bge{slug}_{a.dataset}_{a.split}.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"))
    print(f"wrote {dest}  ({len(out):,} records)")

    # Report the slice this file exists to define, so a broken run is obvious here
    # rather than three steps later inside the scorer.
    below = 0
    for rec, q in zip(out, queries):
        golds = set(q.get("supporting_documents") or [])
        rank = next((i for i, d in enumerate(rec["predictions"]["document"], 1)
                     if d[0] in golds), 10 ** 9)
        below += rank > 100
    print(f"dissimilar slice: {below:,} of {len(out):,} queries "
          f"({below/max(len(out),1):.1%}) have their best gold below rank 100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
