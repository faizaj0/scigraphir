"""
make_smoke.py -- carve a tiny end-to-end rehearsal set out of the staged CS data.

The smoke test exists because the expensive failures in this pipeline are
SILENT, not loud:

  * a query whose phrases snap to nothing gets dropped by the loader with a
    log line ("Skipping sample ... due to empty start nodes") and simply
    vanishes from the run. The final number is then computed over fewer
    queries than you think, and nothing says so.
  * a stale cache from another corpus loads if the row count happens to match.
  * frames come back empty or malformed for a slice of documents.

None of those raise. All three are cheap to catch at 50 queries and expensive
to discover after 30,881 LLM calls.

WHAT IT BUILDS. A self-contained dataset named `sir4_cs_smoke` with the same
shape as the real thing, so every downstream script runs unmodified with
`--dataset sir4_cs_smoke`. The corpus is the golds of the sampled queries plus
random filler, because a corpus of golds alone cannot expose a ranking failure:
everything in it is a correct answer for somebody.

Usage
-----
    python3 prep/make_smoke.py                    # 50 test + 50 train queries
    python3 prep/make_smoke.py --queries 20 --corpus 300
"""
from __future__ import annotations

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CARGO = os.path.dirname(ROOT)
KG_DATA = f"{CARGO}/kg-construction/data"


def carve(src_dir: str, split: str, n_queries: int, n_corpus: int, seed: int):
    corpus = json.load(open(f"{src_dir}/raw/documents.json"))
    queries = json.load(open(f"{src_dir}/raw/{split}.json"))

    rng = random.Random(seed)
    qs = rng.sample(queries, min(n_queries, len(queries)))

    keep = {g for q in qs for g in q["supporting_documents"] if g in corpus}
    n_gold = len(keep)
    # Filler so the task is a ranking task and not a lookup. Without it every
    # document is a gold for some query and precision is meaningless.
    pool = [d for d in corpus if d not in keep]
    keep |= set(rng.sample(pool, min(max(n_corpus - len(keep), 0), len(pool))))

    sub_corpus = {d: corpus[d] for d in keep}
    # Golds outside the carved corpus would be unanswerable, so drop them and
    # then drop any query left with nothing. Silently keeping them would build
    # a floor of guaranteed zeros into the rehearsal.
    out_q, dropped = [], 0
    for q in qs:
        g = [d for d in q["supporting_documents"] if d in sub_corpus]
        if not g:
            dropped += 1
            continue
        out_q.append({**q, "supporting_documents": g})
    return sub_corpus, out_q, n_gold, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{ROOT}/data/active", help="staged domain (default: data/active)")
    ap.add_argument("--name", default="sir4_cs_smoke")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--corpus", type=int, default=400, help="documents per split, golds included")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=KG_DATA, help="where the scripts look for corpora")
    a = ap.parse_args()

    print(f"source: {os.path.realpath(a.src)}")
    for split in ("train", "test"):
        src = f"{a.src}/tomato_{split}"
        if not os.path.isdir(src):
            print(f"  missing {src}; run prep/stage_sir4.py first")
            return 2
        corpus, queries, n_gold, dropped = carve(src, split, a.queries, a.corpus, a.seed)
        raw = f"{a.out}/{a.name}_{split}/raw"
        os.makedirs(raw, exist_ok=True)
        json.dump(corpus, open(f"{raw}/documents.json", "w"))
        json.dump(queries, open(f"{raw}/{split}.json", "w"))
        mean_g = sum(len(q["supporting_documents"]) for q in queries) / max(len(queries), 1)
        print(f"  {split:5} {len(queries):3} queries  {len(corpus):4} docs "
              f"({n_gold} gold + {len(corpus)-n_gold} filler)  mean {mean_g:.2f} golds/query"
              + (f"  [{dropped} queries dropped: gold not in carved corpus]" if dropped else ""))
        print(f"        -> {a.out}/{a.name}_{split}")

    print(f"\nnext:  export CARGO_DATASET={a.name}   (or pass --dataset {a.name} to each script)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
