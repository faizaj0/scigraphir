"""
make_sir4_subset.py -- the query subset each LLM-retrieval baseline runs on, per SIR-4 field.

WHY A SUBSET. MOOSE-Chem reads the whole corpus once per query (corpus/12 calls per
elimination round), so its cost is proportional to queries x docs. The TOMATO rows used a
250 same / 250 cross stratified sample (seed 42). SIR-4 does not have 250 cross-field
queries in any field (cs 247, biology 141, physics 138, matsci 121), so the rule here is:

    every cross-field query  +  up to N_SAME same-field queries (seed 42)

The cross column, which Table 9.3 is about, is therefore the FULL cross slice in every
field; only the same-field column is sampled. matsci (210 same) runs in full.

Output: <out>/subset_<field>.json  {n_same, n_cross, seed, query_ids, strata}
        strata is {query_id: "same"|"cross"} for the scorer's slice table.

Usage
-----
    python3 llm_baselines/make_sir4_subset.py                      # all four fields, 250 same
    python3 llm_baselines/make_sir4_subset.py --fields cs --n-same 250
    python3 llm_baselines/make_sir4_subset.py --n-same 0            # cross-only (not recommended: no gap column)
    python3 llm_baselines/make_sir4_subset.py --full                # every query, no sampling
"""
from __future__ import annotations

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
S4 = os.path.dirname(HERE)
CARGO = os.path.dirname(S4)
DATA_ROOT = f"{CARGO}/retriever/data"
FIELDS = ["cs", "biology", "physics", "matsci"]


def queries_path(field: str) -> str:
    return f"{DATA_ROOT}/sir4_{field}_test/raw/test.json"


def corpus_path(field: str) -> str:
    return f"{DATA_ROOT}/sir4_{field}_test/raw/documents.json"


def build(field: str, n_same: int, seed: int, full: bool) -> dict:
    qs = json.load(open(queries_path(field)))
    same = [q["id"] for q in qs if q["stratum"] == "same"]
    cross = [q["id"] for q in qs if q["stratum"] == "cross"]
    assert len(same) + len(cross) == len(qs), f"{field}: a query has a stratum other than same/cross"
    if not full:
        rng = random.Random(seed)
        same = sorted(rng.sample(same, min(n_same, len(same))))
    ids = same + cross
    return {"field": field, "n_same": len(same), "n_cross": len(cross), "n_same_available": sum(q["stratum"] == "same" for q in qs),
            "seed": None if full else seed, "full": full, "query_ids": ids,
            "strata": {i: "same" for i in same} | {i: "cross" for i in cross}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", nargs="+", default=FIELDS)
    ap.add_argument("--n-same", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true", help="all queries, no sampling")
    ap.add_argument("--out", default=f"{HERE}/subsets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    total = 0
    for f in a.fields:
        m = build(f, a.n_same, a.seed, a.full)
        tag = "full" if a.full else f"same{a.n_same}_allcross_seed{a.seed}"
        p = f"{a.out}/subset_{f}_{tag}.json"
        json.dump(m, open(p, "w"), indent=1)
        n_docs = len(json.load(open(corpus_path(f))))
        total += len(m["query_ids"])
        print(f"{f:8} same {m['n_same']:4} / {m['n_same_available']:4}   cross {m['n_cross']:4} (all)   "
              f"queries {len(m['query_ids']):4}   docs {n_docs:5}   -> {os.path.relpath(p, S4)}")
    print(f"total queries {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
