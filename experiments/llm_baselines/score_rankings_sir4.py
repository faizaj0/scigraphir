"""
score_rankings_sir4.py -- same/cross R@3, R@5, nDCG@5 for an LLM-baseline ranking on one SIR-4 field.

Input is the runner's rankings file, {query_id: [doi, doi, ...]} best-first (a truncated list
is fine: MOOSE-Star only proposes 25 leaves). Scoring is eval/score_sir4.py's score_one, so
recall is over ALL golds and nDCG@5 is binary-relevance with an ideal of min(#golds, 5),
exactly as for the non-LLM rows of Table 9.3.

Only queries present in the rankings file are scored, so a subset run scores its subset.
Pass --subset to assert that every manifest query was ranked (catches a crashed run).

Usage
-----
    python3 llm_baselines/score_rankings_sir4.py --field cs \\
        --rankings llm_baselines/outputs/moose_chem/cs/rankings.json --method MOOSE-Chem
    # -> prints the row, writes <rankings dir>/scores.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from sir4_llm_data import S4, default_subset, load_queries

sys.path.insert(0, f"{S4}/eval")
from score_sir4 import score_one  # noqa: E402

COLS = [("recall@3", "R@3"), ("recall@5", "R@5"), ("ndcg@5", "nDCG@5"), ("recall@10", "R@10"), ("mrr", "MRR")]


def score(field: str, rankings: dict[str, list[str]], subset: str | None) -> dict:
    queries = {q["query_id"]: q for q in load_queries(field)}
    if subset:
        want = set(json.load(open(subset))["query_ids"])
        missing = want - set(rankings)
        assert not missing, f"{len(missing)} manifest queries have no ranking (run not finished?)"
    per = {}
    for qid, ranked in rankings.items():
        q = queries.get(qid)
        if q is None or not ranked:
            continue
        m = score_one(list(ranked), set(q["golds"]), [])
        m["stratum"] = q["stratum"]
        per[qid] = m
    out = {}
    for slc in ("same", "cross", "all"):
        rows = [m for m in per.values() if slc == "all" or m["stratum"] == slc]
        out[slc] = {"n": len(rows)} | {k: (100 * sum(r[k] for r in rows) / len(rows) if rows else float("nan"))
                                       for k, _ in COLS}
    return out


def fmt_row(method: str, s: dict) -> str:
    return (f"| {method} | " + " | ".join(f"{s['same'][k]:.2f}" for k, _ in COLS[:3]) + " | "
            + " | ".join(f"{s['cross'][k]:.2f}" for k, _ in COLS[:3])
            + f" | {100 * (s['cross']['recall@5'] - s['same']['recall@5']) / max(s['same']['recall@5'], 1e-9):+.2f}% |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--method", default=None)
    ap.add_argument("--subset", default="default", help="manifest to check completeness against; 'none' to skip")
    a = ap.parse_args()
    subset = None if a.subset == "none" else (default_subset(a.field) if a.subset == "default" else a.subset)
    rankings = json.load(open(a.rankings))
    s = score(a.field, rankings, subset)
    s["field"], s["method"], s["rankings"] = a.field, a.method or os.path.basename(os.path.dirname(a.rankings)), a.rankings
    print(f"{a.field}: n same {s['same']['n']}, cross {s['cross']['n']}")
    print("| Method | same R@3 | same R@5 | same nDCG@5 | cross R@3 | cross R@5 | cross nDCG@5 | dR@5 |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|")
    print(fmt_row(s["method"], s))
    out = os.path.join(os.path.dirname(a.rankings), "scores.json")
    json.dump(s, open(out, "w"), indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
