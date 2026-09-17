"""
table91_row.py -- print one Table 9.1 row from score_sir4.py --json-out files.

Table 9.1 reports Recall@3, Recall@5 and nDCG@5 on the `same` and `cross`
slices of TOMATO, all x100, plus the cross-vs-same gap

    dR@5 = (Recall@5[cross] - Recall@5[same]) / Recall@5[same] * 100

which is how every gap in the printed table was derived (SciGraphIR: 33.22 vs
44.11 -> -24.69%). Nothing here recomputes a metric; it only reads the
aggregates score_sir4.py wrote, so the numbers are exactly the ones that
scorer printed.

Usage
-----
    python3 eval/table91_row.py --name "Handcrafted scorer + OpenIE graph (v1, ep20)"                                 --scores results/v1_fusion_openie/scores.json
    python3 eval/table91_row.py --name A --scores a.json --name B --scores b.json
"""
from __future__ import annotations

import argparse
import json


def row(name: str, scores: dict) -> str:
    same, cross = scores["same"], scores["cross"]
    r5s, r5c = same["recall@5"], cross["recall@5"]
    gap = (r5c - r5s) / r5s * 100 if r5s else float("nan")
    cells = [same["recall@3"], cross["recall@3"],
             r5s, r5c,
             same["ndcg@5"], cross["ndcg@5"]]
    return (f"| {name} | " + " | ".join(f"{100 * c:.2f}" for c in cells)
            + f" | {gap:+.2f}% |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--scores", action="append", required=True,
                    help="score_sir4.py --json-out file, one per --name, same order")
    a = ap.parse_args()
    if len(a.name) != len(a.scores):
        ap.error("give one --scores per --name")

    print("| Method | R@3 same | R@3 cross | R@5 same | R@5 cross "
          "| nDCG@5 same | nDCG@5 cross | dR@5 gap |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|")
    for nm, p in zip(a.name, a.scores):
        s = json.load(open(p))
        for k in ("same", "cross"):
            if k not in s:
                raise SystemExit(f"{p}: no `{k}` slice; was --queries the TOMATO "
                                 f"test.json with `stratum` on every row?")
        print(row(nm, s))
        print(f"    n: same {s['same']['n']}, cross {s['cross']['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
