"""
pick_qualitative.py -- choose the worked examples for the qualitative section.

A good example for a reviewer is a CROSS-FIELD query whose gold the dense retrievers bury
and SciGraphIR ranks at the top. This script reads the saved rankings of every system on one
field's test set, computes each system's best-gold rank per query, and lists the queries
that satisfy: primary rank <= --max-primary-rank, every --against system's rank >= that
plus --min-gap. Ordered by the smallest against-rank (the hardest case for the baselines
that SciGraphIR still solves). Nothing here touches a model; it only reads prediction files.

    python3 eval/pick_qualitative.py --queries .../raw/test.json --docs .../raw/documents.json \\
        --pred scigraphir=/p/predictions_sir4_biology_test_v16sc.json \\
        --pred openie=/p/predictions_sir4_biology_test.json \\
        --pred qwen3=/p/predictions_qwen3_sir4_biology_test.json --pred bge=... \\
        --primary scigraphir --against qwen3,bge --stratum cross --top 12 --out picks.json
"""
from __future__ import annotations

import argparse
import json
import os

BIG = 10 ** 6


def ranked_docs(rec: dict) -> list[str]:
    p = rec.get("predictions", rec)
    docs = p.get("document", p) if isinstance(p, dict) else p
    return [d[0] if isinstance(d, (list, tuple)) else d for d in docs]


def best_gold_rank(ranked: list[str], golds: set[str]) -> tuple[int, str | None]:
    for i, d in enumerate(ranked, 1):
        if d in golds:
            return i, d
    return BIG, None


def title_of(text: str, n: int = 110) -> str:
    t = text.strip().split(". ")[0]
    return (t[:n] + "...") if len(t) > n else t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--pred", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--primary", default="scigraphir")
    ap.add_argument("--against", default="qwen3,bge", help="systems that must rank the gold low")
    ap.add_argument("--prefer-worse", default="openie,control",
                    help="systems that should ALSO rank the gold below the primary when present (soft)")
    ap.add_argument("--stratum", default="cross", help="cross | same | any")
    ap.add_argument("--max-primary-rank", type=int, default=10)
    ap.add_argument("--min-gap", type=int, default=10)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--channels", default=None,
                    help="scan output of interpret_paths (paths=0): adds the model's OWN per-channel ranks")
    ap.add_argument("--rank-by", default="dense", choices=["dense", "graph"],
                    help="dense: gap between the best dense baseline and the primary; graph: gap between the "
                         "model's scorer channel and its fused ranking (what the graph moved)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    chan = {}
    if a.channels and os.path.exists(a.channels):
        for rec in json.load(open(a.channels)):
            best = min(rec["targets"], key=lambda t: t["rank"]["fused"]) if rec["targets"] else None
            if best: chan[rec["id"]] = {"fused": best["rank"]["fused"], "graph_ch": best["rank"]["graph"],
                                        "scorer_ch": best["rank"]["scorer"], "dense_ch": best["rank"]["dense"], "gold": best["doc"]}
        print(f"[pick] channel ranks for {len(chan)} queries from {a.channels}")

    queries = {q["id"]: q for q in json.load(open(a.queries))}
    docs = json.load(open(a.docs))
    preds = {}
    for spec in a.pred:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[pick] {name}: missing {path} (skipped)"); continue
        preds[name] = {r["id"]: ranked_docs(r) for r in json.load(open(path))}
        print(f"[pick] {name}: {len(preds[name])} queries")
    assert a.primary in preds, f"primary system {a.primary!r} has no predictions"
    against = [x for x in a.against.split(",") if x in preds]
    prefer = [x for x in a.prefer_worse.split(",") if x in preds]

    rows = []
    for qid, q in queries.items():
        if a.stratum != "any" and q.get("stratum") != a.stratum:
            continue
        golds = set(q["supporting_documents"])
        r = {}
        which = {}
        for name, table in preds.items():
            if qid in table:
                r[name], which[name] = best_gold_rank(table[qid], golds)
        if a.primary not in r or r[a.primary] > a.max_primary_rank:
            continue
        # gap = how much lower the BEST dense baseline ranks the gold than the primary system.
        # Every query with a positive gap is a candidate; --min-gap only marks the strong ones,
        # so the list is never empty when the primary beats the baselines anywhere.
        gap = (min(min(r.get(s, BIG), 301) for s in against) - r[a.primary]) if against else 0
        if gap <= 0:
            continue
        soft = sum(1 for s in prefer if r.get(s, BIG) > r[a.primary])
        gold = which[a.primary]
        c = chan.get(qid, {})
        ggap = (c["scorer_ch"] - c["fused"]) if c else None        # ranks the graph channel moved the gold
        if a.rank_by == "graph" and (ggap is None or ggap <= 0):
            continue
        for k in ("graph_ch", "scorer_ch", "dense_ch"):
            if c: r[k] = c[k]
        rows.append({"id": qid, "stratum": q.get("stratum"), "ranks": r, "gap": gap, "graph_gap": ggap,
                     "strong": gap >= a.min_gap, "prefer_worse": soft, "gold": gold,
                     "gold_title": title_of(docs.get(gold, "")), "n_golds": len(golds), "question": q["question"]})
    if a.rank_by == "graph":
        rows.sort(key=lambda x: (-x["graph_gap"], -x["gap"], x["ranks"][a.primary]))
    else:
        rows.sort(key=lambda x: (-x["gap"], -x["prefer_worse"], x["ranks"][a.primary]))
    names = [a.primary] + [n for n in preds if n != a.primary] + (["graph_ch", "scorer_ch", "dense_ch"] if chan else [])
    strong = sum(x["strong"] for x in rows)
    print(f"\n{len(rows)} candidates ({a.stratum} queries, {a.primary} rank <= {a.max_primary_rank}, "
          f"{'/'.join(against)} rank below it); {strong} with gap >= {a.min_gap}; "
          f"gap >= 5: {sum(x['gap'] >= 5 for x in rows)}, >= 20: {sum(x['gap'] >= 20 for x in rows)}. Top {a.top}:\n")
    print("| # | query id | dense gap | graph gap | " + " | ".join(names) + " | gold (title) | question |")
    print("|--:|---|--:|--:|" + "--:|" * len(names) + "---|---|")
    for i, x in enumerate(rows[:a.top], 1):
        rk = " | ".join(str(x["ranks"].get(n, "--")) if x["ranks"].get(n, BIG) < BIG else ">300" for n in names)
        print(f"| {i} | {x['id']} | {x['gap']}{'*' if x['strong'] else ''} | {x['graph_gap'] if x['graph_gap'] is not None else '--'} | {rk} | {x['gold_title'][:70]} | {x['question'][:90].replace('|', '/')}... |")
    if a.out:
        json.dump({"primary": a.primary, "against": against, "systems": names, "candidates": rows[:max(a.top, 50)]},
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
