"""
score_sir4.py -- retrieval metrics for SIR-4, which is MULTI-GOLD.

WHY THIS EXISTS. Every eval cell inherited from the TOMATO notebooks does:

    gold = r["supporting_documents"]
    gd   = gold[0] if isinstance(gold, list) else gold     # <-- one gold
    rank = ranked.index(gd) + 1

TOMATO carries exactly one gold per row, so that is correct there. SIR-4 carries
a mean of 4.45 golds per query (range 1 to 10) on a single row, so the inherited
code silently scores against the first of four-and-a-half. It does not crash and
the number it prints looks reasonable, which is what makes it dangerous.

METRICS (as specified for the CS run):

  MRR            reciprocal rank of the HIGHEST-ranked labelled positive.
                 "Did the retriever surface anything useful, and how fast."

  MGRR           mean reciprocal rank over ALL labelled positives.
                 "How highly did it rank the complete flat gold list."

  Recall@k       fraction of ALL labelled positives inside the top k.
                 "How much of the decomposition did it find."

  CompleteSet@k  1 if at least one complete set from sets.json is a subset of
                 the top k, else 0. "Did it find a WHOLE working explanation."

CompleteSet@k is the SIR-4-specific metric. It is only definable because SIR-4
records every validated decomposition, not just one; no other benchmark in the
MOOSE lineage carries the alternative sets. It is also strictly harder than
Recall@k: a run can retrieve 6 of 7 golds and still score 0, because six pieces
of two different explanations do not make one explanation.

SLICES. `same` / `cross` come from the query row. `similar` / `dissimilar` need a
dense baseline: a query is `dissimilar` when plain BGE ranks its BEST gold below
100. Best-gold, not first-gold, so the slice definition is consistent with MRR.
Without --bge those two slices are skipped rather than guessed.

Usage
-----
    python3 eval/score_sir4.py --pred preds.json --queries .../test.json \\
                               --sets .../sets.json --bge predictions_bge.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

# 15 is here for ResearchBench: it retains 20% of a 75-candidate pool, so Recall@15
# is the cutoff its published numbers are reported at. Costs nothing to compute and
# nothing downstream reads KS positionally.
# 25 is here for the semantic-scorer comparison, whose reporting set spans
# Recall@3 to Recall@100. Adding a cutoff is safe: nothing downstream reads KS
# positionally, and every consumer selects metrics by name or via --cols.
KS = (1, 2, 3, 5, 10, 15, 20, 25, 50, 100)
BIG = 10 ** 9


def ranked_docs(rec: dict) -> list[str]:
    """Ranked document ids from a prediction record, tolerating both shapes."""
    p = rec.get("predictions", rec)
    docs = p.get("document", p) if isinstance(p, dict) else p
    out = []
    for d in docs:
        out.append(d[0] if isinstance(d, (list, tuple)) else d)
    return out


def best_gold_rank(ranked: list[str], golds: set[str]) -> int:
    for i, d in enumerate(ranked, 1):
        if d in golds:
            return i
    return BIG


def score_one(ranked: list[str], golds: set[str], sets_: list[list[str]]) -> dict:
    r = best_gold_rank(ranked, golds)
    # Standard query-level MRR uses the first relevant result.  The previous version
    # stored the mean reciprocal rank over every gold under `mrr`; that is a useful
    # multi-gold diagnostic, but it is not conventional MRR.  Preserve it explicitly
    # as MGRR so old values remain reproducible without overloading a standard name.
    #
    # A gold outside the truncated prediction list contributes 0 rather than 1/rank.
    # With top-300 of 20,322 the difference is below the fourth decimal, and pretending
    # to know a rank the file does not contain would be worse.
    #
    # `mrr_best` is retained as a deprecated alias for compatibility with any files
    # or notebooks that already requested it explicitly.
    pos = {d: i + 1 for i, d in enumerate(ranked)}
    ranks = [pos.get(g, BIG) for g in golds]
    standard_mrr = 1.0 / r if r < BIG else 0.0
    mean_gold_rr = ((sum(1.0 / x for x in ranks if x < BIG) / len(golds))
                    if golds else 0.0)
    # Average precision over the flat gold set: mean over golds of precision at that gold's
    # rank, a gold outside the list contributing 0. Aggregated over queries this is mAP,
    # the metric MIR (ACL 2025) reports, so its published rows line up with ours.
    hits, ap = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in golds:
            hits += 1
            ap += hits / i
    out = {
        "mrr": standard_mrr,
        "mgrr": mean_gold_rr,
        "mrr_best": standard_mrr,
        "map": ap / len(golds) if golds else 0.0,
    }
    for k in KS:
        topk = set(ranked[:k])
        # Recall over ALL positives, not just the first one.
        out[f"recall@{k}"] = len(topk & golds) / len(golds) if golds else 0.0
        out[f"hits@{k}"] = float(r <= k)
        # A whole validated set inside the top k. Sets larger than k can never
        # qualify, which is correct: you cannot fit 7 papers into a top-5.
        out[f"completeset@{k}"] = float(any(set(s) <= topk for s in sets_)) if sets_ else 0.0

    # Binary-relevance nDCG@5 over the flat set of labelled inspiration papers.
    # CompleteSet@5 below remains the dependency-aware companion metric.
    rel = [float(d in golds) for d in ranked[:5]]
    dcg = sum(v / math.log2(i + 2) for i, v in enumerate(rel))
    ideal_n = min(len(golds), 5)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    out["ndcg@5"] = dcg / idcg if idcg else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions json (list of records)")
    ap.add_argument("--queries", required=True, help="the split's {split}.json")
    ap.add_argument("--sets", default=None, help="sets.json from the SIR-4 export (for CompleteSet@k)")
    ap.add_argument("--bge", default=None, help="dense baseline predictions, for the similar/dissimilar split")
    ap.add_argument("--name", default="run")
    ap.add_argument("--json-out", default=None)
    # PER-QUERY OUTPUT EXISTS FOR PAIRED TESTS. The aggregate below reduces every
    # slice to a mean, and two means cannot be compared with a paired test no
    # matter how the arms were run. When two arms rank the SAME queries over the
    # SAME candidate graphs -- which is the whole design of the ResearchBench
    # transfer experiment -- an unpaired z-test throws away the pairing and is
    # strictly weaker. Dump the per-query rows and bootstrap the difference.
    ap.add_argument("--per-query-out", default=None,
                    help="write {query_id: {metric: value, 'stratum': ...}} before aggregation")
    # PER-GOLD STRATA EXIST BECAUSE RESEARCHBENCH LABELS GOLDS, NOT QUERIES.
    # SIR-4 carries one `stratum` per query, so the slice table below works. On
    # ResearchBench a query has a LIST of per-gold labels (`gold_strata`,
    # positionally aligned with supporting_documents): one query can contribute a
    # same-field gold and a cross-field gold at once. Collapsing that to a query
    # label would have to pick a rule -- any-cross? all-cross? majority? -- and
    # every rule discards information the benchmark actually provides. Instead
    # answer the question the labels support directly: of the labelled cross-field
    # golds, what fraction were retrieved by rank k. That is a different
    # denominator from the slice table and is reported separately.
    ap.add_argument("--gold-strata", action="store_true",
                    help="also report recall by PER-GOLD stratum (needs gold_strata on each query)")
    # The default column set is the one the SIR-4 results notebook reads by name.
    # ResearchBench wants a different set (Recall@3 and @15 are ITS operating points,
    # at 4% and 20% of a 75-candidate pool), and quietly widening the default would
    # change every SIR-4 table already produced. Let the caller ask instead.
    ap.add_argument("--cols", default=None,
                    help="comma-separated metric keys to aggregate; default is the SIR-4 set")
    a = ap.parse_args()

    queries = {q["id"]: q for q in json.load(open(a.queries))}
    preds = json.load(open(a.pred))
    if isinstance(preds, dict):
        preds = list(preds.values())

    sets_by_q: dict[str, list[list[str]]] = {}
    if a.sets:
        raw = json.load(open(a.sets))
        for qid, v in raw.items():
            sets_by_q[qid] = [list(s) for s in (v.get("sets") if isinstance(v, dict) else v)]

    bge_rank: dict[str, int] = {}
    if a.bge:
        for rec in json.load(open(a.bge)):
            q = queries.get(rec["id"])
            if q:
                bge_rank[rec["id"]] = best_gold_rank(ranked_docs(rec), set(q["supporting_documents"]))

    slices: dict[str, list[dict]] = defaultdict(list)
    per_query: dict[str, dict] = {}
    gold_rows: list[tuple[str, int]] = []          # (stratum, rank) one per gold
    seen, missing = set(), 0
    for rec in preds:
        qid = rec.get("id")
        q = queries.get(qid)
        if q is None or qid in seen:
            missing += q is None
            continue
        seen.add(qid)
        golds = set(q["supporting_documents"])
        m = score_one(ranked_docs(rec), golds, sets_by_q.get(qid, []))
        # SIR-4 has one query-level `stratum`.  ResearchBench has per-gold
        # labels plus the conservative derived `domain_slice` written by the
        # corrected pool builder.  Map the latter only for the query-level
        # table; the per-gold table below keeps the original aligned labels.
        query_stratum = q.get("stratum")
        if not query_stratum:
            query_stratum = {
                "same_only": "same",
                "cross_involved": "cross",
            }.get(q.get("domain_slice"))
        names = ["all"] + ([query_stratum] if query_stratum else [])
        if qid in bge_rank:
            sim = "dissimilar" if bge_rank[qid] > 100 else "similar"
            names.append(sim)
            if query_stratum:
                names.append(f"{query_stratum}+{sim}")
        for n in names:
            slices[n].append(m)
        if a.gold_strata:
            # One row per (query, gold). `ranked.index` is O(n) but this runs once
            # per gold over a list already in memory, and clarity beats a rank map
            # that would have to be kept in sync with ranked_docs().
            order = ranked_docs(rec)
            pos = {d: i + 1 for i, d in enumerate(order)}
            strata = q.get("gold_strata") or []
            for i, g in enumerate(q["supporting_documents"]):
                gold_rows.append((strata[i] if i < len(strata) else "unlabelled",
                                  pos.get(g, BIG)))
        if a.per_query_out is not None:
            # `slices` is carried alongside so a paired test can be restricted to a
            # slice (cross-field only, one discipline) without re-deriving which
            # queries belong to it. `discipline` is ResearchBench-only and absent
            # on SIR-4; recorded when present rather than required.
            per_query[qid] = {**m, "slices": names,
                              "n_golds": len(golds),
                              **({"domain_slice": q["domain_slice"]}
                                 if "domain_slice" in q else {}),
                              **({"discipline": q["discipline"]} if "discipline" in q else {})}

    order = ["all", "same", "cross", "similar", "dissimilar", "same+dissimilar", "cross+dissimilar"]
    cols = ["mrr", "ndcg@5", "recall@3", "recall@5", "completeset@5"]
    if a.cols:
        cols = [c.strip() for c in a.cols.split(",") if c.strip()]
        probe = score_one([], set(), [])          # the metric keys score_one emits
        unknown = [c for c in cols if c not in probe]
        if unknown:
            return print(f"unknown metric(s) {unknown}; available: "
                         f"{sorted(probe)}") or 2
    print(f"\n### {a.name}   ({len(seen)} queries scored"
          + (f", {missing} predictions had no matching query" if missing else "") + ")")
    if not a.sets:
        print("    CompleteSet@k is 0 everywhere: --sets not given")
    if not a.bge:
        print("    similar/dissimilar slices skipped: --bge not given")
    print("| slice | n | " + " | ".join(cols) + " |")
    print("|---|--:|" + "--:|" * len(cols))
    result = {}
    for nm in order:
        rows = slices.get(nm)
        if not rows:
            continue
        agg = {c: sum(r[c] for r in rows) / len(rows) for c in cols}
        result[nm] = {"n": len(rows), **agg}
        print(f"| {nm} | {len(rows)} | " + " | ".join(f"{agg[c]:.4f}" for c in cols) + " |")

    if a.gold_strata:
        if not gold_rows:
            print("\n--gold-strata: no gold_strata on any query; nothing to report")
        else:
            by = defaultdict(list)
            for st, r in gold_rows:
                by[st].append(r)
                by["all"].append(r)
            gk = [1, 5, 10, 20, 100]
            print(f"\n### per-GOLD recall by stratum   "
                  f"(denominator is golds, not queries; {len(gold_rows):,} golds)")
            print("| stratum | golds | " + " | ".join(f"R@{k}" for k in gk) + " | median rank |")
            print("|---|--:|" + "--:|" * (len(gk) + 1))
            gs_out = {}
            for st in ("all", "same", "cross", "ambiguous", "unmatched",
                       "unknown", "unlabelled"):
                rs = by.get(st)
                if not rs:
                    continue
                row = {f"recall@{k}": sum(r <= k for r in rs) / len(rs) for k in gk}
                med = sorted(rs)[len(rs) // 2]
                gs_out[st] = {"n_golds": len(rs), **row,
                              "median_rank": None if med >= BIG else med}
                print(f"| {st} | {len(rs)} | "
                      + " | ".join(f"{row[f'recall@{k}']:.4f}" for k in gk)
                      + f" | {'>corpus' if med >= BIG else med} |")
            result["_per_gold"] = gs_out
            print("NOTE: this table's rows are golds, so it does NOT sum to the "
                  "query-level table above.")

    if a.json_out:
        json.dump(result, open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")
    if a.per_query_out:
        json.dump(per_query, open(a.per_query_out, "w"), indent=1)
        print(f"wrote {a.per_query_out}  ({len(per_query)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
