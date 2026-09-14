"""
paired_bootstrap.py -- paired comparison of two arms over the same queries.

WHY PAIRED. Both arms rank the same 1,367 queries over byte-identical cached
candidate graphs. That pairing is the most valuable property of the design and an
unpaired test throws it away: query difficulty varies enormously here (2 to 6 golds,
12 disciplines), and unpaired tests charge that variance against the difference you
care about. The paired bootstrap resamples QUERIES, so per-query difficulty cancels.

WHY BOOTSTRAP RATHER THAN A t-TEST. MRR, recall@k and completeset@k are bounded,
heavily tied and nowhere near normal -- completeset@k is 0/1, recall@k takes about
four distinct values at 2.35 golds per query. A bootstrap makes no distributional
claim; it just resamples.

NULL RESULTS ARE RESULTS. This prints the CI whether or not it crosses zero, and says
so in words. An interval spanning zero after 1,367 paired queries is a finding about
the size of the effect, not a failed run to be re-cut until it clears.

Usage
-----
    python3 transfer/paired_bootstrap.py \
        --a out/sir4_biology_perquery.json --a-name "SIR-4 Biology" \
        --b out/tomato_perquery.json       --b-name "TOMATO-Star" \
        --slice all --metrics mrr,recall@5,completeset@5
    # restrict to the predeclared strict zero-shot subset
    python3 transfer/paired_bootstrap.py ... --exclude-disciplines Biology,"Cell Biology",Chemistry
"""
from __future__ import annotations

import argparse
import json
import os
import random

# SUPERSEDED, kept only so `--exclude-disciplines strict` reproduces the original
# definition for the write-up's comparison. Dropping ResearchBench's Biology, Cell
# Biology and Chemistry buckets leaves 990 queries, but 177 of them read biomedical:
# the buckets do not describe content (Math is 29% biomedical, Law 27%). The live
# definition is the frozen `strict_zeroshot` list in subsets.json, written by
# transfer/declare_subsets.py before any model was run. Use --subset.
STRICT_EXCLUDE = ("Biology", "Cell Biology", "Chemistry")
SUBSETS = "kg-construction/data/researchbench_test/subsets.json"


def load(path: str) -> dict:
    d = json.load(open(path))
    if not isinstance(d, dict):
        raise SystemExit(f"{path}: expected the --per-query-out mapping, got {type(d).__name__}")
    legacy = 0
    for qid, row in d.items():
        # score_sir4.py historically wrote mean gold reciprocal rank as `mrr`,
        # but also retained conventional first-relevant MRR as `mrr_best`.
        # Normalise old files in memory so a disconnected Colab runtime can reuse
        # its saved per-query results without rerunning model inference.
        if "mgrr" not in row and "mrr_best" in row:
            row["mgrr"] = row.get("mrr", 0.0)
            row["mrr"] = row["mrr_best"]
            legacy += 1
        if "mrr_best" in row and abs(row.get("mrr", 0.0) - row["mrr_best"]) > 1e-12:
            raise SystemExit(f"{path}: query {qid!r} has ambiguous MRR semantics")
    if legacy:
        print(f"{path}: normalised standard MRR from mrr_best for "
              f"{legacy:,} legacy rows")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="per-query json for arm A (the claim)")
    ap.add_argument("--b", required=True, help="per-query json for arm B (the comparator)")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--metrics", default="mrr,hits@1,hits@5,recall@1,recall@5,"
                                         "recall@10,recall@20,completeset@5,completeset@10")
    ap.add_argument("--slice", default="all",
                    help="restrict to queries whose `slices` list contains this")
    ap.add_argument("--subset", default=None,
                    help="name of a frozen subset in subsets.json, e.g. strict_zeroshot")
    ap.add_argument("--subsets-file", default=None)
    ap.add_argument("--exclude-disciplines", default=None,
                    help=f"SUPERSEDED, for comparison only; 'strict' = drop {list(STRICT_EXCLUDE)}")
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)

    # An arm missing queries is a silent denominator change, which is the exact bug
    # class this experiment's validity checks exist to catch. Refuse rather than
    # quietly intersect.
    only_a, only_b = set(A) - set(B), set(B) - set(A)
    if only_a or only_b:
        raise SystemExit(
            f"arms cover different queries: {len(only_a)} only in {a.a_name}, "
            f"{len(only_b)} only in {a.b_name}. Both arms must score every query "
            f"(zero-seed queries included, never dropped).")

    ex, keep, label = (), None, "all queries"
    if a.subset:
        # Read the frozen list. Never recompute it here: a subset that can be
        # recomputed at report time is a subset that can be tuned at report time.
        here = os.path.dirname(os.path.abspath(__file__))
        path = a.subsets_file or os.path.join(
            os.path.dirname(os.path.dirname(here)), SUBSETS)
        if not os.path.exists(path):
            raise SystemExit(f"no frozen subsets at {path}; "
                             f"run transfer/declare_subsets.py first")
        subs = json.load(open(path))["subsets"]
        if a.subset not in subs:
            raise SystemExit(f"{a.subset!r} is not declared; have {list(subs)}")
        keep, label = set(subs[a.subset]), f"subset={a.subset}"
    elif a.exclude_disciplines:
        ex = (STRICT_EXCLUDE if a.exclude_disciplines.strip().lower() == "strict"
              else tuple(s.strip() for s in a.exclude_disciplines.split(",")))
        label = f"excluding {list(ex)}"

    qids = [q for q in sorted(A)
            if a.slice in A[q].get("slices", ["all"])
            and A[q].get("discipline") not in ex
            and (keep is None or q in keep)]
    if not qids:
        raise SystemExit(f"no queries left after slice={a.slice!r} {label}")
    if keep is not None and len(qids) < len(keep):
        print(f"note: {len(keep) - len(qids)} of the subset's queries are absent "
              f"from the scored arms (slice filter, or never scored)\n")

    metrics = [m.strip() for m in a.metrics.split(",")]
    metrics = [m for m in metrics if m in A[qids[0]] and m in B[qids[0]]]

    print(f"\n### {a.a_name}  vs  {a.b_name}")
    print(f"slice={a.slice}  {label}  n={len(qids)}  bootstrap={a.iters:,}\n")
    print(f"| metric | {a.a_name} | {a.b_name} | diff | 95% CI | verdict |")
    print("|---|--:|--:|--:|---|---|")

    rng = random.Random(a.seed)
    # One resampled index set per iteration, SHARED across metrics, so the columns
    # of the table are mutually consistent rather than each telling its own story.
    draws = [[rng.randrange(len(qids)) for _ in range(len(qids))] for _ in range(a.iters)]

    out = {}
    for m in metrics:
        da = [A[q][m] for q in qids]
        db = [B[q][m] for q in qids]
        diffs = [x - y for x, y in zip(da, db)]
        obs = sum(diffs) / len(diffs)
        boot = sorted(sum(diffs[i] for i in idx) / len(idx) for idx in draws)
        lo, hi = boot[int(.025 * a.iters)], boot[int(.975 * a.iters)]
        crosses = lo <= 0 <= hi
        verdict = "no difference" if crosses else (
            f"**{a.a_name} better**" if obs > 0 else f"**{a.b_name} better**")
        print(f"| {m} | {sum(da)/len(da):.4f} | {sum(db)/len(db):.4f} | "
              f"{obs:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {verdict} |")
        out[m] = {"a": sum(da) / len(da), "b": sum(db) / len(db), "diff": obs,
                  "ci95": [lo, hi], "crosses_zero": crosses, "n": len(qids)}

    print(f"\npaired bootstrap over {len(qids)} queries, resampling queries.")
    print("A CI spanning zero means the difference is not resolvable at this sample "
          "size. That is a result; report it.")
    if a.json_out:
        json.dump({"a_name": a.a_name, "b_name": a.b_name, "slice": a.slice,
                   "subset": a.subset, "excluded": list(ex), "n": len(qids), "iters": a.iters,
                   "metrics": out}, open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
