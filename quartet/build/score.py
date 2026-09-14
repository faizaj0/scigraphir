#!/usr/bin/env python3
"""Score a retriever against QUARTET, with the family of valid decompositions.

    python3 build/score.py --benchmark data/benchmark/cs_test_pilot200
    python3 build/score.py --benchmark <dir> --predictions preds.json

With no --predictions this runs BM25 over the benchmark's own corpus, so the
numbers are reproducible from the released files alone and a new split can be
sanity-checked before any model is involved.

THE METRICS, AND WHY THERE ARE THREE.

A QUARTET query does not have a gold LIST, it has a family of gold SETS: several
decompositions, each independently validated as sufficient to explain the same
hypothesis. Three questions follow, and they are not the same question.

  PrimarySet@k    did the top k contain the WHOLE of D1?
                      1[ D1 subset of TopK ]
                  The flat TOMATO-Star reading, made set-shaped.

  CompleteSet@k   did the top k contain a whole valid decomposition, ANY of them?
                      1[ exists S in M : S subset of TopK ]
                  The headline. Recovering two of three inspirations explains
                  nothing, and only a set-completion metric says so.

  CompleteSet@k - PrimarySet@k    is the alternative credit, and it is
                  NON-NEGATIVE BY CONSTRUCTION, because D1 is a member of M.
                  It is exactly the share of queries where a retriever recovered
                  a valid explanation that the single-gold benchmark calls a
                  failure. That number is the thesis result.

  UnionRecall@k   the fraction of the union of all valid decompositions inside
                  the top k. Reported for continuity with ordinary recall, and
                  NEVER differenced against PrimarySet@k.

WHY THE EARLIER PAIR WAS WRONG. The first version reported `Any@k`, described as
"did the top k contain any member", but implemented as
`|union & TopK| / |union|`: fractional recall over the union, not an indicator.
Differencing it against recall over D1 compares two fractions with different
denominators, so "alternative credit" came out NEGATIVE (-10.9 at k=1) whenever
the union was larger than D1, which is whenever there are alternatives at all.
The quantity was meaningless and its sign was backwards. Both members of the
headline pair are now indicators over the same k, so the comparison is sound.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

KS = (1, 5, 10, 25, 50, 100)


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


# ------------------------------------------------------------------------ bm25
class BM25:
    def __init__(self, docs: dict, k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs)
        self.tok = {d: norm(t).split() for d, t in docs.items()}
        self.n = len(self.tok) or 1
        self.avg = sum(len(t) for t in self.tok.values()) / self.n
        df = Counter()
        for t in self.tok.values():
            df.update(set(t))
        self.idf = {w: math.log(1 + (self.n - c + 0.5) / (c + 0.5))
                    for w, c in df.items()}
        self.tf = {d: Counter(t) for d, t in self.tok.items()}
        self.k1, self.b = k1, b

    def rank(self, q: str) -> list:
        qt = norm(q).split()
        out = []
        for d, c in self.tf.items():
            L = len(self.tok[d]) or 1
            s = 0.0
            for w in qt:
                f = c.get(w, 0)
                if f:
                    s += self.idf.get(w, 0) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append((s, d))
        out.sort(key=lambda p: -p[0])
        return [d for _, d in out]


# ----------------------------------------------------------------------- score
def complete_at(sets: list, ranked: list, k: int) -> int:
    """1 if any whole valid decomposition sits inside the top k."""
    top = set(ranked[:k])
    return int(any(set(s) <= top for s in sets if s))


def set_at(one: set, ranked: list, k: int) -> int:
    """1 if this ONE set sits entirely inside the top k."""
    return int(bool(one) and one <= set(ranked[:k]))


def recall_at(golds: set, ranked: list, k: int) -> float:
    if not golds:
        return 0.0
    return len(golds & set(ranked[:k])) / len(golds)


def ndcg_at(golds: set, ranked: list, k: int) -> float:
    if not golds:
        return 0.0
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in golds)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(golds), k)))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--predictions", type=Path,
                    help="{query_id: [doc_id, ...]} ranked best first. "
                         "Omitted, BM25 over the benchmark's own corpus is used")
    ap.add_argument("--slice", default="stratum",
                    choices=["stratum", "domain_distance", "none"])
    ap.add_argument("--strict-strata", action="store_true",
                    help="also exclude queries whose primary golds are only "
                         "PARTLY labelled. A query with one 'same' gold and one "
                         "unlabelled gold is not known to be same-domain")
    ap.add_argument("--out", type=Path, help="write the numbers as JSON")
    a = ap.parse_args()

    ev = json.loads((a.benchmark / "eval.json").read_text())
    docs = json.loads((a.benchmark / "raw" / "documents.json").read_text())
    if not ev or "quartet" not in ev[0] or "sets" not in ev[0].get("quartet", {}):
        sys.exit(f"{a.benchmark / 'eval.json'} is not the set-aware file. "
                 f"Re-export with the current build/05_export.py.")
    prim_path = a.benchmark / "eval_primary.json"
    primary = defaultdict(set)
    if prim_path.exists():
        for r in json.loads(prim_path.read_text()):
            primary[r["id"].rsplit("::", 1)[0]].update(r["supporting_documents"])

    if a.predictions:
        preds = json.loads(a.predictions.read_text())
        if isinstance(preds, list):        # a predictions-style list of rows
            preds = {r["id"]: [d for d, *_ in r.get("predictions", [])]
                     if r.get("predictions") and isinstance(r["predictions"][0], list)
                     else r.get("predictions", []) for r in preds}
        run = "predictions"
    else:
        bm = BM25(docs)
        preds = {r["id"]: bm.rank(r["question"]) for r in ev}
        run = "BM25"

    buckets: dict = defaultdict(lambda: defaultdict(list))
    missing = 0
    for r in ev:
        ranked = preds.get(r["id"])
        if ranked is None:
            missing += 1
            continue
        sets = [s for s in r["quartet"]["sets"] if s]
        union = set(r["supporting_documents"])
        pset = primary.get(r["id"]) or set(r["quartet"]["primary_set"])
        keys = ["all"]
        if a.slice == "stratum":
            # A row-level `stratum` with no evidence behind it is a compatibility
            # placeholder Stage 5 has to emit, not a measurement. Counting those
            # in the `same` bucket biases every same/cross comparison towards
            # same, by exactly the share of golds OpenAlex could not label.
            # Measured on the first export: 15 of 158 primary golds, 9.5%,
            # affecting 8 of 66 queries. They get their own bucket instead.
            q = r.get("quartet", {})
            labelled = q.get("stratum_labelled")
            if labelled is None:                 # exported before the flag
                labelled = bool(r.get("stratum"))
            if a.strict_strata and not q.get("stratum_complete", True):
                labelled = False
            keys.append(r.get("stratum") or "unlabelled" if labelled
                        else "unlabelled")
        elif a.slice == "domain_distance":
            bands = {m.get("domain_distance") for m in
                     r["quartet"]["per_document"].values() if m.get("domain_distance")}
            keys += sorted(bands)
        for kx in keys:
            for k in KS:
                buckets[kx][f"CompleteSet@{k}"].append(complete_at(sets, ranked, k))
                buckets[kx][f"PrimarySet@{k}"].append(set_at(pset, ranked, k))
                buckets[kx][f"UnionRecall@{k}"].append(recall_at(union, ranked, k))
            buckets[kx]["nDCG@10"].append(ndcg_at(union, ranked, 10))
            buckets[kx]["n_sets"].append(len(sets))

    print(f"\n{'=' * 78}\n{run} over {len(docs)} documents, {len(ev)} queries"
          + (f", {missing} with no prediction" if missing else ""))
    if len(docs) < 1000:
        print(f"  NOTE: {len(docs)} documents is small. Random Any@100 is "
              f"{min(100, 100 * 100 // len(docs))}%, so deep-recall numbers here "
              f"are not comparable\n  to a full-scale corpus. Ordering across "
              f"slices is the readable part.")
    n_unlab = len(buckets.get("unlabelled", {}).get("nDCG@10", []))
    if n_unlab and a.slice == "stratum":
        print(f"  {n_unlab} of {len(ev)} queries have no domain evidence and are "
              f"reported ONLY under 'unlabelled'.\n  They are NOT in 'same'"
              + (", and partly-labelled queries are excluded too (--strict-strata)."
                 if a.strict_strata else
                 ". Add --strict-strata to also drop partly-labelled ones."))
    for kx in sorted(buckets, key=lambda x: (x != "all", x == "unlabelled", x)):
        b = buckets[kx]
        n = len(b["nDCG@10"])
        print(f"\n  {kx}   n={n}   mean valid sets/query "
              f"{sum(b['n_sets']) / max(n, 1):.2f}")
        # CompleteSet and PrimarySet are the same indicator over M and over {D1},
        # and D1 is in M, so the difference cannot be negative. It is the share
        # of queries where a valid explanation was recovered that the single-gold
        # reading scores as a failure.
        print(f"    {'k':<6}{'CompleteSet':>13}{'PrimarySet':>13}"
              f"{'  alt credit':>14}{'UnionRecall':>13}")
        for k in KS:
            c = 100 * sum(b[f"CompleteSet@{k}"]) / max(n, 1)
            p = 100 * sum(b[f"PrimarySet@{k}"]) / max(n, 1)
            u = 100 * sum(b[f"UnionRecall@{k}"]) / max(n, 1)
            assert c + 1e-9 >= p, "CompleteSet < PrimarySet: D1 is not in M"
            print(f"    {k:<6}{c:>12.1f}%{p:>12.1f}%{c - p:>+13.1f}{u:>12.1f}%")
        print(f"    nDCG@10 (union)  {100 * sum(b['nDCG@10']) / max(n, 1):.1f}")

    if a.out:
        a.out.write_text(json.dumps(
            {kx: {m: (sum(v) / len(v) if v else 0.0) for m, v in b.items()}
             for kx, b in buckets.items()}, indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
