#!/usr/bin/env python3
"""
walk_prior.py -- the parameter-free walk prior of the graph channel, computed locally for every dataset
and graph, per stratum.

Same construction as FusionSFTTrainer._prior_doc: symmetrise the stage-1 graph, row-normalise by degree,
start from the query's seed nodes (stage1 test.json "start_nodes", all types, uniform mass), run
T = 3 steps of x <- (1 - a) A x + a x0 with a = 0.15, rank the DOCUMENT nodes by the mass they hold.
Then nDCG@5 / R@5 / R@10 against the query's supporting_documents, split by stratum. The trainer prints
this only on the whole test set; here it is split by same / cross too.

    python3 eval/walk_prior.py [--data ../kg-construction/data] [--datasets sir4_cs,...] [--md out.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp

csv.field_size_limit(10 ** 9)
DATASETS = ["sir4_cs", "sir4_biology", "sir4_physics", "sir4_matsci", "tomato", "mir"]


def load_graph(stage1: str):
    names, types = [], []
    with open(f"{stage1}/nodes.csv", newline="") as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            names.append(row[0]); types.append(row[1])
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)
    src, dst = [], []
    with open(f"{stage1}/edges.csv", newline="") as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            a, b = idx.get(row[0]), idx.get(row[2])
            if a is not None and b is not None:
                src.append(a); dst.append(b)
    src, dst = np.array(src), np.array(dst)
    rr = np.concatenate([src, dst]); cc = np.concatenate([dst, src])          # symmetrised
    deg = np.bincount(rr, minlength=N).astype(np.float64)
    w = 1.0 / np.maximum(deg[rr], 1.0)
    A = sp.csr_matrix((w, (cc, rr)), shape=(N, N))                            # A[c, r] = 1/deg[r]: mass flows r -> c
    docs = np.array([i for i, t in enumerate(types) if t == "document"])
    return idx, A, docs, names


def ndcg5(ranked_docs: list, golds: set) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked_docs[:5]) if d in golds)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(golds), 5)))
    return dcg / idcg if idcg else 0.0


def walk(A, idx, docs, queries, T=3, alpha=0.15, chunk=512):
    N = A.shape[0]
    out = []
    for c0 in range(0, len(queries), chunk):
        qs = queries[c0:c0 + chunk]
        X0 = np.zeros((N, len(qs)), dtype=np.float32)
        for j, q in enumerate(qs):
            seeds = [idx[n] for v in q["start_nodes"].values() for n in v if n in idx]
            if seeds:
                X0[seeds, j] = 1.0 / len(seeds)
        X = X0.copy()
        for _ in range(T):
            X = (1.0 - alpha) * (A @ X) + alpha * X0
        M = X[docs]                                                            # [n_docs, q]
        for j, q in enumerate(qs):
            has_seed = X0[:, j].sum() > 0
            top = docs[np.argsort(-M[:, j], kind="stable")[:10]]
            out.append((has_seed, top))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kg-construction", "data"))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    lines = ["| dataset | graph | stratum | queries | seeded | walk nDCG@5 | R@5 | R@10 |", "|---|---|---|--:|--:|--:|--:|--:|"]
    for ds in a.datasets.split(","):
        for g, label in ((f"{ds}_test_v16sc", "SciAffordGraph"), (f"{ds}_test", "OpenIE")):
            stage1 = f"{a.data}/{g}/processed/stage1"
            if not os.path.exists(f"{stage1}/test.json"):
                print(f"[skip] {g}: no stage1/test.json", file=sys.stderr); continue
            idx, A, docs, names = load_graph(stage1)
            queries = json.load(open(f"{stage1}/test.json"))
            res = walk(A, idx, docs, queries)
            per = []
            for q, (seeded, top) in zip(queries, res):
                golds = set(q["supporting_documents"]) if isinstance(q["supporting_documents"], list) else set(eval(q["supporting_documents"]))
                ranked = [names[i] for i in top]
                per.append((q.get("stratum") or "same", seeded, ndcg5(ranked, golds),
                            any(d in golds for d in ranked[:5]), any(d in golds for d in ranked[:10])))
            strata = ["all"] + sorted({p[0] for p in per}, key=lambda s: (s != "same", s))
            if len(strata) == 2: strata = ["all"]
            for s in strata:
                sub = [p for p in per if s == "all" or p[0] == s]
                lines.append(f"| {ds} | {label} | {s} | {len(sub)} | {100 * np.mean([p[1] for p in sub]):.1f}% | "
                             f"{100 * np.mean([p[2] for p in sub]):.2f} | {100 * np.mean([p[3] for p in sub]):.1f} | {100 * np.mean([p[4] for p in sub]):.1f} |")
            print(f"{g}: {len(queries)} queries, {len(docs)} docs, {A.nnz // 2} edges", file=sys.stderr)
    lines.append("")
    lines.append("walk = 3-step personalised PageRank (restart 0.15) from the query's seed nodes over the symmetrised, degree-normalised "
                 "stage-1 graph, documents ranked by mass; no parameters. R@k = share of queries with a gold in the top k "
                 "(any-gold). seeded = queries with at least one seed node in the graph; unseeded queries score 0.")
    text = "\n".join(lines); print(text)
    if a.md: open(a.md, "w").write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
