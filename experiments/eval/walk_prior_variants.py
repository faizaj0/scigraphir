#!/usr/bin/env python3
"""
walk_prior_variants.py -- cheap construction levers for the frame graph, judged by the same untrained walk.

Variants (all on the stage-1 test graph, no training):
  base        SciAffordGraph as built
  direct      + one-hop paper->frame edges for the mechanism frames a paper already owns through its
              methods / tasks / findings (contributes->achieves, ->overcomes, ->limited_by, ->works_via;
              addresses->limited_by; reports->concerns / ->explains). Makes function/limitation seeds
              1 hop from papers instead of 2.
  hybrid      + the OpenIE graph's entity->paper mention edges, entities with OpenIE degree <= CAP only
              (specific names, no hubs), and the query's OpenIE entity seeds added to its frame seeds.
  invdeg      restart mass per seed proportional to 1/degree instead of uniform (specific seeds count more).
  direct+hybrid, direct+hybrid+invdeg
Prints walk nDCG@5 per dataset x stratum x variant.

    python3 eval/walk_prior_variants.py [--datasets ...] [--cap 30] [--md out.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk_prior import DATASETS, ndcg5  # noqa: E402

csv.field_size_limit(10 ** 9)


def read_graph(stage1):
    names, types = [], []
    with open(f"{stage1}/nodes.csv", newline="") as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            names.append(row[0]); types.append(row[1])
    idx = {n: i for i, n in enumerate(names)}
    edges = []
    with open(f"{stage1}/edges.csv", newline="") as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            a, b = idx.get(row[0]), idx.get(row[2])
            if a is not None and b is not None:
                edges.append((a, row[1], b))
    return names, types, idx, edges


def direct_edges(types, edges):
    """paper -> frame shortcuts through the paper's own methods / tasks / findings"""
    out = defaultdict(set)                     # method/task/finding node -> frames it exposes
    for a, r, b in edges:
        if r in ("achieves", "overcomes", "limited_by", "works_via", "concerns", "explains"):
            out[a].add(b)
    new = []
    for a, r, b in edges:
        if types[a] == "document" and r in ("contributes", "addresses", "reports"):
            for f in out.get(b, ()):
                new.append((a, "direct_" + r, f))
    return new


def build(N, edges, extra=()):
    src = np.array([a for a, _, b in edges] + [a for a, _, b in extra])
    dst = np.array([b for a, _, b in edges] + [b for a, _, b in extra])
    rr = np.concatenate([src, dst]); cc = np.concatenate([dst, src])
    deg = np.bincount(rr, minlength=N).astype(np.float64)
    w = 1.0 / np.maximum(deg[rr], 1.0)
    return sp.csr_matrix((w, (cc, rr)), shape=(N, N)), deg


def run_walk(A, deg, docs, seeds_per_q, invdeg=False, T=3, alpha=0.15, chunk=512):
    N = A.shape[0]; tops = []
    for c0 in range(0, len(seeds_per_q), chunk):
        qs = seeds_per_q[c0:c0 + chunk]
        X0 = np.zeros((N, len(qs)), dtype=np.float32)
        for j, seeds in enumerate(qs):
            if not seeds: continue
            wts = np.array([1.0 / max(deg[s], 1.0) for s in seeds]) if invdeg else np.ones(len(seeds))
            X0[seeds, j] = (wts / wts.sum()).astype(np.float32)
        X = X0.copy()
        for _ in range(T):
            X = (1.0 - alpha) * (A @ X) + alpha * X0
        M = X[docs]
        for j in range(len(qs)):
            tops.append(docs[np.argsort(-M[:, j], kind="stable")[:5]])
    return tops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "retriever", "data"))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--cap", type=int, default=30, help="max OpenIE degree of an entity admitted into the hybrid graph")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    VARIANTS = ["base", "direct", "hybrid", "invdeg", "direct+hybrid", "direct+hybrid+invdeg"]
    L = [f"| dataset | stratum | queries | " + " | ".join(VARIANTS) + " |", "|---|---|--:|" + "--:|" * len(VARIANTS)]
    def out(s): print(s); L.append(s)
    for ds in a.datasets.split(","):
        fs1 = f"{a.data}/{ds}_test_v16sc/processed/stage1"; os1 = f"{a.data}/{ds}_test/processed/stage1"
        if not (os.path.exists(f"{fs1}/test.json") and os.path.exists(f"{os1}/test.json")): continue
        names, types, idx, edges = read_graph(fs1)
        docs = np.array([i for i, t in enumerate(types) if t == "document"])
        queries = json.load(open(f"{fs1}/test.json"))
        oq = {q["id"]: q for q in json.load(open(f"{os1}/test.json"))}
        # OpenIE mention edges with a degree cap; entities become new nodes appended to the frame graph
        onames, otypes, oidx, oedges = read_graph(os1)
        odeg = np.zeros(len(onames)); 
        for u, _, v in oedges: odeg[u] += 1; odeg[v] += 1
        ent_new = {}                                               # OpenIE entity name -> new node id
        N = len(names); hyb = []
        for u, r, v in oedges:
            if r != "is_mentioned_in" or otypes[u] != "entity" or odeg[u] > a.cap: continue
            d = idx.get(onames[v])
            if d is None: continue
            e = ent_new.setdefault(onames[u], N + len(ent_new))
            hyb.append((e, "mentioned_in", d))
        N2 = N + len(ent_new)
        dedges = direct_edges(types, edges)
        graphs = {"base": build(N2, edges), "direct": build(N2, edges, dedges), "hybrid": build(N2, edges, hyb),
                  "direct+hybrid": build(N2, edges, dedges + hyb)}
        f_seeds = [sorted({idx[n] for v in q["start_nodes"].values() for n in v if n in idx}) for q in queries]
        h_seeds = [sorted(set(s) | {ent_new[n] for n in oq.get(q["id"], {}).get("start_nodes", {}).get("entity", []) if n in ent_new})
                   for q, s in zip(queries, f_seeds)]
        golds = [set(q["supporting_documents"]) for q in queries]
        res = {}
        for v in VARIANTS:
            key = v.replace("+invdeg", ""); key = "base" if key == "invdeg" else key
            A, deg = graphs[key]
            seeds = h_seeds if "hybrid" in v else f_seeds
            tops = run_walk(A, deg, docs, seeds, invdeg="invdeg" in v)
            res[v] = [ndcg5([names[i] for i in t], g) for t, g in zip(tops, golds)]
        strata = ["all"] + sorted({q.get("stratum") or "same" for q in queries}, key=lambda s: (s != "same", s))
        if len(strata) == 2: strata = ["all"]
        for s in strata:
            sel = [i for i, q in enumerate(queries) if s == "all" or (q.get("stratum") or "same") == s]
            out(f"| {ds} | {s} | {len(sel)} | " + " | ".join(f"{100 * np.mean([res[v][i] for i in sel]):.2f}" for v in VARIANTS) + " |")
        print(f"  {ds}: frame edges {len(edges):,} | direct +{len(dedges):,} | hybrid +{len(hyb):,} mention edges from {len(ent_new):,} entities (cap {a.cap})", file=sys.stderr)
    L.append("")
    L.append(f"walk nDCG@5 (%), 3-step PPR restart 0.15 from the query's seeds. direct = paper->frame shortcuts for the mechanism frames a paper already "
             f"owns via its methods/tasks/findings; hybrid = + OpenIE entity->paper mention edges for entities of OpenIE degree <= {a.cap}, and the "
             "query's OpenIE entity seeds; invdeg = restart mass per seed proportional to 1/degree.")
    text = "\n".join(L); print(text)
    if a.md: open(a.md, "w").write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
