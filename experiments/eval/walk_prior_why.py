#!/usr/bin/env python3
"""
walk_prior_why.py -- why the untrained walk prefers one graph over the other, per dataset and stratum.

For every dataset, graph and stratum: how many seed nodes a query has and how specific they are
(degree), how far the golds sit from the seed set (BFS distance 1 / 2 / 3 / further), how large the
2-hop neighbourhood is (the flood the gold competes with), and the walk's nDCG@5 conditioned on the
gold's distance. Then qualitative pairs: queries where the frame-graph walk ranks the gold in the top 5
and the OpenIE walk buries it, and the reverse, each with the shortest seed->gold route on both graphs
and the degree of every node on it.

    python3 eval/walk_prior_why.py [--datasets ...] [--md results/qualitative/walk_prior_why.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict, deque

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk_prior import load_graph, DATASETS  # noqa: E402

csv.field_size_limit(10 ** 9)


def load_edges(stage1, idx):
    rel = {}
    with open(f"{stage1}/edges.csv", newline="") as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            a, b = idx.get(row[0]), idx.get(row[2])
            if a is not None and b is not None:
                rel[(a, b)] = row[1]
    return rel


def ntype(name, doc_set):
    if name in doc_set: return "paper"
    return name[1:name.index("]")] if name.startswith("[") and "]" in name else "entity"


def walk_ranks(A, idx, docs, queries, T=3, alpha=0.15, chunk=512):
    """rank of every gold under the walk, and the seed list, per query"""
    N = A.shape[0]; out = []
    doc_pos = {d: i for i, d in enumerate(docs)}
    for c0 in range(0, len(queries), chunk):
        qs = queries[c0:c0 + chunk]
        X0 = np.zeros((N, len(qs)), dtype=np.float32); seeds_of = []
        for j, q in enumerate(qs):
            seeds = sorted({idx[n] for v in q["start_nodes"].values() for n in v if n in idx})
            seeds_of.append(seeds)
            if seeds: X0[seeds, j] = 1.0 / len(seeds)
        X = X0.copy()
        for _ in range(T):
            X = (1.0 - alpha) * (A @ X) + alpha * X0
        M = X[docs]
        for j, q in enumerate(qs):
            golds = [idx[g] for g in q["supporting_documents"] if g in idx]
            col = M[:, j]
            ranks = {g: int((col > col[doc_pos[g]]).sum()) + 1 for g in golds if g in doc_pos}
            out.append((seeds_of[j], ranks))
    return out


def bfs_dist(Ab, seeds, targets, max_d=3):
    """min hops from the seed set to each target (undirected), by sparse frontier expansion"""
    N = Ab.shape[0]
    if not seeds: return {t: None for t in targets}, np.zeros(N, dtype=bool)
    x = np.zeros(N, dtype=bool); x[seeds] = True
    seen = x.copy(); dist = {t: (0 if t in set(seeds) else None) for t in targets}
    for d in range(1, max_d + 1):
        x = (Ab @ x.astype(np.float32)) > 0
        x &= ~seen; seen |= x
        for t in targets:
            if dist[t] is None and x[t]: dist[t] = d
        if all(v is not None for v in dist.values()): break
    return dist, seen


def bfs_path(adj, seeds, target, max_d=4):
    parent = {s: None for s in seeds}; q = deque(seeds); depth = {s: 0 for s in seeds}
    while q:
        u = q.popleft()
        if u == target:
            path = [u]
            while parent[path[-1]] is not None: path.append(parent[path[-1]])
            return path[::-1]
        if depth[u] >= max_d: continue
        for v in adj[u]:
            if v not in parent:
                parent[v] = u; depth[v] = depth[u] + 1; q.append(v)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "retriever", "data"))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    L = []
    def out(s=""): print(s); L.append(s)

    out("# Why the walk prefers one graph: seeds, distances, floods\n")
    out("| dataset | graph | stratum | seeds/query | median seed degree | hub seeds (deg>200) | gold at d=1 | d=2 | d=3 | d>3 | docs within 2 hops (median) | walk nDCG@5 all | d=1 queries | d=2 queries | d>=3 queries |")
    out("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    qual = {}
    for ds in a.datasets.split(","):
        per_graph = {}
        for g, label in ((f"{ds}_test_v16sc", "SciAffordGraph"), (f"{ds}_test", "OpenIE")):
            stage1 = f"{a.data}/{g}/processed/stage1"
            if not os.path.exists(f"{stage1}/test.json"): continue
            idx, A, docs, names = load_graph(stage1)
            Ab = (A > 0).astype(np.float32).tocsr()
            deg = np.asarray(Ab.sum(axis=1)).ravel()
            doc_set = set(docs.tolist())
            queries = json.load(open(f"{stage1}/test.json"))
            wr = walk_ranks(A, idx, docs, queries)
            rows = []
            for q, (seeds, ranks) in zip(queries, wr):
                golds = list(ranks)
                if not golds: continue
                dist, seen = bfs_dist(Ab, seeds, golds)
                flood = int(seen[docs].sum())
                best = min(ranks.values())
                nd = sum(1.0 / math.log2(r + 1) for r in ranks.values() if r <= 5) / sum(1.0 / math.log2(i + 2) for i in range(min(len(golds), 5)))
                dmin = min((d for d in dist.values() if d is not None), default=None)
                rows.append({"id": q["id"], "stratum": q.get("stratum") or "same", "n_seeds": len(seeds),
                             "seed_deg": [float(deg[s]) for s in seeds], "dist": dist, "dmin": dmin, "flood": flood,
                             "ndcg": nd, "best": best, "ranks": ranks, "seeds": seeds, "question": q["question"]})
            per_graph[label] = (rows, idx, names, deg, doc_set, stage1)
            strata = ["all"] + sorted({r["stratum"] for r in rows}, key=lambda s: (s != "same", s))
            if len(strata) == 2: strata = ["all"]
            for s in strata:
                sub = [r for r in rows if s == "all" or r["stratum"] == s]
                dd = [d for r in sub for d in r["dist"].values()]
                sd = [x for r in sub for x in r["seed_deg"]]
                def pct(cond): return 100 * np.mean([cond(d) for d in dd])
                def nd_at(cond):
                    v = [r["ndcg"] for r in sub if cond(r["dmin"])]
                    return f"{100 * np.mean(v):.1f} (n={len(v)})" if v else "--"
                out(f"| {ds} | {label} | {s} | {np.mean([r['n_seeds'] for r in sub]):.1f} | {np.median(sd):.0f} | "
                    f"{100 * np.mean([x > 200 for x in sd]):.1f}% | {pct(lambda d: d == 1):.1f}% | {pct(lambda d: d == 2):.1f}% | "
                    f"{pct(lambda d: d == 3):.1f}% | {pct(lambda d: d is None or d > 3):.1f}% | {np.median([r['flood'] for r in sub]):.0f} | "
                    f"{100 * np.mean([r['ndcg'] for r in sub]):.2f} | {nd_at(lambda d: d == 1)} | {nd_at(lambda d: d == 2)} | {nd_at(lambda d: d is not None and d >= 3)} |")
            print(f"  {g}: done", file=sys.stderr)
        if len(per_graph) == 2:
            qual[ds] = per_graph
    out()
    out("seeds = the query's start nodes in that graph (frames extracted from the query on SciAffordGraph, entities linked from the "
        "query text on OpenIE). degree = undirected degree in the stage-1 graph; a hub seed spreads its mass over hundreds of "
        "neighbours. gold at d=k = share of golds whose shortest route from ANY seed is k hops (BFS, undirected). docs within 2 hops "
        "= the papers a 2-step walk can reach at all, i.e. the field the gold competes in. The last three columns are the walk's "
        "nDCG@5 on queries whose nearest gold is at that distance.\n")

    # seed-type breakdown on the frame graph: which frame types carry the distance-1 links
    out("## Which seed types touch a gold directly (SciAffordGraph, distance-1 links)\n")
    out("| dataset | stratum | seed types per query (mean) | share of d=1 golds reached via task / function / limitation / method / entity / other |")
    out("|---|---|---|---|")
    for ds, pg in qual.items():
        rows, idx, names, deg, doc_set, stage1 = pg["SciAffordGraph"]
        Ab_adj = None
        _, A_, _, _ = load_graph(stage1); Ab = (A_ > 0).tocsr()
        for s in (["all", "same", "cross"] if any(r["stratum"] == "cross" for r in rows) else ["all"]):
            sub = [r for r in rows if s == "all" or r["stratum"] == s]
            tcount = Counter(); ttot = Counter()
            for r in sub:
                for sd in r["seeds"]: ttot[ntype(names[sd], doc_set)] += 1
                for g, d in r["dist"].items():
                    if d != 1: continue
                    nb = set(Ab.indices[Ab.indptr[g]:Ab.indptr[g + 1]])
                    for sd in r["seeds"]:
                        if sd in nb: tcount[ntype(names[sd], doc_set)] += 1
            tot = sum(tcount.values()) or 1
            types = ["task", "function", "limitation", "method", "entity"]
            out(f"| {ds} | {s} | " + ", ".join(f"{t} {ttot[t] / len(sub):.1f}" for t in types if ttot[t]) + " | "
                + " / ".join(f"{100 * tcount[t] / tot:.0f}%" for t in types) + f" / {100 * sum(v for k, v in tcount.items() if k not in types) / tot:.0f}% |")
    out()

    # qualitative pairs
    out("## Qualitative: where one graph's walk finds the gold and the other buries it\n")
    for ds, pg in qual.items():
        fr, fidx, fnames, fdeg, fdocs, fs1 = pg["SciAffordGraph"]; op, oidx, onames, odeg, odocs, os1 = pg["OpenIE"]
        docs_txt = json.load(open(f"{a.data}/{ds}_test/raw/documents.json"))
        frel, orel = load_edges(fs1, fidx), load_edges(os1, oidx)
        fadj, oadj = defaultdict(list), defaultdict(list)
        for (u, v) in frel: fadj[u].append(v); fadj[v].append(u)
        for (u, v) in orel: oadj[u].append(v); oadj[v].append(u)
        fb = {r["id"]: r for r in fr}; ob = {r["id"]: r for r in op}
        def route(adj, names, deg, rel, idx_, seeds, gold_name, doc_set):
            g = idx_.get(gold_name)
            if g is None: return "gold not in graph"
            p = bfs_path(adj, seeds, g)
            if not p: return "no route within 4 hops"
            parts = []
            for u, v in zip(p, p[1:]):
                r = rel.get((u, v)) or ("inv. " + rel.get((v, u), "?"))
                parts.append(f"{lab(names[u], doc_set, docs_txt)} [deg {int(deg[u])}] --{r}--> ")
            return "".join(parts) + lab(names[p[-1]], doc_set, docs_txt) + f" [deg {int(deg[p[-1]])}]"
        def lab(n, doc_set, docs_txt):
            return ("[paper] " + docs_txt.get(n, n).split(". ")[0][:70]) if n in docs_txt else n[:70]
        pairs = []
        for qid in fb.keys() & ob.keys():
            f, o = fb[qid], ob[qid]
            pairs.append((qid, f["best"], o["best"], f["stratum"]))
        def show(title, sel, first, second, fi, si):
            out(f"### {ds}: {title}\n")
            for qid, fbest, obest, strat in sel[: a.n_examples]:
                f, o = fb[qid], ob[qid]
                gold = min(f["ranks"], key=f["ranks"].get); gold_name = fnames[gold]
                out(f"**{qid}** ({strat}-field) | walk rank of the gold: SciAffordGraph {f['ranks'][gold]}, OpenIE {o['ranks'].get(oidx.get(gold_name), '--') if oidx.get(gold_name) is not None else '--'}")
                out(f"- query: {f['question'][:300]}")
                out(f"- gold: {docs_txt.get(gold_name, gold_name)[:220]}")
                out(f"- SciAffordGraph seeds: {len(f['seeds'])} (degrees median {np.median(f['seed_deg']):.0f}); route: {route(fadj, fnames, fdeg, frel, fidx, f['seeds'], gold_name, fdocs)}")
                out(f"- OpenIE seeds: {len(o['seeds'])} (degrees median {np.median(o['seed_deg']) if o['seed_deg'] else 0:.0f}); route: {route(oadj, onames, odeg, orel, oidx, o['seeds'], gold_name, odocs)}")
                out()
        fw = sorted([p for p in pairs if p[1] <= 5 and p[2] > 50], key=lambda p: (-p[2], p[1]))
        ow = sorted([p for p in pairs if p[2] <= 5 and p[1] > 50], key=lambda p: (-p[1], p[2]))
        out(f"### {ds}: counts\n\nframe top-5 while OpenIE past 50: {len(fw)} queries; OpenIE top-5 while frame past 50: {len(ow)} queries "
            f"(cross-field: {sum(p[3] == 'cross' for p in fw)} vs {sum(p[3] == 'cross' for p in ow)})\n")
        show("frame graph finds it, OpenIE buries it", fw, "frame", "openie", 1, 2)
        show("OpenIE finds it, frame graph buries it", ow, "openie", "frame", 2, 1)
    text = "\n".join(L)
    if a.md: open(a.md, "w").write(text + "\n"); print("wrote", a.md, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
