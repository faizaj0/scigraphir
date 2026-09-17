"""
audit_graph.py -- step 10. Check a built SciAfford graph before anything is trained on it.

The builder prints node and edge counts and nothing else, so every property that
actually decides whether training is meaningful is currently unobserved. The one
that matters most is SEED COVERAGE: a query with no start nodes is dropped by the
GFM-RAG loader with a single log line, so the reported metric is silently computed
over fewer queries than the split contains.

Sections
  A structural   dangling edges, self loops, duplicates, isolated docs, gold in graph
  B seeds        seeds per query, zero-seed queries, per type, per stratum
  C reachability can a gold be reached from the query's seeds in one hop
  D hubs         canonical nodes that seed a large share of all queries
  E spread       distribution of the snap cosine (needs BGE; --spread)
  F sample       affordance representations + seeds + golds dumped to a file for hand reading

Exit status is 1 if a HARD check fails: dangling edges, or any zero-seed query.
Those two make the training run wrong rather than merely worse.

Usage
-----
    python3 eval/audit_graph.py --dataset sir4_cs --split test
    python3 eval/audit_graph.py --dataset sir4_cs --split train --spread
    python3 eval/audit_graph.py --dataset sir4_cs --split test --sample 40
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # experiments/
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, REPO_ROOT)
from scigraphir_paths import banner, corpus_dir, affordances_path, graph_name, set_dataset  # noqa: E402

csv.field_size_limit(10 ** 8)
KG = f"{REPO_ROOT}/retriever"
SEED_T = ("task", "function", "method", "limitation")


def pct(n, d):
    return f"{n/d:.1%}" if d else "n/a"


def dist(counts: list[int]) -> str:
    """min/median/mean/p90/max, without pulling in numpy for six numbers."""
    if not counts:
        return "empty"
    s = sorted(counts)
    n = len(s)
    return (f"min {s[0]}  med {s[n//2]}  mean {sum(s)/n:.1f}  "
            f"p90 {s[int(n*0.9)]}  max {s[-1]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("SCIGRAPHIR_DATASET", "tomato"))
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--suffix", default="v16sc")
    ap.add_argument("--sample", type=int, default=30, help='affordance representations dumped for hand reading')
    ap.add_argument("--spread", action="store_true",
                    help="also report the snap-cosine distribution (loads BGE)")
    ap.add_argument("--spread-queries", type=int, default=300)
    a = ap.parse_args()
    set_dataset(a.dataset)
    print(banner())
    random.seed(0)

    s1 = f"{KG}/data/{graph_name(a.split, a.suffix)}/processed/stage1"
    if not os.path.isdir(s1):
        print(f"no graph at {s1} -- build it first", file=sys.stderr)
        return 2

    ntype = {r["name"]: r["type"] for r in csv.DictReader(open(f"{s1}/nodes.csv"))}
    edges = [(r["source"], r["relation"], r["target"])
             for r in csv.DictReader(open(f"{s1}/edges.csv"))]
    queries = json.load(open(f"{s1}/{a.split}.json"))
    corpus = json.load(open(f"{corpus_dir(a.split)}/raw/documents.json"))
    docs = {n for n, t in ntype.items() if t == "document"}

    hard_fail = []
    print(f"\n{'='*66}\n  {graph_name(a.split, a.suffix)}\n{'='*66}")

    # ---- A. structural -------------------------------------------------
    print("\n[A] structural")
    print(f"  nodes {len(ntype):,}   edges {len(edges):,}   queries {len(queries):,}")
    tc = Counter(ntype.values())
    for t, c in tc.most_common():
        print(f"    {t:12} {c:>8,}")

    dangling = sum(1 for s, _, t in edges if s not in ntype or t not in ntype)
    selfloop = sum(1 for s, _, t in edges if s == t)
    dupe = len(edges) - len(set(edges))
    print(f"  dangling endpoints {dangling:,}   self loops {selfloop:,}   duplicate rows {dupe:,}")
    if dangling:
        hard_fail.append(f"{dangling} edges reference a node not in nodes.csv")

    deg = Counter()
    for s, _, t in edges:
        deg[s] += 1
        deg[t] += 1
    isolated = [d for d in docs if deg[d] == 0]
    print(f"  documents with no edge: {len(isolated):,} ({pct(len(isolated), len(docs))})")
    print(f"  document nodes {len(docs):,} of {len(corpus):,} in the corpus "
          f"({pct(len(docs), len(corpus))})")

    golds = {g for q in queries for g in (q.get("supporting_documents") or [])}
    missing_gold = golds - docs
    print(f"  gold documents present in the graph: "
          f"{pct(len(golds) - len(missing_gold), len(golds))}"
          f"   missing {len(missing_gold):,}")

    rel = Counter(r for _, r, _ in edges)
    print(f"  relations {len(rel)}: " + ", ".join(f"{r}({c:,})" for r, c in rel.most_common()))

    # ---- B. seeds ------------------------------------------------------
    print("\n[B] seeds")
    per_q, per_type, zero = [], Counter(), []
    for q in queries:
        sn = q.get("start_nodes") or {}
        n = sum(len(v) for v in sn.values())
        per_q.append(n)
        for t, v in sn.items():
            per_type[t] += len(v)
        if n == 0:
            zero.append(q["id"])
    print(f"  seeds per query: {dist(per_q)}")
    for t, c in per_type.most_common():
        print(f"    {t:12} {c:>8,} total   {c/max(len(queries),1):.1f}/query")
    print(f"  ZERO-SEED queries: {len(zero):,} ({pct(len(zero), len(queries))})")
    if zero:
        print(f"    e.g. {zero[:5]}")
        hard_fail.append(f"{len(zero)} queries have no start nodes and will be "
                         f"dropped by the loader")

    st = defaultdict(lambda: [0, 0])
    for q, n in zip(queries, per_q):
        k = q.get("stratum") or "?"
        st[k][0] += 1
        st[k][1] += (n == 0)
    print("  by stratum:")
    for k, (tot, z) in sorted(st.items(), key=lambda kv: -kv[1][0]):
        mean = sum(n for q, n in zip(queries, per_q) if (q.get("stratum") or "?") == k) / max(tot, 1)
        print(f"    {k:10} {tot:6,} queries   {mean:5.1f} seeds/query   zero-seed {z}")

    # ---- C. reachability ----------------------------------------------
    # One hop OUT of a seed. A gold reachable this way can be promoted by the
    # graph channel; one that is not can only ever be found by the handcrafted scorer.
    print("\n[C] reachability (gold adjacent to a seed node)")
    nbr = defaultdict(set)
    for s, _, t in edges:
        if ntype.get(t) == "document":
            nbr[s].add(t)
        if ntype.get(s) == "document":
            nbr[t].add(s)
    hit_any, hit_all, gold_tot, gold_hit = 0, 0, 0, 0
    for q in queries:
        seeds = {n for v in (q.get("start_nodes") or {}).values() for n in v}
        reach = set()
        for sd in seeds:
            reach |= nbr.get(sd, set())
        g = set(q.get("supporting_documents") or [])
        if not g:
            continue
        gold_tot += len(g)
        gold_hit += len(g & reach)
        hit_any += bool(g & reach)
        hit_all += (g <= reach)
    n = sum(1 for q in queries if q.get("supporting_documents"))
    print(f"  queries with >=1 gold reachable: {hit_any:,} ({pct(hit_any, n)})")
    print(f"  queries with ALL golds reachable: {hit_all:,} ({pct(hit_all, n)})")
    print(f"  golds reachable: {gold_hit:,} of {gold_tot:,} ({pct(gold_hit, gold_tot)})")
    print('  (an unreachable gold is invisible to the graph channel; the handcrafted scorer '
          "can still rank it)")

    # ---- D. hub seeds --------------------------------------------------
    print("\n[D] over-seeded nodes")
    seed_use = Counter()
    for q in queries:
        for v in (q.get("start_nodes") or {}).values():
            seed_use.update(set(v))
    if seed_use:
        share = [(nd, c) for nd, c in seed_use.most_common(10)]
        print(f"  distinct nodes used as a seed: {len(seed_use):,}")
        print(f"  top seeds by query share (a node seeding most queries carries "
              f"no signal):")
        for nd, c in share:
            print(f"    {c:6,} ({pct(c, len(queries)):>6})  deg {deg[nd]:>6,}  {nd[:70]}")
        over = sum(1 for _, c in seed_use.items() if c > 0.10 * len(queries))
        print(f"  nodes seeding >10% of queries: {over}")

    # ---- E. similarity spread -----------------------------------------
    if a.spread:
        print("\n[E] snap-cosine spread")
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except Exception as e:                                    # noqa: BLE001
            print(f"  skipped: {e}")
        else:
            qf = {json.loads(l)["id"]: (json.loads(l).get("affordance", json.loads(l).get("frame")) or {})
                  for l in open(affordances_path("query", a.split))}
            samp = random.sample(queries, min(a.spread_queries, len(queries)))
            pairs = []
            for q in samp:
                fr = qf.get(q["id"], {})
                phr = list(fr.get("task") or []) + list(fr.get("needs") or [])
                for cm in (fr.get("current_methods") or []):
                    if isinstance(cm, dict) and cm.get("name"):
                        phr.append(cm["name"])
                seeds = [n for v in (q.get("start_nodes") or {}).values() for n in v]
                if phr and seeds:
                    pairs.append((phr, seeds))
            if not pairs:
                print("  no phrase/seed pairs to score")
            else:
                model = SentenceTransformer("BAAI/bge-large-en-v1.5")
                model.max_seq_length = 512
                uphr = sorted({p for ph, _ in pairs for p in ph})
                usd = sorted({s for _, sd in pairs for s in sd})
                strip = lambda x: x.split("] ", 1)[-1]            # noqa: E731
                E1 = model.encode(uphr, normalize_embeddings=True,
                                  batch_size=128, show_progress_bar=False)
                E2 = model.encode([strip(s) for s in usd], normalize_embeddings=True,
                                  batch_size=128, show_progress_bar=False)
                i1 = {p: i for i, p in enumerate(uphr)}
                i2 = {s: i for i, s in enumerate(usd)}
                best = []
                for ph, sd in pairs:
                    S = E1[[i1[p] for p in ph]] @ E2[[i2[s] for s in sd]].T
                    best.extend(S.max(1).tolist())
                best.sort()
                m = len(best)
                print(f"  {m:,} phrase-to-nearest-seed cosines over {len(pairs)} queries")
                print("  " + "  ".join(f"p{p}={best[int(m*p/100)]:.3f}"
                                       for p in (5, 25, 50, 75, 95)))
                below = sum(1 for x in best if x < 0.60)
                print(f"  below the 0.60 snap threshold: {below:,} ({pct(below, m)})")
                print("  (a mass sitting just above 0.60 means the threshold, not the "
                      "vocabulary, is choosing the seeds)")

    # ---- F. hand-reading sample ---------------------------------------
    out = f"{ROOT}/data/audit_{a.dataset}_{a.split}_sample.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    qf = {json.loads(l)["id"]: (json.loads(l).get("affordance", json.loads(l).get("frame")) or {})
          for l in open(affordances_path("query", a.split))}
    df = {json.loads(l)["id"]: (json.loads(l).get("affordance", json.loads(l).get("frame")) or {})
          for l in open(affordances_path("doc", a.split))}
    with open(out, "w") as f:
        for q in random.sample(queries, min(a.sample, len(queries))):
            f.write("=" * 78 + f"\n{q['id']}   stratum={q.get('stratum')}\n")
            f.write(f"Q: {(q.get('question') or '')[:600]}\n\n")
            f.write('problem requirement representations:\n' + json.dumps(qf.get(q["id"], {}), indent=1) + "\n\n")
            f.write("seeds:\n")
            for t, v in sorted((q.get("start_nodes") or {}).items()):
                f.write(f"  {t}: {v}\n")
            f.write("\ngolds:\n")
            for g in (q.get("supporting_documents") or []):
                f.write(f"  {g}  in_graph={g in docs}\n")
                f.write(f"    text : {(corpus.get(g) or '')[:220]}\n")
                f.write(f"    affordance: {json.dumps(df.get(g, {}))[:400]}\n")
            f.write("\n")
    print(f"\n[F] {min(a.sample, len(queries))} cases -> {out}")
    print("    read 30 to 50 of these by hand; nothing else in this report "
          'catches a affordance representation that is fluent and wrong')

    if hard_fail:
        print("\nHARD FAILURES:")
        for h in hard_fail:
            print("  -", h)
        return 1
    print("\nno hard failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
