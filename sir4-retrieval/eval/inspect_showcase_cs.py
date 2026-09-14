#!/usr/bin/env python3
"""
inspect_showcase_cs.py -- readable dump of chosen (query, gold) pairs from path-interpretation files.

    python3 eval/inspect_showcase_cs.py --dir results/qualitative/drive_scan_sir4_cs \
        --title "Neuroticism" --title "Memory reconsolidation" --pair 10.48550_arxiv.2601.22474=q:ac2856c33ccdf7be

--dir holds hops_frame_ccmp.json, hops_frame_ccmp_off.json, [hops_frame_nocc.json], hops_openie.json (the Drive
scan folder); with no --dir it reads the local cs_hops_frame_{on,off}.json / cs_hops_openie.json. --title matches
the gold's title; --pair pins an exact query=gold. A path whose first hop does not start at one of the query's
seed frames (the "stability of motion" placeholder node) is marked ARTEFACT, not hidden.
"""
import argparse, collections, csv, json, os
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q = f"{R}/sir4-retrieval/results/qualitative"
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=None); ap.add_argument("--dataset", default="sir4_cs")
ap.add_argument("--title", action="append", default=[]); ap.add_argument("--pair", action="append", default=[])
ap.add_argument("--quartet", default=f"{R}/quartet/data.nosync/benchmark/cs_test_final/eval.json")
a = ap.parse_args(); D = a.dataset
docs = json.load(open(f"{R}/kg-construction/data/{D}_test/raw/documents.json"))
queries = {q["id"]: q for q in json.load(open(f"{R}/kg-construction/data/{D}_test/raw/test.json"))}
qf = {}
if os.path.exists(a.quartet):
    for x in json.load(open(a.quartet)):
        for d, m in (x.get("quartet", {}).get("per_document") or {}).items(): qf[(x["id"], d)] = m
adj = collections.defaultdict(list)
with open(f"{R}/kg-construction/data/{D}_test_v16sc/processed/stage1/edges.csv", newline="") as fh:
    rd = csv.reader(fh); next(rd)
    for row in rd:
        if len(row) >= 3: adj[row[0]].append((row[1], row[2])); adj[row[2]].append(("inv." + row[1], row[0]))
if a.dir:
    ARMS = [("frame graph + CCMP (gate on)", f"{a.dir}/hops_frame_ccmp.json"), ("frame graph, same weights, gate OFF", f"{a.dir}/hops_frame_ccmp_off.json"),
            ("frame graph, no CCMP (own run)", f"{a.dir}/hops_frame_nocc.json"), ("OpenIE graph model", f"{a.dir}/hops_openie.json")]
else:
    ARMS = [("frame graph + CCMP (gate on)", f"{Q}/cs_hops_frame_on.json"), ("frame graph, same weights, gate OFF", f"{Q}/cs_hops_frame_off.json"),
            ("OpenIE graph model", f"{Q}/cs_hops_openie.json")]
ARMS = [(n, p) for n, p in ARMS if os.path.exists(p)]
assert ARMS, (f"no hops_*.json under {a.dir}: download cargo-gfmrag/outputs/scan/{D}/ from Drive "
              f"(hops_frame_ccmp.json, hops_frame_ccmp_off.json, hops_openie.json, showcase_{D}_candidates.json) into that folder")
IDX = {n: {(r["id"], t["doc"]): (r, t) for r in json.load(open(p)) for t in r["targets"]} for n, p in ARMS}
main = ARMS[0][0]
def lab(n): return f"[paper] {docs[n].split('. ')[0][:70]}" if n in docs else n
def fmt(p, seeds):
    out = ["      ARTEFACT: first hop does not start at a seed frame (placeholder node); not a route"] if p["hops"][0]["head"] not in seeds else []
    for h in p["hops"]:
        g = f"   (gate {h['gate']:.2f}, layer {h.get('layer')})" if "gate" in h else (f"   (layer {h.get('layer')})" if "layer" in h else "")
        out.append(f"      {lab(h['head'])}\n         --{h['rel']}-->{g}")
    out.append(f"      {lab(p['hops'][-1]['tail'])}"); return "\n".join(out)
pairs = [tuple(s.split("=", 1)) for s in a.pair]
for tt in a.title:
    hits = [k for k in IDX[main] if docs.get(k[1], "").startswith(tt) or tt.lower() in docs.get(k[1], "")[:120].lower()]
    if not hits: print(f"ABSENT in {ARMS[0][1]}: {tt}")
    pairs += hits
W = 110
for i, (qid, g) in enumerate(pairs, 1):
    q = queries.get(qid, {}); m = qf.get((qid, g), {})
    print("=" * W); print(f"{i}. {docs.get(g, g).split('. ')[0][:90]}   <-   {q.get('question', '')[:110]}..."); print("=" * W)
    print(f"query id: {qid}   gold id: {g}\nQUARTET label for this gold: {m.get('field_pair')}   stratum={m.get('stratum')}   | golds of this query: {len(q.get('supporting_documents', []))}\n")
    print("QUERY:\n  " + q.get("question", "")); print(); print("GOLD:\n  " + docs.get(g, g)[:1000]); print()
    print("GOLD'S EDGES IN THE FRAME GRAPH (degree %d):" % len(adj[g]))
    for rel, t_ in sorted(adj[g]): print(f"  --{rel}--> {t_}")
    print()
    for n, _ in ARMS:
        r, t = IDX[n].get((qid, g), (None, None))
        if not t: print(f"--- {n}: not in this file\n"); continue
        print(f"--- {n}\n  ranks: {t['rank']}   | shortest seed->gold route: {t.get('min_hops')} hops   | dense cos {t.get('dense_cos')}")
        if n == main:
            print(f"  seeds ({len(r['seeds'])}): " + " ; ".join(r["seeds"])); print("  scorer views:")
            for v in t.get("views", []): print(f"    view {v['view']} match {v['match']:.3f}: {v.get('text')}")
        if not t["paths"]: print("  paths: none recorded")
        for j, p in enumerate(t["paths"], 1): print(f"  path {j}  weight {p['weight']:.3f}  ({len(p['hops'])} hops)"); print(fmt(p, set(r["seeds"])))
        print()
