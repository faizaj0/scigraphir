#!/usr/bin/env python3
"""
route_graph.py -- draw the local subgraph the reasoner used for one (query, gold) pair.

All seed-origin routes recorded for the pair (gate on, plus gate-off-only routes dashed) are merged into one
graph, laid out in layers from the query's seed nodes (left) to the gold (right). Nodes are coloured by affordance representation
type; edges carry the relation, are coloured by the CCMP gate on the sender (orange > 1.05, blue < 0.95) and
widened by the weight of the best path through them. The gold's remaining edges (not on any route) are shown in
grey so the reader sees what else the paper touches; a domain node shows its degree.

  python3 eval/route_graph.py --dataset sir4_cs --dir results/qualitative/drive_scan_sir4_cs --prefix hops_ \\
      --pair 10.48550_arxiv.2602.20408=10.3758/bf03202751 --out results/qualitative/showcase2_drive/graph_creativity
"""
import argparse, collections, csv, json, os, textwrap
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILL = {"function": "#d6f0e0", "limitation": "#fce8c8", "method": "#e4dcf6", "task": "#fce8c8", "finding": "#faf3cd",
        "mechanism": "#e4dcf6", "domain": "#ececee", "paper": "#d6e4f7", "entity": "#f3f3f3"}


def ntype(n, docs):
    if n in docs: return "paper"
    return n[1:n.index("]")] if n.startswith("[") and "]" in n else "entity"


def label(n, docs, w=24):
    if n in docs: return "\n".join(textwrap.wrap("[paper] " + docs[n].split(". ")[0], w)[:3])
    return "\n".join(textwrap.wrap(n, w)[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--prefix", default="hops_")
    ap.add_argument("--pair", required=True); ap.add_argument("--out", required=True); ap.add_argument("--max-paths", type=int, default=6)
    ap.add_argument("--title", default="")
    a = ap.parse_args(); q, g = a.pair.split("=", 1); D = a.dataset
    docs = json.load(open(f"{R}/retriever/data/{D}_test/raw/documents.json"))
    queries = {x["id"]: x for x in json.load(open(f"{R}/retriever/data/{D}_test/raw/test.json"))}
    adj = collections.defaultdict(list)
    with open(f"{R}/retriever/data/{D}_test_v16sc/processed/stage1/edges.csv", newline="") as fh:
        rd = csv.reader(fh); next(rd)
        for row in rd:
            if len(row) >= 3: adj[row[0]].append((row[1], row[2])); adj[row[2]].append(("inv." + row[1], row[0]))
    def load(name):
        p = f"{a.dir}/{a.prefix}{name}.json"
        if not os.path.exists(p): return None
        for r in json.load(open(p)):
            if r["id"] != q: continue
            for t in r["targets"]:
                if t["doc"] == g: return r, t
        return None
    on = load("frame_ccmp"); off = load("frame_ccmp_off")
    assert on, f"pair not in {a.dir}/{a.prefix}frame_ccmp.json"
    r, t = on; seeds = set(r["seeds"])
    def valid(p):
        h = p["hops"]
        if not h or h[0]["head"] not in seeds or not all(x["tail"] == y["head"] for x, y in zip(h, h[1:])): return False
        ns = [h[0]["head"]] + [x["tail"] for x in h]
        return len(ns) == len(set(ns))          # simple path only
    P_on = [p for p in t["paths"] if valid(p)][: a.max_paths]
    P_off = [p for p in (off[1]["paths"] if off else []) if valid(p)][: a.max_paths]
    key = lambda p: tuple((h["head"], h["rel"], h["tail"]) for h in p["hops"])
    on_keys = {key(p) for p in P_on}
    # merge: edge -> (best weight, gate, on/off-only)
    E = {}; depth = {}
    for p in P_on:
        for i, h in enumerate(p["hops"]):
            k = (h["head"], h["rel"], h["tail"]); w, gt, kind = E.get(k, (0, None, "on"))
            E[k] = (max(w, p["weight"]), h.get("gate", gt), "on"); depth[h["head"]] = min(depth.get(h["head"], i), i); depth[h["tail"]] = min(depth.get(h["tail"], i + 1), i + 1)
    for p in P_off:
        if key(p) in on_keys: continue
        for i, h in enumerate(p["hops"]):
            k = (h["head"], h["rel"], h["tail"])
            if k not in E: E[k] = (p["weight"], None, "off"); depth[h["head"]] = min(depth.get(h["head"], i), i); depth[h["tail"]] = min(depth.get(h["tail"], i + 1), i + 1)
    nodes = set(depth); depth[g] = max(depth.values()) + (0 if g in depth and depth[g] == max(depth.values()) else 0)
    gd = max(depth.values()); depth[g] = gd
    # the gold's other edges (not on a route)
    others = [(rel, n) for rel, n in adj[g] if (n, rel.replace("inv.", "inverse_"), g) not in E and (n, rel, g) not in E and not any(k[0] == n and k[2] == g for k in E)]
    # --- layout: columns by depth, rows spread; gold alone in its column
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    cols = collections.defaultdict(list)
    for n in nodes: cols[depth[n]].append(n)
    for c in cols: cols[c] = sorted(cols[c], key=lambda n: (ntype(n, docs) != "domain", n))
    nbrs = collections.defaultdict(set)
    for (h, _, tl) in E: nbrs[h].add(tl); nbrs[tl].add(h)
    for _sweep in range(3):
        for c in sorted(cols):
            rowof = {n: i for cc in cols for i, n in enumerate(cols[cc])}
            def bary(n):
                v = [rowof[m] for m in nbrs[n] if m in rowof and depth.get(m) != c]
                return sum(v) / len(v) if v else rowof[n]
            cols[c] = sorted(cols[c], key=bary)
    if g in cols[gd] and len(cols[gd]) > 1: cols[gd].remove(g); cols[gd + 1] = [g]; depth[g] = gd + 1; gd += 1
    W, H = 3.6, 1.55; pos = {}
    maxrows = max(len(v) for v in cols.values())
    for c, ns in cols.items():
        for i, n in enumerate(ns): pos[n] = (c * W, (len(ns) - 1) / 2 - i) * 1 if False else (c * W, ((len(ns) - 1) / 2 - i) * H * 1.6)
    # gold's other edges: put their nodes in a column right of the gold
    for i, (rel, n) in enumerate(others[:6]):
        if n not in pos: pos[n] = ((gd + 1) * W, ((min(len(others), 6) - 1) / 2 - i) * H * 1.3)
    fig, ax = plt.subplots(figsize=(3.2 * (gd + 2) + 2, 1.05 * max(maxrows, min(len(others), 6), 3) * 1.6 + 2.6))
    def box(n, x, y, grey=False, gold=False):
        tp = ntype(n, docs); fc = "#f4f4f4" if grey else FILL.get(tp, "#f3f3f3")
        extra = ""
        if tp == "domain": extra = f"\n(degree {len(adj[n])})"
        ax.add_patch(FancyBboxPatch((x - 1.45, y - 0.55), 2.9, 1.1, boxstyle="round,pad=0.08", fc=fc, ec="#1f4e9a" if gold else ("#aaaaaa" if grey else "#555555"), lw=2.0 if gold else 0.8))
        ax.text(x, y, label(n, docs, 26) + extra, ha="center", va="center", fontsize=6.6, color="#777777" if grey else "black")
    for n, (x, y) in pos.items(): box(n, x, y, grey=(n not in nodes and n != g), gold=(n == g))
    wmax = max((w for w, _, _ in E.values()), default=1) or 1
    for (h, rel, tl), (w, gt, kind) in E.items():
        (x1, y1), (x2, y2) = pos[h], pos[tl]
        col = "#333333" if gt is None or abs(gt - 1) <= 0.05 else ("#d9701a" if gt > 1 else "#1f6fb4")
        span = abs(depth.get(tl, 0) - depth.get(h, 0)); rad = 0.0 if span <= 1 else (0.28 if y2 >= y1 else -0.28)
        ax.annotate("", xy=(x2 - 1.45, y2), xytext=(x1 + 1.45, y1), arrowprops=dict(arrowstyle="-|>", lw=0.8 + 2.2 * (w / wmax) if kind == "on" else 0.9, color=col if kind == "on" else "#999999", ls="-" if kind == "on" else "--", shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2 + (0.0 if rad == 0 else (0.9 if rad > 0 else -0.9) * abs(x2 - x1) / W * 0.35)
        txt = rel.replace("inverse_", "inv. ").replace("_", " ") + (f"  [gate {gt:.2f}]" if gt is not None and kind == "on" and abs(gt - 1) > 0.05 else "")
        ax.text(xm, ym + 0.22, txt, fontsize=6, ha="center", va="bottom", style="italic", color=col if kind == "on" else "#999999", bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
    for rel, n in others[:6]:
        (x1, y1), (x2, y2) = pos[g], pos[n]
        ax.annotate("", xy=(x2 - 1.45, y2), xytext=(x1 + 1.45, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#bbbbbb", shrinkA=0, shrinkB=0))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.2, rel.replace("inv.", "inv. ").replace("_", " "), fontsize=5.6, ha="center", va="bottom", style="italic", color="#999999", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 2.0, max(xs) + 2.0); ax.set_ylim(min(ys) - 1.3, max(ys) + 1.9); ax.axis("off")
    rk = t["rank"]; rko = off[1]["rank"] if off else {}
    ax.text(min(xs) - 1.9, max(ys) + 1.75, (a.title or f"{docs.get(g, g).split('. ')[0][:80]}") + f"\nquery: {queries.get(q, {}).get('question', '')[:150]}...", fontsize=8.5, va="top", weight="bold")
    ax.text(min(xs) - 1.9, min(ys) - 1.0,
            f"seeds of the query: {len(seeds)} affordance nodes ({len({k[0] for k in E if k[0] in seeds})} on a route)   |   routes drawn: {len(P_on)} with CCMP, {sum(1 for p in P_off if key(p) not in on_keys)} gate-off only (dashed)   |   "
            f"rank: cosine {rk.get('dense')}, scorer {rk.get('scorer')}, graph channel {rk.get('graph')} (gate off {rko.get('graph')}), SciGraphIR {rk.get('fused')} (gate off {rko.get('fused')})   |   "
            f"grey: the gold's other edges ({len(others)})", fontsize=7, va="top", color="#333333")
    for ext in ("png", "pdf"): fig.savefig(f"{a.out}.{ext}", dpi=170, bbox_inches="tight")
    print("wrote", a.out + ".png")


if __name__ == "__main__":
    main()
