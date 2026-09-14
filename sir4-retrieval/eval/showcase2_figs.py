#!/usr/bin/env python3
"""
showcase2_figs.py -- figures for hand-picked path interpretations (the "showcase 2" notebook).

Reads hops_pick_<arm>.json written by interpret_paths for a PICK of (query, gold) pairs and draws, per dataset:

  <out>_ladder.{png,pdf}    one row per pair: the gold's rank at every stage of the cumulative ablation
                            (Qwen3 cosine -> multi-view scorer -> +OpenIE graph -> +SciAffordGraph, gate off ->
                            +CCMP), log scale, with the graph channel alone (gate off / on) as hollow markers and
                            any dense/lexical baselines as grey ticks. The "difference with the graph branch" figure.
  <out>_routes_<n>.{png,pdf} one panel per pair (4 per page): the top interpreted route under the frame graph with
                            CCMP, boxes typed by frame kind, the relation on every hop and the CCMP gate coloured
                            (orange > 1.05, blue < 0.95), the weight with CCMP and with the gate off, and the
                            OpenIE graph's top route (or "no route") underneath for contrast.
  <out>.md                  the readable dump (query, gold, field label, ranks per arm, seeds, views, every path).

usage:
  showcase2_figs.py --dataset sir4_cs --dir <folder with hops_pick_*.json> --pick golds_pick.json \\
      --queries raw/test.json --docs raw/documents.json [--quartet eval.json] [--pred qwen3=...] --out <prefix>
"""
from __future__ import annotations
import argparse, json, os, textwrap

BIG = 10 ** 6
OPENIE_FUSED = False     # the OpenIE model's fused rank is its own scorer's rank when its graph has no route; off by default
ARMS = [("frame_ccmp", "SciAffordGraph + CCMP"), ("frame_ccmp_off", "SciAffordGraph, gate off"),
        ("frame_nocc", "SciAffordGraph, no CCMP (own run)"), ("openie", "OpenIE graph")]
SYS = {"bm25": "BM25", "bge": "BGE-large", "qwen3": "Qwen3-Emb.", "reasonir": "ReasonIR-8B", "specter2": "SPECTER2", "scincl": "SciNCL"}
FILL = {"function": "#d6f0e0", "limitation": "#fce8c8", "method": "#e4dcf6", "task": "#fce8c8", "finding": "#faf3cd",
        "mechanism": "#e4dcf6", "domain": "#ececee", "paper": "#d6e4f7", "entity": "#f3f3f3", "gold": "#c9dcf5"}


def short(s, n):
    s = " ".join(str(s).split()); return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def ntype(name, docs):
    if name in docs: return "paper"
    if name.startswith("[") and "]" in name: return name[1: name.index("]")]
    return "entity"


def node_text(name, docs, w=22):
    if name in docs: return "\n".join(textwrap.wrap("[paper] " + docs[name].split(". ")[0], w)[:3])
    return "\n".join(textwrap.wrap(name, w)[:3])


def load_arm(path):
    out = {}
    for r in json.load(open(path)):
        for t in r.get("targets", []):
            out[(r["id"], t["doc"])] = {"rank": t["rank"], "paths": t.get("paths", []), "views": t.get("views", []),
                                        "seeds": r.get("seeds", []), "question": r.get("question", ""), "min_hops": t.get("min_hops"),
                                        "stratum": r.get("stratum")}
    return out


def valid(p, seeds):
    h = p.get("hops") or []
    return bool(h) and h[0]["head"] in seeds and all(a["tail"] == b["head"] for a, b in zip(h, h[1:]))


def ranked_docs(rec):
    p = rec.get("predictions", rec); d = p.get("document", p) if isinstance(p, dict) else p
    return [x[0] if isinstance(x, (list, tuple)) else x for x in d]


def draw_ladder(rows, out, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(rows); fig, ax = plt.subplots(figsize=(11, 0.62 * n + 1.8))
    stages = [("dense", "Qwen3 cosine", "o", "#7f7f7f"), ("scorer", "multi-view scorer", "s", "#9467bd"),
              ("off_fused", "+ SciAffordGraph (gate off)", "^", "#2ca02c"), ("fused", "+ CCMP (SciGraphIR)", "*", "#1f77b4")]
    if OPENIE_FUSED: stages.insert(2, ("openie_fused", "OpenIE model (own scorer + entity graph)", "D", "#ff7f0e"))
    for i, r in enumerate(rows):
        y = n - 1 - i
        xs = [r.get(k) for k, *_ in stages]
        pts = [(x, k) for x, (k, *_) in zip(xs, stages) if x]
        if len(pts) > 1: ax.plot([x for x, _ in pts], [y] * len(pts), color="#bbbbbb", lw=1, zorder=1)
        for x, (k, lab, mk, col) in zip(xs, stages):
            if x: ax.scatter([x], [y], marker=mk, s=90 if mk == "*" else 46, color=col, zorder=3, label=lab if i == 0 else None)
        for k, lab, mk, col in (("off_graph", "SciAffordGraph channel alone, gate off", "^", "#2ca02c"), ("graph", "SciAffordGraph channel alone, with CCMP", "*", "#1f77b4"),
                                ("openie_graph", "OpenIE entity-graph channel alone", "D", "#ff7f0e")):
            if r.get(k): ax.scatter([r[k]], [y], marker=mk, s=90 if mk == "*" else 46, facecolors="none", edgecolors=col, linewidths=1.4, zorder=2, label=lab if i == 0 else None)
        for b, v in (r.get("base") or {}).items():
            if v and v < BIG: ax.scatter([v], [y], marker="|", s=70, color="#999999", zorder=2, label="dense / lexical baselines" if (i == 0 and b == list(r["base"])[0]) else None)
    ax.set_yticks(range(n)); ax.set_yticklabels([r["label"] for r in rows][::-1], fontsize=8.5)
    ax.set_xscale("log"); ax.set_xlim(0.8, 6000); ax.set_xlabel("rank of the gold inspiration (log scale; lower is better)")
    ax.axvline(10, color="#dddddd", lw=1, ls="--"); ax.text(10, -0.75, "top-10", fontsize=7.5, color="#888888", ha="center", va="top")
    ax.grid(True, axis="x", ls=":", alpha=0.5); ax.set_title(title, fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16 - 0.9 / max(n, 4)), ncol=3, fontsize=8, frameon=False)
    ax.text(0.0, -0.10 - 0.9 / max(n, 4), "filled = the cumulative ablation (one scorer, add the frame graph, add CCMP); hollow = each graph's channel on its own", transform=ax.transAxes, fontsize=7.5, color="#555555", va="top")
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(f"{out}_ladder.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_routes(rows, out, docs, per_page=4):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    pages = [rows[i: i + per_page] for i in range(0, len(rows), per_page)]
    for pg, chunk in enumerate(pages, 1):
        fig, axes = plt.subplots(len(chunk), 1, figsize=(16, 4.3 * len(chunk)), squeeze=False)
        for ax, r in zip(axes[:, 0], chunk):
            ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 12)
            ax.text(0, 11.75, r["label"].replace("\n", "   |   "), fontsize=10, weight="bold", va="top")
            ax.text(0, 10.75, "query: " + short(r["question"], 200), fontsize=8, va="top", color="#333333")
            ax.text(0, 9.95, f"rank: cosine {r.get('dense')}  |  scorer {r.get('scorer')}  |  +OpenIE graph {r.get('openie_fused')}  |  +SciAffordGraph, gate off {r.get('off_fused')}  |  +CCMP {r.get('fused')}      graph channel alone: {r.get('off_graph')} (gate off)  ->  {r.get('graph')} (with CCMP)",
                    fontsize=8, va="top", color="#333333")
            def chain(p, yc, ylab, seeds, grey=False, tag=""):
                if not p:
                    ax.text(0, ylab, tag + "no route within the reasoner's depth", fontsize=8.5, va="top", color="#888888"); return
                art = not valid(p, seeds)
                hops = p["hops"]; nb = len(hops) + 1; gap = 0.9; w = min(13.5, (98.5 - gap * (nb - 1)) / nb); x0 = 0.5
                names = [h["head"] for h in hops] + [hops[-1]["tail"]]
                lab = tag + f"w = {p['weight']:.2f}" + (f"  ({r['off_w']:.2f} with the gate off)" if r.get("off_w") is not None and not grey else "")
                if art: lab += "   [first hop is not a seed frame: placeholder artefact, not a route]"
                ax.text(0, ylab, lab, fontsize=7.8, va="top", color="#888888" if grey else "#111111")
                for j, nm in enumerate(names):
                    t = ntype(nm, docs); gold = (j == nb - 1); x = x0 + j * (w + gap)
                    ax.add_patch(FancyBboxPatch((x, yc - 1.1), w, 2.2, boxstyle="round,pad=0.12",
                                                fc="#f0f0f0" if grey else FILL.get("gold" if gold else t, "#f3f3f3"),
                                                ec="#1f4e9a" if gold else ("#999999" if grey else "#555555"), lw=1.8 if gold else 0.8))
                    ax.text(x + w / 2, yc, node_text(nm, docs, 20 if w > 11 else 16), fontsize=6.4 if w > 11 else 5.8, ha="center", va="center", color="#666666" if grey else "black")
                    if j < nb - 1:
                        h = hops[j]; g = h.get("gate"); xm = x + w + gap / 2
                        col = "#333333" if g is None or abs(g - 1) <= 0.05 else ("#d9701a" if g > 1 else "#1f6fb4")
                        ax.annotate("", xy=(x + w + gap, yc), xytext=(x + w, yc), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#999999" if grey else "#333333"))
                        ax.text(xm, yc + 1.25, short(h["rel"].replace("inverse_", "inv. ").replace("_", " "), 20), fontsize=6.2, ha="center", va="bottom", style="italic", color="#666666" if grey else "#222222")
                        if g is not None and not grey: ax.text(xm, yc - 1.25, f"gate {g:.2f}", fontsize=6.2, ha="center", va="top", color=col, weight="bold" if abs(g - 1) > 0.05 else "normal")
            chain(r.get("path"), 6.6, 9.0, set(r.get("seeds", [])), tag="SciAffordGraph + CCMP, top route:   ")
            chain(r.get("openie_path"), 2.0, 4.35, set(r.get("openie_seeds", [])), grey=True, tag="OpenIE graph, top route:   ")
        fig.suptitle("Path interpretations (page %d/%d): boxes are typed frames, arrows carry the relation, the CCMP gate on the sender is orange when > 1.05 and blue when < 0.95" % (pg, len(pages)), fontsize=9.5, y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        for ext in ("png", "pdf"): fig.savefig(f"{out}_routes_{pg}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)
    return len(pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--pick", default=None)
    ap.add_argument("--queries", required=True); ap.add_argument("--docs", required=True); ap.add_argument("--quartet", default=None)
    ap.add_argument("--pred", action="append", default=[]); ap.add_argument("--out", required=True); ap.add_argument("--prefix", default="hops_pick_")
    ap.add_argument("--candidates", default=None, help="showcase_<dataset>_candidates.json: baseline ranks per gold (instead of --pred files)")
    ap.add_argument("--openie-fused", action="store_true", help="also plot the OpenIE model's fused rank (a different trained model, its own scorer)")
    a = ap.parse_args()
    global OPENIE_FUSED; OPENIE_FUSED = a.openie_fused
    docs = json.load(open(a.docs)); queries = {q["id"]: q for q in json.load(open(a.queries))}
    arms = {k: load_arm(f"{a.dir}/{a.prefix}{k}.json") for k, _ in ARMS if os.path.exists(f"{a.dir}/{a.prefix}{k}.json")}
    assert "frame_ccmp" in arms, f"{a.dir}/{a.prefix}frame_ccmp.json is missing"
    qf = {}
    if a.quartet and os.path.exists(a.quartet):
        for x in json.load(open(a.quartet)):
            for d, m in (x.get("quartet", {}).get("per_document") or {}).items(): qf[(x["id"], d)] = m
    cand_base = {}
    if a.candidates and os.path.exists(a.candidates):
        for c in json.load(open(a.candidates))["candidates"]:
            cand_base[(c["id"], c["gold"])] = {k: v for k, v in c["ranks"].items() if k in ("qwen3", "bge", "reasonir", "bm25", "specter2", "scincl")}
    preds = {}
    for spec in a.pred:
        nm, path = spec.split("=", 1)
        if os.path.exists(path): preds[nm] = {r["id"]: ranked_docs(r) for r in json.load(open(path))}
    if a.pick and os.path.exists(a.pick):
        pk = json.load(open(a.pick)); order = [(q, g) for q, gs in pk.items() for g in gs if (q, g) in arms["frame_ccmp"]]
    else:
        order = list(arms["frame_ccmp"])
    rows, md = [], [f"# {a.dataset}: showcase 2, hand-picked path interpretations ({len(order)} pairs)", ""]
    for q, g in order:
        on = arms["frame_ccmp"][(q, g)]; off = arms.get("frame_ccmp_off", {}).get((q, g)); oi = arms.get("openie", {}).get((q, g))
        m = qf.get((q, g), {}); fp = m.get("field_pair"); st = m.get("stratum") or on["stratum"]
        p = next((p for p in on["paths"] if valid(p, set(on["seeds"]))), on["paths"][0] if on["paths"] else None)
        off_w = None
        if off and p:
            same = [pp for pp in off["paths"] if [(h["head"], h["rel"], h["tail"]) for h in pp["hops"]] == [(h["head"], h["rel"], h["tail"]) for h in p["hops"]]]
            off_w = same[0]["weight"] if same else None
        base = dict(cand_base.get((q, g), {}))
        for nm, tab in preds.items():
            lst = tab.get(q); base[nm] = (lst.index(g) + 1) if lst and g in lst else (BIG if lst else None)
        row = {"q": q, "g": g, "label": f"{short(docs.get(g, g).split('. ')[0], 48)}\n{short(fp or st or '', 40)}",
               "question": queries.get(q, {}).get("question", on["question"]), "dense": on["rank"].get("dense"), "scorer": on["rank"].get("scorer"),
               "graph": on["rank"].get("graph"), "fused": on["rank"].get("fused"), "off_fused": off["rank"].get("fused") if off else None,
               "off_graph": off["rank"].get("graph") if off else None, "openie_fused": oi["rank"].get("fused") if oi else None,
               "openie_graph": oi["rank"].get("graph") if oi else None, "path": p, "off_w": off_w, "seeds": on["seeds"], "base": base,
               "openie_path": (oi["paths"][0] if oi and oi["paths"] else None), "openie_seeds": oi["seeds"] if oi else []}
        rows.append(row)
        md += [f"## {docs.get(g, g).split('. ')[0][:100]}", f"query {q} -> gold {g} | QUARTET {fp} / {st} | shortest route {on.get('min_hops')} hops", "",
               f"**Query:** {row['question']}", "", f"**Gold:** {short(docs.get(g, g), 600)}", "",
               f"**Ranks:** cosine {row['dense']}, scorer {row['scorer']}, +OpenIE {row['openie_fused']} (graph {row['openie_graph']}), "
               f"+SciAffordGraph gate off {row['off_fused']} (graph {row['off_graph']}), +CCMP {row['fused']} (graph {row['graph']})"
               + ("; baselines " + ", ".join(f"{SYS.get(k, k)} {v if v < BIG else '>300'}" for k, v in base.items() if v) if base else ""), "",
               "**Seeds:** " + "; ".join(on["seeds"]), ""]
        for v in on["views"]: md.append(f"- view {v['view']} match {v['match']:.3f}: {v.get('text')}")
        for k, lab in ARMS:
            t = arms.get(k, {}).get((q, g))
            if not t: continue
            md.append(f"\n**{lab}** ranks {t['rank']}")
            for pp in t["paths"]:
                flag = "" if valid(pp, set(t["seeds"])) else " [ARTEFACT: not from a seed]"
                md.append(f"- w={pp['weight']:.2f}{flag}: " + " -> ".join(f"{node_text(h['head'], docs, 200).replace(chr(10), ' ')} --{h['rel']}" + (f" (gate {h['gate']:.2f})" if 'gate' in h else "") for h in pp["hops"]) + f" --> {node_text(pp['hops'][-1]['tail'], docs, 200).replace(chr(10), ' ')}")
        md.append("")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    draw_ladder(rows, a.out, f"{a.dataset}: rank of the gold at each stage of the cumulative ablation ({len(rows)} hand-picked pairs)")
    n = draw_routes(rows, a.out, docs)
    print(f"wrote {a.out}.md, _ladder.png/.pdf, _routes_1..{n}.png/.pdf")


if __name__ == "__main__":
    main()
