#!/usr/bin/env python3
"""
route_tikz.py -- the routes SciGraphIR takes for one (query, gold) pair, as a TikZ figure in the thesis style
(fig_graph_cs_creativity.tex): the query at the top, one column per top-weighted route, typed affordance nodes, the
relation on every edge, the CCMP gate where it differs from 1, the gold at the bottom, and the rank ladder
+ path weights (with / without the gate) underneath.

  python3 eval/route_tikz.py --dataset sir4_cs --dir results/qualitative/drive_scan_sir4_cs --prefix hops_ \\
      --pair 10.48550_arxiv.2603.03985=10.1111/j.1749-6632.2010.05443.x --n-routes 2 \\
      --query-short "How can the retrospective memory of a streaming-video language model be measured?" \\
      --gold-short "memory that persists returns to an unstable state on retrieval and must re-stabilise" \\
      --out results/qualitative/showcase2_drive/tikz_reconsolidation --compile

Writes <out>.tex (a \\begin{figure} block to \\input, needs xcolor + tikz{positioning,arrows.meta,calc} + float)
and, with --compile, <out>_standalone.pdf/.png for preview (tectonic + pdftoppm).
"""
import argparse, collections, csv, json, os, re, subprocess, shutil, textwrap
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STYLE = {"function": "fun", "method": "met", "paper": "pap", "domain": "dom", "finding": "fin", "limitation": "lim", "task": "tsk", "mechanism": "met", "entity": "ent"}


def tex(s):
    return re.sub(r"([&%$#_{}])", r"\\\1", str(s)).replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")


def ntype(n, docs):
    if n in docs: return "paper"
    return n[1:n.index("]")] if n.startswith("[") and "]" in n else "entity"


def body(n, docs, dom):
    if n in docs:
        return "\\emph{" + tex(docs[n].split(". ")[0][:95]) + "}"
    return tex(n[n.index("]") + 2:] if n.startswith("[") and "]" in n else n)


def rel(r):
    inv = r.startswith("inverse_"); r = r[8:] if inv else r
    return tex(r.replace("_", " ")) + ("$^{-1}$" if inv else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--prefix", default="hops_")
    ap.add_argument("--pair", required=True); ap.add_argument("--n-routes", type=int, default=2); ap.add_argument("--out", required=True)
    ap.add_argument("--query-short", default=None); ap.add_argument("--gold-short", default=None); ap.add_argument("--label", default=None)
    ap.add_argument("--candidates", default=None); ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-openie", action="store_true", help="do not draw the OpenIE graph's route as a comparison column")
    a = ap.parse_args(); q, g = a.pair.split("=", 1); D = a.dataset
    docs = json.load(open(f"{R}/retriever/data/{D}_test/raw/documents.json"))
    queries = {x["id"]: x for x in json.load(open(f"{R}/retriever/data/{D}_test/raw/test.json"))}
    dom = collections.defaultdict(list)
    with open(f"{R}/retriever/data/{D}_test_v16sc/processed/stage1/edges.csv", newline="") as fh:
        rd = csv.reader(fh); next(rd)
        for row in rd:
            if len(row) >= 3 and row[1] == "in_field": dom[row[0]].append(row[2].replace("[domain] ", ""))
    qf = {}
    qp = f"{R}/sir-4/data/benchmark/{D.replace('sir4_', '')}_test_final/eval.json"
    if os.path.exists(qp):
        for x in json.load(open(qp)):
            for d, m in (x.get("quartet", {}).get("per_document") or {}).items(): qf[(x["id"], d)] = m
    def load(name):
        p = f"{a.dir}/{a.prefix}{name}.json"
        if not os.path.exists(p): return None
        for r in json.load(open(p)):
            if r["id"] == q:
                for t in r["targets"]:
                    if t["doc"] == g: return r, t
        return None
    on = load("frame_ccmp"); off = load("frame_ccmp_off"); oie = load("openie")
    assert on, "pair not in the CCMP hops file"
    r, t = on; seeds = set(r["seeds"])
    def simple(p):
        h = p["hops"]
        if not h or h[0]["head"] not in seeds or not all(x["tail"] == y["head"] for x, y in zip(h, h[1:])): return False
        ns = [h[0]["head"]] + [x["tail"] for x in h]; return len(ns) == len(set(ns))
    routes = [p for p in t["paths"] if simple(p)][: a.n_routes]
    assert routes, "no seed-origin simple route recorded for this pair"
    key = lambda p: tuple((h["head"], h["rel"], h["tail"]) for h in p["hops"])
    offw = {key(p): p["weight"] for p in (off[1]["paths"] if off else [])}
    base = {}
    if a.candidates and os.path.exists(a.candidates):
        for c in json.load(open(a.candidates))["candidates"]:
            if c["id"] == q and c["gold"] == g: base = {k: v for k, v in c["ranks"].items() if k in ("bm25", "bge", "qwen3", "reasonir", "specter2", "scincl")}
    # ---- the OpenIE graph's top route for the same pair (comparison column)
    oroute = None; oseeds = set()
    if oie and not a.no_openie:
        oseeds = set(oie[0]["seeds"])
        def osimple(p):
            h = p["hops"]
            if not h or h[0]["head"] not in oseeds or not all(x["tail"] == y["head"] for x, y in zip(h, h[1:])): return False
            ns = [h[0]["head"]] + [x["tail"] for x in h]; return len(ns) == len(set(ns)) and h[-1]["tail"] == g
        oroute = next((p for p in oie[1]["paths"] if osimple(p)), None)
    draw_oie = bool(oie) and not a.no_openie
    # ---- geometry: affordance representation routes left/centre, the OpenIE column (if any) on the right
    n = len(routes); ncol = n + (1 if draw_oie else 0)
    xs = {1: [0.0], 2: [-4.6, 4.6], 3: [-6.3, 0.0, 6.3], 4: [-7.8, -2.6, 2.6, 7.8]}[min(ncol, 4)]
    tw = {1: 56, 2: 56, 3: 44, 4: 36}[min(ncol, 4)]; dy = 2.35
    qtext = a.query_short or (queries.get(q, {}).get("question", r["question"]).split("?")[0] + "?")
    import math
    qlines = max(2, math.ceil(len(qtext) / 58)); qs = -(0.35 + 0.215 * qlines)     # estimated y of q.south (96mm box, \small)
    yh = qs - 1.2; y0 = qs - 2.2                                                     # headers just under the seed line, first row under the headers
    maxlen = max([len(p["hops"]) for p in routes] + ([len(oroute["hops"])] if oroute else [1]))
    ygold = y0 - dy * (maxlen - 1) - 3.3
    gtitle = docs.get(g, g).split(". ")[0]; gdom = "; ".join(dom.get(g, [])) or (qf.get((q, g), {}).get("field_pair") or "").split("->")[-1].strip()
    year = qf.get((q, g), {}).get("year")
    L = []
    L.append("% generated by eval/route_tikz.py from " + a.dir + "/" + a.prefix + "frame_ccmp.json (CCMP gate on) and " + a.prefix + "frame_ccmp_off.json;\n"
             "% the routes are the reasoner's top-weighted seed-origin paths (gradient beam search), nothing else is drawn.\n"
             "% Preamble needs: \\usepackage{tikz} \\usetikzlibrary{positioning,arrows.meta,calc} \\usepackage{xcolor} \\usepackage{float}")
    L.append(r"""\providecolor{hlblue}{RGB}{214,228,247}
\providecolor{hlgreen}{RGB}{214,240,224}
\providecolor{hlorange}{RGB}{252,232,200}
\providecolor{hlpurple}{RGB}{228,220,246}
\providecolor{hlgrey}{RGB}{236,236,238}
\providecolor{hlyellow}{RGB}{250,243,205}
\providecommand{\ty}[1]{{\scriptsize\scshape\bfseries\color{black!75}#1}}
\begin{figure}[H]
\centering
\begin{tikzpicture}[
  font=\small, every node/.style={align=left},
  fr/.style={draw=black!70, line width=0.5pt, rounded corners=2.5pt, inner sep=5pt, text width=%dmm},
  fun/.style={fr, fill=hlgreen}, met/.style={fr, fill=hlpurple}, pap/.style={fr, fill=hlblue},
  dom/.style={fr, fill=hlgrey}, fin/.style={fr, fill=hlyellow}, lim/.style={fr, fill=hlorange}, tsk/.style={fr, fill=hlorange},
  ent/.style={fr, fill=white, draw=black!45, text=black!70}, hdr/.style={font=\footnotesize\scshape, color=black!60, inner sep=1pt},
  qry/.style={fr, fill=white, dashed, text width=96mm, align=center},
  gold/.style={fr, fill=hlblue, line width=1.1pt, text width=96mm, align=center},
  e/.style={-{Latex[length=2.2mm]}, line width=0.7pt, color=black!80},
  seed/.style={-{Latex[length=2mm]}, line width=0.6pt, color=black!45, dashed},
  rel/.style={font=\footnotesize\itshape, inner sep=2pt},
  gate/.style={font=\scriptsize, color=orange!70!black, inner sep=2pt},
]""" % tw)
    L.append("%% ---- query (top)\n\\node[qry] (q) at (0,0) {\\ty{query (%s)}\\\\ ``%s''};" % (tex(D.replace("sir4_", "SIR-4 ").replace("cs", "computer science")), tex(qtext)))
    names = {}
    for i, p in enumerate(routes):
        x = xs[i]; chain = [p["hops"][0]["head"]] + [h["tail"] for h in p["hops"]][:-1]     # nodes before the gold
        L.append(f"% ---- route {i + 1}")
        for k, nd in enumerate(chain):
            nid = f"r{i}n{k}"; names[(i, k)] = nid; tp = ntype(nd, docs); sty = STYLE.get(tp, "ent")
            head = tp + (f" ({tex(dom[nd][0])})" if tp == "paper" and dom.get(nd) else "")
            L.append(f"\\node[{sty}] ({nid}) at ({x:.1f},{y0 - dy * k:.2f}) {{\\ty{{{head}}}\\\\ {body(nd, docs, dom)}}};")
    L.append(f"% ---- gold (bottom)\n\\node[gold] (g) at (0,{ygold:.2f}) {{\\ty{{gold paper{(' (' + tex(gdom) + ')') if gdom else ''}{(', ' + str(year)) if year else ''}}}\\\\ \\emph{{{tex(gtitle)}}}"
             + (f"\\\\[1pt] {{\\footnotesize {tex(a.gold_short)}}}" if a.gold_short else "") + "};")
    if draw_oie:
        xo = xs[n]; L.append("% ---- OpenIE entity graph (comparison column)")
        L.append(f"\\node[hdr] at ({xo:.1f},{yh:.2f}) {{OpenIE entity graph}};")
        if oroute:
            ochain = [oroute["hops"][0]["head"]] + [h["tail"] for h in oroute["hops"]][:-1]
            for k, nd in enumerate(ochain):
                nid = f"o{k}"; names[("o", k)] = nid
                ohead = "paper" + (f" ({tex(dom[nd][0])})" if dom.get(nd) else "") if nd in docs else "entity"
                L.append(f"\\node[ent] ({nid}) at ({xo:.1f},{y0 - dy * k:.2f}) {{\\ty{{{ohead}}}\\\\ {body(nd, docs, dom)}}};")
        else:
            L.append(f"\\node[ent, dashed, align=center] (o0) at ({xo:.1f},{y0:.2f}) {{\\ty{{no route}}\\\\ {{\\footnotesize no seed-to-gold route within the reasoner's depth}}}};")
        L.append(f"\\node[hdr] at ({sum(xs[:n]) / n:.1f},{yh:.2f}) {{SciAfford graph}};")
    L.append("% ---- seed edges")
    for i in range(n):
        L.append(f"\\draw[seed] (q.south) -- ++(0,-0.45) -| " + ('node[pos=0.04, right, font=\\scriptsize, color=black!55] {seed nodes of the query} ' if i == 0 else "") + f"({names[(i, 0)]}.north);")
    if draw_oie and oroute:
        L.append(f"\\draw[seed] (q.south) -- ++(0,-0.45) -| node[pos=0.5, above, font=\\scriptsize, color=black!55] {{seed entities}} (o0.north);")
    for i, p in enumerate(routes):
        L.append(f"% ---- route {i + 1} edges")
        side = "right" if xs[i] <= 0 else "left"; gside = "left" if side == "right" else "right"
        for k, h in enumerate(p["hops"]):
            gt = h.get("gate"); has_gate = gt is not None and abs(gt - 1) > 0.05
            if k < len(p["hops"]) - 1:
                gl = f" node[gate, {gside}] {{CCMP gate {gt:.2f}}}" if has_gate else ""
                L.append(f"\\draw[e] ({names[(i, k)]}) -- node[rel, {side}] {{{rel(h['rel'])}}}{gl} ({names[(i, k + 1)]});")
            else:   # the diagonal into the gold: relation early on the edge, gate later, so the two labels never meet
                anchor = f"(g.north -| {xs[i] * 0.7:.1f},0)" if n > 1 else "(g.north)"
                gl = f" node[gate, {gside}, pos=0.72] {{CCMP gate {gt:.2f}}}" if has_gate else ""
                L.append(f"\\draw[e] ({names[(i, k)]}.south) -- node[rel, {side}, pos=0.3] {{{rel(h['rel'])}}}{gl} {anchor};")
    if draw_oie and oroute:
        L.append("% ---- OpenIE route edges")
        xo = xs[n]
        for k, h in enumerate(oroute["hops"]):
            if k < len(oroute["hops"]) - 1:
                L.append(f"\\draw[e, color=black!50] (o{k}) -- node[rel, left, color=black!60] {{{rel(h['rel'])}}} (o{k + 1});")
            else:
                L.append(f"\\draw[e, color=black!50] (o{k}.south) -- node[rel, left, pos=0.3, color=black!60] {{{rel(h['rel'])}}} (g.north -| {xo * 0.55:.1f},0);")
    L.append("\\end{tikzpicture}\n\n\\vspace{4pt}\n{\\footnotesize\n\\begin{tabular}{@{}l l@{}}")
    rk = t["rank"]; rko = off[1]["rank"] if off else {}; rki = oie[1]["rank"] if oie else {}
    L.append("\\textbf{rank of the gold} & Qwen3 cosine \\textbf{%s} $\\rightarrow$ multi-view scorer \\textbf{%s} $\\rightarrow$ {+}\\,SciAfford graph \\textbf{%s} $\\rightarrow$ {+}\\,CCMP \\textbf{%s} \\\\"
             % (rk.get("dense"), rk.get("scorer"), rko.get("fused", "--"), rk.get("fused")))
    L.append("\\textbf{path weights} & " + ";\\; ".join(f"route {i + 1} ({'left' if xs[i] < 0 else ('right' if xs[i] > 0 else 'centre')}) \\textbf{{{p['weight']:.2f}}} with CCMP" + (f", {offw[key(p)]:.2f} without" if key(p) in offw else "") for i, p in enumerate(routes)) + " \\\\")
    L.append("\\textbf{graph channel alone} & SciAfford graph \\textbf{%s} with CCMP, \\textbf{%s} without%s \\\\" % (rk.get("graph"), rko.get("graph", "--"),
             (f";\\; OpenIE entity graph \\textbf{{{rki.get('graph')}}}" + (f" (top route weight {(abs(oroute['weight']) if abs(oroute['weight']) < 0.005 else oroute['weight']):.2f})" if oroute else " (no route)") if rki else "")))
    if base:
        lab = {"bm25": "BM25", "bge": "BGE-large", "qwen3": "Qwen3-Embedding", "reasonir": "ReasonIR-8B", "specter2": "SPECTER2", "scincl": "SciNCL"}
        L.append("\\textbf{baselines} & " + ", ".join(f"{lab[k]} {('$>$300' if v >= 10**6 else v)}" for k, v in base.items() if v) + " \\\\")
    L.append("\\end{tabular}}")
    fp = qf.get((q, g), {}).get("field_pair")
    L.append("\\caption{Routes taken by \\textsc{SciGraphIR} from the query's seed nodes to \\emph{%s}%s. Each column is one of the reasoner's top-weighted routes "
             "(gradient beam search over the per-layer edge weights, as in NBFNet and GFM-RAG); dashed arrows are the seeding step, $r^{-1}$ the inverse relation, "
             "and the CCMP gate on a hop's sender is shown where it differs from 1. Weights are given with CCMP and, where the same route is recorded, with the gate switched off on the same weights. "
             "The right-hand column is the OpenIE entity graph built from the same corpus: its reasoner reaches the paper, when it does, through chains of co-mention rather than through the need the query states.}"
             % (tex(gtitle), (" (" + tex(fp).replace("->", "$\\rightarrow$") + ")") if fp else ""))
    L.append("\\label{fig:%s}\n\\end{figure}" % (a.label or "graph-" + re.sub(r"[^a-z0-9]+", "-", os.path.basename(a.out).lower()).strip("-")))
    open(a.out + ".tex", "w").write("\n".join(L) + "\n"); print("wrote", a.out + ".tex")
    if a.compile:
        wrap = ("\\documentclass[11pt]{article}\\usepackage[margin=8mm,paperwidth=200mm,paperheight=%dmm]{geometry}\\usepackage{xcolor,tikz,float,booktabs}"
                "\\usetikzlibrary{positioning,arrows.meta,calc}\\pagestyle{empty}\\begin{document}\\input{%s}\\end{document}" % (int(abs(ygold) * 10 + 190), os.path.basename(a.out) + ".tex"))
        sa = a.out + "_standalone.tex"; open(sa, "w").write(wrap)
        pr = subprocess.run(["tectonic", "-o", os.path.dirname(a.out) or ".", sa], capture_output=True, text=True)
        if pr.returncode != 0: print(pr.stdout[-3000:], pr.stderr[-3000:]); raise SystemExit("tectonic failed")
        pdf = a.out + "_standalone.pdf"
        if shutil.which("pdftoppm"):
            subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", pdf, a.out + "_standalone"])
            try:                                   # trim the white margins of the tall standalone page
                from PIL import Image, ImageChops
                im = Image.open(a.out + "_standalone.png").convert("RGB")
                bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
                if bbox: im.crop((max(0, bbox[0] - 30), max(0, bbox[1] - 30), min(im.width, bbox[2] + 30), min(im.height, bbox[3] + 30))).save(a.out + "_standalone.png")
            except Exception as e:
                print("(no crop:", e, ")")
            print("wrote", a.out + "_standalone.png")
        print("wrote", pdf)


if __name__ == "__main__":
    main()
