#!/usr/bin/env python3
"""
route_matrix_tikz.py -- the routes SciGraphIR takes for one (query, gold) pair, drawn in the idiom of
figures/fig_equivalent_decompositions.tex: two text cards on top (research question, i* abstract) with short
spans highlighted in the colour of the node type they ground to, then a `matrix of nodes` grid whose rows are
the routes (the reasoner's top routes, plus the OpenIE entity graph's route for contrast) and whose columns
are hops; q and i* are the endpoint circles. Ranks and baselines go in the legend line.

  python3 eval/route_matrix_tikz.py --dataset sir4_cs --dir results/qualitative/drive_scan_sir4_cs --prefix hops_ \\
      --pair 10.48550_arxiv.2603.03985=10.1111/j.1749-6632.2010.05443.x --n-routes 2 \\
      --question "How can we ...?" --survey "[...] do not systematically measure memory degradation over time ..." \\
      --doc "Memory consolidation refers to ..." --field-q "Streaming video understanding" --field-d "Neuroscience" \\
      --span "R1H1=measure memory degradation over time" --span "R1H2=return to a transient unstable state" \\
      --out figures/fig_route_matrix_reconsolidation --compile

Node ids for --span are R<row>H<hop> (row 1 = route 1, hop 1 = the seed frame); run once without --span to
see the ids printed. A span's text must occur verbatim in one of the three texts; it is highlighted there
in the colour of that node's type with a superscript number, and the same number badges the cell.
"""
import argparse, collections, csv, json, os, re, shutil, subprocess
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TYPE_COL = {"function": "cFun", "finding": "cFin", "method": "cMet", "mechanism": "cMet", "limitation": "cLim", "task": "cLim", "domain": "cDom", "paper": "cDoc", "entity": "cEnt"}


def tex(s):
    return re.sub(r"([&%$#_{}])", r"\\\1", str(s)).replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")


def ntype(n, docs):
    if n in docs: return "paper"
    return n[1:n.index("]")] if n.startswith("[") and "]" in n else "entity"


def rel(r):
    inv = r.startswith("inverse_"); r = r[8:] if inv else r
    return "\\textsc{" + tex(r.replace("_", " ")) + "}" + ("$^{-1}$" if inv else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--prefix", default="hops_")
    ap.add_argument("--pair", required=True); ap.add_argument("--n-routes", type=int, default=2); ap.add_argument("--out", required=True)
    ap.add_argument("--question", default=None); ap.add_argument("--survey", default=None); ap.add_argument("--doc", default=None)
    ap.add_argument("--field-q", default=""); ap.add_argument("--field-d", default=""); ap.add_argument("--year", default=None)
    ap.add_argument("--span", action="append", default=[]); ap.add_argument("--no-openie", action="store_true")
    ap.add_argument("--candidates", default=None); ap.add_argument("--compile", action="store_true"); ap.add_argument("--max-hops", type=int, default=6)
    a = ap.parse_args(); q, g = a.pair.split("=", 1); D = a.dataset
    docs = json.load(open(f"{R}/retriever/data/{D}_test/raw/documents.json"))
    queries = {x["id"]: x for x in json.load(open(f"{R}/retriever/data/{D}_test/raw/test.json"))}
    qf = {}
    qp = f"{R}/benchmark/data.nosync/benchmark/{D.replace('sir4_', '')}_test_final/eval.json"
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
    def simple(p, seeds):
        h = p["hops"]
        if not h or h[0]["head"] not in seeds or not all(x["tail"] == y["head"] for x, y in zip(h, h[1:])): return False
        ns = [h[0]["head"]] + [x["tail"] for x in h]; return len(ns) == len(set(ns)) and h[-1]["tail"] == g and len(h) <= a.max_hops
    routes = [p for p in on[1]["paths"] if simple(p, set(on[0]["seeds"]))][: a.n_routes]
    assert routes, "no seed-origin simple route recorded"
    key = lambda p: tuple((h["head"], h["rel"], h["tail"]) for h in p["hops"])
    offw = {key(p): p["weight"] for p in (off[1]["paths"] if off else [])}
    orow = None
    if oie and not a.no_openie:
        orow = next((p for p in oie[1]["paths"] if simple(p, set(oie[0]["seeds"]))), None)
    rows = [("route %d" % (i + 1), p, False) for i, p in enumerate(routes)] + ([("entity graph", orow, True)] if orow else [])
    oie_none = bool(oie) and not a.no_openie and orow is None
    ncol = max(len(p["hops"]) for _, p, _ in rows)            # hops = nodes before the gold
    import math
    tw = 26 if ncol <= 4 else (22 if ncol <= 5 else 19)
    def est_lines(nd):
        tp = ntype(nd, docs); txt = docs[nd].split(". ")[0][:70] if tp == "paper" else (nd[nd.index("]") + 2:] if nd.startswith("[") else nd)
        return 1 + math.ceil(len(txt) / max(8, tw / 1.75)) + (1 if tp == "paper" else 0)
    rowh = []
    for _, p, _ in rows:
        chain = [p["hops"][0]["head"]] + [h["tail"] for h in p["hops"]][:-1]
        rowh.append(max(17.0, max(3.4 * est_lines(nd) + 6 for nd in chain)))
    base = {}
    if a.candidates and os.path.exists(a.candidates):
        for c in json.load(open(a.candidates))["candidates"]:
            if c["id"] == q and c["gold"] == g: base = {k: v for k, v in c["ranks"].items() if k in ("bm25", "bge", "qwen3", "reasonir")}
    # ---- node ids and spans
    ids = {}
    for ri, (_, p, _) in enumerate(rows, 1):
        chain = [p["hops"][0]["head"]] + [h["tail"] for h in p["hops"]][:-1]
        for hi, nd in enumerate(chain, 1): ids[f"R{ri}H{hi}"] = nd
    print("node ids:"); [print(f"  {k:6} {ntype(v, docs):10} {v[:80]}") for k, v in ids.items()]
    spans = []
    for s in a.span:
        nid, txt = s.split("=", 1); assert nid in ids, f"unknown node id {nid}"; spans.append((nid, txt))
    num = {nid: i + 1 for i, (nid, _) in enumerate(dict.fromkeys(n for n, _ in spans).items())}
    def mark(text):
        out = tex(text)
        for nid, txt in spans:
            t = tex(txt)
            if t in out: out = out.replace(t, "\\spanmark{%s}{%d}{%s}" % (TYPE_COL[ntype(ids[nid], docs)], num[nid], t), 1)
        return out
    question = a.question or queries.get(q, {}).get("question", on[0]["question"]).split("?")[0] + "?"
    _sents = docs[g].split(". ")
    doc_text = a.doc or ((". ".join(x.rstrip(".") for x in _sents[1:3]) + ".") if len(_sents) > 1 else "(title only: no abstract in the corpus)")
    year = a.year or qf.get((q, g), {}).get("year") or ""
    gtitle = docs[g].split(". ")[0]
    # ---- TeX
    L = [r"""% generated by eval/route_matrix_tikz.py; idiom of figures/fig_equivalent_decompositions.tex
% Build:  tectonic -X compile <this file> --outfmt pdf
\documentclass[tikz,border=8pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{soul}
\usetikzlibrary{positioning,fit,matrix,arrows.meta,calc}
\definecolor{ink}   {RGB}{ 38, 42, 48}
\definecolor{muted} {RGB}{124,130,140}
\definecolor{rule}  {RGB}{206,210,218}
\definecolor{paperf}{RGB}{247,248,250}
\definecolor{cFun}{RGB}{ 32,128,112}\definecolor{cFunl}{RGB}{202,230,223}
\definecolor{cFin}{RGB}{164,120, 34}\definecolor{cFinl}{RGB}{245,231,199}
\definecolor{cMet}{RGB}{ 42,105,160}\definecolor{cMetl}{RGB}{206,224,241}
\definecolor{cLim}{RGB}{112, 82,150}\definecolor{cLiml}{RGB}{222,212,238}
\definecolor{cDom}{RGB}{110,116,126}\definecolor{cDoml}{RGB}{236,237,240}
\definecolor{cDoc}{RGB}{196, 78, 82}\definecolor{cDocl}{RGB}{250,235,236}
\definecolor{cEnt}{RGB}{124,130,140}\definecolor{cEntl}{RGB}{240,241,243}
\definecolor{cGate}{RGB}{217,112, 26}
\newcommand{\bnum}[2]{\begingroup\setlength{\fboxsep}{1.3pt}\colorbox{#1}{\textcolor{white}{\scriptsize\bfseries #2}}\endgroup}
\newcommand{\spanmark}[3]{{\sethlcolor{#1l}\hl{#3}}\textsuperscript{\textcolor{#1}{\bfseries #2}}}
\newcommand{\ty}{\fontsize{6.5}{7.5}\selectfont\scshape\bfseries}
\begin{document}
\begin{tikzpicture}[
  >={Stealth[length=2.4mm,width=1.9mm]}, line width=0.9pt, draw=ink,
  endp/.style = {circle, draw=ink, fill=white, line width=1.1pt, minimum size=9.5mm, inner sep=0pt, font=\normalsize},
  chip/.style = {draw=rule, fill=paperf, rounded corners=2pt, align=center, text width=%(tw)dmm, inner sep=1.5mm,
                 minimum height=17mm, font=\scriptsize, text=ink, execute at begin node={\hyphenpenalty=10000\exhyphenpenalty=10000}},
  blank/.style= {draw=none, fill=none, minimum height=17mm, text width=%(tw)dmm},
  hd/.style   = {draw=none, fill=none, align=center, text width=%(tw)dmm, inner sep=1mm, minimum height=0mm, font=\scriptsize\itshape, text=muted},
  card/.style = {draw=ink, fill=white, rounded corners=2.5pt, align=left, text width=165mm, inner sep=3.6mm, font=\small},
  rlbl/.style = {font=\footnotesize, text=ink, inner sep=0pt, align=right},
  rel/.style  = {font=\fontsize{6.5}{7.5}\selectfont, text=ink, inner sep=1pt, fill=white, align=center},
  gate/.style = {font=\fontsize{6.5}{7.5}\selectfont, text=cGate, inner sep=1pt, fill=white},
  ]""".replace("%(tw)d", str(26 if ncol <= 4 else (22 if ncol <= 5 else 19)))]
    # matrix
    cells = []
    hdr = " & ".join("|[hd]| {%s}" % ("seed frame" if c == 1 else f"hop {c}") for c in range(1, ncol + 1)) + " \\\\"
    cells.append(hdr)
    for ri, (lab, p, grey) in enumerate(rows, 1):
        chain = [p["hops"][0]["head"]] + [h["tail"] for h in p["hops"]][:-1]
        cs = []
        for hi in range(1, ncol + 1):
            if hi > len(chain): cs.append("|[blank]| {}"); continue
            nd = chain[hi - 1]; tp = ntype(nd, docs); col = TYPE_COL[tp]; nid = f"R{ri}H{hi}"
            badge = ("\\bnum{%s}{%d}\\ " % (col, num[nid])) if nid in num else ""
            if tp == "paper":
                yr = qf.get((q, nd), {}).get("year") or ""
                ttl = docs[nd].split(". ")[0]
                cut = len(ttl) > 70
                if cut: ttl = ttl[:70].rsplit(" ", 1)[0]
                body = "\\textcolor{muted}{\\ty paper}\\\\ %s%s%s" % (badge, tex(ttl), "\\,\\ldots" if cut else "") + ("\\\\[1pt]\\textcolor{muted}{%s}" % yr if yr else "")
            else:
                name = nd[nd.index("]") + 2:] if nd.startswith("[") and "]" in nd else nd
                body = "{\\ty\\color{%s}%s}\\\\ %s%s" % (col, tp if tp != "entity" else "entity", badge, tex(name))
            style = ("chip, draw=%s, line width=1.0pt" % col if not grey else "chip, draw=rule, text=muted") + ", minimum height=%.1fmm" % rowh[ri - 1]
            cs.append("|[%s]| {%s}" % (style, body))
        cells.append(" & ".join(cs) + " \\\\")
    if oie_none:
        cells.append("|[chip, draw=rule, dashed, text=muted, minimum height=17mm]| {{\\ty entity graph}\\\\ no seed-to-gold route within the reasoner's depth}" + " & |[blank]| {}" * (ncol - 1) + " \\\\")
    L.append("\\matrix (m) [matrix of nodes, nodes={chip}, row sep=9mm, column sep=7mm] {\n  " + "\n  ".join(cells) + "\n};")
    # arrows within rows, with relation labels (emitted after the endpoints exist)
    ARR = []
    for ri, (lab, p, grey) in enumerate(rows, 1):
        mr = ri + 1
        for hi, h in enumerate(p["hops"], 1):
            gt = h.get("gate"); has_gate = gt is not None and abs(gt - 1) > 0.05 and not grey
            style = "draw=muted" if grey else "draw=ink"; up = rowh[ri - 1] / 2 + 1.2
            if hi < len(p["hops"]):
                gl = " node[gate, below=%.1fmm, pos=.5]{gate %.2f}" % (up, gt) if has_gate else ""
                ARR.append("\\draw[->, %s] (m-%d-%d.east) -- node[rel, above=%.1fmm, pos=.5]{%s}%s (m-%d-%d.west);" % (style, mr, hi, up, rel(h["rel"]), gl, mr, hi + 1))
            else:
                gl = " node[gate, below=2.2mm, pos=.3]{gate %.2f}" % gt if has_gate else ""
                ARR.append("\\draw[->, %s] (m-%d-%d.east) to[out=0, in=180] node[rel, above=0.6mm, pos=.4]{%s}%s (hs);" % (style, mr, hi, rel(h["rel"]), gl))
    # row labels, endpoints
    for ri, (lab, p, grey) in enumerate(rows, 1):
        w = p["weight"]; wo = offw.get(key(p))
        sub = ("weight %.1f" % w + (" (%.1f off)" % wo if wo is not None else "")) if not grey else ("weight %.2f" % abs(w))
        L.append("\\node[rlbl, left=4mm of m-%d-1] (r%d) {%s\\\\[1pt]{\\scriptsize\\color{muted}%s}};" % (ri + 1, ri, tex(lab) if not grey else "entity graph\\\\{\\scriptsize\\color{muted}(OpenIE)}", sub))
    if oie_none:
        L.append("\\node[rlbl, left=4mm of m-%d-1] (r%d) {entity graph\\\\{\\scriptsize\\color{muted}(OpenIE)}};" % (len(rows) + 2, len(rows) + 1))
    nr = len(rows) + (1 if oie_none else 0)
    L.append("\\coordinate (lmid) at ($(r1.west)!0.5!(r%d.west)$);" % nr)
    L.append("\\node[endp] (qn) at ([xshift=-13mm] lmid) {$q$};")
    L.append("\\coordinate (rmid) at ($(m-2-%d.east)!0.5!(m-%d-%d.east)$);" % (ncol, nr + 1, ncol))
    L.append("\\node[endp] (hs) at ([xshift=24mm] rmid) {$i^{\\star}$};")
    for ri in range(1, nr + 1):
        ang = 0 if nr == 1 else int(28 - 56 * (ri - 1) / (nr - 1))
        L.append("\\draw[->, draw=muted] (qn) to[out=%d, in=180] (r%d.west);" % (ang, ri))
    L += ARR
    # cards
    L.append("\\node[fit=(qn)(hs)(m), inner sep=0pt] (span) {};")
    L.append("\\node[card, anchor=south] (target) at ([yshift=5mm] span.north) {\\textbf{$i^{\\star}$}\\ \\ \\textcolor{muted}{%s}\\ \\ \\emph{%s}%s\\\\[2pt] ``%s''};"
             % (tex(a.field_d), tex(gtitle), (" (%s)" % year if year else ""), mark(doc_text)))
    qcard = "\\textbf{Research Question}\\ \\ \\textcolor{muted}{%s}\\ \\ ``%s''" % (tex(a.field_q), mark(question))
    if a.survey: qcard += "\\\\[3pt]\\textbf{Background Survey}\\ \\ ``%s''" % mark(a.survey)
    L.append("\\node[card, anchor=south, fill=paperf] (query) at ([yshift=3mm] target.north) {%s};" % qcard)
    # legend
    rk = on[1]["rank"]; rko = off[1]["rank"] if off else {}; rki = oie[1]["rank"] if oie else {}
    lab = {"bm25": "BM25", "bge": "BGE-large", "qwen3": "Qwen3-Embedding", "reasonir": "ReasonIR-8B"}
    bl = ", ".join(f"{lab[k]} {('$>$300' if v >= 10**6 else v)}" for k, v in base.items() if v)
    L.append("\\node[below=5mm of m.south, anchor=north, text=muted, align=center, text width=170mm, font=\\footnotesize] (leg) "
             "{Rows are the reasoner's top-weighted routes from a seed frame of $q$ to $i^{\\star}$; the last row is the OpenIE entity graph%s. "
             "Rank of $i^{\\star}$: SciAffordGraph channel \\textbf{%s} (%s with the CCMP gate off), \\textsc{SciGraphIR} \\textbf{%s} (%s off), multi-view scorer %s, Qwen3 cosine %s%s; entity-graph channel %s.%s};"
             % ("'s route on the same corpus" if orow else " built from the same corpus, which has no route to $i^{\\star}$", rk.get("graph"), rko.get("graph", "--"), rk.get("fused"), rko.get("fused", "--"), rk.get("scorer"), rk.get("dense"), ("; " + bl) if bl else "", rki.get("graph", "--"),
                " Orange: the CCMP gate on the hop's sender where it differs from 1." if any(abs(h.get("gate", 1) - 1) > 0.05 for _, p, gr in rows if not gr for h in p["hops"]) else ""))
    L.append("\\end{tikzpicture}\n\\end{document}")
    open(a.out + ".tex", "w").write("\n".join(L) + "\n"); print("wrote", a.out + ".tex")
    if a.compile:
        pr = subprocess.run(["tectonic", "-X", "compile", a.out + ".tex", "--outfmt", "pdf"], capture_output=True, text=True)
        if pr.returncode != 0: print(pr.stdout[-2500:], pr.stderr[-2500:]); raise SystemExit("tectonic failed")
        if shutil.which("pdftoppm"): subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", a.out + ".pdf", a.out + "_preview"]); print("wrote", a.out + "_preview.png")


if __name__ == "__main__":
    main()
