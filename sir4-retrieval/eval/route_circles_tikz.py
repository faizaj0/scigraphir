#!/usr/bin/env python3
"""
route_circles_tikz.py -- the routes SciGraphIR takes for one (query, gold) pair, drawn as a graph of circles:
q grounds into a dashed panel holding EVERY seed frame of the query as a small dot coloured by type (the seeds
that start a drawn route are enlarged and numbered); each route runs left to right as circles coloured by node
type with its label underneath and the relation on the arrow; the OpenIE entity graph's route (or "no route")
is the last row; i* is the endpoint. Text cards with numbered spans on top, ranks in the legend line.
Same CLI as route_matrix_tikz.py (--span ids are R<row>H<hop>).
"""
import argparse, json, math, os, re, shutil, subprocess
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TYPE_COL = {"function": "cFun", "finding": "cFin", "method": "cMet", "mechanism": "cMet", "limitation": "cLim", "task": "cLim", "domain": "cDom", "paper": "cDoc", "entity": "cEnt"}


def frame_arm(d, prefix="hops_", want="auto"):
    """which SciAffordGraph arm the hops files hold: the merged graph (hyb_ccmp, the current SciAffordGraph) when present,
    else the older frame-only graph (frame_ccmp). Returns (arm name, label suffix)."""
    if want == "auto": want = "hyb_ccmp" if os.path.exists(f"{d}/{prefix}hyb_ccmp.json") else "frame_ccmp"
    return want, ("merged graph" if want.startswith("hyb") else "frame-only graph, outdated")


def tex(s):
    s = re.sub(r"([&%$#_{}])", r"\\\1", str(s)).replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
    return s.replace("\u2014", "\\textemdash{}").replace("\u2013", "\\textendash{}").replace("\u2018", "`").replace("\u2019", "'").replace("\u201c", "``").replace("\u201d", "''")


def ntype(n, docs):
    if n in docs: return "paper"
    return n[1:n.index("]")] if n.startswith("[") and "]" in n else "entity"


def rel(r):
    inv = r.startswith("inverse_"); r = r[8:] if inv else r
    w = r.replace("_", " "); lines = [w]
    if len(w) > 13 and " " in w:                       # two lines, split near the middle
        ws = w.split(" "); k = max(1, len(ws) // 2); lines = [" ".join(ws[:k]), " ".join(ws[k:])]
    return "\\\\".join("\\textsc{" + tex(l) + "}" for l in lines) + ("$^{-1}$" if inv else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--prefix", default="hops_")
    ap.add_argument("--pair", required=True); ap.add_argument("--n-routes", type=int, default=2); ap.add_argument("--out", required=True)
    ap.add_argument("--question", default=None); ap.add_argument("--survey", default=None); ap.add_argument("--doc", default=None)
    ap.add_argument("--field-q", default=""); ap.add_argument("--field-d", default=""); ap.add_argument("--year", default=None)
    ap.add_argument("--span", action="append", default=[]); ap.add_argument("--no-openie", action="store_true")
    ap.add_argument("--candidates", default=None); ap.add_argument("--compile", action="store_true"); ap.add_argument("--max-hops", type=int, default=6)
    ap.add_argument("--docs", default=None, help="documents.json (default: the CARGO tree)"); ap.add_argument("--queries", default=None, help="test.json")
    ap.add_argument("--quartet", default=None, help="QUARTET eval.json for the field pair / year (default: the CARGO tree)")
    ap.add_argument("--no-strip", action="store_true", help="omit panel (c), the rank strip; ranks then belong in the path table (route_table_tex.py)")
    ap.add_argument("--arm", default="auto", help="SciAffordGraph arm: auto (hyb_ccmp if present, else frame_ccmp), hyb_ccmp or frame_ccmp")
    a = ap.parse_args(); q, g = a.pair.split("=", 1); D = a.dataset
    docs = json.load(open(a.docs or f"{R}/kg-construction/data/{D}_test/raw/documents.json"))
    queries = {x["id"]: x for x in json.load(open(a.queries or f"{R}/kg-construction/data/{D}_test/raw/test.json"))}
    qf = {}
    qp = a.quartet or f"{R}/quartet/data.nosync/benchmark/{D.replace('sir4_', '')}_test_final/eval.json"
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
    ARM, ARMLAB = frame_arm(a.dir, a.prefix, a.arm); print("SciAffordGraph arm:", ARM, f"({ARMLAB})")
    on = load(ARM); off = load(f"{ARM}_off"); oie = load("openie")
    assert on, "pair not in the CCMP hops file"
    def simple(p, seeds):
        h = p["hops"]
        if not h or h[0]["head"] not in seeds or not all(x["tail"] == y["head"] for x, y in zip(h, h[1:])): return False
        ns = [h[0]["head"]] + [x["tail"] for x in h]; return len(ns) == len(set(ns)) and h[-1]["tail"] == g and len(h) <= a.max_hops
    routes = [p for p in on[1]["paths"] if simple(p, set(on[0]["seeds"]))][: a.n_routes]
    assert routes, "no seed-origin simple route recorded"
    key = lambda p: tuple((h["head"], h["rel"], h["tail"]) for h in p["hops"])
    offw = {key(p): p["weight"] for p in (off[1]["paths"] if off else [])}
    orow = next((p for p in oie[1]["paths"] if simple(p, set(oie[0]["seeds"]))), None) if (oie and not a.no_openie) else None
    oie_none = bool(oie) and not a.no_openie and orow is None
    rows = [("route %d" % (i + 1), p, False) for i, p in enumerate(routes)] + ([("entity graph", orow, True)] if orow else [])
    ncol = max(len(p["hops"]) for _, p, _ in rows)
    seeds = list(dict.fromkeys(on[0]["seeds"]))
    base = {}
    if a.candidates and os.path.exists(a.candidates):
        for c in json.load(open(a.candidates))["candidates"]:
            if c["id"] == q and c["gold"] == g: base = {k: v for k, v in c["ranks"].items() if k in ("bm25", "bge", "qwen3", "reasonir")}
    ids = {}
    for ri, (_, p, _) in enumerate(rows, 1):
        chain = [p["hops"][0]["head"]] + [h["tail"] for h in p["hops"]][:-1]
        for hi, nd in enumerate(chain, 1): ids[f"R{ri}H{hi}"] = nd
    print("node ids:"); [print(f"  {k:6} {ntype(v, docs):10} {v[:80]}") for k, v in ids.items()]
    spans = []
    for s in a.span:
        nid, txt = s.split("=", 1); assert nid in ids, f"unknown node id {nid}"; spans.append((nid, txt))
    num = {nid: i + 1 for i, nid in enumerate(dict.fromkeys(n for n, _ in spans))}
    def mark(text):
        out = tex(text)
        for nid, txt in spans:
            t = tex(txt)
            if t in out: out = out.replace(t, "\\spanmark{%s}{%d}{%s}" % (TYPE_COL[ntype(ids[nid], docs)], num[nid], t), 1)
        return out
    question = a.question or queries.get(q, {}).get("question", on[0]["question"]).split("?")[0] + "?"
    fp = (qf.get((q, g), {}).get("field_pair") or "")                 # high-level fields from the benchmark, e.g. "Computer Science -> Neuroscience"
    if not a.field_q: a.field_q = fp.split("->")[0].strip() if "->" in fp else fp
    if not a.field_d: a.field_d = fp.split("->")[-1].strip() if "->" in fp else fp
    _s = docs[g].split(". ")
    doc_text = a.doc or ((". ".join(x.rstrip(".") for x in _s[1:3]) + ".") if len(_s) > 1 else "(title only: no abstract in the corpus)")
    year = a.year or qf.get((q, g), {}).get("year") or ""; gtitle = _s[0]
    # ---- geometry (cm)
    dx = 3.0 if ncol <= 4 else (2.75 if ncol == 5 else 2.5); lw = {3.0: 26, 2.75: 24, 2.5: 22}[dx]     # column pitch, label width (mm)
    dy = 2.9; x0 = 0.0
    nrows = len(rows) + (1 if oie_none else 0); ent_last = bool(oie_none or (rows and rows[-1][2]))
    ys = [-(r * dy) - (0.5 if (ent_last and r == nrows - 1) else 0.0) for r in range(nrows)]   # extra gap above the entity row
    xq = -3.2
    L = [r"""% generated by eval/route_circles_tikz.py
% Build:  tectonic -X compile <this file> --outfmt pdf
\documentclass[tikz,border=8pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{soul}
\usetikzlibrary{positioning,fit,arrows.meta,calc,backgrounds,shapes.geometric}
\definecolor{ink}   {RGB}{ 38, 42, 48}
\definecolor{muted} {RGB}{124,130,140}
\definecolor{rule}  {RGB}{206,210,218}
\definecolor{paperf}{RGB}{247,248,250}
\definecolor{band}  {RGB}{241,243,246}
\definecolor{cFun}{RGB}{ 32,128,112}\definecolor{cFunl}{RGB}{202,230,223}
\definecolor{cFin}{RGB}{164,120, 34}\definecolor{cFinl}{RGB}{245,231,199}
\definecolor{cMet}{RGB}{ 42,105,160}\definecolor{cMetl}{RGB}{206,224,241}
\definecolor{cLim}{RGB}{112, 82,150}\definecolor{cLiml}{RGB}{222,212,238}
\definecolor{cDom}{RGB}{110,116,126}\definecolor{cDoml}{RGB}{236,237,240}
\definecolor{cDoc}{RGB}{196, 78, 82}\definecolor{cDocl}{RGB}{250,235,236}
\definecolor{cEnt}{RGB}{124,130,140}\definecolor{cEntl}{RGB}{240,241,243}
\definecolor{cGate}{RGB}{217,112, 26}
\definecolor{cUp}  {RGB}{ 32,128,112}
\definecolor{cDown}{RGB}{196, 78, 82}
\newcommand{\spanmark}[3]{{\sethlcolor{#1l}\hl{#3}}\textsuperscript{\textcolor{#1}{\bfseries #2}}}
\newcommand{\ty}{\fontsize{6.2}{7}\selectfont\scshape\bfseries}
\newcommand{\hd}{\fontsize{6.5}{7}\selectfont\scshape}
\begin{document}
\begin{tikzpicture}[
  >={Stealth[length=2.2mm,width=1.8mm]}, line width=0.9pt, draw=ink,
  endp/.style = {circle, draw=ink, fill=white, line width=1.1pt, minimum size=9.5mm, inner sep=0pt, font=\normalsize},
  circ/.style = {circle, minimum size=7.6mm, inner sep=0pt, line width=1.1pt, font=\scriptsize\bfseries, text=white},
  lbl/.style  = {align=center, font=\scriptsize, text=ink, text width=%(lw)dmm, inner sep=0.6mm,
                 execute at begin node={\hyphenpenalty=10000\exhyphenpenalty=10000}},
  card/.style = {draw=rule, fill=white, line width=0.6pt, rounded corners=2pt, align=left, text width=166mm, inner sep=3.4mm, font=\small},
  rlbl/.style = {font=\footnotesize, text=ink, inner sep=0pt, align=left},
  rel/.style  = {font=\fontsize{6.2}{7}\selectfont, text=ink, inner sep=0.8pt, fill=white, align=center},
  gate/.style = {font=\fontsize{6.2}{7}\selectfont, inner sep=0.8pt, fill=white},
  hdr/.style  = {font=\hd, text=muted, inner sep=0pt},
  pan/.style  = {font=\small\bfseries, text=ink, inner sep=0pt, anchor=south west},
  route/.style= {rounded corners=5pt},
]""".replace("%(lw)d", str(lw))]
    # ---- circles and labels
    def circle(name, nd, x, y, grey, badge=None):
        tp = ntype(nd, docs); col = TYPE_COL[tp]
        fill = "white" if grey else col; draw = "rule" if grey else col; txt = ("\\textcolor{muted}{%s}" % badge) if (grey and badge) else (badge or "")
        L.append("\\node[circ, draw=%s, fill=%s] (%s) at (%.2f,%.2f) {%s};" % (draw, fill if not grey else "cEntl", name, x, y, txt))
        if tp == "paper":
            ttl = docs[nd].split(". ")[0]; cut = len(ttl) > 64
            if cut: ttl = ttl[:64].rsplit(" ", 1)[0]
            yr = qf.get((q, nd), {}).get("year") or ""
            body = "{\\ty\\color{%s}paper}\\\\ %s%s%s" % ("muted" if grey else col, tex(ttl), "\\,\\ldots" if cut else "", ("\\\\\\textcolor{muted}{%s}" % yr) if yr else "")
        else:
            nm = nd[nd.index("]") + 2:] if nd.startswith("[") and "]" in nd else nd
            body = "{\\ty\\color{%s}%s}\\\\ %s" % ("muted" if grey else col, tp, tex(nm))
        L.append("\\node[lbl%s, anchor=north] (l%s) at (%s.south) {%s};" % (", text=muted" if grey else "", name, name, body))
    # seed band behind the first column (drawn first so everything sits on top of it)
    ybot_rows = ys[-1] - (3.55 if oie_none else 2.75)     # below the entity-seed list
    L.append("\\fill[band] (%.2f,%.2f) rectangle (%.2f,%.2f);" % (x0 - 0.75, ys[0] + 0.92, x0 + 0.75, ybot_rows))
    for ri, (lab, p, grey) in enumerate(rows, 1):
        y = ys[ri - 1]; chain = [p["hops"][0]["head"]] + [h["tail"] for h in p["hops"]][:-1]
        for hi, nd in enumerate(chain, 1):
            nid = f"R{ri}H{hi}"; circle(f"n{ri}_{hi}", nd, x0 + (hi - 1) * dx, y, grey, badge=str(num[nid]) if nid in num else None)
    if oie_none:
        L.append("\\node[circ, draw=rule, fill=cEntl, dashed] (n%d_1) at (%.2f,%.2f) {};" % (nrows, x0, ys[-1]))
        L.append("\\node[lbl, text=muted, anchor=north] (ln%d_1) at (n%d_1.south) {{\\ty entity graph}\\\\ no seed-to-gold route within the reasoner's depth};" % (nrows, nrows))
    # ---- endpoints and the merge bus into i*
    ymid = (ys[0] + ys[-1]) / 2
    L.append("\\node[endp] (qn) at (%.2f,%.2f) {$q$};" % (xq, ymid))
    xend = x0 + (ncol - 1) * dx + 3.1; xm = xend - 1.35
    L.append("\\node[endp] (hs) at (%.2f,%.2f) {$i^{\\star}$};" % (xend, ymid))
    L.append("\\draw[->, draw=muted] (qn.east) -- (%.2f,%.2f);" % (x0 - 0.75, ymid))
    # ---- column headers (panel b)
    yh = ys[0] + 1.18
    L.append("\\draw[draw=rule, line width=0.6pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (x0 - 0.75, yh - 0.22, xend + 0.5, yh - 0.22))
    L.append("\\node[hdr] at (%.2f,%.2f) {seed};" % (x0, yh))
    for c in range(2, ncol + 1): L.append("\\node[hdr] at (%.2f,%.2f) {hop %d};" % (x0 + (c - 1) * dx, yh, c - 1))
    L.append("\\node[hdr] at (%.2f,%.2f) {target};" % (xend, yh))
    # ---- seed counts and the other entity seeds, under the band
    n = len(seeds); frame_rows = [ri for ri, (_, p, gr) in enumerate(rows, 1) if not gr]
    ent_row = len(rows) if (rows and rows[-1][2]) else (nrows if oie_none else None)
    foot = "frames %d/%d" % (len(frame_rows), n)
    if ent_row:
        oseeds = list(dict.fromkeys(oie[0]["seeds"])); used = rows[-1][1]["hops"][0]["head"] if (rows and rows[-1][2]) else None
        others = [x for x in oseeds if x != used][:3]; more = len(oseeds) - len(others) - (1 if used else 0)
        foot += " \\,\\textperiodcentered\\, entities %d/%d" % (1 if used else 0, len(oseeds))
        if others:
            items = "\\\\ ".join("\\textcolor{rule}{\\rule[0.4ex]{1.6mm}{0.5pt}}\\, " + tex(x) for x in others) + ((" \\\\ \\textcolor{rule}{\\rule[0.4ex]{1.6mm}{0.5pt}}\\, +%d more" % more) if more > 0 else "")
            L.append("\\node[anchor=north, font=\\fontsize{6}{7}\\selectfont, text=muted, align=left, text width=%dmm, inner sep=0.4mm] at (ln%d_1.south) {%s};" % (lw, ent_row, items))
    L.append("\\node[anchor=north, font=\\fontsize{6}{7}\\selectfont, text=muted, align=center, text width=44mm, inner sep=0pt] at (%.2f,%.2f) {%s};" % (x0, ybot_rows - 0.08, foot))
    # ---- row labels
    for ri, (lab, p, grey) in enumerate(rows, 1):
        w = p["weight"]; y = ys[ri - 1]
        if grey: L.append("\\node[rlbl, anchor=south west] at (%.2f,%.2f) {\\textsc{entity graph} {\\scriptsize\\color{muted}OpenIE \\,\\textperiodcentered\\, $w = %.2f$}};" % (x0 + dx - 0.45, y + 0.5, abs(w)))
        else: L.append("\\node[rlbl, anchor=south west] at (%.2f,%.2f) {\\textsc{%s} {\\scriptsize\\color{muted}$w = %.1f$}};" % (x0 + dx - 0.45, y + 0.5, tex(lab), w))
    if oie_none: L.append("\\node[rlbl, anchor=south west] at (%.2f,%.2f) {\\textsc{entity graph} {\\scriptsize\\color{muted}OpenIE}};" % (x0 + dx - 0.45, ys[-1] + 0.5))
    # ---- edges: relation above the arrow; the CCMP gate colours the hop (orange = amplified, dashed blue-grey = damped)
    order = []
    for ri, (lab, p, grey) in enumerate(rows, 1):
        y = ys[ri - 1]
        for hi, h in enumerate(p["hops"], 1):
            gt = h.get("gate"); amp = (gt is not None and gt > 1.05 and not grey); damp = (gt is not None and gt < 0.95 and not grey)
            style = "draw=muted" if grey else "draw=ink, line width=1.1pt"
            gl = (" node[gate, text=%s, below=0.4mm, pos=%s]{$g$ %.2f}" % ("cGate" if amp else ("cMet" if damp else "muted"), ".5" if hi < len(p["hops"]) else ".62", gt)) if (gt is not None and not grey) else ""
            if hi < len(p["hops"]):
                cmd = "\\draw[->, %s] (n%d_%d.east) -- node[rel, above=0.3mm, pos=.5]{%s}%s (n%d_%d.west);" % (style, ri, hi, rel(h["rel"]), gl, ri, hi + 1)
            else:   # last hop: horizontal to the merge bus, then down/up the bus into i* (straight when the row is at bus height)
                xlast = x0 + (hi - 1) * dx; ps = ".3" if (xm - xlast) > 3.0 else ".5"
                gl = gl.replace("pos=.62", "pos=%s" % (".62" if (xm - xlast) > 3.0 else ".5"))
                if abs(y - ymid) < 0.6:
                    cmd = "\\draw[->, %s] (n%d_%d.east) -- node[rel, above=0.4mm, pos=%s]{%s}%s (hs.west);" % (style, ri, hi, ps, rel(h["rel"]), gl)
                else:
                    cmd = "\\draw[->, %s, route] (n%d_%d.east) -- node[rel, above=0.4mm, pos=%s]{%s}%s (%.2f,%.2f) -- (%.2f,%.2f) -- (hs.west);" % (style, ri, hi, ps, rel(h["rel"]), gl, xm, y, xm, ymid)
            order.append((0 if grey else (2 if amp else 1), cmd))
    for _, cmd in sorted(order, key=lambda t: t[0]): L.append(cmd)
    # ---- cards (panel a)
    ytop = ys[0] + 1.75
    L.append("\\node[card, anchor=south] (target) at (%.2f,%.2f) {\\textbf{$i^{\\star}$}\\ \\ \\textcolor{muted}{%s}\\ \\ \\emph{%s}%s\\\\[2pt] ``%s''};"
             % ((xq + xend) / 2, ytop, tex(a.field_d), tex(gtitle), (" (%s)" % year if year else ""), mark(doc_text)))
    qcard = "\\textbf{Research Question}\\ \\ \\textcolor{muted}{%s}\\ \\ ``%s''" % (tex(a.field_q), mark(question))
    if a.survey: qcard += "\\\\[3pt]\\textbf{Background Survey}\\ \\ ``%s''" % mark(a.survey)
    L.append("\\node[card, anchor=south, fill=paperf] (query) at ([yshift=2.5mm] target.north) {%s};" % qcard)
    L.append("\\node[pan] at ([xshift=-1mm, yshift=1.2mm] query.north west) {(a)};")
    L.append("\\node[pan] at (%.2f,%.2f) {(b)};" % (xq - 0.9, yh - 0.12))
    if a.no_strip:   # only the hop key under the routes
        key = ("$g$ = CCMP gate on the hop's sender, normalised by the frontier mean (\\textcolor{cGate}{${>}1.05$: amplified}, \\textcolor{cMet}{${<}0.95$: damped}) \\quad "
               "\\textcolor{muted}{\\rule[0.35ex]{4mm}{0.8pt}} entity-graph route")
        L.append("\\node[anchor=north, font=\\fontsize{6}{7}\\selectfont, text=muted, align=center, text width=178mm] at (%.2f,%.2f) {%s};" % ((x0 + xend) / 2, ybot_rows - 0.45, key))
    else:
        # ---- rank strip (panel c): log axis 1..5000; gate-off (hollow) -> with CCMP (filled), arrow green if the rank improves, red if it worsens
        rk = on[1]["rank"]; rko = off[1]["rank"] if off else {}; rki = oie[1]["rank"] if oie else {}
        xa = x0 - 0.2; xb = xend - 0.2; W = xb - xa; yst = ybot_rows - 2.15
        pos = lambda r: xa + W * math.log10(max(1.0, min(float(r), 5000.0))) / math.log10(5000.0)
        yG, yM, yF = yst + 0.62, yst, yst - 0.62
        L.append("\\node[pan] at (%.2f,%.2f) {(c)};" % (xq - 0.9, yG + 0.42))
        L.append("\\draw[draw=rule, line width=0.6pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (x0 - 0.75, yG + 0.55, xend + 0.5, yG + 0.55))
        L.append("\\node[hdr, anchor=west] at (%.2f,%.2f) {rank of $i^{\\star}$ (log scale)};" % (xa, yG + 0.72))
        L.append("\\node[hdr, anchor=center] at (%.2f,%.2f) {CCMP $\\Delta$};" % (xend, yG + 0.72))
        for yy in (yG, yM, yF): L.append("\\draw[draw=rule, line width=0.6pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (xa, yy, xb, yy))
        for t in (1, 10, 100, 1000):
            L.append("\\draw[draw=muted] (%.2f,%.2f) -- (%.2f,%.2f);" % (pos(t), yF - 0.50, pos(t), yF - 0.60))
            L.append("\\node[anchor=north, font=\\fontsize{5.5}{6}\\selectfont, text=muted] at (%.2f,%.2f) {%d};" % (pos(t), yF - 0.60, t))
        for yy, t in ((yG, "graph channel"), (yM, "text retrievers"), (yF, "\\textsc{SciGraphIR} (fused)")):
            L.append("\\node[anchor=east, font=\\fontsize{6.2}{7}\\selectfont, text=ink] at (%.2f,%.2f) {%s};" % (xa - 0.15, yy, t))
        def mk(shape, x, y, fill, draw, size="2.6mm"):
            L.append("\\node[%s, fill=%s, draw=%s, line width=0.8pt, minimum size=%s, inner sep=0pt] at (%.2f,%.2f) {};" % (shape, fill, draw, size, x, y))
        def numlab(x, y, r, above=True, col="ink"):
            L.append("\\node[anchor=%s, font=\\fontsize{5.5}{6}\\selectfont, text=%s] at (%.2f,%.2f) {%s};" % ("south" if above else "north", col, x, y + (0.16 if above else -0.16), r))
        if rk.get("dense"): mk("circle", pos(rk["dense"]), yM, "muted", "muted"); numlab(pos(rk["dense"]), yM, rk["dense"], True)
        if rk.get("scorer"): mk("rectangle", pos(rk["scorer"]), yM, "ink", "ink", "2.3mm"); numlab(pos(rk["scorer"]), yM, rk["scorer"], True)
        for k, v in base.items():
            if v: L.append("\\draw[draw=muted, line width=0.9pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (pos(v), yM - 0.14, pos(v), yM + 0.14))
        def pair(ro, rn, yy, above):
            """hollow circle (gate off) -> filled circle (with CCMP); the arrow and the delta tag carry the sign"""
            if ro and rn and ro != rn and abs(pos(ro) - pos(rn)) > 0.42:      # markers too close for an arrow: the tag carries the sign
                c = "cUp" if rn < ro else "cDown"
                L.append("\\draw[->, draw=%s, line width=1.0pt] (%.2f,%.2f) -- (%.2f,%.2f);" % (c, pos(ro) + (-0.15 if rn < ro else 0.15), yy, pos(rn) + (0.15 if rn < ro else -0.15), yy))
            if ro: mk("circle", pos(ro), yy, "white", "cMet", "2.7mm")
            if rn: mk("circle", pos(rn), yy, "cMet", "cMet", "2.7mm")
            if ro and rn and ro != rn and abs(pos(ro) - pos(rn)) < 0.75:
                L.append("\\node[anchor=%s, font=\\fontsize{5.5}{6}\\selectfont, text=ink] at (%.2f,%.2f) {\\textcolor{muted}{%s} $\\rightarrow$ %s};" % ("south" if above else "north", (pos(ro) + pos(rn)) / 2, yy + (0.16 if above else -0.16), ro, rn))
            else:
                if ro and ro != rn: numlab(pos(ro), yy, ro, above, "muted")
                if rn: numlab(pos(rn), yy, rn, above)
            if ro and rn:
                d = rn - ro; c = "cUp" if d < 0 else ("cDown" if d > 0 else "muted")
                L.append("\\node[anchor=center, font=\\fontsize{7}{8}\\selectfont\\bfseries, text=%s] at (%.2f,%.2f) {%s};" % (c, xend, yy, ("$%+d$" % d) if d else "$0$"))
        if rki.get("graph"): mk("diamond", pos(rki["graph"]), yG, "white", "cEnt", "2.9mm"); numlab(pos(rki["graph"]), yG, rki["graph"])
        pair(rko.get("graph"), rk.get("graph"), yG, True)
        pair(rko.get("fused"), rk.get("fused"), yF, False)
        key = ("\\tikz\\node[circle, fill=muted, minimum size=2mm, inner sep=0pt]{}; cosine \\quad "
               "\\tikz\\node[rectangle, fill=ink, minimum size=1.9mm, inner sep=0pt]{}; scorer \\quad "
               "\\tikz\\draw[draw=muted, line width=0.9pt] (0,0) -- (0,0.22); dense baselines \\quad "
               "\\tikz\\node[diamond, draw=cEnt, fill=white, minimum size=2.4mm, inner sep=0pt]{}; entity graph \\quad "
               "\\tikz\\node[circle, draw=cMet, fill=white, minimum size=2.1mm, inner sep=0pt]{}; CCMP gate off \\quad "
               "\\tikz\\node[circle, fill=cMet, minimum size=2.1mm, inner sep=0pt]{}; with CCMP \\quad "
               "\\textcolor{cUp}{$\\rightarrow$} better \\ \\textcolor{cDown}{$\\rightarrow$} worse \\quad "
               "\\textcolor{cGate}{\\rule[0.35ex]{4mm}{1.2pt}} hop amplified \\ \\textcolor{cMet}{\\rule[0.35ex]{1.3mm}{1pt}\\,\\rule[0.35ex]{1.3mm}{1pt}} damped")
        L.append("\\node[anchor=north, font=\\fontsize{6}{7}\\selectfont, text=muted, align=center, text width=178mm] at (%.2f,%.2f) {%s};" % ((xa + xb) / 2, yF - 0.98, key))
    L.append("\\end{tikzpicture}\n\\end{document}")
    open(a.out + ".tex", "w").write("\n".join(L) + "\n"); print("wrote", a.out + ".tex")
    if a.compile:
        pr = subprocess.run(["tectonic", "-X", "compile", a.out + ".tex", "--outfmt", "pdf"], capture_output=True, text=True)
        if pr.returncode != 0: print(pr.stdout[-2500:], pr.stderr[-2500:]); raise SystemExit("tectonic failed")
        if shutil.which("pdftoppm"): subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", a.out + ".pdf", a.out + "_preview"]); print("wrote", a.out + "_preview.png")


if __name__ == "__main__":
    main()
