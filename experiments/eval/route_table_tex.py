#!/usr/bin/env python3
"""
route_table_tex.py -- path interpretations as a table, in the format of GFM-RAG Table 4 (NBFNet-style paths listed
as weight: (head, relation, tail) -> (...), r^-1 = inverse relation), one block per (query, gold) pair:

    Query (field)        "..."
    Inspiration (field)  Title (year)
    Rank of i*           cosine . scorer . entity graph . SciAfford graph gate off -> with CCMP . SciGraphIR gate off -> with CCMP
    Routes               w: (node, rel, node) -> (node, rel[gate g], i*)      the SciAfford graph's top routes, CCMP gate on the hop where it acts
    Entity graph         w: (...)                                              the OpenIE graph's route, or "no seed-to-gold route"

Same inputs as route_circles_tikz.py (hops_frame_ccmp / _off / openie in --dir, optional --candidates for the baselines).

    python3 eval/route_table_tex.py --dataset sir4_cs --dir results/qualitative/drive_scan_sir4_cs         --pair 10.48550_arxiv.2603.03985=10.1111/j.1749-6632.2010.05443.x --pair ... --n-routes 2 --out ../figures/tab_route_paths_cs --compile
"""
import argparse, json, os, re, shutil, subprocess
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def frame_arm(d, prefix="hops_", want="auto"):
    """which SciAfford graph arm the hops files hold: the merged graph (hyb_ccmp, the current SciAfford graph) when present,
    else the older SciAfford graph (frame_ccmp). Returns (arm name, label suffix)."""
    if want == "auto": want = "hyb_ccmp" if os.path.exists(f"{d}/{prefix}hyb_ccmp.json") else "frame_ccmp"
    return want, ("merged graph" if want.startswith("hyb") else 'SciAfford graph, outdated')


def tex(s):
    s = re.sub(r"([&%$#_{}])", r"\\\1", str(s)).replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
    return s.replace("\u2014", "\\textemdash{}").replace("\u2013", "\\textendash{}").replace("\u2018", "`").replace("\u2019", "'").replace("\u201c", "``").replace("\u201d", "''")


def ntype(n, docs):
    if n in docs: return "paper"
    return n[1:n.index("]")] if n.startswith("[") and "]" in n else "entity"


def rel(r):
    inv = r.startswith("inverse_"); r = r[8:] if inv else r
    return tex(r.replace("_", " ")) + ("$^{-1}$" if inv else "")


def node(n, docs, gold, maxlen=46):
    if n == gold: return "$i^{\\star}$"
    tp = ntype(n, docs)
    if tp == "paper":
        t = docs[n].split(". ")[0]
        if len(t) > maxlen: t = t[:maxlen].rsplit(" ", 1)[0] + "\\,\\ldots"
        return "{\\scriptsize\\textsc{paper}} " + tex(t)
    nm = n[n.index("]") + 2:] if n.startswith("[") and "]" in n else n
    return ("{\\scriptsize\\textsc{%s}} " % tp if tp != "entity" else "") + tex(nm)


def path_tex(p, docs, gold, gates=True):
    parts = []
    for h in p["hops"]:
        g = h.get("gate"); gs = ""
        if gates and g is not None:
            gs = "\\,{\\color{%s}\\scriptsize $g$\\,%.2f}" % ("cGate" if g > 1.05 else ("cMet" if g < 0.95 else "muted"), g)
        parts.append("(%s, %s%s, %s)" % (node(h["head"], docs, gold), rel(h["rel"]), gs, node(h["tail"], docs, gold)))
    return " $\\rightarrow$ ".join(parts)


def simple(p, seeds, g, max_hops):
    h = p["hops"]
    if not h or h[0]["head"] not in seeds or not all(x["tail"] == y["head"] for x, y in zip(h, h[1:])): return False
    ns = [h[0]["head"]] + [x["tail"] for x in h]; return len(ns) == len(set(ns)) and h[-1]["tail"] == g and len(h) <= max_hops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--prefix", default="hops_")
    ap.add_argument("--pair", action="append", required=True, help="query=gold, repeatable; one block each")
    ap.add_argument("--n-routes", type=int, default=2); ap.add_argument("--max-hops", type=int, default=6)
    ap.add_argument("--candidates", default=None); ap.add_argument("--docs", default=None); ap.add_argument("--queries", default=None); ap.add_argument("--sir4", "--quartet", dest='sir4', default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--compile", action="store_true"); ap.add_argument("--label", default="tab:route-paths")
    ap.add_argument("--arm", default="auto", help="SciAfford graph arm: auto (hyb_ccmp if present, else frame_ccmp), hyb_ccmp or frame_ccmp")
    a = ap.parse_args(); D = a.dataset
    docs = json.load(open(a.docs or f"{R}/retriever/data/{D}_test/raw/documents.json"))
    queries = {x["id"]: x for x in json.load(open(a.queries or f"{R}/retriever/data/{D}_test/raw/test.json"))}
    qf = {}
    qp = a.sir4 or f"{R}/sir-4/data/benchmark/{D.replace('sir4_', '')}_test_final/eval.json"
    if os.path.exists(qp):
        for x in json.load(open(qp)):
            for d, m in (x.get("quartet", {}).get("per_document") or {}).items(): qf[(x["id"], d)] = m
    def load(name):
        p = f"{a.dir}/{a.prefix}{name}.json"
        return {(r["id"], t["doc"]): (r, t) for r in json.load(open(p)) for t in r["targets"]} if os.path.exists(p) else {}
    ARM, ARMLAB = frame_arm(a.dir, a.prefix, a.arm); print("SciAfford graph arm:", ARM, f"({ARMLAB})")
    ON, OFF, OIE = load(ARM), load(f"{ARM}_off"), load("openie")
    cand = {}
    if a.candidates and os.path.exists(a.candidates):
        for c in json.load(open(a.candidates))["candidates"]: cand[(c["id"], c["gold"])] = c["ranks"]
    blocks = []
    for pair in a.pair:
        q, g = pair.split("=", 1); on = ON[(q, g)]; off = OFF.get((q, g)); oie = OIE.get((q, g))
        rk, rko = on[1]["rank"], (off[1]["rank"] if off else {}); rki = oie[1]["rank"] if oie else {}
        fp = qf.get((q, g), {}).get("field_pair") or ""; fq, fd = (fp.split("->") + [""])[:2] if "->" in fp else (fp, fp)
        year = qf.get((q, g), {}).get("year") or ""
        question = queries.get(q, {}).get("question", on[0]["question"]).split("?")[0] + "?"
        title = docs[g].split(". ")[0]
        routes = [p for p in on[1]["paths"] if simple(p, set(on[0]["seeds"]), g, a.max_hops)][: a.n_routes]
        orow = next((p for p in oie[1]["paths"] if simple(p, set(oie[0]["seeds"]), g, a.max_hops)), None) if oie else None
        def arrow(o, n): return ("%s $\\rightarrow$ %s" % (o, n)) if (o and n and o != n) else str(n)
        ranks = ["cosine %s" % rk.get("dense"), "scorer %s" % rk.get("scorer")]
        if rki.get("graph"): ranks.append("entity graph %s" % rki["graph"])
        ranks.append("\\textsc{SciAfford graph} %s" % arrow(rko.get("graph"), rk.get("graph")))
        ranks.append("\\textsc{SciGraphIR} %s" % arrow(rko.get("fused"), rk.get("fused")))
        rows = ["\\textbf{Query} \\newline {\\scriptsize\\color{muted}%s} & ``%s'' \\\\" % (tex(fq.strip()), tex(question)),
                "\\textbf{Inspiration $i^{\\star}$} \\newline {\\scriptsize\\color{muted}%s} & \\emph{%s}%s \\\\" % (tex(fd.strip()), tex(title), (" (%s)" % year) if year else ""),
                "\\textbf{Rank of $i^{\\star}$} & %s \\\\" % " \\,$\\cdot$\\, ".join(ranks)]
        if routes:
            rows.append("\\textbf{Routes} & " + " \\newline ".join("%.2f: %s" % (p["weight"], path_tex(p, docs, g)) for p in routes) + " \\\\")
        else:
            rows.append("\\textbf{Routes} & \\textcolor{muted}{no seed-to-gold route within the reasoner's depth} \\\\")
        if oie:
            rows.append("\\textbf{Entity graph} & " + (("%.2f: %s" % (abs(orow["weight"]), path_tex(orow, docs, g, gates=False))) if orow else "\\textcolor{muted}{no seed-to-gold route within the reasoner's depth}") + " \\\\")
        blocks.append("\n".join(rows))
    body = "\n\\midrule\n".join(blocks)
    frag = ("% generated by eval/route_table_tex.py -- needs \\definecolor{cGate}{RGB}{217,112,26} \\definecolor{cMet}{RGB}{42,105,160} \\definecolor{muted}{RGB}{124,130,140}\n"
            "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.15\\textwidth} p{0.83\\textwidth}@{}}\n\\toprule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    open(a.out + ".tex", "w").write(frag); print("wrote", a.out + ".tex")
    if a.compile:
        doc = ("\\documentclass[10pt]{article}\\usepackage[margin=12mm,paperwidth=190mm,paperheight=400mm]{geometry}\\usepackage[T1]{fontenc}\\usepackage{lmodern}"
               "\\usepackage{amsmath,amssymb,booktabs,xcolor,array}\\definecolor{cGate}{RGB}{217,112,26}\\definecolor{cMet}{RGB}{42,105,160}\\definecolor{muted}{RGB}{124,130,140}"
               "\\pagestyle{empty}\\begin{document}\\small\\noindent\n" + frag + "\\end{document}\n")
        pv = a.out + "_preview.tex"; open(pv, "w").write(doc)
        pr = subprocess.run(["tectonic", "-X", "compile", pv, "--outfmt", "pdf"], capture_output=True, text=True)
        if pr.returncode != 0: print(pr.stdout[-2500:], pr.stderr[-2500:]); raise SystemExit("tectonic failed")
        if shutil.which("pdftoppm"):
            subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", a.out + "_preview.pdf", a.out + "_preview"])
            try:
                from PIL import Image, ImageChops
                im = Image.open(a.out + "_preview.png").convert("RGB"); bg = Image.new("RGB", im.size, (255, 255, 255))
                bb = ImageChops.difference(im, bg).getbbox()
                if bb: im.crop((max(0, bb[0] - 30), max(0, bb[1] - 30), min(im.width, bb[2] + 30), min(im.height, bb[3] + 30))).save(a.out + "_preview.png")
            except Exception as e: print("crop skipped:", e)
            print("wrote", a.out + "_preview.png")


if __name__ == "__main__":
    main()
