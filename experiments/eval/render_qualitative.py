"""
render_qualitative.py -- turn interpret_paths.py outputs into the qualitative section's
markdown and LaTeX: per example, the query, the gold inspiration, every system's rank of it,
the scorer's best-matching hypothetical-answer views, and the top reasoning paths on the
SciAfford graph (with the CCMP gate along each hop) next to the top paths on the OpenIE graph.

    python3 eval/render_qualitative.py --picks picks.json --docs documents.json \\
        --paths sciafford=paths_frame.json --paths openie=paths_openie.json [--paths control=...] \\
        --n 3 --out-md qualitative.md --out-tex qualitative_paths.tex
"""
from __future__ import annotations

import argparse
import json
import os
import re

SYS_LABEL = {"bm25": "BM25", "bge": "BGE-large", "qwen3": "Qwen3-Emb.", "specter2": "SPECTER2",
             "scincl": "SciNCL", "reasonir": "ReasonIR-8B", "dense": "Qwen3 cosine", "scorer": "Multi-view scorer",
             "openie": "+ Graph (OpenIE)", "control": "+ Graph (SciGraph)", "scigraphir": "+ Graph + CCMP (SciGraphIR)",
             "graph_ch": "graph channel alone (inside SciGraphIR)", "scorer_ch": "scorer channel (inside SciGraphIR)",
             "dense_ch": "Qwen3 cosine (inside SciGraphIR)"}
ARM_LABEL = {"frame": 'SciAfford graph (+ CCMP)', "control": 'SciAfford graph (no CCMP, separate control weights)',
             "frame_nogate": 'SciAfford graph (same weights, CCMP gate off at inference)',
             "openie": "OpenIE entity graph"}


def tex_escape(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    for c in "&%$#_{}":
        s = s.replace(c, "\\" + c)
    return s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def short(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n - 3].rstrip() + "..."


def node_label(name: str, docs: dict, n: int = 60) -> str:
    if name in docs:
        return "[paper] " + short(docs[name].split(". ")[0], n)
    return short(name, n)


def rel_label(r: str) -> str:
    return r.replace("inverse_", "inv. ").replace("_", " ")


def fmt_path(p: dict, docs: dict, with_gate: bool) -> str:
    parts = []
    for h in p["hops"]:
        g = f" (gate {h['gate']:.2f})" if with_gate and "gate" in h else ""
        parts.append(f"{node_label(h['head'], docs)}{g} --{rel_label(h['rel'])}-->")
    parts.append(node_label(p["hops"][-1]["tail"], docs))
    return " ".join(parts)


def fmt_path_tex(p: dict, docs: dict, with_gate: bool) -> str:
    parts = []
    for h in p["hops"]:
        g = f" {{\\scriptsize({h['gate']:.2f})}}" if with_gate and "gate" in h else ""
        parts.append(f"{tex_escape(node_label(h['head'], docs, 48))}{g} $\\xrightarrow{{\\text{{{tex_escape(rel_label(h['rel']))}}}}}$")
    parts.append(tex_escape(node_label(p["hops"][-1]["tail"], docs, 48)))
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--paths", action="append", default=[], help="arm=json (sciafford|sciafford_nogate|control|openie), repeatable")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--paths-per-arm", type=int, default=2)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-tex", required=True)
    a = ap.parse_args()

    picks = json.load(open(a.picks))
    docs = json.load(open(a.docs))
    arms = {}
    for spec in a.paths:
        name, path = spec.split("=", 1)
        name = {"sciafford": "frame", "sciafford_nogate": "frame_nogate"}.get(name, name)
        if os.path.exists(path):
            arms[name] = {r["id"]: r for r in json.load(open(path))}
    systems = picks["systems"]
    cands = picks["candidates"][:a.n]

    md, tex = [], []
    md.append("# Qualitative examples: cross-field inspiration retrieval\n")
    for k, c in enumerate(cands, 1):
        qid, gold = c["id"], c["gold"]
        md.append(f"## Example {k}: `{qid}` ({c['stratum']}-field, {c['n_golds']} golds)\n")
        md.append(f"**Query.** {short(c['question'], 700)}\n")
        md.append(f"**Gold inspiration.** {short(docs.get(gold, gold), 400)}\n")
        md.append("| system | rank of the gold |"); md.append("|---|--:|")
        for s in systems:
            r = c["ranks"].get(s)
            md.append(f"| {SYS_LABEL.get(s, s)} | {r if r is not None and r < 10**6 else '>300'} |")
        md.append("")
        for arm, table in arms.items():
            rec = table.get(qid)
            if not rec:
                md.append(f"*{ARM_LABEL.get(arm, arm)}: no interpretation for this query.*\n"); continue
            tgt = next((t for t in rec["targets"] if t["doc"] == gold), rec["targets"][0] if rec["targets"] else None)
            if tgt is None:
                continue
            md.append(f"### {ARM_LABEL.get(arm, arm)}\n")
            md.append("channel ranks of this gold: " + ", ".join(f"{k2} {v}" for k2, v in tgt["rank"].items()) + "\n")
            if arm == "frame" and tgt.get("views"):
                md.append("Scorer views that matched the gold best (hypothetical answers written for the query):\n")
                for v in tgt["views"]:
                    md.append(f"- view {v['view']} match {v['match']:.3f}: {short(v.get('text') or '(text unavailable)', 220)}")
                md.append(f"- direct Qwen3 cosine of the gold: {tgt['dense_cos']:.3f}\n")
            with_gate = arm == "frame" and any("gate" in h for p in tgt["paths"] for h in p["hops"])
            if not tgt["paths"]:
                md.append("(no path found within the reasoner's depth)\n")
            for p in tgt["paths"][:a.paths_per_arm]:
                md.append(f"- w={p['weight']:.3f}: {fmt_path(p, docs, with_gate)}")
            if with_gate:
                md.append(f"\nfrontier-mean responsibility per layer: " + ", ".join(f"{x:.2f}" for x in tgt["frontier_mean_resp"]))
            md.append("")
    # ---- LaTeX: one table in the GFM-RAG Table 4 layout, plus a rank table
    tex.append("% generated by render_qualitative.py; edit freely")
    tex.append("\\begin{table}[t]\n\\centering\n\\resulttablestyle\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.2}")
    tex.append("\\begin{tabular}{@{}p{0.27\\linewidth} p{0.22\\linewidth} p{0.47\\linewidth}@{}}\n\\toprule")
    tex.append("\\textbf{Query (cross-field)} & \\textbf{Retrieved inspiration} & \\textbf{Top reasoning path (CCMP gate per hop)} \\\\\n\\midrule")
    for c in cands:
        qid, gold = c["id"], c["gold"]
        rec = arms.get("frame", {}).get(qid)
        tgt = next((t for t in (rec or {}).get("targets", []) if t["doc"] == gold), None)
        pth = fmt_path_tex(tgt["paths"][0], docs, True) if tgt and tgt["paths"] else "--"
        ranks = " / ".join(f"{SYS_LABEL.get(s, s)} {c['ranks'].get(s, '--')}" for s in ("qwen3", "bge", "scigraphir") if s in c["ranks"])
        tex.append(f"{tex_escape(short(c['question'], 260))} & {tex_escape(short(docs.get(gold, gold).split('. ')[0], 120))} "
                   f"{{\\scriptsize (rank: {tex_escape(ranks)})}} & {pth} \\\\\n\\addlinespace")
    tex.append("\\bottomrule\n\\end{tabular}")
    tex.append("\\caption{Path interpretations on SIR-4. For each cross-field query, the highest-weighted path from a query "
               'seed node to the gold paper under the trained reasoner (gradient beam search over per-layer edge weights, '
               "as in NBFNet and GFM-RAG); numbers in parentheses are the CCMP gate applied to each hop's sender "
               "(1.0 = frontier mean; $>$1 amplified, $<$1 suppressed). Ranks are the position of the gold in each "
               "system's ranking.}\n\\label{tab:path-interpretations}\n\\end{table}")
    # rank table
    tex.append("\n\\begin{table}[t]\n\\centering\n\\resulttablestyle")
    tex.append("\\begin{tabular}{@{}l" + "r" * len(systems) + "@{}}\n\\toprule")
    tex.append("Example & " + " & ".join(tex_escape(SYS_LABEL.get(s, s)) for s in systems) + " \\\\\n\\midrule")
    for k, c in enumerate(cands, 1):
        tex.append(f"{k} & " + " & ".join((str(c['ranks'][s]) if c['ranks'].get(s, 10**6) < 10**6 else "$>$300") if s in c["ranks"] else "--" for s in systems) + " \\\\")
    tex.append("\\bottomrule\n\\end{tabular}\n\\caption{Rank of the gold inspiration under each system for the examples above.}\n\\label{tab:qual-ranks}\n\\end{table}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out_md)), exist_ok=True)
    open(a.out_md, "w").write("\n".join(md)); open(a.out_tex, "w").write("\n".join(tex))
    print("\n".join(md)); print(f"\nwrote {a.out_md}\nwrote {a.out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
