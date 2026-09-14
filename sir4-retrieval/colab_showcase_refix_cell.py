# ===== LIVE FIX: rewrite eval/showcase.py and redraw every dataset from the cached hops files =====
# Paste into the RUNNING kernel and run. No engine, no Qwen3, no path search: it only re-reads the
# hops_*.json already on Drive. Fixes in this version:
#   * the hop figure's legend is keyed on the arm LABEL, so an arm keeps its colour across panels and
#     the shared legend no longer mislabels the datasets that have fewer arms;
#   * the panel title shows the real gold count and says "no floor" when min_hops is absent;
#   * the candidate table no longer emits a stray empty column when no baseline predictions are found;
#   * a failure in the figure can no longer destroy the .md / .tex / _hops.json.
import os, sys, json, glob, time, subprocess
S4        = globals().get("S4",        "/content/cargo/sir4-retrieval")
DRIVE     = globals().get("DRIVE",     "/content/drive/MyDrive/cargo-gfmrag")
DATA_ROOT = globals().get("DATA_ROOT", "/content/cargo/kg-construction/data")
DATASETS  = globals().get("DATASETS") or [globals().get("DATASET")]
DATASETS  = [d for d in DATASETS if d]
assert DATASETS, "no DATASETS / DATASET in this kernel; set DATASETS = ['mir', ...] before running"
TOP, N_TEX = 30, 4

os.makedirs(f"{S4}/eval", exist_ok=True)
open(f"{S4}/eval/showcase.py", "w").write(r"""#!/usr/bin/env python3
'''
showcase.py -- find the worked examples that show cross-domain reasoning, and draw the hop figure.

Reads interpret_paths.py outputs with paths (one file per arm; the FIRST --arm is the model being
showcased, normally the frame graph + CCMP) and produces, for one dataset:

  <out>.md               candidates ranked for a reader: cross-field golds the dense retrievers bury
                         that the full model ranks at the top, with a readable multi-hop route through
                         a mechanism frame (function / limitation / method), not a domain hub. Per
                         candidate: the query, the gold and its field, every rank, the top routes on
                         every graph with the CCMP gate per hop.
  <out>.tex              the top --n-tex candidates in the GFM-RAG Table 4 layout (query / inspiration /
                         ranks / paths), one block per example.
  <out>_candidates.json  the ranked candidate list (machine-readable).
  <out>_hops.{pdf,png}   the hop figure (GFM-RAG Fig. 6 analogue): share of golds by length of the
                         top path under each arm, against the shortest seed->gold route on the graph
                         (a structural floor, NOT a ground-truth reasoning path; the caption must say so),
                         one panel per stratum, with the mean gap to the floor printed per arm.
  <out>_hops.json        the distribution behind the figure, for --combine.

--combine a_hops.json b_hops.json ... draws one row of panels (one per dataset, cross stratum) into
--out.{pdf,png}, the multi-dataset figure for the thesis.

usage (one dataset):
  showcase.py --dataset sir4_cs --queries raw/test.json --docs raw/documents.json \\
      --edges processed/stage1/edges.csv \\
      --arm "SciGraphIR (frame graph + CCMP)=hops_frame_ccmp.json" \\
      --arm "frame graph, CCMP gate off=hops_frame_ccmp_off.json" --arm "OpenIE graph=hops_openie.json" \\
      [--pred qwen3=predictions_qwen3_sir4_cs_test.json --pred bge=...] [--quartet eval.json] \\
      --out results/qualitative/showcase_sir4_cs --top 30 --n-tex 4
'''
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics as st
from collections import Counter, defaultdict

BIG = 10 ** 6
MECH = ("function", "limitation", "method", "finding")     # frame types that carry a mechanism
SYS_LABEL = {"bm25": "BM25", "bge": "BGE-large", "qwen3": "Qwen3-Emb.", "specter2": "SPECTER2", "scincl": "SciNCL",
             "reasonir": "ReasonIR-8B", "dense": "Qwen3 cosine", "scorer": "multi-view scorer", "graph": "graph channel",
             "fused": "SciGraphIR"}


# ----------------------------------------------------------------------------------------------- helpers
def tex_escape(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", str(s)).replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")


def short(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def title_of(doc_id: str, docs: dict, n: int = 110) -> str:
    return short(docs.get(doc_id, doc_id).split(". ")[0], n)


def ntype(name: str, docs: dict) -> str:
    if name in docs:
        return "paper"
    if name.startswith("[") and "]" in name:
        return name[1 : name.index("]")]
    return "entity"


def node_label(name: str, docs: dict, n: int = 60) -> str:
    return "[paper] " + title_of(name, docs, n) if name in docs else short(name, n)


def rel_label(r: str) -> str:
    return r.replace("inverse_", "inv. ").replace("_", " ")


def valid(p: dict, seeds: set) -> bool:
    h = p.get("hops") or []
    return bool(h) and h[0]["head"] in seeds and all(a["tail"] == b["head"] for a, b in zip(h, h[1:]))


def fmt_path(p: dict, docs: dict, gate: bool) -> str:
    parts = []
    for h in p["hops"]:
        g = f" (gate {h['gate']:.2f})" if gate and "gate" in h else ""
        parts.append(f"{node_label(h['head'], docs)}{g} --{rel_label(h['rel'])}-->")
    parts.append(node_label(p["hops"][-1]["tail"], docs))
    return " ".join(parts)


def fmt_path_tex(p: dict, docs: dict, gate: bool) -> str:
    parts = []
    for h in p["hops"]:
        g = f" {{\\scriptsize({h['gate']:.2f})}}" if gate and "gate" in h else ""
        parts.append(f"{tex_escape(node_label(h['head'], docs, 48))}{g} $\\xrightarrow{{\\text{{{tex_escape(rel_label(h['rel']))}}}}}$")
    parts.append(tex_escape(node_label(p["hops"][-1]["tail"], docs, 48)))
    return " ".join(parts)


def ranked_docs(rec) -> list:
    p = rec.get("predictions", rec)
    d = p.get("document", p) if isinstance(p, dict) else p
    return [x[0] if isinstance(x, (list, tuple)) else x for x in d]


def fnum(x, spec: str = ".2f") -> str:
    '''Format a number, or 'n/a' when a statistic is undefined (no valid route, or no min_hops in the file).'''
    return "n/a" if x is None else format(x, spec)


def load_arm(path: str) -> dict:
    '''{(qid, gold): target-with-context} for one interpret_paths output.'''
    out = {}
    for r in json.load(open(path)):
        seeds = set(r.get("seeds", []))
        for t in r.get("targets", []):
            ps = [p for p in t.get("paths", []) if valid(p, seeds)]
            out[(r["id"], t["doc"])] = {"rank": t["rank"], "min_hops": t.get("min_hops"), "paths": ps,
                                        "n_raw_paths": len(t.get("paths", [])), "views": t.get("views", []),
                                        "dense_cos": t.get("dense_cos"), "stratum": r.get("stratum") or "same",
                                        "question": r.get("question", ""), "seeds": r.get("seeds", [])}
    if out and all(t["min_hops"] is None for t in out.values()):
        print(f"[showcase] WARNING {os.path.basename(path)}: no target carries min_hops; the engine that wrote it predates "
              "the structural floor (apply gfm_overlay.zip and rerun the path stage). The hop figure will show no floor.")
    return out


def doc_domains(edges_csv: str | None) -> dict:
    '''document id -> '[domain] ...' node names from the frame graph's in_field edges.'''
    dom = defaultdict(list)
    if not edges_csv or not os.path.exists(edges_csv):
        return dom
    import csv
    with open(edges_csv, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 3 and row[1] == "in_field":
                dom[row[0]].append(row[2].replace("[domain] ", ""))
    return dom


def quartet_fields(path: str | None) -> dict:
    '''(qid, gold) -> 'Computer Science -> Engineering' from the QUARTET export, when available.'''
    out = {}
    if not path or not os.path.exists(path):
        return out
    for x in json.load(open(path)):
        for d, m in (x.get("quartet", {}).get("per_document") or {}).items():
            if m.get("field_pair"):
                out[(x["id"], d)] = {"field_pair": m["field_pair"], "stratum": m.get("stratum")}
    return out


def route_kind(p: dict | None, docs: dict) -> str:
    '''bridge = passes a mechanism frame; hub = only papers/domain/task/entity nodes; none = no valid path.'''
    if not p:
        return "none"
    inner = [ntype(h["head"], docs) for h in p["hops"][1:]] + [ntype(p["hops"][0]["head"], docs)]
    if any(t in MECH for t in inner):
        return "bridge"
    if any(t == "domain" for t in inner):
        return "hub"
    return "other"


# ----------------------------------------------------------------------------------------------- hop figure
def hop_stats(arms: list, docs: dict, max_hops: int) -> dict:
    '''Per stratum and arm: distribution of top-path length, the shortest-route floor, mean gap, unreachable.'''
    strata = sorted({t["stratum"] for _, tab in arms for t in tab.values()}, key=lambda s: (s != "same", s))
    out = {"max_hops": max_hops, "strata": strata, "arms": [lab for lab, _ in arms], "dist": {}}
    for s in strata:
        out["dist"][s] = {}
        first = arms[0][1]
        floor = [t["min_hops"] for t in first.values() if t["stratum"] == s and t["min_hops"] is not None]
        cf = Counter(min(h, max_hops) for h in floor)
        out["dist"][s]["floor"] = {"n": len(floor), "pct": [100 * cf.get(h, 0) / max(1, len(floor)) for h in range(1, max_hops + 1)],
                                   "mean": st.mean(floor) if floor else None}
        for lab, tab in arms:
            sub = [t for t in tab.values() if t["stratum"] == s]
            hops = [len(t["paths"][0]["hops"]) for t in sub if t["paths"]]
            gaps = [len(t["paths"][0]["hops"]) - t["min_hops"] for t in sub if t["paths"] and t["min_hops"] is not None]
            c = Counter(min(h, max_hops) for h in hops)
            out["dist"][s][lab] = {"n_golds": len(sub), "n_paths": len(hops), "unreachable": len(sub) - len(hops),
                                   "pct": [100 * c.get(h, 0) / max(1, len(hops)) for h in range(1, max_hops + 1)],
                                   "mean": st.mean(hops) if hops else None, "mean_gap": st.mean(gaps) if gaps else None,
                                   "mae": st.mean(abs(g) for g in gaps) if gaps else None}
    return out


def draw_hops(panels: list, out_prefix: str, title_fs: float = 10) -> None:
    '''panels: [(title, stats_dict, stratum)] -> one row of panels, GFM-RAG Fig. 6 style.

    Colour and marker are keyed on the ARM LABEL, not its position, because datasets do not all have
    the same arms (a missing checkpoint drops one) and a shared legend taken from the first panel would
    then mislabel every other panel's curves. The legend is the union over panels.

    Every number printed here can be undefined (an arm with no valid route has no mean gap, a file
    written before min_hops existed has no floor), so all of them go through fnum() and a panel with
    nothing to draw is dropped rather than crashing the run.
    '''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(t, S, s) for t, S, s in panels if s in (S.get("dist") or {})]
    if not panels:
        print("[showcase] no stratum carries any interpreted gold; the hop figure is skipped")
        return
    labels = []                       # arm labels in first-seen order: the colour key
    for _, S, _s in panels:
        for lab in S.get("arms") or []:
            if lab not in labels:
                labels.append(lab)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n + 0.4, 3.0), squeeze=False)
    mk = ["s", "D", "^", "v", "P", "X"]
    col = ["#1f77b4", "#7f7f7f", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    for ax, (title, S, s) in zip(axes[0], panels):
        H = S.get("max_hops") or 6; xs = list(range(1, H + 1)); D = S["dist"][s]
        fl = D.get("floor") or {}
        if fl.get("n"):
            ax.plot(xs, fl["pct"], marker="o", color="#2ca02c", lw=1.8, ms=6,
                    label=f"shortest route on the {(S.get('arms') or [''])[0].split('(')[-1].rstrip(')').split('+')[0].strip() or 'first'} graph (floor)")
        drawn = 0
        for lab in S.get("arms") or []:
            d = D.get(lab)
            if not d or not d.get("n_paths"):
                continue
            k = labels.index(lab)
            ax.plot(xs, d["pct"], marker=mk[k % len(mk)], color=col[k % len(col)], lw=1.6, ms=5.5, label=lab)
            ax.text(0.03, 0.95 - 0.085 * drawn,
                    f"{short(lab, 30)}: mean gap {fnum(d.get('mean_gap'), '+.2f')}, no valid route {d.get('unreachable', 0)}",
                    transform=ax.transAxes, fontsize=6.6, va="top", color=col[k % len(col)])
            drawn += 1
        ax.set_xticks(xs); ax.set_xlabel("hops"); ax.set_ylabel("share of golds (%)")
        # n = golds behind the floor when there is one, otherwise golds interpreted under the first arm
        n_g = fl.get("n") or max((d.get("n_golds", 0) for k_, d in D.items() if k_ != "floor"), default=0)
        ax.set_title(f"{title} ({s}-field, n={n_g}{'' if fl.get('n') else ', no floor'})", fontsize=title_fs)
        ax.grid(True, ls="--", alpha=0.4); ax.set_ylim(0, max(50, ax.get_ylim()[1]))
    seen, h, l = {}, [], []           # union legend, deduplicated by label
    for ax in axes[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in seen:
                seen[ll] = 1; h.append(hh); l.append(ll)
    if l:
        fig.legend(h, l, loc="upper center", ncol=min(4, len(l)), fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combine", nargs="*", default=None, help="<out>_hops.json files; draws the multi-dataset figure to --out")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--queries"); ap.add_argument("--docs"); ap.add_argument("--edges", default=None)
    ap.add_argument("--arm", action="append", default=[], help="label=hops json; the first is the showcased model")
    ap.add_argument("--pred", action="append", default=[], help="name=predictions json of a baseline (rank of the gold)")
    ap.add_argument("--quartet", default=None, help="QUARTET eval.json for the field pair of each gold (SIR-4 only)")
    ap.add_argument("--stratum", default="auto", help="cross | same | any | auto (cross when the dataset has it)")
    ap.add_argument("--max-fused", type=int, default=25); ap.add_argument("--max-graph", type=int, default=5)
    ap.add_argument("--min-dense", type=int, default=25, help="the gold must be at least this deep under raw cosine")
    ap.add_argument("--min-hops", type=int, default=2); ap.add_argument("--max-hops", type=int, default=6)
    ap.add_argument("--top", type=int, default=30); ap.add_argument("--n-tex", type=int, default=4)
    ap.add_argument("--paths-per-arm", type=int, default=2)
    ap.add_argument("--sample-qids", default=None, help="json list of query ids; the hop figure uses only these (unbiased sample)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.combine is not None:
        panels = []
        for p in a.combine:
            S = json.load(open(p))
            if not S.get("strata"):
                print(f"[showcase] {os.path.basename(p)} has no interpreted gold; skipped")
                continue
            s = "cross" if "cross" in S["strata"] else S["strata"][0]
            panels.append((S.get("dataset", os.path.basename(p)), S, s))
        draw_hops(panels, a.out)
        print("wrote", a.out + ".pdf")
        return 0

    assert a.arm and a.queries and a.docs, "--arm, --queries and --docs are required"
    docs = json.load(open(a.docs))
    queries = {q["id"]: q for q in json.load(open(a.queries))}
    arms = []
    for spec in a.arm:
        lab, path = spec.split("=", 1)
        arms.append((lab, load_arm(path)))
    main_lab, main_tab = arms[0]
    preds = {}
    for spec in a.pred:
        nm, path = spec.split("=", 1)
        if os.path.exists(path):
            preds[nm] = {r["id"]: ranked_docs(r) for r in json.load(open(path))}
    dom = doc_domains(a.edges)
    qf = quartet_fields(a.quartet)
    strata_present = {t["stratum"] for t in main_tab.values()}
    stratum = a.stratum if a.stratum != "auto" else ("cross" if "cross" in strata_present else "any")

    # ---- candidates ------------------------------------------------------------------------------
    cands = []
    for (qid, gold), t in main_tab.items():
        # a "cross" query can carry same-field golds too; use the gold's own label when QUARTET has it
        g_strat = (qf.get((qid, gold)) or {}).get("stratum") or t["stratum"]
        if stratum != "any" and g_strat != stratum:
            continue
        rk = t["rank"]
        gk = lambda c: rk.get(c, BIG)      # a channel the scan did not write = "did not retrieve it", never a KeyError
        # eligible: cosine buries the gold and either the graph channel or the full model recovers it
        if gk("dense") < a.min_dense or (gk("graph") > a.max_graph and gk("fused") > a.max_fused):
            continue
        p = t["paths"][0] if t["paths"] else None
        kind = route_kind(p, docs)
        hops = len(p["hops"]) if p else 0
        gates = [h.get("gate") for h in p["hops"]] if p else []
        row = {"id": qid, "gold": gold, "stratum": g_strat, "question": queries.get(qid, {}).get("question", t["question"]),
               "gold_title": title_of(gold, docs), "gold_domain": "; ".join(dom.get(gold, [])) or None,
               "field_pair": (qf.get((qid, gold)) or {}).get("field_pair"), "ranks": dict(rk), "route": kind, "hops": hops,
               "max_gate": max((g for g in gates if g is not None), default=None), "min_hops": t["min_hops"],
               "dense_cos": t["dense_cos"], "n_golds": len(queries.get(qid, {}).get("supporting_documents", []) or [gold])}
        for lab, tab in arms[1:]:
            o = tab.get((qid, gold))
            row["ranks"][f"{lab}|fused"] = o["rank"].get("fused") if o else None
            row["ranks"][f"{lab}|graph"] = o["rank"].get("graph") if o else None
        for nm, tab in preds.items():
            lst = tab.get(qid)
            row["ranks"][nm] = (lst.index(gold) + 1) if lst and gold in lst else (BIG if lst else None)
        # amazingness: a real route through a mechanism, deep under cosine, top under the model,
        # and missed by the OpenIE graph and the dense baselines when we know them
        openie = [v for k, v in row["ranks"].items() if k.endswith("|fused") and "OpenIE" in k and v]
        base = [v for k, v in row["ranks"].items() if k in preds and v]
        row["tier"] = "A" if gk("fused") <= 10 else ("B" if gk("graph") <= 5 else "C")   # A: model top-10; B: graph top-5 only
        row["score"] = ((2 if kind == "bridge" else 0) + (1 if hops >= a.min_hops else 0)
                        + (3 if gk("fused") <= 5 else 2 if gk("fused") <= 10 else 1 if gk("fused") <= 25 else 0)
                        + (1 if gk("graph") <= 5 else 0)
                        + min(3.0, __import__("math").log10(max(1, gk("dense"))))
                        + (1 if openie and min(openie) > 25 else 0) + (1 if base and min(base) > 25 else 0)
                        + (0.5 if row["max_gate"] and row["max_gate"] > 1.05 else 0))
        cands.append(row)
    cands.sort(key=lambda r: (-r["score"], -r["ranks"].get("dense", BIG), r["ranks"].get("fused", BIG)))
    top = cands[: a.top]

    # ---- markdown -------------------------------------------------------------------------------
    md = [f"# {a.dataset or 'dataset'}: showcase candidates ({stratum}-field golds; cosine >= {a.min_dense} and "
          f"(graph <= {a.max_graph} or fused <= {a.max_fused}); tier A = model top-10, B = graph top-5 only, C = rest)", "",
          f"{len(main_tab)} golds with interpretations under '{main_lab}'; {len(cands)} pass the filter; "
          f"routes: {Counter(r['route'] for r in cands)}", "",
          "Read the top rows first. 'bridge' = the top route passes a function / limitation / method / finding frame; "
          "'hub' = it only passes papers and a domain node (the failure signature); gates > 1 are hops CCMP amplified.", ""]
    # the extra columns (other arms, then baselines) are built as one list: with no baseline predictions
    # on disk, joining two groups with " | " left a stray empty column and a malformed markdown table
    extra_hdr = [short(lab, 18) for lab, _ in arms[1:]] + [SYS_LABEL.get(n, n) for n in preds]
    md.append("| # | score | tier | route | hops | cosine | scorer | graph | fused | "
              + "".join(h + " | " for h in extra_hdr) + "gold | field | query |")
    md.append("|--:|--:|---|---|--:|--:|--:|--:|--:|" + "---:|" * len(extra_hdr) + "---|---|---|")
    for i, r in enumerate(top, 1):
        rk = r["ranks"]; f = lambda v: "--" if v is None else (">300" if v >= BIG else str(v))
        extra = [f(rk.get(f"{lab}|fused")) for lab, _ in arms[1:]] + [f(rk.get(n)) for n in preds]
        md.append(f"| {i} | {r['score']:.1f} | {r['tier']} | {r['route']} | {r['hops']} | {f(rk.get('dense'))} | {f(rk.get('scorer'))} | {f(rk.get('graph'))} | {f(rk.get('fused'))} | "
                  + "".join(c + " | " for c in extra)
                  + f"{short(r['gold_title'], 60)} | {short(r['field_pair'] or r['gold_domain'] or '', 40)} | {short(r['question'], 80).replace('|', '/')} |")
    md.append("")
    for i, r in enumerate(top, 1):
        qid, gold = r["id"], r["gold"]
        md += [f"## {i}. {r['id']}  ->  {r['gold_title']}", "",
               f"**Field:** {r['field_pair'] or r['gold_domain'] or 'unknown'} | **stratum:** {r['stratum']} | "
               f"**golds for this query:** {r['n_golds']} | **route:** {r['route']}, {r['hops']} hops (shortest {r['min_hops']})", "",
               f"**Query:** {short(r['question'], 700)}", "",
               f"**Inspiration (gold):** {short(docs.get(gold, gold), 500)}", "",
               "**Ranks of this gold:** " + ", ".join(f"{SYS_LABEL.get(k, k)} {v if v < BIG else '>300'}" for k, v in r["ranks"].items() if v is not None), ""]
        t = main_tab[(qid, gold)]
        if t["views"]:
            md.append("**Scorer views that matched best (hypothetical answers written for the query):**")
            for v in t["views"]:
                md.append(f"- view {v['view']} match {v['match']:.3f}: {short(v.get('text') or '(text unavailable)', 240)}")
            md.append("")
        for lab, tab in arms:
            o = tab.get((qid, gold))
            md.append(f"**{lab}**" + (f" (ranks: {', '.join(f'{k} {v}' for k, v in o['rank'].items())})" if o else ": no interpretation"))
            if o:
                gate = any("gate" in h for p in o["paths"] for h in p["hops"])
                if not o["paths"]:
                    md.append("- (no valid path within the reasoner's depth)")
                for p in o["paths"][: a.paths_per_arm]:
                    md.append(f"- w={p['weight']:.2f}: {fmt_path(p, docs, gate)}")
            md.append("")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    json.dump({"dataset": a.dataset, "stratum": stratum, "n_interpreted": len(main_tab), "n_pass": len(cands), "candidates": cands},
              open(a.out + "_candidates.json", "w"), indent=1)

    # ---- LaTeX (GFM-RAG Table 4 layout, one block per example) ------------------------------------
    tex = ["% generated by showcase.py; edit freely",
           "\\begin{table}[t]\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.15}",
           "\\begin{tabular}{@{}p{0.13\\linewidth} p{0.85\\linewidth}@{}}\n\\toprule"]
    for r in top[: a.n_tex]:
        qid, gold = r["id"], r["gold"]
        field = r["field_pair"] or r["gold_domain"] or ""
        rk = r["ranks"]
        rank_txt = ", ".join(f"{SYS_LABEL.get(k, k)} {v if v < BIG else '$>$300'}" for k, v in rk.items()
                             if v is not None and k in ("dense", "scorer", "graph", "fused", "qwen3", "bge"))
        for lab, _ in arms[1:]:
            v = rk.get(f"{lab}|fused")
            if v:
                rank_txt += f", {tex_escape(lab)} {v}"
        tex.append(f"\\textbf{{Query}} & {tex_escape(short(r['question'], 420))} \\\\")
        tex.append(f"\\textbf{{Inspiration}} & {tex_escape(r['gold_title'])}" + (f" {{\\scriptsize({tex_escape(field)})}}" if field else "") + " \\\\")
        tex.append(f"\\textbf{{Rank}} & {{\\scriptsize {rank_txt}}} \\\\")
        lines = []
        for lab, tab in arms:
            o = tab.get((qid, gold))
            if not o:
                continue
            gate = any("gate" in h for p in o["paths"] for h in p["hops"])
            for p in o["paths"][: (a.paths_per_arm if lab == main_lab else 1)]:
                lines.append(f"{{\\scriptsize {tex_escape(lab)}}} {p['weight']:.2f}: {fmt_path_tex(p, docs, gate)}")
            if not o["paths"]:
                lines.append(f"{{\\scriptsize {tex_escape(lab)}}}: no route within {a.max_hops} hops")
        tex.append("\\textbf{Paths} & " + " \\newline ".join(lines) + " \\\\\n\\midrule")
    if tex[-1].endswith("\\midrule"):
        tex[-1] = tex[-1][: -len("\n\\midrule")]
    tex.append("\\bottomrule\n\\end{tabular}")
    tex.append(f"\\caption{{Path interpretations on {tex_escape(a.dataset)}: cross-field queries whose gold inspiration the dense "
               "retrievers bury and SciGraphIR ranks at the top. Paths are the highest-weighted routes from a query seed frame to "
               "the gold under each graph (gradient beam search over per-layer edge weights, as in NBFNet and GFM-RAG); numbers in "
               "parentheses are the CCMP gate on each hop's sender (1 = frontier mean; $>$1 amplified). Ranks are the position of "
               "the gold under each channel and system.}")
    tex.append(f"\\label{{tab:showcase-{a.dataset.replace('_', '-')}}}\n\\end{{table}}")
    open(a.out + ".tex", "w").write("\n".join(tex) + "\n")

    # ---- hop figure ------------------------------------------------------------------------------
    if a.sample_qids and os.path.exists(a.sample_qids):
        keep = set(json.load(open(a.sample_qids)))
        arms_fig = [(lab, {k: v for k, v in tab.items() if k[0] in keep}) for lab, tab in arms]
    else:
        arms_fig = arms
    S = hop_stats(arms_fig, docs, a.max_hops); S["dataset"] = a.dataset; S["sampled"] = bool(a.sample_qids)
    json.dump(S, open(a.out + "_hops.json", "w"), indent=1)
    try:
        draw_hops([(a.dataset, S, s) for s in S["strata"]], a.out + "_hops")
    except Exception as e:      # the markdown, the LaTeX and the distribution are already on disk
        print(f"[showcase] the hop figure failed ({type(e).__name__}: {e}); {a.out}.md/.tex/_hops.json are written, "
              "redraw with --combine once the cause is fixed")
    print("\n".join(md[:8 + min(len(top), 12)]))
    print("\nhop figure:")
    for s in S["strata"]:
        D = S["dist"][s]
        print(f"  {s}: floor mean {fnum(D['floor']['mean'])} (n={D['floor']['n']})" + "".join(
            f" | {lab}: mean {fnum(d['mean'])} gap {fnum(d['mean_gap'], '+.2f')} no valid route {d['unreachable']}"
            for lab, d in D.items() if lab != "floor" and d["n_paths"]))
    print(f"\nwrote {a.out}.md, .tex, _candidates.json, _hops.json, _hops.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""")
print(f"wrote {S4}/eval/showcase.py (repo version of 2026-09-08 17:41)\n")

ARM_LABEL = {"frame_ccmp": "SciGraphIR (frame graph + CCMP)", "frame_ccmp_off": "frame graph, CCMP gate off",
             "frame_nocc": "frame graph, no CCMP", "openie": "OpenIE graph"}

def _run(cmd):
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=S4, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        c = os.read(p.stdout.fileno(), 8192)
        if not c: break
        sys.stdout.write(c.decode("utf-8", "replace")); sys.stdout.flush()
    p.wait(); print(f"[{time.time()-t0:.0f}s, exit {p.returncode}]")
    return p.returncode

done, missing = [], []
for d in DATASETS:
    out = f"{DRIVE}/outputs/scan/{d}"
    hops = {n: f"{out}/hops_{n}.json" for n in ARM_LABEL if os.path.exists(f"{out}/hops_{n}.json")}
    if not hops:
        missing.append(d); print(f"[skip] {d}: no hops_*.json under {out}"); continue
    print(f"\n======== {d}: {', '.join(hops)} ========")
    show = f"{out}/showcase_{d}"
    cmd = [sys.executable, "-u", "eval/showcase.py", "--dataset", d,
           "--queries", f"{DATA_ROOT}/{d}_test/raw/test.json",
           "--docs",    f"{DATA_ROOT}/{d}_test/raw/documents.json",
           "--edges",   f"{DATA_ROOT}/{d}_test_v16sc/processed/stage1/edges.csv",
           "--out", show, "--top", str(TOP), "--n-tex", str(N_TEX)]
    if os.path.exists(f"{out}/qids_sample.json"):
        cmd += ["--sample-qids", f"{out}/qids_sample.json"]
    for n in ("frame_ccmp", "frame_ccmp_off", "frame_nocc", "openie"):
        if n in hops: cmd += ["--arm", f"{ARM_LABEL[n]}={hops[n]}"]
    for tag in ("qwen3", "bge", "reasonir", "bm25"):
        p_ = f"{DRIVE}/outputs/baselines/{d}/predictions_{tag}_{d}_test.json"
        if os.path.exists(p_): cmd += ["--pred", f"{tag}={p_}"]
    done.append(d) if _run(cmd) == 0 else missing.append(d)

# the multi-dataset figure, then show everything
hops_json = sorted(glob.glob(f"{DRIVE}/outputs/scan/*/showcase_*_hops.json"))
if hops_json:
    print("\n======== combined hop figure ========")
    _run([sys.executable, "-u", "eval/showcase.py", "--combine", *hops_json,
          "--out", f"{DRIVE}/outputs/scan/fig_hops_all"])
from IPython.display import Image, display
for p_ in [f"{DRIVE}/outputs/scan/fig_hops_all.png"] + [f"{DRIVE}/outputs/scan/{d}/showcase_{d}_hops.png" for d in done]:
    if os.path.exists(p_): print(os.path.relpath(p_, DRIVE)); display(Image(p_))
print("\nredrawn:", done, "| skipped or failed:", missing)
