#!/usr/bin/env python3
"""
table4.py -- GFM-RAG Table-4 style path interpretations, best examples first.

Reads interpret_paths.py outputs with paths (one file per arm). The FIRST --arm is the model being
showcased (normally the OpenIE-graph reasoner); every later arm is shown underneath the same
query/gold so the reader can compare routes on another graph. For an arm whose checkpoint has a
CCMP head, every hop carries the gate the model applied to that hop's SENDER, and the table prints
it after the triple (g = 1 is the frontier mean, g > 1 amplified, g < 1 suppressed).

For one dataset it writes:

  <out>.md               every candidate ranked for a reader: the query, the inspiration (gold),
                         its rank under every channel and baseline, the query entities the routes
                         start from, and the top distinct simple paths on every graph.
  <out>.tex              the top --n-tex examples in the GFM-RAG Table 4 layout: Question /
                         Inspiration / Rank / Seeds / Paths, one row block per example, paths as
                         (head, relation, tail) triples joined by arrows, r^{-1} for inverse edges.
  <out>_candidates.json  the ranked candidates with the rendered paths (machine-readable; --combine
                         reads these to build the multi-dataset table).

--combine a_candidates.json b_candidates.json ... --out <prefix> builds ONE table with a header row
per dataset and --n-tex examples each (the thesis table).

What makes an example "best" (documented so the caption can say it):
  eligible   the gold sits at or below --min-dense under raw Qwen3 cosine (dense retrieval buries
             it) and the showcased model recovers it (graph channel <= --max-graph or fused
             <= --max-fused), the gold belongs to the requested stratum, and the showcased arm has
             at least one valid SIMPLE path (no node repeated) of --min-hops .. --max-hops hops.
  selection score (used only to choose readable qualitative examples; it is not an evaluation
             metric or evidence that one arm performs better overall)
             rank gap log(dense) - log(best of graph/fused)           (how much the graph adds)
             + 2 * share of typed relations on the best path          (mechanism, not co-mention)
             + 0.5 if the best path has 2..4 hops, - 0.5 per hop past 4  (readable)
             - 1.0 per intermediate paper on the best path            (paper -> entity -> paper hopping)
             - 1.0 per domain-hub node on the best path               (SciAfford graph failure signature)
             + 0.5 * log10(weight of the best path), clipped to [-3, 2] (a strong route, not beam noise)
  diversity  at most one example per query in the ranked list.

usage:
  table4.py --dataset sir4_cs --queries raw/test.json --docs raw/documents.json \\
      --edges <SciAfford graph>/processed/stage1/edges.csv [--sir4 eval.json] \\
      --arm "OpenIE graph=hops_t4_openie.json" --arm "SciAfford graph + CCMP=hops_t4_frame_ccmp.json" \\
      [--pred qwen3=predictions_qwen3_sir4_cs_test.json ...] --out outputs/scan/sir4_cs/table4_sir4_cs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict

BIG = 10 ** 6
# relations that only say "co-occurs" (OpenIE) or "same field" (SciAfford graph): not a mechanism
UNTYPED = {"is_mentioned_in", "equivalent", "in_field", "mentions"}
SYS_LABEL = {"bm25": "BM25", "bge": "BGE-large", "qwen3": "Qwen3-Emb.", "specter2": "SPECTER2", "scincl": "SciNCL",
             "reasonir": "ReasonIR-8B", "dense": "Qwen3 cosine rank", "scorer": "semantic scorer", "graph": "graph channel",
             "fused": "SciGraphIR"}
DATASET_LABEL = {"tomato": "TOMATO-Star", "sir4_cs": "SIR-4 Computer Science", "sir4_biology": "SIR-4 Biology",
                 "sir4_physics": "SIR-4 Physics", "sir4_matsci": "SIR-4 Materials Science", "mir": "MIR"}


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
        return name[1: name.index("]")]
    return "entity"


def node_text(name: str, docs: dict, n: int, tex: bool = False) -> str:
    """GFM-RAG prints node names in lower case; papers are printed by title (quoted) so the reader can follow."""
    if name in docs:
        t = title_of(name, docs, n)
        return "``" + tex_escape(t) + "''" if tex else "“" + t + "”"
    return tex_escape(short(name, n)) if tex else short(name, n)


def split_rel(r: str) -> tuple[str, bool]:
    """('builds on', True) for 'inverse_builds_on'-style names; the flag says the edge was walked backwards."""
    inv = r.startswith("inverse_") or r.startswith("inv_")
    base = r.split("_", 1)[1] if inv else r
    return base.replace("_", " "), inv


def is_typed(r: str) -> bool:
    return split_rel(r)[0].replace(" ", "_") not in UNTYPED


def valid(p: dict, seeds: set) -> bool:
    h = p.get("hops") or []
    return bool(h) and h[0]["head"] in seeds and all(a["tail"] == b["head"] for a, b in zip(h, h[1:]))


def simple(p: dict) -> bool:
    """No node visited twice (the beam search happily walks A -> B -> A -> gold)."""
    nodes = [p["hops"][0]["head"]] + [h["tail"] for h in p["hops"]]
    return len(nodes) == len(set(nodes))


def path_key(p: dict) -> tuple:
    return tuple((h["head"], h["rel"], h["tail"]) for h in p["hops"])


def distinct_simple(paths: list, k: int, min_hops: int = 1, max_hops: int = 99) -> list:
    out, seen = [], set()
    for p in sorted(paths, key=lambda x: -x["weight"]):
        if not simple(p) or not (min_hops <= len(p["hops"]) <= max_hops):
            continue
        key = path_key(p)
        if key in seen:
            continue
        seen.add(key); out.append(p)
        if len(out) >= k:
            break
    return out


def path_features(p: dict, docs: dict) -> dict:
    hops = p["hops"]
    inner = [h["tail"] for h in hops[:-1]]                  # intermediate nodes (not the seed, not the gold)
    return {"hops": len(hops),
            "typed": sum(is_typed(h["rel"]) for h in hops) / max(1, len(hops)),
            "inner_papers": sum(1 for x in inner if ntype(x, docs) == "paper"),
            "domain_hubs": sum(1 for x in inner if ntype(x, docs) == "domain"),
            "max_gate": max((h["gate"] for h in hops if h.get("gate") is not None), default=None),
            "min_gate": min((h["gate"] for h in hops if h.get("gate") is not None), default=None)}


def fmt_path_md(p: dict, docs: dict) -> str:
    parts = []
    for h in p["hops"]:
        rel, inv = split_rel(h["rel"])
        g = f" [g={h['gate']:.2f}]" if h.get("gate") is not None else ""
        parts.append(f"({node_text(h['head'], docs, 60)}, {rel}{'⁻¹' if inv else ''}, {node_text(h['tail'], docs, 60)}){g}")
    return " → ".join(parts)


def fmt_path_tex(p: dict, docs: dict, n: int = 46) -> str:
    parts = []
    for h in p["hops"]:
        rel, inv = split_rel(h["rel"])
        g = f"{{\\scriptsize\\,g={h['gate']:.2f}}}" if h.get("gate") is not None else ""
        parts.append(f"({node_text(h['head'], docs, n, tex=True)}, {tex_escape(rel)}{'$^{-1}$' if inv else ''}, "
                     f"{node_text(h['tail'], docs, n, tex=True)}){g}")
    return " $\\rightarrow$ ".join(parts)


def fw(w: float) -> str:
    """Path weight: two decimals, or two significant digits when it is tiny (a 0.00 tells the reader nothing)."""
    return f"{w:.2f}" if w >= 0.01 else f"{w:.2g}"


def ranked_docs(rec) -> list:
    p = rec.get("predictions", rec)
    d = p.get("document", p) if isinstance(p, dict) else p
    return [x[0] if isinstance(x, (list, tuple)) else x for x in d]


def load_arm(path: str) -> dict:
    """{(qid, gold): target-with-context} for one interpret_paths output; only valid paths are kept."""
    out = {}
    for r in json.load(open(path)):
        seeds = set(r.get("seeds", []))
        for t in r.get("targets", []):
            ps = [p for p in t.get("paths", []) if valid(p, seeds)]
            out[(r["id"], t["doc"])] = {"rank": t["rank"], "min_hops": t.get("min_hops"), "paths": ps,
                                        "views": t.get("views", []), "dense_cos": t.get("dense_cos"),
                                        "stratum": r.get("stratum") or "same", "question": r.get("question", ""),
                                        "seeds": r.get("seeds", []), "has_gate": any("gate" in h for p in ps for h in p["hops"])}
    return out


def doc_domains(edges_csv: str | None) -> dict:
    """document id -> domain names from the SciAfford graph's in_field edges (a field label when SIR-4 is absent)."""
    dom = defaultdict(list)
    if not edges_csv or not os.path.exists(edges_csv):
        return dom
    import csv
    with open(edges_csv, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 3 and row[1] == "in_field":
                dom[row[0]].append(row[2].replace("[domain] ", ""))
    return dom


def sir4_fields(path: str | None) -> dict:
    out = {}
    if not path or not os.path.exists(path):
        return out
    for x in json.load(open(path)):
        for d, m in (x.get("quartet", {}).get("per_document") or {}).items():
            if m.get("field_pair"):
                out[(x["id"], d)] = {"field_pair": m["field_pair"], "stratum": m.get("stratum")}
    return out


def rank_txt(ranks: dict, preds_ranks: dict, extra: list) -> str:
    """Render shared baselines followed by each arm's semantic/graph/fused ranks."""
    parts = [f"{SYS_LABEL['dense']} {ranks['dense']}"] if ranks.get("dense") is not None else []
    for nm, v in preds_ranks.items():
        parts.append(f"{SYS_LABEL.get(nm, nm)} {v if v else '>300'}")
    return ", ".join(parts) + ("; semantic/graph/fused ranks: " + ", ".join(extra) if extra else "")


# ----------------------------------------------------------------------------------------------- rendering
def render_tex_block(c: dict, n_paths: int) -> list:
    """The rows of one example in the Table-4 layout (Question / Inspiration / Rank / Seeds / Paths)."""
    rows = [f"\\textbf{{Question}} & {tex_escape(short(c['question'], 380))} \\\\",
            f"\\textbf{{Inspiration}} & {tex_escape(c['gold_title'])}"
            + (f" {{\\scriptsize({tex_escape(c['field'])})}}" if c.get("field") else "") + " \\\\",
            f"\\textbf{{Rank}} & {{\\scriptsize {tex_escape(c['rank_txt'])}}} \\\\",
            f"\\textbf{{Seeds}} & {{\\scriptsize {tex_escape(', '.join(c['seeds_used']))}}} \\\\"]
    lines = []
    for arm in c["arms"]:
        head = f"{{\\scriptsize\\textit{{{tex_escape(arm['label'])}}}}}"
        if not arm["paths"]:
            lines.append(f"{head}: no valid route")
            continue
        for p in arm["paths"][:n_paths]:
            lines.append(f"{head} {fw(p['weight'])}: {p['tex']}")
    rows.append("\\textbf{Paths} & " + " \\newline ".join(lines) + " \\\\")
    return rows


def tex_table(blocks: list, label: str, caption: str) -> str:
    out = ["% generated by eval/table4.py; edit freely", "\\begin{table*}[t]", "\\centering\\footnotesize",
           "\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.15}",
           "\\begin{tabular}{@{}p{0.11\\linewidth} p{0.87\\linewidth}@{}}", "\\toprule"]
    for i, b in enumerate(blocks):
        if i:
            out.append("\\midrule")
        out += b
    out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}"]
    return "\n".join(out) + "\n"


CAPTION = ("Path interpretations{where}: queries whose gold inspiration dense retrieval buries (Qwen3 cosine rank "
           "$\\geq$ {min_dense}) and at least one graph reasoner ({main}) recovers. Paths are the highest-weighted routes from a query entity "
           "(a seed) to the inspiration, found by gradient beam search over the per-layer edge weights of the graph "
           "channel, as in NBFNet and GFM-RAG; $r^{{-1}}$ is the inverse of relation $r$. For an arm with CCMP, "
           "$g$ after a triple is the gate applied to that hop's sender (1 = frontier mean; $>$1 amplified, $<$1 "
           "suppressed). Each arm reports semantic/graph/fused ranks; [G] denotes graph rank $\\leq 5$ and [F] "
           "denotes fused rank $\\leq 10$. Lower ranks are better. The selection score used to choose examples is "
           "a presentation heuristic, not an evaluation metric.")


def caption_for(datasets: list, main, min_dense: int) -> str:
    where = " on " + " and ".join(DATASET_LABEL.get(d, d) for d in datasets) if datasets else ""
    arms = main if isinstance(main, list) else [main]
    return CAPTION.format(where=where, min_dense=min_dense, main=tex_escape(", ".join(arms)))


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combine", nargs="*", default=None, help="<out>_candidates.json files -> one multi-dataset table")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--queries"); ap.add_argument("--docs"); ap.add_argument("--edges", default=None)
    ap.add_argument("--arm", action="append", default=[], help="label=hops json; the first is the showcased model")
    ap.add_argument("--pred", action="append", default=[], help="name=predictions json of a baseline (rank of the gold)")
    ap.add_argument("--sir4", "--quartet", dest='sir4', default=None, help="SIR-4 eval.json for the field pair of each gold (SIR-4 only)")
    ap.add_argument("--stratum", default="auto", help="cross | same | any | auto (cross when the dataset has it)")
    ap.add_argument("--min-dense", type=int, default=25); ap.add_argument("--max-graph", type=int, default=5)
    ap.add_argument("--max-fused", type=int, default=10)
    ap.add_argument("--min-hops", type=int, default=2, help="the best path must have at least this many hops")
    ap.add_argument("--max-hops", type=int, default=6)
    ap.add_argument("--top", type=int, default=30, help="candidates in the markdown")
    ap.add_argument("--n-tex", type=int, default=3, help="examples in the LaTeX table (per dataset)")
    ap.add_argument("--paths", type=int, default=3, help="distinct simple paths shown per arm")
    ap.add_argument("--recover", default="any", help="any: a gold is eligible when ANY arm recovers it; main: only the first arm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.combine is not None:
        blocks, datasets, main_lab = [], [], ""
        for p in a.combine:
            C = json.load(open(p))
            if not C.get("candidates"):
                print(f"[table4] {os.path.basename(p)}: no candidate; skipped"); continue
            d = C["dataset"]; datasets.append(d); main_lab = main_lab or C.get("arms") or C["main_arm"]
            blocks.append([f"\\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{{tex_escape(DATASET_LABEL.get(d, d))}}}}} \\\\ \\midrule"]
                          + [row for c in C["candidates"][: a.n_tex] for row in render_tex_block(c, a.paths) + ["\\addlinespace"]])
            blocks[-1] = blocks[-1][:-1]
        open(a.out + ".tex", "w").write(tex_table(blocks, "tab:table4-all", caption_for(datasets, main_lab, a.min_dense)))
        print("wrote", a.out + ".tex", "|", ", ".join(datasets))
        return 0

    assert a.arm and a.queries and a.docs, "--arm, --queries and --docs are required"
    docs = json.load(open(a.docs))
    queries = {q["id"]: q for q in json.load(open(a.queries))}
    arms = []
    for spec in a.arm:
        lab, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[table4] arm '{lab}': {path} missing; skipped"); continue
        arms.append((lab, load_arm(path)))
    assert arms, "no arm file exists"
    main_lab, main_tab = arms[0]
    preds = {}
    for spec in a.pred:
        nm, path = spec.split("=", 1)
        if os.path.exists(path):
            preds[nm] = {r["id"]: ranked_docs(r) for r in json.load(open(path))}
    dom = doc_domains(a.edges)
    qf = sir4_fields(a.sir4)
    if a.dataset.startswith("sir4_") and not qf:
        raise FileNotFoundError(
            "SIR-4 qualitative reporting requires the official SIR-4 eval.json via --sir4; "
            "refusing to infer cross-field labels from graph domain nodes."
        )
    strata_present = ({m.get("stratum") for m in qf.values() if m.get("stratum")}
                      if a.dataset.startswith("sir4_") else
                      {t["stratum"] for t in main_tab.values()})
    stratum = a.stratum if a.stratum != "auto" else ("cross" if "cross" in strata_present else "any")

    def arm_score(t, rk):
        """score of one arm's route to the gold, or None when that arm has no usable simple path"""
        gk = lambda c: rk.get(c, BIG)
        best = distinct_simple(t["paths"], 1, a.min_hops, a.max_hops)
        if not best:
            return None, None
        f = path_features(best[0], docs)
        sc = (math.log1p(gk("dense")) - math.log1p(min(gk("graph"), gk("fused")))
              + 2.0 * f["typed"] + (0.5 if 2 <= f["hops"] <= 4 else 0.0) - 0.5 * max(0, f["hops"] - 4)
              - 1.0 * f["inner_papers"] - 1.0 * f["domain_hubs"]
              + 0.5 * max(-3.0, min(2.0, math.log10(max(best[0]["weight"], 1e-9)))))
        return sc, f

    def recovery_flags(rk):
        return rk.get("graph", BIG) <= a.max_graph, rk.get("fused", BIG) <= a.max_fused

    def recovers(rk):
        graph_ok, fused_ok = recovery_flags(rk)
        return graph_ok or fused_ok

    def arm_rank_cell(label, rk):
        graph_ok, fused_ok = recovery_flags(rk)
        flags = "+".join(x for x, ok in (("G", graph_ok), ("F", fused_ok)) if ok)
        triplet = f"{rk.get('scorer', '--')}/{rk.get('graph', '--')}/{rk.get('fused', '--')}"
        return f"{label} {triplet}" + (f" [{flags}]" if flags else "")

    keys = sorted({k for _, tab in arms for k in tab})
    if a.dataset.startswith("sir4_"):
        missing_official = [k for k in keys
                            if not (qf.get(k, {}).get("stratum") and qf.get(k, {}).get("field_pair"))]
        if missing_official:
            print(f"[table4] omitted {len(missing_official)} query-gold pairs without complete official "
                  "SIR-4 stratum/field_pair metadata")
            missing_official = set(missing_official)
            keys = [k for k in keys if k not in missing_official]
            if not keys:
                raise ValueError("No interpreted query-gold pair has complete official SIR-4 metadata.")
    cands, n_elig = [], 0
    for (qid, gold) in keys:
        recs = [(lab, tab.get((qid, gold))) for lab, tab in arms]
        t0 = next((o for _, o in recs if o), None)
        g_strat = (qf.get((qid, gold)) or {}).get("stratum") or t0["stratum"]
        if stratum != "any" and g_strat != stratum:
            continue
        dense = min((o["rank"].get("dense", BIG) for _, o in recs if o), default=BIG)
        rec_by = [lab for lab, o in recs if o and recovers(o["rank"])]
        if dense < a.min_dense or not (rec_by if a.recover == "any" else (main_lab in rec_by)):
            continue
        n_elig += 1
        scored = [(lab, *arm_score(o, o["rank"])) for lab, o in recs if o and lab in rec_by]
        scored = [(lab, sc, f) for lab, sc, f in scored if sc is not None]
        if not scored:
            continue
        best_lab, score, f = max(scored, key=lambda x: x[1])
        arm_rows, extra, all_path_lengths = [], [], []
        for lab, o in recs:
            ps = distinct_simple(o["paths"], a.paths, 1, a.max_hops) if o else []
            rr = dict(o["rank"]) if o else {}
            graph_ok, fused_ok = recovery_flags(rr)
            arm_rows.append({"label": lab, "rank": rr, "has_gate": bool(o and o["has_gate"]),
                             "recovers": graph_ok or fused_ok, "graph_recovers": graph_ok,
                             "fused_recovers": fused_ok,
                             "paths": [{"weight": p["weight"], "hops": len(p["hops"]), "md": fmt_path_md(p, docs), "tex": fmt_path_tex(p, docs)}
                                       for p in ps]})
            if o:
                extra.append(arm_rank_cell(lab, o["rank"]))
                all_path_lengths.extend(len(p["hops"]) for p in o["paths"] if simple(p))
        pr = {}
        for nm, tab in preds.items():
            lst = tab.get(qid)
            pr[nm] = (lst.index(gold) + 1) if lst and gold in lst else None
        seeds_used = []
        for ar in arm_rows:
            for p in ar["paths"]:
                s0 = p["md"].split(",", 1)[0].lstrip("(")
                if s0 not in seeds_used:
                    seeds_used.append(s0)
        rk = {"dense": dense}
        best_obj = next(o for lab, o in recs if lab == best_lab and o)
        field = (qf.get((qid, gold)) or {}).get("field_pair") or "; ".join(dom.get(gold, [])) or ""
        cands.append({"id": qid, "gold": gold, "stratum": g_strat, "score": round(score, 3), "best_arm": best_lab, "features": f,
                      "recovered_by": rec_by,
                      "question": queries.get(qid, {}).get("question", t0["question"]), "gold_title": title_of(gold, docs),
                      "gold_abstract": short(docs.get(gold, ""), 400), "field": field, "ranks": rk, "pred_ranks": pr,
                      "rank_txt": rank_txt(rk, pr, extra), "seeds_used": seeds_used, "seeds": best_obj["seeds"],
                      "min_hops": min(all_path_lengths) if all_path_lengths else None,
                      "views": best_obj["views"][:1], "arms": arm_rows})
    cands.sort(key=lambda c: -c["score"])
    top, seen_q = [], set()
    for c in cands:
        if c["id"] in seen_q:
            continue
        seen_q.add(c["id"]); top.append(c)
        if len(top) >= a.top:
            break

    # ---- markdown ---------------------------------------------------------------------------------
    md = [f"# {a.dataset}: Table-4 candidates ({stratum} golds; Qwen cosine rank >= {a.min_dense} and, under {'any arm' if a.recover == 'any' else main_lab}, "
          f"graph <= {a.max_graph} or fused <= {a.max_fused}); arms: {', '.join(lab for lab, _ in arms)}; best simple path {a.min_hops}..{a.max_hops} hops)", "",
          f"{len(keys)} interpreted golds; {n_elig} eligible by rank; {len(cands)} with a usable route; "
          f"{len(top)} listed (one per query). Selection score is a presentation heuristic: best over recovering arms of rank gap + 2*typed share + readability - papers/hubs + log weight; "
          "[G] marks graph recovery and [F] marks fused recovery.", "",
          "| # | selection score | best arm | selected-path hops | typed share | Qwen cosine rank | " + " | ".join(f"{lab} s/g/f" for lab, _ in arms)
          + " | gold | field | query |",
          "|--:|--:|---|--:|--:|--:|" + "---:|" * len(arms) + "---|---|---|"]
    for i, c in enumerate(top, 1):
        rk = c["ranks"]; f = c["features"]
        per_cells = []
        for ar in c["arms"]:
            flags = "+".join(x for x, ok in (("G", ar["graph_recovers"]), ("F", ar["fused_recovers"])) if ok)
            rr = ar["rank"]
            cell = f"{rr.get('scorer', '--')}/{rr.get('graph', '--')}/{rr.get('fused', '--')}"
            per_cells.append(cell + (f" [{flags}]" if flags else ""))
        per = " | ".join(per_cells)
        md.append(f"| {i} | {c['score']:.1f} | {short(c['best_arm'], 18)} | {f['hops']} | {f['typed']:.2f} | {rk.get('dense')} | "
                  f"{per} | {short(c['gold_title'], 60)} | {short(c['field'], 40)} | {short(c['question'], 80).replace('|', '/')} |")
    for i, c in enumerate(top, 1):
        md += ["", f"## {i}. {c['id']}  ->  {c['gold_title']}", "",
               f"**Field:** {c['field'] or 'unknown'} | **stratum:** {c['stratum']} | **selection score (not an evaluation metric):** {c['score']} "
               f"(best arm: {c['best_arm']}; recovered by: {', '.join(c['recovered_by'])}) | "
               f"**shortest valid seed-to-gold route across displayed arms:** {c['min_hops']} hops", "",
               f"**Question:** {short(c['question'], 700)}", "",
               f"**Inspiration (gold):** {c['gold_abstract']}", "",
               f"**Rank of this gold:** {c['rank_txt']}", "",
               f"**Seeds the routes start from:** {', '.join(c['seeds_used'])}", ""]
        if c["views"]:
            v = c["views"][0]
            md += [f"**Best semantic view for {c['best_arm']} (match {v.get('match', 0):.2f}):** {short(v.get('text') or '', 220)}", ""]
        for ar in c["arms"]:
            rr = ar["rank"]
            md.append(f"**{ar['label']}**" + (f" (ranks: semantic {rr.get('scorer')}, graph {rr.get('graph')}, fused {rr.get('fused')}, Qwen cosine {rr.get('dense')})" if rr else "")
                      + (" — gate per hop in [g=…]" if ar["has_gate"] else ""))
            md += [f"- {fw(p['weight'])}: {p['md']}" for p in ar["paths"]] or ["- (no valid simple route)"]
            md.append("")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")

    # ---- LaTeX + json ------------------------------------------------------------------------------
    blocks = [render_tex_block(c, a.paths) for c in top[: a.n_tex]]
    open(a.out + ".tex", "w").write(tex_table(blocks, f"tab:table4-{a.dataset.replace('_', '-')}",
                                              caption_for([a.dataset] if a.dataset else [], [lab for lab, _ in arms], a.min_dense)))
    json.dump({"dataset": a.dataset, "main_arm": main_lab, "arms": [lab for lab, _ in arms], "stratum": stratum, "n_interpreted": len(keys), "n_eligible": n_elig,
               "n_with_route": len(cands), "filters": {"min_dense": a.min_dense, "max_graph": a.max_graph, "max_fused": a.max_fused,
               "min_hops": a.min_hops, "max_hops": a.max_hops}, "candidates": top}, open(a.out + "_candidates.json", "w"), indent=1)
    print("\n".join(md[: 6 + min(len(top), 12)]))
    print(f"\nwrote {a.out}.md, .tex, _candidates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
