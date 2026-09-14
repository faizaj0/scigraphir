"""The downstream table in the thesis Table 9.3 layout, for either dataset.

API-FREE.  Rows = arms (closed-book, retrievers, oracle); columns = Top-1 hit
(share of queries whose fed document is a gold), and the mean matched score
(0-12) Overall / Same / Cross, over the PAIRED query set (queries judged under
every arm shown).  Stars: the reference arm (--ref, default scigraphir) vs the
row, paired bootstrap 95% CI excluding zero both Overall and on Cross, as in the
thesis caption.  Also prints every pairwise CI (for the prose), the
hit-conditioned split (hit vs miss mean per arm, pooled), and with --by-field
one block per SIR-4 field.

Run:
  python -m analysis.downstream.downstream_table --tag gpt-4o-batch --latex                       # TOMATO
  DOWNSTREAM_DATASET=sir4 python -m analysis.downstream.downstream_table --tag gpt-4o-batch --latex --by-field
Writes results/<downstream dir>/table__{tag}.json and table__{tag}.tex.
"""
from __future__ import annotations

import argparse
import json
import random

from analysis.downstream._common import DATASET, INPUTS_PATH, OUT_DIR, index_by, read_jsonl
from analysis.downstream.aggregate import _bootstrap_ci, _matched, _mean

ORDER = ["none", "random", "bm25", "bge", "qwen3", "reasonir", "moose_chem", "lattice",
         "ours", "scigraphir", "oracle"]
LABEL = {"none": "Closed-book", "random": "Random doc", "bm25": "BM25", "bge": "BGE",
         "qwen3": "Qwen3", "reasonir": "ReasonIR-8B", "moose_chem": "MOOSE-Chem",
         "lattice": "LATTICE", "ours": "Ours (June)", "scigraphir": "SciGraphIR",
         "oracle": "Oracle (gold)"}
FIELD_NAME = {"cs": "Computer Science", "biology": "Biology", "physics": "Physics",
              "matsci": "Materials Science"}
BOUNDS = ("none", "random", "oracle")          # never bolded


def collect(tag, arms):
    data = {a: _matched(a, tag) for a in arms}
    data = {a: d for a, d in data.items() if d}
    present = [a for a in arms if a in data]
    if not present:
        raise SystemExit(f"no matched_*__{tag}.jsonl under {OUT_DIR}")
    comps = {a: index_by(read_jsonl(OUT_DIR / f"compositions_{a}.jsonl")) for a in present}
    inputs = index_by(read_jsonl(INPUTS_PATH))
    shared = sorted(set.intersection(*(set(data[a]) for a in present)))
    return data, comps, inputs, present, shared


def stats(data, comps, present, qs, ref, n_boot):
    out = {}
    for a in present:
        tot = [data[a][q]["total"] for q in qs]
        if a == "none":
            hit = None
        elif a == "oracle":
            hit = 1.0
        else:
            hit = _mean([1.0 if comps[a].get(q, {}).get("is_gold") else 0.0 for q in qs])
        out[a] = {"mean": _mean(tot), "hit": hit, "n": len(qs)}
    if ref in present:
        for a in present:
            if a == ref:
                continue
            diffs = [data[ref][q]["total"] - data[a][q]["total"] for q in qs]
            lo, hi = _bootstrap_ci(diffs, n_boot=n_boot)
            out[a]["vs_ref"] = {"delta": _mean(diffs), "ci95": [lo, hi],
                                "sig": not (lo <= 0 <= hi)}
    return out


def hit_split(data, comps, present, qs):
    """hit-vs-miss means per retriever arm and pooled."""
    out, pooled = {}, {True: [], False: []}
    for a in present:
        if a in BOUNDS:
            continue
        by = {True: [], False: []}
        for q in qs:
            by[bool(comps[a].get(q, {}).get("is_gold"))].append(data[a][q]["total"])
        out[a] = {"hit": _mean(by[True]), "n_hit": len(by[True]),
                  "miss": _mean(by[False]), "n_miss": len(by[False])}
        for k in by:
            pooled[k].extend(by[k])
    out["pooled"] = {"hit": _mean(pooled[True]), "n_hit": len(pooled[True]),
                     "miss": _mean(pooled[False]), "n_miss": len(pooled[False])}
    return out


def blocks_for(qs, inputs, data, present):
    def stratum(q):
        return inputs.get(q, {}).get("stratum") or data[present[0]][q]["stratum"]
    return [("Overall", qs),
            ("Same", [q for q in qs if stratum(q) == "same"]),
            ("Cross", [q for q in qs if stratum(q) == "cross"])]


def console(title, table, present, ref):
    print(f"\n{title}")
    print(f"  {'Retrieval':<14}{'Top-1 hit':>10}{'Overall':>9}{'Same':>8}{'Cross':>8}   "
          f"{ref} vs row: Overall / Same / Cross  (delta [95% CI], * = CI excludes 0)")
    for a in present:
        o, s, c = (table[b][a] for b in ("Overall", "Same", "Cross"))
        hit = "--" if o["hit"] is None else f"{100 * o['hit']:.1f}%"
        cis = ""
        if a != ref and "vs_ref" in o:
            cis = "   ".join(f"{b['vs_ref']['delta']:+.2f} [{b['vs_ref']['ci95'][0]:+.2f}, "
                            f"{b['vs_ref']['ci95'][1]:+.2f}]{'*' if b['vs_ref']['sig'] else ' '}"
                            for b in (o, s, c))
        print(f"  {LABEL.get(a, a):<14}{hit:>10}{o['mean']:>9.2f}{s['mean']:>8.2f}"
              f"{c['mean']:>8.2f}   {cis}")
    n = {b: table[b][present[0]]["n"] for b in ("Overall", "Same", "Cross")}
    print(f"  n paired: overall {n['Overall']}  same {n['Same']}  cross {n['Cross']}")


def star(a, table, ref):
    if a == ref or "vs_ref" not in table["Overall"][a]:
        return ""
    return r"$^{*}$" if (table["Overall"][a]["vs_ref"]["sig"]
                         and table["Cross"][a]["vs_ref"]["sig"]) else ""


def latex_rows(table, present, ref):
    lines = []
    retr = [a for a in present if a not in BOUNDS]
    best = {b: (max(retr, key=lambda a: table[b][a]["mean"]) if retr else None)
            for b in ("Overall", "Same", "Cross")}
    for a in present:
        if a == "oracle":
            lines.append(r"\midrule")
        o = table["Overall"][a]
        hit = "--" if o["hit"] is None else (r"100\%" if a == "oracle" else f"{100 * o['hit']:.1f}\\%")
        cells = []
        for b in ("Overall", "Same", "Cross"):
            v = f"{table[b][a]['mean']:.2f}"
            cells.append(rf"\textbf{{{v}}}" if best[b] == a else v)
        lines.append(f"{LABEL.get(a, a)}{star(a, table, ref)} & {hit} & " + " & ".join(cells) + r" \\")
    return lines


def latex_main(table, present, ref, n, tag):
    ds = "SIR-4 (four fields pooled)" if DATASET == "sir4" else "TOMATO"
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{Matched score (0--12) of hypotheses composed from the top-1 retrieved document, "
        rf"{ds}, $n={n['Overall']}$ ({n['Same']} same-field, {n['Cross']} cross-field). "
        r"Top-1 hit = share of queries whose retrieved document is a gold inspiration. "
        rf"Paired bootstrap 95\% CIs; $^{{*}}$ {LABEL.get(ref, ref)} vs.\ the row is significant overall and on cross-domain. "
        rf"Judge tag: {tag}.}}",
        r"\label{tab:downstream" + ("-sir4" if DATASET == "sir4" else "") + "}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Retrieval & Top-1 hit & Overall & Same & Cross \\", r"\midrule",
        *latex_rows(table, present, ref),
        r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def latex_by_field(per_field, present, ref, tag):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Downstream matched score (0--12) per SIR-4 field; same layout as the pooled table. "
             rf"$^{{*}}$ {LABEL.get(ref, ref)} vs.\ the row significant overall and on cross-field within the field. Judge tag: {tag}.}}",
             r"\label{tab:downstream-sir4-fields}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Retrieval & Top-1 hit & Overall & Same & Cross \\"]
    for field, (table, n) in per_field.items():
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{FIELD_NAME.get(field, field)}}} "
                     rf"($n={n['Overall']}$: {n['Same']} same / {n['Cross']} cross)}} \\")
        lines.append(r"\midrule")
        lines.extend(latex_rows(table, present, ref))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt-4o-batch")
    ap.add_argument("--arms", nargs="+", default=ORDER,
                    help="arms to show, in this order (absent ones are skipped)")
    ap.add_argument("--ref", default="scigraphir", help="arm the stars are computed against")
    ap.add_argument("--by-field", action="store_true", help="add one block per field (SIR-4)")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    data, comps, inputs, present, shared = collect(args.tag, args.arms)
    print(f"dataset={DATASET}  tag={args.tag}  arms={present}  paired queries={len(shared)}")

    blocks = blocks_for(shared, inputs, data, present)
    table = {name: stats(data, comps, present, qs, args.ref, args.n_boot) for name, qs in blocks}
    n = {name: len(qs) for name, qs in blocks}
    console("POOLED", table, present, args.ref)
    hs = hit_split(data, comps, present, shared)
    print("\n  hit-conditioned (Overall):")
    for a, v in hs.items():
        print(f"    {LABEL.get(a, a) if a != 'pooled' else 'POOLED':<14} "
              f"hit {v['hit']:5.2f} (n={v['n_hit']})   miss {v['miss']:5.2f} (n={v['n_miss']})")

    summary = {"dataset": DATASET, "tag": args.tag, "ref": args.ref, "arms": present,
               "n": n, "pooled": table, "hit_conditioned": hs, "by_field": {}}
    per_field = {}
    if args.by_field:
        fields = []
        for q in shared:
            f = inputs.get(q, {}).get("field")
            if f and f not in fields:
                fields.append(f)
        for f in fields:
            qs_f = [q for q in shared if inputs[q].get("field") == f]
            b = blocks_for(qs_f, inputs, data, present)
            t = {name: stats(data, comps, present, qs, args.ref, args.n_boot) for name, qs in b}
            nf = {name: len(qs) for name, qs in b}
            per_field[f] = (t, nf)
            console(f"FIELD {FIELD_NAME.get(f, f)}", t, present, args.ref)
            summary["by_field"][f] = {"n": nf, "table": t,
                                      "hit_conditioned": hit_split(data, comps, present, qs_f)}

    (OUT_DIR / f"table__{args.tag}.json").write_text(json.dumps(summary, indent=2))
    tex = latex_main(table, present, args.ref, n, args.tag)
    if per_field:
        tex += "\n\n" + latex_by_field(per_field, present, args.ref, args.tag)
    (OUT_DIR / f"table__{args.tag}.tex").write_text(tex + "\n")
    print(f"\nwrote {OUT_DIR / f'table__{args.tag}.json'} and table__{args.tag}.tex")
    if args.latex:
        print("\n" + tex)


if __name__ == "__main__":
    main()
