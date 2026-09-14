"""Per-criterion Idea-Arena Elo table for a single head-to-head pair.

API-FREE.  Reads an idea_arena jsonl and, for ONE pair of arms (default
BM25 vs MOOSE-Chem), prints a table in the form:

    Metric          Naïve Retrieval   MI Retrieval
    IdeaArena
    Novelty               996             1004
    Significance          985             1015
    ...
    Overall               990             1010

Each row is a 2-player Bradley-Terry Elo over that criterion's DIRECT matchups
(both orders, ties = 0.5), anchored so the two ratings are symmetric around 1000
(they sum to exactly 2000).  "Overall" pools all 5 criteria.  With ties counted
as half-points the 2-player BT strength ratio is just the points ratio, so

    Elo_hi = 1000 + round(200 * log10(pts_hi / pts_lo)),  Elo_lo = 2000 - Elo_hi.

Run (canonical = gpt-4o judge):
  cd TOMATO-Star
  python -m analysis.downstream.arena_criteria_table --tag gpt-4o
  python -m analysis.downstream.arena_criteria_table --tag gpt-4o --naive none --mi moose_chem

Output: stdout table (Overall / Same / Cross) + a markdown block to paste.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

from analysis.downstream._common import OUT_DIR, read_jsonl
from analysis.downstream.idea_arena import CRITERIA, CRIT_DISPLAY
from analysis.downstream.aggregate import (
    _coi_points_pair, _bradley_terry_elo, ARM_ORDER, ARM_DISPLAY)


def _pair_points(rows, naive, mi):
    """Direct-matchup points for (naive, mi) per criterion, both orders.
    Returns (pts[arm][crit], n_records)."""
    pts = {naive: defaultdict(float), mi: defaultdict(float)}
    n = 0
    for r in rows:
        if {r["arm0"], r["arm1"]} != {naive, mi}:
            continue
        a0, a1, res = _coi_points_pair(r)
        n += 1
        for crit in CRITERIA:
            p0, p1 = res[crit]
            pts[a0][crit] += p0
            pts[a1][crit] += p1
    return pts, n


def _elo_pair(pts_hi, pts_lo):
    """2-player anchored Elo: hi = 1000 + d, lo = 1000 - d, d integer."""
    hi = max(pts_hi, 1e-9)
    lo = max(pts_lo, 1e-9)
    d = round(200.0 * math.log10(hi / lo))
    return 1000 + d, 1000 - d


def _table(rows, naive, mi, label):
    pts, n = _pair_points(rows, naive, mi)
    if n == 0:
        return None
    naive_disp = ARM_DISPLAY.get(naive, naive)
    mi_disp = ARM_DISPLAY.get(mi, mi)
    print(f"\n[{label}]  n={n} direct matchups   "
          f"(Naive={naive_disp}, MI={mi_disp})")
    print(f"  {'Metric':<16}{'Naive (' + naive_disp + ')':>22}"
          f"{'MI (' + mi_disp + ')':>22}")
    print("  IdeaArena")
    rows_out = []
    tot_naive = tot_mi = 0.0
    for crit in CRITERIA:
        pn, pm = pts[naive][crit], pts[mi][crit]
        tot_naive += pn
        tot_mi += pm
        e_mi, e_naive = _elo_pair(pm, pn)          # mi is the "hi" column
        print(f"  {CRIT_DISPLAY[crit]:<16}{e_naive:>22d}{e_mi:>22d}")
        rows_out.append((CRIT_DISPLAY[crit], e_naive, e_mi))
    e_mi, e_naive = _elo_pair(tot_mi, tot_naive)
    print(f"  {'Overall':<16}{e_naive:>22d}{e_mi:>22d}")
    rows_out.append(("Overall", e_naive, e_mi))
    return {"naive": naive_disp, "mi": mi_disp, "n": n, "rows": rows_out}


def _markdown(res, label):
    print(f"\n#### IdeaArena Elo — {label}  (n={res['n']})\n")
    print(f"| Metric | Naïve Retrieval ({res['naive']}) | "
          f"MI Retrieval ({res['mi']}) |")
    print("|---|---|---|")
    for name, e_naive, e_mi in res["rows"]:
        bold = "**" if name == "Overall" else ""
        print(f"| {bold}{name}{bold} | {bold}{e_naive}{bold} | "
              f"{bold}{e_mi}{bold} |")


# --------------------------------------------------------------------------
# Multi-method table: per-criterion multi-player Bradley-Terry Elo, ALL arms
# --------------------------------------------------------------------------
def _multi_elo(rows, arms):
    """Per-criterion multi-player BT Elo (anchored to mean 1000 across the arms
    present in each criterion), plus a pooled 'overall'.  Returns
    {criterion|'overall': {arm: elo}} restricted to `arms`."""
    out = {}
    for crit in CRITERIA + ["overall"]:
        wins = defaultdict(float)            # arm -> points on this criterion
        games = defaultdict(float)           # (i,j) -> symmetric point-units
        for r in rows:
            a0, a1, res = _coi_points_pair(r)
            if a0 not in arms or a1 not in arms:
                continue
            if crit == "overall":
                p0 = sum(res[c][0] for c in CRITERIA)
                p1 = sum(res[c][1] for c in CRITERIA)
            else:
                p0, p1 = res[crit]
            g = p0 + p1                       # 2 per criterion, 10 for overall
            wins[a0] += p0; wins[a1] += p1
            games[(a0, a1)] += g; games[(a1, a0)] += g
        present = [a for a in arms if a in wins]
        out[crit] = _bradley_terry_elo(wins, games, present) if present else {}
    return out


def _print_multi(elo, arms, label):
    print(f"\n[{label}]  multi-player Elo (anchored mean 1000)")
    hdr = "  " + f"{'Metric':<16}" + "".join(
        f"{ARM_DISPLAY.get(a, a):>13}" for a in arms)
    print(hdr)
    print("  IdeaArena")
    for crit in CRITERIA + ["overall"]:
        name = "Overall" if crit == "overall" else CRIT_DISPLAY[crit]
        cells = "".join(f"{round(elo[crit].get(a, float('nan'))):>13}"
                        if a in elo[crit] else f"{'--':>13}" for a in arms)
        print(f"  {name:<16}{cells}")


def _latex_multi(elo, arms, tag, label):
    cols = "l" + "c" * len(arms)
    disp = [ARM_DISPLAY.get(a, a) for a in arms]
    judge = tag or "gpt-4o-mini"
    safe = label.lower().replace("-", "")
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Idea Arena per-criterion Bradley--Terry Elo "
        rf"({label}, official CoI judge = {judge}, both orders; multi-player "
        rf"round-robin, anchored to mean~1000). Higher is better; the best "
        rf"method per row is \textbf{{bold}}.}}",
        rf"  \label{{tab:arena-elo-{safe}}}",
        rf"  \begin{{tabular}}{{{cols}}}",
        r"    \toprule",
        "    Metric & " + " & ".join(disp) + r" \\",
        r"    \midrule",
    ]
    for crit in CRITERIA:
        row_elo = elo[crit]
        best = max((a for a in arms if a in row_elo),
                   key=lambda a: row_elo[a], default=None)
        cells = []
        for a in arms:
            if a not in row_elo:
                cells.append("--")
            else:
                v = f"{round(row_elo[a])}"
                cells.append(rf"\textbf{{{v}}}" if a == best else v)
        lines.append(f"    {CRIT_DISPLAY[crit]} & " + " & ".join(cells) + r" \\")
    lines.append(r"    \midrule")
    row_elo = elo["overall"]
    best = max((a for a in arms if a in row_elo),
               key=lambda a: row_elo[a], default=None)
    cells = []
    for a in arms:
        if a not in row_elo:
            cells.append("--")
        else:
            v = f"{round(row_elo[a])}"
            cells.append(rf"\textbf{{{v}}}" if a == best else v)
    lines.append(r"    \textbf{Overall} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _latex_combined(blocks, arms, tag):
    """One grouped table: each `blocks` entry is (super_label, elo_by_crit).
    Methods repeat under every super-column; row-best is bolded within each
    block.  `arms` are the columns inside each block."""
    disp = [ARM_DISPLAY.get(a, a) for a in arms]
    nm = len(arms)
    judge = tag or "gpt-4o-mini"
    labels = " and ".join(lbl for lbl, _ in blocks)
    colspec = "l " + " ".join(["c" * nm for _ in blocks])

    head_super = " & " + " & ".join(
        rf"\multicolumn{{{nm}}}{{c}}{{{lbl}}}" for lbl, _ in blocks) + r" \\"
    cmids, start = [], 2
    for _ in blocks:
        end = start + nm - 1
        cmids.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    head_meth = "    Metric & " + " & ".join(
        " & ".join(disp) for _ in blocks) + r" \\"

    def row(crit, name):
        cells = []
        for _, elo in blocks:
            row_elo = elo[crit]
            best = max((a for a in arms if a in row_elo),
                       key=lambda a: row_elo[a], default=None)
            for a in arms:
                if a not in row_elo:
                    cells.append("--")
                else:
                    v = f"{round(row_elo[a])}"
                    cells.append(rf"\textbf{{{v}}}" if a == best else v)
        nm_disp = rf"\textbf{{{name}}}" if name == "Overall" else name
        return f"    {nm_disp} & " + " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Idea Arena per-criterion Bradley--Terry Elo by domain "
        rf"({labels}; official CoI judge = {judge}, both orders; multi-player "
        rf"round-robin, anchored to mean~1000 within each block). Higher is "
        rf"better; the best method per row within each block is \textbf{{bold}}.}}",
        r"  \label{tab:arena-elo-domain}",
        rf"  \begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        "  " + head_super,
        "    " + " ".join(cmids),
        head_meth,
        r"    \midrule",
    ]
    for crit in CRITERIA:
        lines.append(row(crit, CRIT_DISPLAY[crit]))
    lines.append(r"    \midrule")
    lines.append(row("overall", "Overall"))
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt-4o",
                    help="judge namespace (reads idea_arena__{tag}.jsonl); "
                         "empty for the un-tagged gpt-4o-mini file")
    ap.add_argument("--naive", default="bm25",
                    help="arm for the 'Naïve Retrieval' column (pair mode)")
    ap.add_argument("--mi", default="moose_chem",
                    help="arm for the 'MI Retrieval' column (pair mode)")
    ap.add_argument("--no-strata", action="store_true",
                    help="print Overall only")
    ap.add_argument("--latex", action="store_true",
                    help="emit a multi-player Elo LaTeX table over ALL arms "
                         "present (instead of the 2-arm pair table)")
    ap.add_argument("--combined", action="store_true",
                    help="with --latex: ONE grouped table with Same-domain and "
                         "Cross-domain as super-columns (plus Overall row)")
    ap.add_argument("--with-overall-block", action="store_true",
                    help="with --combined: prepend an Overall super-column "
                         "block as well")
    ap.add_argument("--arms", default="",
                    help="comma-separated arms/columns for --latex mode; "
                         "default = every arm present, in floor->ceiling order")
    args = ap.parse_args()

    suffix = f"__{args.tag}" if args.tag else ""
    path = OUT_DIR / f"idea_arena{suffix}.jsonl"
    rows = [r for r in read_jsonl(path)
            if not r.get("failed") and r.get("orderA") and r.get("orderB")
            and r.get("arm0") and r.get("arm1")]
    if not rows:
        raise SystemExit(f"No usable rows in {path}")

    strata = ([("Overall", None)] if args.no_strata
              else [("Overall", None), ("Same-domain", "same"),
                    ("Cross-domain", "cross")])

    # ---- multi-method LaTeX mode ----------------------------------------
    if args.latex:
        present = [a for a, _ in ARM_ORDER
                   if any(r["arm0"] == a or r["arm1"] == a for r in rows)]
        if args.arms:
            want = [a.strip() for a in args.arms.split(",") if a.strip()]
            arms = [a for a in want if a in present]
        else:
            arms = present
        print("=" * 72)
        print(f"IDEA-ARENA multi-player Elo  (file: {path.name})")
        print(f"  arms: {', '.join(ARM_DISPLAY.get(a, a) for a in arms)}")
        print(f"  Elo = multi-player Bradley-Terry, anchored to mean 1000")
        print("=" * 72)

        elos = {}
        for label, key in strata:
            rs = [r for r in rows if key is None or r["stratum"] == key]
            elos[label] = _multi_elo(rs, arms)
            _print_multi(elos[label], arms, f"{label} (n={len(rs)})")

        print("\n" + "=" * 72)
        print("LATEX (paste into the paper)")
        print("=" * 72 + "\n")
        if args.combined:
            blocks = []
            if args.with_overall_block and "Overall" in elos:
                blocks.append(("Overall", elos["Overall"]))
            for lbl in ("Same-domain", "Cross-domain"):
                if lbl in elos:
                    blocks.append((lbl, elos[lbl]))
            print(_latex_combined(blocks, arms, args.tag))
        else:
            print("\n\n".join(
                _latex_multi(elos[label], arms, args.tag, label)
                for label, _ in strata))
        return

    # ---- default: 2-arm pair table --------------------------------------
    print("=" * 72)
    print(f"IDEA-ARENA per-criterion Elo  (file: {path.name})")
    print(f"  pair: Naive={args.naive}  vs  MI={args.mi}")
    print(f"  Elo = 2-player Bradley-Terry, anchored symmetric around 1000")
    print("=" * 72)

    results = {}
    for label, key in strata:
        rs = [r for r in rows if key is None or r["stratum"] == key]
        res = _table(rs, args.naive, args.mi, label)
        if res:
            results[label] = res

    print("\n" + "=" * 72)
    print("MARKDOWN (paste into the deck)")
    print("=" * 72)
    for label, res in results.items():
        _markdown(res, label)


if __name__ == "__main__":
    main()
