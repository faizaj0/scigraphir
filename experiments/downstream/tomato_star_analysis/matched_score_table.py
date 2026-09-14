"""Matched-Score table, per-dimension, stratified by domain -> LaTeX.

API-FREE.  Reads matched_{arm}{__tag}.jsonl for every arm present and reports the
reference-based recall scores (Motivation / Mechanism / Methodology, each 0-4, and
Total 0-12) as means over the PAIRED query set (queries scored under every arm),
split into Same-domain / Cross-domain.  Mirrors aggregate.matched_report but emits
a single grouped LaTeX table (Same | Cross super-columns), parallel to
arena_criteria_table --combined.

Run (canonical = gpt-4o judge):
  cd TOMATO-Star
  python -m analysis.downstream.matched_score_table --tag gpt-4o --latex --combined

Output: stdout console preview + LaTeX.
"""
from __future__ import annotations

import argparse

from analysis.downstream.aggregate import _matched, _mean, ARM_ORDER, ARM_DISPLAY

DIMS = [("motivation", "Motivation"), ("mechanism", "Mechanism"),
        ("methodology", "Methodology")]
ROWS = DIMS + [("total", "Total (0--12)")]


def _collect(tag):
    data = {arm: _matched(arm, tag) for arm, _ in ARM_ORDER}
    data = {a: d for a, d in data.items() if d}
    present = [a for a, _ in ARM_ORDER if a in data]
    if not present:
        raise SystemExit(f"No matched_*{('__' + tag) if tag else ''}.jsonl found.")
    shared = None
    for a in present:
        ks = set(data[a])
        shared = ks if shared is None else (shared & ks)
    shared = sorted(shared or [])
    return data, present, shared


def _stats(data, present, qs):
    """{arm: {row_key: mean}} over query ids qs."""
    out = {}
    for a in present:
        s = {d: _mean([data[a][q]["scores"][d] for q in qs]) for d, _ in DIMS}
        s["total"] = _mean([data[a][q]["total"] for q in qs])
        out[a] = s
    return out


def _print_block(stats, present, label, n):
    print(f"\n[{label}]  n={n} paired queries")
    print("  " + f"{'Metric':<16}" + "".join(
        f"{ARM_DISPLAY.get(a, a):>13}" for a in present))
    for key, name in ROWS:
        print(f"  {name.replace(' (0--12)', ''):<16}" + "".join(
            f"{stats[a][key]:>13.2f}" for a in present))


def _latex_combined(blocks, present, tag):
    """blocks: list of (super_label, stats, n).  present = arm columns."""
    disp = [ARM_DISPLAY.get(a, a) for a in present]
    nm = len(present)
    judge = tag or "gpt-4o-mini"
    colspec = "l " + " ".join(["c" * nm for _ in blocks])
    head_super = " & " + " & ".join(
        rf"\multicolumn{{{nm}}}{{c}}{{{lbl}~($n={n}$)}}"
        for lbl, _, n in blocks) + r" \\"
    cmids, start = [], 2
    for _ in blocks:
        end = start + nm - 1
        cmids.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    head_meth = "    Metric & " + " & ".join(
        " & ".join(disp) for _ in blocks) + r" \\"

    def row(key, name):
        cells = []
        for _, stats, _n in blocks:
            best = max(present, key=lambda a: stats[a][key])
            for a in present:
                v = f"{stats[a][key]:.2f}"
                cells.append(rf"\textbf{{{v}}}" if a == best else v)
        nm_disp = rf"\textbf{{{name}}}" if key == "total" else name
        return f"    {nm_disp} & " + " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Matched-Score (reference-based recall vs the gold delta "
        rf"hypothesis; judge = {judge}) by domain. Motivation / Mechanism / "
        rf"Methodology are each 0--4 (Total 0--12), averaged over the paired "
        rf"query set. Higher is better; the best arm per row within each block "
        rf"is \textbf{{bold}}.}}",
        r"  \label{tab:matched-score-domain}",
        rf"  \begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        "  " + head_super,
        "    " + " ".join(cmids),
        head_meth,
        r"    \midrule",
    ]
    for key, name in DIMS:
        lines.append(row(key, name))
    lines.append(r"    \midrule")
    lines.append(row("total", "Total (0--12)"))
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _latex_single(stats, present, tag, label, n):
    return _latex_combined([(label, stats, n)], present, tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt-4o",
                    help="judge namespace (reads matched_{arm}__{tag}.jsonl); "
                         "empty for the un-tagged gpt-4o-mini files")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--combined", action="store_true",
                    help="with --latex: ONE grouped table, Same|Cross "
                         "super-columns")
    ap.add_argument("--with-overall-block", action="store_true",
                    help="with --combined: prepend an Overall super-column")
    args = ap.parse_args()

    data, present, shared = _collect(args.tag)
    print("=" * 72)
    print(f"MATCHED-SCORE  (tag={args.tag or 'gpt-4o-mini'})")
    print(f"  arms: {', '.join(ARM_DISPLAY.get(a, a) for a in present)}")
    print(f"  paired queries (scored under all arms): {len(shared)}")
    print("=" * 72)

    strata = [("Overall", None), ("Same-domain", "same"),
              ("Cross-domain", "cross")]
    first = present[0]
    block_data = {}
    for label, key in strata:
        qs = [q for q in shared
              if key is None or data[first][q]["stratum"] == key]
        if not qs:
            continue
        st = _stats(data, present, qs)
        block_data[label] = (st, len(qs))
        _print_block(st, present, label, len(qs))

    if not args.latex:
        return

    print("\n" + "=" * 72)
    print("LATEX (paste into the paper)")
    print("=" * 72 + "\n")
    if args.combined:
        blocks = []
        if args.with_overall_block and "Overall" in block_data:
            st, n = block_data["Overall"]
            blocks.append(("Overall", st, n))
        for lbl in ("Same-domain", "Cross-domain"):
            if lbl in block_data:
                st, n = block_data[lbl]
                blocks.append((lbl, st, n))
        print(_latex_combined(blocks, present, args.tag))
    else:
        out = []
        for label, _ in strata:
            if label in block_data:
                st, n = block_data[label]
                out.append(_latex_single(st, present, args.tag, label, n))
        print("\n\n".join(out))


if __name__ == "__main__":
    main()
