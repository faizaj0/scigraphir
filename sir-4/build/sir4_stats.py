#!/usr/bin/env python3
"""SIR-4 dataset statistics: the LaTeX table and the two figures, in TOMATO-Star's style.

Produces the direct counterparts of the TOMATO-Star exploration in
`retriever/tomato_star/explore.py`, so the two datasets can sit side by side in the thesis:

    fig_sir4_lengths.png   query and document length densities, train vs test
                           (counterpart of Figure 10.1)
    fig_sir4_crossfield_matrix.png
                           target-field by inspiration-field composition of
                           cross-field golds, train and test
    tab_sir4_stats.tex     the statistics table
    sir4_stats.json        every number in machine-readable form

WHERE THE TWO DATASETS DIFFER, AND WHY THE TABLE NEEDS AN EXTRA COLUMN.

TOMATO-Star reports I+ = 1.0: one gold inspiration per query, by construction.
That is the assumption SIR-4 exists to test, so reproducing its table verbatim
would hide the result. Two columns are added:

    |D1|   golds in the PRIMARY decomposition. Already > 1, because SIR-4 keeps
           the full inspiration set rather than splitting it into one query per
           inspiration.
    |M|    VALID DECOMPOSITIONS per query, the family the uniqueness sweep found.
           |M| = 1 means the primary set is the only valid one. |M| > 1 means a
           single-gold benchmark would score a correct alternative as a miss.

THE FIELD FIGURE IS NOT A COPY.

TOMATO-Star colours gold fields biomedical against distant, which works because
every source paper is biomedical. SIR-4 has four target fields, so a marginal
source-field distribution hides direction. The replacement figure is a
row-normalised target-by-source matrix over cross-field golds. It reports where
each target field's cross-field inspirations come from while keeping train and
evaluation directly comparable.

    python3 build/sir4_stats.py
    python3 build/sir4_stats.py --domains cs biology --out figures/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matched to the TOMATO-Star figures (10.1, 10.2) so the two datasets read as one
# body of work: sans-serif, a light box on all four sides, a faint grid behind the
# data, and the legend outside the axes for the categorical figure.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#BFBFBF",
    "axes.grid": True,
    "grid.color": "#E8E8E8", "grid.linewidth": 0.7,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    # TrueType outlines, so PDF text stays vector, selectable and searchable.
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# A document shorter than this carries a title and no abstract: the paper
# resolved, but no index held its abstract. They are real corpus members and stay
# in every COUNT, but they are excluded from the length figure, where a spike of
# 15% of the corpus at zero words dominates a panel that is about the shape of
# the abstract distribution.
MIN_DOC_WORDS = 25

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"
# The thesis figure directory, one level above the benchmark build: these sit
# beside the TOMATO-Star figures they are the counterpart of, not in a folder of
# their own that LaTeX would have to be pointed at separately.
FIGS = ROOT.parent / "figures"

# Tags differ by domain because cs was built before the effort setting changed.
SPLITS = {
    "cs":      {"test": "cs_test_final",       "train": "cs_train_final"},
    "biology": {"test": "biology_test_low",    "train": "biology_train_low"},
    "physics": {"test": "physics_test_low",    "train": "physics_train_low"},
    "matsci":  {"test": "matsci_test_low",     "train": "matsci_train_low"},
}

# TOMATO-Star's palette, so the figures read as a pair with Figures 10.1-10.2.
BLUE, RED = "#4C72B0", "#C44E52"
ORANGE = "#C1701C"

# OpenAlex field names run long enough to truncate mid-word on a bar chart.
# Abbreviated the way TOMATO-Star's own figure abbreviates them, so the two
# read consistently.
SHORT = {
    "Biochemistry, Genetics and Molecular Biology": "Biochem. & Genetics",
    "Agricultural and Biological Sciences": "Agric. & Bio. Sci.",
    "Pharmacology, Toxicology and Pharmaceutics": "Pharmacology & Tox.",
    "Immunology and Microbiology": "Immunol. & Microbio.",
    "Earth and Planetary Sciences": "Earth & Planetary Sci.",
    "Economics, Econometrics and Finance": "Econ. & Finance",
    "Physics and Astronomy": "Physics & Astronomy",
    "Environmental Science": "Environmental Sci.",
    "Computer Science": "Computer Science",
    "Materials Science": "Materials Science",
    "Health Professions": "Health Professions",
    "Chemical Engineering": "Chemical Eng.",
}

# Short enough to sit HORIZONTALLY under a heatmap column. Rotated labels are
# the main reason a matrix like this reads as cluttered.
TIGHT = {
    "Mathematics": "Maths",
    "Physics and Astronomy": "Physics",
    "Engineering": "Eng.",
    "Computer Science": "CS",
    "Medicine": "Medicine",
    "Decision Sciences": "Decision",
    "Chemistry": "Chem.",
    "Social Sciences": "Social",
    "Biochemistry, Genetics and Molecular Biology": "Biochem.",
    "Economics, Econometrics and Finance": "Econ.",
    "Environmental Science": "Environ.",
    "Neuroscience": "Neuro.",
    "Agricultural and Biological Sciences": "Agri.",
    "Materials Science": "Mat. Sci.",
    "Earth and Planetary Sciences": "Earth",
    "Immunology and Microbiology": "Immuno.",
}

DOMAIN_NAME = {
    "cs": "Computer science",
    "biology": "Biology",
    "physics": "Physics",
    "matsci": "Materials science",
}


def load(name: str) -> tuple[list, dict] | None:
    d = BENCH / name
    if not (d / "eval.json").exists():
        return None
    ev = json.loads((d / "eval.json").read_text())
    docs = json.loads((d / "raw" / "documents.json").read_text())
    return ev, docs


def collect(names: list[str]) -> dict:
    """Pool several splits into one set of arrays."""
    qlen, sets, d1, strat = [], [], [], Counter()
    fields, docs = Counter(), {}
    for n in names:
        got = load(n)
        if not got:
            continue
        ev, dd = got
        docs.update(dd)
        for r in ev:
            q = r.get("quartet") or {}
            qlen.append(len(str(r.get("question") or "").split()))
            sets.append(len(q.get("sets") or []))
            d1.append(len(q.get("primary_set") or []))
            strat[r.get("stratum")] += 1
            # `field_pair` is "<target field> -> <gold field>"; the gold side is
            # what the figure plots, and the two sides decide the colour.
            for rec in (q.get("per_document") or {}).values():
                fp = str(rec.get("field_pair") or "")
                if "->" not in fp:
                    continue
                tgt, gold = (x.strip() for x in fp.split("->", 1))
                if gold:
                    fields[(gold, tgt == gold)] += 1
    dlen = np.array([len(str(t).split()) for t in docs.values()])
    return {"qlen": np.array(qlen), "sets": np.array(sets), "d1": np.array(d1),
            "strat": strat, "fields": fields,
            "dlen": dlen,
            # Same array with title-only records removed, for the length figure.
            "dlen_full": dlen[dlen >= MIN_DOC_WORDS],
            "n_title_only": int((dlen < MIN_DOC_WORDS).sum()),
            "n_docs": len(docs)}


def _tidy(ax, axis="y"):
    """TOMATO-Star's frame: light box on all four sides, faint grid behind."""
    ax.set_axisbelow(True)
    ax.grid(False)
    if axis:
        ax.grid(axis=axis, color="#E8E8E8", lw=0.7)
    ax.tick_params(labelsize=7.5, length=3, pad=2)


def panel(ax, tr, te, title, xlabel, clip=None, subtitle=None):
    """Train and evaluation overlaid, dashed means. TOMATO-Star Figure 10.1's shape."""
    both = np.concatenate([x for x in (tr, te) if len(x)]) if (len(tr) or len(te)) else np.array([0])
    clip = clip or float(np.percentile(both, 99.5))
    bins = np.linspace(0, clip, 44)
    # Truncate the 0.5% tail rather than clipping it: np.clip piles every long
    # document into the final bin and invents a mode at the axis edge.
    for arr, colour, lab in ((tr, BLUE, "Train"), (te, RED, "Test")):
        if not len(arr):
            continue
        ax.hist(arr[arr <= clip], bins=bins, density=True, color=colour,
                alpha=0.62, lw=0, label=f"{lab} (n={len(arr):,})")
        ax.axvline(float(np.mean(arr)), color=colour, ls="--", lw=1.0, alpha=0.9)

    ax.set_title(title, fontsize=8.5, pad=13 if subtitle else 6)
    if subtitle:
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, fontsize=6.5,
                color="#777777", ha="center", va="bottom")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_xlim(0, clip)
    ax.margins(y=0.03)
    ax.legend(fontsize=6.8, loc="upper right", framealpha=0.85,
              edgecolor="#DDDDDD", fancybox=False, borderpad=0.5,
              handlelength=1.2, handletextpad=0.5)
    _tidy(ax, axis=None)


def cross_field_counts(doms: list[str], split: str) -> Counter:
    """Count cross-field gold occurrences by benchmark target and source field."""
    pairs = Counter()
    for domain in doms:
        got = load(SPLITS[domain][split])
        if not got:
            continue
        ev, _ = got
        for row in ev:
            per_document = ((row.get("quartet") or {}).get("per_document") or {})
            for rec in per_document.values():
                if rec.get("stratum") != "cross":
                    continue
                field_pair = str(rec.get("field_pair") or "")
                if "->" not in field_pair:
                    continue
                _, source = (part.strip() for part in field_pair.split("->", 1))
                if source:
                    pairs[(DOMAIN_NAME[domain], source)] += 1
    return pairs


def cross_field_matrix(pairs: Counter, rows: list[str], columns: list[str]) -> tuple[np.ndarray, list[int]]:
    """Return row-normalised percentages, aggregating unlisted sources as Other."""
    matrix = np.zeros((len(rows), len(columns) + 1), dtype=float)
    totals = []
    for r, target in enumerate(rows):
        total = sum(n for (t, _), n in pairs.items() if t == target)
        totals.append(total)
        if not total:
            continue
        used = 0
        for c, source in enumerate(columns):
            n = pairs.get((target, source), 0)
            matrix[r, c] = 100 * n / total
            used += n
        matrix[r, -1] = 100 * (total - used) / total
    return matrix, totals


def cross_field_panel(ax, matrix: np.ndarray, rows: list[str], totals: list[int],
                      columns: list[str], title: str, vmax: float,
                      show_field_names: bool = True):
    """Plot one target-by-source matrix; every target row sums to 100%."""
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9.5, loc="left", pad=7)
    ax.set_xticks(range(len(columns) + 1))
    ax.set_xticklabels([SHORT.get(c, c) for c in columns] + ["Other"],
                       rotation=38, ha="right", rotation_mode="anchor", fontsize=6.8)
    ax.set_yticks(range(len(rows)))
    if show_field_names:
        ylabels = [f"{name}  ($n$={n:,})" for name, n in zip(rows, totals)]
        ax.set_ylabel("Target-paper field", fontsize=8.2, labelpad=7)
    else:
        ylabels = [f"$n$={n:,}" for n in totals]
    ax.set_yticklabels(ylabels, fontsize=7.2)
    ax.tick_params(length=0, pad=2)
    for side in ax.spines.values():
        side.set_visible(False)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            if value < 0.5:
                continue
            colour = "white" if value >= vmax * 0.52 else "#222222"
            ax.text(c, r, f"{value:.0f}", ha="center", va="center",
                    fontsize=6.8, color=colour)
    ax.set_xlabel("Inspiration-paper field", fontsize=8.2, labelpad=3)
    return image


def latex_domains(rows: list[dict], out: Path) -> str:
    """Per-domain breakdown. The pooled table hides that the domains differ a lot.

    Cross-domain share in particular ranges 3x across the four, and it is the
    column a reader of a cross-domain retrieval chapter goes looking for.
    """
    body = []
    last = None
    display_domain = {
        "cs": "CS",
        "biology": "Biology",
        "physics": "Physics",
        "matsci": "Materials",
    }
    for r in rows:
        sep = r["domain"] != last
        last = r["domain"]
        name = display_domain.get(r["domain"], r["domain"]) if sep else ""
        if sep and body:
            body.append(r"\addlinespace[2pt]")
        body.append(
            f"{name:<9} & {r['split']:<5} & {r['Q']:>6,} & {r['I']:>6,} & "
            f"{r['d1']:.1f} & {r['M']:.2f} & {r['cross']:.1f}\\% & "
            f"{r['qlen']:.0f} & {r['dlen']:.0f} \\\\".replace(",", "{,}"))
    tex = r"""\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.12}
\caption{SIR-4 statistics by domain. $|D_1|$ is the primary-set size,
$|\mathcal{M}|$ the number of valid decompositions recovered by the bounded
sweep (a lower bound), and \emph{Cross} the share of labelled gold-document
occurrences whose OpenAlex field sets are disjoint from the target's.}
\label{tab:sir4-domains}
\begin{tabular}{ll rr rr r rr}
\toprule
 & & \multicolumn{2}{c}{Total number} & \multicolumn{2}{c}{Gold structure} & &
\multicolumn{2}{c}{Mean length} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}
Domain & Split & $|\mathcal{Q}|$ & $|\mathcal{I}|$ & $|D_1|$ & $|\mathcal{M}|$ &
Cross & $|q|$ & $|d|$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out.write_text(tex)
    return tex


def latex(rows: list[dict], out: Path) -> str:
    body = []
    for r in rows:
        body.append(
            f"{r['split']:<7} & {r['Q']:>7,} & {r['I']:>7,} & {r['d1']:.1f} & "
            f"{r['M']:.2f} & {r['qlen']:.1f} & {r['dlen']:.1f} \\\\".replace(",", "{,}"))
    tex = r"""\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.15}
\caption{SIR-4 inspiration-retrieval statistics. Each retrieval query $q$
contains the research question and background; the verified hypothesis
$h^\star$ is hidden and used only to construct the gold. Each
$d\in\mathcal{I}$ contains an inspiration-paper title and abstract. $|D_1|$ is
the primary-set size and $|\mathcal{M}|$ the number of valid decompositions
recovered by the bounded sweep, and is therefore a lower bound. Lengths are in
whitespace-separated words.}
\label{tab:sir4-stats}
\begin{tabular}{l rr rr rr}
\toprule
 & \multicolumn{2}{c}{Total number} & \multicolumn{2}{c}{Gold structure} & \multicolumn{2}{c}{Mean length} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Split & $|\mathcal{Q}|$ & $|\mathcal{I}|$ & $|D_1|$ & $|\mathcal{M}|$ & $|q|$ & $|d|$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out.write_text(tex)
    return tex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="*", default=list(SPLITS))
    ap.add_argument("--out", type=Path, default=FIGS)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    doms = [d for d in a.domains if d in SPLITS]
    missing = [d for d in a.domains if d not in SPLITS]
    if missing:
        print(f"  unknown domain(s) ignored: {', '.join(missing)}", file=sys.stderr)

    tr_names = [SPLITS[d]["train"] for d in doms]
    te_names = [SPLITS[d]["test"] for d in doms]
    tr, te = collect(tr_names), collect(te_names)
    if not len(tr["qlen"]) and not len(te["qlen"]):
        sys.exit("no benchmarks found; has export run?")

    # ---- figure 1: lengths, the counterpart of Fig 10.1
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 2.95))
    panel(ax[0], tr["qlen"], te["qlen"], "(a) IR query length",
          "Query length (words)")
    # Title-only records are excluded here and only here. Left in, a spike of 15%
    # of the corpus at zero words swamped the panel and hid the abstract-length
    # distribution the panel exists to show. Stated on the axis so the exclusion
    # is visible rather than silently applied.
    short = tr["n_title_only"] + te["n_title_only"]
    tot = len(tr["dlen"]) + len(te["dlen"])
    # In the axis label, not floating in the corner where it collided with the
    # legend. A note that overlaps what it annotates is worse than no note.
    panel(ax[1], tr["dlen_full"], te["dlen_full"], "(b) Inspiration document length",
          "Document length (words)",
          subtitle=f"excludes {short:,} title-only records "
                   f"({short / max(tot,1) * 100:.0f}% of the corpus)")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(a.out / "fig_sir4_lengths.pdf")          # vector, for the thesis
    fig.savefig(a.out / "fig_sir4_lengths.png", dpi=200)  # raster, for previewing
    plt.close(fig)

    # ---- figure 2: directional composition of cross-field inspirations
    #
    # ONE pooled panel, not two. Train and evaluation agree on every large cell
    # (cs<-Maths 31/17, matsci<-Physics 54/40, biology<-Medicine 25/25), so a
    # side-by-side pair spent half the figure's width restating agreement while
    # squeezing eleven columns into each half. Pooling doubles the cell size and
    # the agreement belongs in a sentence.
    target_rows = [DOMAIN_NAME[d] for d in doms]
    pairs = Counter()
    for split in ("train", "test"):
        for k, v in cross_field_counts(doms, split).items():
            pairs[k] += v
    source_totals = Counter()
    for (_, source), count in pairs.items():
        source_totals[source] += count
    # The ten largest source fields. Mathematics was previously excluded here as
    # "not one of the four domains analysed", which is the wrong test: columns
    # are where cross-field inspirations COME FROM, not which corpora were built,
    # and Mathematics is the single largest source overall.
    source_columns = [source for source, _ in source_totals.most_common(10)]
    matrix, totals = cross_field_matrix(pairs, target_rows, source_columns)

    # Small multiples in TOMATO-Star Figure 10.2's idiom: horizontal share bars,
    # value labels at the bar ends, a two-colour categorical split, and a single
    # legend beneath the whole figure.
    #
    # A 4x11 heatmap was tried first and abandoned: about thirty of its
    # forty-four cells are under 5%, so it rendered as mostly white and spent its
    # area on near-zeroes. One ranked panel per corpus answers the only question
    # the figure asks -- where does this domain borrow from.
    #
    # ONE colour at the same alpha as the length figure's Test series, so a bar
    # here and a histogram there are visibly the same ink. An earlier version
    # split the bars by whether the source field was itself one of SIR-4's four
    # corpora; that encoded a second variable the panels were not asking about
    # and cost a legend to explain it. Rank alone carries the message.
    TOP_N = 6

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.2))
    for ax, target in zip(axes.ravel(), target_rows):
        row = Counter({src: n for (t, src), n in pairs.items() if t == target})
        total = sum(row.values()) or 1
        top = row.most_common(TOP_N)
        other = total - sum(n for _, n in top)
        items = [(SHORT.get(k, k), n / total * 100) for k, n in top]
        if other > 0:
            items.append(("All other fields", other / total * 100))
        items = items[::-1]

        y = list(range(len(items)))
        ax.barh(y, [v for _, v in items], color=RED, alpha=0.62,
                height=0.62, lw=0)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for lab, _ in items], fontsize=7)
        span = max(v for _, v in items)
        for i, (_, v) in enumerate(items):
            ax.text(v + span * 0.025, i, f"{v:.1f}%", va="center", fontsize=6.5,
                    color="#444444")
        ax.set_xlim(0, span * 1.22)
        ax.set_xlabel("Share (%)", fontsize=8)
        ax.set_title(f"{target}  (n={total:,})", fontsize=8.5)
        _tidy(ax, axis=None)

    fig.tight_layout(h_pad=2.4, w_pad=2.6)
    fig.savefig(a.out / "fig_sir4_crossfield_matrix.pdf")
    fig.savefig(a.out / "fig_sir4_crossfield_matrix.png", dpi=200)
    plt.close(fig)

    # ---- table
    rows = []
    for lab, names in (("Train", tr_names), ("Test", te_names), ("All", tr_names + te_names)):
        c = collect(names) if lab == "All" else (tr if lab == "Train" else te)
        if not len(c["qlen"]):
            continue
        rows.append({"split": lab, "Q": len(c["qlen"]), "I": c["n_docs"],
                     "d1": float(c["d1"].mean()), "M": float(c["sets"].mean()),
                     "qlen": float(c["qlen"].mean()), "dlen": float(c["dlen"].mean())})
    tex = latex(rows, a.out / "tab_sir4_stats.tex")

    # ---- per-domain breakdown
    drows = []
    for d in doms:
        for lab, key in (("Train", "train"), ("Test", "test")):
            c = collect([SPLITS[d][key]])
            if not len(c["qlen"]):
                continue
            mf = BENCH / SPLITS[d][key] / "manifest.json"
            st = json.loads(mf.read_text())["stratum"]
            gtot = st["same"] + st["cross"] or 1
            drows.append({"domain": d, "split": lab, "Q": len(c["qlen"]),
                          "I": c["n_docs"], "d1": float(c["d1"].mean()),
                          "M": float(c["sets"].mean()),
                          "cross": st["cross"] / gtot * 100,
                          "qlen": float(c["qlen"].mean()),
                          "dlen": float(c["dlen"].mean())})
    latex_domains(drows, a.out / "tab_sir4_domains.tex")
    print(f"\n{'domain':<9}{'split':<7}{'|Q|':>8}{'|I|':>8}{'|D1|':>7}{'|M|':>7}"
          f"{'cross':>8}{'|q|':>7}{'|d|':>7}")
    for r in drows:
        print(f"{r['domain']:<9}{r['split']:<7}{r['Q']:>8,}{r['I']:>8,}{r['d1']:>7.1f}"
              f"{r['M']:>7.2f}{r['cross']:>7.1f}%{r['qlen']:>7.0f}{r['dlen']:>7.0f}")

    stats = {"domains": doms, "by_domain": drows,
             "rows": rows,
             "stratum_train": dict(tr["strat"]), "stratum_test": dict(te["strat"])}
    (a.out / "sir4_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"{'split':<7}{'|Q|':>9}{'|I|':>9}{'|D1|':>7}{'|M|':>7}{'|q|':>8}{'|d|':>8}")
    for r in rows:
        print(f"{r['split']:<7}{r['Q']:>9,}{r['I']:>9,}{r['d1']:>7.1f}"
              f"{r['M']:>7.2f}{r['qlen']:>8.1f}{r['dlen']:>8.1f}")
    print(f"\nwrote -> {a.out}/")
    for f in ("fig_sir4_lengths.pdf", "fig_sir4_lengths.png",
              "fig_sir4_crossfield_matrix.pdf", "fig_sir4_crossfield_matrix.png",
              "tab_sir4_stats.tex", "sir4_stats.json"):
        print(f"    {f}")


if __name__ == "__main__":
    main()
