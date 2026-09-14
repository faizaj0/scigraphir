#!/usr/bin/env python3
"""
ablation_ladder_fig.py -- the cumulative ablation as a ladder: x = stage, y = share of golds within the top k, one line per
dataset. Rows: cross-field golds, same-field golds. Columns: k = 1, 5, 10. Stages (each a full ranking of the corpus):
  1 Qwen3 cosine  ->  2 multi-view scorer  ->  3 scorer + OpenIE entity graph (that model's own scorer, hollow marker)
  ->  4 scorer + SciAffordGraph, CCMP gate off (same weights as 5)  ->  5 scorer + SciAffordGraph + CCMP = SciGraphIR

    python3 eval/ablation_ladder_fig.py --out ../figures/fig_ablation_ladder
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); S4 = os.path.dirname(HERE)
INK, MUTED, RULE = "#262a30", "#7c828c", "#ced2da"
DATASETS = [("sir4_cs", "SIR-4 CS", "#2a69a0", "o"), ("sir4_biology", "SIR-4 Biology", "#208070", "s"), ("sir4_physics", "SIR-4 Physics", "#705296", "^"),
            ("sir4_matsci", "SIR-4 MatSci", "#a47822", "D"), ("tomato", "TOMATO", "#c44e52", "v"), ("mir", "MIR", "#6e747e", "P")]
_QC = f"{S4}/results/qualitative/quartet_cache"
QUARTET = {"sir4_cs": f"{_QC}/cs_test_final.eval.json", **{f"sir4_{f}": f"{_QC}/{f}_test_low.eval.json" for f in ("biology", "physics", "matsci")}}
STAGES = ["1\ncosine", "2\nscorer", "3\n+ entity\ngraph", "4\n+ frame\ngraph", "5\n+ CCMP"]
KS = [1, 5, 10]


def openie_path(d):
    for name in ("hops_openie.json", "hops_t4_openie.json"):
        if os.path.exists(f"{d}/{name}"): return f"{d}/{name}"
    return None


def frame_arm(d, prefix="hops_", want="auto"):
    """which SciAffordGraph arm the hops files hold: the merged graph (hyb_ccmp, the current SciAffordGraph) when present,
    else the older frame-only graph (frame_ccmp). Returns (arm name, label suffix)."""
    if want == "auto": want = "hyb_ccmp" if os.path.exists(f"{d}/{prefix}hyb_ccmp.json") else "frame_ccmp"
    return want, ("merged graph" if want.startswith("hyb") else "frame-only graph, outdated")


def load(path):
    return {(r["id"], t["doc"]): {"stratum": r.get("stratum") or "unlabelled", "rank": t.get("rank") or {}} for r in json.load(open(path)) for t in r["targets"]}


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(RULE); ax.spines["bottom"].set_color(RULE); ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.7)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--arm", default="auto", help="auto | hyb_ccmp | frame_ccmp"); a = ap.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.6), gridspec_kw={"wspace": 0.28, "hspace": 0.62})
    for ax in axes.ravel(): style(ax)
    rows = ["| dataset | stratum | n | k | cosine | scorer | + entity graph (own scorer) | + SciAffordGraph gate off | + CCMP (SciGraphIR) |", "|---|---|---|---|---|---|---|---|---|"]
    handles = {}
    for ds, label, col, mk in DATASETS:
        d = f"{S4}/results/qualitative/drive_scan_{ds}"
        ARM, ARMLAB = frame_arm(d, "hops_", a.arm); F = load(f"{d}/hops_{ARM}.json"); O = load(f"{d}/hops_{ARM}_off.json") if os.path.exists(f"{d}/hops_{ARM}_off.json") else {}
        _ep = openie_path(d); E = load(_ep) if _ep else {}
        if ds in QUARTET and os.path.exists(QUARTET[ds]):
            qz = {}
            for x in json.load(open(QUARTET[ds])):
                for doc, m in (x.get("quartet", {}).get("per_document") or {}).items(): qz[(x["id"], doc)] = m.get("stratum")
            for k, v in F.items(): v["stratum"] = qz.get(k) or "unlabelled"
        for ri, s in enumerate(("cross", "same")):
            keys = [k for k, v in F.items() if v["stratum"] == s and all(v["rank"].get(c) for c in ("fused", "scorer", "dense"))]
            if len(keys) < 30: continue
            ke = [k for k in keys if k in E and E[k]["rank"].get("fused")]; ko = [k for k in keys if k in O and O[k]["rank"].get("fused")]
            stage_ranks = [[F[k]["rank"]["dense"] for k in keys], [F[k]["rank"]["scorer"] for k in keys], [E[k]["rank"]["fused"] for k in ke],
                           [O[k]["rank"]["fused"] for k in ko], [F[k]["rank"]["fused"] for k in keys]]
            for ci, K in enumerate(KS):
                ax = axes[ri][ci]
                ys = [100.0 * sum(r <= K for r in rk) / len(rk) if rk else float("nan") for rk in stage_ranks]
                xs = list(range(5))
                ln, = ax.plot(xs, ys, color=col, marker=mk, ms=5, lw=1.6, label=label)
                if ys[2] == ys[2]: ax.plot([2], [ys[2]], marker=mk, ms=6, mfc="white", mec=col, mew=1.3, ls="")     # hollow: a different model's scorer
                handles[ds] = ln
                rows.append(f"| {ds} | {s} | {len(keys)} | {K} | " + " | ".join("" if y != y else f"{y:.1f}" for y in ys) + " |")
    for ri, s in enumerate(("cross", "same")):
        for ci, K in enumerate(KS):
            ax = axes[ri][ci]; ax.set_xticks(range(5)); ax.set_xticklabels(STAGES, fontsize=7.4, color=INK, linespacing=1.15); ax.set_xlim(-0.4, 4.4)
            ax.set_ylabel(f"% of golds within the top {K}", fontsize=8.2, color=INK)
            ax.set_title(f"({'abcdef'[ri * 3 + ci]}) {s}-field golds: top {K}", fontsize=8.8, color=INK, loc="left")
            for x in (1.5, 3.5): ax.axvline(x, color=RULE, lw=0.6, ls=":")
    order = [ds for ds, _, _, _ in DATASETS if ds in handles]
    lab = {ds: l for ds, l, _, _ in DATASETS}
    fig.legend([handles[d] for d in order], [lab[d] for d in order], loc="lower center", ncol=6, fontsize=8.8, frameon=False, handlelength=2.6, columnspacing=2.0, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.02, 0.99, "Cumulative ablation, final ranking at each stage:  1 Qwen3 cosine  ->  2 multi-view scorer  ->  3 scorer + OpenIE entity graph  ->  "
             "4 scorer + SciAffordGraph, CCMP gate off  ->  5 + CCMP = SciGraphIR", fontsize=8.8, color=INK, ha="left")
    fig.text(0.02, 0.962, "Hollow marker: stage 3 is a separately trained model with its own scorer. Stages 4 and 5 share weights and differ only in the gate. "
             "SIR-4: per-gold QUARTET stratum; TOMATO, MIR: query stratum.", fontsize=7.8, color=MUTED, ha="left")
    fig.savefig(a.out + ".pdf", bbox_inches="tight"); fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight")
    open(a.out + ".md", "w").write("\n".join(rows) + "\n"); print("wrote", a.out + ".png", a.out + ".md")


if __name__ == "__main__":
    main()
