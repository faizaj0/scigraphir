#!/usr/bin/env python3
"""
ablation_head_fig.py -- the head of the FINAL ranking under the cumulative ablation: share of golds ranked within k for k = 1..10,
one small panel per dataset, cross-field golds (top row) and same-field golds (bottom row). Stages, each a full ranking:
Qwen3 cosine -> multi-view scorer -> scorer + OpenIE entity graph (that model's fused score) -> scorer + SciAfford graph with the
CCMP gate off (same weights) -> scorer + SciAfford graph + CCMP (SciGraphIR). Also writes a table of R@1 / R@3 / R@5 / R@10.

    python3 eval/ablation_head_fig.py --out ../figures/fig_ablation_head
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); S4 = os.path.dirname(HERE)
INK, MUTED, RULE = "#262a30", "#7c828c", "#ced2da"
C_FRAME, C_ENT, C_SC, C_COS = "#2a69a0", "#d9701a", "#705296", "#8a9099"
DATASETS = [("sir4_cs", "SIR-4 CS"), ("sir4_biology", "SIR-4 Biology"), ("sir4_physics", "SIR-4 Physics"), ("sir4_matsci", "SIR-4 MatSci"), ("tomato", "TOMATO"), ("mir", "MIR")]
_QC = f"{S4}/results/qualitative/quartet_cache"
SIR4_EXPORTS = {"sir4_cs": f"{_QC}/cs_test_final.eval.json", **{f"sir4_{f}": f"{_QC}/{f}_test_low.eval.json" for f in ("biology", "physics", "matsci")}}
KS = list(range(1, 11))
SPEC = {"cosine": ("1  Qwen3 cosine", "#9aa0a8", ":", "D", False, 1.3),
        "scorer": ("2  multi-view scorer", "#705296", "-.", "^", True, 1.4),
        "entity": ("3  scorer + OpenIE entity graph (fused; that model's own scorer)", "#d9701a", "-", "s", True, 1.4),
        "frame_off": ("4  scorer + SciAfford graph, CCMP gate off (fused; same weights as 5)", "#7fb3d5", "--", "o", False, 1.5),
        "frame_on": ("5  scorer + SciAfford graph + CCMP = SciGraphIR (fused)", "#1f4e79", "-", "o", True, 2.0)}




def openie_path(d, prefix="hops_"):
    """the OpenIE-arm hops file: the showcase scan if present, else the Table-4 run (TOMATO: July v1 OpenIE model, subset of queries)"""
    for name in (f"{prefix}openie.json", f"{prefix}t4_openie.json", "hops_t4_openie.json"):
        if os.path.exists(f"{d}/{name}"): return f"{d}/{name}"
    return None



def frame_arm(d, prefix="hops_", want="auto"):
    """which SciAfford graph arm the hops files hold: the merged graph (hyb_ccmp, the current SciAfford graph) when present,
    else the older SciAfford graph (frame_ccmp). Returns (arm name, label suffix)."""
    if want == "auto": want = "hyb_ccmp" if os.path.exists(f"{d}/{prefix}hyb_ccmp.json") else "frame_ccmp"
    return want, ("merged graph" if want.startswith("hyb") else 'SciAfford graph, outdated')


def load(path):
    return {(r["id"], t["doc"]): {"stratum": r.get("stratum") or "unlabelled", "rank": t.get("rank") or {}} for r in json.load(open(path)) for t in r["targets"]}


def share(ranks, ks):
    n = len(ranks); return [100.0 * sum(r <= k for r in ranks) / n for k in ks] if n else [float("nan")] * len(ks)


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(RULE); ax.spines["bottom"].set_color(RULE); ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.7)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--arm", default="auto", help="auto | hyb_ccmp | frame_ccmp"); a = ap.parse_args()
    rows = ["| dataset | stratum | curve | n | R@1 | R@3 | R@5 | R@10 |", "|---|---|---|---|---|---|---|---|"]
    data = {}
    for ds, label in DATASETS:
        d = f"{S4}/results/qualitative/drive_scan_{ds}"
        ARM, ARMLAB = frame_arm(d, "hops_", a.arm); F = load(f"{d}/hops_{ARM}.json"); O = load(f"{d}/hops_{ARM}_off.json") if os.path.exists(f"{d}/hops_{ARM}_off.json") else {}
        _ep = openie_path(d); E = load(_ep) if _ep else {}
        unit = "query stratum"
        if ds in SIR4_EXPORTS and os.path.exists(SIR4_EXPORTS[ds]):
            qz = {}
            for x in json.load(open(SIR4_EXPORTS[ds])):
                for doc, m in (x.get("quartet", {}).get("per_document") or {}).items(): qz[(x["id"], doc)] = m.get("stratum")
            for k, v in F.items(): v["stratum"] = qz.get(k) or "unlabelled"
            unit = "gold stratum"
        for s in ("cross", "same"):
            keys = [k for k, v in F.items() if v["stratum"] == s and all(v["rank"].get(c) for c in ("fused", "scorer", "dense"))]
            if len(keys) < 30: continue
            ke = [k for k in keys if k in E and E[k]["rank"].get("fused")]; ko = [k for k in keys if k in O and O[k]["rank"].get("fused")]
            data[(ds, s)] = {"n": len(keys), "unit": unit, "curves": [("cosine", [F[k]["rank"]["dense"] for k in keys]), ("scorer", [F[k]["rank"]["scorer"] for k in keys]),
                             ("entity", [E[k]["rank"]["fused"] for k in ke]), ("frame_off", [O[k]["rank"]["fused"] for k in ko]), ("frame_on", [F[k]["rank"]["fused"] for k in keys])]}
    for s in ("cross", "same"):
        dss = [(ds, label) for ds, label in DATASETS if (ds, s) in data]
        ncol = 3; nrow = (len(dss) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(12.6, 4.1 * nrow + 0.6), squeeze=False, gridspec_kw={"wspace": 0.26, "hspace": 0.45}); axes = axes.ravel()
        handles = {}
        for i_, (ds, label) in enumerate(dss):
            ax = axes[i_]; style(ax); rec = data[(ds, s)]; ends = []
            for key, ranks in rec["curves"]:
                if not ranks: continue
                lab, col, ls, mk, filled, lw = SPEC[key]; ys = share(ranks, KS)
                ln, = ax.plot(KS, ys, color=col, ls=ls, marker=mk, ms=4.6 if filled else 5.0, lw=lw, mfc=col if filled else "white", mec=col, mew=1.2)
                handles[key] = ln; ends.append((ys[9], col, key))
                rows.append(f"| {ds} | {s} ({rec['unit']}) | {lab} | {len(ranks)} | {ys[0]:.1f} | {ys[2]:.1f} | {ys[4]:.1f} | {ys[9]:.1f} |")
            # value labels at k = 10, spread so they do not overlap
            ends.sort(); ylo, yhi = ax.get_ylim(); step = (yhi - ylo) * 0.045; last = None
            for y, col, key in ends:
                yy = y if last is None or y - last >= step else last + step
                ax.text(10.35, yy, f"{y:.1f}", fontsize=7.4, color=col, va="center", ha="left", fontweight="bold" if key == "frame_on" else "normal"); last = yy
            ax.set_xticks(KS); ax.set_xlim(0.6, 11.6)
            ax.set_title(f"{label}: {s}-field golds, n = {rec['n']}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel("rank k of the gold", fontsize=8.4, color=INK); ax.set_ylabel("% of golds ranked within k", fontsize=8.4, color=INK)
        for ax in axes[len(dss):]: ax.set_axis_off()
        order = [k for k in ("cosine", "scorer", "entity", "frame_off", "frame_on") if k in handles]
        fig.legend([handles[k] for k in order], [SPEC[k][0] for k in order], loc="lower center", ncol=2, fontsize=8.6, frameon=False, handlelength=3.4, columnspacing=2.0, bbox_to_anchor=(0.5, -0.02 - 0.02 * (3 - nrow)))
        fig.text(0.02, 0.99, f"Cumulative ablation, final ranking at each stage, {s}-field golds: share of golds within the top k, k = 1 to 10 (numbers at the right: k = 10)", fontsize=9.6, color=INK, ha="left")
        fig.text(0.02, 0.965, "SIR-4 fields: per-gold SIR-4 stratum; TOMATO and MIR: query stratum. Stage 3 is a separately trained model; stages 4 and 5 share weights.", fontsize=8, color=MUTED, ha="left")
        out = f"{a.out}_{s}"; fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", out + ".png")
    open(a.out + ".md", "w").write("\n".join(rows) + "\n"); print("wrote", a.out + ".md")


if __name__ == "__main__":
    main()
