#!/usr/bin/env python3
"""
graph_head_fig.py -- the head of the ranking only: share of golds ranked within k for k = 1..10, graph channel alone,
one small panel per dataset, cross-field golds (top row) and same-field golds (bottom row). Curves: SciAfford graph
reasoner with the CCMP gate on, the same weights with the gate off, the OpenIE entity-graph reasoner, and the text
scorer / Qwen3 cosine as references. Also writes a table of R@1 / R@3 / R@5 / R@10 for every curve.

    python3 eval/graph_head_fig.py --out ../figures/fig_graph_channel_head
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
SPEC = {"frame_on": ("SciAfford graph reasoner, CCMP gate on", C_FRAME, "-", "o", True),
        "frame_off": ("SciAfford graph reasoner, CCMP gate off (same weights)", C_FRAME, "--", "o", False),
        "entity": ("OpenIE entity-graph reasoner", C_ENT, "-", "s", True),
        "scorer": ("text scorer, no graph (reference)", C_SC, "-.", "^", True),
        "cosine": ("Qwen3 cosine (reference)", C_COS, ":", "D", False)}




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
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(2.6 * len(DATASETS), 6.4), gridspec_kw={"wspace": 0.32, "hspace": 0.55})
    handles = {}; rows = ["| dataset | stratum | curve | n | R@1 | R@3 | R@5 | R@10 |", "|---|---|---|---|---|---|---|---|"]
    for ci, (ds, label) in enumerate(DATASETS):
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
        for ri, s in enumerate(("cross", "same")):
            ax = axes[ri][ci]; style(ax)
            keys = [k for k, v in F.items() if v["stratum"] == s and all(v["rank"].get(c) for c in ("graph", "scorer", "dense"))]
            if len(keys) < 30:
                ax.text(0.5, 0.5, f"{label}\nno {s}-field golds", ha="center", va="center", fontsize=8, color=MUTED, transform=ax.transAxes); ax.set_axis_off(); continue
            ke = [k for k in keys if k in E and E[k]["rank"].get("graph")]; ko = [k for k in keys if k in O and O[k]["rank"].get("graph")]
            curves = [("frame_on", [F[k]["rank"]["graph"] for k in keys]), ("frame_off", [O[k]["rank"]["graph"] for k in ko]),
                      ("entity", [E[k]["rank"]["graph"] for k in ke]), ("scorer", [F[k]["rank"]["scorer"] for k in keys]), ("cosine", [F[k]["rank"]["dense"] for k in keys])]
            for key, ranks in curves:
                if not ranks: continue
                lab, col, ls, mk, filled = SPEC[key]; ys = share(ranks, KS)
                ln, = ax.plot(KS, ys, color=col, ls=ls, marker=mk, ms=3.6 if filled else 4.0, lw=1.4, mfc=col if filled else "white", mec=col, mew=1.0)
                handles[key] = ln
                rows.append(f"| {ds} | {s} ({unit}) | {lab} | {len(ranks)} | {ys[0]:.1f} | {ys[2]:.1f} | {ys[4]:.1f} | {ys[9]:.1f} |")
            ax.set_xticks(KS); ax.set_xlim(0.6, 10.4); ax.set_ylim(0, 80)
            ax.set_title(f"{label}\n{s}-field golds, n = {len(keys)}", fontsize=7.6, color=INK, loc="left")
            if ci == 0: ax.set_ylabel("% of golds ranked within k", fontsize=8, color=INK)
            if ri == 1 or ds == "mir": ax.set_xlabel("rank k of the gold", fontsize=8, color=INK)
    order = [k for k in ("frame_on", "frame_off", "entity", "scorer", "cosine") if k in handles]
    fig.legend([handles[k] for k in order], [SPEC[k][0] for k in order], loc="lower center", ncol=3, fontsize=8.2, frameon=False, handlelength=3.2, columnspacing=1.8, bbox_to_anchor=(0.5, -0.05))
    fig.text(0.01, 0.975, "The head of the ranking: share of golds within the top k, k = 1 to 10, graph channel alone (SIR-4 fields use the per-gold SIR-4 stratum; TOMATO and MIR the query stratum)", fontsize=9.2, color=INK, ha="left")
    fig.savefig(a.out + ".pdf", bbox_inches="tight"); fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight")
    open(a.out + ".md", "w").write("\n".join(rows) + "\n"); print("wrote", a.out + ".png", a.out + ".md")


if __name__ == "__main__":
    main()
