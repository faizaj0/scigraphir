#!/usr/bin/env python3
"""
graph_channel_curves.py -- where each scorer ranks the gold, as cumulative curves (share of golds ranked within k, log k),
on ALL golds and on the SEMANTICALLY DIFFICULT golds (Qwen3 cosine ranks them beyond --buried), which is what the graph
channel is for. Two rows: the graph channel alone, and the final ranking.

  (a) graph channel alone, all golds          SciAffordGraph reasoner . OpenIE entity-graph reasoner . scorer . cosine
  (b) graph channel alone, buried golds       same curves on golds with cosine rank > --buried
  (c) final ranking, all golds                SciGraphIR (fused) . OpenIE model (fused) . multi-view scorer . cosine
  (d) final ranking, buried golds

One figure per stratum (cross / same) from hops_frame_ccmp.json (+ hops_openie.json); QUARTET eval.json gives the
per-gold stratum for SIR-4, else the query stratum in the hops file. Ranks are the scan ranks stored per gold:
rank = {fused, graph, scorer, dense}; the OpenIE model's fused/scorer are its own.

    python3 eval/graph_channel_curves.py --dir results/qualitative/drive_scan_sir4_cs \
        --quartet ../benchmark/data.nosync/benchmark/cs_test_final/eval.json --out ../figures/fig_graph_channel_curves_cs
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, RULE = "#262a30", "#7c828c", "#ced2da"
C_FRAME, C_ENT, C_SC, C_COS = "#2a69a0", "#d9701a", "#705296", "#8a9099"
KS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]




def openie_path(d, prefix="hops_"):
    """the OpenIE-arm hops file: the showcase scan if present, else the Table-4 run (TOMATO: July v1 OpenIE model, subset of queries)"""
    for name in (f"{prefix}openie.json", f"{prefix}t4_openie.json", "hops_t4_openie.json"):
        if os.path.exists(f"{d}/{name}"): return f"{d}/{name}"
    return None



def frame_arm(d, prefix="hops_", want="auto"):
    """which SciAffordGraph arm the hops files hold: the merged graph (hyb_ccmp, the current SciAffordGraph) when present,
    else the older frame-only graph (frame_ccmp). Returns (arm name, label suffix)."""
    if want == "auto": want = "hyb_ccmp" if os.path.exists(f"{d}/{prefix}hyb_ccmp.json") else "frame_ccmp"
    return want, ("merged graph" if want.startswith("hyb") else "frame-only graph, outdated")


def load(path):
    out = {}
    for r in json.load(open(path)):
        for t in r["targets"]: out[(r["id"], t["doc"])] = {"stratum": r.get("stratum") or "unlabelled", "rank": t.get("rank") or {}, "n_doc": r.get("n_doc")}
    return out


def curve(ranks, ks):
    n = len(ranks); return [100.0 * sum(r <= k for r in ranks) / n for k in ks] if n else [float("nan")] * len(ks)


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(RULE); ax.spines["bottom"].set_color(RULE); ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.grid(axis="both", color=RULE, lw=0.5, alpha=0.7); ax.set_xscale("log"); ax.set_ylim(0, 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--prefix", default="hops_"); ap.add_argument("--quartet", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--buried", type=int, default=100); ap.add_argument("--title", default="")
    ap.add_argument("--arm", default="auto", help="SciAffordGraph arm: auto (hyb_ccmp if present, else frame_ccmp), hyb_ccmp or frame_ccmp")
    a = ap.parse_args()
    ARM, ARMLAB = frame_arm(a.dir, a.prefix, a.arm); print("SciAffordGraph arm:", ARM, f"({ARMLAB})")
    F = load(f"{a.dir}/{a.prefix}{ARM}.json")
    _ep = openie_path(a.dir, a.prefix); E = load(_ep) if _ep else {}
    O = load(f"{a.dir}/{a.prefix}{ARM}_off.json") if os.path.exists(f"{a.dir}/{a.prefix}{ARM}_off.json") else {}   # same weights, gate off
    unit = "query stratum (hops file)"
    if a.quartet and os.path.exists(a.quartet):
        qz = {}
        for x in json.load(open(a.quartet)):
            for d, m in (x.get("quartet", {}).get("per_document") or {}).items(): qz[(x["id"], d)] = m.get("stratum")
        for k, v in F.items(): v["stratum"] = qz.get(k) or "unlabelled"
        unit = "gold stratum (QUARTET)"
    n_doc = max((v["n_doc"] or 0) for v in F.values()) or 5000; ks = [k for k in KS if k < n_doc] + [n_doc]
    labels = {v["stratum"] for v in F.values()} - {"unlabelled"}
    strata = [s for s in ("cross", "same") if s in labels] or sorted(labels) or ["all"]     # SIR-4: cross/same; TOMATO/MIR: their own labels
    if strata == ["all"]:
        for v in F.values(): v["stratum"] = "all"
    summary = {"unit": unit, "buried": a.buried, "n_doc": n_doc}
    # one figure per dataset: columns = strata (cross | same), rows = all golds | golds cosine buries beyond --buried
    SPEC = {"frame_on": (f"SciAffordGraph reasoner ({ARMLAB}), CCMP gate on", C_FRAME, "-", "o", True),
            "frame_off": (f"SciAffordGraph reasoner ({ARMLAB}), CCMP gate off (same weights)", C_FRAME, "--", "o", False),
            "entity": ("OpenIE entity-graph reasoner", C_ENT, "-", "s", True),
            "scorer": ("text scorer, no graph (reference)", C_SC, "-.", "^", True),
            "cosine": ("Qwen3 cosine (reference)", C_COS, ":", "D", False)}
    ncol = len(strata); nrow = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol + 0.3, 3.8 * nrow + 0.5), squeeze=False, gridspec_kw={"wspace": 0.26, "hspace": 0.5})
    handles = {}; letters = "abcdef"
    for ci, s in enumerate(strata):
        keys = [k for k, v in F.items() if v["stratum"] == s and all(v["rank"].get(c) for c in ("fused", "graph", "scorer", "dense"))]
        subsets = [("all golds", keys), (f"golds cosine buries beyond rank {a.buried}", [k for k in keys if F[k]["rank"]["dense"] > a.buried])]
        for ri, (sub, ks_) in enumerate(subsets):
            ax = axes[ri][ci]; style(ax)
            ke = [k for k in ks_ if k in E]; ko = [k for k in ks_ if k in O]
            curves = [("frame_on", [F[k]["rank"]["graph"] for k in ks_]), ("frame_off", [O[k]["rank"]["graph"] for k in ko]),
                      ("entity", [E[k]["rank"]["graph"] for k in ke]), ("scorer", [F[k]["rank"]["scorer"] for k in ks_]), ("cosine", [F[k]["rank"]["dense"] for k in ks_])]
            title = f"({letters[ri * ncol + ci]}) {s}-field: {sub}\n(n = {len(ks_)})" if s != "all" else f"({letters[ri]}) {sub} (n = {len(ks_)})"
            for key, ranks in curves:
                if not ranks: continue
                lab, col, ls, mk, filled = SPEC[key]; ys = curve(ranks, ks)
                ln, = ax.plot(ks, ys, color=col, ls=ls, marker=mk, ms=4.2 if filled else 4.6, lw=1.5, mfc=col if filled else "white", mec=col, mew=1.1, label=lab)
                handles[key] = ln
                summary[f"{s}/graph channel alone, {sub}/{lab}"] = {"n": len(ranks), **{f"R@{k}": round(y, 1) for k, y in zip(ks, ys) if k in (5, 10, 50, 100)}}
            ax.set_title(title, fontsize=8.2, color=INK, loc="left")
            ax.set_xlabel("rank k of the gold (log scale)", fontsize=8, color=INK); ax.set_ylabel("% of golds ranked within k", fontsize=8, color=INK)
            if ri == 1: ax.axvline(a.buried, color=RULE, lw=0.8); ax.text(a.buried * 1.15, 3, f"cosine > {a.buried} by construction", fontsize=6.4, color=MUTED)
    order = [k for k in ("frame_on", "frame_off", "entity", "scorer", "cosine") if k in handles]
    fig.legend([handles[k] for k in order], [SPEC[k][0] for k in order], loc="lower center", ncol=3 if ncol > 1 else 2, fontsize=8, frameon=False, handlelength=3.4, columnspacing=1.8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(a.title or f"graph channel alone ({unit})", fontsize=9.5, color=INK, x=0.02, y=0.995, ha="left")
    fig.savefig(a.out + ".pdf", bbox_inches="tight"); fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", a.out + ".png")
    json.dump(summary, open(a.out + "_summary.json", "w"), indent=1)
    for k, v in summary.items():
        if isinstance(v, dict): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
