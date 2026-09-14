#!/usr/bin/env python3
"""
graph_summary_fig.py -- the graph reasoner across every dataset in one figure (line graphs, one line per dataset).

Rows: cross-field golds (top) and same-field golds (bottom). Columns:
  (1) reach against route length: P(graph-channel rank <= K | length of the top route)   -> reach is a distance effect, everywhere
  (2) frame graph minus entity graph: difference in the share of golds the two graph reasoners rank within k (log k)
  (3) CCMP gate on minus gate off, same weights: difference in the share of golds the graph channel ranks within k

Inputs per dataset: results/qualitative/drive_scan_<ds>/hops_frame_ccmp.json, hops_frame_ccmp_off.json, hops_openie.json;
per-gold QUARTET strata for the SIR-4 fields (cs_test_final, <field>_test_low), query strata for TOMATO / MIR.

    python3 eval/graph_summary_fig.py --out ../figures/fig_graph_reasoning_summary
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); S4 = os.path.dirname(HERE); R = os.path.dirname(S4)
INK, MUTED, RULE = "#262a30", "#7c828c", "#ced2da"
DATASETS = [("sir4_cs", "SIR-4 CS", "#2a69a0"), ("sir4_biology", "SIR-4 Biology", "#208070"), ("sir4_physics", "SIR-4 Physics", "#705296"),
            ("sir4_matsci", "SIR-4 MatSci", "#a47822"), ("tomato", "TOMATO", "#c44e52"), ("mir", "MIR", "#6e747e")]
_QC = f"{S4}/results/qualitative/quartet_cache"
QUARTET = {"sir4_cs": f"{_QC}/cs_test_final.eval.json", **{f"sir4_{f}": f"{_QC}/{f}_test_low.eval.json" for f in ("biology", "physics", "matsci")}}
KS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
MARK = {"sir4_cs": "o", "sir4_biology": "s", "sir4_physics": "^", "sir4_matsci": "D", "tomato": "v", "mir": "P"}


def valid(p, seeds, gold, max_hops=6):
    h = p["hops"]
    if not h or h[0]["head"] not in seeds or not all(x["tail"] == y["head"] for x, y in zip(h, h[1:])): return False
    ns = [h[0]["head"]] + [x["tail"] for x in h]; return len(ns) == len(set(ns)) and h[-1]["tail"] == gold and len(h) <= max_hops




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


def load(path, with_paths=False):
    out = {}
    for r in json.load(open(path)):
        seeds = set(r["seeds"])
        for t in r["targets"]:
            rec = {"stratum": r.get("stratum") or "unlabelled", "rank": t.get("rank") or {}}
            if with_paths:
                ps = [p for p in t.get("paths", []) if valid(p, seeds, t["doc"])]; rec["hops"] = len(ps[0]["hops"]) if ps else None
            out[(r["id"], t["doc"])] = rec
    return out


def share(ranks, ks):
    n = len(ranks); return [100.0 * sum(r <= k for r in ranks) / n for k in ks] if n else [float("nan")] * len(ks)


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(RULE); ax.spines["bottom"].set_color(RULE); ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.7)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=10); ap.add_argument("--arm", default="auto", help="auto | hyb_ccmp | frame_ccmp"); a = ap.parse_args(); K = a.k
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), gridspec_kw={"wspace": 0.36, "hspace": 0.58})
    for ax in axes.ravel(): style(ax)
    summary = {}; handles = {}
    for ds, label, col in DATASETS:
        d = f"{S4}/results/qualitative/drive_scan_{ds}"
        if not (os.path.exists(f"{d}/hops_frame_ccmp.json") or os.path.exists(f"{d}/hops_hyb_ccmp.json")): continue
        ARM, ARMLAB = frame_arm(d, "hops_", a.arm)
        F = load(f"{d}/hops_{ARM}.json", with_paths=True)
        O = load(f"{d}/hops_{ARM}_off.json") if os.path.exists(f"{d}/hops_{ARM}_off.json") else {}
        _ep = openie_path(d); E = load(_ep) if _ep else {}
        if ds in QUARTET and os.path.exists(QUARTET[ds]):
            qz = {}
            for x in json.load(open(QUARTET[ds])):
                for doc, m in (x.get("quartet", {}).get("per_document") or {}).items(): qz[(x["id"], doc)] = m.get("stratum")
            for k, v in F.items(): v["stratum"] = qz.get(k) or "unlabelled"
        for ri, s in enumerate(("cross", "same")):
            keys = [k for k, v in F.items() if v["stratum"] == s and v["rank"].get("graph")]
            if len(keys) < 30: continue
            # (1) reach against route length
            hs = list(range(1, 7)); ys = []
            for h in hs:
                ks_ = [k for k in keys if F[k]["hops"] == h]
                ys.append(100.0 * sum(F[k]["rank"]["graph"] <= K for k in ks_) / len(ks_) if len(ks_) >= 5 else float("nan"))
            ax = axes[ri][0]; ln, = ax.plot(hs, ys, color=col, marker=MARK[ds], ms=4.2, lw=1.5, label=label); handles[ds] = ln
            summary[f"{ds}/{s}/reach_by_hops"] = dict(zip(map(str, hs), ys))
            # (2) frame graph minus entity graph
            ke = [k for k in keys if k in E and E[k]["rank"].get("graph")]
            if ke:
                diff = [f - e for f, e in zip(share([F[k]["rank"]["graph"] for k in ke], KS), share([E[k]["rank"]["graph"] for k in ke], KS))]
                axes[ri][1].plot(KS, diff, color=col, marker=MARK[ds], ms=4.2, lw=1.5); summary[f"{ds}/{s}/frame_minus_entity"] = dict(zip(map(str, KS), diff))
            # (3) gate on minus gate off
            ko = [k for k in keys if k in O and O[k]["rank"].get("graph")]
            if ko:
                diff = [f - o for f, o in zip(share([F[k]["rank"]["graph"] for k in ko], KS), share([O[k]["rank"]["graph"] for k in ko], KS))]
                axes[ri][2].plot(KS, diff, color=col, marker=MARK[ds], ms=4.2, lw=1.5); summary[f"{ds}/{s}/gate_on_minus_off"] = dict(zip(map(str, KS), diff))
    for ri, s in enumerate(("cross", "same")):
        ax = axes[ri][0]; ax.set_xticks(range(1, 7)); ax.set_ylim(0, 102)
        ax.set_xlabel("length of the top route to the gold (hops)", fontsize=8, color=INK); ax.set_ylabel(f"% of golds with graph rank $\\leq$ {K}", fontsize=8, color=INK)
        ax.set_title(f"({'ad'[ri]}) {s}-field golds: reach against route length", fontsize=8.6, color=INK, loc="left")
        for ci, (ttl, yl) in enumerate([("frame graph $-$ entity graph", "difference in % of golds within k\n(SciAffordGraph $-$ OpenIE graph, points)"),
                                        ("CCMP gate on $-$ gate off (same weights)", "difference in % of golds within k\n(gate on $-$ gate off, points)")], start=1):
            ax = axes[ri][ci]; ax.set_xscale("log"); ax.axhline(0, color=INK, lw=0.8)
            ax.set_xlabel("rank k of the gold (log scale)", fontsize=8, color=INK); ax.set_ylabel(yl, fontsize=7.8, color=INK)
            ax.set_title(f"({'abcdef'[ri * 3 + ci]}) {s}-field golds: {ttl}", fontsize=8.6, color=INK, loc="left")
            ax.text(0.98, 0.04, "above 0: first is better", transform=ax.transAxes, fontsize=6.6, color=MUTED, ha="right")
    lo = min(min(ax.get_ylim()[0] for ax in axes[:, 1]), min(ax.get_ylim()[0] for ax in axes[:, 2])); hi = max(max(ax.get_ylim()[1] for ax in axes[:, 1]), max(ax.get_ylim()[1] for ax in axes[:, 2]))
    for ax in list(axes[:, 1]) + list(axes[:, 2]): ax.set_ylim(min(lo, -5), max(hi, 5))
    order = [ds for ds, _, _ in DATASETS if ds in handles]
    fig.legend([handles[d] for d in order], [dict((x[0], x[1]) for x in DATASETS)[d] for d in order], loc="lower center", ncol=6, fontsize=8.5, frameon=False, handlelength=3, columnspacing=1.8, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.02, 0.975, "The graph reasoner across six datasets, graph channel alone (MIR has same-field golds only; TOMATO has no entity-graph run)", fontsize=9.5, color=INK, ha="left")
    fig.savefig(a.out + ".pdf", bbox_inches="tight"); fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight")
    json.dump(summary, open(a.out + "_summary.json", "w"), indent=1); print("wrote", a.out + ".png")


if __name__ == "__main__":
    main()
