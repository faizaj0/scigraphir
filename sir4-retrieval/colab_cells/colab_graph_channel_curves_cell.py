# ===== Graph-channel curves for EVERY dataset with hops files on Drive (paste into any Colab kernel; needs only Drive) =====
# Per dataset and stratum: share of golds the GRAPH CHANNEL ALONE ranks within k (log k), on all golds and on the golds Qwen3 cosine
# buries beyond rank 100. Curves: SciAffordGraph reasoner +CCMP, the same weights with the gate off, the OpenIE entity-graph reasoner,
# and the text scorer / cosine as references. Reads outputs/scan/<dataset>/hops_frame_ccmp.json (+ hops_frame_ccmp_off.json, hops_openie.json).
# SIR-4 fields use the per-gold QUARTET stratum when the CARGO tree is unpacked, else the query stratum stored in the hops file.
import os, sys, glob, json, subprocess
if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive; drive.mount("/content/drive")
DRIVE = globals().get("DRIVE", "/content/drive/MyDrive/cargo-gfmrag"); SCAN = f"{DRIVE}/outputs/scan"
_root = globals().get("CARGO_ROOT", "/content/cargo")
os.makedirs("/content/eval", exist_ok=True)
open("/content/eval/graph_channel_curves.py", "w").write(r'''#!/usr/bin/env python3
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
        --quartet ../quartet/data.nosync/benchmark/cs_test_final/eval.json --out ../figures/fig_graph_channel_curves_cs
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, RULE = "#262a30", "#7c828c", "#ced2da"
C_FRAME, C_ENT, C_SC, C_COS = "#2a69a0", "#d9701a", "#705296", "#8a9099"
KS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


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
    ap.add_argument("--final", action="store_true", help="also draw the final-ranking row (fused scores)")
    a = ap.parse_args()
    F = load(f"{a.dir}/{a.prefix}frame_ccmp.json")
    E = load(f"{a.dir}/{a.prefix}openie.json") if os.path.exists(f"{a.dir}/{a.prefix}openie.json") else {}
    O = load(f"{a.dir}/{a.prefix}frame_ccmp_off.json") if os.path.exists(f"{a.dir}/{a.prefix}frame_ccmp_off.json") else {}   # same weights, gate off
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
    for s in strata:
        keys = [k for k, v in F.items() if v["stratum"] == s and all(v["rank"].get(c) for c in ("fused", "graph", "scorer", "dense"))]
        subsets = [("all golds", keys), (f"golds cosine buries beyond rank {a.buried}", [k for k in keys if F[k]["rank"]["dense"] > a.buried])]
        nrow = 2 if a.final else 1
        fig, axes = plt.subplots(nrow, 2, figsize=(9.2, 3.9 * nrow + 0.4), gridspec_kw={"wspace": 0.24, "hspace": 0.5}); axes = axes.ravel()
        for ax in axes: style(ax)
        # curve specs: label, colour, line style, marker, filled
        SPEC = {"frame_on": ("SciAffordGraph reasoner, +CCMP", C_FRAME, "-", "o", True), "frame_off": ("SciAffordGraph reasoner, CCMP gate off (same weights)", C_FRAME, "--", "o", False),
                "entity": ("OpenIE entity-graph reasoner", C_ENT, "-", "s", True), "scorer": ("multi-view scorer, text only (reference)", C_SC, "-.", "^", True),
                "cosine": ("Qwen3 cosine (reference)", C_COS, ":", "D", False)}
        handles = {}
        for j, (sub, ks_) in enumerate(subsets):
            ke = [k for k in ks_ if k in E]; ko = [k for k in ks_ if k in O]
            rows = [(j, f"({'ab'[j]}) graph channel alone: {sub}\n(n = {len(ks_)})", [
                        ("frame_on", [F[k]["rank"]["graph"] for k in ks_]), ("frame_off", [O[k]["rank"]["graph"] for k in ko]),
                        ("entity", [E[k]["rank"]["graph"] for k in ke]), ("scorer", [F[k]["rank"]["scorer"] for k in ks_]), ("cosine", [F[k]["rank"]["dense"] for k in ks_])])]
            if a.final:
                rows.append((2 + j, f"({'cd'[j]}) final ranking: {sub}\n(n = {len(ks_)})", [
                        ("frame_on", [F[k]["rank"]["fused"] for k in ks_]), ("frame_off", [O[k]["rank"]["fused"] for k in ko]),
                        ("entity", [E[k]["rank"]["fused"] for k in ke]), ("scorer", [F[k]["rank"]["scorer"] for k in ks_]), ("cosine", [F[k]["rank"]["dense"] for k in ks_])]))
            for ai, title, curves in rows:
                ax = axes[ai]
                for key, ranks in curves:
                    if not ranks: continue
                    lab, col, ls, mk, filled = SPEC[key]
                    ys = curve(ranks, ks)
                    ln, = ax.plot(ks, ys, color=col, ls=ls, marker=mk, ms=4.2 if filled else 4.6, lw=1.5, mfc=col if filled else "white", mec=col, mew=1.1, label=lab)
                    handles[key] = ln
                    summary[f"{s}/{title.split(chr(10))[0]}/{lab}"] = {"n": len(ranks), **{f"R@{k}": round(y, 1) for k, y in zip(ks, ys) if k in (5, 10, 50, 100)}}
                ax.set_title(title, fontsize=8.2, color=INK, loc="left"); ax.set_xlabel("rank k of the gold (log scale)", fontsize=8, color=INK)
                ax.set_ylabel("% of golds ranked within k", fontsize=8, color=INK)
                if j == 1: ax.axvline(a.buried, color=RULE, lw=0.8); ax.text(a.buried * 1.15, 3, f"cosine > {a.buried} by construction", fontsize=6.4, color=MUTED)
        order = [k for k in ("frame_on", "frame_off", "entity", "scorer", "cosine") if k in handles]
        fig.legend([handles[k] for k in order], [SPEC[k][0] for k in order], loc="lower center", ncol=3, fontsize=7.6, frameon=False, handlelength=3.2, columnspacing=1.6, bbox_to_anchor=(0.5, -0.13 if nrow == 1 else -0.05))
        ttl = a.title or f"{s}-field golds, {unit}"
        fig.suptitle(ttl, fontsize=9, color=INK, x=0.02, y=1.0 if nrow == 1 else 0.975, ha="left")
        out = f"{a.out}_{s}"; fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); plt.close(fig)
        print("wrote", out + ".png")
    json.dump(summary, open(a.out + "_summary.json", "w"), indent=1)
    for k, v in summary.items():
        if isinstance(v, dict): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
''')
from IPython.display import Image, display
allsum = {}
for d in sorted(glob.glob(f"{SCAN}/*/")):
    ds = os.path.basename(d.rstrip("/"))
    if not os.path.exists(f"{d}/hops_frame_ccmp.json"): continue
    out = f"{d}/fig_graph_channel_curves_{ds}"
    cmd = [sys.executable, "/content/eval/graph_channel_curves.py", "--dir", d, "--out", out, "--title", f"{ds}"]
    _qt = f"{_root}/quartet/data.nosync/benchmark/{ds.replace('sir4_', '')}_test_final/eval.json"
    if os.path.exists(_qt): cmd += ["--quartet", _qt]
    r = subprocess.run(cmd, capture_output=True, text=True); print("=" * 30, ds); print(r.stdout[-3000:], r.stderr[-1500:])
    for p_ in sorted(glob.glob(out + "_*.png")): print(os.path.relpath(p_, DRIVE)); display(Image(p_))
    if os.path.exists(out + "_summary.json"): allsum[ds] = json.load(open(out + "_summary.json"))
json.dump(allsum, open(f"{SCAN}/graph_channel_curves_all_summary.json", "w"), indent=1)
# one table: R@10 / R@50 of every curve, per dataset and stratum, all golds and buried golds
rows = ["| dataset | stratum | subset | curve | n | R@10 | R@50 | R@100 |", "|---|---|---|---|---|---|---|---|"]
for ds, sm in allsum.items():
    for key, v in sm.items():
        if not isinstance(v, dict) or "R@10" not in v: continue
        st, title, curve = key.split("/", 2); subset = "buried" if "buries" in title else "all"
        rows.append(f"| {ds} | {st} | {subset} | {curve} | {v['n']} | {v['R@10']} | {v['R@50']} | {v['R@100']} |")
open(f"{SCAN}/graph_channel_curves_all.md", "w").write("\n".join(rows)); print("\n".join(rows)); print("saved:", f"{SCAN}/graph_channel_curves_all.md")
