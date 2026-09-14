#!/usr/bin/env python3
"""
plot_hops.py -- hop-count distribution of the reasoner's top path, GFM-RAG Figure 6 style.

For every (query, gold) in an interpret_paths.py output, take the number of hops in the
highest-weighted path from the query's seed frames to the gold (the prediction) and the
fewest hops from any seed to the gold in the graph (`min_hops`, the structural reference:
SIR-4 has no annotated reasoning chains, so the shortest seed-to-gold distance stands in
for GFM-RAG's ground-truth hop count). One panel per stratum; one line per arm plus the
dashed reference for each graph; MAE between each arm and its own graph's reference.

usage:
  plot_hops.py --arm "SciAffordGraph + CCMP=hops_frame_on.json:frame" \
               --arm "SciAffordGraph, no CCMP=hops_frame_off.json:frame" \
               --arm "OpenIE graph=hops_openie.json:openie" \
               --out fig_hops_cs.pdf [--title "SIR-4 CS"] [--max-hops 6]

Each --arm is  label=path[:graphkey]; arms sharing a graphkey share one reference line
(their min_hops are identical because the graph is the same).
"""
import argparse, json, os
from collections import Counter, defaultdict

import numpy as np


def load(path):
    rows = []
    for r in json.load(open(path)):
        for t in r.get("targets", []):
            hops = len(t["paths"][0]["hops"]) if t.get("paths") else None
            rows.append({"id": r["id"], "doc": t["doc"], "stratum": r.get("stratum") or "same",
                         "pred": hops, "ref": t.get("min_hops")})
    return rows


def dist(values, max_hops):
    c = Counter(v for v in values if v is not None)
    n = sum(c.values())
    xs = list(range(1, max_hops + 1))
    return xs, [100.0 * c.get(x, 0) / n if n else 0.0 for x in xs], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="label=path[:graphkey]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--max-hops", type=int, default=6)
    ap.add_argument("--strata", default="same,cross")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = []
    for spec in a.arm:
        label, rest = spec.split("=", 1)
        path, _, gkey = rest.partition(":")
        arms.append((label, gkey or label, load(path)))
    strata = a.strata.split(",")
    stratum_label = {"same": "same-field", "cross": "cross-field", "all": "all"}

    style = [("#1f77b4", "s", "-"), ("#1f77b4", "s", "--"), ("#ff7f0e", "^", "-"), ("#9467bd", "D", "-")]
    ref_style = {"frame": ("#2ca02c", "o"), "openie": ("#7f7f7f", "o")}

    fig, axes = plt.subplots(1, len(strata), figsize=(4.2 * len(strata), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    summary = []
    for ax, st in zip(axes, strata):
        drawn_ref = set()
        for k, (label, gkey, rows) in enumerate(arms):
            sub = [r for r in rows if st == "all" or r["stratum"] == st]
            reached = [r for r in sub if r["pred"] is not None]
            xs, ys, n = dist([r["pred"] for r in reached], a.max_hops)
            col, mk, ls = style[k % len(style)]
            ax.plot(xs, ys, marker=mk, color=col, ls=ls, mfc=col if ls == "-" else "white", label=label, lw=1.6, ms=6)
            if gkey not in drawn_ref:
                rx, ry, rn = dist([r["ref"] for r in sub], a.max_hops)
                rc, rmk = ref_style.get(gkey, ("#2ca02c", "o"))
                ax.plot(rx, ry, marker=rmk, color=rc, ls=":", label=f"shortest seed$\\to$gold ({gkey} graph)", lw=1.6, ms=6)
                drawn_ref.add(gkey)
            paired = [(r["pred"], r["ref"]) for r in reached if r["ref"] is not None]
            mae = float(np.mean([abs(p - q) for p, q in paired])) if paired else float("nan")
            extra = float(np.mean([p - q for p, q in paired])) if paired else float("nan")
            mean_pred = float(np.mean([r["pred"] for r in reached])) if reached else float("nan")
            summary.append({"stratum": st, "arm": label, "golds": len(sub), "reached": len(reached),
                            "mean_hops": mean_pred, "mae_vs_shortest": mae, "extra_hops": extra})
            ax.text(0.98, 0.95 - 0.09 * k, f"{label}: MAE {mae:.2f}, reached {100*len(reached)/max(len(sub),1):.0f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=col)
        ax.set_title(f"{a.title + ', ' if a.title else ''}{stratum_label.get(st, st)}", fontsize=10)
        ax.set_xlabel("hops in the top reasoning path")
        ax.set_xticks(range(1, a.max_hops + 1))
        ax.grid(True, ls="--", alpha=0.4)
    axes[0].set_ylabel("percentage of golds (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    base, _ = os.path.splitext(a.out)
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    json.dump(summary, open(base + "_summary.json", "w"), indent=1)
    print(f"{'stratum':<8}{'arm':<26}{'golds':>6}{'reached':>8}{'mean':>7}{'MAE':>7}{'extra':>7}")
    for s in summary:
        print(f"{s['stratum']:<8}{s['arm']:<26}{s['golds']:>6}{s['reached']:>8}{s['mean_hops']:>7.2f}{s['mae_vs_shortest']:>7.2f}{s['extra_hops']:>7.2f}")
    print("wrote", a.out, "and", base + ".png")


if __name__ == "__main__":
    main()
