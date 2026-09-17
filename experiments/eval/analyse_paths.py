#!/usr/bin/env python3
"""
analyse_paths.py -- what the reasoner's top paths look like, and whether they matter.

Reads interpret_paths.py outputs (one per arm) and produces:
  (A) route composition: which node types the top path goes through, split by how well the
      graph channel ranked the gold (<=5 / 6-50 / >50) and by stratum (same / cross);
  (B) hop-length distribution of the top path per arm and stratum, with the shortest
      seed->gold graph distance as a dotted reference (a floor, not a ground truth);
  (C) graph-channel rank against fused rank (the fusion bottleneck);
  (D) path necessity, when the records carry it: rank of the gold after removing the top-1 /
      top-3 paths' edges vs removing the same number of random edges;
  (E) CCMP selectivity, when the records carry a distractor: gate values on hops of paths to
      the gold vs hops of paths to the top-ranked wrong document.

usage:
  analyse_paths.py --arm "SciAfford graph + CCMP=hops_frame_on.json" --arm "SciAfford graph, no CCMP=hops_frame_off.json" \\
                   --arm "OpenIE graph=hops_openie.json" --docs documents.json --out fig_paths_cs
Decoded paths that do not start at a seed or do not chain are artefacts and are skipped.
"""
import argparse, json, os, statistics as st
from collections import Counter, defaultdict

import numpy as np

MECH = ("function", "limitation")


def ntype(name, docs):
    if name in docs:
        return "paper"
    if name.startswith("[") and "]" in name:
        return name[1:name.index("]")]
    return "entity"


def valid(p, seeds):
    h = p["hops"]
    return bool(h) and h[0]["head"] in seeds and all(a["tail"] == b["head"] for a, b in zip(h, h[1:]))


def load(path, docs):
    rows, dist = [], []
    for r in json.load(open(path)):
        seeds = set(r["seeds"]); stratum = r.get("stratum") or "same"
        for t in r.get("targets", []):
            ps = [p for p in t.get("paths", []) if valid(p, seeds)]
            p = ps[0] if ps else None
            types = [ntype(h["head"], docs) for h in p["hops"]] if p else []
            rows.append({"id": r["id"], "doc": t["doc"], "stratum": stratum, "rank": t["rank"], "min_hops": t.get("min_hops"),
                         "hops": len(p["hops"]) if p else None, "weight": p["weight"] if p else None, "types": types,
                         "rels": [h["rel"] for h in p["hops"]] if p else [],
                         "gates": [h["gate"] for h in p["hops"] if "gate" in h] if p else [],
                         "necessity": t.get("necessity"), "reached": t.get("paths") is not None and len(t.get("paths", [])) > 0})
        d = r.get("distractor")
        if d:
            ps = [p for p in d.get("paths", []) if valid(p, seeds)]
            dist.append({"id": r["id"], "stratum": stratum, "rank": d["rank"], "gates": [h["gate"] for h in ps[0]["hops"] if "gate" in h] if ps else [],
                         "hops": len(ps[0]["hops"]) if ps else None, "necessity": d.get("necessity")})
    return rows, dist


def route_class(row):
    """One label per top path: how the reasoner got to the gold."""
    ty = row["types"]
    if not ty:
        return "no valid path"
    if all(x in ("entity", "paper") for x in ty):          # OpenIE entity graph: untyped mentions
        return "direct entity mention" if row["hops"] == 1 else "co-mention chain"
    if "domain" in ty:
        return "via domain hub"
    if row["hops"] == 1:
        return "direct typed link"
    if any(x in MECH for x in ty):
        return "via function/limitation"
    if "paper" in ty:
        return "via bridge paper"
    return 'via method/task affordance representations'


ROUTE_ORDER = ["direct typed link", "via function/limitation", 'via method/task affordance representations', "via bridge paper", "via domain hub",
               "direct entity mention", "co-mention chain", "no valid path"]
BUCKETS = [("graph rank $\\leq$5", lambda r: r <= 5), ("6 to 50", lambda r: 5 < r <= 50), ("$>$50", lambda r: r > 50)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="label=path")
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True, help="output prefix (writes .pdf, .png, _tables.md)")
    ap.add_argument("--max-hops", type=int, default=6)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = json.load(open(a.docs))
    arms = []
    for spec in a.arm:
        label, path = spec.split("=", 1)
        rows, dist = load(path, docs)
        arms.append((label, rows, dist))
    md = []
    strata = ["same", "cross"]
    have_nec = any(r["necessity"] for _, rows, _ in arms for r in rows)
    have_dist = any(dist for _, _, dist in arms)
    ncol = 3 + int(have_nec) + int(have_dist)
    fig, axes = plt.subplots(2, ncol, figsize=(4.1 * ncol, 6.6))
    palette = {"direct typed link": "#2ca02c", "via function/limitation": "#98df8a", 'via method/task affordance representations': "#1f77b4",
               "via bridge paper": "#aec7e8", "via domain hub": "#d62728", "direct entity mention": "#ffbb78", "co-mention chain": "#ff7f0e", "no valid path": "#bbbbbb"}

    # ---- (A) route composition by outcome, first arm (SciAfford graph, CCMP on) and last arm (OpenIE)
    md.append("## A. Route composition of the top path, by graph-channel outcome\n")
    for si, s in enumerate(strata):
        ax = axes[si, 0]
        labels, bottoms = [], []
        x = 0; xt = []; xl = []
        for label, rows, _ in (arms[0], arms[-1]):
            for bname, f in BUCKETS:
                sub = [r for r in rows if r["stratum"] == s and f(r["rank"]["graph"])]
                n = len(sub)
                if n == 0:
                    x += 1; continue
                c = Counter(route_class(r) for r in sub)
                md.append(f"{label} | {s} | {bname.replace('$','')} | n={n} | " + ", ".join(f"{k} {100*c[k]/n:.0f}%" for k in ROUTE_ORDER if c[k]))
                b = 0
                for k in ROUTE_ORDER:
                    v = 100 * c[k] / n
                    if v:
                        ax.bar(x, v, bottom=b, color=palette[k], width=0.8, label=k if (si == 0 and x == 0) or k not in labels else None)
                        labels.append(k); b += v
                xt.append(x); xl.append(f"{bname}\nn={n}"); x += 1
            x += 0.6
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7)
        ax.set_ylabel("share of golds (%)"); ax.set_ylim(0, 100)
        ax.set_title(f"(A) route type, {s}-field   [left: {arms[0][0]}, right: {arms[-1][0]}]", fontsize=8.5)
    handles, labs = [], []
    for k in ROUTE_ORDER:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=palette[k])); labs.append(k)
    axes[0, 0].legend(handles, labs, fontsize=6.5, loc="upper right", ncol=2)

    # ---- (B) hop-length distribution
    md.append("\n## B. Hop length of the top path (valid paths only)\n")
    mk = ["s", "s", "^", "D"]; ls = ["-", "--", "-", "-"]; col = ["#1f77b4", "#1f77b4", "#ff7f0e", "#9467bd"]
    for si, s in enumerate(strata):
        ax = axes[si, 1]
        xs = list(range(1, a.max_hops + 1))
        for k, (label, rows, _) in enumerate(arms):
            sub = [r for r in rows if r["stratum"] == s and r["hops"]]
            c = Counter(r["hops"] for r in sub); n = len(sub)
            ax.plot(xs, [100 * c.get(h, 0) / n for h in xs], marker=mk[k % 4], ls=ls[k % 4], color=col[k % 4], mfc=col[k % 4] if ls[k % 4] == "-" else "white", label=label, lw=1.5, ms=5)
            mh = [r["min_hops"] for r in rows if r["stratum"] == s and r["min_hops"] is not None]
            md.append(f"{label} | {s} | golds {sum(r['stratum']==s for r in rows)} | valid top path {n} | mean hops {st.mean(r['hops'] for r in sub):.2f} | shortest seed->gold mean {st.mean(mh):.2f} | unreachable within {a.max_hops}: {sum(r['stratum']==s and r['min_hops'] is None for r in rows)}")
            if k in (0, len(arms) - 1):
                cm = Counter(mh)
                ax.plot(xs, [100 * cm.get(h, 0) / len(mh) for h in xs], ls=":", color=col[k % 4], lw=1.2, alpha=0.8, label=f"shortest seed$\\to$gold ({label.split(',')[0].split(' +')[0]})")
        ax.set_xticks(xs); ax.set_xlabel("hops in the top path"); ax.set_ylabel("share of golds (%)")
        ax.set_title(f"(B) path length, {s}-field", fontsize=8.5); ax.grid(True, ls="--", alpha=0.4)
        if si == 0: ax.legend(fontsize=6.5)

    # ---- (C) graph rank vs fused rank (first arm)
    md.append("\n## C. Graph-channel rank vs fused rank (first arm)\n")
    label, rows, _ = arms[0]
    for si, s in enumerate(strata):
        ax = axes[si, 2]
        sub = [r for r in rows if r["stratum"] == s]
        g = np.array([r["rank"]["graph"] for r in sub]); f = np.array([r["rank"]["fused"] for r in sub]); d = np.array([r["rank"]["dense"] for r in sub])
        ax.scatter(g, f, s=9, alpha=0.5, color="#1f77b4", label="gold")
        lim = max(g.max(), f.max()); ax.plot([1, lim], [1, lim], ls=":", color="grey", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("graph-channel rank of the gold"); ax.set_ylabel("fused rank")
        ax.set_title(f"(C) graph vs fused rank, {s}-field", fontsize=8.5); ax.grid(True, ls="--", alpha=0.4)
        lost = int(((g <= 5) & (f > 5)).sum()); resc = int(((g <= 5) & (d > 5)).sum()); kept = int(((g <= 5) & (f <= 5)).sum())
        md.append(f"{label} | {s} | golds {len(sub)} | graph<=5 {int((g<=5).sum())} | of which fused<=5 {kept}, fused>5 {lost} | graph<=5 while dense>5 {resc} | medians graph {int(np.median(g))} fused {int(np.median(f))} dense {int(np.median(d))}")
        ax.text(0.02, 0.96, f"graph$\\leq$5: {int((g<=5).sum())}, of which fused$>$5: {lost}", transform=ax.transAxes, fontsize=7, va="top")

    # ---- (D) necessity
    if have_nec:
        md.append("\n## D. Path necessity (rank of the gold after removing the top path's edges)\n")
        for si, s in enumerate(strata):
            ax = axes[si, 3]
            xt, xl = [], []; x = 0
            for k, (label, rows, _) in enumerate(arms):
                sub = [r for r in rows if r["stratum"] == s and r["necessity"] and r["hops"]]
                if not sub:
                    continue
                base = np.array([r["rank"]["graph"] for r in sub])
                series = []
                for key in ("top1", "top3", "random"):
                    after = np.array([r["necessity"][key].get("graph", np.nan) for r in sub], dtype=float)
                    ratio = after / base
                    series.append((key, np.nanmedian(ratio), float(np.nanmean(after > base)), np.nanmedian(after - base)))
                md.append(f"{label} | {s} | n={len(sub)} | " + " | ".join(f"{key}: median rank x{med:.2f}, worsened {100*fw:.0f}%, median drop {int(dd)}" for key, med, fw, dd in series))
                ax.bar([x, x + 1, x + 2], [se[2] * 100 for se in series], color=["#d62728", "#ff9896", "#bbbbbb"], width=0.8)
                for i_, se in enumerate(series):
                    ax.text(x + i_, se[2] * 100 + 1, f"x{se[1]:.1f}", ha="center", fontsize=6.5)
                xt += [x, x + 1, x + 2]; xl += [f"top-1\n{label[:14]}", "top-3", "random"]; x += 4
            ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.5); ax.set_ylim(0, 105)
            ax.set_ylabel("golds whose graph rank worsens (%)"); ax.set_title(f"(D) path necessity, {s}-field", fontsize=8.5)

    # ---- (E) CCMP selectivity
    if have_dist:
        md.append("\n## E. CCMP gate on gold-path hops vs distractor-path hops (first arm with gates)\n")
        for si, s in enumerate(strata):
            ax = axes[si, 3 + int(have_nec)]
            for label, rows, dist in arms:
                gg = [x for r in rows if r["stratum"] == s for x in r["gates"]]
                gd = [x for d in dist if d["stratum"] == s for x in d["gates"]]
                if not gg or not gd or st.pstdev(gg) == 0:
                    continue
                bins = np.linspace(min(gg + gd), max(gg + gd), 30)
                ax.hist(gg, bins=bins, alpha=0.55, color="#2ca02c", label=f"hops on paths to the gold (n={len(gg)})", density=True)
                ax.hist(gd, bins=bins, alpha=0.55, color="#d62728", label=f"hops on paths to the top wrong doc (n={len(gd)})", density=True)
                md.append(f"{label} | {s} | gold hops n={len(gg)} mean gate {st.mean(gg):.3f} >1: {100*np.mean(np.array(gg)>1.005):.0f}% | distractor hops n={len(gd)} mean gate {st.mean(gd):.3f} >1: {100*np.mean(np.array(gd)>1.005):.0f}%")
                break
            ax.axvline(1.0, color="k", lw=0.8, ls=":"); ax.set_xlabel("CCMP gate on the hop's sender (1 = frontier mean)"); ax.set_ylabel("density")
            ax.set_title(f"(E) CCMP selectivity, {s}-field", fontsize=8.5)
            if si == 0: ax.legend(fontsize=6.5)

    fig.tight_layout()
    fig.savefig(a.out + ".pdf", bbox_inches="tight"); fig.savefig(a.out + ".png", dpi=170, bbox_inches="tight")
    open(a.out + "_tables.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md)); print("wrote", a.out + ".pdf/.png/_tables.md")


if __name__ == "__main__":
    main()
