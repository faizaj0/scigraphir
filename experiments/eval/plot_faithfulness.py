#!/usr/bin/env python3
"""
plot_faithfulness.py -- three-panel figure on the reasoner's paths, from interpret_paths.py
outputs produced with +interp.necessity=1 +interp.distractor=1:

  (a) route composition: share of top paths (same-field vs cross-field golds) that pass through
      each affordance representation type on the SciAfford graph, and the mean path length; OpenIE routes are untyped
      entity chains and are reported in the printed table.
  (b) path necessity: the gold's graph-channel rank before and after removing the edges of its
      top-1 / top-3 paths, against removing the same number of random edges (affordance representation vs OpenIE).
  (c) CCMP selectivity: the gate applied on hops of paths to the gold vs hops of paths to the
      query's top-ranked wrong document (same model, same queries).

usage:
  plot_faithfulness.py --sciafford hops_frame_on.json [--sciafford-off hops_frame_off.json]
                       --openie hops_openie.json --docs documents.json --out fig_faith_cs.pdf
"""
import argparse, json, os, re
from collections import Counter, defaultdict

import numpy as np

MECH = {"function", "limitation", "method"}
CATS = ["mechanism", "task", "finding", "domain hub", "bridge paper", "untyped entity"]


def node_type(name, docs):
    if name in docs:
        return "paper"
    m = re.match(r"\[([a-z_ ]+)\]", name)
    return m.group(1).strip() if m else "entity"


def path_cats(path, docs):
    """Categories touched by a path, excluding its final node (the target)."""
    nodes = [h["head"] for h in path["hops"]] + [path["hops"][-1]["tail"]]
    inner = nodes[:-1]
    ty = [node_type(n, docs) for n in inner]
    out = set()
    for t in ty:
        if t in MECH: out.add("mechanism")
        elif t == "task": out.add("task")
        elif t == "finding": out.add("finding")
        elif t == "domain": out.add("domain hub")
        elif t == "paper": out.add("bridge paper")
        else: out.add("untyped entity")
    return out, ty


def load(path):
    return json.load(open(path)) if path and os.path.exists(path) else []


def rows_of(recs):
    for r in recs:
        st = "cross" if r.get("stratum") == "cross" else "same"
        for t in r.get("targets", []):
            yield r, st, t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sciafford", "--frame", dest="frame", required=True); ap.add_argument("--sciafford-off", "--frame-off", dest="frame_off", default=None)
    ap.add_argument("--openie", required=True); ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    docs = json.load(open(a.docs))
    F, FO, O = load(a.frame), load(a.frame_off), load(a.openie)
    summary = {}

    # ---------- (a) route composition
    comp = {st: Counter() for st in ("same", "cross")}; n_paths = Counter(); hops = defaultdict(list)
    rel_frame, rel_openie = Counter(), Counter()
    for r, st, t in rows_of(F):
        if not t["paths"]: continue
        cats, ty = path_cats(t["paths"][0], docs); n_paths[st] += 1; hops[("frame", st)].append(len(t["paths"][0]["hops"]))
        for c in cats: comp[st][c] += 1
        for h in t["paths"][0]["hops"]: rel_frame[h["rel"].replace("inverse_", "")] += 1
    for r, st, t in rows_of(O):
        if not t["paths"]: continue
        hops[("openie", st)].append(len(t["paths"][0]["hops"]))
        for h in t["paths"][0]["hops"]: rel_openie[h["rel"].replace("inverse_", "")] += 1
    summary["composition"] = {st: {c: comp[st][c] / max(n_paths[st], 1) for c in CATS} for st in comp}
    summary["paths_counted"] = dict(n_paths)
    summary["mean_hops"] = {f"{g}/{st}": float(np.mean(v)) for (g, st), v in hops.items() if v}
    summary["reach"] = {}
    for name, recs in (("frame", F), ("openie", O)):
        for st in ("same", "cross"):
            tt = [t for r, s, t in rows_of(recs) if s == st]
            summary["reach"][f"{name}/{st}"] = sum(1 for t in tt if t["paths"]) / max(len(tt), 1)
    summary["top_relations_frame"] = rel_frame.most_common(8); summary["top_relations_openie"] = rel_openie.most_common(8)

    # ---------- (b) necessity
    nec = {}
    for name, recs in (("SciAfford graph", F), ("OpenIE graph", O)):
        before, top1, top3, rnd, keep5 = [], [], [], [], Counter()
        for r, st, t in rows_of(recs):
            n = t.get("necessity")
            if not n or "graph" not in n.get("top1", {}): continue
            b = t["rank"]["graph"]; before.append(b); top1.append(n["top1"]["graph"]); top3.append(n["top3"]["graph"]); rnd.append(n["random"]["graph"])
            for k, v in (("before", b), ("top1", n["top1"]["graph"]), ("top3", n["top3"]["graph"]), ("random", n["random"]["graph"])):
                keep5[k] += int(v <= 5)
        if before:
            nec[name] = {"n": len(before), "median": {"before": float(np.median(before)), "top1": float(np.median(top1)),
                                                      "top3": float(np.median(top3)), "random": float(np.median(rnd))},
                         "share_top5": {k: keep5[k] / len(before) for k in ("before", "top1", "top3", "random")},
                         "median_drop_top1": float(np.median(np.array(top1) - np.array(before))),
                         "share_dropped_top1": float(np.mean(np.array(top1) > np.array(before)))}
    summary["necessity"] = nec

    # ---------- (c) CCMP selectivity
    gold_g, dis_g, gold_max, dis_max = [], [], [], []
    for r in F:
        for t in r.get("targets", []):
            for p in t["paths"][:1]:
                g = [h["gate"] for h in p["hops"] if "gate" in h]
                gold_g += g
                if g: gold_max.append(max(g))
        d = r.get("distractor")
        if d:
            for p in d["paths"][:1]:
                g = [h["gate"] for h in p["hops"] if "gate" in h]
                dis_g += g
                if g: dis_max.append(max(g))
    def gstats(v):
        v = np.array(v); return {"n_hops": int(v.size), "mean": float(v.mean()) if v.size else None,
                                 "share_gt1": float((v > 1.005).mean()) if v.size else None, "share_lt1": float((v < 0.995).mean()) if v.size else None}
    summary["ccmp"] = {"gold_hops": gstats(gold_g), "distractor_hops": gstats(dis_g),
                       "gold_path_max_gate_mean": float(np.mean(gold_max)) if gold_max else None,
                       "distractor_path_max_gate_mean": float(np.mean(dis_max)) if dis_max else None}
    if gold_max and dis_max:
        summary["ccmp"]["share_gold_path_max_gate_gt_distractor"] = float(np.mean(
            [gm > dm for gm, dm in zip(gold_max, dis_max)])) if len(gold_max) == len(dis_max) else None

    # ---------- figure
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    ax = axes[0]; x = np.arange(len(CATS)); w = 0.38
    for i, (st, col) in enumerate((("same", "#7f7f7f"), ("cross", "#1f77b4"))):
        vals = [100 * summary["composition"][st][c] for c in CATS]
        ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=f"{'same-field' if st == 'same' else 'cross-field'} golds (n={n_paths[st]})")
    ax.set_xticks(x); ax.set_xticklabels(CATS, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% of top paths passing through"); ax.set_title("(a) what the SciAfford graph routes go through", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right"); ax.grid(True, axis="y", ls="--", alpha=0.4)
    mh = summary["mean_hops"]
    ax.set_xlabel('mean hops, SciAfford graph: ' + ", ".join(f"{st} {mh.get(f'frame/{st}', float('nan')):.2f}" for st in ("same", "cross"))
                  + ";  OpenIE graph: " + ", ".join(f"{st} {mh.get(f'openie/{st}', float('nan')):.2f}" for st in ("same", "cross")), fontsize=7.5)

    ax = axes[1]; keys = ["before", "top1", "top3", "random"]; labels = ["as is", "top-1 path\nremoved", "top-3 paths\nremoved", "same # random\nedges removed"]
    x = np.arange(len(keys)); w = 0.38
    for i, (name, col) in enumerate((("SciAfford graph", "#1f77b4"), ("OpenIE graph", "#ff7f0e"))):
        if name not in nec: continue
        vals = [nec[name]["median"][k] for k in keys]
        ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=f"{name} (n={nec[name]['n']})")
        for xi, k in zip(x, keys):
            ax.text(xi + (i - 0.5) * w, nec[name]["median"][k] * 1.08, f"{100 * nec[name]['share_top5'][k]:.0f}%", ha="center", fontsize=6.5, color=col)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("median graph-channel rank of the gold (log)"); ax.set_title("(b) path necessity: rank after removing the explained path", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.text(0.02, 0.97, "labels: share of golds still in the top 5", transform=ax.transAxes, va="top", fontsize=7)

    ax = axes[2]
    if gold_g and dis_g:
        lo = min(min(gold_g), min(dis_g)); hi = max(max(gold_g), max(dis_g)); bins = np.linspace(min(lo, 0.85), max(hi, 1.15), 31)
        ax.hist(gold_g, bins=bins, alpha=0.6, color="#1f77b4", label=f"paths to the gold ({len(gold_g)} hops)", density=True)
        ax.hist(dis_g, bins=bins, alpha=0.6, color="#d62728", label=f"paths to the top wrong paper ({len(dis_g)} hops)", density=True)
        ax.axvline(1.0, color="k", lw=0.8, ls=":")
        c = summary["ccmp"]
        ax.text(0.02, 0.97, f"mean gate  gold {c['gold_hops']['mean']:.3f}  wrong {c['distractor_hops']['mean']:.3f}\n"
                            f"share > 1  gold {100 * c['gold_hops']['share_gt1']:.0f}%  wrong {100 * c['distractor_hops']['share_gt1']:.0f}%",
                transform=ax.transAxes, va="top", fontsize=7.5)
    ax.set_xlabel("CCMP gate on the hop's sender (1 = frontier mean)"); ax.set_ylabel("density")
    ax.set_title("(c) CCMP selectivity: gate on gold routes vs wrong routes", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right"); ax.grid(True, ls="--", alpha=0.4)
    if a.title: fig.suptitle(a.title, fontsize=10)
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight")
    base = os.path.splitext(a.out)[0]; fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    json.dump(summary, open(base + "_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1)); print("wrote", a.out, base + ".png", base + "_summary.json")


if __name__ == "__main__":
    main()
