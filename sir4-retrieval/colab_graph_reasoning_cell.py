# ===== Graph-reasoning routes for EVERY dataset with hops files on Drive (paste into any Colab kernel; needs Drive + the unpacked bundle) =====
# Per dataset, line graphs: (a) P(graph rank <= 10) against the length of the top route, cross vs same, SciAffordGraph vs OpenIE entity graph;
# (b) the node type reached at each hop; (c) route weight and success against the largest sender degree on the route (hub inflation);
# (d) the CCMP gate against the sender's degree. Reads outputs/scan/<dataset>/hops_frame_ccmp.json (+ hops_openie.json); degrees come from
# the frame graph's processed/stage1/edges.csv in the unpacked bundle (the build whose targets carry "[domain]" nodes).
import os, sys, glob, json, subprocess
if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive; drive.mount("/content/drive")
DRIVE = globals().get("DRIVE", "/content/drive/MyDrive/cargo-gfmrag"); SCAN = f"{DRIVE}/outputs/scan"
_root = globals().get("CARGO_ROOT", "/content/cargo"); DATA = globals().get("DATA_ROOT", f"{_root}/kg-construction/data")
os.makedirs("/content/eval", exist_ok=True)
open("/content/eval/graph_reasoning_fig.py", "w").write(r'''#!/usr/bin/env python3
"""
graph_reasoning_fig.py -- what the graph reasoner's routes look like, over every gold of a dataset's hops files.
Line graphs; no gold reasoning paths exist, so routes are characterised against the gold's graph-channel rank and
against the structure of the graph (node degree).

  (a) success against route length: P(graph rank <= 10 | length of the top route), cross- and same-field golds,
      SciAffordGraph vs the OpenIE entity graph                             -> the reach of the reasoner is a distance effect
  (b) anatomy of the route: the node type reached at each hop position (share of routes), cross-field golds
  (c) hub inflation: median weight of the top route and P(graph rank <= 10) against the largest sender degree on the route
  (d) the CCMP gate against the sender's degree (mean, 95% bootstrap CI): does credit go to specific senders or to hubs?

Inputs: hops_frame_ccmp.json (+ hops_openie.json) from the showcase notebooks; --edges = the frame graph's
processed/stage1/edges.csv (for degrees); QUARTET eval.json gives the per-gold stratum for SIR-4 (else the query stratum).
Routes that do not start at a seed, do not chain, or revisit a node are decoding artefacts and are skipped.

    python3 eval/graph_reasoning_fig.py --dir results/qualitative/drive_scan_sir4_cs --docs ../kg-construction/data/sir4_cs_test/raw/documents.json \
        --edges ../kg-construction/data/sir4_cs_test_v16sc/processed/stage1/edges.csv \
        --quartet ../quartet/data.nosync/benchmark/cs_test_final/eval.json --out ../figures/fig_graph_reasoning_cs
"""
import argparse, collections, csv, json, math, os, random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, RULE = "#262a30", "#7c828c", "#ced2da"
C_CROSS, C_SAME, C_ENT, C_W, C_P, C_GATE = "#2a69a0", "#9aa0a8", "#7c828c", "#2a69a0", "#208070", "#d9701a"
TYPE_C = {"method": "#2a69a0", "paper": "#c44e52", "domain": "#6e747e", "limitation": "#705296", "task": "#9b86b8", "function": "#208070", "finding": "#a47822", "gold": "#262a30"}


def ntype(n, docs):
    if n in docs: return "paper"
    return n[1:n.index("]")] if n.startswith("[") and "]" in n else "entity"


def valid(p, seeds, gold, max_hops):
    h = p["hops"]
    if not h or h[0]["head"] not in seeds or not all(x["tail"] == y["head"] for x, y in zip(h, h[1:])): return False
    ns = [h[0]["head"]] + [x["tail"] for x in h]; return len(ns) == len(set(ns)) and h[-1]["tail"] == gold and len(h) <= max_hops


def load(path, max_hops):
    out = {}
    for r in json.load(open(path)):
        seeds = set(r["seeds"])
        for t in r["targets"]:
            ps = [p for p in t.get("paths", []) if valid(p, seeds, t["doc"], max_hops)]
            out[(r["id"], t["doc"])] = {"stratum": r.get("stratum") or "unlabelled", "rank": t.get("rank") or {}, "path": ps[0] if ps else None}
    return out


def boot_mean(v, B=1000, seed=0):
    rnd = random.Random(seed); n = len(v); ms = []
    for _ in range(B): ms.append(sum(rnd.choice(v) for _ in range(n)) / n)
    ms.sort(); return ms[int(0.025 * B)], ms[int(0.975 * B)]


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(RULE); ax.spines["bottom"].set_color(RULE); ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--prefix", default="hops_"); ap.add_argument("--docs", required=True)
    ap.add_argument("--edges", default=None, help="frame graph processed/stage1/edges.csv (degrees for panels c, d)")
    ap.add_argument("--quartet", default=None); ap.add_argument("--out", required=True); ap.add_argument("--title", default=""); ap.add_argument("--max-hops", type=int, default=6); ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args(); K = a.k
    docs = json.load(open(a.docs))
    F = load(f"{a.dir}/{a.prefix}frame_ccmp.json", a.max_hops)
    E = load(f"{a.dir}/{a.prefix}openie.json", a.max_hops) if os.path.exists(f"{a.dir}/{a.prefix}openie.json") else {}
    deg = collections.Counter()
    if a.edges and os.path.exists(a.edges):
        with open(a.edges) as f:
            for row in csv.DictReader(f): deg[row["source"]] += 1; deg[row["target"]] += 1
    unit = "query stratum (hops file)"
    if a.quartet and os.path.exists(a.quartet):
        qz = {}
        for x in json.load(open(a.quartet)):
            for d, m in (x.get("quartet", {}).get("per_document") or {}).items(): qz[(x["id"], d)] = m.get("stratum")
        for k, v in F.items(): v["stratum"] = qz.get(k) or "unlabelled"
        unit = "gold stratum (QUARTET)"
    strata = [s for s in ("cross", "same") if any(v["stratum"] == s for v in F.values())] or ["all"]
    if strata == ["all"]:
        for v in F.values(): v["stratum"] = "all"
    keys = {s: [k for k, v in F.items() if v["stratum"] == s and v["rank"].get("graph")] for s in strata}
    s0 = strata[0]; summary = {"unit": unit, "n": {s: len(keys[s]) for s in strata}, "k": K}
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), gridspec_kw={"wspace": 0.38, "hspace": 0.62})
    axes = axes.ravel()
    for ax in axes: style(ax)
    # ---- (a) success against route length
    ax = axes[0]; hs = list(range(1, a.max_hops + 1))
    for s, col, lab in ((s0, C_CROSS, f"{s0}-field golds"), (strata[1] if len(strata) > 1 else None, C_SAME, "same-field golds")):
        if not s: continue
        ys, lo, hi, ns = [], [], [], []
        for h in hs:
            ks = [k for k in keys[s] if F[k]["path"] and len(F[k]["path"]["hops"]) == h]
            ok = sum(F[k]["rank"]["graph"] <= K for k in ks); ns.append(len(ks))
            ys.append(ok / len(ks) if ks else float("nan")); l, u = wilson(ok, len(ks)); lo.append(l); hi.append(u)
        ax.fill_between(hs, lo, hi, color=col, alpha=0.12, lw=0); ax.plot(hs, ys, marker="o", ms=4, lw=1.5, color=col, label=f"SciAffordGraph, {lab}")
        for h, n, y in zip(hs, ns, ys):
            if n and s == s0: ax.text(h, -0.09, f"n={n}", fontsize=5.6, color=MUTED, ha="center")
        summary[f"a:{s}"] = {str(h): {"n": n, "p": y} for h, n, y in zip(hs, ns, ys)}
    if E:
        ys = []
        for h in hs:
            ks = [k for k in keys[s0] if k in E and E[k]["path"] and len(E[k]["path"]["hops"]) == h]
            ys.append(sum(E[k]["rank"]["graph"] <= K for k in ks) / len(ks) if ks else float("nan"))
        ax.plot(hs, ys, marker="s", ms=3.5, lw=1.2, ls="--", color=C_ENT, label=f"OpenIE entity graph, {s0}-field golds"); summary["a:entity"] = dict(zip(map(str, hs), ys))
    nor = sum(1 for k in keys[s0] if not F[k]["path"]); summary["a:no_route"] = nor
    ax.set_xticks(hs); ax.set_ylim(-0.02, 1.05); ax.set_xlabel(f"length of the top route (hops)      no route within {a.max_hops} hops: {nor} {s0}-field golds", fontsize=7.6, color=INK)
    ax.set_ylabel(f"P(graph-channel rank $\\leq$ {K})", fontsize=8, color=INK); ax.legend(fontsize=6.4, frameon=False, loc="upper right")
    ax.set_title("(a) reach falls with route length, not with the field boundary", fontsize=8, color=INK, loc="left")
    # ---- (b) anatomy of the route: node type reached at each hop, cross-field golds
    ax = axes[1]; pos = collections.defaultdict(collections.Counter)
    for k in keys[s0]:
        p = F[k]["path"]
        if not p: continue
        for i, h in enumerate(p["hops"]): pos[i + 1]["gold" if h["tail"] == k[1] else ntype(h["tail"], docs)] += 1
    hs2 = sorted(pos); tot = {h: sum(pos[h].values()) for h in hs2}
    for t, col in TYPE_C.items():
        ys = [100.0 * pos[h][t] / tot[h] for h in hs2]
        if max(ys) < 4: continue
        ax.plot(hs2, ys, marker="o", ms=3.5, lw=1.4, color=col, label=("the gold $i^{\\star}$" if t == "gold" else t), ls="-" if t != "gold" else "-.")
        summary[f"b:{t}"] = dict(zip(map(str, hs2), ys))
    ax.set_xticks(hs2); ax.set_xticklabels([f"hop {h}\nn={tot[h]}" for h in hs2], fontsize=6.4); ax.set_ylabel(f"% of routes ({s0}-field golds)", fontsize=8, color=INK)
    ax.set_xlabel("position along the top route", fontsize=7.6, color=INK); ax.legend(fontsize=6.2, frameon=False, ncol=2, loc="upper right", columnspacing=0.8)
    ax.set_title("(b) what the route reaches at each hop", fontsize=8, color=INK, loc="left")
    # ---- (c) hub inflation: weight and success against the largest sender degree on the route
    if deg:
        ax = axes[2]; bins = collections.defaultdict(list)
        for k in keys[s0]:
            p = F[k]["path"]
            if not p: continue
            m = max(deg.get(h["head"], 1) for h in p["hops"]); bins[round(math.log10(max(1, m)) * 2) / 2].append((p["weight"], F[k]["rank"]["graph"]))
        xs = sorted(b for b in bins if len(bins[b]) >= 8)
        medw = [sorted(w for w, _ in bins[b])[len(bins[b]) // 2] for b in xs]; pk = [sum(r <= K for _, r in bins[b]) / len(bins[b]) for b in xs]
        ax.plot(xs, medw, marker="o", ms=4, lw=1.5, color=C_W, label="median weight of the top route (left)")
        ax.set_ylabel("route weight", fontsize=8, color=C_W); ax.tick_params(axis="y", colors=C_W)
        ax2 = ax.twinx(); ax2.plot(xs, pk, marker="s", ms=3.5, lw=1.5, color=C_P, ls="--", label=f"P(graph rank $\\leq$ {K}) (right)")
        ax2.set_ylim(0, 1.05); ax2.set_ylabel(f"P(graph-channel rank $\\leq$ {K})", fontsize=8, color=C_P); ax2.tick_params(axis="y", colors=C_P, labelsize=7.5)
        for sp in ("top",): ax2.spines[sp].set_visible(False)
        ax2.spines["right"].set_color(RULE); ax2.spines["left"].set_color(RULE)
        ax.set_xticks(xs); ax.set_xticklabels([f"$10^{{{b:g}}}$\nn={len(bins[b])}" for b in xs], fontsize=6.4)
        ax.set_xlabel(f"largest sender degree on the route ({s0}-field golds)", fontsize=7.6, color=INK)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=6.2, frameon=False, loc="upper left")
        ax.set_title("(c) hub inflation: heavy routes through hubs do not rank the gold", fontsize=8, color=INK, loc="left")
        summary["c"] = {f"{b:g}": {"n": len(bins[b]), "median_weight": w, "p": p_} for b, w, p_ in zip(xs, medw, pk)}
        # ---- (d) the gate against the sender's degree, all hops of the top routes
        ax = axes[3]; gb = collections.defaultdict(list)
        for s in strata:
            for k in keys[s]:
                p = F[k]["path"]
                if not p: continue
                for h in p["hops"]:
                    g = h.get("gate"); d = deg.get(h["head"], 0)
                    if g is not None and d > 0: gb[round(math.log10(d) * 2) / 2].append(g)
        xs = sorted(b for b in gb if len(gb[b]) >= 20)
        if xs:
            mean = [sum(gb[b]) / len(gb[b]) for b in xs]; ci = [boot_mean(gb[b]) for b in xs]
            ax.fill_between(xs, [c[0] for c in ci], [c[1] for c in ci], color=C_GATE, alpha=0.15, lw=0)
            ax.plot(xs, mean, marker="o", ms=4, lw=1.5, color=C_GATE, label="mean gate (95% CI)")
            ax.axhline(1.0, color=MUTED, lw=0.8, ls=":"); ax.text(xs[0], 1.003, "1 = frontier mean", fontsize=6.2, color=MUTED, va="bottom")
            ax3 = ax.twinx(); amp = [100.0 * sum(g > 1.05 for g in gb[b]) / len(gb[b]) for b in xs]
            ax3.plot(xs, amp, marker="^", ms=3.5, lw=1.2, ls="--", color=MUTED, label="share of hops amplified ($g > 1.05$, right)")
            ax3.set_ylim(0, 100); ax3.set_ylabel("% of hops amplified", fontsize=8, color=MUTED); ax3.tick_params(axis="y", colors=MUTED, labelsize=7.5)
            ax3.spines["top"].set_visible(False); ax3.spines["right"].set_color(RULE)
            ax.set_xticks(xs); ax.set_xticklabels([f"$10^{{{b:g}}}$\nn={len(gb[b])}" for b in xs], fontsize=6.4)
            ax.set_xlabel("degree of the hop's sender (all golds, all hops of the top route)", fontsize=7.6, color=INK)
            ax.set_ylabel("CCMP gate on the sender", fontsize=8, color=C_GATE); ax.tick_params(axis="y", colors=C_GATE)
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=6.2, frameon=False, loc="upper left")
            ax.set_title("(d) the CCMP gate rises with the sender's degree", fontsize=8, color=INK, loc="left")
            summary["d"] = {f"{b:g}": {"n": len(gb[b]), "mean": m, "ci": c, "amplified": a_} for b, m, c, a_ in zip(xs, mean, ci, amp)}
    else:
        for ax in axes[2:]: ax.text(0.5, 0.5, "no edges.csv: degrees unavailable", ha="center", va="center", fontsize=8, color=MUTED, transform=ax.transAxes); ax.set_axis_off()
    if a.title: fig.suptitle(a.title, fontsize=9, color=INK, x=0.02, y=0.98, ha="left")
    fig.savefig(a.out + ".pdf", bbox_inches="tight"); fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight")
    json.dump(summary, open(a.out + "_summary.json", "w"), indent=1)
    print("wrote", a.out + ".png"); print(json.dumps(summary, indent=1)[:5000])


if __name__ == "__main__":
    main()
''')
def frame_edges(ds):
    """the frame-graph build for this dataset: an edges.csv whose second line has a typed '[...]' target"""
    for p in sorted(glob.glob(f"{DATA}/{ds}_test*/processed/stage1/edges.csv")):
        with open(p) as f: f.readline(); ln = f.readline()
        if ",[" in ln: return p
    return None
from IPython.display import Image, display
for d in sorted(glob.glob(f"{SCAN}/*/")):
    ds = os.path.basename(d.rstrip("/"))
    if not os.path.exists(f"{d}/hops_frame_ccmp.json"): continue
    docs = f"{DATA}/{ds}_test/raw/documents.json"
    if not os.path.exists(docs): print(ds, ": documents.json not found (unpack the bundle first)"); continue
    cmd = [sys.executable, "/content/eval/graph_reasoning_fig.py", "--dir", d, "--docs", docs, "--out", f"{d}/fig_graph_reasoning_{ds}", "--title", ds]
    e = frame_edges(ds)
    if e: cmd += ["--edges", e]
    else: print(ds, ": no frame-graph edges.csv found; panels (c) and (d) will be empty")
    _qt = f"{_root}/quartet/data.nosync/benchmark/{ds.replace('sir4_', '')}_test_final/eval.json"
    if os.path.exists(_qt): cmd += ["--quartet", _qt]
    r = subprocess.run(cmd, capture_output=True, text=True); print("=" * 30, ds); print(r.stdout[-2500:], r.stderr[-1500:])
    if os.path.exists(f"{d}/fig_graph_reasoning_{ds}.png"): display(Image(f"{d}/fig_graph_reasoning_{ds}.png"))
