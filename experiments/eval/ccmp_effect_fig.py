#!/usr/bin/env python3
"""
ccmp_effect_fig.py -- what the CCMP gate does, measured without gold reasoning paths.

We have no ground-truth paths (GFM-RAG Fig. 6 compares predicted hops against gold hops; SIR-4 has none), but we do
have gold documents. So the effect of CCMP is measured as a PAIRED comparison on the same weights: the rank of every
gold document with the gate on vs off, plus the gate values the model assigned along its own top routes.

    (a) graph-channel rank of every gold, gate off (x) vs with CCMP (y), log-log; below the diagonal = CCMP helped
    (b) distribution of the rank change (log10 on/off) for the graph channel and the fused score, cross- vs same-field
    (c) the gate along the top route to each gold: by the type of node the hop enters, and last hop vs inner hops

Inputs are the hops files written by the showcase notebook (hops_frame_ccmp.json, hops_frame_ccmp_off.json), which
carry rank = {fused, graph, scorer, dense} per gold and the gate per hop of every recorded path; the QUARTET eval.json
gives the stratum (cross / same). Run:

    python3 eval/ccmp_effect_fig.py --dir results/qualitative/drive_scan_sir4_cs --docs ../retriever/data/sir4_cs_test/raw/documents.json \
        --quartet ../benchmark/data.nosync/benchmark/cs_test_final/eval.json --out ../figures/fig_ccmp_effect_cs
"""
import argparse, collections, json, math, os, random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED, RULE = "#262a30", "#7c828c", "#ced2da"
C_CROSS, C_SAME, C_GATE, C_UP, C_DOWN = "#2a69a0", "#9aa0a8", "#d9701a", "#208070", "#c44e52"
TYPES = ["paper", "method", "domain"]          # node types the hops enter, most to least frequent


def ntype(n, docs):
    if n in docs: return "paper"
    return n[1:n.index("]")] if n.startswith("[") and "]" in n else "entity"


def load(path):
    return {(r["id"], t["doc"]): t for r in json.load(open(path)) for t in r["targets"]}


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot_mean(v, B=2000, seed=0):
    rnd = random.Random(seed); n = len(v); ms = []
    for _ in range(B): ms.append(sum(rnd.choice(v) for _ in range(n)) / n)
    ms.sort(); return ms[int(0.025 * B)], ms[int(0.975 * B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--prefix", default="hops_"); ap.add_argument("--docs", required=True)
    ap.add_argument("--quartet", default=None); ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
    a = ap.parse_args()
    on = load(f"{a.dir}/{a.prefix}frame_ccmp.json"); off = load(f"{a.dir}/{a.prefix}frame_ccmp_off.json")
    docs = json.load(open(a.docs))
    strat = {}
    if a.quartet and os.path.exists(a.quartet):
        for x in json.load(open(a.quartet)):
            for d, m in (x.get("quartet", {}).get("per_document") or {}).items(): strat[(x["id"], d)] = m.get("stratum")
    keys = [k for k in on if k in off and on[k]["rank"].get("graph") and off[k]["rank"].get("graph")]
    grp = {"cross": [k for k in keys if strat.get(k) == "cross"], "same": [k for k in keys if strat.get(k) == "same"]}
    if not grp["cross"] and not grp["same"]: grp = {"all": keys}
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0], "wspace": 0.42})
    for ax in axes:
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(RULE); ax.spines["bottom"].set_color(RULE); ax.tick_params(colors=MUTED, labelsize=8)
    # ---- (a) paired scatter, graph channel
    ax = axes[0]; lim = (0.8, 6000)
    ax.plot(lim, lim, color=RULE, lw=0.8, zorder=1)
    for name, col, z in (("same", C_SAME, 2), ("cross", C_CROSS, 3), ("all", C_CROSS, 3)):
        ks = grp.get(name, [])
        if not ks: continue
        ax.scatter([off[k]["rank"]["graph"] for k in ks], [on[k]["rank"]["graph"] for k in ks], s=9, color=col, alpha=0.55 if name == "same" else 0.75, lw=0, zorder=z)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("graph-channel rank of the gold, gate off", fontsize=8.5, color=INK); ax.set_ylabel("rank with CCMP", fontsize=8.5, color=INK)
    ax.text(1.0, 3500, "CCMP hurts", fontsize=7.5, color=MUTED, ha="left", va="top"); ax.text(3500, 1.0, "CCMP helps", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    handles = []
    for name, col in (("cross", C_CROSS), ("same", C_SAME), ("all", C_CROSS)):
        ks = grp.get(name, [])
        if not ks: continue
        b = sum(on[k]["rank"]["graph"] < off[k]["rank"]["graph"] for k in ks); w = sum(on[k]["rank"]["graph"] > off[k]["rank"]["graph"] for k in ks)
        summary[f"graph_{name}"] = {"n": len(ks), "better": b, "worse": w, "same": len(ks) - b - w}
        handles.append(Line2D([], [], marker="o", ls="", color=col, ms=4, label=f"{name}-field golds (n={len(ks)}): {b} better, {w} worse, {len(ks)-b-w} same"))
    ax.legend(handles=handles, fontsize=6.6, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a) graph channel, same weights", fontsize=9, color=INK, loc="left", pad=30)
    # ---- (b) ECDF of the rank change
    ax = axes[1]
    for ch, ls in (("graph", "-"), ("fused", "--")):
        for name, col in (("cross", C_CROSS), ("same", C_SAME), ("all", C_CROSS)):
            ks = [k for k in grp.get(name, []) if on[k]["rank"].get(ch) and off[k]["rank"].get(ch)]
            if not ks: continue
            d = sorted(math.log10(on[k]["rank"][ch]) - math.log10(off[k]["rank"][ch]) for k in ks)
            ax.step(d, [(i + 1) / len(d) for i in range(len(d))], where="post", color=col, ls=ls, lw=1.3)
            med = d[len(d) // 2]; lo, hi = boot_mean(d)
            summary[f"dlog_{ch}_{name}"] = {"n": len(d), "median": med, "mean": sum(d) / len(d), "mean_ci95": [lo, hi]}
    ax.axvline(0, color=RULE, lw=0.8); ax.set_xlim(-2.2, 2.2); ax.set_ylim(0, 1)
    ax.set_xlabel("log$_{10}$ rank with CCMP $-$ log$_{10}$ rank gate off", fontsize=8.5, color=INK); ax.set_ylabel("fraction of golds", fontsize=8.5, color=INK)
    ax.text(-2.1, 0.93, "$\\leftarrow$ CCMP helps", fontsize=7.5, color=MUTED, ha="left"); ax.text(2.1, 0.93, "CCMP hurts $\\rightarrow$", fontsize=7.5, color=MUTED, ha="right")
    hb = [Line2D([], [], color=INK, ls="-", lw=1.3, label="graph channel"), Line2D([], [], color=INK, ls="--", lw=1.3, label="fused score")]
    if grp.get("cross"): hb += [Line2D([], [], color=C_CROSS, lw=3, label="cross-field"), Line2D([], [], color=C_SAME, lw=3, label="same-field")]
    ax.legend(handles=hb, fontsize=6.6, frameon=False, loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.6, bbox_to_anchor=(1.02, 0.02))
    ax.set_title("(b) rank change per gold", fontsize=9, color=INK, loc="left", pad=30)
    # ---- (c) the gate along the top route: by the node type the hop enters, and last hop vs inner
    ax = axes[2]
    bytype = collections.defaultdict(list); pos = {"inner\nhops": [], "last hop\ninto the gold": []}
    for k in keys:
        ps = on[k]["paths"]
        if not ps: continue
        hops = ps[0]["hops"]
        for i, h in enumerate(hops):
            g = h.get("gate")
            if g is None: continue
            bytype[ntype(h["tail"], docs)].append(g); pos["last hop\ninto the gold" if i == len(hops) - 1 else "inner\nhops"].append(g)
    cats = [t for t in TYPES if len(bytype.get(t, [])) >= 20] + list(pos)
    vals = [bytype[t] if t in bytype else pos[t] for t in cats]
    xs = list(range(len(cats))); ticks = []
    for x, v, c in zip(xs, vals, cats):
        m = sum(v) / len(v); lo, hi = boot_mean(v); amp = sum(g > 1.05 for g in v) / len(v); damp = sum(g < 0.95 for g in v) / len(v)
        col = C_GATE if m > 1.0 else C_CROSS
        ax.bar(x, m - 1.0, bottom=1.0, width=0.62, color=col, alpha=0.85, lw=0)
        ax.plot([x, x], [lo, hi], color=INK, lw=0.9)
        ax.text(x, 1.068, f"{amp:.0%}$\\uparrow$\n{damp:.0%}$\\downarrow$", fontsize=6.4, color=MUTED, ha="center", va="bottom", linespacing=1.1)
        ticks.append((c if "\n" in c else f"into a\n{c}") + f"\nn={len(v)}")
        summary[f"gate_{c.replace(chr(10), ' ')}"] = {"n": len(v), "mean": m, "ci95": [lo, hi], "frac_amplified": amp, "frac_damped": damp}
    ax.axhline(1.0, color=RULE, lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(ticks, fontsize=7, color=INK)
    ax.set_ylim(0.965, 1.085); ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.text(-0.5, 1.084, "share of hops with gate $>1.05$ / $<0.95$:", fontsize=6.4, color=MUTED, ha="left", va="top")
    ax.set_ylabel("CCMP gate on the sender (1 = frontier mean)", fontsize=8.5, color=INK)
    ax.set_title("(c) where the gate acts on the top route", fontsize=9, color=INK, loc="left", pad=30)
    if a.title: fig.suptitle(a.title, fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.savefig(a.out + ".pdf", bbox_inches="tight"); fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight")
    json.dump(summary, open(a.out + "_summary.json", "w"), indent=1)
    print("wrote", a.out + ".pdf", a.out + ".png", a.out + "_summary.json")
    for k, v in summary.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
