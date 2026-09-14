#!/usr/bin/env python3
"""
gate_decomp_fig.py -- where does the CCMP gate act? Reads gate_decomp_<dataset>.json written by
trainer.gate_decomposition() (colab_ccmp_gate_decomp.ipynb) and draws the decomposition.

For every (query, gold) the file holds the top routes under the full gate and, for six inference-time
gating conditions on the SAME trained weights, the weight of those same routes evaluated directly along
their edges, the gold's graph score and its graph / fused rank:

    gate_on                       the model as run
    gate_off                      every gate 1 (the "no CCMP" arm)
    gate_layers_attributed        gate only at the layers the route is attributed to (0 .. L-1)
    gate_layers_later             gate only at the later layers (L .. 5)
    gate_route_senders_only       gate only on the route's own sender nodes, 1 everywhere else
    gate_all_but_route_senders    gate on every node except the route's senders

Outputs (prefix --out):
    <out>.pdf/.png   panel A: one example (--example gold id, default the first target): route-1
                     weight and graph rank under each condition.
                     panel B: over all golds, median ratio of route-1 weight to the gate-off weight
                     per condition, with the interquartile range, and the share of golds whose graph
                     rank improves over gate-off.
    <out>.md         the same numbers as a table, plus the example's routes with their gates.

usage: gate_decomp_fig.py --decomp outputs/scan/sir4_biology/gate_decomp_sir4_biology.json \\
           --docs kg-construction/data/sir4_biology_test/raw/documents.json \\
           [--example 10.1007/s11263-023-01831-9] --out results/qualitative/fig_gate_decomp_biology
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st

CONDS = [("gate_off", "gate off\n(no CCMP)"), ("gate_layers_attributed", "gate only at\nroute layers\n0..L-1"),
         ("gate_layers_later", "gate only at\nlater layers\nL..5"), ("gate_route_senders_only", "gate only on\nroute\nsenders"),
         ("gate_all_but_route_senders", "gate only on\nother\nnodes"), ("gate_on", "gate on\n(full CCMP)")]


def title_of(doc_id, docs, n=70):
    t = docs.get(doc_id, doc_id).split(". ")[0]
    return t if len(t) <= n else t[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True); ap.add_argument("--docs", required=True)
    ap.add_argument("--example", default=None, help="gold id of the example for panel A (default: first target)")
    ap.add_argument("--route", type=int, default=0, help="which baseline route to follow (0 = top)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    docs = json.load(open(a.docs))
    data = json.load(open(a.decomp))
    targets = [(r, t) for r in data for t in r["targets"]]
    assert targets, "no targets in the decomposition file"

    # ---- aggregate over golds: ratio of the followed route's weight to its gate-off weight, per condition
    agg = {c: [] for c, _ in CONDS}; rank_better = {c: 0 for c, _ in CONDS}; rank_n = 0
    for r, t in targets:
        C = t["conditions"]
        w_off = C["gate_off"]["route_weights_direct"]
        if len(w_off) <= a.route or w_off[a.route] is None or w_off[a.route] <= 0:
            continue
        rank_n += 1
        for c, _ in CONDS:
            w = C[c]["route_weights_direct"][a.route]
            if w is not None and w > 0:
                agg[c].append(w / w_off[a.route])
            if C[c]["rank"]["graph"] < C["gate_off"]["rank"]["graph"]:
                rank_better[c] += 1

    # ---- the example
    ex = next(((r, t) for r, t in targets if t["doc"] == a.example), targets[0]) if a.example else targets[0]
    r_ex, t_ex = ex; C = t_ex["conditions"]
    route = t_ex["routes"][a.route] if len(t_ex["routes"]) > a.route else None

    # ---- markdown
    md = [f"# CCMP gate decomposition: {a.decomp}", "",
          f"{len(targets)} golds; {rank_n} with a positive gate-off weight on route {a.route + 1}. Same trained weights in every row; only the inference-time gate mask changes.", "",
          "| condition | median w / w_off | IQR | share of golds with a better graph rank than gate off |", "|---|--:|---|--:|"]
    for c, lab in CONDS:
        v = agg[c]
        if v:
            q1, q3 = st.quantiles(v, n=4)[0], st.quantiles(v, n=4)[2]
            md.append(f"| {lab.replace(chr(10), ' ')} | {st.median(v):.3f} | {q1:.2f} to {q3:.2f} | {rank_better[c] / max(1, rank_n):.2f} |")
    md += ["", f"## Example: {r_ex['id']} -> {title_of(t_ex['doc'], docs)}", "",
           f"attributed layers 0..{t_ex['attributed_layers'] - 1}; {t_ex['n_route_senders']} route senders; frontier-mean responsibility per layer "
           + ", ".join(f"{x:.3f}" for x in t_ex["frontier_mean_resp"]), ""]
    if route:
        md.append("route " + str(a.route + 1) + ": " + " -> ".join(f"{h['head']} [{h['rel']}] (g {h.get('gate', 1):.4f}, resp {h.get('resp', float('nan')):.3f})" for h in route["hops"]) + f" -> gold; beam weight {route['weight_beam']:.2f}")
    md += ["", "| condition | route weight | graph rank | fused rank | gold graph score | gap to the top graph score |", "|---|--:|--:|--:|--:|--:|"]
    for c, lab in CONDS:
        w = C[c]["route_weights_direct"][a.route] if len(C[c]["route_weights_direct"]) > a.route else None
        md.append(f"| {lab.replace(chr(10), ' ')} | {'n/a' if w is None else f'{w:.2f}'} | {C[c]['rank']['graph']} | {C[c]['rank']['fused']} | {C[c]['graph_score']:.3f} | {C[c]['graph_score_gap_to_top']:.3f} |")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.05, 1]})
    labs = [lab for _, lab in CONDS]; xs = list(range(len(CONDS)))
    # A: example
    ax = axes[0]
    w = [C[c]["route_weights_direct"][a.route] if len(C[c]["route_weights_direct"]) > a.route else None for c, _ in CONDS]
    rk = [C[c]["rank"]["graph"] for c, _ in CONDS]
    cols = ["#7f7f7f", "#9ecae1", "#3182bd", "#a1d99b", "#31a354", "#e6550d"]
    ax.bar(xs, [0 if v is None else v for v in w], color=cols, edgecolor="black", linewidth=0.5)
    for x, v, k in zip(xs, w, rk):
        ax.text(x, (0 if v is None else v) * 1.01, f"rank {k}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7); ax.set_ylabel(f"weight of route {a.route + 1} (same route, same weights)", fontsize=8)
    ax.set_title(f"A. {title_of(t_ex['doc'], docs, 60)}", fontsize=9, loc="left")
    ax.grid(axis="y", ls="--", alpha=0.4)
    # B: aggregate
    ax = axes[1]
    med = [st.median(agg[c]) if agg[c] else float("nan") for c, _ in CONDS]
    lo = [st.quantiles(agg[c], n=4)[0] if len(agg[c]) > 3 else float("nan") for c, _ in CONDS]
    hi = [st.quantiles(agg[c], n=4)[2] if len(agg[c]) > 3 else float("nan") for c, _ in CONDS]
    ax.bar(xs, med, color=cols, edgecolor="black", linewidth=0.5,
           yerr=[[m - l for m, l in zip(med, lo)], [h - m for m, h in zip(med, hi)]], capsize=3)
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    for x, c in zip(xs, [c for c, _ in CONDS]):
        ax.text(x, 0.03, f"{rank_better[c] / max(1, rank_n):.0%}\nbetter\nrank", ha="center", va="bottom", fontsize=7, color="white" if x != 0 else "black")
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7); ax.set_ylabel("route-1 weight / gate-off weight (median, IQR)", fontsize=8)
    ax.set_title(f"B. all {rank_n} golds", fontsize=9, loc="left"); ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight")
    # ---- panel C (separate file): gate attribution along the example's route, per sender and layer
    ga = t_ex.get("gate_attribution") or {}
    if route and "route_senders" in ga:
        senders = [h["head"] for h in route["hops"]]
        fig2, ax = plt.subplots(figsize=(6.5, 3.2))
        nL = len(next(iter(ga["route_senders"].values()))["contrib"])
        width = 0.8 / max(1, len(senders))
        for si, sname in enumerate(senders):
            rec = ga["route_senders"].get(sname)
            if not rec: continue
            xs2 = [l + (si - (len(senders) - 1) / 2) * width for l in range(nL)]
            ax.bar(xs2, rec["contrib"], width=width * 0.95, label=f"{sname[:38]} (g per layer: " + ", ".join(f"{g:.2f}" for g in rec["gate"]) + ")")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(nL)); ax.set_xticklabels([f"layer {l}" for l in range(nL)])
        ax.set_ylabel("CCMP contribution to the gold's score\n∂s/∂g × (g − 1)", fontsize=8)
        ax.set_title(f"C. where CCMP acts on route {a.route + 1}: sum over route senders {ga.get('sum_contrib_route_senders', float('nan')):.3f}, "
                     f"all nodes {ga.get('sum_contrib_all_nodes', float('nan')):.3f}, actual Δscore on−off {ga.get('delta_score_on_minus_off', float('nan')):.3f}", fontsize=8, loc="left")
        ax.legend(fontsize=6.5, loc="best"); ax.grid(axis="y", ls="--", alpha=0.4)
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(f"{a.out}_attr.{ext}", dpi=200, bbox_inches="tight")
        md.append("\n## Gate attribution (route " + str(a.route + 1) + ")\n")
        md.append("| sender | " + " | ".join(f"layer {l}" for l in range(nL)) + " | total |")
        md.append("|---|" + "--:|" * (nL + 1))
        for sname in senders:
            rec = ga["route_senders"].get(sname)
            if rec: md.append(f"| {sname[:50]} | " + " | ".join(f"{c:+.3f} (g {g:.2f})" for c, g in zip(rec["contrib"], rec["gate"])) + f" | {rec['contrib_total']:+.3f} |")
        md.append(f"\nsum over all nodes {ga.get('sum_contrib_all_nodes', 0):+.3f} vs actual Δscore (gate on − off) {ga.get('delta_score_on_minus_off', 0):+.3f}; per-layer sum over all nodes: " + ", ".join(f"{v:+.3f}" for v in ga.get("per_layer_sum_all", [])))
        md.append("\ntop nodes by |contribution| (on route marked *): " + "; ".join(f"{'*' if n['on_route'] else ''}{n['node'][:40]} {n['contrib_total']:+.3f}" for n in ga.get("top_nodes", [])[:10]))
        open(a.out + ".md", "w").write("\n".join(md) + "\n")
    if False:
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight")
    print("\n".join(md[:12])); print(f"\nwrote {a.out}.md/.pdf/.png")


if __name__ == "__main__":
    main()
