"""
graph_channel_scores.py -- the graph channel's OWN scores for fusion runs, from their console logs.

The trainer reports the fused metric; the per-epoch component diagnostics patched into
sft_training (section 3d of every fusion notebook) print, at each evaluation, the three
channels' nDCG@5, the parameter-free walk prior, the graph's recall@10 bucketed by the
gold's semantic rank, and the graph@1 hit rate. This reads those lines from the FINAL
evaluation (best epoch) of each run and prints them side by side, optionally next to the
scored fused metrics from score_sir4.py.

Same parser as colab_routing_<dataset>.ipynb section 6 (final_diag), so the columns match.

Usage
-----
    # local copies of the run dirs (or the Drive mount on Colab)
    python3 eval/graph_channel_scores.py <run_dir> [<run_dir> ...]
    python3 eval/graph_channel_scores.py --root /content/drive/MyDrive/cargo-gfmrag/outputs/mir \\
        mir_qwenmlp_graph_e10_b2 mir_qwenmlp_ccmp_e10_b2 mir_openie_qwenmlp_graph_e10_b2
    # add --scores <dir> to pull the fused nDCG@5 / R@5 from <dir>/<key>_scores.json
    #   (key = --key-map run=key, e.g. mir_qwenmlp_ccmp_e10_b2=scigraphir_ccmp)
    # --md writes a markdown table next to printing it
"""
from __future__ import annotations

import argparse
import json
import os
import re


def final_diag(log_path: str) -> dict:
    """[diag]/[prior]/[bucket]/[hit] lines from the FINAL evaluation (best epoch), plus per-epoch series."""
    log = open(log_path, errors="ignore").read()
    tail = log[log.rfind("Running final evaluation"):] if "Running final evaluation" in log else log
    d: dict = {}
    m = re.search(r"\[diag\] nDCG@5\s+semantic ([\d.]+) \| graph ([\d.]+) \| fused ([\d.]+)\s+gamma mean ([\d.]+)", tail)
    if m:
        d.update(sem=float(m[1]), graph=float(m[2]), fused_diag=float(m[3]), gamma=float(m[4]))
    m = re.search(r"\[prior\] parameter-free walk nDCG@5 ([\d.]+)\s+trained graph ([\d.]+)", tail)
    if m:
        d.update(walk=float(m[1]))
    m = re.search(r"\[bucket\][^\n]*", tail)
    if m:
        d["bucket_line"] = m[0].strip()
        for name, val, n in re.findall(r"(1-10|11-100|101-1k|1k\+) ([\d.]+) \(n=(\d+)\)", m[0]):
            d[f"bucket_{name}"] = float(val)
            d[f"bucket_{name}_n"] = int(n)
        if "bucket_1k+" in d:
            d.update(buried=d["bucket_1k+"], buried_n=d["bucket_1k+_n"])
    m = re.search(r"\[hit\] gold rate by rank -- graph 1..3: ([\d.]+)", tail)
    if m:
        d.update(graph_at1=float(m[1]))
    best = re.findall(r"New best model! document_ndcg@5: ([\d.]+) at epoch (\d+)", log)
    if best:
        d.update(best_epoch=int(best[-1][1]), best_fused=float(best[-1][0]))
    # per-epoch series of the three channels (the final evaluation repeats the line; exclude it)
    head = log[:log.rfind("Running final evaluation")] if "Running final evaluation" in log else log
    d["epochs"] = [(float(a), float(b), float(c)) for a, b, c, _ in re.findall(
        r"\[diag\] nDCG@5\s+semantic ([\d.]+) \| graph ([\d.]+) \| fused ([\d.]+)\s+gamma mean ([\d.]+)", head)]
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories (each holding console.log), or names under --root")
    ap.add_argument("--root", default="", help="prefix for the run names")
    ap.add_argument("--scores", default="", help="scores dir with <key>_scores.json from score_sir4.py")
    ap.add_argument("--key-map", default="", help="comma list run_name=scores_key (default: run name)")
    ap.add_argument("--slice", default="all")
    ap.add_argument("--md", default="", help="write the table here as markdown")
    a = ap.parse_args()
    kmap = dict(kv.split("=", 1) for kv in a.key_map.split(",") if "=" in kv)

    rows = []
    for r in a.runs:
        rd = os.path.join(a.root, r) if a.root else r
        log = os.path.join(rd, "console.log")
        if not os.path.exists(log):
            print(f"[skip] {rd}: no console.log"); continue
        d = final_diag(log)
        fused = None
        if a.scores:
            sp = os.path.join(a.scores, f"{kmap.get(os.path.basename(r), os.path.basename(r))}_scores.json")
            if os.path.exists(sp):
                fused = json.load(open(sp)).get(a.slice)
        rows.append((os.path.basename(r.rstrip("/")), d, fused))

    def pc(x, k=1):
        return "--" if x is None else (f"{100 * x:.2f}" if k else f"{x:.3f}")
    hdr = ["run", "fused nDCG@5", "fused R@5", "scorer nDCG@5", "graph nDCG@5", "walk nDCG@5", "graph - walk",
           "graph R@10 buried (1k+)", "graph@1", "gamma", "best ep"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for name, d, fused in rows:
        fn = fused["ndcg@5"] if fused else d.get("best_fused", d.get("fused_diag"))
        fr = fused["recall@5"] if fused else None
        gw = d["graph"] - d["walk"] if "graph" in d and "walk" in d else None
        lines.append(f"| {name} | {pc(fn)} | {pc(fr)} | {pc(d.get('sem'))} | {pc(d.get('graph'))} | {pc(d.get('walk'))} | "
                     + ("--" if gw is None else f"{100 * gw:+.2f}")
                     + f" | {pc(d.get('buried'))}" + (f" (n={d['buried_n']})" if "buried_n" in d else "")
                     + f" | {pc(d.get('graph_at1'))} | {pc(d.get('gamma'), 0)} | {d.get('best_epoch', '--')} |")
    lines.append("")
    lines.append("fused = the ranking the model outputs (scorer + gated graph); from score_sir4 when --scores is given, else the "
                 "trainer's best-epoch value. scorer/graph = each channel ranked ALONE at the best epoch. walk = a "
                 "zero-parameter personalised PageRank from the same seeds, the graph channel's prior. buried = the graph's "
                 "recall@10 on golds the scorer ranked past 1,000. All percentages.")
    for name, d, _ in rows:
        if d.get("bucket_line"):
            lines.append(f"\n{name}: {d['bucket_line']}")
        if d.get("epochs"):
            lines.append(f"{name} per-epoch (scorer / graph / fused nDCG@5): "
                         + "  ".join(f"ep{i + 1} {100 * s:.1f}/{100 * g:.1f}/{100 * f:.1f}" for i, (s, g, f) in enumerate(d["epochs"])))
    text = "\n".join(lines)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n"); print("\nwrote", a.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
