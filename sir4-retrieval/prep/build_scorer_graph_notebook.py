"""
build_scorer_graph_notebook.py -- one notebook, all four SIR-4 domains, the
scorer x graph 2x2.

    |                   | graph off | graph on |
    |-------------------|-----------|----------|
    | operator          | 5d current| section 9|
    | multi-view scorer | 5d mlp    | section 9b|

WHAT IT READS. Only score JSONs already on Drive, written by eval/score_sir4.py
during the per-domain fusion notebooks:

    {DRIVE}/outputs/sir4_<dom>/semantic/scores_semantic_current_<loss>.json
    {DRIVE}/outputs/sir4_<dom>/semantic/scores_semantic_mlp_<loss>.json
    {DRIVE}/outputs/sir4_<dom>/scores_fusion_qwenop_<epoch>.json
    {DRIVE}/outputs/sir4_<dom>/scores_fusion_qwenmlp_<epoch>.json

WHAT IT DOES NOT DO. No GPU, no encoder, no bundle unpack, no corpora, no
training, no rescoring. It cannot invent a missing arm and does not try; the
audit cell names what is absent and the table drops it.

WHY THE OPERATOR ROW IS 5d's `current` AND NOT cargo_operator.py. 5d refits the
operator's own formula on the same split under the same multi-gold-corrected
objective as the multi-view arm, so a column delta is attributable to the
architecture. Reading 5b instead would fold in a different loss and a different
fit slice.

A KNOWN GAP IN OLDER RUNS. score_sir4.py writes only the columns named by
--cols, and section 10 of build_notebook.py did not pass it until 2026-08-13.
Fusion score JSONs written before that carry mrr, ndcg@5, recall@3, recall@5 and
completeset@5 only. The audit reports each file's metric coverage and the table
intersects, so a stale file narrows the table instead of raising halfway down it.
Re-run section 10 to widen it.

Usage
-----
    python3 prep/build_scorer_graph_notebook.py
    python3 prep/build_scorer_graph_notebook.py --loss fixedloss --epoch-tag epoch20
"""
from __future__ import annotations

import argparse
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(True)}


CONFIG = '''from google.colab import drive
drive.mount("/content/drive")

import os, json, glob
DRIVE   = "/content/drive/MyDrive/cargo-gfmrag"
DOMAINS = ["cs", "biology", "physics", "matsci"]
LOSS    = "__LOSS__"        # the --loss the semantic arms were trained under
EPOCH   = "__EPOCH__"       # the fusion suffix, without the _b{BATCH} part
OUT     = f"{DRIVE}/outputs/_scorer_x_graph"
os.makedirs(OUT, exist_ok=True)

# The 2x2. Order matters: the table and the contrasts below both read it.
ARMS = ["operator", "multi-view", "operator+graph", "multi-view+graph"]

def score_paths(d):
    """Where score_sir4.py put each arm's JSON for one domain."""
    root = f"{DRIVE}/outputs/sir4_{d}"
    sem  = f"{root}/semantic"
    return {"operator":         f"{sem}/scores_semantic_current_{LOSS}.json",
            "multi-view":       f"{sem}/scores_semantic_mlp_{LOSS}.json",
            "operator+graph":   f"{root}/scores_fusion_qwenop_{EPOCH}.json",
            "multi-view+graph": f"{root}/scores_fusion_qwenmlp_{EPOCH}.json"}

def pred_globs(d):
    """Predictions, so the audit can say whether a rescore is even possible."""
    root = f"{DRIVE}/outputs/sir4_{d}"
    sem  = f"{root}/semantic"
    return {"operator":         [f"{sem}/predictions_semantic_current_{LOSS}_sir4_{d}_test.json"],
            "multi-view":       [f"{sem}/predictions_semantic_mlp_{LOSS}_sir4_{d}_test.json"],
            "operator+graph":   [f"{root}/sir4_{d}_fusion_qwenop_{EPOCH}*/predictions_*_test.json"],
            "multi-view+graph": [f"{root}/sir4_{d}_fusion_qwenmlp_{EPOCH}*/predictions_*_test.json"]}

print("DRIVE  ", DRIVE)
print("reading", f"{DRIVE}/outputs/sir4_<domain>/")
print("writing", OUT)
assert os.path.isdir(DRIVE), f"{DRIVE} not visible -- is the mount the right account?"'''


AUDIT = '''# ---- AUDIT: does every run actually exist on Drive? --------------------------
# Run this FIRST. It answers "are all four arms present for all four domains"
# before any table is built, so a hole shows up as a hole and not as a quietly
# shorter table.
WANT = ["mrr", "ndcg@5", "recall@3", "recall@5", "recall@10",
        "recall@25", "recall@100", "completeset@5"]
SLICES = ["all", "same", "cross", "similar", "dissimilar"]

found, cover, slices_seen, preds = {}, {}, {}, {}
for d in DOMAINS:
    sp, pg = score_paths(d), pred_globs(d)
    for a in ARMS:
        p = sp[a]
        if os.path.exists(p):
            try:
                j = json.load(open(p))
            except Exception as e:
                found[(d, a)] = f"CORRUPT ({type(e).__name__})"
                continue
            found[(d, a)] = "ok"
            ks = {k for v in j.values() if isinstance(v, dict) for k in v if k != "n"}
            cover[(d, a)] = ks
            slices_seen[(d, a)] = [s for s in SLICES if s in j]
        else:
            found[(d, a)] = "missing"
        preds[(d, a)] = any(glob.glob(g) for g in pg[a])

W = 18
print("SCORE FILES".ljust(W) + "".join(f"{d:>12}" for d in DOMAINS))
for a in ARMS:
    print(a.ljust(W) + "".join(f"{found[(d, a)]:>12}" for d in DOMAINS))

print("\\nPREDICTIONS (rescore possible?)".ljust(W) + "")
print("".ljust(W) + "".join(f"{d:>12}" for d in DOMAINS))
for a in ARMS:
    print(a.ljust(W) + "".join(f"{('yes' if preds[(d, a)] else 'no'):>12}" for d in DOMAINS))

print("\\nMETRIC COVERAGE  (of " + str(len(WANT)) + " wanted)")
print("".ljust(W) + "".join(f"{d:>12}" for d in DOMAINS))
for a in ARMS:
    row = ""
    for d in DOMAINS:
        ks = cover.get((d, a))
        row += f"{(str(len(ks & set(WANT))) + '/' + str(len(WANT)) if ks else '-'):>12}"
    print(a.ljust(W) + row)

print("\\nSLICES PRESENT")
print("".ljust(W) + "".join(f"{d:>12}" for d in DOMAINS))
for a in ARMS:
    print(a.ljust(W) + "".join(f"{len(slices_seen.get((d, a), [])):>12}" for d in DOMAINS))

# ---- verdict ----------------------------------------------------------------
gaps = [(d, a) for d in DOMAINS for a in ARMS if found[(d, a)] != "ok"]
if gaps:
    print(f"\\n{len(gaps)} of {len(DOMAINS) * len(ARMS)} cells absent:")
    for d, a in gaps:
        p = score_paths(d)[a]
        hint = ("run section 5d in that domain's notebook" if "graph" not in a
                else "run section 9" if a == "operator+graph" else "run section 9b")
        print(f"  {d:9} {a:18} {found[(d, a)]:8} -> {hint}")
        print(f"            expected {p}")
else:
    print("\\nall 16 cells present")

thin = [(d, a) for (d, a), ks in cover.items() if not set(WANT) <= ks]
if thin:
    miss = sorted({m for (d, a) in thin for m in set(WANT) - cover[(d, a)]})
    print(f"\\n{len(thin)} file(s) were scored with a narrower --cols; missing {miss}.")
    print("   Those metrics are dropped from the table below for EVERY arm, so the")
    print("   comparison stays like-for-like. Re-run section 10 (which now passes")
    print("   --cols) to widen it.")'''


TABLE = '''# ---- THE TABLE: four arms x four domains, one metric set --------------------
# Nothing is computed here. Every number was written by eval/score_sir4.py.
S = {}
for d in DOMAINS:
    for a, p in score_paths(d).items():
        if os.path.exists(p):
            try:
                S[(d, a)] = json.load(open(p))
            except Exception:
                pass
assert S, "no score files found -- run the audit cell and fix the gaps first"

# One metric set for the whole table, or a delta between two domains would be
# comparing different columns.
common = None
for v in S.values():
    ks = {k for x in v.values() if isinstance(x, dict) for k in x if k != "n"}
    common = ks if common is None else (common & ks)
ROWS = [m for m in WANT if m in common]
print(f"metrics: {', '.join(ROWS)}")
dropped = [m for m in WANT if m not in common]
if dropped:
    print(f"dropped (absent from at least one file): {', '.join(dropped)}")

CONTRASTS = [("operator",       "operator+graph",   "graph adds, operator"),
             ("multi-view",     "multi-view+graph", "graph adds, multi-view"),
             ("operator",       "multi-view",       "scorer swap, no graph"),
             ("operator+graph", "multi-view+graph", "scorer swap, with graph")]

lines = [f"# SIR-4 scorer x graph   (loss={LOSS}, fusion={EPOCH})"]
for sl in SLICES:
    doms = [d for d in DOMAINS if all((d, a) in S and sl in S[(d, a)] for a in ARMS)]
    if not doms:
        continue
    lines.append(f"\\n\\n## [{sl}]   complete in: {', '.join(doms)}")
    for d in doms:
        lines.append(f"\\n{d}   n={S[(d, ARMS[0])][sl]['n']}")
        lines.append(f"  {'metric':16}" + "".join(f"{a:>19}" for a in ARMS))
        for m in ROWS:
            lines.append(f"  {m:16}" + "".join(f"{S[(d, a)][sl][m]:>19.4f}" for a in ARMS))
        lines.append(f"  {'':16}" + "".join(f"{'':>19}" for a in ARMS))
        lines.append(f"  {'contrast':26}" + "".join(f"{m:>14}" for m in ROWS))
        for lo, hi, tag in CONTRASTS:
            lines.append(f"  {tag:26}"
                         + "".join(f"{S[(d, hi)][sl][m] - S[(d, lo)][sl][m]:>+14.4f}"
                                   for m in ROWS))
report = "\\n".join(lines)
print(report)

dst = f"{OUT}/scorer_x_graph_all_domains.md"
open(dst, "w").write(report + "\\n")
print(f"\\n\\nwritten to {dst}")'''


SUMMARY = '''# ---- HEADLINE: mean over the domains where all four arms exist ---------------
# A mean over an incomplete set of domains is not a result, so the domains are
# named and an arm missing anywhere is excluded rather than averaged over a
# different denominator than its neighbours.
for sl in SLICES:
    doms = [d for d in DOMAINS if all((d, a) in S and sl in S[(d, a)] for a in ARMS)]
    if len(doms) < 2:
        continue
    print(f"\\n[{sl}]  mean over {len(doms)} domains: {', '.join(doms)}")
    print(f"  {'arm':26}" + "".join(f"{m:>14}" for m in ROWS))
    mean = {}
    for a in ARMS:
        mean[a] = {m: sum(S[(d, a)][sl][m] for d in doms) / len(doms) for m in ROWS}
        print(f"  {a:26}" + "".join(f"{mean[a][m]:>14.4f}" for m in ROWS))
    print(f"  {'':26}" + "".join(f"{'':>14}" for m in ROWS))
    for lo, hi, tag in CONTRASTS:
        print(f"  {tag:26}"
              + "".join(f"{mean[hi][m] - mean[lo][m]:>+14.4f}" for m in ROWS))
    # per-domain win counts, so one strong domain cannot carry a mean
    for lo, hi, tag in CONTRASTS:
        w = sum(S[(d, hi)][sl]["ndcg@5"] > S[(d, lo)][sl]["ndcg@5"] for d in doms)
        print(f"    {tag:24} wins nDCG@5 in {w}/{len(doms)} domains")'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="fixedloss",
                    help="tag on the semantic score files; 'operatorloss' for --loss operator")
    ap.add_argument("--epoch-tag", default="epoch20",
                    help="fusion suffix without the _b{BATCH} part")
    ap.add_argument("--out", default=f"{ROOT}/colab_scorer_x_graph.ipynb")
    a = ap.parse_args()

    cells = [
        md("""# SIR-4: scorer x graph, all four domains, one table

Four arms per domain, from artefacts already on Drive:

| | graph off | graph on |
|---|---|---|
| **operator** | 5d `current` | section 9 |
| **multi-view scorer** | 5d `mlp` | section 9b |

Rows give what the graph adds with the scorer held fixed. Columns give what the scorer
swap buys with the graph half held fixed. Both are printed as explicit deltas.

**No GPU, no encoder, no bundle, no corpora, no training.** This notebook only reads
score JSONs that `eval/score_sir4.py` already wrote. It cannot reconstruct a missing arm
and does not pretend to.

**Run cell 2 (audit) before cell 3.** It reports which of the 16 (domain x arm) cells
exist, whether predictions are present so a rescore is possible, and each file's metric
coverage. Older fusion score files were written before section 10 passed `--cols`, so
they carry five metrics rather than eight; the table intersects across every file so the
comparison stays like-for-like."""),
        code(CONFIG.replace("__LOSS__", a.loss).replace("__EPOCH__", a.epoch_tag)),
        md("## 1. Audit — do all the runs exist?"),
        code(AUDIT),
        md("## 2. The 2x2, per domain"),
        code(TABLE),
        md("## 3. Headline — averaged over the domains that are complete"),
        code(SUMMARY),
    ]

    broken = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        try:
            ast.parse("".join(c["source"]))
        except SyntaxError as e:
            broken.append((i, e.lineno, e.msg))
    if broken:
        for i, l, m in broken:
            print(f"  cell {i} line {l}: {m}")
        raise SystemExit(f"{len(broken)} cell(s) do not parse; {a.out} not written")

    json.dump({"cells": cells,
               "metadata": {"kernelspec": {"display_name": "Python 3",
                                           "language": "python", "name": "python3"},
                            "language_info": {"name": "python"}},
               "nbformat": 4, "nbformat_minor": 0}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({len(cells)} cells, loss={a.loss}, fusion={a.epoch_tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
