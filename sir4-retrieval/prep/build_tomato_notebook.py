"""
build_tomato_notebook.py -- generate the TOMATO-Star architecture comparison notebook.

WHY THIS IS A SEPARATE BUILDER. build_notebook.py emits the SIR-4 graph-training
notebook, whose cell 2 asserts that no dataset name contains "tomato". That guard
exists for a real reason (TOMATO caches silently contaminating SIR-4 runs) and is
not something to relax. This notebook shares nothing with it: no graph, no
reasoner, no fusion, no Drive, no bundle.

IT RUNS LOCALLY. Everything TOMATO needs is already on this laptop and cached:

    corpus      kg-construction/data/tomato_{train,test}/raw/
    answers     kg-construction/construct_v2/cache/probes_{train,test}.jsonl
    embeddings  outputs/caches/op_emb/{train,test}_{doc,query,probe}*.npy   (BGE)
    baseline    kg-construction/eval/predictions_bge.json

so the run needs no encoder, no API key and no GPU. Colab would mean uploading
~470 MB of probe embeddings to compute something that takes minutes here.

TOMATO DIFFERS FROM SIR-4 IN THREE WAYS THAT CHANGE THE READING:

  1. One gold per row, and every row id is unique (9,669 train / 3,132 test).
     So `--loss operator` and `--loss fixed` are ARITHMETICALLY IDENTICAL here:
     with a single gold there are no sibling golds to exclude from the
     denominator, and averaging within a query is averaging over one item. The
     loss flag is a no-op on this corpus; the notebook states that rather than
     running both and reporting a spurious tie.

  2. The same QUESTION TEXT repeats across rows with different golds (3,132 rows
     cover 1,658 distinct questions, 1.89 golds each). Scoring per row is the
     single-gold convention every published TOMATO number in this project uses.
     Pooling by question roughly doubles the absolute values and leaves the
     ordering unchanged, so it is the wrong thing to change midway through a
     comparison.

  3. No sets.json, so CompleteSet@k is undefined and omitted from the columns
     rather than reported as a column of zeros.

    python3 prep/build_tomato_notebook.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = f"{ROOT}/tomato_architecture_comparison.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

cells.append(md("""# TOMATO-Star — which semantic scorer architecture wins?

Runs locally. No Colab, no Drive, no bundle, no encoder: TOMATO's BGE embeddings
are already cached on this machine, so this is arithmetic on cached matrices.

**The question.** Given the same frozen encoder, the same hypothetical answers and
the same training split, does any learned combiner beat the handcrafted
`sum` + `max` scorer?

| arm | what it is | params |
|---|---|--:|
| `dense` | `cos(E(q), E(d))` alone, untrained — the floor | 0 |
| `current` | handcrafted `sum` + `max`, three global weights — the target | 4 |
| `attention` | softmax over the direct view and every answer, weighted average | 34 |
| `deepsets` | `φ([x_hyp, x_dir]) → masked mean → ρ([x_dir, u])`, 15% answer dropout | 62 |
| `mlp` | one MLP on `[x_dir, sorted answers, count]` — sorting replaces summarising | ~200 |

**Three things about TOMATO that differ from SIR-4:**

1. **One gold per row**, unique ids, 9,669 train / 3,132 test. So the corrected
   multi-gold loss and the operator loss are *arithmetically identical* here —
   there are no sibling golds to exclude. Only one is run.
2. The same question text repeats across rows with different golds (3,132 rows,
   1,658 distinct questions). Scoring per row is the single-gold convention every
   published TOMATO number in this project uses.
3. No `sets.json`, so CompleteSet@k is undefined and left out of the columns.

**Fit set.** `--train_fit 0` uses every train query after the dev slice: 9,369 of
9,669. The old 2,500 cap came from fitting four parameters and would discard 73%
of the data here, which is precisely wrong for arms that overfit."""))

cells.append(md("## 1. Paths and a check that nothing needs encoding"))
cells.append(code('''import os, sys, json, subprocess, time

CARGO = (os.environ.get("CARGO_ROOT") or os.path.expanduser("~/Desktop/CARGO"))
S4    = f"{CARGO}/sir4-retrieval"
OUT   = f"{S4}/results/semantic_tomato"
os.makedirs(OUT, exist_ok=True)
os.environ["CARGO_ROOT"] = CARGO
os.environ["CARGO_DATASET"] = "tomato"          # legacy unscoped cache paths

# The BGE baseline defines the similar/dissimilar slices: a query is `dissimilar`
# when plain dense ranks its gold below 100. Without it those two rows are simply
# absent, which is the interesting half of TOMATO, so check for it up front.
QS  = f"{CARGO}/kg-construction/data/tomato_test/raw/test.json"
BGE = f"{CARGO}/kg-construction/eval/predictions_bge.json"

need = {
    "test corpus":  f"{CARGO}/kg-construction/data/tomato_test/raw/documents.json",
    "test queries": QS,
    "train corpus": f"{CARGO}/kg-construction/data/tomato_train/raw/documents.json",
    "train queries": f"{CARGO}/kg-construction/data/tomato_train/raw/train.json",
    "test answers":  f"{CARGO}/kg-construction/construct_v2/cache/probes_test.jsonl",
    "train answers": f"{CARGO}/kg-construction/construct_v2/cache/probes_train.jsonl",
    "BGE baseline":  BGE,
}
miss = [k for k, v in need.items() if not os.path.exists(v)]
assert not miss, f"missing: {miss}"
for k, v in need.items():
    print(f"  ok  {k:14} {v.replace(CARGO, '<cargo>')}")

# EMBEDDINGS MUST ALL BE CACHED. If any are absent the scorer would silently load
# a 1.3 GB encoder and start encoding ~85k texts on CPU, which is the difference
# between three minutes and an hour. Fail here instead.
sys.path.insert(0, CARGO)
import importlib.util, hashlib
spec = importlib.util.spec_from_file_location("op", f"{CARGO}/kg-construction/eval/cargo_operator.py")
op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)
MODEL = "BAAI/bge-large-en-v1.5"
slug, qi = op.model_slug(MODEL), op.query_instruction(MODEL)
E = f"{CARGO}/outputs/caches/op_emb"
for split in ("train", "test"):
    rows = json.load(open(f"{CARGO}/kg-construction/data/tomato_{split}/raw/{split}.json"))
    qk = hashlib.md5("|".join(r["id"] for r in rows).encode()).hexdigest()[:8]
    for f in (f"{split}_doc{slug}.npy",
              f"{split}_query_{qk}{slug}{op.qi_tag(qi)}.npy",
              f"{split}_probe_{qk}{slug}.npy"):
        assert os.path.exists(f"{E}/{f}"), (
            f"missing embedding {f}. Running anyway would encode from scratch.")
        print(f"  ok  cached        {f}")
    ng = [len(r.get("supporting_documents") or []) for r in rows]
    print(f"  {split}: {len(rows):,} rows, {len(set(r['id'] for r in rows)):,} unique ids, "
          f"{sum(ng)/len(ng):.2f} golds/row")'''))

cells.append(md("## 2. Run every arm\n\n"
                "Five arms x three seeds. The first call builds the match matrices "
                "(`H` is 9,669 x 8 x 7,000 float16 on train, about 1 GB on disk, "
                "memory-mapped) and caches them, so re-runs skip straight to training."))
cells.append(code('''def sh(cmd, cwd):
    """Run a command, streaming its output live."""
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, shell=True, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        chunk = os.read(p.stdout.fileno(), 8192)
        if not chunk:
            break
        sys.stdout.write(chunk.decode("utf-8", "replace")); sys.stdout.flush()
    p.wait()
    print(f"\\n[{time.time() - t0:.0f}s, exit {p.returncode}]")
    assert p.returncode == 0, f"FAILED: {cmd}"

# --loss is a no-op on TOMATO (one gold per row), so only one setting is run.
# --train_fit 0 keeps every train query after the 300-query dev slice.
sh("python3 -u eval/semantic_scorer.py --dataset tomato "
   f"--model {MODEL} --arms dense,current,mlp --loss fixed "
   "--train_fit 0 --dev 300 --epochs 30 --patience 5 "
   "--lr 3e-4 --weight_decay 1e-2 --hidden 8 --ds_hidden 4 --ds_dropout 0.15 "
   f"--qbatch 8 --seed 0 --seeds 0,1,2 --out {OUT}", S4)'''))

cells.append(md("## 3. Score every arm with the benchmark scorer\n\n"
                "`CompleteSet@k` is omitted: TOMATO carries no `sets.json`, so it "
                "is undefined rather than zero."))
cells.append(code('''COLS = "mrr,ndcg@5,recall@3,recall@5,recall@10,recall@25,recall@100"
ARMS = (("dense",   "dense query-document only", "dense"),
        ("current", "current scorer",            "current_fixedloss"),
        ("mlp",     "sorted-MLP scorer",         "mlp_fixedloss"))

for arm, label, tag in ARMS:
    pred = f"{OUT}/predictions_semantic_{tag}_tomato_test.json"
    assert os.path.exists(pred), f"missing {pred} -- did cell 2 finish?"
    sh(f"python3 -u eval/score_sir4.py --pred {pred} --queries {QS} --bge {BGE} "
       f"--cols {COLS} --name '{label}' "
       f"--json-out {OUT}/scores_semantic_{tag}.json "
       f"--per-query-out {OUT}/perquery_{tag}.json", S4)'''))

cells.append(md("## 4. The table\n\n"
                "`current` is the reference column. `dense` shows what the "
                "hypothetical answers are worth at all; every learned arm is read "
                "against `current`."))
cells.append(code('''import json
SC   = {a: json.load(open(f"{OUT}/scores_semantic_{t}.json")) for a, _, t in ARMS}
META = json.load(open(f"{OUT}/semantic_comparison_tomato_fixedloss.json"))
ROWS = ["mrr", "ndcg@5", "recall@3", "recall@5", "recall@10", "recall@25", "recall@100"]
BASE = "current"
OTHER = [a for a, _, _ in ARMS if a != BASE]

sp = META["actual_split"]
print(f"### TOMATO-Star test — semantic scorer architecture comparison")
print(f"fit={sp['fit']:,} dev={sp['dev']} of {sp['train_queries']:,} train rows; "
      f"test={sp['test_queries']:,}   encoder={META['encoder']}")
for arm, v in META["quick_summary"].items():
    ps = META["arms"][arm].get("per_seed_dev_ndcg@10") or {}
    print(f"  {arm:12} best dev nDCG@10 {v['dev_ndcg@10']:.4f}"
          + (f"   per-seed {ps}" if ps else ""))

lines = []
for sl in ("all", "same", "cross", "similar", "dissimilar"):
    if any(sl not in SC[a] for a in SC):
        continue
    lines.append(f"\\n[{sl}]  n={SC[BASE][sl]['n']}")
    lines.append(f"{'metric':12}{BASE:>10}" + "".join(f"{a:>11}{'delta':>9}" for a in OTHER))
    for m in ROWS:
        if any(m not in SC[a][sl] for a in SC):
            continue
        row = f"{m:12}{SC[BASE][sl][m]:10.4f}"
        for a in OTHER:
            v = SC[a][sl][m]
            row += f"{v:11.4f}{v - SC[BASE][sl][m]:+9.4f}"
        lines.append(row)
table = "\\n".join(lines)
print(table)

print("\\n--- all-slice verdict vs current ---")
for a in OTHER:
    d = {m: SC[a]["all"][m] - SC[BASE]["all"][m] for m in ROWS}
    print(f"  {a:10} wins {sum(v > 0 for v in d.values())}/{len(d)} metrics; "
          f"mrr {d['mrr']:+.4f}  ndcg@5 {d['ndcg@5']:+.4f}  recall@10 {d['recall@10']:+.4f}")

open(f"{OUT}/tomato_architecture_comparison.md", "w").write(
    f"# TOMATO-Star: semantic scorer architecture comparison\\n\\n"
    f"fit={sp['fit']} dev={sp['dev']} test={sp['test_queries']} "
    f"encoder={META['encoder']}\\n\\n```{table}\\n```\\n")
print(f"\\nwrote {OUT}/tomato_architecture_comparison.md")'''))

cells.append(md("""## 6. Is any difference real?

The deltas above are means over 3,132 rows. Two means cannot be compared by
eye when the gaps are a point or less. Every arm ranked the **same** queries over
the **same** corpus, so the comparison is paired and a paired bootstrap is far
more sensitive than an unpaired test would be.

`--per-query-out` was written in cell 3 for exactly this."""))
cells.append(code('''BOOT = f"{S4}/transfer/paired_bootstrap.py"
if os.path.exists(BOOT):
    for arm, _, tag in ARMS:
        if arm == "current":
            continue
        sh(f"python3 -u transfer/paired_bootstrap.py "
           f"--a {OUT}/perquery_current_fixedloss.json "
           f"--b {OUT}/perquery_{tag}.json "
           f"--a-name current --b-name {arm} --metrics mrr,ndcg@5,recall@10", S4)
else:
    print("paired_bootstrap.py not found; per-query files are written and ready "
          f"in {OUT} for whatever test you prefer")'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

# GENERATED CELLS MUST PARSE BEFORE THIS FILE IS WRITTEN. A cell body is a Python
# string inside a Python file, so an escape correct in one layer is wrong in the
# other; that has shipped twice in the sibling builder.
import ast as _ast

broken = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    out, mag = [], False
    for ln in "".join(c["source"]).split(chr(10)):
        if ln.lstrip().startswith(("!", "%")) or mag:
            mag = ln.rstrip().endswith(chr(92))
            out.append("pass")
        else:
            out.append(ln)
    try:
        _ast.parse(chr(10).join(out))
    except SyntaxError as e:
        broken.append((i, e.lineno, e.msg))
if broken:
    for i, l, m in broken:
        print(f"  cell {i} line {l}: {m}")
    raise SystemExit(f"{len(broken)} generated cell(s) do not parse; {OUT} not written")

json.dump(nb, open(OUT, "w"), indent=1)
print(f"wrote {OUT}  ({len(cells)} cells, {os.path.getsize(OUT)/1024:.0f} KB)")
print("runs locally: no Colab, no Drive, no encoder (TOMATO's BGE cache is complete)")
