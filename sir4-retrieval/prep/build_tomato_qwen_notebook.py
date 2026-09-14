"""
build_tomato_qwen_notebook.py -- TOMATO scorer comparison in QWEN3 space, on Colab.

WHY A COLAB NOTEBOOK AT ALL. The local TOMATO notebook uses BGE because those
embeddings are already cached on the laptop. This one re-encodes TOMATO with
Qwen3-Embedding-0.6B, the encoder every SIR-4 run uses, so a TOMATO row can sit in
the same table as the four SIR-4 domains. 122,962 texts is a GPU job.

WHAT IT DOES NOT NEED, which is most of what the SIR-4 notebooks carry:

  * no graph, so no gfm-rag engine install, no PyG patches, no node index;
  * no isolation guard against the name "tomato" -- this IS tomato;
  * no operator fit, no components npz, no fusion.

It mounts Drive, copies the cached Qwen3 model, unpacks an 18 MB slim bundle,
encodes once, runs the arms, scores, and copies the embeddings back to Drive so a
rerun skips the encode.

    python3 prep/build_tomato_qwen_notebook.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = f"{ROOT}/colab_tomato_qwen_scorers.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(keepends=True)}


cells = []

cells.append(md("""# TOMATO-Star in Qwen3 space — scorer architecture comparison

Same five arms as the SIR-4 notebooks, same encoder, so TOMATO can finally sit in
the same table as matsci / physics / cs / biology.

| arm | what it is | params |
|---|---|--:|
| `dense` | `cos(E(q), E(d))` alone, untrained — the floor | 0 |
| `current` | handcrafted `sum` + `max`, three global weights — the target | 4 |
| `attention` | softmax over the direct view and every answer | 34 |
| `deepsets` | `φ([x_hyp, x_dir]) → masked mean → ρ`, 15% answer dropout | 62 |
| `mlp` | one MLP on `[x_dir, sorted answers, count]` | ~200 |

**No graph anywhere in this notebook.** No engine install, no PyG patch, no node
index, no operator fit, no fusion. It encodes, scores, and compares.

**TOMATO specifics that change the reading**

1. **One gold per row**, unique ids, 9,669 train / 3,132 test. `--loss operator`
   and `--loss fixed` are therefore arithmetically identical here: with a single
   gold there are no sibling golds to exclude from the denominator. Only one runs.
2. The same question text repeats across rows with different golds (3,132 rows
   cover 1,658 distinct questions). Scoring per row is the single-gold convention
   every published TOMATO number in this project uses.
3. No `sets.json`, so CompleteSet@k is undefined and left out rather than
   reported as a column of zeros.

**Cost.** 122,962 texts to encode with Qwen3-0.6B, roughly 10-15 minutes on an
A100. That is the only expensive step, it happens once, and it is copied to Drive
at the end so a rerun skips it.

**Before running:** upload `sir4-retrieval/tomato_bundle.zip` (18 MB, built with
`prep/bundle.py --dataset tomato --slim`) to `MyDrive/cargo-gfmrag/`."""))

cells.append(md("## 1. GPU, Drive, paths"))
cells.append(code('''!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import os, sys, json, shutil, subprocess, time, zipfile
from google.colab import drive
drive.mount('/content/drive')

DRIVE      = "/content/drive/MyDrive/cargo-gfmrag"
CARGO_ROOT = "/content/cargo"
DATASET    = "tomato"
BUNDLE     = f"{DRIVE}/tomato_bundle.zip"
OUT_ROOT   = f"{DRIVE}/outputs/tomato_qwen"
CACHE      = f"{OUT_ROOT}/cache"
os.environ["CARGO_ROOT"], os.environ["CARGO_DATASET"] = CARGO_ROOT, DATASET

S4     = f"{CARGO_ROOT}/sir4-retrieval"
SEM    = f"{S4}/results/semantic_tomato_qwen"
MODEL  = "/content/qwen3"
QS     = f"{CARGO_ROOT}/kg-construction/data/tomato_test/raw/test.json"

assert os.path.exists(BUNDLE), (
    f"missing {BUNDLE}. Build it locally with\\n"
    "  python3 prep/bundle.py --dataset tomato --slim --out tomato_bundle.zip\\n"
    "and upload it to MyDrive/cargo-gfmrag/")
os.makedirs(OUT_ROOT, exist_ok=True)
print("bundle  ", BUNDLE, f"({os.path.getsize(BUNDLE)/1e6:.0f} MB)")
print("writes  ", OUT_ROOT)'''))

cells.append(md("## 2. Qwen3 from Drive\n\n"
                "The same cached copy the SIR-4 notebooks use, so the encoder is "
                "byte-identical across every domain and TOMATO."))
cells.append(code('''QDRIVE = f"{DRIVE}/qwen3"
if os.path.isdir(MODEL) and os.path.exists(f"{MODEL}/model.safetensors"):
    print("qwen3 already on this runtime")
elif os.path.isdir(QDRIVE):
    shutil.copytree(QDRIVE, MODEL, dirs_exist_ok=True)
    print(f"copied qwen3 from Drive: {len(os.listdir(MODEL))} files")
else:
    # Fall back to a fresh download rather than failing: the SIR-4 notebooks
    # populate MyDrive/cargo-gfmrag/qwen3, but this notebook may run first.
    os.makedirs(MODEL, exist_ok=True)
    BASE = "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main"
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt",
              "model.safetensors", "modules.json",
              "1_Pooling/config.json", "sentence_bert_config.json"):
        os.makedirs(os.path.dirname(f"{MODEL}/{f}"), exist_ok=True)
        os.system(f'curl -sSL -f "{BASE}/{f}" -o "{MODEL}/{f}"')
    shutil.copytree(MODEL, QDRIVE, dirs_exist_ok=True)
    print("downloaded qwen3 and cached it to Drive for next time")
assert os.path.exists(f"{MODEL}/model.safetensors"), "qwen3 weights missing"
!pip -q install "sentence-transformers>=3.0" 2>/dev/null | tail -1
print("ready")'''))

cells.append(md("## 3. Unpack the bundle, restore any cached embeddings"))
cells.append(code('''if os.path.exists(CARGO_ROOT):
    shutil.rmtree(CARGO_ROOT)
os.makedirs(CARGO_ROOT, exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    z.extractall(CARGO_ROOT)

need = {
    "test corpus":   f"{CARGO_ROOT}/kg-construction/data/tomato_test/raw/documents.json",
    "test queries":  QS,
    "train corpus":  f"{CARGO_ROOT}/kg-construction/data/tomato_train/raw/documents.json",
    "train queries": f"{CARGO_ROOT}/kg-construction/data/tomato_train/raw/train.json",
    "test answers":  f"{CARGO_ROOT}/kg-construction/construct_v2/cache/probes_test.jsonl",
    "train answers": f"{CARGO_ROOT}/kg-construction/construct_v2/cache/probes_train.jsonl",
    "scorer":        f"{S4}/eval/semantic_scorer.py",
    "evaluator":     f"{S4}/eval/score_sir4.py",
}
miss = [k for k, v in need.items() if not os.path.exists(v)]
assert not miss, f"bundle is missing: {miss}"
for k, v in need.items():
    print(f"  ok  {k:14} {v.replace(CARGO_ROOT, '<root>')}")

# THE ENCODE IS THE ONLY EXPENSIVE STEP, and /content does not survive a runtime
# reset. Restore it if a previous run already paid for it.
EMB = f"{CARGO_ROOT}/outputs/caches/op_emb"
if os.path.isdir(f"{CACHE}/op_emb"):
    shutil.copytree(f"{CACHE}/op_emb", EMB, dirs_exist_ok=True)
    print(f"restored {len(os.listdir(EMB))} cached embedding file(s) from Drive")
else:
    print("no cached embeddings on Drive yet; cell 4 will encode (~10-15 min)")

sys.path.insert(0, CARGO_ROOT)
import cargo_paths as cp
cp.set_dataset(DATASET)
print(cp.banner())'''))

cells.append(md("## 4. Run every arm\n\n"
                "The first call encodes 122,962 texts with Qwen3 and builds the "
                "match matrices, then caches both. `--loss` is a no-op on TOMATO "
                "(one gold per row), so only one setting runs."))
cells.append(code('''def sh(cmd, cwd):
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, shell=True,
                         env=dict(os.environ, PYTHONUNBUFFERED="1"),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        chunk = os.read(p.stdout.fileno(), 8192)
        if not chunk:
            break
        sys.stdout.write(chunk.decode("utf-8", "replace")); sys.stdout.flush()
    p.wait()
    print(f"\\n[{time.time() - t0:.0f}s, exit {p.returncode}]")
    assert p.returncode == 0, f"FAILED: {cmd}"

os.makedirs(SEM, exist_ok=True)
sh(f"python3 -u eval/semantic_scorer.py --dataset {DATASET} --model {MODEL} "
   f"--arms dense,current,mlp --loss fixed "
   f"--train_fit 0 --dev 300 --epochs 30 --patience 5 "
   f"--lr 3e-4 --weight_decay 1e-2 --hidden 8 --ds_hidden 4 --ds_dropout 0.15 "
   f"--mlp_hidden 16 --select_on loss --qbatch 8 --seed 0 --seeds 0,1,2 "
   f"--out {SEM}", S4)

# Park the expensive artefacts on Drive before anything can reset the runtime.
os.makedirs(CACHE, exist_ok=True)
if os.path.isdir(EMB):
    shutil.copytree(EMB, f"{CACHE}/op_emb", dirs_exist_ok=True)
    print(f"[cache] {len(os.listdir(EMB))} embedding file(s) saved to Drive")'''))

cells.append(md("## 5. Score every arm\n\n"
                "No `--sets`: TOMATO has none, so CompleteSet@k is left out. "
                "`--bge` uses the BGE dense predictions purely to *define* the "
                "similar/dissimilar slices, which is how every TOMATO table in "
                "this project splits them."))
cells.append(code('''COLS = "mrr,ndcg@5,recall@3,recall@5,recall@10,recall@25,recall@100"
BGE  = f"{CARGO_ROOT}/kg-construction/eval/predictions_bge.json"
ARMS = (("dense",   "dense query-document only", "dense"),
        ("current", "current scorer",            "current_fixedloss"),
        ("mlp",     "sorted-MLP scorer",         "mlp_fixedloss"))

for arm, label, tag in ARMS:
    pred = f"{SEM}/predictions_semantic_{tag}_{DATASET}_test.json"
    assert os.path.exists(pred), f"missing {pred} -- did cell 4 finish?"
    args = (f"--pred {pred} --queries {QS} --cols {COLS} --name '{label}' "
            f"--json-out {SEM}/scores_semantic_{tag}.json "
            f"--per-query-out {SEM}/perquery_{tag}.json")
    if os.path.exists(BGE):
        args += f" --bge {BGE}"
    else:
        print("no BGE predictions in the bundle -> similar/dissimilar skipped")
    sh(f"python3 -u eval/score_sir4.py {args}", S4)'''))

cells.append(md("## 6. The table"))
cells.append(code('''import json
SC   = {a: json.load(open(f"{SEM}/scores_semantic_{t}.json")) for a, _, t in ARMS}
META = json.load(open(f"{SEM}/semantic_comparison_{DATASET}_fixedloss.json"))
ROWS = COLS.split(",")
BASE = "current"
OTHER = [a for a, _, _ in ARMS if a != BASE]

sp = META["actual_split"]
print(f"### TOMATO-Star test, QWEN3 — {' vs '.join(a for a, _, _ in ARMS)}")
print(f"fit={sp['fit']:,} dev={sp['dev']} of {sp['train_queries']:,} train rows; "
      f"test={sp['test_queries']:,}   encoder={META['encoder']}")
for arm, v in META["quick_summary"].items():
    print(f"  {arm:12} dev nDCG@10 {v['dev_ndcg@10']:.4f}")

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

open(f"{SEM}/tomato_qwen_scorers.md", "w").write(
    f"# TOMATO-Star (Qwen3): scorer architecture comparison\\n\\n"
    f"fit={sp['fit']} dev={sp['dev']} test={sp['test_queries']} "
    f"encoder={META['encoder']}\\n\\n```{table}\\n```\\n")

dst = f"{OUT_ROOT}/semantic"
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(SEM)):
    shutil.copy(f"{SEM}/{f}", f"{dst}/{f}")
print(f"\\ncopied {len(os.listdir(SEM))} file(s) to {dst}")'''))

cells.append(md("## 8. Paired bootstrap\n\n"
                "Every arm ranked the same 3,132 rows over the same corpus, so the "
                "comparison is paired. A delta of a point is not readable from two "
                "means at this size."))
cells.append(code('''if os.path.exists(f"{S4}/transfer/paired_bootstrap.py"):
    for arm, _, tag in ARMS:
        if arm == "current":
            continue
        sh(f"python3 -u transfer/paired_bootstrap.py "
           f"--a {SEM}/perquery_current_fixedloss.json --b {SEM}/perquery_{tag}.json "
           f"--a-name current --b-name {arm} "
           f"--metrics mrr,ndcg@5,recall@10,recall@100", S4)
else:
    print(f"paired_bootstrap.py absent; per-query files are in {SEM}")'''))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

import ast as _ast

broken = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    out, mag = [], False
    for ln in "".join(c["source"]).split(chr(10)):
        if ln.lstrip().startswith(("!", "%")) or mag:
            mag = ln.rstrip().endswith(chr(92)); out.append("pass")
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
print(f"wrote {OUT}  ({len(cells)} cells)")
print("upload sir4-retrieval/tomato_bundle.zip to MyDrive/cargo-gfmrag/ first")
