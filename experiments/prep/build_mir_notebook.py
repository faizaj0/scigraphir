"""
build_mir_notebook.py -- Table 9.1 on MIR: in-benchmark training and evaluation of the
cumulative SciGraphIR ablation next to the six baselines.

THE TABLE. Same rows as Table 9.1's families, minus the ones that need a field split (MIR is
all computational linguistics, so there is no same/cross partition and no gap column):

    lexical                   BM25
    dense                     BGE-large, Qwen3-Embedding, SPECTER2-base, SciNCL
    reasoning-trained dense   ReasonIR-8B
    SciGraphIR, cumulative    Multi-View Semantic Scorer  ->  + Graph Reasoner  ->  + CCMP

Columns: Recall@3, Recall@5, nDCG@5, and mAP, the last because MIR's own paper reports
R@3 / R@5 / mAP, so its published retrievers can sit under the table as reference rows.
Their numbers are on the 284-document restricted corpus; ours are on the ~4.9k-document
extended corpus staged by prep/stage_mir.py, so the comparison is stated, not implied.

PROTOCOL. Train on mir_train_v16sc (1,270 proposals, 4,678 docs), select and evaluate on
mir_test_v16sc (155 proposals, 4,857 docs), predictions taken from the training run's
final predict pass exactly as the TOMATO and SIR-4 rows were. The multi-view scorer row is
the scorer trained alone (section 5d); the graph rows are the fusion with and without CCMP.

Reuses build_rb_zeroshot_notebook.py for the engine cloning, the fusion-source blob, the
inline script overlay and the shell/sync helpers.

Usage
-----
    python3 prep/build_mir_notebook.py
    python3 prep/build_mir_notebook.py --epochs 10 --batch 2 --ccmp residual
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_rb_zeroshot_notebook as rb  # noqa: E402

ROOT, CARGO, FORK = rb.ROOT, rb.CARGO, rb.FORK
md, code = rb.md, rb.code

HEADER = '''# MIR — in-benchmark training and the cumulative ablation (Table 9.1 on MIR)

Train **SciGraphIR** on MIR's training proposals, evaluate on its test proposals over the
extended corpus, next to the six baselines. Same runner and scorer as every other table.

| section | what | cost |
|---|---|---|
| 1-2 | paths, bundle, current scripts, Qwen3 | minutes |
| 3 | six baselines on the test corpus | ~15 min (ReasonIR is the slow one) |
| 4 | engine + fusion sources | 5 min |
| 5 | multi-view scorer (5d), component tables | ~15 min |
| 6 | fusion arms: + graph reasoner, + CCMP | ~1 h each at this corpus size |
| 7 | table: R@3, R@5, nDCG@5, mAP | seconds |

MIR (Garikaparthi et al., ACL 2025) is computational linguistics only, so there is no
same/cross split. Golds are citations with MultiCite intent *Uses* or *Extension*; documents
are abstracts only. MIR's published numbers are on a 284-document restricted corpus and are
printed under the table as reference, not as rows.
'''

# The helpers half of the RB paths cell (sh, sync_dir, start_sync, HF token), verbatim.
_HELP = rb.PATHS[rb.PATHS.index("try:\n    from google.colab import userdata"):]
_HELP = _HELP.replace('env = dict(os.environ, SCIGRAPHIR_ROOT=SCIGRAPHIR_ROOT, SCIGRAPHIR_DATASET=DATASET, PYTHONUNBUFFERED="1")',
                      'env = dict(os.environ, SCIGRAPHIR_ROOT=SCIGRAPHIR_ROOT, SCIGRAPHIR_DATASET=DATASET, PYTHONUNBUFFERED="1")')
_HELP = _HELP[:_HELP.index("for d in (OUT_ROOT, RUNS):")]

PATHS = '''!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import os, sys, re, time, glob, json, math, shutil, zipfile, subprocess, hashlib, threading
from collections import defaultdict
from google.colab import drive
drive.mount('/content/drive')

DRIVE    = "/content/drive/MyDrive/cargo-gfmrag"
DATASET  = "mir"
GRAPH    = "__GRAPH__"                           # frame: mir_*_v16sc (SciAfford) | openie: mir_* (entity+document)
TRAIN, TEST = ((f"{DATASET}_train_v16sc", f"{DATASET}_test_v16sc") if GRAPH == "frame"
               else (f"{DATASET}_train", f"{DATASET}_test"))
BUNDLE   = f"{DRIVE}/{DATASET}_bundle.zip"
OUT_ROOT = f"{DRIVE}/outputs/{DATASET}"          # everything this notebook writes
CACHE    = f"{OUT_ROOT}/cache"                   # embeddings + Qwen3 node indexes, graph-keyed
SCIGRAPHIR_ROOT = "/content/scigraphir"
os.environ["SCIGRAPHIR_ROOT"] = SCIGRAPHIR_ROOT
os.environ["SCIGRAPHIR_DATASET"] = DATASET
DATA_ROOT = f"{SCIGRAPHIR_ROOT}/retriever/data"
S4        = f"{SCIGRAPHIR_ROOT}/experiments"
KGDIR     = f"{SCIGRAPHIR_ROOT}/retriever"
RUNS      = "/content/runs"                      # LOCAL run directories, synced to Drive
OP_MODEL  = "/content/qwen3"
OP_SLUG   = "_content-qwen3"
EPOCHS, BATCH = __EPOCHS__, __BATCH__
CCMP_VARIANT = "__CCMP__"                        # standard (30 Aug targets) or residual (31 Aug)

''' + _HELP + '''
for d in (OUT_ROOT, RUNS, CACHE):
    os.makedirs(d, exist_ok=True)
print("DRIVE   ", DRIVE)
print("reads   ", os.path.basename(BUNDLE), "+ gfm-rag-adapted.zip + qwen3-embedding-0.6b/")
print("graphs  ", TRAIN, "->", TEST)
print("writes  ", OUT_ROOT, "| CCMP variant:", CCMP_VARIANT)
'''

UNPACK = '''# 2. Unpack the MIR bundle to an isolated local root and install the CURRENT repo scripts.
KEEP, PARK = f"{SCIGRAPHIR_ROOT}/outputs/caches", "/content/_caches_keep"
if os.path.isdir(KEEP):
    shutil.rmtree(PARK, ignore_errors=True); shutil.move(KEEP, PARK)
if os.path.exists(SCIGRAPHIR_ROOT):
    shutil.rmtree(SCIGRAPHIR_ROOT)
os.makedirs(SCIGRAPHIR_ROOT, exist_ok=True)
assert os.path.exists(BUNDLE) and zipfile.is_zipfile(BUNDLE), f"missing or corrupt {BUNDLE}"
zipfile.ZipFile(BUNDLE).extractall(SCIGRAPHIR_ROOT); print("unpacked", os.path.basename(BUNDLE))
if os.path.isdir(PARK):
    os.makedirs(os.path.dirname(KEEP), exist_ok=True); shutil.move(PARK, KEEP); print("restored caches")

OVERLAY = json.loads(r\'\'\'__OVERLAY__\'\'\')
def apply_overlay():
    ov = f"{DRIVE}/code_overlay"
    if os.path.isdir(ov):
        shutil.copytree(ov, SCIGRAPHIR_ROOT, dirs_exist_ok=True); print("applied Drive code_overlay")
    for rel, src in OVERLAY.items():
        p = f"{SCIGRAPHIR_ROOT}/{rel}"
        os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(src)
    print(f"installed {len(OVERLAY)} repo scripts captured __BUILT__")
apply_overlay()
sys.path.insert(0, SCIGRAPHIR_ROOT)
import scigraphir_paths as cp
cp.set_dataset(DATASET); print(cp.banner())

QUERIES = f"{DATA_ROOT}/{DATASET}_test/raw/test.json"
for s, g in (("train", TRAIN), ("test", TEST)):
    for label, p in ((f"corpus {s}", f"{cp.corpus_dir(s)}/raw/documents.json"),
                     (f"queries {s}", f"{cp.corpus_dir(s)}/raw/{s}.json"),
                     (f"probes {s}", cp.probes_path(s)),
                     (f"graph {s}", f"{DATA_ROOT}/{g}/processed/stage1/nodes.csv")):
        assert os.path.exists(p), f"missing {label}: {p}"
        print(f"  ok  {label:13} {p.replace(SCIGRAPHIR_ROOT, '<root>')}")
    # The loader opens {graph}/raw/documents.json, a copy of the corpus INSIDE the graph dir.
    # For the OpenIE graph the graph directory IS the corpus directory, so there is no copy.
    _src, _dst = f"{cp.corpus_dir(s)}/raw/documents.json", f"{DATA_ROOT}/{g}/raw/documents.json"
    if os.path.abspath(_src) != os.path.abspath(_dst):
        os.makedirs(os.path.dirname(_dst), exist_ok=True); shutil.copy(_src, _dst)
    assert os.path.exists(_dst), _dst
# Node types and seedless queries. The frame graph audits to zero seedless queries; the OpenIE
# graph may drop a few (four on SIR-4 CS), and the loader drops them SILENTLY, so they are
# recorded here and re-inserted as empty rankings before scoring (section 6).
import csv, collections
csv.field_size_limit(10 ** 7)
SEEDLESS = {}
for s, g in (("train", TRAIN), ("test", TEST)):
    s1 = f"{DATA_ROOT}/{g}/processed/stage1"
    _types = collections.Counter(r["type"] for r in csv.DictReader(open(f"{s1}/nodes.csv")))
    if GRAPH == "openie":
        assert "entity" in _types and "document" in _types, f"{g} is not an entity+document graph: {dict(_types)}"
    _q = json.load(open(f"{s1}/{s}.json"))
    SEEDLESS[s] = [x["id"] for x in _q
                   if isinstance(x.get("start_nodes"), dict) and not any(x["start_nodes"].values())]
    print(f"  {g:18} nodes {dict(_types)}  queries {len(_q):,}  seedless {len(SEEDLESS[s])}")
_m = json.load(open(f"{DATA_ROOT}/{DATASET}_test/MANIFEST.json"))
print("\\nstaged:", json.dumps(_m["splits"], indent=1))
print("gold rule:", _m["gold_rule"], "| test corpus:", _m["test_corpus"], "| train source:", _m["train_source"])

# mAP is in the column list because MIR's paper reports R@3 / R@5 / mAP.
COLS = "mrr,ndcg@5,recall@3,recall@5,recall@10,recall@100,map"
!pip -q install rank_bm25 sentence-transformers "transformers>=4.52.4,<5"
import transformers
assert transformers.__version__.startswith("4."), f"transformers {transformers.__version__}: restart the runtime"
print("ready | transformers", transformers.__version__)
'''

RUN_BASELINES = '''# 3b. The six baselines on the MIR test corpus. Same runner, same flags as every other table.
import shlex
TOPK = 300
OUTB = f"{OUT_ROOT}/baselines"; os.makedirs(OUTB, exist_ok=True)
_SCORER = hashlib.md5(open(f"{S4}/eval/baselines_sir4.py", "rb").read()).hexdigest()[:8]
def arm_sig(model, pool, ins, extra):
    return hashlib.md5(json.dumps({"model": model, "pooling": pool, "instruct": ins, "extra": extra,
                                   "topk": TOPK, "scorer": _SCORER, "dataset": DATASET, "split": "test"},
                                  sort_keys=True).encode()).hexdigest()[:12]
PRED = {}
for lab, tag, model, pool, ins, extra in ARMS:
    dest = f"{S4}/data/predictions_{tag}_{DATASET}_test.json"; man = dest + ".manifest.json"
    sig = arm_sig(model, pool, ins, extra)
    for a, b in ((f"{OUTB}/{os.path.basename(dest)}", dest), (f"{OUTB}/{os.path.basename(man)}", man)):
        if not os.path.exists(b) and os.path.exists(a):
            os.makedirs(os.path.dirname(b), exist_ok=True); shutil.copy(a, b)
    have = None
    if os.path.exists(man):
        try: have = json.load(open(man)).get("sig")
        except Exception: have = None
    if os.path.exists(dest) and have == sig:
        print(f"[skip] {lab}: manifest matches"); PRED[tag] = dest; continue
    cmd = [sys.executable, "-u", "eval/baselines_sir4.py", "--dataset", DATASET, "--split", "test",
           "--model", model, "--pooling", pool, "--tag", tag, "--topk", str(TOPK)]
    if ins:   cmd += ["--instruct", ins]
    if extra: cmd += shlex.split(extra)
    cmd += ["--out", dest]
    rc = sh(cmd, S4, check=False)
    if rc != 0:
        print(f"!! {lab} FAILED rc={rc} -- row reported as missing"); continue
    json.dump({"sig": sig, "model": model, "pooling": pool, "instruct": ins, "extra": extra,
               "topk": TOPK, "scorer_md5": _SCORER}, open(man, "w"), indent=1)
    shutil.copy(dest, f"{OUTB}/{os.path.basename(dest)}"); shutil.copy(man, f"{OUTB}/{os.path.basename(man)}")
    PRED[tag] = dest
print("\\nbaseline predictions:", sorted(PRED))
'''

SCORE = '''# 4. Score. One scorer, one flag set, no sets.json (MIR has no decomposition sets) and no
# similar/dissimilar split (not needed for this table).
SCORES = f"{OUT_ROOT}/scores"; os.makedirs(SCORES, exist_ok=True)
def score(key, pred, label):
    js, pq = f"{SCORES}/{key}_scores.json", f"{SCORES}/{key}_perquery.json"
    sh([sys.executable, "-u", "eval/score_sir4.py", "--pred", pred, "--queries", QUERIES,
        "--name", label, "--cols", COLS, "--json-out", js, "--per-query-out", pq], S4)
    return js
LABEL = {tag: lab for lab, tag, *_ in ARMS}
for tag, pred in PRED.items():
    score(tag, pred, LABEL[tag])
'''

HELPERS = '''# 5a. Caches (graph-keyed) and the operator component tables.
import numpy as np
if os.path.isdir(f"{CACHE}/op_emb"):
    shutil.copytree(f"{CACHE}/op_emb", f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb", dirs_exist_ok=True)
    print("restored embedding caches from Drive")

def restore_index(g):
    src = f"{CACHE}/index/{g}"
    if not os.path.isdir(src):
        print(f"  {g}: no cached index on Drive (built on first use)"); return
    for d in os.listdir(src):
        shutil.copytree(f"{src}/{d}", f"{DATA_ROOT}/{g}/processed/{d}", dirs_exist_ok=True)
    print(f"  {g}: index restored")
def save_index(g):
    pr = f"{DATA_ROOT}/{g}/processed"
    for d in os.listdir(pr):
        if d != "stage1":
            shutil.copytree(f"{pr}/{d}", f"{CACHE}/index/{g}/{d}", dirs_exist_ok=True)
def _sem_ok(p):
    if not os.path.exists(p): return False
    zz = np.load(p, allow_pickle=True)
    return os.path.exists(str(zz["h_path"])) and "qwen" in str(zz["encoder"]).lower()
def opc(g):  return f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
def semc(g): return f"{DATA_ROOT}/{g}/semantic_components{OP_SLUG}.npz"

for s, g in (("train", TRAIN), ("test", TEST)):
    restore_index(g)
    if not os.path.exists(opc(g)):
        c = f"{CACHE}/{g}_operator_components{OP_SLUG}.npz"
        if os.path.exists(c): shutil.copy(c, opc(g))
        else:
            sh(f"python3 -u precompute/precompute_operator_components.py "
               f"--dataset {DATASET} --graph {g} --split {s} --model {OP_MODEL}", KGDIR)
            shutil.copy(opc(g), c)
    zz = np.load(opc(g), allow_pickle=True)
    assert "qwen" in str(zz["encoder"]).lower(), f"{opc(g)} was built with {zz['encoder']!r}"
    print(f"  {g:18} operator components dense {zz['dense'].shape}")
shutil.copytree(f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb", f"{CACHE}/op_emb", dirs_exist_ok=True)
'''

SCORER = '''# 5b. The multi-view scorer (5d recipe): sorted-MLP over the hypothetical-answer match profile
# with a jointly trained popularity predictor, --loss fixed for the multi-gold objective.
# Its test predictions ARE the "Multi-View Semantic Scorer" row; its weights warm-start the
# fusion arms. `dense` (cosine only) and `current` (the handcrafted operator) come out of the
# same run for free and are scored as extra reference rows.
SEM = f"{S4}/results/semantic_{DATASET}"
SEM_DRIVE = f"{OUT_ROOT}/semantic"
SEM_CKPT = f"{SEM}/params_semantic_mlp_fixedloss_{DATASET}.json"
SEM_POP  = f"{SEM}/popnet_semantic_mlp_fixedloss_{DATASET}.pt"
os.makedirs(SEM, exist_ok=True)
if os.path.isdir(SEM_DRIVE) and not os.path.exists(SEM_CKPT):
    shutil.copytree(SEM_DRIVE, SEM, dirs_exist_ok=True); print("scorer restored from Drive")
if not (os.path.exists(SEM_CKPT) and os.path.exists(SEM_POP)):
    sh(f"python3 -u eval/semantic_scorer.py --dataset {DATASET} --model {OP_MODEL} "
       f"--arms dense,current,mlp --loss fixed --train_fit 0 --dev 150 --epochs 30 --patience 10 "
       f"--lr 1e-3 --weight_decay 1e-2 --mlp_hidden 16 --mlp_pop_joint 1 --pop_lambda 1.0 "
       f"--select_on loss --qbatch 32 --seed 0 --seeds 0,1,2 --out {SEM}", S4)
    shutil.copytree(SEM, SEM_DRIVE, dirs_exist_ok=True)
    shutil.copytree(f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb", f"{CACHE}/op_emb", dirs_exist_ok=True)
for p in (SEM_CKPT, SEM_POP):
    assert os.path.exists(p), f"5d did not write {p}"
_st = json.load(open(SEM_CKPT))
print(f"scorer: jmax={_st['jmax']} beta={_st['beta']:.4f} hidden={len(_st['net']['0.weight'])}")

SEM_PRED = {"scorer":   f"{SEM}/predictions_semantic_mlp_fixedloss_{DATASET}_test.json",
            "operator": f"{SEM}/predictions_semantic_current_fixedloss_{DATASET}_test.json",
            "dense":    f"{SEM}/predictions_semantic_dense_{DATASET}_test.json"}
for k, p in SEM_PRED.items():
    assert os.path.exists(p), f"missing {p}"
score("scigraphir_scorer", SEM_PRED["scorer"], "Multi-View Semantic Scorer")
score("ref_operator", SEM_PRED["operator"], "operator scorer (reference)")
score("ref_dense", SEM_PRED["dense"], "Qwen3 cosine, no views (reference)")

# scorer components aligned to each graph's document order
for s, g in (("train", TRAIN), ("test", TEST)):
    if not _sem_ok(semc(g)):
        sh(f"python3 -u precompute/precompute_semantic_components.py "
           f"--dataset {DATASET} --model {OP_MODEL} --graph {g} --split {s}", KGDIR)
    zz = np.load(semc(g), allow_pickle=True)
    print(f"  {g:18} H {tuple(int(x) for x in zz['h_shape'])}  Jmax={int(zz['Jmax'])}")
'''

TRAIN_ARMS = '''# 6. The two fusion arms, trained in-benchmark. "+ Graph Reasoner" = the multi-view scorer
# fused with the graph reasoner, no CCMP. "+ CCMP" adds the responsibility head with the
# Table 9.1 settings. Predictions come out of the training run's final predict pass.
BASE_ENV = dict(WANDB_MODE="disabled", HYDRA_FULL_ERROR="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                OPERATOR_COMPONENTS=opc(TRAIN), OPERATOR_COMPONENTS_TEST=opc(TEST),
                SEMANTIC_COMPONENTS=semc(TRAIN), SEMANTIC_COMPONENTS_TEST=semc(TEST),
                SEMANTIC_CKPT=SEM_CKPT, SEMANTIC_POPNET=SEM_POP, SEM_POP_LAMBDA="1.0",
                FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0",
                PER_GOLD="1", HARDNEG_GRAPH="50", STRAT_TEST=QUERIES)
CCMP_ENV = dict(CCMP="1", CCMP_W="1.0", CCMP_NEG="64", CCMP_M="2000", CCMP_LR="2e-3",
                CCMP_GATE="1", CCMP_GATE_NORM="1", CCMP_ETA="0.5",
                CCMP_POOL="256", CCMP_M_POS="512", CCMP_M_NEG="512",
                CCMP_RESIDUAL="1" if CCMP_VARIANT == "residual" else "0", CCMP_IDENTITY_W="1.0")
for k in list(os.environ):
    if k.startswith(("CCMP", "STRAT_", "CQIG", "RESID_", "MISS_W")): os.environ.pop(k)

def train_cmd(run_dir, epochs, max_steps=None):
    return ("python -u -m gfmrag.workflow.sft_training "
            "--config-path config/gfm_reasoner --config-name sft_training_fusion "
            "text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{TRAIN}] datasets.valid_names=[{TEST}] "
            "model.semantic=mlp model.cqig=false "
            f"trainer.args.num_epoch={epochs} trainer.args.train_batch_size={BATCH} "
            f"+trainer.args.eval_batch_size={BATCH} "
            "+trainer.args.do_predict=true +trainer.args.predict_top_k=300 "
            + (f"trainer.args.max_steps_per_epoch={max_steps} " if max_steps else "")
            + f"hydra.run.dir={run_dir}")

if GRAPH == "openie":
    # ONE arm: the "+ Graph Reasoner" rung on the OpenIE graph, no CCMP, matching the SIR-4
    # OpenIE row (colab_sir4_openie_ablation.ipynb). The run name carries `openie` so it never
    # collides with the frame-graph runs on Drive; the scorer warm start is shared (graph-free).
    ARMS_FUSION = [("scigraphir_openie_graph", f"{DATASET}_openie_qwenmlp_graph_e{EPOCHS}_b{BATCH}",
                    "+ Graph Reasoner (OpenIE graph)", {})]
else:
    ARMS_FUSION = [("scigraphir_graph", f"{DATASET}_qwenmlp_graph_e{EPOCHS}_b{BATCH}", "+ Graph Reasoner", {}),
                   ("scigraphir_ccmp",  f"{DATASET}_qwenmlp_{'ccmpresid' if CCMP_VARIANT == 'residual' else 'ccmp'}_e{EPOCHS}_b{BATCH}",
                    "+ CCMP (SciGraphIR)", CCMP_ENV)]
FUSION_PRED = {}
for key, name, label, extra in ARMS_FUSION:
    local, drive = f"{RUNS}/{name}", f"{OUT_ROOT}/{name}"
    pred_drive = f"{drive}/predictions_{TEST}.json"
    env_ = dict(BASE_ENV, **extra)
    print(f"\\n==================== {label} ====================")
    if os.path.exists(pred_drive) and os.path.exists(f"{drive}/console.log") and \\
            f"Epoch {EPOCHS} completed" in open(f"{drive}/console.log", errors="ignore").read():
        print(f"[skip] finished: {os.path.relpath(drive, DRIVE)}"); FUSION_PRED[key] = pred_drive; continue
    if key in ("scigraphir_graph", "scigraphir_openie_graph"):   # one smoke per notebook: 1 epoch, 3 steps
        smoke = f"{local}_smoke"; os.makedirs(smoke, exist_ok=True)
        rc = sh(train_cmd(smoke, 1, 3).replace("+trainer.args.do_predict=true", "+trainer.args.do_predict=false"),
                "/content/gfm-rag", extra=env_, log=f"{smoke}/console.log", check=False)
        log = open(f"{smoke}/console.log", errors="ignore").read()
        assert rc == 0, f"smoke failed (exit {rc}); read {smoke}/console.log"
        assert "semantic='mlp' warm start" in log, "multi-view scorer not constructed"
        assert "[ccmp] responsibility head" not in log, "a CCMP head was built on the no-CCMP arm"
        for g in (TRAIN, TEST): save_index(g)
        print("smoke passed")
    os.makedirs(local, exist_ok=True); os.makedirs(drive, exist_ok=True)
    json.dump({"arm": label, "dataset": DATASET, "train": TRAIN, "valid": TEST, "epochs": EPOCHS, "batch": BATCH,
               "semantic": "mlp", "ccmp": bool(extra), "ccmp_variant": CCMP_VARIANT if extra else None,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{local}/arm.json", "w"), indent=1)
    finish = start_sync(local, drive, every=600)
    try:
        rc = sh(train_cmd(local, EPOCHS), "/content/gfm-rag", extra=env_, log=f"{local}/console.log", check=False)
    finally:
        finish()
    log = open(f"{local}/console.log", errors="ignore").read()
    assert rc == 0, f"{label}: training failed, exit {rc}"
    if extra:
        assert "[ccmp] responsibility head" in log, "CCMP head not constructed"
        assert ("RESIDUAL on" in log) == (CCMP_VARIANT == "residual"), "wrong CCMP variant ran"
    assert os.path.exists(f"{local}/predictions_{TEST}.json"), "no predictions written"
    sync_dir(local, drive); FUSION_PRED[key] = pred_drive
for key, name, label, _ in ARMS_FUSION:
    pred = FUSION_PRED[key]
    if SEEDLESS.get("test"):
        # A test query the loader dropped (no seed) gets an EMPTY ranking, so it counts as a
        # miss and the row is scored over the full test set, not a silently smaller one.
        preds = json.load(open(pred))
        if isinstance(preds, dict): preds = list(preds.values())
        have = {r["id"] for r in preds}
        _q = json.load(open(QUERIES))
        for x in _q:
            if x["id"] not in have:
                preds.append({"id": x["id"], "stratum": x.get("stratum"),
                              "supporting_documents": x["supporting_documents"], "predictions": {"document": []}})
        pred = f"{SCORES}/{key}_predictions_filled.json"; json.dump(preds, open(pred, "w"))
        print(f"{label}: {len(_q) - len(have)} test quer{'y' if len(_q) - len(have) == 1 else 'ies'} "
              f"without a prediction (seedless: {SEEDLESS['test']}) scored as misses")
    score(key, pred, label)
'''

TABLE = '''# 7. Table 9.1 on MIR: R@3, R@5, nDCG@5, mAP as percentages. MIR's published numbers are
# printed underneath as reference; they are on the 284-document restricted corpus.
ORDER = [(r"\\textit{Sparse lexical}", [("bm25", "BM25")]),
         (r"\\textit{Dense embedding}", [("bge", "BGE-large"), ("qwen3", "Qwen3-Embedding"), ("specter2", "SPECTER2-base"), ("scincl", "SciNCL")]),
         (r"\\textit{Reasoning-trained dense retrieval}", [("reasonir", "ReasonIR-8B")]),
         (r"\\textit{\\textsc{SciGraphIR}: cumulative ablation}",
          [("scigraphir_scorer", "Multi-View Semantic Scorer"),
           ("scigraphir_openie_graph", "+ Graph Reasoner (OpenIE graph)"),
           ("scigraphir_graph", "+ Graph Reasoner (SciAfford graph)"),
           ("scigraphir_ccmp", "+ CCMP (full SciGraphIR)")]),
         (r"\\textit{Reference: scorer inputs}", [("ref_dense", "Qwen3 cosine, no hypothetical answers"), ("ref_operator", "operator scorer (4 scalars)")])]
MET = [("recall@3", "R@3"), ("recall@5", "R@5"), ("ndcg@5", "nDCG@5"), ("map", "mAP")]
lines = []
def out(s=""): print(s); lines.append(s)
out(f"# MIR test ({len(json.load(open(QUERIES)))} proposals, extended corpus), in-benchmark training\\n")
out("| Method | " + " | ".join(l for _, l in MET) + " |"); out("|---|" + "--:|" * len(MET))
for fam, rows in ORDER:
    out(f"| {fam} | | | | |")
    for key, lab in rows:
        p = f"{SCORES}/{key}_scores.json"
        if not os.path.exists(p): out(f"| {lab} | -- | -- | -- | -- |"); continue
        s = json.load(open(p))["all"]
        out(f"| {lab} | " + " | ".join(f"{100 * s[m]:.2f}" for m, _ in MET) + " |")
out()
out("MIR paper (Garikaparthi et al., ACL 2025), restricted corpus of 284 documents, R@3 / R@5 / mAP:")
out("BM25 39.34 / -- / --; SPECTER2 56.61 / -- / --; fine-tuned Stella-1.5B 66.95 / -- / --; + LLM re-ranking 71.45 / -- / --.")
out("Fill the dashes from the paper's Table 2 when transcribing; the corpora differ, so state that in the caption.")
open(f"{OUT_ROOT}/table_mir.md", "w").write("\\n".join(lines)); print("\\nwrote", f"{OUT_ROOT}/table_mir.md")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ccmp", default="standard", choices=["standard", "residual"])
    ap.add_argument("--graph", default="frame", choices=["frame", "openie"],
                    help="frame = mir_*_v16sc (SciAfford graph, both fusion arms); openie = mir_* "
                         "(entity+document OpenIE graph from retriever/run_index.sh, one arm: "
                         "'+ Graph Reasoner (OpenIE graph)', no CCMP, as on SIR-4)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = json.load(open(rb.SRC_NB))["cells"]
    base = json.load(open(rb.BASE_NB))["cells"]
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fusion_files = {f"/content/gfm-rag/{rel}": open(f"{FORK}/{rel}").read() for rel in rb.FUSION_REL}
    assert not any("'''" in v for v in fusion_files.values())
    files_cell = code(
        f"# === write the CARGO-fusion files into the fork (generated from the repo copies {built}) ===\n"
        "import json, os\n"
        f"FILES = json.loads(r'''{json.dumps(fusion_files)}''')\n"
        "for p, c in FILES.items():\n"
        "    os.makedirs(os.path.dirname(p), exist_ok=True)\n"
        "    open(p, 'w').write(c)\n"
        "    print('wrote', p, f'({len(c)} bytes)')\n"
        "import importlib, sys\n"
        "sys.path.insert(0, '/content/gfm-rag')\n"
        "for m in ['gfmrag.models.fusion_reasoner', 'gfmrag.trainers.fusion_trainer']:\n"
        "    importlib.import_module(m); print('import OK:', m)\n"
        "print('fusion files ready')\n")
    overlay = {rel: open(f"{CARGO}/{rel}").read() for rel in rb.OVERLAY_REL}
    assert not any("'''" in v for v in overlay.values())
    assert '"map":' in overlay["experiments/eval/score_sir4.py"], "score_sir4.py has no mAP; the table needs it"
    arms_src = "".join(base[rb.ARMS_CELL]["source"])
    assert "ARMS = [" in arms_src and "ReasonIR-8B" in arms_src

    cells = [md(HEADER), md("## 1. GPU + Drive + paths"),
             code(PATHS.replace("__EPOCHS__", str(a.epochs)).replace("__BATCH__", str(a.batch))
                  .replace("__CCMP__", a.ccmp).replace("__GRAPH__", a.graph)),
             md("## 2. Unpack + install the current scripts"),
             code(UNPACK.replace("__OVERLAY__", json.dumps(overlay)).replace("__BUILT__", built)),
             md("## 2b. Qwen3-Embedding-0.6B (cached on Drive)"), src[rb.QWEN_CELLS[1]],
             md("## 3. Baselines\n### 3a. The arms\n*(harvested verbatim from `sir4_baselines_all.ipynb`)*"),
             code(arms_src + rb.ARMS_TAIL), md("### 3b. Run"), code(RUN_BASELINES),
             md("## 4. Score the baselines"), code(SCORE),
             md("## 4b. Engine + fusion sources\n*(cloned from `tomato_ccmp_ablation.ipynb`; the fusion-source blob is regenerated from the repo)*")]
    for i in rb.ENGINE_CELLS:
        cells.append(files_cell if i == 7 else src[i])
    cells += [md("## 5. Multi-view scorer and component tables"), code(HELPERS), code(SCORER),
              md("## 6. Fusion arms: + graph reasoner, + CCMP\nSmoke first, then two runs of about an hour each. Local run, ten-minute Drive sync, finished arms skip."),
              code(TRAIN_ARMS), md("## 7. The table"), code(TABLE)]
    out = a.out or (f"{ROOT}/notebooks/colab_mir_openie.ipynb" if a.graph == "openie" else f"{ROOT}/notebooks/colab_mir_{a.ccmp}.ipynb")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(cells)} cells | ccmp {a.ccmp} | epochs {a.epochs} batch {a.batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
