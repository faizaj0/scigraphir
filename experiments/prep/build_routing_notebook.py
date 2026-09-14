"""
build_routing_notebook.py -- the CCMP comparison: SciGraphIR with no routing, with two
published routing baselines ported into its own graph reasoner, and with CCMP.

THE QUESTION. CCMP adds a per-node gate on outgoing messages and trains it with explicit
continuation targets. A reviewer will ask whether the gain (if any) comes from having a gate
at all or from CCMP's targets. This notebook answers it with four arms that share every
parameter of the reasoner, the multi-view scorer, the fusion gate and the ranking loss, and
differ in exactly one thing: where the routing signal comes from.

    control   SciGraphIR (scorer + graph reasoner), no routing
    astar     + A*Net-style routing      node-level, hard top-K per layer, learned from the
                                         ranking loss only   (ROUTE=astar)
    attn      + RED-GNN-style attention  edge-level softmax per receiver, learned from the
                                         ranking loss only   (ROUTE=attn)
    ccmp      + CCMP                     node-level soft gate, supervised by continuation
                                         targets             (CCMP=1)

The two baselines are ports of the PRINCIPLE of those papers into this reasoner, not
reimplementations of their code (A*Net also selects edges; RED-GNN grows its own subgraph).
The astar head IS the CCMP head with the CCMP loss switched off, so control vs astar vs ccmp
is a clean ladder: no gate -> gate trained implicitly -> gate trained explicitly.

WHERE TO LOOK. On the fused score every arm will sit within noise, because the fusion gate
gives the graph channel little weight (measured on TOMATO and on every SIR-4 field). The
table therefore reports, next to the fused nDCG@5, the graph channel's OWN nDCG@5, its gap
over a parameter-free random walk, and the graph's recall of golds the semantic scorer
buried (the `1k+` bucket), all parsed from the trainer's final-evaluation diagnostics. Those
are the columns where routing can show; the paired bootstrap on the fused score says whether
anything survives at the head.

Engine: the fusion-source blob is regenerated from the repo, and now includes
ultra/layers.py (per-query edge weights for the attention arm). The CCMP_LR patch cell is
extended so the attention head gets the same separate learning-rate group as the CCMP head.

Usage
-----
    python3 prep/build_routing_notebook.py                          # sir4_matsci, 4 arms
    python3 prep/build_routing_notebook.py --dataset tomato
    python3 prep/build_routing_notebook.py --arms control,astar,attn,ccmp,ccmp_residual
    python3 prep/build_routing_notebook.py --seeds 1024,7 --epochs 10 --batch 2
    python3 prep/build_routing_notebook.py --dataset mir --arms control,ccmp,astar,attn \
        --reuse control=outputs/mir/mir_qwenmlp_graph_e10_b2,ccmp=outputs/mir/mir_qwenmlp_ccmp_e10_b2
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

FUSION_REL = rb.FUSION_REL + ["gfmrag/models/ultra/layers.py"]
GUARD = '    assert "sir4" not in t, f"a sir4 dataset reference survived in {cfg}"\n'

ARM_ORDER = ["control", "astar", "attn", "ccmp", "ccmp_residual"]
ARM_LABEL = {"control": "SciGraphIR (scorer + graph reasoner, no routing)",
             "astar": "+ A*Net-style routing (ranking loss only)",
             "attn": "+ RED-GNN-style attention (ranking loss only)",
             "ccmp": "+ CCMP (continuation targets)",
             "ccmp_residual": "+ CCMP, residual targets"}

HEADER = '''# SciGraphIR routing comparison: no routing vs A*Net-style vs RED-GNN-style vs CCMP

Four arms of the **same** SciGraphIR (multi-view scorer + graph reasoner + fusion gate +
hard-negative ranking loss), trained in-benchmark on `__DATASET__`. They differ in one thing:
where the graph reasoner's routing signal comes from.

| arm | routing | trained by |
|---|---|---|
| control | none | ranking loss |
| astar | node gate, hard top-K of the reached frontier per layer (A*Net principle) | ranking loss only |
| attn | edge attention, softmax over each receiver's incoming edges (RED-GNN principle) | ranking loss only |
| ccmp | node gate, soft and mean-preserving | ranking loss + continuation targets |

The astar arm uses CCMP's own responsibility head with the CCMP loss off, so
control → astar → ccmp is one ladder: no gate → gate learned implicitly → gate supervised.

**Read the graph-channel columns, not only the fused score.** The fusion gate gives the graph
little weight, so fused nDCG@5 will be flat across arms. The table adds the graph's own nDCG@5,
its gap over a parameter-free random walk, and its recall of golds the scorer buried.

| section | what | cost |
|---|---|---|
| 1-2 | paths, bundle, current scripts, Qwen3 | minutes |
| 3 | engine + fusion sources (now including `ultra/layers.py`) | 5 min |
| 4 | multi-view scorer (restored from Drive if the field was run before) | 0-15 min |
| 5 | the arms: smoke each, then train | see the estimate printed in 5 |
| 6 | table + paired bootstrap | minutes |
'''

# The helpers half of the RB paths cell (sh, sync_dir, start_sync, HF token), verbatim.
_HELP = rb.PATHS[rb.PATHS.index("try:\n    from google.colab import userdata"):]
_HELP = _HELP[:_HELP.index("for d in (OUT_ROOT, RUNS):")]

PATHS = '''!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import os, sys, re, time, glob, json, math, shutil, zipfile, subprocess, hashlib, threading
from collections import defaultdict
from google.colab import drive
drive.mount('/content/drive')

DRIVE    = "/content/drive/MyDrive/cargo-gfmrag"
DATASET  = "__DATASET__"
GRAPH    = "__GRAPH__"                               # graph suffix: v16sc = SciAffordGraph, hyb = merged graph (build_hybrid_graph.py)
TRAIN, TEST = f"{DATASET}_train_{GRAPH}", f"{DATASET}_test_{GRAPH}"
BUNDLE   = f"{DRIVE}/{DATASET}_bundle.zip"
EXTRA_BUNDLES = __EXTRA__                            # add-on zips unpacked after the bundle (e.g. tomato_hyb_bundle.zip)
TAG      = "" if GRAPH == "v16sc" else "_" + GRAPH
OUT_ROOT = f"{DRIVE}/outputs/routing{TAG}/{DATASET}"  # everything this notebook writes
CACHE    = f"{DRIVE}/outputs/{DATASET}/cache"        # embeddings + Qwen3 node indexes the field runs cached
SEM_DRIVE = f"{DRIVE}/__SEM_DRIVE__"                 # the multi-view scorer this field already trained
SCIGRAPHIR_ROOT = "/content/scigraphir"
os.environ["SCIGRAPHIR_ROOT"] = SCIGRAPHIR_ROOT
os.environ["SCIGRAPHIR_DATASET"] = DATASET
DATA_ROOT = f"{SCIGRAPHIR_ROOT}/retriever/data"
S4        = f"{SCIGRAPHIR_ROOT}/experiments"
KGDIR     = f"{SCIGRAPHIR_ROOT}/retriever"
RUNS      = "/content/runs"                          # LOCAL run directories, synced to Drive
OP_MODEL  = "/content/qwen3"
OP_SLUG   = "_content-qwen3"
EPOCHS, BATCH = __EPOCHS__, __BATCH__
SEEDS     = __SEEDS__                                # trainer seeds; 1024 is the config default every run used
ARM_KEYS  = __ARMS__                                 # which arms this notebook runs, in this RUN order
ARM_LADDER = [k for k in __ARM_ORDER__ if k in ARM_KEYS]   # table order: no gate -> implicit gates -> CCMP
VALIDATE  = __VALIDATE__                             # arms retrained with VAL_SEED after the table, as a check
VAL_SEED  = __VAL_SEED__
ROUTE_K   = "__ROUTE_K__"                            # astar: nodes kept per layer (CCMP supervises 1024 per layer)
# Finished runs of the SAME arm made by another notebook with the same engine, scorer, seed and
# epochs (Drive-relative). They stand in for that arm instead of retraining it; see section 5.
REUSE     = __REUSE__

''' + _HELP + '''
for d in (OUT_ROOT, RUNS, CACHE):
    os.makedirs(d, exist_ok=True)
print("DRIVE   ", DRIVE)
print("reads   ", os.path.basename(BUNDLE), "+ gfm-rag-adapted.zip + qwen3-embedding-0.6b/")
print("graphs  ", TRAIN, "->", TEST)
print("writes  ", OUT_ROOT, "| arms:", ARM_KEYS, "| seeds:", SEEDS)
'''

UNPACK = '''# 2. Unpack the bundle to an isolated local root and install the CURRENT repo scripts.
KEEP, PARK = f"{SCIGRAPHIR_ROOT}/outputs/caches", "/content/_caches_keep"
if os.path.isdir(KEEP):
    shutil.rmtree(PARK, ignore_errors=True); shutil.move(KEEP, PARK)
if os.path.exists(SCIGRAPHIR_ROOT):
    shutil.rmtree(SCIGRAPHIR_ROOT)
os.makedirs(SCIGRAPHIR_ROOT, exist_ok=True)
assert os.path.exists(BUNDLE) and zipfile.is_zipfile(BUNDLE), f"missing or corrupt {BUNDLE}"
zipfile.ZipFile(BUNDLE).extractall(SCIGRAPHIR_ROOT); print("unpacked", os.path.basename(BUNDLE))
for _z in EXTRA_BUNDLES:
    _z = f"{DRIVE}/{_z}"; assert os.path.exists(_z) and zipfile.is_zipfile(_z), f"missing or corrupt {_z}"
    zipfile.ZipFile(_z).extractall(SCIGRAPHIR_ROOT); print("unpacked", os.path.basename(_z))
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
    os.makedirs(f"{DATA_ROOT}/{g}/raw", exist_ok=True)
    _src, _dst = f"{cp.corpus_dir(s)}/raw/documents.json", f"{DATA_ROOT}/{g}/raw/documents.json"
    if os.path.abspath(_src) != os.path.abspath(_dst):
        shutil.copy(_src, _dst)
_q = json.load(open(QUERIES))
_strata = sorted({q.get("stratum", "same") for q in _q})
print(f"\\ntest queries {len(_q)} | strata present: {_strata}")

COLS = "mrr,ndcg@5,recall@3,recall@5,recall@10,recall@100,map"
!pip -q install sentence-transformers "transformers>=4.52.4,<5"
import transformers
assert transformers.__version__.startswith("4."), f"transformers {transformers.__version__}: restart the runtime"
print("ready | transformers", transformers.__version__)
'''

HELPERS = '''# 4a. Caches (graph-keyed) and the operator component tables.
import numpy as np
if os.path.isdir(f"{CACHE}/op_emb"):
    shutil.copytree(f"{CACHE}/op_emb", f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb", dirs_exist_ok=True)
    print("restored embedding caches from Drive")

def restore_index(g):
    src = f"{CACHE}/index/{g}"
    if not os.path.isdir(src):
        print(f"  {g}: no cached index on Drive (built on first use, 10-45 min)"); return
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
    print(f"  {g:22} operator components dense {zz['dense'].shape}")
shutil.copytree(f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb", f"{CACHE}/op_emb", dirs_exist_ok=True)
'''

SCORER = '''# 4b. The multi-view scorer (5d recipe). Restored from Drive when this field was run before,
# which is the normal case; trained otherwise. Its weights warm-start every arm identically.
SEM = f"{S4}/results/semantic_{DATASET}"
SEM_CKPT = f"{SEM}/params_semantic_mlp_fixedloss_{DATASET}.json"
SEM_POP  = f"{SEM}/popnet_semantic_mlp_fixedloss_{DATASET}.pt"
os.makedirs(SEM, exist_ok=True)
if os.path.isdir(SEM_DRIVE) and not os.path.exists(SEM_CKPT):
    shutil.copytree(SEM_DRIVE, SEM, dirs_exist_ok=True); print("scorer restored from", os.path.relpath(SEM_DRIVE, DRIVE))
if not (os.path.exists(SEM_CKPT) and os.path.exists(SEM_POP)):
    sh(f"python3 -u eval/semantic_scorer.py --dataset {DATASET} --model {OP_MODEL} "
       f"--arms dense,current,mlp --loss fixed --train_fit 0 --dev 300 --epochs 30 --patience 10 "
       f"--lr 1e-3 --weight_decay 1e-2 --mlp_hidden 16 --mlp_pop_joint 1 --pop_lambda 1.0 "
       f"--select_on loss --qbatch 32 --seed 0 --seeds 0,1,2 --out {SEM}", S4)
    shutil.copytree(SEM, f"{OUT_ROOT}/semantic", dirs_exist_ok=True)
    shutil.copytree(f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb", f"{CACHE}/op_emb", dirs_exist_ok=True)
for p in (SEM_CKPT, SEM_POP):
    assert os.path.exists(p), f"scorer not available: {p}"
_st = json.load(open(SEM_CKPT))
print(f"scorer: jmax={_st['jmax']} beta={_st['beta']:.4f} hidden={len(_st['net']['0.weight'])}")

SCORES = f"{OUT_ROOT}/scores"; os.makedirs(SCORES, exist_ok=True)
def score(key, pred, label):
    js, pq = f"{SCORES}/{key}_scores.json", f"{SCORES}/{key}_perquery.json"
    sh([sys.executable, "-u", "eval/score_sir4.py", "--pred", pred, "--queries", QUERIES,
        "--name", label, "--cols", COLS, "--json-out", js, "--per-query-out", pq], S4)
    return js
_sp = f"{SEM}/predictions_semantic_mlp_fixedloss_{DATASET}_test.json"
if os.path.exists(_sp):
    score("scorer_only", _sp, "Multi-View Semantic Scorer alone (reference)")

# scorer components aligned to each graph's document order
for s, g in (("train", TRAIN), ("test", TEST)):
    if not _sem_ok(semc(g)):
        sh(f"python3 -u precompute/precompute_semantic_components.py "
           f"--dataset {DATASET} --model {OP_MODEL} --graph {g} --split {s}", KGDIR)
    zz = np.load(semc(g), allow_pickle=True)
    print(f"  {g:22} H {tuple(int(x) for x in zz['h_shape'])}  Jmax={int(zz['Jmax'])}")
'''

TRAIN_ARMS = '''# 5. The arms. One environment, one variable per arm. Every arm is smoke-tested (1 epoch,
# 3 steps) so a wrong code path fails in minutes, not hours; the smoke log is asserted to
# contain the construction line of exactly the routing it should have.
BASE_ENV = dict(WANDB_MODE="disabled", HYDRA_FULL_ERROR="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                OPERATOR_COMPONENTS=opc(TRAIN), OPERATOR_COMPONENTS_TEST=opc(TEST),
                SEMANTIC_COMPONENTS=semc(TRAIN), SEMANTIC_COMPONENTS_TEST=semc(TEST),
                SEMANTIC_CKPT=SEM_CKPT, SEMANTIC_POPNET=SEM_POP, SEM_POP_LAMBDA="1.0",
                FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0",
                PER_GOLD="1", HARDNEG_GRAPH="50", STRAT_TEST=QUERIES)
# HEAD_LR: the CCMP head has its own optimiser group (CCMP_LR, section 3e). The two baseline
# heads get the SAME group and rate, otherwise "learned from the ranking loss only" would be
# confounded with "learned ten times slower".
HEAD_LR = "2e-3"
CCMP_ENV = dict(CCMP="1", CCMP_W="1.0", CCMP_NEG="64", CCMP_M="2000", CCMP_LR=HEAD_LR,
                CCMP_GATE="1", CCMP_GATE_NORM="1", CCMP_ETA="0.5",
                CCMP_POOL="256", CCMP_M_POS="512", CCMP_M_NEG="512",
                CCMP_RESIDUAL="0", CCMP_IDENTITY_W="1.0")
ARM_DEFS = {
    "control":       ("SciGraphIR (no routing)" if GRAPH == "v16sc" else f"SciGraphIR on the {GRAPH} graph (no CCMP)", {}),
    "astar":         ("+ A*Net-style routing",              dict(ROUTE="astar", ROUTE_K=ROUTE_K, CCMP_LR=HEAD_LR)),
    "attn":          ("+ RED-GNN-style attention",          dict(ROUTE="attn", CCMP_LR=HEAD_LR)),
    "ccmp":          ("+ CCMP",                             dict(CCMP_ENV)),
    "ccmp_residual": ("+ CCMP (residual targets)",          dict(CCMP_ENV, CCMP_RESIDUAL="1")),
}
EXPECT = {   # (must appear, must NOT appear) in the smoke log
    "control":       (["semantic='mlp' warm start"], ["[ccmp] responsibility head", "[route]", "[ccmp targets]"]),
    "astar":         (["[route] mode=astar"], ["[ccmp targets]", "[route] mode=attn"]),
    "attn":          (["[route] mode=attn"], ["[ccmp targets]", "[route] mode=astar", "[ccmp] responsibility head"]),
    "ccmp":          (["[ccmp] responsibility head", "[ccmp targets]", "residual off"], ["[route]"]),
    "ccmp_residual": (["[ccmp] responsibility head", "[ccmp targets]", "RESIDUAL on"], ["[route]"]),
}
for k in list(os.environ):
    if k.startswith(("CCMP", "ROUTE", "STRAT_", "CQIG", "RESID_", "MISS_W")): os.environ.pop(k)

def train_cmd(run_dir, epochs, seed, max_steps=None):
    return ("python -u -m gfmrag.workflow.sft_training "
            "--config-path config/gfm_reasoner --config-name sft_training_fusion "
            "text_emb_model=qwen3_st "
            f"seed={seed} "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{TRAIN}] datasets.valid_names=[{TEST}] "
            "model.semantic=mlp model.cqig=false "
            f"trainer.args.num_epoch={epochs} trainer.args.train_batch_size={BATCH} "
            f"+trainer.args.eval_batch_size={BATCH} "
            "+trainer.args.do_predict=true +trainer.args.predict_top_k=300 "
            + (f"trainer.args.max_steps_per_epoch={max_steps} " if max_steps else "")
            + f"hydra.run.dir={run_dir}")

def run_name(key, seed):
    return f"{DATASET}{TAG}_route_{key}_e{EPOCHS}_b{BATCH}_s{seed}"

_nq = len(json.load(open(f"{DATA_ROOT}/{DATASET}_train/raw/train.json")))
print(f"estimate: ~{_nq / BATCH * 1.05 / 60:.0f} min per epoch per arm at ~1 s/step, "
      f"x {EPOCHS} epochs x {len(ARM_KEYS)} arms x {len(SEEDS)} seed(s) "
      f"(the attention arm runs the unfused message path and is slower; the smoke shows by how much)")

RUN = {}          # (key, seed) -> drive dir with predictions + console.log
# Reused runs: another notebook's finished arm stands in for this one. Only for the first seed,
# and only if it holds predictions and a completed log; anything short of that is retrained.
def _reuse_check(d):
    """Which of the three conditions hold. A freshly mounted Drive can answer os.path.exists
    False for a file inside a directory it has not listed yet (this check passed on 7 Sep and
    failed on 8 Sep on the same directory), so list the directory first and retry briefly."""
    _pred, _log = f"{d}/predictions_{TEST}.json", f"{d}/console.log"
    for _try in range(4):
        try: _ls = sorted(os.listdir(d))
        except OSError as e: _ls = f"unlistable ({e})"
        have_pred, have_log = os.path.exists(_pred), os.path.exists(_log)
        done = have_log and (f"Epoch {EPOCHS} completed" in open(_log, errors="ignore").read())
        if have_pred and done: return True, _ls, (have_pred, have_log, done)
        time.sleep(5)
    return False, _ls, (have_pred, have_log, done)
for key, rel in REUSE.items():
    d = f"{DRIVE}/{rel}"
    ok, _ls, (have_pred, have_log, done) = _reuse_check(d)
    if key in ARM_KEYS and ok:
        RUN[(key, SEEDS[0])] = d; print(f"[reuse] {key:8} <- {rel}")
    else:
        # Stop rather than fall through to a retrain: on TOMATO that is a fourteen-hour mistake
        # caused by a typo in a directory name. Fix REUSE (or remove the arm from it) and rerun.
        why = ("not in ARM_KEYS" if key not in ARM_KEYS else
               f"predictions_{TEST}.json {'present' if have_pred else 'MISSING'}, console.log "
               f"{'present' if have_log else 'MISSING'}, 'Epoch {EPOCHS} completed' "
               f"{'found' if done else 'NOT FOUND'}")
        raise AssertionError(f"[reuse] {key} not usable ({why}): {d}\\n  contents: {_ls}")
def train_arm(key, seed, force=False):
    """Smoke, then the full run, into OUT_ROOT/run_name(key, seed). A finished run on Drive is a
    [skip] unless force=True (the validation section uses force with a fresh seed). Returns the
    Drive dir holding predictions + console.log."""
    label, extra = ARM_DEFS[key]
    name = run_name(key, seed)
    local, drive_ = f"{RUNS}/{name}", f"{OUT_ROOT}/{name}"
    pred_drive = f"{drive_}/predictions_{TEST}.json"
    env_ = dict(BASE_ENV, **extra)
    print(f"\\n==================== {label}  (seed {seed}) ====================")
    if not force and os.path.exists(pred_drive) and os.path.exists(f"{drive_}/console.log") and \\
            f"Epoch {EPOCHS} completed" in open(f"{drive_}/console.log", errors="ignore").read():
        print(f"[skip] finished: {os.path.relpath(drive_, DRIVE)}"); return drive_
    smoke = f"{local}_smoke"; os.makedirs(smoke, exist_ok=True)
    t0 = time.time()
    rc = sh(train_cmd(smoke, 1, seed, 3).replace("+trainer.args.do_predict=true", "+trainer.args.do_predict=false"),
            "/content/gfm-rag", extra=env_, log=f"{smoke}/console.log", check=False)
    log = open(f"{smoke}/console.log", errors="ignore").read()
    assert rc == 0, f"{label}: smoke failed (exit {rc}); read {smoke}/console.log"
    must, must_not = EXPECT[key]
    for s_ in must:     assert s_ in log, f"{label}: smoke log lacks {s_!r}"
    for s_ in must_not: assert s_ not in log, f"{label}: smoke log contains {s_!r}, wrong code path"
    assert "semantic='mlp' warm start" in log, "multi-view scorer not constructed"
    for g in (TRAIN, TEST): save_index(g)
    print(f"smoke passed in {(time.time() - t0) / 60:.1f} min")
    os.makedirs(local, exist_ok=True); os.makedirs(drive_, exist_ok=True)
    json.dump({"arm": key, "label": label, "dataset": DATASET, "train": TRAIN, "valid": TEST, "epochs": EPOCHS,
               "batch": BATCH, "seed": seed, "env": extra, "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{local}/arm.json", "w"), indent=1)
    finish = start_sync(local, drive_, every=600)
    try:
        rc = sh(train_cmd(local, EPOCHS, seed), "/content/gfm-rag", extra=env_, log=f"{local}/console.log", check=False)
    finally:
        finish()
    log = open(f"{local}/console.log", errors="ignore").read()
    assert rc == 0, f"{label}: training failed, exit {rc}"
    for s_ in must:     assert s_ in log, f"{label}: run log lacks {s_!r}"
    assert os.path.exists(f"{local}/predictions_{TEST}.json"), "no predictions written"
    sync_dir(local, drive_); return drive_

# ARM_KEYS is the RUN order (put the arm you want trained first at the front); tables use the
# ladder order in ARM_LADDER so the printed rows do not depend on it.
for seed in SEEDS:
    for key in ARM_KEYS:
        if (key, seed) not in RUN:
            RUN[(key, seed)] = train_arm(key, seed)
print("\\nruns:", {f"{k}/s{s}": os.path.relpath(d, DRIVE) for (k, s), d in RUN.items()})
'''

RESULTS = '''# 6. Score every arm on the fused ranking, parse the graph-channel diagnostics from each run's
# final evaluation, and bootstrap the routing arms against the control and against CCMP.
def final_diag(log_path):
    """The trainer's [diag]/[prior]/[bucket] lines from the FINAL evaluation (best epoch)."""
    log = open(log_path, errors="ignore").read()
    tail = log[log.rfind("Running final evaluation"):] if "Running final evaluation" in log else log
    d = {}
    m = re.search(r"\\[diag\\] nDCG@5\\s+semantic ([\\d.]+) \\| graph ([\\d.]+) \\| fused ([\\d.]+)\\s+gamma mean ([\\d.]+)", tail)
    if m: d.update(sem=float(m[1]), graph=float(m[2]), fused_diag=float(m[3]), gamma=float(m[4]))
    m = re.search(r"\\[prior\\] parameter-free walk nDCG@5 ([\\d.]+)\\s+trained graph ([\\d.]+)", tail)
    if m: d.update(walk=float(m[1]))
    m = re.search(r"\\[bucket\\][^\\n]*1k\\+ ([\\d.]+) \\(n=(\\d+)\\)", tail)
    if m: d.update(buried=float(m[1]), buried_n=int(m[2]))
    m = re.search(r"\\[hit\\] gold rate by rank -- graph 1..3: ([\\d.]+)", tail)
    if m: d.update(graph_at1=float(m[1]))
    best = re.findall(r"New best model! document_ndcg@5: ([\\d.]+) at epoch (\\d+)", log)
    if best: d.update(best_epoch=int(best[-1][1]))
    return d

ROWS = {}
for (key, seed), drive_ in RUN.items():
    rk = f"{key}_s{seed}"
    score(rk, f"{drive_}/predictions_{TEST}.json", f"{ARM_DEFS[key][0]} seed {seed}")
    ROWS[(key, seed)] = dict(json.load(open(f"{SCORES}/{rk}_scores.json")), diag=final_diag(f"{drive_}/console.log"))

SLICES = [s for s in ("all", "same", "cross") if any(s in r for r in ROWS.values())]
lines = []
def out(s=""): print(s); lines.append(s)
out(f"# Routing comparison on {DATASET} test ({len(json.load(open(QUERIES)))} queries), {EPOCHS} epochs, batch {BATCH}\\n")
hdr = ["arm", "seed"] + [f"fused nDCG@5 {s}" for s in SLICES] + ["fused R@5 all", "graph nDCG@5", "walk nDCG@5", "graph - walk",
                                                                   "graph R@10 on buried golds (1k+)", "graph@1", "gamma", "best ep"]
out("| " + " | ".join(hdr) + " |"); out("|" + "---|" * len(hdr))
if os.path.exists(f"{SCORES}/scorer_only_scores.json"):
    s0 = json.load(open(f"{SCORES}/scorer_only_scores.json"))
    out("| scorer alone (reference) | - | " + " | ".join(f"{100 * s0[s]['ndcg@5']:.2f}" if s in s0 else "--" for s in SLICES)
        + f" | {100 * s0['all']['recall@5']:.2f} | -- | -- | -- | -- | -- | -- | -- |")
for key in ARM_LADDER:
    for seed in SEEDS:
        r = ROWS.get((key, seed))
        if r is None: continue
        d = r["diag"]
        f = lambda x, k=1: "--" if x is None else (f"{100 * x:.2f}" if k else f"{x:.3f}")
        gw = None if not ("graph" in d and "walk" in d) else d["graph"] - d["walk"]
        out(f"| {ARM_DEFS[key][0]} | {seed} | " + " | ".join(f"{100 * r[s]['ndcg@5']:.2f}" for s in SLICES)
            + f" | {100 * r['all']['recall@5']:.2f} | {f(d.get('graph'))} | {f(d.get('walk'))} | "
            + ("--" if gw is None else f"{100 * gw:+.2f}") + f" | {f(d.get('buried'))} | {f(d.get('graph_at1'))} | "
            + f(d.get('gamma'), 0) + f" | {d.get('best_epoch', '--')} |")
out()
out("fused = the ranking SciGraphIR outputs (scorer + gated graph). graph = the graph channel alone, its own nDCG@5 at the")
out("best epoch. walk = a zero-parameter random walk on the same seeds, the graph channel's prior. buried = the graph's")
out("recall@10 of golds the scorer ranked past 1,000, the queries the graph exists to fix. All numbers are percentages.")

# Paired bootstrap on the fused ranking: each arm vs the control, each routing baseline vs CCMP.
BOOT = f"{OUT_ROOT}/bootstrap"; os.makedirs(BOOT, exist_ok=True)
def paired(a, b, tag, slice_="all"):
    pa, pb = f"{SCORES}/{a}_perquery.json", f"{SCORES}/{b}_perquery.json"
    if not (os.path.exists(pa) and os.path.exists(pb)): return
    js = f"{BOOT}/{tag}_{slice_}.json"
    sh([sys.executable, "-u", "transfer/paired_bootstrap.py", "--a", pa, "--b", pb, "--a-name", a, "--b-name", b,
        "--metrics", "ndcg@5,recall@5,recall@10", "--slice", slice_, "--iters", "10000", "--json-out", js], S4, check=False)
out("\\n## Paired bootstrap (fused ranking, 10,000 resamples)\\n")
for seed in SEEDS:
    c = f"control_s{seed}"
    for key in ARM_KEYS:
        if key == "control" or (key, seed) not in ROWS: continue
        for sl in SLICES:
            paired(f"{key}_s{seed}", c, f"{key}_vs_control_s{seed}", sl)
    for ck in ("ccmp", "ccmp_residual"):            # each CCMP variant vs each implicit-routing baseline
        if (ck, seed) not in ROWS: continue
        for key in ("astar", "attn"):
            if (key, seed) in ROWS:
                for sl in SLICES:
                    paired(f"{ck}_s{seed}", f"{key}_s{seed}", f"{ck}_vs_{key}_s{seed}", sl)
for p in sorted(glob.glob(f"{BOOT}/*.json")):
    try:
        j = json.load(open(p)); m = j.get("metrics", {})
        out(f"- {os.path.basename(p)[:-5]}: " + "; ".join(
            f"{k} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
            + (" (CI spans 0)" if v.get("crosses_zero") else " (resolved)") for k, v in m.items()))
    except Exception as e:
        out(f"- {os.path.basename(p)}: unreadable ({e})")
open(f"{OUT_ROOT}/table_routing_{DATASET}.md", "w").write("\\n".join(lines)); print("\\nwrote", f"{OUT_ROOT}/table_routing_{DATASET}.md")
'''

THESIS_TABLE = '''# 7. Thesis table (LaTeX). The ladder in ARM_LADDER order: fused nDCG@5 same / cross with the
# relative gap, Recall@5 same / cross, and the paired-bootstrap difference against the no-routing
# control on the cross slice (nDCG@5 points, 95% CI; * = CI excludes zero). Written next to the
# markdown table on Drive.
BS = chr(92)
LAB = {"control": BS + "textsc{SciGraphIR} (no routing)" if GRAPH == "v16sc" else BS + "textsc{SciGraphIR} on the " + GRAPH + " graph (no CCMP)",
       "astar": BS + "quad $+$ A*Net-style routing",
       "attn": BS + "quad $+$ RED-GNN-style attention",
       "ccmp": BS + "quad $+$ CCMP",
       "ccmp_residual": BS + "quad $+$ CCMP (residual targets)"}
_seed = SEEDS[0]
def ci_vs_control(key, sl="cross", m="ndcg@5"):
    p = f"{BOOT}/{key}_vs_control_s{_seed}_{sl}.json"
    if not os.path.exists(p): return "--"
    v = json.load(open(p))["metrics"].get(m)
    if not v: return "--"
    lo, hi = v["ci95"]
    return f"${100 * v['diff']:+.2f}$ [{100 * lo:+.2f}, {100 * hi:+.2f}]" + ("" if v["crosses_zero"] else "$^{*}$")
def pct(x): return f"${100 * x:+.2f}" + BS + "%$"
EOL = " " + BS + BS
tex = [BS + "begin{table}[t]", BS + "centering", BS + "small", BS + "setlength{" + BS + "tabcolsep}{5pt}",
       BS + "renewcommand{" + BS + "arraystretch}{1.2}",
       BS + "begin{tabular}{@{}l cc c cc c@{}}", BS + "toprule",
       "& " + BS + "multicolumn{2}{c}{nDCG@5} & & " + BS + "multicolumn{2}{c}{Recall@5} & $" + BS + "Delta$ vs.\\\\ no routing" + EOL,
       BS + "cmidrule(lr){2-3}" + BS + "cmidrule(lr){5-6}",
       BS + "textbf{Method} & Same & Cross & $" + BS + "Delta$ & Same & Cross & (cross nDCG@5, 95" + BS + "% CI)" + EOL,
       BS + "midrule"]
# Single-stratum datasets (MIR) have no cross slice: fall back to the whole test set in one column set.
_ctrl = ROWS.get(("control", _seed)) or next(iter(ROWS.values()))
HAS_CROSS = "cross" in _ctrl
SL = "cross" if HAS_CROSS else "all"
if not HAS_CROSS:
    tex[5] = BS + "begin{tabular}{@{}l cc c c@{}}"
    tex[7] = "& nDCG@5 & Recall@5 & MRR & $" + BS + "Delta$ vs.\\ control" + EOL
    tex[8] = BS + "cmidrule(lr){2-4}"
    tex[9] = BS + "textbf{Method} & all & all & all & (nDCG@5, 95" + BS + "% CI)" + EOL
def _row(label, r, ci):
    if HAS_CROSS:
        ns, nc = r["same"]["ndcg@5"], r["cross"]["ndcg@5"]
        return (f"{label} & {100 * ns:.2f} & {100 * nc:.2f} & {pct((nc - ns) / ns)} & "
                f"{100 * r['same']['recall@5']:.2f} & {100 * r['cross']['recall@5']:.2f} & {ci}" + EOL)
    a = r["all"]
    return f"{label} & {100 * a['ndcg@5']:.2f} & {100 * a['recall@5']:.2f} & {100 * a.get('mrr', float('nan')):.2f} & {ci}" + EOL
if os.path.exists(f"{SCORES}/scorer_only_scores.json"):
    tex.append(_row("Multi-View Semantic Scorer", json.load(open(f"{SCORES}/scorer_only_scores.json")), "--"))
for key in ARM_LADDER:
    r = ROWS.get((key, _seed))
    if r is None: continue
    tex.append(_row(LAB[key], r, "--" if key == "control" else ci_vs_control(key, sl=SL)))
tex += [BS + "bottomrule", BS + "end{tabular}",
        BS + "caption{" + ("Routing ablation on " if GRAPH == "v16sc" else "Merged-graph arms on ") + DATASET.upper() + " (nDCG@5 and Recall@5, " + BS + "%). All arms share the "
        "multi-view scorer, graph reasoner, fusion gate and ranking loss and differ only in where the routing signal comes "
        "from. " + ("$" + BS + "Delta$ is the relative same-to-cross reduction. " if HAS_CROSS else "") + "The last column is the paired bootstrap over "
        "queries against the control; $^{*}$ marks a 95" + BS + "% CI that excludes zero.}",
        BS + "label{tab:routing-" + DATASET + "}", BS + "end{table}"]
open(f"{OUT_ROOT}/table_routing_{DATASET}.tex", "w").write("\\n".join(tex) + "\\n")
print("\\n".join(tex)); print("\\nwrote", f"{OUT_ROOT}/table_routing_{DATASET}.tex")
'''

VALIDATE_CELL = '''# 8. Validation reruns (after the table): retrain the arms in VALIDATE with VAL_SEED and print
# them next to the runs the table used. Same engine, same data, different seed, so the spread is
# the trainer's run-to-run noise, which the differences in the table must be judged against.
if not VALIDATE:
    print("no validation arms requested")
else:
    VAL = {}
    for key in VALIDATE:
        assert key in ARM_DEFS, f"unknown arm {key}"
        VAL[key] = train_arm(key, VAL_SEED)
        score(f"{key}_s{VAL_SEED}", f"{VAL[key]}/predictions_{TEST}.json", f"{ARM_DEFS[key][0]} seed {VAL_SEED}")
    print(f"\\n| arm | table run (seed {SEEDS[0]}) same / cross nDCG@5 | validation (seed {VAL_SEED}) same / cross | same diff | cross diff |")
    print("|---|--:|--:|--:|--:|")
    for key in VALIDATE:
        a = ROWS.get((key, SEEDS[0])); b = json.load(open(f"{SCORES}/{key}_s{VAL_SEED}_scores.json"))
        if a is None: continue
        print(f"| {ARM_DEFS[key][0]} | {100 * a['same']['ndcg@5']:.2f} / {100 * a['cross']['ndcg@5']:.2f} | "
              f"{100 * b['same']['ndcg@5']:.2f} / {100 * b['cross']['ndcg@5']:.2f} | "
              f"{100 * (b['same']['ndcg@5'] - a['same']['ndcg@5']):+.2f} | {100 * (b['cross']['ndcg@5'] - a['cross']['ndcg@5']):+.2f} |")
    for key in VALIDATE:
        for sl in SLICES:
            paired(f"{key}_s{VAL_SEED}", f"{key}_s{SEEDS[0]}", f"{key}_val_s{VAL_SEED}_vs_s{SEEDS[0]}", sl)
    if all(k in VAL for k in ("control", "ccmp_residual")):
        for sl in SLICES:
            paired(f"ccmp_residual_s{VAL_SEED}", f"control_s{VAL_SEED}", f"ccmp_residual_vs_control_s{VAL_SEED}", sl)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sir4_matsci",
                    help="tomato or a SIR-4 field (sir4_matsci, sir4_cs, sir4_biology, sir4_physics)")
    ap.add_argument("--arms", default="control,astar,attn,ccmp",
                    help=f"comma list from {ARM_ORDER}")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seeds", default="1024", help="trainer seeds, comma list; 1024 is the config default")
    ap.add_argument("--route-k", type=int, default=1024, help="astar: nodes kept per layer")
    ap.add_argument("--reuse", default="",
                    help="comma list of arm=Drive-relative run dir to stand in for finished arms, e.g. "
                         "control=outputs/mir/mir_qwenmlp_graph_e10_b2,ccmp=outputs/mir/mir_qwenmlp_ccmp_e10_b2")
    ap.add_argument("--validate", default="",
                    help="comma list of arms to RETRAIN with --val-seed after the table, as a run-to-run check, "
                         "e.g. control,ccmp_residual")
    ap.add_argument("--val-seed", type=int, default=7)
    ap.add_argument("--graph-suffix", default="v16sc", help="graph directory suffix: v16sc (SciAffordGraph) or hyb (merged graph)")
    ap.add_argument("--extra-bundle", default="", help="comma list of add-on zips on Drive to unpack after the bundle")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    assert all(x in ARM_ORDER for x in arms), f"unknown arm in {arms}; choose from {ARM_ORDER}"
    validate = [x.strip() for x in a.validate.split(",") if x.strip()]
    assert all(x in arms for x in validate), f"--validate names an arm not in --arms: {validate}"
    seeds = [int(x) for x in a.seeds.split(",")]
    assert a.val_seed not in seeds, "--val-seed must differ from --seeds"
    reuse = dict(x.split("=", 1) for x in a.reuse.split(",") if x.strip())
    assert all(k in ARM_ORDER for k in reuse), f"--reuse names an unknown arm: {list(reuse)}"
    sem_drive = ("outputs/tomato_ablations_v1/tomato/semantic" if a.dataset == "tomato"
                 else f"outputs/{a.dataset}/semantic")

    src = json.load(open(rb.SRC_NB))["cells"]
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fusion_files = {f"/content/gfm-rag/{rel}": open(f"{FORK}/{rel}").read() for rel in FUSION_REL}
    assert not any("'''" in v for v in fusion_files.values())
    assert "route_mode" in fusion_files["/content/gfm-rag/gfmrag/models/fusion_reasoner.py"], "engine lacks the ROUTE knob"
    assert "_route_attention" in fusion_files["/content/gfm-rag/gfmrag/models/ultra/models.py"], "engine lacks the attention hook"
    assert "per-query weights" in fusion_files["/content/gfm-rag/gfmrag/models/ultra/layers.py"], "layers.py lacks 2-D edge weights"
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
        "print('fusion files ready (routing baselines included)')\n")
    overlay = {rel: open(f"{CARGO}/{rel}").read() for rel in rb.OVERLAY_REL}
    assert not any("'''" in v for v in overlay.values())

    # The CCMP_LR patch cell, extended so the attention head joins the separate lr group.
    old_keys = '("resp_proj", "resp_emb", "resp_head")'
    new_keys = '("resp_proj", "resp_emb", "resp_head", "attn_node", "attn_rel", "attn_query", "attn_emb", "attn_out")'
    engine = []
    for i in rb.ENGINE_CELLS:
        c = src[i]
        s = "".join(c["source"])
        if "_ccmp_lr = os.environ.get" in s:
            assert s.count(old_keys) == 1, "CCMP_LR patch cell changed; update the key tuple anchor"
            c = code(s.replace(old_keys, new_keys).replace(
                "### 3e", "### 3e").replace("CCMP_LR parameter group already patched",
                                            "CCMP_LR parameter group already patched"))
        if a.dataset.startswith("sir4") and GUARD in "".join(c["source"]):
            c = code("".join(c["source"]).replace(GUARD, ""))     # TOMATO-only guard, inverted on sir4 graphs
        engine.append(files_cell if i == 7 else c)
    assert any(new_keys in "".join(c["source"]) for c in engine), "lr-group cell not extended"

    header = HEADER.replace("__DATASET__", a.dataset)
    if a.graph_suffix != "v16sc":
        header = (f"# SciGraphIR on the merged graph (`{a.dataset}_*_{a.graph_suffix}`): {' vs '.join(arms)}\n\n"
                  f"Same recipe as the routing notebook, but every arm trains and evaluates on the `{a.graph_suffix}` graph "
                  f"(prep/build_hybrid_graph.py). Results: section 5 prints each arm's training log with the per-epoch "
                  f"`[diag]` / `[prior]` lines; section 6 prints the table (fused nDCG@5 / R@5 same and cross, graph channel, "
                  f"walk prior, gamma) with the paired bootstrap and writes `outputs/routing_{a.graph_suffix}/{a.dataset}/table_routing_{a.dataset}.md`; "
                  f"section 7 writes the LaTeX next to it.\n\n" + header.split("\n", 1)[1])
    cells = [md(header), md("## 1. GPU + Drive + paths"),
             code(PATHS.replace("__DATASET__", a.dataset).replace("__SEM_DRIVE__", sem_drive)
                  .replace("__GRAPH__", a.graph_suffix).replace("__EXTRA__", json.dumps([x.strip() for x in a.extra_bundle.split(",") if x.strip()]))
                  .replace("__EPOCHS__", str(a.epochs)).replace("__BATCH__", str(a.batch))
                  .replace("__SEEDS__", json.dumps(seeds)).replace("__ARMS__", json.dumps(arms))
                  .replace("__ARM_ORDER__", json.dumps(ARM_ORDER)).replace("__VALIDATE__", json.dumps(validate))
                  .replace("__VAL_SEED__", str(a.val_seed))
                  .replace("__ROUTE_K__", str(a.route_k)).replace("__REUSE__", json.dumps(reuse))),
             md("## 2. Unpack + install the current scripts"),
             code(UNPACK.replace("__OVERLAY__", json.dumps(overlay)).replace("__BUILT__", built)),
             md("## 2b. Qwen3-Embedding-0.6B (cached on Drive)"), src[rb.QWEN_CELLS[1]],
             md("## 3. Engine + fusion sources\n*(cloned from `tomato_ccmp_ablation.ipynb`; the fusion-source blob is "
                "regenerated from the repo and now carries `ultra/layers.py`; the CCMP_LR cell also covers the attention head)*")]
    cells += engine
    cells += [md("## 4. Multi-view scorer and component tables"), code(HELPERS), code(SCORER),
              md("## 5. The arms\nEach arm: smoke (3 steps, asserts the right routing was built), then the full run. "
                 "Local run, ten-minute Drive sync, finished arms skip on rerun."),
              code(TRAIN_ARMS), md("## 6. Table + paired bootstrap"), code(RESULTS),
              md("## 7. Thesis table (LaTeX)"), code(THESIS_TABLE),
              md("## 8. Validation reruns\nRuns AFTER the table is written. Retrains the arms in `VALIDATE` with a "
                 "fresh seed and prints them next to the table's runs, with a paired bootstrap between the two seeds."),
              code(VALIDATE_CELL)]
    out = a.out or f"{ROOT}/colab_cells/colab_routing_{a.dataset}{'' if a.graph_suffix == 'v16sc' else '_' + a.graph_suffix}.ipynb"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(cells)} cells | dataset {a.dataset} | arms {arms} | seeds {seeds} | "
          f"epochs {a.epochs} batch {a.batch} | route_k {a.route_k} | reuse {reuse or '-'} | "
          f"validate {validate or '-'} (seed {a.val_seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
