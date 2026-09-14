"""
build_joint_notebook.py -- joint Physics+Biology training, zero-shot on CS + MatSci.

THE EXPERIMENT. One model trained on Physics and Biology together, then evaluated
WITHOUT adaptation on the Computer Science and Materials Science test graphs. It
asks whether multi-domain training transfers better than the single-domain models
you already have, and whether any gain lands on cross-field and complete-set
retrieval rather than on the easy queries.

    {Physics, Biology}_train  -->  {Computer Science, Materials Science}_test

HOW "JOINT" IS IMPLEMENTED, AND WHY. GraphDatasetLoader already accepts a LIST of
dataset names: it keeps one graph resident at a time and walks them, so one set of
weights receives gradients from both domains. Peak memory stays at the LARGER
single graph rather than their sum, which is what keeps train_batch_size at 2
instead of forcing 1.

The alternative -- physically merging the two graphs -- was measured and rejected
for this run: physics and biology share 1,113 of 312,390 non-document nodes
(0.9%), so a merged graph is two near-disjoint components, not a connected
multi-domain graph. What merging WOULD add is a shared negative pool (a physics
query competing against biology documents), at roughly double the wall clock and
with real OOM risk on a 339k-node graph. That is a separate experiment, not this
one. State the limitation rather than implying the merge happened.

LEAKAGE CONTROL. valid_names is the Physics and Biology TEST graphs, so checkpoint
selection on document_ndcg@5 never touches CS or MatSci; those two are loaded only
after training ends. Consequence to state in the write-up: the Physics/Biology test
numbers from this model are DEV numbers. Only CS and MatSci are held out.

REQUIRES the fusion_reasoner.py patch that lets OPERATOR_COMPONENTS take a
comma-separated list. Physics and biology have different document counts (10,349
vs 15,588) so one merged table cannot serve both, and the shape assert in
_operator would abort on the first batch of the second graph. The overlay cell
below ships the patched file, and a capability check fails fast if it did not land.

Usage
-----
    python3 prep/build_joint_notebook.py
    python3 prep/build_joint_notebook.py --epochs 10 --batch 2
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SETUP_CELLS = 16

TRAIN_DOMS = ["physics", "biology"]
EVAL_DOMS = ["cs", "matsci"]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(True)}


HELPERS = '''import os, sys, time, subprocess, glob, json, shutil, zipfile
KGDIR = f"{CARGO_ROOT}/kg-construction"
env = dict(os.environ, CARGO_ROOT=CARGO_ROOT, PYTHONUNBUFFERED="1")

def sh(cmd, cwd, extra=None, log=None):
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=dict(env, **(extra or {})), shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    fh = open(log, "w") if log else None
    while True:
        c = os.read(p.stdout.fileno(), 8192)
        if not c: break
        s = c.decode("utf-8", "replace")
        sys.stdout.write(s); sys.stdout.flush()
        if fh: fh.write(s); fh.flush()
    p.wait()
    if fh: fh.close()
    print(f"\\n[{time.time()-t0:.0f}s, exit {p.returncode}]")
    return p.returncode

TRAIN_DOMS = __TRAIN__          # trained on
EVAL_DOMS  = __EVAL__           # held out, zero-shot only
OP_SLUG    = "__OPSLUG__"       # the encoder behind the operator components
EPOCHS     = __EPOCHS__
BATCH      = __BATCH__
RUN        = f"{DRIVE}/outputs/_joint___TAG__/joint_qwenop_ep{EPOCHS}_b{BATCH}"
os.makedirs(RUN, exist_ok=True)
print("train:", TRAIN_DOMS, "\\neval (held out):", EVAL_DOMS,
      "\\nslug:", OP_SLUG or "(BGE)", "\\nrun dir:", RUN)
'''

UNPACK = '''# All four bundles into one CARGO_ROOT. The training domains need their TRAIN and
# TEST graphs; the held-out domains need only their TEST graphs.
#
# FORCE=True re-extracts even when the graphs are already present. Needed after
# re-uploading a bundle: the skip below keys on the GRAPHS, but every bundle also
# carries the shared sir4-retrieval/eval tree, so a newer zip left unextracted
# silently keeps an older score_sir4.py in play.
FORCE = False

os.makedirs(CARGO_ROOT, exist_ok=True)
NEED = {d: ("train", "test") for d in TRAIN_DOMS}
NEED.update({d: ("test",) for d in EVAL_DOMS})

# CHECK EVERY ZIP FIRST. Extracting one at a time means a bad bundle surfaces only
# after the good ones have already been written, and BadZipFile does not say which
# file or why. An interrupted Drive upload leaves a short, unreadable zip behind.
bad = []
for d in NEED:
    z = f"{DRIVE}/sir4_{d}_bundle.zip"
    if not os.path.exists(z):
        bad.append(f"  {d:9} MISSING            {z}"); continue
    mb = os.path.getsize(z) / 1e6
    if not zipfile.is_zipfile(z):
        bad.append(f"  {d:9} NOT A ZIP  {mb:8.1f} MB  (upload was interrupted; re-upload it)")
        continue
    try:
        n = len(zipfile.ZipFile(z).namelist())
        print(f"  {d:9} ok         {mb:8.1f} MB  {n} entries")
    except Exception as e:
        bad.append(f"  {d:9} CORRUPT    {mb:8.1f} MB  {e}")
assert not bad, ("bundle problems, re-upload these from ~/Desktop/CARGO/sir4-retrieval/"
                 " and rerun this cell:\\n" + "\\n".join(bad))

for d, splits in NEED.items():
    have = all(os.path.exists(f"{DATA_ROOT}/sir4_{d}_{s}_v16sc/processed/stage1/nodes.csv")
               for s in splits)
    if have and not FORCE:
        print(f"{d:9} already unpacked"); continue
    zipfile.ZipFile(f"{DRIVE}/sir4_{d}_bundle.zip").extractall(CARGO_ROOT)
    print(f"{d:9} {'re-extracted' if have else 'unpacked'}")

# Fail here rather than three cells later inside a dataloader worker.
for d, splits in NEED.items():
    for s in splits:
        g = f"{DATA_ROOT}/sir4_{d}_{s}_v16sc/processed/stage1/nodes.csv"
        assert os.path.exists(g), f"{d} {s} graph missing after unpack: {g}"

# The loader opens {graph}/raw/documents.json, which the graph build never writes.
for d, splits in NEED.items():
    for s in splits:
        dst = f"{DATA_ROOT}/sir4_{d}_{s}_v16sc/raw"
        os.makedirs(dst, exist_ok=True)
        if not os.path.exists(f"{dst}/documents.json"):
            shutil.copy(f"{DATA_ROOT}/sir4_{d}_{s}/raw/documents.json",
                        f"{dst}/documents.json")
print("\\ngraphs ready")
'''

COMPONENTS = '''# Operator components for every graph this run touches: six in total.
# Restored from the training runs' Drive cache when present, computed otherwise.
# Each is [queries x documents] for ITS OWN corpus, which is exactly why the
# multi-table patch is needed: physics and biology have different column counts.
def comp_path(d, s):
    return f"{DATA_ROOT}/sir4_{d}_{s}_v16sc/operator_components{OP_SLUG}.npz"

for d, splits in NEED.items():
    for s in splits:
        g, npz = f"sir4_{d}_{s}_v16sc", comp_path(d, s)
        if os.path.exists(npz):
            print(f"{d:9} {s:6} present"); continue
        cached = f"{DRIVE}/outputs/sir4_{d}/cache/{g}_operator_components{OP_SLUG}.npz"
        if os.path.exists(cached):
            shutil.copy(cached, npz); print(f"{d:9} {s:6} restored from Drive"); continue
        print(f"{d:9} {s:6} computing (encodes the whole split, several minutes) ...")
        rc = sh(f"python3 -u experiments/probe_greasoner/precompute_operator_components.py "
                f"--dataset sir4_{d} --graph {g} --split {s}"
                + (" --model /content/qwen3" if OP_SLUG else ""), KGDIR)
        assert rc == 0 and os.path.exists(npz), f"components failed for {d} {s}"
        # Cache immediately: a disconnect after six of these is an expensive restart.
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        shutil.copy(npz, cached)

# ENCODER CHECK. Every npz records the encoder that produced it. A BGE table fed to
# a Qwen3 run has identical shapes and would train happily on the wrong features,
# so shape checks cannot catch it; only this can.
import numpy as np
print()
want = "qwen" if OP_SLUG else "bge"
for d, splits in NEED.items():
    for s in splits:
        z = np.load(comp_path(d, s), allow_pickle=True)
        enc = str(z["encoder"]) if "encoder" in z else "(unrecorded)"
        assert want in enc.lower(), (
            f"{comp_path(d, s)} was built with {enc!r}, but OP_SLUG={OP_SLUG!r} "
            f"expects {want}. Delete it and let this cell recompute.")
        print(f"{d:9} {s:6} dense {z['dense'].shape}  {len(z['query_ids'])} queries  {enc}")
'''

CAPABILITY = '''# FAIL FAST IF THE OVERLAY DID NOT LAND. Joint training needs the fusion_reasoner
# that accepts a LIST of component files. Without it the run dies on the first
# batch of the second graph with a shape assert, an hour in.
p = "/content/gfm-rag/gfmrag/models/fusion_reasoner.py"
src = open(p).read()
assert "MULTI-CORPUS TRAINING" in src and 'f"{split}#{j}"' in src, (
    "fusion_reasoner.py on this runtime is the OLD single-table version. Rebuild "
    "the bundle from the current repo and re-upload, then rerun the overlay cell.")
print("fusion_reasoner: multi-corpus components supported")

# Query ids must be disjoint across the two training corpora, or the loader would
# route half a batch to the wrong table. The model asserts this too; checking here
# turns an hour-in crash into a five-second one.
import json as _json
ids = {}
for d in TRAIN_DOMS:
    ids[d] = {q["id"] for q in _json.load(open(f"{DATA_ROOT}/sir4_{d}_train/raw/train.json"))}
    print(f"{d:9} {len(ids[d])} train queries")
import itertools as _it
for _a, _b in _it.combinations(TRAIN_DOMS, 2):      # every pair, not just the first two
    ov = ids[_a] & ids[_b]
    assert not ov, (f"{len(ov)} query ids shared between {_a} and {_b}, "
                    f"e.g. {sorted(ov)[:3]}")
print(f"training query ids are disjoint across all {len(TRAIN_DOMS)} corpora: ok")

# And the held-out domains must not leak into training.
for d in EVAL_DOMS:
    held = {q["id"] for q in _json.load(open(f"{DATA_ROOT}/sir4_{d}_test/raw/test.json"))}
    for t in TRAIN_DOMS:
        bad = held & ids[t]
        assert not bad, f"{len(bad)} {d} test queries appear in {t} training data"
    print(f"{d:9} {len(held)} held-out queries, none in training: ok")
'''

SMOKE = '''# SMOKE TEST: the same command, 1 epoch, 3 steps per graph. Roughly 10 minutes.
#
# It is here because the dataset SWITCH is the only genuinely new thing in this run.
# Everything up to the first batch of the second graph is a path the single-domain
# runs already exercise nightly; the switch is not. This cell reaches it in minutes
# instead of finding out forty minutes into a seven-hour job.
#
# What it proves: both component tables load, both graphs build, the loader hands
# over from one to the other, and validation aggregates across two datasets. What it
# does not prove: anything about quality. Ignore every number it prints.
SMOKE_DIR = f"{RUN}_smoke"
os.makedirs(SMOKE_DIR, exist_ok=True)
TRAIN_G = [f"sir4_{d}_train_v16sc" for d in TRAIN_DOMS]
VALID_G = [f"sir4_{d}_test_v16sc"  for d in TRAIN_DOMS]
COMP_TR = ",".join(comp_path(d, "train") for d in TRAIN_DOMS)
COMP_TE = ",".join(comp_path(d, "test")  for d in TRAIN_DOMS)

rc = sh("python -u -m gfmrag.workflow.sft_training "
        "--config-path config/gfm_reasoner --config-name sft_training_fusion "
        "text_emb_model=qwen3_st "
        f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
        f"datasets.train_names=[{','.join(TRAIN_G)}] "
        f"datasets.valid_names=[{','.join(VALID_G)}] "
        "datasets.max_datasets_in_memory=1 "
        "trainer.args.num_epoch=1 trainer.args.max_steps_per_epoch=3 "
        f"trainer.args.train_batch_size={BATCH} "
        f"+trainer.args.eval_batch_size={BATCH} "
        f"hydra.run.dir={SMOKE_DIR}",
        "/content/gfm-rag", log=f"{SMOKE_DIR}/console.log", extra=dict(
            WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            OPERATOR_COMPONENTS=COMP_TR, OPERATOR_COMPONENTS_TEST=COMP_TE,
            FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0"))
assert rc == 0, f"SMOKE FAILED, exit {rc}. Do not start the overnight run."

# Read the log back rather than trusting exit 0: a crash inside a dataloader worker
# can still exit clean, and "it ran" is not the same as "it used both graphs".
log = open(f"{SMOKE_DIR}/console.log", errors="ignore").read()
for g in TRAIN_G:
    assert g in log, f"{g} never appears in the log; the loader did not reach it"
tables = log.count("[fusion] loaded OPERATOR_COMPONENTS")
assert tables >= 4, (f"only {tables} operator tables loaded, expected 4 "
                     f"(2 train + 2 test). The multi-table patch is not active.")
for g in VALID_G:
    assert f"{g}/document_" in log, (
        f"no per-dataset metrics for {g}; validation did not cover both graphs")
print(f"\\nSMOKE PASSED: {tables} component tables, both training graphs reached, "
      f"validation covered {len(VALID_G)} datasets.\\nSafe to start the overnight run.")
'''

TRAIN = '''# JOINT TRAINING. train_names is a LIST: the loader keeps one graph resident at a
# time and walks both, so one set of weights sees both domains while peak memory
# stays at the larger single graph. That is what keeps train_batch_size at 2.
#
# valid_names is the physics and biology TEST graphs. CS and MatSci are NOT here,
# by design: checkpoint selection must not see the held-out domains.
# Recomputed rather than inherited from the smoke cell, so this cell still runs
# standalone after a runtime restart.
TRAIN_G = [f"sir4_{d}_train_v16sc" for d in TRAIN_DOMS]
VALID_G = [f"sir4_{d}_test_v16sc"  for d in TRAIN_DOMS]
COMP_TR = ",".join(comp_path(d, "train") for d in TRAIN_DOMS)
COMP_TE = ",".join(comp_path(d, "test")  for d in TRAIN_DOMS)

cmd = ("python -u -m gfmrag.workflow.sft_training "
       "--config-path config/gfm_reasoner --config-name sft_training_fusion "
       "text_emb_model=qwen3_st "
       f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
       f"datasets.train_names=[{','.join(TRAIN_G)}] "
       f"datasets.valid_names=[{','.join(VALID_G)}] "
       # One graph in memory at a time. The config default is 10, which would hold
       # both and give back the headroom batch 2 depends on. No `+` prefix: the key
       # already exists, and Hydra rejects `+` on an existing key.
       "datasets.max_datasets_in_memory=1 "
       f"trainer.args.num_epoch={EPOCHS} "
       f"trainer.args.train_batch_size={BATCH} "
       f"+trainer.args.eval_batch_size={BATCH} "
       f"hydra.run.dir={RUN}")

# STRAT_TEST / STRAT_BGE are deliberately NOT set. They drive the per-epoch
# stratified eval, which assumes ONE validation corpus; with two it would label
# against the wrong metadata. Unset, the slice patch falls back to empty dicts and
# every query lands in `cross`, `dissim` AND `cross+dissim` at once, so those three
# per-epoch numbers will be identical and should be IGNORED. The whole-set metrics
# on each line are correct, and the real slices come from cell 13's scoring.
print("training on", TRAIN_G)
print("validating on", VALID_G, "(CS and MatSci deliberately absent)\\n")
rc = sh(cmd, "/content/gfm-rag", log=f"{RUN}/console.log", extra=dict(
    WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    OPERATOR_COMPONENTS=COMP_TR, OPERATOR_COMPONENTS_TEST=COMP_TE,
    FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0"))
assert rc == 0, f"training failed, exit {rc} (-9 means the host ran out of RAM)"
CKPT = f"{RUN}/model_best.pth"
assert os.path.exists(CKPT), f"training finished but wrote no {CKPT}"
print("\\ncheckpoint:", CKPT)
'''

ZEROSHOT = '''# ZERO-SHOT. Load the joint checkpoint, predict on each held-out graph. No
# training, no tuning, no adaptation: the only per-domain input is that domain's
# own operator components, which are a property of its corpus and not of the model.
PRED = {}
for d in EVAL_DOMS:
    g   = f"sir4_{d}_test_v16sc"
    out = f"{RUN}/zeroshot_{d}"
    dst = f"{out}/predictions_{g}.json"
    if os.path.exists(dst):
        print(f"{d}: cached"); PRED[d] = dst; continue
    os.makedirs(out, exist_ok=True)
    npz = comp_path(d, "test")
    cmd = ("python -u -m gfmrag.workflow.sft_training "
           "--config-path config/gfm_reasoner --config-name sft_training_fusion "
           "text_emb_model=qwen3_st "
           f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
           f"datasets.train_names=[{g}] datasets.valid_names=[{g}] "
           "trainer.args.do_train=false trainer.args.do_eval=false "
           "+trainer.args.eval_batch_size=2 "
           "+trainer.args.do_predict=true +trainer.args.predict_top_k=300 "
           f"+trainer.args.resume_from_checkpoint={CKPT} "
           f"hydra.run.dir={out}")
    rc = sh(cmd, "/content/gfm-rag", extra=dict(
        WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        OPERATOR_COMPONENTS=npz, OPERATOR_COMPONENTS_TEST=npz,
        FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0"))
    assert rc == 0, f"zero-shot on {d} failed, exit {rc}"
    PRED[d] = dst
print("\\n", PRED)
'''

SCORE = '''# Score the held-out domains. Same scorer and same columns as every other SIR-4
# table, so these numbers drop straight into the comparison.
def sets_path(d):
    tag = "cs_test_final" if d == "cs" else f"{d}_test_low"
    return f"{CARGO_ROOT}/quartet/data/benchmark/{tag}/sets.json"

COLS = "mrr,ndcg@5,recall@3,recall@5,completeset@5"
for d in EVAL_DOMS:
    q   = f"{DATA_ROOT}/sir4_{d}_test/raw/test.json"
    bge = f"{S4}/data/predictions_bge_sir4_{d}_test.json"
    if not os.path.exists(bge):
        for src in (f"{DRIVE}/outputs/sir4_{d}/cache/predictions_bge_sir4_{d}_test.json",
                    f"{DRIVE}/outputs/sir4_{d}/predictions_bge_test.json"):
            if os.path.exists(src):
                os.makedirs(os.path.dirname(bge), exist_ok=True)
                shutil.copy(src, bge); break
    args = f"--pred {PRED[d]} --queries {q} --cols {COLS}"
    if os.path.exists(sets_path(d)): args += f" --sets {sets_path(d)}"
    if os.path.exists(bge):          args += f" --bge {bge}"
    sh(f"python3 -u eval/score_sir4.py {args} --name 'joint P+B -> {d} (zero-shot)' "
       f"--json-out {RUN}/zeroshot_{d}_scores.json "
       f"--per-query-out {RUN}/zeroshot_{d}_perquery.json", S4)
print(f"\\nwrote scores to {RUN}")
'''

COMPARE = '''# THE COMPARISON THE EXPERIMENT IS FOR. Joint Physics+Biology against the
# single-domain models, on the two held-out domains. The single-domain numbers come
# from the transfer matrix if it has been run; otherwise this prints the joint
# results alone and says so, rather than inventing a baseline.
MET = [("mrr","MRR"),("ndcg@5","nDCG@5"),("recall@3","R@3"),
       ("recall@5","R@5"),("completeset@5","CGS@5")]
MTX = f"{DRIVE}/outputs/_transfer_matrix"
lines = []
def out(s=""):
    print(s); lines.append(s)

for slc in ("all", "same", "cross", "similar", "dissimilar"):
    rows = []
    for d in EVAL_DOMS:
        js = f"{RUN}/zeroshot_{d}_scores.json"
        if not os.path.exists(js): continue
        j = json.load(open(js))
        if slc not in j: continue
        row = {"arm": "joint P+B", "target": d, **j[slc]}
        rows.append(row)
        for tr in TRAIN_DOMS:                       # single-domain, same target
            m = f"{MTX}/{tr}__on__{d}_scores.json"
            if os.path.exists(m):
                mj = json.load(open(m))
                if slc in mj:
                    rows.append({"arm": f"{tr} only", "target": d, **mj[slc]})
    if not rows: continue
    out(f"### {slc}   (zero-shot on held-out domains)")
    out("| target | arm | n | " + " | ".join(l for _, l in MET) + " |")
    out("|---|---|--:|" + "--:|" * len(MET))
    for d in EVAL_DOMS:
        sub = [r for r in rows if r["target"] == d]
        best = {k: max((r[k] for r in sub if k in r), default=None) for k, _ in MET}
        for r in sub:
            cells = [("**%.4f**" % r[k]) if k in r and r[k] == best[k]
                     else (f"{r[k]:.4f}" if k in r else "-") for k, _ in MET]
            out(f"| {d} | {r['arm']} | {r['n']} | " + " | ".join(cells) + " |")
    out()

if not glob.glob(f"{MTX}/*__on__*_scores.json"):
    out("Single-domain baselines are absent: run colab_sir4_transfer_matrix.ipynb "
        "to produce physics-only and biology-only zero-shot numbers on the same "
        "two targets. Until then the table above is the joint arm alone.")
open(f"{RUN}/joint_vs_single.md", "w").write("\\n".join(lines))
print(f"\\nwrote {RUN}/joint_vs_single.md")
'''

PATCH = '''# base_trainer._load_checkpoint runs from _setup_model(), which __init__ calls
# BEFORE it builds the AMP scaler, so a resume dies with a bare AttributeError on
# self.scaler. The optimizer load three lines above already guards with hasattr.
# The zero-shot cells below all resume, so this is required, not optional.
p = "/content/gfm-rag/gfmrag/trainers/base_trainer.py"
src = open(p).read()
OLD = 'if "scaler" in state:'
NEW = 'if "scaler" in state and hasattr(self, "scaler"):'
if NEW in src:
    print("already patched: scaler guard")
elif OLD in src:
    open(p, "w").write(src.replace(OLD, NEW, 1)); print("patched: scaler load is now guarded")
else:
    raise SystemExit("could not find the scaler load -- engine changed, check by hand")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-domain", default="biology",
                    help="training notebook to clone the setup cells from")
    ap.add_argument("--epochs", type=int, default=10,
                    help="physics peaked at epoch 7 and was flat after 3; 10 is the "
                         "same budget its own run used")
    ap.add_argument("--batch", type=int, default=2,
                    help="biology's graph needed 2 on an 80GB A100 and it is the "
                         "larger of the two")
    ap.add_argument("--op-slug", default="_content-qwen3")
    # ALL FOUR is the foundation-model configuration: nothing is held out inside
    # SIR-4, so ResearchBench becomes the only zero-shot evidence. The two-domain
    # default keeps cs and matsci held out, which is the weaker model but the only
    # within-benchmark transfer test. They answer different questions; keep both.
    ap.add_argument("--train", default=",".join(TRAIN_DOMS),
                    help="comma-separated training domains")
    ap.add_argument("--eval", dest="evald", default=",".join(EVAL_DOMS),
                    help="comma-separated held-out domains; empty for none")
    ap.add_argument("--tag", default=None,
                    help="run-directory suffix; defaults to the training domains")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    train = [d.strip() for d in a.train.split(",") if d.strip()]
    evald = [d.strip() for d in a.evald.split(",") if d.strip()]
    assert train, "--train needs at least one domain"
    leak = sorted(set(train) & set(evald))
    assert not leak, (f"{leak} appear in BOTH --train and --eval; a held-out domain "
                      f"that was trained on is not held out")
    tag = a.tag or ("all" if len(train) == 4 else "".join(d[0] for d in train))
    if a.out is None:
        a.out = f"{ROOT}/colab_sir4_joint_{tag}.ipynb"

    src = f"{ROOT}/colab_train_sir4_{a.from_domain}_fusion.ipynb"
    if not os.path.exists(src):
        return print(f"missing {src}") or 1
    nb = json.load(open(src))
    cells = [json.loads(json.dumps(c)) for c in nb["cells"][:SETUP_CELLS]]

    # The cloned setup is per-dataset: its isolation guard and bundle unpack both
    # assume one DATASET. Drop those two; this notebook unpacks four bundles itself.
    keep = []
    for c in cells:
        s = "".join(c["source"])
        if "need = [" in s or "z.extractall(CARGO_ROOT)" in s:
            continue
        s = s.replace('DATASET = "sir4_%s"' % a.from_domain,
                      'DATASET = "sir4_physics"  # only for scoped cache paths')
        c["source"] = s.splitlines(True)
        keep.append(c)
    cells = keep

    # RE-INLINE THE ENGINE FILES FROM THE REPO.
    #
    # The overlay cell carries fusion_reasoner.py, fusion_trainer.py and the config
    # as a JSON blob, and build_notebook.py does NOT generate it -- it copies the
    # cell forward from an older notebook (`cells.append(reuse[7])`). So the blob is
    # frozen at whatever was inlined when that notebook was first made, and edits to
    # the repo never reach Colab through it. The bundles do not carry these files
    # either (bundle.py's CODE list stops at kg-construction/eval and experiments).
    #
    # Joint training needs the multi-corpus fusion_reasoner, so refresh the blob from
    # the working tree at build time. Everything else in the cell is left alone.
    kg = os.path.join(os.path.dirname(ROOT), "kg-construction", "gfm-rag", "gfmrag")
    LIVE = {
        "/content/gfm-rag/gfmrag/models/fusion_reasoner.py":
            os.path.join(kg, "models", "fusion_reasoner.py"),
    }
    refreshed = []
    for c in cells:
        s = "".join(c["source"])
        m = re.search(r"FILES = json\.loads\(r'''(.*?)'''\)", s, re.S)
        if not m:
            continue
        blob = json.loads(m.group(1))
        for dest, live in LIVE.items():
            if dest not in blob:
                return print(f"overlay blob has no {dest}; engine layout changed") or 1
            body = open(live).read()
            # The blob sits inside an r'''...''' literal, so a triple-single-quote in
            # any file would terminate it early and produce a syntactically valid but
            # truncated cell. json.dumps escapes quotes but not this.
            if "'''" in body:
                return print(f"{live} contains ''' which would break the r'''...''' "
                             f"wrapper; change the quoting in build_joint_notebook.py") or 1
            if blob[dest] != body:
                refreshed.append(f"{os.path.basename(live)} "
                                 f"{len(blob[dest])} -> {len(body)} bytes")
            blob[dest] = body
        s = s[:m.start(1)] + json.dumps(blob) + s[m.end(1):]
        c["source"] = s.splitlines(True)
    for r in refreshed:
        print(f"  refreshed overlay: {r}")
    if not refreshed:
        print("  overlay already current")

    cells[0] = md(
        "# Joint Physics+Biology training, zero-shot on CS + Materials Science\n\n"
        "One model, two training domains, two held-out domains.\n\n"
        "```\n{Physics, Biology}_train  -->  {Computer Science, Materials Science}_test\n```\n\n"
        "**Leakage control.** Validation and checkpoint selection use the Physics "
        "and Biology test graphs only. CS and MatSci are loaded after training "
        "ends and are never seen by model selection. Their numbers are the only "
        "held-out ones; Physics and Biology test numbers here are dev numbers.\n\n"
        "**What 'joint' means.** `train_names` is a list, and the loader keeps one "
        "graph resident at a time, so one set of weights receives gradients from "
        "both domains while peak memory stays at the larger single graph. The two "
        "graphs are *not* merged: they share 1,113 of 312,390 non-document nodes "
        "(0.9%), so merging would produce two near-disjoint components at double "
        "the wall clock. The cost of not merging is that a Physics query never "
        "competes against a Biology document. Say so in the write-up.\n\n"
        f"Budget: {a.epochs} epochs at batch {a.batch}, roughly 40 min per epoch "
        "across both graphs, so about 7 hours plus setup.")

    cells += [
        md("## 6. Helpers, paths and the run directory"),
        code(HELPERS.replace("__TRAIN__", repr(train))
                    .replace("__EVAL__", repr(evald))
                    .replace("__TAG__", tag)
                    .replace("__OPSLUG__", a.op_slug)
                    .replace("__EPOCHS__", str(a.epochs))
                    .replace("__BATCH__", str(a.batch))),
        md("## 7. Unpack all four bundles"), code(UNPACK),
        md("## 8. Operator components for all six graphs"), code(COMPONENTS),
        md("## 9. Capability and leakage checks (fail in seconds, not in an hour)"),
        code(CAPABILITY),
        md("## 10. Guard the checkpoint-resume path"), code(PATCH),
        md("## 11. Smoke test both graphs (10 min)\n\nThe dataset switch is the "
           "only new code path in this run. Reach it in minutes, not forty minutes "
           "into a seven-hour job. **Do not start cell 13 until this passes.**"),
        code(SMOKE),
        md("## 12. Joint training\n\nThe long cell. Roughly 7 hours; the log streams "
           "here and is mirrored to `console.log` on Drive so a disconnected browser "
           "does not lose it."),
        code(TRAIN),
        md("## 13. Zero-shot on the held-out domains"), code(ZEROSHOT),
        md("## 14. Score"), code(SCORE),
        md("## 15. Joint versus single-domain"), code(COMPARE),
    ]

    nb["cells"] = cells
    json.dump(nb, open(a.out, "w"), indent=1)

    # Every code cell must parse. Colab magics are not Python, so blank them (and any
    # backslash continuation they carry) before compiling.
    bad = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        lines, out_lines, skip = "".join(c["source"]).splitlines(), [], False
        for ln in lines:
            if skip:
                out_lines.append("")
                skip = ln.rstrip().endswith("\\")
                continue
            if ln.lstrip().startswith(("!", "%")):
                out_lines.append("")
                skip = ln.rstrip().endswith("\\")
            else:
                out_lines.append(ln)
        try:
            ast.parse("\n".join(out_lines))
        except SyntaxError as e:
            bad += 1
            print(f"  CELL {i} DOES NOT PARSE: {e}")

    checks = [
        ("train_names is a list", "datasets.train_names=[" in TRAIN and "','.join(TRAIN_G)" in TRAIN),
        ("components list joined", 'COMP_TR = ",".join' in TRAIN),
        ("one graph in memory", "max_datasets_in_memory=1" in TRAIN),
        ("valid excludes held-out", "VALID_G = [f\"sir4_{d}_test_v16sc\"  for d in TRAIN_DOMS]" in TRAIN),
        ("capability check", "MULTI-CORPUS TRAINING" in CAPABILITY),
        ("leakage check", "appear in {t} training data" in CAPABILITY),
        ("scaler guard", "hasattr(self" in PATCH),
        ("log mirrored to Drive", 'log=f"{RUN}/console.log"' in TRAIN),
        ("smoke checks both graphs", "the loader did not reach it" in SMOKE),
        ("smoke counts 4 tables", "tables >= 4" in SMOKE),
        ("smoke is 3 steps", "max_steps_per_epoch=3" in SMOKE),
        ("zips validated first", "is_zipfile" in UNPACK),
        ("FORCE re-extract", "FORCE = False" in UNPACK),
    ]
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"\nwrote {a.out}  ({len(nb['cells'])} cells, "
          f"{a.epochs} epochs, batch {a.batch}, slug {a.op_slug})")
    print("ALL CHECKS PASS" if not bad and all(k for _, k in checks) else "PROBLEMS ABOVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
