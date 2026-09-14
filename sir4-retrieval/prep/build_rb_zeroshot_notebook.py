"""
build_rb_zeroshot_notebook.py -- zero-shot cross-benchmark transfer on ResearchBench.

ONE NOTEBOOK, ONE BENCHMARK, EVERY ARM. ResearchBench (20,322 documents, 1,367
queries) is never trained on, tuned on or selected on by anything here. Every arm
ranks the same full corpus and is scored by the same eval/score_sir4.py call with
the same flags, so a difference between two rows is the retriever and nothing else.

    lexical                   BM25
    dense                     BGE-large, Qwen3-Embedding, SPECTER2-base, SciNCL
    reasoning-trained dense   ReasonIR-8B
    SciGraphIR (TOMATO-Star)  frozen checkpoint from Drive, multi-view scorer + graph + CCMP
    SciGraphIR (SIR-4, 4 fields)  TRAINED HERE on cs+biology+physics+matsci, then frozen

The baseline arms are the SAME six arm definitions as sir4_baselines_all.ipynb,
harvested from that notebook at build time rather than retyped, so the rows here and
the rows in Table 9.3 cannot define a baseline differently.

WHAT IS CLONED AND WHY. The engine install, the fusion-source blob, the config
rewrite, the PyG fix, the torchvision shim, the per-epoch diagnostics patch and the
CCMP_LR patch are cloned by index from tomato_ccmp_ablation.ipynb, the notebook that
trained the TOMATO-Star checkpoint this notebook loads. The fusion-source blob is
REGENERATED from the repo copies (like build_notebook.py does), because the repo's
fusion_trainer.py has moved on since that notebook was built and the trained weights
must be loaded by the current model code.

THE SIR-4 ARM'S SCORER WARM START. semantic_scorer.py fits one corpus at a time. The
multi-view scorer (~200 parameters plus the popularity predictor) is therefore
warm-started on ONE SIR-4 field (SEM_WARM_DOM, default cs, the largest train set) and
then trained jointly with the graph on all four fields inside the fusion, exactly as
the TOMATO-Star arm's scorer was warm-started on TOMATO and trained inside its fusion.
State this in the write-up: the scorer's initialisation saw one field, its training saw
four.

Usage
-----
    python3 prep/build_rb_zeroshot_notebook.py
    python3 prep/build_rb_zeroshot_notebook.py --epochs 10 --batch 2 --warm-dom cs
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)            # sir4-retrieval
CARGO = os.path.dirname(ROOT)
FORK = f"{CARGO}/kg-construction/gfm-rag"

SRC_NB = f"{ROOT}/tomato_ccmp_ablation.ipynb"     # engine + patches, and the run that made the TOMATO checkpoint
BASE_NB = f"{ROOT}/sir4_baselines_all.ipynb"      # the six baseline arm definitions
ENGINE_CELLS = list(range(5, 18))                 # 5..17: engine, FILES(7, regenerated), config, pyg, shim, diagnostics, CCMP_LR
QWEN_CELLS = [18, 19]
ARMS_CELL = 6

# The fusion sources, from the repo. Same list as build_notebook.py.
FUSION_REL = ["gfmrag/models/fusion_reasoner.py",
              "gfmrag/models/cqig.py",
              "gfmrag/models/ultra/models.py",
              "gfmrag/trainers/fusion_trainer.py",
              "gfmrag/trainers/base_trainer.py",
              "gfmrag/workflow/config/gfm_reasoner/sft_training_fusion.yaml"]

# Repo scripts the notebook shells out to, shipped inline so the row cannot depend on
# the age of a bundle on Drive. The researchbench bundle (8 Aug) predates
# baselines_sir4.py, semantic_scorer.py and precompute_semantic_components.py entirely.
OVERLAY_REL = ["cargo_paths.py",
               "sir4-retrieval/eval/baselines_sir4.py",
               "sir4-retrieval/eval/bge_sir4.py",
               "sir4-retrieval/eval/score_sir4.py",
               "sir4-retrieval/eval/semantic_scorer.py",
               "sir4-retrieval/eval/report_domain_results.py",
               "sir4-retrieval/transfer/paired_bootstrap.py",
               "kg-construction/eval/cargo_operator.py",
               "kg-construction/experiments/probe_greasoner/precompute_operator_components.py",
               "kg-construction/experiments/probe_greasoner/precompute_semantic_components.py"]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(True)}


# =============================================================================
# cell bodies (plain strings; __TOKENS__ are substituted at build time)
# =============================================================================

PATHS = '''!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import os, sys, re, time, glob, json, math, shutil, zipfile, subprocess, hashlib, threading
from collections import defaultdict
from google.colab import drive
drive.mount('/content/drive')

DRIVE    = "/content/drive/MyDrive/cargo-gfmrag"
DATASET  = "researchbench"
RBG      = f"{DATASET}_test_v16sc"      # the one ResearchBench graph: 20,322 docs, 1,367 queries
TRAIN = TEST = RBG                      # read by the cloned config-rewrite cell; every run overrides on the CLI
BUNDLE   = f"{DRIVE}/{DATASET}_bundle.zip"
OUT_ROOT = f"{DRIVE}/outputs/rb_zeroshot"          # everything this notebook writes
CACHE_RB = f"{DRIVE}/outputs/researchbench/cache"  # graph-keyed artefacts the earlier RB runs cached
CARGO_ROOT = "/content/cargo"
os.environ["CARGO_ROOT"] = CARGO_ROOT
os.environ["CARGO_DATASET"] = DATASET
DATA_ROOT = f"{CARGO_ROOT}/kg-construction/data"
S4        = f"{CARGO_ROOT}/sir4-retrieval"
KGDIR     = f"{CARGO_ROOT}/kg-construction"
RUNS      = "/content/runs"             # LOCAL run directories; synced to Drive, never written per-line on Drive
OP_MODEL  = "/content/qwen3"
OP_SLUG   = "_content-qwen3"
S4_DOMS   = ["cs", "biology", "physics", "matsci"]

# ---- the two SciGraphIR arms --------------------------------------------------------
# A. TOMATO-Star weights, FROZEN. The default is the run behind Table 9.1's "+ CCMP" row
#    (same R@5 44.11 / cross 33.22, nDCG@5 34.24 / 24.53). The later runs in the same
#    Drive folder are the post-thesis updates; switch by name, nothing else changes:
#      tomato_fusion_qwenmlp_ccmp_epoch10_b2            17 Aug   Table 9.1 row
#      tomato_fusion_qwenmlp_ccmp_epoch10_b2_newccmp    30 Aug
#      tomato_fusion_qwenmlp_ccmp_epoch10_b2_ccmp2      30 Aug
#      tomato_fusion_qwenmlp_ccmpresid_epoch10_b2       31 Aug   residual CCMP
TOMATO_OUT = f"{DRIVE}/outputs/tomato_ablations_v1/tomato"
TOMATO_RUN = "__TOMATO_RUN__"
TOMATO_SEM = f"{TOMATO_OUT}/semantic"     # the multi-view scorer 5d wrote (shapes only; weights come from the checkpoint)
# B. SIR-4 weights do not exist yet: section 8 trains them on all four fields.
SEM_WARM_DOM = "__WARM__"    # field that warm-starts the multi-view scorer before joint training
EPOCHS, BATCH = __EPOCHS__, __BATCH__
# PER-GRAPH BATCH. CS (243k nodes) OOMs at batch 2 with the scorer + CCMP on an 80 GB A100 and
# every August CS run trained at 1; the other fields fit at BATCH. The loader is rebuilt per
# graph each epoch, so the engine patch in section 8 reads this override by graph name.
BATCH_BY_DATASET = {"sir4_cs_train_v16sc": 1}

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    os.environ.setdefault("HF_TOKEN", "")

env = dict(os.environ, CARGO_ROOT=CARGO_ROOT, CARGO_DATASET=DATASET, PYTHONUNBUFFERED="1")

def sh(cmd, cwd, extra=None, log=None, check=True):
    """Run a command with LIVE output. Reads raw chunks, not lines: tqdm redraws with a
    carriage return and a line-buffered reader stays silent for a whole bar."""
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=dict(env, **(extra or {})), shell=isinstance(cmd, str),
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
    if check:
        assert p.returncode == 0, f"FAILED (exit {p.returncode}): {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}"
    return p.returncode

def sync_dir(src, dst):
    """Copy new/changed files src -> dst, tolerating a flapping Drive mount (Errno 107)."""
    n = 0
    try:
        for root, _, files in os.walk(src):
            rel = os.path.relpath(root, src)
            d = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(d, exist_ok=True)
            for f in files:
                s, t = os.path.join(root, f), os.path.join(d, f)
                if (not os.path.exists(t) or os.path.getsize(t) != os.path.getsize(s)
                        or os.path.getmtime(s) > os.path.getmtime(t) + 1):
                    shutil.copy2(s, t); n += 1
    except OSError as e:
        print(f"[sync] {e} -- will retry next round")
    return n

def start_sync(src, dst, every=600):
    """Background thread: sync every `every` seconds until stop() is called, then once more."""
    stop = threading.Event()
    def loop():
        while not stop.wait(every):
            print(f"[sync] {sync_dir(src, dst)} file(s) -> {os.path.relpath(dst, DRIVE)}", flush=True)
    th = threading.Thread(target=loop, daemon=True); th.start()
    def finish():
        stop.set(); th.join(timeout=5)
        print(f"[sync] final: {sync_dir(src, dst)} file(s) -> {os.path.relpath(dst, DRIVE)}")
    return finish

for d in (OUT_ROOT, RUNS):
    os.makedirs(d, exist_ok=True)
print("DRIVE      ", DRIVE)
print("reads      ", os.path.basename(BUNDLE), "+ sir4_*_bundle.zip (section 8) + gfm-rag-adapted.zip + qwen3-embedding-0.6b/")
print("TOMATO arm ", f"{TOMATO_OUT}/{TOMATO_RUN}/model_best.pth")
print("writes     ", OUT_ROOT)
'''

UNPACK = '''# 2. Unpack ResearchBench to an isolated local root, then install the CURRENT repo scripts.
# The researchbench bundle on Drive is from 8 Aug and predates baselines_sir4.py,
# semantic_scorer.py and precompute_semantic_components.py entirely, so the scripts this
# notebook calls are shipped INLINE (captured from the repo when the notebook was built)
# and written over whatever any bundle carries. A Drive code_overlay is applied first and
# the inline copies last, so the repo state at build time is what runs.
KEEP, PARK = f"{CARGO_ROOT}/outputs/caches", "/content/_caches_keep"
if os.path.isdir(KEEP):
    shutil.rmtree(PARK, ignore_errors=True); shutil.move(KEEP, PARK)
if os.path.exists(CARGO_ROOT):
    shutil.rmtree(CARGO_ROOT)
os.makedirs(CARGO_ROOT, exist_ok=True)
assert os.path.exists(BUNDLE) and zipfile.is_zipfile(BUNDLE), f"missing or corrupt {BUNDLE}"
zipfile.ZipFile(BUNDLE).extractall(CARGO_ROOT); print("unpacked", os.path.basename(BUNDLE))
if os.path.isdir(PARK):
    os.makedirs(os.path.dirname(KEEP), exist_ok=True); shutil.move(PARK, KEEP); print("restored caches")

OVERLAY = json.loads(r\'\'\'__OVERLAY__\'\'\')
def apply_overlay():
    ov = f"{DRIVE}/code_overlay"
    if os.path.isdir(ov):
        shutil.copytree(ov, CARGO_ROOT, dirs_exist_ok=True); print("applied Drive code_overlay")
    for rel, src in OVERLAY.items():
        p = f"{CARGO_ROOT}/{rel}"
        os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(src)
    print(f"installed {len(OVERLAY)} repo scripts captured __BUILT__")
apply_overlay()

sys.path.insert(0, CARGO_ROOT)
import cargo_paths as cp
cp.set_dataset(DATASET); print(cp.banner())
assert cp.CARGO == CARGO_ROOT, f"cargo_paths resolved {cp.CARGO}, expected {CARGO_ROOT}"

RB_RAW  = f"{DATA_ROOT}/{DATASET}_test/raw"
QUERIES = f"{RB_RAW}/test.json"
SETS    = f"{DATA_ROOT}/{DATASET}_test/sets.json"
SUBSETS = f"{DATA_ROOT}/{DATASET}_test/subsets.json"
for label, p in (("corpus", f"{RB_RAW}/documents.json"), ("queries", QUERIES),
                 ("probes", cp.probes_path("test")),
                 ("graph", f"{DATA_ROOT}/{RBG}/processed/stage1/nodes.csv"),
                 ("sets", SETS), ("subsets", SUBSETS)):
    assert os.path.exists(p), f"missing {label}: {p}"
    print(f"  ok  {label:8} {p.replace(CARGO_ROOT, '<root>')}")
# The graph loader opens {graph}/raw/documents.json, a copy of the corpus INSIDE the graph dir.
os.makedirs(f"{DATA_ROOT}/{RBG}/raw", exist_ok=True)
shutil.copy(f"{RB_RAW}/documents.json", f"{DATA_ROOT}/{RBG}/raw/documents.json")

_q = json.load(open(QUERIES))
_sl = defaultdict(int)
for x in _q: _sl[x.get("domain_slice")] += 1
_g = [len(x["supporting_documents"]) for x in _q]
print(f"\\n{len(json.load(open(f'{RB_RAW}/documents.json'))):,} docs | {len(_q):,} queries | "
      f"golds/query {sum(_g)/len(_g):.2f} | domain_slice {dict(_sl)}")
STRICT = set(json.load(open(SUBSETS))["subsets"]["strict_zeroshot"])
print(f"strict zero-shot subset: {len(STRICT)} queries (frozen in subsets.json before any model ran)")

# Recall@3 and @15 are ResearchBench's own operating points; R@5 and nDCG@5 match the SIR-4 tables.
RB_COLS = ("mrr,mgrr,ndcg@5,hits@1,hits@5,recall@1,recall@3,recall@5,"
           "recall@10,recall@15,recall@20,completeset@5,completeset@10")

# Section 3 needs these before the engine is installed; the pin matches the engine's.
!pip -q install rank_bm25 sentence-transformers "transformers>=4.52.4,<5"
import transformers
assert transformers.__version__.startswith("4."), (
    f"transformers {transformers.__version__} is outside 4.x; restart the runtime and re-run")
print("ready | transformers", transformers.__version__)
'''

ARMS_TAIL = '''
# The Drive-cached Qwen3 weights are the same checkpoint as the HF id; use them when present.
if os.path.isdir(OP_MODEL):
    ARMS = [(l, t, (OP_MODEL if t == "qwen3" else m), p, i, e) for l, t, m, p, i, e in ARMS]
    print("Qwen3 arm reads", OP_MODEL)
'''

RUN_BASELINES = '''# 3b. Run every baseline on the full 20,322-document corpus. Same runner, same flags as
# sir4_baselines_all.ipynb. Signature-gated: a finished arm on Drive is a [skip].
import shlex
TOPK = 300
OUTB = f"{OUT_ROOT}/baselines"
os.makedirs(OUTB, exist_ok=True)
_SCORER = hashlib.md5(open(f"{S4}/eval/baselines_sir4.py", "rb").read()).hexdigest()[:8]

def arm_sig(model, pool, ins, extra):
    return hashlib.md5(json.dumps(
        {"model": model, "pooling": pool, "instruct": ins, "extra": extra,
         "topk": TOPK, "scorer": _SCORER, "dataset": DATASET, "split": "test"},
        sort_keys=True).encode()).hexdigest()[:12]

PRED = {}
for lab, tag, model, pool, ins, extra in ARMS:
    dest = f"{S4}/data/predictions_{tag}_{DATASET}_test.json"
    man  = dest + ".manifest.json"
    sig  = arm_sig(model, pool, ins, extra)
    for a, b in ((f"{OUTB}/{os.path.basename(dest)}", dest), (f"{OUTB}/{os.path.basename(man)}", man)):
        if not os.path.exists(b) and os.path.exists(a):
            os.makedirs(os.path.dirname(b), exist_ok=True); shutil.copy(a, b)
    have = None
    if os.path.exists(man):
        try: have = json.load(open(man)).get("sig")
        except Exception: have = None
    if os.path.exists(dest) and have == sig:
        print(f"[skip] {lab}: manifest matches"); PRED[tag] = dest; continue
    if os.path.exists(dest):
        print(f"[stale] {lab}: {'no manifest' if have is None else 'signature changed'} -- re-running")
    cmd = [sys.executable, "-u", "eval/baselines_sir4.py", "--dataset", DATASET, "--split", "test",
           "--model", model, "--pooling", pool, "--tag", tag, "--topk", str(TOPK)]
    if ins:   cmd += ["--instruct", ins]          # argv list: the Qwen3 newline reaches the tokenizer intact
    if extra: cmd += shlex.split(extra)
    cmd += ["--out", dest]
    rc = sh(cmd, S4, check=False)
    if rc != 0:
        print(f"!! {lab} FAILED rc={rc} -- row will be reported as missing"); continue
    json.dump({"sig": sig, "model": model, "pooling": pool, "instruct": ins, "extra": extra,
               "topk": TOPK, "scorer_md5": _SCORER}, open(man, "w"), indent=1)
    shutil.copy(dest, f"{OUTB}/{os.path.basename(dest)}"); shutil.copy(man, f"{OUTB}/{os.path.basename(man)}")
    PRED[tag] = dest

# THE SLICE-DEFINITION FILE. score_sir4.py calls a query `dissimilar` when plain BGE ranks
# its best gold below 100. That definition comes from bge_sir4.py (the file every earlier
# ResearchBench number used), NOT from the BGE-large baseline row above, which runs with
# BGE's retrieval instruction. Kept as a separate file so the slice boundary cannot move
# with the row being scored.
BGE_SLICE = f"{OUT_ROOT}/slice/predictions_bge_slice_{DATASET}_test.json"
os.makedirs(os.path.dirname(BGE_SLICE), exist_ok=True)
if not os.path.exists(BGE_SLICE):
    for c in (f"{CACHE_RB}/predictions_bge_{DATASET}_test.json",
              f"{DRIVE}/outputs/{DATASET}/transfer/predictions_bge_{DATASET}_test.json"):
        if os.path.exists(c):
            shutil.copy(c, BGE_SLICE); print("slice file restored from", os.path.relpath(c, DRIVE)); break
if not os.path.exists(BGE_SLICE):
    if os.path.isdir(f"{CACHE_RB}/op_emb"):
        shutil.copytree(f"{CACHE_RB}/op_emb", f"{CARGO_ROOT}/outputs/caches/op_emb", dirs_exist_ok=True)
    sh(f"python3 -u eval/bge_sir4.py --dataset {DATASET} --split test --topk {TOPK} --out {BGE_SLICE}", S4)
assert os.path.exists(BGE_SLICE), BGE_SLICE
print("\\nbaseline predictions:", {k: os.path.basename(v) for k, v in PRED.items()})
'''

SCORE_BASELINES = '''# 4. Score every baseline: one scorer, one flag set. Per-query files are kept for the
# paired tests in section 9.
SCORES = f"{OUT_ROOT}/scores"
os.makedirs(SCORES, exist_ok=True)

def score(key, pred, label):
    js, pq = f"{SCORES}/{key}_scores.json", f"{SCORES}/{key}_perquery.json"
    sh([sys.executable, "-u", "eval/score_sir4.py", "--pred", pred, "--queries", QUERIES,
        "--sets", SETS, "--bge", BGE_SLICE, "--name", label, "--gold-strata",
        "--cols", RB_COLS, "--json-out", js, "--per-query-out", pq], S4)
    return js

LABEL = {tag: lab for lab, tag, *_ in ARMS}
for tag, pred in PRED.items():
    score(tag, pred, LABEL[tag])
print("\\nscored:", sorted(PRED))
'''

RB_COMPONENTS = '''# 6. ResearchBench inputs for the fusion: Qwen3 operator components (not read under the
# multi-view scorer, but the loader expects the path) and the multi-view scorer's own
# components (per-answer match matrix H as a memmap + aligned side arrays). Both are
# aligned to the graph's nodes.csv document order by the precompute scripts.
import numpy as np

if os.path.isdir(f"{CACHE_RB}/op_emb"):
    shutil.copytree(f"{CACHE_RB}/op_emb", f"{CARGO_ROOT}/outputs/caches/op_emb", dirs_exist_ok=True)
    print("restored ResearchBench embedding caches from Drive")

def restore_index(g, cache):
    """The Qwen3 node index is the 30-45 min artefact; earlier runs cached it graph-keyed."""
    src = f"{cache}/index/{g}"
    if not os.path.isdir(src):
        print(f"  {g}: no cached index on Drive (will be built on first use)"); return
    for d in os.listdir(src):
        shutil.copytree(f"{src}/{d}", f"{DATA_ROOT}/{g}/processed/{d}", dirs_exist_ok=True)
    print(f"  {g}: index restored ({', '.join(os.listdir(src))})")

def save_index(g, cache):
    pr = f"{DATA_ROOT}/{g}/processed"
    for d in os.listdir(pr):
        if d != "stage1":
            shutil.copytree(f"{pr}/{d}", f"{cache}/index/{g}/{d}", dirs_exist_ok=True)

restore_index(RBG, CACHE_RB)

# operator components (Qwen3). Cached on Drive by the holdout notebook; recomputed if absent.
RB_OPC = f"{DATA_ROOT}/{RBG}/operator_components{OP_SLUG}.npz"
_c = f"{CACHE_RB}/{RBG}_operator_components{OP_SLUG}.npz"
if not os.path.exists(RB_OPC):
    if os.path.exists(_c):
        shutil.copy(_c, RB_OPC); print("operator components restored from Drive")
    else:
        sh(f"python3 -u experiments/probe_greasoner/precompute_operator_components.py "
           f"--dataset {DATASET} --graph {RBG} --split test --model {OP_MODEL}", KGDIR)
        os.makedirs(os.path.dirname(_c), exist_ok=True); shutil.copy(RB_OPC, _c)
z = np.load(RB_OPC, allow_pickle=True)
assert "qwen" in str(z["encoder"]).lower(), f"{RB_OPC} was built with {z['encoder']!r}, not Qwen3"
print(f"operator components  dense {z['dense'].shape}  {z['encoder']}")

# multi-view scorer components. Encodes the corpus, queries and hypothetical answers with
# Qwen3 on the first run (~10 min for 20k docs), then builds H [Q, Jmax, D] as a memmap.
RB_SEM = f"{DATA_ROOT}/{RBG}/semantic_components{OP_SLUG}.npz"
def _sem_ok(p):
    if not os.path.exists(p): return False
    zz = np.load(p, allow_pickle=True)
    return os.path.exists(str(zz["h_path"])) and "qwen" in str(zz["encoder"]).lower()
if not _sem_ok(RB_SEM):
    sh(f"python3 -u experiments/probe_greasoner/precompute_semantic_components.py "
       f"--dataset {DATASET} --model {OP_MODEL} --graph {RBG} --split test", KGDIR)
assert _sem_ok(RB_SEM), f"semantic components not usable: {RB_SEM}"
zz = np.load(RB_SEM, allow_pickle=True)
print(f"semantic components  H {tuple(int(x) for x in zz['h_shape'])}  dense {zz['dense'].shape}  "
      f"Jmax={int(zz['Jmax'])}  {zz['encoder']}")

# persist the expensive part (embeddings) so a runtime reset costs nothing
os.makedirs(f"{CACHE_RB}/op_emb", exist_ok=True)
shutil.copytree(f"{CARGO_ROOT}/outputs/caches/op_emb", f"{CACHE_RB}/op_emb", dirs_exist_ok=True)
print("embedding caches saved to Drive")
'''

PREDICT_HELPER = '''# 7a. Zero-shot prediction helper. Loads a frozen fusion checkpoint and ranks the whole
# ResearchBench corpus for every query. No training, no tuning, no adaptation: the only
# per-corpus inputs are the corpus's own component tables, which are properties of the
# corpus and not of the model.
import torch, numpy as np

def inspect_ckpt(ckpt):
    """What the checkpoint actually contains, so the model is built to match it.
    base_trainer loads with strict=False, so a head that is not constructed is silently
    dropped -- and a CCMP checkpoint scored without its responsibility head is a different
    model wearing the same name. Read the keys, do not trust the folder name."""
    # weights_only=False: PyTorch 2.6 refuses the numpy scalars the trainer stores next to
    # the weights. The file is our own training output, not a download.
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    resp = [k for k in sd if "resp_" in k]
    sem  = [k for k in sd if k.startswith("sem_")]
    hid  = next((int(sd[k].shape[0]) for k in resp if k.endswith("resp_proj.0.weight")), None)
    jmax = next((int(sd[k].shape[1]) - 2 for k in sem if k.endswith("sem_net.0.weight")), None)
    return {"tensors": len(sd), "resp_keys": len(resp), "sem_keys": len(sem),
            "ccmp_hid": hid, "jmax": jmax}

def predict_rb(ckpt, name, sem_ckpt, sem_pop):
    """Rank ResearchBench with `ckpt`. Returns the predictions path (under OUT_ROOT)."""
    run_local, run_drive = f"{RUNS}/{name}", f"{OUT_ROOT}/{name}"
    pred_drive = f"{run_drive}/predictions_{RBG}.json"
    if os.path.exists(pred_drive):
        print(f"[cached] {os.path.relpath(pred_drive, DRIVE)}"); return pred_drive
    assert os.path.exists(ckpt), f"no checkpoint at {ckpt}"
    info = inspect_ckpt(ckpt)
    print(f"[ckpt] {os.path.relpath(ckpt, DRIVE)}\\n       {info}")
    assert info["sem_keys"] > 0, "checkpoint carries no multi-view scorer (sem_*): not a semantic=mlp run"
    st = json.load(open(sem_ckpt))
    assert int(st["jmax"]) == info["jmax"], (
        f"scorer width mismatch: checkpoint jmax={info['jmax']} but {sem_ckpt} has jmax={st['jmax']}; "
        "the warm-start file must come from the same 5d run as the checkpoint")
    extra = dict(WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                 OPERATOR_COMPONENTS=RB_OPC, OPERATOR_COMPONENTS_TEST=RB_OPC,
                 SEMANTIC_COMPONENTS=RB_SEM, SEMANTIC_COMPONENTS_TEST=RB_SEM,
                 SEMANTIC_CKPT=sem_ckpt, SEMANTIC_POPNET=sem_pop, SEM_POP_LAMBDA="1.0",
                 FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0",
                 PER_GOLD="1", HARDNEG_GRAPH="50")
    for k in list(os.environ):
        if k.startswith("CCMP"): os.environ.pop(k)     # a training cell must not leak its head settings
    if info["resp_keys"]:
        # The head's inference-time settings (gate on/off, mean-norm, eta) are not in the
        # weights. They are the run_model defaults; arm.json overrides when the run wrote one.
        extra.update(CCMP="1", CCMP_HID=str(info["ccmp_hid"]), CCMP_GATE="1", CCMP_GATE_NORM="1", CCMP_ETA="0.5")
        aj = f"{os.path.dirname(ckpt)}/arm.json"
        if os.path.exists(aj):
            a = json.load(open(aj))
            if a.get("ccmp_gate") is not None: extra["CCMP_GATE"] = "1" if a["ccmp_gate"] else "0"
            if a.get("ccmp_eta") is not None: extra["CCMP_ETA"] = str(a["ccmp_eta"])
            print("       arm.json:", {k: v for k, v in a.items() if k.startswith(("ccmp", "semantic", "fusion_form"))})
    os.makedirs(run_local, exist_ok=True)
    rc = sh("python -u -m gfmrag.workflow.sft_training "
            "--config-path config/gfm_reasoner --config-name sft_training_fusion "
            "text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{RBG}] datasets.valid_names=[{RBG}] "
            "model.semantic=mlp model.cqig=false "
            "trainer.args.do_train=false trainer.args.do_eval=false "
            "+trainer.args.eval_batch_size=2 "
            "+trainer.args.do_predict=true +trainer.args.predict_top_k=300 "
            f"+trainer.args.resume_from_checkpoint={ckpt} "
            f"hydra.run.dir={run_local}",
            "/content/gfm-rag", extra=extra, log=f"{run_local}/console.log", check=False)
    log = open(f"{run_local}/console.log", errors="ignore").read()
    assert rc == 0, f"predict failed (exit {rc}); read {run_local}/console.log"
    assert "semantic='mlp' warm start" in log, "the multi-view scorer was never constructed; check SEMANTIC_* env"
    if info["resp_keys"]:
        assert "[ccmp] responsibility head" in log, "CCMP head not constructed: the checkpoint's head weights were dropped"
    pred_local = f"{run_local}/predictions_{RBG}.json"
    assert os.path.exists(pred_local), f"no predictions written under {run_local}"
    save_index(RBG, CACHE_RB)
    sync_dir(run_local, run_drive)
    return pred_drive
print("predict_rb ready")
'''

ARM_A = '''# 7b. Arm A: SciGraphIR trained on TOMATO-Star, frozen, zero-shot on ResearchBench.
TOMATO_CKPT = f"{TOMATO_OUT}/{TOMATO_RUN}/model_best.pth"
SEM_CKPT_T  = f"{TOMATO_SEM}/params_semantic_mlp_fixedloss_tomato.json"
SEM_POP_T   = f"{TOMATO_SEM}/popnet_semantic_mlp_fixedloss_tomato.pt"
for p in (TOMATO_CKPT, SEM_CKPT_T, SEM_POP_T):
    assert os.path.exists(p), f"missing {p}"
print("TOMATO-Star runs available:")
for d in sorted(glob.glob(f"{TOMATO_OUT}/tomato_fusion_qwenmlp_ccmp*")):
    if os.path.exists(f"{d}/model_best.pth"):
        print(f"   {'>' if d.endswith(TOMATO_RUN) else ' '} {os.path.basename(d):50} "
              f"{time.strftime('%d %b', time.localtime(os.path.getmtime(f'{d}/model_best.pth')))}")

PRED_TOMATO = predict_rb(TOMATO_CKPT, f"scigraphir_tomato__{TOMATO_RUN}", SEM_CKPT_T, SEM_POP_T)
score("scigraphir_tomato", PRED_TOMATO, f"SciGraphIR (TOMATO-Star: {TOMATO_RUN}) zero-shot")
'''

ARM_PB = '''# 7c. Arm A2 (optional): SciGraphIR trained on SIR-4 Physics + Biology, the Table 9.3 model
# from colab_sir4_zeroshot.ipynb, frozen, zero-shot on ResearchBench. Runs only when that
# checkpoint exists on Drive; otherwise it is skipped and the table simply lacks the row.
# Its scorer warm-start files are the ones that notebook saved next to the run.
PB_OUT = f"{DRIVE}/outputs/sir4_zeroshot"
_c = sorted(glob.glob(f"{PB_OUT}/scigraphir_physics+biology_*/model_best.pth"), key=os.path.getmtime)
if not _c:
    print("no Physics+Biology checkpoint under", PB_OUT, "-- run colab_sir4_zeroshot.ipynb first; arm skipped")
else:
    PB_CKPT = _c[-1]; PB_RUN = os.path.basename(os.path.dirname(PB_CKPT))
    _arm = json.load(open(f"{os.path.dirname(PB_CKPT)}/arm.json")) if os.path.exists(f"{os.path.dirname(PB_CKPT)}/arm.json") else {}
    _warm = _arm.get("scorer_warm_start", "sir4_biology")          # the field whose 5d run initialised the scorer
    SEM_CKPT_PB = f"{PB_OUT}/semantic_{_warm}/params_semantic_mlp_fixedloss_{_warm}.json"
    SEM_POP_PB  = f"{PB_OUT}/semantic_{_warm}/popnet_semantic_mlp_fixedloss_{_warm}.pt"
    for p in (SEM_CKPT_PB, SEM_POP_PB):
        assert os.path.exists(p), f"missing scorer warm-start file {p}"
    print("Physics+Biology run:", PB_RUN, "| trained on", _arm.get("train"), "| scorer warm start", _warm)
    PRED_PB = predict_rb(PB_CKPT, f"scigraphir_pb__{PB_RUN}", SEM_CKPT_PB, SEM_POP_PB)
    score("scigraphir_pb", PRED_PB, f"SciGraphIR (SIR-4 Physics + Biology: {PB_RUN}) zero-shot")
'''

ARM_MIR = '''# 7d. Arm A3 (optional): SciGraphIR trained on MIR (Methodology Inspiration Retrieval, ACL
# 2025; computational linguistics only, 1,270 training proposals), the "+ CCMP" run from
# colab_mir_standard.ipynb, frozen, zero-shot on ResearchBench. The most different training
# corpus of the three: single field, citation-derived golds, abstracts only. Runs only when
# the checkpoint exists on Drive; otherwise skipped and the table lacks the row.
#
# FROM A FRESH RUNTIME this cell needs only sections 1, 2, 2b, 5, 6 and 7a above. Sections 3,
# 4 and 8 can be left unrun: every finished arm is scored from its per-query file on Drive.
MIR_OUT = f"{DRIVE}/outputs/mir"
MIR_RUN = "mir_qwenmlp_ccmp_e10_b2"                     # standard CCMP, 10 epochs, batch 2 (6 Sep)
MIR_CKPT = f"{MIR_OUT}/{MIR_RUN}/model_best.pth"
SEM_CKPT_M = f"{MIR_OUT}/semantic/params_semantic_mlp_fixedloss_mir.json"
SEM_POP_M  = f"{MIR_OUT}/semantic/popnet_semantic_mlp_fixedloss_mir.pt"
if not os.path.exists(MIR_CKPT):
    print("no MIR checkpoint at", MIR_CKPT, "-- run colab_mir_standard.ipynb first; arm skipped")
    print("   present under", MIR_OUT, ":", sorted(os.listdir(MIR_OUT)) if os.path.isdir(MIR_OUT) else "(dir missing)")
else:
    for p in (SEM_CKPT_M, SEM_POP_M):
        assert os.path.exists(p), f"missing scorer warm-start file {p}"
    _aj = f"{MIR_OUT}/{MIR_RUN}/arm.json"
    if os.path.exists(_aj): print("MIR run:", MIR_RUN, "|", {k: v for k, v in json.load(open(_aj)).items() if k in ("arm", "train", "epochs", "ccmp", "ccmp_variant")})
    PRED_MIR = predict_rb(MIR_CKPT, f"scigraphir_mir__{MIR_RUN}", SEM_CKPT_M, SEM_POP_M)
    score("scigraphir_mir", PRED_MIR, f"SciGraphIR (MIR: {MIR_RUN}) zero-shot")
'''

S4_UNPACK = '''# 8a. The four SIR-4 fields: graphs, corpora, caches. Nothing from ResearchBench enters here.
S4_CACHE = {d: f"{DRIVE}/outputs/sir4_{d}/cache" for d in S4_DOMS}
for d in S4_DOMS:
    z = f"{DRIVE}/sir4_{d}_bundle.zip"
    assert os.path.exists(z) and zipfile.is_zipfile(z), f"{z} missing or not a zip"
    if not os.path.exists(f"{DATA_ROOT}/sir4_{d}_train_v16sc/processed/stage1/nodes.csv"):
        zipfile.ZipFile(z).extractall(CARGO_ROOT); print("unpacked", os.path.basename(z))
apply_overlay()          # the bundles carry older copies of the scripts; the repo's win

def g_of(d, s): return f"sir4_{d}_{s}_v16sc"
for d in S4_DOMS:
    for s in ("train", "test"):
        dst = f"{DATA_ROOT}/{g_of(d, s)}/raw"; os.makedirs(dst, exist_ok=True)
        shutil.copy(f"{DATA_ROOT}/sir4_{d}_{s}/raw/documents.json", f"{dst}/documents.json")
    if os.path.isdir(f"{S4_CACHE[d]}/op_emb"):
        shutil.copytree(f"{S4_CACHE[d]}/op_emb", f"{CARGO_ROOT}/outputs/caches/op_emb", dirs_exist_ok=True)
        print(f"{d:9} embedding caches restored")
    for s in ("train", "test"):
        restore_index(g_of(d, s), S4_CACHE[d])

# operator components (Qwen3) per graph: restored from the training runs' caches, else computed.
import numpy as np
def opc(d, s): return f"{DATA_ROOT}/{g_of(d, s)}/operator_components{OP_SLUG}.npz"
for d in S4_DOMS:
    for s in ("train", "test"):
        p, c = opc(d, s), f"{S4_CACHE[d]}/{g_of(d, s)}_operator_components{OP_SLUG}.npz"
        if not os.path.exists(p):
            if os.path.exists(c): shutil.copy(c, p)
            else:
                sh(f"python3 -u experiments/probe_greasoner/precompute_operator_components.py "
                   f"--dataset sir4_{d} --graph {g_of(d, s)} --split {s} --model {OP_MODEL}",
                   KGDIR, extra={"CARGO_DATASET": f"sir4_{d}"})
                os.makedirs(os.path.dirname(c), exist_ok=True); shutil.copy(p, c)
        zz = np.load(p, allow_pickle=True)
        assert "qwen" in str(zz["encoder"]).lower(), f"{p} was built with {zz['encoder']!r}, not Qwen3"
        print(f"  {g_of(d, s):24} dense {zz['dense'].shape}")

# Every query must seed, or the loader drops it silently and the metric is over fewer queries.
import csv, io
csv.field_size_limit(10 ** 7)
for d in S4_DOMS:
    for s in ("train", "test"):
        s1 = f"{DATA_ROOT}/{g_of(d, s)}/processed/stage1"
        q = json.load(open(f"{s1}/{s}.json"))
        seedless = sum(not any(x["start_nodes"].values()) for x in q)
        assert seedless == 0, f"{g_of(d, s)}: {seedless} seedless queries"
        print(f"  {g_of(d, s):24} {len(q):>6,} queries, all seeded")
'''

S4_SCORER = '''# 8b. Warm-start the multi-view scorer on ONE SIR-4 field (SEM_WARM_DOM). The same 5d
# recipe as the TOMATO-Star run: sorted-MLP over the hypothetical-answer match profile with a
# jointly trained popularity predictor, --loss fixed (multi-gold), one seed. Its weights are
# only the INITIALISATION: the fusion in 8e trains them on all four fields.
SEM_S4 = f"{S4}/results/semantic_sir4_{SEM_WARM_DOM}"
SEM_S4_DRIVE = f"{OUT_ROOT}/semantic_sir4_{SEM_WARM_DOM}"
SEM_CKPT_S4 = f"{SEM_S4}/params_semantic_mlp_fixedloss_sir4_{SEM_WARM_DOM}.json"
SEM_POP_S4  = f"{SEM_S4}/popnet_semantic_mlp_fixedloss_sir4_{SEM_WARM_DOM}.pt"
os.makedirs(SEM_S4, exist_ok=True)
if os.path.isdir(SEM_S4_DRIVE):
    shutil.copytree(SEM_S4_DRIVE, SEM_S4, dirs_exist_ok=True); print("scorer restored from Drive")
if not (os.path.exists(SEM_CKPT_S4) and os.path.exists(SEM_POP_S4)):
    sh(f"python3 -u eval/semantic_scorer.py --dataset sir4_{SEM_WARM_DOM} --model {OP_MODEL} "
       f"--arms mlp --loss fixed --train_fit 0 --dev 300 --epochs 30 --patience 10 "
       f"--lr 1e-3 --weight_decay 1e-2 --mlp_hidden 16 --mlp_pop_joint 1 --pop_lambda 1.0 "
       f"--select_on loss --qbatch 32 --seed 0 --seeds 0 --out {SEM_S4}",
       S4, extra={"CARGO_DATASET": f"sir4_{SEM_WARM_DOM}"})
    shutil.copytree(SEM_S4, SEM_S4_DRIVE, dirs_exist_ok=True)
for p in (SEM_CKPT_S4, SEM_POP_S4):
    assert os.path.exists(p), f"5d did not write {p}"
_st = json.load(open(SEM_CKPT_S4))
print(f"scorer warm start: jmax={_st['jmax']} beta={_st['beta']:.4f} hidden={len(_st['net']['0.weight'])}")
'''

S4_SEMCOMP = '''# 8c. Multi-view scorer components for all eight SIR-4 graphs (H memmaps live under
# /content and are rebuilt from the cached embeddings after a reset; a few minutes each).
def semc(d, s): return f"{DATA_ROOT}/{g_of(d, s)}/semantic_components{OP_SLUG}.npz"
for d in S4_DOMS:
    for s in ("train", "test"):
        if not _sem_ok(semc(d, s)):
            sh(f"python3 -u experiments/probe_greasoner/precompute_semantic_components.py "
               f"--dataset sir4_{d} --model {OP_MODEL} --graph {g_of(d, s)} --split {s}",
               KGDIR, extra={"CARGO_DATASET": f"sir4_{d}"})
        zz = np.load(semc(d, s), allow_pickle=True)
        print(f"  {g_of(d, s):24} H {tuple(int(x) for x in zz['h_shape'])}  Jmax={int(zz['Jmax'])}")
    shutil.copytree(f"{CARGO_ROOT}/outputs/caches/op_emb", f"{S4_CACHE[d]}/op_emb", dirs_exist_ok=True)

TRAIN_G = [g_of(d, "train") for d in S4_DOMS]
VALID_G = [g_of(d, "test") for d in S4_DOMS]
OPC_TR, OPC_TE = ",".join(opc(d, "train") for d in S4_DOMS), ",".join(opc(d, "test") for d in S4_DOMS)
SEM_TR, SEM_TE = ",".join(semc(d, "train") for d in S4_DOMS), ",".join(semc(d, "test") for d in S4_DOMS)

# THE SIR-4 ARM'S ENVIRONMENT. Identical to the TOMATO-Star "+ CCMP" arm's run_model call
# (semantic=mlp, ccmp_w=1.0, ccmp_neg=64, ccmp_m=2000, ccmp_lr=2e-3, gate on, eta 0.5,
# pool 256, +/-512 nodes per layer) plus the multi-gold correction PER_GOLD=1 and the
# graph-negative term HARDNEG_GRAPH=50, which that arm also ran with.
S4_ENV = dict(WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
              PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
              OPERATOR_COMPONENTS=OPC_TR, OPERATOR_COMPONENTS_TEST=OPC_TE,
              SEMANTIC_COMPONENTS=SEM_TR, SEMANTIC_COMPONENTS_TEST=SEM_TE,
              SEMANTIC_CKPT=SEM_CKPT_S4, SEMANTIC_POPNET=SEM_POP_S4, SEM_POP_LAMBDA="1.0",
              FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0",
              PER_GOLD="1", HARDNEG_GRAPH="50",
              CCMP="1", CCMP_W="1.0", CCMP_NEG="64", CCMP_M="2000", CCMP_LR="2e-3",
              CCMP_GATE="1", CCMP_GATE_NORM="1", CCMP_ETA="0.5",
              CCMP_POOL="256", CCMP_M_POS="512", CCMP_M_NEG="512",
              # RESIDUAL CCMP (the 31 Aug variant): supervise only the golds the scorer has not
              # resolved, identity-supervise the rest. "0" = the 30 Aug standard targets.
              CCMP_RESIDUAL="__CCMP_RESID__", CCMP_IDENTITY_W="1.0",
              BATCH_BY_DATASET=json.dumps(BATCH_BY_DATASET))
CCMP_TAG = "ccmpresid" if S4_ENV["CCMP_RESIDUAL"] == "1" else "ccmp"
print("CCMP variant:", "RESIDUAL" if CCMP_TAG == "ccmpresid" else "standard")
# STRAT_TEST / STRAT_BGE deliberately unset: the per-epoch slice labels assume ONE validation
# corpus. The whole-set per-epoch numbers stay correct; the slices come from section 9.
for k in list(os.environ):
    if k.startswith(("STRAT_", "CCMP")): os.environ.pop(k)

def s4_cmd(run_dir, epochs, max_steps=None):
    return ("python -u -m gfmrag.workflow.sft_training "
            "--config-path config/gfm_reasoner --config-name sft_training_fusion "
            "text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{','.join(TRAIN_G)}] datasets.valid_names=[{','.join(VALID_G)}] "
            "datasets.max_datasets_in_memory=1 "        # one graph resident at a time
            # No background prefetch. With 4 workers the loader pulls the OTHER graphs into RAM
            # while one trains, times out at 30 s ("Error loading dataset ... :") and then loads
            # synchronously anyway. Serial loading costs nothing and holds exactly one graph.
            "datasets.data_loading_workers=0 "
            "model.semantic=mlp model.cqig=false "
            f"trainer.args.num_epoch={epochs} trainer.args.train_batch_size={BATCH} "
            f"+trainer.args.eval_batch_size={BATCH} "
            + (f"trainer.args.max_steps_per_epoch={max_steps} " if max_steps else "")
            + f"hydra.run.dir={run_dir}")
print("training on ", TRAIN_G)
print("selecting on", VALID_G, "(dev numbers; ResearchBench is the only clean test)")
'''

BATCH_PATCH = r'''# 8c-ii. Per-graph training batch size. sft_trainer builds one DataLoader per graph per
# epoch with a single train_batch_size; this reads BATCH_BY_DATASET (a JSON dict keyed by
# graph name) and overrides it for the graphs named there. Idempotent; re-running is a no-op.
# Must run again after any reconnect that reinstalled the engine.
STR = "/content/gfm-rag/gfmrag/trainers/sft_trainer.py"
src = open(STR).read()
OLD = ("        batch_size = (\n"
       "            self.args.train_batch_size if is_train else self.args.eval_batch_size\n"
       "        )\n")
NEW = OLD + ("        # PATCH (notebook): per-graph override, BATCH_BY_DATASET='{\"<graph>\": n}'.\n"
             "        import os as _os, json as _json\n"
             "        _bbd = _os.environ.get(\"BATCH_BY_DATASET\", \"\")\n"
             "        if _bbd and is_train:\n"
             "            batch_size = int(_json.loads(_bbd).get(data_name, batch_size))\n"
             "            print(f\"[batch] {data_name}: train batch {batch_size}\", flush=True)\n")
if "BATCH_BY_DATASET" in src:
    print("already patched: per-graph batch size")
elif OLD in src:
    open(STR, "w").write(src.replace(OLD, NEW, 1)); print("patched sft_trainer.py: per-graph batch size")
else:
    raise SystemExit("could not find the batch_size block in sft_trainer.py -- engine changed, check by hand")
'''

S4_SMOKE = '''# 8d. Smoke: 1 epoch, 3 steps per graph, ~10-15 min plus any index build. It exists because
# the four-way dataset switch with EIGHT semantic tables is the only new thing in this run;
# everything else is a path the TOMATO and single-field runs exercise nightly.
SMOKE = f"{RUNS}/scigraphir_sir4_all4_smoke"
os.makedirs(SMOKE, exist_ok=True)
rc = sh(s4_cmd(SMOKE, epochs=1, max_steps=3), "/content/gfm-rag", extra=S4_ENV,
        log=f"{SMOKE}/console.log", check=False)
log = open(f"{SMOKE}/console.log", errors="ignore").read()
assert rc == 0, f"SMOKE FAILED (exit {rc}). Do not start the long run."
for g in TRAIN_G:
    assert g in log, f"{g} never appears in the log; the loader did not reach it"
n_sem = log.count("[fusion] loaded SEMANTIC_COMPONENTS")
assert n_sem >= 8, f"only {n_sem} semantic tables loaded, expected 8 (4 train + 4 test)"
assert "[ccmp] responsibility head" in log, "CCMP head not constructed"
assert ("RESIDUAL on" in log) == (CCMP_TAG == "ccmpresid"), "the trainer did not run the CCMP variant that was asked for"
assert "semantic='mlp' warm start" in log, "multi-view scorer not constructed"
for _g, _b in BATCH_BY_DATASET.items():
    if _g in TRAIN_G:
        assert f"[batch] {_g}: train batch {_b}" in log, f"per-graph batch override for {_g} did not apply; run the 8c-ii patch cell"
for d in S4_DOMS:
    for s in ("train", "test"): save_index(g_of(d, s), S4_CACHE[d])
print(f"\\nSMOKE PASSED: {n_sem} semantic tables, all four training graphs reached, CCMP head built.")
'''

S4_TRAIN = '''# 8e. TRAIN SciGraphIR on all four SIR-4 fields. Hours (the two-field joint run took ~7 h for
# 10 epochs; expect roughly double). Runs LOCALLY with a 10-minute sync to Drive; a finished
# run on Drive is a [skip], so a reconnect never retrains a converged model.
S4_NAME  = f"scigraphir_sir4_all4_qwenmlp_{CCMP_TAG}_e{EPOCHS}_b{BATCH}"
S4_LOCAL, S4_DRIVE = f"{RUNS}/{S4_NAME}", f"{OUT_ROOT}/{S4_NAME}"
S4_CKPT = f"{S4_DRIVE}/model_best.pth"
if os.path.exists(S4_CKPT) and os.path.exists(f"{S4_DRIVE}/console.log") and \
        f"Epoch {EPOCHS} completed" in open(f"{S4_DRIVE}/console.log", errors="ignore").read():
    print(f"[skip] finished run on Drive: {os.path.relpath(S4_CKPT, DRIVE)}")
else:
    os.makedirs(S4_LOCAL, exist_ok=True); os.makedirs(S4_DRIVE, exist_ok=True)
    json.dump({"arm": "SciGraphIR (SIR-4, all four fields)", "train": TRAIN_G, "valid": VALID_G,
               "epochs": EPOCHS, "batch": BATCH, "semantic": "mlp", "scorer_warm_start": f"sir4_{SEM_WARM_DOM}",
               "ccmp": {k: v for k, v in S4_ENV.items() if k.startswith("CCMP")},
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{S4_LOCAL}/arm.json", "w"), indent=1)
    finish = start_sync(S4_LOCAL, S4_DRIVE, every=600)
    try:
        rc = sh(s4_cmd(S4_LOCAL, epochs=EPOCHS), "/content/gfm-rag", extra=S4_ENV,
                log=f"{S4_LOCAL}/console.log", check=False)
    finally:
        finish()
    assert rc == 0, f"training failed, exit {rc} (-9 means the host ran out of RAM)"
    assert os.path.exists(f"{S4_LOCAL}/model_best.pth"), "training finished but wrote no model_best.pth"
    sync_dir(S4_LOCAL, S4_DRIVE)
assert os.path.exists(S4_CKPT), S4_CKPT
print("checkpoint:", S4_CKPT)
'''

S4_PREDICT = '''# 8f. Arm B on ResearchBench: the SIR-4 checkpoint, frozen, zero-shot.
PRED_S4 = predict_rb(S4_CKPT, f"scigraphir_sir4__{S4_NAME}", SEM_CKPT_S4, SEM_POP_S4)
score("scigraphir_sir4", PRED_S4, f"SciGraphIR (SIR-4, 4 fields: {S4_NAME}) zero-shot")
'''

RESULTS = '''# 9. The table. Every arm from its per-query file, one scorer, so a runtime reset never
# requires prediction or scoring to be repeated. Slices: all queries, same-field-only
# queries, queries with a cross-field gold (ResearchBench's own domain_slice), and the
# strict zero-shot subset frozen in subsets.json before any model ran.
SCORES = f"{OUT_ROOT}/scores"
ORDER = [("bm25", "BM25"), ("bge", "BGE-large"), ("qwen3", "Qwen3-Embedding"),
         ("specter2", "SPECTER2-base"), ("scincl", "SciNCL"), ("reasonir", "ReasonIR-8B"),
         ("scigraphir_tomato", "SciGraphIR (TOMATO-Star)"),
         ("scigraphir_pb", "SciGraphIR (SIR-4 Physics+Biology)"),
         ("scigraphir_mir", "SciGraphIR (MIR)"),
         ("scigraphir_sir4", "SciGraphIR (SIR-4, 4 fields)")]
MET = [("mrr", "MRR"), ("ndcg@5", "nDCG@5"), ("recall@3", "R@3"), ("recall@5", "R@5"),
       ("recall@15", "R@15"), ("completeset@5", "CGS@5")]
STRICT = set(json.load(open(SUBSETS))["subsets"]["strict_zeroshot"])

PQ = {}
for key, lab in ORDER:
    p = f"{SCORES}/{key}_perquery.json"
    if os.path.exists(p): PQ[key] = json.load(open(p))
    else: print(f"  {lab}: not run yet ({os.path.basename(p)} missing)")

def agg(rows, sel):
    r = [v for v in rows.values() if sel(v)]
    return {"n": len(r), **{m: sum(x[m] for x in r) / len(r) for m, _ in MET}} if r else None

SLICES = [("all", lambda v: True), ("same-field only", lambda v: "same" in v["slices"]),
          ("cross-field involved", lambda v: "cross" in v["slices"]),
          ("strict zero-shot (923)", None)]
lines = []
def out(s=""): print(s); lines.append(s)
out(f"# ResearchBench zero-shot: full-corpus retrieval over {len(json.load(open(QUERIES))):,} queries\\n")
out(f"TOMATO-Star arm: {TOMATO_RUN}. SIR-4 arm: {globals().get('S4_NAME', '(not trained in this session)')}.\\n")
T = {}
for sname, sel in SLICES:
    out(f"### {sname}")
    out("| method | n | " + " | ".join(l for _, l in MET) + " |")
    out("|---|--:|" + "--:|" * len(MET))
    for key, lab in ORDER:
        if key not in PQ: continue
        rows = PQ[key]
        if sel is None:                      # the frozen subset is a query-id list, not a slice label
            rows, sel_ = {q: v for q, v in rows.items() if q in STRICT}, (lambda v: True)
        else:
            sel_ = sel
        a = agg(rows, sel_)
        if not a: continue
        T.setdefault(sname, {})[lab] = a
        out(f"| {lab} | {a['n']} | " + " | ".join(f"{a[m]:.4f}" for m, _ in MET) + " |")
    out()

# Table 9.3 layout: the cross-field slice, R@3 / R@5 / nDCG@5.
out("### Table 9.3 layout (cross-field involved)")
out("| Method | R@3 | R@5 | nDCG@5 |"); out("|---|--:|--:|--:|")
for key, lab in ORDER:
    a = T.get("cross-field involved", {}).get(lab)
    if a: out(f"| {lab} | {a['recall@3']:.4f} | {a['recall@5']:.4f} | {a['ndcg@5']:.4f} |")
open(f"{OUT_ROOT}/rb_zeroshot_results.md", "w").write("\\n".join(lines))
json.dump(T, open(f"{OUT_ROOT}/rb_zeroshot_results.json", "w"), indent=1)
print("\\nwrote", f"{OUT_ROOT}/rb_zeroshot_results.md")
'''

STATS = '''# 9b. Per-discipline breakdown and paired bootstrap. Paired, because every arm ranked the
# same queries over the same corpus; an unpaired test would throw the pairing away.
arms = ""
for key, lab in ORDER:
    p = f"{SCORES}/{key}_perquery.json"
    if os.path.exists(p): arms += f"--arm '{lab}={p}' "
sh(f"python3 -u eval/report_domain_results.py {arms} "
   f"--md-out {OUT_ROOT}/results_by_discipline.md --json-out {OUT_ROOT}/results_by_discipline.json", S4)

def paired(a, an, b, bn, tag, subset=None):
    if not (os.path.exists(f"{SCORES}/{a}_perquery.json") and os.path.exists(f"{SCORES}/{b}_perquery.json")):
        print(f"[skip] {an} vs {bn}: a per-query file is missing"); return
    sh(f"python3 -u transfer/paired_bootstrap.py "
       f"--a {SCORES}/{a}_perquery.json --a-name '{an}' --b {SCORES}/{b}_perquery.json --b-name '{bn}' "
       f"--metrics mrr,ndcg@5,recall@3,recall@5,recall@15,completeset@5 "
       + (f"--subset {subset} --subsets-file {SUBSETS} " if subset else "")
       + f"--json-out {OUT_ROOT}/paired_{tag}{'_' + subset if subset else ''}.json", S4)

for subset in (None, "strict_zeroshot"):
    paired("scigraphir_sir4", "SciGraphIR (SIR-4)", "scigraphir_tomato", "SciGraphIR (TOMATO-Star)", "sir4_vs_tomato", subset)
    paired("scigraphir_pb", "SciGraphIR (Physics+Biology)", "scigraphir_tomato", "SciGraphIR (TOMATO-Star)", "pb_vs_tomato", subset)
    paired("scigraphir_pb", "SciGraphIR (Physics+Biology)", "qwen3", "Qwen3-Embedding", "pb_vs_qwen3", subset)
    paired("scigraphir_mir", "SciGraphIR (MIR)", "scigraphir_tomato", "SciGraphIR (TOMATO-Star)", "mir_vs_tomato", subset)
    paired("scigraphir_mir", "SciGraphIR (MIR)", "qwen3", "Qwen3-Embedding", "mir_vs_qwen3", subset)
    paired("scigraphir_tomato", "SciGraphIR (TOMATO-Star)", "qwen3", "Qwen3-Embedding", "tomato_vs_qwen3", subset)
    paired("scigraphir_sir4", "SciGraphIR (SIR-4)", "qwen3", "Qwen3-Embedding", "sir4_vs_qwen3", subset)
    paired("scigraphir_sir4", "SciGraphIR (SIR-4)", "reasonir", "ReasonIR-8B", "sir4_vs_reasonir", subset)
'''

HEADER = '''# ResearchBench — zero-shot cross-benchmark transfer

**Nothing here is trained, tuned or selected on ResearchBench.** Every arm ranks the same
20,322-document corpus for the same 1,367 queries and is scored by one `score_sir4.py` call
with one flag set, so a difference between two rows is the retriever and nothing else.

| family | arm | what runs |
|---|---|---|
| lexical | BM25 | section 3, CPU |
| dense | BGE-large, Qwen3-Embedding, SPECTER2-base, SciNCL | section 3, GPU minutes |
| reasoning-trained dense | ReasonIR-8B | section 3, bf16, the slow baseline |
| **SciGraphIR (TOMATO-Star)** | frozen checkpoint from Drive: multi-view scorer + graph reasoner + CCMP | section 7, one predict pass |
| **SciGraphIR (SIR-4, 4 fields)** | **trained here** on cs + biology + physics + matsci, then frozen | section 8, hours |

The baseline arms are the same six definitions as `sir4_baselines_all.ipynb` (harvested at
build time), so this table and Table 9.3 cannot define a baseline differently. The two
disclosures from that notebook apply: SPECTER2 is the base encoder without its retrieval
adapters, and SPECTER2/SciNCL see `Title. Abstract` rather than `title [SEP] abstract`.

**Reading the SIR-4 arm.** Its multi-view scorer is warm-started on one field
(`SEM_WARM_DOM`) because the scorer fitter takes one corpus; the fusion then trains the
scorer, the graph reasoner and the CCMP head jointly on all four. Its SIR-4 test numbers
are dev numbers (they drove checkpoint selection); ResearchBench is its only clean test.

Sections 1-5 need no engine. Stop after section 5 for the baseline table alone.
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    # BATCH 1, NOT 2. CS is the largest graph (243k nodes) and every August CS run with the
    # multi-view scorer trained at batch 1 (sir4_cs_fusion_qwenmlp_*_b1). At batch 2 the
    # four-field smoke OOMed on the first CS step with 74 GB allocated on an 80 GB A100.
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--warm-dom", default="cs", choices=["cs", "biology", "physics", "matsci"])
    ap.add_argument("--ccmp", default="standard", choices=["standard", "residual"],
                    help="CCMP target construction for the trained arm, and which TOMATO-Star checkpoint arm A loads: "
                         "standard = 30 Aug targets (TOMATO run ..._ccmp_epoch10_b2, the Table 9.1 row); "
                         "residual = 31 Aug residual CCMP (TOMATO run tomato_fusion_qwenmlp_ccmpresid_epoch10_b2)")
    ap.add_argument("--out", default=f"{ROOT}/colab_rb_zeroshot.ipynb")
    a = ap.parse_args()

    src = json.load(open(SRC_NB))["cells"]
    base = json.load(open(BASE_NB))["cells"]
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    resid = "1" if a.ccmp == "residual" else "0"
    tomato_run = ("tomato_fusion_qwenmlp_ccmpresid_epoch10_b2" if a.ccmp == "residual"
                  else "tomato_fusion_qwenmlp_ccmp_epoch10_b2")
    global S4_SEMCOMP, PATHS
    S4_SEMCOMP = S4_SEMCOMP.replace("__CCMP_RESID__", resid)
    PATHS = PATHS.replace("__TOMATO_RUN__", tomato_run)

    # --- the fusion sources, regenerated from the repo (same list as build_notebook.py) ---
    fusion_files = {f"/content/gfm-rag/{rel}": open(f"{FORK}/{rel}").read() for rel in FUSION_REL}
    assert not any("'''" in v for v in fusion_files.values()), "fusion source contains '''"
    files_cell = code(
        "# === write the CARGO-fusion files into the fork (generated from the repo copies "
        f"{built}) ===\n"
        "# The TOMATO-Star checkpoint is loaded by THIS model code, so it is the repo's current\n"
        "# fusion_reasoner.py / fusion_trainer.py, not the copies frozen in an older notebook.\n"
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

    # --- repo scripts shipped inline ---
    overlay = {rel: open(f"{CARGO}/{rel}").read() for rel in OVERLAY_REL}
    assert not any("'''" in v for v in overlay.values()), "an overlay script contains '''"
    for rel, need in (("sir4-retrieval/eval/score_sir4.py", ["--gold-strata", "--per-query-out", "--cols"]),
                      ("sir4-retrieval/eval/baselines_sir4.py", ["--pooling", "--trust-remote-code"]),
                      ("sir4-retrieval/eval/semantic_scorer.py", ["SortedMLPScorer", "MatchabilityPredictor"]),
                      ("kg-construction/eval/cargo_operator.py", ["def model_slug", "def load_split"])):
        miss = [t for t in need if t not in overlay[rel]]
        assert not miss, f"{rel} lacks {miss}; the notebook would ship a script that cannot do its job"

    cells = [md(HEADER), md("## 1. GPU + Drive + paths"),
             code(PATHS.replace("__WARM__", a.warm_dom)
                  .replace("__EPOCHS__", str(a.epochs)).replace("__BATCH__", str(a.batch))),
             md("## 2. Unpack ResearchBench + install the current scripts"),
             code(UNPACK.replace("__OVERLAY__", json.dumps(overlay)).replace("__BUILT__", built))]

    # Qwen3 weights (cloned): the dense Qwen3 baseline and every fusion arm read /content/qwen3
    cells.append(md("## 2b. Qwen3-Embedding-0.6B (cached on Drive)\n*(cloned from the TOMATO-Star notebook)*"))
    cells.append(src[QWEN_CELLS[1]])

    cells.append(md("## 3. Baselines — no engine needed\n"
                    "### 3a. The arms\n*(harvested verbatim from `sir4_baselines_all.ipynb`, "
                    "so they cannot differ from Table 9.3's)*"))
    arms_src = "".join(base[ARMS_CELL]["source"])
    assert "ARMS = [" in arms_src and "ReasonIR-8B" in arms_src, "ARMS cell not found where expected"
    cells.append(code(arms_src + ARMS_TAIL))
    cells.append(md("### 3b. Run every arm"))
    cells.append(code(RUN_BASELINES))
    cells.append(md("## 4. Score the baselines"))
    cells.append(code(SCORE_BASELINES))

    cells.append(md("## 5. Engine + fusion sources\n"
                    "*(cloned from `tomato_ccmp_ablation.ipynb`, the notebook that trained the "
                    "TOMATO-Star checkpoint; the fusion-source blob is regenerated from the repo)*"))
    for i in ENGINE_CELLS:
        cells.append(files_cell if i == 7 else src[i])

    cells.append(md("## 6. ResearchBench components for the fusion"))
    cells.append(code(RB_COMPONENTS))
    cells.append(md("## 7. Arm A — SciGraphIR trained on TOMATO-Star, zero-shot"))
    cells.append(code(PREDICT_HELPER))
    cells.append(code(ARM_A))
    cells.append(md("### 7c. Arm A2 (optional) — SciGraphIR trained on SIR-4 Physics + Biology, zero-shot"))
    cells.append(code(ARM_PB))
    cells.append(md("### 7d. Arm A3 (optional) — SciGraphIR trained on MIR, zero-shot"))
    cells.append(code(ARM_MIR))
    cells.append(md("## 8. Arm B — SciGraphIR trained on all four SIR-4 fields, zero-shot\n"
                    "**Hours.** Sections 8a-8c are minutes; 8d is a 10-15 minute smoke; "
                    "8e is the long run and syncs to Drive every 10 minutes."))
    cells.append(code(S4_UNPACK))
    cells.append(code(S4_SCORER))
    cells.append(code(S4_SEMCOMP))
    cells.append(code(BATCH_PATCH))
    cells.append(code(S4_SMOKE))
    cells.append(code(S4_TRAIN))
    cells.append(code(S4_PREDICT))
    cells.append(md("## 9. Results"))
    cells.append(code(RESULTS))
    cells.append(code(STATS))

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"},
                                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(cells)} cells | ccmp {a.ccmp} | fusion sources {sum(len(v) for v in fusion_files.values())/1e3:.0f} KB"
          f" | overlay {len(overlay)} scripts {sum(len(v) for v in overlay.values())/1e3:.0f} KB"
          f" | epochs {a.epochs} batch {a.batch} warm-start field {a.warm_dom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
