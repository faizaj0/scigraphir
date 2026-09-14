"""
build_sir4_zeroshot_notebook.py -- Table 9.3 with the CURRENT model: SciGraphIR (multi-view
scorer + graph reasoner + CCMP) trained on SIR-4 Physics + Biology, evaluated without
adaptation on the cross-field queries of Computer Science and Materials Science, next to
all six baselines.

WHY A NEW RUN. The SciGraphIR row printed in Table 9.3 came from the joint
Physics+Biology run of 9 Aug (build_joint_notebook.py), which is the operator fusion:
no multi-view scorer, no CCMP. Table 9.1's SciGraphIR is the updated model, so the two
tables currently describe different systems under one name. This notebook trains the
updated model under Table 9.3's protocol. The baseline rows are the SAME six arm
definitions as sir4_baselines_all.ipynb, harvested at build time; their predictions are
already on Drive and are reused, so the baseline table is a rescore, not a rerun.

LEAKAGE CONTROL, unchanged from the joint run: valid_names is the Physics and Biology
TEST graphs, so checkpoint selection never touches CS or MatSci. Those two are loaded only
after training ends, and their test queries are asserted disjoint from every training id.

Everything reusable is imported from build_rb_zeroshot_notebook.py (engine cloning, the
fusion-source blob, the inline script overlay, the SIR-4 cache/scorer/smoke/train cells),
with the domain lists substituted, so the two notebooks cannot define the training recipe
differently.

Usage
-----
    python3 prep/build_sir4_zeroshot_notebook.py
    python3 prep/build_sir4_zeroshot_notebook.py --epochs 10 --batch 2 --warm-dom biology
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
from stage_sir4 import DOMAINS as _DOMAINS  # noqa: E402

ROOT, CARGO, FORK = rb.ROOT, rb.CARGO, rb.FORK
md, code = rb.md, rb.code
SETS_MAP = {d: f"quartet/data/benchmark/{_DOMAINS[d][1]}/sets.json" for d in sorted(_DOMAINS)}

HEADER = '''# SIR-4 — zero-shot transfer across fields (Table 9.3, updated model)

Train **SciGraphIR** (multi-view scorer + graph reasoner + CCMP, the Table 9.1 system) on the
**training fields** named in cell 1, then rank the **held-out fields'** test corpora without adaptation. Both target fields are excluded from training and from checkpoint
selection. The table reports the cross-field queries, R@3 / R@5 / nDCG@5, next to the six
baselines of `sir4_baselines_all.ipynb`, whose predictions on Drive are reused and rescored
with the same scorer.

| section | what | cost |
|---|---|---|
| 1-2 | paths, bundles, current scripts, Qwen3 | minutes |
| 3 | baseline arms: restore predictions from Drive, run any that are missing | minutes if cached |
| 4 | engine + fusion sources | 5 min |
| 5 | caches, scorer warm start, component tables | ~30 min |
| 6 | smoke, then **train on Physics + Biology** | hours |
| 7 | zero-shot prediction on CS and MatSci, scoring | ~20 min |
| 8 | Table 9.3 | seconds |

The SciGraphIR row printed in the thesis came from the 9 Aug joint run, which is the
operator fusion without the multi-view scorer or CCMP. This notebook replaces that row with
the updated model; the baseline rows are unchanged.
'''

PATHS = rb.PATHS
for _old, _new in (
        ('DATASET  = "researchbench"', 'DATASET  = "sir4_zeroshot"           # a label for OUT_ROOT; every script gets its own --dataset'),
        ('RBG      = f"{DATASET}_test_v16sc"      # the one ResearchBench graph: 20,322 docs, 1,367 queries',
         'TRAIN_DOMS = __TRAIN__     # trained on\nEVAL_DOMS  = __EVAL__      # held out, zero-shot only\nS4_DOMS    = TRAIN_DOMS + EVAL_DOMS\nRBG = None'),
        ('TRAIN = TEST = RBG                      # read by the cloned config-rewrite cell; every run overrides on the CLI',
         'TRAIN, TEST = f"sir4_{TRAIN_DOMS[0]}_train_v16sc", f"sir4_{TRAIN_DOMS[0]}_test_v16sc"   # config defaults only; every run overrides on the CLI'),
        ('BUNDLE   = f"{DRIVE}/{DATASET}_bundle.zip"', 'BUNDLES  = {d: f"{DRIVE}/sir4_{d}_bundle.zip" for d in S4_DOMS}'),
        ('OUT_ROOT = f"{DRIVE}/outputs/rb_zeroshot"          # everything this notebook writes',
         'OUT_ROOT = f"{DRIVE}/outputs/sir4_zeroshot"        # shared by every direction: runs are named by their training fields, scores by their eval field'),
        ('CACHE_RB = f"{DRIVE}/outputs/researchbench/cache"  # graph-keyed artefacts the earlier RB runs cached',
         'BASE_OUT = {d: f"{DRIVE}/outputs/baselines/sir4_{d}" for d in S4_DOMS}   # sir4_baselines_all.ipynb wrote here'),
        ('S4_DOMS   = ["cs", "biology", "physics", "matsci"]\n', ''),
        ('SEM_WARM_DOM = "__WARM__"    # field that warm-starts the multi-view scorer before joint training',
         'SEM_WARM_DOM = "__WARM__"    # TRAINING field that warm-starts the multi-view scorer (never an eval field)\nassert SEM_WARM_DOM in TRAIN_DOMS, "the scorer warm start must come from a training field"'),
        ('print("reads      ", os.path.basename(BUNDLE), "+ sir4_*_bundle.zip (section 8) + gfm-rag-adapted.zip + qwen3-embedding-0.6b/")',
         'print("reads      ", "sir4_{physics,biology,cs,matsci}_bundle.zip + gfm-rag-adapted.zip + qwen3-embedding-0.6b/")'),
        ('print("TOMATO arm ", f"{TOMATO_OUT}/{TOMATO_RUN}/model_best.pth")', 'print("train on   ", TRAIN_DOMS, "| held out:", EVAL_DOMS)'),
):
    assert _old in PATHS, f"PATHS anchor not found: {_old[:60]!r}"
    PATHS = PATHS.replace(_old, _new)
# The TOMATO checkpoint block is irrelevant here; drop it so nothing reads it by accident.
_a, _b = PATHS.index("# ---- the two SciGraphIR arms"), PATHS.index("SEM_WARM_DOM = ")
PATHS = PATHS[:_a] + "# ---- the run -----------------------------------------------------------------------\n" + PATHS[_b:]

UNPACK = '''# 2. Unpack the four SIR-4 bundles to an isolated local root, then install the CURRENT
# repo scripts over whatever the bundles carry (Drive code_overlay first, inline copies last).
KEEP, PARK = f"{CARGO_ROOT}/outputs/caches", "/content/_caches_keep"
if os.path.isdir(KEEP):
    shutil.rmtree(PARK, ignore_errors=True); shutil.move(KEEP, PARK)
if os.path.exists(CARGO_ROOT):
    shutil.rmtree(CARGO_ROOT)
os.makedirs(CARGO_ROOT, exist_ok=True)
for d in S4_DOMS:
    z = BUNDLES[d]
    assert os.path.exists(z) and zipfile.is_zipfile(z), f"{z} missing or not a zip"
    zipfile.ZipFile(z).extractall(CARGO_ROOT); print("unpacked", os.path.basename(z))
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

def g_of(d, s): return f"sir4_{d}_{s}_v16sc"
QUERIES = {d: f"{DATA_ROOT}/sir4_{d}_test/raw/test.json" for d in S4_DOMS}
SETS    = {d: f"{CARGO_ROOT}/{rel}" for d, rel in __SETS_MAP__.items()}
for d in S4_DOMS:
    cp.set_dataset(f"sir4_{d}")
    for label, p in (("corpus", f"{DATA_ROOT}/sir4_{d}_test/raw/documents.json"), ("queries", QUERIES[d]),
                     ("probes", cp.probes_path("test")), ("graph", f"{DATA_ROOT}/{g_of(d, 'test')}/processed/stage1/nodes.csv"),
                     ("sets", SETS[d])):
        assert os.path.exists(p), f"missing {d} {label}: {p}"
    if d in TRAIN_DOMS:
        for label, p in (("train corpus", f"{DATA_ROOT}/sir4_{d}_train/raw/documents.json"),
                         ("train probes", cp.probes_path("train")),
                         ("train graph", f"{DATA_ROOT}/{g_of(d, 'train')}/processed/stage1/nodes.csv")):
            assert os.path.exists(p), f"missing {d} {label}: {p}"
    print(f"  ok  sir4_{d:8} {'train+test' if d in TRAIN_DOMS else 'test only (held out)'}")

# LEAKAGE: no held-out test query id may appear in any training query set.
_train_ids = set()
for d in TRAIN_DOMS:
    _train_ids |= {x["id"] for x in json.load(open(f"{DATA_ROOT}/sir4_{d}_train/raw/train.json"))}
    _train_ids |= {x["id"] for x in json.load(open(QUERIES[d]))}      # test graphs drive selection
for d in EVAL_DOMS:
    held = {x["id"] for x in json.load(open(QUERIES[d]))}
    assert not (held & _train_ids), f"{len(held & _train_ids)} {d} test queries appear in training/selection data"
    print(f"  {d:8} {len(held)} held-out queries, none seen in training or selection")

STD_COLS = "mrr,ndcg@5,recall@3,recall@5,recall@10,recall@100,completeset@5"
!pip -q install rank_bm25 sentence-transformers "transformers>=4.52.4,<5"
import transformers
assert transformers.__version__.startswith("4."), f"transformers {transformers.__version__}: restart the runtime"
print("ready | transformers", transformers.__version__)
'''

RUN_BASELINES = '''# 3b. Baseline predictions for the two held-out fields. sir4_baselines_all.ipynb already
# wrote every arm to Drive (outputs/baselines/sir4_<field>/); a matching manifest is a
# [skip], anything missing is run here with the same runner and flags.
import shlex
TOPK = 300
_SCORER = hashlib.md5(open(f"{S4}/eval/baselines_sir4.py", "rb").read()).hexdigest()[:8]
def arm_sig(ds, model, pool, ins, extra):
    return hashlib.md5(json.dumps({"model": model, "pooling": pool, "instruct": ins, "extra": extra,
                                   "topk": TOPK, "scorer": _SCORER, "dataset": ds, "split": "test"},
                                  sort_keys=True).encode()).hexdigest()[:12]
PRED = {d: {} for d in EVAL_DOMS}
for d in EVAL_DOMS:
    ds = f"sir4_{d}"
    os.makedirs(BASE_OUT[d], exist_ok=True)
    for lab, tag, model, pool, ins, extra in ARMS:
        dest = f"{S4}/data/predictions_{tag}_{ds}_test.json"; man = dest + ".manifest.json"
        for a, b in ((f"{BASE_OUT[d]}/{os.path.basename(dest)}", dest), (f"{BASE_OUT[d]}/{os.path.basename(man)}", man)):
            if not os.path.exists(b) and os.path.exists(a):
                os.makedirs(os.path.dirname(b), exist_ok=True); shutil.copy(a, b)
        have = None
        if os.path.exists(man):
            try: have = json.load(open(man)).get("sig")
            except Exception: have = None
        if os.path.exists(dest) and (have == arm_sig(ds, model, pool, ins, extra) or have is not None):
            # A manifest from the all-domain notebook with a different scorer md5 means the
            # runner file changed, not the predictions; those files are the published rows.
            print(f"[skip] {d}/{lab}"); PRED[d][tag] = dest; continue
        cmd = [sys.executable, "-u", "eval/baselines_sir4.py", "--dataset", ds, "--split", "test",
               "--model", model, "--pooling", pool, "--tag", tag, "--topk", str(TOPK)]
        if ins:   cmd += ["--instruct", ins]
        if extra: cmd += shlex.split(extra)
        cmd += ["--out", dest]
        rc = sh(cmd, S4, extra={"CARGO_DATASET": ds}, check=False)
        if rc != 0:
            print(f"!! {d}/{lab} FAILED rc={rc}"); continue
        json.dump({"sig": arm_sig(ds, model, pool, ins, extra), "model": model, "pooling": pool,
                   "instruct": ins, "extra": extra, "topk": TOPK, "scorer_md5": _SCORER}, open(man, "w"), indent=1)
        shutil.copy(dest, f"{BASE_OUT[d]}/{os.path.basename(dest)}"); shutil.copy(man, f"{BASE_OUT[d]}/{os.path.basename(man)}")
        PRED[d][tag] = dest
print({d: sorted(v) for d, v in PRED.items()})
'''

SCORE = '''# 3c. Score every baseline on both held-out fields: one scorer, one flag set, sets.json for
# CompleteSet@5. No --bge: the table is same/cross and the similar/dissimilar axis is not used.
SCORES = f"{OUT_ROOT}/scores"
os.makedirs(SCORES, exist_ok=True)
def score(d, key, pred, label):
    js, pq = f"{SCORES}/{d}_{key}_scores.json", f"{SCORES}/{d}_{key}_perquery.json"
    sh([sys.executable, "-u", "eval/score_sir4.py", "--pred", pred, "--queries", QUERIES[d],
        "--sets", SETS[d], "--name", f"{label} [{d}]", "--cols", STD_COLS,
        "--json-out", js, "--per-query-out", pq], S4)
    return js
LABEL = {tag: lab for lab, tag, *_ in ARMS}
for d in EVAL_DOMS:
    for tag, pred in PRED[d].items():
        score(d, tag, pred, LABEL[tag])
'''

HELPERS = '''# 5a. Cache helpers (graph-keyed artefacts the single-field training runs left on Drive).
import numpy as np
S4_CACHE = {d: f"{DRIVE}/outputs/sir4_{d}/cache" for d in S4_DOMS}

def restore_index(g, cache):
    """The Qwen3 node index is the 30-45 min artefact per graph."""
    src = f"{cache}/index/{g}"
    if not os.path.isdir(src):
        print(f"  {g}: no cached index on Drive (built on first use)"); return
    for d in os.listdir(src):
        shutil.copytree(f"{src}/{d}", f"{DATA_ROOT}/{g}/processed/{d}", dirs_exist_ok=True)
    print(f"  {g}: index restored")

def save_index(g, cache):
    pr = f"{DATA_ROOT}/{g}/processed"
    for d in os.listdir(pr):
        if d != "stage1":
            shutil.copytree(f"{pr}/{d}", f"{cache}/index/{g}/{d}", dirs_exist_ok=True)

def _sem_ok(p):
    if not os.path.exists(p): return False
    zz = np.load(p, allow_pickle=True)
    return os.path.exists(str(zz["h_path"])) and "qwen" in str(zz["encoder"]).lower()
print("helpers ready")
'''

# The RB notebook's SIR-4 cache cell, verbatim: it loops S4_DOMS over train+test, which
# here covers the two training fields and the two held-out fields (their train graphs are
# unpacked but never named in any training command).
CACHES = rb.S4_UNPACK.replace("# 8a. The four SIR-4 fields: graphs, corpora, caches. Nothing from ResearchBench enters here.",
                              "# 5b. Graphs, corpora and caches for all four fields. The held-out fields' TRAIN graphs are\n"
                              "# unpacked by the bundle but never named in a training command below.")
for _old, _new in (
        ('S4_CACHE = {d: f"{DRIVE}/outputs/sir4_{d}/cache" for d in S4_DOMS}\n', ''),
        ('for d in S4_DOMS:\n    z = f"{DRIVE}/sir4_{d}_bundle.zip"\n    assert os.path.exists(z) and zipfile.is_zipfile(z), f"{z} missing or not a zip"\n'
         '    if not os.path.exists(f"{DATA_ROOT}/sir4_{d}_train_v16sc/processed/stage1/nodes.csv"):\n'
         '        zipfile.ZipFile(z).extractall(CARGO_ROOT); print("unpacked", os.path.basename(z))\n'
         'apply_overlay()          # the bundles carry older copies of the scripts; the repo\'s win\n\n'
         'def g_of(d, s): return f"sir4_{d}_{s}_v16sc"\n', ''),
        ('import numpy as np\n', '')):
    assert _old in CACHES, f"CACHES anchor not found: {_old[:60]!r}"
    CACHES = CACHES.replace(_old, _new)

SCORER = rb.S4_SCORER.replace("# 8b.", "# 5c.")
SEMCOMP = (rb.S4_SEMCOMP.replace("# 8c.", "# 5d.")
           .replace("for d in S4_DOMS:\n    for s in (\"train\", \"test\"):\n        if not _sem_ok",
                    "for d in TRAIN_DOMS:\n    for s in (\"train\", \"test\"):\n        if not _sem_ok")
           .replace("TRAIN_G = [g_of(d, \"train\") for d in S4_DOMS]", "TRAIN_G = [g_of(d, \"train\") for d in TRAIN_DOMS]")
           .replace("VALID_G = [g_of(d, \"test\") for d in S4_DOMS]", "VALID_G = [g_of(d, \"test\") for d in TRAIN_DOMS]")
           .replace("OPC_TR, OPC_TE = \",\".join(opc(d, \"train\") for d in S4_DOMS), \",\".join(opc(d, \"test\") for d in S4_DOMS)",
                    "OPC_TR, OPC_TE = \",\".join(opc(d, \"train\") for d in TRAIN_DOMS), \",\".join(opc(d, \"test\") for d in TRAIN_DOMS)")
           .replace("SEM_TR, SEM_TE = \",\".join(semc(d, \"train\") for d in S4_DOMS), \",\".join(semc(d, \"test\") for d in S4_DOMS)",
                    "SEM_TR, SEM_TE = \",\".join(semc(d, \"train\") for d in TRAIN_DOMS), \",\".join(semc(d, \"test\") for d in TRAIN_DOMS)")
           .replace('print("selecting on", VALID_G, "(dev numbers; ResearchBench is the only clean test)")',
                    'print("selecting on", VALID_G, "-- the held-out fields", EVAL_DOMS, "are never named here")\n'
                    '# whole token, not substring: "cs" is inside "physics"\n'
                    'assert not any(f"sir4_{d}_" in g for g in TRAIN_G + VALID_G for d in EVAL_DOMS), "a held-out field leaked into training/selection"'))
for _must in ("for d in TRAIN_DOMS:", "TRAIN_G = [g_of(d, \"train\") for d in TRAIN_DOMS]", "a held-out field leaked"):
    assert _must in SEMCOMP, f"SEMCOMP substitution failed: {_must!r}"

SMOKE = (rb.S4_SMOKE.replace("# 8d.", "# 6a.")
         .replace("scigraphir_sir4_all4_smoke", "scigraphir_sir4_pb_smoke")
         .replace("assert n_sem >= 8, f\"only {n_sem} semantic tables loaded, expected 8 (4 train + 4 test)\"",
                  "assert n_sem >= 4, f\"only {n_sem} semantic tables loaded, expected 4 (2 train + 2 test)\"")
         .replace("for d in S4_DOMS:\n    for s in (\"train\", \"test\"): save_index", "for d in TRAIN_DOMS:\n    for s in (\"train\", \"test\"): save_index")
         .replace("all four training graphs reached", "both training graphs reached"))
assert "expected 4 (2 train" in SMOKE and "for d in TRAIN_DOMS:" in SMOKE

TRAIN = (rb.S4_TRAIN.replace("# 8e. TRAIN SciGraphIR on all four SIR-4 fields. Hours (the two-field joint run took ~7 h for\n# 10 epochs; expect roughly double).",
                             "# 6b. TRAIN SciGraphIR on Physics + Biology. Hours (the earlier two-field joint run took ~7 h\n# for 10 epochs at batch 2).")
         .replace('S4_NAME  = f"scigraphir_sir4_all4_qwenmlp_{CCMP_TAG}_e{EPOCHS}_b{BATCH}"',
                  'S4_NAME  = f"scigraphir_{\'+\'.join(TRAIN_DOMS)}_qwenmlp_{CCMP_TAG}_e{EPOCHS}_b{BATCH}"')
         .replace('"arm": "SciGraphIR (SIR-4, all four fields)"', '"arm": "SciGraphIR (SIR-4 Physics + Biology)", "held_out": EVAL_DOMS'))
assert "scigraphir_{'+'.join(TRAIN_DOMS)}_qwenmlp_{CCMP_TAG}" in TRAIN and '"held_out": EVAL_DOMS' in TRAIN

ZEROSHOT = '''# 7. ZERO-SHOT. Load the Physics+Biology checkpoint, rank each held-out field's test corpus.
# No training, no tuning, no adaptation: the only per-field inputs are that field's own
# component tables, which are properties of its corpus and not of the model.
import torch

def inspect_ckpt(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]   # our own file; torch 2.6 default refuses its numpy scalars
    resp = [k for k in sd if "resp_" in k]; sem = [k for k in sd if k.startswith("sem_")]
    return {"tensors": len(sd), "resp_keys": len(resp), "sem_keys": len(sem),
            "ccmp_hid": next((int(sd[k].shape[0]) for k in resp if k.endswith("resp_proj.0.weight")), None),
            "jmax": next((int(sd[k].shape[1]) - 2 for k in sem if k.endswith("sem_net.0.weight")), None)}

info = inspect_ckpt(S4_CKPT); print("[ckpt]", os.path.relpath(S4_CKPT, DRIVE), info)
assert info["sem_keys"] and info["resp_keys"], "checkpoint lacks the multi-view scorer or the CCMP head"
assert int(json.load(open(SEM_CKPT_S4))["jmax"]) == info["jmax"], "scorer width mismatch with the warm-start file"

ZS_PRED = {}
for d in EVAL_DOMS:
    g = g_of(d, "test")
    if not _sem_ok(semc(d, "test")):
        sh(f"python3 -u experiments/probe_greasoner/precompute_semantic_components.py "
           f"--dataset sir4_{d} --model {OP_MODEL} --graph {g} --split test", KGDIR, extra={"CARGO_DATASET": f"sir4_{d}"})
    run_local, run_drive = f"{RUNS}/{S4_NAME}/zeroshot_{d}", f"{S4_DRIVE}/zeroshot_{d}"
    pred_drive = f"{run_drive}/predictions_{g}.json"
    if os.path.exists(pred_drive):
        print(f"[cached] {d}: {os.path.relpath(pred_drive, DRIVE)}"); ZS_PRED[d] = pred_drive; continue
    os.makedirs(run_local, exist_ok=True)
    extra = dict(S4_ENV, OPERATOR_COMPONENTS=opc(d, "test"), OPERATOR_COMPONENTS_TEST=opc(d, "test"),
                 SEMANTIC_COMPONENTS=semc(d, "test"), SEMANTIC_COMPONENTS_TEST=semc(d, "test"),
                 CCMP_HID=str(info["ccmp_hid"]))
    for k in ("CCMP_LR", "CCMP_M", "CCMP_NEG", "CCMP_W", "CCMP_POOL", "CCMP_M_POS", "CCMP_M_NEG"):
        extra.pop(k, None)               # training-only knobs; the head itself must still be built (CCMP=1)
    rc = sh("python -u -m gfmrag.workflow.sft_training "
            "--config-path config/gfm_reasoner --config-name sft_training_fusion "
            "text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{g}] datasets.valid_names=[{g}] "
            "model.semantic=mlp model.cqig=false "
            "trainer.args.do_train=false trainer.args.do_eval=false "
            f"+trainer.args.eval_batch_size={BATCH} "
            "+trainer.args.do_predict=true +trainer.args.predict_top_k=300 "
            f"+trainer.args.resume_from_checkpoint={S4_CKPT} "
            f"hydra.run.dir={run_local}",
            "/content/gfm-rag", extra=extra, log=f"{run_local}/console.log", check=False)
    log = open(f"{run_local}/console.log", errors="ignore").read()
    assert rc == 0, f"predict on {d} failed (exit {rc}); read {run_local}/console.log"
    assert "semantic='mlp' warm start" in log and "[ccmp] responsibility head" in log, "model built without the scorer or the head"
    assert os.path.exists(f"{run_local}/predictions_{g}.json"), "no predictions written"
    save_index(g, S4_CACHE[d]); sync_dir(run_local, run_drive)
    ZS_PRED[d] = pred_drive

for d in EVAL_DOMS:
    score(d, "scigraphir_pb", ZS_PRED[d], "SciGraphIR (Physics + Biology) zero-shot")
'''

TABLE = '''# 8. Table 9.3, updated model. Cross-field queries of the two held-out fields; the same-field
# and all-query slices are printed underneath for the appendix.
ORDER = [("bm25", "BM25", "Sparse lexical", "None"),
         ("bge", "BGE-large", "Dense embedding", "External English retrieval pairs"),
         ("qwen3", "Qwen3-Embedding", "Dense embedding", "External multilingual relevance pairs"),
         ("specter2", "SPECTER2-base", "Dense embedding", "Citation-graph contrastive (scientific)"),
         ("scincl", "SciNCL", "Dense embedding", "Citation-neighbourhood contrastive (scientific)"),
         ("reasonir", "ReasonIR-8B", "Reasoning-trained dense", "Synthetic reasoning-intensive pairs"),
         ("scigraphir_pb", "SciGraphIR", "Ours", "SIR-4 (Physics + Biology)")]
COLS = [("recall@3", "R@3"), ("recall@5", "R@5"), ("ndcg@5", "nDCG@5")]
TITLE = {"cs": "Computer Science", "matsci": "Materials Science", "physics": "Physics", "biology": "Biology"}
lines = []
def out(s=""): print(s); lines.append(s)
for slc in ("cross", "same", "all"):
    out(f"### {slc} queries" + ("   <- Table 9.3" if slc == "cross" else ""))
    out("| Method | Training data | " + " | ".join(l for _, l in COLS) + " |"); out("|---|---|" + "--:|" * len(COLS))
    for d in EVAL_DOMS:
        out(f"| **{TITLE.get(d, d)}** | | | | |")
        for key, lab, fam, data in ORDER:
            p = f"{SCORES}/{d}_{key}_scores.json"
            if not os.path.exists(p):
                out(f"| {lab} | {data} | missing | missing | missing |"); continue
            s = json.load(open(p)).get(slc)
            if not s: continue
            out(f"| {lab} | {data} | " + " | ".join(f"{s[k]:.4f}" for k, _ in COLS) + f" |   n={s['n']}" if key == "bm25" else
                f"| {lab} | {data} | " + " | ".join(f"{s[k]:.4f}" for k, _ in COLS) + " |")
    out()
_md = f"{OUT_ROOT}/table93_" + "+".join(TRAIN_DOMS) + "_to_" + "+".join(EVAL_DOMS) + ".md"
open(_md, "w").write("\\n".join(lines))
print("wrote", _md, "| trained on", TRAIN_DOMS, "| held out", EVAL_DOMS)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--train", default="physics,biology", help="comma-separated training fields")
    ap.add_argument("--eval", dest="evald", default="cs,matsci", help="comma-separated held-out fields")
    ap.add_argument("--warm-dom", default=None,
                    help="TRAINING field that warm-starts the multi-view scorer; default = the training field with the larger corpus")
    ap.add_argument("--ccmp", default="standard", choices=["standard", "residual"],
                    help="standard = 30 Aug CCMP targets; residual = 31 Aug residual CCMP")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    train = [d.strip() for d in a.train.split(",") if d.strip()]
    evald = [d.strip() for d in a.evald.split(",") if d.strip()]
    assert train and evald and not set(train) & set(evald), f"train {train} and eval {evald} must be disjoint and non-empty"
    assert all(d in _DOMAINS for d in train + evald), f"unknown field; known {sorted(_DOMAINS)}"
    SIZE = {"cs": 20203, "biology": 15589, "physics": 10352, "matsci": 4677}      # train documents
    warm = a.warm_dom or max(train, key=SIZE.get)
    assert warm in train, f"--warm-dom {warm} is not a training field"
    global PATHS
    PATHS = PATHS.replace("__TRAIN__", json.dumps(train)).replace("__EVAL__", json.dumps(evald))

    src = json.load(open(rb.SRC_NB))["cells"]
    base = json.load(open(rb.BASE_NB))["cells"]
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    global SEMCOMP
    SEMCOMP = SEMCOMP.replace("__CCMP_RESID__", "1" if a.ccmp == "residual" else "0")
    PATHS = PATHS.replace("__TOMATO_RUN__", "tomato_fusion_qwenmlp_ccmp_epoch10_b2")

    fusion_files = {f"/content/gfm-rag/{rel}": open(f"{FORK}/{rel}").read() for rel in rb.FUSION_REL}
    assert not any("'''" in v for v in fusion_files.values()), "fusion source contains '''"
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
    assert not any("'''" in v for v in overlay.values()), "an overlay script contains '''"

    # The cloned config-rewrite cell asserts no "sir4" name survives; that guard is for TOMATO
    # builds and is inverted here, where the defaults ARE sir4 graphs. Drop that one line.
    cfg_cell = dict(src[9]); cfg_src = "".join(cfg_cell["source"])
    guard = '    assert "sir4" not in t, f"a sir4 dataset reference survived in {cfg}"\n'
    assert guard in cfg_src, "config-rewrite cell changed; check the sir4 guard by hand"
    cfg_cell["source"] = cfg_src.replace(guard, "").splitlines(True)

    cells = [md(HEADER), md("## 1. GPU + Drive + paths"),
             code(PATHS.replace("__WARM__", warm).replace("__EPOCHS__", str(a.epochs)).replace("__BATCH__", str(a.batch))),
             md("## 2. Unpack the four bundles + install the current scripts"),
             code(UNPACK.replace("__OVERLAY__", json.dumps(overlay)).replace("__BUILT__", built)
                  .replace("__SETS_MAP__", json.dumps(SETS_MAP))),
             md("## 2b. Qwen3-Embedding-0.6B (cached on Drive)"), src[rb.QWEN_CELLS[1]],
             md("## 3. Baselines on the held-out fields\n### 3a. The arms\n*(harvested verbatim from "
                "`sir4_baselines_all.ipynb`, so they cannot differ from the published rows)*")]
    arms_src = "".join(base[rb.ARMS_CELL]["source"])
    assert "ARMS = [" in arms_src and "ReasonIR-8B" in arms_src
    cells += [code(arms_src + rb.ARMS_TAIL), md("### 3b. Predictions (restored from Drive, run only if missing)"),
              code(RUN_BASELINES), md("### 3c. Score"), code(SCORE),
              md("## 4. Engine + fusion sources\n*(cloned from `tomato_ccmp_ablation.ipynb`; the fusion-source blob is "
                 "regenerated from the repo)*")]
    for i in rb.ENGINE_CELLS:
        cells.append(files_cell if i == 7 else (cfg_cell if i == 9 else src[i]))
    cells += [md("## 5. Caches, scorer warm start, component tables"),
              code(HELPERS), code(CACHES), code(SCORER), code(SEMCOMP), code(rb.BATCH_PATCH),
              md("## 6. Smoke, then train on Physics + Biology\n**Hours.** Runs locally, syncs to Drive every ten minutes; "
                 "a finished run on Drive is a skip."),
              code(SMOKE), code(TRAIN),
              md("## 7. Zero-shot on Computer Science and Materials Science"), code(ZEROSHOT),
              md("## 8. Table 9.3, updated model"), code(TABLE)]
    a.out = a.out or f"{ROOT}/colab_sir4_zeroshot_{'+'.join(train)}_to_{'+'.join(evald)}_{a.ccmp}.ipynb"
    # The row key and label follow the TRAINING fields. "scigraphir_pb" / "Physics + Biology" are
    # the forward direction's literals; the reverse build (CS + MatSci) must not inherit them.
    _TTL = {"cs": "CS", "matsci": "MatSci", "physics": "Physics", "biology": "Biology"}
    _key = "scigraphir_" + "".join(d[0] for d in train)
    _ttl = " + ".join(_TTL[d] for d in train)
    for c in cells:
        c["source"] = ["".join(c["source"]).replace("scigraphir_pb", _key).replace("Physics + Biology", _ttl)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(cells)} cells | ccmp {a.ccmp} | train {train} -> eval {evald} | {warm}-warm-started scorer | epochs {a.epochs} batch {a.batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
