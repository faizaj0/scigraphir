"""
build_qualitative_notebook.py -- the qualitative section (9.2.5) and the path interpretations
(9.2.6): pick cross-field examples where dense retrieval buries the gold and SciGraphIR finds
it, then explain HOW it found it, hop by hop, with the CCMP gate along the path, next to what
the OpenIE entity graph does on the same query.

WHAT IT PRODUCES, per field:
    picks.json            candidate examples ranked by how badly the dense baselines miss
    paths_frame.json      path interpretations under the SciGraph frame graph + CCMP model
    paths_openie.json     the same under the OpenIE entity-graph model (no CCMP)
    paths_control.json    (optional) frame graph without CCMP, when the routing control exists
    qualitative.md / qualitative_paths.tex   the section's material

HOW A PATH IS FOUND. NBFNet / GFM-RAG recipe: the graph channel's score of the gold paper is
differentiated w.r.t. every layer's edge weights; a beam search over those gradients returns the
top-k highest-weighted paths from the query's seed frames to the gold (Table 4 of GFM-RAG).
Along each path we also read the CCMP responsibility head at the layer the hop was used and
express it as the gate the model applied (frontier mean = 1.0), which is the only part of this
that is ours: the reader sees which hop CCMP amplified and which it suppressed.

MODELS. Frame graph: the CCMP checkpoint trained on this field and its partner (the Table 9.3
zero-shot runs are in-field for their two TRAINING fields). OpenIE graph: the in-field OpenIE
run of colab_sir4_openie_ablation.ipynb. Both current engine. The frame-graph no-CCMP control
is picked up automatically if colab_routing_<field>.ipynb has produced it.

Usage
-----
    python3 prep/build_qualitative_notebook.py --field biology
    python3 prep/build_qualitative_notebook.py --field matsci --n 4
    python3 prep/build_qualitative_notebook.py --field biology --qids 10.1038_x,10.1016_y
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

FUSION_REL = rb.FUSION_REL + ["gfmrag/models/ultra/layers.py", "gfmrag/workflow/interpret_paths.py"]
GUARD = '    assert "sir4" not in t, f"a sir4 dataset reference survived in {cfg}"\n'
OVERLAY_REL = rb.OVERLAY_REL + ["experiments/eval/pick_qualitative.py", "experiments/eval/render_qualitative.py"]

HEADER = '''# Qualitative analysis and path interpretations (SIR-4 `__FIELD__`)

Finds cross-field queries whose gold inspiration the dense retrievers bury and SciGraphIR ranks at
the top, then explains the retrieval: the multi-view scorer's best-matching hypothetical answers,
and the top reasoning paths from the query's seed frames to the gold paper under the trained graph
reasoner (gradient beam search over per-layer edge weights, the NBFNet / GFM-RAG recipe), with the
CCMP gate read at every hop. The same paths are extracted under the OpenIE entity-graph model so
the two constructions can be shown side by side.

| section | what | cost |
|---|---|---|
| 1-2 | paths, bundle, current scripts, Qwen3 | minutes |
| 3 | engine + fusion sources (adds `interpret_paths.py`) | 5 min |
| 4 | component tables for the frame and OpenIE test graphs; scorer files | minutes (cached) |
| 5 | predictions of the frame-graph CCMP model on this field (one predict pass) | ~5 min |
| 6 | scan the model's channel ranks on every cross query, then pick | minutes |
| 7 | path interpretations, both graphs | minutes |
| 8 | markdown + LaTeX | seconds |

Nothing here trains. Every checkpoint it reads already exists on Drive.
'''

_HELP = rb.PATHS[rb.PATHS.index("try:\n    from google.colab import userdata"):]
_HELP = _HELP[:_HELP.index("for d in (OUT_ROOT, RUNS):")]

PATHS = '''!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import os, sys, re, time, glob, json, math, shutil, zipfile, subprocess, hashlib, threading
from collections import defaultdict
from google.colab import drive
drive.mount('/content/drive')

DRIVE    = "/content/drive/MyDrive/cargo-gfmrag"
FIELD    = "__FIELD__"
DATASET  = f"sir4_{FIELD}"
FRAME_TEST  = f"{DATASET}_test_v16sc"                 # SciGraph frame graph
OPENIE_TEST = f"{DATASET}_test"                       # OpenIE entity graph (same corpus, same queries)
TRAIN, TEST = FRAME_TEST, FRAME_TEST                  # config DEFAULT names only (every command overrides them); the cloned
                                                      # engine cell rewrites the yaml with these
BUNDLE   = f"{DRIVE}/{DATASET}_bundle.zip"
OUT_ROOT = f"{DRIVE}/outputs/qualitative/{DATASET}"
CACHE    = f"{DRIVE}/outputs/{DATASET}/cache"
# --- the models -----------------------------------------------------------------------
FIELD_SEM   = f"{DRIVE}/outputs/{DATASET}/semantic"                     # the field's own 5d scorer
FRAME_RUN   = f"{DRIVE}/__FRAME_RUN__"                                  # frame graph + CCMP, trained on this field (+ partner)
FRAME_SEM   = f"{DRIVE}/__FRAME_SEM__"                                  # that run's scorer warm-start files
OPENIE_RUN  = f"{DRIVE}/outputs/sir4_openie/{DATASET}_openie_qwenmlp_graph_e10_b2"   # OpenIE graph, no CCMP, in-field
CONTROL_RUN = f"{DRIVE}/__CONTROL_RUN__"                                # frame graph, no CCMP (optional; routing notebook)
BASELINES   = f"{DRIVE}/outputs/baselines/{DATASET}"
N_EXAMPLES  = __N__
QIDS_OVERRIDE = __QIDS__                              # [] = take the top N candidates of section 6
SCIGRAPHIR_ROOT = "/content/scigraphir"
os.environ["SCIGRAPHIR_ROOT"] = SCIGRAPHIR_ROOT
os.environ["SCIGRAPHIR_DATASET"] = DATASET
DATA_ROOT = f"{SCIGRAPHIR_ROOT}/retriever/data"
S4        = f"{SCIGRAPHIR_ROOT}/experiments"
KGDIR     = f"{SCIGRAPHIR_ROOT}/retriever"
RUNS      = "/content/runs"
OP_MODEL  = "/content/qwen3"
OP_SLUG   = "_content-qwen3"

''' + _HELP + '''
for d in (OUT_ROOT, RUNS, CACHE):
    os.makedirs(d, exist_ok=True)
for lab, p in (("frame ckpt", f"{FRAME_RUN}/model_best.pth"), ("frame scorer", f"{FRAME_SEM}/params_semantic_mlp_fixedloss___WARMDS__.json"),
               ("openie ckpt", f"{OPENIE_RUN}/model_best.pth"), ("field scorer", f"{FIELD_SEM}/params_semantic_mlp_fixedloss_{DATASET}.json")):
    print(f"  {'ok ' if os.path.exists(p) else 'MISSING'}  {lab:13} {os.path.relpath(p, DRIVE)}")
print("  " + ("ok " if os.path.exists(f"{CONTROL_RUN}/model_best.pth") else "-- ") + " control ckpt (optional)", os.path.relpath(CONTROL_RUN, DRIVE))
print("writes ", OUT_ROOT)
'''

UNPACK = '''# 2. Unpack the bundle (it carries BOTH graphs for this field) and install the current scripts.
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
DOCS    = f"{DATA_ROOT}/{DATASET}_test/raw/documents.json"
PROBES  = cp.probes_path("test")
for lab, p in (("queries", QUERIES), ("documents", DOCS), ("probes", PROBES),
               ("frame graph", f"{DATA_ROOT}/{FRAME_TEST}/processed/stage1/nodes.csv"),
               ("openie graph", f"{DATA_ROOT}/{OPENIE_TEST}/processed/stage1/nodes.csv")):
    assert os.path.exists(p), f"missing {lab}: {p}"
    print(f"  ok  {lab:13} {p.replace(SCIGRAPHIR_ROOT, '<root>')}")
for g in (FRAME_TEST, OPENIE_TEST):
    os.makedirs(f"{DATA_ROOT}/{g}/raw", exist_ok=True)
    dst = f"{DATA_ROOT}/{g}/raw/documents.json"
    if os.path.abspath(DOCS) != os.path.abspath(dst): shutil.copy(DOCS, dst)
_q = json.load(open(QUERIES))
print(f"\\ntest queries {len(_q)}: cross {sum(q.get('stratum') == 'cross' for q in _q)}, same {sum(q.get('stratum') == 'same' for q in _q)}")
!pip -q install sentence-transformers "transformers>=4.52.4,<5"
import transformers
assert transformers.__version__.startswith("4."), f"transformers {transformers.__version__}: restart the runtime"
print("ready | transformers", transformers.__version__)
'''

COMPONENTS = '''# 4. Component tables for both TEST graphs (operator + scorer views) and the scorer files.
# Only the test-split embedding files come down from Drive, nothing already present is copied
# twice, and only NEW files go back up.
import numpy as np, fnmatch
def copy_new(src, dst, pattern="*"):
    """Copy files under src matching pattern that are absent (or a different size) at dst."""
    if not os.path.isdir(src): return 0
    n = 0
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for f in files:
            if not fnmatch.fnmatch(f, pattern): continue
            s_, d_ = f"{root}/{f}", f"{dst}/{rel}/{f}"
            if os.path.exists(d_) and os.path.getsize(d_) == os.path.getsize(s_): continue
            os.makedirs(os.path.dirname(d_), exist_ok=True); shutil.copy(s_, d_); n += 1
    return n
t0 = time.time()
EMB_LOCAL = f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb"
n = copy_new(f"{CACHE}/op_emb", EMB_LOCAL, "test_*")
print(f"embedding cache: {n} test-split files restored  ({time.time() - t0:.0f}s)")

def restore_index(g):
    src = f"{CACHE}/index/{g}"
    if not os.path.isdir(src):
        print(f"  {g}: no cached index on Drive (built on first use, ~5-15 min for a test graph)"); return
    n = sum(copy_new(f"{src}/{d}", f"{DATA_ROOT}/{g}/processed/{d}") for d in os.listdir(src))
    print(f"  {g}: index restored ({n} files, {time.time() - t0:.0f}s)")
def save_index(g):
    pr = f"{DATA_ROOT}/{g}/processed"
    for d in os.listdir(pr):
        if d != "stage1": copy_new(f"{pr}/{d}", f"{CACHE}/index/{g}/{d}")
def _sem_ok(p):
    if not os.path.exists(p): return False
    zz = np.load(p, allow_pickle=True)
    return os.path.exists(str(zz["h_path"])) and "qwen" in str(zz["encoder"]).lower()
def opc(g):  return f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
def semc(g): return f"{DATA_ROOT}/{g}/semantic_components{OP_SLUG}.npz"

SEM_LOCAL = f"{S4}/results/semantic_{DATASET}"
os.makedirs(SEM_LOCAL, exist_ok=True)
if os.path.isdir(FIELD_SEM): copy_new(FIELD_SEM, SEM_LOCAL); print("field scorer restored")
FIELD_CKPT_SEM = f"{SEM_LOCAL}/params_semantic_mlp_fixedloss_{DATASET}.json"
FIELD_POP_SEM  = f"{SEM_LOCAL}/popnet_semantic_mlp_fixedloss_{DATASET}.pt"
FRAME_CKPT_SEM = f"{FRAME_SEM}/params_semantic_mlp_fixedloss___WARMDS__.json"
FRAME_POP_SEM  = f"{FRAME_SEM}/popnet_semantic_mlp_fixedloss___WARMDS__.pt"
for p in (FIELD_CKPT_SEM, FIELD_POP_SEM, FRAME_CKPT_SEM, FRAME_POP_SEM):
    assert os.path.exists(p), f"missing scorer file {p}"

for g in (FRAME_TEST, OPENIE_TEST):
    restore_index(g)
    if not os.path.exists(opc(g)):
        c = f"{CACHE}/{g}_operator_components{OP_SLUG}.npz"
        if os.path.exists(c): shutil.copy(c, opc(g))
        else:
            sh(f"python3 -u precompute/precompute_operator_components.py "
               f"--dataset {DATASET} --graph {g} --split test --model {OP_MODEL}", KGDIR)
            shutil.copy(opc(g), c)
    if not _sem_ok(semc(g)):      # the H memmap never survives a runtime reset; seconds to rebuild for a test split
        sh(f"python3 -u precompute/precompute_semantic_components.py "
           f"--dataset {DATASET} --model {OP_MODEL} --graph {g} --split test", KGDIR)
    zz = np.load(semc(g), allow_pickle=True)
    print(f"  {g:24} H {tuple(int(x) for x in zz['h_shape'])}  Jmax={int(zz['Jmax'])}")
n = copy_new(EMB_LOCAL, f"{CACHE}/op_emb", "test_*")
print(f"done in {time.time() - t0:.0f}s; {n} new embedding files written back to Drive")
'''

PREDICT = '''# 5. Full rankings of the frame-graph CCMP model on this field's test set (its training notebook
# only predicted on its held-out fields), plus the control if present. OpenIE rankings come from
# that run's own predict pass. Baselines and scorer-only rankings are read from Drive.
import torch
def inspect_ckpt(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    resp = [k for k in sd if "resp_" in k]; sem = [k for k in sd if k.startswith("sem_")]
    hid = next((int(sd[k].shape[0]) for k in resp if k.endswith("resp_proj.0.weight")), None)
    jmax = next((int(sd[k].shape[1]) - 2 for k in sem if k.endswith("sem_net.0.weight")), None)
    return {"tensors": len(sd), "resp_keys": len(resp), "sem_keys": len(sem), "ccmp_hid": hid, "jmax": jmax}

def model_env(ckpt, graph, sem_ckpt, sem_pop):
    info = inspect_ckpt(ckpt)
    st = json.load(open(sem_ckpt))
    assert int(st["jmax"]) == info["jmax"], f"scorer width mismatch: ckpt jmax={info['jmax']} vs {sem_ckpt} jmax={st['jmax']}"
    env_ = dict(WANDB_MODE="disabled", HYDRA_FULL_ERROR="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                OPERATOR_COMPONENTS=opc(graph), OPERATOR_COMPONENTS_TEST=opc(graph),
                SEMANTIC_COMPONENTS=semc(graph), SEMANTIC_COMPONENTS_TEST=semc(graph),
                SEMANTIC_CKPT=sem_ckpt, SEMANTIC_POPNET=sem_pop, SEM_POP_LAMBDA="1.0",
                FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0",
                PER_GOLD="1", HARDNEG_GRAPH="50", STRAT_TEST=QUERIES)
    for k in list(os.environ):
        if k.startswith(("CCMP", "ROUTE")): os.environ.pop(k)
    if info["resp_keys"]:
        env_.update(CCMP="1", CCMP_HID=str(info["ccmp_hid"]), CCMP_GATE="1", CCMP_GATE_NORM="1", CCMP_ETA="0.5")
        aj = f"{os.path.dirname(ckpt)}/arm.json"
        if os.path.exists(aj):
            a_ = json.load(open(aj))
            if a_.get("ccmp_gate") is not None: env_["CCMP_GATE"] = "1" if a_["ccmp_gate"] else "0"
            if a_.get("ccmp_eta") is not None: env_["CCMP_ETA"] = str(a_["ccmp_eta"])
    return env_, info

def hydra_common(graph):
    return ("--config-path config/gfm_reasoner --config-name sft_training_fusion text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{graph}] datasets.valid_names=[{graph}] model.semantic=mlp model.cqig=false ")

def predict_with(ckpt, name, graph, sem_ckpt, sem_pop):
    run_local, run_drive = f"{RUNS}/pred_{name}", f"{OUT_ROOT}/pred_{name}"
    pred_drive = f"{run_drive}/predictions_{graph}.json"
    if os.path.exists(pred_drive):
        print(f"[cached] {os.path.relpath(pred_drive, DRIVE)}"); return pred_drive
    env_, info = model_env(ckpt, graph, sem_ckpt, sem_pop)
    print(f"[ckpt] {os.path.relpath(ckpt, DRIVE)} {info}")
    os.makedirs(run_local, exist_ok=True)
    rc = sh("python -u -m gfmrag.workflow.sft_training " + hydra_common(graph) +
            "trainer.args.do_train=false trainer.args.do_eval=false +trainer.args.eval_batch_size=2 "
            "+trainer.args.do_predict=true +trainer.args.predict_top_k=300 "
            f"+trainer.args.resume_from_checkpoint={ckpt} hydra.run.dir={run_local}",
            "/content/gfm-rag", extra=env_, log=f"{run_local}/console.log", check=False)
    assert rc == 0, f"predict failed (exit {rc}); read {run_local}/console.log"
    log = open(f"{run_local}/console.log", errors="ignore").read()
    if info["resp_keys"]: assert "[ccmp] responsibility head" in log, "CCMP head not constructed"
    assert os.path.exists(f"{run_local}/predictions_{graph}.json")
    save_index(graph); sync_dir(run_local, run_drive)
    return pred_drive

PRED = {}
FRAME_CKPT = f"{FRAME_RUN}/model_best.pth"
PRED["scigraphir"] = predict_with(FRAME_CKPT, "frame_ccmp", FRAME_TEST, FRAME_CKPT_SEM, FRAME_POP_SEM)
CONTROL_CKPT = f"{CONTROL_RUN}/model_best.pth"
if os.path.exists(CONTROL_CKPT):
    PRED["control"] = predict_with(CONTROL_CKPT, "frame_control", FRAME_TEST, FIELD_CKPT_SEM, FIELD_POP_SEM)
else:
    print("no frame-graph control checkpoint (run colab_routing notebook to add that row); skipped")
OPENIE_CKPT = f"{OPENIE_RUN}/model_best.pth"
_op = f"{OPENIE_RUN}/predictions_{OPENIE_TEST}.json"
PRED["openie"] = _op if os.path.exists(_op) else predict_with(OPENIE_CKPT, "openie", OPENIE_TEST, FIELD_CKPT_SEM, FIELD_POP_SEM)
for tag in ("bm25", "bge", "qwen3", "specter2", "scincl", "reasonir"):
    p = f"{BASELINES}/predictions_{tag}_{DATASET}_test.json"
    if os.path.exists(p): PRED[tag] = p
    else: print(f"  baseline {tag}: no predictions at {os.path.relpath(p, DRIVE)}")
for tag, fn in (("scorer", f"predictions_semantic_mlp_fixedloss_{DATASET}_test.json"), ("dense", f"predictions_semantic_dense_{DATASET}_test.json")):
    p = f"{FIELD_SEM}/{fn}"
    if os.path.exists(p): PRED[tag] = p
    else: print(f"  {tag}: no predictions at {os.path.relpath(p, DRIVE)}")
print("\\nsystems with rankings:", sorted(PRED))
'''

PICK = """# 6. Pick the examples in two steps.
# 6a. SCAN: the frame-graph CCMP model's own per-channel ranks of every cross-field gold, with the
#     CCMP gate ON and, from the SAME weights, with the gate OFF at inference (no path search).
# 6b. PICK: golds where every stage improves the rank, raw cosine > scorer channel > fused with the
#     gate off (+ graph) > fused with the gate on (+ CCMP), strictly, final rank <= 10. Ordered by the
#     CCMP step, then the graph step. Falls back to the graph-gap ordering when no gold is monotone.
CROSS_QIDS = [q["id"] for q in json.load(open(QUERIES)) if q.get("stratum") == "cross"]
json.dump(CROSS_QIDS, open(f"{OUT_ROOT}/qids_cross.json", "w"))
def scan(name, gate):
    out = f"{OUT_ROOT}/channels_{name}.json"
    if os.path.exists(out): print(f"[cached] {os.path.relpath(out, DRIVE)}"); return out
    _env, _info = model_env(FRAME_CKPT, FRAME_TEST, FRAME_CKPT_SEM, FRAME_POP_SEM)
    _env["CCMP_GATE"] = "1" if gate else "0"
    _rl = f"{RUNS}/scan_{name}"; os.makedirs(_rl, exist_ok=True)
    rc = sh("python -u -m gfmrag.workflow.interpret_paths " + hydra_common(FRAME_TEST) +
            f"+interp.ckpt={FRAME_CKPT} +interp.qids_file={OUT_ROOT}/qids_cross.json +interp.out={out} "
            f"+interp.probes={PROBES} +interp.paths=0 +interp.max_golds=4 +interp.top_views=3 hydra.run.dir={_rl}",
            "/content/gfm-rag", extra=_env, log=f"{_rl}/console.log", check=False)
    assert rc == 0 and os.path.exists(out), f"scan {name} failed (exit {rc}); read {_rl}/console.log"
    save_index(FRAME_TEST)
    return out
SCAN_ON, SCAN_OFF = scan("frame_gate_on", True), scan("frame_gate_off", False)
print("scanned", len(json.load(open(SCAN_ON))), "cross queries, gate on and off")

BIG = 10**6
def ranked_docs(rec):
    p = rec.get("predictions", rec); d = p.get("document", p) if isinstance(p, dict) else p
    return [x[0] if isinstance(x, (list, tuple)) else x for x in d]
def best_gold_rank(ranked, golds):
    for i, d in enumerate(ranked, 1):
        if d in golds: return i, d
    return BIG, None
queries = {q["id"]: q for q in json.load(open(QUERIES))}; docs_ = json.load(open(DOCS))
preds = {k: {r["id"]: ranked_docs(r) for r in json.load(open(v))} for k, v in PRED.items()}
on  = {r["id"]: {t["doc"]: t["rank"] for t in r["targets"]} for r in json.load(open(SCAN_ON))}
off = {r["id"]: {t["doc"]: t["rank"] for t in r["targets"]} for r in json.load(open(SCAN_OFF))}
def rank_of(ranked, d):
    return ranked.index(d) + 1 if d in ranked else BIG
def row(qid, gold, rk, ro):
    q = queries[qid]; golds = set(q["supporting_documents"]); r = {}
    for name, tab in preds.items():                      # rank of THIS gold under every system
        if qid in tab: r[name] = rank_of(tab[qid], gold)
    r.update(dense_ch=rk["dense"], scorer_ch=rk["scorer"], graph_ch=rk["graph"], fused_nogate=ro["fused"], fused_gate=rk["fused"])
    return {"id": qid, "stratum": q.get("stratum"), "ranks": r, "gap": rk["dense"] - rk["fused"],
            "graph_gap": rk["scorer"] - ro["fused"], "ccmp_gap": ro["fused"] - rk["fused"], "strong": True,
            "prefer_worse": int(r.get("openie", BIG) > rk["fused"]), "gold": gold,
            "gold_title": docs_.get(gold, "").strip().split(". ")[0][:110], "n_golds": len(golds), "question": q["question"]}
mono, anyg = [], []
for qid, golds_on in on.items():
    for gold, rk in golds_on.items():
        ro = off.get(qid, {}).get(gold)
        if not ro or rk["fused"] > 10: continue
        steps = [rk["dense"], rk["scorer"], ro["fused"], rk["fused"]]
        if all(x > y for x, y in zip(steps, steps[1:])): mono.append(row(qid, gold, rk, ro))
        elif rk["scorer"] > rk["fused"]: anyg.append(row(qid, gold, rk, ro))
mono.sort(key=lambda x: (-x["ccmp_gap"], -x["graph_gap"], -x["gap"])); anyg.sort(key=lambda x: (-x["graph_gap"], -x["gap"]))
rows = mono if mono else anyg
print(f"{len(mono)} golds improve at EVERY stage; {len(anyg)} more where the graph (+gate) improves on the scorer\\n")
print("| # | query id | cosine | scorer | +graph (gate off) | +CCMP (gate on) | graph alone | openie | qwen3 | bge | gold | question |")
print("|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|")
for i, x in enumerate(rows[:20], 1):
    r = x["ranks"]; f = lambda k: str(r.get(k)) if r.get(k, BIG) < BIG else ">300"
    print(f"| {i} | {x['id']} | {r['dense_ch']} | {r['scorer_ch']} | {r['fused_nogate']} | {r['fused_gate']} | {r['graph_ch']} | {f('openie')} | {f('qwen3')} | {f('bge')} | {x['gold_title'][:55]} | {x['question'][:70].replace('|','/')}... |")
PICKS = f"{OUT_ROOT}/picks.json"
names = ["scigraphir"] + [n for n in preds if n != "scigraphir"] + ["dense_ch", "scorer_ch", "fused_nogate", "fused_gate", "graph_ch"]
json.dump({"primary": "scigraphir", "against": ["qwen3", "bge"], "systems": names, "candidates": rows[:50]}, open(PICKS, "w"), indent=1)
QIDS = QIDS_OVERRIDE or list(dict.fromkeys(c["id"] for c in rows[:N_EXAMPLES]))
assert QIDS, "the graph lifted no cross-field gold into the top 10 on this field"
GOLDS = {}
for c in rows:
    if c["id"] in QIDS: GOLDS.setdefault(c["id"], []).append(c["gold"])
json.dump(QIDS, open(f"{OUT_ROOT}/qids.json", "w")); json.dump(GOLDS, open(f"{OUT_ROOT}/golds.json", "w"))
print("\\nexamples to interpret:", QIDS, "\\npinned golds:", GOLDS)
"""

INTERPRET = '''# 7. Path interpretations for the chosen queries under each model, on its own graph.
def interpret_with(ckpt, name, graph, sem_ckpt, sem_pop, num_beam=10, path_topk=5):
    out = f"{OUT_ROOT}/paths_{name}.json"
    run_local = f"{RUNS}/interp_{name}"; os.makedirs(run_local, exist_ok=True)
    env_, info = model_env(ckpt, graph, sem_ckpt, sem_pop)
    print(f"\\n[interpret:{name}] {os.path.relpath(ckpt, DRIVE)} on {graph} {info}")
    rc = sh("python -u -m gfmrag.workflow.interpret_paths " + hydra_common(graph) +
            f"+interp.ckpt={ckpt} +interp.qids_file={OUT_ROOT}/qids.json +interp.out={out} "
            f"+interp.probes={PROBES} +interp.num_beam={num_beam} +interp.path_topk={path_topk} "
            f"+interp.golds_file={OUT_ROOT}/golds.json +interp.max_golds=2 +interp.top_views=3 hydra.run.dir={run_local}",
            "/content/gfm-rag", extra=env_, log=f"{run_local}/console.log", check=False)
    assert rc == 0, f"interpret failed (exit {rc}); read {run_local}/console.log"
    assert os.path.exists(out), f"no output at {out}"
    save_index(graph)
    return out

PATHS_OUT = {"frame": interpret_with(FRAME_CKPT, "frame", FRAME_TEST, FRAME_CKPT_SEM, FRAME_POP_SEM)}
if os.path.exists(CONTROL_CKPT):
    PATHS_OUT["control"] = interpret_with(CONTROL_CKPT, "control", FRAME_TEST, FIELD_CKPT_SEM, FIELD_POP_SEM)
PATHS_OUT["openie"] = interpret_with(OPENIE_CKPT, "openie", OPENIE_TEST, FIELD_CKPT_SEM, FIELD_POP_SEM)
print("\\npath files:", {k: os.path.relpath(v, DRIVE) for k, v in PATHS_OUT.items()})
'''

RENDER = '''# 8. Markdown for reading and LaTeX for the thesis (GFM-RAG Table 4 layout + a rank table).
cmd = [sys.executable, "-u", "eval/render_qualitative.py", "--picks", PICKS, "--docs", DOCS,
       "--n", str(len(QIDS)), "--paths-per-arm", "2",
       "--out-md", f"{OUT_ROOT}/qualitative.md", "--out-tex", f"{OUT_ROOT}/qualitative_paths.tex"]
for k, v in PATHS_OUT.items(): cmd += ["--paths", f"{k}={v}"]
sh(cmd, S4)
print("\\nfiles under", os.path.relpath(OUT_ROOT, DRIVE), ":", sorted(os.listdir(OUT_ROOT)))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="biology", choices=["cs", "biology", "physics", "matsci"])
    ap.add_argument("--n", type=int, default=4, help="examples to interpret (top of the candidate list)")
    ap.add_argument("--qids", default="", help="comma list of query ids to force instead of the top N")
    ap.add_argument("--frame-run", default=None, help="Drive-relative run dir of the frame-graph CCMP checkpoint")
    ap.add_argument("--frame-sem", default=None, help="Drive-relative dir of that run's scorer warm-start files")
    ap.add_argument("--control-run", default=None, help="Drive-relative run dir of a frame-graph no-CCMP checkpoint")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pair = "physics+biology" if a.field in ("physics", "biology") else "cs+matsci"
    warm = "biology" if a.field in ("physics", "biology") else "cs"
    frame_run = a.frame_run or f"outputs/sir4_zeroshot/scigraphir_{pair}_qwenmlp_ccmp_e10_b2"
    frame_sem = a.frame_sem or f"outputs/sir4_zeroshot/semantic_sir4_{warm}"
    control_run = a.control_run or f"outputs/routing/sir4_{a.field}/sir4_{a.field}_route_control_e10_b2_s1024"
    qids = [x.strip() for x in a.qids.split(",") if x.strip()]

    src = json.load(open(rb.SRC_NB))["cells"]
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fusion_files = {f"/content/gfm-rag/{rel}": open(f"{FORK}/{rel}").read() for rel in FUSION_REL}
    assert not any("'''" in v for v in fusion_files.values())
    assert "def interpret(" in fusion_files["/content/gfm-rag/gfmrag/trainers/fusion_trainer.py"], "trainer lacks interpret()"
    assert "_reach_layers" in fusion_files["/content/gfm-rag/gfmrag/models/ultra/models.py"], "reasoner lacks reach capture"
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
        "for m in ['gfmrag.models.fusion_reasoner', 'gfmrag.trainers.fusion_trainer', 'gfmrag.workflow.interpret_paths']:\n"
        "    importlib.import_module(m); print('import OK:', m)\n"
        "print('fusion files ready (path interpretation included)')\n")
    overlay = {rel: open(f"{CARGO}/{rel}").read() for rel in OVERLAY_REL}
    assert not any("'''" in v for v in overlay.values())

    # The cloned config-rewrite cell asserts no "sir4" name survives; that guard is for TOMATO
    # builds and is inverted here, where the defaults ARE sir4 graphs. Drop that one line.
    engine = []
    for i in rb.ENGINE_CELLS:
        c = src[i]
        s_ = "".join(c["source"])
        if GUARD in s_:
            c = code(s_.replace(GUARD, ""))
        engine.append(files_cell if i == 7 else c)
    assert not any(GUARD in "".join(c["source"]) for c in engine), "sir4 guard still present"
    warm_ds = f"sir4_{warm}"
    cells = [md(HEADER.replace("__FIELD__", a.field)), md("## 1. GPU + Drive + paths"),
             code(PATHS.replace("__FIELD__", a.field).replace("__FRAME_RUN__", frame_run).replace("__FRAME_SEM__", frame_sem)
                  .replace("__CONTROL_RUN__", control_run).replace("__N__", str(a.n)).replace("__QIDS__", json.dumps(qids))
                  .replace("__WARMDS__", warm_ds)),
             md("## 2. Unpack + install the current scripts"),
             code(UNPACK.replace("__OVERLAY__", json.dumps(overlay)).replace("__BUILT__", built)),
             md("## 2b. Qwen3-Embedding-0.6B (cached on Drive)"), src[rb.QWEN_CELLS[1]],
             md("## 3. Engine + fusion sources\n*(cloned from `tomato_ccmp_ablation.ipynb`; the fusion-source blob is regenerated "
                "from the repo and carries `interpret_paths.py`)*")]
    cells += engine
    cells += [md("## 4. Component tables and scorer files"), code(COMPONENTS.replace("__WARMDS__", warm_ds)),
              md("## 5. Rankings of every system on this field"), code(PREDICT),
              md("## 6. Pick the examples"), code(PICK),
              md("## 7. Path interpretations"), code(INTERPRET),
              md("## 8. Render"), code(RENDER)]
    out = a.out or f"{ROOT}/notebooks/colab_qualitative_{a.field}.ipynb"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(cells)} cells | field {a.field} | frame run {frame_run} | n {a.n} | qids {qids or '(top N)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
