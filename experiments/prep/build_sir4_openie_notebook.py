"""
build_sir4_openie_notebook.py -- the "+ Graph Reasoner (OpenIE graph)" row of the SIR-4
in-field table, for all four fields.

THE ROW. The cumulative ablation in the SIR-4 table goes: multi-view scorer -> + graph
reasoner (frame graph) -> + CCMP. This notebook adds the construction control between the
first two: the SAME multi-view scorer fused with the SAME graph reasoner, trained the same
way, on the OpenIE entity graph instead of the frame graph. CCMP is off, so the row sits at
the "+ Graph Reasoner" rung and the only thing that differs from the frame-graph row is the
graph. Four in-field runs: train on sir4_<field>_train (OpenIE), evaluate on
sir4_<field>_test (OpenIE), same-field and cross-field slices, nDCG@5.

THE GRAPHS EXIST. retriever/data/sir4_<field>_{train,test}/processed/stage1 holds an
entity + document graph per field (built 18 Aug, inside every bundle on Drive). GFM-RAG
trained on them in August; SciGraphIR never has. They are smaller than the frame graphs
(CS train 205k nodes vs 243k), so nothing here needs more memory than the runs that fit.

SEEDS. Every query seeds on the OpenIE graphs except four in CS (3 train, 1 test). The
loader drops a seedless query silently; the scoring cell here re-inserts any test query
without a prediction as an EMPTY ranking, so it counts as a miss and the denominator stays
the full test set. Say so in the caption rather than reporting over 1,051 queries.

Reuses build_rb_zeroshot_notebook.py for the engine cloning, the fusion-source blob, the
inline script overlay and the shell/sync helpers, so the training recipe cannot drift.

Usage
-----
    python3 prep/build_sir4_openie_notebook.py
    python3 prep/build_sir4_openie_notebook.py --fields cs,matsci --epochs 10 --batch 2
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
SETS_MAP = {d: f"benchmark/data/benchmark/{_DOMAINS[d][1]}/sets.json" for d in sorted(_DOMAINS)}

HEADER = '''# SIR-4 — "+ Graph Reasoner (OpenIE graph)": the construction control, four fields

The SIR-4 cumulative ablation goes *multi-view scorer* → *+ graph reasoner* → *+ CCMP*, all
on the frame graph. This notebook adds one row: the **same scorer and the same graph
reasoner, trained the same way, on the OpenIE entity graph** instead. CCMP is off, so the
row sits at the "+ Graph Reasoner" rung and the only difference from that row is the graph.

| step | what | cost |
|---|---|---|
| 1-3 | paths, bundles, current scripts, Qwen3, engine | 10 min |
| 4 | per field: caches, scorer warm start, component tables for the OpenIE graphs | 20-45 min per field (index build) |
| 5 | per field: **train in-field**, predict on the field's test corpus | hours per field, field-serial, finished runs skip |
| 6 | table: same-field and cross-field nDCG@5 per field, macro gap, next to the baseline rows | seconds |

Protocol is the in-field one of the SIR-4 table: train on `sir4_<field>_train`, select and
evaluate on `sir4_<field>_test`, same/cross from the query's `stratum`. The OpenIE graphs
were built on 18 August and are in every bundle; GFM-RAG trained on them, SciGraphIR never has.
'''

PATHS = rb.PATHS
for _old, _new in (
        ('DATASET  = "researchbench"', 'DATASET  = "__LABEL___openie"        # a label for OUT_ROOT; every script gets its own --dataset'),
        ('RBG      = f"{DATASET}_test_v16sc"      # the one ResearchBench graph: 20,322 docs, 1,367 queries',
         'FIELDS   = __FIELDS__                    # field-serial; a finished field is a skip\n'
         'DS_OF    = __DS_OF__                     # field -> dataset name (sir4_<field>, or tomato)\nRBG = None'),
        ('TRAIN = TEST = RBG                      # read by the cloned config-rewrite cell; every run overrides on the CLI',
         'TRAIN, TEST = "__TRAIN__", "__TEST__"   # config defaults only; every run overrides on the CLI'),
        ('BUNDLE   = f"{DRIVE}/{DATASET}_bundle.zip"', 'BUNDLES  = {d: f"{DRIVE}/{DS_OF[d]}_bundle.zip" for d in FIELDS}'),
        ('OUT_ROOT = f"{DRIVE}/outputs/rb_zeroshot"          # everything this notebook writes',
         'OUT_ROOT = f"{DRIVE}/outputs/__LABEL___openie"     # everything this notebook writes'),
        ('CACHE_RB = f"{DRIVE}/outputs/researchbench/cache"  # graph-keyed artefacts the earlier RB runs cached',
         'BASE_OUT = {d: f"{DRIVE}/outputs/baselines/{DS_OF[d]}" for d in FIELDS}   # baseline score files, for the table\n'
         'S4_CACHE = {d: f"{DRIVE}/outputs/{DS_OF[d]}/cache" for d in FIELDS}      # embeddings + indexes the field runs cached'),
        ('S4_DOMS   = ["cs", "biology", "physics", "matsci"]\n', ''),
        ('SEM_WARM_DOM = "__WARM__"    # field that warm-starts the multi-view scorer before joint training\n', ''),
        ('print("reads      ", os.path.basename(BUNDLE), "+ sir4_*_bundle.zip (section 8) + gfm-rag-adapted.zip + qwen3-embedding-0.6b/")',
         'print("reads      ", "sir4_<field>_bundle.zip + gfm-rag-adapted.zip + qwen3-embedding-0.6b/")'),
        ('print("TOMATO arm ", f"{TOMATO_OUT}/{TOMATO_RUN}/model_best.pth")', 'print("fields     ", FIELDS, "| OpenIE graphs", {d: f"{DS_OF[d]}_{{train,test}}" for d in FIELDS})'),
):
    assert _old in PATHS, f"PATHS anchor not found: {_old[:60]!r}"
    PATHS = PATHS.replace(_old, _new)
_a, _b = PATHS.index("# ---- the two SciGraphIR arms"), PATHS.index("EPOCHS, BATCH = ")
PATHS = (PATHS[:_a] + "# ---- the run: one in-field training per field, scorer + graph reasoner, NO CCMP -------\n"
         + PATHS[_b:])
PATHS = PATHS.replace("__TOMATO_RUN__", "n/a").replace("EPOCHS, BATCH = __EPOCHS__, __BATCH__",
                      "EPOCHS, BATCH = __EPOCHS__, __BATCH__\n"
                      "# CS is the largest graph and every August CS run with the multi-view scorer trained at\n"
                      "# batch 1; batch 2 OOMs on an 80 GB A100. The other fields fit at BATCH.\n"
                      "BATCH_OF = {\"cs\": 1}\n"
                      "def batch_of(d): return BATCH_OF.get(d, BATCH)")

UNPACK = '''# 2. Unpack the bundles to an isolated local root and install the CURRENT repo scripts.
KEEP, PARK = f"{SCIGRAPHIR_ROOT}/outputs/caches", "/content/_caches_keep"
if os.path.isdir(KEEP):
    shutil.rmtree(PARK, ignore_errors=True); shutil.move(KEEP, PARK)
if os.path.exists(SCIGRAPHIR_ROOT):
    shutil.rmtree(SCIGRAPHIR_ROOT)
os.makedirs(SCIGRAPHIR_ROOT, exist_ok=True)
for d in FIELDS:
    z = BUNDLES[d]
    assert os.path.exists(z) and zipfile.is_zipfile(z), f"{z} missing or not a zip"
    zipfile.ZipFile(z).extractall(SCIGRAPHIR_ROOT); print("unpacked", os.path.basename(z))
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
import csv, io, collections
csv.field_size_limit(10 ** 7)

def g_of(d, s): return f"{DS_OF[d]}_{s}"        # the OpenIE graphs: NO _v16sc suffix
QUERIES = {d: f"{DATA_ROOT}/{DS_OF[d]}_test/raw/test.json" for d in FIELDS}
SETS    = {d: f"{SCIGRAPHIR_ROOT}/{rel}" for d, rel in __SETS_MAP__.items() if d in FIELDS}
SEEDLESS = {}
for d in FIELDS:
    cp.set_dataset(DS_OF[d])
    for s in ("train", "test"):
        s1 = f"{DATA_ROOT}/{g_of(d, s)}/processed/stage1"
        for p in (f"{s1}/nodes.csv", f"{s1}/edges.csv", f"{s1}/{s}.json", cp.probes_path(s),
                  f"{DATA_ROOT}/{DS_OF[d]}_{s}/raw/documents.json"):
            assert os.path.exists(p), f"missing {p}"
        types = collections.Counter(r["type"] for r in csv.DictReader(open(f"{s1}/nodes.csv")))
        assert "entity" in types and "document" in types, f"{g_of(d, s)} is not an entity+document graph: {dict(types)}"
        q = json.load(open(f"{s1}/{s}.json"))
        SEEDLESS[(d, s)] = [x["id"] for x in q if not any(x["start_nodes"].values())]
        print(f"  {g_of(d, s):22} entity {types['entity']:>8,}  document {types['document']:>7,}  "
              f"queries {len(q):>6,}  seedless {len(SEEDLESS[(d, s)])}")
        # The loader opens {graph}/raw/documents.json. For the OpenIE graphs the graph directory
        # IS the corpus directory (sir4_<field>_<split>), so the file is already in place; only a
        # differently named graph would need the copy.
        _src, _dst = f"{DATA_ROOT}/{DS_OF[d]}_{s}/raw/documents.json", f"{DATA_ROOT}/{g_of(d, s)}/raw/documents.json"
        if os.path.abspath(_src) != os.path.abspath(_dst):
            os.makedirs(os.path.dirname(_dst), exist_ok=True); shutil.copy(_src, _dst)
        assert os.path.exists(_dst), _dst
    if d in SETS: assert os.path.exists(SETS[d]), f"missing {SETS[d]}"
STD_COLS = "mrr,ndcg@5,recall@3,recall@5,recall@10,recall@100,completeset@5"   # completeset needs sets.json (SIR-4 only)
'''

HELPERS = '''# 4a. Helpers: cache restore/save (graph-keyed, so the OpenIE graphs never collide with the
# frame graphs' artefacts), and the per-field preparation.
import numpy as np

def restore_index(g, cache):
    src = f"{cache}/index/{g}"
    if not os.path.isdir(src):
        print(f"  {g}: no cached index on Drive (built on first use, 20-45 min)"); return
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

def opc(d, s):  return f"{DATA_ROOT}/{g_of(d, s)}/operator_components{OP_SLUG}.npz"
def semc(d, s): return f"{DATA_ROOT}/{g_of(d, s)}/semantic_components{OP_SLUG}.npz"

def prep_field(d):
    """Embeddings, index, operator + scorer components for BOTH OpenIE graphs of one field,
    and the field's own multi-view scorer warm start (the same 5d recipe as the frame-graph
    run of that field). Returns the environment for run/predict."""
    ds = DS_OF[d]
    ex = {"SCIGRAPHIR_DATASET": ds}
    if os.path.isdir(f"{S4_CACHE[d]}/op_emb"):
        shutil.copytree(f"{S4_CACHE[d]}/op_emb", f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb", dirs_exist_ok=True)
        print(f"  {d}: embedding caches restored (documents/queries/answers are the same text as the frame-graph run)")
    for s in ("train", "test"):
        restore_index(g_of(d, s), S4_CACHE[d])
        # operator components: keyed by GRAPH name, so the OpenIE ones get their own file.
        c = f"{S4_CACHE[d]}/{g_of(d, s)}_operator_components{OP_SLUG}.npz"
        if not os.path.exists(opc(d, s)):
            if os.path.exists(c): shutil.copy(c, opc(d, s))
            else:
                sh(f"python3 -u precompute/precompute_operator_components.py "
                   f"--dataset {ds} --graph {g_of(d, s)} --split {s} --model {OP_MODEL}", KGDIR, extra=ex)
                os.makedirs(os.path.dirname(c), exist_ok=True); shutil.copy(opc(d, s), c)
        zz = np.load(opc(d, s), allow_pickle=True)
        assert "qwen" in str(zz["encoder"]).lower(), f"{opc(d, s)} was built with {zz['encoder']!r}"
    # the field's multi-view scorer (warm start; the fusion trains it further)
    sem_dir = f"{S4}/results/semantic_{ds}"
    sem_ckpt = f"{sem_dir}/params_semantic_mlp_fixedloss_{ds}.json"
    sem_pop  = f"{sem_dir}/popnet_semantic_mlp_fixedloss_{ds}.pt"
    os.makedirs(sem_dir, exist_ok=True)
    for src in (f"{OUT_ROOT}/semantic_{ds}", f"{DRIVE}/outputs/{ds}/semantic", f"{DRIVE}/outputs/{ds}_ablations_v1/{ds}/semantic"):
        if os.path.isdir(src) and not os.path.exists(sem_ckpt):
            shutil.copytree(src, sem_dir, dirs_exist_ok=True); print(f"  {d}: scorer restored from {os.path.relpath(src, DRIVE)}")
    if not (os.path.exists(sem_ckpt) and os.path.exists(sem_pop)):
        sh(f"python3 -u eval/semantic_scorer.py --dataset {ds} --model {OP_MODEL} "
           f"--arms mlp --loss fixed --train_fit 0 --dev 300 --epochs 30 --patience 10 "
           f"--lr 1e-3 --weight_decay 1e-2 --mlp_hidden 16 --mlp_pop_joint 1 --pop_lambda 1.0 "
           f"--select_on loss --qbatch 32 --seed 0 --seeds 0 --out {sem_dir}", S4, extra=ex)
        shutil.copytree(sem_dir, f"{OUT_ROOT}/semantic_{ds}", dirs_exist_ok=True)
    for p in (sem_ckpt, sem_pop):
        assert os.path.exists(p), f"5d did not write {p}"
    # scorer components aligned to the OpenIE graphs' document order (H memmap is shared per corpus)
    for s in ("train", "test"):
        if not _sem_ok(semc(d, s)):
            sh(f"python3 -u precompute/precompute_semantic_components.py "
               f"--dataset {ds} --model {OP_MODEL} --graph {g_of(d, s)} --split {s}", KGDIR, extra=ex)
        zz = np.load(semc(d, s), allow_pickle=True)
        print(f"  {g_of(d, s):22} H {tuple(int(x) for x in zz['h_shape'])}  Jmax={int(zz['Jmax'])}")
    shutil.copytree(f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb", f"{S4_CACHE[d]}/op_emb", dirs_exist_ok=True)
    # THE "+ GRAPH REASONER" ENVIRONMENT: the run_model(semantic="mlp") defaults of the CCMP
    # notebook -- additive gate, operator-hard-negative objective, PER_GOLD=1, HARDNEG_GRAPH=50,
    # and NO CCMP variable at all, so the head is never built.
    return dict(WANDB_MODE="disabled", HYDRA_FULL_ERROR="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                SCIGRAPHIR_DATASET=ds,
                OPERATOR_COMPONENTS=opc(d, "train"), OPERATOR_COMPONENTS_TEST=opc(d, "test"),
                SEMANTIC_COMPONENTS=semc(d, "train"), SEMANTIC_COMPONENTS_TEST=semc(d, "test"),
                SEMANTIC_CKPT=sem_ckpt, SEMANTIC_POPNET=sem_pop, SEM_POP_LAMBDA="1.0",
                FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0",
                PER_GOLD="1", HARDNEG_GRAPH="50",
                STRAT_TEST=QUERIES[d])
print("helpers ready")
'''

TRAIN_LOOP = '''# 5. Field-serial: prepare, train in-field on the OpenIE graph, predict on the OpenIE test
# graph. Predictions come out of the training run itself (do_predict on valid_names, exactly
# as the frame-graph rows were produced), so there is no separate predict pass.
for k in list(os.environ):
    if k.startswith(("CCMP", "STRAT_", "CQIG", "RESID_", "MISS_W")): os.environ.pop(k)   # nothing leaks between fields

def run_cmd(d, run_dir, epochs, max_steps=None):
    return ("python -u -m gfmrag.workflow.sft_training "
            "--config-path config/gfm_reasoner --config-name sft_training_fusion "
            "text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{g_of(d, 'train')}] datasets.valid_names=[{g_of(d, 'test')}] "
            "model.semantic=mlp model.cqig=false "
            f"trainer.args.num_epoch={epochs} trainer.args.train_batch_size={batch_of(d)} "
            f"+trainer.args.eval_batch_size={batch_of(d)} "
            "+trainer.args.do_predict=true +trainer.args.predict_top_k=300 "
            + (f"trainer.args.max_steps_per_epoch={max_steps} " if max_steps else "")
            + f"hydra.run.dir={run_dir}")

RUN = {}
for d in FIELDS:
    name = f"{DS_OF[d]}_openie_qwenmlp_graph_e{EPOCHS}_b{batch_of(d)}"
    local, drive = f"{RUNS}/{name}", f"{OUT_ROOT}/{name}"
    pred_drive = f"{drive}/predictions_{g_of(d, 'test')}.json"
    print(f"\\n==================== {d} ====================")
    if os.path.exists(pred_drive) and os.path.exists(f"{drive}/model_best.pth"):
        print(f"[skip] finished: {os.path.relpath(drive, DRIVE)}"); RUN[d] = pred_drive; continue
    ex = prep_field(d)
    # smoke first: 1 epoch, 3 steps. Catches a shape or path error before the hours-long run.
    smoke = f"{local}_smoke"; os.makedirs(smoke, exist_ok=True)
    rc = sh(run_cmd(d, smoke, epochs=1, max_steps=3).replace("+trainer.args.do_predict=true", "+trainer.args.do_predict=false"),
            "/content/gfm-rag", extra=ex, log=f"{smoke}/console.log", check=False)
    log = open(f"{smoke}/console.log", errors="ignore").read()
    assert rc == 0, f"{d}: smoke failed (exit {rc}); read {smoke}/console.log"
    assert "semantic='mlp' warm start" in log, f"{d}: multi-view scorer not constructed"
    assert "[ccmp] responsibility head" not in log, f"{d}: a CCMP head was built; this row must have none"
    for s in ("train", "test"): save_index(g_of(d, s), S4_CACHE[d])
    print(f"{d}: smoke passed, starting the {EPOCHS}-epoch run")
    os.makedirs(local, exist_ok=True); os.makedirs(drive, exist_ok=True)
    json.dump({"arm": "multi-view scorer + graph reasoner (OpenIE graph), no CCMP", "field": d,
               "train": g_of(d, "train"), "valid": g_of(d, "test"), "epochs": EPOCHS, "batch": batch_of(d),
               "semantic": "mlp", "ccmp": False, "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{local}/arm.json", "w"), indent=1)
    finish = start_sync(local, drive, every=600)
    try:
        rc = sh(run_cmd(d, local, epochs=EPOCHS), "/content/gfm-rag", extra=ex, log=f"{local}/console.log", check=False)
    finally:
        finish()
    assert rc == 0, f"{d}: training failed, exit {rc} (-9 = host RAM)"
    assert os.path.exists(f"{local}/predictions_{g_of(d, 'test')}.json"), f"{d}: no predictions written"
    sync_dir(local, drive)
    RUN[d] = pred_drive
print("\\npredictions:", {d: os.path.relpath(p, DRIVE) for d, p in RUN.items()})
'''

SCORE = '''# 5b. Score. A test query the loader dropped (no seed) gets an EMPTY ranking here, so it
# counts as a miss and every field is scored over its full test set.
SCORES = f"{OUT_ROOT}/scores"
os.makedirs(SCORES, exist_ok=True)
for d, pred in RUN.items():
    preds = json.load(open(pred))
    if isinstance(preds, dict): preds = list(preds.values())
    have = {r["id"] for r in preds}
    q = json.load(open(QUERIES[d]))
    missing = [x for x in q if x["id"] not in have]
    for x in missing:
        preds.append({"id": x["id"], "stratum": x.get("stratum"), "supporting_documents": x["supporting_documents"],
                      "predictions": {"document": []}})
    filled = f"{SCORES}/{d}_openie_graph_predictions.json"
    json.dump(preds, open(filled, "w"))
    print(f"{d}: {len(missing)} test quer{'y' if len(missing) == 1 else 'ies'} without a prediction "
          f"(seedless: {SEEDLESS[(d, 'test')]}) scored as misses")
    sh([sys.executable, "-u", "eval/score_sir4.py", "--pred", filled, "--queries", QUERIES[d]]
       + (["--sets", SETS[d]] if d in SETS else [])
       + ["--name", f"+ Graph Reasoner (OpenIE graph) [{d}]", "--cols", STD_COLS if d in SETS else STD_COLS.replace(",completeset@5", ""),
        "--json-out", f"{SCORES}/{d}_openie_graph_scores.json",
        "--per-query-out", f"{SCORES}/{d}_openie_graph_perquery.json"], S4)
'''

TABLE = '''# 6. The row, in the SIR-4 table's layout: nDCG@5 (and R@5 underneath), same-field and
# cross-field per field, macro gap = (mean cross - mean same) / mean same over the fields.
# Baseline rows are read from the all-domain baselines notebook's score files on Drive; the
# frame-graph SciGraphIR rows are the thesis numbers and are not recomputed here.
BASE = [("BM25", "scores_BM25.json"), ("BGE-large", "scores_BGE-large.json"),
        ("Qwen3-Embedding", "scores_Qwen3-Embedding.json"), ("SPECTER2", "scores_SPECTER2-base.json"),
        ("SciNCL", "scores_SciNCL.json"), ("ReasonIR-8B", "scores_ReasonIR-8B.json"),
        ("GFM-RAG", "scores_GFM-RAG.json"), ("G-Reasoner", "scores_G-Reasoner.json")]
ROWS = [(lab, {d: f"{BASE_OUT[d]}/{f}" for d in FIELDS}) for lab, f in BASE]
ROWS.append(("+ Graph Reasoner (OpenIE graph)", {d: f"{SCORES}/{d}_openie_graph_scores.json" for d in FIELDS}))
TITLE = {"cs": "CS", "biology": "Bio", "physics": "Phys.", "matsci": "MS", "tomato": "TOMATO"}
lines = []
def out(s=""): print(s); lines.append(s)
for metric, mname in (("ndcg@5", "nDCG@5"), ("recall@5", "R@5")):
    out(f"### {mname} (%)")
    out("| Method | " + " | ".join(f"same {TITLE[d]}" for d in FIELDS) + " | "
        + " | ".join(f"cross {TITLE[d]}" for d in FIELDS) + " | macro gap |")
    out("|---|" + "--:|" * (2 * len(FIELDS) + 1))
    for lab, files in ROWS:
        same, cross = [], []
        for d in FIELDS:
            p = files[d]
            if not os.path.exists(p): same.append(None); cross.append(None); continue
            s = json.load(open(p)); same.append(100 * s["same"][metric]); cross.append(100 * s["cross"][metric])
        if all(v is None for v in same):
            out(f"| {lab} | " + " | ".join("--" for _ in range(2 * len(FIELDS))) + " | -- |"); continue
        ok = [i for i, v in enumerate(same) if v is not None]
        gap = (sum(cross[i] for i in ok) / len(ok) - sum(same[i] for i in ok) / len(ok)) / (sum(same[i] for i in ok) / len(ok))
        out(f"| {lab} | " + " | ".join(f"{v:.2f}" if v is not None else "--" for v in same) + " | "
            + " | ".join(f"{v:.2f}" if v is not None else "--" for v in cross) + f" | {100 * gap:+.2f}% |")
    out()
out("Macro gap = (mean over fields of cross - mean of same) / mean of same, the convention of the printed table.")
open(f"{OUT_ROOT}/openie_row.md", "w").write("\\n".join(lines))
print("wrote", f"{OUT_ROOT}/openie_row.md")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sir4", choices=["sir4", "tomato"], help="sir4 = field-serial over --fields; tomato = one run on tomato_{train,test}")
    ap.add_argument("--fields", default="cs,biology,physics,matsci")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.dataset == "tomato":
        fields, ds_of, label, train, test = ["tomato"], {"tomato": "tomato"}, "tomato", "tomato_train", "tomato_test"
    else:
        fields = [f.strip() for f in a.fields.split(",") if f.strip()]
        assert fields and all(f in _DOMAINS for f in fields), f"unknown field in {fields}; known {sorted(_DOMAINS)}"
        ds_of, label, train, test = {f: f"sir4_{f}" for f in fields}, "sir4", "sir4_physics_train", "sir4_physics_test"
    a.out = a.out or (f"{ROOT}/notebooks/colab_tomato_openie_ablation.ipynb" if a.dataset == "tomato" else f"{ROOT}/notebooks/colab_sir4_openie_ablation.ipynb")
    paths = (PATHS.replace("__FIELDS__", json.dumps(fields)).replace("__DS_OF__", json.dumps(ds_of)).replace("__LABEL__", label)
             .replace("__TRAIN__", train).replace("__TEST__", test).replace("__EPOCHS__", str(a.epochs)).replace("__BATCH__", str(a.batch)))
    header = HEADER if a.dataset != "tomato" else HEADER.replace("# SIR-4 — \"+ Graph Reasoner (OpenIE graph)\": the construction control, four fields",
                                                                  "# TOMATO-Star — \"+ Graph Reasoner (OpenIE graph)\": the construction control").replace(
        "Protocol is the in-field one of the SIR-4 table: train on `sir4_<field>_train`, select and\nevaluate on `sir4_<field>_test`, same/cross from the query's `stratum`.",
        "Protocol is the Table 9.1 one: train on `tomato_train` (OpenIE), select and evaluate on `tomato_test` (OpenIE), "
        "same/cross from the query's `stratum`; the scorer warm start is the Table 9.1 scorer (`outputs/tomato_ablations_v1/tomato/semantic`).")

    src = json.load(open(rb.SRC_NB))["cells"]
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
    cfg_cell = dict(src[9]); cfg_src = "".join(cfg_cell["source"])
    guard = '    assert "sir4" not in t, f"a sir4 dataset reference survived in {cfg}"\n'
    assert guard in cfg_src, "config-rewrite cell changed; check the sir4 guard by hand"
    cfg_cell["source"] = cfg_src.replace(guard, "").splitlines(True)

    cells = [md(header), md("## 1. GPU + Drive + paths"), code(paths),
             md("## 2. Unpack + install the current scripts + verify the OpenIE graphs"),
             code(UNPACK.replace("__OVERLAY__", json.dumps(overlay)).replace("__BUILT__", built).replace("__SETS_MAP__", json.dumps(SETS_MAP))),
             md("## 2b. Qwen3-Embedding-0.6B (cached on Drive)"), src[rb.QWEN_CELLS[1]],
             md("## 3. Engine + fusion sources\n*(cloned from `tomato_ccmp_ablation.ipynb`; the fusion-source blob is regenerated from the repo)*")]
    for i in rb.ENGINE_CELLS:
        cells.append(files_cell if i == 7 else (cfg_cell if i == 9 else src[i]))
    cells += [md("## 4. Per-field preparation"), code(HELPERS),
              md("## 5. Train in-field on the OpenIE graphs, field by field\n**Hours per field.** Local run, ten-minute Drive sync, finished fields skip."),
              code(TRAIN_LOOP), code(SCORE),
              md("## 6. The row"), code(TABLE)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(cells)} cells | fields {fields} | epochs {a.epochs} batch {a.batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
