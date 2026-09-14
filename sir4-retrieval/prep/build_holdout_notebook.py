"""
build_holdout_notebook.py -- did joint training generalise?

Two questions, two parts, and only the second needs a GPU.

PART A  HELD-OUT vs IN-DOMAIN.  The joint Physics+Biology model evaluated on CS and
        Materials Science, against the models TRAINED on CS and Materials Science.
        Both prediction files already exist on Drive, so this is a rescore: no GPU,
        about a minute.

        It rescores rather than reading the saved scores.json files because those
        were written by different versions of score_sir4: the older runs stored the
        mean reciprocal rank over ALL golds under `mrr`, the newer ones store
        standard best-gold MRR. Reading them side by side would compare 0.31 against
        0.74 and call it a collapse. The scorer is inlined here so the numbers cannot
        depend on which bundle a runtime happens to hold.

PART B  RESEARCHBENCH.  The same joint checkpoint on a benchmark from outside SIR-4
        entirely: 20,322 documents, 1,367 queries, built and audited separately.
        This is the HGNET/SPHERE evidence shape -- train on your own data, evaluate
        zero-shot on someone else's benchmark -- and it is the only arm where BGE
        previously beat every trained model.

        Needs a GPU. ResearchBench's operator components must be rebuilt with Qwen3
        (the existing ones are BGE-era and the joint model was trained on Qwen3
        features), then one predict run over a 267k-node graph.

Usage
-----
    python3 prep/build_holdout_notebook.py
"""
from __future__ import annotations

import argparse
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SETUP_CELLS = 15          # through the helpers cell of the joint notebook


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(True)}


PATHS = '''import os, re, sys, time, glob, json, math, shutil, zipfile, subprocess
from collections import defaultdict

DRIVE      = "/content/drive/MyDrive/cargo-gfmrag"
CARGO_ROOT = "/content/cargo"
DATA_ROOT  = f"{CARGO_ROOT}/kg-construction/data"
S4         = f"{CARGO_ROOT}/sir4-retrieval"
KGDIR      = f"{CARGO_ROOT}/kg-construction"
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

OP_SLUG = "_content-qwen3"
ALL_DOMS = ["cs", "biology", "physics", "matsci"]
JOINT    = f"{DRIVE}/outputs/__RUN__"
CKPT     = f"{JOINT}/model_best.pth"
OUT      = f"{JOINT}/report"
os.makedirs(OUT, exist_ok=True)
assert os.path.exists(CKPT), f"no checkpoint at {CKPT}"

# WHAT THIS RUN TRAINED ON, read from ITS OWN hydra config rather than assumed.
# The difference decides how every SIR-4 number may be described: a domain in
# train_names is a DEV number (it drove checkpoint selection), a domain absent
# from it is genuinely held out. Getting that backwards would overclaim.
cfgp = f"{JOINT}/.hydra/config.yaml"
assert os.path.exists(cfgp), f"no {cfgp}; cannot tell what this run trained on"
_cfg = open(cfgp).read()
def _names(key):
    m = re.search(key + r":\\n((?:\\s+- .*\\n)+)", _cfg)
    return [l.strip("- \\n") for l in m.group(1).splitlines()] if m else []
TRAIN_G  = _names("train_names")
VALID_G  = _names("valid_names")
TRAINED  = [d for d in ALL_DOMS if any(f"sir4_{d}_" in g for g in TRAIN_G)]
HELD_OUT = [d for d in ALL_DOMS if d not in TRAINED]

# The arm label is DERIVED, never typed. This notebook started life pointed at a
# Physics+Biology run; a hardcoded 'joint P+B' would still read that way while
# scoring all-domain weights, and a mislabelled row in a results table is worse
# than a missing one.
JOINT_LABEL = ("joint (all %d domains)" % len(TRAINED)) if len(TRAINED) == len(ALL_DOMS) \
              else ("joint (%s)" % "+".join(TRAINED) if TRAINED else "joint (unknown)")
print("arm label :", JOINT_LABEL)

print("checkpoint :", time.strftime("%d %b %H:%M", time.localtime(os.path.getmtime(CKPT))))
print("trained on :", TRAINED or "(none matched -- check train_names below)")
print("held out   :", HELD_OUT or "NONE inside SIR-4; ResearchBench is the only clean test")
print("train_names:", TRAIN_G)
print("valid_names:", VALID_G)
print("writing to :", OUT)
'''

PATCH = '''# EVERY predict cell in this notebook resumes from a checkpoint, a path the
# training runs never take. base_trainer._load_checkpoint is reached from
# _setup_model(), which __init__ calls BEFORE it builds the AMP scaler, so
# self.scaler does not exist yet and the resume dies with a bare AttributeError.
# The optimizer load three lines above already guards with hasattr; the scaler
# line was missed. Re-installing the engine restores the stock file, so this must
# run again after any reconnect that rebuilt /content/gfm-rag.
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

SCORER = '''# The scorer, inlined so the numbers cannot depend on which score_sir4.py a
# runtime happens to hold. Verified identical to the repo's eval/score_sir4.py on
# 400 random cases across 13 metrics.
BIG = 10 ** 9

def ranked_docs(rec):
    p = rec.get("predictions", rec)
    docs = p.get("document", p) if isinstance(p, dict) else p
    return [d[0] if isinstance(d, (list, tuple)) else d for d in docs]

def best_gold_rank(ranked, golds):
    for i, d in enumerate(ranked, 1):
        if d in golds:
            return i
    return BIG

def score_one(ranked, golds, sets_):
    r   = best_gold_rank(ranked, golds)
    pos = {d: i + 1 for i, d in enumerate(ranked)}
    rks = [pos.get(g, BIG) for g in golds]
    out = {"mrr":  1.0 / r if r < BIG else 0.0,
           "mgrr": (sum(1.0 / x for x in rks if x < BIG) / len(golds)) if golds else 0.0}
    for k in (1, 3, 5, 10, 15, 20, 100):
        topk = set(ranked[:k])
        out[f"recall@{k}"]      = len(topk & golds) / len(golds) if golds else 0.0
        out[f"hits@{k}"]        = float(r <= k)
        out[f"completeset@{k}"] = float(any(set(s) <= topk for s in sets_)) if sets_ else 0.0
    rel  = [float(d in golds) for d in ranked[:5]]
    dcg  = sum(v / math.log2(i + 2) for i, v in enumerate(rel))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(golds), 5)))
    out["ndcg@5"] = dcg / idcg if idcg else 0.0
    return out

def score_run(pred_path, queries, sets_by_q, bge_rank):
    preds = json.load(open(pred_path))
    if isinstance(preds, dict):
        preds = list(preds.values())
    slices, seen = defaultdict(list), set()
    for rec in preds:
        qid = rec.get("id"); q = queries.get(qid)
        if q is None or qid in seen:
            continue
        seen.add(qid)
        m = score_one(ranked_docs(rec), set(q["supporting_documents"]), sets_by_q.get(qid, []))
        # SIR-4 has a query-level `stratum`; ResearchBench has per-gold labels plus
        # the conservative `domain_slice` its pool builder derives from them.
        st = q.get("stratum") or {"same_only": "same",
                                  "cross_involved": "cross"}.get(q.get("domain_slice"))
        names = ["all"] + ([st] if st else [])
        if qid in bge_rank:
            names.append("dissimilar" if bge_rank[qid] > 100 else "similar")
        for n in names:
            slices[n].append(m)
    return {s: {"n": len(rows), **{k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}}
            for s, rows in slices.items()}
print("scorer ready")
'''

COMPS = '''# Corpora and operator components for all four SIR-4 test graphs. Restored from
# Drive where the training runs cached them; recomputed only if absent.
for d in ALL_DOMS:
    if os.path.exists(f"{DATA_ROOT}/sir4_{d}_test_v16sc/processed/stage1/nodes.csv"):
        print(f"{d:9} unpacked"); continue
    z = f"{DRIVE}/sir4_{d}_bundle.zip"
    assert os.path.exists(z) and zipfile.is_zipfile(z), f"{z} missing or not a zip"
    zipfile.ZipFile(z).extractall(CARGO_ROOT); print(f"{d:9} unpacked")

for d in ALL_DOMS:
    dst = f"{DATA_ROOT}/sir4_{d}_test_v16sc/raw"
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(f"{dst}/documents.json"):
        shutil.copy(f"{DATA_ROOT}/sir4_{d}_test/raw/documents.json", f"{dst}/documents.json")

import numpy as np
for d in ALL_DOMS:
    g   = f"sir4_{d}_test_v16sc"
    npz = f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
    if not os.path.exists(npz):
        cached = f"{DRIVE}/outputs/sir4_{d}/cache/{g}_operator_components{OP_SLUG}.npz"
        if os.path.exists(cached):
            shutil.copy(cached, npz)
        else:
            print(f"{d:9} computing components ...")
            rc = sh(f"python3 -u experiments/probe_greasoner/precompute_operator_components.py "
                    f"--dataset sir4_{d} --graph {g} --split test --model /content/qwen3", KGDIR)
            assert rc == 0 and os.path.exists(npz), f"components failed for {d}"
            os.makedirs(os.path.dirname(cached), exist_ok=True); shutil.copy(npz, cached)
    # The encoder is the only thing that distinguishes a BGE table from a Qwen3 one:
    # the shapes are identical, so a mismatch would train nothing and just score wrong.
    z = np.load(npz, allow_pickle=True)
    enc = str(z["encoder"]) if "encoder" in z else "(unrecorded)"
    assert "qwen" in enc.lower(), f"{npz} was built with {enc!r}, not Qwen3"
    print(f"{d:9} components  {z['dense'].shape}  {enc}")
'''


DENSE = '''# TRAINING-FREE BASELINES, one per encoder. These are the arms the trained model
# has to beat to justify itself, and on ResearchBench BGE previously beat every
# trained arm, so they are not a formality.
#
# Encoder-only: no graph, no GNN, no checkpoint. A few minutes per domain and
# cached afterwards. bge_sir4.py gives a non-default model its own cache tag and
# output filename, so Qwen3 cannot silently reload BGE's .npy.
QWEN = "/content/qwen3"
DENSE_PRED = {}
for d in ALL_DOMS:
    DENSE_PRED[d] = {}
    for label, model, tag in (("BGE dense", None, ""),
                              ("Qwen3 dense", QWEN, "_content-qwen3")):
        f = f"{S4}/data/predictions_bge{tag}_sir4_{d}_test.json"
        if not os.path.exists(f):
            # BGE's may already be cached on Drive from the training runs.
            for src in (f"{DRIVE}/outputs/sir4_{d}/cache/{os.path.basename(f)}",
                        f"{DRIVE}/outputs/sir4_{d}/predictions_bge_test.json" if not tag else ""):
                if src and os.path.exists(src):
                    os.makedirs(os.path.dirname(f), exist_ok=True)
                    shutil.copy(src, f); break
        if not os.path.exists(f):
            if model and not os.path.isdir(model):
                print(f"  {d:9} {label:12} skipped, {model} absent"); continue
            print(f"  {d:9} {label:12} encoding ...")
            rc = sh(f"python3 -u eval/bge_sir4.py --dataset sir4_{d} --split test --topk 300"
                    + (f" --model {model}" if model else ""), S4)
            if rc != 0 or not os.path.exists(f):
                print(f"  {d:9} {label:12} FAILED, skipped"); continue
            os.makedirs(f"{DRIVE}/outputs/sir4_{d}/cache", exist_ok=True)
            shutil.copy(f, f"{DRIVE}/outputs/sir4_{d}/cache/{os.path.basename(f)}")
        DENSE_PRED[d][label] = f
        print(f"  {d:9} {label:12} ok")

# A Qwen3 file byte-identical to BGE's means the encoder never changed, and a
# duplicate row is indistinguishable from a real result in the final table.
import hashlib
for d, arms in DENSE_PRED.items():
    if len(arms) == 2:
        h = [hashlib.md5(open(x, "rb").read()).hexdigest() for x in arms.values()]
        assert h[0] != h[1], f"{d}: Qwen3 and BGE predictions are identical; check model_slug"
print("\\ndense baselines ready")
'''


PREDICT = '''# The joint checkpoint on every SIR-4 test graph. joint_all wrote no predictions,
# so these have to be produced: four predict runs, no training, roughly 5-10 min
# each on graphs of 18k-57k nodes. Cached, so a reconnect does not repeat them.
def comp_path(d, s="test"):
    return f"{DATA_ROOT}/sir4_{d}_{s}_v16sc/operator_components{OP_SLUG}.npz"

JPRED = {}
for d in ALL_DOMS:
    g   = f"sir4_{d}_test_v16sc"
    rd  = f"{OUT}/pred_{d}"
    dst = f"{rd}/predictions_{g}.json"
    if os.path.exists(dst):
        print(f"{d:9} cached"); JPRED[d] = dst; continue
    npz = comp_path(d)
    assert os.path.exists(npz), f"missing {npz}; run the components cell"
    os.makedirs(rd, exist_ok=True)
    rc = sh("python -u -m gfmrag.workflow.sft_training "
            "--config-path config/gfm_reasoner --config-name sft_training_fusion "
            "text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{g}] datasets.valid_names=[{g}] "
            "trainer.args.do_train=false trainer.args.do_eval=false "
            "+trainer.args.eval_batch_size=2 "
            "+trainer.args.do_predict=true +trainer.args.predict_top_k=300 "
            f"+trainer.args.resume_from_checkpoint={CKPT} "
            f"hydra.run.dir={rd}",
            "/content/gfm-rag", extra=dict(
                WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                OPERATOR_COMPONENTS=npz, OPERATOR_COMPONENTS_TEST=npz,
                FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0"))
    assert rc == 0, f"predict on {d} failed, exit {rc}"
    JPRED[d] = dst
print("\\n", {k: os.path.relpath(v, OUT) for k, v in JPRED.items()})
'''


PART_A = '''# PART A: ONE joint model against FOUR domain specialists, on all four SIR-4
# test sets. Rescored from predictions with one scorer, so the arms are comparable.
#
# READ THE LABEL COLUMN. For a domain this run TRAINED on, the test graph was also
# its validation set, so the number is a DEV number and is optimistic. For a domain
# it did not train on, the number is genuinely held out. joint_all trained on all
# four, so every row here is dev and ResearchBench in Part B is the only clean test.
def sets_path(d):
    tag = "cs_test_final" if d == "cs" else f"{d}_test_low"
    return f"{CARGO_ROOT}/quartet/data/benchmark/{tag}/sets.json"

def specialist_pred(d):
    """Newest qwenop run for this domain. qwenop matters: a BGE-era checkpoint is a
    different model, not a different training set, and would not be a fair rival."""
    hits = [p for p in glob.glob(f"{DRIVE}/outputs/sir4_{d}/*/predictions_sir4_{d}_test_v16sc.json")
            if "qwenop" in p.lower()]
    return max(hits, key=os.path.getmtime) if hits else None

A = {}
for d in ALL_DOMS:
    q = f"{DATA_ROOT}/sir4_{d}_test/raw/test.json"
    if not os.path.exists(q):
        print(f"  {d:9} corpus missing, skipped"); continue
    queries = {r["id"]: r for r in json.load(open(q))}
    sets_by_q = {}
    if os.path.exists(sets_path(d)):
        raw = json.load(open(sets_path(d)))
        for qid, v in raw.items():
            sets_by_q[qid] = [list(s) for s in (v.get("sets") if isinstance(v, dict) else v)]

    bge = f"{S4}/data/predictions_bge_sir4_{d}_test.json"
    if not os.path.exists(bge):
        for src in (f"{DRIVE}/outputs/sir4_{d}/cache/predictions_bge_sir4_{d}_test.json",
                    f"{DRIVE}/outputs/sir4_{d}/predictions_bge_test.json"):
            if os.path.exists(src):
                os.makedirs(os.path.dirname(bge), exist_ok=True); shutil.copy(src, bge); break
    bge_rank = {}
    if os.path.exists(bge):
        for rec in json.load(open(bge)):
            qq = queries.get(rec["id"])
            if qq:
                bge_rank[rec["id"]] = best_gold_rank(ranked_docs(rec),
                                                     set(qq["supporting_documents"]))
    else:
        print(f"  ! {d}: no BGE baseline, similar/dissimilar slices skipped")

    arms = {JOINT_LABEL: JPRED.get(d),
            f"{d} specialist": specialist_pred(d)}
    arms.update(DENSE_PRED.get(d, {}))
    A[d] = {}
    for label, pth in arms.items():
        if not pth or not os.path.exists(pth):
            print(f"  {d:9} {label:22} MISSING"); continue
        A[d][label] = score_run(pth, queries, sets_by_q, bge_rank)
    print(f"  {d:9} {'DEV (trained on)' if d in TRAINED else 'HELD OUT'}  "
          f"{len(A[d])} arms")

MET = [("mrr","MRR"),("ndcg@5","nDCG@5"),("recall@3","R@3"),
       ("recall@5","R@5"),("completeset@5","CS@5")]

def cross_same_table(SC, out, label_of=lambda k: k, title="same vs cross"):
    """One row per arm: same, cross, and the gap between them.

    The per-slice tables above answer 'how good is each arm'. This answers the
    question the benchmark is actually about: does the arm hold up when the useful
    paper comes from ANOTHER field. A large positive gap means it is riding
    same-domain vocabulary overlap."""
    out(f"### {title}")
    out("| arm | metric | same | cross | cross - same |")
    out("|---|---|--:|--:|--:|")
    for key, lab in MET:
        for k, sc in SC.items():
            if "same" not in sc or "cross" not in sc:
                continue
            a, b = sc["same"][key], sc["cross"][key]
            out(f"| {label_of(k)} | {lab} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    out()
    out(f"n = same {next((sc['same']['n'] for sc in SC.values() if 'same' in sc), 0)}, "
        f"cross {next((sc['cross']['n'] for sc in SC.values() if 'cross' in sc), 0)}.")
    out()

lines = []
def out(s=""):
    print(s); lines.append(s)

out("## One joint model vs four domain specialists")
out()
out(f"Trained on: {', '.join(TRAINED) or 'nothing matched'}. "
    f"Held out inside SIR-4: {', '.join(HELD_OUT) or 'NONE'}.")
out()
for d in ALL_DOMS:
    if d not in A or not A[d]: continue
    tag = "DEV, this domain was in training" if d in TRAINED else "HELD OUT"
    for slc in ("all", "same", "cross", "similar", "dissimilar"):
        rows = [(lab, sc[slc]) for lab, sc in A[d].items() if slc in sc]
        if not rows: continue
        out(f"### {d} / {slc}   (n={rows[0][1]['n']}, {tag})")
        out("| arm | " + " | ".join(l for _, l in MET) + " |")
        out("|---|" + "--:|" * len(MET))
        best = {k: max(r[k] for _, r in rows) for k, _ in MET}
        for lab, r in rows:
            out(f"| {lab} | " + " | ".join(
                (f"**{r[k]:.4f}**" if r[k] == best[k] else f"{r[k]:.4f}") for k, _ in MET) + " |")
        out()

for d in ALL_DOMS:
    if d in A and A[d]:
        cross_same_table(A[d], out, title=f"{d}: same vs cross")

# THE HEADLINE. One generalist minus its domain specialist. Near zero across four
# domains is the foundation-model claim: one model replaces four.
out("### joint minus specialist")
out("| domain | label | slice | " + " | ".join(l for _, l in MET) + " | wins |")
out("|---|---|---|" + "--:|" * len(MET) + "--:|")
for d in ALL_DOMS:
    j = A.get(d, {}).get(JOINT_LABEL)
    sp = A.get(d, {}).get(f"{d} specialist")
    if not (j and sp): continue
    for slc in ("all", "cross"):
        if slc not in j or slc not in sp: continue
        ds = [j[slc][k] - sp[slc][k] for k, _ in MET]
        out(f"| {d} | {'dev' if d in TRAINED else 'held out'} | {slc} | "
            + " | ".join(f"{x:+.4f}" for x in ds)
            + f" | {sum(1 for x in ds if x > 0)}/{len(ds)} |")
out()
out("Near zero means one model does the work of four. Positive means the joint "
    "model is BETTER than the specialist, which would say the extra domains help "
    "rather than dilute.")
open(f"{OUT}/joint_vs_specialists.md", "w").write("\\n".join(lines))
json.dump(A, open(f"{OUT}/joint_vs_specialists.json", "w"), indent=1)
print(f"\\nwrote {OUT}/joint_vs_specialists.md")
'''


RB_PREP = '''# PART B setup: ResearchBench. A benchmark from outside SIR-4 entirely.
RB   = "researchbench"
RBG  = f"{RB}_test_v16sc"
RBRUN = f"{OUT}/rb_joint"
os.makedirs(RBRUN, exist_ok=True)

if not os.path.exists(f"{DATA_ROOT}/{RBG}/processed/stage1/nodes.csv"):
    z = f"{DRIVE}/{RB}_bundle.zip"
    assert os.path.exists(z), f"missing {z}"
    assert zipfile.is_zipfile(z), f"{z} is not a zip; re-upload it"
    zipfile.ZipFile(z).extractall(CARGO_ROOT); print("researchbench unpacked")
else:
    print("researchbench already unpacked")

dst = f"{DATA_ROOT}/{RBG}/raw"
os.makedirs(dst, exist_ok=True)
if not os.path.exists(f"{dst}/documents.json"):
    shutil.copy(f"{DATA_ROOT}/{RB}_test/raw/documents.json", f"{dst}/documents.json")

# QWEN3 COMPONENTS, NOT THE BGE ONES. The joint model learned its operator scalars
# against Qwen3 dense/S/M distributions. ResearchBench's existing components are
# BGE-era from the earlier transfer experiment; feeding those here would measure
# encoder mismatch, not transfer. Shapes are identical either way, so only the
# recorded encoder name can catch it.
RB_NPZ = f"{DATA_ROOT}/{RBG}/operator_components{OP_SLUG}.npz"
cached = f"{DRIVE}/outputs/{RB}/cache/{RBG}_operator_components{OP_SLUG}.npz"
if os.path.exists(RB_NPZ):
    print("components present")
elif os.path.exists(cached):
    shutil.copy(cached, RB_NPZ); print("components restored from Drive")
else:
    print("computing Qwen3 components for 20,322 docs + 1,367 queries (10-20 min) ...")
    rc = sh(f"python3 -u experiments/probe_greasoner/precompute_operator_components.py "
            f"--dataset {RB} --graph {RBG} --split test --model /content/qwen3", KGDIR)
    assert rc == 0 and os.path.exists(RB_NPZ), "components failed"
    os.makedirs(os.path.dirname(cached), exist_ok=True); shutil.copy(RB_NPZ, cached)

import numpy as np
z = np.load(RB_NPZ, allow_pickle=True)
enc = str(z["encoder"]) if "encoder" in z else "(unrecorded)"
assert "qwen" in enc.lower(), (
    f"{RB_NPZ} was built with {enc!r}, not Qwen3. Delete it and rerun this cell.")
print(f"components  dense {z['dense'].shape}  {len(z['query_ids'])} queries  {enc}")
'''

RB_RUN = '''# The joint checkpoint on ResearchBench. 267,226 nodes and 685,789 edges, so this
# is the slow cell: expect roughly 30-60 min at eval_batch_size 2.
RB_PRED = f"{RBRUN}/predictions_{RBG}.json"
if os.path.exists(RB_PRED):
    print("cached:", RB_PRED)
else:
    rc = sh("python -u -m gfmrag.workflow.sft_training "
            "--config-path config/gfm_reasoner --config-name sft_training_fusion "
            "text_emb_model=qwen3_st "
            f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
            f"datasets.train_names=[{RBG}] datasets.valid_names=[{RBG}] "
            "trainer.args.do_train=false trainer.args.do_eval=false "
            "+trainer.args.eval_batch_size=2 "
            "+trainer.args.do_predict=true +trainer.args.predict_top_k=300 "
            f"+trainer.args.resume_from_checkpoint={CKPT} "
            f"hydra.run.dir={RBRUN}",
            "/content/gfm-rag", log=f"{RBRUN}/console.log", extra=dict(
                WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                OPERATOR_COMPONENTS=RB_NPZ, OPERATOR_COMPONENTS_TEST=RB_NPZ,
                FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0"))
    assert rc == 0, f"ResearchBench predict failed, exit {rc}"
assert os.path.exists(RB_PRED), f"no predictions at {RB_PRED}"
print("predictions:", RB_PRED)
'''

RB_SCORE = '''# Score the joint model on ResearchBench and put it next to the arms already run.
# recall@3 and recall@15 are ResearchBench's own operating points (4% and 20% of a
# 75-candidate pool); recall@5 and nDCG@5 keep it comparable with the SIR-4 tables.
RBQ = f"{DATA_ROOT}/{RB}_test/raw/test.json"
queries = {r["id"]: r for r in json.load(open(RBQ))}
sets_json = f"{S4}/data/{RB}_sets.json"
sets_by_q = {}
if os.path.exists(sets_json):
    raw = json.load(open(sets_json))
    for qid, v in raw.items():
        sets_by_q[qid] = [list(s) for s in (v.get("sets") if isinstance(v, dict) else v)]

# DENSE BASELINES, BUILT IF ABSENT. The earlier ResearchBench experiment wrote
# these under /content and never persisted them, so looking in two fixed places
# left RB_BGE = None, which silently dropped the BGE arm AND the similar/dissimilar
# split. BGE is the arm that beat every trained model last time, so a table quietly
# missing it is worse than a table that takes ten more minutes to produce.
QWEN = "/content/qwen3"
def rb_dense(label, model, tag):
    """Return a ResearchBench dense prediction file, encoding once if needed."""
    local = f"{S4}/data/predictions_bge{tag}_{RB}_test.json"
    if os.path.exists(local):
        return local
    # Anywhere on Drive, under any of the names past runs used.
    pats = [f"{DRIVE}/**/predictions_bge{tag}_{RB}_test.json"]
    if tag:
        pats.append(f"{DRIVE}/**/predictions_qwen3_{RB}_test.json")
    for pat in pats:
        for c in sorted(glob.glob(pat, recursive=True), key=os.path.getmtime, reverse=True):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            shutil.copy(c, local)
            print(f"  {label:12} restored from {os.path.relpath(c, DRIVE)}")
            return local
    if model and not os.path.isdir(model):
        print(f"  {label:12} skipped, {model} absent"); return None
    print(f"  {label:12} encoding 20,322 docs + 1,367 queries ...")
    rc = sh(f"python3 -u eval/bge_sir4.py --dataset {RB} --split test --topk 300"
            + (f" --model {model}" if model else ""), S4)
    if rc != 0 or not os.path.exists(local):
        print(f"  {label:12} FAILED"); return None
    dst = f"{DRIVE}/outputs/{RB}/cache/{os.path.basename(local)}"
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(local, dst)
    return local

RB_BGE  = rb_dense("BGE dense",   None,  "")
RB_QWEN = rb_dense("Qwen3 dense", QWEN,  "_content-qwen3")
assert RB_BGE, ("no BGE baseline for ResearchBench. It defines the similar/"
                "dissimilar split and is the arm that beat every trained model "
                "last time, so the table is not worth reading without it.")

# The similar/dissimilar split: a query is `dissimilar` when plain BGE ranks its
# BEST gold below 100.
bge_rank = {}
for rec in json.load(open(RB_BGE)):
    q = queries.get(rec["id"])
    if q:
        bge_rank[rec["id"]] = best_gold_rank(ranked_docs(rec), set(q["supporting_documents"]))

ARMS = {f"{JOINT_LABEL} zero-shot": RB_PRED, "BGE dense": RB_BGE}
if RB_QWEN:
    import hashlib
    if (hashlib.md5(open(RB_QWEN, "rb").read()).hexdigest()
            == hashlib.md5(open(RB_BGE, "rb").read()).hexdigest()):
        print("  ! Qwen3 file is byte-identical to BGE's; dropping it rather than "
              "printing a duplicate row as a result")
    else:
        ARMS["Qwen3 dense"] = RB_QWEN
# Prior single-domain transfer runs, if the earlier experiment left them behind.
for c in sorted(glob.glob(f"{DRIVE}/outputs/{RB}/**/predictions_{RBG}.json", recursive=True)):
    tag = os.path.basename(os.path.dirname(c))
    if RBRUN.endswith(tag): continue
    ARMS[f"prior: {tag}"] = c

B, RBMET = {}, [("mrr","MRR"),("ndcg@5","nDCG@5"),("recall@3","R@3"),
                ("recall@5","R@5"),("recall@15","R@15")]
for lab, p in ARMS.items():
    try:
        B[lab] = score_run(p, queries, sets_by_q, bge_rank)
        print(f"  {lab:28} {os.path.relpath(p, DRIVE)}")
    except Exception as e:
        print(f"  {lab:28} SKIPPED ({e})")

lines = []
def out2(s=""):
    print(s); lines.append(s)
out2("## ResearchBench: 20,322 documents, 1,367 queries, outside SIR-4 entirely")
out2()
for slc in ("all", "same", "cross", "similar", "dissimilar"):
    rows = [(lab, sc[slc]) for lab, sc in B.items() if slc in sc]
    if not rows: continue
    out2(f"### {slc}   (n={rows[0][1]['n']})")
    out2("| arm | " + " | ".join(l for _, l in RBMET) + " |")
    out2("|---|" + "--:|" * len(RBMET))
    best = {k: max(r[k] for _, r in rows) for k, _ in RBMET}
    for lab, r in rows:
        out2(f"| {lab} | " + " | ".join(
            (f"**{r[k]:.4f}**" if r[k] == best[k] else f"{r[k]:.4f}") for k, _ in RBMET) + " |")
    out2()
if any("same" in sc and "cross" in sc for sc in B.values()):
    cross_same_table(B, out2, title="ResearchBench: same vs cross")
else:
    out2("No same/cross split: ResearchBench queries carry neither `stratum` nor "
         "`domain_slice`. Rebuild the pool with the corrected builder to get it.")
    out2()

out2(f"Arm under test: {JOINT_LABEL}, trained on {', '.join(TRAINED)} and never "
     f"on ResearchBench. BGE beat every trained arm in the earlier ResearchBench "
     f"experiment; whether this one changes that is what the table answers.")
open(f"{OUT}/researchbench.md", "w").write("\\n".join(lines))
json.dump(B, open(f"{OUT}/researchbench.json", "w"), indent=1)
print(f"\\nwrote {OUT}/researchbench.md")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="_joint_all/joint_qwenop_ep10_b2",
                    help="run dir under {DRIVE}/outputs")
    ap.add_argument("--from-notebook", default=f"{ROOT}/colab_sir4_joint_pb.ipynb")
    ap.add_argument("--out", default=f"{ROOT}/colab_sir4_holdout_report.ipynb")
    a = ap.parse_args()
    if not os.path.exists(a.from_notebook):
        return print(f"missing {a.from_notebook}; build the joint notebook first") or 1

    nb = json.load(open(a.from_notebook))
    cells = [json.loads(json.dumps(c)) for c in nb["cells"][:SETUP_CELLS]]
    cells[0] = md(
        "# Did joint training generalise?\n\n"
        "**Part A, no GPU.** The joint Physics+Biology model on CS and Materials "
        "Science, against the models trained on those domains. Both prediction "
        "files are already on Drive, so this is a rescore.\n\n"
        "**Part B, GPU.** The same checkpoint on ResearchBench: 20,322 documents "
        "and 1,367 queries, from outside SIR-4 entirely.\n\n"
        "Every arm is rescored with one inlined scorer. The saved `scores.json` "
        "files are not comparable to each other: older runs stored the mean "
        "reciprocal rank over all golds under `mrr`, newer ones store standard "
        "best-gold MRR, and reading them side by side would show 0.31 against 0.74 "
        "for a model that did not change.")

    cells += [
        md("## A1. Paths"), code(PATHS.replace("__RUN__", a.run)),
        md("## A2. Guard the checkpoint-resume path\n\n"
           "Every predict cell here resumes from a checkpoint. Rerun this after "
           "any reconnect that rebuilt `/content/gfm-rag`."),
        code(PATCH),
        md("## A3. Scorer"), code(SCORER),
        md("## A4. Corpora and operator components (all four SIR-4 test graphs)"),
        code(COMPS),
        md("## A5. Training-free baselines (BGE and Qwen3 dense)\n\nEncoder only, "
           "no graph and no checkpoint. These are what the trained model has to "
           "beat; on ResearchBench BGE previously beat every trained arm."),
        code(DENSE),
        md("## A6. Predict on all four SIR-4 test sets\n\nFour predict runs, "
           "5-10 min each. The joint_all run wrote no predictions, so these have "
           "to be produced. Cached, so a reconnect does not repeat them."),
        code(PREDICT),
        md("## A7. One joint model vs specialists and baselines\n\nRead the **label** "
           "column. A domain this run trained on gives a DEV number, because its "
           "test graph was also the validation set. Only ResearchBench in Part B "
           "is clean for a model trained on all four."),
        code(PART_A),
        md("## B1. ResearchBench setup\n\nRebuilds ResearchBench's operator "
           "components with **Qwen3**. The existing ones are BGE-era from the "
           "earlier experiment and would measure encoder mismatch, not transfer."),
        code(RB_PREP),
        md("## B2. Predict on ResearchBench\n\nThe slow cell: 267,226 nodes, "
           "685,789 edges. Roughly 30 to 60 minutes."),
        code(RB_RUN),
        md("## B3. ResearchBench table"), code(RB_SCORE),
    ]
    nb["cells"] = cells
    json.dump(nb, open(a.out, "w"), indent=1)

    bad = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        lines, keep, skip = "".join(c["source"]).splitlines(), [], False
        for ln in lines:
            if skip:
                keep.append(""); skip = ln.rstrip().endswith("\\"); continue
            if ln.lstrip().startswith(("!", "%")):
                keep.append(""); skip = ln.rstrip().endswith("\\")
            else:
                keep.append(ln)
        try:
            ast.parse("\n".join(keep))
        except SyntaxError as e:
            bad += 1; print(f"  CELL {i} DOES NOT PARSE: {e}")

    checks = [
        ("part A rescores only", "sh(" not in PART_A),
        ("predict cell caches", "cached" in PREDICT),
        ("dense baselines both encoders", "Qwen3 dense" in DENSE),
        ("dense contamination guard", "byte-identical" in DENSE or "identical" in DENSE),
        ("dev vs held-out label", "TRAINED else" in PART_A),
        ("components are Qwen3", '"qwen" in enc.lower()' in COMPS),
        ("specialist requires qwenop", '"qwenop" in p.lower()' in PART_A),
        ("joint minus specialist", "joint minus specialist" in PART_A),
        ("RB components are Qwen3", '"qwen" in enc.lower()' in RB_PREP),
        ("RB uses the joint ckpt", "resume_from_checkpoint={CKPT}" in RB_RUN),
        ("one scorer for every arm", "def score_run" in SCORER),
        ("scaler guard shipped", "hasattr(self" in PATCH),
        ("arm label derived", "JOINT_LABEL" in PATHS and "JOINT_LABEL" in RB_SCORE
                              and "JOINT_LABEL" in PART_A),
        ("no hardcoded P+B", "joint P+B" not in PART_A + RB_SCORE),
        ("cross vs same table", "cross_same_table" in PART_A and "cross_same_table" in RB_SCORE),
        ("RB strata fallback", "cross_involved" in SCORER),
        ("RB builds its own BGE", "def rb_dense" in RB_SCORE),
        ("RB asserts BGE present", "assert RB_BGE" in RB_SCORE),
        ("MRR and MGRR both kept", '"mgrr"' in SCORER),
    ]
    for n, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {n}")
    print(f"\nwrote {a.out}  ({len(nb['cells'])} cells)")
    print("ALL CHECKS PASS" if not bad and all(k for _, k in checks) else "PROBLEMS ABOVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
