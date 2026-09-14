"""
build_matrix_notebook.py -- the 4x4 SIR-4 cross-domain transfer matrix.

Every checkpoint evaluated on every domain's test set. The diagonal is in-domain
(already produced by training); the twelve off-diagonal cells are zero-shot
cross-domain transfer, and they need no new data, no extraction and no training.

WHY THIS IS WORTH RUNNING. The ResearchBench experiment shows SIR-4 supervision
transfers better than TOMATO's, but it cannot say whether that is a property of
SIR-4 or of one lucky domain. A full matrix answers that directly: if the
off-diagonal holds up across all twelve pairs, SIR-4 supports domain
generalisation. It is also the same evidence shape HGNET used for SPHERE
(train Physics+Biology, evaluate zero-shot on Comp. Sci. and Mat. Sci.).

WHAT MAKES IT POSSIBLE. relations.csv is byte-identical across all four graphs
(md5 262b4a26683898e97011a14cfe1ddf51), so a checkpoint's DistMult relation
embeddings are meaningful on any of them without reindexing, and node features
are Qwen3 text embeddings, which are corpus-agnostic.

WHAT TO WATCH. Each checkpoint's operator scalars w and beta were fitted on its OWN
corpus's score distributions. Off-diagonal cells therefore combine two effects:
genuine transfer, and calibration mismatch. That is uniform across the matrix so
the comparison is internally consistent, but it means an off-diagonal number is a
lower bound on transferable skill, not a measurement of it.

Usage
-----
    python3 prep/build_matrix_notebook.py
    python3 prep/build_matrix_notebook.py --from-domain biology --op-slug _content-qwen3
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


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(True)}


HELPERS = '''import os, sys, time, subprocess, glob, json, shutil, zipfile
KGDIR = f"{CARGO_ROOT}/kg-construction"
env = dict(os.environ, CARGO_ROOT=CARGO_ROOT, PYTHONUNBUFFERED="1")

def sh(cmd, cwd, extra=None):
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=dict(env, **(extra or {})), shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        c = os.read(p.stdout.fileno(), 8192)
        if not c: break
        sys.stdout.write(c.decode("utf-8", "replace")); sys.stdout.flush()
    p.wait(); print(f"\\n[{time.time()-t0:.0f}s, exit {p.returncode}]")
    return p.returncode

DOMS = __DOMS__
# The encoder the CHECKPOINTS were trained with. The qwenop retrain made
# "_content-qwen3" correct; pass --op-slug "" to build a BGE-era matrix from the
# older checkpoints instead. Mismatch here is silent, so the cell below checks it.
OP_SLUG = "__OPSLUG__"
OUT_M   = f"{DRIVE}/outputs/_transfer_matrix"
os.makedirs(OUT_M, exist_ok=True)
print("domains:", DOMS, "\\noperator slug:", OP_SLUG or "(BGE)", "\\nwriting to:", OUT_M)
'''

UNPACK = '''# Unpack every domain's bundle. They mirror the repo, so one CARGO_ROOT holds all
# four corpora and graphs at once.
os.makedirs(CARGO_ROOT, exist_ok=True)
for d in DOMS:
    g = f"{DATA_ROOT}/sir4_{d}_test_v16sc/processed/stage1/nodes.csv"
    if os.path.exists(g):
        print(f"{d:9} already unpacked"); continue
    z = f"{DRIVE}/sir4_{d}_bundle.zip"
    assert os.path.exists(z), f"missing {z}"
    zipfile.ZipFile(z).extractall(CARGO_ROOT)
    print(f"{d:9} unpacked")

# The loader opens {graph}/raw/documents.json, which the graph build never writes.
for d in DOMS:
    dst = f"{DATA_ROOT}/sir4_{d}_test_v16sc/raw"
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(f"{dst}/documents.json"):
        shutil.copy(f"{DATA_ROOT}/sir4_{d}_test/raw/documents.json", f"{dst}/documents.json")

# Operator components for every TARGET domain's test split. Restored from Drive if
# the training run cached them, computed otherwise.
for d in DOMS:
    g = f"sir4_{d}_test_v16sc"
    npz = f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
    if os.path.exists(npz):
        print(f"{d:9} components present"); continue
    cached = f"{DRIVE}/outputs/sir4_{d}/cache/{g}_operator_components{OP_SLUG}.npz"
    if os.path.exists(cached):
        shutil.copy(cached, npz); print(f"{d:9} components restored from Drive"); continue
    # WARN, do not assert. A domain that has not been retrained yet has no cached
    # components, and computing them here would need the Qwen3 model and ~15
    # minutes to produce something for a checkpoint that does not exist. The
    # matrix cell's own assertion is the real gate; the comparison cell below
    # simply reports such a domain's operator and qwen3 arms as unavailable.
    print(f"{d:9} components NOT cached (domain not retrained yet?) -- "
          f"operator/qwen3 arms will be skipped for it")

# BGE predictions define score_sir4's similar/dissimilar axis.
for d in DOMS:
    p = f"{S4}/data/predictions_bge_sir4_{d}_test.json"
    if os.path.exists(p): continue
    for src in (f"{DRIVE}/outputs/sir4_{d}/cache/predictions_bge_sir4_{d}_test.json",
                f"{DRIVE}/outputs/sir4_{d}/predictions_bge_test.json"):
        if os.path.exists(src):
            os.makedirs(os.path.dirname(p), exist_ok=True); shutil.copy(src, p); break
    print(f"{d:9} BGE preds", "ok" if os.path.exists(p) else "MISSING (slices will be skipped)")
'''

CKPT = '''# One checkpoint per domain. Reject any run whose directory name says it is not
# the additive-gate fusion: routed / leverD / lossv2 / nodistill / smoke are
# different objectives and would make the matrix compare apples to oranges.
BANNED = ("smoke", "routed", "leverd", "lossv2", "nodistill")

# ENCODER ERA. A checkpoint learned its operator scalars (w0,w1,w2,beta) and its
# gate against ONE encoder's dense/S/M distributions. Feeding it the other
# encoder's components measures encoder mismatch, not domain transfer -- and it
# does so silently, because nothing about the tensor shapes changes.
#
# Picking by mtime alone is not enough: a domain that has not been retrained yet
# still has an old BGE model_best.pth sitting there, and it would take its place
# in the matrix looking exactly like the valid rows. So when OP_SLUG selects
# Qwen3, require the run directory to say qwenop; when it selects BGE, require
# that it does not.
REQUIRE = "qwenop" if OP_SLUG else None
def era_ok(run):
    return ("qwenop" in run.lower()) if REQUIRE else ("qwenop" not in run.lower())

# FINISHED, NOT MERELY PRESENT. model_best.pth is REWRITTEN every time an epoch
# improves, so a training run that is still going has one on disk right now and
# its mtime is the newest of any run. Picking it would silently freeze a
# mid-training snapshot into the matrix. Every completed training notebook writes
# scores.json at the end, so its presence is the completion marker.
def finished(run_dir):
    return os.path.exists(f"{run_dir}/scores.json")

CKPTS, wrong_era, in_flight = {}, {}, {}
for d in DOMS:
    hits = sorted(glob.glob(f"{DRIVE}/outputs/sir4_{d}/*/model_best.pth"),
                  key=os.path.getmtime, reverse=True)
    hits = [h for h in hits
            if not any(b in os.path.basename(os.path.dirname(h)).lower() for b in BANNED)]
    era_hits = [h for h in hits if era_ok(os.path.basename(os.path.dirname(h)))]
    ok = [h for h in era_hits if finished(os.path.dirname(h))]
    CKPTS[d] = ok[0] if ok else None
    if not ok and era_hits:                   # right era, still training or crashed
        in_flight[d] = os.path.basename(os.path.dirname(era_hits[0]))
    elif not era_hits and hits:               # exists, but from the other encoder era
        wrong_era[d] = os.path.basename(os.path.dirname(hits[0]))
    when = (time.strftime("%d %b %H:%M", time.localtime(os.path.getmtime(CKPTS[d])))
            if CKPTS[d] else "")
    print(f"{d:9} {'FOUND' if CKPTS[d] else 'not usable':10} {when}  "
          f"{os.path.basename(os.path.dirname(CKPTS[d])) if CKPTS[d] else ''}")

era = "Qwen3 (qwenop)" if REQUIRE else "BGE"
missing = [d for d in DOMS if not CKPTS[d]]
if wrong_era:
    print(f"\\nrejected, wrong encoder era (need {era}):")
    for d, r in wrong_era.items():
        print(f"  {d:9} newest usable run is {r}")
if in_flight:
    print("\\nrejected, no scores.json so the run has not finished:")
    for d, r in in_flight.items():
        print(f"  {d:9} {r}")
assert not missing, (
    f"no finished {era} checkpoint for {missing}. Wait for those runs, or drop "
    f"them with --domains. Do NOT mix eras or use a mid-training model_best.pth: "
    f"the matrix would look complete and be wrong.")
print(f"\\nall {len(DOMS)} checkpoints are finished and {era}-era: ok")
'''

RUN = '''# THE MATRIX. rows = the domain a checkpoint was TRAINED on,
#             cols = the domain it is EVALUATED on.
# Diagonal cells reuse the predictions the training run already wrote: identical
# code path, so recomputing them would burn GPU to reproduce the same file.
EVAL_BS = 2          # 8 OOMs on the big graphs; batch size cannot change a ranking

# CACHE ON THE INPUTS, NOT THE FILENAME. This used to be `if os.path.exists(dst)`,
# which meant retraining a domain and re-running the matrix printed "cached" and
# returned the OLD checkpoint's predictions. Nothing downstream could tell.
def stamp_of(*paths):
    return "|".join(f"{os.path.basename(p)}@{int(os.path.getmtime(p))}"
                    for p in paths if os.path.exists(p))

def fresh(dst, stamp):
    s = dst + ".src"
    return os.path.exists(dst) and os.path.exists(s) and open(s).read() == stamp

def seal(dst, stamp):
    open(dst + ".src", "w").write(stamp)
    return dst

def predict(train_dom, eval_dom):
    g   = f"sir4_{eval_dom}_test_v16sc"
    run = f"{OUT_M}/{train_dom}__on__{eval_dom}"
    dst = f"{run}/predictions_{g}.json"
    npz = f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
    # The checkpoint AND the components: either changing changes the ranking.
    stamp = stamp_of(CKPTS[train_dom], npz)
    if fresh(dst, stamp):
        print(f"  {train_dom} -> {eval_dom}: cached"); return dst
    if os.path.exists(dst):
        print(f"  {train_dom} -> {eval_dom}: STALE (checkpoint or components "
              f"changed since it was written), recomputing")
    if train_dom == eval_dom:                      # reuse the training run's own file
        for c in sorted(glob.glob(f"{DRIVE}/outputs/sir4_{eval_dom}/*/predictions_{g}.json"),
                        key=os.path.getmtime, reverse=True):
            if not any(b in c.lower() for b in ("smoke", "routed", "lossv2", "nodistill")):
                os.makedirs(run, exist_ok=True); shutil.copy(c, dst)
                print(f"  {train_dom} -> {eval_dom}: reused {os.path.basename(os.path.dirname(c))}")
                return seal(dst, stamp)
    os.makedirs(run, exist_ok=True)
    cmd = ("python -u -m gfmrag.workflow.sft_training "
           "--config-path config/gfm_reasoner --config-name sft_training_fusion "
           "text_emb_model=qwen3_st "
           f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
           f"datasets.train_names=[{g}] datasets.valid_names=[{g}] "
           "trainer.args.do_train=false trainer.args.do_eval=false "
           f"+trainer.args.eval_batch_size={EVAL_BS} "
           "+trainer.args.do_predict=true +trainer.args.predict_top_k=300 "
           f"+trainer.args.resume_from_checkpoint={CKPTS[train_dom]} "
           f"hydra.run.dir={run}")
    rc = sh(cmd, "/content/gfm-rag", extra=dict(
        WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        OPERATOR_COMPONENTS=npz, OPERATOR_COMPONENTS_TEST=npz,
        FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0"))
    assert rc == 0, f"{train_dom} -> {eval_dom} failed, exit {rc} (-9 = host OOM)"
    return seal(dst, stamp)

PRED = {}
for te in DOMS:
    print(f"\\n===== evaluating on {te} =====")
    for tr in DOMS:
        PRED[(tr, te)] = predict(tr, te)
'''

SCORE = '''# Score every cell with the one scorer, so the matrix is internally comparable.
def sets_path(d):
    tag = "cs_test_final" if d == "cs" else f"{d}_test_low"
    return f"{CARGO_ROOT}/quartet/data/benchmark/{tag}/sets.json"

COLS = "mrr,ndcg@5,recall@3,recall@5,completeset@5"
SC = {}
for (tr, te), pred in PRED.items():
    js = f"{OUT_M}/{tr}__on__{te}_scores.json"
    # Keyed on the predictions file, which is itself keyed on the checkpoint. A
    # filename-only check here would re-serve the previous checkpoint's scores
    # even after predict() correctly recomputed the ranking.
    stamp = stamp_of(pred)
    if not fresh(js, stamp):
        if os.path.exists(js):
            print(f"  {tr}->{te}: predictions changed, re-scoring")
        q   = f"{DATA_ROOT}/sir4_{te}_test/raw/test.json"
        bge = f"{S4}/data/predictions_bge_sir4_{te}_test.json"
        args = f"--pred {pred} --queries {q} --cols {COLS}"
        if os.path.exists(sets_path(te)): args += f" --sets {sets_path(te)}"
        if os.path.exists(bge):           args += f" --bge {bge}"
        rc = sh(f"python3 -u eval/score_sir4.py {args} --name '{tr}->{te}' "
                f"--json-out {js} --per-query-out {OUT_M}/{tr}__on__{te}_perquery.json", S4)
        assert rc == 0, f"scoring {tr}->{te} failed"
        seal(js, stamp)
    SC[(tr, te)] = json.load(open(js))
print("scored", len(SC), "cells")
'''

TABLE = '''# The matrix, one table per metric. Diagonal is in-domain; everything else is
# zero-shot. The `mean off-diag` column is the number that says whether a
# checkpoint generalises, and it is the one to quote.
MET = [("mrr","MRR"),("ndcg@5","nDCG@5"),("recall@3","R@3"),
       ("recall@5","R@5"),("completeset@5","CGS@5")]
lines = []
def out(s=""):
    print(s); lines.append(s)

for key, lab in MET:
    out(f"### {lab}   (rows = trained on, cols = evaluated on)")
    out("| trained \\\\ eval | " + " | ".join(DOMS) + " | mean off-diag |")
    out("|---|" + "--:|" * (len(DOMS) + 1))
    for tr in DOMS:
        vals, off = [], []
        for te in DOMS:
            v = SC[(tr, te)].get("all", {}).get(key, float("nan"))
            vals.append(f"*{v:.3f}*" if tr == te else f"{v:.3f}")
            if tr != te: off.append(v)
        out(f"| **{tr}** | " + " | ".join(vals)
            + f" | {sum(off)/len(off):.3f} |")
    # how much is lost by leaving the training domain, per target column
    out("| | " + " | ".join("" for _ in DOMS) + " | |")
    drops = []
    for te in DOMS:
        ind = SC[(te, te)]["all"][key]
        oth = [SC[(tr, te)]["all"][key] for tr in DOMS if tr != te]
        drops.append(f"{(sum(oth)/len(oth)) - ind:+.3f}")
    out("| *transfer gap* | " + " | ".join(drops) + " | |")
    out()
out("*italic* = in-domain. `transfer gap` = mean off-diagonal minus in-domain for "
    "that test set; near zero means a checkpoint from another domain is almost as "
    "good, which is the generalisation claim.")
open(f"{OUT_M}/matrix.md", "w").write("\\n".join(lines))
print(f"\\nwrote {OUT_M}/matrix.md")
'''


PATCH = '''# EVERY cell in this notebook resumes from a checkpoint, a path the training runs
# never take. base_trainer._load_checkpoint is reached from _setup_model(), which
# __init__ calls BEFORE it builds the AMP scaler, so `self.scaler` does not exist
# yet and the resume dies with a bare AttributeError. The optimizer load three
# lines above already guards with hasattr; the scaler line was missed.
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
    ap.add_argument("--from-domain", default="biology")
    ap.add_argument("--domains", default="cs,biology,physics,matsci")
    ap.add_argument("--op-slug", default="_content-qwen3",
                    help='"" for BGE-trained checkpoints, "_content-qwen3" after a Qwen3 retrain')
    ap.add_argument("--out", default=f"{ROOT}/colab_sir4_transfer_matrix.ipynb")
    a = ap.parse_args()
    doms = [d.strip() for d in a.domains.split(",")]

    src = f"{ROOT}/colab_train_sir4_{a.from_domain}_fusion.ipynb"
    if not os.path.exists(src):
        return print(f"missing {src}") or 1
    nb = json.load(open(src))
    cells = [json.loads(json.dumps(c)) for c in nb["cells"][:SETUP_CELLS]]

    # Retarget: this notebook owns no single dataset, and the cloned isolation guard
    # and bundle-unpack logic are per-dataset. Neutralise both -- cell 5 below does
    # its own unpacking for all four domains.
    keep = []
    for i, c in enumerate(cells):
        s = "".join(c["source"])
        if "need = [" in s or "z.extractall(CARGO_ROOT)" in s:
            continue                      # per-dataset guard / unpack, replaced below
        s = s.replace('DATASET = "sir4_%s"' % a.from_domain, 'DATASET = "sir4_cs"  # unused here')
        c["source"] = s.splitlines(True)
        keep.append(c)
    cells = keep

    cells[0] = md("# SIR-4 cross-domain transfer matrix\n\n"
                  "Every checkpoint on every domain's test set. The diagonal is "
                  "in-domain; the twelve off-diagonal cells are **zero-shot**.\n\n"
                  "No new data, no extraction, no training. `relations.csv` is "
                  "byte-identical across all four graphs, so a checkpoint's relation "
                  "embeddings are meaningful on any of them.\n\n"
                  "**Read the `transfer gap` row.** Near zero means a checkpoint from "
                  "another domain is almost as good as the native one, which is the "
                  "generalisation claim. Large negative means the skill is "
                  "domain-specific.")
    cells += [
        md("## Helpers and paths"), code(HELPERS.replace("__DOMS__", repr(doms))
                                         .replace("__OPSLUG__", a.op_slug)),
        md("## Unpack all four domains, restore components and BGE baselines"), code(UNPACK),
        # BEFORE the matrix machinery, deliberately. This cell needs no GPU, no
        # engine and no checkpoints, so it produces a table for whatever domains
        # have finished while the matrix cells below are still blocked waiting for
        # the rest. Putting it last would make the two arrive together.
        md("## Baselines vs the trained fusion\n\n"
           "BM25, BGE, Qwen3, operator and fusion for every domain, all through one "
           "`score_sir4.py` call. No GPU: the Qwen3 and operator arms are "
           "reconstructed from the cached components, and BM25 is lexical. Any "
           "domain still training simply reports its missing arms."),
        code(open(f"{ROOT}/eval/compare_arms.py").read()),
        md("## Patch the resume path in the engine"), code(PATCH),
        md("## Locate one checkpoint per domain"), code(CKPT),
        md("## Run the matrix (12 zero-shot cells; the diagonal is reused)"), code(RUN),
        md("## Score every cell"), code(SCORE),
        md("## The matrix"), code(TABLE),
        # After the matrix, because it reads the matrix's own score files. The
        # matrix says how every pair scores; this says which SOURCE is worth
        # training on, and whether any of them beat not training at all.
        md("## Which source transfers best, against the untrained floor\n\n"
           "The untrained arms do not appear as rows in the matrix because they "
           "do not transfer: they score the same on a target whatever the source. "
           "That makes them the floor every off-diagonal cell has to clear."),
        code(open(f"{ROOT}/eval/transfer_ranking.py").read()),
    ]

    broken = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        outl, mag = [], False
        for ln in "".join(c["source"]).split("\n"):
            if ln.lstrip().startswith(("!", "%")) or mag:
                mag = ln.rstrip().endswith("\\"); outl.append("pass")
            else:
                outl.append(ln)
        try:
            ast.parse("\n".join(outl))
        except SyntaxError as e:
            broken.append((i, e.lineno, e.msg))
    if broken:
        for i, l, m in broken:
            print(f"  cell {i} line {l}: {m}")
        raise SystemExit(f"{len(broken)} cell(s) do not parse; {a.out} not written")

    json.dump({"cells": cells, "metadata": nb.get("metadata", {}),
               "nbformat": 4, "nbformat_minor": 0}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({len(cells)} cells, domains {doms}, "
          f"op slug {a.op_slug or '(BGE)'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
