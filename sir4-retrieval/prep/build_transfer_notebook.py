"""
build_transfer_notebook.py -- the ResearchBench transfer notebook.

CLONES ITS SETUP FROM A GENERATED SIR-4 NOTEBOOK rather than restating it. Cells 0-15
of those notebooks (GPU check, isolation guard, engine install, the fusion source
patch, the PyG version fix, Qwen3, bundle unpack, cache save/restore) took sixteen
separate fixes to get right. Re-typing them here would fork that work and the fork
would rot.

CELLS 16 ONWARD ARE DELIBERATELY NOT REUSED. Cell 17 of the training notebooks runs
cargo_operator.py, which FITS w and beta on the corpus and patches them into
fusion_reasoner.py as a warm start. Doing that on ResearchBench would tune the model
on the benchmark this experiment claims never to train on. This notebook computes the
operator COMPONENTS (dense/S/M/total_S, which are corpus statistics, not parameters)
and takes w and beta from each frozen checkpoint.

Usage
-----
    python3 prep/build_transfer_notebook.py
    python3 prep/build_transfer_notebook.py --from-domain biology
"""
from __future__ import annotations

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SETUP_CELLS = 16          # 0..15 inclusive; 16 is the "5b Phase 3" header


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(True)}


HELPERS = '''# Shell helper with live output. Same reader as the training notebooks: raw chunks,
# not lines, because tqdm redraws with a carriage return and a line-buffered reader
# stays silent for the whole bar.
import os, sys, time, subprocess, glob, json, shutil
KGDIR = f"{CARGO_ROOT}/kg-construction"
env = dict(os.environ, CARGO_ROOT=CARGO_ROOT, CARGO_DATASET=DATASET, PYTHONUNBUFFERED="1")

def sh(cmd, cwd, extra=None):
    t0 = time.time()
    e = dict(env, **(extra or {}))
    p = subprocess.Popen(cmd, cwd=cwd, env=e, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        c = os.read(p.stdout.fileno(), 8192)
        if not c: break
        sys.stdout.write(c.decode("utf-8", "replace")); sys.stdout.flush()
    p.wait(); print(f"\\n[{time.time()-t0:.0f}s, exit {p.returncode}]")
    return p.returncode

GRAPH  = f"{DATASET}_test_v16sc"
NPZ    = f"{DATA_ROOT}/{GRAPH}/operator_components.npz"
SETS   = f"{DATA_ROOT}/{DATASET}_test/sets.json"
QUERIES= f"{DATA_ROOT}/{DATASET}_test/raw/test.json"
print("graph  ", GRAPH)
print("queries", QUERIES)

# CODE OVERLAY. The bundle is 65 MB and rebuilt rarely, so the scripts inside it go
# stale against the repo. Re-uploading the whole zip to change one scorer is the
# wrong trade. Anything placed at {DRIVE}/code_overlay/<repo-relative path> is copied
# over the unpacked bundle and wins, e.g.
#     MyDrive/cargo-gfmrag/code_overlay/sir4-retrieval/eval/score_sir4.py
_ov = f"{DRIVE}/code_overlay"
if os.path.isdir(_ov):
    _n = 0
    for _dp, _, _fs in os.walk(_ov):
        for _f in _fs:
            _src = os.path.join(_dp, _f)
            _dst = os.path.join(CARGO_ROOT, os.path.relpath(_src, _ov))
            os.makedirs(os.path.dirname(_dst), exist_ok=True)
            shutil.copy(_src, _dst); _n += 1
            print("  overlay:", os.path.relpath(_src, _ov))
    print(f"applied {_n} overlay file(s)")
else:
    print("no code_overlay on Drive (fine; the bundle's own copies are used)")

# Fail on the missing FLAG, not on a bare exit code 2 from argparse.
_sc = open(f"{S4}/eval/score_sir4.py").read()
_need = ["--per-query-out", "--gold-strata", "--cols"]
_miss = [f for f in _need if f not in _sc]
assert not _miss, (f"score_sir4.py on this runtime lacks {_miss}. It came from a bundle "
                   f"predating those flags. Upload the current file to "
                   f"{DRIVE}/code_overlay/sir4-retrieval/eval/score_sir4.py and re-run.")
print("scorer supports:", ", ".join(_need))
'''

STEP45 = '''# Steps 4 and 5 from the plan. Both encode 20,322 documents, which is why they are
# here and not on the laptop.
#
# NOTE WHAT IS *NOT* RUN: cargo_operator.py. That script FITS w and beta. Fitting
# them on ResearchBench would be tuning on the evaluation benchmark. Only the
# components are computed here -- dense/S/M/total_S are corpus statistics, and every
# checkpoint brings its own learned w and beta.

# 4. operator components, ordered to the master graph's document nodes
if os.path.exists(NPZ):
    print("operator components already built")
else:
    sh(f"python3 -u experiments/probe_greasoner/precompute_operator_components.py "
       f"--graph {GRAPH} --split test", KGDIR)

# 5. BGE baseline. A reported arm, and it defines score_sir4's similar/dissimilar axis.
BGE_PRED = f"{S4}/data/predictions_bge_{DATASET}_test.json"
if os.path.exists(BGE_PRED):
    print("BGE baseline already built")
else:
    sh(f"python3 -u eval/bge_sir4.py --dataset {DATASET} --split test --topk 300", S4)

# 6. audit. Exits non-zero on dangling edges or zero-seed queries.
sh(f"python3 -u eval/audit_graph.py --dataset {DATASET} --split test --sample 40", S4)

# sets.json for CompleteGoldSet@k (our interpretation of RB golds as one set)
if not os.path.exists(SETS):
    sh("python3 -u transfer/make_rb_sets.py", S4)
'''

STRICT = '''# strict=False HIDES A SILENT FAILURE MODE.
# base_trainer._load_checkpoint calls load_state_dict(state["model"], strict=False)
# and DISCARDS the returned _IncompatibleKeys. If a checkpoint's gate or operator
# scalars fail to load -- different key names, a shape change, a partial save -- the
# run continues with randomly initialised weights for those tensors and reports a
# perfectly normal-looking number. On a transfer experiment that is indistinguishable
# from "the other benchmark trained a worse model".
#
# Patch it to report. The plan's validity check "checkpoint loading reports no
# unexplained missing parameters" is not satisfiable without this.
BT = "/content/gfm-rag/gfmrag/trainers/base_trainer.py"
src = open(BT).read()
OLD = 'self.model.load_state_dict(state["model"], strict=False)'
NEW = ('_inc = self.model.load_state_dict(state["model"], strict=False)\\n'
       '                logger.info(f"checkpoint keys: {len(_inc.missing_keys)} missing, "\\n'
       '                            f"{len(_inc.unexpected_keys)} unexpected")\\n'
       '                for _k in _inc.missing_keys[:20]:    logger.warning(f"  MISSING    {_k}")\\n'
       '                for _k in _inc.unexpected_keys[:20]: logger.warning(f"  UNEXPECTED {_k}")')
if "_inc = self.model.load_state_dict" in src:
    print("already patched: key-mismatch reporting")
elif OLD in src:
    src = src.replace(OLD, NEW, 1); print("patched: checkpoint load now reports key mismatches")
else:
    raise SystemExit("could not find the load_state_dict call -- engine changed, check by hand")

# SECOND PATCH: the resume path crashes before it can be used.
#   _load_checkpoint:  if "scaler" in state: self.scaler.load_state_dict(...)
# is called from _setup_model(), which runs BEFORE the AMP scaler is created, so
# `self.scaler` does not exist yet and resuming dies with a bare AttributeError.
# The optimizer load three lines above already guards with hasattr; the scaler line
# was simply missed, and nothing caught it because every run so far trained from
# scratch. Predict-only inference is the first thing to take this path.
OLD_S = 'if "scaler" in state:'
NEW_S = 'if "scaler" in state and hasattr(self, "scaler"):'
if NEW_S in src:
    print("already patched: scaler guard")
elif OLD_S in src:
    src = src.replace(OLD_S, NEW_S, 1); print("patched: scaler load is now guarded")
else:
    raise SystemExit("could not find the scaler load -- engine changed, check by hand")

open(BT, "w").write(src)
'''

CKPT = '''# Locate the frozen checkpoints on Drive. Nothing is trained in this notebook.
def find_ckpt(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None

ARMS = {
    # name                  Drive glob
    "sir4_biology": f"{DRIVE}/outputs/sir4_biology/*/model_best.pth",
    # PINNED, NOT GLOBBED. outputs/v16sc holds NINE runs and only this one is the
    # additive-gate fusion with the operator-hard-negative objective that every SIR-4
    # checkpoint was trained with. The others -- routed_leverD, routed_lossv2,
    # nodistill, plus three smoke runs -- are different objectives. A glob sorted by
    # mtime could silently pick one of those, and the experiment would then attribute
    # an objective difference to the benchmark. That is the most damaging thing that
    # could quietly go wrong here, and nothing downstream would reveal it.
    "tomato":       f"{DRIVE}/outputs/v16sc/v16sc_fusion_epoch20/model_best.pth",
    "sir4_cs":      f"{DRIVE}/outputs/sir4_cs/*/model_best.pth",
    "sir4_physics": f"{DRIVE}/outputs/sir4_physics/*/model_best.pth",
    "sir4_matsci":  f"{DRIVE}/outputs/sir4_matsci/*/model_best.pth",
    # The multi-domain arm: one checkpoint trained on physics AND biology together.
    # This is the foundation-model claim in its weakest testable form -- if joint
    # supervision transfers to a corpus neither domain came from, that is evidence
    # single-domain training cannot give. Absent until colab_sir4_joint_pb.ipynb
    # has run, and simply skipped rather than fatal.
    "joint_pb":     f"{DRIVE}/outputs/_joint_pb/*/model_best.pth",
    # Trained on ALL FOUR SIR-4 domains. Nothing is held out inside SIR-4, so
    # ResearchBench is the only zero-shot evidence this arm has -- which is
    # exactly what makes it the foundation-model configuration.
    "joint_all":    f"{DRIVE}/outputs/_joint_all/*/model_best.pth",
}
BANNED = ("smoke", "routed", "leverd", "lossv2", "nodistill")
CKPTS = {}
for name, pat in ARMS.items():
    p = find_ckpt(pat)
    # Belt and braces: even a pinned path is worth checking, because the SIR-4 globs
    # are still globs and a stray smoke run in one of those directories would be
    # picked up the same way.
    if p and any(b in os.path.basename(os.path.dirname(p)).lower() for b in BANNED):
        raise SystemExit(f"{name} resolved to {p} -- that run is not the "
                         f"additive-gate fusion. Pin the exact directory for this arm.")
    CKPTS[name] = p
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p))) if p else ""
    print(f"{name:15} {'FOUND' if p else 'not found':10} {when}  {p or pat}")

# PRIMARY is biology because TOMATO-Star is 89.5% biomedical (measured over its 7,000
# training documents), so matching the source discipline keeps the comparison about
# the benchmark rather than the field.
PRIMARY = ["sir4_biology", "tomato"]
ROBUST  = ["sir4_cs", "sir4_physics", "sir4_matsci", "joint_pb", "joint_all"]
ROBUST  = [k for k in ROBUST if CKPTS.get(k)]     # joint_pb only once it exists
assert all(CKPTS.get(k) for k in PRIMARY), "the two primary checkpoints must be present"
'''

INFER = '''# FULL-CORPUS PROTOCOL: rank all 20,322 papers. No candidate masking.
#
# This is the protocol no benchmark in the MOOSE/ResearchBench lineage measures -- all
# of them score reranking of a pre-filtered pool of 15 or 75. It needs no new
# inference code, only do_train=false plus resume_from_checkpoint, so it is the first
# result available. The candidate-induced 75 protocol comes later and needs the
# per-query subgraph machinery.
OUT_T = f"{OUT_ROOT}/transfer"; os.makedirs(OUT_T, exist_ok=True)

def predict(name, ckpt, top_k=300, eval_bs=2):
    run_dir = f"{OUT_T}/{name}"; os.makedirs(run_dir, exist_ok=True)
    done = f"{run_dir}/predictions_{GRAPH}.json"
    if os.path.exists(done):
        print(f"{name}: already predicted"); return done
    cmd = ("python -u -m gfmrag.workflow.sft_training "
           "--config-path config/gfm_reasoner --config-name sft_training_fusion "
           "text_emb_model=qwen3_st "
           f"datasets.cfgs.root={DATA_ROOT} datasets.cfgs.force_reload=False "
           f"datasets.train_names=[{GRAPH}] datasets.valid_names=[{GRAPH}] "
           "trainer.args.do_train=false trainer.args.do_eval=false "
           # eval_batch_size defaults to 8, which OOMs: ~9.2 GiB per query
           # over this graph. Batch size changes nothing about the ranking
           # (no gradient, no in-batch interaction), so this is a pure
           # memory dial and not a deviation worth recording.
           # The leading + is required: eval_batch_size is not declared in
           # sft_training_fusion.yaml's trainer.args, and Hydra refuses to
           # override a key that is not already in the struct.
           f"+trainer.args.eval_batch_size={eval_bs} "
           "+trainer.args.do_predict=true "
           f"+trainer.args.predict_top_k={top_k} "
           f"+trainer.args.resume_from_checkpoint={ckpt} "
           f"hydra.run.dir={run_dir}")
    # Both point at the SAME npz: there is one ResearchBench corpus and it is the
    # evaluation set. train_names is set to it only because the loader wants a value.
    rc = sh(cmd, "/content/gfm-rag", extra=dict(
        WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        OPERATOR_COMPONENTS=NPZ, OPERATOR_COMPONENTS_TEST=NPZ,
        FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0"))
    assert rc == 0, f"{name} failed, exit {rc} (-9 = killed, out of memory)"
    return done

PREDS = {}
for name in PRIMARY + ROBUST:
    if CKPTS.get(name):
        print(f"\\n===== {name} =====")
        PREDS[name] = predict(name, CKPTS[name])
'''

SCORE = '''# Score every arm through score_sir4.py so standard MRR and the multi-gold
# metrics use one implementation throughout the comparison.
# Dump per-query rows for the paired test.
PREDS["bge"] = BGE_PRED
# Recall@3 and @15 are ResearchBench's own operating points (4% and 20%
# of a 75-candidate pool). The rest are for our analysis.
RB_COLS = ("mrr,mgrr,ndcg@5,hits@1,hits@5,recall@1,recall@3,recall@5,"
           "recall@10,recall@15,recall@20,completeset@5,completeset@10")
SC = {}
for name, pred in PREDS.items():
    if not os.path.exists(pred):
        print(f"{name}: no predictions, skipped"); continue
    js, pq = f"{OUT_T}/{name}_scores.json", f"{OUT_T}/{name}_perquery.json"
    sh(f"python3 -u eval/score_sir4.py --pred {pred} --queries {QUERIES} "
       f"--sets {SETS} --bge {BGE_PRED} --name '{name}' --gold-strata "
       f"--cols {RB_COLS} "
       f"--json-out {js} --per-query-out {pq}", S4)
    SC[name] = js
print("\\nscored:", list(SC))
'''

COMPARE = '''# The comparison. Paired, because every arm ranked the same queries.
OUT_T = f"{OUT_ROOT}/transfer"
sh(f"python3 -u transfer/paired_bootstrap.py "
   f"--a {OUT_T}/sir4_biology_perquery.json --a-name 'SIR-4 Biology' "
   f"--b {OUT_T}/tomato_perquery.json       --b-name 'TOMATO-Star' "
   f"--json-out {OUT_T}/paired_all.json", S4)

# Same test on the frozen strict zero-shot subset: 923 queries whose own text does
# not read biomedical. Excluding by RB discipline LABEL was the original plan and
# is wrong -- its buckets do not describe content, and 177 biomedical queries
# survive inside the label-based 990. The list is frozen in subsets.json.
sh(f"python3 -u transfer/paired_bootstrap.py "
   f"--a {OUT_T}/sir4_biology_perquery.json --a-name 'SIR-4 Biology' "
   f"--b {OUT_T}/tomato_perquery.json       --b-name 'TOMATO-Star' "
   f"--subset strict_zeroshot --json-out {OUT_T}/paired_strict.json", S4)
'''

TABLE = '''# All arms, all slices, one table. Read the saved per-query files directly,
# so a runtime reset never requires prediction or scoring to be repeated.
import json
OUT_T = f"{OUT_ROOT}/transfer"
MET = [("mrr","MRR"),("ndcg@5","nDCG@5"),("hits@1","H@1"),("hits@5","H@5"),
       ("recall@3","R@3"),("recall@5","R@5"),("recall@15","R@15"),
       ("completeset@5","CGS@5"),("completeset@10","CGS@10")]
ARM_ORDER = ["sir4_biology", "tomato", "joint_all", "joint_pb", "sir4_cs",
             "sir4_physics", "sir4_matsci", "bge", "qwen3"]
P = {}
for name in ARM_ORDER:
    path = f"{OUT_T}/{name}_perquery.json"
    if not os.path.exists(path):
        continue
    rows = json.load(open(path))
    legacy_mean, legacy_standard = 0, 0
    for qid, row in rows.items():
        if "mgrr" not in row and "mrr_best" in row:
            row["mgrr"] = row.get("mrr", 0.0)
            row["mrr"] = row["mrr_best"]
            legacy_mean += 1
        elif "mgrr" not in row and "mrr_best" not in row and "mrr" in row:
            legacy_standard += 1
        assert "mrr" in row, f"{path}: no MRR value for {qid}"
        if "mrr_best" in row:
            assert abs(row["mrr"] - row["mrr_best"]) < 1e-12, \
                f"{path}: ambiguous MRR semantics for {qid}"
    if legacy_mean:
        print(f"{name}: normalised standard MRR from mrr_best for "
              f"{legacy_mean:,} legacy rows")
    if legacy_standard:
        print(f"{name}: using existing standard mrr for "
              f"{legacy_standard:,} early-schema rows")
    P[name] = rows

lines = []
for sl in ("all","same","cross","similar","dissimilar"):
    rows = []
    for name, per_query in P.items():
        selected = [r for r in per_query.values() if sl in r.get("slices", ["all"])]
        if selected:
            rows.append((name, {"n": len(selected), **{
                key: sum(r[key] for r in selected) / len(selected) for key, _ in MET
            }}))
    if not rows:
        continue
    lines += [f"### {sl}   n={rows[0][1]['n']}",
              "| arm | " + " | ".join(l for _, l in MET) + " |",
              "|---|" + "--:|"*len(MET)]
    for n, m in rows:
        lines.append(f"| {n} | " + " | ".join(f"{m[k]:.4f}" if k in m else "-" for k, _ in MET) + " |")
    lines.append("")
print("\\n".join(lines))
open(f"{OUT_T}/results.md","w").write("\\n".join(lines))
print(f"\\nwrote {OUT_T}/results.md")
print("NOTE: CGS = CompleteGoldSet, our reading of RB golds as one accepted set.")
'''

DOMAIN_TABLE = '''# Query-level performance within each ResearchBench discipline.
# Keep this comparison fixed: it tests the primary SIR-4 Biology checkpoint against
# the matched TOMATO-Star checkpoint and the untuned dense baseline.
OUT_T = f"{OUT_ROOT}/transfer"
domain_md = f"{OUT_T}/results_by_domain.md"
domain_json = f"{OUT_T}/results_by_domain.json"
arms = (f"--arm 'BGE={OUT_T}/bge_perquery.json' "
        f"--arm 'SIR-4 Biology={OUT_T}/sir4_biology_perquery.json' "
        f"--arm 'TOMATO-Star={OUT_T}/tomato_perquery.json' ")
# Optional arms, added only when their files exist, so this cell never fails
# because an experiment upstream has not been run yet.
for lab, key in (("Qwen3-Embedding", "qwen3"),
                 ("Joint all-4", "joint_all"),
                 ("Joint Physics+Biology", "joint_pb")):
    if os.path.exists(f"{OUT_T}/{key}_perquery.json"):
        arms += f"--arm '{lab}={OUT_T}/{key}_perquery.json' "
    else:
        print(f"skipping {lab}: no {key}_perquery.json yet")
sh(f"python3 -u eval/report_domain_results.py {arms}"
   f"--md-out {domain_md} --json-out {domain_json}", S4)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-domain", default="biology",
                    help="which generated SIR-4 notebook to clone setup cells from")
    ap.add_argument("--out", default=f"{ROOT}/colab_rb_transfer.ipynb")
    a = ap.parse_args()

    source_notebook = f"{ROOT}/colab_train_sir4_{a.from_domain}_fusion.ipynb"
    if not os.path.exists(source_notebook):
        return print(f"missing {source_notebook}; "
                     f"run prep/build_notebook.py --domain {a.from_domain}") or 1
    nb = json.load(open(source_notebook))
    cells = [json.loads(json.dumps(c)) for c in nb["cells"][:SETUP_CELLS]]

    # Retarget the cloned cells at ResearchBench.
    src_ds = f"sir4_{a.from_domain}"
    for c in cells:
        s = "".join(c["source"])
        s = s.replace(f'DATASET = "{src_ds}"', 'DATASET = "researchbench"')

        # RESEARCHBENCH IS EVALUATION-ONLY: there is no train split and no train
        # graph. `TRAIN` appears in five of the cloned cells (the isolation guard,
        # the config rewrite, and three loops in save_cache/restore_cache), so
        # deleting the lines that mention it is a game of whack-a-mole -- the first
        # attempt filtered on the literal "_train" and missed
        # `data/{TRAIN}/processed/stage1/nodes.csv`, which failed the guard on
        # Colab. Point TRAIN at the test graph instead. Every `(TRAIN, TEST)` loop
        # then processes one graph twice, which is idempotent, and the config
        # rewrite sets train_names and valid_names to the same graph, which is
        # exactly what predict-only inference wants anyway.
        s = s.replace('TRAIN, TEST = f"{DATASET}_train_v16sc", f"{DATASET}_test_v16sc"',
                      'TRAIN = TEST = f"{DATASET}_test_v16sc"   # eval-only corpus: one graph')

        # Redefining TRAIN is necessary but NOT sufficient: cell 15 iterates the
        # splits as LITERAL strings and feeds them to cargo_paths, which resolves
        # cp.corpus_dir("train") to researchbench_train regardless of what TRAIN
        # holds. Those two loops are why the verification cell failed on Colab after
        # the guard was already fixed. Collapse them to the one split that exists.
        s = s.replace('for s in ("train", "test"):', 'for s in ("test",):')
        s = s.replace('for split, g in (("train", TRAIN), ("test", TEST)):',
                      'for split, g in (("test", TEST),):')

        # SIR-4's sets.json lives in the quartet export; ResearchBench's is written
        # by transfer/make_rb_sets.py next to the corpus. Left alone, cell 15 prints
        # "ABSENT" for a file that exists, which is a false alarm about the one
        # metric this experiment adds.
        s = re.sub(r'SETS = f"\{CARGO_ROOT\}/quartet/data/benchmark/[^"]*sets\.json"',
                   'SETS = f"{DATA_ROOT}/{DATASET}_test/sets.json"', s)

        # The two remaining train-side requirements name a raw corpus and a probe
        # cache that genuinely do not exist, so those lines do go.
        if "need = [" in s:
            s = "\n".join(l for l in s.split("\n") if "_train" not in l)
        c["source"] = s.splitlines(True)

    cells[0] = md("# ResearchBench transfer — frozen SIR-4 vs frozen TOMATO\n\n"
                  "Nothing is trained here. Two frozen checkpoints rank the same "
                  "ResearchBench queries, and the difference is tested pairwise.\n\n"
                  "Cells 1-5 are cloned from the SIR-4 training notebook so the engine "
                  "setup stays identical. **Cell 5b of that notebook is deliberately "
                  "absent**: it fits the operator's `w` and `beta`, and fitting them "
                  "here would tune the model on the evaluation benchmark.\n\n"
                  "See `transfer/PLAN.md`.")

    cells += [
        md("## 6. Helpers and paths"), code(HELPERS),
        md("## 7. Steps 4-6: operator components, BGE baseline, audit, sets.json\n\n"
           "The three encode-heavy steps, plus the graph audit."), code(STEP45),
        md("## 8. Make checkpoint loading report key mismatches"), code(STRICT),
        md("## 9. Locate the frozen checkpoints"), code(CKPT),
        md("## 10. Predict — full-corpus protocol (all 20,322 papers)"), code(INFER),
        md("## 11. Score every arm"), code(SCORE),
        md("## 12. Paired comparison"), code(COMPARE),
        md("## 12b. Qwen3-Embedding dense arm\n\n"
           "The second untuned dense baseline, and the `--` row in Table 7.5. "
           "Encoded here with the instruction pinned in the cell rather than taken "
           "from the bundle, so this row and the Qwen3 row in Table 7.3 use the "
           "same prompt."),
        code(open(f"{ROOT}/transfer/rb_qwen3_arm.py").read()),
        md("## 13. Results table"), code(TABLE),
        md("## 14. Performance by ResearchBench discipline"), code(DOMAIN_TABLE),
    ]

    # COMPILE EVERY CODE CELL BEFORE WRITING THE FILE.
    # A cell template is a Python string inside a Python file, so an escape that is
    # right in one layer is wrong in the other: `\n` in a non-raw template becomes a
    # LITERAL newline in the emitted cell, which inside an f-string is a syntax error.
    # That shipped once, and the only reason it was not caught is that this check
    # lived in a throwaway script instead of here. IPython magics (!cmd, %magic) are
    # not Python, so blank them before parsing rather than exempting whole cells.
    import ast
    broken = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        cell_source = "".join(c["source"])
        # A magic can span lines with a trailing backslash (`!pip install a \` ...).
        # Blanking only the first line leaves its continuations dangling, which parses
        # as an unexpected indent -- a false positive that would make this check
        # untrustworthy and therefore ignored.
        out_lines, in_magic = [], False
        for ln in cell_source.split("\n"):
            is_magic = ln.lstrip().startswith(("!", "%"))
            if is_magic or in_magic:
                in_magic = ln.rstrip().endswith("\\")
                out_lines.append("pass")
            else:
                out_lines.append(ln)
        stripped = "\n".join(out_lines)
        try:
            ast.parse(stripped)
        except SyntaxError as e:
            broken.append((i, e.lineno, e.msg,
                           (stripped.split("\n")[e.lineno - 1] if e.lineno else "")[:80]))
    if broken:
        for i, ln, msg, txt in broken:
            print(f"  cell {i} line {ln}: {msg}\n      {txt}")
        raise SystemExit(f"{len(broken)} generated cell(s) do not parse; not writing "
                         f"{a.out}. The previous file is left untouched.")

    out = {"cells": cells, "metadata": nb.get("metadata", {}),
           "nbformat": 4, "nbformat_minor": 0}
    json.dump(out, open(a.out, "w"), indent=1)

    # Guard the CLONED cells only. `sir4_biology` appears legitimately further down as
    # the name of a checkpoint arm and its path on Drive -- that is the whole point of
    # the experiment. What must not survive is a cloned setup cell still pointing its
    # corpus, graph or cache at the source domain.
    cloned = json.dumps(cells[:SETUP_CELLS])
    assert src_ds not in cloned, (
        f"a {src_ds} reference survived in the cloned setup cells; "
        f"the notebook would read the wrong corpus")
    # Catch the failure that actually happened rather than trusting the edit above.
    # The check is on RESOLVED names, not on the substring "_train": cell 9 legitimately
    # contains `.replace("tomato_train_v16sc", TRAIN)` -- that is the TOMATO literal
    # being replaced, not a path anything reads -- and the fusion_reasoner source blob
    # mentions it in a docstring. What must not survive is a path that resolves to a
    # researchbench train artefact, because that fails the isolation guard on Colab
    # minutes into a run.
    bad = [i for i, c in enumerate(cells[:SETUP_CELLS])
           if "researchbench_train" in "".join(c["source"])]
    assert not bad, (f"cells {bad} still resolve a researchbench train path; "
                     f"ResearchBench has no train split")
    assert 'TRAIN = TEST' in cloned, "the TRAIN/TEST redefinition did not apply"

    # THE CHECK THAT WOULD HAVE CAUGHT BOTH COLAB FAILURES. Redefining TRAIN fixes
    # every reference that goes through the variable; it does nothing for code that
    # iterates the splits as literal strings and hands them to cargo_paths, because
    # cp.corpus_dir("train") resolves from the literal, not from TRAIN. Grep the
    # cloned cells for a literal "train" used as a split value.
    import re as _re
    lit = _re.compile(r'''\(\s*["']train["']|["']train["']\s*,\s*["']test["']''')
    hits = [(i, ln.strip()[:70])
            for i, c in enumerate(cells[:SETUP_CELLS])
            for ln in "".join(c["source"]).split("\n") if lit.search(ln)]
    assert not hits, ("a literal 'train' split survives in the cloned cells; it will "
                      "resolve to a corpus that does not exist:\n  "
                      + "\n  ".join(f"cell {i}: {t}" for i, t in hits))
    print(f"wrote {a.out}")
    print(f"  {len(cells)} cells ({SETUP_CELLS} cloned from "
          f"{os.path.basename(source_notebook)}, "
          f"{len(cells)-SETUP_CELLS} transfer-specific)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
