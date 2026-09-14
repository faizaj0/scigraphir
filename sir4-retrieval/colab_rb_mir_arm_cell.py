# MIR-trained SciGraphIR, zero-shot on ResearchBench, in ONE cell from a fresh runtime.
# Replays the setup cells of the colab_rb_zeroshot notebook you already have in Drive (paths, bundle,
# Qwen3, engine, fusion sources, patches, scorer helper, ResearchBench components, predict helper),
# then runs the MIR arm (one predict pass over 20,322 docs, ~10 min) and rebuilds the results table.
import os, json, glob, shutil, time
import sys
# Colab now runs Python 3.13 and the engine's editable install refuses it, so subprocesses must be
# told where gfmrag is and which interpreter to use. Set BEFORE the replayed paths cell, which
# snapshots os.environ for its sh() helper.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
os.environ["PYTHONPATH"] = "/content/gfm-rag" + os.pathsep + os.environ.get("PYTHONPATH", "")
from google.colab import drive
drive.mount('/content/drive')
NB_PATH = ""   # optional: full path of the colab_rb_zeroshot*.ipynb to replay; empty = first match
_c = sorted(glob.glob("/content/drive/MyDrive/**/colab_rb_zeroshot*.ipynb", recursive=True),
            key=lambda p: (not p.endswith("colab_rb_zeroshot.ipynb"), "standard" not in p, p))
NB_PATH = NB_PATH or (_c[0] if _c else "")
assert NB_PATH, "no colab_rb_zeroshot*.ipynb under /content/drive/MyDrive; set NB_PATH"
print("replaying setup cells from", NB_PATH)
_code = [''.join(c["source"]) for c in json.load(open(NB_PATH))["cells"] if c["cell_type"] == "code"]
# Colab saves pasted cells into the notebook; a pasted replay cell mentions every marker below, so drop it.
_code = [s for s in _code if "replaying setup cells from" not in s and "def _pick(" not in s]
def _pick(pred, which="first", optional=False):
    hits = [s for s in _code if pred(s)]
    if not hits and optional: return None
    assert hits, "a setup cell is missing from this notebook; use a newer build"
    return hits[0] if which == "first" else hits[-1]
def _run(label, src, fatal=True):
    if src is None: print(f"[skip] {label}: not in this notebook"); return
    print(f"\n======== replay: {label} ========")
    r = get_ipython().run_cell(src, store_history=False)
    if not r.success and not fatal:
        print(f"[warn] setup cell '{label}' failed; continuing (its check is advisory here)"); return
    assert r.success, f"setup cell '{label}' failed; see the traceback above"
_run("paths",            _pick(lambda s: s.startswith("!nvidia-smi")))
_run("unpack",           _pick(lambda s: s.startswith("# 2. Unpack ResearchBench")))
_run("qwen3",            _pick(lambda s: s.startswith("import os, shutil") and "qwen3" in s.lower()))
ARMS, PRED = [], {}      # section 3 (baselines) is not replayed; the scorer cell's loop over them becomes a no-op
_run("score helper",     _pick(lambda s: s.startswith("# 4. Score every baseline")))      # defines score(); baselines re-score from Drive in seconds
_run("engine",           _pick(lambda s: s.startswith("import os, sys, torch") and "gfm-rag-adapted.zip" in s))
_run("pins",             _pick(lambda s: '_im.version("wandb")' in s or 'wandb.__version__.startswith("0.18.")' in s, optional=True), fatal=False)
_run("fusion sources",   _pick(lambda s: s.startswith("# === write the CARGO-fusion files")))
_run("config defaults",  _pick(lambda s: s.startswith("# Cell 3a is reused VERBATIM")))
_run("PyG version fix",  _pick(lambda s: s.startswith("# The ULTRA layers vendored"), "last"))
_run("torchvision shim", _pick(lambda s: s.startswith("# THE TRAINING SUBPROCESS IS A FRESH PYTHON"), "last"))
_run("diagnostics patch",_pick(lambda s: s.startswith("STF = ")))
_run("CCMP_LR patch",    _pick(lambda s: s.startswith("# Idempotent: re-running is a no-op")))
_run("RB components",    _pick(lambda s: s.startswith("# 6. ResearchBench inputs for the fusion")))
_run("predict helper",   _pick(lambda s: s.startswith("# 7a. Zero-shot prediction helper")))
# the scorer also wants the BGE slice file that section 3b built; it is on Drive from the earlier run
BGE_SLICE = f"{OUT_ROOT}/slice/predictions_bge_slice_{DATASET}_test.json"
assert os.path.exists(BGE_SLICE), f"missing {BGE_SLICE}: run section 3b of the ResearchBench notebook once"
# the MIR arm itself (embedded, so an older notebook build without section 7d still works)
_run("MIR arm", r'''# 7d. Arm A3 (optional): SciGraphIR trained on MIR (Methodology Inspiration Retrieval, ACL
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
''')
_run("results table",    _pick(lambda s: s.startswith("# 9. The table")))
_run("per-discipline + bootstrap", _pick(lambda s: s.startswith("# 9b."), optional=True))
