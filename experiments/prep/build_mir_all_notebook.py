"""
build_mir_all_notebook.py -- every remaining MIR row in ONE Colab notebook.

Composes, from the three existing builders, one run with a single setup and one table:

    section 3    six dense baselines                 (build_mir_notebook, skip on manifest)
    section 3c   HyDE + MuGI                          (build_llm_expansion_notebook, ~$0.20)
    section 4    engine install, STOCK patches only
    section 5    G-Reasoner + GFM-RAG                 (build_baselines_notebook ALL-mode cells)
    section 6    CARGO fusion sources + patches
    section 7    multi-view scorer + component tables (restored from Drive)
    section 8    "+ Graph Reasoner (OpenIE graph)"    (build_mir_notebook --graph openie)
    section 9    the whole MIR table

ORDER IS THE POINT. The two graph baselines train in section 5, BEFORE the fusion sources are
written into the engine in section 6, so they run the stock engine exactly as the SIR-4
baseline rows did. Every block skips finished work (manifests, signatures, Drive run dirs),
so a disconnect costs only the run in flight.

Each harvested block keeps its own variable names; a short "bridge" cell before it defines
those names from this notebook's paths, and a "restore" cell after it undoes the one clash
(both harvested scorers rebind `SCORES`, which this notebook uses as the scores directory).

Usage
-----
    python3 prep/build_mir_all_notebook.py            # -> colab_mir_all.ipynb
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_ap = argparse.ArgumentParser()
_ap.add_argument("--epochs", type=int, default=10)
_ap.add_argument("--batch", type=int, default=2)
_ap.add_argument("--out", default=None)
_args = _ap.parse_args()

# The other builders parse sys.argv at import; hand each the arguments it needs, then restore.
_argv = sys.argv
sys.argv = ["build_baselines_notebook.py", "--dataset", "mir"]
import build_baselines_notebook as bb          # noqa: E402  (writes mir_baselines.ipynb, harmless)
sys.argv = ["build_llm_expansion_notebook.py", "--datasets", "mir"]
import build_llm_expansion_notebook as lx      # noqa: E402  (writes llm_expansion_baselines_mir.ipynb)
sys.argv = _argv
import build_mir_notebook as bm                # noqa: E402
import build_rb_zeroshot_notebook as rb        # noqa: E402

ROOT, CARGO, FORK = rb.ROOT, rb.CARGO, rb.FORK
md, code = rb.md, rb.code


def _find(cell_list, marker):
    hits = [c for c in cell_list if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(hits) == 1, f"marker {marker!r} matched {len(hits)} cells, expected 1"
    return hits[0]


HEADER = '''# MIR — everything still missing, in one run

One setup, one bundle, one scorer, one table. Blocks, in order:

| section | what | cost |
|---|---|---|
| 1-2 | paths, bundle (must carry the `mir_{train,test}` OpenIE graphs), scripts, Qwen3 | minutes |
| 3 | six dense baselines (skip on manifest) | ~15 min first time |
| 3c | **HyDE + MuGI** (gpt-4o-mini, N=4, Qwen3-0.6B encoder) | ~$0.20, ~15 min |
| 4 | engine install + stock patches | 5 min |
| 5 | **G-Reasoner** on the frame graph, **GFM-RAG** on the OpenIE graph (stock engine) | ~1 h each |
| 6 | CARGO fusion sources + patches | 1 min |
| 7 | multi-view scorer (restored from Drive) + component tables for the OpenIE graph | ~10 min |
| 8 | **+ Graph Reasoner (OpenIE graph)** | ~1 h |
| 9 | the MIR table, every row | seconds |

The graph baselines run BEFORE the fusion sources are written (section 6), so they use the
stock engine exactly as the SIR-4 baseline rows did. Everything is idempotent: finished arms
are skipped from their Drive outputs, so rerun from the top after a disconnect.

Namespaces on Drive: dense arms and SciGraphIR rows under `outputs/mir/`, HyDE/MuGI and the
graph baselines under `outputs/baselines/mir/` (the same places the separate notebooks use).
'''

BRIDGE_LX = '''# 3c-i. Bridge for the HyDE/MuGI cells (harvested from llm_expansion_baselines_mir.ipynb).
# They expect per-dataset dicts and a local work dir; the bundle is already unpacked.
import collections, hashlib
DATASETS = {DATASET: (BUNDLE, None)}
SPLIT  = "test"
OUTRT  = {DATASET: f"{DRIVE}/outputs/baselines/{DATASET}"}
CACHED = f"{DRIVE}/outputs/baselines/llm_expansions"
WRK    = "/content/llmx"
for d in list(OUTRT.values()) + [CACHED, WRK]:
    os.makedirs(d, exist_ok=True)
DOCS = {DATASET: f"{DATA_ROOT}/{DATASET}_test/raw/documents.json"}
QS   = {DATASET: QUERIES}
SETS = {DATASET: None}
SCORER = f"{S4}/eval/score_sir4.py"
_q = json.load(open(QS[DATASET]))
print(f"{DATASET}: {len(json.load(open(DOCS[DATASET]))):,} docs, {len(_q)} query rows, "
      f"{len({r['question'] for r in _q})} unique texts")
'''

RESTORE_LX = '''# 3c-v. Free the encoder and restore this notebook's names (the harvested scorer rebinds SCORES).
import gc, torch
try: del _model
except NameError: pass
gc.collect(); torch.cuda.empty_cache()
LX_SCORES = SCORES                      # {dataset: {label: scores}} from the HyDE/MuGI block
SCORES = f"{OUT_ROOT}/scores"           # this notebook's scores directory
print("HyDE/MuGI scored:", sorted(LX_SCORES.get(DATASET, {})))
'''

BRIDGE_BB = '''# 5-i. Bridge for the graph-baseline cells (harvested from mir_baselines.ipynb, ALL-mode).
import hashlib, threading, subprocess, csv, collections
# Every training subprocess below is launched as `python -m gfmrag...`. Make sure that resolves
# to THIS kernel's interpreter (where the engine is pip-installed), not /usr/bin/python3: in the
# combined notebook the bare name resolved to the system python and the run died with
# "No module named 'gfmrag'" (8 Sep).
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
# And make the engine importable from ANY working directory. The fusion notebooks only ever
# launch training with cwd=/content/gfm-rag, which is why the `pip install -e` not taking
# effect for a fresh interpreter went unnoticed; the harvested baseline runner launches from
# /content, so it needs PYTHONPATH.
os.environ["PYTHONPATH"] = "/content/gfm-rag" + os.pathsep + os.environ.get("PYTHONPATH", "")
_chk = subprocess.run(["python", "-c", "import gfmrag, sys; print(sys.executable, gfmrag.__file__)"],
                      capture_output=True, text=True, cwd="/content")
assert _chk.returncode == 0, f"`python` cannot import gfmrag from /content: {_chk.stderr[-400:]}"
print("training python:", _chk.stdout.strip())
DOMS, DSETS = [DATASET], [DATASET]
SPLIT  = "test"
OUTRT  = {DATASET: f"{DRIVE}/outputs/baselines/{DATASET}"}
CORPUS = {DATASET: f"{DATA_ROOT}/{DATASET}_test/raw"}
TOPK   = 300
PRED   = {DATASET: {}}                  # {dataset: {label: predictions path}}; dense arms are scored elsewhere
os.makedirs(OUTRT[DATASET], exist_ok=True)
assert os.path.exists(f"{DATA_ROOT}/{DATASET}_test/processed/stage1/nodes.csv"), (
    "no OpenIE graph in the bundle: build it (MIR_RUNBOOK.md section 9), re-bundle, re-upload")
# Qwen3 node indexes cached on Drive by earlier runs (the frame graphs by colab_mir_standard, the
# OpenIE graphs by a previous pass of this notebook): restoring them saves 20-45 min per graph.
ALL_GRAPHS = (f"{DATASET}_train_v16sc", f"{DATASET}_test_v16sc", f"{DATASET}_train", f"{DATASET}_test")
def _restore_index(g):
    src = f"{CACHE}/index/{g}"
    if not os.path.isdir(src):
        print(f"  {g}: no cached index on Drive (built on first use, 20-45 min)"); return
    for d in os.listdir(src):
        shutil.copytree(f"{src}/{d}", f"{DATA_ROOT}/{g}/processed/{d}", dirs_exist_ok=True)
    print(f"  {g}: index restored")
def _save_index(g):
    pr = f"{DATA_ROOT}/{g}/processed"
    if not os.path.isdir(pr): return
    for d in os.listdir(pr):
        if d != "stage1":
            shutil.copytree(f"{pr}/{d}", f"{CACHE}/index/{g}/{d}", dirs_exist_ok=True)
for g in ALL_GRAPHS: _restore_index(g)
'''

RESTORE_BB = '''# 5-v. Restore this notebook's names (the harvested scorer rebinds SCORES).
BB_SCORES = SCORES                      # {dataset: {label: scores}} from the graph-baseline block
SCORES = f"{OUT_ROOT}/scores"
for g in ALL_GRAPHS: _save_index(g)     # the node indexes the baselines built are reused by sections 7-8
print("graph baselines scored:", sorted(BB_SCORES.get(DATASET, {})))
'''

TABLE = '''# 9. The MIR table, every row, R@3 / R@5 / nDCG@5 / mAP as percentages, `all` slice.
def _sc(p):
    return json.load(open(p))["all"] if os.path.exists(p) else None
B = f"{DRIVE}/outputs/baselines/{DATASET}"
ROWS = [(r"\\textit{Sparse lexical}",          [("BM25", f"{SCORES}/bm25_scores.json")]),
        (r"\\textit{Dense embedding}",         [("BGE-large", f"{SCORES}/bge_scores.json"),
                                               ("Qwen3-Embedding", f"{SCORES}/qwen3_scores.json"),
                                               ("SPECTER2-base", f"{SCORES}/specter2_scores.json"),
                                               ("SciNCL", f"{SCORES}/scincl_scores.json")]),
        (r"\\textit{LLM-expanded dense}",      [("HyDE", f"{B}/scores_HyDE_gpt-4o-mini.json"),
                                               ("MuGI", f"{B}/scores_MuGI_gpt-4o-mini.json")]),
        (r"\\textit{Reasoning-trained dense}", [("ReasonIR-8B", f"{SCORES}/reasonir_scores.json")]),
        (r"\\textit{Graph retrieval}",         [("GFM-RAG", f"{B}/scores_GFM-RAG.json"),
                                               ("G-Reasoner", f"{B}/scores_G-Reasoner.json")]),
        (r"\\textit{\\textsc{SciGraphIR}}",     [("Multi-View Semantic Scorer", f"{SCORES}/scigraphir_scorer_scores.json"),
                                               ("+ Graph Reasoner (OpenIE graph)", f"{SCORES}/scigraphir_openie_graph_scores.json"),
                                               ("+ Graph Reasoner (SciAfford graph)", f"{SCORES}/scigraphir_graph_scores.json"),
                                               ("+ CCMP (full SciGraphIR)", f"{SCORES}/scigraphir_ccmp_scores.json")])]
MET = [("recall@3", "R@3"), ("recall@5", "R@5"), ("ndcg@5", "nDCG@5"), ("map", "mAP")]
lines = []
def out(s=""): print(s); lines.append(s)
out(f"# MIR test ({len(json.load(open(QUERIES)))} proposals, extended corpus)\\n")
out("| Method | " + " | ".join(l for _, l in MET) + " |"); out("|---|" + "--:|" * len(MET))
for fam, rows in ROWS:
    out(f"| {fam} | | | | |")
    for lab, p in rows:
        s = _sc(p)
        out(f"| {lab} | " + (" | ".join(f"{100 * s[m]:.2f}" if s.get(m) is not None else "--" for m, _ in MET)
                            if s else " | ".join("--" for _ in MET)) + " |")
out()
out("Frame-graph SciGraphIR rows come from the earlier colab_mir_standard run on Drive; MOOSE-Star and LATTICE")
out("are scored locally (MIR_RUNBOOK.md section 7 and results/mir/). Dashes = not run yet.")
open(f"{OUT_ROOT}/table_mir_all.md", "w").write("\\n".join(lines)); print("\\nwrote", f"{OUT_ROOT}/table_mir_all.md")
'''


def main() -> int:
    a = _args
    src = json.load(open(rb.SRC_NB))["cells"]
    base = json.load(open(rb.BASE_NB))["cells"]
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fusion_files = {f"/content/gfm-rag/{rel}": open(f"{FORK}/{rel}").read() for rel in rb.FUSION_REL}
    assert not any("'''" in v for v in fusion_files.values())
    files_cell = code(
        f"# === write the CARGO-fusion files into the fork (generated from the repo copies {built}) ===\n"
        "# Runs AFTER the graph baselines on purpose: they trained on the stock engine.\n"
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
    arms_src = "".join(base[rb.ARMS_CELL]["source"])
    assert "ARMS = [" in arms_src and "ReasonIR-8B" in arms_src

    # harvested cells
    lx_install = _find(lx.cells, '!pip -q install "openai>=1.40"')
    lx_gen     = _find(lx.cells, 'GEN_MODEL = "gpt-4o-mini"')
    lx_ret     = _find(lx.cells, 'ENCODER = "Qwen/Qwen3-Embedding-0.6B"')
    lx_score   = _find(lx.cells, "ARM_TAGS = {")
    bb_ship    = _find(bb.ac, "ENTITY_OK = {}")
    bb_gre     = _find(bb.ac, "def run_baseline")
    bb_gfm     = _find(bb.ac, "if not ENTITY_OK.get(ds)")
    bb_score   = _find(bb.ac, "SETS = {ds:")
    assert '"map"' in "".join(bb_score["source"]) and '"map"' in "".join(lx_score["source"])

    paths = (bm.PATHS.replace("__EPOCHS__", str(a.epochs)).replace("__BATCH__", str(a.batch))
             .replace("__CCMP__", "standard").replace("__GRAPH__", "openie"))
    cells = [md(HEADER), md("## 1. GPU + Drive + paths"), code(paths),
             md("## 2. Unpack + install the current scripts"),
             code(bm.UNPACK.replace("__OVERLAY__", json.dumps(overlay)).replace("__BUILT__", built)),
             md("## 2b. Qwen3-Embedding-0.6B (cached on Drive)"), src[rb.QWEN_CELLS[1]],
             md("## 3. Dense baselines\n### 3a. The arms\n*(harvested verbatim from `sir4_baselines_all.ipynb`)*"),
             code(arms_src + rb.ARMS_TAIL), md("### 3b. Run (skips on manifest)"), code(bm.RUN_BASELINES),
             md("### 3b-ii. Score"), code(bm.SCORE),
             md("### 3c. HyDE + MuGI\n*(harvested from `llm_expansion_baselines_mir.ipynb`; same encoder as the "
                "Qwen3-Embedding row, generations cached on Drive per query text)*"),
             code(BRIDGE_LX), lx_install, lx_gen, lx_ret, lx_score, code(RESTORE_LX),
             md("## 4. Engine install + stock patches\n*(engine verbatim from the TOMATO notebook; the pins, "
                "PyG fix and torchvision shim are the ones the baselines notebook applies)*"),
             src[5], src[6], bb.PIN_CELL, bb.PYG_CELL, bb.SHIM_CELL,
             md("## 5. Graph baselines on the STOCK engine\n*(harvested from `mir_baselines.ipynb`)*\n\n"
                "G-Reasoner trains on `mir_{train,test}_v16sc`; GFM-RAG on the `mir_{train,test}` OpenIE "
                "entity graph. Signature-gated: a finished run on Drive is a `[skip]`."),
             code(BRIDGE_BB), bb_ship, md("### 5c. G-Reasoner"), bb_gre, md("### 5d. GFM-RAG"), bb_gfm,
             md("### 5e. Score"), bb_score, code(RESTORE_BB),
             md("## 6. CARGO fusion sources + patches\n*(cloned from `tomato_ccmp_ablation.ipynb`; the "
                "fusion-source blob is regenerated from the repo)*")]
    for i in rb.ENGINE_CELLS:
        if i in (5, 6):
            continue                      # already installed in section 4
        cells.append(files_cell if i == 7 else src[i])
    cells += [md("## 7. Multi-view scorer (restored from Drive) + component tables for the OpenIE graph"),
              code(bm.HELPERS), code(bm.SCORER),
              md("## 8. + Graph Reasoner (OpenIE graph)\nSmoke first, then one run of about an hour. "
                 "Seedless test queries are re-inserted as empty rankings before scoring."),
              code(bm.TRAIN_ARMS), md("## 9. The table"), code(TABLE)]

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
            raise SystemExit(f"cell {i} line {e.lineno}: {e.msg}")

    out = a.out or f"{ROOT}/notebooks/colab_mir_all.ipynb"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(cells)} cells | epochs {a.epochs} batch {a.batch} | graph openie")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
