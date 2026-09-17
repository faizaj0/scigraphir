"""
build_mir_zeroshot_notebook.py -- Table 9.3 on MIR: SciGraphIR checkpoints trained elsewhere,
frozen, ranking MIR's test corpus with no training, tuning or adaptation.

THE ROWS. The same two sources as the ResearchBench zero-shot table (Table 9.3):

    SciGraphIR trained on TOMATO-Star     outputs/tomato_ablations_v1/tomato/<tomato run>/model_best.pth
    SciGraphIR trained on SIR-4 (4 fields) outputs/rb_zeroshot/scigraphir_sir4_all4_qwenmlp_ccmp_e10_b2/model_best.pth

Each is paired with the multi-view scorer warm start of ITS OWN source (the params json +
popnet from that source's 5d run), exactly as build_rb_zeroshot_notebook.py pairs them. The
target-side inputs are properties of MIR's corpus, not of the model: MIR's own hypothetical answers, the
handcrafted scorer and semantic component tables on mir_test_v16sc, and the SciAfford graph itself.

MIR is a single field, so the table has one slice (`all`; every query is stratum "same") and
the MIR columns R@3 / R@5 / nDCG@5 / mAP. The in-benchmark rows (results/mir/table_mir.md,
colab_mir_standard.ipynb) are printed above the zero-shot rows for contrast when their score
files are on Drive.

Reuses build_rb_zeroshot_notebook.py's component-build and predict-only cells VERBATIM (so the
mechanics cannot differ from Table 9.3's) and build_mir_notebook.py's paths/unpack cells (so
the MIR plumbing cannot differ from the in-benchmark run's).

Usage
-----
    python3 prep/build_mir_zeroshot_notebook.py
    python3 prep/build_mir_zeroshot_notebook.py --ccmp residual     # TOMATO source = residual-target run
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
import build_mir_notebook as bm  # noqa: E402

ROOT, REPO_ROOT, FORK = rb.ROOT, rb.REPO_ROOT, rb.FORK
md, code = rb.md, rb.code

HEADER = '''# MIR — zero-shot cross-benchmark transfer (Table 9.3 on MIR)

Two **frozen** SciGraphIR checkpoints, trained on TOMATO-Star and on all four SIR-4 fields,
rank MIR's 4,857-document test corpus for its 155 proposals. No training, no tuning, no
adaptation. The only MIR-side inputs are MIR's own hypothetical answers and the component
tables computed from its corpus, which are properties of the corpus, not of the model.

| section | what | cost |
|---|---|---|
| 1-2 | paths, bundle, current scripts, Qwen3 | minutes |
| 3 | engine + fusion sources | 5 min |
| 4 | MIR component tables on the SciAfford graph (restored from the in-benchmark run's caches) | minutes |
| 5 | two predict-only passes, one per source checkpoint | minutes each |
| 6 | table: R@3, R@5, nDCG@5, mAP, next to the in-benchmark rows | seconds |

MIR is one field (computational linguistics): one slice, no same/cross split, no sets.json.
'''

SOURCES = '''# 4b. The frozen sources. Each checkpoint is paired with the multi-view scorer warm start of
# its OWN training run, exactly as Table 9.3 pairs them (build_rb_zeroshot_notebook.py).
TOMATO_OUT = f"{DRIVE}/outputs/tomato_ablations_v1/tomato"
RBZ_OUT    = f"{DRIVE}/outputs/rb_zeroshot"
SOURCES = {
    "scigraphir_tomato": ("SciGraphIR, trained on TOMATO-Star",
                          f"{TOMATO_OUT}/__TOMATO_RUN__/model_best.pth",
                          f"{TOMATO_OUT}/semantic/params_semantic_mlp_fixedloss_tomato.json",
                          f"{TOMATO_OUT}/semantic/popnet_semantic_mlp_fixedloss_tomato.pt"),
    "scigraphir_sir4":   ("SciGraphIR, trained on SIR-4 (CS + Bio + Phys + MS)",
                          f"{RBZ_OUT}/scigraphir_sir4_all4_qwenmlp_ccmp_e10_b2/model_best.pth",
                          f"{RBZ_OUT}/semantic_sir4_cs/params_semantic_mlp_fixedloss_sir4_cs.json",
                          f"{RBZ_OUT}/semantic_sir4_cs/popnet_semantic_mlp_fixedloss_sir4_cs.pt"),
}
for key, (label, ckpt, sj, sp) in SOURCES.items():
    for p in (ckpt, sj, sp):
        print(f"  {'ok ' if os.path.exists(p) else 'MISSING'}  {key:18} {os.path.relpath(p, DRIVE)}")
_missing = [k for k, (_, c, j, p) in SOURCES.items() if not all(os.path.exists(x) for x in (c, j, p))]
assert not _missing, f"source checkpoint or scorer files missing on Drive for {_missing}; see the list above"
'''

ARMS = '''# 5. Zero-shot: rank MIR with each frozen source, then score with the MIR column set.
SCORES = f"{OUT_ROOT}/scores"; os.makedirs(SCORES, exist_ok=True)
def score(key, pred, label):
    js, pq = f"{SCORES}/{key}_scores.json", f"{SCORES}/{key}_perquery.json"
    sh([sys.executable, "-u", "eval/score_sir4.py", "--pred", pred, "--queries", QUERIES,
        "--name", label, "--cols", COLS, "--json-out", js, "--per-query-out", pq], S4)
    return js
PRED = {}
for key, (label, ckpt, sj, sp) in SOURCES.items():
    print(f"\\n==================== {label} ====================")
    PRED[key] = predict_rb(ckpt, f"{DATASET}_zeroshot_{key}", sj, sp)
    score(key, PRED[key], f"{label}, zero-shot on MIR")
print("\\nscored:", sorted(PRED))
'''

TABLE = '''# 6. The table. In-benchmark rows (colab_mir_standard.ipynb, outputs/mir/scores) for contrast,
# then the two zero-shot rows. R@3 / R@5 / nDCG@5 / mAP as percentages over the `all` slice.
MET = [("recall@3", "R@3"), ("recall@5", "R@5"), ("ndcg@5", "nDCG@5"), ("map", "mAP")]
IN_BENCH = f"{DRIVE}/outputs/{DATASET}/scores"
ROWS = [(r"\\textit{Baselines (training-free)}", None),
        ("BM25", f"{IN_BENCH}/bm25_scores.json"), ("BGE-large", f"{IN_BENCH}/bge_scores.json"),
        ("Qwen3-Embedding", f"{IN_BENCH}/qwen3_scores.json"), ("SPECTER2-base", f"{IN_BENCH}/specter2_scores.json"),
        ("SciNCL", f"{IN_BENCH}/scincl_scores.json"), ("ReasonIR-8B", f"{IN_BENCH}/reasonir_scores.json"),
        (r"\\textit{\\textsc{SciGraphIR}, trained on MIR (in-benchmark, for reference)}", None),
        ("SciGraphIR (MIR-trained)", f"{IN_BENCH}/scigraphir_ccmp_scores.json"),
        (r"\\textit{\\textsc{SciGraphIR}, zero-shot (never trained on MIR)}", None)] + \\
       [(SOURCES[k][0], f"{SCORES}/{k}_scores.json") for k in SOURCES]
lines, tex = [], []
def out(s=""): print(s); lines.append(s)
out(f"# Zero-shot transfer to MIR ({len(json.load(open(QUERIES)))} proposals, {DATASET}_test corpus), nDCG/recall/mAP (%)\\n")
out("| Method | " + " | ".join(l for _, l in MET) + " |"); out("|---|" + "--:|" * len(MET))
BS = chr(92)
for lab, p in ROWS:
    if p is None:
        out(f"| {lab} | | | | |"); tex.append(BS + "midrule" if tex else "")
        tex.append(BS + "multicolumn{5}{@{}l}{" + lab + "}" + " " + BS + BS); continue
    if not os.path.exists(p):
        out(f"| {lab} | -- | -- | -- | -- |"); tex.append(f"{lab} & -- & -- & -- & --" + " " + BS + BS); continue
    s = json.load(open(p))["all"]
    out(f"| {lab} | " + " | ".join(f"{100 * s[m]:.2f}" for m, _ in MET) + " |")
    tex.append(f"{lab} & " + " & ".join(f"{100 * s[m]:.2f}" for m, _ in MET) + " " + BS + BS)
open(f"{OUT_ROOT}/table_mir_zeroshot.md", "w").write("\\n".join(lines) + "\\n")
tex = [BS + "begin{tabular}{@{}l cccc@{}}", BS + "toprule",
       BS + "textbf{Method} & Recall@3 & Recall@5 & nDCG@5 & mAP" + " " + BS + BS, BS + "midrule"] + \\
      [t for t in tex if t != ""] + [BS + "bottomrule", BS + "end{tabular}"]
open(f"{OUT_ROOT}/table_mir_zeroshot.tex", "w").write("\\n".join(tex) + "\\n")
print("\\nwrote", f"{OUT_ROOT}/table_mir_zeroshot.md", "and .tex")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccmp", default="standard", choices=["standard", "residual"],
                    help="which TOMATO-Star checkpoint is the TOMATO source: standard = the Table 9.1 row "
                         "(tomato_fusion_qwenmlp_ccmp_epoch10_b2); residual = tomato_fusion_qwenmlp_ccmpresid_epoch10_b2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tomato_run = ("tomato_fusion_qwenmlp_ccmpresid_epoch10_b2" if a.ccmp == "residual"
                  else "tomato_fusion_qwenmlp_ccmp_epoch10_b2")

    src = json.load(open(rb.SRC_NB))["cells"]
    built = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fusion_files = {f"/content/gfm-rag/{rel}": open(f"{FORK}/{rel}").read() for rel in rb.FUSION_REL}
    assert not any("'''" in v for v in fusion_files.values())
    files_cell = code(
        f"# === write the SciGraphIR-fusion files into the fork (generated from the repo copies {built}) ===\n"
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
    overlay = {rel: open(f"{REPO_ROOT}/{rel}").read() for rel in rb.OVERLAY_REL}
    assert not any("'''" in v for v in overlay.values())
    assert '"map":' in overlay["experiments/eval/score_sir4.py"], "score_sir4.py has no mAP; the table needs it"

    # Paths: the in-benchmark MIR cell (SciAfford graph, MIR caches), plus the names the verbatim
    # RB component/predict cells expect (RBG = target graph, CACHE_RB = target caches) and a
    # zero-shot output root so nothing collides with the in-benchmark run.
    paths = (bm.PATHS.replace("__EPOCHS__", "10").replace("__BATCH__", "2")
             .replace("__CCMP__", a.ccmp).replace("__GRAPH__", "frame")
             .replace('OUT_ROOT = f"{DRIVE}/outputs/{DATASET}"          # everything this notebook writes',
                      'OUT_ROOT = f"{DRIVE}/outputs/{DATASET}_zeroshot" # everything this notebook writes')
             .replace('CACHE    = f"{OUT_ROOT}/cache"                   # embeddings + Qwen3 node indexes, graph-keyed',
                      'CACHE    = f"{DRIVE}/outputs/{DATASET}/cache"     # the in-benchmark run\'s caches (op_emb, node index)'))
    assert "outputs/{DATASET}_zeroshot" in paths and 'outputs/{DATASET}/cache' in paths, "PATHS anchors moved in build_mir_notebook.py"
    paths += ('RBG, CACHE_RB = TEST, CACHE                      # names the verbatim RB component/predict cells use\n'
              'print("zero-shot target", RBG, "| sources: TOMATO-Star (' + tomato_run + '), SIR-4 all4")\n')
    components = (rb.RB_COMPONENTS.replace("ResearchBench inputs", "MIR inputs")
                  .replace("restored ResearchBench embedding caches", "restored MIR embedding caches"))
    predict = rb.PREDICT_HELPER.replace("ResearchBench corpus", "MIR corpus").replace("Rank ResearchBench", "Rank MIR")
    assert "def predict_rb" in predict and "def inspect_ckpt" in predict

    cells = [md(HEADER), md("## 1. GPU + Drive + paths"), code(paths),
             md("## 2. Unpack + install the current scripts"),
             code(bm.UNPACK.replace("__OVERLAY__", json.dumps(overlay)).replace("__BUILT__", built)),
             md("## 2b. Qwen3-Embedding-0.6B (cached on Drive)"), src[rb.QWEN_CELLS[1]],
             md("## 3. Engine + fusion sources\n*(cloned from `tomato_ccmp_ablation.ipynb`; the fusion-source blob is regenerated from the repo)*")]
    for i in rb.ENGINE_CELLS:
        cells.append(files_cell if i == 7 else src[i])
    cells += [md('## 4. MIR component tables on the SciAfford graph, and the frozen sources'),
              code(components), code(SOURCES.replace("__TOMATO_RUN__", tomato_run)),
              md("## 5. Zero-shot prediction and scoring"), code(predict), code(ARMS),
              md("## 6. The table"), code(TABLE)]
    out = a.out or f"{ROOT}/notebooks/colab_mir_zeroshot.ipynb"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(cells)} cells | TOMATO source {tomato_run} | SIR-4 source rb_zeroshot/scigraphir_sir4_all4_qwenmlp_ccmp_e10_b2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
