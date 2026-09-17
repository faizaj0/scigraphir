"""
build_showcase_notebook.py -- a self-contained Colab notebook per dataset for the qualitative showcase:
graph-channel scan of every gold, path interpretations (NBFNet / GFM-RAG recipe) for every cross-field
gold, the ranked "amazing example" list, the GFM-RAG Table-4 LaTeX and the Fig-6-style hop figure.

The setup cells (paths, bundle, Qwen3, engine, fusion sources, patches) are taken from the dataset's
existing notebook (colab_qualitative_<field> / colab_routing_tomato / colab_mir_all), so the notebook
runs on its own from a fresh runtime; the analysis cells are the sections of colab_showcase_cell.py.

    python3 prep/build_showcase_notebook.py --dataset sir4_cs        # -> colab_showcase_sir4_cs.ipynb
    python3 prep/build_showcase_notebook.py --all
"""
import argparse
import ast
import sys
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import overlay_cell  # noqa: E402
SRC_NB = {"mir": "colab_mir_all.ipynb", "tomato": "colab_routing_tomato.ipynb",
          **{f"sir4_{f}": f"colab_qualitative_{f}.ipynb" for f in ("cs", "biology", "physics", "matsci")}}
SETUP = [("!nvidia-smi", "first"), ("# 2. Unpack", "first"), ("import os, shutil", "first"), ("import os, sys, torch", "first"),
         ("# === write the SciGraphIR-fusion files", "first"), ("# Cell 3a is reused VERBATIM", "first"),
         ("# The ULTRA layers vendored", "last"), ("# THE TRAINING SUBPROCESS IS A FRESH PYTHON", "last"),
         ("STF = ", "first"), ("# Idempotent: re-running is a no-op", "first")]
SETUP_TITLES = ["1. GPU, Drive, paths", "2. Unpack the bundle + current scripts", "2b. Qwen3-Embedding (cached on Drive)",
                "3. Engine", "3b. Fusion sources", "3c. Config defaults", "3d. PyG version fix", "3e. torchvision shim",
                "3f. Diagnostics patch", "3g. CCMP_LR patch"]
# section markers of colab_showcase_cell.py, in order, with the notebook title of each analysis cell
SECTIONS = [("# the interpretation entry point", "4. interpret_paths.py (the interpretation entry point)"),
            ("# --- graphs, checkpoints, output dir", "5. Graphs, checkpoints, output directory"),
            ("# --- component tables", "6. Component tables + scorer files"),
            ("# --- checkpoint environment", "7. Checkpoint environment"),
            ("# --- scan every gold", "8. Scan every gold under every arm (no path search)"),
            ("# --- the table", "9. Graph-channel table per stratum"),
            ("# ================= SHOWCASE", "10. Path interpretations for every cross-field gold (+ same-field sample)"),
            ("os.makedirs(f\"{S4}/eval\", exist_ok=True)", "11. Showcase: ranked examples, LaTeX table, hop figure")]


def tolerate_pyg_fix(src: str) -> str:
    """The PyG version-fix cell asserts it patched something; the overlay may already have. Make it a no-op then."""
    return src.replace('assert hits, "version-check pattern not found -- has the engine changed?"', 'if not hits:   # the embedded overlay (3b-bis) ships ultra/layers.py with its own isdigit() fix, so there may be nothing left to patch\n    already = [os.path.join(r_, f_) for r_, _, fs in os.walk("/content/gfm-rag/gfmrag") for f_ in fs\n               if f_.endswith(".py") and "pyg_version" in open(os.path.join(r_, f_)).read() and "isdigit()" in open(os.path.join(r_, f_)).read()]\n    assert already, "version-check pattern not found and no isdigit() fix present -- has the engine changed?"\n    print("  already patched by the overlay:", already)')


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def build(dataset: str) -> str:
    cell = open(f"{ROOT}/colab_cells/colab_showcase_cell.py").read()
    cell = re.sub(r'^DATASET = "[^"]+"', f'DATASET = "{dataset}"', cell, count=1, flags=re.M)
    # head: everything before the replay block, minus the drive mount (the source paths cell mounts Drive)
    head = cell[: cell.index("# --- replay the source notebook's setup cells")]
    head = head.replace("from google.colab import drive\ndrive.mount('/content/drive')\n", "")
    head = "# Dataset, environment and the checkpoint spec. Run first: the PATH/PYTHONPATH lines must precede cell 1.\n" + head[head.index("DATASET = "):]
    # the checks that followed the replay, kept as the start of the first analysis cell
    tail = cell[cell.index("assert DATASET == SCAN_DATASET"):]
    src = [''.join(c["source"]) for c in json.load(open(f"{ROOT}/notebooks/{SRC_NB[dataset]}"))["cells"] if c["cell_type"] == "code"]
    setup = []
    for (mark, which), title in zip(SETUP, SETUP_TITLES):
        hits = [s for s in src if s.startswith(mark)]
        assert hits, f"{SRC_NB[dataset]}: no cell starts with {mark!r}"
        setup.append((title, tolerate_pyg_fix(hits[0] if which == "first" else hits[-1])))
    # split the analysis part at the section markers
    idx = [tail.index(m) for m, _ in SECTIONS]
    assert idx == sorted(idx), "section markers out of order"
    parts = [tail[: idx[0]]] + [tail[a:b] for a, b in zip(idx, idx[1:] + [len(tail)])]
    checks = parts[0]
    analysis = [(t, s) for (_, t), s in zip(SECTIONS, parts[1:])]
    analysis[0] = (analysis[0][0], checks + analysis[0][1])

    header = md(f"# Qualitative showcase: {dataset}\n\n"
                "Finds the examples that show cross-domain scientific reasoning and draws the figures, in the style of "
                "GFM-RAG's path interpretations (Table 4) and hop-distribution figure (Fig. 6).\n\n"
                "1. every gold of every test query ranked under the graph channel alone, the multi-view scorer, Qwen3 cosine "
                'and the fused score, per arm (SciAfford graph + CCMP, same weights with the gate off, no-CCMP control, OpenIE graph);\n'
                "2. NBFNet-style gradient beam search from the query's seed nodes to every gold of every cross-field query "
                "(plus a same-field sample), with the CCMP gate on every hop;\n"
                "3. `eval/showcase.py`: candidates ranked for a reader (cosine buries the gold, the model recovers it through a "
                "mechanism route), the Table-4 LaTeX, the hop figure.\n\n"
                f"Needs on Drive: `{dataset}_bundle.zip` with both graphs, the finished checkpoints listed in cell 0, the scorer "
                "files, and `gfm-rag-adapted.zip`. Outputs: `outputs/scan/" + dataset + "/`.\n\n"
                "Cells 1 to 3g are the setup cells of `" + SRC_NB[dataset] + "` verbatim.")
    cells = [header, md("## 0. Dataset, environment, checkpoint spec"), code(head)]
    for title, s in setup:
        cells += [md(f"## {title}"), code(s)]
        if title.startswith("3b."):
            cells += [md("## 3b-bis. Current fusion sources (embedded overlay)"), code(overlay_cell.cell())]
    for title, s in analysis:
        cells += [md(f"## {title}"), code(s.rstrip() + "\n")]
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        lines, mag = [], False                       # shell magics may continue over lines with a trailing backslash
        for ln in c["source"].split("\n"):
            if ln.lstrip().startswith(("!", "%")) or mag:
                mag = ln.rstrip().endswith("\\"); lines.append("pass")
            else:
                lines.append(ln)
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as e:
            raise SystemExit(f"{dataset}: cell {i} line {e.lineno}: {e.msg}")
    out = f"{ROOT}/notebooks/colab_showcase_{dataset}.ipynb"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(cells)} cells")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SRC_NB), default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for d in (sorted(SRC_NB) if a.all or not a.dataset else [a.dataset]):
        build(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
