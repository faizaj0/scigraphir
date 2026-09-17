"""
build_showcase2_notebook.py -- "showcase 2": a short Colab notebook that runs path interpretations for a HAND-PICKED
list of (query, gold) pairs under every arm (SciAfford graph + CCMP, same weights with the gate off, no-CCMP control if
it exists, OpenIE graph) with a wide beam, then draws the ladder figure (rank at every stage of the cumulative
ablation, graph channel on/off) and the route figures (typed boxes, relation per hop, CCMP gate per hop, weight with
and without the gate, OpenIE route for contrast). No corpus-wide scan, so it runs in ~20-30 minutes.

    python3 prep/build_showcase2_notebook.py --dataset sir4_cs     # -> colab_showcase2_sir4_cs.ipynb

Setup cells come verbatim from the dataset's source notebook (as in build_showcase_notebook.py); the graphs /
components / model_env sections come from colab_showcase_cell.py; the pick cell is colab_paths_pick_cell.py with
the dataset's PICK; the figure cell embeds eval/showcase2_figs.py.
"""
import argparse, ast, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import overlay_cell  # noqa: E402
from build_showcase_notebook import SRC_NB, SETUP, SETUP_TITLES, md, code, tolerate_pyg_fix  # noqa: E402

# the hand-picked pairs per dataset: query id -> gold ids. Comments say why each is there.
PICKS = {
    "sir4_cs": {
        "10.48550_arxiv.2601.22474": ["q:ac2856c33ccdf7be", "10.1037/h0061626"],     # LLM latent learning -> Tolman 1930 / 1948: scorer-led (cosine 957, scorer 1, fused 1); graph 238 gate off, 1965 with CCMP; routes end at the psychology domain hub. A SciGraphIR example, not a graph or CCMP one.
        "10.48550_arxiv.2602.20408": ["10.3758/bf03202751"],                          # LLM idea diversity -> constraining effects of examples (graph 2, CCMP 25->18)
        "10.48550_arxiv.2602.04572": ["10.2307/1906951"],                             # GenAI + Q&A forum -> Two-Person Cooperative Games (graph 1765->372 with CCMP)
        "10.48550_arxiv.2601.20761": ["10.1561/3600000002"],                          # sequential quantum tomography -> e-values (2-hop method route)
        "10.48550_arxiv.2602.21585": ["q:4f02a72cb830786c"],                          # test-time LLM optimisation -> Double Thompson Sampling (graph channel 1)
        "10.48550_arxiv.2602.18621": ["10.1017/cbo9780511805967", "10.1007/s40687-024-00471-w"],   # sandpile groups -> Enumerative Combinatorics (shares-purpose route, CCMP 10->8)
        "10.48550_arxiv.2602.05843": ["10.48550/arxiv.1809.01999"],                   # LM agents exploring latent dynamics -> Recurrent World Models (graph 2477->184 with CCMP)
        "10.48550_arxiv.2603.03985": ["10.1111/j.1749-6632.2010.05443.x", "10.1063/1.4822124"],    # real-time interaction -> memory reconsolidation (graph 2) / Ebbinghaus memristor
        "10.48550_arxiv.2602.00032": ["10.1007/s00146-022-01443-w"],                  # text-to-image bias -> word embeddings are biased (CCMP 6->4)
        "10.48550_arxiv.2602.07181": ["10.1037/0022-3514.89.3.449"],                  # personality traits as user model -> neuroticism / trait-consistent affect (all baselines >300)
        "10.48550_arxiv.2602.22831": ["10.1038/s41586-018-0637-6"],                   # LLM moral triage -> The Moral Machine experiment (OpenIE direct mention; SciAfford graph no bridge)
        "10.48550_arxiv.2602.13723": ["10.48550/arxiv.2502.05368"],                   # requirements-to-system -> Otter (graph 5, CCMP 11->9)
        "10.48550_arxiv.2603.08337": ['10.1017/cbo9780511804441'],                     # DeFi swap routing -> Convex Optimization (fused 8->6, graph 1575->44 with CCMP)
    },
}
SECTION_MARKS = ["# --- graphs, checkpoints, output dir", "# --- component tables", "# --- checkpoint environment", "# --- scan every gold"]


def build(dataset: str) -> str:
    cell = open(f"{ROOT}/colab_cells/colab_showcase_cell.py").read()
    cell = re.sub(r'^DATASET = "[^"]+"', f'DATASET = "{dataset}"', cell, count=1, flags=re.M)
    head = cell[: cell.index("# --- replay the source notebook's setup cells")]
    head = head.replace("from google.colab import drive\ndrive.mount('/content/drive')\n", "")
    head = "# Dataset, environment and the checkpoint spec. Run first: the PATH/PYTHONPATH lines must precede cell 1.\n" + head[head.index("DATASET = "):]
    tail = cell[cell.index("assert DATASET == SCAN_DATASET"):]
    idx = [tail.index(m) for m in SECTION_MARKS]; assert idx == sorted(idx)
    interp = tail[: idx[0]]                                    # post-replay checks + interpret_paths.py
    graphs, components, model_env = tail[idx[0]: idx[1]], tail[idx[1]: idx[2]], tail[idx[2]: idx[3]]
    src = [''.join(c["source"]) for c in json.load(open(f"{ROOT}/{SRC_NB[dataset]}"))["cells"] if c["cell_type"] == "code"]
    setup = []
    for (mark, which), title in zip(SETUP, SETUP_TITLES):
        hits = [s for s in src if s.startswith(mark)]; assert hits, f"{SRC_NB[dataset]}: no cell starts with {mark!r}"
        setup.append((title, tolerate_pyg_fix(hits[0] if which == "first" else hits[-1])))
    # the pick cell: colab_paths_pick_cell.py with this dataset's PICK
    pick_src = open(f"{ROOT}/colab_cells/colab_paths_pick_cell.py").read()
    body = pick_src[pick_src.index("NUM_BEAM, PATH_TOPK, TOP_VIEWS"):]
    pick_lines = ["PICK = {   # query id -> gold ids; edit freely, then re-run this cell and the next"]
    for q, gs in PICKS.get(dataset, {}).items(): pick_lines.append(f"    {q!r}: {gs!r},")
    pick_lines.append("}")
    pick_cell = ("# ===== Path interpretations for the HAND-PICKED (query, gold) pairs, every arm, CCMP gate on vs off, wide beam =====\n"
                 "# Writes SCAN_OUT/hops_pick_<arm>.json and prints a rank table + every route per pair. 3-5 min per arm.\n"
                 + "\n".join(pick_lines) + "\n" + body)
    # the figure cell: embed eval/showcase2_figs.py, run it, show the figures
    figs = open(f"{ROOT}/eval/showcase2_figs.py").read()
    assert "'''" not in figs, "showcase2_figs.py must not contain ''' (embedded in r'''...''')"
    fig_cell = ("# ===== Figures: the rank ladder per pair (graph channel on/off, baselines) and the interpreted routes with gates =====\n"
                "import os, sys, json\n"
                "os.makedirs(f\"{S4}/eval\", exist_ok=True)\n"
                "open(f\"{S4}/eval/showcase2_figs.py\", \"w\").write(r'''" + figs.replace('"""', "'''") .replace("'''", '"""') + "''')\n"
                "BASE = f\"{DRIVE}/outputs/baselines/{DATASET}\"\n"
                "OUT2 = f\"{SCAN_OUT}/showcase2_{DATASET}\"\n"
                "cmd = [sys.executable, \"-u\", \"eval/showcase2_figs.py\", \"--dataset\", DATASET, \"--dir\", SCAN_OUT, \"--pick\", PK_G,\n"
                "       \"--queries\", QUERIES, \"--docs\", f\"{DATA_ROOT}/{DATASET}_test/raw/documents.json\", \"--out\", OUT2]\n"
                "_qt = f\"{SCIGRAPHIR_ROOT}/sir-4/data/benchmark/{DATASET.replace('sir4_', '')}_test_final/eval.json\"\n"
                "if os.path.exists(_qt): cmd += [\"--sir4\", _qt]\n"
                "for tag in (\"qwen3\", \"bge\", \"reasonir\", \"bm25\", \"specter2\", \"scincl\"):\n"
                "    p_ = f\"{BASE}/predictions_{tag}_{DATASET}_test.json\"\n"
                "    if os.path.exists(p_): cmd += [\"--pred\", f\"{tag}={p_}\"]\n"
                "sh(cmd, S4)\n"
                "from IPython.display import Image, display\n"
                "import glob as _glob\n"
                "for p_ in [f\"{OUT2}_ladder.png\"] + sorted(_glob.glob(f\"{OUT2}_routes_*.png\")):\n"
                "    print(os.path.relpath(p_, DRIVE)); display(Image(p_))\n"
                "print(open(f\"{OUT2}.md\").read()[:6000])\n")
    def embed(name):
        src = open(f"{ROOT}/eval/{name}").read(); assert "\'\'\'" not in src, f"{name} must not contain \'\'\'"
        return f"open(f\"{{S4}}/eval/{name}\", \"w\").write(r\'\'\'" + src + "\'\'\')\n"
    effect_cell = ("# ===== CCMP effect over EVERY gold of the path-interpretation sample, paired (same weights, gate on vs off) =====\n"
                   "# No gold reasoning paths exist (GFM-RAG Fig. 6 compares predicted hops with gold hops), but gold DOCUMENTS do: (a) graph-channel\n"
                   "# rank of every gold gate off vs on, (b) the distribution of the rank change for the graph channel and the fused score, cross- vs\n"
                   "# same-field, (c) the gate along the top route (into which node types it amplifies / damps; last hop vs inner hops).\n"
                   "# Uses the corpus-wide hops files of the showcase run (SCAN_OUT/hops_frame_ccmp*.json) when present, else the pick files.\n"
                   "import os, sys, json\n"
                   "os.makedirs(f\"{S4}/eval\", exist_ok=True)\n"
                   + embed("ccmp_effect_fig.py") +
                   "PFX = \"hops_\" if os.path.exists(f\"{SCAN_OUT}/hops_frame_ccmp_off.json\") else \"hops_pick_\"\n"
                   "print(\"hops files:\", PFX + \"frame_ccmp*.json\", \"(corpus-wide showcase run)\" if PFX == \"hops_\" else \"(picked pairs only: few points)\")\n"
                   "OUTE = f\"{SCAN_OUT}/fig_ccmp_effect_{DATASET}\"\n"
                   "cmd = [sys.executable, \"-u\", \"eval/ccmp_effect_fig.py\", \"--dir\", SCAN_OUT, \"--prefix\", PFX, \"--docs\", f\"{DATA_ROOT}/{DATASET}_test/raw/documents.json\", \"--out\", OUTE]\n"
                   "_qt = f\"{SCIGRAPHIR_ROOT}/sir-4/data/benchmark/{DATASET.replace('sir4_', '')}_test_final/eval.json\"\n"
                   "if os.path.exists(_qt): cmd += [\"--sir4\", _qt]\n"
                   "sh(cmd, S4)\n"
                   "from IPython.display import Image, display\n"
                   "display(Image(OUTE + \".png\")); print(json.dumps(json.load(open(OUTE + \"_summary.json\")), indent=1)[:3000])\n")
    tex_cell = ("# ===== LaTeX for the thesis: the route figure (TikZ, panels a+b) per picked pair and the path table (GFM-RAG Table-4 format) =====\n"
                "# Writes SCAN_OUT/showcase2_<dataset>_circles_<n>.tex and SCAN_OUT/tab_route_paths_<dataset>.tex. Compiles them only if tectonic is\n"
                "# on PATH (it is not on Colab): download the .tex files and run  tectonic -X compile <file>  locally, or pass them to route_circles_tikz.py --compile.\n"
                "import os, sys, json, shutil\n"
                "os.makedirs(f\"{S4}/eval\", exist_ok=True)\n"
                + embed("route_circles_tikz.py") + embed("route_table_tex.py") +
                "PICKD = json.load(open(PK_G)); pairs = [f\"{q}={g}\" for q, gs in PICKD.items() for g in gs]\n"
                "common = [\"--dataset\", DATASET, \"--dir\", SCAN_OUT, \"--prefix\", \"hops_pick_\", \"--docs\", f\"{DATA_ROOT}/{DATASET}_test/raw/documents.json\", \"--queries\", QUERIES]\n"
                "_qt = f\"{SCIGRAPHIR_ROOT}/sir-4/data/benchmark/{DATASET.replace('sir4_', '')}_test_final/eval.json\"\n"
                "if os.path.exists(_qt): common += [\"--sir4\", _qt]\n"
                "comp = [\"--compile\"] if shutil.which(\"tectonic\") else []\n"
                "for i, pr in enumerate(pairs, 1):\n"
                "    out = f\"{SCAN_OUT}/showcase2_{DATASET}_circles_{i}\"\n"
                "    rc = sh([sys.executable, \"eval/route_circles_tikz.py\"] + common + [\"--pair\", pr, \"--n-routes\", \"2\", \"--no-strip\", \"--out\", out] + comp, S4, check=False)\n"
                "    print((\"ok  \" if rc == 0 else \"skip\"), i, pr, \"->\", os.path.relpath(out + \".tex\", DRIVE))\n"
                "tab = f\"{SCAN_OUT}/tab_route_paths_{DATASET}\"\n"
                "sh([sys.executable, \"eval/route_table_tex.py\"] + common + sum(([\"--pair\", pr] for pr in pairs), []) + [\"--n-routes\", \"2\", \"--out\", tab] + comp, S4)\n"
                "print(open(tab + \".tex\").read()[:4000])\n")
    header = md(f"# Showcase 2: hand-picked path interpretations, {dataset}\n\n"
                "Runs the NBFNet-style gradient beam search from each query's seed nodes to a hand-picked gold under every arm "
                '(SciAfford graph + CCMP, the same weights with the gate off, the no-CCMP control where a checkpoint exists, the OpenIE graph) '
                "with a wide beam, then draws:\n\n"
                "1. **the ladder**: the gold's rank at every stage of the cumulative ablation (Qwen3 cosine, multi-view scorer, + OpenIE graph, "
                "+ SciAfford graph with the gate off, + CCMP), with the graph channel alone (gate off / on) as hollow markers and the dense/lexical "
                "baselines as grey ticks: the difference the graph branch and CCMP make, per example;\n"
                "2. **the routes**: for each pair the top interpreted route (typed boxes, relation per hop, CCMP gate per hop, weight with and "
                "without the gate) with the OpenIE graph's route underneath for contrast.\n\n"
                "No corpus-wide scan: about 20-30 minutes on a fresh runtime. Edit `PICK` in the pick cell to change the examples. "
                f"Needs on Drive: `{dataset}_bundle.zip`, the checkpoints listed in cell 0, the scorer files, `gfm-rag-adapted.zip`, "
                f"`qwen3-embedding-0.6b/`. Outputs: `outputs/scan/{dataset}/hops_pick_*.json` and `showcase2_{dataset}_*`.\n\n"
                "Cells 1 to 3g are the setup cells of `" + SRC_NB[dataset] + "` verbatim.")
    cells = [header, md("## 0. Dataset, environment, checkpoint spec"), code(head)]
    for title, s in setup:
        cells += [md(f"## {title}"), code(s)]
        if title.startswith("3b."): cells += [md("## 3b-bis. Current fusion sources (embedded overlay)"), code(overlay_cell.cell())]
    cells += [md("## 4. interpret_paths.py (the interpretation entry point)"), code(interp.rstrip() + "\n"),
              md("## 5. Graphs, checkpoints, output directory"), code(graphs.rstrip() + "\n"),
              md("## 6. Component tables + scorer files (cached on Drive)"), code(components.rstrip() + "\n"),
              md("## 7. Checkpoint environment"), code(model_env.rstrip() + "\n"),
              md("## 8. Path interpretations for the picked pairs"), code(pick_cell),
              md("## 9. Figures"), code(fig_cell),
              md("## 10. CCMP effect over every gold (paired: same weights, gate on vs off)"), code(effect_cell),
              md("## 11. LaTeX for the thesis: route figures (TikZ) and the path table"), code(tex_cell)]
    for i, c in enumerate(cells):
        if c["cell_type"] != "code": continue
        lines, mag = [], False
        for ln in c["source"].split("\n"):
            if mag: mag = ln.rstrip().endswith("\\"); continue
            if ln.lstrip().startswith(("!", "%")): mag = ln.rstrip().endswith("\\"); lines.append(" " * (len(ln) - len(ln.lstrip())) + "pass")
            else: lines.append(ln)
        try: ast.parse("\n".join(lines))
        except SyntaxError as e: raise SystemExit(f"{dataset}: cell {i} line {e.lineno}: {e.msg}")
    out = f"{ROOT}/notebooks/colab_showcase2_{dataset}.ipynb"
    json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"},
               "nbformat": 4, "nbformat_minor": 5}, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(cells)} cells"); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=sorted(PICKS), default="sir4_cs"); a = ap.parse_args()
    build(a.dataset); return 0


if __name__ == "__main__":
    raise SystemExit(main())
