#!/usr/bin/env python3
"""Build colab_ccmp_path_sender_effect.ipynb: the path-sender gate effect (Delta_P) notebook.

Reuses the transparent setup of colab_table4_openie.ipynb (build_table4_notebook.py: bundles,
engine install, visible %%writefile source cells, Qwen3, checkpoint selection, component tables)
and adds two source files (eval/ccmp_path_sender_effect.py, eval/ccmp_path_sender_workflow.py),
the driver cell (colab_ccmp_path_sender_cell.py) and a summary cell.

    python3 prep/build_path_sender_notebook.py
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_table4_notebook as base  # noqa: E402

S4 = base.S4
OUT = S4 / "notebooks/colab_ccmp_path_sender_effect.ipynb"
# The gfm-rag-adapted.zip on Drive is the 21 Jun build (1,355,696 bytes, 251 files). Compared file by file with the
# repo engine (13 Sep 2026) it lacks two files and ships two stale ones:
#   - gfmrag/__init__.py and gfmrag/text_emb_models/__init__.py import the vLLM Qwen3 embedder unconditionally;
#     vllm is not on Colab, so `import gfmrag` fails (the pipeline uses the sentence-transformers model, qwen3_st);
#   - config/text_emb_model/qwen3_st.yaml is missing;
#   - config/wandb/default.yaml is missing, although sft_training_fusion.yaml lists `wandb: default` in its
#     defaults, so hydra stops with MissingConfigException before main() runs.
# (graph_index_dataset.py also differs, by an AQC seed-weight feature the cached bundles never use; it is left as on
# Drive so the run matches the 8 Sep scans.) The install cell repairs these four files right after extraction, so
# re-running it can never leave a broken engine behind; section 4 writes the same files again as visible cells.
REPAIR_FILES = [
    ("gfmrag/__init__.py", base.ENGINE / "gfmrag/__init__.py"),
    ("gfmrag/text_emb_models/__init__.py", base.ENGINE / "gfmrag/text_emb_models/__init__.py"),
    ("gfmrag/workflow/config/text_emb_model/qwen3_st.yaml", base.ENGINE / "gfmrag/workflow/config/text_emb_model/qwen3_st.yaml"),
    ("gfmrag/workflow/config/wandb/default.yaml", base.ENGINE / "gfmrag/workflow/config/wandb/default.yaml"),
]
EXTRA_SOURCES = REPAIR_FILES + [
    ("gfmrag/workflow/config/gfm_rag/sft_training_nodefeat.yaml", base.ENGINE / "gfmrag/workflow/config/gfm_rag/sft_training_nodefeat.yaml"),
    ("gfmrag/models/gfm_rag_v1/model_nodefeat.py", base.ENGINE / "gfmrag/models/gfm_rag_v1/model_nodefeat.py"),
    ("gfmrag/models/gfm_rag_v1/rankers.py", base.ENGINE / "gfmrag/models/gfm_rag_v1/rankers.py"),
    ("gfmrag/graph_index_datasets/graph_index_dataset_v1.py", base.ENGINE / "gfmrag/graph_index_datasets/graph_index_dataset_v1.py"),
    ("gfmrag/workflow/ccmp_path_sender_core.py", S4 / "eval/ccmp_path_sender_effect.py"),
    ("gfmrag/workflow/ccmp_path_sender.py", S4 / "eval/ccmp_path_sender_workflow.py"),
]

INTRO = r"""# CCMP path-sender gate effect (Δ_P)

For each existing query → target-paper example this notebook

1. runs the trained SciGraphIR checkpoint with CCMP on, extracts the **top-3 simple paths** from the query's seed nodes to the target with the GFM-RAG / NBFNet edge-gradient beam search, and keeps those paths fixed;
2. measures the target's **graph score and graph rank with CCMP on and with every gate set to 1** (same checkpoint);
3. records the normalised CCMP gates of the native run, **replays them** (must reproduce the native scores), then for each path **resets only that path's unique sender nodes' gates to 1 at every layer**, keeps every other recorded gate fixed (no renormalisation, no recomputation), and recomputes the target score:

   Δ_P = graph score (CCMP on) − graph score (after resetting that path's senders).

Δ_P is a *path-sender gate effect*, not the exclusive contribution of the path: senders act on other routes too, and overlapping paths must not be summed. The attribution column (mean edge gradient) is independent of the gates. A higher target score does not necessarily improve its rank; ranks are reported separately.

`PREFER_MERGED = False` selects the frame-graph checkpoints that produced the existing examples (`outputs/sir4_zeroshot/scigraphir_<pair>_qwenmlp_ccmp_e10_b2` on `<dataset>_test_v16sc`). Both float32 and bfloat16 are run from fresh loads; report float32 and use bfloat16 only to confirm that the stored 8 Sep paths reproduce. Inference and gradients only; no training. All source files are visible `%%writefile` cells.
"""

READOUT = r"""### Reading the result

- One compact table per example: path, attribution (mean edge gradient, GFM-RAG convention), Δ_P (graph score), graph rank after the reset, verdict. Above it: query, target paper, graph rank with CCMP on / all gates 1.
- `supported`: resetting the senders' gates lowered the target's graph score by more than the numerical tolerance, so those senders' gating supported the score in this intervention. `reduced`: the reset raised the score. `numerically_unresolved`: |Δ_P| is within the repeat variation of the forward pass.
- The union row (all three paths' senders reset together) is a separate measurement; it is not the sum of the Δ_P values, and shared senders are listed.
- Frozen replay must reproduce the native run (bit-identical applied gates, identical ranks, scores within the recorded allowance) before any Δ_P is reported.
- These are selected examples, not a population estimate; the gate ablation at inference is not the training-time CCMP effect.

Download `outputs/ccmp_path_sender/latest_*` into `experiments/results/qualitative/ccmp_path_sender_effect/drive/` and re-render locally with `python3 eval/ccmp_path_sender_effect.py <results.json> <out_dir>`.
"""


ENGINE_INSTALL_LIVE = r'''# Install the adapted GFM-RAG checkout and runtime dependencies (live output).
archive = Path(DRIVE) / "gfm-rag-adapted.zip"
assert archive.is_file() and zipfile.is_zipfile(archive), f"missing or corrupt {archive}"
engine_directory = Path("/content/gfm-rag")
if engine_directory.exists():
    shutil.rmtree(engine_directory)
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall("/content")
assert (engine_directory / "gfmrag").is_dir(), "archive did not create /content/gfm-rag"

# The engine's pyproject pins python <3.13 and current Colab runtimes are 3.13. The pin is metadata
# only (the code runs), so tell pip to ignore it; if the editable install still fails, a .pth file
# makes the package importable (PYTHONPATH already covers the workflow subprocesses).
if sh([sys.executable, "-m", "pip", "install", "--no-deps", "--ignore-requires-python", "-e",
       str(engine_directory)], "/content", check=False):
    import site
    Path(site.getsitepackages()[0], "gfmrag_dev.pth").write_text(str(engine_directory) + "\n")
    print("editable install failed; registered /content/gfm-rag through a .pth file instead")
# TORCH IS DELIBERATELY NOT IN THIS LIST: Colab ships a matched torch/torchvision pair.
sh([sys.executable, "-m", "pip", "install", "torch-geometric", "sentence-transformers", "transformers",
    "hydra-core", "omegaconf", "easydict", "ninja", "faiss-cpu", "pymetis", "wandb", "tqdm", "numpy",
    "pandas", "python-dotenv", "langchain-community"], "/content")

import torch
import torchvision
from transformers import PreTrainedModel   # the import that has to work; never uninstall torchvision
assert torch.cuda.is_available(), "Select an A100 or L4 GPU runtime"
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| GPU", torch.cuda.get_device_name())

# Repair the Drive zip (21 Jun build) in place: two files missing, two importing vLLM unconditionally (see the
# %%writefile cells of section 4, which write the same repo copies again). Without this, `import gfmrag` fails
# with "No module named 'vllm'" and hydra cannot find 'wandb/default'.
import json
REPAIR = json.loads(r"""__REPAIR_JSON__""")
for relative, content in REPAIR.items():
    path = engine_directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
print("repaired", len(REPAIR), "files of the Drive zip:", ", ".join(REPAIR))
sh([sys.executable, "-c", "import gfmrag, sys; print('gfmrag import ok from', gfmrag.__file__, '| python', sys.version.split()[0])"],
   "/content")
'''

IMPORT_CHECK = r'''# Import check: the source cells above rewrote the repaired files with identical repo copies.
sh([sys.executable, "-c", "import gfmrag, sys; print('gfmrag import ok from', gfmrag.__file__, '| python', sys.version.split()[0])"],
   "/content")
'''


def build():
    cases_file = S4 / "eval/ccmp_path_sender_cases.json"
    assert cases_file.is_file(), "run python3 eval/ccmp_path_sender_existing.py first (writes the cases file)"
    cases = json.loads(cases_file.read_text())
    cells = base.common_cells(INTRO, ["float32", "bfloat16"])
    config = "".join(cells[2]["source"])
    for old, new in (("PREFER_MERGED = True", "PREFER_MERGED = False"), ("TOP_PATHS = 5", "TOP_PATHS = 3")):
        assert config.count(old) == 1, old
        config = config.replace(old, new)
    cells[2] = base.code(config)
    assert "".join(cells[6]["source"]).startswith("# Install the adapted GFM-RAG checkout"), "install cell moved"
    repair = {relative: source.read_text() for relative, source in REPAIR_FILES}
    repair_json = json.dumps(repair, indent=1)
    assert '"""' not in repair_json, "repair file contents would break the raw triple-quoted literal"
    install = ENGINE_INSTALL_LIVE.replace("__REPAIR_JSON__", repair_json)
    cells[6] = base.code(install)   # Python 3.13 runtimes: --ignore-requires-python, live output, zip repair
    for relative, source in EXTRA_SOURCES:
        cells.append(base.writefile_cell(f"/content/gfm-rag/{relative}", source))
    cells.append(base.code(IMPORT_CHECK))   # after every %%writefile cell, before the engine is used
    init = base.initialization_cell(cases)
    init_src = "".join(init["source"]).replace(
        'print("Loaded", len(CASES), "question/target examples. The path notebook ignores their old paths.")',
        'print("Loaded", len(CASES), "query/target examples; paths are re-discovered from the checkpoint, the stored '
        '8 Sep paths are only a reproduction reference.")')
    init = base.code(init_src)
    driver, final = (S4 / "colab_cells/colab_ccmp_path_sender_cell.py").read_text().split("# --- combined summary", 1)
    final = final.partition("\n")[2]
    cells += [base.md("## 6. Select checkpoints and prepare graph-aligned inputs"), init,
              base.code((S4 / "colab_cells/colab_analysis_setup_cell.py").read_text()),
              base.md("## 7. Run the path-sender gate effect"), base.code(driver),
              base.md("## 8. Summary across datasets and precisions"), base.code("# Combined summary\n" + final),
              base.md(READOUT)]
    extra_hash = hashlib.sha256("".join(s.read_text() for _, s in EXTRA_SOURCES).encode()).hexdigest()
    notebook = base.finalize_notebook(cells, "ccmp-path-sender", {"mode": "path_senders", "extra_source_sha256": extra_hash,
                                                                  "cases": [c["name"] for c in cases]})
    OUT.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {OUT}: {len(cells)} transparent cells, {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    build()
