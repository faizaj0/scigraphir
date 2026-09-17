#!/usr/bin/env python3
"""Build transparent, self-contained Colab notebooks for CCMP analysis.

Every installed source file is a visible %%writefile cell. There are no base64
blobs, encoded archives, SOURCE_FILES dictionaries, or executable source strings.
"""
import hashlib
import json
from pathlib import Path
import pprint


S4 = Path(__file__).resolve().parents[1]
REPO_ROOT = S4.parent
ENGINE = REPO_ROOT / "retriever/gfm-rag"
OUT = S4 / "notebooks/colab_table4_openie.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


INTRO = r"""# Why does CCMP change path attribution?

This notebook tests whether suppressing messages outside selected paths is sufficient to increase those paths' attribution. It runs inference and gradients only.

All implementation is visible below as normal Python and `%%writefile` cells. There are no encoded source blobs or runtime source decoding.

Merged-graph checkpoints are preferred together with their matching `_test_hyb` graph and field scorer. If no merged checkpoint exists under `outputs/sir4_hyb/`, the notebook prints an explicit fallback and uses the previous affordance representation checkpoint.

The main contrast is `outside_suppress − off`. A positive value means that suppressing outside messages was sufficient to raise attribution for that fixed path in that example. Path weights are mean edge gradients, not probabilities or products of gates. Retrieval scores and ranks are reported separately.
"""


READOUT = r"""### Reading the result

- `outside_suppress − off > 0` supports the competing-message suppression explanation for that path and example.
- `native − frozen` measures the contribution of differentiating through the gates while replaying identical applied gates.
- `path_only − off` measures the effect of retaining native gates only on the selected path nodes.
- A larger path weight does not imply a better retrieval rank.

Frozen replay requires bit-identical raw/applied gates and identical target ranks. CUDA sparse reductions can vary between equivalent bfloat16 passes, so score checks record repeat variation and an explicit dtype-and-score-scale allowance. Float32 retains the tight configured tolerance. These are numerical checks, not confidence intervals.
"""


CONFIG_TEMPLATE = r'''# Edit only this block for the usual run.
DATASETS = ["sir4_biology", "sir4_cs"]
PREFER_MERGED = True
PRECISIONS = __PRECISIONS__
PROTECTION = "nodes_all_layers"
RESUME = True
ATOL, RTOL = 1e-5, 1e-4
TOP_PATHS = 5
BEAM_SIZE = 10

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
os.environ["PYTHONPATH"] = "/content/gfm-rag" + os.pathsep + os.environ.get("PYTHONPATH", "")

from google.colab import drive
drive.mount("/content/drive")

DRIVE = "/content/drive/MyDrive/cargo-gfmrag"
SCIGRAPHIR_ROOT = "/content/scigraphir"
DATA_ROOT = f"{SCIGRAPHIR_ROOT}/retriever/data"
S4 = f"{SCIGRAPHIR_ROOT}/experiments"
KGDIR = f"{SCIGRAPHIR_ROOT}/retriever"
RUNS = "/content/runs"
OP_MODEL = "/content/qwen3"
OP_SLUG = "_content-qwen3"
NEED_ENGINE = True
os.environ["SCIGRAPHIR_ROOT"] = SCIGRAPHIR_ROOT
sys.path.insert(0, SCIGRAPHIR_ROOT)
os.makedirs(RUNS, exist_ok=True)

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    os.environ.setdefault("HF_TOKEN", "")

env = dict(os.environ, SCIGRAPHIR_ROOT=SCIGRAPHIR_ROOT, PYTHONUNBUFFERED="1")


def sh(command, cwd, extra=None, log=None, check=True):
    """Run a command and stream its output."""
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(env, **(extra or {})),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_stream = open(log, "w") if log else None
    for raw_line in iter(process.stdout.readline, b""):
        line = raw_line.decode("utf-8", "replace")
        print(line, end="")
        if log_stream:
            log_stream.write(line)
            log_stream.flush()
    process.wait()
    if log_stream:
        log_stream.close()
    print(f"[{time.time() - started:.0f}s, exit {process.returncode}]")
    if check and process.returncode:
        raise RuntimeError("Command failed: " + " ".join(map(str, command)))
    return process.returncode


def sir4_spec(field):
    pair = "physics+biology" if field in ("physics", "biology") else "cs+matsci"
    warm = "biology" if field in ("physics", "biology") else "cs"
    dataset = f"sir4_{field}"
    return {
        "frame": f"{dataset}_test_v16sc",
        "sem": {
            "field": (f"outputs/{dataset}/semantic", dataset),
            "frame": (f"outputs/sir4_zeroshot/semantic_sir4_{warm}", f"sir4_{warm}"),
        },
        "arms": [(
            "frame_ccmp",
            f"outputs/sir4_zeroshot/scigraphir_{pair}_qwenmlp_ccmp_e10_b2",
            "frame", "frame", True,
        )],
    }


SPEC = {f"sir4_{field}": sir4_spec(field)
        for field in ("cs", "biology", "physics", "matsci")}
assert DATASETS and all(dataset in SPEC for dataset in DATASETS)
assert all(precision in ("bfloat16", "float32") for precision in PRECISIONS)
assert PROTECTION in ("nodes_all_layers", "sender_layer")
assert BEAM_SIZE >= TOP_PATHS >= 1
print("Datasets:", DATASETS)
print("Precisions:", PRECISIONS)
print("Prefer merged checkpoints:", PREFER_MERGED)
'''


UNPACK = r'''# Restore the selected corpora under /content/scigraphir.
FORCE_UNPACK = False
marker = Path(SCIGRAPHIR_ROOT) / ".unpacked.json"
available = set(json.loads(marker.read_text())) if marker.exists() and not FORCE_UNPACK else set()
missing = [dataset for dataset in DATASETS if dataset not in available]

if missing:
    cache_directory = Path(SCIGRAPHIR_ROOT) / "outputs/caches"
    parked_cache = Path("/content/_cargo_caches_keep")
    if available:
        for dataset in missing:
            archive = Path(DRIVE) / f"{dataset}_bundle.zip"
            assert archive.is_file() and zipfile.is_zipfile(archive), f"missing or corrupt {archive}"
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(SCIGRAPHIR_ROOT)
            print("unpacked", archive.name)
    else:
        if cache_directory.exists():
            if parked_cache.exists():
                shutil.rmtree(parked_cache)
            shutil.move(cache_directory, parked_cache)
        cargo_directory = Path(SCIGRAPHIR_ROOT)
        if cargo_directory.exists():
            shutil.rmtree(cargo_directory)
        cargo_directory.mkdir(parents=True)
        for dataset in DATASETS:
            archive = Path(DRIVE) / f"{dataset}_bundle.zip"
            assert archive.is_file() and zipfile.is_zipfile(archive), f"missing or corrupt {archive}"
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(SCIGRAPHIR_ROOT)
            print("unpacked", archive.name)
        if parked_cache.exists():
            cache_directory.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(parked_cache, cache_directory)
    marker.write_text(json.dumps(sorted(available | set(DATASETS))))
else:
    print("bundles already unpacked:", sorted(available))
'''


ENGINE_INSTALL = r'''# Install the adapted GFM-RAG checkout and runtime dependencies.
archive = Path(DRIVE) / "gfm-rag-adapted.zip"
assert archive.is_file() and zipfile.is_zipfile(archive), f"missing or corrupt {archive}"
engine_directory = Path("/content/gfm-rag")
if engine_directory.exists():
    shutil.rmtree(engine_directory)
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall("/content")
assert (engine_directory / "gfmrag").is_dir(), "archive did not create /content/gfm-rag"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e",
                str(engine_directory)], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch-geometric", "sentence-transformers", "transformers", "hydra-core",
    "omegaconf", "easydict", "ninja", "faiss-cpu", "pymetis", "wandb",
    "tqdm", "numpy", "pandas", "python-dotenv", "langchain-community",
], check=True)

import torch
assert torch.cuda.is_available(), "Select an A100 or L4 GPU runtime"
print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name())
'''


PROJECT_SOURCES = [
    ("/content/scigraphir/scigraphir_paths.py", REPO_ROOT / "scigraphir_paths.py"),
    ('/content/scigraphir/retriever/eval/handcrafted_scorer.py',
     REPO_ROOT / 'retriever/eval/handcrafted_scorer.py'),
    ("/content/scigraphir/experiments/eval/semantic_scorer.py",
     S4 / "eval/semantic_scorer.py"),
    ('/content/scigraphir/retriever/precompute/precompute_handcrafted_components.py',
     REPO_ROOT / 'retriever/precompute/precompute_handcrafted_components.py'),
    ("/content/scigraphir/retriever/precompute/precompute_semantic_components.py",
     REPO_ROOT / "retriever/precompute/precompute_semantic_components.py"),
]

ENGINE_SOURCES = [
    ("gfmrag/models/cqig.py", ENGINE / "gfmrag/models/cqig.py"),
    ("gfmrag/models/fusion_reasoner.py", ENGINE / "gfmrag/models/fusion_reasoner.py"),
    ("gfmrag/models/gfm_reasoner/model.py", ENGINE / "gfmrag/models/gfm_reasoner/model.py"),
    ("gfmrag/models/ultra/base_nbfnet.py", ENGINE / "gfmrag/models/ultra/base_nbfnet.py"),
    ("gfmrag/models/ultra/layers.py", ENGINE / "gfmrag/models/ultra/layers.py"),
    ("gfmrag/models/ultra/models.py", ENGINE / "gfmrag/models/ultra/models.py"),
    ("gfmrag/trainers/base_trainer.py", ENGINE / "gfmrag/trainers/base_trainer.py"),
    ("gfmrag/trainers/fusion_trainer.py", ENGINE / "gfmrag/trainers/fusion_trainer.py"),
    ("gfmrag/workflow/config/gfm_reasoner/sft_training_fusion.yaml",
     ENGINE / "gfmrag/workflow/config/gfm_reasoner/sft_training_fusion.yaml"),
    ("gfmrag/workflow/ccmp_checkpoint_selection.py", S4 / "eval/ccmp_checkpoint_selection.py"),
    ("gfmrag/workflow/ccmp_mechanism_core.py", S4 / "eval/ccmp_mechanism.py"),
    ("gfmrag/workflow/ccmp_mechanism.py", S4 / "eval/ccmp_mechanism_workflow.py"),
    ("gfmrag/workflow/scientific_paths.py", S4 / "eval/scientific_path_interpretations.py"),
]


def writefile_cell(destination, source):
    return code(f"%%writefile {destination}\n{source.read_text()}")


def common_cells(intro, precisions):
    config = CONFIG_TEMPLATE.replace("__PRECISIONS__", repr(precisions))
    showcase = json.loads((S4 / "notebooks/colab_showcase_all.ipynb").read_text())
    qwen = "".join(showcase["cells"][6]["source"])
    qwen = qwen.replace("# Only runs when cell 1 found a scan or a path search still to do; drawing the figures from\n"
                        "# cached interpretations needs neither the engine nor Qwen3.\n",
                        "# Restore Qwen3-Embedding from Drive, or download and cache it once.\n")
    cells = [md(intro), md("## 1. Configuration"), code(config),
             md("## 2. Restore data bundles"), code(UNPACK),
             md("## 3. Install the runtime"), code(ENGINE_INSTALL),
             md("## 4. Actual source code used by this notebook\n\nEach following cell writes one visible source file. You can search, edit and review it directly.")]
    for destination, source in PROJECT_SOURCES:
        cells.append(writefile_cell(destination, source))
    for relative, source in ENGINE_SOURCES:
        cells.append(writefile_cell(f"/content/gfm-rag/{relative}", source))
    cells += [md("## 5. Restore the embedding model"), code(qwen)]
    return cells


def initialization_cell(cases):
    cases_literal = pprint.pformat(cases, width=110, sort_dicts=False)
    return code(r'''# Import the visible modules written above and patch the analysis hook.
import importlib.util

core_path = Path("/content/gfm-rag/gfmrag/workflow/ccmp_mechanism_core.py")
core_spec = importlib.util.spec_from_file_location("ccmp_mechanism_core", core_path)
_core = importlib.util.module_from_spec(core_spec)
core_spec.loader.exec_module(_core)
_core.patch_engine("/content/gfm-rag/gfmrag/models/ultra/models.py")

selection_path = Path("/content/gfm-rag/gfmrag/workflow/ccmp_checkpoint_selection.py")
selection_spec = importlib.util.spec_from_file_location("ccmp_checkpoint_selection", selection_path)
_selection = importlib.util.module_from_spec(selection_spec)
selection_spec.loader.exec_module(_selection)

CASES = __CASES__
assert all(any(case["dataset"] == dataset for case in CASES) for dataset in DATASETS)
print("Loaded", len(CASES), "question/target examples. The path notebook ignores their old paths.")
'''.replace("__CASES__", cases_literal))


def finalize_notebook(cells, prefix, metadata):
    for index, cell in enumerate(cells):
        cell["id"] = f"{prefix}-{index:02d}"
        cell["source"] = "".join(cell["source"]).splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    source_hash = hashlib.sha256("".join(
        source.read_text() for _, source in PROJECT_SOURCES + ENGINE_SOURCES).encode()).hexdigest()
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU",
                         "ccmp_mechanism": {"version": "transparent-v2",
                                            "source_sha256": source_hash, **metadata}}}


def build():
    cases = json.loads((S4 / "eval/ccmp_mechanism_cases.json").read_text())
    driver, final = (S4 / "colab_cells/colab_ccmp_mechanism_cell.py").read_text().split(
        "# --- combined precision comparison", 1)
    final = final.partition("\n")[2]
    cells = common_cells(INTRO, ["bfloat16", "float32"])
    cells += [md("## 6. Select checkpoints and prepare graph-aligned inputs"),
              initialization_cell(cases), code((S4 / "colab_cells/colab_analysis_setup_cell.py").read_text()),
              md("## 7. Run fixed-path interventions"), code(driver),
              md("## 8. Compare precision and inspect evidence"),
              code("# Combined precision comparison\n" + final), md(READOUT)]
    notebook = finalize_notebook(cells, "ccmp-transparent", {"mode": "interventions"})
    OUT.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {OUT}: {len(cells)} transparent cells, {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    build()
