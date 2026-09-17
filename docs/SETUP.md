# Setup and data

[Repository home](../README.md) · [Code guide](README.md) · [Experiments](../experiments/README.md)

## Environment

Use **Python 3.12** for the combined pipeline. The graph engine declares
`>=3.12,<3.13`, and the graph builder uses f-string syntax that fails on Python 3.11.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export SCIGRAPHIR_ROOT="$PWD"
python -m pip install -r requirements.txt
```

For graph training, install PyTorch and PyTorch Geometric appropriate to the GPU runtime,
then install the fork:

```bash
python -m pip install -e retriever/gfm-rag
```

The root requirements are currently unpinned. The engine has its own
[pyproject.toml](../retriever/gfm-rag/pyproject.toml) and Poetry lockfile. These instructions
describe the current dependency declarations. A portable, locked full-pipeline environment
still needs to be established.

| Variable | Purpose |
|---|---|
| `SCIGRAPHIR_ROOT` | Absolute path to this checkout. Set it for engine and Colab runs. |
| `SCIGRAPHIR_DATASET` | Optional default dataset for scripts using the shared path resolver. |
| `OPENAI_API_KEY` | Affordance representation extraction, hypothetical-answer generation, and relevant LLM experiments. |
| `DEEPSEEK_API_KEY` | SIR-4 decomposition and alternative-set construction. |
| `EXTERNAL_REPOS` | Third-party checkouts used by dataset adapters and baseline/downstream wrappers. |

Cached affordance representations and answers allow graph construction and scoring preparation without
repeating the corresponding LLM calls. GPU compute and model downloads may still be needed.

## Data and artifacts

The repository includes the [SIR-4 dataset](../sir-4/dataset/README.md), stored as compressed
JSON with checksums. It contains all eight available field/split exports, including corpora,
queries, gold inspiration sets and manifests. The available Physics training export differs
from the count printed in the thesis; the dataset guide records the exact counts.

From the repository root, verify and unpack it using only the Python standard library:

```bash
python sir-4/prepare_data.py
python experiments/prep/stage_sir4.py --domain physics --verify-only
python experiments/prep/stage_sir4.py --domain physics
```

Use `--domain cs|biology|physics|matsci` on `prepare_data.py` to unpack one field, or
`--verify-only` to check the packaged files without writing runtime data. Staging creates
the corpus/query layout consumed by retrieval scripts.

Extraction caches, embeddings, model checkpoints, most per-query predictions and third-party
benchmark data are separate. [make_release_data.py](../experiments/prep/make_release_data.py)
packages these existing local SIR-4 artifacts for sharing; it does not obtain missing ones.

| Artifact | Expected location |
|---|---|
| Included SIR-4 dataset | [sir-4/dataset/](../sir-4/dataset/README.md) |
| Staged corpus and queries | `retriever/data/<dataset>_<split>/raw/` |
| Paper/problem requirement representations caches | `sciafford/cache/<dataset>/` |
| Hypothetical-answer caches | `scigraphir_paths.answers_path(split)` resolves the dataset cache. |
| Default SciAfford graph | `retriever/data/<dataset>_<split>_hyb/processed/stage1/` |
| Affordance component | `retriever/data/<dataset>_<split>_v16sc/processed/stage1/` |
| OpenIE component | `retriever/data/<dataset>_<split>/processed/stage1/` |
| Benchmark exports and validated sets | `sir-4/data/benchmark/<export>/` |
| Embeddings and training outputs | `outputs/` and experiment-specific runtime directories |

TOMATO retains legacy cache names; use [scigraphir_paths.py](../scigraphir_paths.py) rather
than constructing its paths by hand. The [benchmark guide](../sir-4/README.md) lists the
four SIR-4 export names.

## Choose a workflow

| Task | Entry point |
|---|---|
| Prepare SIR-4 data and graphs | [SIR-4 preparation runbook](../experiments/RUNBOOK.md) |
| Build affordance representations and graph structure | [SciAfford](../sciafford/README.md) |
| Train or compare retrieval models | [Notebook index](../experiments/notebooks/README.md) |
| Prepare MIR | [MIR runbook](../experiments/MIR_RUNBOOK.md) |
| Score saved rankings | [Evaluation guide](../experiments/eval/README.md) |
| Construct SIR-4 from source papers | [Benchmark](../sir-4/README.md) |

For the full SIR-4 method, use the [default merged-graph notebook](../experiments/notebooks/colab_sir4_hyb.ipynb).
Its `_hyb` graphs come from [build_hybrid_graph.py](../experiments/prep/build_hybrid_graph.py);
the construction sequence is in the [graph guide](../sciafford/README.md).

The Colab workflows expect a dataset bundle, `gfm-rag-adapted.zip`, and, for resumed runs,
the matching checkpoints on Drive. [bundle.py](../experiments/prep/bundle.py) creates the
dataset bundle; it does not create the separate engine archive.

Checked-in notebooks also embed and patch engine source. In the reviewed Physics CCMP
notebook, four embedded engine files differ from the standalone source. Record the notebook,
engine version, graph, scorer checkpoint, and configuration together when reproducing a run.

## Verification

The offline reference-matching regression can run without data downloads or LLM calls once
its Python dependencies are installed:

```bash
python -B sir-4/build/test_matching.py
```

The analysis suite uses PyTorch and small CPU graph fixtures:

```bash
python -B -m unittest discover -s experiments/eval -p 'test_*.py' -v
```

## Reproduction work still required

- Publish the remaining experiment artifacts and a manifest tying each table to its graph,
  scorer, checkpoint, predictions, and configuration. The SIR-4 data is already included.
- Consolidate notebook snapshots and runtime patches into the source implementation.

The source directory is now `sir-4/`. Rebuild dataset bundles created with the earlier
`benchmark/` layout. Dataset keys such as `sir4_physics` and the inner `data/benchmark/`
export directory retain their existing names.
