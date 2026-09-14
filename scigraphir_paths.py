"""
scigraphir_paths.py -- the one place that decides which corpus a script works on and where
that corpus's files live.

Every pipeline script takes --dataset (tomato, sir4_cs, sir4_biology, sir4_physics,
sir4_matsci, mir, researchbench) and resolves corpora, graphs and caches through the helpers
here, so two corpora can never share a cache file. Caches are keyed by dataset and split:

    corpus_dir("test")           ->  retriever/data/sir4_cs_test
    graph_dir("test")            ->  retriever/data/sir4_cs_test_v16sc
    frames_path("doc", "test")   ->  sciafford/cache/sir4_cs/frames_doc_test.jsonl
    probes_path("test")          ->  retriever/probes/cache/sir4_cs/probes_test.jsonl
    emb_dir()                    ->  outputs/caches/op_emb/sir4_cs

The repository root is SCIGRAPHIR_ROOT when set (Colab unpacks the bundle to
/content/scigraphir), otherwise the directory that holds this file.

TOMATO-Star keeps its original, unscoped cache names (frames_doc.jsonl, probes_test.jsonl,
outputs/caches/op_emb/test_doc.npy) so the published TOMATO-Star numbers reproduce from the
existing caches; every other dataset is scoped into a subdirectory named after it.

Usage in a script:

    import os, sys
    sys.path.insert(0, os.environ.get("SCIGRAPHIR_ROOT") or REPO_ROOT)
    from scigraphir_paths import add_dataset_arg, set_dataset, corpus_dir, frames_path

    set_dataset(args.dataset)
    raw = corpus_dir(args.split)              # .../retriever/data/sir4_cs_test
"""
from __future__ import annotations

import os

# Repo root. SCIGRAPHIR_ROOT lets the same scripts run somewhere that is not this
# checkout -- Colab unzips the bundle to /content/scigraphir. Unset, the root is the
# directory that contains this file, i.e. the repository root.
ROOT = os.environ.get("SCIGRAPHIR_ROOT") or os.path.dirname(os.path.abspath(__file__))
RETRIEVER = f"{ROOT}/retriever"
SCIAFFORD = f"{ROOT}/sciafford"

#: Corpus currently being processed. "tomato" reproduces every legacy path.
DATASET = os.environ.get("SCIGRAPHIR_DATASET", "tomato")


def set_dataset(name: str | None) -> str:
    """Set the active corpus. `None` or "" leaves the current value alone."""
    global DATASET
    if name:
        DATASET = name
    return DATASET


def is_legacy() -> bool:
    """True when paths must stay byte-identical to the pre-refactor strings."""
    return DATASET == "tomato"


def _scoped(root: str) -> str:
    """Cache root, with a per-dataset subdirectory for anything but TOMATO."""
    d = root if is_legacy() else f"{root}/{DATASET}"
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# corpora and graphs
# --------------------------------------------------------------------------
def corpus_name(split: str) -> str:
    """Directory name of a raw corpus, e.g. `tomato_test` / `sir4_cs_test`."""
    return f"{DATASET}_{split}"


def corpus_dir(split: str) -> str:
    """Absolute path to a raw corpus directory (holds raw/documents.json)."""
    return f"{RETRIEVER}/data/{corpus_name(split)}"


def graph_name(split: str, suffix: str = "v16sc") -> str:
    """Directory name of a built graph, e.g. `sir4_cs_train_v16sc`."""
    return f"{DATASET}_{split}_{suffix}"


def graph_dir(split: str, suffix: str = "v16sc") -> str:
    return f"{RETRIEVER}/data/{graph_name(split, suffix)}"


# --------------------------------------------------------------------------
# caches
# --------------------------------------------------------------------------
def frames_path(side: str, split: str) -> str:
    """Extracted-frame cache. `side` is "doc" or "query".

    TOMATO's test frames are stored WITHOUT a split suffix and its train frames
    WITH one -- an asymmetry from when only a test split existed. That is
    preserved exactly for TOMATO and dropped for every other dataset, where the
    split is always in the name.
    """
    root = _scoped(f"{SCIAFFORD}/cache")
    if is_legacy() and split == "test":
        return f"{root}/frames_{side}.jsonl"
    return f"{root}/frames_{side}_{split}.jsonl"


def graph_cache_dir() -> str:
    """Scratch for graph-construction caches, e.g. `ds_{split}_emb_{type}.npy`.

    Those were keyed by split alone too. A SIR-4 build silently loaded TOMATO's
    concept embeddings from `ds_test_emb_method.npy`, which is part of why a
    contaminated build produced TOMATO-shaped output without complaining.
    """
    return _scoped(f"{SCIAFFORD}/cache")


def probes_path(split: str) -> str:
    """LLM probe cache used by the operator's S and M terms."""
    return f"{_scoped(f'{RETRIEVER}/probes/cache')}/probes_{split}.jsonl"


def emb_dir() -> str:
    """BGE embedding cache (`{split}_doc.npy`, `{split}_query_{hash}.npy`, ...).

    The most dangerous of the three: filenames carry only the split, and a
    document matrix from the wrong corpus fails no assertion that the callers
    make.
    """
    return _scoped(f"{ROOT}/outputs/caches/op_emb")


def add_dataset_arg(ap) -> None:
    """Attach the standard `--dataset` flag to an argparse parser."""
    ap.add_argument(
        "--dataset", default=os.environ.get("SCIGRAPHIR_DATASET", "tomato"),
        help="corpus to operate on (default tomato; e.g. sir4_cs). "
             "Anything but 'tomato' also scopes every cache under this name.")


def banner() -> str:
    return (f"[scigraphir_paths] dataset={DATASET} "
            f"{'(legacy TOMATO paths)' if is_legacy() else '(scoped caches)'}")
