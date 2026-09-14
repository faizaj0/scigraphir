"""
cargo_paths.py -- one place that decides WHICH corpus a pipeline script is
working on, and where that corpus's caches live.

THE PROBLEM THIS SOLVES. Five scripts in the construction pipeline had the
TOMATO corpus baked in as the literal string "tomato", and their caches were
keyed by SPLIT ALONE:

    kg-construction-v16/cache/frames_doc.jsonl          <- which corpus?
    kg-construction/construct_v2/cache/probes_test.jsonl
    outputs/caches/op_emb/test_doc.npy

Point those scripts at a second corpus and a leftover TOMATO `test_doc.npy`
loads silently: the shape check passes whenever the two corpora happen to have a
similar document count, and the run produces plausible, meaningless numbers. No
exception, no warning. That is the failure this module exists to prevent.

THE CONTRACT. With the default dataset "tomato" every path returned here is
byte-identical to the hardcoded string it replaced, so existing caches still
resolve and published TOMATO numbers stay reproducible. Any other dataset name
additionally scopes every cache into a subdirectory named after it, so two
corpora can never share a cache file.

    DATASET=tomato   ->  .../cache/frames_doc.jsonl          (unchanged)
    DATASET=sir4_cs  ->  .../cache/sir4_cs/frames_doc_test.jsonl

Scripts call `set_dataset()` once after parsing args, then use the helpers.

    import sys, os
    sys.path.insert(0, os.path.expanduser("~/Desktop/CARGO"))
    from cargo_paths import set_dataset, corpus_dir, frames_path

    set_dataset(args.dataset)
    raw = corpus_dir(args.split)              # .../data/sir4_cs_test/raw
"""
from __future__ import annotations

import os

# Repo root. CARGO_ROOT lets the same scripts run somewhere that is not this
# checkout -- Colab unzips the bundle to /content/cargo. Unset, the root is the
# directory that contains this file, i.e. the repository root.
CARGO = os.environ.get("CARGO_ROOT") or os.path.dirname(os.path.abspath(__file__))
KG = f"{CARGO}/kg-construction"
V16 = f"{CARGO}/kg-construction-v16"

#: Corpus currently being processed. "tomato" reproduces every legacy path.
DATASET = os.environ.get("CARGO_DATASET", "tomato")


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
    return f"{KG}/data/{corpus_name(split)}"


def graph_name(split: str, suffix: str = "v16sc") -> str:
    """Directory name of a built graph, e.g. `sir4_cs_train_v16sc`."""
    return f"{DATASET}_{split}_{suffix}"


def graph_dir(split: str, suffix: str = "v16sc") -> str:
    return f"{KG}/data/{graph_name(split, suffix)}"


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
    root = _scoped(f"{V16}/cache")
    if is_legacy() and split == "test":
        return f"{root}/frames_{side}.jsonl"
    return f"{root}/frames_{side}_{split}.jsonl"


def graph_cache_dir() -> str:
    """Scratch for graph-construction caches, e.g. `ds_{split}_emb_{type}.npy`.

    Those were keyed by split alone too. A SIR-4 build silently loaded TOMATO's
    concept embeddings from `ds_test_emb_method.npy`, which is part of why a
    contaminated build produced TOMATO-shaped output without complaining.
    """
    return _scoped(f"{V16}/cache")


def probes_path(split: str) -> str:
    """LLM probe cache used by the operator's S and M terms."""
    return f"{_scoped(f'{KG}/construct_v2/cache')}/probes_{split}.jsonl"


def emb_dir() -> str:
    """BGE embedding cache (`{split}_doc.npy`, `{split}_query_{hash}.npy`, ...).

    The most dangerous of the three: filenames carry only the split, and a
    document matrix from the wrong corpus fails no assertion that the callers
    make.
    """
    return _scoped(f"{CARGO}/outputs/caches/op_emb")


def add_dataset_arg(ap) -> None:
    """Attach the standard `--dataset` flag to an argparse parser."""
    ap.add_argument(
        "--dataset", default=os.environ.get("CARGO_DATASET", "tomato"),
        help="corpus to operate on (default tomato; e.g. sir4_cs). "
             "Anything but 'tomato' also scopes every cache under this name.")


def banner() -> str:
    return (f"[cargo_paths] dataset={DATASET} "
            f"{'(legacy TOMATO paths)' if is_legacy() else '(scoped caches)'}")
