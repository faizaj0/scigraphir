"""
precompute_semantic_components.py — cache what the LEARNED semantic scorer needs inside the
graph fusion, aligned to the graph's nodes.csv document order.

The handcrafted scorer version of this script (precompute_handcrafted_components.py) caches S and M, the two
handcrafted summaries of the hypothetical-answer matches. The sorted-MLP does not use summaries:
it reads the whole ordered match profile, so it needs the FULL per-answer matrix

    H  [Q, Jmax, n_doc]   float16   ReLU(cos(hypothetical answer j, document d))

plus the direct query-document similarity, the answer-validity mask, and the document embeddings
that the background matchability predictor p_hat(d) = softplus(g(E(d))) reads.

H IS NOT COPIED. semantic_scorer.py already materialises exactly this matrix as a memmap under
outputs/caches/semantic/<dataset>/, and at CS scale it is 1.76 GB — duplicating it per graph would
cost more disk than every other artefact in the project combined, and would introduce a second copy
that can go stale. This script instead records the memmap's PATH and the column permutation `col`
that maps corpus document order to nodes.csv document order, and the fusion applies `col` to the
few rows it slices per batch. Everything small (dense, mask, doc_emb, total_S) is reordered here
and stored outright, because those are cheap and reordering them per batch would not be.

Run it AFTER the semantic scorer (notebook section 5d), which is what builds the memmap and trains
the checkpoint the fusion warm-starts from. If the memmap is absent this rebuilds it from the
cached embeddings; if the embeddings are absent too it will encode, which is the slow path.

  python precompute/precompute_semantic_components.py       --dataset sir4_physics --model /content/qwen3 --graph sir4_physics_train_v16sc --split train
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

# SCIGRAPHIR_ROOT so this runs off-laptop (Colab unzips to /content/scigraphir). Unset, it
# resolves to exactly the path this replaced.
_ROOT = os.environ.get("SCIGRAPHIR_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BASE = f"{_ROOT}/retriever"
sys.path.insert(0, BASE)
sys.path.insert(0, _ROOT)
from scigraphir_paths import add_dataset_arg, banner, corpus_dir, emb_dir, set_dataset  # noqa: E402

csv.field_size_limit(10 ** 7)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="graph dir supplying nodes.csv document order")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--force", action="store_true", help="rebuild H even if cached")
    add_dataset_arg(ap)
    a = ap.parse_args()
    set_dataset(a.dataset)
    print(banner())

    op = _load("op", f"{BASE}/eval/handcrafted_scorer.py")
    sem = _load("sem", f"{_ROOT}/experiments/eval/semantic_scorer.py")
    slug = op.model_slug(a.model)

    # ---- the graph's document order -------------------------------------------------
    s1 = f"{BASE}/data/{a.graph}/processed/stage1"
    docnodes = [n["name"] for n in csv.DictReader(open(f"{s1}/nodes.csv")) if n["type"] == "document"]
    corpus = json.load(open(f"{corpus_dir(a.split)}/raw/documents.json"))
    doc_ids = list(corpus)
    d2i = {d: i for i, d in enumerate(doc_ids)}
    assert all(n in d2i for n in docnodes), "graph has document nodes not in documents.json"
    col = np.asarray([d2i[name] for name in docnodes], dtype=np.int64)   # corpus -> nodes.csv

    # ---- the semantic scorer's own inputs, in CORPUS order ---------------------------
    cache = f"{_ROOT}/outputs/caches/semantic/{a.dataset}"
    data = sem.build_inputs(op, a.model, a.split, cache, force=a.force)
    meta = data["meta"]
    Q, Jmax, D = data["H"].shape
    assert D == len(doc_ids), f"H has {D} document columns, corpus has {len(doc_ids)}"
    assert len(data["queries"]) == Q

    # The memmap path is reconstructed from the SAME key build_inputs used, so a cache
    # written under a different encoder or a reordered corpus cannot be picked up here.
    tag = f"{a.split}_{meta['qkey']}_{meta['dockey']}{slug}"
    h_path = f"{cache}/semantic_H_{tag}.f16"
    assert os.path.exists(h_path), f"H memmap missing: {h_path}"

    # ---- document embeddings, for the background matchability predictor ---------------------------
    de = np.load(f"{emb_dir()}/{a.split}_doc{slug}.npy").astype(np.float32)
    assert de.shape[0] == D, f"doc embeddings {de.shape[0]} rows != {D} documents"

    qids = [q["id"] for q in data["queries"]]
    raw_qids = [q["id"] for q in json.load(open(f"{corpus_dir(a.split)}/raw/{a.split}.json"))]
    assert qids == raw_qids, "build_inputs query order differs from the raw split order"
    assert meta["qkey"] == hashlib.md5("|".join(qids).encode()).hexdigest()[:8]

    out = f"{BASE}/data/{a.graph}/semantic_components{slug}.npz"
    np.savez_compressed(
        out,
        # reordered to nodes.csv document order, exactly like the handcrafted scorer components
        dense=np.asarray(data["dense"])[:, col].astype(np.float16),
        doc_emb=de[col].astype(np.float16),
        total_S=np.asarray(data["total_S"])[col].astype(np.float32),
        mask=np.asarray(data["mask"]),                       # [Q, Jmax] bool, order-independent
        # H stays where semantic_scorer.py put it; the fusion applies `col` per batch
        col=col, h_path=np.array(h_path), h_shape=np.array([Q, Jmax, D]),
        query_ids=np.array(qids), Jmax=np.array(Jmax),
        encoder=np.array(a.model), query_instruct=np.array(meta["query_instruct"]),
        dockey=np.array(meta["dockey"]),
    )
    sz = os.path.getsize(out) / 1e6
    print(f"[sem-comp] saved {out}  ({sz:.0f} MB)")
    print(f"[sem-comp]   dense {(Q, len(col))} f16 | doc_emb {(len(col), de.shape[1])} f16 "
          f"| mask {(Q, Jmax)} bool")
    print(f"[sem-comp]   H referenced at {h_path} ({os.path.getsize(h_path)/1e6:.0f} MB, not copied)")
    print(f"[sem-comp] env:  SEMANTIC_COMPONENTS{'_TEST' if a.split == 'test' else ''}={out}")


if __name__ == "__main__":
    main()
