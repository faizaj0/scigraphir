"""Run plain BGE and query-seeded PPR on one SIR-4 graph.

PPR uses the graph's stored start_nodes, an untyped symmetric adjacency, a
uniform restart over unique seeds, and ranks document nodes by stationary
mass.  Predictions are emitted in the schema consumed by score_sir4.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
KG = ROOT / "retriever"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(KG))

from scigraphir_paths import corpus_dir, emb_dir, set_dataset  # noqa: E402
from eval.operator_scorer import BGE_QI, cached_encode  # noqa: E402


KS = (1, 5, 10, 100)


def predictions(rows: list[dict], doc_ids: list[str], scores: np.ndarray,
                positive_only: bool = False, topn: int = 100) -> list[dict]:
    out = []
    for qi, q in enumerate(rows):
        s = scores[qi]
        order = np.argsort(-s, kind="stable")
        if positive_only:
            order = order[s[order] > 0]
        order = order[:topn]
        out.append({
            "id": q["id"],
            "stratum": q.get("stratum"),
            "supporting_documents": q.get("supporting_documents", []),
            "predictions": {
                "document": [doc_ids[int(j)] for j in order]
            },
        })
    return out


def score(scores: np.ndarray, rows: list[dict], doc_ids: list[str],
          sets_by_q: dict[str, list[list[str]]], bge_best: list[int] | None,
          positive_only: bool) -> tuple[dict, list[int]]:
    """Exact full-corpus metrics; zero-mass PPR documents are unretrieved."""
    pos = {d: i for i, d in enumerate(doc_ids)}
    buckets: dict[str, list[dict]] = {}
    best_ranks = []
    for qi, q in enumerate(rows):
        s = scores[qi]
        order = np.argsort(-s, kind="stable")
        if positive_only:
            order = order[s[order] > 0]
        gold = {pos[d] for d in q.get("supporting_documents", []) if d in pos}
        rank = next((r for r, d in enumerate(order, 1) if int(d) in gold), 10 ** 9)
        best_ranks.append(rank)
        row = {"mrr": 1.0 / rank if rank < 10 ** 9 else 0.0}
        complete = [{pos[d] for d in set_ if d in pos} for set_ in sets_by_q.get(q["id"], [])]
        complete = [x for x in complete if x]
        for k in KS:
            top = set(map(int, order[:k]))
            row[f"hits@{k}"] = float(rank <= k)
            row[f"recall@{k}"] = len(top & gold) / len(gold) if gold else 0.0
            row[f"completeset@{k}"] = float(any(x <= top for x in complete)) if complete else 0.0
        names = ["all", q.get("stratum") or "?"]
        if bge_best is not None:
            difficulty = "dissimilar" if bge_best[qi] > 100 else "similar"
            names.extend((difficulty, f"{q.get('stratum')}+{difficulty}"))
        for name in names:
            buckets.setdefault(name, []).append(row)

    cols = ("mrr", "hits@1", "hits@5", "recall@5", "recall@10",
            "recall@100", "completeset@10", "completeset@100")
    result = {}
    for name, vals in buckets.items():
        result[name] = {"n": len(vals), **{
            c: float(np.mean([v[c] for v in vals])) for c in cols
        }}
    return result, best_ranks


def print_scores(label: str, result: dict) -> None:
    cols = ("mrr", "hits@1", "hits@5", "recall@5", "recall@10",
            "recall@100", "completeset@10", "completeset@100")
    print(f"\n### {label}")
    print("| slice | n | " + " | ".join(cols) + " |")
    print("|---|--:|" + "--:|" * len(cols))
    for name in ("all", "same", "cross", "similar", "dissimilar",
                 "same+dissimilar", "cross+dissimilar"):
        if name not in result:
            continue
        row = result[name]
        print(f"| {name} | {row['n']} | " + " | ".join(f"{row[c]:.4f}" for c in cols) + " |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sir4_cs")
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--graph", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--sets", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    set_dataset(a.dataset)

    graph_name = a.graph or f"{a.dataset}_{a.split}_v16sc"
    stage = KG / "data" / graph_name / "processed" / "stage1"
    raw = Path(corpus_dir(a.split)) / "raw"
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "experiments" / "results" / graph_name
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = json.load((raw / "documents.json").open())
    queries = json.load((raw / f"{a.split}.json").open())
    graph_queries = {q["id"]: q for q in json.load((stage / f"{a.split}.json").open())}
    doc_ids = list(corpus)
    doc_pos = {d: i for i, d in enumerate(doc_ids)}

    nodes = list(csv.DictReader((stage / "nodes.csv").open()))
    node_names = [n["name"] for n in nodes]
    node_pos = {n: i for i, n in enumerate(node_names)}
    graph_doc_nodes = [n["name"] for n in nodes if n["type"] == "document"]
    if set(graph_doc_nodes) != set(doc_ids):
        raise ValueError("Graph and corpus document sets differ")

    rr, cc = [], []
    for e in csv.DictReader((stage / "edges.csv").open()):
        s, t = node_pos[e["source"]], node_pos[e["target"]]
        if s != t:
            rr.extend((s, t))
            cc.extend((t, s))
    adj = sp.csr_matrix(
        (np.ones(len(rr), np.float32), (rr, cc)),
        shape=(len(nodes), len(nodes)),
    )
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    degree = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    degree[degree == 0] = 1.0
    transition = (sp.diags(1.0 / degree) @ adj).tocsr()
    graph_doc_rows = np.asarray([node_pos[d] for d in doc_ids], dtype=np.int64)

    ppr_scores = np.zeros((len(queries), len(doc_ids)), np.float32)
    seed_counts, zero_seed = [], 0
    for start in tqdm(range(0, len(queries), a.batch), desc="PPR"):
        batch = queries[start:start + a.batch]
        restart = np.zeros((len(batch), len(nodes)), np.float32)
        for bi, q in enumerate(batch):
            row = graph_queries[q["id"]]
            seeds = sorted({
                n for values in (row.get("start_nodes") or {}).values()
                for n in values if n in node_pos
            })
            seed_counts.append(len(seeds))
            if seeds:
                restart[bi, [node_pos[n] for n in seeds]] = 1.0 / len(seeds)
            else:
                zero_seed += 1
        x = restart.copy()
        for _ in range(a.iters):
            x = a.alpha * restart + (1.0 - a.alpha) * (x @ transition)
        ppr_scores[start:start + len(batch)] = x[:, graph_doc_rows]

    # BGE uses exactly the same encoder/instruction as the dense operator arm.
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cpu" if a.cpu else (
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = SentenceTransformer(a.model, device=device, local_files_only=True)
    model.max_seq_length = 512
    de = cached_encode(
        model, [corpus[d] for d in doc_ids], f"{a.split}_doc", batch=64, chunk=256
    )
    import hashlib
    qkey = hashlib.md5("|".join(q["id"] for q in queries).encode()).hexdigest()[:8]
    qe = cached_encode(
        model, [q["question"] for q in queries], f"{a.split}_query_{qkey}",
        instruct=BGE_QI, batch=64, chunk=256
    )
    bge_scores = qe @ de.T

    sets_by_q = {}
    if a.sets:
        source = json.load(open(a.sets))
        for qid, value in source.items():
            sets_by_q[qid] = value.get("sets", []) if isinstance(value, dict) else value
    bge_result, bge_best = score(
        bge_scores, queries, doc_ids, sets_by_q, None, positive_only=False
    )
    ppr_result, _ = score(
        ppr_scores, queries, doc_ids, sets_by_q, bge_best, positive_only=True
    )
    # Re-score BGE with its own difficulty partition so both tables have the
    # identical BGE-defined similar/dissimilar slices.
    bge_result, _ = score(
        bge_scores, queries, doc_ids, sets_by_q, bge_best, positive_only=False
    )
    print_scores("BGE", bge_result)
    print_scores("PPR", ppr_result)

    bge_out = out_dir / "predictions_bge.json"
    ppr_out = out_dir / "predictions_ppr.json"
    json.dump(predictions(queries, doc_ids, bge_scores), bge_out.open("w"))
    # Zero PPR mass means unreachable; omit such documents instead of resolving
    # their tied rank by arbitrary corpus order.
    json.dump(predictions(queries, doc_ids, ppr_scores, positive_only=True), ppr_out.open("w"))
    json.dump({"BGE": bge_result, "PPR": ppr_result},
              (out_dir / "scores.json").open("w"), indent=2)
    meta = {
        "dataset": a.dataset,
        "split": a.split,
        "graph": graph_name,
        "queries": len(queries),
        "documents": len(doc_ids),
        "nodes": len(nodes),
        "undirected_edges": int(adj.nnz // 2),
        "alpha": a.alpha,
        "iterations": a.iters,
        "restart": "uniform over unique stored start_nodes",
        "mean_seeds": float(np.mean(seed_counts)),
        "zero_seed_queries": zero_seed,
        "mean_positive_ppr_documents": float(np.mean((ppr_scores > 0).sum(axis=1))),
        "bge_predictions": str(bge_out),
        "ppr_predictions": str(ppr_out),
    }
    json.dump(meta, (out_dir / "run.json").open("w"), indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
