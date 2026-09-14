"""Baseline retrievers: BM25 (lexical) and BGE-large (dense). No API key needed."""
from __future__ import annotations
import re
import numpy as np
from tqdm import tqdm
from .config import CACHE

_WORD_RE = re.compile(r"[a-zA-Z]+")
_STOP = set("""a an and are as at be by for from has have in is it of on or that the to
was were will with this these those we our using used use based via study paper""".split())


def _tokens(text: str) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _WORD_RE.finditer(text))
            if len(t) > 2 and t not in _STOP]


def rank_bm25(queries: list[dict], corpus: dict[str, str]) -> dict[str, list[str]]:
    from rank_bm25 import BM25Okapi
    keys = list(corpus)
    print(f"[bm25] tokenising {len(keys)} docs...")
    bm = BM25Okapi([_tokens(corpus[k]) for k in tqdm(keys, desc="bm25 docs")])
    rankings: dict[str, list[str]] = {}
    for q in tqdm(queries, desc="bm25 queries"):
        scores = bm.get_scores(_tokens(q.get("query_text", q["b_text"])))
        order = np.argsort(-scores)
        rankings[q["query_id"]] = [keys[i] for i in order]
    return rankings


def _bge_embed(texts: list[str], model_name: str, tag: str,
               instr: str = "", batch_size: int = 64) -> np.ndarray:
    import os
    cache_f = CACHE / f"bge_{tag}.npy"
    if cache_f.exists():
        emb = np.load(cache_f)
        if emb.shape[0] == len(texts):
            print(f"[bge] cached {tag}: {emb.shape}")
            return emb
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bge] encoding {len(texts)} {tag} on {dev}...")
    m = SentenceTransformer(model_name, device=dev)
    m.max_seq_length = 512
    emb = m.encode([instr + t for t in texts], batch_size=batch_size,
                   normalize_embeddings=True, show_progress_bar=True).astype("float32")
    np.save(cache_f, emb)
    print(f"[bge] cached -> {cache_f.name} {emb.shape}")
    return emb


def rank_bge(queries: list[dict], corpus: dict[str, str],
             model_name: str = "BAAI/bge-large-en-v1.5") -> dict[str, list[str]]:
    keys = list(corpus)
    Q_INSTR = "Represent this sentence for searching relevant passages: "
    doc_emb = _bge_embed([corpus[k] for k in keys], model_name, "docs")
    q_emb   = _bge_embed([q.get("query_text", q["b_text"]) for q in queries], model_name, "queries", instr=Q_INSTR)
    rankings: dict[str, list[str]] = {}
    B = 256
    for i in tqdm(range(0, len(queries), B), desc="bge rank"):
        sims = q_emb[i:i+B] @ doc_emb.T          # cosine (both normalised)
        order = np.argsort(-sims, axis=1)
        for j, q in enumerate(queries[i:i+B]):
            rankings[q["query_id"]] = [keys[c] for c in order[j]]
    return rankings
