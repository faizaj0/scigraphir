"""
cargo_ann.py -- top-k nearest neighbours over unit-normalised embeddings, with
an exact backend and a FAISS backend behind one interface.

WHY. Graph construction has three quadratic blocks: canonicalisation
(single-linkage over pairs above tau), mutual-kNN soft edges, and seed snapping.
Each was a chunked `E @ E.T`. That is fine at TOMATO scale and does not survive
SIR-4 CS:

    corpus              largest type   pairs        score matrix
    TOMATO test         ~15,600        245 M        ~1 GB
    SIR-4 CS train      ~104,000       10.8 B       ~43 GB

THE EXACT BACKEND IS STILL THE DEFAULT, and is bit-identical to the code it
replaced. FAISS is opt-in per run (`--ann faiss`), because HNSW is approximate
and silently switching would change published TOMATO numbers. Callers that want
the old behaviour get it by doing nothing.

APPROXIMATION, STATED PLAINLY. HNSW may miss a true neighbour. It is used here
only at high similarity thresholds (canonicalisation 0.85, soft edges 0.80),
which is where graph-based ANN recall is strongest, and `k` is taken generously
so the threshold, not `k`, is what prunes. Two mitigations that matter:

  * canonicalisation is single-linkage, so a missed A-B link usually survives
    via a shared neighbour C;
  * `neighbours()` returns similarities, so every caller re-applies its own
    threshold exactly. ANN affects only which candidates were considered.

If a run must be exact at scale, use backend="exact" and accept the wall-clock.
"""
from __future__ import annotations

import numpy as np

#: k below which the exact backend is used regardless, because building an
#: index costs more than the scan. Also keeps tiny types deterministic.
_SMALL_N = 4096


def _exact(emb: np.ndarray, k: int, chunk: int = 1024):
    """Chunked full matmul. Same arithmetic as the code this replaced."""
    n = len(emb)
    kk = min(k, n - 1)
    idx = np.zeros((n, kk), np.int32)
    sim = np.zeros((n, kk), np.float32)
    embT = emb.T
    for i0 in range(0, n, chunk):
        S = emb[i0:i0 + chunk] @ embT                     # (b, n)
        for r in range(S.shape[0]):
            row = S[r]
            row[i0 + r] = -1.0                            # drop self
            cand = np.argpartition(-row, kk - 1)[:kk]
            order = cand[np.argsort(-row[cand])]
            idx[i0 + r] = order
            sim[i0 + r] = row[order]
    return idx, sim


def _faiss(emb: np.ndarray, k: int, m: int = 32, ef: int = 128):
    import faiss

    n, d = emb.shape
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = max(ef, k * 4)
    index.add(emb)
    # k+1 because the query is in the index and returns itself first.
    sim, idx = index.search(emb, min(k + 1, n))
    out_i = np.zeros((n, min(k, n - 1)), np.int32)
    out_s = np.zeros((n, min(k, n - 1)), np.float32)
    for r in range(n):
        keep = [(j, s) for j, s in zip(idx[r], sim[r]) if j != r and j >= 0][:out_i.shape[1]]
        for c, (j, s) in enumerate(keep):
            out_i[r, c], out_s[r, c] = j, s
        for c in range(len(keep), out_i.shape[1]):        # pad short rows
            out_i[r, c], out_s[r, c] = r, -1.0
    return out_i, out_s


def neighbours(emb: np.ndarray, k: int, backend: str = "exact", verbose: bool = False):
    """Top-`k` neighbours of every row, self excluded, sorted by similarity.

    Args:
        emb: (n, d) float32, rows unit-normalised. Inner product == cosine.
        k: neighbours per row.
        backend: "exact" (default, reproduces the legacy matmul), "faiss"
            (HNSW, approximate), or "auto" (faiss above `_SMALL_N` if importable).

    Returns:
        (idx, sim), each (n, min(k, n-1)). Padded rows carry sim == -1.0, which
        every threshold rejects.
    """
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    n = len(emb)
    if n < 2:
        return np.zeros((n, 0), np.int32), np.zeros((n, 0), np.float32)

    use = backend
    if backend == "auto":
        use = "exact"
        if n > _SMALL_N:
            try:
                import faiss  # noqa: F401
                use = "faiss"
            except ImportError:
                pass
    if use == "faiss" and n <= _SMALL_N:
        use = "exact"                                     # index not worth building
    if use == "faiss":
        try:
            import faiss  # noqa: F401
        except ImportError:
            # Falling back silently would make a 43 GB allocation look like a
            # hang, so say so.
            print(f"[ann] faiss not importable; falling back to exact for n={n:,} "
                  f"(pip install faiss-cpu)")
            use = "exact"

    if verbose:
        print(f"[ann] {use} backend, n={n:,} k={k}")
    return (_faiss(emb, k) if use == "faiss" else _exact(emb, k))


def add_ann_arg(ap) -> None:
    """Attach the standard `--ann` flag to an argparse parser."""
    ap.add_argument(
        "--ann", default="exact", choices=["exact", "faiss", "auto"],
        help="neighbour backend. 'exact' (default) reproduces legacy numbers; "
             "'faiss' is approximate HNSW, needed above ~20k nodes per type.")
