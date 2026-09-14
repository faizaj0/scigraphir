# Operator and TOMATO-Star baselines

| File | What |
|---|---|
| `cargo_operator.py` | The handcrafted semantic scorer: `S_op = w0 z(dense) + w1 z(S / dem^b) + w2 z(M / dem^b)` where `dense` is the query-document cosine, `S` and `M` are the sum and max of the hypothetical-answer cosines, and `dem` is a leave-one-out popularity term that penalises documents which match everybody's answers. Fitted by InfoNCE on a training slice. It also defines the encoder (BGE-large or Qwen3-Embedding), the query instruction and the embedding cache that the learned scorer, the SIR-4 BGE baseline and the precompute scripts all import, so every arm sees identical vectors. |
| `score.py` | TOMATO-Star scorer (single gold per row): nDCG@10 and Recall@k, split same / cross. Any predictions file in the shared schema works. |
| `bge.py`, `bm25.py`, `hyde.py`, `qwen3_dense.py`, `fuse_rrf.py` | TOMATO-Star baselines writing that schema. |

Predictions schema (one record per query):

```json
{"id": "...", "stratum": "cross", "supporting_documents": ["doi"], "predictions": {"document": [["doi", 0.91], ...]}}
```

SIR-4 and MIR are multi-gold; use `sir4-retrieval/eval/score_sir4.py` and
`sir4-retrieval/eval/baselines_sir4.py` for them.
