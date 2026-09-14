# Handcrafted semantic scorer (the operator)

`operator_scorer.py` is the handcrafted scorer that preceded the learned multi-view scorer:

```
S_op = w0 z(dense) + w1 z(S / dem^b) + w2 z(M / dem^b)
```

`dense` is the query-document cosine, `S` and `M` are the sum and max of the
hypothetical-answer cosines, and `dem` is a leave-one-out popularity term that penalises
documents which match everybody's answers. The four scalars are fitted by InfoNCE on a
training slice and are learned again inside the fusion.

The file also defines the encoder (BGE-large or Qwen3-Embedding), the query instruction and
the embedding cache that the learned scorer (`../../experiments/eval/semantic_scorer.py`), the
dense baselines (`../../experiments/eval/bge_sir4.py`) and the precompute scripts import, so
every arm sees identical vectors.

Scoring lives in `../../experiments/eval/score_sir4.py`; predictions use one schema:

```json
{"id": "...", "stratum": "cross", "supporting_documents": ["doi"], "predictions": {"document": [["doi", 0.91], ...]}}
```
