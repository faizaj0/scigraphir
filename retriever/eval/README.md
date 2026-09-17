# Shared embeddings and handcrafted scorer

[Retriever](../README.md) · [Final semantic model](../../experiments/eval/semantic_scorer.py)

[handcrafted_scorer.py](handcrafted_scorer.py) combines two roles:

- Shared encoder selection, query instructions, and embedding caches used by semantic
  scoring, precomputation, and dense baselines.
- The earlier handcrafted semantic scorer, selected as `handcrafted` in fusion experiments.

Its score combines direct query similarity with the sum and maximum of hypothetical-answer
similarities, corrected by a leave-one-out background matchability term:

```text
s = w0 * z(dense) + w1 * z(S / background matchability^beta) + w2 * z(M / background matchability^beta)
```

The final thesis semantic model is `SortedMLPScorer` in
[semantic_scorer.py](../../experiments/eval/semantic_scorer.py). It reuses the encoding/cache
helpers here. The [code guide](../../docs/README.md#2-multi-view-semantic-scoring) traces the
complete semantic branch.

Retrieval metrics are implemented separately in
[score_sir4.py](../../experiments/eval/score_sir4.py). A prediction record uses this shape:

```json
{
  "id": "query-id",
  "stratum": "cross",
  "supporting_documents": ["gold-document-id"],
  "predictions": {"document": [["document-id", 0.91]]}
}
```
