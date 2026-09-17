# Graph-aligned semantic inputs

[Retriever](../README.md) · [Semantic code guide](../../docs/README.md#2-multi-view-semantic-scoring)

These scripts prepare semantic inputs in the graph's document order. The fusion model
can then recompute scores while training and mine semantic hard negatives without encoding
the corpus on every training step.

| Script | Use | Output |
|---|---|---|
| [precompute_semantic_components.py](precompute_semantic_components.py) | Final learned multi-view scorer. | Per-answer similarity-cache references, document embeddings, answer mask, and corpus-to-graph column mapping. |
| [precompute_handcrafted_components.py](precompute_handcrafted_components.py) | Handcrafted handcrafted scorer comparison. | Direct, sum, and max similarities; background matchability totals; query IDs. |

Inputs are the staged corpus, answers/embeddings, and built graph. Outputs are component
files under `retriever/data/<graph>/`; encoder-specific filename suffixes may be added.
The [experiment notebooks](../../experiments/notebooks/README.md) supply the exact paths.

The engine reads learned-scorer inputs through `SEMANTIC_COMPONENTS` and
`SEMANTIC_COMPONENTS_TEST`, and handcrafted scorer inputs through `HANDCRAFTED_COMPONENTS` and
`HANDCRAFTED_COMPONENTS_TEST`. These are inputs to the scorer, not its trained weights.

Use a component file with the graph and document ordering it was built for. See
[setup and data](../../docs/SETUP.md) for artifact locations and
[semantic_scorer.py](../../experiments/eval/semantic_scorer.py) for cache construction.
