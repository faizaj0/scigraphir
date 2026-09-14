# Precomputed semantic components for the fusion

The graph reasoner trains on Colab; the semantic branch's inputs are cached once per graph
so the fusion can recompute the semantic score live with learnable scalars and mine hard
negatives from it.

| Script | Output |
|---|---|
| `precompute_operator_components.py --graph <graph> --split <split>` | `data/<graph>/operator_components.npz`: `dense`, `S`, `M` (float16, queries x documents in the graph's `nodes.csv` order), `total_S`, `query_ids`. Read through `OPERATOR_COMPONENTS[_TEST]`. |
| `precompute_semantic_components.py --dataset <ds> --model <encoder> --graph <graph> --split <split>` | `semantic_components.npz`: the path of the per-answer similarity memmap built by `semantic_scorer.py`, the document embeddings the popularity predictor reads, the answer mask and the corpus-to-graph column permutation. Read through `SEMANTIC_COMPONENTS[_TEST]`. |

Both run inside the Colab notebooks after the scorer cell (sections 5b and 5d); the notebooks
save the results to Drive because they are the expensive artefacts.
