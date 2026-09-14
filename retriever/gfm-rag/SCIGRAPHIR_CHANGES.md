# What SciGraphIR changes in the GFM-RAG / G-Reasoner engine

This directory is a fork of [`RManLuo/gfm-rag`](https://github.com/RManLuo/gfm-rag) at commit
`36cef5a155888723b9875df2fa96ef6e148d55ae` (v2.0.0, 2026-04-20; `VENDORED_COMMIT.txt`),
Apache-2.0 (`LICENSE`). The upstream `README.md` is kept unchanged. Everything below was added
for the thesis. Files not listed are upstream code.

## Added files

| File | Purpose |
|---|---|
| `gfmrag/models/fusion_reasoner.py` | `FusionGraphReasoner`: wraps the query-conditioned `GraphReasoner`, adds the semantic channel (handcrafted operator or the learned multi-view scorer, `semantic: operator | mlp`), the query-adaptive fusion router (`s_fused = z_sem + gamma_q ReLU(z_graph)`, `gamma_q = softplus(gate(phi_q))`), and the **CCMP responsibility head** (one projection per layer, shared trunk, layer embedding). Thesis Ch. 6 and 7. |
| `gfmrag/trainers/fusion_trainer.py` | `FusionSFTTrainer`: the joint objective. Per-gold ranking loss over a candidate set of golds, semantic hard negatives, graph hard negatives and random negatives (`FUSION_OBJECTIVE=hardneg`, `HARDNEG_HUB`, `HARDNEG_RAND`, `PER_GOLD`); the graph-alone auxiliary ranking loss (`AUX_W`); **CCMP target construction** (`_ccmp_targets`: finite-horizon absorbing-walk hitting probabilities toward gold vs. reachable semantic-error endpoints, targets `y` and confidences `c`) and the confidence-weighted BCE (`_ccmp_loss`); `interpret()` (NBFNet gradient beam search with the CCMP gate recorded per hop) and `gate_decomposition()`. |
| `gfmrag/models/cqig.py` | Cross-Query Informativeness Gating, an alternative gate explored during the project. Imported by the trainer but off by default (`cqig: false`) and not part of the final system. |
| `gfmrag/workflow/interpret_paths.py` | Loads a trained fusion checkpoint and writes path interpretations (used by the qualitative and Table-4 style analyses). |
| `gfmrag/workflow/config/gfm_reasoner/sft_training_fusion.yaml` | The SciGraphIR training config (QueryNBFNet, 6 x 1024, DistMult, sum aggregation, `FusionGraphReasoner`, `FusionSFTTrainer`). |
| `gfmrag/workflow/config/gfm_reasoner/sft_training_gfmrag.yaml`, `gfm_rag/sft_training_nodefeat.yaml`, `gfmrag/models/gfm_rag_v1/model_nodefeat.py` | The GFM-RAG baseline run through the same trainer, with and without text node features. |
| `gfmrag/workflow/config/text_emb_model/qwen3_st.yaml` | Qwen3-Embedding-0.6B through sentence-transformers, the frozen encoder of the semantic branch. |

## Modified upstream files

| File | Change |
|---|---|
| `gfmrag/models/ultra/models.py` | **CCMP gate**: the predicted responsibility of a reached node scales its outgoing messages at each layer, normalised over the reached frontier so the mean gate is one (`g = (1 - eta) + eta * y_bar`); unreached nodes keep gate 1. Also the two routing baselines used as controls (`ROUTE=astar`: A*Net-style hard top-K node selection; `ROUTE=attn`: RED-GNN-style edge attention) and the gate masks used by the gate-decomposition analysis. |
| `gfmrag/models/ultra/layers.py` | Per-query edge weights through the fused message-passing kernel, and the separate-gradient path used for path attribution. |
| `gfmrag/trainers/base_trainer.py` | Hooks for the fusion trainer (prediction dumps, CCMP bookkeeping). |
| `gfmrag/models/gfm_reasoner/model.py`, `gfmrag/models/gfm_rag_v1/rankers.py` | Small changes so the wrapped reasoner exposes document logits and hidden states to the fusion. |
| `gfmrag/graph_index_datasets/graph_index_dataset.py`, `graph_index_dataset_v1.py` | Load the typed SciAfford / merged graphs and multi-gold queries; text-featured document nodes. |
| `gfmrag/graph_index_construction/*` (`kg_constructor.py`, `dpr_el_model.py`, `llm_ner_model.py`) | Mac-friendly OpenIE indexing (DPR entity linker, threshold and worker settings) for the OpenIE control graph. |
| `gfmrag/workflow/sft_training.py`, `gfmrag/utils/qa_utils.py`, `gfmrag/text_emb_models/__init__.py`, `gfmrag/__init__.py` | Wiring for the new model, trainer, encoder and configs. |
| `pyproject.toml` | CPU FAISS for local construction; otherwise upstream dependencies. |

Upstream `docs/` and `scripts/` were dropped from this copy; they are unchanged in the
upstream repository. Two explored model variants (per-document convex routing and a
semantic-prior reasoner) were removed from this copy because the thesis does not use them.
