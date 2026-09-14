# kg-construction: engine, semantic scorer inputs, corpora

Everything the retriever needs at training and inference time, apart from the SciAfford
graph builder (which lives in `../kg-construction-v16/`).

| Path | What it is | Thesis |
|---|---|---|
| `gfm-rag/` | The graph reasoning engine: a fork of GFM-RAG / G-Reasoner (`RManLuo/gfm-rag` at commit `36cef5a`) carrying the SciGraphIR fusion model, the CCMP head and gate, the joint training objective and path interpretation. See `gfm-rag/SCIGRAPHIR_CHANGES.md` for exactly which files were added or modified. | Ch. 7 |
| `construct_v2/gen_probes.py` | One LLM call per query generating up to 8 short hypothetical answers ("probes"): specific methods the problem would need, and hypothetical abstracts from other fields whose mechanism is analogous. These are the views of the multi-view semantic scorer. | Ch. 6 |
| `eval/cargo_operator.py` | The handcrafted semantic scorer (dense + probe-sum + probe-max with a leave-one-out anti-hub denominator). It owns the encoder, query instruction and embedding cache that every other component reuses, and it warm-starts the learned scorer. | Ch. 6 |
| `eval/score.py`, `eval/bge.py`, `eval/bm25.py`, `eval/hyde.py`, `eval/qwen3_dense.py`, `eval/fuse_rrf.py` | TOMATO-Star scorer and single-gold baselines. SIR-4 (multi-gold) uses `../sir4-retrieval/eval/score_sir4.py` and `baselines_sir4.py` instead. | Ch. 9 |
| `experiments/probe_greasoner/precompute_operator_components.py` | Caches dense, probe-sum, probe-max and popularity terms per query, aligned to the graph's document order, so the fusion can recompute the operator live with learnable scalars. | Ch. 7 |
| `experiments/probe_greasoner/precompute_semantic_components.py` | Same for the learned scorer: records the per-answer similarity memmap, the document embeddings and the column permutation the fusion needs. | Ch. 7 |
| `experiments/researchbench/` | Builds the ResearchBench pooled corpus and its OpenAlex domain labels for the zero-shot transfer experiment. | Ch. 9 |
| `prepare/` | TOMATO-Star staging: selects the 7,000-document training subset (`train_selection.json`) and writes the corpus files. | Ch. 9 |
| `train/colab_train_v16sc_fusion_greasoner.ipynb` | The original TOMATO-Star training notebook. The SIR-4 notebook generators reuse its engine-install and model cells verbatim. | Ch. 9 |
| `run_index.sh` | OpenIE entity-graph index with the stock engine (construction control). | Ch. 4 |

## Data layout (created at runtime, git-ignored)

```
kg-construction/data/<dataset>_<split>/raw/documents.json        {doc_id: "Title. Abstract"}
kg-construction/data/<dataset>_<split>/raw/<split>.json          queries: id, question, supporting_documents, stratum
kg-construction/data/<dataset>_<split>/processed/stage1/         OpenIE graph (run_index.sh)
kg-construction/data/<dataset>_<split>_v16sc/processed/stage1/   SciAfford graph (build_greasoner_dataset.py)
kg-construction/data/<dataset>_<split>_hyb/processed/stage1/     merged graph (build_hybrid_graph.py)
kg-construction/construct_v2/cache/<dataset>/probes_<split>.jsonl  hypothetical answers
```

`<dataset>` is `tomato`, `sir4_cs`, `sir4_biology`, `sir4_physics`, `sir4_matsci`, `mir` or
`researchbench`. `../cargo_paths.py` is the single resolver for these paths; every script
takes `--dataset`.

## Installing the engine

```bash
pip install -e kg-construction/gfm-rag      # needs torch and torch_geometric first
```

The Colab notebooks install it from `gfm-rag-adapted.zip` on Drive (cells 3a to 3c) and
run training with `python -m gfmrag.workflow.sft_training --config-path config/gfm_reasoner
--config-name sft_training_fusion ...`. The per-arm flags (`CCMP=1`, `FUSION_OBJECTIVE=hardneg`,
`HARDNEG_HUB`, `HARDNEG_RAND`, `PER_GOLD`, `ROUTE=astar|attn`) are environment variables read by
`gfm-rag/gfmrag/trainers/fusion_trainer.py` and `gfm-rag/gfmrag/models/ultra/models.py`.
