# retriever: engine, semantic-branch inputs, corpora

Everything the retriever needs at training and inference time, apart from the SciAfford
graph builder (`../sciafford/`) and the experiment tooling (`../experiments/`).

| Path | What it is | Thesis |
|---|---|---|
| `gfm-rag/` | The graph reasoning engine: a fork of GFM-RAG / G-Reasoner (`RManLuo/gfm-rag` at commit `36cef5a`) carrying the SciGraphIR fusion model, the CCMP head and gate, the joint training objective and path interpretation. `gfm-rag/SCIGRAPHIR_CHANGES.md` lists exactly which files were added or modified. | Ch. 7 |
| `probes/gen_probes.py` | One LLM call per query generating up to 8 short hypothetical answers: specific methods the problem would need, and hypothetical abstracts from other fields whose mechanism is analogous. These are the views of the multi-view semantic scorer. | Ch. 6 |
| `eval/operator_scorer.py` | The handcrafted semantic scorer (dense + probe-sum + probe-max with a leave-one-out anti-hub denominator). It owns the encoder, query instruction and embedding cache that every other component reuses, and it warm-starts the learned scorer in `../experiments/eval/semantic_scorer.py`. | Ch. 6 |
| `precompute/` | Caches the semantic branch's per-query inputs aligned to a graph's document order, so the fusion recomputes the semantic score live with learnable scalars and mines hard negatives from it. | Ch. 7 |
| `researchbench/` | Builds the ResearchBench pooled corpus and its OpenAlex domain labels for the zero-shot transfer experiment. | Ch. 9 |
| `tomato_star/` | Stages the TOMATO-Star benchmark into the corpus layout (resolution of inspiration domains, same / cross strata, the 7,000-document training subset). | Ch. 9 |
| `train/` | The original TOMATO-Star training notebook; the notebook generators reuse its engine-install and model cells verbatim. | Ch. 9 |
| `run_index.sh` | OpenIE entity-graph index with the stock engine (construction control). | Ch. 4 |

## Data layout (created at runtime, git-ignored)

```
retriever/data/<dataset>_<split>/raw/documents.json         {doc_id: "Title. Abstract"}
retriever/data/<dataset>_<split>/raw/<split>.json           queries: id, question, supporting_documents, stratum
retriever/data/<dataset>_<split>/processed/stage1/          OpenIE graph (run_index.sh)
retriever/data/<dataset>_<split>_v16sc/processed/stage1/    SciAfford graph (sciafford/build_greasoner_dataset.py)
retriever/data/<dataset>_<split>_hyb/processed/stage1/      merged graph (experiments/prep/build_hybrid_graph.py)
retriever/probes/cache/<dataset>/probes_<split>.jsonl       hypothetical answers
```

`<dataset>` is `tomato`, `sir4_cs`, `sir4_biology`, `sir4_physics`, `sir4_matsci`, `mir` or
`researchbench`. `../scigraphir_paths.py` is the single resolver for these paths; every script
takes `--dataset`.

## Installing and running the engine

```bash
pip install -e retriever/gfm-rag      # needs torch and torch_geometric first
```

The Colab notebooks install it from `gfm-rag-adapted.zip` on Drive (cells 3a to 3c) and run
training with `python -m gfmrag.workflow.sft_training --config-path config/gfm_reasoner
--config-name sft_training_fusion ...`. The per-arm switches (`CCMP=1`,
`FUSION_OBJECTIVE=hardneg`, `HARDNEG_HUB`, `HARDNEG_RAND`, `PER_GOLD`, `ROUTE=astar|attn`) are
environment variables read by `gfm-rag/gfmrag/trainers/fusion_trainer.py` and
`gfm-rag/gfmrag/models/ultra/models.py`. The engine locates the semantic scorer through
`SCIGRAPHIR_ROOT`, so set it before training.
