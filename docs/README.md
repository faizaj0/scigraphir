# Code guide

[Repository home](../README.md) · [Setup and data](SETUP.md)

The repository contains the retrieval method, the benchmark used to train and evaluate it,
and the experiments that support the thesis. Read the method first; use the experiments
to see how it was evaluated.
The [thesis method figure](figures/scigraphir-method.pdf) provides a visual overview.

## 1. SciAfford: turn papers into a graph

Start with [sciafford/README.md](../sciafford/README.md).

| File | Role |
|---|---|
| [extract_affordances.py](../sciafford/extract_affordances.py) | Extracts structured affordance representations from papers and research questions. |
| [build_greasoner_dataset.py](../sciafford/build_greasoner_dataset.py) | Builds the affordance component: typed concepts, relations and query seeds. |
| [run_index.sh](../retriever/run_index.sh) | Builds the OpenIE component supplying entity context and query entity seeds. |
| [build_hybrid_graph.py](../experiments/prep/build_hybrid_graph.py) | Assembles the default SciAfford graph from both components and adds direct paper-to-affordance links. |
| [audit_graph.py](../experiments/eval/audit_graph.py) | Checks the resulting graph and query seeds before training. |

The default graph combines affordance structure, OpenIE entity seeds, entity-to-paper
mention edges and direct paper-to-affordance links. Its output is
`retriever/data/<dataset>_<split>_hyb/processed/stage1/`, containing `nodes.csv`,
`edges.csv`, `relations.csv` and seeded queries. The `_v16sc` graph is the affordance
component and is also used in component comparisons.

## 2. Multi-view semantic scoring

| File or class | Role |
|---|---|
| [generate_answers.py](../retriever/hypothetical_answers/generate_answers.py) | Generates hypothetical answers for each research problem. |
| [handcrafted_scorer.py](../retriever/eval/handcrafted_scorer.py) | Shared encoder/cache helpers and the earlier handcrafted scorer. |
| [semantic_scorer.py](../experiments/eval/semantic_scorer.py) | Contains the final `SortedMLPScorer`, its `MatchabilityPredictor`, training, and alternative scorer architectures. |
| [precompute_semantic_components.py](../retriever/precompute/precompute_semantic_components.py) | Aligns cached semantic inputs to the graph's document order. |

Read `SortedMLPScorer` and `MatchabilityPredictor` for the thesis model. Other scorer
classes in the same file are comparisons. The final scorer is selected with the `mlp` arm.

The semantic branch compares each document to the separate hypothetical answers. The document
MLP estimates background matchability for PMI-style specificity correction. The ranking MLP
maps the sorted similarity profile and direct query similarity to a score.

## 3. CCMP and adaptive fusion

The base reasoner and SciGraphIR extensions share the [GFM-RAG fork](../retriever/gfm-rag/).
The [fork change map](../retriever/gfm-rag/SCIGRAPHIR_CHANGES.md) distinguishes their origins.

| File | Read for |
|---|---|
| [fusion_reasoner.py](../retriever/gfm-rag/gfmrag/models/fusion_reasoner.py) | `FusionGraphReasoner`, the semantic branch, adaptive fusion router, and CCMP responsibility head. |
| [fusion_trainer.py](../retriever/gfm-rag/gfmrag/trainers/fusion_trainer.py) | `FusionSFTTrainer`, hard-negative ranking, `_ccmp_targets`, `_ccmp_hit`, and `_ccmp_loss`. |
| [ultra/models.py](../retriever/gfm-rag/gfmrag/models/ultra/models.py) | `QueryNBFNet` propagation, CCMP message gates, and routing controls. |
| [ultra/layers.py](../retriever/gfm-rag/gfmrag/models/ultra/layers.py) | Message-passing layers and per-query edge weights. |
| [sft_training_fusion.yaml](../retriever/gfm-rag/gfmrag/workflow/config/gfm_reasoner/sft_training_fusion.yaml) | Base architecture and training configuration; notebooks supply experiment overrides. |

CCMP uses training labels to supervise intermediate nodes according to their continuation
toward relevant versus misleading documents. Predicted responsibilities gate propagation.
The router combines the semantic and graph branches using
`z_sem + gamma_q * ReLU(z_graph)`.

`cqig.py` implements an explored alternative. A*Net-style and RED-GNN-style routing are
comparison arms. Their presence in the engine does not make them part of the final CCMP method.

## 4. SIR-4 and EAID

The [included dataset](../sir-4/dataset/README.md) contains the four fields and both splits.
The [benchmark guide](../sir-4/README.md) covers collection through export. The key
contribution code is [03_new_method.py](../sir-4/build/03_new_method.py), which fixes the
target and constructs its inspiration decomposition, and
[03b_uniqueness.py](../sir-4/build/03b_uniqueness.py), which searches for alternatives.
[03_decompose.py](../sir-4/build/03_decompose.py) supplies shared validation checks.

[05_export.py](../sir-4/build/05_export.py) writes the benchmark records and validated
inspiration sets. [score_sir4.py](../experiments/eval/score_sir4.py) consumes those sets for
CompleteSet@k alongside ordinary retrieval metrics.

## 5. Experiments and evidence

Use the [experiment guide](../experiments/README.md) to choose a notebook, baseline, or
analysis. The [reported results](../experiments/results/THESIS_RESULTS.md) reproduce the final
thesis PDF; the [results index](../experiments/results/README.md) also links supporting run records. The
[downstream study](../experiments/downstream/README.md) examines the hypotheses composed
from retrieved inspirations.

## Terminology and compatibility identifiers

Use **SciGraphIR** for the complete framework, **SciAfford** for the affordance-lifted
graph construction, and **SIR-4** for the benchmark. **EAID** means Equivalence-Aware
Inspiration Decomposition. **CCMP** means Contrastive Continuation Message Passing.

The implementation preserves the identifiers below so existing exports, caches and
checkpoints remain usable. They are storage or configuration names, not alternative names
for the thesis contributions.

| Name | Meaning |
|---|---|
| `sciafford` / `*_v16sc` | Affordance extraction/build code / saved affordance-component graph suffix. |
| `*_hyb` | Default SciAfford graph, combining affordance structure, OpenIE entity context and direct paper-to-affordance links. |
| `answers` | Hypothetical-answer records. |
| `handcrafted` | Earlier handcrafted semantic scorer, also used for shared embedding helpers. |
| `mlp` | Learned multi-view semantic scorer. |
| `CCMP` / `resp_*` | Contrastive Continuation Message Passing / responsibility-head parameters. |
| `quartet` | Retained SIR-4 export metadata key; the exporter and readers share this schema. |
| `cargo-gfmrag`, `quartet/`, `quartet_cache` | Existing Drive/export/cache locations used to reproduce saved experiments. |
| `Pop`, `pop_*`, `popnet`, `SEMANTIC_POPNET` | Background matchability objects, configuration and checkpoint keys. |
| `same` / `cross` | Dataset-specific field-relation labels; see the relevant evaluation protocol. |

[scigraphir_paths.py](../scigraphir_paths.py) resolves corpus, graph, and cache paths.
Most retrieval scripts select a corpus with `--dataset`; benchmark scripts generally use
`--domain` and `--split`, and LLM baseline wrappers use `--field`.

Analysis commands use `--sir4` for a SIR-4 evaluation export; `--quartet` remains accepted.
The semantic comparison uses `--matchability` and `--mlp-matchability`; the earlier
`--popularity` and `--mlp_popularity` options remain accepted and map to the same settings.

Commands and code use **hypothetical answers**, **affordance representations**, and the
**handcrafted semantic scorer**. Existing cache filenames, graph arm identifiers, and
checkpoint tensor keys retain their storage names so saved runs remain usable.
