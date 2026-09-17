# Retriever

[Repository home](../README.md) · [Code guide](../docs/README.md) · [Setup](../docs/SETUP.md)

The retrieval system has a semantic branch, a graph branch, and an adaptive fusion router.
The default graph combines affordance structure, OpenIE entity context and direct
paper-to-affordance links; its saved graph names end in `_hyb`.
This directory contains the graph engine and most retrieval support code. The final
semantic scorer currently lives in `experiments/eval/`; it is linked explicitly below.

## Read the thesis method

| Component | Start here | Purpose |
|---|---|---|
| Hypothetical answers | [hypothetical_answers/](hypothetical_answers/README.md) | Generate separate semantic views of a research problem. |
| Final semantic scorer | [semantic_scorer.py](../experiments/eval/semantic_scorer.py) | `SortedMLPScorer` and `MatchabilityPredictor`; select the `mlp` arm. |
| Shared embeddings and earlier handcrafted scorer | [eval/](eval/README.md) | Encoder/cache helpers and the handcrafted comparison scorer. |
| Graph-aligned semantic inputs | [precompute/](precompute/README.md) | Cache inputs in the document order expected by the graph. |
| Fusion router and CCMP head | [fusion_reasoner.py](gfm-rag/gfmrag/models/fusion_reasoner.py) | Combine semantic and graph scores and predict node responsibilities. |
| Training and CCMP targets | [fusion_trainer.py](gfm-rag/gfmrag/trainers/fusion_trainer.py) | Ranking objectives, hard negatives, continuation targets, and CCMP loss. |
| Message propagation and gating | [ultra/models.py](gfm-rag/gfmrag/models/ultra/models.py) | QueryNBFNet and the gate applied to outgoing messages. |

The [engine change map](gfm-rag/SCIGRAPHIR_CHANGES.md) separates SciGraphIR additions from
the vendored GFM-RAG code. Its [upstream README](gfm-rag/README.md) describes the upstream
project; use this guide and the change map for SciGraphIR.

## Supporting directories

| Path | Role |
|---|---|
| [tomato_star/](tomato_star/README.md) | TOMATO-Star corpus preparation and labels. |
| [researchbench/](researchbench/) | ResearchBench corpus preparation and labels, using the SIR-4 reference matcher. |
| [train/](train/README.md) | Source notebook used by experiment generators. |
| [run_index.sh](run_index.sh) | OpenIE component for the default graph; also used as a standalone control. |
| `data/` | Runtime corpora and graphs; excluded from Git. |

SciAfford graph construction is in [sciafford/](../sciafford/README.md). Training runs,
baselines, and evaluations are indexed under [experiments/](../experiments/README.md).

## Data flow and execution

Corpora are stored at `data/<dataset>_<split>/raw/`. The default SciAfford graph uses
`data/<dataset>_<split>_hyb/processed/stage1/`. The `_v16sc` directories contain the
affordance component used to construct it and for component comparisons.
[scigraphir_paths.py](../scigraphir_paths.py) retains component defaults for older callers;
pass `suffix="hyb"` when resolving the default full-method graph.

Install the engine using the [setup instructions](../docs/SETUP.md). Its training entry
point is `python -m gfmrag.workflow.sft_training`; the base
[fusion config](gfm-rag/gfmrag/workflow/config/gfm_reasoner/sft_training_fusion.yaml) is
overridden by the [experiment notebooks](../experiments/notebooks/README.md).

The base config defaults to the `handcrafted` scorer, and CCMP is activated by
`CCMP=1`. A bare invocation therefore does not select the full thesis method. Full-method
experiments also select `semantic=mlp`, provide its checkpoints and cached components, and
configure the ranking objective. Start with the
[default SIR-4 graph workflow](../experiments/notebooks/colab_sir4_hyb.ipynb) and follow
its complete settings, including the `_hyb` graph names.
