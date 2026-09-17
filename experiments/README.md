# Experiments

[Repository home](../README.md) · [Code guide](../docs/README.md) · [Setup](../docs/SETUP.md)

This directory connects the thesis method to its evidence: training runs, comparisons,
retrieval metrics, transfer studies, downstream hypothesis evaluation, and figures.
For the full SIR-4 method, start with [colab_sir4_hyb.ipynb](notebooks/colab_sir4_hyb.ipynb):
it uses the default SciAfford graph combining affordance structure and OpenIE entity context.

## Choose a task

| Task | Start here |
|---|---|
| Prepare a dataset and build a Colab bundle | [Preparation tools](prep/README.md), [SIR-4 runbook](RUNBOOK.md), [MIR runbook](MIR_RUNBOOK.md) |
| Train SciGraphIR or compare ablations | [Notebook index](notebooks/README.md) |
| Score predictions or inspect a graph | [Evaluation and analysis](eval/README.md) |
| Read the final semantic scorer implementation | [semantic_scorer.py](eval/semantic_scorer.py): `SortedMLPScorer`, `MatchabilityPredictor` |
| Run LLM retrieval comparisons | [LLM baselines](llm_baselines/README.md) |
| Evaluate composed hypotheses | [Downstream study](downstream/README.md) |
| Read the reported thesis results | [Reported tables](results/THESIS_RESULTS.md), [source and run records](results/README.md) |

## What belongs where

| Directory | Contents |
|---|---|
| [prep/](prep/README.md) | Dataset adapters, graph variants, packaging, and notebook generators. |
| [notebooks/](notebooks/README.md) | Colab workflows grouped by experiment. |
| [eval/](eval/README.md) | Retrieval metrics, semantic scorer experiments, graph diagnostics, path analysis, and plots. |
| [transfer/](transfer/) | ResearchBench subsets, paired bootstrap, and transfer preparation. |
| [llm_baselines/](llm_baselines/README.md) | MOOSE-Chem, MOOSE-Star, and LATTICE adapters and runners. |
| [downstream/](downstream/README.md) | Hypothesis composition, judging, and aggregation. |
| [colab_cells/](colab_cells/) | Notebook fragments and paste-in analysis cells; some use notebook-only syntax. |
| [results/](results/README.md) | Reported thesis tables, source fingerprints, and supporting experiment records. |

The semantic scorer is a method contribution currently stored among experiment tools.
Use the [code guide](../docs/README.md#2-multi-view-semantic-scoring) to distinguish its final
model classes from the alternative architectures evaluated in the same file.

## Reading the experimental evidence

1. [Notebook index](notebooks/README.md): identify the dataset and comparison.
2. [Evaluation guide](eval/README.md): identify the metric and analysis script.
3. [Results index](results/README.md): read PDF-verified reported tables or supporting experiment records.

Per-query predictions and checkpoints are generally external runtime artifacts. A checked-in
table is a saved result, not a complete reproducibility package. Current requirements and
known execution gaps are recorded in [setup and data](../docs/SETUP.md).
