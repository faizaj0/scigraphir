# Experiment notebooks

[Experiments](../README.md) · [Setup](../../docs/SETUP.md) · [Results](../results/README.md)

The 23 notebooks here are Colab experiment workflows. Choose by research question below.
They expect dataset bundles and engine/model artifacts on Drive; see [setup](../../docs/SETUP.md).

## Default full-method workflow

Start with [SIR-4 on the default SciAfford graph](colab_sir4_hyb.ipynb). It runs all
four fields using the merged affordance/OpenIE graph (`_hyb`), the learned multi-view
semantic scorer, and comparisons with and without CCMP.

## Component and dataset experiments

| Experiment | Notebook |
|---|---|
| SIR-4, computer science — affordance component | [sir4_cs_ccmp_vs_control](sir4_cs_ccmp_vs_control.ipynb) |
| SIR-4, biology — affordance component | [sir4_biology_ccmp_vs_control](sir4_biology_ccmp_vs_control.ipynb) |
| SIR-4, physics — affordance component | [sir4_physics_ccmp_vs_control](sir4_physics_ccmp_vs_control.ipynb) |
| SIR-4, materials science — affordance component | [sir4_matsci_ccmp_vs_control](sir4_matsci_ccmp_vs_control.ipynb) |
| TOMATO-Star | [Scorer/fusion ablations](tomato_ablations.ipynb), [CCMP ablation](tomato_ccmp_ablation.ipynb) |
| MIR | [Combined experiment](colab_mir_all.ipynb), [baselines](mir_baselines.ipynb) |

## Baselines and graph controls

| Comparison | Notebook |
|---|---|
| Dense and lexical baselines | [SIR-4](sir4_baselines_all.ipynb), [TOMATO-Star](tomato_baselines.ipynb) |
| LLM query expansion | [SIR-4 expansion baselines](llm_expansion_baselines_sir4.ipynb) |
| OpenIE graph | [SIR-4](colab_sir4_openie_ablation.ipynb), [TOMATO-Star](colab_tomato_openie_ablation.ipynb) |
| A*Net-style / RED-GNN-style routing versus CCMP | [SIR-4 CS](colab_routing_sir4_cs.ipynb), [TOMATO-Star](colab_routing_tomato.ipynb) |

## Transfer

| Train → evaluate | Notebook |
|---|---|
| Physics + biology → CS + materials science | [Forward transfer](colab_sir4_zeroshot_physics+biology_to_cs+matsci_standard.ipynb) |
| CS + materials science → physics + biology | [Reverse transfer](colab_sir4_zeroshot_cs+matsci_to_physics+biology_standard.ipynb) |
| Trained checkpoints → ResearchBench | [ResearchBench transfer](colab_rb_zeroshot.ipynb) |

## Interpretation and mechanism studies

| Question | Notebook |
|---|---|
| Which scientific paths does the trained model use? | [Scientific paths](colab_scientific_paths.ipynb) |
| How do gate interventions change a fixed path's attribution? | [CCMP mechanism study](colab_table4_openie.ipynb) |
| Which layers and senders change a route's weight? | [Gate decomposition](colab_ccmp_gate_decomp.ipynb) |
| Which examples illustrate branch and model differences? | [Worked examples](colab_showcase_all.ipynb) |

The filename `colab_table4_openie` predates the current mechanism workflow. Read the
[mechanism guide](../eval/ccmp_mechanism_README.md) for its present experiment.

## Generators and source versions

[Preparation tools](../prep/README.md) links the `build_*_notebook.py` generators and the
[training template](../../retriever/train/README.md) they reuse. Notebooks contain source
snapshots and runtime patches, so the checked-in notebook, regenerated notebook, and direct
engine invocation can execute different code. Record the notebook and generator revision
with each result. These source snapshots need consolidation before the notebooks can
be reduced to launchers that import the repository implementation.
