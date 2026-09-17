# Evaluation and analysis

[Experiments](../README.md) · [Notebooks](../notebooks/README.md) · [Results](../results/README.md)

This directory currently contains four roles: semantic model development, retrieval metrics,
graph/path analysis, and presentation of results. Use the groups below to find the relevant code.

## Semantic method and comparisons

[semantic_scorer.py](semantic_scorer.py) contains the thesis `SortedMLPScorer` and
`MatchabilityPredictor`, their fitting code, and alternative scorer architectures.
The final model uses arm `mlp`. See the [code guide](../../docs/README.md#2-multi-view-semantic-scoring)
for its relationship to encoding, precomputation, and fusion.

## Retrieval metrics and baselines

| Script | Purpose |
|---|---|
| [score_sir4.py](score_sir4.py) | Recall@k, MRR, MGRR, mAP, nDCG@5, and CompleteSet@k, with dataset-appropriate slices. |
| [bge_sir4.py](bge_sir4.py), [baselines_sir4.py](baselines_sir4.py) | Dense and other retrieval baselines. |
| [run_ppr_bge.py](run_ppr_bge.py) | Personalized PageRank comparison. |
| [compare_arms.py](compare_arms.py), [all_domains_table.py](all_domains_table.py), [report_domain_results.py](report_domain_results.py) | Compare saved arms and summarize fields. |

Example, from the repository root after obtaining predictions and benchmark exports:

```bash
python experiments/eval/score_sir4.py \
  --pred /path/to/predictions.json \
  --queries retriever/data/sir4_physics_test/raw/test.json \
  --sets sir-4/data/benchmark/physics_test_low/sets.json \
  --name scigraphir
```

`--sets` supplies the validated inspiration sets required for CompleteSet@k. MRR uses the
first relevant result; MGRR averages reciprocal rank over all gold documents. Keep these
metrics distinct when comparing older saved tables.

## Graph and path analysis

| Task | Files |
|---|---|
| Validate graph structure and seeds | [audit_graph.py](audit_graph.py) |
| Compare graph and semantic channels | [graph_channel_scores.py](graph_channel_scores.py), [graph_channel_curves.py](graph_channel_curves.py) |
| Inspect parameter-free walk behavior | [walk_prior.py](walk_prior.py), [walk_prior_variants.py](walk_prior_variants.py), [walk_prior_why.py](walk_prior_why.py) |
| Measure paired CCMP retrieval differences | [ccmp_paired_stats.py](ccmp_paired_stats.py), [ccmp_rank_delta.py](ccmp_rank_delta.py) |
| Discover scientific paths | [scientific path guide](scientific_path_interpretations_README.md) |
| Intervene on gates along fixed paths | [CCMP mechanism guide](ccmp_mechanism_README.md), [ccmp_path_sender_effect.py](ccmp_path_sender_effect.py) |
| Select matching graph, scorer, and checkpoint | [ccmp_checkpoint_selection.py](ccmp_checkpoint_selection.py) |

Path attribution and retrieval performance answer different questions. The linked analysis
guides document the conditions and controls used for each intervention.

## Tables and figures

[export_thesis_results.py](export_thesis_results.py) transcribes the final PDF into the
[reported results page](../results/THESIS_RESULTS.md), JSON and LaTeX exports. Its `--check`
mode verifies those files against the reviewed PDF without changing them. This workflow
is separate from evaluating predictions.

Files ending in `_fig.py`, `route_*.py`, `showcase*.py`, and table builders render analysis
outputs. Start from the [results index](../results/README.md) to identify the desired table,
then use the [notebook index](../notebooks/README.md) for its experiment.

## Tests

`test_ccmp_checkpoint_selection.py`, `test_ccmp_mechanism.py`,
`test_ccmp_path_sender_effect.py`, and `test_scientific_path_interpretations.py` cover the
analysis tooling. [Verification notes](../../docs/SETUP.md#verification) give the command,
the latest local result, and the missing artifact requirements.
