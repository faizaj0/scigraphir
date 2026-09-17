# Downstream hypothesis evaluation

[Experiments](../README.md) · [Code guide](../../docs/README.md) · [Results](../results/README.md)

The [reported downstream table](../results/THESIS_RESULTS.md#downstream-hypothesis-quality)
contains the TOMATO-Star and pooled SIR-4 values from the final thesis PDF.

This study asks whether better retrieved inspirations help a fixed
composer produce better hypotheses. It includes reference-based matched scoring and
reference-free pairwise judging, with closed-book and oracle comparison arms.

## Code map

The [tomato_star_analysis/](tomato_star_analysis/) directory contains a copy of the downstream
analysis package used during the project.

| Stage | Files in the copied package |
|---|---|
| Build inputs and attach retrieval arms | [build_inputs.py](tomato_star_analysis/build_inputs.py), [build_inputs_sir4.py](tomato_star_analysis/build_inputs_sir4.py), [augment_arms.py](tomato_star_analysis/augment_arms.py) |
| Compose hypotheses | [compose.py](tomato_star_analysis/compose.py) |
| Reference-based judging | [score_matched.py](tomato_star_analysis/score_matched.py), [score_matched_batch.py](tomato_star_analysis/score_matched_batch.py) |
| Reference-free pairwise judging | [idea_arena.py](tomato_star_analysis/idea_arena.py), [idea_arena_batch.py](tomato_star_analysis/idea_arena_batch.py) |
| Aggregate and render tables | [aggregate.py](tomato_star_analysis/aggregate.py), [downstream_table.py](tomato_star_analysis/downstream_table.py) |

## Current execution path

[run_downstream_sir4.sh](run_downstream_sir4.sh) currently invokes `analysis.downstream`
inside an external TOMATO-Star checkout under `$EXTERNAL_REPOS`. It does not invoke the
copied package beside this README. The copied package also retains external-layout
assumptions in [_common.py](tomato_star_analysis/_common.py).

This dependency needs consolidation before the directory can serve as a standalone runner.
Existing experiment instructions,
artifact IDs, and historical run details are preserved in
[REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md); they are provenance notes, not a fresh-clone
quickstart.

## Protocol details to preserve

For the SIR-4 study, a top-1 retrieval hit means the retrieved paper is any member of the
gold union. The reference used for judging depends on whether the retrieved paper is a
validated inspiration, so the reference policy must be recorded with the result.

The original subset uses all cross-field queries plus sampled same-field queries, with
materials science evaluated in full. [Historical protocol notes](REPRODUCTION_NOTES.md)
give the exact arms and counts. Composition and judging make paid API calls; input assembly
and table aggregation operate on existing artifacts.
