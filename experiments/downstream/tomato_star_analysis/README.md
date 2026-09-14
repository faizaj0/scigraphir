# Downstream hypothesis-quality package (copy)

These scripts were developed inside a checkout of the TOMATO-Star repository at
`TOMATO-Star/analysis/downstream/` and are copied here so the thesis code is in one place.
`_common.py` resolves paths relative to that location (`parents[2]` is the TOMATO-Star root)
and imports the MOOSE-Star prompt store and scoring rubric from a sibling `MOOSE-Star`
checkout, so **run them from the TOMATO-Star checkout** under `$EXTERNAL_REPOS`, as
`../run_downstream_sir4.sh` does. `build_inputs_sir4.py` reads this repository's predictions
through `SCIGRAPHIR_ROOT`.

Pipeline (thesis Chapter 9, downstream results):

| Step | Script | What |
|---|---|---|
| build | `build_inputs.py` (TOMATO-Star) / `build_inputs_sir4.py` (SIR-4) | one row per query with the top-1 retrieved document of every arm (BM25, Qwen3, ReasonIR, MOOSE-Chem, LATTICE, SciGraphIR), plus closed-book and oracle arms |
| compose | `compose.py` | `gpt-4o-mini` with the MOOSE-Star delta-hypothesis prompt, fed only the top-1 document |
| judge | `score_matched.py`, `score_matched_batch.py` | reference-based Matched-Score rubric (0 to 12) against the true hypothesis, `gpt-4o` |
| arena | `idea_arena.py`, `idea_arena_batch.py` | reference-free pairwise judging (the protocol the thesis shows rewards weaker grounding) |
| tables | `aggregate.py`, `downstream_table.py`, `matched_score_table.py`, `arena_criteria_table.py` | paired-bootstrap ladders and the thesis tables |
