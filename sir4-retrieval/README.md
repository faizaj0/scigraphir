# sir4-retrieval: experiments

Everything that turns a staged corpus into thesis tables: staging, the learned semantic
scorer, multi-gold metrics, the Colab notebooks, the baselines, zero-shot transfer, the
LLM-retrieval baselines, the downstream study and the qualitative analyses. Despite the
name it covers every dataset (TOMATO-Star, SIR-4, MIR, ResearchBench); `--dataset` selects.

Start with **`RUNBOOK.md`** (one SIR-4 field end to end) and `MIR_RUNBOOK.md`.

| Path | What |
|---|---|
| `prep/stage_sir4.py`, `prep/stage_mir.py` | Verify a benchmark export (duplicate ids, gold in corpus, empty texts) and stage it under `kg-construction/data/<dataset>_<split>/raw/`. |
| `prep/run_domain.py` | Idempotent driver for stage, extract, probes, graph, audit, bundle and notebook for one or all fields; paid steps need `--spend`. |
| `prep/build_hybrid_graph.py` | Merged-graph variant: SciAfford graph plus OpenIE entity seeds, mention edges and paper-to-frame shortcuts. |
| `prep/bundle.py` | Zips code, corpora, graphs and caches for Colab; the zip mirrors this repository so `CARGO_ROOT=/content/cargo` resolves every path. |
| `prep/build_*_notebook.py` | Generators for every notebook in `notebooks/`. Their docstrings are the authoritative description of each experiment. |
| `eval/semantic_scorer.py` | **Contribution 2a.** Controlled comparison of semantic scorer architectures on frozen encoder outputs; the thesis scorer is `SortedMLPScorer` (arm `mlp`): sorted per-answer similarity profile, learned popularity predictor `p_hat(d)` with exponent `beta`, per-gold full-corpus InfoNCE plus the matchability anchor loss. |
| `eval/score_sir4.py` | Multi-gold metrics: Recall@k, MRR, MGRR, nDCG@5, **CompleteSet@k** (needs `sets.json` from the SIR-4 export), same / cross and similar / dissimilar slices. |
| `eval/audit_graph.py` | Structural audit of a built graph (fails on dangling edges or zero-seed queries). |
| `eval/bge_sir4.py`, `eval/baselines_sir4.py`, `eval/run_ppr_bge.py` | Dense, lexical and walk baselines on any dataset, written in the shared predictions schema. |
| `eval/compare_arms.py`, `eval/all_domains_table.py`, `eval/report_domain_results.py`, `eval/zeroshot_baselines_table.py`, `eval/table91_row.py` | Table builders from per-query scores. |
| `eval/walk_prior*.py`, `eval/graph_channel_*.py`, `eval/ccmp_*.py`, `eval/gate_decomp_fig.py` | Graph-channel diagnostics: parameter-free walk prior, graph-alone curves, CCMP paired statistics and mechanism analyses (see `eval/ccmp_mechanism_README.md`). |
| `eval/scientific_path_interpretations.py`, `eval/table4.py`, `eval/showcase.py`, `eval/route_*.py`, `eval/pick_qualitative.py`, `eval/render_qualitative.py` | Path interpretations (NBFNet gradient beam search with the CCMP gate per hop), Table-4 style tables and the qualitative figures (see `eval/scientific_path_interpretations_README.md`). |
| `eval/test_*.py` | Unit tests for the analysis tooling (`python3 -m unittest discover -s sir4-retrieval/eval -p 'test_*.py'`). 25 tests; two need the merged graphs and Drive scan outputs in place, and `test_top_one_hop_is_not_hidden_and_invalid_paths_are_rejected` fails on the current path-attribution code (known, 14 Sep 2026). |
| `transfer/` | ResearchBench zero-shot protocol: pooled corpus, subsets, paired bootstrap. |
| `llm_baselines/` | MOOSE-Chem, MOOSE-Star and LATTICE runners on SIR-4 (import the original repositories). |
| `downstream/` | **Contribution 4.** Hypothesis composition and judging pipeline. |
| `notebooks/` | The Colab notebooks behind the thesis tables (see `notebooks/README.md`). |
| `results/` | Final tables (`main_table/`, `zeroshot_baselines/`, `mir/`, `sir4_llm_baselines/`, `llm_baselines_tomato/`) and qualitative outputs. |
| `colab_*_cell.py`, `colab_cells/` | Paste-in Colab cells. The ones at this level are inputs to the notebook generators in `prep/`; `colab_cells/` holds the stand-alone ones. Not importable modules. |

`PLAN.md` records the CS port plan and the six silent-failure bugs the smoke test caught; the
lesson (caches keyed by corpus, never by split alone) is what `cargo_paths.py` enforces.
