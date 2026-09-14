# LLM-retrieval baselines on TOMATO-Star: nDCG@5 rescored from saved runs (2026-09-06)

All three rows are the 500-query stratified subset (250 same / 250 cross, seed 42,
`TOMATO-Star/results/stratified_250same_250cross_seed42.json`; strata from
`llm-guided-hierarchical-search/tomato_star_baselines_bundle/query_strata.json`).
No LLM calls were made; every number is a rescore of a saved per-query ranking.

| Method | same R@3 | same R@5 | same nDCG@5 | cross R@3 | cross R@5 | cross nDCG@5 | source |
|---|--:|--:|--:|--:|--:|--:|---|
| MOOSE-Chem (gpt-4o-mini, approach3-soft) | 20.0 | 23.6 | 19.48 | 13.6 | 18.0 | 14.01 | `TOMATO-Star/outputs/caches/moose_chem/rankings_moose-chem-gpt-4o-mini-approach3-soft_stratified_*.json` (tiered survivorship ranking) |
| LATTICE (gpt-4o-mini, NumI=40, beam 2) | 20.0 | 22.4 | 17.17 | 7.6 | 10.0 | 6.90 | `llm-guided-hierarchical-search/results/TOMATO-Star/full/*NumI=40*MaxBS=2.pkl` via `compute_r1_r5.py` |
| MOOSE-Star (MS-IR-7B, sglang) | 5.2 | 6.4 | 4.96 | 4.0 | 4.4 | 3.73 | Drive `TOMATO-STAR/results/msir7b_stratified_n500_p50{,_cross}/results_incremental.jsonl`, `propose_rank` |
| MOOSE-Star (gpt-4o-mini) | 2.8 | 3.6 | 2.65 | 2.4 | 2.4 | 2.05 | `TOMATO-Star/results/gpt4omini_stratified_n500_p25/results.jsonl`, `propose_rank` |

MOOSE-Chem and MOOSE-Star gpt-4o-mini R@1/R@5 reproduce the thesis table exactly;
MS-IR-7B same R@1 is 3.2 here vs 3.6 in the table (later results file).
LATTICE beam-5 variant: same 26.4 / 19.1, cross 15.6 / 10.6 (R@5 / nDCG@5).

LATTICE R@3 recomputed 2026-09-06 with compute_recall(top-3 leaf paths) from the same beam-2 run
(R@5 / nDCG@5 reproduced to 2 dp in the same pass).
