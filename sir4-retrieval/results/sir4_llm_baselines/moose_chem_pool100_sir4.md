# MOOSE-Chem (gpt-4o-mini, approach3-soft) on SIR-4, Qwen3 top-100 POOL per query, subset (all cross + 250 same), 8 Sep 2026
Same-field n = 250 per field (MatSci 210 = all); cross-field = every cross query. ~$12 total. Pool recall@100 (ceiling): CS 64.2, Bio 82.4, Phys 83.5, MS 82.2.

| field | same R@3 | same R@5 | same nDCG@5 | cross R@3 | cross R@5 | cross nDCG@5 |
|---|--:|--:|--:|--:|--:|--:|
| CS | 12.98 | 23.50 | 25.18 | 12.06 | 20.05 | 23.74 |
| Biology | 20.54 | 36.80 | 40.06 | 14.29 | 26.03 | 28.41 |
| Physics | 22.70 | 39.45 | 41.27 | 18.15 | 31.80 | 33.77 |
| MatSci | 17.56 | 30.89 | 33.27 | 18.20 | 29.78 | 32.68 |

Macro gap nDCG@5: same 34.94 -> cross 29.65 = -15.14%. Source: llm_baselines/outputs/moose_chem/<field>_pool100/scores.json
