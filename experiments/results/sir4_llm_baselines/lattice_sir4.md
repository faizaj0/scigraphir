# LATTICE (gpt-4o-mini, top-down tree, NumI=40, beam 2) on SIR-4, subset (all cross + 250 same), live calls, 8 Sep 2026
Same-field n = 250 sampled per field (MatSci 210 = all); cross-field = every cross query. ~$60 live incl. false starts.

| field | same R@3 | same R@5 | same nDCG@5 | cross R@3 | cross R@5 | cross nDCG@5 |
|---|--:|--:|--:|--:|--:|--:|
| CS | 16.24 | 19.77 | 23.54 | 12.29 | 14.92 | 20.03 |
| Biology | 28.15 | 31.07 | 38.56 | 23.59 | 27.89 | 34.48 |
| Physics | 34.55 | 42.16 | 47.30 | 28.24 | 31.97 | 37.66 |
| MatSci | 27.65 | 34.71 | 38.91 | 25.50 | 31.63 | 36.78 |

Macro gap nDCG@5: same 37.08 -> cross 32.24 = -13.05%. Source: llm_baselines/outputs/lattice/<field>/scores.json
