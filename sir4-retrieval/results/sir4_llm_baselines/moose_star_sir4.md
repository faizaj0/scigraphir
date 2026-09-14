# MOOSE-Star (gpt-4o-mini, best-first, 5 proposals) on SIR-4, FULL test sets, run 6 Sep 2026
Scored with eval/score_sir4.py's score_one (recall over all golds, binary nDCG@5), same as every other row.
Physics scored 830 of 834 queries and MatSci 330 of 331 (API errors; rerunning the runner retries them).

| field | same n | same R@3 | same R@5 | same nDCG@5 | cross n | cross R@3 | cross R@5 | cross nDCG@5 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CS | 805 | 2.32 | 3.05 | 3.83 | 247 | 1.02 | 1.46 | 1.96 |
| Biology | 686 | 3.67 | 4.16 | 5.57 | 141 | 3.08 | 3.40 | 4.67 |
| Physics | 693 | 3.71 | 4.47 | 5.55 | 137 | 3.21 | 3.60 | 5.00 |
| MatSci | 209 | 5.61 | 6.44 | 8.03 | 121 | 5.68 | 7.31 | 9.02 |

Macro gap, nDCG@5: mean same 5.75, mean cross 5.16, delta -10.14%.  Macro gap, R@5: 4.53 -> 3.94, -12.97%.
For reference, TOMATO MOOSE-Star (500-query subset): same nDCG@5 4.96, cross 3.73.
Source: llm_baselines/outputs/moose_star/<field>/scores.json
