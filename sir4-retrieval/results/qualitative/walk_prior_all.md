| dataset | graph | stratum | queries | seeded | walk nDCG@5 | R@5 | R@10 |
|---|---|---|--:|--:|--:|--:|--:|
| sir4_cs | SciAffordGraph | all | 1052 | 100.0% | 19.47 | 47.1 | 57.7 |
| sir4_cs | SciAffordGraph | same | 805 | 100.0% | 20.22 | 47.6 | 58.0 |
| sir4_cs | SciAffordGraph | cross | 247 | 100.0% | 17.04 | 45.7 | 56.7 |
| sir4_cs | OpenIE | all | 1052 | 99.9% | 13.11 | 37.4 | 51.0 |
| sir4_cs | OpenIE | same | 805 | 99.9% | 13.30 | 37.8 | 51.1 |
| sir4_cs | OpenIE | cross | 247 | 100.0% | 12.50 | 36.0 | 50.6 |
| sir4_biology | SciAffordGraph | all | 827 | 100.0% | 25.56 | 60.3 | 70.4 |
| sir4_biology | SciAffordGraph | same | 686 | 100.0% | 26.33 | 60.1 | 70.0 |
| sir4_biology | SciAffordGraph | cross | 141 | 100.0% | 21.82 | 61.7 | 72.3 |
| sir4_biology | OpenIE | all | 827 | 100.0% | 28.67 | 67.1 | 76.8 |
| sir4_biology | OpenIE | same | 686 | 100.0% | 29.38 | 67.9 | 77.3 |
| sir4_biology | OpenIE | cross | 141 | 100.0% | 25.19 | 63.1 | 74.5 |
| sir4_physics | SciAffordGraph | all | 834 | 100.0% | 24.73 | 57.1 | 68.0 |
| sir4_physics | SciAffordGraph | same | 696 | 100.0% | 25.86 | 59.2 | 69.3 |
| sir4_physics | SciAffordGraph | cross | 138 | 100.0% | 19.02 | 46.4 | 61.6 |
| sir4_physics | OpenIE | all | 834 | 100.0% | 22.86 | 59.0 | 70.0 |
| sir4_physics | OpenIE | same | 696 | 100.0% | 23.45 | 59.5 | 71.4 |
| sir4_physics | OpenIE | cross | 138 | 100.0% | 19.92 | 56.5 | 63.0 |
| sir4_matsci | SciAffordGraph | all | 331 | 100.0% | 23.16 | 53.8 | 68.0 |
| sir4_matsci | SciAffordGraph | same | 210 | 100.0% | 24.57 | 57.6 | 70.0 |
| sir4_matsci | SciAffordGraph | cross | 121 | 100.0% | 20.71 | 47.1 | 64.5 |
| sir4_matsci | OpenIE | all | 331 | 100.0% | 21.79 | 55.0 | 70.1 |
| sir4_matsci | OpenIE | same | 210 | 100.0% | 22.69 | 56.7 | 71.4 |
| sir4_matsci | OpenIE | cross | 121 | 100.0% | 20.22 | 52.1 | 67.8 |
| tomato | SciAffordGraph | all | 3132 | 100.0% | 14.42 | 21.1 | 28.6 |
| tomato | SciAffordGraph | same | 2843 | 100.0% | 14.84 | 21.7 | 29.3 |
| tomato | SciAffordGraph | cross | 289 | 100.0% | 10.35 | 15.2 | 21.5 |
| tomato | OpenIE | all | 3132 | 100.0% | 13.80 | 19.0 | 25.7 |
| tomato | OpenIE | same | 2843 | 100.0% | 14.25 | 19.6 | 26.6 |
| tomato | OpenIE | cross | 289 | 100.0% | 9.33 | 13.1 | 16.6 |
| mir | SciAffordGraph | all | 155 | 100.0% | 12.67 | 23.2 | 30.3 |
| mir | OpenIE | all | 155 | 100.0% | 13.84 | 23.2 | 29.7 |

walk = 3-step personalised PageRank (restart 0.15) from the query's seed nodes over the symmetrised, degree-normalised stage-1 graph, documents ranked by mass; no parameters. R@k = share of queries with a gold in the top k (any-gold). seeded = queries with at least one seed node in the graph; unseeded queries score 0.
