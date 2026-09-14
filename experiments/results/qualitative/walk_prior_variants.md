| dataset | stratum | queries | base | direct | hybrid | invdeg | direct+hybrid | direct+hybrid+invdeg |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| sir4_cs | all | 1052 | 19.47 | 15.94 | 15.88 | 16.15 | 18.28 | 14.71 |
| sir4_cs | same | 805 | 20.22 | 15.57 | 16.28 | 16.27 | 18.44 | 15.06 |
| sir4_cs | cross | 247 | 17.04 | 17.14 | 14.55 | 15.74 | 17.75 | 13.56 |
| sir4_biology | all | 827 | 25.56 | 24.22 | 32.29 | 23.58 | 34.45 | 29.47 |
| sir4_biology | same | 686 | 26.33 | 24.53 | 33.00 | 24.00 | 35.19 | 29.88 |
| sir4_biology | cross | 141 | 21.82 | 22.70 | 28.84 | 21.57 | 30.86 | 27.47 |
| sir4_physics | all | 834 | 24.73 | 21.56 | 26.41 | 21.18 | 28.27 | 24.21 |
| sir4_physics | same | 696 | 25.86 | 22.04 | 27.46 | 21.93 | 29.20 | 24.86 |
| sir4_physics | cross | 138 | 19.02 | 19.15 | 21.11 | 17.41 | 23.59 | 20.94 |
| sir4_matsci | all | 331 | 23.16 | 21.10 | 24.91 | 20.73 | 28.42 | 23.77 |
| sir4_matsci | same | 210 | 24.57 | 21.81 | 26.17 | 22.01 | 30.16 | 25.26 |
| sir4_matsci | cross | 121 | 20.71 | 19.85 | 22.72 | 18.50 | 25.40 | 21.17 |
| tomato | all | 3132 | 14.42 | 18.75 | 17.96 | 13.70 | 21.91 | 17.55 |
| tomato | same | 2843 | 14.84 | 19.05 | 18.49 | 13.89 | 22.46 | 17.86 |
| tomato | cross | 289 | 10.35 | 15.81 | 12.81 | 11.86 | 16.49 | 14.49 |
| mir | all | 155 | 12.67 | 11.55 | 15.18 | 9.86 | 15.73 | 13.97 |

walk nDCG@5 (%), 3-step PPR restart 0.15 from the query's seeds. direct = paper->frame shortcuts for the mechanism frames a paper already owns via its methods/tasks/findings; hybrid = + OpenIE entity->paper mention edges for entities of OpenIE degree <= 30, and the query's OpenIE entity seeds; invdeg = restart mass per seed proportional to 1/degree.

## Cap sweep and seed ablation (walk nDCG@5 %)

| dataset | stratum | base | hyb cap10 | hyb cap30 | hyb cap100 | hyb no cap | dir+hyb cap10 | dir+hyb cap30 | dir+hyb cap100 | dir+hyb no cap | dir+hyb edges only (frame seeds) cap30 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| sir4_cs | all | 19.5 | 15.3 | 15.9 | 16.1 | 16.0 | 17.3 | 18.3 | 18.4 | 18.6 | 15.6 |
| sir4_cs | cross | 17.0 | 13.6 | 14.6 | 15.0 | 15.0 | 16.8 | 17.7 | 18.0 | 18.2 | 16.8 |
| sir4_biology | all | 25.6 | 29.7 | 32.3 | 32.7 | 32.7 | 31.8 | 34.4 | 34.8 | 34.8 | 23.4 |
| sir4_biology | cross | 21.8 | 27.2 | 28.8 | 29.2 | 29.0 | 28.5 | 30.9 | 30.9 | 30.8 | 21.4 |
| sir4_physics | all | 24.7 | 25.3 | 26.4 | 26.7 | 26.6 | 26.8 | 28.3 | 28.4 | 28.3 | 21.1 |
| sir4_physics | cross | 19.0 | 20.9 | 21.1 | 21.5 | 21.5 | 22.6 | 23.6 | 23.6 | 23.5 | 18.0 |
| sir4_matsci | all | 23.2 | 24.3 | 24.9 | 25.1 | 25.1 | 26.8 | 28.4 | 28.2 | 28.3 | 21.2 |
| sir4_matsci | cross | 20.7 | 21.5 | 22.7 | 22.7 | 22.7 | 23.6 | 25.4 | 25.0 | 25.1 | 19.1 |
| tomato | all | 14.4 | 16.8 | 18.0 | 18.0 | 18.0 | 21.2 | 21.9 | 22.0 | 22.0 | 18.6 |
| tomato | cross | 10.4 | 11.9 | 12.8 | 12.7 | 12.9 | 16.2 | 16.5 | 16.8 | 16.9 | 16.5 |
| mir | all | 12.7 | 15.4 | 15.2 | 14.8 | 15.2 | 15.3 | 15.7 | 15.7 | 15.5 | 11.1 |

The gain needs the query's ENTITY SEEDS: mention edges with frame seeds only (last column) fall below base on every SIR-4 field.

## Directed mention edges (entity -> paper only), cap 30

| dataset | stratum | base | dir+hyb symmetric | hyb directed | dir+hyb directed | OpenIE alone |
|---|---|--:|--:|--:|--:|--:|
| sir4_cs | all | 19.5 | 18.3 | 15.5 | 18.7 | 13.1 |
| sir4_cs | cross | 17.0 | 17.7 | 14.3 | 18.2 | 12.5 |
| sir4_biology | all | 25.6 | 34.4 | 31.8 | 34.8 | 28.7 |
| sir4_biology | cross | 21.8 | 30.9 | 27.2 | 31.9 | 25.2 |
| sir4_physics | all | 24.7 | 28.3 | 26.6 | 29.5 | 22.9 |
| sir4_physics | cross | 19.0 | 23.6 | 21.9 | 25.2 | 19.9 |
| sir4_matsci | all | 23.2 | 28.4 | 24.7 | 29.5 | 21.8 |
| sir4_matsci | cross | 20.7 | 25.4 | 22.8 | 27.5 | 20.2 |
| tomato | all | 14.4 | 21.9 | 17.8 | 22.5 | 13.8 |
| tomato | cross | 10.4 | 16.5 | 13.5 | 18.2 | 9.3 |
| mir | all | 12.7 | 15.7 | 16.3 | 15.9 | 13.8 |
