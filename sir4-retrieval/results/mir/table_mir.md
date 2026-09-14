# MIR test (155 proposals, extended corpus of 4,857 abstracts), in-benchmark training

| Method | R@3 | R@5 | nDCG@5 | mAP |
|---|--:|--:|--:|--:|
| BM25 | 20.85 | 27.77 | 20.07 | 18.52 |
| BGE-large | 24.90 | 31.68 | 23.68 | 22.31 |
| Qwen3-Embedding | 31.03 | 35.71 | 27.47 | 25.95 |
| SPECTER2-base | 23.35 | 26.60 | 21.09 | 21.20 |
| SciNCL | 18.87 | 25.02 | 18.70 | 18.73 |
| ReasonIR-8B | 27.14 | 37.59 | 27.67 | 25.38 |
| Multi-View Semantic Scorer | 29.85 | 40.55 | 30.93 | 28.94 |
| + Graph Reasoner | 30.92 | 40.55 | 31.22 | 29.55 |
| + CCMP (full SciGraphIR) | 32.22 | 41.78 | 31.68 | 29.57 |

Best baseline per column: Qwen3 (R@3, mAP), ReasonIR (R@5, nDCG@5).
Full SciGraphIR over best baseline: R@3 +1.19, R@5 +4.19, nDCG@5 +4.01, mAP +3.62.
Cumulative: scorer over Qwen3 (its input) nDCG@5 +3.46; graph +0.29; CCMP +0.46.
Run: colab_mir_standard.ipynb, 2026-09-06, CCMP standard, 10 epochs batch 2.

LLM-retrieval rows (8 Sep 2026, all 155 proposals): LATTICE R@3 15.72 / R@5 17.62 / nDCG@5 15.61 / mAP 14.90;
MOOSE-Star R@3 0.81 / R@5 2.10 / nDCG@5 1.33 / mAP 1.02 (4 of 155 queries have a gold among its 5 proposals; ~3.8 calls/query).
MOOSE-Chem pool-100 (Qwen3 top-100 per query, pool recall@100 73.66%), 8 Sep, all 155: R@3 11.02 / R@5 21.69 / nDCG@5 14.71 / mAP 14.90; 44/155 queries have a gold in the top 5; ~$1.
