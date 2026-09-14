# Zero-shot: trained on CS + MatSci, evaluated on Physics and Biology (run 6 Sep 2026, ccmp standard, best epoch 7)
nDCG@5 (%). Same / cross / gap = (mean cross - mean same) / mean same over the two held-out fields.

| Method | Phys same | Bio same | Phys cross | Bio cross | gap |
|---|--:|--:|--:|--:|--:|
| BM25 | 49.21 | 49.69 | 37.57 | 39.63 | -21.94% |
| BGE-large | 57.09 | 61.18 | 45.92 | 49.70 | -19.14% |
| Qwen3-Embedding | 58.34 | 61.62 | 51.06 | 48.12 | -17.33% |
| SPECTER2 | 43.23 | 51.04 | 33.73 | 38.19 | -23.71% |
| SciNCL | 43.06 | 51.03 | 34.33 | 38.22 | -22.89% |
| ReasonIR-8B | 57.00 | 56.90 | 45.02 | 46.57 | -19.59% |
| SciGraphIR (CS + MatSci) | 52.40 | 56.70 | 43.99 | 45.17 | -18.27% |

R@10 (%): SciGraphIR is best on Physics cross (54.46 vs Qwen3 52.44) and Physics all (63.54 vs 63.52),
second on Biology cross (50.91 vs BGE 52.46). Head ranking loses, depth recall holds.
Full tables (cross/same/all, R@3 R@5 nDCG@5 MRR R@10): Drive outputs/sir4_zeroshot/table93_cs+matsci_to_physics+biology.md
