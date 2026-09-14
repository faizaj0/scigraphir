# Zero-shot baseline rows for Table 9.3 (SIR-4 CS + MatSci, cross-field queries)

Decoded on 2026-09-05 from Drive `cargo-gfmrag/outputs/baselines/sir4_{cs,matsci}/scores_<arm>.json`,
written by `sir4_baselines_all.ipynb` (one runner, one scorer, `--topk 300`, slices all/same/cross).
Regenerate the table with `python3 eval/zeroshot_baselines_table.py`.

Provenance check against the printed Table 9.3: BGE-large matches to four decimals on both
fields. BM25 and Qwen3-Embedding differ (thesis rows came from the transfer-matrix runner, a
different BM25 tokeniser and a different Qwen3 instruction). Use one source for all six rows.
