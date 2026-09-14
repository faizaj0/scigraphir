# Downstream hypothesis quality on SIR-4 (Table 9.3 layout, four subsets)

Same protocol as thesis 9.2.6 on TOMATO: composer gpt-4o-mini with the MOOSE-Star delta prompt fed the
top-1 retrieved document; judge gpt-4o matched-score rubric (0-12); closed-book and oracle bound the range.
Queries = the four LLM-baseline subsets (all cross + 250 same, seed 42): cs 497 / bio 391 / phys 388 / matsci 331 = 1,607.

Code lives in `~/Desktop/Project_Week9/TOMATO-Star/analysis/downstream/` (same package as the TOMATO run);
`DOWNSTREAM_DATASET=sir4` re-points every script at `results/downstream_sir4/`. New: `build_inputs_sir4.py`,
`downstream_table.py` (Table 9.3 layout for either dataset; reproduces the TOMATO table from the existing files).

## What differs from TOMATO (state in the caption)
- Paper-level retrieval, several golds per paper (mean 4.3), gold list = union of all validated inspiration sets.
  **Top-1 hit = top-1 is any gold** (same convention as the SIR-4 recall metrics).
- **Judge reference**: TOMATO judged against the component the query's gold contributes. Here a composition made
  from a gold document is judged against that gold's own contribution text (its QUARTET `delta`; identical to
  `hypothesis_components[order]`, and every alternative-set gold has one too); a non-gold composition and the
  closed-book arm are judged against the primary inspiration's delta (order 0); the oracle feeds the primary
  inspiration and is judged against the same delta. (`--reference hypothesis` = judge everything against the
  full fine-grained hypothesis instead; lower scores, one reference per query.)
- Previous hypothesis is always empty (TOMATO step 0). 10-16% of documents are title-only and are fed as such.
- Top-1 hit rates are far higher than TOMATO's 13-17% (Qwen3 63.8%, MOOSE-Chem 53.0%, LATTICE 49.9% pooled),
  so the design ceiling (hit-rate gap x gold-vs-nongold gap) is much larger.

## 1. Files still needed (once)
```bash
pip install gdown
cd ~/Desktop/CARGO/sir4-retrieval/data
gdown 1BYKatK1xlqT_kL5Kgjgb58BljubKV1CD -O predictions_reasonir_sir4_cs_test.json
gdown 1vx7nrycFqu01soE8WIFH2m1OZ_Ckrij6 -O predictions_reasonir_sir4_biology_test.json
gdown 1fZAERzRh_Yx5U2fOEdHsYsR-SrhPTlIH -O predictions_reasonir_sir4_physics_test.json
gdown 1OJDXARPgLS-fxwI0FovS4KML19GWu1XI -O predictions_reasonir_sir4_matsci_test.json
```
SciGraphIR per-query SIR-4 predictions are on Drive only. Pick ONE set and save it as
`predictions_scigraphir_sir4_{field}_test.json` in the same folder (all are the CARGO prediction-list format):

| set | cs | biology | physics | matsci |
|---|---|---|---|---|
| OpenIE graph, current engine, in-field, 6 Sep (complete, recommended) | `1Lewi0tdqZY43Hy8c4ubX7aeL8D4Nmve7` | `1K_2U_-3mmGJmqKHtTaY9A36D9sb7AYYS` | `1ptZ4R9gw0Iim7c8VpZ8X-kmIQF1dvnN_` | `1BtkUIiUYzJjTK2HJ29pwRmgseJ5CxoJn` |
| frame graph + CCMP, current engine (`pred_frame_ccmp`, 8 Sep) | `11w2f5iGYL8XNtyV2t3cyz08XTOXl7hZ9` | `1TRn7uihMkHlDoET6YuTEJdiNSbePx5z1` | not on Drive | not on Drive |
| merged graph (`_hyb`, 8-10 Sep) | not run | `1SQpRdZr299eyLn4bZ41s22jMQDwPsQHy` | `1tu5_uQbTzr0Ue9ZnQBVzk5d1FpING8gj` | `164gsR4zuaJ6wT7oq6cFmC5TtS5y9DHwv` |
| Aug thesis-row frame+CCMP (cs only located): ccmpstrong / ccmpctrl | `1DauJNHQsnyWsm0HVAce7o2l9Qr_sIW1s` / `1VudULEJfNoguoa4d8y9o9P2SYKOkzqua` | | | |

Local already: Qwen3 (`data/predictions_qwen3_sir4_*`), MOOSE-Chem pool-100 and LATTICE subset rankings.

## 2. Run (in order; `export OPENAI_API_KEY=...` first)
```bash
bash run_downstream_sir4.sh build          # prints per-field top-1 hit rates per arm; no API
bash run_downstream_sir4.sh compose        # ~7 calls-worth per query, gpt-4o-mini, ~$7, ~15 min
bash run_downstream_sir4.sh judge-build    # prints the exact batch cost estimate; no API
bash run_downstream_sir4.sh judge-submit   # gpt-4o batch, ~$28
bash run_downstream_sir4.sh judge-fetch    # polls; writes matched files + summary__gpt-4o-batch.json
bash run_downstream_sir4.sh table          # Table 9.3 layout, pooled + per field -> results/downstream_sir4/table__gpt-4o-batch.tex
```
Drop LATTICE with `ARMS="none qwen3 reasonir moose_chem scigraphir oracle"` (saves ~$5).
Everything is resumable; `build` can be re-run after the downloads without losing arms already attached.

## Outputs (`TOMATO-Star/results/downstream_sir4/`)
`inputs_sir4.jsonl`, `compositions_{arm}.jsonl`, `matched_{arm}__gpt-4o-batch.jsonl`, `summary__gpt-4o-batch.json`
(ladder + paired bootstrap CIs + hit-conditioned split), `table__gpt-4o-batch.{tex,json}` (pooled table and a
per-field table, stars = SciGraphIR vs row significant overall and cross).
