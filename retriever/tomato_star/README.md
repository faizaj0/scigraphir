# TOMATO-Star staging

Turns a TOMATO-Star checkout (Yang and Bing, 2026) into the corpus layout the pipeline
reads: `retriever/data/tomato_{train,test}/raw/{documents.json, {train,test}.json}`.
The other benchmarks are staged by `experiments/prep/stage_sir4.py` (SIR-4),
`experiments/prep/stage_mir.py` (MIR) and `retriever/researchbench/` (ResearchBench).

`config.py` expects the checkout at `$EXTERNAL_REPOS/TOMATO-Star` (default
`<repo>/external/TOMATO-Star`) with `data/{train,test}.jsonl` and
`outputs/domain_outputs/`. Intermediate files go to `<repo>/outputs/caches/`.

Run the modules from `retriever/`, in this order; each docstring says what it reads and writes.

| Step | Command | What |
|---|---|---|
| 1 | `python3 -m tomato_star.data` | test-split retrieval task: corpus, queries, golds, same / cross strata |
| 2 | `python3 -m tomato_star.resolve` | resolve gold-inspiration domains that OpenAlex could not (`OPENAI_API_KEY`) |
| 3 | `python3 -m tomato_star.build_train` | carve train and dev bundles from the 107k-paper train split |
| 4 | `python3 -m tomato_star.resolve_train` | resolve train-split domains the same way |
| 5 | `python3 -m tomato_star.train_strata` | same / cross labels per training query (`train_strata.jsonl`) |
| 6 | `python3 tomato_star/select_train_subset.py` | the 7,000-document training corpus (`train_selection.json`, included, seed 42) |
| 7 | `python3 tomato_star/make_inputs.py` | write the staged corpora |

`explore.py` reproduces the dataset statistics and figures; `eval.py` and `baselines.py`
are the original stratum-aware scorer and the BM25 / BGE baselines used during exploration.
