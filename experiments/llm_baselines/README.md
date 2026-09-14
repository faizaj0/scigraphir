# LLM-retrieval baselines on SIR-4 (MOOSE-Chem, LATTICE, MOOSE-Star)

Same three methods and settings as the TOMATO rows of the main table
(`results/llm_baselines_tomato/README.md`), run per SIR-4 field with the field's own test
corpus as the candidate pool, scored with `eval/score_sir4.py` (recall over all golds,
binary nDCG@5), reported as same / cross R@3, R@5, nDCG@5.

All three runners import the existing code in `$EXTERNAL_REPOS/{TOMATO-Star,
MOOSE-Chem, MOOSE-Star, llm-guided-hierarchical-search}` unchanged; only the loader is new
(`sir4_llm_data.py`). One exception: LATTICE's `src/prompts.py` gets four alias lines so the
SIR-4 field names map to the inspiration relevance definition (else it silently uses the
StackExchange one). `sir4_to_lattice.py --patch-prompts` does it, idempotently.

## Query subset

SIR-4 does not have 250 cross-field queries per field, so the subset is
**all cross-field queries + 250 sampled same-field queries** (seed 42; matsci runs in full).
The cross column is therefore exact; only the same column is a sample.

| field   | docs  | same (of) | cross | queries |
|---------|------:|----------:|------:|--------:|
| cs      | 4,181 | 250 / 805 |   247 |     497 |
| biology | 3,453 | 250 / 686 |   141 |     391 |
| physics | 3,144 | 250 / 696 |   138 |     388 |
| matsci  | 1,296 | 210 / 210 |   121 |     331 |
| total   |       |           |   647 |   1,607 |

`python3 llm_baselines/make_sir4_subset.py` writes `subsets/subset_<field>_same250_allcross_seed42.json`
(already done). `--full` for every query (3,044).

## Data conventions (sir4_llm_data.py)

- query: `research_question` = text through the first `?`; `background_survey` = the rest.
  Every SIR-4 question has that shape, matching TOMATO's two fields.
- corpus: `title` = text before the first `". "`, `abstract` = the rest. 10-16% of docs are
  title-only (abstract ""). MOOSE-Star's tree builder drops abstract-less papers, so those get
  the placeholder "No abstract available." there.
- titles are made unique with " (2)" suffixes (1-5 duplicates per field) because MOOSE-Chem
  and MOOSE-Star key on the title.

## Cost and time (gpt-4o-mini, subset protocol)

| method | calls / query | subset cost | full-set cost | wall-clock |
|---|---|--:|--:|---|
| MOOSE-Chem approach3-soft | 171 (matsci) to 540 (cs) | ~$300 (cs 121, bio 79, phys 71, matsci 25) | ~$600 | ~2 h/field at 8 x 32 workers |
| LATTICE NumI=40 beam 2 | 80 + tree (N/20 + 0.17 N) | ~$60 (~$35 with BATCH=1) | ~$110 (~$60 with BATCH=1) | ~1 h/field at 200 concurrent; BATCH=1: 40 batch jobs/field, minutes to hours each |
| MOOSE-Star best-first, 5 proposals (default; 25 gives identical @5) | ~8 | ~$6 | ~$12 | ~20 min/field at 16 workers |

`run_moose_chem_sir4.py --field X --estimate` prints the exact call count for any subset.

## Commands (run in `experiments/llm_baselines/`, `export OPENAI_API_KEY=...`)

MOOSE-Star (cheapest; do first, it validates the loader and scorer end to end):
```bash
python3 build_moose_star_tree_sir4.py --field matsci            # SPECTER2 k-means tree, no LLM, minutes on CPU
python3 run_moose_star_sir4.py --field matsci --limit 3          # smoke
python3 run_moose_star_sir4.py --field matsci --num-workers 16
python3 score_rankings_sir4.py --field matsci --rankings outputs/moose_star/matsci/rankings.json --method "MOOSE-Star (gpt-4o-mini)"
```

LATTICE (one script: data, tree, search, score; resumable at every step):
```bash
bash run_lattice_sir4.sh matsci tree                              # build the tree, inspect trees/SIR-4/matsci/build.log
CONC=200 bash run_lattice_sir4.sh matsci                          # search + score
BEAM=5 CONC=200 bash run_lattice_sir4.sh matsci                   # optional beam-5 variant
BATCH=1 bash run_lattice_sir4.sh matsci                           # half price via the OpenAI Batch API (search only)
```

BATCH=1 uses `OpenAIBatchAPI` (added to LATTICE's `src/llm_apis.py`, flag `--openai_batch` in
`src/hyperparams.py`, wired in `src/run.py`): identical request bodies and model, one batch job per
iteration, missing/invalid lines re-run live, batch ids saved in `results/SIR-4/<field>/batch_state-*.json`
so a killed run re-attaches instead of paying twice. Smoke it first: `python3 smoke_openai_batch.py` in
the LATTICE repo (3 prompts, under a cent).

MOOSE-Chem (the expensive one; pilot on matsci, then cs):
```bash
python3 run_moose_chem_sir4.py --field matsci --estimate
python3 run_moose_chem_sir4.py --field matsci --limit 3                 # smoke, ~$0.25
python3 run_moose_chem_sir4.py --field matsci --workers 8 --window-workers 32
python3 run_moose_chem_sir4.py --field cs --workers 8 --window-workers 32
python3 score_rankings_sir4.py --field cs --rankings outputs/moose_chem/cs/rankings.json --method MOOSE-Chem
```

Every runner writes `outputs/<method>/<field>/rankings.json` = `{query_id: [doi, ...]}` best
first and resumes from it; `score_rankings_sir4.py` writes `scores.json` next to it and prints
the table row. `--subset none` on any runner = every query of the field.

## MOOSE-Star and multiple golds

The published best-first search stops at the first proposal equal to the single gold and
reports `propose_rank`. SIR-4 has ~4 golds per query, so `run_moose_star_sir4.py` runs the
search to the proposal cap without stopping and scores the proposal order as a ranking.
For one gold this is identical to `propose_rank`. The cap defaults to 5: the search is
deterministic, so the first 5 proposals (all the table needs) are the same as under
TOMATO's cap of 25, at ~40% fewer calls. `--max-proposals 25` if you want R@10. The TOMATO row is MS-IR-7B; this is gpt-4o-mini (the TOMATO
gpt-4o-mini variant scored same R@5 3.6 / cross 2.4), label it as such.
