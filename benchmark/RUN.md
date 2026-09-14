# QUARTET: commands

Run stages in order. Each reads the previous stage's output directory.
All stages resume: rerun the same command after a crash and it skips finished work.

## Setup

```bash
cd $SCIGRAPHIR_ROOT/quartet
export DEEPSEEK_API_KEY=sk-...          # from https://platform.deepseek.com
export OPENALEX_MAILTO=faizajalil0@gmail.com
```

---

## Stage 1: collect papers from OpenAlex

Free. One domain and split at a time, never concurrently: parallel collectors exceed
the polite-pool rate and OpenAlex returns empty pages that look like a finished result set.

```bash
python3 build/01_collect.py --domain cs --split train
```

```bash
python3 build/01_collect.py --domain cs --split test
```

Domains: `cs` `physics` `biology` `maths` `matsci`

**Status:** all five test splits done, 55,023 papers. Train splits blocked on the
OpenAlex daily budget, which resets at midnight UTC. Top up at
<https://openalex.org/pricing> if you do not want to wait; all five train splits cost
about $0.15.

---

## Stage 2: fetch full text

Free. arXiv LaTeX source for cs/physics/maths/matsci, PubMed Central XML for biology.
No PDFs, no OCR, no GPU.

**Use `--headroom`, not the bare command.** Stage 1 collects the whole eligible pool on
purpose, so its output is several times the benchmark target: 19,051 CS train papers
against a target of 6,000. Fetching all of it is 4.6x the work for no extra benchmark.
`--headroom` reads the targets from `config/domains.yaml`, including the materials
science override, so the number cannot drift as you move between domains.

```bash
python3 build/02_fulltext.py --domain cs --split both --headroom 1.4 --workers 12
```

1.4 covers the papers with no machine-readable text (measured: 4-9% for cs, physics,
maths and biology, 40% for matsci) plus margin for the Stage 3 quality gates.

A small reproducible sample, for a smoke test:

```bash
python3 build/02_fulltext.py --domain cs --split test --sample 100 --seed 42
```

**Check the warning at the end.** If many papers come back with an empty bibliography,
stop: inspirations must be grounded in the reference list, so decomposition would
either return nothing or invent citations.

Then audit the output. Every check corresponds to a bug that shipped at some point, so
a clean report means those failures are absent. It exits non-zero on a fatal one.

```bash
python3 build/audit_fulltext.py --domain cs --split both
```

Run it after the smoke test and again after the full fetch, before paying for Stage 3.

---

## Stage 3: decompose

Costs money. `deepseek-v4-pro` with thinking for decomposition, `v4-flash` for the gates.

Inspect the prompts without calling anything:

```bash
python3 build/03_decompose.py --domain cs --split test --limit 1 --dry-run
```

The 100-paper pilot, with the same-domain-bias ablation:

```bash
python3 build/03_decompose.py --domain cs --split test --probe-cross
```

Full domain:

```bash
python3 build/03_decompose.py --domain cs --split train
```

Lower `--workers` if DeepSeek returns 429s:

```bash
python3 build/03_decompose.py --domain cs --split test --workers 4
```

| Scope | Papers | Cost |
|---|---|---|
| Pilot with ablation | 100 | ~$2.80 |
| Computer Science, whole domain | 7,000 | ~$96 |
| All five domains | 35,000 | ~$480 |

Peak hours (09:00–12:00 and 14:00–18:00 Beijing time) cost 2x. Run outside them.

---

## Not written yet

- `04_resolve.py` — Semantic Scholar lookup turning `supposed_title` into
  `found_title` / `found_abstract` / `found_doi`. **Without this there is no corpus,
  so no benchmark.** Next thing to write.
- `05_stratify.py` — same/cross labelling and the 50/50 balance
- `06_export.py` — `documents.json` and `train.json` / `test.json`

---

## Where things land

```
data/01_collected/{domain}_{split}.jsonl     paper list + OpenAlex metadata
data/02_fulltext/{domain}_{split}.jsonl      + fulltext, bibliography
data/03_decomposed/{domain}_{split}.jsonl    + research_question, background_survey,
                                               inspiration[], detailed_breakdown
```
