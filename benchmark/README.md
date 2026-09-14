# QUARTET (the SIR-4 benchmark)

> **QUARTET was the working name of SIR-4**, the benchmark of thesis Chapter 8. The code below
> builds it. Equivalence-Aware Inspiration Decomposition (EAID) is `build/03_new_method.py`
> (fixed-target decomposition) with `build/03b_uniqueness.py` (leave-one-out search for
> alternative validated inspiration sets); the four validation criteria (necessity, sufficiency,
> background disjointness, non-redundancy) are the shared checker in `build/03_decompose.py`;
> `build/05_export.py` writes every validated set to `sets.json`, which
> `../experiments/eval/score_sir4.py` reads for CompleteSet@k. The README text that follows
> is the construction-time documentation and still uses the working name and the five-domain
> plan; the released benchmark has four fields (mathematics was dropped).


A four-discipline benchmark for **corpus-scale cross-domain inspiration retrieval**:
Computer Science, Physics, Biology, Materials Science.

Extraction follows TOMATO-Star (Yang & Bing, 2026). Evaluation construction differs,
because TOMATO-Star builds a 15-candidate selection task and QUARTET measures
retrieval over a pooled corpus.

## Why it exists

Every benchmark in this lineage scores re-ranking inside a pre-filtered candidate set
that already contains the answer:

| Benchmark | Candidates per query | Random baseline |
|---|---|---|
| MOOSE-Chem | 300 (mostly random filler) | - |
| TOMATO-Star | 15 (1 positive, 14 negatives) | 6.70% |
| ResearchBench | 75, scored at top-20% and top-4% | ~4% at top-4% |

None of them measures whether a system can find an inspiration in a corpus. QUARTET does.

Second gap: TOMATO-Star's 108,717 papers all come from NCBI, so biology, chemistry,
medicine, medical imaging, psychology and cognitive science are all life sciences.
Cross-domain retrieval cannot be measured within one scientific culture. QUARTET's four
disciplines do not share a vocabulary.

## Design at a glance

| | |
|---|---|
| Source papers | 9,383 train + 1,658 test **per discipline**, 44,164 total |
| Corpus | 7,000 train + 3,033 test per discipline, **pooled at eval time** |
| Split | Jan 2020 to Sept 2025 train, **October 2025 test** (TOMATO-Star's split verbatim) |
| Query | One per (paper, inspiration step), id `{year}_{id}::{step}` |
| Gold source | The paper's own **citations**, resolved via Semantic Scholar |
| Domain labels | OpenAlex `primary_topic.field`, one field per discipline, **no unions** |
| Stratum balance | **50/50 same/cross**, pooled, by down-sampling same-domain eval rows |
| Corpus composition | Golds only; documents dropped from eval stay in as distractors |

All parameters live in [`config/domains.yaml`](config/domains.yaml). Change them there,
not in the code or the thesis text.

## Two things to know before reading results

**QUARTET is balanced, not natural.** The real cross-domain rate is about 9%. QUARTET
forces 50%. That is a deliberate stress test to give the cross-domain stratum enough
statistical power, and it overstates how often cross-domain retrieval is needed in
practice. Never report the aggregate without saying so.

**The biology split collides with TOMATO-Star's test set.** Both are October 2025 NCBI
papers. This is not model contamination (the model never trains on TOMATO-Star test),
but QUARTET-Biology and TOMATO-Star test are not independent evaluations.

## Construction stages

| Stage | What | Status |
|---|---|---|
| 0 | Configuration and domain assignment rule | done, `config/domains.yaml` |
| 1 | Paper collection per discipline | not started |
| 2 | Full-text acquisition (arXiv source, PMC XML, fallbacks) | not started |
| 3 | Decomposition into (background, hypothesis, inspirations) | not started |
| 4 | Gold resolution via Semantic Scholar | not started |
| 5 | Four quality gates: necessity, sufficiency, disjointness, non-redundancy | not started |
| 6 | Query construction, one per (paper, step) | not started |
| 7 | Corpus assembly, pooling, stratification, 50/50 balance | not started |
| 8 | Validation: fidelity vs TOMATO-Star + human spot-check | not started |
| 9 | Contamination audit vs TOMATO-Star | reusable from `retriever/researchbench/` |
| 10 | Release in `raw/documents.json` + `raw/{train,test}.json` | not started |

Neither TOMATO-Star nor ResearchBench released construction code. ResearchBench's
repo (`ankitala/ResearchBench`) is an evaluation toolkit plus a data standardiser;
MOOSE-Star's covers training and inference only. Stages 1 to 7 are a reimplementation
from the papers, which is why Stage 8 exists.

## Layout

```
benchmark/
  config/domains.yaml            single source of truth
  build/                         construction scripts (01_collect ... 05_export, EAID in 03_new_method / 03b_uniqueness)
  audit/                         LLM audit of the decompositions
  data/                          outputs (git-ignored; the release zip unpacks the exports here)
```
