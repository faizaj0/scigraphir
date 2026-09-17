# SIR-4: Data Collection and Split

Stage 1 of construction. Implemented in [`01_collect.py`](01_collect.py), parameterised
by [`../config/domains.yaml`](../config/domains.yaml).

## What this stage produces

- `data/01_collected/{domain}_{split}.jsonl` — one row per eligible paper
- `data/01_collected/collection_report.json` — drop counts, purity histogram, headroom

It does **not** sample down to the 6,000 / 1,000 targets. Balancing to 50/50
same/cross can only happen after extraction, because a paper's inspirations, and
therefore their domains, are unknown until it has been decomposed and resolved. So
Stage 1 collects the full eligible pool and a later stage subsamples it.

## Source

Papers are enumerated from **OpenAlex**, filtered to works that carry a
**structured-full-text record**: an arXiv location for Computer Science, Physics and
Materials Science, or a PubMed Central location for Biology.

This is a deliberate restriction and it is what removes MinerU from the pipeline.
TOMATO-Star converted PDFs to Markdown with MinerU because it worked from NCBI PDFs;
a PDF carries no structure, so layout detection and OCR are unavoidable. arXiv returns
the author's LaTeX source and PubMed Central returns JATS XML, both with sections
already marked up. No GPU stage exists in SIR-4.

## Temporal split

| Split | Window |
|---|---|
| Train | 2020-01-01 to 2025-09-30 |
| Test | 2025-10-01 to 2026-03-31 |

The split point of 1 October 2025 is TOMATO-Star's, adopted verbatim so that the
transfer result is not confounded by a different temporal regime. The test window is
widened from one month to six because a single month does not supply enough Materials
Science papers with structured full text: arXiv carries 127 in October 2025 against a
target of 1,000.

## Filters, in the order applied

| # | Filter | Rationale |
|---|---|---|
| 1 | `primary_topic.field` equals the domain's field, exactly | One field per domain, no unions. Asymmetric mappings distort cross-domain rates. |
| 2 | `type:article` and carries an arXiv or PMC location | Structured full text; removes the PDF and GPU stage |
| 3 | `purity >= 0.66` | See below |
| 4 | Not dated exactly 1 January | OpenAlex's dump for imprecise dates |
| 5 | Has a DOI and a reconstructable abstract of 200+ characters | Needed downstream |

### Purity

OpenAlex assigns each work up to three topics, each belonging to a field.
`primary_topic` is simply the highest-scoring one.

```
purity = (topics in the primary field) / (total topics)
```

Topic **scores** cannot be used to detect ambiguity. They are unnormalised and often
cluster near 1.0 — one real work scores 1.000, 0.998 and 0.997 across its three topics
— so a score-margin rule flags confident papers as ambiguous and misses genuinely
mixed ones.

At purity 1/3 the field label is a minority verdict, one topic outvoting two. A
measured example: *"Machine learning for perovskite materials design and discovery"* is
labelled **Engineering** on the strength of one topic, while its other two topics are
both **Materials Science**. If that paper became a gold inspiration for a Materials
Science query, the pair would be scored **cross-domain** when it is really
**same-domain**: a fabricated row in the exact stratum the thesis measures. Such rows
are the most damaging error available, because they inflate the metric the whole
contribution rests on.

We therefore keep purity 3/3 and 2/3 and drop 1/3. The cost is that genuinely
interdisciplinary papers are discarded, but those are precisely the papers whose single
domain label carries least information. `n_topics` is recorded per paper so the filter
can be tightened later without recollecting.

### The 1 January filter

OpenAlex assigns works with an imprecise publication date to 1 January. Measured: 1,086
Computer Science works claim 2026-01-01 against a daily average of about 115 for the
rest of that month, roughly ten times a normal day.

Those works could originate from any point in the year. Since the entire contamination
argument is *"the test set postdates the extraction model's knowledge cutoff"*, a paper
whose date cannot be trusted silently breaks it. They are dropped.

## Measured yields

Full enumeration of the test window (2025-10-01 to 2026-03-31), all five domains, no
sampling. Run 2026-07-31.

| Domain | Enumerated | Retained | Survival | Target | Headroom |
|---|---|---|---|---|---|
| Biology | 46,369 | 23,662 | 51.0% | 1,000 | 23.7x |
| Physics | 16,346 | 13,318 | 81.5% | 1,000 | 13.3x |
| Computer Science | 17,065 | 13,058 | 76.5% | 1,000 | 13.1x |
| Mathematics | 5,890 | 3,958 | 67.2% | 1,000 | 4.0x |
| **Materials Science** | 2,090 | **1,027** | 49.1% | 1,000 | **1.03x** |
| **Total** | 87,760 | **55,023** | 62.7% | 5,000 | |

Drops, as a share of works enumerated:

| Domain | Low purity | 1 January | No abstract | No DOI |
|---|---|---|---|---|
| Physics | 13.9% | 1.5% | 3.0% | 0.1% |
| Computer Science | 16.5% | 5.0% | 2.0% | <0.1% |
| Mathematics | 19.3% | 6.1% | 7.4% | 0 |
| Biology | 41.2% | 4.8% | 3.0% | <0.1% |
| Materials Science | 45.7% | 1.7% | 3.4% | 0 |

Two disciplines lose over 40% of their works to the purity filter. Both are
intrinsically boundary-spanning: Materials Science straddles engineering, chemistry and
physics, and OpenAlex's *Biochemistry, Genetics and Molecular Biology* field overlaps
heavily with *Medicine*. This is a property of the disciplines, not of the filter, and
it is precisely why an unfiltered field label would be unsafe to build a same/cross
metric on.

## Open issue: Materials Science

Materials Science clears the 1,000-paper test target by 27 papers. The 50/50 balance
requires `1 / (2c)` over-collection, where `c` is the natural cross-domain rate, so at
any plausible `c` this pool is too small.

Options, measured:

| Option | Test headroom | Cost |
|---|---|---|
| Keep as is, decompose all 1,027, balance what emerges | 1.03x | none; final row count below target |
| 12-month test window | ~1.5x | test window no longer tight |
| Add PDFs via MinerU | ample (8,241 OA works/month) | 20-40 GPU-hours, reinstates the GPU stage |
| Drop to four domains | n/a | loses a discipline |

Every other domain has 4x headroom or better, so this decision affects Materials
Science alone.

## Usage

```bash
python build/01_collect.py --domain cs --split test --limit 600   # pilot
python build/01_collect.py --domain matsci --split test           # one pool
python build/01_collect.py --all                                  # everything
```
