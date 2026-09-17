# SIR-4 and EAID

[Repository home](../README.md) · [Code guide](../docs/README.md) · [Evaluation](../experiments/eval/README.md)

SIR-4 is a benchmark for retrieving scientific inspirations from a
corpus across computer science, biology, physics, and materials science.
Equivalence-Aware Inspiration Decomposition (**EAID**) constructs inspiration sets for a
fixed target and searches for alternative sets that explain the same target.

**The dataset is included:** [dataset guide, counts and schema](dataset/README.md).
It contains the available train and test exports for all four fields.

## Where the contribution lives

| Code | Role |
|---|---|
| [03_new_method.py](build/03_new_method.py) | Fixed-target decomposition and integrated alternative-set search. |
| [03b_uniqueness.py](build/03b_uniqueness.py) | Additional controls and search for valid substitute inspiration sets. |
| [03_decompose.py](build/03_decompose.py) | Shared checks for necessity, sufficiency, background disjointness, and non-redundancy. |
| [05_export.py](build/05_export.py) | Corpus/query export and validated inspiration sets in `sets.json`. |
| [score_sir4.py](../experiments/eval/score_sir4.py) | Retrieval metrics and CompleteSet@k over the exported sets. |

## Construction pipeline

| Stage | Files | Produces |
|---|---|---|
| Collect papers | [01_collect.py](build/01_collect.py), [domains.yaml](config/domains.yaml) | Candidate source papers by field and date. |
| Acquire full text | [02_fulltext.py](build/02_fulltext.py), [02b_openalex_refs.py](build/02b_openalex_refs.py) | Structured text and reference information. |
| Decompose and search | [03_new_method.py](build/03_new_method.py), [03b_uniqueness.py](build/03b_uniqueness.py) | Fixed targets, primary inspirations, and candidate alternatives. |
| Resolve and label | [04_resolve.py](build/04_resolve.py), [labels.py](build/labels.py) | Matched source papers and field-relation labels. |
| Export | [05_export.py](build/05_export.py) | Corpora, queries, validated sets, and manifests. |
| Validate references | [test_matching.py](build/test_matching.py) | Reference-matching regression checks. |

For source construction, read each script's arguments before running it. Collection and
resolution contact external services; decomposition uses DeepSeek. Existing exported data
can be staged for retrieval without repeating construction:

```bash
# From the repository root: unpack the included data, then validate and stage it.
python sir-4/prepare_data.py --domain physics
python experiments/prep/stage_sir4.py --domain physics --verify-only
python experiments/prep/stage_sir4.py --domain physics
```

## Exports used by the retrieval experiments

These names come from [stage_sir4.py](../experiments/prep/stage_sir4.py).

| Field | Dataset key | Train export | Test export |
|---|---|---|---|
| Computer science | `sir4_cs` | `cs_train_final` | `cs_test_final` |
| Biology | `sir4_biology` | `biology_train_low` | `biology_test_low` |
| Physics | `sir4_physics` | `physics_train_low` | `physics_test_low` |
| Materials science | `sir4_matsci` | `matsci_train_low` | `matsci_test_low` |

Compressed exports are versioned under [dataset/](dataset/README.md). The preparation
command verifies their checksums and unpacks runtime copies to `sir-4/data/benchmark/`.
Those unpacked copies are ignored by Git. Use the dataset manifest for actual counts and
the documented Physics training discrepancy. The construction configuration contains earlier
planning choices, including mathematics; it is not the manifest of the included benchmark.

## Export compatibility and collection methodology

The export metadata key `quartet` is retained for compatibility with saved SIR-4 records.
Use `--sir4` when passing an export to an analysis command; `--quartet` remains an alias.
[Collection methodology](build/METHODOLOGY.md) records the source-selection rationale
and earlier collection measurements. Use the exported manifests for final counts and settings.
