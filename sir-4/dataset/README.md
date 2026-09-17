# SIR-4 dataset

[Benchmark and construction](../README.md) · [Repository home](../../README.md) · [Setup](../../docs/SETUP.md)

This directory contains the complete available **SIR-4** train and test exports for
computer science, biology, physics and materials science. Each export contains its
candidate paper corpus, research queries, gold inspiration sets, primary-only labels
and construction metadata. These are dataset records, not model predictions.

The packaged files occupy about **81 MB**; unpacking creates about **502 MB** of JSON.
They require no separate download, API key or Git LFS installation.

## Use the data

From the repository root:

```bash
python sir-4/prepare_data.py
python experiments/prep/stage_sir4.py --domain physics --verify-only
python experiments/prep/stage_sir4.py --domain physics
```

The first command checks SHA-256 hashes and unpacks all fields to
`sir-4/data/benchmark/`. Use `--domain physics` to unpack just that field, or
`--verify-only` to verify the packaged data without writing files. Existing identical
exports are reused; differing exports are left unchanged and reported as a conflict.
`--out /path/to/exports` writes another unpacked copy.

The staging command converts the exports to the retrieval input layout under
`retriever/data/sir4_<field>_<split>/raw/`. Replace `physics` with `cs`, `biology` or
`matsci` for the other fields. Unpacked and staged copies are ignored by Git.

## Included splits

| Field | Split | Export directory | Queries | Candidate documents |
|---|---|---|---:|---:|
| Computer science | Train | `cs_train_final` | 5,445 | 20,203 |
| Computer science | Test | `cs_test_final` | 1,052 | 4,181 |
| Biology | Train | `biology_train_low` | 3,954 | 15,589 |
| Biology | Test | `biology_test_low` | 827 | 3,453 |
| Physics | Train | `physics_train_low` | 3,087 | 10,352 |
| Physics | Test | `physics_test_low` | 834 | 3,144 |
| Materials science | Train | `matsci_train_low` | 1,304 | 4,677 |
| Materials science | Test | `matsci_test_low` | 331 | 1,296 |

There are **13,790 training queries** and **3,044 test queries**. Corpus sizes count
records within each split; documents can recur across splits and fields.

**Physics training version:** the thesis PDF reports 3,120 queries and 10,471 documents.
The available export and the saved training bundle both contain 3,087 queries and
10,352 documents. This repository includes that available export unchanged, rather
than filling in missing records or claiming an exact thesis match. The other seven
splits match the thesis query and corpus counts. Matching counts do not establish
checkpoint or result reproducibility; those require the corresponding run artifacts.

## Files and schema

Each export directory contains:

| Packaged file | Contents |
|---|---|
| `raw/documents.json.gz` | Object mapping document identifiers to retrieval text, usually title and abstract; some entries are title-only. |
| `eval.json.gz` | One record per research query, including the union of gold documents and EAID metadata. |
| `sets.json.gz` | Query identifier → valid complete inspiration sets and search/status metadata. |
| `eval_primary.json.gz` | Primary decomposition only, flattened into one row per inspiration occurrence. This is a separate evaluation view. |
| `manifest.json` | Export counts, construction settings, label coverage and provenance. |

The `.gz` files contain ordinary UTF-8 JSON. For example:

```python
import gzip
import json

with gzip.open("sir-4/dataset/physics_test_low/eval.json.gz", "rt", encoding="utf-8") as f:
    queries = json.load(f)
print(queries[0]["question"])
print(queries[0]["supporting_documents"])
```

An `eval.json` record includes `id`, `question`, `answer`, `supporting_documents`,
`target_nodes`, `stratum`, and `quartet`. The retained `quartet` key contains EAID
metadata such as the fixed hypothesis, background, decomposition and per-document
labels. The retrieval input uses the question and candidate corpus; target hypotheses,
decompositions and gold labels are supervision/evaluation data. Staging removes the
EAID metadata from the retrieval query records.

For set-aware evaluation, use `eval.json` with `sets.json`. A query may have multiple
accepted inspiration sets; `supporting_documents` is their union. CompleteSet@k checks
whether a retrieved list contains a complete accepted set. The `sets` list in each
`sets.json` entry preserves those alternatives. Do not substitute the flattened
`eval_primary.json` rows for the set-aware queries.

`same` and `cross` describe field relationships. A query-level label does not replace
the per-document labels needed for per-gold analyses. Missing or ambiguous field labels
are retained as exported. Construction and resolution choices are documented in the
[benchmark guide](../README.md).

## Version and integrity

[MANIFEST.json](MANIFEST.json) records the dataset version, split counts, every packaged
and unpacked file's size and SHA-256, and the thesis PDF fingerprint used for the count
comparison. Query, corpus and label payloads are byte-identical to the selected source
exports. Export manifests retain their contents except that machine-specific absolute
source paths are reduced to relative construction paths; their original hashes are also
recorded. Gzip headers use a fixed timestamp for reproducible packaging.

```bash
python sir-4/prepare_data.py --verify-only
```

The benchmark package excludes paper full-text downloads, LLM extraction caches,
embeddings, generated graphs, checkpoints and predictions. Those are separate inputs or
outputs of the [retrieval experiments](../../experiments/README.md). The construction
scripts remain available for inspecting how SIR-4 and EAID were built.
