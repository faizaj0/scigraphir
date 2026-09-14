# CS in-domain fusion run: plan

**Goal.** Port the CARGO operator+graph fusion retriever onto SIR-4 and report it
on the CS test split.

**Setting.** Within-domain, CS only.

```
cs_train docs (20,203) -> cs_train graph -> train  -> parameters
cs_test  docs ( 4,181) -> cs_test  graph -> evaluate
```

Queries: 5,445 train (split 90/10 into train/validation), 1,052 test. Train and
test corpora share 524 documents (12.5% of test); query ids are fully disjoint.

**Working directory.** `$SCIGRAPHIR_ROOT/experiments/`

---

## Phase 0. Local prep (free, hours of work, no spend)

| # | step | status |
|---|---|---|
| 1 | **Stage SIR-4.** `prep/stage_sir4.py --domain cs`. Verifies ids, gold-in-corpus, empties. | done |
| 2 | **Generalise the scripts.** See below. | done |
| 3 | **Smoke test on 50 queries.** End to end. Nothing paid runs until it passes. | **PASS** |
| 4 | **Zip code + staged data to Drive.** `prep/bundle.py`. | done |

### What the smoke test caught

Six real bugs, every one of which would otherwise have surfaced during or after
the paid run. Five were silent.

| bug | how it would have failed |
|---|---|
| `sys` used before import in two scripts | loud, instant |
| `extract_frames` wrote to the unscoped cache | **silent**: 898 SIR-4 frames appended into the TOMATO caches |
| `build_greasoner_dataset` fell back to the bare TOMATO frames for any dataset | **silent**: built `sir4_cs_smoke_test_v16sc` from 3,182 TOMATO queries and reported success |
| concept-embedding cache `ds_{split}_emb_{type}.npy` unscoped | **silent**: loaded TOMATO embeddings, which is why the bogus graph had TOMATO shapes |
| `gen_probes` resolved its INPUT path before `set_dataset()` | **silent**: began generating probes for 3,132 TOMATO queries into the SIR-4 cache |
| `precompute_operator_components` assumed a prior run had embedded the same query subset | loud, but only after everything upstream was paid for |

The lesson generalises: the embedding and frame caches are keyed by SPLIT, never
by corpus, so a wrong-corpus load passes every check the code makes. There is
now an automated audit that every path-resolving call inside `main()` happens
after `set_dataset()`.

### Smoke-test result

| split | graph | queries | seeds | operator matrix | aligned |
|---|--:|--:|--:|---|---|
| test | 398 docs | 50 | **50/50** | dense (50, 398) | yes |
| train | 400 docs | 50 | **50/50** | dense (50, 400) | yes |

Seed coverage at the fixed 0.60 threshold is 100% on both splits, so the
threshold works for CS vocabulary and no query will be silently dropped.

Retrieval numbers from the smoke run are **not** meaningful: 400 documents
instead of 4,181, and n=8 on the cross slice.

Step 2 has three parts:

- **2a. Untie five scripts from TOMATO.** They hardcoded `tomato_{split}` and
  keyed caches by split alone, so a stale TOMATO `test_doc.npy` would load
  silently into a CS run and pass every shape check. Now resolved through
  `$SCIGRAPHIR_ROOT/scigraphir_paths.py`, which keeps TOMATO paths byte-identical by
  default and scopes every cache under the dataset name otherwise.
  Scripts: `extract_frames.py` (done), `build_greasoner_dataset.py`,
  `operator_scorer.py`, `precompute_operator_components.py`, `bge.py`.
- **2b. FAISS: MEASURED, NOT NEEDED. Dropped.** I claimed the all-pairs
  construction would not scale, citing a 43 GB score matrix. Both halves were
  wrong. The existing code already chunks the matmul, so memory was never
  quadratic, and the arithmetic is fast on BLAS. Measured at the true size of
  the CS-train `limitation` type:

  | n | d | backend | time | peak RSS |
  |--:|--:|---|--:|--:|
  | 104,000 | 1024 | exact | **2.5 min** | **1.3 GB** |
  | 40,000 | 1024 | exact | 22.6 s | |
  | 40,000 | 1024 | faiss HNSW | 24.7 s | |
  | 20,000 | 1024 | faiss HNSW | 11.8 s vs 5.3 s exact | |

  FAISS is *slower* below ~40k because building the index costs more than the
  scan. Six types at CS scale is ~10 minutes total. an ANN helper (since removed) was written
  and tested (pair recall 1.0000 at tau 0.80) and stays available as an escape
  hatch, but the working construction code is left alone. Revisit only if a
  type exceeds ~150k nodes.
- **2c. Multi-gold scorer.** SIR-4 averages 4.45 golds per query; every existing
  eval cell scores `gold[0]`.

## Phase 1. LLM extraction (paid, the dominant cost)

| # | step | volume |
|---|---|---|
| 5 | **Document frames.** task, domain, methods, functions, limitations, mechanisms, findings | 24,384 docs |
| 6 | **Query frames.** task, needs, current methods, limitations | 6,497 queries |
| 7 | **Probes**, for the operator's `S` and `M` terms | 6,497 queries |

Estimate the per-call cost on the 50-query smoke test before launching these.

## Phase 2. Graph construction (free)

| # | step |
|---|---|
| 8 | **Build the graph per split.** Typed nodes and edges from the frames, merge duplicate concepts at cosine 0.85, add mutual-8NN typed soft links. Via FAISS. |
| 9 | **Attach query seeds.** Up to three unweighted, same-type canonical nodes per query phrase, at a fixed threshold of 0.60. No entity, document or field seeds. |
| 10 | **Audit.** Structural checks, plus seed coverage, similarity spread, over-seeded nodes, and 30 to 50 frames read by hand. |

## Phase 3. Operator (free)

| # | step |
|---|---|
| 11 | BGE embeddings for documents, queries, probes |
| 12 | Fit `w0, w1, w2, β` on a train slice, select on a held-out train slice |
| 13 | Cache the ingredients as `operator_components.npz` so those four stay trainable inside the fusion |

## Phase 4. Baseline (free)

| # | step |
|---|---|
| 14 | **BGE-only predictions on cs_test.** Defines the *dissimilar* slice (gold ranked below 100 by plain BGE), so it is not optional. |

## Phase 5. Training (A100)

| # | step |
|---|---|
| 15 | Qwen3 node-embedding index. ~313k train and ~65k test node names. One-off. |
| 16 | Smoke run at 0 epochs: builds the index, catches shape errors. |
| 17 | **Train the additive-gate fusion model**, original operator-hard-negative objective. Not lever D, not loss v2. One run. |
| 18 | Report once on the 1,052 test queries. Metrics below. |

### Metrics (step 18)

| metric | definition |
|---|---|
| training mask | `supporting_documents` |
| MRR | rank of the highest-ranked labelled positive |
| Recall@k | proportion of *all* labelled positives retrieved in the top k |
| **CompleteSet@k** | whether at least one complete set from `sets.json` is contained in the top k |

`CompleteSet@k` is the SIR-4-specific one: it asks whether the retriever returned
a *whole* validated decomposition, not just some of its parts. Nothing in the
MOOSE lineage measures this, because no other benchmark carries the alternative
sets needed to define it.

---

## Decisions already made

| decision | why |
|---|---|
| **Additive-gate fusion, original hard-negative objective** | The validated arm, and the best head numbers measured on TOMATO (MRR .3299, R@1 .2273, R@5 .4368). Lever D traded head for tail (MRR .3141, dissim R@5 .0290 -> .0412) and loss v2 lost on both ends; neither is used here. Porting the known-good system keeps the SIR-4 numbers directly comparable to the TOMATO ones. |
| **One training run, not three** | The dense baseline is the operator, which is training-free and computed every batch anyway. Graph-only is a genuine second run; add it only if a reviewer asks "how much is just the graph". |
| **Keep the probes** | They are the operator's `S` and `M` terms, not a graph channel. Dropping them reduces the dense arm to plain BGE, which is 12.4 points of cross R@5 worse. Ablate them as their own row instead. |
| **Up to three unweighted seeds per phrase, fixed tau 0.60** | Same-type only. This is the seeding that produced the validated TOMATO numbers, so keeping it holds one more variable fixed. Dropping the entity channel still cuts seeds/query from ~66 to ~27. No threshold sweep. |
| **No entity, document or field seeds** | The entity channel depended on the old v6r graph and was the largest single channel, which made the graph's contribution unattributable. Every CS paper is CS, so a field seed carries no information. |
| **A100 40GB is enough, 80GB is comfortable** | cs_train is ~313k nodes / 727k edges. At 6 layers, 1024 dims, bf16, batch 4: ~2.6 GB per layer of node states, ~16 GB held, 20-25 GB peak with backward. Drop batch to 2 or enable `split_graph_training` if it OOMs. |

## Known properties of the data

- **Multi-gold.** 4.45 golds per query on average, range 1 to 10.
- `supporting_documents` **already equals** the union over all validated
  decompositions, verified 1052/1052. `sets.json` is still needed for set-aware
  evaluation, not for the positive mask.
- **The corpus is exactly the union of golds.** 4,181 documents, 4,181 distinct
  golds. There are no background distractors.

## Open

- Whether to also build the CS **v6r** graph, which would restore the entity
  seed channel. Currently: no.
