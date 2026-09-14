# ResearchBench transfer: does SIR-4 supervision transfer better than TOMATO-Star?

## The question

> Does training the same fusion retriever on SIR-4 produce better zero-shot transfer
> to ResearchBench than training it on TOMATO-Star?

ResearchBench is never used for training, tuning, threshold selection or early
stopping. Both checkpoints are frozen.

```
ResearchBench                 build one              derive each query's
1,367 queries / 20,322 docs -> master graph      ->  75-paper subgraph (cached)
                                                          |
                                        +-----------------+-----------------+
                                        v                                   v
                             frozen TOMATO fusion                frozen SIR-4 fusion
                                        |                                   |
                                        +----------> rank the same 75 <-----+
                                                          |
                                              set-aware metrics + paired bootstrap
```

---

## Design decisions, and the evidence for them

### Primary comparator is SIR-4 **Biology**, not CS

TOMATO-Star is overwhelmingly biomedical. Keyword-majority classification over its
7,000 training documents:

| label | docs | share |
|---|--:|--:|
| **biomed / neuro** | 6,263 | **89.5%** |
| unclassified | 386 | 5.5% |
| CS / ML | 148 | 2.1% |
| physics | 106 | 1.5% |
| chem / matsci | 91 | 1.3% |
| earth / env | 6 | 0.1% |

Pairing TOMATO against SIR-4 CS would confound *source discipline* with *supervision
quality*. SIR-4 Biology matches TOMATO's source distribution, so a difference is
attributable to the benchmark rather than the field.

The other three checkpoints are run on the identical cached graphs as a robustness
row, not as candidates for a best-of selection.

### The strict zero-shot subset is predeclared at 990 queries

Excluded because they sit inside one or both training distributions: Biology (114),
Cell Biology (150), Chemistry (113).

| discipline | queries |
|---|--:|
| Physics | 130 |
| Energy Science | 115 |
| Environmental Science | 114 |
| Material Science | 114 |
| Business | 113 |
| Earth Science | 113 |
| Math | 109 |
| Law | 96 |
| Astronomy | 86 |
| **total** | **990** |

Material Science is **retained**: TOMATO is 1.3% chem/matsci, which is not a material
amount, and SIR-4 Biology contains none. This was the audit the plan called for.

**Declare this subset before running anything.** Per-domain numbers are reported for
all 12 disciplines regardless of outcome.

### The protocol is corpus-calibrated, not fully candidate-local

The operator branch normalises across the document axis
(`zr` in [operator_scorer.py:182](../../retriever/eval/operator_scorer.py:182))
and its learned `w` and `beta` were fitted against full-corpus statistics. So:

- operator score **and** the gate are computed over all 20,322 documents, then the
  *result* is sliced to the candidates
- graph reasoning runs **only** on the candidate-induced subgraph
- `total_S` is sliced, never recomputed. It is a per-document vector
  (`total_S = S.sum(0)` over queries,
  [precompute_operator_components.py:100](../../retriever/precompute/precompute_operator_components.py:100)),
  so slicing preserves full-pool popularity for those documents

$$s_q^{75} = o_q[C_q] + \gamma_q\,\mathrm{ReLU}\!\left(z(g_q^{75})\right)$$

Described in the paper as: *a corpus-calibrated fusion retriever whose graph reasoning
and final ranking are restricted to the query's 75 candidates.* Not as a fully
candidate-local model, which it is not and cannot be.

A **candidate-renormalised** variant (all normalisation recomputed over the 75) is a
robustness row, never the primary protocol.

Also state: canonical nodes and soft edges are built once over the complete
**unlabelled** pool, so node identities carry corpus-level information. No gold-label
information enters construction.

### Metrics

All golds sit inside their candidate sets (3,211 / 3,211 = 100%), so this protocol
measures ranking only, never retrieval. Random-ranker references over 75 candidates:
recall@k = k/75, CompleteGoldSet@5/@10 = 0.3% / 1.3%.

Report MRR, hits@1, hits@5, recall@1/5/10/20, CompleteGoldSet@5/@10, split overall /
same-field / cross-field. Compute all of them; move any that turn out empirically
redundant to an appendix, rather than pruning on a priori reasoning.

`CompleteGoldSet@k` is `CompleteSet@k` with a gold family of size one:
$\mathcal M_{RB}(q)=\{S_1\}$ versus $\mathcal M_{SIR4}(q)=\{S_1,\dots,S_m\}$.
ResearchBench never validated its golds *as a set*; treating them as one accepted
decomposition is our interpretation and must be stated as such.

### Zero-seed queries are scored, not dropped

If no local seed clears 0.60 inside the candidate graph, the graph contribution is
zero and the model falls back to its operator branch. `zr` carries an epsilon in the
denominator, so an all-zero graph row gives `(0-0)/(0+1e-6) = 0` and `relu(0) = 0`:
no NaN. Verify the graph-branch normalisation in `fusion_reasoner.py` has the same
guard, since it is a separate code path.

Forcing a seed to satisfy an assertion would inject an unrelated concept, which is
worse than the problem it solves.

### The operator row is a diagnostic, and it is not optional in practice

On matsci the converged fusion was a **no-op** over the operator: R@10 +0.0004,
CS@10 exactly 0.0000, nothing significant on any slice. If that holds generally, then
TOMATO-fusion versus SIR-4-fusion is a comparison of two *operators* with differently
fitted `w`/`beta` and a dormant graph channel, and the claim collapses from "SIR-4
trains better retrievers" to "SIR-4 fits better operator scalars".

Include it, labelled a diagnostic ablation rather than a main method. It costs one
scoring pass over predictions already being produced.

---

## File map

```
experiments/transfer/
├── PLAN.md                     this file
├── prep_researchbench.sh       phase 1 commands, in order
├── build_candidate_views.py    master graph -> 1,367 cached subgraph views
├── transfer_infer.py           frozen checkpoint -> ranking of exactly the candidates
├── paired_bootstrap.py         paired CI over the 1,367 queries
└── colab_rb_transfer.ipynb     generated; the GPU half
```

Touched elsewhere:

| file | change |
|---|---|
| `sciafford/build_greasoner_dataset.py` | emit `node_provenance.json` |
| `retriever/gfm-rag/gfmrag/models/fusion_reasoner.py` | accept per-batch components |
| `experiments/eval/score_sir4.py` | `--per-query-out` |
| `experiments/prep/bundle.py` | ship `transfer/` |

---

## Phase 1 — prepare ResearchBench once (local + paid)

Shared by every checkpoint. Runs once.

**Step 0. Fix the corpus path.** The pool builder writes `data/researchbench/raw`,
but `scigraphir_paths.corpus_dir(split)` resolves `data/{DATASET}_{split}`. Move it:

```bash
mv $SCIGRAPHIR_ROOT/retriever/data/researchbench $SCIGRAPHIR_ROOT/retriever/data/researchbench_test
```

**Step 1. Frames.** 20,322 documents + 1,367 queries, about **$7**.

**Step 2. Probes.** 1,367 queries, small.

**Step 3. Master graph** at `--tau_canon 0.95`, plus `node_provenance.json`.

**Step 4. Operator components** over the master document order.

**Step 5. Audit** the master graph before spending anything else.

Commands live in `prep_researchbench.sh`. Every one prints the
`[scigraphir_paths] dataset=...` banner: if it says anything but `researchbench`, stop.

## Phase 2 — candidate views (local, free)

`build_candidate_views.py` reads the master graph plus `node_provenance.json` and
`candidates.json`, and writes 1,367 cached views:

- $V_q = C_q \cup \{\text{concepts with provenance in } C_q\}$
- keep only edges with both endpoints in $V_q$
- `torch_geometric.utils.subgraph` with relabelling, master features and relation
  embeddings preserved
- re-snap query seeds inside $V_q$: same type, top 3, threshold 0.60
- record the local seed count, including zeros

Cached once and byte-identical for every checkpoint. That is what makes the
comparison paired.

## Phase 3 — frozen inference (Colab, GPU)

`transfer_infer.py`, one checkpoint load, loop over cached views, one query per
forward pass because the graph differs per query.

| checkpoint | role |
|---|---|
| SIR-4 Biology | primary |
| TOMATO-Star | primary comparator |
| SIR-4 CS / Physics / Matsci | robustness |
| operator alone | diagnostic |
| BGE | training-free reference |

Emits a predictions JSON per arm containing all 74-75 ranked candidates.

## Phase 4 — scoring and comparison (local, free)

```bash
python3 eval/score_sir4.py --pred <arm> --queries ... --sets <rb sets.json> \
    --per-query-out <arm>_perquery.json --json-out <arm>_scores.json
python3 transfer/paired_bootstrap.py --a sir4_biology --b tomato --metric mrr
```

Paired bootstrap over 1,367 queries: mean difference SIR-4 minus TOMATO, 95% CI,
whether it crosses zero. Paired is available and far stronger than unpaired here,
because both arms see identical graphs.

---

## Validity checks, asserted before any result is reported

- all 1,367 queries scored; zero-seed queries included, never dropped
- every ranking contains exactly its 74-75 candidates, no non-candidate appears
- every gold present in its candidate list
- candidate-graph document count equals candidate count
- both checkpoints consumed identical cached views
- no NaNs from zero graph scores
- no ResearchBench gold label influenced construction or parameter selection
- checkpoint loading reports no unexplained missing parameters
- contaminated and clean-query results reported separately
  (L1 = 13 pool docs / 7 gold docs; L2 = 93 / 51; L3 = 381 / 142)
- local seed distribution reported against the master graph's

## Known limitation, stated up front

The Biology checkpoint is a **single-field** SIR-4 model. It cannot support a claim
that one model learned jointly from all four SIR-4 fields. If a pooled four-field
SIR-4 model is trained later it becomes the headline system, and its strict unseen
subset shrinks to Astronomy, Business, Earth Science, Energy, Environmental Science,
Law and Math.

## Open items

- [ ] `fusion_reasoner.py` needs an inference-only path accepting per-batch
      components instead of one fixed-width matrix
- [ ] confirm the graph-branch normalisation carries the epsilon guard
- [ ] the same master graph later supports the **full-corpus** protocol over all
      20,322 papers, which is the contribution no benchmark in this lineage measures
