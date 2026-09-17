# Runbook: MIR (Methodology Inspiration Retrieval) through the pipeline

The current default SciAfford graph combines affordance structure, OpenIE entity context
and paper-to-affordance links, using the `_hyb` suffix. This runbook also records earlier
`_v16sc` and OpenIE-only experiments. Their graph paths identify those recorded runs;
use the [graph guide](../sciafford/README.md) for the default construction.

Dataset name is `mir`. Every command takes `--dataset mir`; check the
`[scigraphir_paths] dataset=mir` banner on each. The staged layout (done):

```
retriever/data/mir_train/raw/{documents.json,train.json}   4,678 docs, 1,270 queries
retriever/data/mir_test/raw/{documents.json,test.json}     4,857 docs,   155 queries
```

Test queries are never in training. The test corpus contains the whole training corpus
plus dev/test citations, so 4,678 documents are shared between the two splits.

## 0. Stage  (done; re-run is free and idempotent)

```bash
cd $SCIGRAPHIR_ROOT && python3 experiments/prep/stage_mir.py
```

## 1. Affordance representations  (PAID; about $2-3 total at CS's rate of ~$0.4 per 1,000 documents)

Train side first:

```bash
cd $SCIGRAPHIR_ROOT/sciafford && python3 extract_affordances.py --dataset mir --side doc --split train --workers 128
```

```bash
cd $SCIGRAPHIR_ROOT/sciafford && python3 extract_affordances.py --dataset mir --side query --split train --workers 128
```

**Seed the test document cache from the train one before extracting test documents.**
The extractor skips any id already in its cache file, and 4,678 of the 4,857 test
documents are the same papers as the training corpus, so this halves the paid work:

```bash
cd $SCIGRAPHIR_ROOT/sciafford && cp cache/mir/frames_doc_train.jsonl cache/mir/frames_doc_test.jsonl
```

Then the test side picks up only the ~180 documents not in train:

```bash
cd $SCIGRAPHIR_ROOT/sciafford && python3 extract_affordances.py --dataset mir --side doc --split test --workers 128
```

```bash
cd $SCIGRAPHIR_ROOT/sciafford && python3 extract_affordances.py --dataset mir --side query --split test --workers 128
```

Check counts: doc affordance representations should be 4,678 (train) and 4,857 (test); problem requirement representations 1,270 and 155.

```bash
wc -l $SCIGRAPHIR_ROOT/sciafford/cache/mir/*.jsonl
```

## 2. hypothetical answers  (PAID, small)

```bash
cd $SCIGRAPHIR_ROOT/retriever && python3 hypothetical_answers/generate_answers.py --dataset mir --split train --workers 128
```

```bash
cd $SCIGRAPHIR_ROOT/retriever && python3 hypothetical_answers/generate_answers.py --dataset mir --split test --workers 128
```

## 3. Graphs  (free, minutes at this size)

Same settings as SIR-4: canonicalisation threshold 0.95, no entity or hypothetical answer seeds.

```bash
cd $SCIGRAPHIR_ROOT/sciafford && for s in test train; do python3 build_greasoner_dataset.py --dataset mir --split $s --tau_canon 0.95 --no_entity_seeds --no_probe_seeds; done
```

## 4. Audit  (free)

```bash
cd $SCIGRAPHIR_ROOT/experiments && for s in test train; do python3 eval/audit_graph.py --dataset mir --split $s --spread --sample 40; done
```

Must exit clean: no dangling edges, zero seedless queries.

## 5. Bundle  (free)

```bash
cd $SCIGRAPHIR_ROOT/experiments && python3 prep/bundle.py --dataset mir
```

The bundler prints "no sets.json will be bundled" for `mir`: correct, MIR has no
decomposition sets, so CompleteSet@k is not defined here.

Upload `experiments/mir_bundle.zip` to `MyDrive/cargo-gfmrag/`.

## 6. Colab

The MIR notebook (built by `prep/build_mir_notebook.py`) runs the six baselines, the
multi-view scorer, and the cumulative ablation (scorer, + graph reasoner, + CCMP), then
prints the main retrieval table with R@3, R@5, nDCG@5 and mAP. Training is minutes to an
hour per arm at this corpus size.

## Caption facts

- Golds are citation-derived (MultiCite Uses / Extention), the same construct as MOOSE-Chem.
- Corpus is 4,857 abstracts, between the 75-candidate re-ranking benchmarks and the
  20k-document full-corpus setting; no titles in the release.
- Single field (computational linguistics): no same/cross split.
- MIR's published numbers are on a 284-document restricted corpus; report them as such.

## 7. MOOSE-Star on MIR  (PAID, about $2; ~20 min)

The LLM-baseline loader accepts `--field mir` (abstract-only corpus: the stand-in title is
the first sentence, the abstract is the whole text; all 155 test proposals, no subset).
Run in `experiments/llm_baselines/` with `OPENAI_API_KEY` exported.

```bash
cd $SCIGRAPHIR_ROOT/experiments/llm_baselines && python3 build_moose_star_tree_sir4.py --field mir
```

```bash
cd $SCIGRAPHIR_ROOT/experiments/llm_baselines && python3 run_moose_star_sir4.py --field mir --subset none --limit 3
```

```bash
cd $SCIGRAPHIR_ROOT/experiments/llm_baselines && python3 run_moose_star_sir4.py --field mir --subset none --num-workers 16
```

```bash
cd $SCIGRAPHIR_ROOT/experiments/llm_baselines && python3 score_rankings_sir4.py --field mir --subset none --rankings outputs/moose_star/mir/rankings.json --method "MOOSE-Star (gpt-4o-mini)"
```

Read the `all` slice (every MIR query is stratum "same"). mAP is not in that scorer's
columns; R@3, R@5 and nDCG@5 fill the MOOSE-Star row of `results/mir/table_mir.tex`.

## 8. HyDE + MuGI on MIR  (PAID, about $0.20; ~15 min on a T4)

Same recipe as the TOMATO rows (gpt-4o-mini, N=4 generations, Qwen3-Embedding-0.6B
encoder). The notebook reads `mir_bundle.zip` as it is now; no re-bundle needed.

```bash
cd $SCIGRAPHIR_ROOT/experiments && python3 prep/build_llm_expansion_notebook.py --datasets mir
```

Upload and run `experiments/notebooks/llm_expansion_baselines_mir.ipynb`. Scores land in
`outputs/baselines/mir/scores_HyDE_gpt-4o-mini.json` and `..._MuGI_...` on Drive; read the
`all` slice. Carry the caption caveat (the expansion LLM may know the gold papers) into the
MIR table.

## 9. OpenIE graph for MIR  (PAID, about $2-4 of gpt-4o-mini; ~40 min at WORKERS=64)

Needed by BOTH the GFM-RAG baseline (it ranks entity nodes) and the
"+ Graph Reasoner (OpenIE graph)" SciGraphIR row. Same indexer and settings as the SIR-4
OpenIE graphs of 18 Aug (gpt-4o-mini NER + triples, BGE-large linking at 0.9). The staged
`mir_{train,test}/raw/` is already the layout it expects.

Where the time went (matsci train, 4,677 docs, 18 Aug, 10 threads): OpenIE 30 min (API
latency bound), BGE entity embedding + similar-entity search 14 min (local), query NER +
linking ~10 min. Two fixes since (8 Sep): `WORKERS=64` shrinks the two API stages to a few
minutes each on tier 5, and the entity linker now encodes on MPS instead of CPU (7.6x faster
on this Mac) and no longer encodes the phrase list twice, so the local 14 min becomes ~2-3.
Expect ~10 min for train and ~5 min for the seeded test split. Use the same WORKERS for both
splits: the value is part of the cache fingerprint. `EL_DEVICE=cpu` reproduces the old path.

Faster still, if the ~$2 does not matter: skip the seeding step and run both splits at once
in two terminals (each is then ~10 min, total wall ~10 min, cost ~$4.50 instead of ~$2.40).

```bash
cd $SCIGRAPHIR_ROOT/retriever && WORKERS=64 bash run_index.sh mir_train
```

The vendored OpenIE model has no retry, so a transient API error drops that document from
the cache. Check the log's `Total number of processed data` equals 4,678; if it is short,
run the same command again, which extracts only the missing passages and rebuilds the graph:

```bash
cd $SCIGRAPHIR_ROOT/retriever && grep -E "Number of passages|Total number of processed" logs/index_mir_train.log
```

**Seed the test OpenIE cache from the train one before indexing the test split.** The cache
is keyed by passage text under `tmp/kg_construction/<fingerprint>/<dataset>/`, and 4,678 of
the 4,857 test documents are the same papers with the same ids, so this leaves ~180
documents to extract. The fingerprint depends on WORKERS, so take the directory from the
train run rather than typing it:

```bash
cd $SCIGRAPHIR_ROOT/retriever && FP=$(ls -td tmp/kg_construction/*/mir_train | head -1 | xargs dirname) && mkdir -p "$FP/mir_test" && cp "$FP/mir_train/openie_results.jsonl" "$FP/mir_test/" && echo "seeded $FP/mir_test"
```

```bash
cd $SCIGRAPHIR_ROOT/retriever && WORKERS=64 bash run_index.sh mir_test
```

The test log should say `Number of passages which require processing: ~179`, not 4,857.

Check: `data/mir_{train,test}/processed/stage1/nodes.csv` exist with both `entity` and
`document` types, and `logs/index_mir_{train,test}.log` end cleanly.

```bash
cd $SCIGRAPHIR_ROOT/retriever && for s in train test; do python3 -c "import csv,collections,sys; csv.field_size_limit(10**7); print('$s', dict(collections.Counter(r['type'] for r in csv.DictReader(open('data/mir_$s/processed/stage1/nodes.csv')))))"; done
```

Then re-bundle (the bundler ships the whole corpus directory, so the new `processed/stage1`
rides along with no flag) and upload `mir_bundle.zip` again:

```bash
cd $SCIGRAPHIR_ROOT/experiments && python3 prep/bundle.py --dataset mir
```

## 9b. Everything in ONE notebook  (GPU, ~3.5 h; replaces sections 8, 10 and 11)

One setup, one table. Composed from the three builders below, so the arms are identical;
the graph baselines train before the fusion sources are written, so they stay stock.
Needs the re-bundled zip from section 9 on Drive.

```bash
cd $SCIGRAPHIR_ROOT/experiments && python3 prep/build_mir_all_notebook.py
```

Upload and run `experiments/notebooks/colab_mir_all.ipynb` top to bottom. Every block skips finished
work from Drive, so rerun from the top after a disconnect. The final cell writes
`outputs/mir/table_mir_all.md` with every MIR row. Sections 8, 10 and 11 remain the
per-block notebooks if you would rather run one piece at a time.

## 9c. Graph-channel scan (every gold, every arm), any dataset, one cell

`colab_graph_scan_cell.py` is a single paste-in cell. Set `DATASET` at the top
(`tomato | mir | sir4_cs | sir4_biology | sir4_physics | sir4_matsci`), paste it into a fresh
runtime and run. It replays the setup cells of a notebook already in Drive's "Colab Notebooks"
(colab_mir_all for MIR, colab_routing_tomato for TOMATO, colab_qualitative_<field> for SIR-4;
the notebook must carry `interpret()`, i.e. built on/after 6 Sep), then ranks every gold of
every test query under each arm with no path search and prints the table per stratum
(same / cross / all). Output: `outputs/scan/<dataset>/graph_channel_scan_<dataset>.{md,json}`.

Arms (skipped when the checkpoint is missing): SciAfford graph + CCMP, the same weights with the
CCMP gate off, SciAfford graph no-CCMP control, OpenIE graph. TOMATO has no current-engine OpenIE
checkpoint, so it gets three arms. Do not run this cell in a kernel where another replay cell
(e.g. `colab_rb_mir_arm_cell.py`) has run: each replay rewrites `/content/gfm-rag`'s fusion
sources with its own notebook's blob, and the RB blob lacks `interpret()`.

`colab_mir_scan_cell.py` is the MIR-only predecessor (same output layout).

## 9c-bis. TOMATO OpenIE arm (needed for the affordance representation-vs-OpenIE comparison on TOMATO)

The only OpenIE-graph weights on Drive for TOMATO are the July v1 run (BGE handcrafted scorer, old
engine, no scorer), so the scan/showcase cells skip the OpenIE arm on TOMATO. To add it:
`python3 prep/build_sir4_openie_notebook.py --dataset tomato` -> `colab_tomato_openie_ablation.ipynb`
(same recipe as the SIR-4 OpenIE row: multi-view scorer warm start from
`outputs/tomato_ablations_v1/tomato/semantic`, graph reasoner on `tomato_{train,test}` OpenIE
graphs, no CCMP, 10 epochs, batch 2; `tomato_bundle.zip` already ships both OpenIE graphs with
their stage1 QA json). Writes `outputs/tomato_openie/tomato_openie_qwenmlp_graph_e10_b2/`, which
the scan and showcase cells then pick up as the `openie` arm. Hours on an A100.

## 9c-speed. Why a scan or showcase can take hours, and the fix

`interpret_paths` with `+interp.paths=0` is a fast rank scan ONLY if the engine's `interpret()` has the
`do_paths` argument. A notebook whose fusion blob predates it ignores the flag and runs the gradient
beam search on every gold (biology: 767 s per arm instead of ~60 s, and the same again in the path
stage). The paste-in cells therefore apply `MyDrive/cargo-gfmrag/gfm_overlay.zip` (the current fusion
sources, built from the fork: `experiments/gfm_overlay.zip`) after the replayed blob and refuse to
run without a current `interpret()`. Upload the zip once; rebuild it when the fork changes.

The path stage now searches only (a) a random sample of 80 cross + 80 same queries, used for the hop
figure (`qids_sample.json`, unbiased), and (b) the candidate golds the scan flagged (cosine >= 25 and
graph <= 5 or fused <= 25), pinned via `golds_paths.json` so only those golds are searched, with
num_beam 6 and path_topk 3. Budget: minutes per arm instead of hours.

**Update (8 Sep, later).** The notebooks no longer need the Drive upload: every `colab_showcase_*.ipynb`
carries `gfm_overlay.zip` inside cell 3b-bis (base64) and extracts it over the engine after the replayed
fusion-sources cell, then asserts `do_paths`, `golds=None` and `min_hops` in `fusion_trainer.py`. The
paste-in cells still read `MyDrive/cargo-gfmrag/gfm_overlay.zip` and now assert `min_hops` too.

Symptoms of a stale engine, seen in the 8 Sep 14:13 CS run: the path stage prints `num_beam: 10`,
`path_topk: 5`, `max_golds: 8` and "N cross + 120 same" (old stage), then `eval/showcase.py` dies in
`draw_hops` with `unsupported format string passed to NoneType.__format__` because no target carries
`min_hops` (no structural floor, so `mean_gap` is None). `showcase.py` now prints `n/a` and a warning
instead of crashing, but the fix is the overlay: delete `outputs/scan/<dataset>/hops_*.json` written by
the old engine and rerun the path stage. Rebuild after any engine change:
`python3 prep/build_showcase_cell.py && python3 prep/build_showcase_notebook.py && python3 prep/build_showcase_all_notebook.py`
(prep/overlay_cell.py reads experiments/gfm_overlay.zip; regenerate that zip from retriever/gfm-rag first).

## 9c-ter. Merged graph on TOMATO (affordance representation + entity seeds + mention edges + mechanism shortcuts)

Built locally by `python3 prep/build_hybrid_graph.py --dataset tomato` (cap 30, seed entities only):
`retriever/data/tomato_{train,test}_hyb/`, zipped as `tomato_hyb_bundle.zip` (31 MB). Walk
prior on the test graph 21.7 / cross 17.0 vs 14.4 / 10.4 on the SciAfford graph. Upload the zip to
`MyDrive/cargo-gfmrag/` next to `tomato_bundle.zip`, then run `colab_routing_tomato_hyb.ipynb`
(built by `build_routing_notebook.py --dataset tomato --graph-suffix hyb --extra-bundle tomato_hyb_bundle.zip
--arms control,ccmp`): two arms, no CCMP and CCMP, 10 epochs, batch 2, ~15-18 h each on an A100
(local training, 10-min Drive sync, resumes). Writes `outputs/routing_hyb/tomato/tomato_hyb_route_{control,ccmp}_e10_b2_s1024/`.
The scan / showcase cells and notebooks then add the rows `merged graph, no CCMP`, `merged graph + CCMP`
and `merged graph, gate off`, with gold-by-gold win rates against the affordance representation and OpenIE arms.

## 9d. Showcase: "amazing" cross-domain examples + hop figure

One notebook for everything: `colab_showcase_all.ipynb` (built by `prep/build_showcase_all_notebook.py`).
Edit `DATASETS` in cell 1 (default: the four SIR-4 fields, TOMATO, MIR), run top to bottom on an
A100. Setup happens once (all bundles unpacked side by side, Qwen3, engine, fusion sources with
`interpret()`); cell 5 loops over the datasets and runs, per dataset:

- the graph-channel scan of every gold under every arm (affordance representation + CCMP, gate off, no-CCMP control,
  OpenIE; missing checkpoints are skipped) -> `outputs/scan/<dataset>/graph_channel_scan_<dataset>.md`;
- path interpretations (gradient beam search, NBFNet / GFM-RAG recipe) for every gold of every
  cross-field query plus 120 same-field queries -> `hops_<arm>.json`;
- `eval/showcase.py` -> `showcase_<dataset>.md` (candidates ranked for a reader: cosine rank >= 25 and
  graph <= 5 or fused <= 25; tier A = full model top-10, B = graph top-5 only, C = rest; route
  "bridge" = passes a function / limitation / method / finding affordance representation, "hub" = only papers and a
  domain node), `showcase_<dataset>.tex` (GFM-RAG Table 4 layout), `showcase_<dataset>_hops.{pdf,png,json}`
  (Fig. 6 analogue: top-path length per arm against the shortest seed->gold route, a structural
  FLOOR, not a ground-truth reasoning path; say so in the caption).

Cell 6 draws `outputs/scan/fig_hops_all.pdf`, one panel per dataset. Everything is cached on
Drive, so re-running after a new checkpoint (e.g. the TOMATO OpenIE arm, 9c-bis) only adds rows.
Per-dataset variants: `colab_showcase_<dataset>.ipynb` (build_showcase_notebook.py) and the
paste-in `colab_showcase_cell.py`. Locally, `--sir4 ../sir-4/data/benchmark/<field>_test_final/eval.json`
adds the SIR-4 field pair per gold and the gold-level stratum.

## 10. G-Reasoner + GFM-RAG on MIR  (GPU, ~1 h each)

One notebook, the SIR-4 all-domains plumbing over `mir`. G-Reasoner trains on the
`mir_*_v16sc` affordance representation graphs and runs today; GFM-RAG is gated on the OpenIE graph from
section 9 and opens by itself once the re-bundled zip is on Drive. Sections 1-4 re-run the
six dense arms into `outputs/baselines/mir/` (about 15 min; the MIR fusion notebook keeps
its own copies elsewhere, both are fine).

```bash
cd $SCIGRAPHIR_ROOT/experiments && python3 prep/build_baselines_notebook.py --dataset mir
```

Upload and run `experiments/notebooks/mir_baselines.ipynb`. Rows come out of section 7 (read the
`all` slice; `map` is in the columns). The cross-domain summary at the end prints dashes:
it is the SIR-4 layout, not a failure.

## 12. Zero-shot transfer to MIR  (GPU, ~20 min, free)

Zero-shot transfer on MIR: the TOMATO-Star-trained and the SIR-4 four-field SciGraphIR checkpoints,
frozen, rank MIR's test corpus. No training. Needs only what is already on Drive: the two
checkpoints with their scorer warm starts (`outputs/tomato_ablations_v1/tomato/...`,
`outputs/rb_zeroshot/...`), the current `mir_bundle.zip`, and the in-benchmark run's caches
under `outputs/mir/cache`. Uses the SciAfford graph `mir_test_v16sc`, never the OpenIE graph.

```bash
cd $SCIGRAPHIR_ROOT/experiments && python3 prep/build_mir_zeroshot_notebook.py
```

Upload and run `experiments/notebooks/colab_mir_zeroshot.ipynb`. Outputs land in
`outputs/mir_zeroshot/`: `scores/scigraphir_{tomato,sir4}_scores.json`, and
`table_mir_zeroshot.{md,tex}` with the six baselines and the MIR-trained SciGraphIR row read
from `outputs/mir/scores` for contrast. Read the `all` slice (single field). Caption facts:
neither source ever saw MIR; the SIR-4 source's scorer was warm-started on CS alone and the
reasoner trained jointly on four fields, as in the zero-shot transfer experiment.

## 11. SciGraphIR on the OpenIE graph  (GPU, ~1 h)

The "+ Graph Reasoner (OpenIE graph)" row: multi-view scorer + graph reasoner on the
`mir_*` OpenIE graph, no CCMP, exactly the SIR-4 recipe. Needs section 9's re-bundled zip.
The scorer warm start is restored from `outputs/mir/semantic` on Drive (it never reads a
graph), so only the component tables and the one fusion run are new work.

```bash
cd $SCIGRAPHIR_ROOT/experiments && python3 prep/build_mir_notebook.py --graph openie
```

Upload and run `experiments/notebooks/colab_mir_openie.ipynb`. The run is
`outputs/mir/mir_openie_qwenmlp_graph_e10_b2`; the score file is
`outputs/mir/scores/scigraphir_openie_graph_scores.json`. Any test query the OpenIE graph
left seedless is re-inserted as an empty ranking before scoring, so the row is over all 155
queries. Its nDCG@5 fills the dashed OpenIE row of the MIR table in
`results/main_table/main_results_three_tables.tex`.
