# Runbook: SIR-4 preparation and training

Every command takes `--dataset sir4_<domain>` or `--domain <domain>`. Domains:
`cs`, `biology`, `physics`, `matsci`. Substitute throughout; the examples use
`physics`.

Two domains can run at once. Each has its own caches, bundle, Drive output
directory and notebook, and nothing is shared except two read-only files on
Drive (`gfm-rag-adapted.zip`, the cached Qwen3 model).

**Check the `[scigraphir_paths] dataset=...` banner on every command.** It prints on
start. If it says `tomato`, stop.

---

## Default graph workflow

The full method uses the **default SciAfford graph** assembled by
[build_hybrid_graph.py](prep/build_hybrid_graph.py): affordance structure, OpenIE query
entity seeds, entity-to-paper mention edges and direct paper-to-affordance links.
Its graph names end in `_hyb`.

The component preparation recipe below produces the `_v16sc` inputs and per-field
bundles. Follow the [graph construction guide](../sciafford/README.md#build-the-default-graph)
to also construct the OpenIE inputs and assemble the default graph. Once both components
exist, the final construction and checks are:

```bash
cd "$SCIGRAPHIR_ROOT"
python experiments/prep/build_hybrid_graph.py --dataset sir4_physics --cap 30
for split in train test; do
  python experiments/eval/audit_graph.py --dataset sir4_physics \
    --split "$split" --suffix hyb --sample 40
done
```

Use [colab_sir4_hyb.ipynb](notebooks/colab_sir4_hyb.ipynb) for the full-method SIR-4 runs.
It expects the per-field bundles, the learned scorer warm starts, the engine archive,
and `sir4_hyb_bundle.zip` on Drive. That additional zip must retain repository-relative
paths for each `_hyb` graph's `raw/` and `processed/stage1/` files. For all four fields,
create it after building all eight merged graphs:

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path.cwd()  # repository root
paths = []
for field in ("cs", "biology", "physics", "matsci"):
    for split in ("train", "test"):
        graph = root / "retriever/data" / f"sir4_{field}_{split}_hyb"
        paths.append(graph / "raw/documents.json")
        paths.extend(graph / "processed/stage1" / name
                     for name in ("nodes.csv", "edges.csv", "relations.csv", f"{split}.json"))
missing = [str(path) for path in paths if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing graph inputs: " + ", ".join(missing))
with ZipFile(root / "experiments/sir4_hyb_bundle.zip", "w", ZIP_DEFLATED) as bundle:
    for path in paths:
        bundle.write(path, path.relative_to(root))
```

The remaining sections retain the earlier component-graph workflow and its run records.
Its `_v16sc` paths describe that component; the default graph and full-method notebook
are the `_hyb` workflow above.

## A. Local component preparation

### 1. Stage the export  (seconds, free)

```bash
cd "$SCIGRAPHIR_ROOT"
python3 sir-4/prepare_data.py --domain physics
python3 experiments/prep/stage_sir4.py --domain physics
```

The first command unpacks the [included dataset](../sir-4/dataset/README.md) after
checking its hashes. The available Physics training split contains 3,087 queries;
its difference from the thesis count is recorded in the dataset guide.

Staging verifies duplicate ids, gold-in-corpus, empty questions and documents, then
writes the corpus to both `experiments/data/<domain>/` and
`retriever/data/sir4_<domain>_{train,test}/`, which is where the pipeline
reads. It refuses to stage if any check fails.

### 2. Extract affordance representations  (PAID, the dominant cost)

Four runs: two sides x two splits.

```bash
cd $SCIGRAPHIR_ROOT/sciafford && for s in test train; do for side in doc query; do python3 extract_affordances.py --dataset sir4_physics --side $side --split $s --workers 128; done; done
```

Resumable. Re-running picks up only what is missing, because a failed item is
deliberately NOT cached. Cost scales with document count: CS was 24,384
documents for about $9.55.

`--workers 128` suits an OpenAI tier-5 account. These calls are I/O-bound, so
concurrency costs threads rather than memory and will not repeat the swap
problem that local BGE encoding caused. The script already holds one shared
client with `max_retries=0` and its own exponential backoff with jitter, so a
rate-limit burst backs off rather than failing the item. If you see sustained
429s, drop to 64; going much above 128 buys little, because the tail is
dominated by slow individual calls rather than queueing.

### 3. hypothetical answers  (PAID, small)

```bash
cd $SCIGRAPHIR_ROOT/retriever && for s in test train; do python3 hypothetical_answers/generate_answers.py --dataset sir4_physics --split $s --workers 128; done
```

About 7 to 8 short search phrases per query. These are the handcrafted scorer's `S` and
`M` terms, not a graph channel.

**Check before moving on:** affordance representation and hypothetical answer line counts against corpus size.

```bash
wc -l $SCIGRAPHIR_ROOT/sciafford/cache/sir4_physics/*.jsonl $SCIGRAPHIR_ROOT/retriever/probes/cache/sir4_physics/*.jsonl
```

A shortfall of a handful of documents is normal (title-only records, or items
that exhausted their retries); re-run step 2 to pick the latter up.

### 4. Build the train and test affordance components  (free)

```bash
cd $SCIGRAPHIR_ROOT/sciafford && for s in test train; do python3 build_greasoner_dataset.py --dataset sir4_physics --split $s --tau_canon 0.95 --no_entity_seeds --no_answer_seeds; done
```

`--tau_canon 0.95` is not the default and matters. At the default 0.85, 66,043
of 88,095 CS limitation nodes merged into a single node. 0.95 is the only
threshold measured where the worst merge in the graph is still the same concept.

The two `--no_*_seeds` flags drop the entity and document seed channels, which
depended on the old v6r graph.

To choose the threshold for a new domain, sweep it offline first from the cached
concept embeddings rather than rebuilding: the sweep predicted the built graph
exactly on CS.

### 5. Validate both affordance components  (free)

```bash
cd $SCIGRAPHIR_ROOT/experiments && for s in test train; do python3 eval/audit_graph.py --dataset sir4_physics --split $s --spread --sample 40; done
```

Exits non-zero on the two failures that make a run wrong rather than merely
worse: dangling edges, and any zero-seed query. A zero-seed query is dropped by
the loader with one log line, so the final metric would be computed over fewer
queries than reported.

Also watch `nodes seeding >10% of queries` (should be 0) and gold reachability
(a one-hop lower bound, not a ceiling). It writes 40 cases to
`data/audit_sir4_<domain>_<split>_sample.txt` for hand reading, which nothing
else substitutes for.

### 6. Bundle and generate the notebook  (free)

```bash
cd $SCIGRAPHIR_ROOT/experiments && python3 prep/bundle.py --dataset sir4_physics && python3 prep/build_notebook.py --domain physics --batch 1
```

The zip mirrors the repo, so Colab unzips it to `/content/scigraphir`, sets
`SCIGRAPHIR_ROOT`, and every script resolves exactly as it does locally. The bundler
names anything missing rather than shipping a thin zip that fails an upload and
a runtime later.

`--batch 1` fits a 40 GB A100. Use `--batch 2` only on 80 GB.

---

## B. Earlier component-graph Colab workflow

Upload `experiments/sir4_<domain>_bundle.zip` to `MyDrive/cargo-gfmrag/`,
open `colab_train_sir4_<domain>_fusion.ipynb`, and run the cells in order.

| cell | what | time |
|---|---|---|
| 1, 2 | paths, isolation guard | seconds |
| 3a, 3b, 3c | engine install, config rewrite, PyG version fix | 2 min |
| 4 | Qwen3 model (cached on Drive after the first domain) | 1 min |
| 5 | unpack bundle, verify every path | 1 min |
| **5b** | **Phase 3**: fit the handcrafted scorer, cache its components | ~6 min |
| **5c** | **Phase 4**: BGE baseline, scored immediately | ~2 min |
| 6 | structural audit | seconds |
| 7 | `run_model` | - |
| 8 | smoke at 0 epochs, builds the Qwen3 node index | 6 min |
| 9 | train | hours |
| 10, 11 | multi-gold scoring, per-epoch trajectory | 2 min |

**Save the expensive artefacts to Drive after 5c.** Colab runtimes reset and
`/content` does not survive it. The index and the embeddings are the costly
parts.

**Read the epoch-1 ETA before committing to 20 epochs.** The TOMATO fusion
converged at epoch 5, so 8 may be enough. `save_best_only: true` writes to
Drive, so a disconnect still leaves the best checkpoint.

---

## What to record as deviations

- **Batch size.** TOMATO trained at 4; the SIR-4 graphs do not fit at 4 on a
  40 GB card.
- **`tau_canon 0.95`,** chosen from measured merge coherence, not the 0.85
  default.
- **Warm start** from this corpus's own handcrafted scorer fit, not the TOMATO constants
  hardcoded in `fusion_reasoner.py`.

## Estimated cost of the remaining domains

| domain | docs | queries | extraction |
|---|--:|--:|--:|
| biology | 19,042 | 4,781 | ~$7.50 |
| physics | 13,496 | 3,921 | ~$5.30 |
| matsci | 5,973 | 1,635 | ~$2.30 |

Everything after extraction is free.

## C. Table-4 path interpretations (OpenIE graph / SciGraphIR / SciGraphIR + CCMP, TOMATO-Star + SIR-4 CS)

`colab_table4_openie.ipynb` (generated by `prep/build_table4_notebook.py`, which embeds
`eval/table4.py`) produces the GFM-RAG Table-4 analogue for three arms, OpenIE graph, SciGraphIR (SciAfford graph,
no CCMP) and SciGraphIR + CCMP (gate per hop): per dataset
`outputs/scan/<dataset>/table4_<dataset>.{md,tex}` + `_candidates.json`, and
`outputs/scan/table4_all.tex` for both datasets in one table. Run cell 1, then 2 to 6; cells 2b-3g
are no-ops when `scan_all_openie.json` and `hops_t4_*.json` are already on Drive.

- Candidates: cross-field golds with Qwen3 cosine rank >= 25 that ANY arm recovers (its graph channel <= 5 or
  fused <= 10), deepest-buried 150 queries max; beam 10, top-5 paths, all three arms (`openie`, `frame_nocc`, `frame_ccmp`).
- Ranking ("best example"): best over the recovering arms of rank gap + typed-relation share + readability
  - hub/paper hops + log path weight; simple paths only (no node twice); one example per query. See the docstring of `eval/table4.py`.
- To re-render locally (different filter, SIR-4 field pairs, more examples), download `hops_t4_*.json` and run:

```bash
python3 eval/table4.py --dataset sir4_cs \
  --queries ../retriever/data/sir4_cs_test/raw/test.json \
  --docs ../retriever/data/sir4_cs_test/raw/documents.json \
  --edges ../retriever/data/sir4_cs_test_v16sc/processed/stage1/edges.csv \
  --sir4 ../sir-4/data/benchmark/cs_test_final/eval.json \
  --arm "OpenIE graph=hops_t4_openie.json" --arm "SciGraphIR=hops_t4_frame_nocc.json" \
  --arm "SciGraphIR + CCMP=hops_t4_frame_ccmp.json" --recover any \
  --n-tex 3 --paths 3 --out results/qualitative/table4_sir4_cs
python3 eval/table4.py --combine results/qualitative/table4_sir4_cs_candidates.json \
  results/qualitative/table4_tomato_candidates.json --n-tex 2 --out results/qualitative/table4_all
```

## D. CCMP gate decomposition (why route weights change with the gate on)

`colab_ccmp_gate_decomp.ipynb` (generated by `prep/build_gate_decomp_notebook.py`; engine: `_gate_layers` /
`_gate_nodes` masks in `models/ultra/models.py`, `FusionSFTTrainer.gate_decomposition()`, `interpret_paths.py
+interp.mode=gate_decomp`). For every (query, gold) the showcase run interpreted with a valid route on the
CCMP arm, it follows the top routes and re-evaluates the SAME routes (directly along their edges) and the gold's
ranks under six gate masks on the same weights: on, off, layers 0..L-1 only, layers L..5 only, route senders only,
all but route senders. Writes `outputs/scan/<d>/gate_decomp_<d>.json` and `fig_gate_decomp_<d>.{pdf,png,md}`
(`eval/gate_decomp_fig.py`). Needs `hops_frame_ccmp.json` on Drive (from the showcase notebook) and the CCMP
checkpoint; ~7 forward/backward passes per gold. Run cell 1, then 2 to 5.
