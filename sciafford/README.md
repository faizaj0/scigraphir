# SciAfford: graph construction

[Repository home](../README.md) · [Code guide](../docs/README.md) · [Retriever](../retriever/README.md)

**The default SciAfford graph combines affordance structure with OpenIE entity context
and direct paper-to-affordance links.** It is assembled by
[build_hybrid_graph.py](../experiments/prep/build_hybrid_graph.py) and stored with the
`_hyb` suffix used by the training workflows.

Affordance representations describe the functions papers provide, limitations they
address, mechanisms they use and methods they build on. Problem requirements identify
starting concepts. OpenIE adds query entity seeds and entity-to-paper mention edges;
direct paper-to-affordance links make contribution capabilities accessible in one hop.

## Files and stages

| Stage | Code | Input → output |
|---|---|---|
| Paper affordances | [extract_affordances.py](extract_affordances.py), `--side doc` | Titles/abstracts → contribution capabilities. |
| Problem requirements | [extract_affordances.py](extract_affordances.py), `--side query` | Research questions → requirements, methods and limitations. |
| Affordance component | [build_greasoner_dataset.py](build_greasoner_dataset.py) | Representations → typed graph and query seeds (`_v16sc`). |
| OpenIE component | [run_index.sh](../retriever/run_index.sh) | Corpus/questions → entities, mention edges and linked query entities. |
| **Default graph** | [build_hybrid_graph.py](../experiments/prep/build_hybrid_graph.py) | Both components → SciAfford graph with entity context and paper-to-affordance links (`_hyb`). |

The affordance builder merges similar concepts, adds typed similarity links and maps
query phrases to graph nodes. The final builder preserves that structure, adds the
OpenIE query entities that pass its degree cap and have paper links, and adds direct
links from papers to the capabilities represented through their methods, tasks and findings.

## Graph vocabulary

| Source | Relation examples | Target |
|---|---|---|
| Paper | `contributes`, `reports`, `addresses`, `in_field` | Method, finding, task, domain |
| Method | `achieves`, `overcomes`, `works_via` | Function, limitation, mechanism |
| Method or finding | `builds_on`, `improves_on` | Method |
| Task or method | `limited_by` | Limitation |
| Finding | `concerns`, `explains` | Function, limitation |
| Entity | `mentioned_in` | Paper |
| Paper | `paper_achieves`, `paper_overcomes`, `paper_works_via`, `paper_limited_by`, `paper_concerns`, `paper_explains` | Affordance concept |

Concept names include their type, such as `[function] partition an input into labeled
regions`. Paper nodes use document IDs. Similarity relations connect concepts of the same type.

## Build the default graph

First follow [setup and data](../docs/SETUP.md) and stage a corpus with
[stage_sir4.py](../experiments/prep/stage_sir4.py). Run from the repository root using
Python 3.12 and the installed graph engine. Affordance extraction and OpenIE construction
use `OPENAI_API_KEY` and make paid calls for uncached items. Existing matching components
can be reused.

```bash
export SCIGRAPHIR_ROOT="$PWD"
for split in train test; do
  for side in doc query; do
    python sciafford/extract_affordances.py --dataset sir4_physics \
      --split "$split" --side "$side" --workers 16
  done
  python sciafford/build_greasoner_dataset.py --dataset sir4_physics \
    --split "$split" --tau_canon 0.95 --no_entity_seeds --no_answer_seeds
  bash retriever/run_index.sh "sir4_physics_${split}"
done

python experiments/prep/build_hybrid_graph.py --dataset sir4_physics --cap 30

for split in train test; do
  python experiments/eval/audit_graph.py --dataset sir4_physics \
    --split "$split" --suffix hyb --sample 40
done
```

The merge step runs locally without further LLM calls. By default it includes only
query-seed entities under the degree cap; `--all-entities` is an optional expansion.

The graph used by the full method is written to
`retriever/data/<dataset>_<split>_hyb/`, with `raw/documents.json` and a
`processed/stage1/` directory containing `nodes.csv`, `edges.csv`, `relations.csv`
and `<split>.json`. The input components remain available for construction and comparisons.

## Training and component comparisons

Start with the [default SIR-4 graph notebook](../experiments/notebooks/colab_sir4_hyb.ipynb).
It runs the default graph with and without CCMP using the learned multi-view semantic scorer.

- The `_v16sc` graph is the affordance component used before the final merge and in comparisons.
- The OpenIE-only graph is a component and a graph-construction control.
- [make_noent_dataset.py](make_noent_dataset.py) prepares a seed ablation of the affordance component.
- The [preparation runbook](../experiments/RUNBOOK.md) describes component bundles and the default graph workflow.
