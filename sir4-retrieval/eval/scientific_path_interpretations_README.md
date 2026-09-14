# Scientific path interpretations, explained simply

Open `sir4-retrieval/colab_scientific_paths.ipynb` in Colab, select an A100 or L4 GPU, and run all cells. This is the notebook to use first: it shows the current model's top graph paths to selected relevant scientific papers. It embeds the code and uses your existing Drive bundles, scorers and checkpoints. It performs inference and gradients, with no training.

`colab_table4_openie.ipynb` is the separate, more detailed experiment about **why CCMP changes a fixed path's attribution**. Path discovery and gate interventions answer different questions.

## What the original Table 4 means

The [GFM-RAG paper, Section 4.8](https://proceedings.neurips.cc/paper_files/paper/2025/file/33ca0b1102b54c191a9a45a05adafaf4-Paper-Conference.pdf) shows graph routes that receive high attribution for a selected prediction. Its first example can be read as:

> song → singer → the singer's full name → football club

Each parenthesized triple is one graph edge: `(head, relation, tail)`. The next triple starts where the previous one ends. An inverse relation means walking backwards along an existing edge. For example, `(Elton John, named a stand after⁻¹, Watford)` reads as “Watford named a stand after Elton John.” It does not mean the club itself was named after him. The second path's `owned` edge answers the ownership question more directly, although its attribution is lower.

The number `1.095` is an attribution score, not 109.5% confidence. Higher attribution does not guarantee a logically complete or factually correct explanation. A generated answer can also depend on retrieved text whose details do not appear in the graph path. Multiple graph layers allow multiple hops within one retrieval forward pass.

## How the code calculates a path weight

For the selected target paper, let `s` be its graph score. Give each graph edge a differentiable scalar multiplier `a`. At the normal forward pass those edge multipliers are 1. Autograd measures how sensitive `s` is to those multipliers. It does not differentiate words such as “owned.”

For a path with `m` edges, the local engine uses:

\[
w(p)=\frac{1}{m}\sum_{l=1}^{m}\frac{\partial s}{\partial a^{(l)}_{e_l}}.
\]

For example, edge gradients `[0.9, 1.2, 0.6]` give path weight `0.9`. This is an illustrative calculation, not a measured result from your checkpoint. Scores can be signed and should not be compared as calibrated confidence across questions, targets or checkpoints.

The engine's NBFNet beam search accumulates edge-gradient scores and then ranks candidate paths by their length-averaged score. “Longest” refers to weighted path score, not simply the most hops. A finite beam is approximate. The notebook removes cyclic, disconnected, non-seed-starting and wrong-target paths, and checks that each displayed triple resolves to one exact graph edge. It reports the highest-scoring valid candidates it found; it does not guarantee the global optimum over all possible paths.

For compatibility with your existing figures, the implementation retains the engine's legacy gradient convention: successive layer edge-weight tensors are linked clones, rather than independent detached perturbations. The formula refers to those actual autograd variables. Changing this convention would define a different attribution experiment and require new measurements.

The CCMP gate `g` is a separate quantity. It multiplies a sender node's message at a layer: below 1 suppresses, above 1 amplifies. The displayed path weight is **not the product of gates**. The output records both raw gates and the values actually applied after the message-dtype cast; values very close to 1 may round to exactly 1 in bfloat16.

## What would support a cross-domain scientific reasoning claim?

For your biology question, a useful mechanism bridge to examine is:

> protein–RNA prediction with class imbalance → class-imbalance limitation → optimal-transport method → a computer-vision paper

The question supplies the biological context. A displayed graph path starts at the question's linked seed concepts; it need not contain a node for the whole question. This route is a candidate pattern to check, not a claim that the current merged checkpoint's beam has already selected it.

The notebook starts with your three selected question/target pairs: biology/optimal transport, creativity/fixation, and memory reconsolidation. It discards their previously curated paths and discovers new ones from the selected checkpoint. It shows:

- The actual question, query dataset field, target paper title, target's graph domain labels, and benchmark stratum.
- The target's actual graph, final fused, learned-scorer and dense-retrieval ranks (1 is best), plus the top retrieved papers.
- Overall top paths, including single-hop paths, with weights and sender gates.
- A separately labelled subset requiring at least two hops. Its P-numbers retain their rank among all valid beam candidates.
- Flags for routes containing domain hubs, aliases, or no recognized mechanism/function/limitation relation. These flags are inspection aids, not scientific validators.

The target is pinned to a benchmark-relevant paper, not necessarily the model's top retrieval. A path only through a shared domain node is weaker evidence than a specific mechanism linking a problem to a method. Check the original papers to verify the extracted relations and the proposed relevance to the query.

A defensible qualitative claim, **if supported by the measured rows**, is: “The model retrieves relevant papers across scientific domains through multi-hop paths connecting shared problems and mechanisms.” These examples alone do not establish dataset-wide generalization or prove that a retrieved method works in the new domain. For a broad performance claim, report held-out cross-domain retrieval results alongside the qualitative paths.

## Settings and outputs

The relevant configuration is:

```python
DATASETS = ["sir4_biology", "sir4_cs"]
PREFER_MERGED = True
PRECISIONS = ["bfloat16"]
TOP_PATHS = 5
BEAM_SIZE = 10
```

Merged-graph CCMP checkpoints are preferred under `outputs/sir4_hyb/`, with matching `_test_hyb` graphs and field scorers. The standard e10 runs are preferred, then other available merged CCMP runs. A frame checkpoint is an explicit fallback only if no merged checkpoint exists. A present merged checkpoint with missing or incompatible assets causes an error. The notebook prints and records its exact selection.

Results are written to `outputs/scientific_paths/<dataset>/<merged-or-frame>/<precision>/<manifest-hash>/` on Drive:

| File | Use |
|---|---|
| `table4.md` | Readable questions, ranks and paths, also displayed in the notebook |
| `table4.tex` | Table 4 in LaTeX; the top two overall paths per example |
| `top_paths.tikz.tex` | Editable TikZ strips for the top two overall paths, with weights and gates |
| `path_hops.csv` | Every displayed hop's edge gradient, raw gate and applied gate, when paths exist |
| `results.json` | All reported paths, metadata, ranks, retrieved papers and verification results |
| `discovered_cases_for_interventions.json` | The top two discovered paths per case, fixed for later CCMP controls |
| `manifest.json`, `gates_*.pt` | Reproducibility metadata and full native gate tensors |

The LaTeX table needs `booktabs,tabularx`. The TikZ file needs `tikz` and `\usetikzlibrary{arrows.meta}`. These are editable exports; check the layout after measuring real paths, whose label lengths vary.

## Reusing the extraction function

The complete implementation is `eval/scientific_path_interpretations.py`; Colab installs it as `gfmrag.workflow.scientific_paths`. The notebook handles loading the model, graph and question batch. Inside an already-loaded workflow, the call is:

```python
from gfmrag.workflow.scientific_paths import interpret_scientific_paths

result, gates = interpret_scientific_paths(
    model=trainer.model,
    graph=graph,
    batch=batch,        # one question, including its linked seed mask and gold mask
    src=src,            # loaded graph dataset, with node/relation names
    case=case,          # name, dataset, query_id, gold_id
    dtype=trainer.dtype,
    documents=documents,
    top_k=5,
    beam_size=10,
)
for path in result["top_paths"]:
    print(path["name"], path["weight"])
    for hop in path["hops"]:
        print(hop["head"], hop["rel"], hop["tail"], "g =", hop["g_applied"])
```

To run the gate experiment on these newly discovered paths, add a cell in `colab_table4_openie.ipynb` **after installation and before the intervention driver**:

```python
import json

files = [
    # Paste the full Drive paths printed by the discovery notebook here.
]
assert files, "Choose the discovered_cases_for_interventions.json files first"
CASES = [case for filename in files for case in json.load(open(filename))]
DATASETS = sorted({case["dataset"] for case in CASES})
```

The workflow rejects discovered paths from a different graph or checkpoint. Keeping these paths fixed is essential: selecting new top paths after every intervention would mix changes in attribution with changes in which paths are being compared.

## Validation

```sh
python3 sir4-retrieval/prep/build_scientific_paths_notebook.py
python3 -m unittest discover -s sir4-retrieval/eval -p 'test_ccmp*.py' -v
python3 -m unittest discover -s sir4-retrieval/eval -p 'test_scientific_path_interpretations.py' -v
```

Local tests execute the real patched message-passing method and real NBFNet beam search with small CPU layers. They also exercise strict checkpoint loading, workflow exports, cache reuse and metadata invalidation. These tests do not execute the Drive checkpoints or the full Colab CUDA dependency stack. The actual scientific measurements require running the notebook in Colab.
