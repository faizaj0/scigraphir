#!/usr/bin/env python3
"""Build the transparent notebook that discovers current-model scientific paths."""
import json

from build_table4_notebook import S4, common_cells, code, finalize_notebook, initialization_cell, md


INTRO = r"""# Show the model's scientific reasoning paths

This notebook answers: **Which graph routes receive the most attribution when the current model scores a relevant paper from another scientific domain?**

All implementation is visible below as ordinary Python and `%%writefile` cells. There are no base64 blobs, encoded overlays, or runtime source decoding.

For every selected question/target pair, the notebook loads the available merged-graph CCMP checkpoint and matching graph/scorer where possible, computes gradients of the target paper's graph score with respect to layer edge weights, and applies the engine's NBFNet beam search. Previously saved paths are not used to select the result.

It reports the target's actual graph, fused, scorer and dense ranks. Overall top paths include one-hop paths; a separately labelled subset requires at least two hops. Cyclic, disconnected, placeholder and wrong-target routes are rejected rather than replaced by curated examples.

A useful cross-domain route connects a concrete problem, limitation or function in the query domain to a method or finding in the target paper's domain. A domain hub, alias, large weight, or benchmark `cross` label alone is weak evidence. Validate the extracted relations against the papers before using a path in the manuscript.
"""


READOUT = r"""### How to read the output

Follow each triple from left to right. A relation marked with `⁻¹` traverses an existing graph edge backwards. The path weight is the mean of its edge gradients for the selected target paper; it is not confidence, probability, or a product of gates.

The target paper is pinned from the benchmark's relevant documents. Its reported rank tells you whether the model actually retrieved it near the top. The paths explain sensitivity of its graph score; they are not generated chain-of-thought and do not prove that the retrieved method works in the query domain.

The notebook exports `table4.md`, `table4.tex`, `top_paths.tikz.tex`, `path_hops.csv`, `results.json`, and `discovered_cases_for_interventions.json` under the printed Drive result directory.
"""


def build():
    cases = json.loads((S4 / "eval/ccmp_mechanism_cases.json").read_text())
    cells = common_cells(INTRO, ["bfloat16"])
    cells += [md("## 6. Select checkpoints and prepare graph-aligned inputs"),
              initialization_cell(cases), code((S4 / "colab_cells/colab_analysis_setup_cell.py").read_text()),
              md("## 7. Discover and display top paths"),
              code((S4 / "colab_cells/colab_scientific_paths_cell.py").read_text()), md(READOUT)]
    notebook = finalize_notebook(cells, "scientific-transparent",
                                 {"mode": "paths", "top_k": 5, "beam_size": 10})
    output = S4 / "notebooks/colab_scientific_paths.ipynb"
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {output}: {len(cells)} transparent cells, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    build()
