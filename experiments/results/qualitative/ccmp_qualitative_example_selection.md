# Recommended qualitative example

Use **protein-RNA binding-site prediction -> optimal transport for class-imbalanced visual recognition** as the main figure. This was selected after comparing the available SIR-4 CS, biology, and physics path dumps (886 query records); materials-science dumps in this directory are empty. This is a selection of illustrative cases, not an aggregate performance result.

| Example | CCMP graph rank off -> on | Final fused rank off -> on | Qwen3 cosine | Multi-view scorer | Why use it |
|---|---:|---:|---:|---:|---|
| Protein-RNA -> optimal transport | 436 -> 3 | 10 -> 8 | 113 | 13 | Best main figure: two direct two-hop paths connect the query's class-imbalance need to the paper's method; no domain-node detour. |
| LLM creativity -> constraining effects of examples | 4 -> 2 | 25 -> 18 | 169 | 49 | Intuitive second example and larger fused-rank gain; the leading path goes through a broad cognitive-psychology domain node. |
| Streaming-video memory -> reconsolidation | 3 -> 2 | 37 -> 28 | 380 | 93 | Strong fused-rank gain among these cases, but weaker direct correspondence between the query and the reconsolidation seed, and a worse final rank. |

## Recommended figure

`fig_ccmp_biology_ot_iclr.tex` is a self-contained editable TikZ source. Compile with `tectonic fig_ccmp_biology_ot_iclr.tex`. The PDF uses the local ICLR template's 5.5-inch text width. To insert native TikZ in a manuscript, reuse the color and style definitions plus the `tikzpicture` inside a normal `figure` environment. The source contains no raster graphics.

The figure shows the two highest-weighted simple frame paths and the highest-weighted simple OpenIE path. The second OpenIE path differs only in the direction of the equivalent relation and is omitted for clarity. Frame path weights, off -> on: 9.28 -> 12.44 and 8.06 -> 11.31. OpenIE graph rank is 3,329 and fused rank is 13.

All numbers were read from the same query/gold pair in `drive_scan_sir4_biology/hops_frame_ccmp.json`, `hops_frame_ccmp_off.json`, and `hops_openie.json`. Exact supporting records are saved in `fig_ccmp_biology_ot_iclr_evidence.json`.

## Interpretation limits that matter for the figure

- The shown frame-hop gates are approximately 0.99991, 1.00027, and 1.00147. The figure now labels these on the hops to four decimal places: 0.9999, 1.0003, and 1.0015. CCMP off applies g = 1. It also shows complete-path weights inside the numbered path seed boxes. The rank change is a whole-model comparison and is not explained solely by the two displayed paths.
- Gradient path weights are attributions, not probabilities; their scales differ across graph models.
- OpenIE is a separate arm. Its scorer rank is 11 here, whereas the frame arm's scorer rank is 13. The figure therefore does not portray every row as one cumulative ablation.
- The creativity OpenIE dump contains zero-weight traces that do not terminate at the selected gold. This supports saying that the stored traces provide no valid seed-to-gold explanation, but does not establish that no such route exists anywhere in the graph.
