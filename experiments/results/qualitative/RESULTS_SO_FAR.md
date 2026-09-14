# Results so far (9 Sep 2026)

SciAffordGraph = the merged graph (frame + entity seeds + mention edges + shortcuts). Sections 1 and 2 are on the merged graph. Section 3 is per-gold analysis that still runs on the older frame-only graph and will be regenerated once the merged-graph scan exists.

## 1. Merged graph, fused ranking (current engine)

### SIR-4 MatSci (331 queries: 210 same, 121 cross)

| Method | nDCG@5 same | nDCG@5 cross | R@5 same | R@5 cross |
|---|--:|--:|--:|--:|
| Multi-View Semantic Scorer (thesis) | 53.14 | 47.33 | 49.40 | 43.15 |
| + Graph Reasoner, frame graph (thesis) | 54.65 | 49.71 | 50.87 | 45.86 |
| + CCMP = SciGraphIR (thesis) | 54.95 | 49.90 | 51.55 | 46.07 |
| + Graph Reasoner, OpenIE graph (6 Sep, current engine) | 55.01 | 49.02 | 51.16 | 44.37 |
| merged graph, no CCMP (8 Sep, current engine) | 53.99 | 48.49 | 49.86 | 44.12 |
| merged graph + CCMP (8 Sep, current engine) | 55.20 | 49.53 | 51.13 | 45.62 |

Paired bootstrap, cross-field nDCG@5: merged CCMP vs merged control +0.0103 [+0.0001, +0.0212]; merged CCMP vs OpenIE +0.0051 [-0.0138, +0.0230]; merged control vs OpenIE -0.0053 [-0.0242, +0.0126]. Thesis rows are the older-engine 31 Aug tables.

### MIR (155 queries, all same-field)

| Method | nDCG@5 | R@5 |
|---|--:|--:|
| Multi-View Semantic Scorer | 30.93 | 40.55 |
| + Graph Reasoner, frame graph (thesis) | 31.22 | 40.55 |
| + CCMP, frame graph (thesis) | 31.68 | 41.78 |
| merged graph, no CCMP (8 Sep) | 32.70 | 43.13 |
| merged graph + CCMP (8 Sep) | 32.69 | 41.62 |

Graph channel alone on the merged graph: 19.92 nDCG@5 (no CCMP) / 18.76 (CCMP) vs walk prior 16.20; recall@10 of scorer-buried golds 0.00 in both arms. CCMP vs control: -0.0001 nDCG@5 [-0.019, +0.020].

### TOMATO (3,132 rows; 2,843 same, 289 cross)

| Method | graph | nDCG@5 same | nDCG@5 cross | R@5 same | R@5 cross |
|---|---|--:|--:|--:|--:|
| Multi-View Semantic Scorer | - | 32.33 | 22.01 | 41.75 | 29.76 |
| + Graph Reasoner, no routing (thesis) | frame | 34.30 | 24.30 | | |
| + CCMP (thesis) | frame | 34.28 | 25.23 | | |
| + Graph Reasoner, OpenIE graph (July v1 model: BGE operator scorer, not the multi-view scorer) | OpenIE | 34.31 | 21.04 | 44.35 | 27.68 |
| + Graph Reasoner, no routing (9 Sep) | merged | 34.86 | 24.74 | 44.95 | 33.56 |
| + CCMP (9 Sep) | merged | 34.56 | 25.05 | 44.67 | 34.60 |

Graph channel alone (merged): 26.41 nDCG@5 (no CCMP) / 26.30 (CCMP) vs walk prior 21.74; graph R@10 on scorer-buried golds 2.70 both arms; best epoch 8 both.
Paired bootstrap, merged CCMP vs merged control, nDCG@5: all -0.0024 [-0.0062, +0.0015]; cross +0.0031 [-0.0087, +0.0152]; same -0.0030 [-0.0070, +0.0012]. Frame-graph CCMP vs no routing (thesis table): cross +0.93 [+0.20, +1.77].

### Status of the other merged-graph runs (Drive, 9 Sep 10:00)

| dataset | control | CCMP | scored |
|---|---|---|---|
| TOMATO | finished | finished 9 Sep 10:00 | yes |
| SIR-4 Physics | training | not started | no |
| SIR-4 Biology, CS | queued after physics | | no |

## 2. What the merged graph changes and what it does not

- MIR: +1.0 to +1.5 nDCG@5 and +2.6 R@5 over the frame-graph rows; first construction change to beat the frame graph at the fused level.
- MatSci: level with the frame graph and with the OpenIE row (all differences inside the bootstrap CI); CCMP adds +1.0 cross-field nDCG@5 inside the merged runs (CI clears zero).
- On both datasets the graph channel still finds none of the golds the scorer buries past rank 1,000.

## 3. Per-gold analyses on the OUTDATED frame-only graph (hops files of the showcase scan; SIR-4 arms are the zero-shot pair checkpoints)

These are the figures built today. They describe the frame-only reasoner and must be regenerated from hops_hyb_*.json.

### Graph channel alone, share of golds within rank 10 (%), cross-field golds

| dataset | stratum | n | frame, gate on | frame, gate off | entity graph | scorer | cosine |
|---|---|--:|--:|--:|--:|--:|--:|
| biology | cross | 200 | 25.5 | 18.0 | 31.0 | 41.5 | 37.5 |
| biology | same | 891 | 35.4 | 19.1 | 39.7 | 64.9 | 63.4 |
| cs | cross | 383 | 18.8 | 19.1 | 13.6 | 33.4 | 29.0 |
| cs | same | 1257 | 26.7 | 26.7 | 13.9 | 42.0 | 41.9 |
| matsci | cross | 256 | 30.5 | 29.7 | 28.1 | 47.3 | 38.3 |
| matsci | same | 711 | 37.3 | 37.6 | 32.5 | 61.3 | 59.4 |
| mir | same | 235 | 27.7 | 27.7 | 23.8 | 48.1 | 41.3 |
| physics | cross | 240 | 24.2 | 14.2 | 21.2 | 43.8 | 38.3 |
| physics | same | 731 | 38.0 | 15.3 | 39.7 | 67.4 | 67.0 |
| tomato | cross | 289 | 28.7 | 29.1 |  | 39.8 | 25.3 |
| tomato | same | 120 | 46.7 | 47.5 |  | 56.7 | 49.2 |

### Cumulative ablation, final ranking, share of golds within rank 10 (%)

| dataset | stratum | n | cosine | scorer | + entity graph | + frame graph (gate off) | + CCMP |
|---|---|--:|--:|--:|--:|--:|--:|
| sir4_cs | cross | 383 | 29.0 | 33.4 | 33.2 | 32.9 | 32.4 |
| sir4_cs | same | 1257 | 41.9 | 42.0 | 41.6 | 42.3 | 41.9 |
| sir4_biology | cross | 200 | 37.5 | 41.5 | 47.0 | 43.5 | 43.0 |
| sir4_biology | same | 891 | 63.4 | 64.9 | 65.8 | 64.9 | 65.3 |
| sir4_physics | cross | 240 | 38.3 | 43.8 | 45.4 | 46.7 | 44.6 |
| sir4_physics | same | 731 | 67.0 | 67.4 | 68.0 | 67.9 | 67.9 |
| sir4_matsci | cross | 256 | 38.3 | 47.3 | 52.3 | 36.7 | 35.5 |
| sir4_matsci | same | 711 | 59.4 | 61.3 | 61.9 | 46.1 | 44.7 |
| tomato | cross | 289 | 25.3 | 39.8 |  | 40.1 | 40.1 |
| tomato | same | 120 | 49.2 | 56.7 |  | 58.3 | 58.3 |
| mir | same | 235 | 41.3 | 48.1 | 50.2 | 48.1 | 48.1 |

MatSci stages 4 and 5 collapse because the showcase arm for SIR-4 is the zero-shot pair checkpoint (scigraphir_cs+matsci), not the in-field model; treat those two cells as a provenance artefact.

### Gate on vs off, same weights, graph channel (better / worse / same golds)

| dataset | stratum | n | better | worse | same | mean Δlog10 rank [95% CI] |
|---|---|--:|--:|--:|--:|---|
| sir4_cs | cross | 383 | 128 | 194 | 61 | -0.012 [-0.082, +0.054] |
| sir4_cs | same | 1257 | 658 | 347 | 252 | -0.332 [-0.370, -0.295] |
| sir4_biology | cross | 200 | 109 | 80 | 11 | -0.392 [-0.544, -0.246] |
| sir4_biology | same | 891 | 640 | 181 | 70 | -0.868 [-0.939, -0.799] |
| sir4_physics | cross | 240 | 137 | 92 | 11 | -0.337 [-0.471, -0.211] |
| sir4_physics | same | 731 | 556 | 121 | 54 | -0.966 [-1.037, -0.895] |
| sir4_matsci | cross | 256 | 75 | 109 | 72 | +0.040 [-0.016, +0.097] |
| sir4_matsci | same | 711 | 317 | 185 | 209 | -0.171 [-0.207, -0.140] |
| tomato | cross | 289 | 83 | 137 | 69 | +0.038 [+0.022, +0.054] |
| tomato | same | 120 | 35 | 42 | 43 | +0.005 [-0.018, +0.029] |
| mir | same | 235 | 98 | 80 | 57 | +0.013 [-0.020, +0.047] |

Fused ranking: within ±0.02 log10 rank on every dataset and stratum (the gate never reaches the final ranking).

### Route structure (replicates on all six datasets)
- P(graph rank ≤ 10 | top route length): ~1.0 at 1 hop, 0.5-0.8 at 2 hops, ≤ 0.15 at 3 hops, ~0 beyond; cross- and same-field curves coincide.
- The CCMP gate rises with the sender's degree on every dataset (1.06 on CS, 1.3-1.5 on biology/physics for degree ≥ 100).
- Hub inflation (heavy hub routes that do not rank the gold) holds on CS only.

## 4. Files
- Merged graph: results/qualitative/hyb_scores/ (MatSci scores + bootstrap, MIR routing table).
- Frame-graph figures: figures/fig_graph_channel_curves_<ds>.png, fig_graph_channel_head.png, fig_ablation_head_{cross,same}.png, fig_ablation_ladder.png, fig_graph_reasoning_summary.png, fig_graph_reasoning_<ds>.png, fig_route_circles_*.png, tab_route_paths_*.tex.
- Tables: results/qualitative/graph_channel_curves_all.md, ccmp_paired_stats_all.md, figures/fig_ablation_ladder.md, fig_graph_channel_head.md.
