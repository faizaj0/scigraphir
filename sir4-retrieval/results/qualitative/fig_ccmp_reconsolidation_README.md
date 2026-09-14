# Reconsolidation qualitative figure

The revised figure uses a 5.5-inch-wide vector drawing with embedded fonts and a tight standalone crop.

## Files and compilation

`fig_ccmp_reconsolidation_standalone.pdf` is the revised figure with caption. The editable sources are `fig_ccmp_reconsolidation_artwork.tex` and `fig_ccmp_reconsolidation_caption.tex`. Compile the standalone wrapper from this directory with:

```sh
tectonic fig_ccmp_reconsolidation_standalone.tex
```

For a manuscript, place the figure wrapper, artwork, and caption beside the main TeX file and input `fig_ccmp_reconsolidation.tex`. The preamble needs `xcolor,tikz,graphicx,float,lmodern,helvet` and the TikZ libraries `arrows.meta,calc`. Alternatively, use the `import` package and `\import{path/to/qualitative/}{fig_ccmp_reconsolidation.tex}` to resolve nested inputs from another directory. Adjust float placement `[H]` to match the paper.

Original files are preserved under `archive/ccmp_reconsolidation_before_iclr_20260909/`.

## Data fidelity

Query: `10.48550_arxiv.2603.03985`. Gold: `10.1111/j.1749-6632.2010.05443.x`.

Ranks, path identities, route weights, and gates were checked against `drive_scan_sir4_cs/hops_frame_ccmp.json`, `hops_frame_ccmp_off.json`, and `hops_openie.json`.

The figure retains the top two simple frame paths; the cyclic CCMP-on path with weight 7.73 is excluded, as in the original. OpenIE weights are approximately -0.001304 each and are shown as approximately zero. Gate values describe CCMP on; the off-run diagnostic logs also contain predicted gates. The multi-view scorer rank remains 93, as in the source figure. The OpenIE arm has scorer rank 102 in its diagnostics, so the comparison is not presented as a strictly cumulative ablation.

The final PDF was rendered and visually inspected, checked for text inside page bounds, and confirmed to contain vector graphics and embedded fonts.
