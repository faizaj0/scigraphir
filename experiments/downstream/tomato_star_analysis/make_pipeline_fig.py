"""Render the downstream experiment-setup pipeline as a presentation figure.

API-FREE.  Produces a clean horizontal flow:

    Query -> COMPOSE (fixed) -> Hypothesis -> Evaluate (2 judges)

with the retrieved TOP-1 inspiration feeding UP into the composer, highlighted as
"the only thing that changes between arms".  The point of the slide is the control:
composer + prompt are identical across arms, so any output difference is caused by
retrieval.

Run:
  cd TOMATO-Star
  python -m analysis.downstream.make_pipeline_fig

Output: results/downstream/figures/pipeline.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "downstream" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, title, sub="", fc="#FFFFFF", ec="#555555",
        lw=1.6, tsize=12, ssize=9.5, tcolor="#111111"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=tcolor, zorder=3)
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center",
                fontsize=ssize, color=tcolor, zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=tcolor, zorder=3)


def arrow(ax, x0, y0, x1, y1, color="#B4BAC2", lw=2.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0,
                                mutation_scale=18), zorder=1)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # pastel palette  (fill, soft edge)
    LAV = ("#EDE7F6", "#C3B4E0")     # query
    GRAY = ("#F2F3F5", "#CDD2D8")    # hypothesis
    BLUE = ("#E4EEF8", "#A6C8E8")    # compose
    GREEN = ("#E7F3EA", "#AFD6B5")   # evaluate
    ORANGE = ("#FCEBD7", "#EFC597")  # retrieve (gentle highlight)
    INK = "#46494D"                  # soft body text
    PEACH_TXT = "#B9763C"            # muted terracotta for the callout

    # --- top row: Query -> Compose -> Hypothesis -> Evaluate ---------------
    ytop, h = 60, 18
    box(ax, 1, ytop, 24, h, "Query",
        "research question,\nbackground, prior hypothesis",
        fc=LAV[0], ec=LAV[1], lw=1.6, tcolor=INK)
    box(ax, 33, ytop, 22, h, "Compose",
        "gpt-4o-mini\n(fixed prompt)", fc=BLUE[0], ec=BLUE[1], lw=1.6,
        tcolor=INK)
    box(ax, 62, ytop, 15, h, "Hypothesis", fc=GRAY[0], ec=GRAY[1], lw=1.6,
        tcolor=INK)
    box(ax, 81, ytop, 18, h, "Evaluate",
        "Matched-Score\nIdea Arena", fc=GREEN[0], ec=GREEN[1],
        lw=1.6, ssize=9.0, tcolor=INK)

    ymid = ytop + h / 2
    cx_compose = 33 + 22 / 2
    arrow(ax, 25, ymid, 33, ymid)
    arrow(ax, 55, ymid, 62, ymid)
    arrow(ax, 77, ymid, 81, ymid)

    # --- feeder: retrieved inspiration -> up into Compose ------------------
    rb_w, rb_h = 30, 15
    rb_x = cx_compose - rb_w / 2          # centred under Compose
    rb_y = 33
    box(ax, rb_x, rb_y, rb_w, rb_h, "Retrieve top-1 inspiration",
        "varies across conditions", fc=ORANGE[0], ec=ORANGE[1], lw=1.6,
        tsize=12, ssize=9.5, tcolor=PEACH_TXT)
    arrow(ax, cx_compose, rb_y + rb_h, cx_compose, ytop,
          color="#E3B27E", lw=2.2)

    # retrieval conditions, centred under the feeder (no emphasis)
    ax.text(cx_compose, 24,
            "Retrieval conditions:   No-insp  ·  BM25  ·  MOOSE-Chem  ·  Oracle",
            ha="center", va="center", fontsize=10.5, color="#7A6450")
    ax.text(cx_compose, 18.5,
            "empty   ·   lexical   ·   cross-domain   ·   gold",
            ha="center", va="center", fontsize=9, color="#A9947E")

    # title + neutral caption
    ax.text(50, 96,
            "Controlled hypothesis-composition experiment",
            ha="center", va="center", fontsize=14.5, fontweight="bold",
            color="#3A3D41")
    ax.text(50, 88.5,
            "The composer and prompt are held fixed across all conditions; "
            "only the retrieved inspiration varies,\nso any difference in the "
            "composed hypothesis is attributable to retrieval.",
            ha="center", va="center", fontsize=10, color="#6B7178",
            linespacing=1.4)

    out = OUT / "pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
