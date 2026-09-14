#!/usr/bin/env python3
"""Figure: path-sender gate effect (Delta_P) per example, float32 headline.

Bars = target paper's graph score with all gates 1, with CCMP on, and after resetting only one
path's sender gates to 1 (recorded gates elsewhere). Rank above each bar. Reads the downloaded
summary.csv files when present (results/qualitative/ccmp_path_sender_effect/drive/latest_*_float32/),
otherwise the numbers pasted from the 13 Sep 2026 Colab run (same run, rounded to 3 decimals).

    python3 eval/fig_path_sender_effect.py            # writes figures/fig_path_sender_effect.{pdf,png}
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1] / "results/qualitative/ccmp_path_sender_effect"
OUT = HERE / "figures"

# (example, title, field pair, off score, off rank, on score, on rank, [(path, delta_P, rank after reset), ...])
PASTED = [
    ("biology_optimal_transport", "Biology query → computer-vision paper",
     "protein–RNA binding sites → optimal-transport imbalance", 14.792, 456, 48.342, 3,
     [("P1", 5.613, 4), ("P2", 5.404, 4), ("P3", 5.615, 4)]),
    ("creativity_fixation", "CS query → psychology paper",
     "LLM idea diversity → constraining effects of examples", 55.328, 4, 62.089, 3,
     [("P1", 2.347, 4), ("P2", 2.511, 4), ("P3", 0.184, 3)]),
    ("memory_reconsolidation", "CS query → neuroscience paper",
     "video-LLM memory benchmark → memory reconsolidation", 73.473, 3, 88.723, 2,
     [("P1", 10.015, 3), ("P2", 10.012, 3), ("P3", 1.599, 2)]),
    ("dueling_bandits", "CS query → decision-sciences paper",
     "preference-based test-time optimisation → double TS", 79.528, 1, 80.753, 1,
     [("P1", 0.251, 1), ("P2", 0.248, 1), ("P3", 0.178, 1)]),
]


def from_summary_files():
    """Same layout from the downloaded float32 summary.csv files, if they exist."""
    rows = []
    for f in sorted((HERE / "drive").glob("latest_*_float32/summary.csv")):
        with open(f) as stream:
            rows += list(csv.DictReader(stream))
    if not rows:
        return None
    by_case = {}
    for r in rows:
        c = by_case.setdefault(r["case"], {"paths": []})
        c["off"], c["off_rank"] = float(r["graph_score_off"]), int(r["graph_rank_off"])
        c["on"], c["on_rank"] = float(r["graph_score_on"]), int(r["graph_rank_on"])
        c["paths"].append((r["path"], float(r["delta_P"]), int(r["graph_rank_after_reset"])))
    out = []
    for name, title, pair, *_ in PASTED:
        if name in by_case:
            c = by_case[name]
            out.append((name, title, pair, c["off"], c["off_rank"], c["on"], c["on_rank"], sorted(c["paths"])))
    return out or None


def draw(data, path_stem):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), constrained_layout=True)
    axes = axes.flatten()
    colours = {"off": "#b0b0b0", "on": "#1f4e79", "reset": "#7fa7d0"}
    for ax, (name, title, pair, off, off_rank, on, on_rank, paths) in zip(axes, data):
        labels = ["all gates 1", "CCMP on"] + [f"reset {p}\nsenders" for p, _, _ in paths]
        values = [off, on] + [on - d for _, d, _ in paths]
        ranks = [off_rank, on_rank] + [r for _, _, r in paths]
        cols = [colours["off"], colours["on"]] + [colours["reset"]] * len(paths)
        bars = ax.bar(range(len(values)), values, color=cols, width=0.72)
        top = max(values)
        for i, (b, v, r) in enumerate(zip(bars, values, ranks)):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015 * top, f"rank {r}", ha="center", va="bottom", fontsize=7.5)
            if i >= 2:
                ax.text(b.get_x() + b.get_width() / 2, v * 0.5, f"Δ_P = {on - v:.2f}",
                        ha="center", va="center", fontsize=7.5, color="white", rotation=90)
        ax.axhline(on, color=colours["on"], lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, top * 1.14)
        ax.set_title(f"{title}\n{pair}", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("graph score of the target paper", fontsize=8.5)
    fig.suptitle("Path-sender gate effect Δ_P (float32): graph score with all gates 1, with CCMP on, and after\n"
                 "resetting one path's sender gates to 1 with every other recorded gate fixed; rank above each bar", fontsize=9)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{path_stem}.{ext}", dpi=200)
    print("wrote", OUT / f"{path_stem}.pdf", "and .png")


if __name__ == "__main__":
    data = from_summary_files()
    print("source:", "downloaded summary.csv files" if data else "numbers pasted from the 13 Sep run")
    draw(data or PASTED, "fig_path_sender_effect")
