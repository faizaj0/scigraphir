"""
zeroshot_baselines_table.py -- Table 9.3's baseline rows, all six arms, from files that
already exist. No GPU, no re-encoding.

WHERE THE NUMBERS COME FROM. sir4_baselines_all.ipynb scored every baseline on every
SIR-4 field and wrote one file per arm under
    <Drive>/outputs/baselines/sir4_<field>/scores_<arm>.json
with `all` / `same` / `cross` slices. The cross slice of cs and matsci is exactly the
setting of Table 9.3 (held-out fields, cross-field queries only), so the three rows the
table lacks (SPECTER2-base, SciNCL, ReasonIR-8B) are a read, not a run.

TWO ROWS WILL NOT MATCH THE PRINTED TABLE. The BGE-large row matches to four decimals,
which shows the scoring is the same. The BM25 and Qwen3 rows differ slightly because the
thesis table took those two from a different runner (the transfer-matrix arms: a
different BM25 tokeniser, and Qwen3 under the handcrafted scorer's instruction rather than the
model card's). The file this script prints is one runner for all six; if the table is
regenerated from it, all six rows share one definition. Say which source was used.

Usage
-----
    python3 eval/zeroshot_baselines_table.py                       # repo copy of the scores
    python3 eval/zeroshot_baselines_table.py --scores-root "/Volumes/GoogleDrive/My Drive/cargo-gfmrag/outputs/baselines"
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_ROOT = f"{ROOT}/results/zeroshot_baselines"     # decoded from Drive on 2026-09-05

ARMS = [("BM25", "Sparse lexical", "None"),
        ("BGE-large", "Dense embedding", "External English retrieval pairs"),
        ("Qwen3-Embedding", "Dense embedding", "External multilingual relevance pairs"),
        ("SPECTER2-base", "Dense embedding", "Citation-graph contrastive (scientific)"),
        ("SciNCL", "Dense embedding", "Citation-neighbourhood contrastive (scientific)"),
        ("ReasonIR-8B", "Reasoning-trained dense", "Synthetic reasoning-intensive retrieval pairs")]
FIELDS = [("cs", "Computer Science"), ("matsci", "Materials Science")]
COLS = [("recall@3", "R@3"), ("recall@5", "R@5"), ("ndcg@5", "nDCG@5")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-root", default=DEFAULT_ROOT,
                    help="directory holding sir4_<field>/scores_<arm>.json")
    ap.add_argument("--slice", default="cross", choices=["cross", "same", "all"],
                    help="Table 9.3 is the cross slice")
    ap.add_argument("--md-out", default=None)
    a = ap.parse_args()

    lines = [f"| Method | Family | Training data | " + " | ".join(l for _, l in COLS) + " |",
             "|---|---|---|" + "--:|" * len(COLS)]
    for fld, title in FIELDS:
        lines.append(f"| **{title}** | | | | | |")
        for arm, fam, data in ARMS:
            p = f"{a.scores_root}/sir4_{fld}/scores_{arm}.json"
            if not os.path.exists(p):
                lines.append(f"| {arm} | {fam} | {data} | missing | missing | missing |")
                continue
            s = json.load(open(p))[a.slice]
            lines.append(f"| {arm} | {fam} | {data} | " + " | ".join(f"{s[k]:.4f}" for k, _ in COLS)
                         + " |")
        n = json.load(open(f"{a.scores_root}/sir4_{fld}/scores_BM25.json"))[a.slice]["n"] \
            if os.path.exists(f"{a.scores_root}/sir4_{fld}/scores_BM25.json") else "?"
        lines.append(f"| | | *n = {n} {a.slice}-field queries* | | | |")
    out = "\n".join(lines)
    print(out)
    if a.md_out:
        open(a.md_out, "w").write(out + "\n")
        print(f"\nwrote {a.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
