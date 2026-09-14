"""
Stage 0a - select the train subset for the single KG index.

Natural distribution: random N gold docs at a fixed seed (default 7000 @ seed 42),
then every train query whose gold is in that sample. Reproduces ~9,521 queries / 825 cross.

Reads  : outputs/caches/train_strata.jsonl   (per-query: source_id, step_idx, gold_key, stratum)
Writes : kg-construction/prepare/train_selection.json
No API key needed.
"""

import argparse
import json
import logging
import os
import random
from collections import Counter

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("select_train_subset")


def main() -> None:
    base = (os.environ.get("CARGO_ROOT") or os.path.expanduser("~/Desktop/CARGO"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", default=f"{base}/outputs/caches/train_strata.jsonl")
    ap.add_argument("--out", default=f"{base}/kg-construction/prepare/train_selection.json")
    ap.add_argument("--n_docs", type=int, default=7000, help="number of unique gold docs in the corpus")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log.info("reading %s", args.strata)
    rows = []
    with open(args.strata) as f:
        for line in tqdm(f, desc="train_strata"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info("loaded %d train queries", len(rows))

    all_golds = sorted({r["gold_key"] for r in rows})  # sorted -> deterministic before shuffle
    log.info("unique golds in train: %d", len(all_golds))

    rng = random.Random(args.seed)
    rng.shuffle(all_golds)
    chosen = set(all_golds[: args.n_docs])
    log.info("sampled %d golds @ seed %d", len(chosen), args.seed)

    queries = [r for r in rows if r["gold_key"] in chosen]
    strat = Counter(q["stratum"] for q in queries)
    cross_pct = 100 * strat.get("cross", 0) / max(1, len(queries))
    log.info("selected queries: %d | strata: %s | cross %.1f%%", len(queries), dict(strat), cross_pct)

    out = {
        "config": {"n_docs": args.n_docs, "seed": args.seed},
        "gold_keys": sorted(chosen),
        "queries": [
            {k: q[k] for k in ("source_id", "step_idx", "gold_key", "stratum")} for q in queries
        ],
        "stats": {
            "n_docs": len(chosen),
            "n_queries": len(queries),
            **{f"n_{k}": v for k, v in strat.items()},
            "cross_pct": round(cross_pct, 2),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    log.info("wrote selection -> %s", args.out)


if __name__ == "__main__":
    main()
