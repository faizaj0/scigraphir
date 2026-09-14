"""
score_lattice_sir4.py -- turn a finished LATTICE run on SIR-4/<field> into rankings.json and score it.

Mirrors llm-guided-hierarchical-search/compute_r1_r5.py: load the saved eval samples for the
HP string, take each sample's get_top_predictions (leaf paths ordered by calibrated
relevance), map leaf ids d_NNNN -> DOI through id_map.json, and write
outputs/lattice/<field>/rankings.json for score_rankings_sir4.py. Also asserts that no gold
was dropped from the tree (run.py silently drops golds missing from the tree, which would
inflate LATTICE's own recall; our scorer uses the raw gold list so it is immune, but the
assertion tells you the tree lost documents).

Usage
-----
    python3 llm_baselines/score_lattice_sir4.py --field cs                    # beam 2, NumI 40 (table config)
    python3 llm_baselines/score_lattice_sir4.py --field cs --max-beam-size 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys

from sir4_llm_data import HERE, LATTICE_DIR, OUT_ROOT

sys.path.insert(0, f"{LATTICE_DIR}/src")


def run_args(field: str, num_iters: int, num_eval: int, beam: int) -> str:
    return (f"--dataset SIR-4 --subset {field} --tree_version top-down --traversal_prompt_version 5 "
            f"--reasoning_in_traversal_prompt -1 --num_leaf_calib 10 --pl_tau 5.0 --relevance_chain_factor 0.5 "
            f"--llm_api_backend openai --llm gpt-4o-mini --num_iters {num_iters} --num_eval_samples {num_eval} "
            f"--max_beam_size {beam}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--num-iters", type=int, default=40)
    ap.add_argument("--max-beam-size", type=int, default=2)
    ap.add_argument("--num-eval-samples", type=int, default=None, help="default: rows in examples.jsonl")
    a = ap.parse_args()

    from hyperparams import HyperParams
    from tree_objects import SemanticNode
    from utils import compute_node_registry, load_exp, setup_logger

    data = f"{LATTICE_DIR}/data/SIR-4/{a.field}"
    examples = [json.loads(l) for l in open(f"{data}/examples.jsonl")]
    id_map = json.load(open(f"{data}/id_map.json"))
    n_eval = a.num_eval_samples or len(examples)
    hp = HyperParams.from_args(args=run_args(a.field, a.num_iters, n_eval, a.max_beam_size))
    logger = setup_logger("score_lattice", f"/tmp/score_lattice_{a.field}.log", logging.WARNING)
    print(f"hp string: {hp}")

    tree_dict = pickle.load(open(f"{LATTICE_DIR}/trees/SIR-4/{a.field}/tree-top-down.pkl", "rb"))
    root = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
    reg = compute_node_registry(root)
    results_dir = f"{LATTICE_DIR}/results/SIR-4/{a.field}/"
    samples, dfs = load_exp(results_dir, hp, root, reg, logger)
    print(f"loaded {len(samples)} samples, {len(dfs)} iterations")
    assert len(samples) == n_eval, f"{len(samples)} samples in results, expected {n_eval}"

    rankings, dropped = {}, 0
    for i, s in enumerate(samples):
        ex = examples[i]
        dropped += len(ex["gold_ids"]) - len(s.gold_paths)
        # PredictionNode.path is a tuple of CHILD INDICES from the root (0, 3, 1, ...), not node
        # objects and not registry indices: resolve it by walking the tree. Predictions that stop
        # at an internal node (the leaf-relevance fn can still surface them) are skipped.
        def resolve(path):
            node = root
            for idx in path:
                ch = node.child or []
                if idx >= len(ch):
                    return None
                node = ch[idx]
            return node if not (node.child or []) else None
        ranked, seen = [], set()
        for x in s.get_top_predictions(rel_fn=s.get_rel_fn(leaf=True)):
            n = resolve(tuple(x[0].path))
            if n is None:
                continue
            key = str(n.id).split("] ", 1)[-1]
            if key in id_map and id_map[key] not in seen:
                seen.add(id_map[key]); ranked.append(id_map[key])
        rankings[ex["query_id"]] = ranked
    if dropped:
        print(f"WARNING: {dropped} golds were missing from the LATTICE tree (its own recall is inflated; ours is not)")
    out_dir = f"{OUT_ROOT}/lattice/{a.field}" + ("" if a.max_beam_size == 2 else f"_beam{a.max_beam_size}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rankings, open(f"{out_dir}/rankings.json", "w"))
    print(f"wrote {out_dir}/rankings.json ({len(rankings)} queries, "
          f"median ranked docs {sorted(len(v) for v in rankings.values())[len(rankings)//2]})")
    return subprocess.call([sys.executable, f"{HERE}/score_rankings_sir4.py", "--field", a.field,
                            "--rankings", f"{out_dir}/rankings.json", "--method",
                            f"LATTICE (gpt-4o-mini, NumI={a.num_iters}, beam {a.max_beam_size})", "--subset", "none"])


if __name__ == "__main__":
    raise SystemExit(main())
