"""Report query-level retrieval performance by ResearchBench discipline.

The input files are the ``--per-query-out`` mappings written by
``eval/score_sir4.py``. Legacy files are losslessly normalised in memory because
they already store conventional MRR under ``mrr_best``: their old ``mrr`` becomes
``mgrr``, while ``mrr_best`` becomes ``mrr``. Rankings are never recomputed.

Example
-------
python3 eval/report_domain_results.py \
  --arm "BGE=/path/bge_perquery.json" \
  --arm "SIR-4 Biology=/path/sir4_biology_perquery.json" \
  --arm "TOMATO-Star=/path/tomato_perquery.json"
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


METRICS = (
    ("mrr", "MRR"),
    ("ndcg@5", "nDCG@5"),
    ("recall@5", "R@5"),
    ("completeset@5", "CGS@5"),
)


def parse_arm(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("arm must be LABEL=PATH")
    label, path = spec.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("arm must be LABEL=PATH")
    return label.strip(), path.strip()


def load_arm(path: str) -> dict[str, dict]:
    rows = json.load(open(path))
    if not isinstance(rows, dict) or not rows:
        raise SystemExit(f"{path}: expected a non-empty per-query mapping")
    legacy_mean, legacy_standard = 0, 0
    for qid, row in rows.items():
        if "mgrr" not in row and "mrr_best" in row:
            row["mgrr"] = row.get("mrr", 0.0)
            row["mrr"] = row["mrr_best"]
            legacy_mean += 1
        elif "mgrr" not in row and "mrr_best" not in row and "mrr" in row:
            # The earliest scorer emitted only conventional first-relevant MRR.
            # There is no separate all-gold quantity to preserve in this schema.
            legacy_standard += 1
        missing = [key for key, _ in METRICS if key not in row]
        if missing:
            raise SystemExit(f"{path}: query {qid!r} lacks {missing}")
        if "mrr" not in row:
            raise SystemExit(f"{path}: query {qid!r} has no MRR value")
        if "mrr_best" in row and abs(row["mrr"] - row["mrr_best"]) > 1e-12:
            raise SystemExit(f"{path}: query {qid!r} does not use standard MRR")
        if not row.get("discipline"):
            raise SystemExit(f"{path}: query {qid!r} has no ResearchBench discipline")
    if legacy_mean:
        print(f"{path}: normalised standard MRR from mrr_best for "
              f"{legacy_mean:,} legacy rows")
    if legacy_standard:
        print(f"{path}: using the existing standard mrr field for "
              f"{legacy_standard:,} early-schema rows")
    return rows


def aggregate(arms: list[tuple[str, dict[str, dict]]]) -> dict:
    reference_label, reference = arms[0]
    qids = set(reference)
    for label, rows in arms[1:]:
        if set(rows) != qids:
            raise SystemExit(
                f"{label} and {reference_label} cover different query sets: "
                f"{len(rows)} versus {len(reference)}"
            )
        mismatch = [qid for qid in qids
                    if rows[qid]["discipline"] != reference[qid]["discipline"]]
        if mismatch:
            raise SystemExit(f"{label}: discipline mismatch for {mismatch[0]!r}")

    by_domain: dict[str, list[str]] = defaultdict(list)
    for qid, row in reference.items():
        by_domain[row["discipline"]].append(qid)

    result = {}
    for domain in sorted(by_domain):
        ids = by_domain[domain]
        result[domain] = {"n": len(ids), "arms": {}}
        for label, rows in arms:
            result[domain]["arms"][label] = {
                key: sum(rows[qid][key] for qid in ids) / len(ids)
                for key, _ in METRICS
            }

        # SAME / CROSS WITHIN THE DISCIPLINE. score_sir4 records each query's
        # slice memberships, so this is a partition of the ids above, not a
        # rescore. Taken from the reference arm because every arm ranked the same
        # queries and the slices are a property of the query, not the ranking.
        for slc in ("same", "cross"):
            sub = [q for q in ids if slc in (reference[q].get("slices") or [])]
            if not sub:
                continue
            result[domain].setdefault("slices", {})[slc] = {
                "n": len(sub),
                "arms": {label: {key: sum(rows[q][key] for q in sub) / len(sub)
                                 for key, _ in METRICS}
                         for label, rows in arms},
            }
    return result


def markdown(result: dict, labels: list[str]) -> str:
    lines = [
        "### Performance by ResearchBench discipline",
        "",
        "| ResearchBench domain | n | Model | "
        + " | ".join(label for _, label in METRICS) + " |",
        "|---|--:|---|" + "--:|" * len(METRICS),
    ]
    for domain, block in result.items():
        for index, label in enumerate(labels):
            values = block["arms"][label]
            lines.append(
                f"| {domain if index == 0 else ''} | "
                f"{block['n'] if index == 0 else ''} | {label} | "
                + " | ".join(f"{values[key]:.4f}" for key, _ in METRICS)
                + " |"
            )
    lines += ["", "MRR is the reciprocal rank of the first relevant paper. "
              "CGS treats ResearchBench's accepted gold list as one complete set."]

    # Table 7.5's shape: same and cross paired under each metric, one block per
    # discipline. Built here rather than reshaped by hand, because transcribing
    # eight numbers per arm is where a transcription error would enter unnoticed.
    if any("slices" in block for block in result.values()):
        lines += ["", "### Same vs cross within each discipline", ""]
        header = " | ".join(f"{lab} Same | {lab} Cross" for _, lab in METRICS)
        lines += [f"| domain | Model | n same | n cross | {header} |",
                  "|---|---|--:|--:|" + "--:|" * (2 * len(METRICS))]
        for domain, block in result.items():
            sl = block.get("slices") or {}
            if not sl:
                continue
            ns = sl.get("same", {}).get("n", 0)
            nc = sl.get("cross", {}).get("n", 0)
            for index, label in enumerate(labels):
                cells = []
                for key, _ in METRICS:
                    for slc in ("same", "cross"):
                        v = sl.get(slc, {}).get("arms", {}).get(label, {}).get(key)
                        cells.append("--" if v is None else f"{v:.4f}")
                lines.append(
                    f"| {domain if index == 0 else ''} | {label} | "
                    f"{ns if index == 0 else ''} | {nc if index == 0 else ''} | "
                    + " | ".join(cells) + " |")
        lines += ["", "`same` and `cross` describe whether the gold inspiration "
                  "comes from the target's own field. Cross counts are small in "
                  "several disciplines; read those columns as indicative."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", type=parse_arm, required=True,
                        help="LABEL=PATH; repeat in the desired display order")
    parser.add_argument("--md-out")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    if len(args.arm) < 2:
        raise SystemExit("provide at least two --arm arguments")
    labels = [label for label, _ in args.arm]
    if len(labels) != len(set(labels)):
        raise SystemExit("arm labels must be unique")

    arms = [(label, load_arm(path)) for label, path in args.arm]
    result = aggregate(arms)
    table = markdown(result, labels)
    print(table)
    if args.md_out:
        open(args.md_out, "w").write(table + "\n")
        print(f"\nwrote {args.md_out}")
    if args.json_out:
        json.dump(result, open(args.json_out, "w"), indent=1)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
