#!/usr/bin/env python3
"""Align an existing corrected label file to ResearchBench's retrieval gold rows.

This is an offline repair for the first v2 run, which labelled the decomposition
file (1,369 records) rather than the retrieval split (1,367 queries).  Exact
keys are retained.  A differently formatted title is inherited only when its
candidate abstract matches an already labelled inspiration abstract from the
same target; otherwise it is conservatively marked unmatched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "rb_labels", HERE / "label_researchbench_domains.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb", type=Path, default=M.RB_DEFAULT)
    ap.add_argument("--out", type=Path, default=M.OUT_DEFAULT)
    args = ap.parse_args()

    labels_path = args.out / "domain_labels_v2.jsonl"
    report_path = args.out / "domain_labels_v2_report.json"
    if not labels_path.exists():
        raise SystemExit(f"missing {labels_path}")

    old = M.load_jsonl(labels_path)
    retrieval = M.load_jsonl(args.rb / "retrieve" / "retrieve.jsonl")
    papers = {p["sample_id"]: p for p in M.load_jsonl(args.rb / "papers" / "papers.jsonl")}
    exact = {(r["source_paper_id"], M.norm_title(r["inspiration_title"])): r for r in old}
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in old:
        by_query[row["source_paper_id"]].append(row)

    # Abstracts attached to the decomposition titles supply identity evidence
    # for harmless title variants in the retrieval candidate file.
    old_abstract: dict[tuple[str, str], str] = {}
    for sid, paper in papers.items():
        for inspiration in paper.get("inspirations") or []:
            old_abstract[(sid, M.norm_title(inspiration.get("title")))] = (
                inspiration.get("abstract") or ""
            )

    repaired: list[dict] = []
    verdicts_by_query: dict[str, list[str]] = defaultdict(list)
    inherited = unresolved = 0
    for query in retrieval:
        sid = query["sample_id"]
        named = {M.norm_title(t) for t in query.get("gold_titles") or []}
        golds: dict[str, dict] = {}
        for candidate in query.get("candidates") or []:
            key = M.norm_title(candidate.get("title"))
            if candidate.get("label") == "gold" or key in named:
                golds.setdefault(key, candidate)

        for key, candidate in golds.items():
            row = exact.get((sid, key))
            if row is not None:
                row = dict(row)
                row["alignment_basis"] = "exact_retrieval_title"
            else:
                candidate_abstract = candidate.get("abstract") or ""
                matches = []
                for prior in by_query.get(sid, []):
                    abstract = old_abstract.get(
                        (sid, M.norm_title(prior.get("inspiration_title"))), ""
                    )
                    score = (M.R.similarity(candidate_abstract, abstract)
                             if candidate_abstract and abstract else 0.0)
                    matches.append((score, prior))
                score, prior = max(matches, default=(0.0, None), key=lambda x: x[0])
                if prior is not None and score >= 0.8:
                    row = dict(prior)
                    row["inspiration_title"] = candidate.get("title") or ""
                    row["alignment_basis"] = "same_abstract_title_variant"
                    row["alignment_similarity"] = round(score, 3)
                    inherited += 1
                else:
                    # Preserve the target-side evidence from any row for this
                    # query, but make no claim about the unresolved inspiration.
                    prior = (by_query.get(sid) or [{}])[0]
                    row = {
                        "schema_version": M.SCHEMA_VERSION,
                        "source_paper_id": sid,
                        "source_discipline": query.get("discipline"),
                        "target_doi": query.get("doi"),
                        "target_openalex_id": prior.get("target_openalex_id"),
                        "target_title": prior.get("target_title"),
                        "target_domain": prior.get("target_domain"),
                        "target_primary_field": prior.get("target_primary_field"),
                        "target_subfield": prior.get("target_subfield"),
                        "target_topic_fields": prior.get("target_topic_fields") or {},
                        "inspiration_title": candidate.get("title") or "",
                        "inspiration_openalex_id": None,
                        "matched_title": None,
                        "title_similarity": None,
                        "identity_basis": "unmatched",
                        "inspiration_domain": None,
                        "inspiration_primary_field": None,
                        "inspiration_subfield": None,
                        "inspiration_topic_fields": {},
                        "shared_fields": [],
                        "n_shared_fields": 0,
                        "label_basis": None,
                        "label_policy": M.R.L.DEFAULT_POLICY,
                        "verdict": "unmatched",
                        "alignment_basis": "new_retrieval_gold_unresolved",
                    }
                    unresolved += 1
            repaired.append(row)
            verdicts_by_query[sid].append(row["verdict"])

    if len(repaired) != 3211 or len(verdicts_by_query) != 1367:
        raise SystemExit(
            f"refusing write: expected 3,211 rows/1,367 queries, got "
            f"{len(repaired):,}/{len(verdicts_by_query):,}"
        )
    keys = {(r["source_paper_id"], M.norm_title(r["inspiration_title"])) for r in repaired}
    if len(keys) != len(repaired):
        raise SystemExit("refusing write: repaired labels contain duplicate query/title keys")

    backup = args.out / "domain_labels_v2.pre_alignment.jsonl"
    if not backup.exists():
        shutil.copy2(labels_path, backup)
    M.atomic_jsonl(labels_path, repaired)

    old_report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report = {
        **old_report,
        "papers": len(verdicts_by_query),
        "inspiration_occurrences": len(repaired),
        "verdicts": dict(Counter(r["verdict"] for r in repaired)),
        "query_slices": dict(Counter(M.query_slice(v) for v in verdicts_by_query.values())),
        "alignment": {
            "exact_rows": len(repaired) - inherited - unresolved,
            "same_abstract_title_variants": inherited,
            "new_unresolved_rows": unresolved,
            "backup": str(backup),
        },
    }
    M.atomic_json(report_path, report)
    print("verdicts:", report["verdicts"])
    print("query slices:", report["query_slices"])
    print("alignment:", report["alignment"])
    print(f"wrote {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
