#!/usr/bin/env python3
"""Offline regression checks for corrected ResearchBench field labels."""
from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "rb_domain_labels", HERE / "label_researchbench_domains.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def label(target_fields, inspiration_fields):
    target_primary = next(iter(target_fields), None)
    inspiration_primary = next(iter(inspiration_fields), None)
    ev = M.R.L.label_evidence(
        {x: 1 for x in target_fields}, {x: 1 for x in inspiration_fields},
        target_primary, inspiration_primary,
    )
    return M.R.L.apply_policy(ev)["domain_relation"]


def main() -> int:
    assert label(["Computer Science", "Mathematics"], ["Mathematics"]) == "same"
    assert label(["Physics and Astronomy"], ["Biochemistry, Genetics and Molecular Biology"]) == "cross"
    assert label([], ["Mathematics"]) is None

    assert M.query_slice(["same", "same"]) == "same_only"
    assert M.query_slice(["same", "cross"]) == "cross_involved"
    assert M.query_slice(["cross", "unknown"]) == "cross_involved"
    assert M.query_slice(["same", "unknown"]) == "unlabelled"
    assert M.query_slice(["unmatched"]) == "unlabelled"

    wrong = M.R.similarity(
        "Summary of the HIPAA Privacy Rule",
        "Standards and guidelines for the interpretation of sequence variants: "
        "a joint consensus recommendation of the American College of Medical "
        "Genetics and Genomics and the Association for Molecular Pathology",
    )
    assert not M.R.accepted(wrong)
    print("all corrected-domain-label checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
