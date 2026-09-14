#!/usr/bin/env python3
"""Build the fixed, stratified sample used for the SIR-4 LLM audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data.nosync"
SEED = "sir4-llm-audit-v1"
N_PER_STRATUM = 5
SOURCES = (
    ("cs", "train", "cs_train_final"),
    ("cs", "test", "cs_test_final"),
    ("biology", "train", "biology_train_low"),
    ("biology", "test", "biology_test_low"),
    ("physics", "train", "physics_train_low"),
    ("physics", "test", "physics_test_low"),
    ("matsci", "train", "matsci_train_low"),
    ("matsci", "test", "matsci_test_low"),
)


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def stable_rank(source_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{source_id}".encode()).hexdigest()


def normalise_doi(value: str | None) -> str:
    return (value or "").lower().replace("https://doi.org/", "").replace("/", "_")


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "llm_audit_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    packets = []

    for domain, split, stem in SOURCES:
        exported = {
            row["id"] for row in json.loads((ROOT / "benchmark" / stem / "eval.json").read_text())
        }
        records = {
            row["source_id"]: row
            for row in read_jsonl(ROOT / "04_resolved" / f"{stem}.jsonl")
            if row.get("source_id") in exported
        }
        fulltexts = list(read_jsonl(ROOT / "02_fulltext" / f"{domain}_{split}.jsonl"))
        by_doi = {normalise_doi(row.get("doi")): row for row in fulltexts}
        by_title = {row.get("title", "").casefold(): row for row in fulltexts}

        selected = sorted(records.values(), key=lambda row: stable_rank(row["source_id"]))[
            :N_PER_STRATUM
        ]
        for row in selected:
            source = by_doi.get(normalise_doi(row.get("doi"))) or by_title.get(
                row.get("title", "").casefold()
            )
            if source is None:
                raise RuntimeError(f"Missing full text for {row['source_id']}")
            packets.append(
                {
                    "audit_id": len(packets) + 1,
                    "seed": SEED,
                    "domain": domain,
                    "split": split,
                    "source_id": row["source_id"],
                    "title": row["title"],
                    "target_abstract": row.get("abstract"),
                    "target_fulltext": source.get("fulltext"),
                    "bibliography": source.get("bibliography"),
                    "research_question": row.get("research_question"),
                    "background": row.get("background_survey"),
                    "hypothesis": row.get("fine_grained_hypothesis"),
                    "sets": (row.get("uniqueness") or {}).get("M", []),
                    "automated_status": (row.get("uniqueness") or {}).get(
                        "status_after_resolution"
                    ),
                    "budget": (row.get("uniqueness") or {}).get("budget"),
                }
            )

    sample_path = output_dir / "sample.jsonl"
    with sample_path.open("w") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False) + "\n")

    manifest = {
        "seed": SEED,
        "sampling": "SHA-256 rank within field x split; first five exported targets",
        "n_per_stratum": N_PER_STRATUM,
        "n_total": len(packets),
        "fields": ["cs", "biology", "physics", "matsci"],
        "splits": ["train", "test"],
        "sample_file": sample_path.name,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
