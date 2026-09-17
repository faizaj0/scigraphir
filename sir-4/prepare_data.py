#!/usr/bin/env python3
"""Verify and unpack the SIR-4 dataset included with the repository.

Run from the repository root:
    python sir-4/prepare_data.py
    python sir-4/prepare_data.py --domain physics --verify-only

Uses only the Python standard library. Existing files with different content
are never overwritten. Runtime JSON files are written to sir-4/data/benchmark/.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
OUTPUT = HERE / "data" / "benchmark"
DOMAINS = ("cs", "biology", "physics", "matsci")
CHUNK = 1024 * 1024


def digest_stream(stream, sink=None):
    size, digest = 0, hashlib.sha256()
    while chunk := stream.read(CHUNK):
        size += len(chunk)
        digest.update(chunk)
        if sink is not None:
            sink.write(chunk)
    return size, digest.hexdigest()


def digest_file(path):
    with path.open("rb") as source:
        return digest_stream(source)


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path leaves its root: {relative}")
    return path


def prepare(domain="all", *, dataset=DATASET, output=OUTPUT, verify_only=False):
    dataset, output = Path(dataset), Path(output)
    manifest = json.loads((dataset / "MANIFEST.json").read_text())
    splits = [s for s in manifest["splits"] if domain == "all" or s["domain"] == domain]
    if not splits:
        raise ValueError(f"No dataset splits for {domain!r}")
    exports = {s["export"] for s in splits}
    files = [f for f in manifest["files"] if f["export"] in exports]

    # Check the complete selection before creating any runtime files.
    for entry in files:
        asset = checked_path(dataset, entry["asset"])
        if digest_file(asset) != (entry["asset_bytes"], entry["asset_sha256"]):
            raise ValueError(f"Dataset asset failed its checksum: {asset}")
        destination = checked_path(output, f'{entry["export"]}/{entry["path"]}')
        if not verify_only and destination.exists():
            if digest_file(destination) != (entry["bytes"], entry["sha256"]):
                raise FileExistsError(
                    f"Existing export differs; left unchanged: {destination}. "
                    "Choose an empty --out directory to unpack another copy."
                )

    for entry in files:
        asset = checked_path(dataset, entry["asset"])
        destination = checked_path(output, f'{entry["export"]}/{entry["path"]}')
        opener = gzip.open if entry["compression"] == "gzip" else open
        if entry["compression"] not in ("gzip", "none"):
            raise ValueError(f'Unknown compression: {entry["compression"]}')
        if verify_only or destination.exists():
            with opener(asset, "rb") as source:
                result = digest_stream(source)
            if result != (entry["bytes"], entry["sha256"]):
                raise ValueError(f"Unpacked content failed its checksum: {asset}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as sink:
                temp = Path(sink.name)
                with opener(asset, "rb") as source:
                    result = digest_stream(source, sink)
            if result != (entry["bytes"], entry["sha256"]):
                raise ValueError(f"Unpacked content failed its checksum: {asset}")
            # Exclusive creation preserves files introduced after the preflight check.
            os.link(temp, destination)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    action = "Verified" if verify_only else "Prepared"
    for split in splits:
        print(f'{action} {split["domain"]}/{split["split"]}: '
              f'{split["queries"]:,} queries, {split["documents"]:,} documents')
        if not split["matches_thesis_counts"]:
            print(f'  Version note: thesis reports {split["thesis_queries"]:,} queries '
                  f'and {split["thesis_documents"]:,} documents; this is the available export.')
    if not verify_only:
        print(f"Dataset ready at {output}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=("all", *DOMAINS), default="all")
    parser.add_argument("--out", type=Path, default=OUTPUT, help="destination for runtime JSON exports")
    parser.add_argument("--verify-only", action="store_true", help="verify packaged bytes and contents without unpacking")
    args = parser.parse_args()
    try:
        prepare(args.domain, output=args.out, verify_only=args.verify_only)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Dataset preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
