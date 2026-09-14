#!/usr/bin/env python3
"""Resolve ResearchBench gold papers and label same- versus cross-field cases.

The legacy labels compared a ResearchBench directory name with the first
OpenAlex title-search result.  That makes the result depend on the directory in
which a target was placed and silently labels bad title matches.  This script
instead applies the same evidence and policy used by SIR-4:

    same  iff the verified target and inspiration share an OpenAlex topic field
    cross iff their verified OpenAlex topic-field sets are disjoint

Both papers are resolved independently.  Targets use their DOI.  Inspirations
reuse a legacy OpenAlex identifier only when its returned title passes SIR-4's
strong title matcher; all other titles are searched again with SIR-4's resolver.
Missing or ambiguous evidence remains unlabelled.

The output is intentionally evidence-rich.  Every verdict stores both field
sets, their overlap, the identity route, and the label basis, so the policy can
be audited without another API call.

Example:
    python retriever/researchbench/label_researchbench_domains.py \
      --out retriever/data/researchbench_test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
CARGO = HERE.parents[2]
RB_DEFAULT = Path(os.environ.get("EXTERNAL_REPOS") or Path(__file__).resolve().parents[2] / "external") / "ResearchBench"
OUT_DEFAULT = CARGO / "retriever" / "data" / "researchbench_test"
SCHEMA_VERSION = 2
FIELD_SELECT = "id,doi,display_name,primary_topic,topics"


def load_resolver():
    """Load SIR-4's tested matcher and field-labelling policy."""
    path = CARGO / "quartet" / "build" / "04_resolve.py"
    spec = importlib.util.spec_from_file_location("sir4_resolve_for_rb", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SIR-4 resolver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load_resolver()

sys.path.insert(0, str(HERE))
from norm import norm_title  # noqa: E402  (local construction normaliser)


class NoSemanticScholar:
    """Keep this deterministic OpenAlex-only pass free of S2 rate-limit calls."""

    def match(self, _title):
        return None

    def by_id(self, _ident):
        return None


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def norm_doi(value) -> str:
    return str(value or "").strip().lower().removeprefix("https://doi.org/").removeprefix("doi:")


def work_id(value) -> str:
    return str(value or "").rstrip("/").rsplit("/", 1)[-1]


def topic_fields(work: dict | None) -> dict[str, int]:
    return dict(Counter(
        (topic.get("field") or {}).get("display_name")
        for topic in ((work or {}).get("topics") or [])
        if (topic.get("field") or {}).get("display_name")
    ))


def hierarchy(work: dict | None) -> tuple[str | None, str | None, str | None]:
    primary = (work or {}).get("primary_topic") or {}
    return (
        (primary.get("domain") or {}).get("display_name"),
        (primary.get("field") or {}).get("display_name"),
        (primary.get("subfield") or {}).get("display_name"),
    )


def query_slice(verdicts: list[str]) -> str:
    """Conservative query label used by result tables.

    One verified cross-field gold is enough to establish that a query contains
    cross-field inspiration.  A query is same-field only when *every* gold was
    resolved and labelled same.  Everything else is unlabelled.
    """
    if "cross" in verdicts:
        return "cross_involved"
    if verdicts and all(v == "same" for v in verdicts):
        return "same_only"
    return "unlabelled"


def fetch_by_openalex_ids(oa, ids: list[str], works: dict[str, dict],
                          cache_path: Path) -> dict[str, dict]:
    """Fetch field metadata in OR-filtered batches of 50, with resumption."""
    clean = sorted({work_id(x) for x in ids if work_id(x)})
    missing = [oid for oid in clean if oid not in works]
    for start in range(0, len(missing), oa.BATCH):
        chunk = missing[start:start + oa.BATCH]
        query = oa_params(oa, {
            "filter": "ids.openalex:" + "|".join(chunk),
            "per-page": len(chunk),
        })
        payload = oa.http.get(f"{R.OA}/works?{query}") or {}
        for work in payload.get("results") or []:
            oid = work_id(work.get("id"))
            if oid:
                works[oid] = work
        save_work_cache(cache_path, works)
    return {oid: works[oid] for oid in clean if oid in works}


def load_legacy(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for row in load_jsonl(path):
        key = R.norm(row.get("query_title") or row.get("title_norm"))
        if key:
            out[key] = row
    return out


def load_work_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
        if blob.get("schema_version") == SCHEMA_VERSION:
            return blob.get("works") or {}
    except (json.JSONDecodeError, AttributeError):
        pass
    return {}


def save_work_cache(path: Path, works: dict[str, dict]) -> None:
    atomic_json(path, {"schema_version": SCHEMA_VERSION, "works": works})


def oa_params(oa, extra: dict) -> str:
    params = {**extra, "select": FIELD_SELECT}
    if oa.mailto:
        params["mailto"] = oa.mailto
    if oa.key:
        params["api_key"] = oa.key
    return urllib.parse.urlencode(params)


def fetch_by_dois(oa, dois: list[str], works: dict[str, dict],
                  cache_path: Path) -> dict[str, dict]:
    by_doi = {
        norm_doi(work.get("doi")): work
        for work in works.values() if norm_doi(work.get("doi"))
    }
    missing = sorted(set(dois) - set(by_doi))
    for start in range(0, len(missing), oa.BATCH):
        chunk = missing[start:start + oa.BATCH]
        query = oa_params(oa, {
            "filter": "doi:" + "|".join(chunk),
            "per-page": len(chunk),
        })
        payload = oa.http.get(f"{R.OA}/works?{query}") or {}
        for work in payload.get("results") or []:
            oid = work_id(work.get("id"))
            if oid:
                works[oid] = work
            doi = norm_doi(work.get("doi"))
            if doi:
                by_doi[doi] = work
        save_work_cache(cache_path, works)
    return {doi: by_doi[doi] for doi in dois if doi in by_doi}


def broad_title_search(title: str, oa, cache: dict) -> dict:
    """Fallback to OpenAlex's general search, with strict local verification.

    ``title.search`` is precise but misses a substantial number of real papers.
    General search has higher recall, so its ranking is never trusted directly:
    every candidate must still clear SIR-4's asymmetric strong-title matcher.
    """
    key = "rb_broad::" + R.norm(title)
    if key in cache:
        return cache[key]
    params = {
        "search": title,
        "per-page": 8,
        "select": "id,doi,display_name,publication_year,type,cited_by_count",
    }
    if oa.mailto:
        params["mailto"] = oa.mailto
    if oa.key:
        params["api_key"] = oa.key
    payload = oa.http.get(f"{R.OA}/works?{urllib.parse.urlencode(params)}") or {}
    candidates = []
    for work in payload.get("results") or []:
        score = R.similarity(title, work.get("display_name") or "")
        if R.accepted(score):
            candidates.append((score, work))
    if not candidates:
        result = {"basis": "unmatched", "similarity": None,
                  "matched_title": None, "openalex_id": None}
    else:
        best_score = max(score for score, _ in candidates)
        best = [(score, work) for score, work in candidates
                if abs(score - best_score) < 1e-12]
        candidate_titles = {R.norm(work.get("display_name") or "") for _, work in best}
        if len(candidate_titles) > 1:
            result = {"basis": "ambiguous", "similarity": round(best_score, 3),
                      "matched_title": None, "openalex_id": None}
        else:
            # Identical-title records are usually preprint/publisher duplicates;
            # retain the best documented and most cited record.
            _, chosen = max(best, key=lambda item: (
                bool(item[1].get("doi")) and "arxiv" not in str(item[1].get("doi")).lower(),
                item[1].get("cited_by_count") or 0,
            ))
            result = {
                "basis": "broad_title_verified",
                "similarity": round(best_score, 3),
                "matched_title": chosen.get("display_name"),
                "openalex_id": chosen.get("id"),
            }
    cache[key] = result
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb", type=Path, default=RB_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--legacy-cache", type=Path, default=None,
                    help="old title cache used only for strongly verified OpenAlex IDs")
    ap.add_argument("--delay", type=float, default=0.12)
    ap.add_argument("--limit", type=int, default=None,
                    help="paper limit for a smoke test")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    papers_path = args.rb / "papers" / "papers.jsonl"
    retrieve_path = args.rb / "retrieve" / "retrieve.jsonl"
    if not papers_path.exists() or not retrieve_path.exists():
        raise SystemExit(f"missing {papers_path} or {retrieve_path}")
    papers = load_jsonl(papers_path)
    retrieval = load_jsonl(retrieve_path)
    if args.limit:
        retrieval = retrieval[:args.limit]
    papers_by_id = {paper["sample_id"]: paper for paper in papers}

    legacy_path = args.legacy_cache or args.rb / "scripts" / "openalex_cache.jsonl"
    legacy = load_legacy(legacy_path)
    http = R.Http(args.out / "openalex_verified_title_cache.json", args.delay)
    oa = R.OpenAlex(http)
    no_s2 = NoSemanticScholar()
    work_cache_path = args.out / "openalex_field_work_cache.json"
    work_cache = load_work_cache(work_cache_path)

    # One target record per DOI.  This removes the contradiction caused by the
    # same DOI appearing under several ResearchBench discipline directories.
    target_dois = sorted({norm_doi(q.get("doi")) for q in retrieval if norm_doi(q.get("doi"))})
    targets = fetch_by_dois(oa, target_dois, work_cache, work_cache_path)

    occurrences: list[tuple[dict, dict]] = []
    inspiration_by_key: dict[str, dict] = {}
    # Use the exact retrieval gold rows, not the larger decomposition file.  It
    # is this set that must align positionally with raw/test.json.
    for query in retrieval:
        paper = papers_by_id.get(query["sample_id"])
        if paper is None:
            raise SystemExit(f"no paper metadata for retrieval query {query['sample_id']}")
        named_golds = {norm_title(title) for title in query.get("gold_titles") or []}
        golds: dict[str, dict] = {}
        for candidate in query.get("candidates") or []:
            title = candidate.get("title") or ""
            canonical = norm_title(title)
            if candidate.get("label") == "gold" or canonical in named_golds:
                golds.setdefault(canonical, {
                    "title": title,
                    "abstract": candidate.get("abstract") or "",
                })
        for inspiration in golds.values():
            title = inspiration["title"]
            key = R.norm(title)
            if not key:
                continue
            occurrences.append((paper, inspiration))
            inspiration_by_key.setdefault(key, inspiration)

    # Reuse only legacy identities whose returned title clears the tested
    # strong-match threshold.  The old field verdict itself is never reused.
    candidate_ids: dict[str, str] = {}
    identity: dict[str, dict] = {}
    for key, inspiration in inspiration_by_key.items():
        old = legacy.get(key) or {}
        score = R.similarity(inspiration.get("title") or "", old.get("matched_title") or "")
        if old.get("matched") and old.get("openalex_id") and R.accepted(score):
            candidate_ids[key] = work_id(old["openalex_id"])
            identity[key] = {
                "basis": "legacy_id_title_verified",
                "similarity": round(score, 3),
                "matched_title": old.get("matched_title"),
            }

    works = fetch_by_openalex_ids(
        oa, list(candidate_ids.values()), work_cache, work_cache_path
    )

    # A reused ID must still agree with the current full OpenAlex record.  Any
    # missing or changed record goes through a fresh title search.
    resolved: dict[str, dict] = {}
    pending: list[str] = []
    for key, inspiration in inspiration_by_key.items():
        work = works.get(candidate_ids.get(key, ""))
        score = R.similarity(inspiration.get("title") or "", (work or {}).get("display_name") or "")
        if work and R.accepted(score):
            resolved[key] = work
            identity[key]["similarity"] = round(score, 3)
            identity[key]["matched_title"] = work.get("display_name")
        else:
            pending.append(key)

    print(f"targets: {len(targets):,}/{len(target_dois):,} resolved by DOI")
    print(f"inspirations: {len(resolved):,}/{len(inspiration_by_key):,} reused after title verification")
    print(f"fresh title searches required: {len(pending):,}")

    searched_ids: dict[str, str] = {}
    for index, key in enumerate(pending, 1):
        inspiration = inspiration_by_key[key]
        result = R.resolve_title(
            inspiration.get("title") or "", oa, no_s2, http.cache
        )
        ident = result.get("identity") or {}
        # If several distinct works share the exact title and the title alone
        # cannot choose between them, keep the field label unassigned.
        ambiguous = ident.get("basis") == "title_only" and ident.get("n_title_matches", 0) > 1
        if result.get("match_quality") != "none" and result.get("openalex_id") and not ambiguous:
            searched_ids[key] = work_id(result["openalex_id"])
            identity[key] = {
                "basis": ident.get("basis") or result.get("route"),
                "similarity": result.get("similarity"),
                "matched_title": result.get("found_title"),
            }
        else:
            fallback = ({"basis": "ambiguous", "similarity": result.get("similarity"),
                         "matched_title": result.get("found_title"), "openalex_id": None}
                        if ambiguous else broad_title_search(
                            inspiration.get("title") or "", oa, http.cache
                        ))
            identity[key] = {
                "basis": fallback.get("basis"),
                "similarity": fallback.get("similarity"),
                "matched_title": fallback.get("matched_title"),
            }
            if fallback.get("openalex_id"):
                searched_ids[key] = work_id(fallback["openalex_id"])
        if index % 25 == 0:
            http.save()
            print(f"  searched {index:,}/{len(pending):,}")

    fresh = fetch_by_openalex_ids(
        oa, list(searched_ids.values()), work_cache, work_cache_path
    )
    works.update(fresh)
    for key, oid in searched_ids.items():
        work = works.get(oid)
        title = inspiration_by_key[key].get("title") or ""
        if work and R.accepted(R.similarity(title, work.get("display_name") or "")):
            resolved[key] = work
        else:
            identity[key] = {"basis": "unmatched", "similarity": None,
                             "matched_title": None}

    output: list[dict] = []
    by_query: dict[str, list[str]] = defaultdict(list)
    for paper, inspiration in occurrences:
        title = inspiration.get("title") or ""
        key = R.norm(title)
        doi = norm_doi(paper.get("doi"))
        target = targets.get(doi)
        source = resolved.get(key)
        id_info = identity.get(key) or {"basis": "unmatched"}

        if source is None:
            verdict = "ambiguous" if id_info.get("basis") == "ambiguous" else "unmatched"
            ev = R.L.label_evidence(None, None, None, None)
            applied = R.L.apply_policy(ev, R.L.DEFAULT_POLICY)
        else:
            td, tf, ts = hierarchy(target)
            sd, sf, ss = hierarchy(source)
            ev = R.L.label_evidence(
                topic_fields(target), topic_fields(source), tf, sf, td, sd
            )
            applied = R.L.apply_policy(ev, R.L.DEFAULT_POLICY)
            verdict = applied["domain_relation"] or "unknown"

        td, tf, ts = hierarchy(target)
        sd, sf, ss = hierarchy(source)
        row = {
            "schema_version": SCHEMA_VERSION,
            "source_paper_id": paper.get("sample_id"),
            "source_discipline": paper.get("discipline"),
            "target_doi": doi or None,
            "target_openalex_id": (target or {}).get("id"),
            "target_title": paper.get("title"),
            "target_domain": td,
            "target_primary_field": tf,
            "target_subfield": ts,
            "target_topic_fields": topic_fields(target),
            "inspiration_title": title,
            "inspiration_openalex_id": (source or {}).get("id"),
            "matched_title": (source or {}).get("display_name") or id_info.get("matched_title"),
            "title_similarity": id_info.get("similarity"),
            "identity_basis": id_info.get("basis"),
            "inspiration_domain": sd,
            "inspiration_primary_field": sf,
            "inspiration_subfield": ss,
            "inspiration_topic_fields": topic_fields(source),
            "shared_fields": ev.get("shared_fields") or [],
            "n_shared_fields": ev.get("n_shared", 0),
            "label_basis": applied.get("label_basis"),
            "label_policy": applied.get("label_policy"),
            "verdict": verdict,
        }
        output.append(row)
        by_query[paper.get("sample_id")].append(verdict)

    labels_path = args.out / "domain_labels_v2.jsonl"
    atomic_jsonl(labels_path, output)
    http.save()

    verdict_counts = Counter(row["verdict"] for row in output)
    slices = Counter(query_slice(v) for v in by_query.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": R.L.DEFAULT_POLICY,
        "definition": "same iff verified target and inspiration share an OpenAlex topic field",
        "papers": len(retrieval),
        "inspiration_occurrences": len(output),
        "unique_inspiration_titles": len(inspiration_by_key),
        "verdicts": dict(verdict_counts),
        "query_slices": dict(slices),
        "openalex_calls": http.calls,
        "openalex_rate_limits": http.rate_limited,
        "labels_path": str(labels_path),
    }
    atomic_json(args.out / "domain_labels_v2_report.json", report)

    print("\nverdicts:", dict(verdict_counts))
    print("query slices:", dict(slices))
    print(f"OpenAlex calls: {http.calls:,}; rate limits: {http.rate_limited:,}")
    print(f"wrote {labels_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R.BudgetExhausted as exc:
        message = str(exc)
        retry = ""
        try:
            retry_after = json.loads(message).get("retryAfter")
            if retry_after:
                hours = float(retry_after) / 3600
                retry = f" (approximately {hours:.1f} hours)"
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        print(
            "\nOpenAlex did not run the relabelling because its API budget is "
            "exhausted.\n"
            f"The anonymous allowance should reset after the reported delay{retry}.\n"
            "Either rerun after the reset or set OPENALEX_API_KEY to an OpenAlex "
            "key with available credit.\n"
            "No partial label file was installed; cached completed batches will "
            "be reused on the next run.",
            file=sys.stderr,
        )
        raise SystemExit(3)
