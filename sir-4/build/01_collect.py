"""
Stage 1 -- paper collection and temporal split.

Builds the eligible paper pool for each SIR-4 domain from OpenAlex, applies the
construction filters, assigns each paper to train or test by publication date, and
writes one JSONL per (domain, split) plus a report of exactly what was dropped and why.

This stage does NOT sample down to the 6,000/1,000 targets. Balancing the benchmark to
50/50 same/cross can only happen after extraction, because a paper's inspirations --
and therefore their domains -- are unknown until it has been decomposed and resolved.
So we collect the full eligible pool and let a later stage subsample it.

Eligibility, in order applied:

  1. primary_topic.field is the domain's field, exactly. No unions.
  2. type:article, and the work carries a structured-full-text record: an arXiv
     location (CS, Physics, Materials Science) or a PubMed Central one (Biology).
     This is what removes MinerU from the pipeline entirely.
  3. purity >= 0.66, i.e. at least 2 of the work's 3 topics sit in the primary field.
     See config/domains.yaml for why this replaces a topic-score margin.
  4. Not dated exactly 1 January -- OpenAlex's dump for works with imprecise dates.
  5. Has a reconstructable abstract and a DOI.

Usage:
  python build/01_collect.py --domain cs --split train
  python build/01_collect.py --all
  python build/01_collect.py --domain cs --split test --limit 200   # pilot
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "domains.yaml"
OUT = ROOT / "data" / "01_collected"

OPENALEX = "https://api.openalex.org/works"
MAILTO = os.environ.get("OPENALEX_MAILTO", "faizajalil0@gmail.com")
UA = f"scigraphir-sir4/0.1 (mailto:{MAILTO})"

# Since 2026-02-13 OpenAlex requires an API key, and prepaid credit is attached to the
# key rather than the account. Requests sent with only `mailto` draw on the anonymous
# free allowance ($1/day) no matter how much credit the account holds, which is how we
# ran dry while topped up. Get a key at https://openalex.org/settings/api
API_KEY = os.environ.get("OPENALEX_API_KEY", "")

# Keep the response small. abstract_inverted_index is the bulk of it but we need it.
SELECT = ",".join([
    "id", "doi", "title", "publication_date", "type",
    "primary_topic", "topics", "locations", "best_oa_location",
    "abstract_inverted_index", "cited_by_count", "referenced_works_count",
])


# --------------------------------------------------------------------------- helpers
def load_config() -> dict:
    """Load the config, stringifying dates.

    yaml.safe_load turns bare 2025-10-01 into a datetime.date, which is fine in an
    f-string but not JSON-serialisable when it reaches the report.
    """
    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    for split in cfg["split"].values():
        if isinstance(split, dict):
            for k, v in split.items():
                split[k] = str(v)
    return cfg


def invert_abstract(index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}. Rebuild the text."""
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, spots in index.items():
        for p in spots:
            positions.append((p, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def purity(work: dict) -> tuple[float, str | None, dict]:
    """Fraction of the work's topics sitting in its primary field.

    Returns (purity, primary_field_name, {field: count}). With OpenAlex's 3 topics
    this is 1/3, 2/3 or 1. Score margins are useless here -- topic scores are
    unnormalised and often all sit above 0.99.
    """
    topics = work.get("topics") or []
    if not topics:
        return 0.0, None, {}
    primary = ((work.get("primary_topic") or {}).get("field") or {}).get("display_name")
    counts = collections.Counter(
        (t.get("field") or {}).get("display_name") for t in topics
    )
    if not primary:
        return 0.0, None, dict(counts)
    return counts[primary] / len(topics), primary, dict(counts)


def fulltext_id(work: dict, source_id: str) -> str | None:
    """The arXiv or PMC identifier, from whichever location matches the source."""
    want = source_id.rsplit("/", 1)[-1]
    for loc in work.get("locations") or []:
        src = (loc.get("source") or {})
        sid = (src.get("id") or "").rsplit("/", 1)[-1]
        if sid == want:
            return loc.get("landing_page_url") or loc.get("pdf_url") or src.get("display_name")
    return None


def paginate(params: dict, cap: int, session: requests.Session, state: dict):
    """Cursor-paginate OpenAlex, yielding works. OpenAlex caps per-page at 200.

    An empty page or a null next_cursor is NOT proof the result set is exhausted: under
    rate limiting OpenAlex can return either while thousands of records remain. Running
    five collectors concurrently truncated four of them silently. So we record
    meta.count from the first page, retry empty pages, and hand the caller enough state
    to assert completeness rather than assume it.
    """
    cursor, seen, empty_retries = "*", 0, 0
    while cursor and seen < cap:
        params = {**params, "per-page": 200, "cursor": cursor}
        r = None
        for attempt in range(6):
            try:
                r = session.get(OPENALEX, params=params, headers={"User-Agent": UA}, timeout=90)
                if r.status_code == 429:
                    # Two different things return 429. A throttle is worth retrying;
                    # an exhausted daily budget is not, and retrying just burns time.
                    body = {}
                    try:
                        body = r.json()
                    except ValueError:
                        pass
                    if "budget" in str(body.get("message", "")).lower():
                        raise RuntimeError(
                            "OpenAlex daily budget exhausted: "
                            f"{body.get('message', '')} "
                            f"(resets in {body.get('retryAfter', '?')}s). "
                            "Wait for the reset or add funds at https://openalex.org/pricing."
                        )
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code in (500, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 5:
                    raise
                print(f"    retry {attempt + 1} after {exc.__class__.__name__}", file=sys.stderr)
                time.sleep(2 ** attempt)
        if r is None or r.status_code != 200:
            detail = ""
            try:
                detail = f" -- {r.json().get('message', '')}" if r is not None else ""
            except ValueError:
                detail = f" -- {r.text[:200]}" if r is not None else ""
            raise RuntimeError(
                f"OpenAlex refused after retries: "
                f"{r.status_code if r is not None else 'no response'}{detail}")

        # Stay inside OpenAlex's polite-pool rate. Concurrent collectors exceeded it.
        time.sleep(0.15)

        payload = r.json()
        if state.get("total") is None:
            state["total"] = (payload.get("meta") or {}).get("count", 0)

        results = payload.get("results") or []
        if not results:
            # Could be genuine exhaustion or a rate-limited blank. Retry before believing it.
            if empty_retries < 3 and seen < state["total"]:
                empty_retries += 1
                print(f"    empty page at {seen:,}/{state['total']:,}, retrying "
                      f"({empty_retries}/3)", file=sys.stderr)
                time.sleep(5 * empty_retries)
                continue
            break
        empty_retries = 0

        for w in results:
            yield w
            seen += 1
            if seen >= cap:
                state["seen"] = seen
                return
        cursor = (payload.get("meta") or {}).get("next_cursor")
    state["seen"] = seen


# --------------------------------------------------------------------------- collect
def collect(domain: dict, split_name: str, cfg: dict, limit: int | None) -> dict:
    split = cfg["split"][split_name]
    src_id = domain["fulltext"]["source_id"]
    cap = limit or cfg["targets"]["pool_cap_per_domain_split"]
    min_purity = cfg["domain_assignment"]["min_purity"]
    drop_jan1 = cfg["split"]["drop_january_first"]

    params = {
        "filter": ",".join([
            f"primary_topic.field.id:fields/{domain['openalex_field_id']}",
            f"locations.source.id:{src_id.rsplit('/', 1)[-1]}",
            f"from_publication_date:{split['start']}",
            f"to_publication_date:{split['end']}",
            "type:article",
        ]),
        "select": SELECT,
        "mailto": MAILTO,
    }
    if API_KEY:
        params["api_key"] = API_KEY

    drops = collections.Counter()
    purity_hist = collections.Counter()
    session = requests.Session()
    state: dict = {"total": None, "seen": 0}

    # Stream to disk rather than accumulating. At the 60,000 cap, holding rows in
    # memory is roughly half a gigabyte per domain, and a crash loses all of it.
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{domain['id']}_{split_name}.jsonl"
    # Write to a sibling temp file and swap in only on success. Opening `path` in "w"
    # truncated it before the first request, so a run that died on page 1 -- an
    # exhausted OpenAlex budget, a dropped connection -- destroyed a pool that had
    # taken hours to collect. A failed run must now leave the previous file untouched.
    tmp = path.with_suffix(".jsonl.partial")
    fh = open(tmp, "w")

    print(f"[{domain['id']}/{split_name}] querying OpenAlex "
          f"({split['start']} to {split['end']}, cap {cap:,}) ...")

    # Total is unknown until the first page returns meta.count, so the bar starts
    # untotalled and gets its total set once.
    bar = tqdm(desc=f"{domain['id']}/{split_name}", unit="work", dynamic_ncols=True)
    for w in paginate(params, cap, session, state):
        drops["seen"] += 1
        if bar.total is None and state.get("total"):
            bar.total = min(state["total"], cap)
            bar.refresh()
        bar.update(1)

        pur, primary, field_counts = purity(w)
        purity_hist[round(pur, 2)] += 1

        if primary != domain["openalex_field"]:
            drops["wrong_primary_field"] += 1
            continue
        if pur < min_purity:
            drops["low_purity"] += 1
            continue

        date = w.get("publication_date") or ""
        if drop_jan1 and date.endswith("-01-01"):
            drops["january_first"] += 1
            continue

        doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
        if not doi:
            drops["no_doi"] += 1
            continue

        abstract = invert_abstract(w.get("abstract_inverted_index"))
        if len(abstract) < 200:
            drops["no_abstract"] += 1
            continue

        ft = fulltext_id(w, src_id)
        if not ft:
            drops["no_fulltext_location"] += 1
            continue

        drops["kept"] += 1
        fh.write(json.dumps({
            "doi": doi,
            "openalex_id": w["id"],
            "title": (w.get("title") or "").strip(),
            "abstract": abstract,
            "publication_date": date,
            "domain": domain["id"],
            "split": split_name,
            "primary_field": primary,
            "purity": round(pur, 3),
            # A work with a single topic scores purity 1.0 on thin evidence. Recorded
            # so the filter can be tightened later without recollecting.
            "n_topics": len(w.get("topics") or []),
            "topic_fields": field_counts,
            "fulltext_route": domain["fulltext"]["primary"],
            "fulltext_url": ft,
            "referenced_works_count": w.get("referenced_works_count", 0),
            "cited_by_count": w.get("cited_by_count", 0),
        }) + "\n")

        if drops["kept"] % 500 == 0:
            fh.flush()
            bar.set_postfix(kept=drops["kept"], purity_drop=drops["low_purity"])

    bar.close()
    fh.close()
    # Only now, with the walk finished, replace the previous pool. If the loop above
    # raised, we never reach here and the old file survives; the .partial left behind
    # is overwritten from scratch by the next run, so it cannot contaminate anything.
    # The old pool is kept as .prev: a run can finish "successfully" having enumerated
    # far less than last time (rate limiting truncates pagination without erroring),
    # and a re-collection should never be the only copy.
    if path.exists() and path.stat().st_size > 0:
        path.replace(path.with_suffix(".jsonl.prev"))
    tmp.replace(path)

    target = cfg["targets"]["per_domain"][f"{split_name}_source_papers"]
    headroom = drops["kept"] / target if target else float("inf")
    total = state["total"] or 0
    capped = drops["seen"] >= cap
    # Completeness: did we actually walk the whole result set, or stop early?
    complete = capped or drops["seen"] >= total
    report = {
        "domain": domain["id"],
        "split": split_name,
        "window": [split["start"], split["end"]],
        "target": target,
        "openalex_total": total,
        "enumerated": drops["seen"],
        "kept": drops["kept"],
        "complete": complete,
        "capped": capped,
        "headroom_vs_target": round(headroom, 2),
        "drops": dict(drops),
        "purity_histogram": {str(k): v for k, v in sorted(purity_hist.items())},
        "path": str(path),
    }

    print(f"[{domain['id']}/{split_name}] kept {drops['kept']:,} of {drops['seen']:,} enumerated"
          f" (OpenAlex reports {total:,})   target {target:,}   headroom {headroom:.1f}x")
    for k, v in drops.most_common():
        if k not in ("seen", "kept"):
            print(f"    dropped {k:<24} {v:,}")
    if not complete:
        print(f"    !! INCOMPLETE: enumerated {drops['seen']:,} of {total:,} available "
              f"({100*drops['seen']/max(total,1):.0f}%). Pagination stopped early, most "
              f"likely rate limiting. Re-run this domain ALONE, not alongside others.")
    if headroom < 2.0 and complete and not capped:
        print(f"    !! headroom {headroom:.1f}x is below 2x. The 50/50 balance needs "
              f"1/(2c) over-collection; at a natural cross rate under 25% this pool "
              f"will not be large enough.")
    return report


# --------------------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    by_id = {d["id"]: d for d in cfg["domains"]}

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(by_id), help="single domain")
    ap.add_argument("--split", choices=["train", "test"], help="single split")
    ap.add_argument("--all", action="store_true", help="every domain and split")
    ap.add_argument("--limit", type=int, help="cap results; use for a pilot run")
    a = ap.parse_args()

    if not API_KEY:
        print("!! OPENALEX_API_KEY is not set. Requests will use the anonymous free\n"
              "   allowance ($1/day) and any prepaid credit on your account will be\n"
              "   ignored, because credit is attached to the key. Get one at\n"
              "   https://openalex.org/settings/api then:\n"
              "     export OPENALEX_API_KEY=...\n", file=sys.stderr)

    if a.all:
        jobs = [(d, s) for d in cfg["domains"] for s in ("train", "test")]
    elif a.domain:
        splits = [a.split] if a.split else ["train", "test"]
        jobs = [(by_id[a.domain], s) for s in splits]
    else:
        ap.error("pass --domain or --all")

    reports = [collect(d, s, cfg, a.limit) for d, s in jobs]

    # Merge into any existing report rather than replacing it. Runs happen one domain
    # at a time (concurrent collectors exceed the rate limit), so overwriting left the
    # report describing only the last run and hid whether the others completed.
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "collection_report.json"
    merged: dict[tuple[str, str], dict] = {}
    if report_path.exists():
        try:
            for r in json.load(open(report_path)):
                merged[(r["domain"], r["split"])] = r
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    for r in reports:
        merged[(r["domain"], r["split"])] = r
    with open(report_path, "w") as fh:
        json.dump([merged[k] for k in sorted(merged)], fh, indent=1)

    print(f"\n{'domain':<10} {'split':<6} {'kept':>9} {'target':>8} {'headroom':>9}")
    for r in reports:
        print(f"{r['domain']:<10} {r['split']:<6} {r['kept']:>9,} "
              f"{r['target']:>8,} {r['headroom_vs_target']:>8.1f}x")
    print(f"\nwrote {OUT}/collection_report.json")


if __name__ == "__main__":
    main()
