"""
Stage 2b -- take the reference list from OpenAlex instead of parsing it out of LaTeX.

Stage 2 recovers references by parsing the paper's own bibliography. That works, but it
inherits every quirk of every LaTeX bibliography package, and it cannot invent what the
source omits. Two failures survive it:

  - 4.7% of references carry no title at all, because physics, astronomy and much of
    materials science cite as "PRX Quantum 2, 010343 (2021)". On 26% of matsci papers
    that is more than half the bibliography.
  - 3.1% are full-name author runs that no regex can separate from a title, because
    "Aharon Ben-Tal and Arkadi Nemirovski" and "Distinguishing Separable and Entangled
    States" are both capitalised words joined by "and".

Neither is recoverable downstream. Semantic Scholar and Crossref both answer EVERY
bibliographic query with their best guess and no confidence signal, so a titleless string
does not fail at Stage 4 -- it silently resolves to the wrong paper. Measured:
"Physical Review Letters 103, 210501" comes back as volume 102, article 129901.

OpenAlex holds the same reference lists as structured records. On the three worst physics
papers here, where 75-79% of the parsed bibliography was unusable, OpenAlex had 63, 89 and
96 references, each an ID rather than a string. That removes the problem at its source and
makes Stage 4's resolution step unnecessary for those references: the work is already
identified, so there is nothing to match.

This does NOT replace the parsed bibliography. That one keeps the link between an in-text
citation marker and a specific entry, which OpenAlex has no way to express. Both are
stored; Stage 3 reads whichever it needs.

Two passes, both batched:
  1. paper DOI  -> referenced_works (50 DOIs per request)
  2. work ID    -> title, doi, year (100 IDs per request, deduplicated corpus-wide)

Usage:
  export OPENALEX_API_KEY=...
  python build/02b_openalex_refs.py --domain all --split both --sample 40   # measure first
  python build/02b_openalex_refs.py --domain all --split both
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "domains.yaml"
OUT = ROOT / "data" / "02_fulltext"
CACHE = ROOT / "data" / "02b_openalex_cache.jsonl"

API = "https://api.openalex.org/works"
MAILTO = os.environ.get("QUARTET_MAILTO", "faizajalil0@gmail.com")
UA = {"User-Agent": f"scigraphir-sir4/0.1 (mailto:{MAILTO})"}
# OpenAlex's OR filter takes a documented 50 values. In practice it often accepts more,
# and IDs-per-request is the only real throughput lever here: the request rate is pinned
# at 9/sec by politeness, so 50 per request caps the run at 450 works/sec no matter how
# many workers run. probe_id_batch() finds the real limit with one request.
DOI_BATCH = 50
ID_BATCH = 50


def probe_id_batch(cli, ids: list[str], want: int = 100) -> int:
    """Largest OR-filter size the API actually honours, from `want` down to 50.

    Verified by COUNT, not by HTTP status: an over-long filter can return 200 with a
    silently truncated result set, which would look like missing works rather than a
    rejected request.
    """
    if len(ids) < want:
        return min(len(ids), ID_BATCH)
    probe = [w.rsplit("/", 1)[-1] for w in ids[:want]]
    res = cli.get({"filter": "openalex_id:" + "|".join(probe),
                   "select": "id", "per-page": want})
    got = len((res or {}).get("results", []))
    # Allow for ids OpenAlex has merged or withdrawn; only a large shortfall means the
    # filter was truncated rather than the works being absent.
    return want if got >= want * 0.9 else ID_BATCH

RE_DOI = re.compile(r"10\.\d{4,9}/\S+")
# Same detectors the auditor uses, so "unusable" means one thing across the pipeline.
JOURNAL_ONLY = re.compile(r"^[A-Z][A-Za-z.\s&]{2,40}\.?\s*\d{1,4}\s*[,(]")
AUTHORY = re.compile(
    r"^[^a-z]{0,4}(?:[A-Z][A-Za-z'`\-]+,?\s+(?:[A-Z]\.\s*){1,3}[;,&]?\s*){2,}"
    r"|^(?:[A-Z][a-z]+\s+(?:[A-Z]\.\s*)?[A-Z][A-Za-z'`\-]+,\s*){2,}")


def unusable(v: str) -> bool:
    return bool(JOURNAL_ONLY.match(v) or AUTHORY.match(v))


def refuse_if_fetching(targets: set[str], known: list[str]) -> None:
    """Refuse only if a running Stage 2 writes to a file we are about to replace.

    Stage 2 holds its output open in append mode for the whole run, so replacing one of
    those files unlinks the inode it is writing into and every paper it fetches after
    that is silently discarded. But that is a per-FILE hazard, not a per-process one:
    a Stage 2 run on the test split and this script on the train split never touch the
    same file. Refusing on the process alone blocked a pair that was perfectly safe.
    """
    try:
        out = subprocess.run(["pgrep", "-af", "02_fulltext.py"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return
    for line in out.stdout.splitlines():
        if "02_fulltext.py" not in line:
            continue
        dm = re.search(r"--domain[= ]+(\S+)", line)
        sm = re.search(r"--split[= ]+(\S+)", line)
        if not dm or not sm:
            # Cannot tell what it writes, so assume the worst.
            sys.exit(f"REFUSING TO RUN: a Stage 2 process is running and its scope could "
                     f"not be read:\n  {line.strip()}\nStop it first: pkill -f 02_fulltext.py")
        doms = known if dm.group(1) == "all" else [x.strip() for x in dm.group(1).split(",")]
        spls = ["train", "test"] if sm.group(1) == "both" else [sm.group(1)]
        busy = {f"{d}_{s}.jsonl" for d in doms for s in spls}
        if clash := busy & targets:
            sys.exit(f"REFUSING TO RUN: Stage 2 is writing to {', '.join(sorted(clash))}.\n"
                     f"  {line.strip()}\n"
                     f"Replacing those files would make every paper it fetches from now "
                     f"on vanish. Wait for it, or narrow this run's --domain/--split.")


class RateLimiter:
    """Global token bucket. OpenAlex asks for no more than 10 requests per second, and
    that is a per-client limit, so it has to be shared across threads rather than being a
    sleep inside each one. Sleeping per worker serialises the whole thing: the run this
    was written for managed 2.5 requests/sec against an allowance of 10."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if sleep_for:
            time.sleep(sleep_for)


class Client:
    """Batched OpenAlex reads with budget-aware failure.

    OpenAlex answers 429 for BOTH throttling and an exhausted prepaid budget, and the two
    need opposite responses: back off, or stop immediately. Stage 1 spent a run retrying
    against a spent budget before that was noticed, so the body is read here.
    """

    def __init__(self, key: str, per_second: float = 9.0):
        self.key = key
        self.limiter = RateLimiter(per_second)
        self._local = threading.local()
        self._lock = threading.Lock()
        self.requests = 0

    @property
    def s(self) -> requests.Session:
        # One Session per thread: Session is not documented as thread-safe, and sharing
        # one across workers produces sporadic connection-pool errors under load.
        sess = getattr(self._local, "sess", None)
        if sess is None:
            sess = requests.Session()
            self._local.sess = sess
        return sess

    def get(self, params: dict) -> dict | None:
        if self.key:
            params = {**params, "api_key": self.key}
        for attempt in range(6):
            self.limiter.wait()
            try:
                r = self.s.get(API, params=params, headers=UA, timeout=60)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            with self._lock:
                self.requests += 1
            if r.status_code == 200:
                return r.json()
            body = r.text[:400]
            if "Insufficient budget" in body or "budget" in body.lower():
                sys.exit(f"\nOpenAlex budget exhausted after {self.requests:,} requests.\n"
                         f"{body}\nTop up at https://openalex.org/pricing, or wait for the "
                         f"midnight UTC reset. Progress is cached, so rerunning resumes.")
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            return None
        return None


def load_cache() -> dict:
    """work id -> {title, doi, year}. Cached because the same work is cited by many
    papers: deduplication is most of the saving, and a resumed run costs nothing."""
    cache = {}
    if CACHE.exists():
        for line in open(CACHE):
            try:
                w = json.loads(line)
                cache[w["id"]] = w
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="domain id, comma list, or 'all'")
    ap.add_argument("--split", required=True, choices=["train", "test", "both"])
    ap.add_argument("--sample", type=int,
                    help="measure coverage on N papers per file and write nothing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--id-batch", type=int,
                    help="ids per OpenAlex request; probed automatically if unset")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent OpenAlex requests (default 8). The shared limiter "
                         "caps the aggregate rate at 9/sec regardless, so this only "
                         "overlaps waiting.")
    a = ap.parse_args()

    key = os.environ.get("OPENALEX_API_KEY", "")
    if not key:
        print("!! OPENALEX_API_KEY is not set. The free pool is small and shared by IP, "
              "so this will stop early. export it first.")
    known = [d["id"] for d in yaml.safe_load(open(CONFIG))["domains"]]
    domains = known if a.domain == "all" else [d.strip() for d in a.domain.split(",")]
    splits = ["train", "test"] if a.split == "both" else [a.split]
    if not a.sample:
        refuse_if_fetching({f"{d}_{s}.jsonl" for d in domains for s in splits}, known)
    cli = Client(key)

    # ---------------------------------------------------------------- pass 1: the lists
    jobs: list[tuple[Path, list[dict]]] = []
    for d in domains:
        for s in splits:
            path = OUT / f"{d}_{s}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(l) for l in open(path) if l.strip()]
            if a.sample:
                rows = random.Random(a.seed).sample(rows, min(a.sample, len(rows)))
            jobs.append((path, rows))
    if not jobs:
        sys.exit("no Stage 2 output found")

    by_doi: dict[str, list[str]] = {}
    todo = [r for _, rows in jobs for r in rows]
    bar = tqdm(total=len(todo), desc="reference lists", unit="paper", dynamic_ncols=True)
    for i in range(0, len(todo), DOI_BATCH):
        batch = todo[i: i + DOI_BATCH]
        dois = [(r.get("doi") or "").replace("https://doi.org/", "").lower()
                for r in batch]
        dois = [d for d in dois if d]
        if dois:
            res = cli.get({"filter": "doi:" + "|".join(dois),
                           "select": "doi,referenced_works",
                           "per-page": DOI_BATCH})
            for w in (res or {}).get("results", []):
                k = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                by_doi[k] = w.get("referenced_works") or []
        bar.update(len(batch))
    bar.close()

    hit = sum(1 for r in todo
              if by_doi.get((r.get("doi") or "").replace("https://doi.org/", "").lower()))
    print(f"\n{hit:,}/{len(todo):,} papers have an OpenAlex reference list "
          f"({100*hit/max(len(todo),1):.0f}%)")
    if not hit:
        sys.exit("no reference lists returned; check the API key and budget")

    # ---------------------------------------------------------------- pass 2: the titles
    cache = load_cache()
    wanted = {w for ws in by_doi.values() for w in ws} - set(cache)
    print(f"{len(wanted):,} referenced works to look up "
          f"({len(cache):,} already cached)")
    if wanted:
        # Concurrent, because the work is entirely network latency. Sequentially this ran
        # at 2.5 requests/sec against OpenAlex's allowance of 10, so 1.28M works would
        # have taken 2.8 hours. The shared limiter keeps the aggregate rate legal no
        # matter how many workers are in flight.
        wl = sorted(wanted)
        batch = a.id_batch or probe_id_batch(cli, wl)
        print(f"using {batch} ids per request "
              f"(~{int(batch * 9):,} works/sec at the 9 req/sec limit)")
        chunks = [wl[i: i + batch] for i in range(0, len(wl), batch)]
        bar = tqdm(total=len(wl), desc="work titles", unit="work", dynamic_ncols=True)
        write_lock = threading.Lock()

        def fetch(chunk):
            ids = [w.rsplit("/", 1)[-1] for w in chunk]
            res = cli.get({"filter": "openalex_id:" + "|".join(ids),
                           "select": "id,title,doi,publication_year",
                           "per-page": batch})
            return [{"id": w["id"], "title": w.get("title") or "",
                     "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                     "year": w.get("publication_year")}
                    for w in (res or {}).get("results", [])], len(chunk)

        with open(CACHE, "a") as fh, ThreadPoolExecutor(max_workers=a.workers) as pool:
            for fut in as_completed([pool.submit(fetch, c) for c in chunks]):
                try:
                    recs, n = fut.result()
                except Exception as exc:
                    bar.write(f"  !! {exc.__class__.__name__}: {exc}")
                    continue
                with write_lock:
                    for rec in recs:
                        cache[rec["id"]] = rec
                        fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    bar.update(n)
        bar.close()

    # ---------------------------------------------------------------- report and write
    stats = collections.Counter()
    # Coverage PER FILE, not just in aggregate. OpenAlex builds reference lists from
    # publisher deposits, which lag by months, so the test window (October 2025 onward)
    # is far less complete than the training window. An aggregate 72% hid a split from
    # ~88% on train to ~18% on test -- and train and test drawing their references from
    # different sources would make the two halves of the benchmark incomparable.
    print(f"\n{'file':<24}{'papers':>8}{'coverage':>10}{'OA refs':>9}{'parsed':>8}"
          f"{'unusable parsed':>17}{'rescued by OA':>15}")
    for path, rows in jobs:
        out_lines, n_oa, n_parsed, n_bad, n_resc, n_paper, n_cov = [], 0, 0, 0, 0, 0, 0
        for r in rows:
            k = (r.get("doi") or "").replace("https://doi.org/", "").lower()
            n_cov += bool(by_doi.get(k))
            refs = [cache[w] for w in by_doi.get(k, []) if w in cache and cache[w]["title"]]
            parsed = list((r.get("bibliography") or {}).values())
            bad = sum(1 for v in parsed if unusable(v))
            n_paper += 1; n_oa += len(refs); n_parsed += len(parsed); n_bad += bad
            # "Rescued" counts only where OpenAlex supplies titles the parse could not:
            # a paper with unusable strings that now has a clean structured list.
            if bad and refs:
                n_resc += bad
            r["openalex_references"] = refs
            r["openalex_reference_count"] = len(refs)
            out_lines.append(json.dumps(r) + "\n")
        stats["papers"] += n_paper; stats["oa"] += n_oa; stats["cov"] += n_cov
        stats["parsed"] += n_parsed; stats["bad"] += n_bad; stats["rescued"] += n_resc
        cov = f"{100*n_cov/max(n_paper,1):.0f}%"
        print(f"{path.name:<24}{n_paper:>8,}{cov:>10}{n_oa:>9,}{n_parsed:>8,}"
              f"{n_bad:>16,}{n_resc:>15,}")
        if not a.sample:
            # Write then rename, so an interrupted run cannot truncate the file.
            tmp = path.with_suffix(".jsonl.partial")
            with open(tmp, "w") as fh:
                fh.writelines(out_lines)
            os.replace(tmp, path)

    print(f"\n{cli.requests:,} OpenAlex requests (~${cli.requests * 0.0001:.2f})")
    if a.sample:
        print("SAMPLE MODE: nothing was written. Drop --sample to apply.")
    else:
        print(f"added `openalex_references` to {stats['papers']:,} papers; "
              f"{stats['rescued']:,} previously unusable references now have a title "
              f"and an OpenAlex id")


if __name__ == "__main__":
    main()
