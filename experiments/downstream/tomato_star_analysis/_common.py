"""Shared helpers for the downstream hypothesis-composition experiment.

Experiment: does better *inspiration retrieval* yield better *composed hypotheses*?
We hold the composer fixed (gpt-4o-mini) and feed it the TOP-1 retrieved inspiration
under two arms -- BM25 vs MOOSE-Chem -- then evaluate the composed delta hypothesis
with (a) the Matched-Score rubric (reference-based, vs the GT delta) and
(b) Idea Arena (reference-free, pairwise BM25-vs-MOOSE-Chem).

This module reuses the *real* MOOSE-Star prompts/scoring so there is no copy-drift:
  - composition prompt: prompt_store.instruction_prompts(
        "prepare_HC_sft_data_to_go_comprehensive_v2_delta")
  - scoring rubric:     scoring_utils.{SCORING_RUBRIC, RERANKER_PROMPT_TEMPLATE, parse_scores}
  - delta extraction:   common_utils.{extract_between_markers, extract_answer_content}
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- paths -----------------------------------------------------------------
PROJ = Path(__file__).resolve().parents[2]          # .../TOMATO-Star
MOOSE_STAR = PROJ.parent / "MOOSE-Star"

import sys
sys.path.insert(0, str(MOOSE_STAR / "utils"))

from prompt_store import instruction_prompts                         # noqa: E402
from scoring_utils import (                                          # noqa: E402
    SCORING_RUBRIC,
    RERANKER_PROMPT_TEMPLATE,
    parse_scores,
)
from common_utils import extract_between_markers, extract_answer_content  # noqa: E402

# The 6-segment composition prompt (Research Q / Background / Prev hyp /
# Insp title / Insp abstract / response instructions).
COMPOSE_PROMPT = instruction_prompts(
    "prepare_HC_sft_data_to_go_comprehensive_v2_delta"
)

# Dataset switch.  DOWNSTREAM_DATASET=tomato (default) | sir4.  Every script in
# this package reads and writes under OUT_DIR / INPUTS_PATH, so the one env var
# re-points the whole pipeline (build -> compose -> judge -> aggregate -> table)
# at another dataset without touching the TOMATO files.
import os
DATASET = os.environ.get("DOWNSTREAM_DATASET", "tomato").strip().lower()
if DATASET == "tomato":
    OUT_DIR = PROJ / "results" / "downstream"
    INPUTS_PATH = OUT_DIR / "inputs_500.jsonl"
elif DATASET == "sir4":
    OUT_DIR = PROJ / "results" / "downstream_sir4"
    INPUTS_PATH = OUT_DIR / "inputs_sir4.jsonl"
else:
    raise SystemExit(f"DOWNSTREAM_DATASET must be 'tomato' or 'sir4', got {DATASET!r}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical complete MOOSE-Chem ranking (the only variant with all 500 queries;
# the plain/approach3 caches are 5-10 key smoke tests).
MC_RANKINGS = (
    PROJ / "outputs" / "caches" / "moose_chem"
    / "rankings_moose-chem-gpt-4o-mini-approach3-soft_stratified_250same_250cross_seed42.json"
)


# --- OpenAI ----------------------------------------------------------------
_client = None
_client_lock = threading.Lock()


def client():
    """Lazily build a single OpenAI client (reads OPENAI_API_KEY from env)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from openai import OpenAI
                _client = OpenAI()
    return _client


def chat(prompt: str, model: str = "gpt-4o-mini",
         temperature: float = 0.0, max_tokens: int = 4096,
         retries: int = 5) -> str:
    """One chat completion with exponential-backoff retries."""
    last = None
    for attempt in range(retries):
        try:
            resp = client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat failed after {retries} attempts: {last}")


# --- composition helpers ---------------------------------------------------
def build_compose_prompt(research_question: str, background_survey: str,
                         prev_hypothesis: str, insp_title: str,
                         insp_abstract: str) -> str:
    p = COMPOSE_PROMPT
    return (p[0] + research_question + p[1] + background_survey +
            p[2] + prev_hypothesis + p[3] + insp_title +
            p[4] + insp_abstract + p[5])


def extract_delta(response: str) -> str:
    """Pull the delta hypothesis out of a composer response (same logic as the
    MOOSE-Star rubric harness)."""
    d = extract_between_markers(response, r"Delta\s*Hypothesis")
    if d:
        return d.strip()
    return (extract_answer_content(response) or "").strip()


# --- jsonl io --------------------------------------------------------------
def read_jsonl(path) -> list[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def index_by(rows: list[dict], key: str = "query_id") -> dict:
    return {r[key]: r for r in rows}


def done_ids(path, key: str = "query_id") -> set:
    return {r[key] for r in read_jsonl(path)}


class JsonlWriter:
    """Thread-safe append-only jsonl writer for resumable runs."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._fh = self.path.open("a")

    def write(self, obj: dict):
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        self._fh.close()


def run_concurrent(items, fn, writer: "JsonlWriter", workers: int = 20,
                   desc: str = "") -> int:
    """Map fn over items concurrently; write each non-None result immediately.

    fn(item) -> dict | None.  Returns the number of records written.

    Progress: prints a 'starting' line up front, then the first completion
    (so you can confirm calls are succeeding within seconds), then ~every 5%
    with live rate / elapsed / ETA, then a final line.  Per-item failures are
    caught and counted (the run keeps going; failed items stay un-written so a
    resume will retry them) and the first few error messages are surfaced.
    """
    n = len(items)
    if n == 0:
        return 0
    written = 0
    errors = 0
    written_lock = threading.Lock()
    start = time.time()
    step = max(1, n // 20)            # ~5% increments
    print(f"  [{desc}] starting {n} items @ {workers} workers ...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
            except Exception as e:      # noqa: BLE001
                errors += 1
                res = None
                if errors <= 3:
                    print(f"  [{desc}] ERROR ({errors}): {e}", flush=True)
            if res is not None:
                writer.write(res)
                with written_lock:
                    written += 1
            if i == 1 or i % step == 0 or i == n:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (n - i) / rate if rate > 0 else 0.0
                err = f"  errors={errors}" if errors else ""
                print(f"  [{desc}] {i}/{n} ({100 * i / n:3.0f}%)  "
                      f"written={written}{err}  {rate:4.1f}/s  "
                      f"elapsed {elapsed:4.0f}s  eta {eta:4.0f}s", flush=True)
    if errors:
        print(f"  [{desc}] DONE with {errors} error(s); "
              f"re-run the same command to retry the {errors} un-written item(s).",
              flush=True)
    return written
