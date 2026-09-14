"""Build per-query inputs for the SIR-4 downstream hypothesis-composition study.

API-FREE.  SIR-4 port of build_inputs.py + augment_arms.py in one script.  Run with
DOWNSTREAM_DATASET=sir4: every downstream script then reads/writes
results/downstream_sir4/ (inputs_sir4.jsonl, compositions_*, matched_*, batch/).

Units = the four subset manifests used for every SIR-4 LLM-baseline row (all
cross-field queries + 250 sampled same-field queries, seed 42):
cs 497 / biology 391 / physics 388 / matsci 331 = 1,607 queries.

What differs from TOMATO, and how it is handled
  * SIR-4 retrieval is paper-level: one ranking per paper, several gold
    inspirations per paper (mean 4.3), and the gold list is the UNION of every
    validated inspiration set (the primary decomposition plus the uniqueness
    sweep's alternative sets).  Top-1 hit = the top-1 is ANY gold, matching the
    SIR-4 recall metrics.
  * Judge reference (--reference component, the default).  TOMATO judged each
    composition against the component its query's gold contributes
    (hypothesis_components[step]).  SIR-4 has that text for every gold: the
    inspiration's `delta` (identical to hypothesis_components[order] for the
    primary set; each alternative set carries its own deltas).  So a composition
    made from a GOLD document is judged against that gold's delta; a composition
    made from a non-gold document, and the closed-book arm, is judged against the
    primary inspiration's delta (order 0).  The oracle arm feeds the primary
    inspiration and is judged against that same order-0 delta.
    --reference hypothesis instead judges every arm against the paper's full
    fine-grained hypothesis (one reference per query; lower absolute scores
    because one document can only cover one of several inspirations).
  * prev_hypothesis is always "No previous hypothesis." (= TOMATO step 0).
  * Corpus documents are (title, abstract) from the field's documents.json;
    10-16% are title-only and are fed with an empty abstract, exactly as the
    retrievers saw them.

Record schema (superset of TOMATO's, so compose / score / aggregate run unchanged):
  query_id, field, source_id (doi), step_idx=0, stratum, research_question,
  background_survey, prev_hypothesis, gt_delta, gt_hypothesis, reference_mode,
  gold_key, golds, n_golds, gold_deltas {gold: delta},
  gold_top1 = {key, title, abstract, is_gold=True, gt_delta}
  {arm}_top1 = {key, title, abstract, is_gold[, gt_delta when is_gold]}

Run (from TOMATO-Star):
  DOWNSTREAM_DATASET=sir4 python -m analysis.downstream.build_inputs_sir4 \
      --arm qwen3 --arm moose_chem --arm lattice --arm reasonir --arm scigraphir
--arm NAME uses the default path template (DEFAULT_ARM_PATHS, {field} substituted);
--arm NAME=/path/with/{field}.json overrides it.  A missing file skips that arm
with a warning.  Re-running keeps every arm already in the file and adds or
overwrites the arms given; --rebuild drops them all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from analysis.downstream._common import DATASET, INPUTS_PATH, read_jsonl

if DATASET != "sir4":
    raise SystemExit("build_inputs_sir4 must run with DOWNSTREAM_DATASET=sir4")

CARGO = Path(os.environ.get("CARGO_DIR", Path.home() / "Desktop" / "CARGO"))
S4 = CARGO / "sir4-retrieval"
sys.path.insert(0, str(S4 / "llm_baselines"))
import sir4_llm_data as S  # noqa: E402  (queries, corpus, subset conventions)

FIELDS = ["cs", "biology", "physics", "matsci"]
FIELD_NAME = {"cs": "Computer Science", "biology": "Biology",
              "physics": "Physics", "matsci": "Materials Science"}
# QUARTET stage-4 (resolved) files: the research question, background survey,
# fine-grained hypothesis, the primary inspiration list with per-inspiration
# deltas, and the uniqueness sweep's alternative sets (uniqueness.M).
QUARTET_DIR = CARGO / "quartet" / "data.nosync" / "04_resolved"
QUARTET_FILE = {"cs": "cs_test_final", "biology": "biology_test_low",
                "physics": "physics_test_low", "matsci": "matsci_test_low"}
SUBSET = str(S4 / "llm_baselines" / "subsets" / "subset_{field}_same250_allcross_seed42.json")

DEFAULT_ARM_PATHS = {
    "qwen3":      str(S4 / "data" / "predictions_qwen3_sir4_{field}_test.json"),
    "moose_chem": str(S4 / "llm_baselines" / "outputs" / "moose_chem" / "{field}_pool100" / "rankings.json"),
    "lattice":    str(S4 / "llm_baselines" / "outputs" / "lattice" / "{field}" / "rankings.json"),
    "reasonir":   str(S4 / "data" / "predictions_reasonir_sir4_{field}_test.json"),
    "scigraphir": str(S4 / "data" / "predictions_scigraphir_sir4_{field}_test.json"),
}
PREV_HYPOTHESIS = "No previous hypothesis."


# --- identifiers (copied from quartet/build/05_export.py so ids match the corpus) ---
def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def doc_id(x: dict) -> str | None:
    doi = x.get("found_doi")
    if doi:
        return str(doi).strip().lower()
    title = x.get("found_title")
    if not title:
        return None
    return "q:" + hashlib.sha1(_norm(title).encode()).hexdigest()[:16]


# --- rankings --------------------------------------------------------------
def load_rankings(path: Path) -> dict[str, list[str]]:
    """Accepted formats (auto-detected), keys lower-cased:
      A. CARGO prediction list [{id, predictions:{document:[[key,score],...]}}] (score desc)
      B. ranked dict {query_id: [key, ...]}  (MOOSE-Chem / LATTICE rankings.json)"""
    d = json.load(open(path))
    if isinstance(d, dict):
        return {q: [str(k).strip().lower() for k in ks] for q, ks in d.items()}
    out = {}
    for r in d:
        docs = sorted(r["predictions"]["document"], key=lambda x: -x[1])
        out[r["id"]] = [str(k).strip().lower() for k, _ in docs]
    return out


def top1(keys: list[str], corpus: dict, golds: set, gold_deltas: dict) -> dict | None:
    """First ranked key that resolves to a corpus document with any text."""
    for k in keys:
        if k in corpus and (corpus[k][0] or corpus[k][1]):
            title, abstract = corpus[k]
            d = {"key": k, "title": title, "abstract": abstract, "is_gold": k in golds}
            if d["is_gold"] and gold_deltas.get(k):
                d["gt_delta"] = gold_deltas[k]
            return d
    return None


# --- base records ----------------------------------------------------------
def build_base(field: str, reference: str):
    corpus = {k.lower(): v for k, v in S.load_corpus(field).items()}
    queries = {q["query_id"]: q for q in S.load_queries(field)}
    man = json.load(open(SUBSET.format(field=field)))
    ids, strata = man["query_ids"], man["strata"]
    want = set(ids)
    recs = {}
    with open(QUARTET_DIR / f"{QUARTET_FILE[field]}.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            qid = r["doi"].replace("/", "_")
            if qid in want:
                recs[qid] = r
    missing_recs = [q for q in ids if q not in recs]
    if missing_recs:
        raise SystemExit(f"{field}: {len(missing_recs)} subset queries missing from "
                         f"{QUARTET_FILE[field]}.jsonl, e.g. {missing_recs[:3]}")

    rows, st = [], Counter()
    for qid in ids:
        r, q = recs[qid], queries[qid]
        insp = sorted(r.get("inspiration") or [], key=lambda i: i.get("order", 0))
        if not insp:
            raise SystemExit(f"{field}/{qid}: no inspirations in the QUARTET record")
        # every gold's contribution text: primary set first, then alternatives
        gold_deltas: dict[str, str] = {}
        for i in insp:
            k = doc_id(i)
            if k and (i.get("delta") or "").strip():
                gold_deltas.setdefault(k, i["delta"].strip())
        for m in (r.get("uniqueness") or {}).get("M") or []:
            for i in m.get("inspiration") or []:
                k = doc_id(i)
                if k and (i.get("delta") or "").strip():
                    gold_deltas.setdefault(k, i["delta"].strip())
        golds = [str(g).strip().lower() for g in q["golds"]]
        primary = insp[0]
        gold_key = doc_id(primary)
        delta0 = (primary.get("delta") or "").strip()
        hyp = (r.get("fine_grained_hypothesis") or "").strip()
        if not delta0 or not hyp:
            raise SystemExit(f"{field}/{qid}: empty primary delta or hypothesis")
        gt = delta0 if reference == "component" else hyp
        no_delta = [g for g in golds if g not in gold_deltas]
        for g in no_delta:                       # never seen so far; judged vs the default
            gold_deltas[g] = gt
        st["queries"] += 1
        st["golds"] += len(golds)
        st["golds_without_delta"] += len(no_delta)
        st[f"stratum_{strata[qid]}"] += 1
        if gold_key in corpus:
            g_title, g_abs = corpus[gold_key]
        else:
            st["primary_gold_not_in_corpus"] += 1
            g_title = (primary.get("found_title") or "").strip()
            g_abs = (primary.get("found_abstract") or "").strip()
        if not g_abs:
            st["primary_gold_title_only"] += 1
        if gold_key not in golds:
            st["primary_gold_not_in_gold_list"] += 1
        rq = (r.get("research_question") or q["research_question"] or "").strip()
        bs = (r.get("background_survey") or q["background_survey"] or "").strip()
        rows.append({
            "query_id": qid, "field": field, "source_id": r["doi"], "step_idx": 0,
            "stratum": strata[qid],
            "research_question": rq, "background_survey": bs,
            "prev_hypothesis": PREV_HYPOTHESIS,
            "gt_delta": gt, "gt_hypothesis": hyp, "reference_mode": reference,
            "gold_key": gold_key, "golds": golds, "n_golds": len(golds),
            "gold_deltas": gold_deltas if reference == "component" else {},
            "gold_top1": {"key": gold_key, "title": g_title, "abstract": g_abs,
                          "is_gold": True, "gt_delta": gt},
        })
    return rows, corpus, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", nargs="+", default=FIELDS, choices=FIELDS)
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME  or  NAME=/path/with/{field}.json  (repeatable)")
    ap.add_argument("--reference", choices=["component", "hypothesis"], default="component")
    ap.add_argument("--rebuild", action="store_true",
                    help="drop every arm already in the inputs file")
    args = ap.parse_args()

    existing = [] if args.rebuild else read_jsonl(INPUTS_PATH)
    if existing and existing[0].get("reference_mode") != args.reference:
        raise SystemExit(f"{INPUTS_PATH} was built with --reference "
                         f"{existing[0].get('reference_mode')}; pass --rebuild to change it")
    carried = {r["query_id"]: {k: v for k, v in r.items()
                               if k.endswith("_top1") and k != "gold_top1"}
               for r in existing}
    carried_names = sorted({k[:-5] for d in carried.values() for k in d})

    rows_all, corpora = [], {}
    print(f"reference mode: {args.reference}")
    for field in args.fields:
        rows, corpus, st = build_base(field, args.reference)
        corpora[field] = corpus
        for r in rows:
            r.update(carried.get(r["query_id"], {}))
        rows_all.extend(rows)
        print(f"{field:8} queries {st['queries']} (same {st['stratum_same']} / cross "
              f"{st['stratum_cross']})  golds/query {st['golds'] / st['queries']:.2f}  "
              f"golds without delta {st['golds_without_delta']}  "
              f"primary gold title-only {st['primary_gold_title_only']}"
              + (f"  NOT IN CORPUS {st['primary_gold_not_in_corpus']}"
                 if st["primary_gold_not_in_corpus"] else "")
              + (f"  primary not in gold list {st['primary_gold_not_in_gold_list']}"
                 if st["primary_gold_not_in_gold_list"] else ""))
    if carried_names:
        print(f"carried arms from the existing file: {carried_names}")

    for spec in args.arm:
        name, _, tmpl = spec.partition("=")
        tmpl = tmpl or DEFAULT_ARM_PATHS.get(name)
        if not tmpl:
            raise SystemExit(f"unknown arm '{name}': give NAME=/path/with/{{field}}.json")
        fld = f"{name}_top1"
        tot_hit = tot_n = 0
        for field in args.fields:
            p = Path(tmpl.format(field=field))
            frows = [r for r in rows_all if r["field"] == field]
            if not p.exists():
                print(f"{name:<11} {field:8} SKIPPED (no file: {p})")
                continue
            rank = load_rankings(p)
            hit = miss = unres = 0
            for r in frows:
                ks = rank.get(r["query_id"])
                if ks is None:
                    miss += 1
                    r[fld] = {"key": "(missing)", "title": "", "abstract": "", "is_gold": False}
                    continue
                t = top1(ks, corpora[field], set(r["golds"]), r["gold_deltas"])
                if t is None:
                    unres += 1
                    t = {"key": "(unresolved)", "title": "", "abstract": "", "is_gold": False}
                r[fld] = t
                hit += int(t["is_gold"])
            n = len(frows)
            tot_hit += hit
            tot_n += n
            print(f"{name:<11} {field:8} top-1 hit {hit}/{n} ({100 * hit / n:.1f}%)"
                  + (f"  missing queries {miss}" if miss else "")
                  + (f"  unresolved {unres}" if unres else "")
                  + f"   [{p.name}]")
        if tot_n:
            print(f"{name:<11} {'ALL':8} top-1 hit {tot_hit}/{tot_n} ({100 * tot_hit / tot_n:.1f}%)")

    INPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INPUTS_PATH.open("w") as f:
        for r in rows_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    arms = sorted({k[:-5] for r in rows_all for k in r if k.endswith("_top1")})
    print(f"\nWrote {INPUTS_PATH}: {len(rows_all)} records, arms {arms}")
    missing_arms = [a for a in ("qwen3", "reasonir", "moose_chem", "scigraphir") if a not in arms]
    if missing_arms:
        print(f"still missing for the Table 9.3 layout: {missing_arms} "
              f"(download the prediction files, then re-run with --arm NAME)")


if __name__ == "__main__":
    main()
