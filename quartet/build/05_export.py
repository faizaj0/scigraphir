#!/usr/bin/env python3
"""Stage 5. Export a resolved QUARTET split as a retrieval benchmark.

Stage 4 leaves one JSONL of papers, each carrying a validated decomposition and,
where the uniqueness sweep found any, a family M of alternative decompositions.
That is not something a retriever can be run against. This turns it into the
three files the pipeline consumes.

    <out>/raw/documents.json    {doc_id: "Title. Abstract"}    the corpus
    <out>/eval.json             THE evaluation: one row per query, carrying every
                                valid decomposition as a set
    <out>/eval_primary.json     D1 only, flat, TOMATO-Star shape. Comparability
                                only. NOT the headline result
    <out>/sets.json             M keyed by paper, for analysis
    <out>/manifest.json         counts, and everything excluded and why

WHY THREE FILES AND NOT TWO.

TOMATO-Star's eval format stores golds as a FLAT LIST. A query with three golds
is three rows, ::0 ::1 ::2. M is not a flat list, it is a family of SETS:

    M  = { {A,B} , {A,C} }        two ways to explain the same hypothesis
    flattened -> {A, B, C}

Under the flat reading a retriever that returns only C scores a hit, though {C}
alone explains nothing. Flattening overstates, and it cannot distinguish
"recovered a complete valid explanation" from "recovered one member of one".

WHY THE FLAT FILE CANNOT BE THE HEADLINE.

The first version emitted only the flat file and put every alternative's members
into the corpus anyway. An alternative-only inspiration was then: in the corpus,
validated as part of a sufficient decomposition in sets.json, and ABSENT from
eval.json, so the evaluator counted retrieving it as a false positive. Measured:
63 of 236 documents, 27% of the corpus, across 48 of 73 papers. A retriever that
recovers D2 instead of D1 has succeeded, and that file called it a miss. That is
the exact error the uniqueness test exists to expose, so it must not be built
into the scoring.

eval.json is therefore one row per QUERY, carrying M as sets. Its
`supporting_documents` is the union, which removes the false negative and is the
floor; `quartet.sets` is what CompleteSet@k reads, and that is the metric that
can tell "recovered a complete valid explanation" from "recovered one member of
one". build/score.py computes both.

WHAT COUNTS AS A DOCUMENT.

An inspiration is corpus-eligible when Stage 4 resolved it AND it has an
abstract. Roughly a fifth of resolved golds have no abstract anywhere machine
readable, mostly older papers, and those are emitted as title-only documents and
flagged rather than dropped: the drop is not random, it hits old papers hardest,
and old papers are where the cross-domain borrowings live. `--drop-title-only`
removes them if you would rather have the cleaner corpus.

    python build/05_export.py --input  data/_snap/run_B_resolved.jsonl \
                              --out    data/benchmark/cs_test
    python build/05_export.py --input ... --drop-title-only --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L          # the same/cross definition, shared with Stage 4

ROOT = Path(__file__).resolve().parent.parent


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def doc_id(x: dict) -> str | None:
    """Stable identifier for a resolved inspiration.

    A DOI when there is one, because that is what the existing corpus is keyed
    on and what makes a document comparable across splits. Some papers resolve
    with a title and abstract but no DOI, mostly S2 hits; those get a synthetic
    id rather than being dropped, since the text is what a retriever indexes and
    the text is present. The prefix keeps them obviously distinguishable from
    real DOIs in any downstream analysis.
    """
    doi = x.get("found_doi")
    if doi:
        return str(doi).strip().lower()
    title = x.get("found_title")
    if not title:
        return None
    return "q:" + hashlib.sha1(norm(title).encode()).hexdigest()[:16]


def doc_text(x: dict) -> str:
    """What the retriever indexes. Title and abstract, exactly as TOMATO has it."""
    title = str(x.get("found_title") or "").strip().rstrip(".")
    abstract = str(x.get("found_abstract") or "").strip()
    return f"{title}. {abstract}".strip() if abstract else f"{title}."


def question_of(rec: dict) -> str:
    """The retrieval query: the research question, then the background.

    Matching TOMATO, whose `question` field is the two run together. The
    background is the part that carries the search signal; the question orients
    it. Neither names a citation, which the Stage 3 disjointness check enforces.
    """
    rq = str(rec.get("research_question") or "").strip()
    bg = str(rec.get("background_survey") or "").strip()
    return f"{rq} {bg}".strip()


def stratum_of(x: dict) -> str | None:
    """Collapsed to the two values the existing eval code understands.

    `domain_distance` has three bands and they are worth keeping, but `stratum`
    in the prediction files only ever holds "same" or "cross". Emitting a third
    value here would break the consumer for the sake of a distinction that is
    preserved anyway, one field over, in `domain_distance`.

    Stage 4's binary label was renamed same_domain/cross_domain ->
    same_field/cross_field, because that is the level it compares. Both spellings
    are read here so a file resolved before that change still exports.
    """
    d = x.get("domain_relation")
    if d in ("same", "cross"):
        return d                       # written by the label policy, use as-is
    # Files written before build/labels.py existed. Both older spellings compared
    # one forced field label per side; kept only so an old run still exports.
    if d in ("same_field", "same_domain"):
        return "same"
    if d in ("cross_field", "cross_domain"):
        return "cross"
    legacy = x.get("domain_distance")
    if legacy in ("same_field", "same_primary"):
        return "same"
    return "cross" if legacy else None


def confidence_of(x: dict) -> str | None:
    """Whether the field label behind `stratum` is worth counting.

    Stage 4 marks a field "ambiguous" when only one of the work's three OpenAlex
    topics carries it. On the first run 87 of 190 cross-field labels rested on a
    plurality of one, so a same/cross split that counts them is reporting the
    taxonomy's noise as a property of the benchmark. The row still exports, with
    the flag attached, so the split can be taken over the clear labels alone.
    """
    return x.get("field_confidence")


def quartet_of(x: dict) -> dict:
    """The per-document provenance both eval files carry."""
    return {
        "domain_distance": x.get("domain_distance"),
        "field_pair": x.get("field_pair"),
        "abstract_source": x.get("abstract_source"),
        "match_quality": x.get("match_quality"),
        "similarity": x.get("similarity"),
        "order": x.get("order"),
        "stratum": stratum_of(x),
        "stratum_labelled": stratum_of(x) is not None,
        "field_confidence": confidence_of(x),
        "label_basis": x.get("label_basis"),
        "field_purity": x.get("field_purity"),
        # The facts the same/cross verdict was derived from, so build/relabel.py
        # can re-cut the split here without going back to the Stage 4 file.
        "label_evidence": x.get("label_evidence"),
        "label_policy": x.get("label_policy"),
        "field_source": x.get("field_source"),
        "openalex_id": x.get("openalex_id"),
        "identity": x.get("identity"),
        "year": x.get("year"),
    }


def usable(x: dict, drop_title_only: bool) -> bool:
    return bool(doc_id(x)) and bool(
        x.get("found_abstract") or not drop_title_only)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="a Stage 4 output, i.e. resolved")
    ap.add_argument("--out", type=Path, required=True,
                    help="directory to write raw/documents.json, eval.json, sets.json")
    ap.add_argument("--split", default=None,
                    help="recorded in the manifest; defaults to the input stem")
    ap.add_argument("--drop-title-only", action="store_true",
                    help="exclude golds resolved without an abstract. Cleaner "
                         "corpus, but the exclusion is not random")
    ap.add_argument("--keep-partial", action="store_true",
                    help="keep a paper whose decomposition is only partly "
                         "resolvable. Off by default: scoring against half a "
                         "decomposition is not scoring against a decomposition")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.input) if l.strip()]
    ok = [r for r in recs if r.get("decomposed")]
    excluded: dict[str, list] = defaultdict(list)

    documents: dict[str, str] = {}
    title_only: set[str] = set()
    rows: list[dict] = []
    set_rows: list[dict] = []
    sets_out: dict[str, dict] = {}

    for r in ok:
        prim = r.get("inspiration") or []
        if not prim:
            excluded["no_inspirations"].append(r.get("doi"))
            continue

        # A decomposition is exported whole or not at all. A row whose gold set
        # is missing a member scores the retriever against a set that no check
        # ever validated, and it reads as a retrieval failure rather than as the
        # resolution failure it actually is.
        if not all(usable(x, a.drop_title_only) for x in prim):
            n_ok = sum(1 for x in prim if usable(x, a.drop_title_only))
            if not a.keep_partial:
                excluded["primary_incomplete"].append(
                    (r.get("doi"), f"{n_ok}/{len(prim)} golds resolvable"))
                continue
            prim = [x for x in prim if usable(x, a.drop_title_only)]
            if not prim:
                excluded["primary_none"].append(r.get("doi"))
                continue

        pid = r.get("source_id") or str(r.get("doi") or "").replace("/", "_")
        question = question_of(r)
        if not question:
            excluded["no_question"].append(r.get("doi"))
            continue

        for x in prim:
            did = doc_id(x)
            documents.setdefault(did, doc_text(x))
            if not x.get("found_abstract"):
                title_only.add(did)

        for j, x in enumerate(prim):
            did = doc_id(x)
            rows.append({
                "id": f"{pid}::{j}",
                "question": question,
                "answer": "",
                "stratum": stratum_of(x) or "same",
                "supporting_documents": [did],
                "target_nodes": {"document": [did]},
                "quartet": quartet_of(x),
            })

        # ---- the family, with its set structure kept
        M = (r.get("uniqueness") or {}).get("M") or []
        members, dropped = [], 0
        meta: dict[str, dict] = {}
        # doc id -> what that inspiration contributes to h, in dependency order.
        deltas: dict[str, dict] = {}
        set_deltas: list[list] = []

        def note_delta(x):
            did = doc_id(x)
            if did and did not in deltas:
                deltas[did] = {"order": x.get("order"),
                               "delta": x.get("delta"),
                               "insp": x.get("insp_concise") or x.get("insp"),
                               "motivation": x.get("motivation"),
                               "relation": x.get("relation")}
            return did
        for m in M:
            insps = m.get("inspiration") or []
            ids = [doc_id(x) for x in insps]
            if not ids or not all(ids):
                dropped += 1          # an alternative Stage 4 could not resolve
                continue
            if not all(usable(x, a.drop_title_only) for x in insps):
                dropped += 1          # a member with no usable document
                continue
            for x in insps:
                did = note_delta(x)
                if did not in documents:
                    documents[did] = doc_text(x)
                    if not x.get("found_abstract"):
                        title_only.add(did)
                meta.setdefault(did, quartet_of(x))
            members.append(ids)
            # The ordered chain this set claims composes to h, so a reader can
            # follow one alternative without reassembling it from two files.
            set_deltas.append([{"doc": doc_id(x), "order": x.get("order"),
                                "delta": x.get("delta")} for x in insps])

        # The primary decomposition is a valid set by construction, so it belongs
        # in the family whether or not the sweep happened to re-derive it.
        prim_ids = [doc_id(x) for x in prim]
        if not any(set(mm) == set(prim_ids) for mm in members):
            members.insert(0, prim_ids)
        for x in prim:
            meta.setdefault(doc_id(x), quartet_of(x))
            note_delta(x)
        if len(set_deltas) < len(members):      # the primary set was inserted
            set_deltas.insert(0, [{"doc": doc_id(x), "order": x.get("order"),
                                   "delta": x.get("delta")} for x in prim])

        # ---- the SET-AWARE row. One per query, not one per gold.
        #
        # This is the QUARTET evaluation. eval_primary.json below keeps the
        # flat TOMATO-Star shape for comparability, but it cannot be the headline
        # file: it lists only D1's members as relevant, while the corpus contains
        # every alternative's members too. A retriever that returns a DIFFERENT
        # valid decomposition is then marked wrong for returning a decomposition
        # this benchmark itself validated as sufficient. Measured on the first
        # export: 63 of 236 documents (27%) were alternative-only, affecting 48
        # of 73 papers. Penalising those is precisely the error the uniqueness
        # test exists to expose, so it must not be built into the scoring.
        union = sorted({d for mm in members for d in mm})
        set_rows.append({
            "id": pid,
            "question": question,
            "answer": "",
            # COMPATIBILITY ONLY. The consumer's `stratum` is never null, so an
            # unlabelled query still has to say something, and "same" is the
            # conservative default because it cannot inflate a cross-domain
            # claim. It is a placeholder, not a finding: `stratum_labelled`
            # below says whether it rests on anything, and build/score.py drops
            # unlabelled rows from every domain-stratified number rather than
            # counting them as same. Measured on the first export: 15 of 158
            # primary golds (9.5%) had no field label, affecting 8 of 66
            # queries, 6 of which had no labelled primary gold at all.
            "stratum": ("cross" if any(stratum_of(x) == "cross" for x in prim)
                        else "same"),
            # ANY member of ANY valid decomposition is relevant. This is the
            # floor: it removes the false negative but says nothing about
            # whether a COMPLETE explanation was recovered.
            "supporting_documents": union,
            "target_nodes": {"document": union},
            "quartet": {
                # WHAT THE ALTERNATIVES ARE ALTERNATIVES *TO*.
                #
                # Without this the released benchmark is a list of document ids
                # and the reader has to take on trust that two different sets
                # explain the same thing. The claim of this benchmark is that
                # M(b, h) has more than one member; h is the thing held fixed
                # across every member, so h has to ship with it. The deltas are
                # what each inspiration contributes, so a reader can follow
                #     background + delta_1 + ... + delta_k  ==  h
                # for any set in the family, and check the claim rather than
                # believe it.
                "hypothesis": r.get("fine_grained_hypothesis"),
                "research_question": r.get("research_question"),
                "background_survey": r.get("background_survey"),
                "hypothesis_components": r.get("hypothesis_components"),
                "deltas": deltas,
                "set_deltas": set_deltas,
                # The structure CompleteSet@k needs: a hit is a whole set inside
                # the top k, not one member of one.
                "sets": members,
                "primary_set": prim_ids,
                "n_sets": len(members),
                "unresolvable_alternatives": dropped,
                "uniqueness": (r.get("uniqueness") or {}).get("uniqueness"),
                "status_after_resolution":
                    (r.get("uniqueness") or {}).get("status_after_resolution"),
                "alternative_only": sorted(set(union) - set(prim_ids)),
                # Does the row-level `stratum` rest on any evidence at all, and
                # on evidence for EVERY primary gold? A query with one labelled
                # "same" gold and one unlabelled gold is not known to be
                # same-domain, so the strict slice needs both flags.
                "stratum_labelled": any(stratum_of(x) is not None for x in prim),
                "stratum_complete": all(stratum_of(x) is not None for x in prim),
                "n_unlabelled_primary": sum(1 for x in prim
                                            if stratum_of(x) is None),
                "per_document": meta,
                "label_policy": next((x.get("label_policy") for x in prim
                                      if x.get("label_policy")), None),
            },
        })
        if members:
            u = r.get("uniqueness") or {}
            sets_out[pid] = {
                "sets": members,
                "n_valid_sets": len(members),
                "unresolvable_alternatives": dropped,
                # Both verdicts. `uniqueness` is Stage 3's, decided on text.
                # `status_after_resolution` is that verdict restated against the
                # decompositions whose papers actually resolved, and it can read
                # "resolution_incomplete": fewer than two alternatives survived
                # lookup, which is a fact about OpenAlex and not evidence that the
                # decomposition is unique. Only ever report the second.
                "uniqueness": u.get("uniqueness"),
                "status_after_resolution": u.get("status_after_resolution"),
                "budget": u.get("budget"),
            }

    # -------------------------------------------------------------------- report
    n_q = len({row["id"].rsplit("::", 1)[0] for row in rows})
    print(f"\n{'=' * 68}\nEXPORT\n")
    print(f"  papers in                 {len(recs):>6}")
    print(f"  decomposed                {len(ok):>6}")
    print(f"  queries exported          {n_q:>6}")
    print(f"  eval rows                 {len(rows):>6}   "
          f"({len(rows) / max(n_q, 1):.2f} golds per query)")
    print(f"  corpus documents          {len(documents):>6}")
    print(f"    of which title-only     {len(title_only):>6}   "
          f"({100 * len(title_only) / max(len(documents), 1):.0f}%)")

    strat = Counter(row["stratum"] for row in rows)
    unlab = sum(1 for row in rows if not row["quartet"]["stratum_labelled"])
    print(f"\n  stratum   same {strat['same']:>5}   cross {strat['cross']:>5}"
          f"   (of which {unlab} unlabelled, defaulted to same)")
    band = Counter(row["quartet"]["domain_distance"] for row in rows
                   if row["quartet"]["domain_distance"])
    for k, v in band.most_common():
        print(f"    {k:<32}{v:>5}")

    lb = Counter(row["quartet"].get("label_basis") for row in rows)
    if lb["primary_fallback"]:
        fbc = Counter(row["stratum"] for row in rows
                      if row["quartet"].get("label_basis") == "primary_fallback")
        print(f"\n  label basis:  topics {lb['topics']}   "
              f"primary_fallback {lb['primary_fallback']} "
              f"(same {fbc['same']}, cross {fbc['cross']})   "
              f"none {lb[None]}")
        print(f"    the fallback rule is coarser; filter "
              f"quartet.label_basis == 'topics' for a clean split")

    conf = Counter(row["quartet"]["field_confidence"] for row in rows)
    amb_cross = sum(1 for row in rows if row["stratum"] == "cross"
                    and row["quartet"]["field_confidence"] == "ambiguous")
    n_unlab = sum(1 for r in set_rows if not r["quartet"]["stratum_labelled"])
    n_part = sum(1 for r in set_rows if r["quartet"]["stratum_labelled"]
                 and not r["quartet"]["stratum_complete"])
    rows_unlab = sum(1 for row in rows if not row["quartet"]["stratum_labelled"])
    print(f"\n  STRATUM EVIDENCE")
    print(f"    primary gold rows with no field label  {rows_unlab:>4}"
          f"  ({100 * rows_unlab / max(len(rows), 1):.1f}%)")
    print(f"    queries with NO labelled primary gold  {n_unlab:>4}"
          f"  ({100 * n_unlab / max(len(set_rows), 1):.1f}%)")
    print(f"    queries labelled only in part          {n_part:>4}")
    print(f"    -> those default to stratum='same' for compatibility and are")
    print(f"       EXCLUDED from same/cross results by build/score.py")

    print(f"\n  field label confidence:  "
          f"clear {conf['clear']}   ambiguous {conf['ambiguous']}   "
          f"unknown {conf[None]}")
    if strat["cross"]:
        print(f"    {amb_cross}/{strat['cross']} cross rows rest on an ambiguous "
              f"field; take the split over field_confidence == 'clear'")

    alt_only = {d for r in set_rows for d in r["quartet"]["alternative_only"]}
    pen = sum(1 for r in set_rows if r["quartet"]["alternative_only"])
    nsets = Counter(r["quartet"]["n_sets"] for r in set_rows)
    print(f"\n  SET-AWARE eval.json       {len(set_rows):>6} queries")
    print(f"    valid sets per query    {dict(sorted(nsets.items()))}")
    print(f"    documents reachable via an alternative but NOT in D1:")
    print(f"      {len(alt_only):>6} documents ({100 * len(alt_only) / max(len(documents), 1):.0f}% of the corpus)")
    print(f"      {pen:>6} of {len(set_rows)} queries affected")
    print(f"    the flat eval_primary.json scores every one of those as a MISS,")
    print(f"    which is why it is not the headline file")

    print(f"\n  papers with a family      {len(sets_out):>6}")
    if sets_out:
        tot = sum(s["n_valid_sets"] for s in sets_out.values())
        drop = sum(s["unresolvable_alternatives"] for s in sets_out.values())
        after = Counter(s["status_after_resolution"] for s in sets_out.values())
        print(f"    valid sets exported     {tot:>6}   "
              f"({tot / len(sets_out):.2f} per paper)")
        print(f"    alternatives dropped    {drop:>6}   (Stage 4 could not resolve them)")
        print(f"    uniqueness, AFTER resolution:")
        for k, v in after.most_common():
            print(f"      {str(k):<30}{v:>5}/{len(sets_out)}")

    if excluded:
        print(f"\n  EXCLUDED:")
        for k, v in sorted(excluded.items(), key=lambda kv: -len(kv[1])):
            print(f"    {k:<24}{len(v):>5}")
        for doi, why in excluded.get("primary_incomplete", [])[:8]:
            print(f"        {why:<26}{doi}")

    if not rows:
        print("\nnothing to export", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------- write
    if a.dry_run:
        print("\n(dry run, nothing written)")
        return
    (a.out / "raw").mkdir(parents=True, exist_ok=True)
    (a.out / "raw" / "documents.json").write_text(json.dumps(documents, ensure_ascii=False))
    # eval.json is the SET-AWARE file and the QUARTET evaluation.
    # eval_primary.json is D1 only, in TOMATO-Star's flat shape, for the
    # comparability claim and for nothing else.
    (a.out / "eval.json").write_text(json.dumps(set_rows, ensure_ascii=False))
    (a.out / "eval_primary.json").write_text(json.dumps(rows, ensure_ascii=False))
    (a.out / "sets.json").write_text(json.dumps(sets_out, ensure_ascii=False, indent=1))
    (a.out / "manifest.json").write_text(json.dumps({
        "split": a.split or a.input.stem,
        "source": str(a.input),
        "papers_in": len(recs), "decomposed": len(ok),
        "queries": n_q, "eval_rows": len(rows),
        "documents": len(documents), "title_only_documents": len(title_only),
        "families": len(sets_out),
        "set_aware_queries": len(set_rows),
        "alternative_only_documents": len(alt_only),
        "queries_with_alternative_only_golds": pen,
        "eval_files": {"eval.json": "set-aware, one row per query, THE evaluation",
                       "eval_primary.json": "D1 only, flat, TOMATO-comparable"},
        "stratum": dict(strat), "domain_distance": dict(band),
        "field_confidence": {str(k): v for k, v in conf.items()},
        "label_basis": {str(k): v for k, v in lb.items()},
        "cross_rows_on_ambiguous_field": amb_cross,
        "queries_without_stratum_evidence": n_unlab,
        "queries_partially_labelled": n_part,
        "primary_rows_without_stratum_evidence": rows_unlab,
        "uniqueness_after_resolution": {
            str(k): v for k, v in
            Counter(s["status_after_resolution"] for s in sets_out.values()).items()},
        "excluded": {k: len(v) for k, v in excluded.items()},
        "options": {"drop_title_only": a.drop_title_only,
                    "keep_partial": a.keep_partial},
    }, indent=1))
    print(f"\nwrote -> {a.out}")
    for f in ("raw/documents.json", "eval.json", "eval_primary.json",
              "sets.json", "manifest.json"):
        print(f"    {f:<24}{(a.out / f).stat().st_size:>10,} bytes")


if __name__ == "__main__":
    main()
