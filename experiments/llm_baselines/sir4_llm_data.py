"""
sir4_llm_data.py -- one loader for the three LLM-retrieval baselines on SIR-4.

Each SIR-4 field ships retriever/data/sir4_<field>_test/raw/{documents.json, test.json}:
    documents.json  {doi: "Title. Abstract"}          (10-11% of docs are title-only)
    test.json       [{id, question, supporting_documents:[doi,...], stratum:"same"|"cross"}]

The three runners were written for TOMATO-Star, whose query is a research question plus a
background survey and whose corpus is (title, abstract) pairs keyed by DOI. This module
presents SIR-4 the same way:

    query   research_question = text up to and including the first "?"
            background_survey = the rest (every SIR-4 question has this shape; the survey
            paragraph starts right after the question mark)
    corpus  title = text before the first ". ", abstract = the rest ("" when title-only).
            Titles are made unique (MOOSE-Chem and MOOSE-Star key on the title) by
            suffixing " (2)", " (3)" to the 1-5 duplicates per field.

Nothing here calls an API.
"""
from __future__ import annotations

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
S4 = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(S4)
DATA_ROOT = f"{REPO_ROOT}/retriever/data"
FIELDS = ["cs", "biology", "physics", "matsci"]

# Sibling repos the runners import from (edit here if they move).
PROJECT_WEEK9 = os.environ.get("EXTERNAL_REPOS") or os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "external")
TOMATO_STAR_DIR = f"{PROJECT_WEEK9}/TOMATO-Star"
MOOSE_CHEM_DIR = f"{PROJECT_WEEK9}/MOOSE-Chem"
MOOSE_STAR_DIR = f"{PROJECT_WEEK9}/MOOSE-Star"
LATTICE_DIR = f"{PROJECT_WEEK9}/llm-guided-hierarchical-search"

OUT_ROOT = f"{HERE}/outputs"          # all rankings / trees / scores land under here


# `mir` is accepted everywhere a field is: MIR (ACL 2025) staged by prep/stage_mir.py. Its
# corpus is abstracts only, so the "title" the runners key on is the abstract's first
# sentence; every query is stratum "same"; there is no subset (155 test proposals, run all).
def _ds(field: str) -> str:
    return "mir_test" if field == "mir" else f"sir4_{field}_test"


def queries_path(field: str) -> str:
    return f"{DATA_ROOT}/{_ds(field)}/raw/test.json"


def corpus_path(field: str) -> str:
    return f"{DATA_ROOT}/{_ds(field)}/raw/documents.json"


def split_title_abstract(text: str) -> tuple[str, str]:
    text = text.strip()
    title, sep, abstract = text.partition(". ")
    if not sep:
        return text.rstrip("."), ""
    return title, abstract.strip()


def mir_title_abstract(text: str) -> tuple[str, str]:
    """MIR ships abstracts without titles. The runners need a title (MOOSE-Star keys and
    embeds on it), so the stand-in is the first sentence when it has >= 6 words, else the
    first 12 words; the abstract is the WHOLE text, so no content is lost to the split."""
    text = text.strip().lstrip(". ;:-").strip()
    first = text.partition(". ")[0].strip()
    title = first if len(first.split()) >= 6 else " ".join(text.split()[:12])
    return title, text


def split_question(question: str) -> tuple[str, str]:
    """research_question (through the first '?'), background_survey (the rest)."""
    q = question.strip()
    i = q.find("?")
    if i < 0:
        return q, ""
    return q[: i + 1].strip(), q[i + 1:].strip()


def load_corpus(field: str) -> dict[str, tuple[str, str]]:
    """{doi: (unique_title, abstract)} in sorted-DOI order."""
    raw = json.load(open(corpus_path(field)))
    seen: Counter = Counter()
    out: dict[str, tuple[str, str]] = {}
    for doi in sorted(raw):
        if field == "mir":
            title, abstract = mir_title_abstract(raw[doi])
        else:
            title, abstract = split_title_abstract(raw[doi])
        key = title.lower()
        seen[key] += 1
        if seen[key] > 1:
            title = f"{title} ({seen[key]})"
        out[doi] = (title, abstract)
    return out


def load_queries(field: str) -> list[dict]:
    """[{query_id, research_question, background_survey, question, golds, stratum}]"""
    out = []
    for q in json.load(open(queries_path(field))):
        rq, bs = split_question(q["question"])
        out.append({"query_id": q["id"], "research_question": rq, "background_survey": bs,
                    "question": q["question"], "golds": list(q["supporting_documents"]),
                    "stratum": q["stratum"]})
    return out


def load_subset(path: str | None, field: str, queries: list[dict]) -> list[dict]:
    """Restrict to a make_sir4_subset.py manifest (manifest order); None = every query."""
    if path is None:
        return queries
    man = json.load(open(path))
    assert man["field"] == field, f"{path} is for {man['field']}, not {field}"
    by_id = {q["query_id"]: q for q in queries}
    return [by_id[i] for i in man["query_ids"]]


def default_subset(field: str) -> str | None:
    if field == "mir":
        return None                                   # 155 queries: always the full set
    return f"{HERE}/subsets/subset_{field}_same250_allcross_seed42.json"


def check(field: str) -> None:
    corpus, queries = load_corpus(field), load_queries(field)
    titles = [t for t, _ in corpus.values()]
    assert len(set(titles)) == len(titles), "titles not unique after disambiguation"
    missing = sum(1 for q in queries for g in q["golds"] if g not in corpus)
    no_abs = sum(1 for _, a in corpus.values() if not a)
    no_bs = sum(1 for q in queries if not q["background_survey"])
    print(f"{field:8} docs {len(corpus):5} (title-only {no_abs:4})   queries {len(queries):5} "
          f"(same {sum(q['stratum']=='same' for q in queries)}, cross {sum(q['stratum']=='cross' for q in queries)})   "
          f"golds missing {missing}   queries without survey {no_bs}")


if __name__ == "__main__":
    import sys
    for f in (sys.argv[1:] or FIELDS):
        check(f)
