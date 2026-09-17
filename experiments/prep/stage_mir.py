"""
stage_mir.py -- Phase 0 for MIR (Methodology Inspiration Retrieval, ACL 2025): turn the
released CSVs into the corpus + query layout every pipeline script reads via scigraphir_paths.

WHAT MIR IS. One row per (citing paper, cited paper) citation. The citing paper carries a
`proposal` (problem + motivation, no method) and the cited paper carries an abstract. A
citation is an INSPIRATION when its MultiCite intent is `Uses` or `Extention`, which the
release marks as `inspirational == 1`. That is the paper's own gold definition and it is
kept exactly: the notebook never re-derives it from intents.

WHAT A QUERY IS. One citing paper. Its proposal text is identical across every row of that
paper (verified: 0 papers with more than one proposal text in any split), so the query id is
the citing paper id and the question is the proposal. `gen_id` (s2 / llm) is how a cited
paper's id was resolved and is ignored.

WHAT THE CORPORA ARE. Two of MIR's own settings, selected with --test-corpus:
    extended    every paper cited by any training, dev or test proposal (~4.9k docs). The
                paper's "extended corpus" is training-cited plus test golds; adding the
                non-gold test citations gives natural hard negatives for free and is the
                default here.
    restricted  only the papers cited by test proposals (284 docs). Matches the paper's
                restricted-corpus numbers; too small to call retrieval.
The training corpus is every paper cited by a training proposal.

--train-source augmented (default) uses MIR's augmented training file: 1,511 proposals with
4,688 cited papers, versus 800 / 1,232 in the original train split. Same construction,
extended to 2024, and the file MIR's follow-up work trains on.

DOCUMENTS ARE ABSTRACTS ONLY. The release carries no titles. Ten cited ids have a junk
abstract (a single punctuation mark); those documents are dropped and removed from every
gold list, and any query left without a gold is dropped and counted.

SINGLE FIELD. Everything is computational linguistics, so every query gets stratum "same":
the scorer's `all` and `same` slices coincide and there is no cross-field slice.

Usage
-----
    python3 prep/stage_mir.py                       # augmented train, extended test corpus
    python3 prep/stage_mir.py --verify-only
    python3 prep/stage_mir.py --train-source train --test-corpus restricted
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # experiments/
REPO_ROOT = os.path.dirname(ROOT)
SRC = f"{ROOT}/data/mir"                     # the four CSVs from the MIR GitHub repository
KG_DATA = f"{REPO_ROOT}/retriever/data"    # where scigraphir_paths.corpus_dir(split) resolves
MIN_ABSTRACT = 100                           # chars; below this the "abstract" is punctuation

FILES = {"train": "train_chronological_df.csv", "augmented": "augmented_train_chronological.csv",
         "dev": "dev_chronological_df.csv", "test": "test_chronological_df.csv"}


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(f"{SRC}/{FILES[name]}")
    if "citing_paper_id" in df.columns:          # the augmented file names the column differently
        df = df.rename(columns={"citing_paper_id": "paper_id"})
    for c in ("paper_id", "cited_paper_id", "proposal", "cited_paper_abstracts", "inspirational"):
        assert c in df.columns, f"{FILES[name]} lacks column {c}"
    df["cited_paper_abstracts"] = df["cited_paper_abstracts"].astype(str)
    df["proposal"] = df["proposal"].astype(str)
    return df


def corpus_of(*dfs: pd.DataFrame) -> tuple[dict, list]:
    """{cited id: abstract} over the given affordance representations, dropping junk abstracts. Returns (corpus, dropped ids)."""
    corpus, dropped = {}, []
    for df in dfs:
        for cid, txt in df.groupby("cited_paper_id")["cited_paper_abstracts"].first().items():
            txt = txt.strip()
            if len(txt) < MIN_ABSTRACT:
                dropped.append(cid)
                continue
            corpus[cid] = txt
    return corpus, sorted(set(dropped))


def queries_of(df: pd.DataFrame, corpus: dict) -> tuple[list, dict]:
    """One query per citing paper: proposal text, unique inspirational citations as golds."""
    out, stats = [], {"papers": df["paper_id"].nunique(), "no_gold": 0, "gold_dropped_junk": 0}
    for pid, g in df.groupby("paper_id", sort=False):
        props = g["proposal"].unique()
        assert len(props) == 1, f"{pid}: {len(props)} distinct proposal texts"
        golds = sorted(set(g.loc[g["inspirational"] == 1, "cited_paper_id"]))
        kept = [d for d in golds if d in corpus]
        stats["gold_dropped_junk"] += len(golds) - len(kept)
        if not kept:
            stats["no_gold"] += 1
            continue
        out.append({"id": pid, "question": props[0].strip(), "answer": "",
                    "supporting_documents": kept, "stratum": "same"})
    return out, stats


def verify(corpus: dict, queries: list, tag: str) -> list[str]:
    problems = []
    ids = [q["id"] for q in queries]
    if len(ids) != len(set(ids)):
        problems.append(f"{tag}: duplicate query ids")
    if any(not q["question"] for q in queries):
        problems.append(f"{tag}: empty question")
    if any(not t for t in corpus.values()):
        problems.append(f"{tag}: empty document")
    missing = [q["id"] for q in queries if any(g not in corpus for g in q["supporting_documents"])]
    if missing:
        problems.append(f"{tag}: {len(missing)} queries have a gold outside the corpus")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-source", default="augmented", choices=["train", "augmented"])
    ap.add_argument("--test-corpus", default="extended", choices=["extended", "restricted"])
    ap.add_argument("--verify-only", action="store_true", help="report, write nothing")
    a = ap.parse_args()
    for f in FILES.values():
        assert os.path.exists(f"{SRC}/{f}"), f"missing {SRC}/{f}; download the MIR CSVs first"

    tr, dv, te = load(a.train_source), load("dev"), load("test")

    train_corpus, junk_tr = corpus_of(tr)
    if a.test_corpus == "extended":
        test_corpus, junk_te = corpus_of(tr, dv, te)
    else:
        test_corpus, junk_te = corpus_of(te)
    train_q, st_tr = queries_of(tr, train_corpus)
    test_q, st_te = queries_of(te, test_corpus)

    stats = {}
    for split, corpus, qs, st, junk in (("train", train_corpus, train_q, st_tr, junk_tr),
                                        ("test", test_corpus, test_q, st_te, junk_te)):
        golds = [len(q["supporting_documents"]) for q in qs]
        gold_ids = {g for q in qs for g in q["supporting_documents"]}
        stats[split] = {"documents": len(corpus), "queries": len(qs), "citing_papers": st["papers"],
                        "dropped_no_gold": st["no_gold"], "golds_dropped_junk_abstract": st["gold_dropped_junk"],
                        "junk_documents_dropped": len(junk),
                        "golds_per_query": round(sum(golds) / len(golds), 3), "max_golds": max(golds),
                        "distinct_gold_docs": len(gold_ids),
                        "mean_doc_chars": round(sum(len(t) for t in corpus.values()) / len(corpus)),
                        "mean_query_chars": round(sum(len(q["question"]) for q in qs) / len(qs))}
        print(f"[{split:5}] {len(corpus):>6,} docs | {len(qs):>5,} queries of {st['papers']:,} citing papers "
              f"({st['no_gold']} dropped: no gold) | golds/query {stats[split]['golds_per_query']} "
              f"(max {max(golds)}) | junk docs dropped {len(junk)}")
    overlap = len(set(train_corpus) & set(test_corpus))
    tq = {q["id"] for q in train_q}; leak = [q["id"] for q in test_q if q["id"] in tq]
    print(f"test corpus ∩ train corpus: {overlap:,} docs (shared candidates, expected) | "
          f"test queries also in train: {len(leak)} (must be 0)")
    assert not leak, "test citing papers appear in training"

    problems = verify(train_corpus, train_q, "train") + verify(test_corpus, test_q, "test")
    if problems:
        print("\n".join("PROBLEM: " + p for p in problems), file=sys.stderr)
        return 1
    if a.verify_only:
        print("verify-only: nothing written"); return 0

    for split, corpus, qs in (("train", train_corpus, train_q), ("test", test_corpus, test_q)):
        raw = f"{KG_DATA}/mir_{split}/raw"
        os.makedirs(raw, exist_ok=True)
        json.dump(corpus, open(f"{raw}/documents.json", "w"))
        json.dump(qs, open(f"{raw}/{split}.json", "w"))
        print(f"wrote {raw}/documents.json + {split}.json")
    manifest = {"dataset": "mir", "source": "MIR (Garikaparthi et al., ACL 2025), github.com/Anikethh/Methodology-Inspiration-Retrieval",
                "licence": "CC BY-NC-SA 4.0", "train_source": a.train_source, "test_corpus": a.test_corpus,
                "gold_rule": "inspirational == 1 (MultiCite intents Uses / Extention)",
                "query_text": "proposal (problem + motivation, no method)", "document_text": "abstract only (no titles in the release)",
                "min_abstract_chars": MIN_ABSTRACT, "stratum": "all 'same' (single field)", "splits": stats}
    json.dump(manifest, open(f"{KG_DATA}/mir_test/MANIFEST.json", "w"), indent=1)
    json.dump(manifest, open(f"{KG_DATA}/mir_train/MANIFEST.json", "w"), indent=1)
    print("MANIFEST written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
