"""
tomato_index.py — build the contamination index that the ResearchBench transfer claim rests on.

We train on TOMATO-Star and evaluate frozen on ResearchBench. Any ResearchBench document the
model has already seen in training makes "zero-shot" false, so we have to be able to name the
overlap exactly rather than assert it is zero.

Three layers, from narrowest to widest. Report all three; the honest headline is L3.

  L1  graph corpus     the documents.json the v16sc graph was actually built over (7,000 docs)
  L2  training bundle  the wider document pool the training examples were sampled from
  L3  full TOMATO-Star every source paper and every retrieved inspiration in data/train.jsonl,
                       i.e. everything the *dataset* covers, whether or not our run touched it

L1/L2 store documents as "Title. Abstract" with an unreliable separator (titles that already
end in a period produce ".."; 19% of the v16sc train corpus has no ".. " at all), so their
titles cannot be recovered by splitting. But they are *keyed by DOI*, and L1/L2 are subsets of
TOMATO-Star, so we never need their titles: L3 gives a clean title -> DOI map, and layer
membership is then an exact DOI lookup. No prefix or fuzzy matching anywhere in the primary
path -- an earlier prefix-matching version returned false positives on short generic titles
("Functional Analysis") and on a Chinese-language title that normalised to two characters.

Output: a JSON index mapping normalised title keys to DOIs, plus per-layer DOI sets. Cached so
the 1.9 GB stream runs once.

Usage:
  python researchbench/tomato_index.py --out data/researchbench/tomato_index.json
"""
import argparse
import json
import os
import sys

from norm import norm_title, norm_doi

BASE = (os.environ.get("SCIGRAPHIR_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
TOMATO_STAR = os.path.join(os.environ.get("EXTERNAL_REPOS") or os.path.join(BASE, "external"), "TOMATO-Star")

# L1: the corpus the v16sc graph was built over -- what the model provably saw.
L1_PATHS = [
    f"{BASE}/drive_data_v16sc/tomato_train_v16sc/raw/documents.json",
    f"{BASE}/retriever/data/tomato_train/raw/documents.json",
]
# L2: the wider bundle training examples were drawn from.
L2_PATHS = [
    f"{BASE}/outputs/caches/train_bundle_big.json",
]
# L3: the full upstream dataset.
L3_PATHS = [
    f"{TOMATO_STAR}/data/train.jsonl",
]


def load_l1_l2(paths, label):
    """documents.json is {doi: 'Title. Abstract'}; train_bundle_big is {documents: [{id, content}]}.

    We keep only the DOI keys. Titles are recovered through L3, so the unreliable
    title/abstract separator never matters. Also keep the normalised texts, which the
    consumer may use as a secondary recall check for papers absent from L3.
    """
    for path in paths:
        if not os.path.exists(path):
            continue
        blob = json.load(open(path))
        if isinstance(blob, dict) and "documents" in blob:
            items = [(d["id"], d["content"]) for d in blob["documents"]]
        else:
            items = list(blob.items())
        print(f"[{label}] {path}  ({len(items):,} docs)")
        return {norm_doi(k) for k, _ in items}, [norm_title(v) for _, v in items]
    print(f"[{label}] !! not found, skipped: {paths}")
    return set(), []


def load_l3(paths):
    """Stream TOMATO-Star train.jsonl into a title_key -> [doi] map.

    This is the join that lets L1/L2 membership be an exact DOI lookup rather than a
    guess at where a title ends.
    """
    for path in paths:
        if not os.path.exists(path):
            continue
        size = os.path.getsize(path)
        print(f"[L3] streaming {path}  ({size/1e9:.2f} GB) ...")
        dois = set()
        title2doi = {}          # normalised title -> set of DOIs
        n = 0
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1

                def add(title, doi):
                    tk, dk = norm_title(title), norm_doi(doi)
                    if dk:
                        dois.add(dk)
                    if tk:
                        title2doi.setdefault(tk, set()).add(dk)

                # the source paper itself
                add(row.get("title"), row.get("doi"))
                # its retrieved inspirations -- these are documents in the retrieval corpus.
                # supposed_title is the LLM's guess at the cited paper; found_title is what
                # Semantic Scholar actually returned. Index both: a ResearchBench document
                # matching either has been seen.
                insp = row.get("inspiration")
                if isinstance(insp, str):
                    try:
                        insp = json.loads(insp)
                    except json.JSONDecodeError:
                        insp = []
                for i in insp or []:
                    if not isinstance(i, dict):
                        continue
                    add(i.get("found_title"), i.get("found_doi"))
                    add(i.get("supposed_title"), i.get("found_doi"))
                if n % 20000 == 0:
                    print(f"       {n:,} rows  |  {len(title2doi):,} titles  {len(dois):,} dois",
                          flush=True)
        title2doi.pop("", None)
        dois.discard("")
        print(f"[L3] {n:,} source papers -> {len(title2doi):,} titles, {len(dois):,} DOIs")
        return dois, title2doi
    print(f"[L3] !! not found, skipped: {paths}")
    return set(), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{BASE}/retriever/data/researchbench/tomato_index.json")
    a = ap.parse_args()

    l1_dois, l1_texts = load_l1_l2(L1_PATHS, "L1")
    l2_dois, l2_texts = load_l1_l2(L2_PATHS, "L2")
    l3_dois, l3_title2doi = load_l3(L3_PATHS)

    if not (l1_texts or l2_texts or l3_title2doi):
        sys.exit("no TOMATO source found -- cannot build a contamination index")

    resolved = sum(1 for v in l3_title2doi.values() if any(v))
    print(f"[L3] titles with at least one DOI: {resolved:,} / {len(l3_title2doi):,}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({
        # primary path: exact title match against L3, then DOI lookup into L1/L2.
        "l3_title2doi": {k: sorted(x for x in v if x) for k, v in l3_title2doi.items()},
        "l3_dois": sorted(l3_dois),
        "l1_dois": sorted(l1_dois),
        "l2_dois": sorted(l2_dois),
        # secondary recall check only, for pool documents absent from L3. Consumers must
        # apply a minimum key length; short generic titles prefix-match spuriously.
        "l1_texts": l1_texts, "l2_texts": l2_texts,
        "sources": {"l1": L1_PATHS, "l2": L2_PATHS, "l3": L3_PATHS},
    }, open(a.out, "w"))
    print(f"\n[wrote] {a.out}")
    print(f"  L1 {len(l1_texts):,} docs / {len(l1_dois):,} dois")
    print(f"  L2 {len(l2_texts):,} docs / {len(l2_dois):,} dois")
    print(f"  L3 {len(l3_title2doi):,} titles / {len(l3_dois):,} dois")


if __name__ == "__main__":
    main()
