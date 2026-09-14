"""Build dev_strata.json (cross/same labels for the dev bundle) from the EXISTING train domain
labels — no API calls.  Mirrors the test cross definition: cross = BIO source AND a DIST gold.

A dev query is one paper (multi-gold).  We label it 'cross' if its OpenAlex source domain is
biomedical (Life/Health Sciences) AND at least one of its gold inspirations is in a distant
domain (Physical/Social Sciences) -- the same BIO->DIST jump used for the test split.

Inputs (already on disk, both train-split-covered):
  domains/source_domains.jsonl        (source_id -> openalex_domain)
  domains/inspiration_domains.jsonl   ((found_doi/title) -> openalex_domain)
  outputs/caches/dev_bundle_big.json  (the carved dev queries, in order)

Run:  cd CARGO && python -m cargo.build_dev_strata
Out:  outputs/caches/dev_strata.json   {dev_query_index: "cross"|"same"}  (upload to Drive next to the bundles)
"""
from __future__ import annotations
import json
from collections import Counter
from tqdm import tqdm
from .config import CACHE
from .data import SRC_DOMAINS, INSP_DOMAINS, BIOMEDICAL, DISTANT


def _key(found_doi: str | None, found_title: str | None) -> str | None:
    return (found_doi or "").strip().lower() or (found_title or "").strip().lower() or None


def _grp(d):
    return "BIO" if d in BIOMEDICAL else ("DIST" if d in DISTANT else None)


def main():
    # 1. source domain per train paper
    src_dom: dict[str, str | None] = {}
    for line in tqdm(open(SRC_DOMAINS), desc="source_domains"):
        r = json.loads(line)
        if r.get("split") == "train":
            src_dom[r["source_id"]] = r.get("openalex_domain")
    print(f"[dev_strata] {len(src_dom)} train source domains")

    # 2. gold-key -> domain (train inspirations), keyed exactly like build_train._key()
    gold_dom: dict[str, str | None] = {}
    for line in tqdm(open(INSP_DOMAINS), desc="inspiration_domains"):
        r = json.loads(line)
        if r.get("split") != "train":
            continue
        k = _key(r.get("found_doi"), r.get("found_title"))
        if k and r.get("openalex_domain"):
            gold_dom.setdefault(k, r["openalex_domain"])
    print(f"[dev_strata] {len(gold_dom)} train inspiration-key domains")

    # 3. label each dev query (paper): cross = BIO source AND >=1 DIST gold
    bundle = json.load(open(CACHE / "dev_bundle_big.json"))
    examples = bundle["examples"]
    strata: dict[str, str] = {}
    miss_src = miss_all_gold = 0
    for i, ex in enumerate(tqdm(examples, desc="label dev")):
        sg = _grp(src_dom.get(ex["source_id"]))
        if sg is None:
            miss_src += 1
        gold_groups = [_grp(gold_dom.get(k)) for k in ex["gold_ids"]]
        if all(g is None for g in gold_groups):
            miss_all_gold += 1
        is_cross = (sg == "BIO") and any(g == "DIST" for g in gold_groups)
        strata[str(i)] = "cross" if is_cross else "same"

    out = CACHE / "dev_strata.json"
    json.dump(strata, open(out, "w"))
    c = Counter(strata.values())
    print(f"[dev_strata] {len(examples)} dev queries -> cross {c['cross']} / same {c['same']}")
    print(f"[dev_strata] unresolved: source-domain {miss_src}, all-golds-domainless {miss_all_gold} "
          f"(these default to 'same')")
    print(f"[dev_strata] wrote -> {out}")


if __name__ == "__main__":
    main()
