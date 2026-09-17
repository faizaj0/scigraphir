"""Resolve the TRAIN-split inspirations/sources OpenAlex left without a domain.

Mirror of retriever.tomato_star.resolve (which is test-only). The domain files leave a small residual
unlabelled (openalex_found=False or domain-null): ~2,016 gold inspirations + ~71 source papers.
We classify each into one of the 4 OpenAlex top-level domains from its title+abstract via
gpt-4o-mini (SAME prompt/model as retriever.tomato_star.resolve), so the train same/cross affordance representation carries no
mislabelled positives and cross is not under-counted.

Writes:
  outputs/caches/train_gold_domain_overrides.json    { gold_key:   domain }
  outputs/caches/train_source_domain_overrides.json  { source_id:  domain }
Text comes from data/train.jsonl (titles + abstracts). RESUMABLE: re-running loads the
existing override files and skips anything already done (also checkpoints every ~200).

Run:
  OPENAI_API_KEY=... python -m retriever.tomato_star.resolve_train          # the real run
  python -m retriever.tomato_star.resolve_train --dry-run                   # harvest + estimate only, no API
Then rebuild the affordance representation:
  python -m retriever.tomato_star.train_strata --split train --write
Cost: ~2,087 gpt-4o-mini calls, roughly $0.2-0.4.
"""
from __future__ import annotations
import argparse, json, os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import SRC_DOMAINS, INSP_DOMAINS, TRAIN_JSONL, CACHE
from .resolve import _classify  # identical prompt/model as the test resolver


def _gold_key(found_doi: str | None, found_title: str | None) -> str | None:
    """found_doi (lowercased) else found_title (lowercased) -- identical to retriever.tomato_star.data._norm_key."""
    doi = (found_doi or "").strip().lower()
    if doi:
        return doi
    t = (found_title or "").strip().lower()
    return t or None


def collect_unresolved(split: str = "train"):
    """Unique gold_keys + source_ids with no OpenAlex domain in the precomputed files."""
    gold_keys: set[str] = set()
    with INSP_DOMAINS.open() as f:
        for line in tqdm(f, desc="scan inspiration_domains"):
            r = json.loads(line)
            if r.get("split") != split or r.get("openalex_domain") is not None:
                continue
            k = _gold_key(r.get("found_doi"), r.get("found_title"))
            if k:
                gold_keys.add(k)
    src_ids: set[str] = set()
    with SRC_DOMAINS.open() as f:
        for line in tqdm(f, desc="scan source_domains"):
            r = json.loads(line)
            if r.get("split") != split or r.get("openalex_domain") is not None:
                continue
            src_ids.add(r["source_id"])
    print(f"[collect] unresolved: {len(gold_keys)} gold keys (unique), {len(src_ids)} source papers")
    return gold_keys, src_ids


def build_text_maps(gold_keys: set[str], src_ids: set[str]):
    """Stream train.jsonl ONCE; pull title+abstract for the needed golds and sources only."""
    gold_text: dict[str, str] = {}
    src_text: dict[str, str] = {}
    with TRAIN_JSONL.open() as f:
        for line in tqdm(f, desc="read train.jsonl (titles+abstracts)"):
            paper = json.loads(line)
            sid = paper.get("source_id")
            if sid in src_ids and sid not in src_text:
                src_text[sid] = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).strip()
            insps = paper.get("inspiration", [])
            if isinstance(insps, str):
                try: insps = json.loads(insps)
                except Exception: insps = []
            for insp in insps:
                k = _gold_key(insp.get("found_doi"), insp.get("found_title"))
                if k in gold_keys and k not in gold_text:
                    gold_text[k] = ((insp.get("found_title") or "") + " " +
                                    (insp.get("found_abstract") or "")).strip()
    miss_g, miss_s = len(gold_keys) - len(gold_text), len(src_ids) - len(src_text)
    print(f"[text] golds with text {len(gold_text)}/{len(gold_keys)} (missing {miss_g}); "
          f"sources {len(src_text)}/{len(src_ids)} (missing {miss_s})")
    if miss_g or miss_s:
        print("[text] missing items have no text in train.jsonl -> cannot classify (left unresolved)")
    return gold_text, src_text


def _resolve(client, text_map: dict[str, str], out_path, label: str, workers: int) -> dict:
    done: dict[str, str] = {}
    if out_path.exists():
        done = json.loads(out_path.read_text())
        print(f"[{label}] resuming from {len(done)} already-resolved")
    todo = {k: t for k, t in text_map.items() if t and k not in done}
    print(f"[{label}] classifying {len(todo)} (skipping {len(text_map) - len(todo)})")
    if not todo:
        return done
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_classify, client, k, t) for k, t in todo.items()]
        for i, fu in enumerate(tqdm(as_completed(futs), total=len(futs), desc=f"classify {label}")):
            k, d = fu.result()
            if d:
                done[k] = d
            if (i + 1) % 200 == 0:  # periodic checkpoint
                out_path.write_text(json.dumps(done, indent=2))
    out_path.write_text(json.dumps(done, indent=2))
    print(f"[{label}] resolved {len(done)}/{len(text_map)} -> {out_path.name}")
    print(f"[{label}] domain mix: {dict(Counter(done.values()))}")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--dry-run", action="store_true", help="harvest text + cost estimate, no API calls")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    gold_keys, src_ids = collect_unresolved(args.split)
    if not gold_keys and not src_ids:
        print("[resolve_train] nothing unresolved; done.")
        return
    gold_text, src_text = build_text_maps(gold_keys, src_ids)

    n = sum(1 for t in gold_text.values() if t) + sum(1 for t in src_text.values() if t)
    print(f"[estimate] ~{n} gpt-4o-mini calls -> roughly ${0.0001 * n:.2f}-${0.0002 * n:.2f}")
    if args.dry_run:
        print("[dry-run] stopping before any API call. Drop --dry-run (with OPENAI_API_KEY set) to resolve.")
        return

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    _resolve(client, gold_text, CACHE / f"{args.split}_gold_domain_overrides.json", "gold", args.max_workers)
    _resolve(client, src_text, CACHE / f"{args.split}_source_domain_overrides.json", "source", args.max_workers)
    print(f"[resolve_train] done. Now rerun:  python -m retriever.tomato_star.train_strata --split {args.split} --write")


if __name__ == "__main__":
    main()
