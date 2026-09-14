"""Resolve gold-inspiration domains OpenAlex couldn't find (openalex_found=False).

These are mostly arXiv / preprint / very-recent papers absent from OpenAlex. We classify
each into one of the 4 OpenAlex top-level domains from its title+abstract via gpt-4o-mini,
so they stop being dropped from the same/cross split.

Writes: outputs/caches/gold_domain_overrides.json  { gold_key: domain }
Run:    cd CARGO && OPENAI_API_KEY=... python -m cargo.resolve
Cost:   ~40 calls, well under $0.05.
"""
from __future__ import annotations
import os, json, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import CACHE
from .data import build

DOMAINS = ["Life Sciences", "Health Sciences", "Physical Sciences", "Social Sciences"]
SYS = (
    "You classify a scientific paper into EXACTLY ONE OpenAlex top-level domain.\n"
    "- 'Life Sciences': biology, biochemistry, molecular/cell biology, neuroscience, genetics, immunology, physiology.\n"
    "- 'Health Sciences': clinical medicine, surgery, public health, nursing, pharmacy, dentistry, epidemiology.\n"
    "- 'Physical Sciences': computer science, AI/ML/deep learning, engineering, mathematics, statistics, physics, chemistry, materials.\n"
    "- 'Social Sciences': psychology, economics, sociology, education, political science.\n"
    'Return JSON: {"domain": "<one of the four exact strings>"}.'
)


def _classify(client, key: str, text: str):
    for _ in range(4):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini", temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": text[:1600]}])
            d = json.loads(r.choices[0].message.content).get("domain", "").strip()
            if d in DOMAINS:
                return key, d
        except Exception:
            time.sleep(2)
    return key, None


def main():
    from openai import OpenAI
    queries, corpus, golds = build(use_cache=True)
    none_keys = sorted({q["gold_key"] for q in queries if q["gold_domain"] is None})
    print(f"[resolve] {len(none_keys)} unresolved gold inspirations to classify")
    if not none_keys:
        print("[resolve] nothing to do."); return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_classify, client, k, corpus.get(k, "")) for k in none_keys]
        for f in tqdm(as_completed(futs), total=len(futs), desc="classify golds"):
            k, d = f.result()
            if d:
                out[k] = d

    path = CACHE / "gold_domain_overrides.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[resolve] resolved {len(out)}/{len(none_keys)} -> {path.name}")
    print(f"[resolve] domains: {dict(Counter(out.values()))}")
    print("[resolve] now rerun:  python -m cargo.data   (rebuild)  then  python -m scripts.03_baselines")


if __name__ == "__main__":
    main()
