"""
Generate CARGO probes (methods + cross-field analogs) per query.

Faithful port of the CARGO operator-notebook probe prompt. Each query -> up to 8 short probe
passages:
  - 3-4 "methods": a SPECIFIC named technique the research needs, written as it would appear in
    that method paper's abstract;
  - 3-4 "analogs": a 1-2 sentence hypothetical abstract from ANOTHER field whose mechanism is
    structurally analogous (this is the cross-domain bridge).

Probes feed two downstream uses:
  (1) the dense OPERATOR  S_op = w0 z(dense) + w1 z(S/dem^b) + w2 z(M/dem^b)  (eval/operator.py);
  (2) probe-based graph SEEDING (probe entities -> graph start_nodes).

Cached/resumable jsonl per split (re-run is free; only missing queries hit the API).
Run (your key, gpt-4o-mini; Tier-4 -> high --workers):
  cd kg-construction
  OPENAI_API_KEY=...  python -m construct_v2.gen_probes --split test  --workers 64
  OPENAI_API_KEY=...  python -m construct_v2.gen_probes --split train --workers 64
"""
import sys
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

HERE = os.path.dirname(__file__)

SYS = (
    "Given a research problem, propose concrete bridges to papers that could inspire its solution. "
    "Return JSON with two keys.\n"
    '"methods": 3-4 items. Each = 1-2 sentences describing a SPECIFIC, NAMED quantitative/computational/'
    "experimental technique the research would need, written as it would appear in THAT method paper's "
    'abstract (e.g. "a structural equation model estimating latent variables via maximum likelihood").\n'
    '"analogs": 3-4 items. Each = a 1-2 sentence abstract of a SPECIFIC hypothetical paper from ANOTHER '
    "field whose concrete mechanism is structurally analogous to this problem. Name the field and the "
    "mechanism concretely.\n"
    'Every item concrete and specific. Never use vague words like "complex system", "optimization", "framework".'
)


def gen_one(client, q, model):
    """One query -> list of up to 8 probe strings (methods + analogs). Retries; falls back to the query."""
    for _ in range(4):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.7, max_tokens=500,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": q[:1500]}])
            d = json.loads(r.choices[0].message.content)
            pl = [str(x) for x in d.get("methods", [])] + [str(x) for x in d.get("analogs", [])]
            pl = [p for p in pl if p.strip()][:8]
            return pl or [q[:200]]
        except Exception:
            time.sleep(2)
    return [q[:200]]            # label-free fallback: the raw query as a single probe


# One resolver for corpus + caches; default "tomato" reproduces legacy paths.
ROOT = os.environ.get("CARGO_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))   # repo root
sys.path.insert(0, ROOT)
from cargo_paths import (add_dataset_arg, banner, corpus_name,  # noqa: E402
                         probes_path, set_dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--name", default=None, help="dataset dir (default tomato_<split>)")
    ap.add_argument("--root", default=os.path.join(ROOT, "kg-construction", "data"))
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=64)
    add_dataset_arg(ap)
    a = ap.parse_args()

    # set_dataset FIRST. It used to sit below the query load, so the INPUT path
    # still resolved to `tomato_{split}` while the OUTPUT went to the scoped
    # cache: a --dataset sir4_cs_smoke run started generating probes for 3,132
    # TOMATO queries and writing them into the SIR-4 probe file. Resolve the
    # dataset before anything reads a path.
    set_dataset(a.dataset)
    print(banner())

    name = a.name or corpus_name(a.split)
    qfile = os.path.join(a.root, name, "raw", f"{a.split}.json")
    queries = json.load(open(qfile))
    print(f"[probes] {len(queries)} {a.split} queries from {qfile}  (model={a.model}, workers={a.workers})")

    cache = probes_path(a.split)
    done = {}
    if os.path.exists(cache):
        for line in open(cache):
            r = json.loads(line); done[r["id"]] = r["probes"]
    todo = [q for q in queries if q["id"] not in done]
    print(f"[probes] {len(done)} already cached, {len(todo)} to generate -> {cache}")

    if todo:
        # Same keepalive point as extract_frames: the SDK pools 1000
        # connections but keeps 100 alive, so past ~100 workers every extra
        # request pays a TLS handshake instead of reusing a warm socket.
        import httpx
        client = OpenAI(                        # reads OPENAI_API_KEY
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=1000,
                                    max_keepalive_connections=1000),
                timeout=httpx.Timeout(90.0, connect=10.0)))
        n_done = n_probes = 0
        with open(cache, "a") as fout, ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(gen_one, client, q["question"], a.model): q["id"] for q in todo}
            for fu in tqdm(as_completed(futs), total=len(futs), desc=f"probes {a.split}"):
                qid = futs[fu]; probes = fu.result()
                n_done += 1; n_probes += len(probes)
                fout.write(json.dumps({"id": qid, "probes": probes}) + "\n"); fout.flush()
                done[qid] = probes
        print(f"[probes] generated {n_done} queries, {n_probes} probe texts "
              f"({n_probes / max(n_done, 1):.1f}/query)")

    total = sum(len(v) for v in done.values())
    print(f"[probes] DONE: {len(done)} queries, {total} probe texts "
          f"({total / max(len(done), 1):.1f}/query) -> {cache}")


if __name__ == "__main__":
    main()
