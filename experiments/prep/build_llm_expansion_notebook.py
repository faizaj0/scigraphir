"""
build_llm_expansion_notebook.py -- HyDE and MuGI rows for the baseline tables.

WHY A SEPARATE BUILDER. Same reasoning as build_baselines_notebook.py: no engine,
no graph, no training. These arms read a corpus and rank it; the only shared piece
is score_sir4.py, which is shelled out to with the same flags as every other row.

THE ARMS. Both are "LLM-expanded dense retrieval": an LLM expands the query, the
SAME encoder as the plain dense row (Qwen/Qwen3-Embedding-0.6B, sentence-transformers,
the same web-search instruct) does the retrieval. Holding the encoder fixed makes the
delta against the Qwen3-Embedding row attributable to the expansion and nothing else.

  HyDE  (Gao et al. 2022): N hypothetical documents are generated for the query;
        each is embedded AS A PASSAGE (no instruct); the query embedding (with
        instruct) and the N passage embeddings are averaged and re-normalised.
  MuGI  (Zhang et al. 2024): N pseudo-references are generated; the query text is
        repeated until its length roughly balances the references (beta = 1), the
        repetition and the references are concatenated, and the result is embedded
        as ONE query (with instruct).

CAVEAT FOR THE TABLE CAPTION. The expansion LLM (gpt-4o-mini) may have seen the
actual inspiration papers in pretraining; an expansion that paraphrases a real
gold abstract is a contamination channel the plain dense rows do not have. The
rows are still fair as "what LLM expansion buys today", but they are not
encoder-only systems and should be grouped separately in the table.

Usage
-----
    python3 prep/build_llm_expansion_notebook.py                 # every dataset below
    python3 prep/build_llm_expansion_notebook.py --datasets mir  # one dataset, own notebook

MIR (added 8 Sep 2026) has no sets.json and every query is stratum "same"; read its
`all` slice. Its bundle carries the three files this notebook extracts.
"""
from __future__ import annotations

import argparse
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# dataset -> (bundle zip basename, sets.json export or None)
ALL_DATASETS = {
    "tomato":       ("tomato_bundle.zip",       None),
    "sir4_biology": ("sir4_biology_bundle.zip", "biology_test_low"),
    "sir4_cs":      ("sir4_cs_bundle.zip",      "cs_test_final"),
    "sir4_matsci":  ("sir4_matsci_bundle.zip",  "matsci_test_low"),
    "sir4_physics": ("sir4_physics_bundle.zip", "physics_test_low"),
    "mir":          ("mir_bundle.zip",          None),
}
_ap = argparse.ArgumentParser()
_ap.add_argument("--datasets", default=",".join(ALL_DATASETS),
                 help="comma list from " + ", ".join(ALL_DATASETS))
_ap.add_argument("--out", default=None)
_a = _ap.parse_args()
DSETS = [d.strip() for d in _a.datasets.split(",") if d.strip()]
_unknown = [d for d in DSETS if d not in ALL_DATASETS]
assert DSETS and not _unknown, f"unknown datasets {_unknown}; choose from {list(ALL_DATASETS)}"
OUT = _a.out or (f"{ROOT}/notebooks/llm_expansion_baselines.ipynb" if len(DSETS) == len(ALL_DATASETS)
                 else f"{ROOT}/notebooks/llm_expansion_baselines_{'_'.join(DSETS)}.ipynb")
_DATASETS_SRC = "DATASETS = {\n" + "".join(
    f'    {repr(d) + ":":16}(f"{{DRIVE}}/{ALL_DATASETS[d][0]}", {ALL_DATASETS[d][1]!r}),\n' for d in DSETS) + "}"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(True)}


cells = []

cells.append(md(f"""# LLM-expanded dense retrieval — HyDE + MuGI on {', '.join(DSETS)}

Two rows for the baseline tables, both on the **same encoder as the plain dense row**
(`Qwen/Qwen3-Embedding-0.6B`, same instruct), so the delta against `Qwen3-Embedding`
is the expansion and nothing else.

| arm | recipe | LLM cost |
|---|---|---|
| HyDE | N hypothetical abstracts, embedded as passages, averaged with the query embedding | ~4,700 unique queries x N=4 x ~250 tok |
| MuGI | N pseudo-references concatenated with a length-balanced repetition of the query | same again |

Everything is cached and idempotent: generations cache to Drive per dataset+method
(a re-run only fills gaps), retrieval skips on a matching manifest, and scoring uses
`score_sir4.py` from the bundles with the same flags as every other arm.

**Caption caveat**: the expansion LLM may know the actual inspiration papers from
pretraining. These rows measure what LLM expansion buys, not an encoder-only system;
group them under their own family header in the table.

Estimated OpenAI spend at the defaults (gpt-4o-mini, N=4, ~250 tok): **$5-8 total**
for both methods over all five datasets. The generation cell prints running usage."""))

cells.append(md("## 1. Drive + datasets"))
cells.append(code('''import os, json, hashlib, shutil, zipfile, subprocess, collections
from google.colab import drive
drive.mount('/content/drive')

DRIVE = "/content/drive/MyDrive/cargo-gfmrag"
# dataset -> (bundle zip, sets.json export or None); stamped at build time from --datasets
''' + _DATASETS_SRC + '''
SPLIT  = "test"
OUTRT  = {ds: f"{DRIVE}/outputs/baselines/{ds}" for ds in DATASETS}
CACHED = f"{DRIVE}/outputs/baselines/llm_expansions"   # generation caches (durable)
WRK    = "/content/llmx"                               # everything hot lives locally
for d in list(OUTRT.values()) + [CACHED, WRK]:
    os.makedirs(d, exist_ok=True)
_missing = [ds for ds, (b, _) in DATASETS.items() if not os.path.exists(b)]
for ds, (b, _) in DATASETS.items():
    print(f"  {ds:14} bundle {'ok' if ds not in _missing else 'MISSING'}")
assert not _missing, f"missing bundles on Drive: {_missing}"'''))

cells.append(md("## 2. Pull ONLY the needed files out of each bundle\n"
                "Targeted member extraction, not `extractall`: the TOMATO bundle is "
                "~1.9 GB and this notebook needs four small files from it."))
cells.append(code('''def _zip(ds):
    zp = DATASETS[ds][0]
    try:
        return zipfile.ZipFile(zp)
    except OSError:                      # Drive FUSE flap: one sequential copy, then local
        lp = f"/content/_{ds}.zip"
        if not os.path.exists(lp): shutil.copy(zp, lp)
        return zipfile.ZipFile(lp)

SCORER = f"{WRK}/experiments/eval/score_sir4.py"
if not os.path.exists(SCORER):        # every bundle carries the scorer; take the first one
    _zip(next(iter(DATASETS))).extract("experiments/eval/score_sir4.py", WRK)

DOCS, QS, SETS = {}, {}, {}
for ds, (_, sx) in DATASETS.items():
    need = [f"retriever/data/{ds}_{SPLIT}/raw/documents.json",
            f"retriever/data/{ds}_{SPLIT}/raw/{SPLIT}.json"]
    if sx: need.append(f"benchmark/data/benchmark/{sx}/sets.json")
    z = None
    for m in need:
        if not os.path.exists(f"{WRK}/{m}"):
            z = z or _zip(ds)
            z.extract(m, WRK)
    DOCS[ds] = f"{WRK}/{need[0]}"
    QS[ds]   = f"{WRK}/{need[1]}"
    SETS[ds] = f"{WRK}/{need[2]}" if sx else None
    _q = json.load(open(QS[ds]))
    _uniq = len({r["question"] for r in _q})
    print(f"  {ds:14} {len(json.load(open(DOCS[ds]))):>7,} docs  "
          f"{len(_q):>5,} query rows  {_uniq:>5,} unique texts  "
          f"strata {dict(collections.Counter(r.get('stratum') for r in _q))}")'''))

cells.append(md("## 3. Install + OpenAI key\n"
                "Key comes from the Colab secret `OPENAI_API_KEY` (Tools > Secrets) "
                "or an interactive prompt. It is never written to disk."))
cells.append(code('''!pip -q install "openai>=1.40" sentence-transformers "transformers>=4.52.4,<5"
import getpass
try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("key: from Colab secret")
except Exception:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
    print("key: entered interactively")'''))

cells.append(md("""## 4. Generate the expansions

One cache file per dataset+method on Drive, keyed by the md5 of the query TEXT, so
TOMATO's repeated queries (3,132 rows, 1,658 texts) are expanded once. The cache is
written locally and synced to Drive every flush; a re-run fills only the gaps, so an
interrupted run costs nothing already paid for."""))
cells.append(code('''GEN_MODEL = "gpt-4o-mini"
N_GEN, MAX_TOK, TEMP = 4, 256, 1.0
PROMPTS = {
    # HyDE: hypothetical documents, written as the thing being retrieved (abstracts).
    "hyde": ("Write a scientific paper abstract that could address the following "
             "research problem. Write it as a factual, technical abstract describing "
             "a method and its mechanism, in the style of a published paper. Do not "
             "restate or refer to the problem; write only the abstract.\\n\\n"
             "Research problem: {q}"),
    # MuGI: pseudo-references, relevant background passages.
    "mugi": ("Generate a short informative passage, in the style of a scientific "
             "paper abstract, that is relevant to the following research query. "
             "Output the passage only.\\n\\nQuery: {q}"),
}

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, time, random
from openai import OpenAI
_cli = OpenAI()
_usage = {"in": 0, "out": 0}
_ulock = threading.Lock()

def _gen_one(text, method):
    msg = [{"role": "user", "content": PROMPTS[method].format(q=text)}]
    for att in range(6):
        try:
            r = _cli.chat.completions.create(model=GEN_MODEL, messages=msg,
                                             n=N_GEN, max_tokens=MAX_TOK,
                                             temperature=TEMP)
            with _ulock:
                _usage["in"] += r.usage.prompt_tokens
                _usage["out"] += r.usage.completion_tokens
            return [c.message.content or "" for c in r.choices]
        except Exception as e:
            if att == 5: raise
            time.sleep(min(60, 2 ** att + random.random()))

def _key(text):
    return hashlib.md5(text.encode()).hexdigest()

for ds in DATASETS:
    texts = sorted({r["question"] for r in json.load(open(QS[ds]))})
    for method in PROMPTS:
        cname = f"{method}_{GEN_MODEL}_{ds}.json"
        drive_c, local_c = f"{CACHED}/{cname}", f"{WRK}/{cname}"
        cache = {}
        for p in (drive_c, local_c):
            if os.path.exists(p):
                try: cache.update(json.load(open(p)))
                except Exception: pass
        todo = [t for t in texts if _key(t) not in cache]
        print(f"{ds}/{method}: {len(texts)} texts, {len(todo)} to generate")
        if not todo:
            json.dump(cache, open(local_c, "w")); continue
        done = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(_gen_one, t, method): t for t in todo}
            for f in as_completed(futs):
                cache[_key(futs[f])] = f.result()
                done += 1
                if done % 200 == 0 or done == len(todo):
                    json.dump(cache, open(local_c, "w"))
                    try: shutil.copy(local_c, drive_c)          # durable checkpoint
                    except OSError as e: print(f"[sync skipped: {e}]")
                    print(f"  {ds}/{method}: {done}/{len(todo)}  "
                          f"tokens in/out {_usage['in']:,}/{_usage['out']:,}")
        json.dump(cache, open(local_c, "w")); shutil.copy(local_c, drive_c)

_cost = _usage["in"] / 1e6 * 0.15 + _usage["out"] / 1e6 * 0.60
print(f"\\nTHIS SESSION'S generation usage: {_usage['in']:,} in / {_usage['out']:,} out"
      f"  ~ ${_cost:.2f} at gpt-4o-mini rates (cached texts cost nothing)")'''))

cells.append(md("""## 5. Retrieve

The corpus is embedded once per dataset (passages, no instruct) and reused by both
methods. Query construction:

- **hyde**: mean of the instructed query embedding and the N generation embeddings
  (embedded as passages), re-normalised.
- **mugi**: `query x r + generations` as one string, `r` chosen so the repeated query
  is roughly as long as the generations (beta = 1), embedded with the query instruct.

Skips on a matching manifest; predictions land in the same per-dataset namespace the
other baseline rows use."""))
cells.append(code('''import numpy as np, torch
from sentence_transformers import SentenceTransformer

ENCODER = "Qwen/Qwen3-Embedding-0.6B"
QWEN_INSTRUCT = ("Instruct: Given a web search query, retrieve relevant passages "
                 "that answer the query\\nQuery:")
TOPK = 300
_model = SentenceTransformer(ENCODER, device="cuda" if torch.cuda.is_available() else "cpu")
_model.max_seq_length = 4096      # MuGI's expanded queries exceed the 512 default

def _sig(method):
    return hashlib.md5(json.dumps(
        {"method": method, "gen_model": GEN_MODEL, "n_gen": N_GEN, "temp": TEMP,
         "max_tok": MAX_TOK, "prompt": PROMPTS[method], "encoder": ENCODER,
         "instruct": QWEN_INSTRUCT, "topk": TOPK}, sort_keys=True).encode()
    ).hexdigest()[:12]

def _embed(texts, prompt=None, bs=32):
    return _model.encode(texts, prompt=prompt, batch_size=bs,
                         normalize_embeddings=True, show_progress_bar=False,
                         convert_to_numpy=True).astype(np.float32)

for ds in DATASETS:
    docs = json.load(open(DOCS[ds]))
    ids, dvec = list(docs), None
    rows = json.load(open(QS[ds]))
    for method in PROMPTS:
        tag  = f"{method}_{GEN_MODEL.replace('-', '')}_qwen3"
        dest = f"{OUTRT[ds]}/predictions_{tag}_{ds}_{SPLIT}.json"
        man  = dest + ".manifest.json"
        sig  = _sig(method)
        if os.path.exists(dest) and os.path.exists(man):
            try:
                if json.load(open(man)).get("sig") == sig:
                    print(f"[skip] {ds}/{method}: manifest matches"); continue
            except Exception: pass
        cache = json.load(open(f"{WRK}/{method}_{GEN_MODEL}_{ds}.json"))
        if dvec is None:
            print(f"{ds}: embedding {len(ids):,} docs once ...")
            dvec = _embed([docs[i] for i in ids], bs=64)
        texts = sorted({r["question"] for r in rows})
        if method == "hyde":
            qv = _embed(texts, prompt=QWEN_INSTRUCT)
            gv = _embed([g for t in texts for g in cache[hashlib.md5(t.encode()).hexdigest()]])
            gv = gv.reshape(len(texts), N_GEN, -1)
            vec = qv + gv.sum(1)                    # mean of 1 query + N docs, then renorm
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        else:
            expanded = []
            for t in texts:
                gens = " ".join(cache[hashlib.md5(t.encode()).hexdigest()])
                r = max(1, round(len(gens) / max(1, len(t))))
                expanded.append((t + " ") * r + gens)
            vec = _embed(expanded, prompt=QWEN_INSTRUCT, bs=8)
        v_by_text = dict(zip(texts, vec))
        out = []
        for r in rows:
            s = v_by_text[r["question"]] @ dvec.T
            top = np.argsort(-s)[:TOPK]
            out.append({"id": r["id"],
                        "predictions": {"document": [[ids[j], float(s[j])] for j in top]}})
        json.dump(out, open(dest, "w"))
        json.dump({"sig": sig, "method": method, "gen_model": GEN_MODEL, "n_gen": N_GEN,
                   "encoder": ENCODER, "topk": TOPK}, open(man, "w"), indent=1)
        print(f"wrote {dest}  ({len(out)} records)")
    dvec = None'''))

cells.append(md("## 6. Score, same scorer, same flags\n"
                "`--sets` where the domain has one (CompleteSet@5), `--bge` where the "
                "plain BGE predictions exist (similar/dissimilar slices)."))
cells.append(code('''# `map` is here for MIR's table (R@3 / R@5 / nDCG@5 / mAP); harmless elsewhere.
STD_COLS = ("mrr", "ndcg@5", "recall@3", "recall@5", "recall@10", "recall@25", "recall@100", "map")
GRID_KS  = (1, 3, 5, 10, 20, 25, 50, 100)
COLS = ",".join(dict.fromkeys(list(STD_COLS) + [f"recall@{k}" for k in GRID_KS]
                              + [f"hits@{k}" for k in GRID_KS] + ["completeset@5"]))
ARM_TAGS = {f"HyDE ({GEN_MODEL})": f"hyde_{GEN_MODEL.replace('-', '')}_qwen3",
            f"MuGI ({GEN_MODEL})": f"mugi_{GEN_MODEL.replace('-', '')}_qwen3"}

SCORES = {ds: {} for ds in DATASETS}
for ds in DATASETS:
    bge = f"{OUTRT[ds]}/predictions_bge_{ds}_{SPLIT}.json"
    for lab, tag in ARM_TAGS.items():
        p = f"{OUTRT[ds]}/predictions_{tag}_{ds}_{SPLIT}.json"
        if not os.path.exists(p):
            print(f"skipping {ds}/{lab}: no predictions"); continue
        jo = f"{OUTRT[ds]}/scores_{lab.replace(' ', '_').replace('(', '').replace(')', '')}.json"
        cmd = (f"python3 -u {SCORER} --pred {p} --queries {QS[ds]} --cols {COLS} "
               f"--name '{ds} {lab}' --json-out {jo}")
        if SETS[ds] and os.path.exists(SETS[ds]): cmd += f" --sets {SETS[ds]}"
        if os.path.exists(bge):                   cmd += f" --bge {bge}"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(jo):
            SCORES[ds][lab] = json.load(open(jo))
        else:
            print(f"FAILED {ds}/{lab}:\\n{r.stdout[-300:]}")
    # the plain dense row for contrast, from the scores the baselines notebooks wrote
    ref = f"{OUTRT[ds]}/scores_Qwen3-Embedding.json"
    if os.path.exists(ref):
        SCORES[ds]["Qwen3-Embedding"] = json.load(open(ref))
    print(f"{ds}: scored {sorted(SCORES[ds])}")'''))

cells.append(md("## 7. The table\n"
                "The plain `Qwen3-Embedding` row is repeated from the baseline tables "
                "as the reference; the deltas under each slice are expansion minus plain."))
cells.append(code('''_cols = list(STD_COLS) + ["completeset@5"]
ORDER = ["Qwen3-Embedding"] + list(ARM_TAGS)
out = []
for ds in DATASETS:
    out.append(f"\\n================ {ds} ================")
    for slc in ("all", "same", "cross", "similar", "dissimilar"):
        labs = [l for l in ORDER if slc in (SCORES[ds].get(l) or {})]
        if not labs: continue
        out.append(f"\\n--- {slc} ---  n={SCORES[ds][labs[0]][slc]['n']}")
        out.append(f"{'arm':26}" + "".join(
            f"{c.replace('recall@', 'R@').replace('completeset@5', 'CS@5'):>9}" for c in _cols))
        for lab in labs:
            sc = SCORES[ds][lab][slc]
            out.append(f"{lab:26}" + "".join(
                f"{100*sc[c]:>9.1f}" if sc.get(c) is not None else f"{'--':>9}" for c in _cols))
        if "Qwen3-Embedding" in labs:
            base = SCORES[ds]["Qwen3-Embedding"][slc]
            for lab in labs:
                if lab == "Qwen3-Embedding": continue
                sc = SCORES[ds][lab][slc]
                out.append(f"{'  ' + lab + ' - dense':26}" + "".join(
                    f"{100*(sc[c]-base[c]):>+9.1f}"
                    if sc.get(c) is not None and base.get(c) is not None else f"{'--':>9}"
                    for c in _cols))
print("\\n".join(out))
dst = f"{DRIVE}/outputs/baselines/llm_expansion_tables.md"
open(dst, "w").write("\\n".join(out) + "\\n")
json.dump(SCORES, open(f"{DRIVE}/outputs/baselines/llm_expansion_scores.json", "w"), indent=1)
print(f"\\nwritten to {dst}")'''))


def write_nb(cell_list, out_path):
    nb = {"cells": cell_list,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    for _i, _c in enumerate(cell_list):
        if _c["cell_type"] != "code":
            continue
        _out, _mag = [], False
        for _ln in "".join(_c["source"]).split("\n"):
            if _ln.lstrip().startswith(("!", "%")) or _mag:
                _mag = _ln.rstrip().endswith("\\")
                _out.append("pass")
            else:
                _out.append(_ln)
        try:
            ast.parse("\n".join(_out))
        except SyntaxError as e:
            raise SystemExit(f"cell {_i} line {e.lineno}: {e.msg}")
    json.dump(nb, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}  ({len(cell_list)} cells, {os.path.getsize(out_path)/1024:.0f} KB)")


write_nb(cells, OUT)
