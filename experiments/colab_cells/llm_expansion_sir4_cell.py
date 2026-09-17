# HyDE + MuGI on the four SIR-4 fields, one cell (the llm_expansion_baselines_sir4 notebook end to end)
import os, json, hashlib, shutil, zipfile, subprocess, collections
from google.colab import drive
drive.mount('/content/drive')

DRIVE = "/content/drive/MyDrive/cargo-gfmrag"
# dataset -> (bundle zip, sets.json export or None); stamped at build time from --datasets
DATASETS = {
    'sir4_biology': (f"{DRIVE}/sir4_biology_bundle.zip", 'biology_test_low'),
    'sir4_cs':      (f"{DRIVE}/sir4_cs_bundle.zip", 'cs_test_final'),
    'sir4_matsci':  (f"{DRIVE}/sir4_matsci_bundle.zip", 'matsci_test_low'),
    'sir4_physics': (f"{DRIVE}/sir4_physics_bundle.zip", 'physics_test_low'),
}
SPLIT  = "test"
OUTRT  = {ds: f"{DRIVE}/outputs/baselines/{ds}" for ds in DATASETS}
CACHED = f"{DRIVE}/outputs/baselines/llm_expansions"   # generation caches (durable)
WRK    = "/content/llmx"                               # everything hot lives locally
for d in list(OUTRT.values()) + [CACHED, WRK]:
    os.makedirs(d, exist_ok=True)
_missing = [ds for ds, (b, _) in DATASETS.items() if not os.path.exists(b)]
for ds, (b, _) in DATASETS.items():
    print(f"  {ds:14} bundle {'ok' if ds not in _missing else 'MISSING'}")
assert not _missing, f"missing bundles on Drive: {_missing}"

# ======================================================================
def _zip(ds):
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
    if sx: need.append(f"sir-4/data/benchmark/{sx}/sets.json")
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
          f"strata {dict(collections.Counter(r.get('stratum') for r in _q))}")

# ======================================================================
!pip -q install "openai>=1.40" sentence-transformers "transformers>=4.52.4,<5"
import getpass
try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("key: from Colab secret")
except Exception:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
    print("key: entered interactively")

# ======================================================================
GEN_MODEL = "gpt-4o-mini"
N_GEN, MAX_TOK, TEMP = 4, 256, 1.0
PROMPTS = {
    # HyDE: hypothetical documents, written as the thing being retrieved (abstracts).
    "hyde": ("Write a scientific paper abstract that could address the following "
             "research problem. Write it as a factual, technical abstract describing "
             "a method and its mechanism, in the style of a published paper. Do not "
             "restate or refer to the problem; write only the abstract.\n\n"
             "Research problem: {q}"),
    # MuGI: pseudo-references, relevant background passages.
    "mugi": ("Generate a short informative passage, in the style of a scientific "
             "paper abstract, that is relevant to the following research query. "
             "Output the passage only.\n\nQuery: {q}"),
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
print(f"\nTHIS SESSION'S generation usage: {_usage['in']:,} in / {_usage['out']:,} out"
      f"  ~ ${_cost:.2f} at gpt-4o-mini rates (cached texts cost nothing)")

# ======================================================================
import numpy as np, torch
from sentence_transformers import SentenceTransformer

ENCODER = "Qwen/Qwen3-Embedding-0.6B"
QWEN_INSTRUCT = ("Instruct: Given a web search query, retrieve relevant passages "
                 "that answer the query\nQuery:")
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
    dvec = None

# ======================================================================
# `map` is here for MIR's table (R@3 / R@5 / nDCG@5 / mAP); harmless elsewhere.
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
            print(f"FAILED {ds}/{lab}:\n{r.stdout[-300:]}")
    # the plain dense row for contrast, from the scores the baselines notebooks wrote
    ref = f"{OUTRT[ds]}/scores_Qwen3-Embedding.json"
    if os.path.exists(ref):
        SCORES[ds]["Qwen3-Embedding"] = json.load(open(ref))
    print(f"{ds}: scored {sorted(SCORES[ds])}")

# ======================================================================
_cols = list(STD_COLS) + ["completeset@5"]
ORDER = ["Qwen3-Embedding"] + list(ARM_TAGS)
out = []
for ds in DATASETS:
    out.append(f"\n================ {ds} ================")
    for slc in ("all", "same", "cross", "similar", "dissimilar"):
        labs = [l for l in ORDER if slc in (SCORES[ds].get(l) or {})]
        if not labs: continue
        out.append(f"\n--- {slc} ---  n={SCORES[ds][labs[0]][slc]['n']}")
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
print("\n".join(out))
dst = f"{DRIVE}/outputs/baselines/llm_expansion_tables.md"
open(dst, "w").write("\n".join(out) + "\n")
json.dump(SCORES, open(f"{DRIVE}/outputs/baselines/llm_expansion_scores.json", "w"), indent=1)
print(f"\nwritten to {dst}")

# ---- Table 9.1 rows: same / cross nDCG@5 per field and the macro gap (mean cross vs mean same) ----
FIELDS = [("sir4_cs", "CS"), ("sir4_biology", "Biology"), ("sir4_physics", "Physics"), ("sir4_matsci", "MS")]
print("\n\n=== Table 9.1 rows (nDCG@5, %) ===")
for lab in ARM_TAGS:
    cells, same, cross = [], [], []
    for ds, _ in FIELDS:
        sc = SCORES.get(ds, {}).get(lab)
        if not sc: cells += ["--", "--"]; continue
        s, c = 100 * sc["same"]["ndcg@5"], 100 * sc["cross"]["ndcg@5"]
        same.append(s); cross.append(c); cells += [f"{s:.2f}", f"{c:.2f}"]
    gap = f"{100 * (sum(cross) / len(cross) - sum(same) / len(same)) / (sum(same) / len(same)):+.2f}%" if same and len(same) == 4 else "--"
    print(f"{lab.split(' ')[0]:6} & " + " & ".join(cells) + f" & {gap}   % CS same/cross, Bio, Phys, MS, macro gap")
