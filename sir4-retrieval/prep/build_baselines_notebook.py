"""
build_baselines_notebook.py -- the retrieval-baseline table, one notebook.

WHY A SEPARATE BUILDER, when build_notebook.py's own doctrine says not to fork.
That doctrine is about the FUSION notebook: one generator so the engine, the loss
and the arm definitions cannot drift into a stale copy. This notebook shares none
of that. No engine, no graph, no reasoner, no fusion, no training, no Qwen3
operator fit. It reads a corpus and ranks it. The only thing it has in common is
score_sir4.py, which it shells out to exactly like every other arm does.

WHAT IT PRODUCES. all / same / cross for every baseline, through the SAME scorer
with the SAME flags, plus the paper table: three (same, cross) pairs at chosen
cutoffs and the mean relative cross-domain drop.

NO similar/dissimilar. Those need a reference dense run to define them, and a
baseline table is exactly where that self-reference is wrong: the arm defining
the slice would be graded on its own definition.

Usage
-----
    python3 prep/build_baselines_notebook.py                    # tomato
    python3 prep/build_baselines_notebook.py --dataset sir4_cs
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CARGO = os.path.dirname(ROOT)
FORK = f"{CARGO}/kg-construction/gfm-rag"
# THE SAME SOURCE build_notebook.py READS. The engine install and the Qwen3 fetch are
# lifted from this notebook rather than retyped, so the two builders cannot drift into
# different engines: a change there lands in both, or in neither.
SRC = f"{CARGO}/kg-construction/train/colab_train_v16sc_fusion_greasoner.ipynb"

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="tomato")
ap.add_argument("--out", default=None)
_a = ap.parse_args()
DSET = _a.dataset
# --dataset all: ONE notebook covering every SIR-4 domain. Dense arms loop the domains at
# runtime; G-Reasoner trains per domain on the existing v16sc graphs; GFM-RAG is a GATED
# section because no SIR-4 domain has the OpenIE entity graph its forward path requires.
# --dataset mir (8 Sep 2026): the ALL-mode plumbing over the single MIR dataset. G-Reasoner
# trains on the mir_*_v16sc frame graphs, GFM-RAG is gated on the mir_* OpenIE graph exactly
# as for SIR-4 (build it with kg-construction/run_index.sh mir_{train,test}, re-bundle, and
# the gate opens with no edit). The single-dataset mode is TOMATO-shaped (OpenIE graph names,
# an entity-node assert) and does not fit MIR, so mir goes through ALL.
MIR = DSET == "mir"
ALL = MIR or DSET in ("all", "sir4_all", "all_sir4")
OUT = _a.out or (f"{ROOT}/mir_baselines.ipynb" if MIR
                 else f"{ROOT}/sir4_baselines_all.ipynb" if ALL
                 else f"{ROOT}/{DSET}_baselines.ipynb")
_ALL_DOMS = ["mir"] if MIR else ["biology", "cs", "matsci", "physics"]
_ALL_DSETS = ["mir"] if MIR else [f"sir4_{d}" for d in _ALL_DOMS]
_SUMMARY = "mir_all.json" if MIR else "sir4_all_domains.json"
_MIR_HEADER = """# MIR — graph retrieval baselines (G-Reasoner, GFM-RAG) + the dense arms

The two graph-retrieval rows of the MIR table, through the SAME runner, scorer and flags
as the SIR-4 rows. Sections 1-4 re-run the dense arms into `outputs/baselines/mir/` (the
MIR fusion notebook keeps its own copies under `outputs/mir/baselines/`; both are fine).

| family | arm | cost |
|---|---|---|
| dense | BM25, BGE-large, Qwen3-Embedding, SPECTER2-base, SciNCL, ReasonIR-8B | ~15 min |
| graph | G-Reasoner | trained here on `mir_{train,test}_v16sc`, ~1 h |
| graph | GFM-RAG | **gated** on the `mir_{train,test}` OpenIE entity graph; ~1 h once it is in the bundle |

MIR is a single field: every query is stratum `same`, so read the `all` slice, and there is
no sets.json, so CompleteSet@5 is not requested. mAP is in the columns because the MIR
table reports R@3 / R@5 / nDCG@5 / mAP. The cross-domain summary at the end prints
dashes here; it is the SIR-4 layout and is left in place so the two notebooks stay one
builder.
"""

_src = json.load(open(SRC))
reuse = {i: _src["cells"][i] for i in (4, 7)}   # 4 = engine install, 7 = Qwen3 fetch
# Cell 5 (the fusion FILES blob) is deliberately NOT reused: these baselines are the
# stock upstream models, so pulling the fusion sources in would be the one thing that
# could stop them being baselines.

# Files that exist nowhere but this repo, shipped inline so the notebook does not depend on
# the age of gfm-rag-adapted.zip on Drive.
_SHIP = [
    # The GFM-RAG config.
    ("gfmrag/workflow/config/gfm_reasoner/sft_training_gfmrag.yaml",
     "/content/gfm-rag/gfmrag/workflow/config/gfm_reasoner/sft_training_gfmrag.yaml"),
    # nDCG. The stock evaluate() implements mrr/recall@k/hits@k and NOTHING for ndcg@,
    # where it falls through to `raise ValueError(unknown metric)`. The GFM-RAG arm asks
    # for ndcg@5, so on the zip's copy it dies at the end of epoch 1, after the Qwen3
    # index. The repo's qa_utils.py has the binary-relevance nDCG that matches
    # score_sir4.py; gfm-rag-adapted.zip was built 2026-07-13 and predates it, and
    # re-zipping is a 1.4 MB upload the notebook should not silently require.
    ("gfmrag/utils/qa_utils.py", "/content/gfm-rag/gfmrag/utils/qa_utils.py"),
]
GRAPH_FILES = {dst: open(f"{FORK}/{src}").read() for src, dst in _SHIP}
assert not any("'''" in v for v in GRAPH_FILES.values()), "shipped file contains '''"
# The whole point of shipping qa_utils.py is the nDCG branch; if a future edit drops it the
# notebook would install a file that crashes exactly as the stale zip does.
assert 'ndcg@' in GRAPH_FILES["/content/gfm-rag/gfmrag/utils/qa_utils.py"], \
    "the repo's qa_utils.py has no ndcg@ branch -- shipping it would not fix the crash"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.splitlines(True)}


cells = []

cells.append(md(f"""# {DSET} — retrieval baselines

Every row through the **same scorer with the same flags**, so a difference between
two rows is the retriever and nothing else.

| family | arm | cost |
|---|---|---|
| lexical | BM25 | CPU, minutes |
| dense | BGE-large, Qwen3-Embedding, SPECTER2, SciNCL | GPU, minutes each |
| reasoning-trained dense | ReasonIR-8B | GPU, bf16, ~16 GB of weights |
| graph | G-Reasoner, GFM-RAG | **trained here**, section 5c — GPU, real training time |
| **ours** | multi-view scorer, and the same scorer + graph | section 5e — read from the ablation notebook's output, nothing trained here |

The last family is **not** a baseline. It is there because the gap between the two `ours`
rows is the graph's whole contribution on top of the learned scorer, which no pair of
baseline rows can show.

**Slices are `all`, `same`, `cross` only.** `similar`/`dissimilar` are defined by a
reference dense run, and in a baseline table that is circular: the arm defining the
slice would be graded on its own definition.

**Pooling is explicit.** SPECTER2 and SciNCL are CLS-pooled. `SentenceTransformer`
mean-pools any checkpoint that ships no ST config, which silently evaluates a
different model and understates both, so this notebook passes `--pooling cls` for
them and lets the others use their own config.

**Two disclosures that belong in the table caption, not in a footnote:**

* the SPECTER2 row is `allenai/specter2_base`, the base encoder **without** the retrieval
  adapters (`proximity` for candidate papers, ad-hoc-query for short queries). It is
  labelled `SPECTER2-base` for that reason and must not be shortened to "SPECTER2";
* SPECTER2 and SciNCL are both trained on `title [SEP] abstract`, and this corpus is
  already flattened to `"Title. Abstract"` with the boundary unrecoverable. Both see the
  same text through a slightly different input format from their papers'. It applies
  equally to both rows, so it cannot flip their comparison, but neither row is that
  paper's published number."""))

cells.append(md("## 1. GPU + Drive"))
cells.append(code(f'''!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import os, sys, json, subprocess
from google.colab import drive
drive.mount('/content/drive')

DRIVE   = "/content/drive/MyDrive/cargo-gfmrag"
DATASET = "{DSET}"
SPLIT   = "test"
BUNDLE  = f"{{DRIVE}}/{{DATASET}}_bundle.zip"
OUT_ROOT = f"{{DRIVE}}/outputs/baselines/{{DATASET}}"
CARGO_ROOT = "/content/cargo"
os.environ["CARGO_ROOT"] = CARGO_ROOT
os.environ["CARGO_DATASET"] = DATASET
S4 = f"{{CARGO_ROOT}}/sir4-retrieval"
os.makedirs(OUT_ROOT, exist_ok=True)
print("bundle  ", BUNDLE)
print("writes  ", OUT_ROOT)'''))

cells.append(md("## 2. Unpack the bundle + install\n"
                "Sections 1-4 read only the corpus and the eval scripts. **Section 5 also reads "
                "the graphs**, so run this against the FULL bundle if you want the graph rows; "
                "a `--slim` bundle carries no graphs and section 5 will say so."))
cells.append(code('''import zipfile, shutil
assert os.path.exists(BUNDLE), f"missing {BUNDLE}"
if os.path.isdir(CARGO_ROOT): shutil.rmtree(CARGO_ROOT)
os.makedirs(CARGO_ROOT, exist_ok=True)
zipfile.ZipFile(BUNDLE).extractall(CARGO_ROOT)

# A code_overlay on Drive wins over the bundle's copies, so a script edited after
# the bundle was built does not need a 400 MB re-upload to take effect.
OV = f"{DRIVE}/code_overlay"
if os.path.isdir(OV):
    shutil.copytree(OV, CARGO_ROOT, dirs_exist_ok=True); print("applied code_overlay")

CORPUS = f"{CARGO_ROOT}/kg-construction/data/{DATASET}_{SPLIT}/raw"
for f in ("documents.json", f"{SPLIT}.json"):
    assert os.path.exists(f"{CORPUS}/{f}"), f"missing {CORPUS}/{f}"
BL = f"{S4}/eval/baselines_sir4.py"
assert os.path.exists(BL), f"missing {BL} -- rebuild the bundle or use code_overlay"

_c = json.load(open(f"{CORPUS}/documents.json")); _q = json.load(open(f"{CORPUS}/{SPLIT}.json"))
_g = [len(x.get("supporting_documents") or []) for x in _q]
import collections
print(f"corpus {len(_c):,} docs | {len(_q):,} queries | golds/query {sum(_g)/len(_g):.2f}")
print("strata:", dict(collections.Counter(x.get("stratum") for x in _q)))

# Sections 1-4 are advertised as needing no graph engine, so they cannot assume the
# engine cell in 5a has run. Colab ships transformers but NOT sentence-transformers,
# and every dense arm imports it -- without this the "no GPU engine needed" claim is
# only true if you happen to run the sections out of order.
# PINNED, not latest. Colab resolves `transformers` to the newest release, which is now
# 5.x; the engine in section 5 and ReasonIR's remote-code architecture are both 4.x-era.
# An unpinned install has already produced a transformers 5.x / wandb 0.28 environment
# here. Sections 1-4 only need transformers, but the pin has to be the same one section 5a
# uses or whichever cell ran last decides the environment.
!pip -q install rank_bm25 sentence-transformers "transformers>=4.52.4,<5"
import transformers
print("ready | transformers", transformers.__version__)
assert transformers.__version__.startswith("4."), (
    f"transformers {transformers.__version__} is outside the supported 4.x range; "
    "restart the runtime after the pinned install so the 4.x wheel is the one imported")'''))

cells.append(md("""## 3. The arms

Each instruction-aware baseline uses its documented stock query instruction rather
than a SciGraphIR-specific prompt. This keeps the rows recognizable as off-the-shelf
baselines. BM25, SPECTER2 and SciNCL take no instruction."""))
cells.append(code('''# Qwen3's STOCK retrieval instruction from its official model card. The newline is
# part of the format and must reach the tokenizer as a real newline, not the two
# characters backslash+n. The run cell below therefore passes argv as a list.
QWEN_INSTRUCT = ("Instruct: Given a web search query, retrieve relevant passages "
                  "that answer the query\\nQuery:")
BGE_INSTRUCT = "Represent this sentence for searching relevant passages: "

# ReasonIR's documented default is an empty instruction. Keep the variable explicit so
# a future task-specific ReasonIR experiment cannot silently change the stock baseline.
REASONIR_INSTRUCT = ""

# (label, tag, model, pooling, instruct, extra flags)
ARMS = [
    ("BM25",            "bm25",     "bm25",                       "auto", "",           ""),
    ("BGE-large",       "bge",      "BAAI/bge-large-en-v1.5",     "st",   BGE_INSTRUCT, ""),
    ("Qwen3-Embedding", "qwen3",    "Qwen/Qwen3-Embedding-0.6B",  "st",   QWEN_INSTRUCT, ""),
    # CLS, NOT MEAN. Both are CLS-pooled; ST's mean-pooling fallback would evaluate a
    # different model than either paper and understate them.
    # SPECTER2-base, NOT SPECTER2. The published retrieval model is this base encoder PLUS
    # a task adapter (proximity for candidate papers, ad-hoc-query for short queries).
    # Loading the base alone is a different, weaker model, so the row is labelled for what
    # it is. To make it the real thing: pip install adapters, load allenai/specter2 onto
    # the base with load_as="proximity", which baselines_sir4.py does not yet support.
    ("SPECTER2-base",   "specter2", "allenai/specter2_base",      "cls",  "",           ""),
    # NO [SEP]. SciNCL and SPECTER2 are both trained on "title [SEP] abstract"; this corpus
    # is already flattened to "Title. Abstract" in documents.json and the title boundary is
    # not recoverable from it without the pre-flattening source. Both rows therefore see a
    # slightly different input format from the one their papers used. Same text, same
    # tokeniser, one missing separator token, and it applies equally to both rows, so it
    # cannot flip their comparison -- but it is a reason not to read either as that paper's
    # published number, and it belongs in the table caption.
    ("SciNCL",          "scincl",   "malteos/scincl",             "cls",  "",           ""),
    # 8B: half precision or it does not fit, and a custom architecture so it needs remote
    # code. BFLOAT16, not float16: the model card's own usage is torch_dtype="auto", which
    # resolves to the bf16 the checkpoint was trained in. Forcing fp16 re-quantises to a
    # format with a much smaller exponent range, which is the standard way an 8B scores
    # below its published numbers. bf16 needs Ampere or newer (A100 yes, T4 no).
    ("ReasonIR-8B",     "reasonir", "reasonir/ReasonIR-8B",       "st",   REASONIR_INSTRUCT,
     "--trust-remote-code --dtype bfloat16 --batch 8"),
]
for lab, tag, m, p, ins, extra in ARMS:
    print(f"  {lab:18} {m}")'''))

cells.append(md("## 4. Run every arm\n"
                "Skips an arm whose predictions already exist **and were produced by this "
                "exact configuration**, so a disconnect costs only the arm that was running.\n\n"
                "The filename carries only the tag, dataset and split, so it cannot tell a "
                "mean-pooled run from a CLS-pooled one, or an old instruction from the current "
                "one. Each prediction file therefore gets a sidecar manifest holding the model, "
                "pooling, instruction, extra flags, top-k and a hash of the scorer script, and "
                "the skip is conditional on that manifest matching. A file with no manifest, or "
                "a manifest that disagrees, is re-run rather than presented as current."))
cells.append(code('''import hashlib, shlex
TOPK = 300   # in the signature below, so a change to it invalidates the cache

# Set True only to accept prediction files that carry no manifest (everything produced
# before this cell existed). They are then reported as though they matched the current
# configuration, which is the exact failure the manifest exists to prevent.
ACCEPT_UNVERIFIED = False

def sh(cmd, cwd=None):
    # Shell strings remain supported for the scorer cells below. Baseline inference uses
    # an argv list so instructions, especially Qwen3's required newline, arrive verbatim.
    p = subprocess.Popen(cmd, shell=isinstance(cmd, str), cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end="")
    return p.wait()

# The scorer script itself is part of the configuration: a change to pooling, instruction
# handling or normalisation inside it changes the numbers without changing any flag.
_SCORER = hashlib.md5(open(f"{S4}/eval/baselines_sir4.py", "rb").read()).hexdigest()[:8]

def arm_sig(model, pool, ins, extra):
    return hashlib.md5(json.dumps(
        {"model": model, "pooling": pool, "instruct": ins, "extra": extra,
         "topk": TOPK, "scorer": _SCORER, "dataset": DATASET, "split": SPLIT},
        sort_keys=True).encode()).hexdigest()[:12]

PRED = {}
for lab, tag, model, pool, ins, extra in ARMS:
    dest = f"{S4}/data/predictions_{tag}_{DATASET}_{SPLIT}.json"
    man  = dest + ".manifest.json"
    sig  = arm_sig(model, pool, ins, extra)
    PRED[lab] = dest
    # RESTORE BEFORE SKIPPING. {S4} is the unpacked bundle under /content and does not
    # survive a runtime reset, so "is the file here" is the wrong question: it recomputes
    # an arm whose result is already sitting on Drive. Re-running the unpack cell has the
    # same effect. Copying back first makes a disconnect cost nothing but the running arm.
    for a, b in ((f"{OUT_ROOT}/{os.path.basename(dest)}", dest),
                 (f"{OUT_ROOT}/{os.path.basename(man)}",  man)):
        if not os.path.exists(b) and os.path.exists(a):
            os.makedirs(os.path.dirname(b), exist_ok=True); shutil.copy(a, b)
    if os.path.exists(dest):
        print(f"[restore] {lab}: recovered from Drive")
    _have = None
    if os.path.exists(man):
        try: _have = json.load(open(man)).get("sig")
        except Exception: _have = None
    if os.path.exists(dest):
        if _have == sig:
            print(f"[skip] {lab}: manifest matches this configuration"); continue
        if _have is None and ACCEPT_UNVERIFIED:
            print(f"[skip] {lab}: NO MANIFEST, accepted because ACCEPT_UNVERIFIED"); continue
        why = "no manifest" if _have is None else f"manifest {_have} != {sig}"
        print(f"[stale] {lab}: {why} -- re-running, the old file will be overwritten")
    cmd = [sys.executable, "-u", "eval/baselines_sir4.py",
           "--dataset", DATASET, "--split", SPLIT, "--model", model,
           "--pooling", pool, "--tag", tag, "--topk", str(TOPK)]
    if ins:
        cmd += ["--instruct", ins]
    if extra:
        cmd += shlex.split(extra)
    cmd += ["--out", dest]
    rc = sh(cmd, S4)
    # NOT an assert: one unavailable checkpoint (a gated repo, an OOM on the 8B)
    # should cost that row, not the whole table.
    if rc != 0:
        print(f"!! {lab} FAILED rc={rc} -- its row will be reported as missing")
    else:
        # Manifest written only on success, and only after the predictions exist, so a
        # half-written file can never be certified as current.
        json.dump({"sig": sig, "model": model, "pooling": pool, "instruct": ins,
                   "extra": extra, "topk": TOPK, "scorer_md5": _SCORER},
                  open(man, "w"), indent=1)
        shutil.copy(dest, f"{OUT_ROOT}/{os.path.basename(dest)}")
        shutil.copy(man,  f"{OUT_ROOT}/{os.path.basename(man)}")'''))

cells.append(md("""## 5. Graph baselines — G-Reasoner and GFM-RAG

**Everything below this point is optional.** Sections 1-4 are the lexical and dense
table and need no GPU engine; stop here if that is all you want. These two arms
train a GNN, so they install the engine and take real GPU time.

Both are the published architectures trained from **random init** on our graph and
split. No pretrained GFM-RAG or G-Reasoner checkpoints are downloaded.

| arm | model | config | dataset class | supervision | ranker |
|---|---|---|---|---|---|
| G-Reasoner | `gfm_reasoner.GraphReasoner` | `sft_training` (stock) | `GraphIndexDataset` | document | n/a |
| GFM-RAG | `gfm_rag_v1.GNNRetriever` | `sft_training_gfmrag` | `GraphIndexDatasetV1` | entity | `idf_topk_ranker` |

**Both run on `tomato_train` / `tomato_test`, the OpenIE graph**, so the two rows
differ in the model and not in the substrate. They cannot both run on v16sc: GFM-RAG's
forward path needs `target_type: entity`, and v16sc has no entity nodes.

Each arm keeps its own reference recipe, which is what a baseline means. That does
leave three differences that belong to the architectures rather than to the setup, and
they should be stated as description, not hidden:

* documents are scored directly as graph nodes by G-Reasoner, and via the ranker from
  entity scores by GFM-RAG;
* `use_ent_emb: early-late-fusion` exists only for G-Reasoner;
* seed-node weighting (`init_nodes_weight`) exists only for GFM-RAG.

Both are evaluated on the same **document** metrics, so the rows compare directly even
though the losses sit at different levels of the graph.

Neither uses the operator, the learned scorer, the gate or any fusion. That is the
point of having them: whatever a fusion arm gains over these rows is what the
project added. Note the fusion arms run on **v16sc**, so that gain includes the graph
construction as well as the model. The construction on its own is isolated by the
separate v1-fusion versus v16-fusion pair, not by this table.

**This needs the FULL bundle**, not `--slim`, because a slim bundle carries no graphs."""))
cells.append(md("### 5a. Engine\n"
                "Reused verbatim from the fusion notebook, so the two builders cannot end up "
                "installing different engines."))
cells.append(reuse[4])

# AFTER the engine cell, because that cell is reused verbatim and installs `transformers`
# and `wandb` with no version bound. Whichever cell runs last decides the environment, so
# pinning only in section 2 would be undone the moment section 5a ran: the engine would pull
# transformers 5.x back in over the 4.x the dense arms were pinned to. Re-pinning here is a
# few seconds of pip resolve and makes the order irrelevant.
cells.append(md("### 5a-i. Re-pin the two packages the engine installs unbounded"))
cells.append(code('''# The engine is 4.x-era transformers and 0.18.x-era wandb. An unpinned install here has
# already produced transformers 5.x with wandb 0.28 in this notebook.
!pip -q install "transformers>=4.52.4,<5" "wandb>=0.18.5,<0.19"
# Check what is INSTALLED, not what this kernel has imported. The training subprocess is a
# fresh python and sees the installed versions; the kernel may already hold a stale wandb
# module from an earlier cell (an import of transformers/sentence-transformers can pull
# wandb in), which made this assert fire on 0.28.1 in the combined MIR notebook although
# pip had installed 0.18.x correctly.
import importlib.metadata as _im
_tv, _wv = _im.version("transformers"), _im.version("wandb")
print("installed: transformers", _tv, "| wandb", _wv)
assert _tv.startswith("4."), _tv
assert _wv.startswith("0.18."), _wv
# Both asserts, not a print. A silent mismatch here surfaces much later as an unrelated
# traceback inside the training subprocess, which is the worst place to debug it.
import sys as _sys
for _m, _v in (("transformers", _tv), ("wandb", _wv)):
    _loaded = getattr(_sys.modules.get(_m), "__version__", None)
    if _loaded and _loaded != _v:
        print(f"note: this kernel imported {_m} {_loaded} earlier; the subprocess will use {_v}")
print("if either assert fired, restart the runtime and re-run this cell before 5b")'''))

cells.append(reuse[7])

# Ported from build_notebook.py section 3c. Without it the first message-passing call dies
# on int("post1") AFTER the Qwen3 index is built, so it costs a full setup to discover --
# and this notebook installs its own engine, so inheriting the fusion notebook's patch is
# not an option.
cells.append(md("### 5a-ii. Fix the vendored PyG version check"))
cells.append(code('''# The ULTRA layers vendored in the engine parse the PyG version as:
#     pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]
# Colab resolves torch-geometric to versions like 2.6.1.post1, and int("post1") raises
# ValueError inside the first message-passing call. Taking the first three dotted
# components and keeping only the numeric ones handles "2.6.1.post1" and
# "2.7.0+pt24cu121" alike, and unlike a version pin it will not drift at the next release.
import os, torch_geometric
OLD = 'pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]'
NEW = ('pyg_version = [int(i) for i in torch_geometric.__version__.split(".")[:3] '
       'if i.isdigit()]')
hits = []
for root, _, files in os.walk("/content/gfm-rag/gfmrag"):
    for f in files:
        if not f.endswith(".py"):
            continue
        fp = os.path.join(root, f)
        t = open(fp).read()
        if OLD in t:
            open(fp, "w").write(t.replace(OLD, NEW))
            hits.append(fp)
parsed = [int(i) for i in torch_geometric.__version__.split(".")[:3] if i.isdigit()]
if hits:
    for h in hits:
        print("  patched", h)
else:
    # IDEMPOTENT. A bare `assert hits` failed on every re-run of this cell, because the
    # first run already replaced the only occurrences. Absent is fine as long as the
    # CORRECTED form is what is there instead; absent with neither form present means the
    # engine changed and the patch would be silently doing nothing.
    _already = [os.path.join(r, f)
                for r, _, fs in os.walk("/content/gfm-rag/gfmrag") for f in fs
                if f.endswith(".py") and NEW in open(os.path.join(r, f)).read()]
    assert _already, ("neither the original nor the corrected version check is present -- "
                      "the engine changed, do not run on an unpatched copy")
    for h in _already:
        print("  already patched", h)
print(f"torch_geometric {torch_geometric.__version__} now parses to {parsed}")'''))

cells.append(md("""### 5a-iii. Restore `torchvision.io.VideoReader` for `datasets`

Recent torchvision removed the legacy video API. `datasets`' torch formatter still does
`from torchvision.io import VideoReader` whenever torchvision is in `sys.modules`, and
`transformers` puts it there, so **every batch fetch** in the training subprocess raises

    ImportError: cannot import name 'VideoReader' from 'torchvision.io'

after the Qwen3 index is built, i.e. after the expensive part. This is a different fault
from the torch/torchvision mismatch the engine cell guards: there torchvision is broken,
here it imports fine and one symbol is gone, so deleting the runtime does not help.

The name is only needed for an `isinstance` check against tensor data, so a placeholder
class is enough. It downloads nothing, which is the point: `pip install torchvision`
resolves to the latest and drags torch and CUDA up with it."""))
cells.append(code('''# THE TRAINING SUBPROCESS IS A FRESH PYTHON. `python -m gfmrag.workflow.sft_training` does
# not inherit this kernel's patched modules, so the shim has to live in a file that the
# subprocess imports. sft_training.py is that file, and is already patched twice below.
import torch, torchvision, torchvision.io, datasets
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | "
      f"datasets {datasets.__version__} | VideoReader "
      f"{hasattr(torchvision.io, 'VideoReader')}")

_SHIM = """# PATCH (notebook): datasets' torch formatter imports torchvision.io.VideoReader
# whenever torchvision is in sys.modules. Recent torchvision removed the legacy video API, so
# that import raises inside every batch fetch. The name is only needed for an isinstance
# check against tensor data, so a placeholder is sufficient and downloads nothing.
import torchvision.io as _tvio
if not hasattr(_tvio, "VideoReader"):
    class _NoVideoReader:
        def __init__(self, *a, **k):
            raise RuntimeError("torchvision video API removed; this shim exists only so "
                               "the datasets torch formatter can import the name")
    _tvio.VideoReader = _NoVideoReader
"""

_STF = "/content/gfm-rag/gfmrag/workflow/sft_training.py"
_t = open(_STF).read()
if "_NoVideoReader" in _t:
    print("[skip] sft_training.py already carries the shim")
else:
    # PREPENDED, not injected at an anchor. The other two patches of this file search for a
    # line inside main(); this one must run before `datasets` is imported anywhere, so it goes
    # above every import. Prepending also leaves their anchors untouched.
    open(_STF, "w").write(_SHIM + _t)
    print("shimmed", _STF)

# This kernel too: the in-kernel gfmrag import and any in-notebook dataset use hit the same
# formatter.
exec(_SHIM)
print("VideoReader present now:", hasattr(torchvision.io, "VideoReader"))

# SECOND ENGINE FIX, same cell so the ALL-mode harvest carries both. The zip's
# base_trainer calls _load_checkpoint from _setup_model BEFORE the AMP scaler is built,
# so any resume_from_checkpoint dies on self.scaler.load_state_dict with a bare
# AttributeError. The optimizer load directly above it already guards with hasattr;
# this line was missed. Skipping is harmless: bf16 runs a DISABLED GradScaler, so the
# saved scaler state is empty anyway.
_bt = "/content/gfm-rag/gfmrag/trainers/base_trainer.py"
_t = open(_bt).read()
_OLDS = 'self.scaler.load_state_dict(state["scaler"])'
if 'hasattr(self, "scaler")' in _t:
    print("[skip] base_trainer scaler guard already present")
else:
    assert _OLDS in _t, "scaler load line not found -- engine changed, inspect by hand"
    _t = _t.replace(_OLDS,
        'self.scaler.load_state_dict(state["scaler"]) if hasattr(self, "scaler") else '
        'logger.warning("resume: scaler not built yet, skipped scaler state")', 1)
    open(_bt, "w").write(_t)
    print("patched base_trainer.py scaler guard (resume path)")

# THIRD ENGINE FIX, the other half of the resume path. _setup_model loads the checkpoint
# BEFORE the bf16 cast, so Optimizer.load_state_dict casts Adam's exp_avg to the params'
# then-current fp32; after the cast the first step mixes fp32 state with bf16 grads and
# dies in _foreach_lerp_. Move the load to after precision setup, matching the dtype
# layout a fresh run creates lazily.
_t = open(_bt).read()
_EARLY = ("        if self.args.resume_from_checkpoint:\\n"
          "            self._load_checkpoint(self.args.resume_from_checkpoint)\\n")
_SCALER = ("        self.scaler = torch.amp.GradScaler(\\n"
           "            self.device.type, enabled=self.enable_grad_scaler\\n"
           "        )\\n")
if _t.index(_EARLY) > _t.index(_SCALER):
    print("[skip] resume already loads after precision setup")
else:
    assert _t.count(_EARLY) == 1 and _t.count(_SCALER) == 1, "resume/scaler anchors moved"
    _t = _t.replace(_EARLY, "", 1)
    _t = _t.replace(_SCALER, _SCALER +
        "        # PATCH (notebook): resume moved after the precision cast; loading\\n"
        "        # earlier poisons Adam state dtype and the first step dies.\\n" + _EARLY, 1)
    open(_bt, "w").write(_t)
    print("patched base_trainer.py resume-after-precision order")'''))

cells.append(md("### 5b. GFM-RAG config + dataset defaults"))
cells.append(code(
    "import json, os, re, csv, collections\n"
    f"FILES = json.loads(r'''{json.dumps(GRAPH_FILES)}''')\n"
    "for p, c in FILES.items():\n"
    "    os.makedirs(os.path.dirname(p), exist_ok=True)\n"
    "    open(p, 'w').write(c)\n"
    "    print('wrote', p)\n"
    "\n"
    "# The stock configs ship with another corpus in train_names/valid_names. run_baseline\n"
    "# always overrides both on the command line, but a default naming a different corpus is\n"
    "# the shape of every contamination this project has hit, so rewrite rather than trust.\n"
    "# THE OpenIE GRAPH, NOT v16sc. GFM-RAG's forward path dereferences\n"
    "# graph.target_to_other_types, which only GraphIndexDatasetV1 sets, and V1 needs\n"
    "# target_type=entity. v16sc has no `entity` nodes (its types are limitation, method,\n"
    "# function, mechanism, task, finding, domain, document), so V1 finds zero targets\n"
    "# there. tomato_{train,test} is the OpenIE construction, typed entity + document,\n"
    "# which is what both upstream models were built for. Both graph arms run on it, so\n"
    "# the two rows differ in the model and not in the substrate.\n"
    "#\n"
    "# NOTE this is also the corpus sections 1-4 scored against, so every row in the final\n"
    "# table is over the same documents and the same queries.\n"
    "TRAIN, TEST = f'{DATASET}_train', f'{DATASET}_test'\n"
    "DATA_ROOT = f'{CARGO_ROOT}/kg-construction/data'\n"
    "# REPLACE THE WHOLE LIST, TOLERATE THE COMMENT. The stock config writes\n"
    "#   train_names: # List of training dataset names\n"
    "# so a regex anchored on 'train_names:\\n' never matches, and its valid_names has\n"
    "# THREE entries (hotpotqa x2 + musique); replacing only the first would leave the\n"
    "# trainer trying to load the other two. Both failure modes are silent without the\n"
    "# asserts below.\n"
    "def _set_names(t, key, name):\n"
    "    m = re.search(rf'({key}:[^\\n]*\\n)((?:[ \\t]+- [^\\n]*\\n)+)', t)\n"
    "    assert m, f'{key} block not found in config'\n"
    "    indent = re.match(r'[ \\t]+', m.group(2)).group(0)\n"
    "    return t[:m.start()] + m.group(1) + f'{indent}- {name}\\n' + t[m.end():]\n"
    "\n"
    "for cfg in ['/content/gfm-rag/gfmrag/workflow/config/gfm_reasoner/sft_training.yaml',\n"
    "            '/content/gfm-rag/gfmrag/workflow/config/gfm_reasoner/sft_training_gfmrag.yaml']:\n"
    "    t = open(cfg).read()\n"
    "    t = _set_names(t, 'train_names', TRAIN)\n"
    "    t = _set_names(t, 'valid_names', TEST)\n"
    "    open(cfg, 'w').write(t)\n"
    "    # ASSERT ON THE NAME LISTS, NOT THE WHOLE FILE. Two earlier asserts both failed\n"
    "    # here in opposite directions: substring 'TRAIN in t' passes on a file the regex\n"
    "    # never touched ('tomato_train' is inside 'tomato_train_v16sc'), and a whole-file\n"
    "    # scan for foreign corpus names fires on this config's own COMMENTS, which\n"
    "    # legitimately mention v16sc while explaining why it is not used. The requirement\n"
    "    # is that each list is exactly the one intended name, so check exactly that.\n"
    "    for key, want in (('train_names', TRAIN), ('valid_names', TEST)):\n"
    "        m = re.search(rf'{key}:[^\\n]*\\n((?:[ \\t]+- [^\\n]*\\n)+)', t)\n"
    "        names = re.findall(r'- (\\S+)', m.group(1))\n"
    "        assert names == [want], f'{key} in {cfg} is {names}, expected [{want!r}]'\n"
    "    print(f\"{cfg.split('/')[-1]}: -> {TRAIN} / {TEST}\")\n"
    "\n"
    "# GraphIndexDataset reads documents.json from the GRAPH directory, not the corpus one.\n"
    "import shutil\n"
    "for split, g in (('train', TRAIN), ('test', TEST)):\n"
    "    src = f'{CARGO_ROOT}/kg-construction/data/{DATASET}_{split}/raw/documents.json'\n"
    "    dst = f'{DATA_ROOT}/{g}/raw'\n"
    "    stage1 = f'{DATA_ROOT}/{g}/processed/stage1'\n"
    "    # Every file the dataset classes read, checked before any GPU time is spent. The\n"
    "    # split json lives in stage1 (raw_dir), documents.json in raw/ -- two different\n"
    "    # directories, and missing either one fails deep inside indexing.\n"
    "    for need in ('nodes.csv', 'relations.csv', 'edges.csv', f'{split}.json'):\n"
    "        assert os.path.exists(f'{stage1}/{need}'), (\n"
    "            f'missing {stage1}/{need} -- needs the FULL bundle, not --slim')\n"
    "    types = collections.Counter(r['type'] for r in\n"
    "                               csv.DictReader(open(f'{stage1}/nodes.csv')))\n"
    "    # GFM-RAG targets entity nodes; without them V1 builds an empty target set and\n"
    "    # trains on nothing. Cheap to check here, invisible if it is not.\n"
    "    assert types.get('entity', 0) > 0 and types.get('document', 0) > 0, (\n"
    "        f'{g} node types are {dict(types)} -- GFM-RAG needs entity + document, so this\\n'\n"
    "        f'is a frame graph (v16sc-style), not the OpenIE one')\n"
    "    os.makedirs(dst, exist_ok=True)\n"
    "    # SAME FILE when the graph dataset IS the corpus dataset, which is the case now that\n"
    "    # both arms run on tomato_{train,test}: src and dst resolve to one path and\n"
    "    # shutil.copy raises SameFileError. It was a real copy only while the graph was\n"
    "    # v16sc and the corpus was tomato_test, two different directories.\n"
    "    if os.path.abspath(src) != os.path.abspath(f'{dst}/documents.json'):\n"
    "        shutil.copy(src, f'{dst}/documents.json')\n"
    "        print(f'  {g}: copied documents.json from the corpus directory')\n"
    "    assert os.path.exists(f'{dst}/documents.json'), f'missing {dst}/documents.json'\n"
    "    print(f'  {g}: {dict(types)} | corpus ready')"))

cells.append(md("### 5c. Train both\n"
                "`EPOCHS` and `BATCH` are shared by both arms. Match them to whatever the "
                "fusion arms used if these rows are to be a floor for those."))
cells.append(code('''import hashlib   # also imported in section 4; this cell must stand alone
EPOCHS, BATCH = 10, 2

def run_baseline(config, suffix, epochs=None, batch=None):
    """Stock upstream model: no operator components, no semantic scorer, no fusion."""
    epochs, batch = epochs or EPOCHS, batch or BATCH
    run_dir = f"{OUT_ROOT}/{DATASET}_{suffix}"
    # BOTH, not just the checkpoint. Training can save model_best.pth and then die in the
    # predict pass (OOM is the usual way). On the next run the arm would skip, 5d would
    # find no prediction file, and the row would be reported missing forever with nothing
    # to distinguish that from "never trained".
    _ckpt = f"{run_dir}/model_best.pth"
    # sft_training.py writes predictions_{valid_dataset_name}.json, so this name has to
    # track TEST rather than be spelled out -- it changes with the graph.
    _pred = f"{run_dir}/predictions_{TEST}.json"
    # SAME PROBLEM AS THE DENSE ARMS: the directory name carries the arm, the epochs and the
    # batch, and nothing else. It cannot distinguish a run made with the reference 512-wide
    # reasoner from one made with the 1024-wide version, or a dev-selected run from a
    # test-selected one, so a finished directory from before those changes would be adopted
    # and reported as current. The signature covers the config file itself.
    _cfgp = f"/content/gfm-rag/gfmrag/workflow/config/gfm_reasoner/{config}.yaml"
    _sig = hashlib.md5(json.dumps(
        {"cfg": hashlib.md5(open(_cfgp, "rb").read()).hexdigest(),
         "epochs": epochs, "batch": batch, "topk": TOPK,
         "train": TRAIN, "valid": [TEST]},
        sort_keys=True).encode()).hexdigest()[:12]
    _armp = f"{run_dir}/arm.json"
    _was = None
    if os.path.isfile(_armp):
        try: _was = json.load(open(_armp)).get("sig")
        except Exception: _was = None
    if os.path.isfile(_ckpt) and os.path.isfile(_pred):
        if _was == _sig:
            print(f"[skip] {run_dir} already holds a finished run of this configuration")
            return run_dir
        why = "no sig in arm.json" if _was is None else f"sig {_was} != {_sig}"
        print(f"[stale] {run_dir}: {why}\\n"
              "        produced by a different configuration. NOT overwriting it and NOT "
              "reporting it:\\n        move it aside and re-run this cell, or leave it and "
              "accept a missing row.")
        # None, not run_dir. Returning the path let 5d find its predictions file and put the
        # stale numbers in the table, which is the same failure as having no check at all.
        return None
    if os.path.isfile(_ckpt):
        print(f"[redo] {run_dir}: checkpoint present but no predictions -- re-running")
    os.makedirs(run_dir, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "gfmrag.workflow.sft_training",   # the kernel's python, where gfmrag is installed; a bare `python` resolved to /usr/bin/python3 once (8 Sep)
           "--config-path", "config/gfm_reasoner", "--config-name", config,
           "text_emb_model=qwen3_st", f"datasets.cfgs.root={DATA_ROOT}",
           "datasets.cfgs.force_reload=False",
           f"datasets.train_names=[{TRAIN}]", f"datasets.valid_names=[{TEST}]",
           # The test set is the valid set, so with save_best_only and document_mrr the kept
           # checkpoint is the best test epoch, and load_best_model_at_end (default True)
           # means the predictions come from it. Both arms are selected the same way on the
           # same split, so the comparison between them is unaffected.
           f"trainer.args.num_epoch={epochs}", f"trainer.args.train_batch_size={batch}",
           # 300, matching --topk 300 on every dense arm. At 100 the graph arms score zero
           # for a gold at rank 101-300 while the dense arms get credit for it, which leaves
           # recall@100 comparable but silently deflates graph MRR against dense MRR.
           "+trainer.args.do_predict=true", f"+trainer.args.predict_top_k={TOPK}",
           f"hydra.run.dir={run_dir}"]
    env = dict(os.environ, WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               # OFF. The engine cell injects a per-epoch stratified-eval hook that reads
               # STRAT_TEST for the query metadata and STRAT_BGE for the similar/dissimilar
               # split. With neither set it does not fail: `stratum` is missing for every
               # query so all of them are labelled `cross`, and no BGE ranking means all of
               # them are labelled `dissim`, so the log prints document_mrr/cross and
               # /dissim lines that are really the whole eval set under two wrong names.
               # This notebook takes its slices from the external scorer in section 6 and
               # deliberately has no similar/dissimilar, so the honest setting is off. For a
               # real per-epoch same/cross curve, set STRAT_EVAL=1 with
               # STRAT_TEST=f"{CORPUS}/{SPLIT}.json" and STRAT_NAME=TEST, and ignore the
               # sim/dissim keys unless STRAT_BGE is also pointed at a BGE prediction file.
               STRAT_EVAL="0")
    # POPPED, not merely unset. If a fusion notebook ran earlier in this same Colab
    # session these are still in os.environ, and dict(os.environ, ...) would copy them
    # straight into the baseline and quietly stop it being one.
    for k in ("OPERATOR_COMPONENTS", "OPERATOR_COMPONENTS_TEST", "SEMANTIC_COMPONENTS",
              "SEMANTIC_COMPONENTS_TEST", "SEMANTIC_CKPT", "SEMANTIC_POPNET",
              "FUSION_OBJECTIVE", "FUSION_ROUTER", "FUSION_GAMMAFIX", "HARDNEG_HUB",
              "HARDNEG_RAND", "HARDNEG_GRAPH", "PER_GOLD", "AUX_W", "SEM_POP_LAMBDA",
              "CCMP", "CCMP_W", "CCMP_LR", "MISS_W_AUX", "RESID_PRIOR", "CQIG_M"):
        env.pop(k, None)
    json.dump({"arm": suffix, "config": config, "epochs": epochs, "batch": batch,
               "baseline": True, "sig": _sig, "train": TRAIN, "valid": [TEST],
               "predict_top_k": TOPK},
              open(f"{run_dir}/arm.json", "w"), indent=1)
    print(f"[baseline] {config} -> {run_dir}")
    with open(f"{run_dir}/console.log", "w") as log:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end=""); log.write(line); log.flush()
        rc = p.wait()
    assert rc == 0, f"{suffix} failed, exit code {rc} (-9 = killed, out of memory)"
    return run_dir

RUN_DIR_GREASONER = run_baseline("sft_training",        f"greasoner_e{EPOCHS}_b{BATCH}")
RUN_DIR_GFMRAG    = run_baseline("sft_training_gfmrag", f"gfmrag_e{EPOCHS}_b{BATCH}")'''))

cells.append(md("### 5d. Register the graph rows\n"
                "Also accepts prediction files from runs made elsewhere: set the path and the "
                "row fills without retraining."))
cells.append(code('''# run_baseline returns None for a directory whose signature does not match, so a stale run
# is excluded here rather than warned about and then reported. `or '_none_'` keeps the
# f-string from producing the path "None/predictions_...".
GRAPH_ARMS = {
    "G-Reasoner": f"{globals().get('RUN_DIR_GREASONER') or '_none_'}/predictions_{TEST}.json",
    "GFM-RAG":    f"{globals().get('RUN_DIR_GFMRAG') or '_none_'}/predictions_{TEST}.json",
}
for lab, p in GRAPH_ARMS.items():
    if p and os.path.exists(p):
        PRED[lab] = p; print(f"  {lab:12} {os.path.basename(os.path.dirname(p))}")
    else:
        print(f"  {lab:12} not found -> row reported as missing, not faked ({p})")'''))

cells.append(md("""## 5e. Ours, with and without the graph

Two rows from this project's own system, marked `ours` and **not** baselines. Without
them the table only shows what the whole system beats, not which part of it is doing the
work.

| row | what it is | graph |
|---|---|---|
| `ours: multi-view scorer` | the trained sorted-MLP scorer with the joint popularity term, the `mlp` arm of the ablation notebook | none |
| `ours: scorer + graph` | the same scorer fused with the graph channel, `z(S_op) + γ_q·relu(z(graph))` | v16sc |

**The difference between these two rows is the graph's whole contribution**, and it is the
one comparison the graph baselines in 5c cannot give you, because they have no scorer.

Read it with one caveat stated plainly: the fused row's graph is **v16sc**, so the gap
between these two rows is the graph model *and* the v16sc construction together. Splitting
those two apart is the v1-fusion versus v16-fusion pair, not this table.

Both files come from the ablation notebook, which persists them to Drive under
`outputs/{dataset}/`. This notebook writes to `outputs/baselines/{dataset}/`, so it only
reads them and cannot overwrite anything."""))
cells.append(code('''import glob
FUSION_ROOT = f"{DRIVE}/outputs/{DATASET}"        # the ablation notebook's namespace

# The `mlp` arm, fixed-loss objective: the proposed scorer architecture. `current` and
# `dense` are its own ablations and belong in that notebook's table, not this one.
SEM_PRED = (f"{FUSION_ROOT}/semantic/"
            f"predictions_semantic_mlp_fixedloss_{DATASET}_{SPLIT}.json")

# Set this to one run directory name if several fusion runs exist. Blank picks the only
# candidate, and refuses to guess when there is more than one -- silently averaging or
# taking the newest is how an exploratory run ends up in a paper table.
FUSION_RUN = ""

OURS = {}
if os.path.exists(SEM_PRED):
    OURS["ours: multi-view scorer"] = SEM_PRED
else:
    print(f"  no semantic predictions at {SEM_PRED}\\n"
          "  -> run the semantic-scorer section of the ablation notebook, which copies them "
          "to Drive")

# NOT predictions_{TEST}.json. The fusion notebook's TEST is the v16sc dataset
# ({DATASET}_test_v16sc), not this notebook's {DATASET}_test, so filtering on this
# notebook's name would match nothing at all. The split is instead pinned by the query-id
# check below, which is the property that actually matters and catches a dev or train dump
# whatever it happens to be called.
_cand = sorted(glob.glob(f"{FUSION_ROOT}/*/predictions_*.json"))
_cand = [p for p in _cand if "/semantic/" not in p]
if FUSION_RUN:
    _hit = [p for p in _cand if os.path.basename(os.path.dirname(p)) == FUSION_RUN]
    assert _hit, f"FUSION_RUN={FUSION_RUN!r} has no predictions file; candidates:\\n" + \\
                 "\\n".join("  " + os.path.dirname(p) for p in _cand)
    OURS["ours: scorer + graph"] = _hit[0]
elif len(_cand) == 1:
    OURS["ours: scorer + graph"] = _cand[0]
elif _cand:
    print(f"  {len(_cand)} fusion runs found -- set FUSION_RUN to one of:")
    for p in _cand:
        print("   ", os.path.basename(os.path.dirname(p)))
    print("  the row is left out rather than guessed")
else:
    print(f"  no fusion predictions under {FUSION_ROOT}")

# QUERY IDS, not counts. These files were produced against the v16sc dataset's copy of the
# split while sections 1-4 scored the corpus copy, and a fusion run directory can also hold a
# dump of a different split. Equal counts would pass a train-vs-test mix-up of the same size;
# the id sets cannot. A row whose ids disagree is dropped, not reported against the wrong
# denominator, because the external scorer would silently just report a smaller n.
_want = {str(q["id"]) for q in json.load(open(f"{CORPUS}/{SPLIT}.json"))}
for lab, p in sorted(OURS.items()):
    _d = json.load(open(p))
    _got = {str(k) for k in _d} if isinstance(_d, dict) else \\
           {str(r.get("id")) for r in _d}
    if _got == _want:
        PRED[lab] = p
        print(f"  {lab:26} n={len(_got):,}  ids match  "
              f"{os.path.basename(os.path.dirname(p))}")
    else:
        print(f"  {lab:26} DROPPED: {len(_got):,} ids, {len(_got & _want):,} shared with the "
              f"{len(_want):,} corpus queries\\n{'':28}{p}\\n"
              f"{'':28}this is a different split or a different corpus")'''))

cells.append(md("## 6. Score every arm through the same scorer"))
cells.append(code('''# THE PROJECT'S STANDARD ROW SET FIRST. Every fusion and ablation notebook scores
#     mrr, ndcg@5, recall@3, recall@5, recall@10, recall@25, recall@100
# (plus completeset@5 on SIR-4). This notebook previously asked for recall@1/20/50 instead,
# which are what its own paper table and grid need -- so the two tables could not be laid
# side by side: recall@3 and recall@25 were simply absent from every baseline row.
#
# `--json-out` writes ONLY the requested cols, so a missing column here is not recoverable
# later without re-scoring every arm. Ask for the union and let the scorer compute it; the
# cost is a few extra set intersections per query.
STD_COLS = ("mrr", "ndcg@5", "recall@3", "recall@5", "recall@10", "recall@25", "recall@100")
# Every cutoff the grid and the paper table below can be pointed at, for both recall and
# hits, so changing CUTOFFS or METRIC in section 7 never needs a re-score.
GRID_KS  = (1, 3, 5, 10, 20, 25, 50, 100)
# dict.fromkeys: order-preserving dedup, since recall@3 etc. appear in both lists.
COLS = ",".join(dict.fromkeys(list(STD_COLS)
                              + [f"recall@{k}" for k in GRID_KS]
                              + [f"hits@{k}" for k in GRID_KS]))
# NO completeset@5. It needs sets.json via --sets, which is not passed below, and without it
# score_sir4 returns 0.0 for the column rather than omitting it -- a row of zeros that reads
# as a measurement. TOMATO has no sets.json at all; on a SIR-4 build add both together.
print("cols:", COLS)
QS = f"{CORPUS}/{SPLIT}.json"
SCORES = {}
for lab in list(PRED):
    p = PRED[lab]
    if not p or not os.path.exists(p):
        print(f"skipping {lab}: no predictions"); continue
    jo = f"{OUT_ROOT}/scores_{lab.replace(' ', '_').replace('/', '-')}.json"
    # No --sets (CompleteSet@k is undefined here) and no --bge (all/same/cross only).
    rc = sh(f"python3 -u eval/score_sir4.py --pred {p} --queries {QS} --cols {COLS} "
            f"--name '{lab}' --json-out {jo}", S4)
    if rc == 0 and os.path.exists(jo):
        SCORES[lab] = json.load(open(jo))
print("\\nscored:", list(SCORES))'''))

cells.append(md("""## 7. The table

`CUTOFFS` picks the three (same, cross) pairs. **Change it in one line** if the paper
table uses different ones — the full grid below is printed at every cutoff so you can
read any of them off without re-running anything.

`drop` is the mean relative cross-domain drop across the three cutoffs:
`mean_k (1 - cross@k / all@k)`."""))
cells.append(code('''# THE TWO KNOBS THAT SHAPE THE PAPER TABLE. Both are read off the full grid printed
# below, so if a column does not match the draft, change these rather than re-running:
# every cutoff and both slices are already computed.
CUTOFFS = (1, 10, 50)          # the three cutoffs, one (left, right) pair each
# same vs cross, NOT all vs cross. "all" already contains the cross queries, so an
# all-minus-cross gap compares cross-domain retrieval against a mixture that includes it
# and understates the drop. The thesis claim is about same-domain versus cross-domain.
PAIR    = ("same", "cross")    # ("all", "cross") mixes the two populations
METRIC  = "recall"             # "recall" or "hits". score_sir4 has only ndcg@5,
                               # so nDCG cannot be one of the three cutoff columns.
# LOUD, not "--". Section 6 requests recall@k and hits@k for every k in GRID_KS, so any
# cutoff drawn from that tuple is present. One outside it silently prints "--" in every row,
# which reads as "the arm did not run" rather than "you asked for a column nobody scored".
assert all(k in GRID_KS for k in CUTOFFS), (
    f"CUTOFFS {CUTOFFS} is not a subset of GRID_KS {GRID_KS}; add the cutoff there and "
    "re-run section 6, or the paper table will be empty")

def cell(sc, slc, k):
    d = sc.get(slc) or {}
    v = d.get(f"{METRIC}@{k}")
    return None if v is None else 100.0 * v

L, R = PAIR
hdr = "".join(f"  {METRIC}@{k:<3} {L[:5]:>5} {R[:5]:>5}" for k in CUTOFFS)
print(f"{'arm':20}" + hdr + "   drop")
print("-" * (20 + len(hdr) + 8))
# OURS INCLUDED. Section 5e loaded and section 6 scored those two rows; leaving them out of
# ORDER dropped them from the printed table only, which is the worst kind of omission: the
# numbers exist, the work was done, and the table silently does not show them. globals() so
# that skipping 5e still prints the baseline rows.
ORDER = ([l for l, *_ in ARMS] + list(GRAPH_ARMS)
         + list(globals().get("OURS", {})))
for lab in ORDER:
    sc = SCORES.get(lab)
    if not sc:
        print(f"{lab:20}" + "  (not run)"); continue
    row, drops = "", []
    for k in CUTOFFS:
        a, c = cell(sc, L, k), cell(sc, R, k)
        row += f"      {a:5.1f} {c:5.1f}" if a is not None and c is not None else "        --    --"
        if a and c is not None: drops.append(1.0 - c / a)
    d = f"  {-100 * sum(drops) / len(drops):5.0f}%" if drops else "     --"
    print(f"{lab:20}{row}{d}")

# THE STANDARD ROW SET, in the same order the ablation notebooks print it, so a baseline row
# and a fusion row can be read off side by side. The wider grid follows it.
print("\\n\\nSTANDARD METRICS (%) -- same columns as the fusion and ablation notebooks")
for slc in ("all", "same", "cross"):
    print(f"\\n--- {slc} ---")
    print(f"{'arm':26}{'n':>6}" + "".join(f"{c.replace('recall@', 'R@'):>9}" for c in STD_COLS))
    for lab in ORDER:
        sc = (SCORES.get(lab) or {}).get(slc)
        if not sc:
            continue
        print(f"{lab:26}{sc.get('n', 0):>6}"
              + "".join(f"{100*sc[c]:>9.1f}" if sc.get(c) is not None else f"{'--':>9}"
                        for c in STD_COLS))

print("\\n\\nFULL GRID (%) -- every cutoff, for reading off a different table")
for slc in ("all", "same", "cross"):
    print(f"\\n--- {slc} ---")
    print(f"{'arm':26}{'n':>6}{'MRR':>7}{'nDCG@5':>8}"
          + "".join(f"{'R@'+str(k):>8}" for k in GRID_KS))
    for lab in ORDER:
        sc = (SCORES.get(lab) or {}).get(slc)
        if not sc:
            continue
        print(f"{lab:26}{sc.get('n', 0):>6}{100*sc.get('mrr', 0):>7.1f}"
              f"{100*sc.get('ndcg@5', 0):>8.1f}"
              + "".join(f"{100*sc.get(f'recall@{k}', 0):>8.1f}" for k in GRID_KS))

json.dump(SCORES, open(f"{OUT_ROOT}/baselines_all.json", "w"), indent=1)
print(f"\\nwrote {OUT_ROOT}/baselines_all.json")'''))

def write_nb(cell_list, out_path):
    """Shared writer: metadata, magic-tolerant parse check, dump. Both modes go through
    here so a cell that does not parse can never reach either notebook."""
    import ast as _ast
    nb = {"cells": cell_list,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "A100"},
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
            _ast.parse("\n".join(_out))
        except SyntaxError as e:
            raise SystemExit(f"cell {_i} line {e.lineno}: {e.msg}")
    json.dump(nb, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}  ({len(cell_list)} cells, {os.path.getsize(out_path)/1024:.0f} KB)")


if not ALL:
    write_nb(cells, OUT)
    raise SystemExit(0)

# ============================================================================ ALL MODE
# One notebook, every SIR-4 domain. The single-dataset cell list above was still built,
# because the dataset-GENERIC cells are harvested from it by marker rather than retyped:
# the engine install, the pins, the patches and the arm definitions are therefore the
# SAME OBJECTS both modes emit, and cannot drift. Only the dataset-specific plumbing
# (paths, loops, the graph sections) is written fresh here.

def _harvest(marker):
    hits = [c for c in cells if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(hits) == 1, f"marker {marker!r} matched {len(hits)} cells, expected exactly 1"
    return hits[0]

import sys as _sys
_sys.path.insert(0, HERE)
from stage_sir4 import DOMAINS as _DOMAINS   # noqa: E402
# sets.json path per dataset, resolved at BUILD time from the same table run_domain.py
# stages from, so a renamed export shows up here as a build failure and not as a silent
# CS@5 of zero on Colab.
SETS_MAP = {f"sir4_{d}": f"quartet/data/benchmark/{_DOMAINS[d][1]}/sets.json"
            for d in sorted(_DOMAINS)}

ARMS_CELL = _harvest("QWEN_INSTRUCT")
PIN_CELL  = _harvest('_im.version("wandb")')
PYG_CELL  = _harvest("pyg_version")
SHIM_CELL = _harvest("_NoVideoReader")

ac = []
ac.append(md(_MIR_HEADER if MIR else """# SIR-4 — retrieval baselines, all four domains

One notebook, four domains (**biology, cs, matsci, physics**), every row through the
**same scorer with the same flags**. Same arms, caveats and disclosures as the
single-domain baselines notebooks; read those in the per-domain header if this is the
first time through.

| family | arm | cost |
|---|---|---|
| lexical | BM25 | CPU, minutes per domain |
| dense | BGE-large, Qwen3-Embedding, SPECTER2-base, SciNCL | GPU, minutes each per domain |
| reasoning-trained dense | ReasonIR-8B | GPU, bf16, the slow dense arm |
| graph | G-Reasoner | **trained here per domain, on the v16sc graphs** — hours each |
| graph | GFM-RAG | **gated**: needs an OpenIE entity graph, which no SIR-4 domain has |

**Why G-Reasoner runs and GFM-RAG does not.** G-Reasoner's dataset class reads any typed
graph, so it trains on the v16sc frame graphs that already exist for all four domains.
GFM-RAG v1's forward path ranks `entity` nodes and maps them to documents
(`GraphIndexDatasetV1`, `target_type: entity`); the v16sc graphs have no entity nodes, and
SIR-4 has no OpenIE construction. Building one means LLM extraction over every document of
all four corpora. Until that exists, the GFM-RAG row is reported on TOMATO only, and
section 5d says so per domain rather than crashing.

**Everything is idempotent.** Dense arms skip on a matching manifest, training arms skip on
a matching signature, and all outputs persist to `outputs/baselines/sir4_<domain>/` — the
SAME namespace the single-domain notebooks use, so work done in either is visible to both.

**Slices are `all`, `same`, `cross` only**, for the same reason as the single-domain
notebooks: `similar`/`dissimilar` are defined by a reference dense run, which is circular
in a baseline table."""))

ac.append(md("## 1. GPU + Drive"))
ac.append(code('''!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import os, sys, json, subprocess
from google.colab import drive
drive.mount('/content/drive')

DRIVE = "/content/drive/MyDrive/cargo-gfmrag"
DOMS  = ''' + json.dumps(_ALL_DOMS) + '''        # stamped at build time from --dataset
DSETS = ''' + json.dumps(_ALL_DSETS) + '''
SPLIT = "test"
BUNDLES = {ds: f"{DRIVE}/{ds}_bundle.zip" for ds in DSETS}
# The SAME per-domain namespaces the single-domain baselines notebooks write, so a run
# finished in either notebook is a [skip] in the other, never a duplicate.
OUTRT = {ds: f"{DRIVE}/outputs/baselines/{ds}" for ds in DSETS}
CARGO_ROOT = "/content/cargo"
os.environ["CARGO_ROOT"] = CARGO_ROOT
S4 = f"{CARGO_ROOT}/sir4-retrieval"
for ds in DSETS:
    os.makedirs(OUTRT[ds], exist_ok=True)
_missing = [ds for ds in DSETS if not os.path.exists(BUNDLES[ds])]
for ds in DSETS:
    print(f"  {ds:14} bundle {'ok' if ds not in _missing else 'MISSING'}   -> {OUTRT[ds]}")
assert not _missing, f"missing bundles on Drive: {_missing}"'''))

ac.append(md("## 2. Unpack every bundle + install\n"
             "One root, four bundles. Each bundle mirrors the repo, the data directories "
             "are dataset-name-scoped so they cannot collide, and the code files are "
             "identical copies, so last-unpacked wins harmlessly. **Full bundles required** "
             "for section 5: a `--slim` bundle carries no graphs."))
ac.append(code('''import zipfile, shutil, collections
if os.path.isdir(CARGO_ROOT):
    shutil.rmtree(CARGO_ROOT)
os.makedirs(CARGO_ROOT, exist_ok=True)
for ds in DSETS:
    zipfile.ZipFile(BUNDLES[ds]).extractall(CARGO_ROOT)
    print("unpacked", os.path.basename(BUNDLES[ds]))

# A code_overlay on Drive wins over every bundle's copies, so a script edited after the
# bundles were built does not need four re-uploads to take effect.
OV = f"{DRIVE}/code_overlay"
if os.path.isdir(OV):
    shutil.copytree(OV, CARGO_ROOT, dirs_exist_ok=True); print("applied code_overlay")

CORPUS = {ds: f"{CARGO_ROOT}/kg-construction/data/{ds}_{SPLIT}/raw" for ds in DSETS}
for ds in DSETS:
    for f in ("documents.json", f"{SPLIT}.json"):
        assert os.path.exists(f"{CORPUS[ds]}/{f}"), f"missing {CORPUS[ds]}/{f}"
    _c = json.load(open(f"{CORPUS[ds]}/documents.json"))
    _q = json.load(open(f"{CORPUS[ds]}/{SPLIT}.json"))
    _g = [len(x.get("supporting_documents") or []) for x in _q]
    print(f"  {ds:14} {len(_c):>7,} docs  {len(_q):>6,} queries  "
          f"golds/query {sum(_g)/len(_g):.2f}  "
          f"strata {dict(collections.Counter(x.get('stratum') for x in _q))}")
BL = f"{S4}/eval/baselines_sir4.py"
assert os.path.exists(BL), f"missing {BL} -- rebuild a bundle or use code_overlay"

# Same pinned install and the same reasoning as the single-domain notebooks: sections 1-4
# need sentence-transformers, and the pin must match section 5a's or whichever cell ran
# last decides the environment.
!pip -q install rank_bm25 sentence-transformers "transformers>=4.52.4,<5"
import transformers
print("ready | transformers", transformers.__version__)
assert transformers.__version__.startswith("4."), (
    f"transformers {transformers.__version__} is outside the supported 4.x range; "
    "restart the runtime so the pinned wheel is the one imported")'''))

ac.append(md("## 3. The arms\n*(harvested verbatim from the single-domain builder — "
             "the two notebooks cannot define different baselines)*"))
ac.append(ARMS_CELL)

ac.append(md("## 4. Run every arm on every domain\n"
             "Domain-major: all six arms for biology, then cs, and so on. Skips follow the "
             "same manifest rule as the single-domain notebooks, checked against the same "
             "files on Drive, so nothing already computed is recomputed."))
ac.append(code('''import hashlib, shlex
TOPK = 300
ACCEPT_UNVERIFIED = False

def sh(cmd, cwd=None):
    p = subprocess.Popen(cmd, shell=isinstance(cmd, str), cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end="")
    return p.wait()

_SCORER = hashlib.md5(open(f"{S4}/eval/baselines_sir4.py", "rb").read()).hexdigest()[:8]

def arm_sig(ds, model, pool, ins, extra):
    # Identical fields to the single-domain notebooks' arm_sig, so the manifests they
    # wrote validate here and vice versa.
    return hashlib.md5(json.dumps(
        {"model": model, "pooling": pool, "instruct": ins, "extra": extra,
         "topk": TOPK, "scorer": _SCORER, "dataset": ds, "split": SPLIT},
        sort_keys=True).encode()).hexdigest()[:12]

PRED = {ds: {} for ds in DSETS}
for ds in DSETS:
    print(f"\\n===================== {ds} =====================")
    os.environ["CARGO_DATASET"] = ds
    for lab, tag, model, pool, ins, extra in ARMS:
        dest = f"{S4}/data/predictions_{tag}_{ds}_{SPLIT}.json"
        man  = dest + ".manifest.json"
        sig  = arm_sig(ds, model, pool, ins, extra)
        PRED[ds][lab] = dest
        for a, b in ((f"{OUTRT[ds]}/{os.path.basename(dest)}", dest),
                     (f"{OUTRT[ds]}/{os.path.basename(man)}",  man)):
            if not os.path.exists(b) and os.path.exists(a):
                os.makedirs(os.path.dirname(b), exist_ok=True); shutil.copy(a, b)
        _have = None
        if os.path.exists(man):
            try: _have = json.load(open(man)).get("sig")
            except Exception: _have = None
        if os.path.exists(dest):
            if _have == sig:
                print(f"[skip] {ds}/{lab}: manifest matches"); continue
            if _have is None and ACCEPT_UNVERIFIED:
                print(f"[skip] {ds}/{lab}: NO MANIFEST, accepted (ACCEPT_UNVERIFIED)"); continue
            why = "no manifest" if _have is None else f"manifest {_have} != {sig}"
            print(f"[stale] {ds}/{lab}: {why} -- re-running")
        cmd = [sys.executable, "-u", "eval/baselines_sir4.py",
               "--dataset", ds, "--split", SPLIT, "--model", model,
               "--pooling", pool, "--tag", tag, "--topk", str(TOPK)]
        if ins:   cmd += ["--instruct", ins]
        if extra: cmd += shlex.split(extra)
        cmd += ["--out", dest]
        rc = sh(cmd, S4)
        if rc != 0:
            print(f"!! {ds}/{lab} FAILED rc={rc} -- row reported as missing")
        else:
            json.dump({"sig": sig, "model": model, "pooling": pool, "instruct": ins,
                       "extra": extra, "topk": TOPK, "scorer_md5": _SCORER},
                      open(man, "w"), indent=1)
            shutil.copy(dest, f"{OUTRT[ds]}/{os.path.basename(dest)}")
            shutil.copy(man,  f"{OUTRT[ds]}/{os.path.basename(man)}")'''))

ac.append(md("""## 5. Graph baseline — G-Reasoner per domain

**Everything below is optional and slow.** Sections 1-4 are the dense table; stop there if
that is all you need today.

G-Reasoner (`GraphReasoner`, stock `sft_training` config) trains from random init on each
domain's **v16sc graphs**, which the full bundles already carry. First run per domain also
builds the Qwen3 node index (~20-45 min); training is hours per domain at batch 2, and the
loop is domain-serial with signature skips, so a disconnect costs only the run in flight.

GFM-RAG is section 5d and is **gated**: it prints what it needs instead of crashing."""))
ac.append(md("### 5a. Engine\n*(reused verbatim from the fusion notebook)*"))
ac.append(reuse[4])
ac.append(md("### 5a-i. Re-pin the two packages the engine installs unbounded"))
ac.append(PIN_CELL)
ac.append(reuse[7])
ac.append(md("### 5a-ii. Fix the vendored PyG version check"))
ac.append(PYG_CELL)
ac.append(md("### 5a-iii. Restore `torchvision.io.VideoReader` for `datasets`"))
ac.append(SHIM_CELL)

ac.append(md("### 5b. Ship repo files + check every domain's graphs"))
ac.append(code(
    "import json, os, csv, collections, shutil\n"
    f"FILES = json.loads(r'''{json.dumps(GRAPH_FILES)}''')\n"
    "for p, c in FILES.items():\n"
    "    os.makedirs(os.path.dirname(p), exist_ok=True)\n"
    "    open(p, 'w').write(c)\n"
    "    print('wrote', p)\n"
    "\n"
    "csv.field_size_limit(10 ** 7)\n"
    "DATA_ROOT = f'{CARGO_ROOT}/kg-construction/data'\n"
    "ENTITY_OK = {}\n"
    "for ds in DSETS:\n"
    "    # v16sc graphs: required for the G-Reasoner arm. The loader reads\n"
    "    # {graph}/raw/documents.json, a copy of the corpus INSIDE the graph directory.\n"
    "    for split in ('train', 'test'):\n"
    "        g  = f'{ds}_{split}_v16sc'\n"
    "        s1 = f'{DATA_ROOT}/{g}/processed/stage1'\n"
    "        assert os.path.exists(f'{s1}/nodes.csv'), (\n"
    "            f'missing {s1}/nodes.csv -- the {ds} bundle is --slim, section 5 needs the '\n"
    "            f'FULL bundle')\n"
    "        src = f'{DATA_ROOT}/{ds}_{split}/raw/documents.json'\n"
    "        dst = f'{DATA_ROOT}/{g}/raw/documents.json'\n"
    "        if not os.path.exists(dst):\n"
    "            os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
    "            shutil.copy(src, dst)\n"
    "    # OpenIE entity graph: the GFM-RAG gate. Exists for TOMATO, for no SIR-4 domain.\n"
    "    s1 = f'{DATA_ROOT}/{ds}_test/processed/stage1'\n"
    "    if os.path.exists(f'{s1}/nodes.csv'):\n"
    "        _t = collections.Counter(r['type'] for r in csv.DictReader(open(f'{s1}/nodes.csv')))\n"
    "        ENTITY_OK[ds] = _t.get('entity', 0) > 0 and _t.get('document', 0) > 0\n"
    "    else:\n"
    "        ENTITY_OK[ds] = False\n"
    "    _e = 'present' if ENTITY_OK[ds] else 'absent -> GFM-RAG gated'\n"
    "    print(f'  {ds:14} v16sc graphs ok   entity graph: {_e}')"))

ac.append(md("### 5c. Train G-Reasoner on every domain\n"
             "Signature-gated exactly like the single-domain notebook: a finished run of "
             "this configuration is a `[skip]`, a directory produced by a different one is "
             "refused rather than adopted."))
ac.append(code('''EPOCHS, BATCH = 10, 2
import threading

def _sync_dir(src, dst):
    """Copy new/changed files src -> dst, tolerating a dead mount. Returns note."""
    try:
        for root, _, files in os.walk(src):
            rel = os.path.relpath(root, src)
            os.makedirs(os.path.join(dst, rel) if rel != "." else dst, exist_ok=True)
            for f in files:
                s = os.path.join(root, f)
                d = os.path.join(dst, rel, f) if rel != "." else os.path.join(dst, f)
                if (not os.path.exists(d) or os.path.getsize(d) != os.path.getsize(s)
                        or os.path.getmtime(s) > os.path.getmtime(d) + 1):
                    shutil.copy2(s, d)
        return "ok"
    except OSError as e:
        return f"skipped ({e})"

def run_baseline(ds, config, suffix, train, valid, epochs=EPOCHS, batch=BATCH):
    """Train LOCALLY, sync to Drive every 10 min. Drive is never on the training hot
    path: a FUSE mount flap killed a run at epoch 9 of 10 through the console-log tee,
    and the trainer's own checkpoint saves were one flap away from the same fate. Now a
    flap costs one sync pass and a runtime disconnect costs <=10 min of training.
    Skip/stale/resume decisions read the DRIVE copy, which is the durable one."""
    drive_dir = f"{OUTRT[ds]}/{ds}_{suffix}"
    run_dir   = f"/content/runs/{ds}_{suffix}"
    _ckpt_d, _pred_d = f"{drive_dir}/model_best.pth", f"{drive_dir}/predictions_{valid}.json"
    _cfgp = f"/content/gfm-rag/gfmrag/workflow/config/gfm_reasoner/{config}.yaml"
    _sig = hashlib.md5(json.dumps(
        {"cfg": hashlib.md5(open(_cfgp, "rb").read()).hexdigest(),
         "epochs": epochs, "batch": batch, "topk": TOPK,
         "train": train, "valid": [valid]}, sort_keys=True).encode()).hexdigest()[:12]
    _was = None
    if os.path.isfile(f"{drive_dir}/arm.json"):
        try: _was = json.load(open(f"{drive_dir}/arm.json")).get("sig")
        except Exception: _was = None
    if os.path.isfile(_ckpt_d) and os.path.isfile(_pred_d):
        if _was == _sig:
            print(f"[skip] {drive_dir}: finished run of this configuration")
            return drive_dir
        print(f"[stale] {drive_dir}: sig mismatch -- NOT overwriting and NOT reporting it")
        return None
    os.makedirs(run_dir, exist_ok=True); os.makedirs(drive_dir, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "gfmrag.workflow.sft_training",   # the kernel's python, where gfmrag is installed; a bare `python` resolved to /usr/bin/python3 once (8 Sep)
           "--config-path", "config/gfm_reasoner", "--config-name", config,
           "text_emb_model=qwen3_st", f"datasets.cfgs.root={DATA_ROOT}",
           "datasets.cfgs.force_reload=False",
           f"datasets.train_names=[{train}]", f"datasets.valid_names=[{valid}]",
           f"trainer.args.num_epoch={epochs}", f"trainer.args.train_batch_size={batch}",
           "+trainer.args.do_predict=true", f"+trainer.args.predict_top_k={TOPK}",
           f"hydra.run.dir={run_dir}"]
    # RESUME, not redo. The checkpoint carries epoch and optimizer state, and selection
    # is best-epoch anyway, so resuming from best loses at most the epochs since it.
    if os.path.isfile(_ckpt_d) and _was == _sig:
        print("[resume] restoring checkpoint from Drive and resuming from it")
        shutil.copy(_ckpt_d, f"{run_dir}/model_best.pth")
        cmd.append(f"trainer.args.resume_from_checkpoint={run_dir}/model_best.pth")
    elif os.path.isfile(_ckpt_d):
        print("[redo] Drive checkpoint is from a DIFFERENT configuration -- from scratch")
    env = dict(os.environ, WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", STRAT_EVAL="0",
               # the engine must be importable from /content in a fresh interpreter; the
               # editable install did not guarantee that on the Sept 2026 image
               PYTHONPATH="/content/gfm-rag" + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for k in ("OPERATOR_COMPONENTS", "OPERATOR_COMPONENTS_TEST", "SEMANTIC_COMPONENTS",
              "SEMANTIC_COMPONENTS_TEST", "SEMANTIC_CKPT", "SEMANTIC_POPNET",
              "FUSION_OBJECTIVE", "FUSION_ROUTER", "FUSION_GAMMAFIX", "HARDNEG_HUB",
              "HARDNEG_RAND", "HARDNEG_GRAPH", "PER_GOLD", "AUX_W", "SEM_POP_LAMBDA",
              "CCMP", "CCMP_W", "CCMP_LR", "MISS_W_AUX", "RESID_PRIOR", "CQIG_M"):
        env.pop(k, None)
    json.dump({"arm": suffix, "config": config, "epochs": epochs, "batch": batch,
               "baseline": True, "sig": _sig, "train": train, "valid": [valid],
               "predict_top_k": TOPK}, open(f"{run_dir}/arm.json", "w"), indent=1)
    print(f"[baseline] {ds}: {config} -> {run_dir}  (syncing to {drive_dir} every 10 min)")
    _stop = threading.Event()
    def _pump():
        while not _stop.wait(600):
            print(f"[sync] {_sync_dir(run_dir, drive_dir)}")
    _t = threading.Thread(target=_pump, daemon=True); _t.start()
    try:
        with open(f"{run_dir}/console.log", "w") as log:
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                print(line, end=""); log.write(line)
            rc = p.wait()
    finally:
        # Runs on crash too, so the last checkpoint reaches Drive either way.
        _stop.set()
        print(f"[sync-final] {_sync_dir(run_dir, drive_dir)}")
    assert rc == 0, f"{ds}/{suffix} failed, exit code {rc} (-9 = killed, out of memory)"
    return drive_dir

for ds in DSETS:
    rd = run_baseline(ds, "sft_training", f"greasoner_e{EPOCHS}_b{BATCH}",
                      f"{ds}_train_v16sc", f"{ds}_test_v16sc")
    if rd:
        p = f"{rd}/predictions_{ds}_test_v16sc.json"
        if os.path.exists(p):
            PRED[ds]["G-Reasoner"] = p'''))

ac.append(md("""### 5d. GFM-RAG — gated on an OpenIE entity graph

GFM-RAG v1 ranks `entity` nodes and maps them to documents, so it needs the OpenIE
construction (`entity` + `document` node types). **No SIR-4 domain has one**; building it
means LLM extraction (NER + OpenIE triples) over every document of the corpus, then the
GFM-RAG index build. Until then this section reports the gate per domain and the thesis
reports GFM-RAG on TOMATO, where the entity graph exists.

If an entity graph is ever built for a domain (`<ds>_{train,test}/processed/stage1` with
entity-typed nodes in the bundle), this cell picks it up on the next run with no edits."""))
ac.append(code('''for ds in DSETS:
    if not ENTITY_OK.get(ds):
        print(f"[gate] {ds}: no OpenIE entity graph -- GFM-RAG row not trainable on this "
              f"domain (see the section header); reported on TOMATO only")
        continue
    rd = run_baseline(ds, "sft_training_gfmrag", f"gfmrag_e{EPOCHS}_b{BATCH}",
                      f"{ds}_train", f"{ds}_test")
    if rd:
        p = f"{rd}/predictions_{ds}_test.json"
        if os.path.exists(p):
            PRED[ds]["GFM-RAG"] = p'''))

ac.append(md("## 6. Score every arm on every domain, same scorer, same flags\n"
             "CompleteSet@5 is the one SIR-4-specific column: 1 if a complete inspiration "
             "set from `sets.json` sits inside the top 5. It is requested ONLY when the "
             "domain's sets.json is present, because score_sir4 writes a hard 0.0 for the "
             "column otherwise, which reads as measured-and-zero."))
ac.append(code('''# `map` is in the list for MIR's table (R@3 / R@5 / nDCG@5 / mAP); harmless for SIR-4.
STD_COLS = ("mrr", "ndcg@5", "recall@3", "recall@5", "recall@10", "recall@25", "recall@100", "map")
GRID_KS  = (1, 3, 5, 10, 20, 25, 50, 100)
_BASE_COLS = ",".join(dict.fromkeys(list(STD_COLS)
                              + [f"recall@{k}" for k in GRID_KS]
                              + [f"hits@{k}" for k in GRID_KS]))
# Stamped at build time from the same table run_domain.py stages from.
SETS = {ds: f"{CARGO_ROOT}/{rel}" for ds, rel in SETS_JSON.items()}
SCORES = {ds: {} for ds in DSETS}
for ds in DSETS:
    QS = f"{CORPUS[ds]}/{SPLIT}.json"
    _has_sets = os.path.exists(SETS.get(ds, ""))
    COLS = _BASE_COLS + (",completeset@5" if _has_sets else "")
    if not _has_sets:
        print(f"{ds}: sets.json MISSING at {SETS.get(ds)} -- CompleteSet@5 skipped, "
              f"not zero-filled")
    for lab, p in list(PRED[ds].items()):
        if not p or not os.path.exists(p):
            print(f"skipping {ds}/{lab}: no predictions"); continue
        jo = f"{OUTRT[ds]}/scores_{lab.replace(' ', '_').replace('/', '-')}.json"
        _extra = f" --sets {SETS[ds]}" if _has_sets else ""
        rc = sh(f"python3 -u eval/score_sir4.py --pred {p} --queries {QS} --cols {COLS}"
                f"{_extra} --name '{ds} {lab}' --json-out {jo}", S4)
        if rc == 0 and os.path.exists(jo):
            SCORES[ds][lab] = json.load(open(jo))
    print(f"{ds}: scored {sorted(SCORES[ds])}")'''.replace("SETS_JSON", json.dumps(SETS_MAP))))

ac.append(md("""## 7. The tables

Per-domain standard metrics first (same columns and order as every fusion and ablation
notebook), then the cross-domain summary grid: one row per arm, one column pair per
domain, at the cutoffs in `CUTOFFS`. `drop` is `mean_k (1 - cross@k / same@k)`."""))
ac.append(code('''CUTOFFS = (1, 10, 50)
PAIR    = ("same", "cross")
METRIC  = "recall"
assert all(k in GRID_KS for k in CUTOFFS), (
    f"CUTOFFS {CUTOFFS} not a subset of GRID_KS {GRID_KS}; add the cutoff and re-run 6")
ORDER = [l for l, *_ in ARMS] + ["G-Reasoner", "GFM-RAG"]

for ds in DSETS:
    print(f"\\n================ {ds} ================")
    print("\\nSTANDARD METRICS (%)")
    for slc in ("all", "same", "cross"):
        print(f"\\n--- {slc} ---")
        _cols = list(STD_COLS) + ["completeset@5"]
        print(f"{'arm':22}{'n':>6}" + "".join(
            f"{c.replace('recall@', 'R@').replace('completeset@5', 'CS@5'):>9}" for c in _cols))
        for lab in ORDER:
            sc = (SCORES[ds].get(lab) or {}).get(slc)
            if not sc:
                continue
            print(f"{lab:22}{sc.get('n', 0):>6}"
                  + "".join(f"{100*sc[c]:>9.1f}" if sc.get(c) is not None else f"{'--':>9}"
                            for c in _cols))

L, R = PAIR
print(f"\\n\\n================ CROSS-DOMAIN SUMMARY ({METRIC}, %) ================")
for k in CUTOFFS:
    print(f"\\n--- {METRIC}@{k}: ({L} / {R}) per domain ---")
    print(f"{'arm':22}" + "".join(f"{d:>16}" for d in DOMS) + f"{'mean drop':>12}")
    for lab in ORDER:
        row, drops, have = f"{lab:22}", [], False
        for ds in DSETS:
            sc = SCORES[ds].get(lab) or {}
            a = (sc.get(L) or {}).get(f"{METRIC}@{k}")
            c = (sc.get(R) or {}).get(f"{METRIC}@{k}")
            if a is None or c is None:
                row += f"{'--':>16}"
            else:
                row += f"{100*a:>8.1f}/{100*c:<7.1f}"; have = True
                if a: drops.append(1.0 - c / a)
        row += (f"{-100*sum(drops)/len(drops):>10.0f}%" if drops else f"{'--':>12}")
        if have:
            print(row)

for ds in DSETS:
    json.dump(SCORES[ds], open(f"{OUTRT[ds]}/baselines_all.json", "w"), indent=1)
json.dump({ds: SCORES[ds] for ds in DSETS},
          open(f"{DRIVE}/outputs/baselines/''' + _SUMMARY + '''", "w"), indent=1)
print(f"\\nwrote per-domain baselines_all.json + {DRIVE}/outputs/baselines/''' + _SUMMARY + '''")'''))

write_nb(ac, OUT)
