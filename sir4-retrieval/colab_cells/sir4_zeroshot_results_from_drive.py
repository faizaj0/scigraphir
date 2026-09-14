# Standalone: read every score file the zero-shot notebook left on Drive and print the tables.
# Needs no GPU, no bundle, no engine. Works for either direction; set TRAIN_DOMS / EVAL_DOMS.
import os, json, glob
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE    = "/content/drive/MyDrive/cargo-gfmrag"
OUT_ROOT = f"{DRIVE}/outputs/sir4_zeroshot"
SCORES   = f"{OUT_ROOT}/scores"
TRAIN_DOMS, EVAL_DOMS = ["cs", "matsci"], ["physics", "biology"]
TITLE = {"cs": "Computer Science", "matsci": "Materials Science", "physics": "Physics", "biology": "Biology"}
ZS_KEYS = ["scigraphir_" + "".join(d[0] for d in TRAIN_DOMS), "scigraphir_pb"]   # new key, then the old hard-coded one
ORDER = [("bm25", "BM25"), ("bge", "BGE-large"), ("qwen3", "Qwen3-Embedding"), ("specter2", "SPECTER2-base"),
         ("scincl", "SciNCL"), ("reasonir", "ReasonIR-8B"), (ZS_KEYS, "SciGraphIR (" + " + ".join(TITLE[d] for d in TRAIN_DOMS) + ")")]
COLS = [("recall@3", "R@3"), ("recall@5", "R@5"), ("ndcg@5", "nDCG@5"), ("mrr", "MRR"), ("recall@10", "R@10")]

# 1. which run produced the zero-shot rows, and how far it got
for run in sorted(glob.glob(f"{OUT_ROOT}/scigraphir_{'+'.join(TRAIN_DOMS)}_*")):
    log = f"{run}/console.log"
    done = [l for l in open(log, errors="ignore")] if os.path.exists(log) else []
    best = [l.strip() for l in done if "New best model" in l]
    ep   = [l.strip() for l in done if "completed - Average loss" in l]
    print(f"[run] {os.path.relpath(run, DRIVE)}")
    print(f"      model_best.pth: {os.path.exists(f'{run}/model_best.pth')} | epochs completed: {len(ep)} | last best: {best[-1][-60:] if best else 'none'}")
    for p in sorted(glob.glob(f"{run}/predictions_*.json")): print("      pred:", os.path.basename(p))

# 2. every score file present for the held-out fields
print("\n[scores on Drive]")
for d in EVAL_DOMS:
    have = sorted(os.path.basename(p) for p in glob.glob(f"{SCORES}/{d}_*_scores.json"))
    print(f"  {d}: {have}")

# 3. the tables, in percent
def load(d, key):
    keys = key if isinstance(key, list) else [key]
    for k in keys:
        p = f"{SCORES}/{d}_{k}_scores.json"
        if os.path.exists(p): return json.load(open(p)), k
    return None, None
lines = []
def out(s=""): print(s); lines.append(s)
for slc in ("cross", "same", "all"):
    out(f"### {slc} queries" + ("   <- Table 9.3 right half" if slc == "cross" else ""))
    out("| Method | " + " | ".join(l for _, l in COLS) + " | n |"); out("|---|" + "--:|" * (len(COLS) + 1))
    for d in EVAL_DOMS:
        out(f"| **{TITLE[d]}** |" + " |" * (len(COLS) + 1))
        for key, lab in ORDER:
            s, used = load(d, key)
            if s is None or not s.get(slc):
                out(f"| {lab} | " + " | ".join("missing" for _ in COLS) + " | |"); continue
            r = s[slc]
            out(f"| {lab} | " + " | ".join(f"{100*r[k]:.2f}" for k, _ in COLS) + f" | {r['n']} |")
    out()
# macro gap per method: (mean cross - mean same) / mean same over the two held-out fields, nDCG@5
out("### macro gap (nDCG@5)")
out("| Method | mean same | mean cross | gap |"); out("|---|--:|--:|--:|")
for key, lab in ORDER:
    ss = [load(d, key)[0] for d in EVAL_DOMS]
    if any(s is None for s in ss): out(f"| {lab} | missing | | |"); continue
    same = sum(s["same"]["ndcg@5"] for s in ss) / len(ss); cross = sum(s["cross"]["ndcg@5"] for s in ss) / len(ss)
    out(f"| {lab} | {100*same:.2f} | {100*cross:.2f} | {100*(cross-same)/same:+.2f}% |")
md = f"{OUT_ROOT}/table93_" + "+".join(TRAIN_DOMS) + "_to_" + "+".join(EVAL_DOMS) + ".md"
open(md, "w").write("\n".join(lines)); print("\nwrote", md)
