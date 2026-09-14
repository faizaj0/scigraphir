# Walk prior vs trained graph channel for EVERY fusion run on Drive, in one cell (CPU runtime is enough).
# Every fusion run's console.log prints, at the final evaluation, the three channels' nDCG@5, the
# parameter-free walk prior (personalised PageRank from the query's seeds, no learned parameters),
# gamma and the buried-gold bucket. This finds every such log under outputs/, parses it, and prints
# one table per dataset: walk prior | trained graph | delta | scorer | fused | gamma | best epoch.
import os, re, glob, json, time
from google.colab import drive
drive.mount('/content/drive')
ROOT = "/content/drive/MyDrive/cargo-gfmrag/outputs"
OUT = f"{ROOT}/scan/walk_prior_all.md"

def parse(log_path):
    log = open(log_path, errors="ignore").read()
    if "[prior] parameter-free walk" not in log or "Running final evaluation" not in log:
        return None
    tail = log[log.rfind("Running final evaluation"):]
    d = {"run": os.path.relpath(os.path.dirname(log_path), ROOT), "mtime": os.path.getmtime(log_path)}
    m = re.search(r"valid_names:\s*\n\s*-\s*(\S+)", log); d["valid"] = m[1] if m else "?"
    m = re.search(r"train_names:\s*\n\s*-\s*(\S+)", log); d["train"] = m[1] if m else "?"
    d["dataset"] = re.sub(r"_(test|train)(_v16sc)?$", "", d["valid"])
    d["graph"] = "SciAffordGraph" if "_v16sc" in d["valid"] else "OpenIE"
    m = re.search(r"\[ccmp\] responsibility head[^\n]*gate=(\w+)", log)
    d["ccmp"] = ("on" if m[1] == "True" else "gate off") if m else "no"
    m = re.search(r"\[route\] mode=(\w+)", log); d["route"] = m[1] if m else ""
    m = re.search(r"\[diag\] nDCG@5\s+\S+ ([\d.]+) \| \S+ ([\d.]+) \| \S+ ([\d.]+)\s+gamma mean ([\d.]+)", tail)
    if m: d.update(sem=float(m[1]), graph_ndcg=float(m[2]), fused=float(m[3]), gamma=float(m[4]))
    m = re.search(r"\[prior\] parameter-free walk nDCG@5 ([\d.]+)\s+trained graph ([\d.]+)", tail)
    if m: d.update(walk=float(m[1]), graph_ndcg=float(m[2]))
    m = re.search(r"1k\+ ([\d.]+) \(n=(\d+)\)", tail)
    if m: d.update(buried=float(m[1]), buried_n=int(m[2]))
    best = re.findall(r"New best model! document_ndcg@5: ([\d.]+) at epoch (\d+)", log)
    if best: d.update(best_epoch=int(best[-1][1]), best_fused=float(best[-1][0]))
    return d

t0 = time.time()
logs = glob.glob(f"{ROOT}/**/console.log", recursive=True)
print(f"{len(logs)} console.log files under outputs/ ({time.time()-t0:.0f}s)")
rows = [r for r in (parse(p) for p in logs) if r and "walk" in r]
print(f"{len(rows)} runs with a walk-prior line\n")

def pc(x): return "--" if x is None else f"{100*x:.2f}"
lines = ["# Walk prior vs trained graph channel, every fusion run on Drive (final evaluation, whole test set, nDCG@5 %)", ""]
def out(s=""): print(s); lines.append(s)
order = ["sir4_cs", "sir4_biology", "sir4_physics", "sir4_matsci", "tomato", "mir"]
for ds in sorted({r["dataset"] for r in rows}, key=lambda x: (order.index(x) if x in order else 99, x)):
    sub = sorted([r for r in rows if r["dataset"] == ds], key=lambda r: (r["graph"], r["ccmp"], -r["mtime"]))
    out(f"## {ds}  (test set {sub[0]['valid']})\n")
    out("| run | trained on | graph | CCMP | routing | walk prior | trained graph | graph - walk | scorer | fused | gamma | buried R@10 | best ep |")
    out("|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in sub:
        gw = r["graph_ndcg"] - r["walk"] if "graph_ndcg" in r else None
        out(f"| {r['run']} | {r['train']} | {r['graph']} | {r['ccmp']} | {r['route']} | {pc(r.get('walk'))} | {pc(r.get('graph_ndcg'))} | "
            f"{'--' if gw is None else f'{100*gw:+.2f}'} | {pc(r.get('sem'))} | {pc(r.get('best_fused', r.get('fused')))} | "
            f"{r.get('gamma', float('nan')):.3f} | {pc(r.get('buried'))}{' (n=%d)' % r['buried_n'] if 'buried_n' in r else ''} | {r.get('best_epoch', '--')} |")
    out()
out("## Summary: latest run per dataset x graph x CCMP\n")
out("| dataset | graph | CCMP | walk prior | trained graph | graph - walk | scorer | fused | gamma | run |")
out("|---|---|---|--:|--:|--:|--:|--:|--:|---|")
seen = {}
for r in sorted(rows, key=lambda r: -r["mtime"]):
    if r["route"]: continue
    k = (r["dataset"], r["graph"], r["ccmp"])
    if k in seen: continue
    seen[k] = r
for k in sorted(seen, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[0], k[1], k[2])):
    r = seen[k]; gw = r["graph_ndcg"] - r["walk"]
    out(f"| {k[0]} | {k[1]} | {k[2]} | {pc(r['walk'])} | {pc(r['graph_ndcg'])} | {100*gw:+.2f} | {pc(r.get('sem'))} | "
        f"{pc(r.get('best_fused', r.get('fused')))} | {r.get('gamma', float('nan')):.3f} | {r['run']} |")
out()
out("walk prior = personalised PageRank from the query's seed nodes over the symmetrised, degree-normalised graph, documents "
    "ranked by the mass they receive; no learned parameters. trained graph = the reasoner's graph channel alone at the best "
    "epoch. scorer = multi-view scorer alone. fused = the model's output ranking. buried = the graph's recall@10 on golds the "
    "scorer ranked past 1,000. All on the run's whole test set (same and cross together).")
os.makedirs(os.path.dirname(OUT), exist_ok=True); open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
