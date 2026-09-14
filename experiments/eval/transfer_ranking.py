# ---- Which trained graph transfers best, against the untrained floor ---------
# The 4x4 matrix says how every pair scores. It does not say which SOURCE is the
# best thing to train on, nor whether any of them beat doing no training at all.
# Both questions are answered here, from files already on Drive.
#
# The untrained arms (bm25/bge/qwen3/operator) do not transfer -- they score the
# same on a target regardless of source -- so they are not rows in the matrix.
# They are the FLOOR every off-diagonal cell has to clear to have meant anything.
import os, json

MET  = [("mrr", "MRR"), ("ndcg@5", "nDCG@5"), ("recall@3", "R@3"),
        ("recall@5", "R@5"), ("completeset@5", "CGS@5")]
HEAD = "ndcg@5"                       # the metric the ranking is sorted on
MTX  = globals().get("OUT_M", f"{DRIVE}/outputs/_transfer_matrix")
ARMC = f"{DRIVE}/outputs/_arm_comparison"
BASE = [("bm25", "BM25"), ("bge", "BGE"), ("qwen3", "Qwen3"), ("operator", "Operator")]
SLICES = ("all", "cross")             # `all` is the headline, `cross` is the claim

rl = []
def out(s=""):
    print(s); rl.append(s)


def load(path, slc):
    """Score file -> one slice, or None. Exact paths only, and non-dicts refused:
    _arm_comparison holds prediction files (lists) beside the score files, and a
    glob over it is how an earlier version of this crashed."""
    if not os.path.exists(path):
        return None
    try:
        j = json.load(open(path))
    except Exception:
        return None
    if not isinstance(j, dict):
        return None
    m = j.get(slc)
    return m if isinstance(m, dict) and "n" in m else None


def cellv(tr, te, slc, k):
    m = load(f"{MTX}/{tr}__on__{te}_scores.json", slc)
    return None if not m else m.get(k)


def basev(arm, te, slc, k):
    m = load(f"{ARMC}/{arm}_sir4_{te}.json", slc)
    return None if not m else m.get(k)


# `mrr` names a different metric in the two scorer versions, so a mixed set would
# rank sources on two different quantities without saying so.
kinds = set()
for q in ([f"{MTX}/{a}__on__{b}_scores.json" for a in DOMS for b in DOMS]
          + [f"{ARMC}/{k}_sir4_{d}.json" for k, _ in BASE for d in DOMS]):
    if not os.path.exists(q):
        continue
    try:
        v = json.load(open(q))
    except Exception:
        continue
    if isinstance(v, dict) and isinstance(v.get("all"), dict):
        kinds.add("current" if "mgrr" in v["all"] else "bundled")
if len(kinds) > 1:
    out(f"!! MIXED SCORERS {sorted(kinds)}: `mrr` means a different thing in each. "
        f"Re-score one set before reading this.")
    out()

for slc in SLICES:
    if not any(cellv(a, b, slc, HEAD) is not None for a in DOMS for b in DOMS):
        continue

    # The floor, per target: the best any untrained retriever manages there.
    floor = {}
    for te in DOMS:
        best = None
        for arm, lab in BASE:
            v = basev(arm, te, slc, HEAD)
            if v is not None and (best is None or v > best[1]):
                best = (lab, v)
        floor[te] = best

    out(f"### {slc}: which source transfers best   (sorted by mean off-diagonal {HEAD})")
    out("| source | in-domain | mean off-diag | retention | vs untrained floor | beats floor |")
    out("|---|--:|--:|--:|--:|--:|")
    rank = []
    for tr in DOMS:
        offs = [(te, cellv(tr, te, slc, HEAD)) for te in DOMS if te != tr]
        offs = [(te, v) for te, v in offs if v is not None]
        if not offs:
            continue
        ind  = cellv(tr, tr, slc, HEAD)
        mean = sum(v for _, v in offs) / len(offs)
        # Delta against the floor on the SAME target, then averaged. Averaging the
        # floors first would let an easy target mask a hard one.
        ds   = [v - floor[te][1] for te, v in offs if floor.get(te)]
        wins = sum(1 for d in ds if d > 0)
        rank.append((mean, tr, ind, mean, (sum(ds)/len(ds)) if ds else None,
                     wins, len(ds)))
    for _, tr, ind, mean, dmean, wins, n in sorted(rank, reverse=True):
        out(f"| **{tr}** | {'-' if ind is None else f'{ind:.4f}'} | {mean:.4f} | "
            + ("-" if not ind else f"{mean/ind*100:.0f}%") + " | "
            + ("-" if dmean is None else f"{dmean:+.4f}") + f" | {wins}/{n} |")
    out()

    # Per target: every source and every untrained arm on one ladder, so it is
    # visible whether the winner is a checkpoint or a retriever that never trained.
    out(f"### {slc}: per target, sources and untrained arms on one ladder ({HEAD})")
    out("| target | rank | arm | " + " | ".join(l for _, l in MET) + " |")
    out("|---|--:|---|" + "--:|" * len(MET))
    for te in DOMS:
        ladder = []
        for arm, lab in BASE:
            m = load(f"{ARMC}/{arm}_sir4_{te}.json", slc)
            if m: ladder.append((m.get(HEAD, -1), f"{lab} (untrained)", m))
        for tr in DOMS:
            m = load(f"{MTX}/{tr}__on__{te}_scores.json", slc)
            if m:
                tag = f"{tr} (in-domain)" if tr == te else f"{tr} -> {te}"
                ladder.append((m.get(HEAD, -1), tag, m))
        for i, (_, lab, m) in enumerate(sorted(ladder, key=lambda x: -x[0]), 1):
            cells = [f"{m[k]:.4f}" if k in m else "-" for k, _ in MET]
            out(f"| {te if i == 1 else ''} | {i} | {lab} | " + " | ".join(cells) + " |")
    out()

out(f"`retention` = mean off-diagonal / in-domain. 100% means a checkpoint from "
    f"another domain does as well as the native one.")
out()
out(f"`vs untrained floor` is the one that decides the claim: it compares each "
    f"source's zero-shot score against the BEST untrained retriever on the same "
    f"target. Negative means training on that domain and transferring lost to a "
    f"method that trained on nothing.")

if not any(os.path.exists(f"{ARMC}/{k}_sir4_{d}.json") for k, _ in BASE for d in DOMS):
    out()
    out("No untrained baselines found: run the comparison cell above first, which "
        "writes them to _arm_comparison/. Without them the floor columns are blank.")

os.makedirs(MTX, exist_ok=True)
open(f"{MTX}/transfer_ranking.md", "w").write("\n".join(rl))
print(f"\nwrote {MTX}/transfer_ranking.md")
