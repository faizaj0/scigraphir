"""Data exploration: corpus/query stats, distributions, lexical overlap, examples.

Recreates the key TOMATO-Star exploration on the FULL test split and saves:
  outputs/results/explore_stats.json
  outputs/results/examples.txt
  figures/explore_*.png
"""
from __future__ import annotations
import json, re
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import FIGS, RESULTS, BIOMEDICAL, DISTANT
from .data import build

_WORD_RE = re.compile(r"[a-zA-Z]+")
def _wset(t): return {w for w in (m.group(0).lower() for m in _WORD_RE.finditer(t)) if len(w) > 2}
def jaccard(a, b):
    A, B = _wset(a), _wset(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def run():
    queries, corpus, golds = build(use_cache=True)
    stats: dict = {}

    # ---- counts ----
    strat = Counter(q["stratum"] for q in queries)
    n_papers = len({q["source_id"] for q in queries})
    stats["counts"] = {
        "test_papers": n_papers, "queries": len(queries), "corpus_docs": len(corpus),
        "same": strat.get("same", 0), "cross": strat.get("cross", 0),
        "unstratified": strat.get(None, 0),
    }
    # inspirations per paper
    per_paper = Counter(q["source_id"] for q in queries)
    ipp = list(per_paper.values())
    stats["insp_per_paper"] = {"mean": float(np.mean(ipp)), "max": int(max(ipp)),
                               "hist": dict(Counter(ipp))}
    # gold-domain distribution (cross-domain breakdown)
    gdom = Counter(q["gold_domain"] for q in queries)
    stats["gold_domain"] = {str(k): v for k, v in gdom.most_common()}

    # ---- lengths (words) ----
    qlen = np.array([len(q["b_text"].split()) for q in queries])
    dlen = np.array([len(t.split()) for t in corpus.values()])
    stats["query_len_words"] = {"mean": float(qlen.mean()), "median": float(np.median(qlen)), "p95": float(np.percentile(qlen, 95))}
    stats["doc_len_words"]   = {"mean": float(dlen.mean()), "median": float(np.median(dlen)), "p95": float(np.percentile(dlen, 95))}

    # ---- lexical jaccard query <-> gold, same vs cross ----
    jac = {"same": [], "cross": []}
    for q in tqdm(queries, desc="lexical jaccard"):
        s = q["stratum"]
        if s in jac:
            jac[s].append(jaccard(q["b_text"], corpus.get(q["gold_key"], "")))
    stats["jaccard_query_gold"] = {s: {"mean": float(np.mean(v)), "median": float(np.median(v))}
                                   for s, v in jac.items() if v}

    # ================= PLOTS =================
    # 1. inspirations per paper
    plt.figure(figsize=(5, 3.2))
    xs = sorted(stats["insp_per_paper"]["hist"])
    plt.bar(xs, [stats["insp_per_paper"]["hist"][x] for x in xs], color="#2166ac")
    plt.xlabel("inspirations per paper"); plt.ylabel("# papers")
    plt.title(f"Inspirations / paper (mean {stats['insp_per_paper']['mean']:.2f})")
    plt.tight_layout(); plt.savefig(FIGS / "explore_insp_per_paper.png", dpi=150); plt.close()

    # 2. query & doc length distributions
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].hist(np.clip(qlen, 0, 400), bins=40, color="#2166ac"); ax[0].set_title("query length (words)"); ax[0].set_xlabel("words")
    ax[1].hist(np.clip(dlen, 0, 400), bins=40, color="#b2182b"); ax[1].set_title("corpus doc length (words)"); ax[1].set_xlabel("words")
    plt.tight_layout(); plt.savefig(FIGS / "explore_lengths.png", dpi=150); plt.close()

    # 3. lexical jaccard same vs cross  (THE key cross-domain signal)
    plt.figure(figsize=(5, 3.4))
    bins = np.linspace(0, 0.4, 40)
    for s, c in [("same", "#2166ac"), ("cross", "#b2182b")]:
        if jac[s]:
            plt.hist(jac[s], bins=bins, alpha=0.55, density=True, color=c,
                     label=f"{s} (med {np.median(jac[s]):.3f})")
    plt.xlabel("Jaccard(query, gold)"); plt.ylabel("density")
    plt.title("Lexical overlap query↔gold"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGS / "explore_lexical_jaccard.png", dpi=150); plt.close()

    # 4. gold-domain distribution
    plt.figure(figsize=(5.5, 3.2))
    items = gdom.most_common()
    labs = [str(k) for k, _ in items]; vals = [v for _, v in items]
    cols = ["#2166ac" if l in BIOMEDICAL else ("#b2182b" if l in DISTANT else "#999999") for l in labs]
    plt.bar(range(len(labs)), vals, color=cols)
    plt.xticks(range(len(labs)), labs, rotation=30, ha="right", fontsize=8)
    plt.ylabel("# gold inspirations"); plt.title("Gold-inspiration domains (blue=bio, red=distant)")
    plt.tight_layout(); plt.savefig(FIGS / "explore_gold_domains.png", dpi=150); plt.close()

    # ---- examples ----
    lines = ["# SciGraphIR / TOMATO-Star — example queries (full test)\n"]
    for s in ("same", "cross"):
        ex = [q for q in queries if q["stratum"] == s][:3]
        lines.append(f"\n===== {s.upper()}-DOMAIN EXAMPLES =====")
        for q in ex:
            g = corpus.get(q["gold_key"], "")
            lines.append(f"\n[{q['query_id']}]  src={q['src_domain']} -> gold={q['gold_domain']}")
            lines.append(f"QUERY: {q['b_text'][:380]}")
            lines.append(f"GOLD : {g[:380]}")
            lines.append(f"jaccard(query,gold) = {jaccard(q['b_text'], g):.3f}")
    (RESULTS / "examples.txt").write_text("\n".join(lines))

    (RESULTS / "explore_stats.json").write_text(json.dumps(stats, indent=2))
    print("\n===== STATS =====")
    print(json.dumps(stats, indent=2))
    print(f"\nfigures -> {FIGS}/explore_*.png")
    print(f"examples -> {RESULTS/'examples.txt'}")
    return stats


if __name__ == "__main__":
    run()
