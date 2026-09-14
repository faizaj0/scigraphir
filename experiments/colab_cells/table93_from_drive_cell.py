# ===== PASTE INTO A FRESH COLAB RUNTIME. Builds Table 9.3 (both directions) from the score
# files already on Drive. Reads only; trains nothing; no engine, no bundle, no GPU needed. =====
import os, json, glob
from google.colab import drive
drive.mount('/content/drive')
DRIVE = "/content/drive/MyDrive/cargo-gfmrag"
ZS    = f"{DRIVE}/outputs/sir4_zeroshot/scores"        # SciGraphIR zero-shot scores (both directions)
BL    = f"{DRIVE}/outputs/baselines"                    # sir4_<field>/scores_<arm>.json, training-free
ARMS  = [("BM25", "BM25"), ("BGE-large", "BGE-large"), ("Qwen3-Embedding", "Qwen3-Embedding"),
         ("SPECTER2-base", "SPECTER2"), ("SciNCL", "SciNCL"), ("ReasonIR-8B", "ReasonIR-8B")]
DIRS  = [("Physics + Biology", ["cs", "matsci"]), ("CS + MatSci", ["physics", "biology"])]
SHORT = {"cs": "CS", "matsci": "MS", "physics": "Phys.", "biology": "Bio"}

def load(p): return json.load(open(p)) if os.path.exists(p) else None
def zs_scores(field):
    c = sorted(glob.glob(f"{ZS}/{field}_scigraphir_*_scores.json"))
    return (load(c[-1]), os.path.basename(c[-1])) if c else (None, None)
def gap(rows):                       # Delta_macro = (mean cross - mean same) / mean same, nDCG@5
    same = [r["same"]["ndcg@5"] for r in rows]; cross = [r["cross"]["ndcg@5"] for r in rows]
    ms, mc = sum(same) / len(same), sum(cross) / len(cross)
    return 100 * (mc - ms) / ms
pct = lambda x: f"{100 * x:.2f}"

lines = []
def out(s=""): print(s); lines.append(s)
tex = {}                                              # (direction, arm label) -> latex cell string
for train_label, evals in DIRS:
    out(f"\n## Trained on {train_label}, zero-shot on {' / '.join(SHORT[f] for f in evals)}  (nDCG@5, %)\n")
    hdr = ["Method"] + [f"{SHORT[f]} same" for f in evals] + [f"{SHORT[f]} cross" for f in evals] + ["gap", "R@3 cross", "R@5 cross"]
    out("| " + " | ".join(hdr) + " |"); out("|" + "---|" * len(hdr))
    table_rows = []
    for arm, label in ARMS:
        rows = [load(f"{BL}/sir4_{f}/scores_{arm}.json") for f in evals]
        table_rows.append((label, rows))
    zs = [zs_scores(f) for f in evals]
    if all(z[0] for z in zs):
        table_rows.append(("SciGraphIR", [z[0] for z in zs]))
        src = ", ".join(z[1] for z in zs)
    else:
        src = "MISSING: " + ", ".join(f for f, z in zip(evals, zs) if not z[0])
    for label, rows in table_rows:
        if any(r is None for r in rows):
            out(f"| {label} | " + " | ".join(["--"] * (len(hdr) - 1)) + " |"); continue
        g = gap(rows)
        cells = ([pct(r["same"]["ndcg@5"]) for r in rows] + [pct(r["cross"]["ndcg@5"]) for r in rows]
                 + [f"{g:+.2f}%", "/".join(pct(r["cross"]["recall@3"]) for r in rows),
                    "/".join(pct(r["cross"]["recall@5"]) for r in rows)])
        out(f"| {label} | " + " | ".join(cells) + " |")
        tex[(train_label, label)] = (" & ".join(pct(r["same"]["ndcg@5"]) for r in rows) + " & "
                                     + " & ".join(pct(r["cross"]["ndcg@5"]) for r in rows)
                                     + f" & \\dg{{${g:+.2f}\\%$}}")
    out(f"\nSciGraphIR source files: {src}")

out("\n## LaTeX cells for table93_zeroshot_sidebyside.tex (same, same, cross, cross, gap), per half\n")
for label in [l for _, l in ARMS] + ["SciGraphIR"]:
    left = tex.get(("Physics + Biology", label), "-- & -- & -- & -- & --")
    right = tex.get(("CS + MatSci", label), "-- & -- & -- & -- & --")
    out(f"{label}\n& {left}\n& {right} \\\\")
os.makedirs(f"{DRIVE}/outputs/sir4_zeroshot", exist_ok=True)
open(f"{DRIVE}/outputs/sir4_zeroshot/table93_both_directions.md", "w").write("\n".join(lines))
print("\nwrote", f"{DRIVE}/outputs/sir4_zeroshot/table93_both_directions.md")
