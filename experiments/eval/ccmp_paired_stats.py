#!/usr/bin/env python3
"""
ccmp_paired_stats.py -- the paired CCMP statistics for EVERY dataset that has hops files: for each gold document, its
rank with the CCMP gate on vs off (same weights) in the graph channel and in the fused score, counted as better / worse /
same and summarised as the mean change in log10 rank with a bootstrap CI. Split by the query stratum stored in the hops
files (cross / same for SIR-4; whatever the dataset's labels are for TOMATO and MIR) plus "all".

Two comparisons per dataset, whichever files exist:
  gate      hops_<arm>.json vs hops_<arm>_off.json          same weights, gate on vs off (inference-time ablation)
  trained   hops_<arm>_ccmp.json vs hops_<arm>_nocc.json    CCMP-trained vs the no-CCMP control (training-time ablation)

    python3 eval/ccmp_paired_stats.py --root /content/drive/MyDrive/.../outputs/scan          # every dataset dir under root
    python3 eval/ccmp_paired_stats.py --dir sir4_cs=results/qualitative/drive_scan_sir4_cs     # one dir, named
Writes <out>.md and <out>.json (default: ccmp_paired_stats in the first dir).
"""
import argparse, glob, json, math, os, random


def load(path):
    out = {}
    for r in json.load(open(path)):
        for t in r["targets"]:
            out[(r["id"], t["doc"])] = (r.get("stratum") or "unlabelled", t.get("rank") or {})
    return out


def boot_mean(v, B=2000, seed=0):
    rnd = random.Random(seed); n = len(v); ms = []
    for _ in range(B): ms.append(sum(rnd.choice(v) for _ in range(n)) / n)
    ms.sort(); return ms[int(0.025 * B)], ms[int(0.975 * B)]


def compare(base, new, channel, keys):
    """base = gate off / control, new = with CCMP; lower rank is better"""
    b = w = s = 0; d = []
    for k in keys:
        rb, rn = base[k][1].get(channel), new[k][1].get(channel)
        if not rb or not rn: continue
        b += rn < rb; w += rn > rb; s += rn == rb; d.append(math.log10(rn) - math.log10(rb))
    if not d: return None
    d.sort(); lo, hi = boot_mean(d)
    return {"n": len(d), "better": b, "worse": w, "same": s, "median_dlog10": d[len(d) // 2], "mean_dlog10": sum(d) / len(d), "mean_ci95": [lo, hi],
            "top10_base": sum(base[k][1].get(channel, 10 ** 9) <= 10 for k in keys), "top10_new": sum(new[k][1].get(channel, 10 ** 9) <= 10 for k in keys)}


def pairs_in(d, prefix):
    out = []
    for f in sorted(glob.glob(f"{d}/{prefix}*_off.json")):
        arm = os.path.basename(f)[len(prefix):-len("_off.json")]
        if os.path.exists(f"{d}/{prefix}{arm}.json"): out.append(("gate", arm, f"{d}/{prefix}{arm}.json", f))
    for f in sorted(glob.glob(f"{d}/{prefix}*_ccmp.json")):
        stem = os.path.basename(f)[len(prefix):-len("_ccmp.json")]
        for ctrl in ("nocc", "control"):
            g = f"{d}/{prefix}{stem}_{ctrl}.json"
            if os.path.exists(g): out.append(("trained", f"{stem}_ccmp vs {stem}_{ctrl}", f, g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="directory whose subdirectories are datasets")
    ap.add_argument("--dir", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--prefix", default="hops_"); ap.add_argument("--out", default=None)
    ap.add_argument("--sir4", "--quartet", dest='sir4', action="append", default=[], help="name=eval.json: per-GOLD stratum from SIR-4; otherwise the query stratum in the hops file is used")
    a = ap.parse_args()
    dirs = [(os.path.basename(p.rstrip("/")), p) for p in sorted(glob.glob(f"{a.root}/*/")) ] if a.root else []
    dirs += [tuple(x.split("=", 1)) for x in a.dir]
    assert dirs, "give --root or --dir"
    res = {}; md = []
    qz = {}
    for x in a.sir4:
        nm, path = x.split("=", 1); m = {}
        for row in json.load(open(path)):
            for doc, meta in (row.get("quartet", {}).get("per_document") or {}).items(): m[(row["id"], doc)] = meta.get("stratum")
        qz[nm] = m
    for name, d in dirs:
        pr = pairs_in(d, a.prefix)
        if not pr: print(f"{name}: no hops pairs under {d}"); continue
        for kind, arm, fnew, fbase in pr:
            new, base = load(fnew), load(fbase); keys = [k for k in new if k in base]
            if name in qz:   # per-gold stratum (a query can have cross- and same-field golds)
                new = {k: (qz[name].get(k) or "unlabelled", v[1]) for k, v in new.items()}
            unit = "stratum of the gold (SIR-4)" if name in qz else "stratum of the query (hops file)"
            strata = sorted({new[k][0] for k in keys})
            groups = [("all", keys)] + [(s, [k for k in keys if new[k][0] == s]) for s in strata if s != "unlabelled"]
            tag = f"{name} · {arm} · " + ("gate on vs off, same weights" if kind == "gate" else "CCMP-trained vs control")
            md.append(f"\n### {tag}\n\n{unit}\n\n| channel | stratum | n | better | worse | same | mean Δlog10 rank [95% CI] | in top-10: base → CCMP |\n|---|---|---|---|---|---|---|---|")
            for ch in ("graph", "fused"):
                for s, ks in groups:
                    r = compare(base, new, ch, ks)
                    if not r: continue
                    res.setdefault(name, {}).setdefault(f"{kind}:{arm}", {})[f"{ch}:{s}"] = r
                    md.append(f"| {ch} | {s} | {r['n']} | {r['better']} | {r['worse']} | {r['same']} | {r['mean_dlog10']:+.3f} [{r['mean_ci95'][0]:+.3f}, {r['mean_ci95'][1]:+.3f}] | {r['top10_base']} → {r['top10_new']} |")
    text = "# CCMP paired statistics per dataset\n\nbetter = the gold's rank is lower (better) with CCMP; Δlog10 < 0 = CCMP helps. Per gold document, paired.\n" + "\n".join(md) + "\n"
    out = a.out or os.path.join(dirs[0][1], "ccmp_paired_stats")
    open(out + ".md", "w").write(text); json.dump(res, open(out + ".json", "w"), indent=1)
    print(text); print("wrote", out + ".md", out + ".json")


if __name__ == "__main__":
    main()
