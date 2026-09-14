#!/usr/bin/env python3
# usage: python3 eval/mine_showcase_cs.py <dir with hops_*.json + showcase_sir4_cs_candidates.json>   (no arg = local results/qualitative/cs_hops_*)
import json, os, csv, re, sys
D=sys.argv[1] if len(sys.argv)>1 else None
from collections import defaultdict
R=os.environ.get("CARGO_ROOT") or os.path.expanduser("~/Desktop/CARGO")
Q=f"{R}/sir4-retrieval/results/qualitative"
BIG=10**6
MECH=("function","limitation","method","finding")
def load(p):
    out={}
    for r in json.load(open(p)):
        seeds=set(r["seeds"])
        for t in r["targets"]:
            ps=[p_ for p_ in t.get("paths",[]) if p_.get("hops") and p_["hops"][0]["head"] in seeds
                and all(a["tail"]==b["head"] for a,b in zip(p_["hops"],p_["hops"][1:]))]
            out[(r["id"],t["doc"])]=dict(rank=t["rank"],paths=ps,views=t.get("views",[]),stratum=r["stratum"],q=r["question"],seeds=r["seeds"])
    return out
if D:
    ON=load(f"{D}/hops_frame_ccmp.json"); OFF=load(f"{D}/hops_frame_ccmp_off.json")
    OIE=load(f"{D}/hops_openie.json") if os.path.exists(f"{D}/hops_openie.json") else {}
    BASE={}
    cj=f"{D}/showcase_sir4_cs_candidates.json"
    if os.path.exists(cj):
        for c in json.load(open(cj))["candidates"]:
            BASE[(c["id"],c["gold"])]={k:v for k,v in c["ranks"].items() if k in ("qwen3","bge","reasonir","bm25")}
else:
    ON=load(f"{Q}/cs_hops_frame_on.json"); OFF=load(f"{Q}/cs_hops_frame_off.json"); OIE=load(f"{Q}/cs_hops_openie.json"); BASE={}
docs=json.load(open(f"{R}/kg-construction/data/sir4_cs_test/raw/documents.json"))
queries={q["id"]:q for q in json.load(open(f"{R}/kg-construction/data/sir4_cs_test/raw/test.json"))}
qwen={}   # predictions file is on a slow mount; the dense column IS the Qwen3 cosine rank
qf={}
for x in json.load(open(f"{R}/quartet/data.nosync/benchmark/cs_test_final/eval.json")):
    for d,m in (x.get("quartet",{}).get("per_document") or {}).items():
        qf[(x["id"],d)]=dict(field_pair=m.get("field_pair"),stratum=m.get("stratum"))
dom=defaultdict(list)
for row in csv.reader(open(f"{R}/kg-construction/data/sir4_cs_test_v16sc/processed/stage1/edges.csv")):
    if len(row)>=3 and row[1]=="in_field": dom[row[0]].append(row[2].replace("[domain] ",""))
def ntype(n): return "paper" if n in docs else (n[1:n.index("]")] if n.startswith("[") and "]" in n else "entity")
def kind(p):
    if not p: return "none"
    inner=[ntype(h["head"]) for h in p["hops"]]
    if any(t in MECH for t in inner): return "bridge"
    return "hub" if any(t=="domain" for t in inner) else "other"
def title(d): return docs.get(d,d).split(". ")[0][:100]
def lab(n): return f"[paper] {title(n)[:55]}" if n in docs else n[:60]
def fmt(p,gate=True):
    s=[]
    for h in p["hops"]:
        g=f" <g={h['gate']:.2f}>" if gate and "gate" in h and abs(h["gate"]-1)>0.02 else ""
        s.append(f"{lab(h['head'])}{g} --{h['rel'].replace('inverse_','inv.')}--> ")
    return "".join(s)+lab(p["hops"][-1]["tail"])
rows=[]
for (qid,gold),t in ON.items():
    rk=t["rank"]; g=lambda c: rk.get(c,BIG)
    fp=(qf.get((qid,gold)) or {}).get("field_pair"); st=(qf.get((qid,gold)) or {}).get("stratum") or t["stratum"]
    if st!="cross" and t["stratum"]!="cross": continue
    o=OFF.get((qid,gold)); oi=OIE.get((qid,gold))
    b=BASE.get((qid,gold),{}); qr=b.get("qwen3")
    base_min=min([v for v in b.values() if v],default=None)   # best of the 4 dense/lexical baselines
    p=t["paths"][0] if t["paths"] else None
    k=kind(p); maxg=max([h.get("gate",1) for h in p["hops"]],default=1) if p else 1
    off_f=o["rank"]["fused"] if o else None; off_g=o["rank"]["graph"] if o else None
    row=dict(qid=qid,gold=gold,fp=fp,dense=g("dense"),scorer=g("scorer"),graph=g("graph"),fused=g("fused"),
             off_fused=off_f,off_graph=off_g,oie_fused=oi["rank"]["fused"] if oi else None,oie_graph=oi["rank"]["graph"] if oi else None,
             qwen=qr,base_min=base_min,base=b,kind=k,hops=len(p["hops"]) if p else 0,maxg=maxg)
    # bucket A: full model top-10, cosine buries, mechanism route
    A = row["fused"]<=10 and row["dense"]>=50 and k=="bridge" and (base_min is None or base_min>=50)
    # bucket B: graph-led rescue (graph<=5, scorer>=20), mechanism route
    B = row["graph"]<=5 and row["scorer"]>=20 and k in ("bridge","hub")
    ccmp_help = (off_f is not None and row["fused"]<off_f) or (off_g is not None and row["graph"]<off_g)
    row["bucket"]=("A" if A else "")+("B" if B else ""); row["ccmp_help"]=ccmp_help
    if A or B: rows.append(row)
rows.sort(key=lambda r:(-len(r["bucket"]),-r["ccmp_help"],r["fused"],-r["dense"]))
print(f"{len(rows)} candidates (A: full model top-10 & cosine>=50 & bridge; B: graph<=5 & scorer>=20 & bridge)\n")
print("| bkt | ccmp | fused/off | graph/off | scorer | cosine | best baseline | oie f/g | hops | gate | field pair | gold |")
for r in rows:
    print(f"| {r['bucket']:2} | {'Y' if r['ccmp_help'] else '-'} | {r['fused']}/{r['off_fused']} | {r['graph']}/{r['off_graph']} | {r['scorer']} | {r['dense']} | {r['base_min'] if r['base_min'] and r['base_min']<BIG else ('>300' if r['base'] else 'n/a')} | {r['oie_fused']}/{r['oie_graph']} | {r['hops']} | {r['maxg']:.2f} | {(r['fp'] or '?')[:40]} | {title(r['gold'])[:60]} |")
print("\n\n================ DETAILS ================\n")
for i,r in enumerate(rows,1):
    qid,gold=r["qid"],r["gold"]; t=ON[(qid,gold)]; o=OFF.get((qid,gold)); oi=OIE.get((qid,gold))
    print(f"### {i}. [{r['bucket']}{' CCMP+' if r['ccmp_help'] else ''}] {title(gold)}   ({r['fp']}; domain nodes: {'; '.join(dom.get(gold,[]))[:80]})")
    print(f"QUERY {qid}: {queries.get(qid,{}).get('question',t['q'])[:900]}")
    print(f"GOLD: {docs.get(gold,'')[:500]}")
    print(f"ranks: fused {r['fused']} (off {r['off_fused']}) | graph {r['graph']} (off {r['off_graph']}) | scorer {r['scorer']} | cosine {r['dense']} | baselines {r['base']} | openie fused {r['oie_fused']} graph {r['oie_graph']}")
    for v in t["views"][:2]: print(f"view {v['view']} ({v['match']:.2f}): {(v.get('text') or '')[:220]}")
    print("SEEDS:", "; ".join(t["seeds"][:8])[:400])
    for j,p in enumerate(t["paths"][:3]): print(f"ON  w={p['weight']:.2f}: {fmt(p)}")
    if o:
        for j,p in enumerate(o["paths"][:2]): print(f"OFF w={p['weight']:.2f}: {fmt(p,gate=False)}")
    if oi and oi["paths"]: print(f"OIE w={oi['paths'][0]['weight']:.2f}: {fmt(oi['paths'][0],gate=False)}")
    print()
