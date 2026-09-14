"""Where do MIR's inspiration papers come from? Venue / arXiv category of every corpus
document via Semantic Scholar (batch) + arXiv (for arXiv-only records). Free APIs, read-only."""
import json, os, re, sys, time, collections, urllib.request, urllib.parse, xml.etree.ElementTree as ET
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'mir')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'mir')
CACHE = f"{OUT}/mir_s2_meta.json"

frames = {s: pd.read_csv(f"{D}/{s}_chronological_df.csv") for s in ("train", "dev", "test")}
ids = sorted(set().union(*[set(f.cited_paper_id.astype(str)) for f in frames.values()]))
print("unique cited papers across splits:", len(ids))

meta = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
todo = [i for i in ids if i not in meta]
print("to fetch from S2:", len(todo))
FIELDS = "title,year,venue,publicationVenue,externalIds,s2FieldsOfStudy,fieldsOfStudy"
for k in range(0, len(todo), 500):
    chunk = todo[k:k + 500]
    req = urllib.request.Request(
        "https://api.semanticscholar.org/graph/v1/paper/batch?fields=" + FIELDS,
        data=json.dumps({"ids": chunk}).encode(), headers={"Content-Type": "application/json"})
    for att in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                rows = json.load(r)
            break
        except Exception as e:
            print("  retry", att, e); time.sleep(10 * (att + 1))
    else:
        sys.exit("S2 batch failed")
    for pid, row in zip(chunk, rows):
        meta[pid] = row or {}
    json.dump(meta, open(CACHE, "w"))
    print(f"  fetched {min(k + 500, len(todo))}/{len(todo)}"); time.sleep(1.5)

# arXiv primary category for records whose venue is arXiv-only
arx = {pid: (m.get("externalIds") or {}).get("ArXiv") for pid, m in meta.items()}
need_cat = [pid for pid, a in arx.items() if a and not meta[pid].get("arxiv_primary")]
print("arXiv ids to categorise:", len(need_cat))
for k in range(0, len(need_cat), 100):
    chunk = need_cat[k:k + 100]
    url = "http://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(arx[p] for p in chunk)
    for att in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                xmlt = r.read()
            break
        except Exception as e:
            print("  retry", att, e); time.sleep(10)
    ns = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}
    by_id = {}
    for e in ET.fromstring(xmlt).findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/abs/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        pc = e.find("ar:primary_category", ns)
        by_id[aid] = pc.get("term") if pc is not None else None
    for p in chunk:
        meta[p]["arxiv_primary"] = by_id.get(re.sub(r"v\d+$", "", arx[p]), None) or "?"
    json.dump(meta, open(CACHE, "w")); time.sleep(3)

CL_VENUES = re.compile(r"(\b(ACL|EMNLP|NAACL|EACL|AACL|COLING|CoNLL|TACL|SemEval|LREC|IJCNLP|SIGDIAL|WMT|NLP)\b|"
                       r"Computational Linguistics|Empirical Methods in Natural Language|Natural Language (Processing|Learning|Generation|Understanding)|"
                       r"Association for Computational|Language Resources|Computational Natural Language|Machine Translation|"
                       r"Semantic Evaluation|Dialogue|Lexical|Linguistic|\*SEM|Findings|Human Language Technology|Parsing Technologies|"
                       r"Language Engineering|Language, Information and Computation|Language Technology|Computational Semantics|TALIP|Computational Logic)", re.I)
ML_VENUES = re.compile(r"\b(NeurIPS|NIPS|ICML|ICLR|AAAI|IJCAI|JMLR|KDD|UAI|AISTATS|Machine Learning)\b|Neural Information Processing|"
                       r"Artificial Intelligence|Learning Representations|Knowledge Discovery|Data Mining", re.I)
CV_VENUES = re.compile(r"\b(CVPR|ICCV|ECCV|TPAMI|Computer Vision)\b", re.I)
SP_VENUES = re.compile(r"\b(ICASSP|Speech|ASRU|SLT|Interspeech)\b|Audio", re.I)
IR_VENUES = re.compile(r"\b(SIGIR|WSDM|WWW|CIKM|ECIR|Information Retrieval|TheWebConf|NTCIR)\b|Web Conference|Web and Social Media|Information and Knowledge Management|Inf\. Sci", re.I)

def field(m):
    v = " ".join(filter(None, [m.get("venue") or "", (m.get("publicationVenue") or {}).get("name") or ""]))
    cat = m.get("arxiv_primary") or ""
    if CL_VENUES.search(v) or cat == "cs.CL": return "CL / NLP"
    if SP_VENUES.search(v) or cat in ("eess.AS", "cs.SD"): return "speech"
    if CV_VENUES.search(v) or cat == "cs.CV": return "computer vision"
    if IR_VENUES.search(v) or cat == "cs.IR": return "information retrieval"
    if ML_VENUES.search(v) or cat in ("cs.LG", "stat.ML", "cs.AI", "cs.NE"): return "ML / AI (general)"
    if cat: return f"other arXiv ({cat})"
    if v.strip(): return "other venue"
    return "unknown"

rows = []
for s, f in frames.items():
    g = f[f.inspirational == 1]
    for pid in g.cited_paper_id.astype(str).unique():
        rows.append((s, "inspiration (gold)", field(meta.get(pid, {}))))
    for pid in f.cited_paper_id.astype(str).unique():
        rows.append((s, "all cited candidates", field(meta.get(pid, {}))))
df = pd.DataFrame(rows, columns=["split", "set", "field"])
for (s, st), grp in df.groupby(["split", "set"]):
    c = grp.field.value_counts(); n = len(grp)
    print(f"\n{s} | {st} | n={n}")
    for k, v in c.items(): print(f"  {k:28} {v:5}  {100*v/n:5.1f}%")
# coarse same/cross for the gold inspirations, pooled
gold = df[df.set == "inspiration (gold)"]
same = (gold.field == "CL / NLP").sum(); unk = (gold.field == "unknown").sum(); n = len(gold)
print(f"\nGOLD INSPIRATIONS, all splits pooled: n={n} | CL/NLP (same-field) {same} = {100*same/n:.1f}% | "
      f"non-CL (cross-field) {n-same-unk} = {100*(n-same-unk)/n:.1f}% | unknown {unk}")
ex = collections.defaultdict(list)
for pid, m in meta.items():
    ex[field(m)].append((m.get("title") or "")[:70] + " | " + (m.get("venue") or (m.get("arxiv_primary") or "")))
other = collections.Counter((m.get("venue") or (m.get("publicationVenue") or {}).get("name") or "") for m in meta.values() if field(m) == "other venue")
print("\nremaining other-venue names (top 25):"); [print(f"  {n:4} {v[:80]}") for v, n in other.most_common(25)]
for k in ("ML / AI (general)", "computer vision", "speech"):
    print(f"\nexamples {k}:"); [print("  -", t) for t in ex[k][:4]]
df.to_csv(f"{OUT}/mir_fields.csv", index=False)
