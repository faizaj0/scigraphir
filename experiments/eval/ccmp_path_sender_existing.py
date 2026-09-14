#!/usr/bin/env python3
"""Existing measurements for the path-sender examples, and the cases file for the Colab run.

Reads the 8 Sep 2026 showcase scan (results/qualitative/drive_scan_<dataset>/hops_frame_ccmp{,_off}.json:
zero-shot pair CCMP checkpoints outputs/sir4_zeroshot/scigraphir_<pair>_qwenmlp_ccmp_e10_b2 on the
<dataset>_test_v16sc frame graphs, bfloat16, beam 10, top-5 paths per target) and writes

    eval/ccmp_path_sender_cases.json                                    the examples for the notebook,
                                                                        with the stored paths as a
                                                                        reproduction reference only
    results/qualitative/ccmp_path_sender_effect/existing_measurements.json / .md

Delta_P is NOT in these files: it needs the Colab run (colab_ccmp_path_sender_effect.ipynb).

    python3 eval/ccmp_path_sender_existing.py
"""
import json
from pathlib import Path

S4 = Path(__file__).resolve().parents[1]
ROOT = S4.parent
OUT_DIR = S4 / "results/qualitative/ccmp_path_sender_effect"
TOP_K = 3

EXAMPLES = [
    {"name": "biology_optimal_transport", "dataset": "sir4_biology",
     "query_id": "10.1021_acsomega.5c12723", "gold_id": "10.1007/s11263-023-01831-9",
     "quartet": {"domain_distance": "disjoint", "field_pair": "Biochemistry, Genetics and Molecular Biology -> Computer Science",
                 "gold_stratum": "cross", "source": "benchmark/data/benchmark/biology_test_low/eval.json"},
     "graph_score_9sep": {"on": 48.750, "off": 15.062,
                          "source": "Drive outputs/scan/sir4_biology/fig_gate_decomp_sir4_biology.md, gate-decomposition run "
                                    "of 9 Sep 2026 (same checkpoint and graph, bfloat16): gate on rank 3 / gate off rank 436"}},
    {"name": "creativity_fixation", "dataset": "sir4_cs",
     "query_id": "10.48550_arxiv.2602.20408", "gold_id": "10.3758/bf03202751",
     "quartet": {"domain_distance": "disjoint", "field_pair": "Computer Science -> Psychology", "gold_stratum": "cross",
                 "source": "benchmark/data/benchmark/cs_test_final/eval.json"}},
    {"name": "memory_reconsolidation", "dataset": "sir4_cs",
     "query_id": "10.48550_arxiv.2603.03985", "gold_id": "10.1111/j.1749-6632.2010.05443.x",
     "quartet": {"domain_distance": "disjoint", "field_pair": "Computer Science -> Neuroscience", "gold_stratum": "cross",
                 "source": "benchmark/data/benchmark/cs_test_final/eval.json"}},
    {"name": "dueling_bandits", "dataset": "sir4_cs",
     "query_id": "10.48550_arxiv.2602.21585", "gold_id": "q:4f02a72cb830786c",
     "quartet": {"domain_distance": "overlap_1", "field_pair": "Computer Science -> Decision Sciences", "gold_stratum": "same",
                 "note": "query stratum is cross, but QUARTET's gold-level label is same (field confidence ambiguous); "
                         "cite as a within-CS decision-theory example, not as cross-field",
                 "source": "benchmark/data/benchmark/cs_test_final/eval.json"}},
]


def is_simple(path):
    nodes = [path["hops"][0]["head"]] + [h["tail"] for h in path["hops"]]
    return len(nodes) == len(set(nodes))


def hop_key(path):
    return tuple((h["head"], h["rel"], h["tail"]) for h in path["hops"])


def title(documents, doc_id):
    value = documents.get(doc_id)
    if isinstance(value, dict):
        return str(value.get("title") or doc_id)
    if isinstance(value, str):
        return value.split(". ")[0].rstrip(".")
    return doc_id


def find_target(hops, query_id, gold_id):
    for rec in hops:
        if rec["id"] == query_id:
            for t in rec["targets"]:
                if t["doc"] == gold_id:
                    return rec, t
    raise KeyError(f"{query_id} -> {gold_id} not in the hops file")


def route(path, documents):
    parts = [title(documents, path["hops"][0]["head"]) if path["hops"][0]["head"] in documents else path["hops"][0]["head"]]
    for h in path["hops"]:
        rel = h["rel"].removeprefix("inverse_").replace("_", " ") + ("⁻¹" if h["rel"].startswith("inverse_") else "")
        tail = title(documents, h["tail"]) if h["tail"] in documents else h["tail"]
        parts += [f"[{rel}]", tail]
    return " → ".join(parts)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases, existing = [], []
    md = ["# Existing measurements for the path-sender examples (8 Sep 2026 showcase scan)", "",
          "Source: `results/qualitative/drive_scan_<dataset>/hops_frame_ccmp.json` (CCMP gate on) and `hops_frame_ccmp_off.json` "
          "(same weights, every gate 1); checkpoints `outputs/sir4_zeroshot/scigraphir_<pair>_qwenmlp_ccmp_e10_b2`, graphs "
          "`<dataset>_test_v16sc`, bfloat16, beam 10. Attribution = beam weight = mean edge gradient along the path (GFM-RAG "
          "convention), read from the gate-on run; the gate-off attribution is listed when the same route was among the gate-off "
          "beam's top 5. Per-hop gates are the normalised CCMP gates of the sender at the layer the hop is attributed to.", "",
          "**Δ_P has not been measured yet.** It needs the frozen-gate replay in `colab_ccmp_path_sender_effect.ipynb`; the "
          "column is left empty on purpose.", ""]
    for ex in EXAMPLES:
        scan = S4 / f"results/qualitative/drive_scan_{ex['dataset']}"
        on = json.loads((scan / "hops_frame_ccmp.json").read_text())
        off = json.loads((scan / "hops_frame_ccmp_off.json").read_text())
        documents = json.loads((ROOT / f"retriever/data/{ex['dataset']}_test/raw/documents.json").read_text())
        rec_on, t_on = find_target(on, ex["query_id"], ex["gold_id"])
        _, t_off = find_target(off, ex["query_id"], ex["gold_id"])
        simple_on = [p for p in t_on["paths"] if is_simple(p)][:TOP_K]
        off_weights = {hop_key(p): p["weight"] for p in t_off["paths"]}
        paths = []
        for i, p in enumerate(simple_on, 1):
            paths.append({"name": f"P{i}", "weight_on": p["weight"], "weight_off": off_weights.get(hop_key(p)),
                          "hops": [{"layer": h["layer"], "head": h["head"], "rel": h["rel"], "tail": h["tail"],
                                    "gate": h.get("gate"), "resp": h.get("resp"), "frontier_mean": h.get("frontier_mean")}
                                   for h in p["hops"]]})
        entry = {"name": ex["name"], "dataset": ex["dataset"], "query_id": ex["query_id"], "gold_id": ex["gold_id"],
                 "gold_title": title(documents, ex["gold_id"]), "question": rec_on["question"],
                 "query_stratum": rec_on.get("stratum"), "quartet": ex["quartet"], "n_seeds": len(rec_on["seeds"]),
                 "ranks_on": t_on["rank"], "ranks_off": t_off["rank"], "dense_cos": t_on.get("dense_cos"),
                 "frontier_mean_resp_on": t_on.get("frontier_mean_resp"), "n_beam_paths_on": len(t_on["paths"]),
                 "n_simple_paths_on": sum(is_simple(p) for p in t_on["paths"]), "top_simple_paths_on": paths,
                 "paper_titles": {n: title(documents, n) for p in paths for h in p["hops"] for n in (h["head"], h["tail"]) if n in documents}}
        if "graph_score_9sep" in ex:
            entry["graph_score_9sep"] = ex["graph_score_9sep"]
        existing.append(entry)
        cases.append({"name": ex["name"], "dataset": ex["dataset"], "query_id": ex["query_id"], "gold_id": ex["gold_id"],
                      "quartet": ex["quartet"],
                      "expected_8sep_bf16": {"note": "reproduction reference only; never an input to the measurement",
                                             "ranks_on": t_on["rank"], "ranks_off": t_off["rank"],
                                             "top_simple_paths_on": [{"name": p["name"], "weight_on": p["weight_on"],
                                                                      "hops": [{k: h[k] for k in ("layer", "head", "rel", "tail")} for h in p["hops"]]}
                                                                     for p in paths]}})
        ro, rf = t_on["rank"], t_off["rank"]
        md += [f"## {ex['name']}", "",
               f"**Query** ({ex['query_id']}, {ex['dataset']}, stratum {rec_on.get('stratum')}): {rec_on['question'][:350]}…", "",
               f"**Target paper**: {entry['gold_title']} (`{ex['gold_id']}`); QUARTET gold-level label: {ex['quartet']['domain_distance']}, "
               f"{ex['quartet']['field_pair']}, stratum {ex['quartet']['gold_stratum']}.", "",
               f"**Graph rank** CCMP on {ro['graph']} / all gates 1 {rf['graph']}; fused rank {ro['fused']} / {rf['fused']}; "
               f"scorer rank {ro['scorer']}; dense (Qwen3 cosine) rank {ro['dense']}."
               + (f" Graph score (9 Sep gate-decomposition run): on {ex['graph_score_9sep']['on']:.3f} / off {ex['graph_score_9sep']['off']:.3f}."
                  if "graph_score_9sep" in ex else " Graph scores: not stored in the scan; measured by the Colab run."), "",
               f"Seeds: {len(rec_on['seeds'])}; beam paths stored: {len(t_on['paths'])}, of which simple: {entry['n_simple_paths_on']}.", "",
               "| Path | Route (seed → … → target) | Attribution, gate on | Attribution, gate off | Per-hop gates (on) | Δ_P |",
               "|---|---|---:|---:|---|---|"]
        for p in paths:
            gates = ", ".join(f"{h['gate']:.4f}" if h["gate"] is not None else "n/a" for h in p["hops"])
            w_off = f"{p['weight_off']:.3f}" if p["weight_off"] is not None else "not in gate-off top 5"
            md.append(f"| {p['name']} | {route(p, documents)} | {p['weight_on']:.3f} | {w_off} | {gates} | pending |")
        md.append("")
    (S4 / "eval/ccmp_path_sender_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (OUT_DIR / "existing_measurements.json").write_text(json.dumps(existing, indent=2) + "\n")
    (OUT_DIR / "existing_measurements.md").write_text("\n".join(md) + "\n")
    print(f"wrote {S4 / 'eval/ccmp_path_sender_cases.json'} ({len(cases)} cases)")
    print(f"wrote {OUT_DIR / 'existing_measurements.json'} and .md")
    for e in existing:
        print(f"  {e['name']}: graph rank on {e['ranks_on']['graph']} / off {e['ranks_off']['graph']}; "
              f"top simple paths {[round(p['weight_on'], 3) for p in e['top_simple_paths_on']]}")


if __name__ == "__main__":
    main()
