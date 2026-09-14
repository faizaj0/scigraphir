"""Table-4-style scientific paths from the CURRENT model, using NBFNet beam search.

The target is a pinned benchmark-relevant PAPER. Rankings are reported even when
it is retrieved poorly. Paths are post-hoc gradient attributions, not generated
answers, proofs, probabilities, or experiments validating scientific transfer.
"""
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ == "gfmrag.workflow":
    from .ccmp_mechanism_core import (run_condition, resolve_paths, score_metrics, atomic_json,
                                      compare_scores, numerical_allowance, inverse_relation_id)
else:
    from ccmp_mechanism import (run_condition, resolve_paths, score_metrics, atomic_json,
                                compare_scores, numerical_allowance, inverse_relation_id)


def document_title(name, documents):
    value = documents.get(name, name)
    if isinstance(value, dict):
        return str(value.get("title") or value.get("text") or name).split("\n")[0]
    return str(value).split("\n")[0].split(". ")[0].rstrip(".")


def graph_domains(node, graph, src):
    if "in_field" not in src.rel2id:
        return []
    mask = (graph.edge_index[0] == node) & (graph.edge_type == src.rel2id["in_field"])
    return sorted({src.id2node[int(i)].removeprefix("[domain] ") for i in graph.edge_index[1, mask].tolist()})


def decode_candidates(raw_paths, raw_weights, edge_grads, graph, batch, src,
                      gold_id, gates, gate_stats, documents, top_k):
    """Keep genuine simple paths; never silently replace a weak path with a curated one."""
    id2rel = {v: k for k, v in src.rel2id.items()}
    valid, rejected, seen = [], [], set()
    for raw, beam_weight in zip(raw_paths, raw_weights):
        raw = tuple(tuple(int(v) for v in hop) for hop in raw)
        if raw in seen:
            continue
        seen.add(raw)
        if not math.isfinite(float(beam_weight)):
            rejected.append("nonfinite beam score")
            continue
        # The beam ran on the kernel's flow graph (edge_index flipped): raw (h, t, r) is the message from
        # sender h to receiver t on the edge (t, r, h). Display the KG fact read from sender to receiver,
        # (h, inverse(r), t); resolve_paths maps it back to that same message edge.
        hops = [{"layer": l, "head": src.id2node[h], "rel": id2rel[inverse_relation_id(r, len(src.rel2id))],
                 "tail": src.id2node[t]} for l, (h, t, r) in enumerate(raw)]
        try:
            _, resolved, _ = resolve_paths({"gold_id": gold_id, "paths": [{"name": "candidate", "hops": hops}]},
                                           src, graph, batch, len(edge_grads), "sender_layer")
        except ValueError as e:
            rejected.append(str(e))
            continue
        detailed = resolved[0]["hops"]
        for hop in detailed:
            l, h = hop["layer"], hop["head_id"]
            gate = gates[l][0, h]
            dtype = getattr(torch, gate_stats[l]["message_dtype"].split(".")[-1])
            hop.update(edge_gradient=float(edge_grads[l][hop["edge_id"]]),
                       g_raw=float(gate), g_applied=float(gate.to(dtype)))
        weight = sum(h["edge_gradient"] for h in detailed) / len(detailed)
        if not math.isfinite(weight):
            raise ValueError("Nonfinite path attribution")
        nodes = [detailed[0]["head"]] + [h["tail"] for h in detailed]
        labels = [h["rel"].removeprefix("inverse_") for h in detailed]
        flags = []
        if len(detailed) == 1:
            flags.append("single hop")
        if any(n.startswith("[domain]") for n in nodes):
            flags.append("contains domain hub")
        if any(r in {"equivalent", "same_as"} for r in labels):
            flags.append("contains alias link")
        mechanism = {"achieves", "overcomes", "works_via", "limited_by", "concerns", "explains",
                     "paper_achieves", "paper_overcomes", "paper_works_via", "paper_limited_by",
                     "paper_concerns", "paper_explains"}
        if not mechanism.intersection(labels):
            flags.append("no mechanism/function/limitation relation")
        valid.append({"weight": weight, "beam_weight": float(beam_weight), "hops": detailed,
                      "flags": flags, "papers_on_path": [n for n in dict.fromkeys(nodes) if n in documents]})
    valid.sort(key=lambda p: (-p["weight"], tuple((h["head"], h["rel"], h["tail"]) for h in p["hops"])))
    for rank, p in enumerate(valid, 1):
        p.update(name=f"P{rank}", rank_in_candidates=rank)
    return {"top_paths": valid[:top_k],
            "top_multihop_paths": [p for p in valid if len(p["hops"]) >= 2][:top_k],
            "valid_candidate_count": len(valid), "rejected_count": len(rejected),
            "rejection_reasons": sorted(set(rejected))}


def interpret_scientific_paths(model, graph, batch, src, case, dtype, documents,
                               top_k=5, beam_size=10):
    """Find the highest-attribution paths returned by the engine's finite beam.

    No historical paths/weights enter selection. Beam search ranks cumulative
    gradient scores, not hop count; candidate weights are divided by path length.
    """
    if top_k < 1 or beam_size < top_k:
        raise ValueError("Require beam_size >= top_k >= 1")
    model.eval()
    em = model.base.entity_model
    if (getattr(em, "resp_proj", None) is None or not em.resp_gate_norm or getattr(em, "route_mode", "")
            or getattr(em, "attn_node", None) is not None or getattr(graph, "dist_context", None) is not None):
        raise ValueError("This notebook requires single-process CCMP, without alternate routing")
    target = src.node2id[case["gold_id"]]
    if not bool(batch["target_nodes_mask"][0, target] > 0):
        raise ValueError("Target must be a benchmark gold paper")
    protected = torch.zeros((len(em.layers), 1, graph.num_nodes), device=graph.edge_index.device, dtype=torch.bool)
    diagnostics = {}
    # Same graph-score forward and gradients as the matched CCMP workflow, but no
    # preselected paths. This is the actual native-CCMP forward pass.
    record, gates, scores, _ = run_condition(model, graph, batch, target, [], protected,
                                            "native", dtype, diagnostics=diagnostics)
    old_gate = em.resp_gate
    try:
        em.resp_gate = True
        with torch.no_grad(), torch.autocast(device_type=graph.edge_index.device.type,
                                            dtype=dtype, enabled=dtype != torch.float32):
            plain = model(graph, batch)
            plain_scores = {"graph": model._raw_doc[0].detach().float().cpu(),
                            "fused": plain[0, model._doc_ids].detach().float().cpu(),
                            "scorer": model._s_op[0].detach().float().cpu()}
        forward_allowance = numerical_allowance(scores, dtype)
        forward_check = compare_scores(scores, plain_scores, 1e-5, 1e-4,
                                       "Path attribution vs uninstrumented inference", forward_allowance)
    finally:
        em.resp_gate = old_gate
    seed_ids = batch["start_nodes_mask"][0].nonzero(as_tuple=True)[0]
    # KERNEL DIRECTION. The gradients above come from the rspmm kernel (layers.py propagate), which
    # receives at edge_index[0] and sends from edge_index[1]. NBFNet's beam search walks edges from
    # edge_index[0] to edge_index[1], so hand it the flipped index: paths then follow the messages the
    # model actually passed, seed -> ... -> target, and "no path leaves the target" keeps its meaning.
    flow = SimpleNamespace(num_nodes=graph.num_nodes, edge_index=graph.edge_index.flip(0),
                           edge_type=graph.edge_type)
    with torch.no_grad():
        distances, back_edges = em.beam_search_distance(
            flow, diagnostics["edge_grads"], seed_ids, target, num_beam=beam_size)
        # Inspect every target candidate across depths before removing cyclic /
        # placeholder paths. Asking for only k initially can leave too few valid paths.
        raw_paths, raw_weights = em.topk_average_length(
            distances, back_edges, target, k=beam_size * len(em.layers))
    found = decode_candidates(raw_paths, raw_weights, diagnostics["edge_grads"], graph, batch, src,
                              case["gold_id"], gates, record["gate_stats"], documents, top_k)
    did, j = diagnostics["doc_ids"].cpu(), diagnostics["target_position"]
    gm = batch["target_nodes_mask"][0, diagnostics["doc_ids"]].cpu() > 0
    ranks = dict(record["channels"])
    ranks["scorer"] = score_metrics(scores["scorer"], j, gm)
    tag_row = getattr(model, "_sem_row", {}).get(str(batch["id"][0]))
    if tag_row is not None:
        tag, row = tag_row
        dense = model._sem_tab[tag]["dense"][row].detach().float().cpu()
        if dense.numel() != did.numel():
            raise ValueError("Dense reference does not match the model's document order")
        ranks["dense"] = score_metrics(dense, j, gm)
    else:
        ranks["dense"] = None
    raw_query = next((q for q in src.raw_test_data if str(q["id"]) == case["query_id"]), None)
    if raw_query is None:
        raise ValueError("Question text / stratum missing")
    top_documents = {k: [{"id": src.id2node[int(did[i])], "title": document_title(src.id2node[int(did[i])], documents),
                          "score": float(v[i])} for i in v.topk(min(top_k, v.numel())).indices.tolist()]
                     for k, v in scores.items() if k in {"graph", "fused"}}
    result = {"case": case, "question": raw_query["question"], "query_field": case["dataset"].removeprefix("sir4_"),
              "benchmark_stratum": raw_query.get("stratum"), "target_id": case["gold_id"],
              "target_title": document_title(case["gold_id"], documents),
              "target_domains": graph_domains(target, graph, src),
              "gold_papers": [{"id": d, "title": document_title(d, documents)} for d in raw_query["supporting_documents"]],
              "channels": ranks, "top_documents": top_documents, "gate_stats": record["gate_stats"], **found,
              "forward_score_check": forward_check, "forward_score_allowance": forward_allowance,
              "beam_size": beam_size, "requested_top_k": top_k,
              "search_note": "Approximate top simple paths among finite-beam candidates; no paths are invented if none qualify.",
              "scope": "Attribution to the graph score of a preselected relevant paper; not an answer-generation trace."}
    paper_ids = {d for p in found["top_paths"] + found["top_multihop_paths"] for d in p["papers_on_path"]}
    result["paper_titles"] = {d: document_title(d, documents) for d in sorted(paper_ids)}
    print(f"[paths] {case['name']}: {found['valid_candidate_count']} valid candidates; "
          f"graph rank {ranks['graph']['rank']}, fused rank {ranks['fused']['rank']}", flush=True)
    for p in result["top_paths"]:
        print(f"  {p['name']} w={p['weight']:.6g}: " + " -> ".join(
            f"({h['head']}, {h['rel']}, {h['tail']})" for h in p["hops"]), flush=True)
    return result, {"native_raw": torch.stack(gates).cpu(),
                    "message_dtypes": [s["message_dtype"] for s in record["gate_stats"]]}


def tex_escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(text))


def relation_text(rel, tex=False):
    inverse = rel.startswith("inverse_")
    value = rel.removeprefix("inverse_").replace("_", " ")
    return (tex_escape(value) + (r"$^{-1}$" if inverse else "")) if tex else value + ("⁻¹" if inverse else "")


def path_text(path, titles, tex=False):
    def name(n):
        s = titles.get(n, n)
        return tex_escape(s) if tex else s.replace("|", "\\|")
    triples = [f"({name(h['head'])}, {relation_text(h['rel'], tex)}, {name(h['tail'])})" for h in path["hops"]]
    return (r" $\rightarrow$ " if tex else " → ").join(triples)


def render_scientific_table(payload, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection = payload["manifest"].get("selection") or {}
    md = ["# Scientific path interpretations", "",
          f"Graph: `{selection.get('graph', '')}`. Checkpoint: `{selection.get('checkpoint', '')}`.", "",
          "Paths are discovered from the current checkpoint. Weights are gradient attributions, not probabilities. "
          "The target is a preselected benchmark-relevant paper; its actual retrieval rank is reported. "
          "The cross stratum is a dataset annotation, and target domains are graph labels.", ""]
    tex = [r"% Include with \usepackage{booktabs,tabularx}. Inverse relations are traversed backwards.",
           r"\begin{table*}[t]", r"\centering\small", r"\renewcommand{\arraystretch}{1.15}",
           r"\begin{tabularx}{\textwidth}{@{}p{0.13\textwidth}X@{}}", r"\toprule"]
    csv_rows, intervention_cases = [], []
    for record in payload["results"]:
        title_map = record["paper_titles"]
        name = record["case"]["name"]
        info = [("Question", record["question"]), ("Query field", record["query_field"]),
                ("Relevant paper", record["target_title"]), ("Paper domain", ", ".join(record["target_domains"]) or "unknown"),
                ("Benchmark stratum", record["benchmark_stratum"] or "unknown")]
        md += [f"## {name}", ""] + [f"**{label}:** {text}\n" for label, text in info]
        ranks = " | ".join(f"{k}: {v['rank']}" for k, v in record["channels"].items() if v is not None)
        md += [f"**Target ranks (1 is best):** {ranks}", "",
               "**Top fused retrieved papers:** " + "; ".join(f"{i}. {d['title']}" for i, d in enumerate(record["top_documents"]["fused"], 1)), "",
               "### Top paths, including single-hop paths", "",
               "| Path | Weight | Hops | Applied gates, in hop order | Graph route | Notes |", "|---|---:|---:|---|---|---|"]
        for p in record["top_paths"]:
            gate_text = ", ".join(f"{h['g_applied']:.6g}" for h in p["hops"])
            md.append(f"| {p['name']} | {p['weight']:.6g} | {len(p['hops'])} | {gate_text} | {path_text(p, title_map)} | {', '.join(p['flags'])} |")
        if not record["top_paths"]:
            md += ["No valid simple path was returned by the configured beam search."]
        md += ["", "### Highest-ranked multi-hop subset", "",
               "This subset requires at least two hops; P-numbers retain their rank among all valid beam candidates.", ""]
        for p in record["top_multihop_paths"]:
            md += [f"**{p['name']} — w={p['weight']:.6g}:** {path_text(p, title_map)}", "",
                   "Applied sender gates: " + ", ".join(f"hop {h['layer']+1}: {h['g_applied']:.6g}" for h in p["hops"]) + ".", ""]
        if not record["top_multihop_paths"]:
            md += ["No multi-hop path was returned. This example does not currently provide a multi-hop path illustration.", ""]
        md += ["**Papers appearing on displayed paths:** " + "; ".join(title_map.values()), "",
               "**Interpretation:** inspect whether the route connects a scientific problem/function/limitation to a relevant "
               "method in another domain. A domain hub or alias alone is weak evidence. These paths do not show that the "
               "retrieved method has experimentally solved the query's problem.", ""]
        for label, text in info:
            tex.append(tex_escape(label) + " & " + tex_escape(text) + r" \\")
        tex.append("Ranks & " + tex_escape(ranks) + r" \\")
        for index, p in enumerate(record["top_paths"][:2]):
            tex.append(("Top paths" if index == 0 else "") + f" & {p['weight']:.4f}: " + path_text(p, title_map, tex=True) + r" \\")
        tex.append(r"\midrule")
        for p in record["top_paths"] + record["top_multihop_paths"]:
            for h in p["hops"]:
                row = {"case": name, "path": p["name"], "weight": p["weight"], "graph": selection.get("graph", ""), **h}
                if row not in csv_rows:
                    csv_rows.append(row)
        if record["top_paths"]:
            c = {k: v for k, v in record["case"].items() if not k.startswith("historical_") and k not in {"paths", "source"}}
            c["paths"] = [{"name": p["name"], "hops": [{k: h[k] for k in ("layer", "head", "rel", "tail")} for h in p["hops"]]}
                          for p in record["top_paths"][:2]]
            c["selection_source"] = {"run_id": payload["run_id"], "graph": selection.get("graph"),
                                     "checkpoint": selection.get("checkpoint"),
                                     "checkpoint_sha256": payload["manifest"].get("checkpoint_sha256"),
                                     "precision": payload["manifest"].get("precision"),
                                     "rule": "top two native simple paths; no multi-hop filtering"}
            c["historical_reference_applicable"] = False
            intervention_cases.append(c)
    tex += [r"\bottomrule\end{tabularx}",
            r"\caption{Gradient-attributed scientific retrieval paths from the current checkpoint. Targets are selected benchmark-relevant papers; ranks report actual retrieval. "
            r"Relation $r^{-1}$ traverses an original edge in reverse. Weights are not probabilities. Beam search is approximate.}",
            r"\label{tab:scientific-paths}", r"\end{table*}"]
    (out / "table4.md").write_text("\n".join(md) + "\n")
    (out / "table4.tex").write_text("\n".join(tex) + "\n")
    if csv_rows:
        with (out / "path_hops.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    atomic_json(out / "discovered_cases_for_interventions.json", intervention_cases)
    # Editable TikZ path strips; use \input inside a document with the listed packages.
    tikz = [r"% Preamble: \usepackage{tikz}\usetikzlibrary{arrows.meta}",
            r"% One strip per discovered top path. Coordinates and boxes are editable."]
    for rec in payload["results"]:
        for p in rec["top_paths"][:2]:
            hops = p["hops"]
            nodes = [hops[0]["head"]] + [h["tail"] for h in hops]
            spacing = 15.0 / max(1, len(hops))
            width = min(2.6, spacing - 0.30)
            tikz += [r"\begin{tikzpicture}[>=Stealth, every node/.style={font=\scriptsize}]",
                     r"\node[anchor=west,align=left,text width=16cm] at (-1.3,2.4) {" + tex_escape(rec["case"]["name"])
                     + f"; {p['name']}; $w={p['weight']:.4f}$" + "};"]
            for i, n in enumerate(nodes):
                label = rec["paper_titles"].get(n, n)
                tikz.append(r"\node[draw,rounded corners=2pt,align=center,text width=" + f"{width:.2f}cm,minimum height=1.0cm"
                            + f"] (n{i}) at ({i*spacing:.2f},0) " + "{" + tex_escape(label) + "};")
            for i, h in enumerate(hops):
                label = relation_text(h["rel"], tex=True) + rf"\\$g={h['g_applied']:.4f}$"
                tikz.append(rf"\draw[->] (n{i}) -- node[above,align=center,font=\tiny,text width={spacing:.2f}cm] "
                            + "{" + label + "}" + f" (n{i+1});")
            tikz += [r"\end{tikzpicture}", r"\par\medskip"]
    (out / "top_paths.tikz.tex").write_text("\n".join(tikz) + "\n")
