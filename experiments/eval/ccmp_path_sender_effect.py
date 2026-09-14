"""Path-sender gate effect (Delta_P) for CCMP: fixed paths, frozen recorded gates, one checkpoint.

Installed by the Colab notebook as gfmrag/workflow/ccmp_path_sender_core.py. Builds on the
matched-intervention machinery in ccmp_mechanism.py (engine hook after gate normalisation,
frozen replay, numerical allowances) and on the finite-beam path discovery in
scientific_path_interpretations.py. No training, inference and gradients only.

Procedure, per (query, target paper) example:

 1. Native CCMP-on forward pass with edge gradients; NBFNet / GFM-RAG gradient beam search
    from the query's seed nodes to the target; the top-K valid simple paths (K = 3 by
    default), ranked by their mean edge gradient, are FIXED for every later condition.
 2. "off": every gate is 1 (CCMP removed at inference). Target graph score and rank.
 3. "frozen": the recorded, normalised native gates are replayed; the scores must reproduce
    the native run (bit-identical applied gates, scores within the measured allowance).
 4. "reset_senders_<P>": the recorded gates are replayed, except that the gates of P's unique
    sender nodes are set to 1 at EVERY layer. Nothing is renormalised and no gate is
    recomputed from the changed hidden states.

        Delta_P = graph score (native CCMP on) - graph score (after resetting P's senders)

Delta_P is a path-sender gate effect, not the exclusive contribution of the path: a sender node
also scales its messages on every other route, and overlapping paths must not be summed. The
GFM-RAG attribution (mean edge gradient along the path) is reported alongside and is never
derived from the gates. A higher target score does not necessarily improve its rank; ranks are
reported separately.
"""
import csv
import json
import math
from pathlib import Path

import torch

if __package__ == "gfmrag.workflow":
    from .ccmp_mechanism_core import (atomic_json, compare_scores, numerical_allowance, resolve_paths,
                                      run_condition, score_deltas)
    from .scientific_paths import document_title, graph_domains, interpret_scientific_paths, tex_escape
else:
    from ccmp_mechanism import (atomic_json, compare_scores, numerical_allowance, resolve_paths,
                                run_condition, score_deltas)
    from scientific_path_interpretations import document_title, graph_domains, interpret_scientific_paths, tex_escape

VERSION = "ccmp-path-sender-v1"
BASE_CONDITIONS = ("off", "native", "frozen")


def sender_ids(resolved_path):
    """Unique sender (head) nodes of a resolved path, in node-id order."""
    return sorted({int(h["head_id"]) for h in resolved_path["hops"]})


def sender_mask(sender_sets, layers, num_nodes, device):
    """[layers, 1, N] boolean mask: True for every listed sender at EVERY layer."""
    mask = torch.zeros((layers, 1, num_nodes), dtype=torch.bool, device=device)
    ids = sorted(set().union(*[set(s) for s in sender_sets])) if sender_sets else []
    if ids:
        mask[:, 0, ids] = True
    return mask


def fixed_paths_from_discovery(discovery, top_k):
    """The top-K discovered simple paths as plain (layer, head, rel, tail) hop lists."""
    return [{"name": p["name"], "hops": [{k: h[k] for k in ("layer", "head", "rel", "tail")} for h in p["hops"]]}
            for p in discovery["top_paths"][:top_k]]


def _check_gate_reset(gates_reset, gates_native, senders, layers):
    """The reset run must apply exactly the native gates off the senders and exactly 1 on them."""
    for layer in range(layers):
        g, g0 = gates_reset[layer], gates_native[layer]
        keep = torch.ones(g.shape[-1], dtype=torch.bool, device=g.device)
        keep[senders] = False
        if not torch.equal(g[..., keep], g0[..., keep]):
            raise RuntimeError("Sender reset changed a gate outside the path's senders")
        if senders and not torch.all(g[..., senders] == 1):
            raise RuntimeError("Sender reset did not set every sender gate to 1")


def path_sender_experiment(model, graph, batch, src, case, dtype, documents, top_k=3, beam_size=10,
                           atol=1e-5, rtol=1e-4, discover=None):
    """Run the whole procedure for one example. Returns (result, gate_archive)."""
    em = model.base.entity_model
    if getattr(em, "resp_proj", None) is None or not em.resp_gate_norm:
        raise RuntimeError("Requires a checkpoint with mean-normalized CCMP heads")
    if getattr(em, "route_mode", "") or getattr(em, "attn_node", None) is not None:
        raise RuntimeError("Disable alternate routing for a CCMP-only experiment")
    if getattr(graph, "dist_context", None) is not None:
        raise RuntimeError("Run on one GPU without graph partitioning")
    if top_k < 1 or beam_size < top_k:
        raise ValueError("Require beam_size >= top_k >= 1")
    model.eval()
    layers = len(em.layers)
    discover = discover or interpret_scientific_paths
    # 1. Discovery under native CCMP; the returned paths are fixed from here on.
    discovery, discovery_gates = discover(model, graph, batch, src, case, dtype, documents,
                                         top_k=top_k, beam_size=beam_size)
    fixed = fixed_paths_from_discovery(discovery, top_k)
    if not fixed:
        raise RuntimeError(f"{case['name']}: no valid simple path from the seeds to the target")
    target, paths, _ = resolve_paths({**case, "paths": fixed}, src, graph, batch, layers, "sender_layer")
    senders = [sender_ids(p) for p in paths]
    num_nodes, device = graph.num_nodes, graph.edge_index.device
    none = torch.zeros((layers, 1, num_nodes), dtype=torch.bool, device=device)
    conditions = [(name, name, none) for name in BASE_CONDITIONS]
    for p, s in zip(paths, senders):
        conditions.append((f"reset_senders_{p['name']}", "outside_full", sender_mask([s], layers, num_nodes, device)))
    if len(paths) > 1:
        conditions.append(("reset_senders_union", "outside_full", sender_mask(senders, layers, num_nodes, device)))
    results, tensors, gate_runs, gates, comps = {}, {}, {}, None, None
    for name, policy, protected in conditions:
        rec, gs, sc, comp = run_condition(model, graph, batch, target, paths, protected, policy, dtype, gates, comps)
        rec["condition"], rec["policy"] = name, policy
        results[name], tensors[name], gate_runs[name] = rec, sc, gs
        if name == "off":
            comps = comp   # the same five off-run non-gold competitors in every condition
        if name == "native":
            gates = gs     # the recorded, normalised gates every later condition replays
        compare_scores({"scorer": tensors["off"]["scorer"]}, {"scorer": sc["scorer"]}, atol, rtol,
                       "Semantic scorer must stay fixed")
        print(f"  {case['name']} / {name}: graph score {rec['channels']['graph']['score']:.6g}, "
              f"graph rank {rec['channels']['graph']['rank']}, fused rank {rec['channels']['fused']['rank']}", flush=True)
    # 3. Frozen replay must apply the exact recorded gates and reproduce the native ranks.
    gate_replay = {
        "raw_bit_identical": all(torch.equal(a, b) for a, b in zip(gate_runs["native"], gate_runs["frozen"])),
        "applied_bit_identical": all(torch.equal(a.to(dtype), b.to(dtype))
                                     for a, b in zip(gate_runs["native"], gate_runs["frozen"])),
        "target_ranks_identical": all(results["native"]["channels"][k]["rank"] == results["frozen"]["channels"][k]["rank"]
                                      for k in ("graph", "fused")),
    }
    if not all(gate_replay.values()):
        raise RuntimeError("Frozen replay did not apply the exact saved native gates")
    for p, s in zip(paths, senders):
        _check_gate_reset(gate_runs[f"reset_senders_{p['name']}"], gates, s, layers)
    if "reset_senders_union" in gate_runs:
        _check_gate_reset(gate_runs["reset_senders_union"], gates, sorted(set().union(*map(set, senders))), layers)
    # Repeat the base conditions to measure run-to-run variation (GPU sparse reductions).
    repeats, repeat_deltas = {}, {}
    for name in BASE_CONDITIONS:
        rec, _, sc, _ = run_condition(model, graph, batch, target, paths, none, name, dtype, gates, comps)
        repeats[name], repeat_deltas[name] = rec, score_deltas(tensors[name], sc)
    native_allowance = numerical_allowance(tensors["native"], dtype, repeat_deltas["native"])
    frozen_allowance = numerical_allowance(tensors["frozen"], dtype, repeat_deltas["frozen"])
    replay_allowance = {k: max(native_allowance[k], frozen_allowance[k]) for k in native_allowance}
    checks = {"gate_replay": gate_replay, "repeat_variation": repeat_deltas,
              "frozen_replay_allowance": replay_allowance,
              "frozen_replay": compare_scores(tensors["native"], tensors["frozen"], atol, rtol,
                                               "Frozen replay", replay_allowance)}
    disc = discovery_gates.get("native_raw") if isinstance(discovery_gates, dict) else None
    if disc is not None and tuple(disc.shape) == tuple(torch.stack(gates).shape):
        checks["discovery_vs_native_gate_max_abs_diff"] = float((disc.cpu().float() - torch.stack(gates).cpu().float()).abs().max())
    # The instrumented forward must agree with plain inference for the two deployable settings.
    old_gate = em.resp_gate
    try:
        for name in ("off", "native"):
            em.resp_gate = name == "native"
            with torch.no_grad(), torch.autocast(device_type=batch["question_embeddings"].device.type,
                                                dtype=dtype, enabled=dtype != torch.float32):
                pred = model(graph, batch)
                plain = {"graph": model._raw_doc[0].float().cpu(), "fused": pred[0, model._doc_ids].float().cpu(),
                         "scorer": model._s_op[0].float().cpu()}
            allowance = numerical_allowance(tensors[name], dtype, repeat_deltas[name])
            checks[name + "_uninstrumented_allowance"] = allowance
            checks[name + "_uninstrumented"] = compare_scores(tensors[name], plain, atol, rtol,
                                                              "Uninstrumented forward", allowance)
    finally:
        em.resp_gate = old_gate
    # 4. Effects. Numerical tolerance = screening only, not a confidence interval.
    native_ch, off_ch = results["native"]["channels"], results["off"]["channels"]
    s_native, f_native = native_ch["graph"]["score"], native_ch["fused"]["score"]
    repeat_graph = max(repeat_deltas[k]["graph"] for k in repeats)
    tol = max(atol, rtol * abs(s_native), 5 * repeat_graph)
    stacked = torch.stack(gates).float().cpu()   # [layers, 1, N]

    def effect(name, sender_set, label):
        c = results[name]["channels"]
        d_graph, d_fused = s_native - c["graph"]["score"], f_native - c["fused"]["score"]
        verdict = "supported" if d_graph > tol else "reduced" if d_graph < -tol else "numerically_unresolved"
        return {"path": label, "condition": name, "n_senders": len(sender_set),
                "senders": [src.id2node[u] for u in sender_set],
                "native_gates_on_senders": {src.id2node[u]: [round(float(stacked[l, 0, u]), 6) for l in range(layers)]
                                            for u in sender_set},
                "graph_score_native": s_native, "graph_score_reset": c["graph"]["score"], "delta_P": d_graph,
                "fused_score_native": f_native, "fused_score_reset": c["fused"]["score"], "delta_P_fused": d_fused,
                "graph_rank_native": native_ch["graph"]["rank"], "graph_rank_reset": c["graph"]["rank"],
                "fused_rank_native": native_ch["fused"]["rank"], "fused_rank_reset": c["fused"]["rank"],
                "numerical_tolerance": tol, "verdict": verdict}

    effects = []
    for i, (p, s) in enumerate(zip(paths, senders)):
        e = effect(f"reset_senders_{p['name']}", s, p["name"])
        e["attribution_native"] = results["native"]["paths"][i]["weight"]
        e["attribution_off"] = results["off"]["paths"][i]["weight"]
        e["attribution_beam"] = next((q["weight"] for q in discovery["top_paths"] if q["name"] == p["name"]), None)
        e["hops"] = [{**{k: h[k] for k in ("layer", "head", "rel", "tail")},
                      "edge_gradient_native": h["edge_gradient"], "g_applied_native": h["g_applied"]}
                     for h in results["native"]["paths"][i]["hops"]]
        effects.append(e)
    union = None
    if "reset_senders_union" in results:
        union = effect("reset_senders_union", sorted(set().union(*map(set, senders))),
                       "+".join(p["name"] for p in paths))
        union["sum_of_individual_delta_P"] = float(sum(e["delta_P"] for e in effects))
    overlap = {f"{a['name']}&{b['name']}": [src.id2node[u] for u in sorted(set(sa) & set(sb))]
               for a, sa in zip(paths, senders) for b, sb in zip(paths, senders) if a["name"] < b["name"]}
    raw_query = next((q for q in src.raw_test_data if str(q["id"]) == case["query_id"]), None) if hasattr(src, "raw_test_data") else None
    result = {"case": case, "version": VERSION, "top_k": top_k, "beam_size": beam_size,
              "question": (raw_query or {}).get("question", ""), "benchmark_stratum": (raw_query or {}).get("stratum"),
              "target_id": case["gold_id"], "target_title": document_title(case["gold_id"], documents),
              "target_domains": graph_domains(target, graph, src),
              "seed_count": int(batch["start_nodes_mask"][0].sum().item()),
              "discovery": {k: discovery.get(k) for k in ("valid_candidate_count", "rejected_count", "rejection_reasons",
                                                          "channels", "gate_stats")},
              "fixed_paths": fixed,
              "paper_titles": {n: document_title(n, documents) for p in fixed for h in p["hops"]
                               for n in (h["head"], h["tail"]) if n in documents},
              "ranks": {"native": native_ch, "off": off_ch},
              "conditions": results, "checks": checks, "repeats": repeats,
              "effects": effects, "union_effect": union, "sender_overlap": overlap,
              "definition": "delta_P = graph score (native CCMP on) - graph score with the path's unique sender gates "
                            "reset to 1 at every layer, all other recorded gates fixed, no renormalisation; a "
                            "path-sender gate effect, not the exclusive contribution of the path; not summable "
                            "across overlapping paths; attribution = mean edge gradient, independent of the gates."}
    for rec in results.values():
        for values in rec["competitors"].values():
            for v in values:
                v["document"] = src.id2node[v["node_id"]]
    archive = {"native_raw": stacked, "sender_masks": {p["name"]: s for p, s in zip(paths, senders)},
               "message_dtypes": [s["message_dtype"] for s in results["native"]["gate_stats"]],
               "paths": paths, "precision": str(dtype)}
    return result, archive


# ----------------------------------------------------------------------------------------------- reporting
def _short(name, titles, limit=70):
    text = titles.get(name, name)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def route_text(hops, titles, tex=False):
    def node(n):
        return tex_escape(_short(n, titles)) if tex else _short(n, titles)
    def rel(r):
        inv = r.startswith("inverse_")
        base = r.removeprefix("inverse_").replace("_", " ")
        return (tex_escape(base) + (r"$^{-1}$" if inv else "")) if tex else base + ("⁻¹" if inv else "")
    parts = [node(hops[0]["head"])]
    for h in hops:
        parts += [f"[{rel(h['rel'])}]", node(h["tail"])]
    return (r" $\rightarrow$ " if tex else " → ").join(parts)


def _fmt(x, digits=3):
    return "n/a" if x is None else f"{x:+.{digits}f}" if isinstance(x, float) else str(x)


def render_path_sender_report(payload, out_dir):
    """report.md (one compact table per example), tables.tex (booktabs), summary.csv, hops.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = payload.get("manifest", {})
    selection = manifest.get("selection") or {}
    md = ["# CCMP path-sender gate effect", "",
          f"Checkpoint: `{selection.get('checkpoint', manifest.get('checkpoint', ''))}`; graph: `{selection.get('graph', '')}`; "
          f"precision: {manifest.get('precision', '')}; run {payload.get('run_id', '')[:12]}.", "",
          "Paths are the top-K simple paths of the NBFNet gradient beam search under native CCMP, fixed for every "
          "condition. Attribution is the mean edge gradient along the path (GFM-RAG convention), not a product of gates. "
          "Delta_P = native graph score minus the graph score after resetting the path's unique sender gates to 1 at every "
          "layer with all other recorded gates fixed (no renormalisation). Delta_P is a path-sender gate effect, not the "
          "path's exclusive contribution; overlapping paths must not be summed. A higher target score does not imply a "
          "better rank; ranks are listed separately. Tolerance is a numerical screen, not a confidence interval.", ""]
    tex = [r"% booktabs required. One table per example. Inverse relations (r^{-1}) traverse an edge backwards.",
           r"% Delta_P: path-sender gate effect (native graph score minus score after resetting the path's sender gates to 1)."]
    summary, hops = [], []
    for rec in payload["results"]:
        case, titles = rec["case"], rec["paper_titles"]
        r_on, r_off = rec["ranks"]["native"], rec["ranks"]["off"]
        chk = rec["checks"]
        replay_ok = all(chk["gate_replay"].values())
        md += [f"## {case['name']}", "",
               f"**Query** ({case['query_id']}, stratum {rec.get('benchmark_stratum')}): {rec['question'][:400]}…", "",
               f"**Target paper**: {rec['target_title']} (`{rec['target_id']}`; graph domain: {', '.join(rec['target_domains']) or 'unknown'})", "",
               f"**Target graph rank** CCMP on {r_on['graph']['rank']} / all gates 1 {r_off['graph']['rank']} "
               f"(graph score {r_on['graph']['score']:.4f} / {r_off['graph']['score']:.4f}); "
               f"fused rank on {r_on['fused']['rank']} / off {r_off['fused']['rank']}.", "",
               f"Frozen replay of the recorded gates reproduces the native run: {'yes' if replay_ok else 'NO'} "
               f"(max |Δscore| graph {chk['frozen_replay']['graph']:.2e}, allowance {chk['frozen_replay_allowance']['graph']:.2e}). "
               f"Seeds: {rec['seed_count']}; valid simple beam candidates: {rec['discovery']['valid_candidate_count']}.", "",
               "| Path | Route (seed → … → target) | Attribution (mean edge gradient) | Δ_P (graph score) | Graph rank after reset | Verdict |",
               "|---|---|---:|---:|---:|---|"]
        for e in rec["effects"]:
            md.append(f"| {e['path']} | {route_text(e['hops'], titles)} | {e['attribution_native']:.3f} | {e['delta_P']:+.3f} | "
                      f"{e['graph_rank_reset']} | {e['verdict']} |")
            summary.append({"case": case["name"], "query_id": case["query_id"], "target_id": rec["target_id"],
                            "precision": manifest.get("precision", ""), "path": e["path"], "n_hops": len(e["hops"]),
                            "n_senders": e["n_senders"], "attribution_native": e["attribution_native"],
                            "attribution_off": e["attribution_off"], "delta_P": e["delta_P"], "delta_P_fused": e["delta_P_fused"],
                            "graph_score_native": e["graph_score_native"], "graph_score_reset": e["graph_score_reset"],
                            "graph_rank_native": e["graph_rank_native"], "graph_rank_reset": e["graph_rank_reset"],
                            "graph_rank_off": r_off["graph"]["rank"], "fused_rank_native": e["fused_rank_native"],
                            "fused_rank_reset": e["fused_rank_reset"], "fused_rank_off": r_off["fused"]["rank"],
                            "tolerance": e["numerical_tolerance"], "verdict": e["verdict"]})
            for h in e["hops"]:
                hops.append({"case": case["name"], "path": e["path"], **h,
                             "native_gates_all_layers": json.dumps(e["native_gates_on_senders"].get(h["head"]))})
        u = rec.get("union_effect")
        if u:
            md += ["", f"All {len(rec['effects'])} paths' senders reset together ({u['n_senders']} nodes): Δ = {u['delta_P']:+.3f} "
                   f"(graph rank {u['graph_rank_reset']}); sum of the individual Δ_P would be {u['sum_of_individual_delta_P']:+.3f}, "
                   "which is not a valid estimate because senders are shared: "
                   + "; ".join(f"{k}: {', '.join(v) if v else 'none'}" for k, v in rec["sender_overlap"].items()) + "."]
        md += ["", f"Numerical tolerance for Δ_P: {rec['effects'][0]['numerical_tolerance']:.2e} "
               f"(max repeat variation of the graph score {max(v['graph'] for v in chk['repeat_variation'].values()):.2e}).", "",
               "Native gates on each path's senders, all layers: "
               + "; ".join(f"{e['path']}: " + ", ".join(f"{s} {g}" for s, g in e["native_gates_on_senders"].items()) for e in rec["effects"]), ""]
        tex += [r"\begin{table}[t]\centering\small", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
                r"Path & Route & Attribution & $\Delta_P$ & Rank after reset \\", r"\midrule"]
        for e in rec["effects"]:
            tex.append(f"{e['path']} & {route_text(e['hops'], titles, tex=True)} & {e['attribution_native']:.2f} & "
                       f"{e['delta_P']:+.2f} & {e['graph_rank_reset']} \\\\")
        tex += [r"\bottomrule", r"\end{tabular}",
                r"\caption{" + tex_escape(case["name"]) + ": query " + tex_escape(case["query_id"]) + r" $\rightarrow$ "
                + tex_escape(rec["target_title"]) + f". Target graph rank {r_on['graph']['rank']} with CCMP on, "
                f"{r_off['graph']['rank']} with all gates set to 1 (graph score {r_on['graph']['score']:.2f} vs "
                f"{r_off['graph']['score']:.2f}). $\\Delta_P$ is the path-sender gate effect (native score minus the score "
                "after resetting the path's sender gates to 1 at every layer, other gates fixed). Attribution is the mean "
                "edge gradient of the path. Precision " + tex_escape(manifest.get("precision", "")) + ".}",
                r"\label{tab:ccmp-path-sender-" + case["name"].replace("_", "-") + "}", r"\end{table}", ""]
    (out / "report.md").write_text("\n".join(md) + "\n")
    (out / "tables.tex").write_text("\n".join(tex) + "\n")
    for name, rows in (("summary.csv", summary), ("hops.csv", hops)):
        if rows:
            with (out / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    atomic_json(out / "fixed_paths.json", {r["case"]["name"]: r["fixed_paths"] for r in payload["results"]})


if __name__ == "__main__":   # local re-rendering: python3 eval/ccmp_path_sender_effect.py results.json out_dir
    import sys
    payload = json.loads(Path(sys.argv[1]).read_text())
    render_path_sender_report(payload, sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).parent)
    print("rendered", sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).parent)
