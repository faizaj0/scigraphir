"""Matched, fixed-path CCMP interventions. Embedded in colab_table4_openie.ipynb.

The intervention is applied AFTER native gate normalization and BEFORE the dtype
cast / outgoing-message multiplication. Counterfactual gates are frozen from the
native-on pass. No counterfactual is renormalized or recomputed from its new states.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

VERSION = "ccmp-mechanism-v1"
CONDITIONS = (
    "off", "native", "frozen", "path_only", "outside_full",
    "outside_suppress", "outside_amplify",
)


def patch_engine(path):
    """Install an opt-in hook in the single-process branch; ordinary runs unchanged."""
    path = Path(path)
    text = path.read_text()
    marker = "# CCMP_MECHANISM_HOOK_V1"
    if marker in text:
        return
    anchor = """                else:
                    _msg = layer_input
                if separate_grad:
                    edge_weight = edge_weight.clone().requires_grad_()
"""
    replacement = """                else:
                    _msg = layer_input
                # CCMP_MECHANISM_HOOK_V1: analysis-only, after normalization.
                _hook = getattr(self, "_ccmp_intervention", None)
                if _hook is not None:
                    _native_gate = (_gt if _heads is not None and self.resp_gate
                                    else torch.ones_like(layer_input[..., 0]).float())
                    _chosen_gate = _hook(_li, _native_gate, layer_input,
                                         self._reach_layers[_li])
                    _msg = layer_input * _chosen_gate.unsqueeze(-1).to(layer_input.dtype)
                if separate_grad:
                    edge_weight = edge_weight.clone().requires_grad_()
"""
    if text.count(anchor) != 1:
        raise RuntimeError("Engine changed: expected exactly one CCMP intervention anchor")
    text = text.replace(anchor, replacement)
    compile(text, str(path), "exec")
    path.write_text(text)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def choose_gate(condition, baseline, protected):
    """Pure counterfactual policy. No normalization after intervention."""
    base = baseline.detach()
    one = torch.ones_like(base)
    if condition == "off":
        return one
    if condition == "frozen":
        return base
    if condition == "path_only":
        return torch.where(protected, base, one)
    if condition == "outside_full":
        return torch.where(protected, one, base)
    if condition == "outside_suppress":
        return torch.where(protected, one, base.clamp(max=1))
    if condition == "outside_amplify":
        return torch.where(protected, one, base.clamp(min=1))
    raise ValueError(condition)


def inverse_relation_id(r, num_relations):
    """Inverse relations fill the second half of the relation table (graph_index_dataset: r + R/2)."""
    if num_relations % 2:
        raise ValueError("Relation table must pair every relation with its inverse")
    return (int(r) + num_relations // 2) % num_relations


def resolve_paths(case, src, graph, batch, layers, protection):
    """Exact strings and edge IDs; fail on missing/ambiguous edges or wrong targets.

    A hop (head, rel, tail) is the KG fact read from the sending node to the receiving node. The
    rspmm kernel that training and inference run receives at edge_index[0] and sends from
    edge_index[1] (layers.py propagate), so the message head -> tail travels on the edge with
    edge_index[0] = tail, edge_index[1] = head and the INVERSE relation of the displayed fact;
    that edge id is what the edge gradients and the sender gates refer to.
    """
    node2id, rel2id = src.node2id, src.rel2id
    target = node2id[case["gold_id"]]
    if not bool(batch["target_nodes_mask"][0, target] > 0):
        raise ValueError(f"Pinned document is not a gold: {case['gold_id']}")
    paths = []
    protected = torch.zeros((layers, 1, graph.num_nodes), dtype=torch.bool,
                            device=graph.edge_index.device)
    for p in case["paths"]:
        hops, nodes = [], []
        for i, hop in enumerate(p["hops"]):
            if hop["layer"] != i or i >= layers:
                raise ValueError("Fixed paths must start at layer 0 and fit the model depth")
            h, t, r = node2id[hop["head"]], node2id[hop["tail"]], rel2id[hop["rel"]]
            if i == 0:
                if not bool(batch["start_nodes_mask"][0, h] > 0):
                    raise ValueError(f"Path does not start at a seed: {hop['head']}")
                nodes.append(h)
            elif nodes[-1] != h:
                raise ValueError("Disconnected fixed path")
            nodes.append(t)
            match = ((graph.edge_index[0] == t) & (graph.edge_index[1] == h)
                     & (graph.edge_type == inverse_relation_id(r, len(rel2id)))).nonzero(as_tuple=True)[0]
            if match.numel() != 1:
                raise ValueError(f"Expected one exact edge, found {match.numel()}: {hop}")
            hops.append({**hop, "head_id": h, "tail_id": t, "rel_id": r,
                         "edge_id": int(match[0])})
            protected[i, 0, h] = True
        if not hops or nodes[-1] != target or len(nodes) != len(set(nodes)):
            raise ValueError("Path must be simple and end at the exact pinned gold")
        if protection == "nodes_all_layers":
            protected[:, 0, nodes] = True
        elif protection != "sender_layer":
            raise ValueError(protection)
        paths.append({"name": p["name"], "hops": hops})
    if not paths:
        raise ValueError("At least one fixed path is required")
    return target, paths, protected


def gate_stats(gate, reached):
    v = gate.detach().float()[reached.bool()]
    if not v.numel():
        return {"n": 0}
    return {"n": v.numel(), "mean": float(v.mean()), "min": float(v.min()),
            "max": float(v.max()), "suppressed": int((v < 1).sum()),
            "amplified": int((v > 1).sum()), "exactly_one": int((v == 1).sum()),
            "suppression_mass": float((1 - v).clamp(min=0).sum()),
            "amplification_mass": float((v - 1).clamp(min=0).sum())}


class GateRecorder:
    def __init__(self, condition, protected, baseline=None):
        self.condition, self.protected, self.baseline = condition, protected, baseline
        self.gates, self.stats = [], []

    def __call__(self, layer, native_gate, layer_input, reached):
        if reached is None or layer != len(self.gates):
            raise RuntimeError("Expected ordered single-query gates with structural reach")
        if self.condition == "native":
            gate = native_gate                 # retain derivatives through the gate
        else:
            base = native_gate if self.baseline is None else self.baseline[layer]
            gate = choose_gate(self.condition, base, self.protected[layer])
        if not torch.isfinite(gate).all():
            raise RuntimeError("Non-finite gate")
        applied = gate.to(layer_input.dtype).float()
        if self.condition.startswith("outside_"):
            if not torch.all(applied[self.protected[layer]] == 1):
                raise RuntimeError("Protected path gate changed")
        self.gates.append(gate.detach().clone())
        self.stats.append({"layer": layer, "message_dtype": str(layer_input.dtype),
                           "raw": gate_stats(gate, reached),
                           "applied": gate_stats(applied, reached),
                           "outside_applied": gate_stats(applied, reached.bool() & ~self.protected[layer]),
                           "protected_count": int(self.protected[layer].sum())})
        return gate


def score_metrics(scores, j, gold_mask):
    v = scores.detach().float()
    if not torch.isfinite(v).all():
        raise RuntimeError("Non-finite document scores")
    nongold = v[~gold_mask]
    return {"score": float(v[j]), "rank": int((v > v[j]).sum()) + 1,
            "ties": int((v == v[j]).sum()),
            "margin_to_best_nongold": float(v[j] - nongold.max()) if nongold.numel() else None}


def run_condition(model, graph, batch, target, paths, protected, condition,
                  dtype, baseline=None, competitors=None, diagnostics=None):
    """Use the true fusion forward, with the legacy visualize edge-gradient convention."""
    em = model.base.entity_model
    recorder = GateRecorder(condition, protected, baseline)
    original_bf = em.bellmanford
    old_gate, old_keep = em.resp_gate, getattr(em, "_keep_reach", False)
    if getattr(em, "_ccmp_intervention", None) is not None:
        raise RuntimeError("Another intervention is already installed")
    captured = {}

    def capture(*args, **kwargs):
        kwargs["separate_grad"] = True
        out = original_bf(*args, **kwargs)
        captured["edge_weights"] = out["edge_weights"]
        return out

    em.bellmanford = capture
    em._ccmp_intervention, em._keep_reach = recorder, True
    em.resp_gate = condition != "off"
    try:
        with torch.enable_grad(), torch.autocast(device_type=batch["question_embeddings"].device.type,
                                               dtype=dtype, enabled=dtype != torch.float32):
            pred = model(graph, batch)
            did = model._doc_ids
            js = (did == target).nonzero(as_tuple=True)[0]
            if js.numel() != 1:
                raise RuntimeError("Pinned gold is not exactly one document node")
            j = int(js[0])
            scores = {"graph": model._raw_doc[0].float(), "fused": pred[0, did].float()}
            edge_grads = torch.autograd.grad(scores["graph"][j], captured["edge_weights"])
        if diagnostics is not None:
            diagnostics.update(edge_grads=tuple(g.detach() for g in edge_grads),
                               doc_ids=did.detach(), target_position=j)
        if len(recorder.gates) != len(em.layers):
            raise RuntimeError("Hook did not record every layer; check engine patch")
        scores = {k: v.detach() for k, v in scores.items()}
        gm = batch["target_nodes_mask"][0, did] > 0
        result = {"condition": condition, "channels": {k: score_metrics(v, j, gm) for k, v in scores.items()},
                  "gate_stats": recorder.stats, "paths": []}
        for p in paths:
            hops = []
            for hop in p["hops"]:
                l, h, e = hop["layer"], hop["head_id"], hop["edge_id"]
                raw = recorder.gates[l][0, h]
                message_dtype = getattr(torch, recorder.stats[l]["message_dtype"].split(".")[-1])
                hops.append({**hop, "edge_gradient": float(edge_grads[l][e]),
                             "g_raw": float(raw), "g_applied": float(raw.to(message_dtype)),
                             "protected": bool(protected[l, 0, h])})
            result["paths"].append({"name": p["name"], "weight": sum(h["edge_gradient"] for h in hops) / len(hops),
                                    "hops": hops})
        if competitors is None:
            competitors = {}
            for k, v in scores.items():
                ix = (~gm).nonzero(as_tuple=True)[0]
                competitors[k] = ix[v[ix].topk(min(5, ix.numel())).indices].tolist()
        result["competitors"] = {
            k: [{"node_id": int(did[ix]), "score": float(v[ix]), "gold_margin": float(v[j] - v[ix])}
                for ix in competitors[k]] for k, v in scores.items()}
        detached = {k: v.detach().float().cpu().clone() for k, v in scores.items()}
        detached["scorer"] = model._s_op[0].detach().float().cpu().clone()
        return result, recorder.gates, detached, competitors
    finally:
        em.bellmanford = original_bf
        em.resp_gate, em._keep_reach = old_gate, old_keep
        del em._ccmp_intervention
        em._resp_pred, em._reach_layers = [], []


def score_deltas(a, b):
    return {k: float((a[k] - b[k]).abs().max()) for k in a}


def numerical_allowance(reference, dtype, repeat_error=None):
    """Absolute allowance for repeated GPU reductions at the requested dtype.

    The scorer is a separate float32 channel. Graph/fused scores inherit the GNN
    arithmetic dtype; bf16 sparse reductions need an allowance proportional to
    their scale even when the applied gates are bit-identical.
    """
    repeat_error = repeat_error or {}
    allowance = {}
    for k, value in reference.items():
        scale = max(1.0, float(value.detach().abs().max()))
        roundoff = 0.0 if dtype == torch.float32 or k == "scorer" else 2 * torch.finfo(dtype).eps * scale
        allowance[k] = max(5 * float(repeat_error.get(k, 0.0)), roundoff)
    return allowance


def compare_scores(a, b, atol, rtol, label, extra_atol=None):
    checks = {}
    extra_atol = extra_atol or {}
    for k in a:
        delta = float((a[k] - b[k]).abs().max())
        checks[k] = delta
        allowance = float(extra_atol.get(k, 0.0))
        if not torch.allclose(a[k], b[k], atol=atol + allowance, rtol=rtol):
            raise RuntimeError(f"{label}: {k} score mismatch, max absolute error {delta}; "
                               f"measured/dtype allowance {allowance}")
    return checks


def experiment(model, graph, batch, src, case, dtype, protection="nodes_all_layers",
               atol=1e-5, rtol=1e-4):
    em = model.base.entity_model
    if getattr(em, "resp_proj", None) is None or not em.resp_gate_norm:
        raise RuntimeError("Requires a checkpoint with mean-normalized CCMP heads")
    if getattr(em, "route_mode", "") or getattr(em, "attn_node", None) is not None:
        raise RuntimeError("Disable alternate routing for a CCMP-only experiment")
    if getattr(graph, "dist_context", None) is not None:
        raise RuntimeError("Run on one GPU without graph partitioning")
    model.eval()
    target, paths, protected = resolve_paths(case, src, graph, batch, len(em.layers), protection)
    results, tensors, gate_runs, gates, comps = {}, {}, {}, None, None
    # Native off remains the actual ungated code path (plus identity hook).
    for condition in CONDITIONS:
        rec, gs, sc, comp = run_condition(model, graph, batch, target, paths, protected,
                                         condition, dtype, gates, comps)
        results[condition], tensors[condition], gate_runs[condition] = rec, sc, gs
        if condition == "off":
            comps = comp  # same five off-run non-gold competitors in all conditions
        if condition == "native":
            gates = gs
        compare_scores({"scorer": tensors["off"]["scorer"]}, {"scorer": sc["scorer"]},
                       atol, rtol, "Semantic scorer must stay fixed")
        print(f"  {case['name']} / {condition}: graph rank {rec['channels']['graph']['rank']}, "
              f"w={[round(p['weight'], 6) for p in rec['paths']]}", flush=True)
    # The causal replay is defined by its applied gates. Check these exactly before
    # allowing for nondeterministic low-precision sparse reductions in its scores.
    gate_replay = {
        "raw_bit_identical": all(torch.equal(a, b) for a, b in zip(gate_runs["native"], gate_runs["frozen"])),
        "applied_bit_identical": all(torch.equal(a.to(dtype), b.to(dtype))
                                     for a, b in zip(gate_runs["native"], gate_runs["frozen"])),
        "target_ranks_identical": all(results["native"]["channels"][k]["rank"]
                                      == results["frozen"]["channels"][k]["rank"]
                                      for k in ("graph", "fused")),
    }
    if not all(gate_replay.values()):
        raise RuntimeError("Frozen replay did not apply the exact saved native gates")
    repeats, repeat_tensors, repeat_deltas = {}, {}, {}
    for condition in ("off", "native", "frozen"):
        rec, _, sc, _ = run_condition(model, graph, batch, target, paths, protected, condition,
                                      dtype, gates, comps)
        repeats[condition], repeat_tensors[condition] = rec, sc
        repeat_deltas[condition] = score_deltas(tensors[condition], sc)
    native_allowance = numerical_allowance(tensors["native"], dtype, repeat_deltas["native"])
    frozen_allowance = numerical_allowance(tensors["frozen"], dtype, repeat_deltas["frozen"])
    replay_allowance = {k: max(native_allowance[k], frozen_allowance[k]) for k in native_allowance}
    checks = {"gate_replay": gate_replay, "repeat_variation": repeat_deltas,
              "frozen_replay_allowance": replay_allowance,
              "frozen_replay": compare_scores(tensors["native"], tensors["frozen"], atol, rtol,
                                               "Frozen replay", replay_allowance)}
    # Verify that instrumentation / separate_grad did not change deployed forward scores.
    old_gate = em.resp_gate
    try:
        for condition in ("off", "native"):
            em.resp_gate = condition == "native"
            with torch.no_grad(), torch.autocast(device_type=batch["question_embeddings"].device.type,
                                                dtype=dtype, enabled=dtype != torch.float32):
                pred = model(graph, batch)
                plain = {"graph": model._raw_doc[0].float().cpu(),
                         "fused": pred[0, model._doc_ids].float().cpu(),
                         "scorer": model._s_op[0].float().cpu()}
            allowance = numerical_allowance(tensors[condition], dtype, repeat_deltas[condition])
            checks[condition + "_uninstrumented_allowance"] = allowance
            checks[condition + "_uninstrumented"] = compare_scores(
                tensors[condition], plain, atol, rtol, "Uninstrumented forward", allowance)
    finally:
        em.resp_gate = old_gate
    effects = []
    for i, p in enumerate(paths):
        w = {k: r["paths"][i]["weight"] for k, r in results.items()}
        if not all(math.isfinite(v) for v in w.values()):
            raise RuntimeError(f"Non-finite path attribution: {p['name']}")
        repeat_error = max(abs(repeats[k]["paths"][i]["weight"] - w[k]) for k in repeats)
        # Numerical screening only; this is NOT a confidence interval or hypothesis test.
        tol = max(atol, rtol * max(abs(x) for x in w.values()), 5 * repeat_error)
        suppression = w["outside_suppress"] - w["off"]
        n_suppressed = sum(s["outside_applied"].get("suppressed", 0)
                           for s in results["outside_suppress"]["gate_stats"])
        verdict = ("no_effective_suppression" if n_suppressed == 0 else
                   "increases" if suppression > tol else
                   "decreases" if suppression < -tol else "numerically_unresolved")
        effects.append({"path": p["name"], "weights": w, "numerical_tolerance": tol,
                        "repeat_attribution_error": repeat_error,
                        "native_minus_off": w["native"] - w["off"],
                        "frozen_minus_off": w["frozen"] - w["off"],
                        "gate_derivative_effect": w["native"] - w["frozen"],
                        "path_only_effect": w["path_only"] - w["off"],
                        "outside_full_effect": w["outside_full"] - w["off"],
                        "outside_suppression_effect": suppression,
                        "outside_amplification_effect": w["outside_amplify"] - w["off"],
                        "outside_suppression_verdict": verdict,
                        "path_outside_interaction": w["frozen"] - w["path_only"] - w["outside_full"] + w["off"],
                        "suppression_amplification_interaction": w["outside_full"] - w["outside_suppress"] - w["outside_amplify"] + w["off"]})
    for rec in results.values():
        for values in rec["competitors"].values():
            for v in values:
                v["document"] = src.id2node[v["node_id"]]
    historical = {}
    refs = case.get("historical_reference", {}) if case.get("historical_reference_applicable", True) else {}
    for condition, ref in refs.items():
        rec = results[condition]
        delta = {p["name"]: p["weight"] - ref["path_weights"][p["name"]] for p in rec["paths"]}
        historical[condition] = {
            "path_weight_deltas": delta,
            "weights_match_tolerance": all(math.isclose(p["weight"], ref["path_weights"][p["name"]],
                                                        abs_tol=atol, rel_tol=rtol) for p in rec["paths"]),
            "rank_deltas": {k: rec["channels"][k]["rank"] - ref["ranks"][k] for k in ("graph", "fused")}}
    # CPU gate tensors are returned separately for reproducibility; never used as old-run cache inputs.
    gate_archive = {"native_raw": torch.stack(gates).cpu(), "protected": protected.cpu(),
                    "message_dtypes": [s["message_dtype"] for s in results["native"]["gate_stats"]],
                    "paths": paths, "precision": str(dtype)}
    return {"case": case, "protection": protection, "conditions": results,
            "checks": checks, "repeats": repeats, "effects": effects,
            "historical_comparison": historical}, gate_archive


def render_report(payload, out_dir):
    """Only current, hash-validated measurements enter the table and TikZ data files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# CCMP fixed-path intervention results", "",
             "All conditions use the same checkpoint, graph, query, gold and fixed paths. "
             "Positive Δw means increased gradient attribution, not necessarily improved ranking.", "",
             "Outside means outside the protected node set (all layers by default). Gates are frozen "
             "from native CCMP, then clamped without renormalization. These selected cases do not estimate "
             "a population effect. Frozen replay requires bit-identical applied gates. Score checks include "
             "measured repeat variation and a recorded bfloat16 scale allowance; these numerical tolerances "
             "are not statistical confidence intervals.", ""]
    selection = payload["manifest"].get("selection") or {}
    if selection:
        lines += [f"Graph: `{selection['graph']}`. Checkpoint family: **{selection['family']}**.", "",
                  f"Checkpoint: `{selection['checkpoint']}`.", "", selection["reason"], ""]
    rows, hops, effects = [], [], []
    for record in payload["results"]:
        case = record["case"]
        lines += [f"## {case['name']} ({payload['manifest']['precision']})", "",
                  f"Protection: `{record['protection']}`. Gold: `{case['gold_id']}`.", "",
                  "| Condition | Graph rank | Fused rank | Graph score | " + " | ".join(p["name"] + " w" for p in case["paths"]) + " |",
                  "|---|---:|---:|---:|" + "---:|" * len(case["paths"])]
        for condition, r in record["conditions"].items():
            ch = r["channels"]
            lines.append(f"| {condition} | {ch['graph']['rank']} | {ch['fused']['rank']} | {ch['graph']['score']:.6g} | "
                         + " | ".join(f"{p['weight']:.6g}" for p in r["paths"]) + " |")
            for p in r["paths"]:
                common = {"case": case["name"], "query_id": case["query_id"], "gold_id": case["gold_id"],
                          "graph": selection.get("graph", ""), "checkpoint_family": selection.get("family", ""),
                          "precision": payload["manifest"]["precision"],
                          "condition": condition, "path": p["name"], "weight": p["weight"]}
                rows.append({**common, **{f"{k}_{m}": v for k, ms in ch.items() for m, v in ms.items()}})
                hops.extend({**common, **h} for h in p["hops"])
        lines += ["", "| Path | Native Δw | Gate derivative Δw | Outside suppression Δw | Suppression result |",
                  "|---|---:|---:|---:|---|"]
        for e in record["effects"]:
            lines.append(f"| {e['path']} | {e['native_minus_off']:.6g} | {e['gate_derivative_effect']:.6g} | "
                         f"{e['outside_suppression_effect']:.6g} | {e['outside_suppression_verdict']} |")
            effects.append({"case": case["name"], **{k: v for k, v in e.items() if k != "weights"}})
        lines += ["", "An increase under outside_suppress supports suppression being sufficient to raise this path's "
                  "attribution under this intervention. It does not show that all other messages were irrelevant, "
                  "or that suppression fully explains native CCMP. Native minus frozen isolates differentiation "
                  "through gates at a matched forward pass. Inspect ranks and gold margins separately.", ""]
        if record.get("historical_comparison"):
            history = record["historical_comparison"]
            matched = all(v["weights_match_tolerance"] and all(d == 0 for d in v["rank_deltas"].values())
                          for v in history.values())
            lines += ["Historical figure check: " + ("off/on weights and graph/fused ranks reproduce at the configured tolerance."
                       if matched else "the old figure's weights or ranks do not reproduce exactly at this precision. "
                       "Use these new matched measurements; inspect historical_comparison in results.json before reusing old percentages."), ""]
        elif case.get("historical_reference_applicable") is False:
            lines += ["The old figure used a frame-only checkpoint/graph. Its weights are retained as provenance, "
                      "not used as a numerical reproduction target for this merged-graph experiment.", ""]
    (out / "report.md").write_text("\n".join(lines))
    for name, data in (("paths.csv", rows), ("hops.csv", hops), ("effects.csv", effects)):
        if data:
            with (out / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    # Simple numeric TSV that pgfplotstable / TikZ can consume without escaped document strings.
    with (out / "tikz_values.tsv").open("w") as f:
        f.write("case\tcondition\tpath\thop\tw\tgraw\tgapplied\n")
        for ci, record in enumerate(payload["results"]):
            for vi, condition in enumerate(CONDITIONS):
                for pi, p in enumerate(record["conditions"][condition]["paths"]):
                    for hi, h in enumerate(p["hops"]):
                        f.write(f"{ci}\t{vi}\t{pi}\t{hi}\t{p['weight']:.9g}\t{h['g_raw']:.9g}\t{h['g_applied']:.9g}\n")
    atomic_json(out / "tikz_index.json", {"cases": [r["case"]["name"] for r in payload["results"]],
                                         "conditions": CONDITIONS,
                                         "paths": [[p["name"] for p in r["case"]["paths"]] for r in payload["results"]]})
