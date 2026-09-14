"""CPU smoke tests: execute the patched REAL bellmanford with tiny differentiable layers.

These test intervention semantics, not the unavailable Drive checkpoints / CUDA kernels.
Run: python3 -m unittest discover -s experiments/eval -p test_ccmp_mechanism.py -v
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import ccmp_mechanism as core

from ccmp_mechanism import (
    CONDITIONS, GateRecorder, choose_gate, experiment, patch_engine, render_report,
    resolve_paths, run_condition, compare_scores, numerical_allowance,
)

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "retriever/gfm-rag/gfmrag/models/ultra/models.py"


def actual_bellmanford():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "models.py"
        p.write_text(ENGINE.read_text())
        patch_engine(p)
        first = p.read_text()
        patch_engine(p)
        assert p.read_text() == first
        tree = ast.parse(first)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QueryNBFNet")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "bellmanford")
    scope = {"torch": torch, "os": os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ENGINE), "exec"), scope)
    return scope["bellmanford"]


class TinyLayer(nn.Module):
    """Same direction as the rspmm kernel: receive at edge_index[0], send from edge_index[1]."""
    def forward(self, message, query, boundary, edge_index, edge_type, size, weight):
        send = message[:, edge_index[1]] * weight[None, :, None]
        sums = torch.zeros_like(message).index_add(1, edge_index[0], send)
        return torch.tanh(0.5 * sums + 0.2 * boundary)


class TinyEntity(nn.Module):
    bellmanford = actual_bellmanford()

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer(), TinyLayer()])
        self.resp_proj = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in self.layers])
        for m in self.resp_proj:
            nn.init.constant_(m.weight, -1.5)
        self.resp_emb = nn.Embedding(3, 1)
        nn.init.zeros_(self.resp_emb.weight)
        self.resp_head = nn.Linear(1, 1, bias=False)
        nn.init.ones_(self.resp_head.weight)
        self.resp_gate, self.resp_gate_norm, self.resp_eta = True, True, 0.5
        self.short_cut, self.concat_hidden = True, True


class TinyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.entity_model = TinyEntity()
        self.register_buffer("initial", torch.tensor([[[1.0], [0.3], [0.0], [0.1], [2.0]]]))

    def forward(self, graph, batch):
        em = self.base.entity_model
        em._ccmp_seeds = batch["start_nodes_mask"]
        out = em.bellmanford(graph, self.initial, batch["question_embeddings"])
        raw = out["node_feature"][..., :-1].sum(-1)
        self._doc_ids = torch.tensor([2, 3])
        self._raw_doc = raw[:, self._doc_ids]
        self._s_op = torch.tensor([[0.2, 0.4]])
        pred = raw.clone()
        pred[:, self._doc_ids] = self._s_op + 0.1 * self._raw_doc
        return pred


def fixture():
    # Message edges in kernel convention (receiver = edge_index[0], sender = edge_index[1]), all typed
    # "inverse_rel", so the displayed facts (sender, "rel", receiver) are seed->method, method->gold,
    # outside->method, outside->competitor, seed->outside.
    graph = SimpleNamespace(num_nodes=5, num_edges=5,
                            edge_index=torch.tensor([[1, 2, 1, 3, 4], [0, 1, 4, 4, 0]]),
                            edge_type=torch.ones(5, dtype=torch.long))
    names = ["seed", "method", "gold", "competitor", "outside"]
    src = SimpleNamespace(node2id={n: i for i, n in enumerate(names)}, id2node=dict(enumerate(names)),
                          rel2id={"rel": 0, "inverse_rel": 1})
    batch = {"question_embeddings": torch.tensor([[0.3]]), "start_nodes_mask": torch.tensor([[1, 0, 0, 0, 1]]),
             "target_nodes_mask": torch.tensor([[0, 0, 1, 0, 0]]), "id": ["query"]}
    case = {"name": "toy", "query_id": "query", "gold_id": "gold", "paths": [{"name": "P1", "hops": [
        {"layer": 0, "head": "seed", "rel": "rel", "tail": "method"},
        {"layer": 1, "head": "method", "rel": "rel", "tail": "gold"}]}]}
    return TinyFusion(), graph, src, batch, case


class MechanismTest(unittest.TestCase):
    def test_after_normalization_policies_do_not_rescale_protected_nodes(self):
        baseline = torch.tensor([[1.2, 0.6, 1.4]], requires_grad=True)
        mask = torch.tensor([[True, False, False]])
        self.assertTrue(torch.equal(choose_gate("outside_suppress", baseline, mask), torch.tensor([[1., .6, 1.]])))
        self.assertTrue(torch.equal(choose_gate("outside_amplify", baseline, mask), torch.tensor([[1., 1., 1.4]])))
        self.assertFalse(choose_gate("frozen", baseline, mask).requires_grad)
        self.assertTrue(torch.equal(choose_gate("path_only", baseline, mask), torch.tensor([[1.2, 1., 1.]])))

    def test_applied_bfloat16_gate_can_be_exactly_one(self):
        raw = torch.tensor([[0.9999, 1.0015, 0.8]])
        record = GateRecorder("native", torch.zeros((1, 1, 3), dtype=torch.bool))
        record(0, raw, torch.ones((1, 3, 1), dtype=torch.bfloat16), torch.ones_like(raw))
        self.assertEqual(record.stats[0]["raw"]["exactly_one"], 0)
        self.assertEqual(record.stats[0]["applied"]["exactly_one"], 2)

    def test_bfloat16_score_replay_uses_a_recorded_scale_allowance(self):
        reference = {"graph": torch.tensor([8.0, 2.0]), "fused": torch.tensor([4.0]),
                     "scorer": torch.tensor([3.0])}
        allowance = numerical_allowance(reference, torch.bfloat16)
        self.assertEqual(allowance["graph"], 0.125)
        self.assertEqual(allowance["scorer"], 0)
        changed = {"graph": torch.tensor([8.125, 2.0]), "fused": torch.tensor([4.0]),
                   "scorer": torch.tensor([3.0])}
        self.assertEqual(compare_scores(reference, changed, 1e-5, 1e-4, "bf16 replay", allowance)["graph"], 0.125)
        changed["graph"] = torch.tensor([8.5, 2.0])
        with self.assertRaisesRegex(RuntimeError, "measured/dtype allowance"):
            compare_scores(reference, changed, 1e-5, 1e-4, "bf16 replay", allowance)

    def test_real_bellmanford_matched_interventions_and_derivative_control(self):
        model, graph, src, batch, case = fixture()
        result, gates = experiment(model, graph, batch, src, case, torch.float32)
        self.assertEqual(set(result["conditions"]), set(CONDITIONS))
        self.assertEqual(result["checks"]["frozen_replay"]["graph"], 0)
        self.assertTrue(result["checks"]["gate_replay"]["applied_bit_identical"])
        self.assertEqual(result["checks"]["off_uninstrumented"]["graph"], 0)
        self.assertGreater(abs(result["effects"][0]["gate_derivative_effect"]), 1e-6)
        self.assertGreater(abs(result["effects"][0]["outside_suppression_effect"]), 1e-6)
        for name in ("outside_full", "outside_suppress", "outside_amplify"):
            for h in result["conditions"][name]["paths"][0]["hops"]:
                self.assertEqual(h["g_applied"], 1)
        self.assertTrue(torch.all(gates["protected"][:, :, [0, 1, 2]]))
        self.assertFalse(hasattr(model.base.entity_model, "_ccmp_intervention"))
        self.assertTrue(model.base.entity_model.resp_gate)
        with tempfile.TemporaryDirectory() as d:
            render_report({"manifest": {"precision": "float32"}, "results": [result]}, d)
            self.assertIn("Outside suppression", (Path(d) / "report.md").read_text())
            self.assertEqual(len((Path(d) / "hops.csv").read_text().splitlines()), 15)
            self.assertTrue((Path(d) / "tikz_values.tsv").exists())

    def test_path_lookup_rejects_duplicate_edges_and_wrong_gold(self):
        model, graph, src, batch, case = fixture()
        graph.edge_index = torch.cat([graph.edge_index, graph.edge_index[:, :1]], dim=1)
        graph.edge_type = torch.cat([graph.edge_type, graph.edge_type[:1]])
        with self.assertRaisesRegex(ValueError, "Expected one exact edge"):
            resolve_paths(case, src, graph, batch, 3, "nodes_all_layers")
        batch["target_nodes_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "not a gold"):
            resolve_paths(case, src, graph, batch, 3, "nodes_all_layers")

    def test_intervention_cleanup_on_failure(self):
        model, graph, src, batch, case = fixture()
        target, paths, mask = resolve_paths(case, src, graph, batch, 3, "nodes_all_layers")
        with self.assertRaises(ValueError):
            run_condition(model, graph, batch, target, paths, mask, "bad", torch.float32)
        self.assertTrue(model.base.entity_model.resp_gate)
        self.assertFalse(hasattr(model.base.entity_model, "_ccmp_intervention"))

    def test_pinned_cases_are_exact_simple_paths_from_saved_traces(self):
        root = ROOT / "experiments"
        for case in json.loads((root / "eval/ccmp_mechanism_cases.json").read_text()):
            source = json.loads((root / case["source"]).read_text())
            query = next(q for q in source if q["id"] == case["query_id"])
            target = next(t for t in query["targets"] if t["doc"] == case["gold_id"])
            for p in case["paths"]:
                canonical = [{k: h[k] for k in ("layer", "head", "rel", "tail")} for h in p["hops"]]
                self.assertTrue(any(canonical == [{k: h[k] for k in ("layer", "head", "rel", "tail")} for h in sp["hops"]]
                                    for sp in target["paths"]))
                nodes = [p["hops"][0]["head"]] + [h["tail"] for h in p["hops"]]
                self.assertEqual(len(nodes), len(set(nodes)))
                self.assertEqual(nodes[-1], case["gold_id"])

    def test_workflow_writes_results_resumes_by_hash_and_loads_strictly(self):
        # Exercise the actual workflow's main/fingerprint functions with tiny data,
        # replacing only Hydra/trainer/data-loader infrastructure unavailable locally.
        source = (ROOT / "experiments/eval/ccmp_mechanism_workflow.py").read_text()
        funcs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        for f in funcs:
            f.decorator_list = []

        def plain(x):
            if isinstance(x, SimpleNamespace):
                return {k: plain(v) for k, v in vars(x).items()}
            if isinstance(x, list):
                return [plain(v) for v in x]
            return x

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model, graph, src, batch, case = fixture()
            graph.to = lambda device: graph
            src.graph = graph
            src.test_data = [{**{k: v[0] for k, v in batch.items() if k != "id"}, "id": "query"}]
            src.processed_dir = str(d / "data")
            Path(src.processed_dir).mkdir()
            src.processed_graph = [str(d / "data/graph.pt")]
            torch.save({"dummy": 1}, src.processed_graph[0])
            torch.save(src.test_data, d / "data/test.pt")
            torch.save({"model": model.state_dict()}, d / "model.pth")
            (d / "cases.json").write_text(json.dumps([case]))
            selection = {"checkpoint": str(d / "model.pth"), "graph": "toy", "family": "merged",
                         "reason": "Merged fixture selected"}
            (d / "selection.json").write_text(json.dumps(selection))
            wf = d / "engine/workflow/ccmp_mechanism.py"
            wf.parent.mkdir(parents=True)
            wf.write_text(source)
            cfg = SimpleNamespace(seed=1024, model=SimpleNamespace(dtype="float32"),
                                  trainer=SimpleNamespace(args=SimpleNamespace(resume_from_checkpoint=None)),
                                  optimizer=SimpleNamespace(), losses=[],
                                  datasets=SimpleNamespace(valid_names=["toy"]),
                                  mechanism=SimpleNamespace(ckpt=str(d / "model.pth"), cases=str(d / "cases.json"),
                                                            selection=str(d / "selection.json"),
                                                            out=str(d / "outputs"), protection="nodes_all_layers",
                                                            resume=True, atol=1e-5, rtol=1e-4))
            class Loader:
                def __init__(self, *a, **kw): pass
                def __iter__(self): return iter([SimpleNamespace(data=src)])
                def shutdown(self): pass

            def instantiate(component, *args, **kw):
                if component is cfg.model:
                    return TinyFusion()
                if component is cfg.trainer:
                    return SimpleNamespace(model=kw["model"], dtype=torch.float32, device=torch.device("cpu"))
                return None

            calls = []
            def count_experiment(*args, **kwargs):
                calls.append(1)
                return experiment(*args, **kwargs)
            scope = {"__file__": str(wf), "Path": Path, "json": json, "os": os, "torch": torch,
                     "np": __import__("numpy"), "random": __import__("random"), "DictConfig": object,
                     "OmegaConf": SimpleNamespace(to_container=lambda cfg, **kw: plain(cfg)),
                     "HydraConfig": SimpleNamespace(get=lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(d)))),
                     "utils": SimpleNamespace(init_multi_dataset=lambda *a: [1]),
                     "GraphDatasetLoader": Loader, "instantiate": instantiate, "experiment": count_experiment,
                     **{k: getattr(core, k) for k in ("VERSION", "atomic_json", "digest", "render_report", "sha256_file")}}
            exec(compile(ast.Module(body=funcs, type_ignores=[]), str(wf), "exec"), scope)
            with contextlib.redirect_stdout(io.StringIO()):
                scope["main"](cfg)
                first = json.loads((d / "outputs/latest.json").read_text())
                scope["main"](cfg)
                self.assertEqual(len(calls), 1, "Identical inputs should resume")
                run_dir = Path(first["results"]).parent
                gate = next(run_dir.glob("gates_*.pt"))
                gate.write_bytes(gate.read_bytes() + b"changed")
                scope["main"](cfg)
                self.assertEqual(len(calls), 2, "A changed gate archive must not resume")
                wf.write_text(source + "\n# Source changed\n")
                scope["main"](cfg)
                self.assertEqual(len(calls), 3, "Changed engine source must invalidate the cache")
                latest = json.loads((d / "outputs/latest.json").read_text())
                self.assertNotEqual(first["run_id"], latest["run_id"])
                payload = json.loads(Path(latest["results"]).read_text())
                self.assertEqual(payload["status"], "complete")
                self.assertEqual(payload["manifest"]["selection"], selection)
                self.assertEqual(len(payload["results"]), 1)
                self.assertTrue(Path(latest["results"]).with_name("hops.csv").exists())
                (d / "selection.json").write_text(json.dumps({**selection, "graph": "wrong_graph"}))
                with self.assertRaisesRegex(ValueError, "differs from workflow configuration"):
                    scope["main"](cfg)
                (d / "selection.json").write_text(json.dumps(selection))
                state = model.state_dict()
                del state["base.entity_model.resp_head.weight"]
                torch.save({"model": state}, d / "model.pth")
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    scope["main"](cfg)


if __name__ == "__main__":
    unittest.main()
