"""Run the real engine's gradient pass AND NBFNet beam search on a small CPU graph."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import ccmp_mechanism as core
import scientific_path_interpretations as scientific
from test_ccmp_mechanism import fixture, ROOT, TinyFusion
from scientific_path_interpretations import interpret_scientific_paths, render_scientific_table, decode_candidates, relation_text


def attach_real_beam(entity_model):
    path = ROOT / "retriever/gfm-rag/gfmrag/models/ultra"
    variadic_tree = ast.parse((path / "variadic.py").read_text())
    vs = {"torch": torch}
    nodes = [n for n in variadic_tree.body if isinstance(n, ast.FunctionDef) and n.name in {"broadcast", "native_scatter"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "variadic.py", "exec"), vs)
    scope = {"torch": torch, "variadic": SimpleNamespace(native_scatter=vs["native_scatter"])}
    tree = ast.parse((path / "base_nbfnet.py").read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"size_to_index", "multi_slice_mask", "scatter_extend", "scatter_topk"}]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseNBFNet")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"beam_search_distance", "topk_average_length"}]
    exec(compile(ast.Module(body=funcs + methods, type_ignores=[]), "base_nbfnet.py", "exec"), scope)
    for m in methods:
        setattr(entity_model, m.name, MethodType(scope[m.name], entity_model))


def scientific_fixture():
    model, graph, src, batch, case = fixture()
    attach_real_beam(model.base.entity_model)
    case = {k: v for k, v in case.items() if k != "paths"}
    case["dataset"] = "sir4_biology"
    src.raw_test_data = [{"id": "query", "question": "Which method could address this scientific problem?",
                          "supporting_documents": ["gold"], "stratum": "cross"}]
    docs = {"gold": "Target scientific paper. Abstract text.", "competitor": "Another paper."}
    return model, graph, src, batch, case, docs


class ScientificPathsTest(unittest.TestCase):
    def test_actual_beam_discovers_new_paths_and_matches_gradient_averages(self):
        model, graph, src, batch, case, docs = scientific_fixture()
        result, _ = interpret_scientific_paths(model, graph, batch, src, case, torch.float32, docs, top_k=3, beam_size=5)
        self.assertTrue(result["top_paths"])
        self.assertEqual(result["forward_score_check"]["graph"], 0)
        self.assertEqual(result["benchmark_stratum"], "cross")
        self.assertEqual(result["target_title"], "Target scientific paper")
        self.assertIsNone(result["channels"]["dense"])
        self.assertGreaterEqual(len(result["top_multihop_paths"]), 1)
        weights = [p["weight"] for p in result["top_paths"]]
        self.assertEqual(weights, sorted(weights, reverse=True))
        for p in result["top_paths"]:
            self.assertAlmostEqual(p["weight"], sum(h["edge_gradient"] for h in p["hops"]) / len(p["hops"]))
            nodes = [p["hops"][0]["head"]] + [h["tail"] for h in p["hops"]]
            self.assertEqual(nodes[-1], "gold")
            self.assertEqual(len(nodes), len(set(nodes)))
        with tempfile.TemporaryDirectory() as d:
            payload = {"run_id": "test", "manifest": {"selection": {"graph": "test_hyb", "checkpoint": "synthetic"}}, "results": [result]}
            render_scientific_table(payload, d)
            text = (Path(d) / "table4.md").read_text()
            self.assertIn("Top paths, including single-hop", text)
            self.assertIn("Target scientific paper", text)
            self.assertTrue((Path(d) / "top_paths.tikz.tex").exists())
            fixed = json.loads((Path(d) / "discovered_cases_for_interventions.json").read_text())
            self.assertEqual(fixed[0]["paths"][0]["hops"][0]["head"], result["top_paths"][0]["hops"][0]["head"])
            self.assertEqual(fixed[0]["selection_source"]["checkpoint"], "synthetic")

    def test_top_one_hop_is_not_hidden_and_invalid_paths_are_rejected(self):
        model, graph, src, batch, case, docs = scientific_fixture()
        graph.edge_index = torch.cat([graph.edge_index, torch.tensor([[0], [2]])], dim=1)
        graph.edge_type = torch.cat([graph.edge_type, torch.tensor([0])])
        grads = [torch.ones(6) for _ in range(3)]
        grads[0][5] = 100
        gates = [torch.ones((1, 5)) for _ in range(3)]
        stats = [{"message_dtype": "torch.float32"} for _ in range(3)]
        result = decode_candidates([[(0, 2, 0)], [(0, 1, 0), (1, 2, 0)], [(1, 2, 0)], [(0, 1, 0)]],
                                   [100, 1, 1, 1], grads, graph, batch, src, "gold", gates, stats, docs, 2)
        self.assertEqual(len(result["top_paths"][0]["hops"]), 1)
        self.assertIn("single hop", result["top_paths"][0]["flags"])
        self.assertEqual(result["top_multihop_paths"][0]["name"], "P2")
        self.assertEqual(result["rejected_count"], 2)

    def test_inverse_relation_is_displayed_as_a_reverse_traversal(self):
        self.assertEqual(relation_text("inverse_overcomes"), "overcomes⁻¹")
        self.assertEqual(relation_text("inverse_overcomes", tex=True), r"overcomes$^{-1}$")

    def test_empty_beam_does_not_invent_a_path(self):
        model, graph, src, batch, case, docs = scientific_fixture()
        model.base.entity_model.topk_average_length = lambda *a, **k: ([], [])
        result, _ = interpret_scientific_paths(model, graph, batch, src, case, torch.float32, docs, top_k=2, beam_size=5)
        self.assertEqual(result["top_paths"], [])
        self.assertEqual(result["top_multihop_paths"], [])

    def test_paths_workflow_exports_and_invalidates_changed_question_metadata(self):
        # Execute the real workflow with the real gradient/beam routines. Replace
        # only unavailable Hydra/trainer infrastructure and the checkpoint scale.
        source = (ROOT / "experiments/eval/ccmp_mechanism_workflow.py").read_text()
        funcs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        for f in funcs:
            f.decorator_list = []

        def plain(value):
            if isinstance(value, SimpleNamespace):
                return {k: plain(v) for k, v in vars(value).items()}
            if isinstance(value, list):
                return [plain(v) for v in value]
            return value

        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            model, graph, src, batch, case, docs = scientific_fixture()
            graph.to = lambda device: graph
            src.graph = graph
            src.test_data = [{**{k: v[0] for k, v in batch.items() if k != "id"}, "id": "query"}]
            src.processed_dir = str(d / "data/toy/processed/stage2")
            src.raw_dir = str(d / "data/toy/processed/stage1")
            for path in (Path(src.processed_dir), Path(src.raw_dir), d / "data/toy/raw"):
                path.mkdir(parents=True)
            src.processed_graph = [str(Path(src.processed_dir) / "graph.pt")]
            torch.save({"dummy": 1}, src.processed_graph[0])
            torch.save(src.test_data, Path(src.processed_dir) / "test.pt")
            question_file = Path(src.raw_dir) / "test.json"
            question_file.write_text(json.dumps(src.raw_test_data))
            (d / "data/toy/raw/documents.json").write_text(json.dumps(docs))
            torch.save({"model": model.state_dict()}, d / "model.pth")
            (d / "cases.json").write_text(json.dumps([case]))
            selection = {"checkpoint": str(d / "model.pth"), "graph": "toy", "family": "merged"}
            (d / "selection.json").write_text(json.dumps(selection))
            wf = d / "engine/workflow/ccmp_mechanism.py"
            wf.parent.mkdir(parents=True)
            wf.write_text(source)
            cfg = SimpleNamespace(seed=1024, model=SimpleNamespace(dtype="float32"),
                                  trainer=SimpleNamespace(args=SimpleNamespace(resume_from_checkpoint=None)),
                                  optimizer=SimpleNamespace(), losses=[],
                                  datasets=SimpleNamespace(valid_names=["toy"], cfgs=SimpleNamespace(root=str(d / "data"))),
                                  mechanism=SimpleNamespace(mode="paths", ckpt=str(d / "model.pth"), cases=str(d / "cases.json"),
                                                            selection=str(d / "selection.json"), out=str(d / "outputs"),
                                                            protection="none", resume=True, top_k=3, beam_size=5))

            class Loader:
                def __init__(self, *a, **kw): pass
                def __iter__(self): return iter([SimpleNamespace(data=src, name="toy")])
                def shutdown(self): pass

            def instantiate(component, *args, **kwargs):
                if component is cfg.model:
                    model = TinyFusion()
                    attach_real_beam(model.base.entity_model)
                    return model
                if component is cfg.trainer:
                    return SimpleNamespace(model=kwargs["model"], dtype=torch.float32, device=torch.device("cpu"))
                return None

            scope = {"__file__": str(wf), "Path": Path, "json": json, "os": os, "torch": torch,
                     "np": __import__("numpy"), "random": __import__("random"), "DictConfig": object,
                     "OmegaConf": SimpleNamespace(to_container=lambda cfg, **kw: plain(cfg)),
                     "HydraConfig": SimpleNamespace(get=lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(d)))),
                     "utils": SimpleNamespace(init_multi_dataset=lambda *a: [1]),
                     "GraphDatasetLoader": Loader, "instantiate": instantiate,
                     **{k: getattr(core, k) for k in ("VERSION", "atomic_json", "digest", "render_report", "sha256_file")}}
            exec(compile(ast.Module(body=funcs, type_ignores=[]), str(wf), "exec"), scope)
            with patch.dict(sys.modules, {"gfmrag.workflow.scientific_paths": scientific}), \
                    patch.object(scientific, "interpret_scientific_paths", wraps=interpret_scientific_paths) as discovery, \
                    contextlib.redirect_stdout(io.StringIO()):
                scope["main"](cfg)
                first = json.loads((d / "outputs/latest.json").read_text())
                payload = json.loads(Path(first["results"]).read_text())
                self.assertEqual(payload["manifest"]["analysis_mode"], "paths")
                self.assertTrue(payload["results"][0]["top_paths"])
                self.assertTrue(Path(first["results"]).with_name("table4.tex").exists())
                scope["main"](cfg)
                self.assertEqual(discovery.call_count, 1)
                src.raw_test_data[0]["question"] = "Updated scientific question"
                question_file.write_text(json.dumps(src.raw_test_data))
                scope["main"](cfg)
                self.assertEqual(discovery.call_count, 2)
                latest = json.loads((d / "outputs/latest.json").read_text())
                self.assertNotEqual(first["run_id"], latest["run_id"])
                payload = json.loads(Path(latest["results"]).read_text())
                self.assertEqual(payload["results"][0]["question"], "Updated scientific question")


if __name__ == "__main__":
    unittest.main()
