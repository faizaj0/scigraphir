"""CPU tests for the path-sender gate effect (Delta_P).

They execute the REAL patched QueryNBFNet.bellmanford with tiny differentiable layers (the
fixture of test_ccmp_mechanism.py) and a stubbed path discovery; the Drive checkpoints and CUDA
kernels are not exercised. Run:
    python3 -m unittest discover -s experiments/eval -p test_ccmp_path_sender_effect.py -v
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

import ccmp_mechanism as core
from ccmp_mechanism import resolve_paths, run_condition
import ccmp_path_sender_effect as ps
from test_ccmp_mechanism import TinyFusion, fixture

ROOT = Path(__file__).resolve().parents[2]
LAYERS = 3
TOY_PATHS = [
    {"name": "P1", "hops": [{"layer": 0, "head": "seed", "rel": "rel", "tail": "method"},
                            {"layer": 1, "head": "method", "rel": "rel", "tail": "gold"}]},
    {"name": "P2", "hops": [{"layer": 0, "head": "outside", "rel": "rel", "tail": "method"},
                            {"layer": 1, "head": "method", "rel": "rel", "tail": "gold"}]},
]


def stub_discovery(model, graph, batch, src, case, dtype, documents, top_k=3, beam_size=10):
    paths = [{"name": p["name"], "weight": 1.0 - 0.1 * i, "hops": [dict(h) for h in p["hops"]]}
             for i, p in enumerate(TOY_PATHS)][:top_k]
    return ({"top_paths": paths, "valid_candidate_count": len(paths), "rejected_count": 0,
             "rejection_reasons": [], "channels": {}, "gate_stats": []}, {})


def toy_case():
    _, _, _, _, case = fixture()
    return {k: v for k, v in case.items() if k != "paths"}


def run_toy(top_k=2):
    model, graph, src, batch, _ = fixture()
    with contextlib.redirect_stdout(io.StringIO()):
        result, archive = ps.path_sender_experiment(model, graph, batch, src, toy_case(), torch.float32, {},
                                                    top_k=top_k, beam_size=10, discover=stub_discovery)
    return model, graph, src, batch, result, archive


class PathSenderTest(unittest.TestCase):
    def test_sender_mask_marks_senders_at_every_layer_and_nothing_else(self):
        m = ps.sender_mask([[0, 1], [4]], LAYERS, 5, torch.device("cpu"))
        self.assertEqual(tuple(m.shape), (LAYERS, 1, 5))
        self.assertTrue(bool(m[:, 0, [0, 1, 4]].all()))
        self.assertFalse(bool(m[:, 0, [2, 3]].any()))
        self.assertFalse(bool(ps.sender_mask([], LAYERS, 5, torch.device("cpu")).any()))

    def test_toy_gate_is_not_the_identity(self):
        _, _, _, _, _, archive = run_toy()
        self.assertTrue(bool((archive["native_raw"] != 1).any()), "the toy CCMP gate must move some node")

    def test_delta_p_equals_native_minus_a_direct_frozen_reset(self):
        model, graph, src, batch, result, archive = run_toy()
        self.assertTrue(all(result["checks"]["gate_replay"].values()))
        self.assertEqual([e["path"] for e in result["effects"]], ["P1", "P2"])
        self.assertEqual(result["fixed_paths"], TOY_PATHS)
        for stats in result["conditions"]["off"]["gate_stats"]:      # off = every gate exactly 1
            self.assertEqual(stats["raw"]["exactly_one"], stats["raw"]["n"])
        target, paths, _ = resolve_paths({**toy_case(), "paths": result["fixed_paths"]}, src, graph, batch,
                                         LAYERS, "sender_layer")
        gates = [archive["native_raw"][l] for l in range(LAYERS)]
        s_native = result["ranks"]["native"]["graph"]["score"]
        for e, p in zip(result["effects"], paths):
            senders = ps.sender_ids(p)
            self.assertEqual(e["senders"], [src.id2node[u] for u in senders])
            self.assertNotIn("gold", e["senders"])
            mask = ps.sender_mask([senders], LAYERS, graph.num_nodes, torch.device("cpu"))
            with contextlib.redirect_stdout(io.StringIO()):
                rec, gs, _, _ = run_condition(model, graph, batch, target, paths, mask, "outside_full",
                                              torch.float32, gates)
            self.assertAlmostEqual(e["graph_score_reset"], rec["channels"]["graph"]["score"], places=6)
            self.assertAlmostEqual(e["delta_P"], s_native - rec["channels"]["graph"]["score"], places=6)
            for l in range(LAYERS):                                     # native off the senders, 1 on them
                keep = torch.ones(graph.num_nodes, dtype=torch.bool)
                keep[senders] = False
                self.assertTrue(torch.equal(gs[l][0, keep], gates[l][0, keep]))
                self.assertTrue(bool((gs[l][0, senders] == 1).all()))
            self.assertEqual(e["native_gates_on_senders"].keys(), set(e["senders"]))
            self.assertEqual(len(next(iter(e["native_gates_on_senders"].values()))), LAYERS)
        # attribution is the native fixed-path weight, never derived from gates
        self.assertEqual(result["effects"][0]["attribution_native"], result["conditions"]["native"]["paths"][0]["weight"])
        # shared senders are recorded; the union is measured, and the naive sum is only reported for contrast
        self.assertEqual(result["sender_overlap"]["P1&P2"], ["method"])
        self.assertEqual(result["union_effect"]["n_senders"], 3)
        self.assertAlmostEqual(result["union_effect"]["sum_of_individual_delta_P"],
                               sum(e["delta_P"] for e in result["effects"]), places=9)
        self.assertEqual(result["union_effect"]["condition"], "reset_senders_union")

    def test_single_path_has_no_union(self):
        _, _, _, _, result, _ = run_toy(top_k=1)
        self.assertEqual([e["path"] for e in result["effects"]], ["P1"])
        self.assertIsNone(result["union_effect"])
        self.assertNotIn("reset_senders_union", result["conditions"])

    def test_report_files(self):
        _, _, _, _, result, _ = run_toy()
        payload = {"manifest": {"precision": "float32", "selection": {"checkpoint": "toy.pth", "graph": "toy"}},
                   "run_id": "r" * 40, "status": "complete", "results": [result]}
        with tempfile.TemporaryDirectory() as d:
            ps.render_path_sender_report(payload, d)
            names = {p.name for p in Path(d).iterdir()}
            self.assertTrue({"report.md", "tables.tex", "summary.csv", "hops.csv", "fixed_paths.json"} <= names)
            report = (Path(d) / "report.md").read_text()
            self.assertIn("| P1 |", report)
            self.assertIn("Δ_P", report)
            self.assertIn("all gates 1", report)
            tex = (Path(d) / "tables.tex").read_text()
            self.assertIn(r"\Delta_P", tex)
            self.assertEqual(len(json.loads((Path(d) / "fixed_paths.json").read_text())["toy"]), 2)
            rows = (Path(d) / "summary.csv").read_text().splitlines()
            self.assertEqual(len(rows), 3)
            self.assertIn("delta_P", rows[0])

    def test_workflow_writes_results_resumes_by_hash_and_loads_strictly(self):
        source = (ROOT / "experiments/eval/ccmp_path_sender_workflow.py").read_text()
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
            model, graph, src, batch, _ = fixture()
            graph.to = lambda device: graph
            src.graph = graph
            src.test_data = [{**{k: v[0] for k, v in batch.items() if k != "id"}, "id": "query"}]
            src.processed_dir = str(d / "data/toy/processed")
            src.raw_dir = str(d / "data/toy/raw")
            Path(src.processed_dir).mkdir(parents=True)
            Path(src.raw_dir).mkdir(parents=True)
            src.processed_graph = [str(d / "data/toy/processed/graph.pt")]
            torch.save({"dummy": 1}, src.processed_graph[0])
            torch.save(src.test_data, d / "data/toy/processed/test.pt")
            (d / "data/toy/raw/test.json").write_text(json.dumps([{"id": "query", "question": "q?", "stratum": "cross"}]))
            (d / "data/toy/raw/documents.json").write_text(json.dumps({"gold": "Gold paper. Abstract."}))
            torch.save({"model": model.state_dict()}, d / "model.pth")
            case = toy_case()
            (d / "cases.json").write_text(json.dumps([case]))
            selection = {"checkpoint": str(d / "model.pth"), "graph": "toy", "family": "frame", "reason": "fixture"}
            (d / "selection.json").write_text(json.dumps(selection))
            wf = d / "engine/workflow/ccmp_path_sender.py"
            wf.parent.mkdir(parents=True)
            wf.write_text(source)
            cfg = SimpleNamespace(seed=1024, model=SimpleNamespace(dtype="float32"),
                                  trainer=SimpleNamespace(args=SimpleNamespace(resume_from_checkpoint=None)),
                                  optimizer=SimpleNamespace(), losses=[],
                                  datasets=SimpleNamespace(valid_names=["toy"], cfgs=SimpleNamespace(root=str(d / "data"))),
                                  mechanism=SimpleNamespace(ckpt=str(d / "model.pth"), cases=str(d / "cases.json"),
                                                            selection=str(d / "selection.json"), out=str(d / "outputs"),
                                                            resume=True, atol=1e-5, rtol=1e-4, top_k=2, beam_size=10))

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

            def counted(*args, **kwargs):
                calls.append(1)
                return ps.path_sender_experiment(*args, discover=stub_discovery, **kwargs)

            scope = {"__file__": str(wf), "Path": Path, "json": json, "os": os, "torch": torch,
                     "np": __import__("numpy"), "random": __import__("random"), "DictConfig": object,
                     "OmegaConf": SimpleNamespace(to_container=lambda cfg, **kw: plain(cfg)),
                     "HydraConfig": SimpleNamespace(get=lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(d)))),
                     "utils": SimpleNamespace(init_multi_dataset=lambda *a: [1]),
                     "GraphDatasetLoader": Loader, "instantiate": instantiate,
                     "path_sender_experiment": counted, "render_path_sender_report": ps.render_path_sender_report,
                     "VERSION": ps.VERSION,
                     **{k: getattr(core, k) for k in ("atomic_json", "digest", "sha256_file")}}
            exec(compile(ast.Module(body=funcs, type_ignores=[]), str(wf), "exec"), scope)
            with contextlib.redirect_stdout(io.StringIO()):
                scope["main"](cfg)
                first = json.loads((d / "outputs/latest.json").read_text())
                self.assertEqual(first["version"], ps.VERSION)
                scope["main"](cfg)
                self.assertEqual(len(calls), 1, "Identical inputs should resume")
                run_dir = Path(first["results"]).parent
                self.assertTrue((run_dir / "report.md").exists() and (run_dir / "tables.tex").exists())
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
                self.assertEqual(payload["manifest"]["analysis_mode"], "path_senders")
                self.assertEqual(payload["manifest"]["top_k"], 2)
                self.assertEqual(payload["results"][0]["target_title"], "Gold paper")
                self.assertEqual(payload["results"][0]["question"], "")   # fixture has no raw_test_data
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
