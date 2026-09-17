"""Checkpoint preference, matched graph/scorer, and fixed-path preservation checks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from ccmp_checkpoint_selection import merged_candidates, select_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def touch(path, content=b"checkpoint"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def graph_files(data, graph):
    for f in ("nodes.csv", "edges.csv", "relations.csv", "test.json"):
        touch(data / graph / "processed/stage1" / f, b"[]")
    touch(data / graph / "raw/documents.json", b"{}")


def fixture(root, dataset="sir4_cs"):
    drive, data = root / "drive", root / "scigraphir/retriever/data"
    spec = {"frame": f"{dataset}_test_v16sc",
            "sem": {"field": (f"outputs/{dataset}/semantic", dataset),
                    "frame": ("outputs/sir4_zeroshot/semantic_sir4_cs", "sir4_cs")},
            "arms": [("frame_ccmp", "outputs/frame_checkpoint", "frame", "frame", True)]}
    touch(drive / "outputs/frame_checkpoint/model_best.pth")
    graph_files(data, spec["frame"])
    graph_files(data, f"{dataset}_test_hyb")
    for stem, ext in (("params", "json"), ("popnet", "pt")):
        touch(drive / f"outputs/sir4_hyb/semantic_{dataset}/{stem}_semantic_mlp_fixedloss_{dataset}.{ext}")
    return drive, data, spec


class CheckpointSelectionTest(unittest.TestCase):
    def test_merged_wins_and_switches_graph_and_scorer_together(self):
        with tempfile.TemporaryDirectory() as d:
            drive, data, spec = fixture(Path(d))
            older = touch(drive / "outputs/sir4_hyb/sir4_cs_hyb_ccmp_e20_b2/model_best.pth")
            chosen = touch(drive / "outputs/sir4_hyb/sir4_cs_hyb_ccmp_e10_b1/model_best.pth")
            selection = select_checkpoint("sir4_cs", spec, drive, data)
            self.assertEqual(selection["checkpoint"], str(chosen))
            self.assertEqual(selection["graph"], "sir4_cs_test_hyb")
            self.assertEqual(selection["scorer_key"], "field")
            self.assertEqual(selection["semantic_specs"]["field"][0], "outputs/sir4_hyb/semantic_sir4_cs")
            self.assertEqual(selection["family"], "merged")
            self.assertEqual(selection, json.loads(json.dumps(selection)))
            chosen.unlink()
            self.assertEqual(select_checkpoint("sir4_cs", spec, drive, data)["checkpoint"], str(older))

    def test_fallback_is_explicit_and_ignores_control_smoke_other_fields(self):
        with tempfile.TemporaryDirectory() as d:
            drive, data, spec = fixture(Path(d))
            for name in ("sir4_cs_hyb_control_e10_b1", "sir4_cs_hyb_ccmp_e10_b1_smoke", "sir4_biology_hyb_ccmp_e10_b2"):
                touch(drive / "outputs/sir4_hyb" / name / "model_best.pth")
            self.assertEqual(merged_candidates("sir4_cs", drive), [])
            selection = select_checkpoint("sir4_cs", spec, drive, data)
            self.assertEqual(selection["family"], "frame")
            self.assertEqual(selection["graph"], "sir4_cs_test_v16sc")
            self.assertEqual(selection["scorer_key"], "frame")
            self.assertIn("FALLBACK", selection["reason"])

    def test_biology_defaults_to_b2_and_explicit_frame_mode_works(self):
        with tempfile.TemporaryDirectory() as d:
            drive, data, spec = fixture(Path(d), "sir4_biology")
            touch(drive / "outputs/sir4_hyb/sir4_biology_hyb_ccmp_e10_b1/model_best.pth")
            chosen = touch(drive / "outputs/sir4_hyb/sir4_biology_hyb_ccmp_e10_b2/model_best.pth")
            self.assertEqual(select_checkpoint("sir4_biology", spec, drive, data)["checkpoint"], str(chosen))
            self.assertEqual(select_checkpoint("sir4_biology", spec, drive, data, prefer_merged=False)["family"], "frame")

    def test_missing_graph_is_unpacked_from_shared_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            drive, data, spec = fixture(Path(d))
            touch(drive / "outputs/sir4_hyb/sir4_cs_hyb_ccmp_e10_b1/model_best.pth")
            missing = data / "sir4_cs_test_hyb/processed/stage1/nodes.csv"
            missing.unlink()
            with zipfile.ZipFile(drive / "sir4_hyb_bundle.zip", "w") as z:
                for f in ("nodes.csv", "edges.csv", "relations.csv", "test.json"):
                    z.writestr(f"retriever/data/sir4_cs_test_hyb/processed/stage1/{f}", "bundle graph")
            self.assertEqual(select_checkpoint("sir4_cs", spec, drive, data)["family"], "merged")
            self.assertEqual(missing.read_text(), "bundle graph")

    def test_existing_merged_checkpoint_with_bad_assets_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as d:
            drive, data, spec = fixture(Path(d))
            chosen = touch(drive / "outputs/sir4_hyb/sir4_cs_hyb_ccmp_e10_b1/model_best.pth")
            metadata = chosen.with_name("arm.json")
            metadata.write_text(json.dumps({"valid": "sir4_cs_test_v16sc"}))
            with self.assertRaisesRegex(ValueError, "expected 'sir4_cs_test_hyb'"):
                select_checkpoint("sir4_cs", spec, drive, data)
            metadata.unlink()
            (data / "sir4_cs_test_hyb/processed/stage1/nodes.csv").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "not silently changed"):
                select_checkpoint("sir4_cs", spec, drive, data)

    def test_fixed_figure_paths_and_seeds_exist_in_real_merged_graphs(self):
        cases = json.loads((ROOT / "experiments/eval/ccmp_mechanism_cases.json").read_text())
        for ds in {c["dataset"] for c in cases}:
            stage = ROOT / "retriever/data" / f"{ds}_test_hyb/processed/stage1"
            with (stage / "edges.csv").open() as f:
                edges = {(r["source"], r["relation"], r["target"]) for r in csv.DictReader(f)}
            queries = {q["id"]: q for q in json.loads((stage / "test.json").read_text())}
            for case in (c for c in cases if c["dataset"] == ds):
                query = queries[case["query_id"]]
                seeds = {n for values in query["start_nodes"].values() for n in values}
                self.assertIn(case["gold_id"], query["supporting_documents"])
                for path in case["paths"]:
                    self.assertIn(path["hops"][0]["head"], seeds)
                    for hop in path["hops"]:
                        rel = hop["rel"]
                        edge = ((hop["tail"], rel[8:], hop["head"]) if rel.startswith("inverse_")
                                else (hop["head"], rel, hop["tail"]))
                        self.assertIn(edge, edges)


if __name__ == "__main__":
    unittest.main()
