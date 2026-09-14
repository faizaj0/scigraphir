"""Prefer SIR-4 merged-graph CCMP checkpoints and their matching test graph/scorer.

Run names and scorer locations follow prep/build_sir4_hyb_notebook.py and
prep/build_sir4_openie_notebook.py. Availability is checked on mounted Drive.
"""
import json
from pathlib import Path
import re
import shutil
import zipfile


def merged_candidates(dataset, drive):
    if dataset not in {"sir4_cs", "sir4_biology", "sir4_physics", "sir4_matsci"}:
        return []
    root = Path(drive) / "outputs/sir4_hyb"
    default_batch = 1 if dataset == "sir4_cs" else 2
    preferred = root / f"{dataset}_hyb_ccmp_e10_b{default_batch}" / "model_best.pth"
    pattern = re.compile(rf"{re.escape(dataset)}_hyb_ccmp_e(\d+)_b(\d+)")
    candidates = []
    for path in root.glob(f"{dataset}_hyb_ccmp_e*_b*/model_best.pth"):
        match = pattern.fullmatch(path.parent.name)
        if match and path.is_file() and path.stat().st_size:
            candidates.append((path, int(match[1]), int(match[2])))
    # Prefer the standard e10/default-batch recipe; then other available runs by
    # epoch and batch, deterministically. Never select by test performance or mtime.
    candidates.sort(key=lambda x: (x[0] != preferred, -x[1], x[2] != default_batch, x[2], str(x[0])))
    return [p for p, _, _ in candidates]


def ensure_graph(dataset, graph, family, drive, data_root):
    root = Path(data_root) / graph
    needed = [f"processed/stage1/{f}" for f in ("nodes.csv", "edges.csv", "relations.csv", "test.json")]
    if not all((root / p).is_file() for p in needed):
        bundle = Path(drive) / "sir4_hyb_bundle.zip"
        if family != "merged" or not bundle.is_file():
            raise FileNotFoundError(f"Selected {family} checkpoint requires {graph}. "
                                    f"Its graph files are missing; merged graphs are supplied by {bundle}. "
                                    "Checkpoint selection was not silently changed.")
        prefix = f"kg-construction/data/{graph}/"
        with zipfile.ZipFile(bundle) as z:
            if not all(prefix + p in z.namelist() for p in needed):
                raise ValueError(f"{bundle} does not contain the complete test graph {graph}")
            # Extract just this graph, without overwriting other datasets or caches.
            for info in z.infolist():
                if not info.filename.startswith(prefix) or info.is_dir():
                    continue
                relative = Path(info.filename[len(prefix):])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Invalid bundle member: {info.filename}")
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as source, dest.open("wb") as target:
                    shutil.copyfileobj(source, target)
    docs = root / "raw/documents.json"
    if not docs.is_file():
        docs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(data_root) / f"{dataset}_test/raw/documents.json", docs)


def select_checkpoint(dataset, spec, drive, data_root, prefer_merged=True):
    """Select once per dataset; every intervention/precision shares this selection.

    A missing merged checkpoint permits fallback. A present checkpoint with
    missing/inconsistent assets is an error, not a silent change of experiment.
    Strict tensor loading and exact graph/path validation happen in the workflow.
    """
    available = merged_candidates(dataset, drive)
    if prefer_merged and available:
        ckpt, graph, family, name, skey = available[0], f"{dataset}_test_hyb", "merged", "merged_ccmp", "field"
        reason = "Merged-graph CCMP checkpoint available; preferred over frame-only checkpoint."
        meta_path = ckpt.parent / "arm.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        expected = {"field": dataset.removeprefix("sir4_"), "graph": "hyb", "valid": graph,
                    "train": f"{dataset}_train_hyb", "ccmp": True, "semantic": "mlp"}
        for key, value in expected.items():
            if key in metadata and metadata[key] != value:
                raise ValueError(f"{meta_path}: {key}={metadata[key]!r}, expected {value!r}")
        scorer_dirs = [f"outputs/sir4_hyb/semantic_{dataset}", spec["sem"]["field"][0],
                       f"outputs/{dataset}_ablations_v1/{dataset}/semantic"]
        scorer = next((rel for rel in scorer_dirs
                       if all((Path(drive) / rel / f"{stem}_semantic_mlp_fixedloss_{dataset}.{ext}").is_file()
                              for stem, ext in (("params", "json"), ("popnet", "pt")))), None)
        if scorer is None:
            raise FileNotFoundError(f"Merged checkpoint found, but its field scorer files are missing: {scorer_dirs}")
        semantic_specs = {"field": (scorer, dataset)}
    else:
        arms = [a for a in spec["arms"] if a[0] == "frame_ccmp"]
        if len(arms) != 1:
            raise ValueError(f"{dataset}: expected one frame CCMP fallback specification")
        name, relative, _, skey, _ = arms[0]
        relative = relative if isinstance(relative, list) else [relative]
        ckpt = next((Path(drive) / r / "model_best.pth" for r in relative
                     if (Path(drive) / r / "model_best.pth").is_file()
                     and (Path(drive) / r / "model_best.pth").stat().st_size), None)
        if ckpt is None:
            raise FileNotFoundError(f"{dataset}: neither a merged nor fallback CCMP checkpoint is available")
        graph, family, metadata = spec["frame"], "frame", {}
        semantic_specs = spec["sem"]
        reason = ("FALLBACK: no merged-graph CCMP checkpoint found under outputs/sir4_hyb; using the previous frame checkpoint."
                  if prefer_merged else "Frame checkpoint explicitly requested (PREFER_MERGED=False).")
    ensure_graph(dataset, graph, family, drive, data_root)
    return json.loads(json.dumps({"dataset": dataset, "family": family, "arm": name, "checkpoint": str(ckpt),
            "graph": graph, "scorer_key": skey, "semantic_specs": semantic_specs,
            "reason": reason, "prefer_merged": prefer_merged,
            "available_merged_checkpoints": [str(p) for p in available], "arm_metadata": metadata}))
