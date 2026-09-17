"""Installed as gfmrag.workflow.ccmp_mechanism by the Colab notebook. No training."""
try:
    import torchvision.io as _tvio
    if not hasattr(_tvio, "VideoReader"):
        class _NoVideoReader:
            def __init__(self, *a, **k):
                raise RuntimeError("torchvision video API removed")
        _tvio.VideoReader = _NoVideoReader
except Exception:
    pass

import json
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gfmrag import utils
from gfmrag.graph_index_datasets import GraphDatasetLoader
from gfmrag.trainers.sft_trainer import SFTLoss
from gfmrag.workflow.ccmp_mechanism_core import (
    VERSION, atomic_json, digest, experiment, render_report, sha256_file,
)


def fingerprint_inputs(cfg, src, cases):
    root = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(root)): sha256_file(p)
              for p in sorted(root.rglob("*.py"))}
    files = list(src.processed_graph) + [str(Path(src.processed_dir) / "test.pt")]
    mode = getattr(cfg.mechanism, "mode", "interventions")
    if mode == "paths":
        files += [str(Path(src.raw_dir) / "test.json"),
                  str(Path(cfg.datasets.cfgs.root) / cfg.datasets.valid_names[0] / "raw/documents.json")]
    selection = None
    if getattr(cfg.mechanism, "selection", None):
        selection = json.loads(Path(cfg.mechanism.selection).read_text())
        if (selection["checkpoint"] != str(cfg.mechanism.ckpt)
                or selection["graph"] != str(cfg.datasets.valid_names[0])):
            raise ValueError("Selected checkpoint/graph differs from workflow configuration")
        files.append(str(cfg.mechanism.selection))
    for key in ('HANDCRAFTED_COMPONENTS_TEST', "SEMANTIC_COMPONENTS_TEST", "SEMANTIC_CKPT", "SEMANTIC_POPNET"):
        p = os.environ.get(key)
        if p:
            files.append(p)
            if key == "SEMANTIC_COMPONENTS_TEST":
                with np.load(p, allow_pickle=True) as z:
                    files.append(str(z["h_path"]))
    inputs = {p: sha256_file(p) for p in sorted(set(files))}
    manifest = {
        "version": VERSION, "cases": cases, "checkpoint": str(cfg.mechanism.ckpt),
        "analysis_mode": mode,
        "checkpoint_sha256": sha256_file(cfg.mechanism.ckpt),
        "selection": selection,
        "engine_sha256": digest(source), "source_files": source, "inputs": inputs,
        "config": OmegaConf.to_container(cfg, resolve=True), "precision": str(cfg.model.dtype),
        "protection": str(cfg.mechanism.protection), "seed": int(cfg.seed),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "environment": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith(("CCMP", "ROUTE", "FUSION_", "SEM_", "SEED_", "CQIG"))},
        "attribution": ("finite-beam path discovery" if mode == "paths" else "fixed-path measurement")
                       + "; mean edge gradients from graph gold score; legacy linked layer clones, same as visualize; not a product of gates",
        "gate_policy": ("native differentiable CCMP gates, normalized by the model and cast to message dtype"
                        if mode == "paths" else
                        "freeze native-on gates; intervene after normalization; cast to message dtype; no renormalization"),
    }
    # RESUME only changes execution, not the experiment identity.
    manifest["config"]["mechanism"].pop("resume", None)
    for case in cases:
        source = case.get("selection_source")
        if source is not None:
            if (source["checkpoint"] != str(cfg.mechanism.ckpt)
                    or source["graph"] != str(cfg.datasets.valid_names[0])
                    or (source.get("checkpoint_sha256") is not None
                        and source["checkpoint_sha256"] != manifest["checkpoint_sha256"])):
                raise ValueError("Discovered paths were selected using a different checkpoint or graph")
    return manifest


@hydra.main(config_path="config/gfm_reasoner", config_name="sft_training_fusion", version_base=None)
def main(cfg: DictConfig):
    mode = getattr(cfg.mechanism, "mode", "interventions")
    if mode not in {"interventions", "paths"}:
        raise ValueError(f"Unknown analysis mode: {mode}")
    if mode == "paths":
        from gfmrag.workflow.scientific_paths import interpret_scientific_paths, render_scientific_table
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("Use a single GPU process")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if str(cfg.model.dtype) == "bfloat16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError("bfloat16 requested: use an Ampere-or-newer GPU or explicitly choose float32")
    if cfg.trainer.args.resume_from_checkpoint is not None:
        raise RuntimeError("Training resume must be disabled for this experiment")
    feat_dim = set(utils.init_multi_dataset(cfg, 1, 0))
    if len(feat_dim) != 1 or len(cfg.datasets.valid_names) != 1:
        raise RuntimeError("Expected one test graph with one feature width")
    model = instantiate(cfg.model, feat_dim=feat_dim.pop())
    state = torch.load(cfg.mechanism.ckpt, map_location="cpu", weights_only=False)["model"]
    # No dropping mismatched tensors, no newly initialized missing CCMP/scorer weights.
    model.load_state_dict(state, strict=True)
    if not any("resp_proj" in k for k in state):
        raise RuntimeError("Checkpoint has no CCMP responsibility head")
    del state
    loader = GraphDatasetLoader(cfg.datasets, cfg.datasets.valid_names, shuffle=False,
                               max_datasets_in_memory=1, data_loading_workers=0)
    losses = [SFTLoss(name=lc.name, loss_fn=instantiate(lc.loss), weight=lc.weight,
                      target_node_type=lc.target_node_type,
                      is_distillation_loss=lc.get("is_distillation_loss", False)) for lc in cfg.losses]
    trainer = instantiate(cfg.trainer, output_dir=HydraConfig.get().runtime.output_dir,
                          model=model, optimizer=instantiate(cfg.optimizer, model.parameters()),
                          loss_functions=losses, train_graph_dataset_loader=loader,
                          eval_graph_dataset_loader=loader)
    if trainer.dtype != getattr(torch, str(cfg.model.dtype)):
        raise RuntimeError("Actual precision differs from requested precision")
    cases = json.loads(Path(cfg.mechanism.cases).read_text())
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise RuntimeError("Cases must be nonempty with unique names")
    completed = []
    try:
        for dataset in loader:
            src = dataset.data
            manifest = fingerprint_inputs(cfg, src, cases)
            run_id = digest(manifest)
            out = Path(cfg.mechanism.out) / run_id[:20]
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(out / "manifest.json", manifest)
            graph = src.graph.to(trainer.device)
            if mode == "paths":
                documents = json.loads((Path(cfg.datasets.cfgs.root) / dataset.name / "raw/documents.json").read_text())
            pos = {str(src.test_data[i]["id"]): i for i in range(len(src.test_data))}
            for case in cases:
                if case["query_id"] not in pos:
                    raise RuntimeError(f"Pinned query missing: {case['query_id']}")
                token = digest(case)[:20]
                result_file, gate_file = out / f"case_{token}.json", out / f"gates_{token}.pt"
                cached = json.loads(result_file.read_text()) if result_file.exists() else None
                if (bool(cfg.mechanism.resume) and cached and cached.get("run_id") == run_id
                        and gate_file.exists() and cached.get("gate_sha256") == sha256_file(gate_file)):
                    print(f"[verified cache] {case['name']} {run_id[:20]}", flush=True)
                    completed.append(cached["result"])
                    continue
                item = src.test_data[pos[case["query_id"]]]
                batch = {k: item[k].unsqueeze(0).to(trainer.device)
                         for k in ("question_embeddings", "start_nodes_mask", "target_nodes_mask")}
                batch["id"] = [case["query_id"]]
                if mode == "paths":
                    result, gates = interpret_scientific_paths(
                        trainer.model, graph, batch, src, case, trainer.dtype, documents,
                        top_k=int(cfg.mechanism.top_k), beam_size=int(cfg.mechanism.beam_size))
                else:
                    result, gates = experiment(trainer.model, graph, batch, src, case, trainer.dtype,
                                               protection=str(cfg.mechanism.protection),
                                               atol=float(cfg.mechanism.atol), rtol=float(cfg.mechanism.rtol))
                tmp = gate_file.with_suffix(".tmp")
                torch.save(gates, tmp)
                tmp.replace(gate_file)
                atomic_json(result_file, {"run_id": run_id, "gate_sha256": sha256_file(gate_file), "result": result})
                completed.append(result)
            payload = {"manifest": manifest, "run_id": run_id, "status": "complete", "results": completed}
            atomic_json(out / "results.json", payload)
            if mode == "paths":
                render_scientific_table(payload, out)
            else:
                render_report(payload, out)
            atomic_json(Path(cfg.mechanism.out) / "latest.json",
                        {"version": VERSION, "run_id": run_id, "status": "complete", "results": str(out / "results.json")})
            print(f"[complete] {out / ('table4.md' if mode == 'paths' else 'report.md')}", flush=True)
        if len(completed) != len(cases):
            raise RuntimeError("Not all requested cases completed")
    finally:
        loader.shutdown()


if __name__ == "__main__":
    main()
