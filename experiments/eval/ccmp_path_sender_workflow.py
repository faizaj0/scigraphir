"""Installed as gfmrag.workflow.ccmp_path_sender by the Colab notebook. No training.

Same construction as ccmp_mechanism_workflow.py (strict checkpoint load, hashed manifest,
verified resume) but the per-case analysis is path_sender_experiment: top-K path discovery under
native CCMP, frozen replay of the recorded gates, and the per-path sender reset (Delta_P).
"""
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
from gfmrag.workflow.ccmp_mechanism_core import atomic_json, digest, sha256_file
from gfmrag.workflow.ccmp_path_sender_core import VERSION, path_sender_experiment, render_path_sender_report


def fingerprint_inputs(cfg, src, cases):
    root = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*.py"))}
    graph_name = str(cfg.datasets.valid_names[0])
    files = list(src.processed_graph) + [str(Path(src.processed_dir) / "test.pt"),
                                         str(Path(src.raw_dir) / "test.json"),
                                         str(Path(cfg.datasets.cfgs.root) / graph_name / "raw/documents.json")]
    selection = None
    if getattr(cfg.mechanism, "selection", None):
        selection = json.loads(Path(cfg.mechanism.selection).read_text())
        if selection["checkpoint"] != str(cfg.mechanism.ckpt) or selection["graph"] != graph_name:
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
        "analysis_mode": "path_senders", "checkpoint_sha256": sha256_file(cfg.mechanism.ckpt),
        "selection": selection, "engine_sha256": digest(source), "source_files": source, "inputs": inputs,
        "config": OmegaConf.to_container(cfg, resolve=True), "precision": str(cfg.model.dtype),
        "top_k": int(cfg.mechanism.top_k), "beam_size": int(cfg.mechanism.beam_size), "seed": int(cfg.seed),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "environment": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith(("CCMP", "ROUTE", "FUSION_", "SEM_", "SEED_", "CQIG"))},
        "attribution": "top-K simple paths of the finite gradient beam under native CCMP, fixed thereafter; "
                       "mean edge gradients of the graph gold score (legacy visualize convention); not a product of gates",
        "gate_policy": "record the native normalised gates; replay them frozen; per path, set the unique senders' gates "
                       "to 1 at every layer; no renormalisation, no recomputation from the changed states",
    }
    manifest["config"]["mechanism"].pop("resume", None)   # resume changes execution, not the experiment
    return manifest


@hydra.main(config_path="config/gfm_reasoner", config_name="sft_training_fusion", version_base=None)
def main(cfg: DictConfig):
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
    model.load_state_dict(state, strict=True)   # no dropped tensors, no freshly initialised heads
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
            documents = json.loads((Path(cfg.datasets.cfgs.root) / str(cfg.datasets.valid_names[0])
                                    / "raw/documents.json").read_text())
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
                result, gates = path_sender_experiment(
                    trainer.model, graph, batch, src, case, trainer.dtype, documents,
                    top_k=int(cfg.mechanism.top_k), beam_size=int(cfg.mechanism.beam_size),
                    atol=float(cfg.mechanism.atol), rtol=float(cfg.mechanism.rtol))
                tmp = gate_file.with_suffix(".tmp")
                torch.save(gates, tmp)
                tmp.replace(gate_file)
                atomic_json(result_file, {"run_id": run_id, "gate_sha256": sha256_file(gate_file), "result": result})
                completed.append(result)
            payload = {"manifest": manifest, "run_id": run_id, "status": "complete", "results": completed}
            atomic_json(out / "results.json", payload)
            render_path_sender_report(payload, out)
            atomic_json(Path(cfg.mechanism.out) / "latest.json",
                        {"version": VERSION, "run_id": run_id, "status": "complete", "results": str(out / "results.json")})
            print(f"[complete] {out / 'report.md'}", flush=True)
        if len(completed) != len(cases):
            raise RuntimeError("Not all requested cases completed")
    finally:
        loader.shutdown()


if __name__ == "__main__":
    main()
