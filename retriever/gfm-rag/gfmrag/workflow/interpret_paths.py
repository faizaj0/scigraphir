"""
interpret_paths.py -- path interpretations for a trained fusion checkpoint.

Same construction as sft_training (config, datasets, model, trainer) with no training: the
checkpoint is loaded, then FusionSFTTrainer.interpret() runs the NBFNet-style gradient beam
search from each requested query's seed nodes to its best-ranked gold and records the CCMP
responsibility along every path. Output: one JSON.

    python -m gfmrag.workflow.interpret_paths --config-path config/gfm_reasoner \\
        --config-name sft_training_fusion text_emb_model=qwen3_st \\
        datasets.cfgs.root=... datasets.train_names=[G] datasets.valid_names=[G] \\
        model.semantic=mlp model.cqig=false \\
        +interp.ckpt=/path/model_best.pth +interp.qids_file=/path/qids.json \\
        +interp.out=/path/paths.json +interp.answers=/path/probes_test.jsonl \\
        hydra.run.dir=/path/run
"""
try:  # same torchvision shim as sft_training
    import torchvision.io as _tvio
    if not hasattr(_tvio, "VideoReader"):
        class _NoVideoReader:
            def __init__(self, *a, **k):
                raise RuntimeError("torchvision video API removed")
        _tvio.VideoReader = _NoVideoReader
except Exception:
    pass
import json
import logging
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gfmrag import utils
from gfmrag.graph_index_datasets import GraphDatasetLoader
from gfmrag.trainers.sft_trainer import SFTLoss

logger = logging.getLogger(__name__)


@hydra.main(config_path="config/gfm_rag", config_name="sft_training", version_base=None)
def main(cfg: DictConfig) -> None:
    utils.init_distributed_mode(cfg.timeout)
    torch.manual_seed(cfg.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    feat_dim = set(utils.init_multi_dataset(cfg, 1, 0))
    assert len(feat_dim) == 1
    model = instantiate(cfg.model, feat_dim=feat_dim.pop())
    ckpt = cfg.interp.ckpt
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    # an older checkpoint (the July v1 fusion) can carry a tensor of another shape (e.g. the 1-feature
    # gate); strict=False does not tolerate that, so such tensors are dropped and reported
    msd = model.state_dict()
    bad = [k for k, v in sd.items() if k in msd and tuple(msd[k].shape) != tuple(v.shape)]
    for k in bad:
        sd.pop(k)
    if bad:
        print(f"[interpret] dropped {len(bad)} tensors whose shape differs from this model: {bad[:5]}", flush=True)
    res = model.load_state_dict(sd, strict=False)
    print(f"[interpret] loaded {ckpt}: {len(sd)} tensors, missing {len(res.missing_keys)}, "
          f"unexpected {len(res.unexpected_keys)}", flush=True)
    if res.unexpected_keys:
        print("[interpret] unexpected keys (first 5):", res.unexpected_keys[:5], flush=True)
    valid_loader = GraphDatasetLoader(cfg.datasets, cfg.datasets.valid_names, shuffle=False,
                                      max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
                                      data_loading_workers=cfg.datasets.data_loading_workers)
    optimizer = instantiate(cfg.optimizer, model.parameters())
    loss_functions = [SFTLoss(name=lc.name, loss_fn=instantiate(lc.loss), weight=lc.weight,
                              target_node_type=lc.target_node_type,
                              is_distillation_loss=lc.get("is_distillation_loss", False))
                      for lc in cfg.losses]
    trainer = instantiate(cfg.trainer, output_dir=output_dir, model=model, optimizer=optimizer,
                          loss_functions=loss_functions, train_graph_dataset_loader=valid_loader,
                          eval_graph_dataset_loader=valid_loader)
    qids = json.load(open(cfg.interp.qids_file))
    golds = json.load(open(cfg.interp.golds_file)) if cfg.interp.get("golds_file") else None
    if str(cfg.interp.get("mode", "interpret")) == "gate_decomp":
        # which gates move a route's weight: same weights, the gate restricted per layer / per sender set
        trainer.gate_decomposition(qids, cfg.interp.out, golds=golds, num_beam=int(cfg.interp.get("num_beam", 10)),
                                   path_topk=int(cfg.interp.get("path_topk", 5)), max_golds=int(cfg.interp.get("max_golds", 2)))
    else:
        trainer.interpret(qids, cfg.interp.out, answers_path=cfg.interp.get("answers", cfg.interp.get("probes")),
                          num_beam=int(cfg.interp.get("num_beam", 10)), path_topk=int(cfg.interp.get("path_topk", 5)),
                          max_golds=int(cfg.interp.get("max_golds", 2)), top_views=int(cfg.interp.get("top_views", 3)),
                          do_paths=bool(int(cfg.interp.get("paths", 1))), golds=golds,
                          necessity=bool(int(cfg.interp.get("necessity", 0))), distractor=bool(int(cfg.interp.get("distractor", 0))),
                          dump_k=int(cfg.interp.get("dump_scores", 0)))
    valid_loader.shutdown()
    utils.synchronize()
    utils.cleanup()


if __name__ == "__main__":
    main()
