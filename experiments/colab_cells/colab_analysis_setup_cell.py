"""Readable Colab setup helpers for the scientific-path and CCMP notebooks.

This file is inserted directly into a notebook code cell. It is deliberately
ordinary Python: no encoded source, exec, eval, or generated code strings.
"""
import copy
import fnmatch
from pathlib import Path

import numpy as np
import torch

import scigraphir_paths as cp


def copy_new(source, destination, pattern="*"):
    """Copy files that are missing or have a different size."""
    if not os.path.isdir(source):
        return 0
    copied = 0
    for root, _, files in os.walk(source):
        relative_root = os.path.relpath(root, source)
        for filename in files:
            if not fnmatch.fnmatch(filename, pattern):
                continue
            source_file = f"{root}/{filename}"
            destination_file = f"{destination}/{relative_root}/{filename}"
            if (os.path.exists(destination_file)
                    and os.path.getsize(destination_file) == os.path.getsize(source_file)):
                continue
            os.makedirs(os.path.dirname(destination_file), exist_ok=True)
            shutil.copy(source_file, destination_file)
            copied += 1
    return copied


def restore_index(graph_name):
    source = f"{CACHE}/index/{graph_name}"
    if not os.path.isdir(source):
        print(f"  {graph_name}: no cached index on Drive (built on first use)")
        return
    copied = sum(copy_new(f"{source}/{directory}",
                          f"{DATA_ROOT}/{graph_name}/processed/{directory}")
                 for directory in os.listdir(source))
    print(f"  {graph_name}: index restored ({copied} files)")


def save_index(graph_name):
    processed = f"{DATA_ROOT}/{graph_name}/processed"
    for directory in os.listdir(processed):
        if directory != "stage1":
            copy_new(f"{processed}/{directory}", f"{CACHE}/index/{graph_name}/{directory}")


def operator_components(graph_name):
    return f"{DATA_ROOT}/{graph_name}/operator_components{OP_SLUG}.npz"


def semantic_components(graph_name):
    return f"{DATA_ROOT}/{graph_name}/semantic_components{OP_SLUG}.npz"


def semantic_components_are_current(path):
    if not os.path.exists(path):
        return False
    with np.load(path, allow_pickle=True) as saved:
        return os.path.exists(str(saved["h_path"])) and "qwen" in str(saved["encoder"]).lower()


def set_dataset(dataset_name):
    """Switch every scoped cache/path to one SIR-4 field."""
    global DATASET, S, CACHE, QUERIES, env
    DATASET = dataset_name
    S = copy.deepcopy(SPEC[dataset_name])
    CACHE = f"{DRIVE}/outputs/{dataset_name}/cache"
    QUERIES = f"{DATA_ROOT}/{dataset_name}_test/raw/test.json"
    os.makedirs(CACHE, exist_ok=True)
    for key in list(os.environ):
        if key.startswith(("CCMP", "ROUTE", "STRAT_", "CQIG", "RESID_", "MISS_W",
                           "FUSION_", "SEED_")):
            os.environ.pop(key)
    os.environ["SCIGRAPHIR_DATASET"] = dataset_name
    env = dict(os.environ, SCIGRAPHIR_ROOT=SCIGRAPHIR_ROOT, SCIGRAPHIR_DATASET=dataset_name,
               PYTHONUNBUFFERED="1")
    cp.set_dataset(dataset_name)
    print(cp.banner())


def prepare_components(graph_names):
    """Restore/build graph-aligned operator and semantic tables."""
    global EMB_LOCAL, SEMF
    started = time.time()
    EMB_LOCAL = f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb"
    restored = copy_new(f"{CACHE}/op_emb", EMB_LOCAL, "test_*")
    print(f"embedding cache: {restored} test-split files restored ({time.time()-started:.0f}s)")

    SEMF = {}
    for key, (relative, scorer_dataset) in S["sem"].items():
        source = f"{DRIVE}/{relative}"
        if key == "field":
            location = f"{S4}/results/semantic_{scorer_dataset}"
            os.makedirs(location, exist_ok=True)
            copy_new(source, location)
        else:
            location = source
        SEMF[key] = (
            f"{location}/params_semantic_mlp_fixedloss_{scorer_dataset}.json",
            f"{location}/popnet_semantic_mlp_fixedloss_{scorer_dataset}.pt",
        )
        for filename in SEMF[key]:
            assert os.path.exists(filename), f"scorer file missing: {filename}"
        with open(SEMF[key][0]) as stream:
            jmax = json.load(stream)["jmax"]
        print(f"  scorer '{key}': jmax {jmax} ({relative})")

    for graph_name in graph_names:
        restore_index(graph_name)
        operator_file = operator_components(graph_name)
        if not os.path.exists(operator_file):
            cached = f"{CACHE}/{graph_name}_operator_components{OP_SLUG}.npz"
            if os.path.exists(cached):
                shutil.copy(cached, operator_file)
            else:
                command = [
                    sys.executable, "-u",
                    "precompute/precompute_operator_components.py",
                    "--dataset", DATASET, "--graph", graph_name,
                    "--split", "test", "--model", OP_MODEL,
                ]
                sh(command, KGDIR)
                shutil.copy(operator_file, cached)

        semantic_file = semantic_components(graph_name)
        if not semantic_components_are_current(semantic_file):
            command = [
                sys.executable, "-u",
                "precompute/precompute_semantic_components.py",
                "--dataset", DATASET, "--model", OP_MODEL,
                "--graph", graph_name, "--split", "test",
            ]
            sh(command, KGDIR)
        with np.load(semantic_file, allow_pickle=True) as saved:
            shape = tuple(int(value) for value in saved["h_shape"])
            jmax = int(saved["Jmax"])
        print(f"  {graph_name}: H {shape}, Jmax={jmax}")

    written = copy_new(EMB_LOCAL, f"{CACHE}/op_emb", "test_*")
    print(f"{written} new embedding files written back to Drive")


def inspect_checkpoint(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    responsibility = [key for key in state if "resp_" in key]
    semantic = [key for key in state if key.startswith("sem_")]
    hidden = next((int(state[key].shape[0]) for key in responsibility
                   if key.endswith("resp_proj.0.weight")), None)
    jmax = next((int(state[key].shape[1]) - 2 for key in semantic
                 if key.endswith("sem_net.0.weight")), None)
    return {"tensors": len(state), "resp_keys": len(responsibility),
            "sem_keys": len(semantic), "ccmp_hid": hidden, "jmax": jmax}


def model_environment(checkpoint, graph_name, scorer_key, gate=True):
    """Construct the exact environment expected by FusionGraphReasoner."""
    info = inspect_checkpoint(checkpoint)
    semantic_checkpoint, semantic_popnet = SEMF[scorer_key]
    with open(semantic_checkpoint) as stream:
        scorer = json.load(stream)
    assert int(scorer["jmax"]) == info["jmax"], (
        f"scorer width mismatch: checkpoint jmax={info['jmax']} vs "
        f"scorer '{scorer_key}' jmax={scorer['jmax']}")
    values = {
        "WANDB_MODE": "disabled",
        "HYDRA_FULL_ERROR": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OPERATOR_COMPONENTS": operator_components(graph_name),
        "OPERATOR_COMPONENTS_TEST": operator_components(graph_name),
        "SEMANTIC_COMPONENTS": semantic_components(graph_name),
        "SEMANTIC_COMPONENTS_TEST": semantic_components(graph_name),
        "SEMANTIC_CKPT": semantic_checkpoint,
        "SEMANTIC_POPNET": semantic_popnet,
        "SEM_POP_LAMBDA": "1.0",
        "FUSION_OBJECTIVE": "hardneg",
        "HARDNEG_HUB": "50",
        "HARDNEG_RAND": "50",
        "AUX_W": "1.0",
        "PER_GOLD": "1",
        "HARDNEG_GRAPH": "50",
        "STRAT_TEST": QUERIES,
    }
    assert info["resp_keys"], "selected checkpoint has no CCMP responsibility head"
    values.update(CCMP="1", CCMP_HID=str(info["ccmp_hid"]),
                  CCMP_GATE="1" if gate else "0", CCMP_GATE_NORM="1", CCMP_ETA="0.5")
    arm_file = f"{os.path.dirname(checkpoint)}/arm.json"
    if os.path.exists(arm_file):
        with open(arm_file) as stream:
            arm = json.load(stream)
        if arm.get("ccmp_eta") is not None:
            values["CCMP_ETA"] = str(arm["ccmp_eta"])
    return values, info


def hydra_common(graph_name):
    return [
        "--config-path", "config/gfm_reasoner",
        "--config-name", "sft_training_fusion",
        "text_emb_model=qwen3_st",
        f"datasets.cfgs.root={DATA_ROOT}",
        "datasets.cfgs.force_reload=False",
        f"datasets.train_names=[{graph_name}]",
        f"datasets.valid_names=[{graph_name}]",
        "model.semantic=mlp",
        "model.cqig=false",
    ]
