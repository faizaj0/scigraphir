# Run all seven conditions plus repeat/replay checks, per fixed case and precision.
import copy
from pathlib import Path
from IPython.display import display, Markdown


RESULTS = []
for d in DATASETS:
    set_dataset(d)
    selection = _selection.select_checkpoint(d, S, DRIVE, DATA_ROOT, prefer_merged=PREFER_MERGED)
    print(f"[checkpoint] {selection['reason']}")
    print(f"[checkpoint] {selection['checkpoint']}\n[graph] {selection['graph']}\n[scorer] {selection['scorer_key']}")
    ckpt, graph, skey = selection["checkpoint"], selection["graph"], selection["scorer_key"]
    GRAPHS = [graph]
    S["sem"] = selection["semantic_specs"]
    selection_file = f"{RUNS}/ccmp_mechanism_selection_{d}.json"
    with open(selection_file, "w") as f:
        json.dump(selection, f, indent=2)
    prepare_components(GRAPHS)
    model_environment_values, info = model_environment(ckpt, graph, skey, gate=True)
    assert info["resp_keys"] and model_environment_values["CCMP_GATE"] == "1"
    selected_cases = copy.deepcopy([c for c in CASES if c["dataset"] == d])
    for case in selected_cases:
        # Original figure measurements came from the frame checkpoint/graph.
        # Keep them as provenance, but do not call merged results a reproduction failure.
        case["historical_reference_applicable"] = selection["family"] == "frame" and not case.get("selection_source")
    case_file = f"{RUNS}/ccmp_mechanism_cases_{d}.json"
    with open(case_file, "w") as f:
        json.dump(selected_cases, f, indent=2)
    for precision in PRECISIONS:
        output_root = f"{DRIVE}/outputs/ccmp_mechanism/{d}/{selection['family']}/{precision}/{PROTECTION}"
        run_dir = f"{RUNS}/ccmp_mechanism/{d}/{selection['family']}/{precision}/{PROTECTION}"
        os.makedirs(run_dir, exist_ok=True)
        command = [sys.executable, "-u", "-m", "gfmrag.workflow.ccmp_mechanism"]
        command += hydra_common(graph)
        command += [f"model.dtype={precision}", "datasets.data_loading_workers=0",
                    f"+mechanism.ckpt={ckpt}", f"+mechanism.cases={case_file}",
                    f"+mechanism.selection={selection_file}",
                    f"+mechanism.out={output_root}", f"+mechanism.protection={PROTECTION}",
                    f"+mechanism.resume={str(RESUME).lower()}",
                    f"+mechanism.atol={ATOL}", f"+mechanism.rtol={RTOL}",
                    f"hydra.run.dir={run_dir}"]
        sh(command, "/content/gfm-rag", extra=model_environment_values, log=f"{run_dir}/console.log")
        latest = json.load(open(f"{output_root}/latest.json"))
        assert latest["status"] == "complete" and latest["version"] == _core.VERSION
        result = json.load(open(latest["results"]))
        assert result["run_id"] == latest["run_id"] and result["manifest"]["cases"] == selected_cases
        assert result["manifest"]["precision"] == precision and result["status"] == "complete"
        assert result["manifest"]["selection"] == selection
        assert len(result["results"]) == len(selected_cases)
        RESULTS.append((d, precision, latest["results"], result))
        display(Markdown(Path(latest["results"]).with_name("report.md").read_text()))
        print("Results, CSV, full gate tensors and TikZ values:", str(Path(latest["results"]).parent))
    save_index(graph)

# --- combined precision comparison -------------------------------------------------
import pandas as pd

effect_rows = []
for dataset, precision, path, payload in RESULTS:
    for result in payload["results"]:
        for effect in result["effects"]:
            effect_rows.append({"dataset": dataset, "precision": precision, "case": result["case"]["name"],
                                "graph": payload["manifest"]["selection"]["graph"],
                                "checkpoint_family": payload["manifest"]["selection"]["family"],
                                "path": effect["path"], "native_delta_w": effect["native_minus_off"],
                                "gate_derivative_delta_w": effect["gate_derivative_effect"],
                                "outside_suppression_delta_w": effect["outside_suppression_effect"],
                                "suppression_result": effect["outside_suppression_verdict"],
                                "results_file": path})
effects = pd.DataFrame(effect_rows)
display(effects.drop(columns="results_file"))
for (case, path), group in effects.groupby(["case", "path"]):
    if len(group) > 1 and group.suppression_result.nunique() > 1:
        print(f"PRECISION-SENSITIVE: {case} / {path}; inspect both runs before making a figure claim.")
print("paths.csv contains weights; hops.csv has g_raw, g_applied and every edge gradient.")
print("The numerical screen is not a statistical test; these cases cannot establish a dataset-wide effect.")
