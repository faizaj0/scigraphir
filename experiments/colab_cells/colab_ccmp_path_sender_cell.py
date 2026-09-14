# Run the path-sender gate effect per dataset and precision: discover the top-K paths under native CCMP
# (fixed thereafter), replay the recorded gates, reset each path's unique sender gates to 1 at every
# layer, measure Delta_P. Outputs land under outputs/ccmp_path_sender/<dataset>/<family>/<precision>/.
import copy
import shutil
from pathlib import Path
from IPython.display import display, Markdown

assert Path("/content/gfm-rag/gfmrag/workflow/ccmp_path_sender_core.py").is_file(), "run the source cells first"
RESULTS = []
for d in DATASETS:
    set_dataset(d)
    selection = _selection.select_checkpoint(d, S, DRIVE, DATA_ROOT, prefer_merged=PREFER_MERGED)
    print(f"[checkpoint] {selection['reason']}")
    print(f"[checkpoint] {selection['checkpoint']}\n[graph] {selection['graph']}\n[scorer] {selection['scorer_key']}")
    ckpt, graph, skey = selection["checkpoint"], selection["graph"], selection["scorer_key"]
    GRAPHS = [graph]
    S["sem"] = selection["semantic_specs"]
    selection_file = f"{RUNS}/ccmp_path_sender_selection_{d}.json"
    with open(selection_file, "w") as f:
        json.dump(selection, f, indent=2)
    prepare_components(GRAPHS)
    model_environment_values, info = model_environment(ckpt, graph, skey, gate=True)
    assert info["resp_keys"] and model_environment_values["CCMP_GATE"] == "1"
    selected_cases = copy.deepcopy([c for c in CASES if c["dataset"] == d])
    assert selected_cases, f"no cases for {d}"
    case_file = f"{RUNS}/ccmp_path_sender_cases_{d}.json"
    with open(case_file, "w") as f:
        json.dump(selected_cases, f, indent=2)
    for precision in PRECISIONS:
        output_root = f"{DRIVE}/outputs/ccmp_path_sender/{d}/{selection['family']}/{precision}"
        run_dir = f"{RUNS}/ccmp_path_sender/{d}/{selection['family']}/{precision}"
        os.makedirs(run_dir, exist_ok=True)
        command = [sys.executable, "-u", "-m", "gfmrag.workflow.ccmp_path_sender"]
        command += hydra_common(graph)
        command += [f"model.dtype={precision}", "datasets.data_loading_workers=0",
                    f"+mechanism.ckpt={ckpt}", f"+mechanism.cases={case_file}",
                    f"+mechanism.selection={selection_file}", f"+mechanism.out={output_root}",
                    f"+mechanism.resume={str(RESUME).lower()}", f"+mechanism.atol={ATOL}", f"+mechanism.rtol={RTOL}",
                    f"+mechanism.top_k={TOP_PATHS}", f"+mechanism.beam_size={BEAM_SIZE}",
                    f"hydra.run.dir={run_dir}"]
        sh(command, "/content/gfm-rag", extra=model_environment_values, log=f"{run_dir}/console.log")
        latest = json.load(open(f"{output_root}/latest.json"))
        assert latest["status"] == "complete" and latest["version"] == "ccmp-path-sender-v1"
        result = json.load(open(latest["results"]))
        assert result["run_id"] == latest["run_id"] and result["manifest"]["cases"] == selected_cases
        assert result["manifest"]["precision"] == precision and result["status"] == "complete"
        assert result["manifest"]["selection"] == selection
        assert len(result["results"]) == len(selected_cases)
        RESULTS.append((d, precision, latest["results"], result))
        run_out = Path(latest["results"]).parent
        display(Markdown((run_out / "report.md").read_text()))
        # flat copy for downloading: outputs/ccmp_path_sender/latest_<dataset>_<family>_<precision>/
        flat = Path(f"{DRIVE}/outputs/ccmp_path_sender/latest_{d}_{selection['family']}_{precision}")
        flat.mkdir(parents=True, exist_ok=True)
        for name in ("results.json", "report.md", "tables.tex", "summary.csv", "hops.csv", "fixed_paths.json", "manifest.json"):
            if (run_out / name).exists():
                shutil.copy(run_out / name, flat / name)
        print("Results, gate tensors and report:", run_out, "| flat copy:", flat)
    save_index(graph)

# --- combined summary
import pandas as pd

rows = []
for dataset, precision, path, payload in RESULTS:
    for rec in payload["results"]:
        on, off = rec["ranks"]["native"], rec["ranks"]["off"]
        for e in rec["effects"]:
            rows.append({"dataset": dataset, "precision": precision, "case": rec["case"]["name"], "path": e["path"],
                         "hops": len(e["hops"]), "attribution": round(e["attribution_native"], 3),
                         "delta_P": round(e["delta_P"], 3), "verdict": e["verdict"],
                         "graph_rank_on": on["graph"]["rank"], "graph_rank_off": off["graph"]["rank"],
                         "graph_rank_after_reset": e["graph_rank_reset"],
                         "graph_score_on": round(on["graph"]["score"], 3), "graph_score_off": round(off["graph"]["score"], 3),
                         "replay_ok": all(rec["checks"]["gate_replay"].values())})
table = pd.DataFrame(rows)
display(table)
for (case, path), group in table.groupby(["case", "path"]):
    if group.verdict.nunique() > 1:
        print(f"PRECISION-SENSITIVE: {case} / {path}: {dict(zip(group.precision, group.verdict))}; report float32 and say so.")
print("Delta_P is a path-sender gate effect (not the path's exclusive contribution); overlapping paths are not summable;")
print("a higher target score does not imply a better rank. These are selected examples, not a population estimate.")
print("Download outputs/ccmp_path_sender/latest_* to experiments/results/qualitative/ccmp_path_sender_effect/drive/.")
