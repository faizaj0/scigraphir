# Discover and DISPLAY top paths from the selected current checkpoint.
import copy
from pathlib import Path
from IPython.display import display, Markdown

RESULTS = []
for d in DATASETS:
    set_dataset(d)
    selection = _selection.select_checkpoint(d, S, DRIVE, DATA_ROOT, prefer_merged=PREFER_MERGED)
    print(selection["reason"], "\nCheckpoint:", selection["checkpoint"], "\nGraph:", selection["graph"])
    ckpt, graph, skey = selection["checkpoint"], selection["graph"], selection["scorer_key"]
    GRAPHS = [graph]
    S["sem"] = selection["semantic_specs"]
    selection_file = f"{RUNS}/scientific_paths_selection_{d}.json"
    Path(selection_file).write_text(json.dumps(selection, indent=2))
    prepare_components(GRAPHS)
    model_environment_values, info = model_environment(ckpt, graph, skey, gate=True)
    assert info["resp_keys"]
    # Keep questions and relevant target papers; discard all old preselected paths.
    selected_cases = [{k: c[k] for k in ("name", "dataset", "query_id", "gold_id")}
                      for c in CASES if c["dataset"] == d]
    case_file = f"{RUNS}/scientific_paths_cases_{d}.json"
    Path(case_file).write_text(json.dumps(selected_cases, indent=2))
    for precision in PRECISIONS:
        output_root = f"{DRIVE}/outputs/scientific_paths/{d}/{selection['family']}/{precision}"
        run_dir = f"{RUNS}/scientific_paths/{d}/{precision}"
        os.makedirs(run_dir, exist_ok=True)
        command = [sys.executable, "-u", "-m", "gfmrag.workflow.ccmp_mechanism"] + hydra_common(graph)
        command += [f"model.dtype={precision}", "datasets.data_loading_workers=0",
                    "+mechanism.mode=paths", f"+mechanism.ckpt={ckpt}", f"+mechanism.cases={case_file}",
                    f"+mechanism.selection={selection_file}", f"+mechanism.out={output_root}",
                    f"+mechanism.resume={str(RESUME).lower()}", f"+mechanism.top_k={TOP_PATHS}",
                    f"+mechanism.beam_size={BEAM_SIZE}", "+mechanism.protection=none",
                    f"hydra.run.dir={run_dir}"]
        sh(command, "/content/gfm-rag", extra=model_environment_values, log=f"{run_dir}/console.log")
        latest = json.load(open(f"{output_root}/latest.json"))
        assert latest["status"] == "complete" and latest["version"] == _core.VERSION
        payload = json.load(open(latest["results"]))
        assert payload["run_id"] == latest["run_id"] and payload["manifest"]["analysis_mode"] == "paths"
        assert payload["manifest"]["cases"] == selected_cases and payload["manifest"]["selection"] == selection
        assert len(payload["results"]) == len(selected_cases)
        result_dir = Path(latest["results"]).parent
        display(Markdown((result_dir / "table4.md").read_text()))
        print("Table 4 LaTeX:", result_dir / "table4.tex")
        print("Editable TikZ:", result_dir / "top_paths.tikz.tex")
        print("Discovered paths to freeze for later CCMP tests:", result_dir / "discovered_cases_for_interventions.json")
        RESULTS.append((d, precision, payload))
    save_index(graph)

# Compact overview: the detailed paths are already displayed above.
import pandas as pd
overview = []
for d, precision, payload in RESULTS:
    for r in payload["results"]:
        overview.append({"case": r["case"]["name"], "query_field": r["query_field"],
                         "target_domain": ", ".join(r["target_domains"]), "stratum": r["benchmark_stratum"],
                         "graph_rank": r["channels"]["graph"]["rank"], "fused_rank": r["channels"]["fused"]["rank"],
                         "scorer_rank": r["channels"]["scorer"]["rank"],
                         "dense_rank": r["channels"]["dense"]["rank"] if r["channels"]["dense"] else None,
                         "top_paths_shown": len(r["top_paths"]), "multihop_paths_shown": len(r["top_multihop_paths"]),
                         "precision": precision, "graph": payload["manifest"]["selection"]["graph"]})
display(pd.DataFrame(overview))
