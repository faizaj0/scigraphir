"""Export the reported quantitative tables from the final thesis PDF.

Uses Poppler's pdftotext. The PDF fingerprint identifies the reviewed source;
this is a transcription workflow, separate from scoring experiment predictions.
Run from the repository root:
    python experiments/eval/export_thesis_results.py --pdf /path/to/FinalThesis.pdf
    python experiments/eval/export_thesis_results.py --pdf /path/to/FinalThesis.pdf --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

RESULTS = Path(__file__).resolve().parents[1] / "results"
SOURCE_SHA256 = "0b4325bcad90cb2444104f5eee621450a3a10903e1ca0406e013a8d477e853c4"
DECIMAL = re.compile(r"[-−]?\d+\.\d+%?")


def normalise(text):
    text = text.replace("S CI G RAPH IR", "SciGraphIR").replace("S CI A FFORD", "SciAfford")
    text = re.sub(r"\s*\[\d+\]", "", text)
    return " ".join(text.split()).replace("−", "-").replace("∗", "*")


def numeric_rows(page, count):
    rows = []
    for line in page.splitlines():
        matches = list(DECIMAL.finditer(line))
        if len(matches) != count:
            continue
        rows.append({"method": normalise(line[:matches[0].start()]),
                     "values": [normalise(m[0]) for m in matches],
                     "line": line, "matches": matches})
    return rows


def clean_rows(rows):
    return [{"method": r["method"], "values": r["values"]} for r in rows]


def read_pdf(pdf):
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("This PDF differs from the reviewed final thesis; verify the source before updating the fingerprint and page mapping.")
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                          check=True, capture_output=True, text=True).stdout
    pages = text.split("\f")
    main = clean_rows(numeric_rows(pages[58], 13))
    both = numeric_rows(pages[59], 14)
    transfer, external = [], []
    for row in both:
        transfer.append({"method": row["method"], "values": row["values"][:10]})
        right_label = normalise(row["line"][row["matches"][9].end():row["matches"][10].start()])
        external.append({"method": right_label, "values": row["values"][10:]})
    for row in transfer:
        if row["method"] in ("zero-shot", "in-field*"):
            row["method"] = "SciGraphIR " + row["method"]
    for row in external:
        if row["method"] in ("TOMATO-Star-trained", "SIR-4-trained"):
            row["method"] = "SciGraphIR " + row["method"]
    downstream = []
    for line in pages[60].splitlines():
        if not re.match(r"\s*(Closed-book|Qwen3|ReasonIR-8B|MOOSE-Chem|S CI G RAPH IR|Oracle)", line):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) == 9:
            downstream.append({"method": normalise(parts[0]),
                               "values": [normalise(v).replace("–", "--") for v in parts[1:]]})
    routing = clean_rows(numeric_rows(pages[92], 3))
    assert [len(x) for x in (main, transfer, external, downstream, routing)] == [17, 8, 8, 6, 5], "Unexpected PDF table structure"
    def table(title, page, columns, rows, caption, tex_path):
        assert all(len(r["values"]) == len(columns) for r in rows)
        return dict(title=title, pdf_page=page, columns=columns, rows=rows,
                    caption=caption, tex_path=tex_path)
    tables = [
        table("Main retrieval comparison", 59,
              ["TOMATO Same", "TOMATO Cross", "TOMATO Gap", "CS Same", "CS Cross",
               "Biology Same", "Biology Cross", "Physics Same", "Physics Cross",
               "MS Same", "MS Cross", "SIR-4 Macro Gap", "MIR All"], main,
              "Retrieval performance (nDCG@5, %). All methods marked † are evaluated on a stratified random sample of 500 queries to reduce LLM inference costs.",
              "main_table/main_results_thesis.tex"),
        table("Transfer between SIR-4 fields", 60,
              ["P+B train: CS Same", "P+B train: MS Same", "P+B train: CS Cross", "P+B train: MS Cross", "P+B train: Gap",
               "CS+MS train: Physics Same", "CS+MS train: Biology Same", "CS+MS train: Physics Cross", "CS+MS train: Biology Cross", "CS+MS train: Gap"], transfer,
              "Zero-shot transfer (nDCG@5, %). P+B means Physics + Biology. Training uses the named source fields and testing uses the remaining fields. The in-field* row is the in-domain reference printed in the thesis.",
              "zeroshot_baselines/table_zeroshot_thesis.tex"),
        table("Transfer to external benchmarks", 60,
              ["ResearchBench Same", "ResearchBench Cross", "ResearchBench Gap", "MIR All"], external,
              "Zero-shot transfer (nDCG@5, %). SciGraphIR trained on all SIR-4 fields or TOMATO-Star, evaluated on ResearchBench and MIR. Baselines are not fine-tuned.",
              "zeroshot_baselines/table_zeroshot_thesis.tex"),
        table("Downstream hypothesis quality", 61,
              ["TOMATO Top-1 hit", "TOMATO Overall", "TOMATO Same", "TOMATO Cross",
               "SIR-4 Top-1 hit", "SIR-4 Overall", "SIR-4 Same", "SIR-4 Cross"], downstream,
              "Downstream utility. Matched score (0-12) of hypotheses composed from the top-1 retrieved document on TOMATO and pooled SIR-4. Top-1 hit is the percentage of queries where the retrieved document is a gold inspiration.",
              "downstream/downstream_thesis.tex"),
        table("Routing comparison on TOMATO-Star", 93,
              ["Same", "Cross", "Gap"], routing,
              "Routing ablation on TOMATO-Star (nDCG@5, %). These are the routing comparison's reported runs.",
              "routing/routing_thesis.tex"),
    ]
    return {"source": {"filename": "FinalThesis.pdf", "sha256": digest,
                       "title": "Graph Reasoning for Cross-Domain Scientific Inspiration Retrieval",
                       "pdf_build_date": "2026-09-14", "page_numbering": "PDF pages, counted from the cover"},
            "value_format": "Strings preserve the precision and percentage signs printed in the PDF; -- means unavailable.",
            "source_notes": [
                "The PDF reports Biology cross-field as 53.21 in the main table and 51.57 in the transfer table's in-field reference. Both are retained as printed.",
                "The main and transfer reference rows also differ for Materials Science. Values and gaps are transcribed, not recomputed or reconciled.",
                "The routing comparison uses its own reported runs; its CCMP row is not substituted for the full-system row in the main comparison.",
                "The sampling statement in the main caption is reproduced from the PDF. Individual run records retain their original sampling and coverage metadata.",
            ], "tables": tables}


def tex_escape(text):
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
                    "†": r"$^{\dagger}$"}
    return "".join(replacements.get(c, c) for c in text)


def render(data):
    md = ["# Reported thesis results", "", "[Results index](README.md) · [Evaluation code](../eval/README.md)", "",
          "Source: **FinalThesis.pdf**, built 14 September 2026. The tables below reproduce the reported values, row labels, units and evaluation groupings.", "",
          "The [source record](reported_results.json) stores the PDF fingerprint, page locations and exact displayed values. These are reported scores; generating them does not rerun the experiments.", ""]
    outputs = {}; tex = {}
    for tab in data["tables"]:
        md += ["## " + tab["title"], "", tab["caption"], "",
               "| Method | " + " | ".join(tab["columns"]) + " |",
               "|---|" + "---:|" * len(tab["columns"])]
        md += ["| " + row["method"] + " | " + " | ".join(row["values"]) + " |" for row in tab["rows"]]
        md += ["", "[LaTeX table](" + tab["tex_path"] + ")", ""]
        lines = ["% Generated by experiments/eval/export_thesis_results.py from the reviewed final PDF.",
                 "% Requires booktabs and graphicx. This file does not recalculate experimental scores.",
                 r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
                 r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "r" * len(tab["columns"]) + "}",
                 r"\toprule", "Method & " + " & ".join(map(tex_escape, tab["columns"])) + r" \\", r"\midrule"]
        lines += [tex_escape(row["method"]) + " & " + " & ".join(map(tex_escape, row["values"])) + r" \\" for row in tab["rows"]]
        lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\caption{" + tex_escape(tab["caption"]) + "}", r"\end{table}", ""]
        tex.setdefault(tab["tex_path"], []).append("\n".join(lines))
    md += ["## Reading the source faithfully", ""] + ["- " + note for note in data["source_notes"]] + [""]
    outputs["THESIS_RESULTS.md"] = "\n".join(md)
    outputs["reported_results.json"] = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    outputs.update({path: "\n".join(parts) for path, parts in tex.items()})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="compare all generated files without writing")
    args = parser.parse_args()
    outputs = render(read_pdf(args.pdf))
    mismatches = []
    for relative, content in outputs.items():
        path = RESULTS / relative
        if args.check:
            if not path.exists() or path.read_text() != content:
                mismatches.append(relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if mismatches:
        raise SystemExit("Reported results differ from the PDF: " + ", ".join(mismatches))
    print(("Verified" if args.check else "Wrote") + f" {len(outputs)} result files from the final PDF.")


if __name__ == "__main__":
    main()
