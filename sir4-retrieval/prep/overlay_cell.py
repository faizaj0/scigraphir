"""overlay_cell.py -- the notebook cell that installs the CURRENT fusion sources (gfm_overlay.zip, embedded
as base64) over the engine the replayed setup cells wrote, and refuses to continue with an old interpret().
The zip is sir4-retrieval/gfm_overlay.zip: fusion_reasoner.py, cqig.py, ultra/models.py, ultra/layers.py,
fusion_trainer.py, base_trainer.py, sft_training_fusion.yaml, interpret_paths.py from kg-construction/gfm-rag."""
import base64
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = f"{ROOT}/gfm_overlay.zip"
NEEDED = ("def interpret(", "do_paths", "golds=None", "min_hops")


def cell() -> str:
    names = zipfile.ZipFile(ZIP).namelist()
    for n in ("gfmrag/trainers/fusion_trainer.py", "gfmrag/workflow/interpret_paths.py"):
        assert n in names, f"{ZIP} lacks {n}"
    ft = zipfile.ZipFile(ZIP).read("gfmrag/trainers/fusion_trainer.py").decode()
    assert all(k in ft for k in NEEDED), f"{ZIP} fusion_trainer.py is stale (needs {NEEDED})"
    blob = base64.b64encode(open(ZIP, "rb").read()).decode()
    return ("# 3b-bis. The CURRENT fusion sources, embedded (gfm_overlay.zip from the repo, built with the notebook).\n"
            "# The replayed cell above writes an older fusion_trainer.py whose interpret() lacks do_paths / golds /\n"
            "# min_hops: without this overlay the rank scan silently runs the slow beam search on every gold and\n"
            "# the hop figure has no structural floor. Nothing to upload; the zip travels inside this cell.\n"
            "import base64 as _b64, io as _io, zipfile as _zf\n"
            f"_ov = _zf.ZipFile(_io.BytesIO(_b64.b64decode(\"{blob}\")))\n"
            "_ov.extractall(\"/content/gfm-rag\"); print(\"overlay:\", _ov.namelist())\n"
            "_ft = open(\"/content/gfm-rag/gfmrag/trainers/fusion_trainer.py\").read()\n"
            f"_need = {NEEDED!r}\n"
            "assert all(k in _ft for k in _need), \"fusion_trainer.py lacks \" + str([k for k in _need if k not in _ft])\n"
            "print(\"engine interpret() is current: do_paths, pinned golds, min_hops\")\n")
