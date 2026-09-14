from .base_model import BaseELModel
# pylate/ColBERT is only needed if the ColBERT linker is actually selected;
# indexing here uses dpr_el_model, so keep `import gfmrag` working without pylate.
try:
    from .colbert_el_model import ColbertELModel
except Exception:
    ColbertELModel = None
from .dpr_el_model import DPRELModel, NVEmbedV2ELModel

__all__ = [
    "BaseELModel",
    "ColbertELModel",
    "DPRELModel",
    "NVEmbedV2ELModel",
]
