from .base_model import BaseTextEmbModel
from .nv_embed import NVEmbedV2

# Qwen3TextEmbModel imports vllm at module load; vllm is an optional heavy backend (only needed if you
# actually use the vLLM-served Qwen3 embedder). Keep `import gfmrag` working without vllm installed.
try:
    from .qwen3_model import Qwen3TextEmbModel
except Exception:  # pragma: no cover - vllm optional
    Qwen3TextEmbModel = None

__all__ = ["BaseTextEmbModel", "NVEmbedV2", "Qwen3TextEmbModel"]
