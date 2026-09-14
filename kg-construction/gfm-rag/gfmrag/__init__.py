from . import trainers

# GFMRetriever / GraphIndexer pull the construction + retrieval stack (entity-linking -> pylate,
# OpenIE/NER -> langchain, etc.). Training (sft_training) needs none of it, so keep `import gfmrag`
# working when those optional deps are absent.
try:
    from .gfmrag_retriever import GFMRetriever
    from .graph_indexer import GraphIndexer
except Exception:  # pragma: no cover - construction/retrieval deps optional
    GFMRetriever = GraphIndexer = None
