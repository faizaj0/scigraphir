"""
GFM-RAG v1 + node text features (G-reasoner-style reasoning core), on the GFM-RAG entity graph.

This is a NON-DESTRUCTIVE variant of GNNRetriever. It changes exactly two things vs the baseline:

  * Init (G-reasoner Eq 3): every node starts as its OWN projected text embedding (graph.x), plus
    the query embedding added at the query-seed entities. Baseline starts non-seed nodes at zero.

  * Readout (G-reasoner Eq 5): the reasoner returns per-node embeddings h^L (return_hidden=True),
    and a small predictor scores each node from [h^L, raw node text h_v]. The query h_q is already
    carried inside h^L (the reasoner concatenates node_query), so concatenating h_v completes the
    (h^L, h_v, h_q) triple. Baseline scores from h^L alone via the reasoner's built-in scalar head.

Everything else is inherited unchanged: DistMult message passing, sum aggregation, the BCE+ListCE
loss, and the IDF doc-ranker (map_entities_to_docs). We still score ENTITIES and aggregate to docs
with the ranker; we do NOT score document nodes directly (that is a further, G-reasoner-only step).

Requires datasets.cfgs.use_node_feat=True so that graph.x holds the node text embeddings.
"""

from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from .model import GNNRetriever


def _readout_in_features(module: nn.Module) -> int:
    """Return the input dimension of the reasoner's readout MLP (= dim of h^L node_feature)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    raise ValueError("No nn.Linear found in entity_model.mlp to infer h^L dimension.")


class GNNRetrieverNF(GNNRetriever):
    """GNNRetriever with node text features at the input (Eq 3) and the predictor (Eq 5)."""

    def __init__(
        self,
        entity_model: Any,
        feat_dim: int,
        ranker: Any,
        init_nodes_weight: bool = True,
        init_nodes_type: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(entity_model, feat_dim, ranker, init_nodes_weight, init_nodes_type)
        dim = self.entity_model.dims[0]                       # GNN input/boundary dimension
        # make the reasoner hand back per-node embeddings h^L instead of a scalar score
        self.entity_model.return_hidden = True
        d_out = _readout_in_features(self.entity_model.mlp)   # dimension of h^L (node_feature)
        # Eq 3: project raw node text (feat_dim) into the GNN boundary space (dim)
        self.node_mlp = nn.Linear(feat_dim, dim)
        # Eq 5: predict relevance from [h^L, raw node text h_v]  (h_q already inside h^L)
        self.predictor = nn.Sequential(
            nn.Linear(d_out + feat_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        graph: Data,
        batch: dict[str, torch.Tensor],
        entities_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert getattr(graph, "x", None) is not None, (
            "GNNRetrieverNF requires node features (set datasets.cfgs.use_node_feat=True)."
        )
        question_emb = batch["question_embeddings"]
        question_entities_mask = batch["start_nodes_mask"]
        question_embedding = self.question_mlp(question_emb)            # (bs, dim)
        batch_size = question_embedding.size(0)
        relation_representations = (
            self.rel_mlp(graph.rel_attr).unsqueeze(0).expand(batch_size, -1, -1)
        )

        if self.init_nodes_weight and entities_weight is None:
            assert self.init_nodes_type is not None, (
                "init_nodes_type must be set if init_nodes_weight is True and entities_weight is None"
            )
            entities_weight = self.get_entities_weight(
                graph.target_to_other_types[self.init_nodes_type]
            )
        if entities_weight is not None:
            question_entities_mask = question_entities_mask * entities_weight.unsqueeze(0)

        # ---- Eq 3: node text (projected) + query at the seed entities ----
        node0 = self.node_mlp(graph.x).unsqueeze(0).expand(batch_size, -1, -1)   # (bs, N, dim)
        seed = torch.einsum("bn, bd -> bnd", question_entities_mask, question_embedding)
        node_input = node0 + seed                                                # (bs, N, dim)

        # reasoner returns per-node embeddings h^L (return_hidden=True)
        h_L = self.entity_model(
            graph, node_input, relation_representations, question_embedding
        )                                                                        # (bs, N, d_out)

        # ---- Eq 5: predictor over [h^L, raw node text h_v] ----
        node_text = graph.x.unsqueeze(0).expand(batch_size, -1, -1)              # (bs, N, feat_dim)
        node_pred = self.predictor(torch.cat([h_L, node_text], dim=-1)).squeeze(-1)  # (bs, N)

        # entities -> docs via the IDF ranker (unchanged from baseline)
        return self.map_entities_to_docs(node_pred, graph)
