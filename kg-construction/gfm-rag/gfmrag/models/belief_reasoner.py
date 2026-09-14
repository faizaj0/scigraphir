"""
belief_reasoner.py — H3: semantic-prior graph reasoning.

There is no fusion here. The semantic scorer does not produce a ranking that is later
combined with a graph ranking; it produces the graph's INITIAL BELIEF over document
nodes, and the graph revises that belief:

    b^(0)_qd = z( s_sem(q, d) )                         semantic prior
    h^(0)_qd = node_embedding_qd + W_b · b^(0)_qd       injected at the document nodes
    h^(L)    = entity_model( graph, h^(0), ... )        query-conditioned propagation
    S(q, d)  = b^(0)_qd + R_theta( h^(L)_qd )           one residual belief update

WHY THIS IS NOT THE FUSION IN ANOTHER COSTUME. The graph's computation now DEPENDS on
the semantic belief: b^(0) enters at layer 0 and propagates, so a plausible paper can
raise the belief of a textually distant paper through a shared method or mechanism. In
the fusion the two channels were computed independently and only met at the end, so no
such path existed. The final `b^(0) + R(...)` looks additive, but the thing being added
is a function of b^(0) rather than a second opinion formed in ignorance of it.

WHY DOCUMENTS NEEDED SEEDING AT ALL. In the stock reasoner the only query-specific
input is `start_nodes_mask`, the linked query entities. Document nodes start at zero, so
the graph has to transport relevance across several hops before it can rank anything,
which is the hop2-to-hop3 cliff. Here every document starts at the best available
estimate and the graph only has to compute the correction.

STABILITY. `R_theta` is initialised near zero, so at step 0 the model IS the semantic
scorer (S ~= b^(0)) and it departs only where that reduces the retrieval loss. This is a
property of the architecture, not an external gate that has to learn to stay shut.

ABLATIONS, both of which this same class provides:
    use_prior=False   -> no injection and no residual: the stock graph reasoner
    use_graph=False   -> R_theta contributes nothing: the semantic scorer alone
"""
import torch
import torch.nn as nn

from gfmrag.models.fusion_reasoner import FusionGraphReasoner


class SemanticPriorReasoner(FusionGraphReasoner):
    def __init__(self, entity_model, feat_dim, belief_scale=1.0, readout_init=1e-3,
                 use_prior=True, use_graph=True, **kwargs):
        # The semantic channel must be the LEARNED scorer: this architecture is about
        # what the graph does with a good prior, and the handcrafted operator is not one.
        kwargs.setdefault("semantic", "mlp")
        super().__init__(entity_model, feat_dim, **kwargs)
        self.use_prior, self.use_graph = bool(use_prior), bool(use_graph)

        # Scalar belief -> hidden dimension. No bias: b=0 must inject nothing, so that a
        # document the scorer is neutral about is not nudged merely by being a document.
        self.belief_proj = nn.Linear(1, feat_dim, bias=False)
        nn.init.normal_(self.belief_proj.weight, std=float(belief_scale))

        # THE ENTITY MODEL DOES NOT RETURN feat_dim. With return_hidden it returns
        #     (sum(hidden_dims) if concat_hidden else hidden_dims[-1]) + input_dim
        # because the final states are concatenated with the per-node query embedding
        # (models.py: `torch.cat([hiddens[-1], node_query], dim=-1)`). For the shipped
        # config that is 1024 + 1024 = 2048, and a readout built at feat_dim dies with a
        # bare matmul shape error several minutes into the run. Derive it instead.
        self.readout_dim = self._entity_out_dim(entity_model, feat_dim)

        # The residual belief update. Near zero, NOT exactly zero: at exactly zero the
        # gradient into the graph is zero on step 1 (dL/dh = dL/dDelta * W_R = 0), so the
        # graph would sit idle for an iteration waiting for W_R to move off the origin.
        self.readout = nn.Linear(self.readout_dim, 1)
        nn.init.normal_(self.readout.weight, std=float(readout_init))
        nn.init.zeros_(self.readout.bias)

        # Injection and capture ride on hooks rather than a copy of GraphReasoner.forward,
        # so an upstream change to the engine cannot silently diverge from this file.
        self._pending = None      # (belief, doc_index) for the current forward
        self._h = None            # entity_model output, captured for the readout
        self.base.entity_model.register_forward_pre_hook(self._inject)
        self.base.entity_model.register_forward_hook(self._capture)

    @staticmethod
    def _entity_out_dim(entity_model, feat_dim):
        """Width of what the entity model returns under return_hidden, from its own dims."""
        dims = getattr(entity_model, "dims", None)
        if not dims or len(dims) < 2:
            return 2 * feat_dim          # the shipped QueryNBFNet shape, as a last resort
        input_dim, hidden = dims[0], list(dims[1:])
        base = sum(hidden) if getattr(entity_model, "concat_hidden", False) else hidden[-1]
        return int(base + input_dim)

    # ------------------------------------------------------------------ hooks
    def _inject(self, _module, args):
        """Add W_b · b to the document rows of the entity model's input."""
        if self._pending is None or not self.use_prior:
            return None
        b, doc = self._pending
        node_embedding = args[1]
        assert node_embedding.shape[-1] == self.belief_proj.out_features, (
            f"entity_model input width {node_embedding.shape[-1]} != belief projection "
            f"{self.belief_proj.out_features}; set feat_dim to match")
        # Cast the OUTPUT, not just the input: under AMP autocast this Linear emits
        # bfloat16 whatever it is fed, and index_add_ refuses a dtype mismatch.
        add = self.belief_proj(b.unsqueeze(-1).float()).to(node_embedding.dtype)
        node_embedding = node_embedding.index_add(1, doc, add)
        return (args[0], node_embedding, *args[2:])

    def _capture(self, _module, _args, output):
        self._h = output

    # ------------------------------------------------------------------ forward
    def forward(self, graph, batch, entities_weight=None):
        doc = graph.nodes_by_type["document"].to(self.belief_proj.weight.device)
        b0 = self._z(self._semantic(batch["id"], doc.numel(), doc.device))   # [B, n_doc]

        self._pending = (b0, doc)
        g = self.base(graph, batch, entities_weight)
        self._pending = None

        assert self._h is not None, "entity_model produced no hidden states to read"
        assert self._h.shape[-1] == self.readout_dim, (
            f"entity_model returned width {self._h.shape[-1]} but the readout was built "
            f"for {self.readout_dim}. Check hidden_dims / input_dim / concat_hidden in "
            f"the entity_model config, or set return_hidden: true.")
        if self.use_graph:
            delta = self.readout(self._h.index_select(1, doc).float()).squeeze(-1)
        else:
            delta = torch.zeros_like(b0)

        # THE SUM MUST HAPPEN IN FLOAT32, EXPLICITLY. Under AMP autocast every nn.Linear
        # emits bfloat16 no matter what dtype it was fed, so both the prior (through the
        # sorted-MLP) and the update (through the readout) arrive as bfloat16. Adding a
        # delta of order 1e-3 to a belief of order 1 in bfloat16, whose epsilon near 1.0
        # is 0.0078, discards the graph's entire contribution. Casting after the fact is
        # too late; the operands have to be widened before they meet.
        b0, delta = b0.float(), delta.float()
        S = b0 + delta

        # float32, for the reason the fusion needed it: at initialisation `delta` is far
        # smaller than one bfloat16 step near 1.0, so a bfloat16 write would round the
        # graph's entire contribution away and the architecture would look inert.
        out = g.clone().float()
        out[:, doc] = S

        self._doc_ids = doc
        # Negatives are mined from the SEMANTIC PRIOR, which is the right lineup: the
        # graph's job is to fix what the prior gets wrong.
        self._s_op = b0.detach()
        # Diagnostics only. There is no standalone graph ranking to train, so the
        # graph-alone auxiliary term is switched off and this is never used as a loss.
        self._raw_doc = delta.detach()
        self._raw_doc_z = None
        self._aux_ok = False
        # The analogue of gamma: how far the graph moved the belief, per query.
        self._gamma = delta.detach().abs().mean(-1)
        return out
