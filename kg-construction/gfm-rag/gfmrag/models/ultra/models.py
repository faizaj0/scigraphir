# mypy: ignore-errors
import os
import torch
from torch import autograd, nn

from . import layers
from .base_nbfnet import BaseNBFNet


class EntityNBFNet(BaseNBFNet):
    """Neural Bellman-Ford Network for entity prediction."""

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):
        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        self.return_hidden = kwargs.get("return_hidden", False)
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    self.dims[0],
                    self.message_func,
                    self.aggregate_func,
                    self.layer_norm,
                    self.activation,
                    dependent=False,
                    project_relations=True,
                )
            )

        feature_dim = (
            sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]
        ) + input_dim
        if not self.return_hidden:
            self.mlp = nn.Sequential()
            mlp = []
            for i in range(self.num_mlp_layers - 1):
                mlp.append(nn.Linear(feature_dim, feature_dim))
                mlp.append(nn.ReLU())
            mlp.append(nn.Linear(feature_dim, 1))
            self.mlp = nn.Sequential(*mlp)

    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(
            batch_size, data.num_nodes, self.dims[0], device=h_index.device
        )
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(
            -1, data.num_nodes, -1
        )  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch):
        h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # if self.training:
        # Edge dropout in the training mode
        # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
        # to make NBFNet iteration learn non-trivial paths
        # data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        )
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(
            data, h_index[:, 0], r_index[:, 0]
        )  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(
            1, index
        )  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)


class QueryNBFNet(EntityNBFNet):
    """
    The entity-level reasoner for UltraQuery-like complex query answering pipelines.

    This class extends EntityNBFNet to handle query-specific reasoning in knowledge graphs.
    Key differences from EntityNBFNet include:

    1. Initial node features are provided during forward pass rather than read from triples batch
    2. Query comes from outer loop
    3. Returns distribution over all nodes (assuming t_index covers all nodes)

    Attributes:
        layers: List of neural network layers for message passing
        short_cut: Boolean flag for using residual connections
        concat_hidden: Boolean flag for concatenating hidden states
        mlp: Multi-layer perceptron for final scoring
        num_beam: Beam size for path search
        path_topk: Number of top paths to return

    Methods:
        bellmanford(data, node_features, query, separate_grad=False):
            Performs Bellman-Ford message passing iterations.
            Args:
                data: Graph data object containing edge information
                node_features: Initial node representations
                query: Query representation
                separate_grad: Whether to track gradients separately for edges
            Returns:
                dict: Contains node features and edge weights

        forward(data, node_features, relation_representations, query):
            Main forward pass of the model.
            Args:
                data: Graph data object
                node_features: Initial node features
                relation_representations: Representations for relations
                query: Query representation
            Returns:
                torch.Tensor: Scores for each node

        visualize(data, sample, node_features, relation_representations, query):
            Visualizes reasoning paths for given entities.
            Args:
                data: Graph data object
                sample: Dictionary containing entity masks
                node_features: Initial node features
                relation_representations: Representations for relations
                query: Query representation
            Returns:
                dict: Contains paths and weights for target entities
    """

    def _route_attention(self, li, h, query, layer, data, edge_weight):
        """RED-GNN-style query-conditioned attention, one weight per (query, edge).

        logit[b, e] = a . tanh(W_n h_b[src(e)] + W_r r_b[type(e)] + W_q q_b + emb[layer])
        alpha       = softmax of logit over every receiver's INCOMING edges, per query
        weight      = alpha * in_degree(receiver)        (ROUTE_ATTN_NORM=1, default)

        The degree factor makes the weights average 1 over each receiver's incoming
        edges, so the aggregation keeps the control's scale and a uniform attention is
        exactly the ungated layer (the output layer is zero-initialised, so that is
        epoch 0). ROUTE_ATTN_NORM=0 gives RED-GNN's raw softmax, which also changes the
        aggregation from a sum to a weighted mean and is therefore a second variable.
        Trained by the ranking loss only. Runs in fp32 like the CCMP head.
        """
        from torch_geometric.utils import softmax as _pyg_softmax
        src, dst = data.edge_index[0], data.edge_index[1]
        B = h.shape[0]
        with torch.autocast(device_type=h.device.type, enabled=False):
            zn = self.attn_node[li](h.float())                          # [B, N, H]
            rel = layer.relation
            if isinstance(rel, nn.Embedding):
                rel = rel.weight
            if rel.dim() == 2:
                rel = rel.unsqueeze(0).expand(B, -1, -1)
            zr = self.attn_rel(rel.float())                             # [B, R, H]
            zq = self.attn_query(query.float()).unsqueeze(1)            # [B, 1, H]
            ze = (zn.index_select(1, src) + zr.index_select(1, data.edge_type) + zq
                  + self.attn_emb.weight[li].float())                   # [B, E, H]
            logit = self.attn_out(torch.tanh(ze)).squeeze(-1)           # [B, E]
            alpha = _pyg_softmax(logit.t().contiguous(), dst, num_nodes=data.num_nodes)  # [E, B]
            if getattr(self, "attn_norm", True):
                deg = torch.bincount(dst, minlength=data.num_nodes).to(alpha.dtype)
                alpha = alpha * deg.index_select(0, dst).unsqueeze(-1)
            w = alpha.t().contiguous()                                  # [B, E]
            w = w * edge_weight.float().unsqueeze(0)
        return w

    def bellmanford(self, data, node_features, query, separate_grad=False):
        import torch.distributed as dist

        dist_context = getattr(data, "dist_context", None)
        is_dist = dist_context is not None and dist.is_initialized()
        boundary_mode = getattr(data, "boundary_mode", False)

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=query.device, dtype=query.dtype)

        hiddens = []
        edge_weights = []

        if is_dist and boundary_mode:
            # ------------------------------------------------------------------
            # Boundary-only AllGather (METIS partition)
            # ------------------------------------------------------------------
            # Scalable variant: only communicate boundary source node states
            # instead of the full hidden tensor.  The compact tensor layout is
            # [local_nodes | boundary_nodes] with edges pre-remapped to this
            # space by partition_graph_metis().
            # ------------------------------------------------------------------
            rank, world_size = dist_context
            num_nodes = data.num_nodes
            local_nodes = data.local_nodes
            boundary_nodes = data.boundary_nodes
            compact_size = data.compact_size
            node2part = data.node2part
            local_N = local_nodes.shape[0]
            boundary_N = boundary_nodes.shape[0]

            # Collect local_nodes from all ranks for scatter-back.
            all_local_nodes_list = [None] * world_size
            dist.all_gather_object(all_local_nodes_list, local_nodes.cpu())
            all_local_N = [len(ns) for ns in all_local_nodes_list]
            max_local_N = max(all_local_N)

            def _scatter_allgather(local_t):
                """AllGather local outputs and scatter to global positions."""
                B, loc_n, D = local_t.shape
                if loc_n < max_local_N:
                    pad = local_t.new_zeros(B, max_local_N - loc_n, D)
                    padded = torch.cat([local_t, pad], dim=1).contiguous()
                else:
                    padded = local_t.contiguous()
                chunks = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(chunks, padded)
                # Scatter each rank's results to correct global positions.
                output = local_t.new_zeros(B, num_nodes, D)
                for r in range(world_size):
                    r_nodes = all_local_nodes_list[r].to(local_t.device)
                    r_data = chunks[r][:, : all_local_N[r], :]
                    output[:, r_nodes, :] = r_data
                return output

            # Precompute boundary exchange info (once per forward).
            boundary_owners = node2part[boundary_nodes]
            all_boundary_nodes_list = [None] * world_size
            dist.all_gather_object(all_boundary_nodes_list, boundary_nodes.cpu())

            local_nodes_set = set(local_nodes.cpu().tolist())
            send_indices = {}
            for r in range(world_size):
                if r == rank:
                    continue
                their_boundary = set(all_boundary_nodes_list[r].tolist())
                needed_from_us = their_boundary & local_nodes_set
                if needed_from_us:
                    needed_t = torch.tensor(sorted(needed_from_us), device=query.device)
                    idx = torch.searchsorted(local_nodes, needed_t)
                    send_indices[r] = idx

            recv_indices = {}
            for r in range(world_size):
                if r == rank:
                    continue
                mask = boundary_owners == r
                if mask.any():
                    recv_indices[r] = mask.nonzero(as_tuple=True)[0]

            # Initial local hidden states and compact boundary condition.
            local_layer_input = node_features[:, local_nodes, :].clone()
            compact_boundary = torch.cat([
                node_features[:, local_nodes, :],
                node_features[:, boundary_nodes, :],
            ], dim=1)

            compact_edge_size = (compact_size, compact_size)

            for layer in self.layers:
                if separate_grad:
                    edge_weight = edge_weight.clone().requires_grad_()

                # Exchange boundary states.
                B, _, D = local_layer_input.shape
                boundary_hidden = local_layer_input.new_zeros(B, boundary_N, D)

                send_data = {}
                for r, idx in send_indices.items():
                    send_data[r] = local_layer_input[:, idx, :].contiguous()

                all_send_data = [None] * world_size
                dist.all_gather_object(all_send_data, send_data)

                for r in range(world_size):
                    if r == rank:
                        continue
                    if rank in all_send_data[r]:
                        states = all_send_data[r][rank].to(query.device)
                        boundary_hidden[:, recv_indices[r], :] = states

                compact_input = torch.cat(
                    [local_layer_input, boundary_hidden], dim=1
                )

                hidden = layer(
                    compact_input,
                    query,
                    compact_boundary,
                    data.edge_index,
                    data.edge_type,
                    compact_edge_size,
                    edge_weight,
                )

                local_hidden = hidden[:, :local_N, :]
                if self.short_cut and local_hidden.shape == local_layer_input.shape:
                    local_hidden = local_hidden + local_layer_input
                hiddens.append(local_hidden)
                edge_weights.append(edge_weight)
                local_layer_input = local_hidden

            node_query_local = (
                query.unsqueeze(1).expand(-1, local_N, -1).contiguous()
            )
            if self.concat_hidden:
                local_output = torch.cat(hiddens + [node_query_local], dim=-1)
            else:
                local_output = torch.cat([hiddens[-1], node_query_local], dim=-1)

            output = _scatter_allgather(local_output)

        elif is_dist:
            # ------------------------------------------------------------------
            # Distributed split-graph inference
            # ------------------------------------------------------------------
            # Strategy (mathematically exact):
            #   1. Each rank owns nodes [local_start, local_end) and the edges
            #      whose *target* falls in that slice (set by partition_graph_edges).
            #   2. Before each layer: AllGather local hidden states â†’ full (B,N,D).
            #   3. Run the layer with the full source states but local-only edges.
            #      The layer output is correct at local target positions; non-local
            #      positions contain boundary-only values and are discarded.
            #   4. Slice to local portion, apply residual, store.
            #   5. After all layers: AllGather the local concatenated output once
            #      to reconstruct the full result on every rank.
            # ------------------------------------------------------------------
            rank, world_size = dist_context
            num_nodes = data.num_nodes
            base_N = (num_nodes + world_size - 1) // world_size  # ceiling division
            local_start = rank * base_N
            local_end = min((rank + 1) * base_N, num_nodes)
            local_N = local_end - local_start

            # Collect each rank's actual local_N once (for uneven last partition).
            local_N_t = torch.tensor(local_N, device=query.device)
            all_local_N_list = [torch.zeros_like(local_N_t) for _ in range(world_size)]
            dist.all_gather(all_local_N_list, local_N_t)
            all_local_N = [int(x.item()) for x in all_local_N_list]
            max_local_N = max(all_local_N)  # == base_N

            def _allgather(local_t: torch.Tensor) -> torch.Tensor:
                """AllGather (B, local_N, D) across ranks into (B, N, D).

                Handles the case where the last rank may have fewer nodes than
                the others by zero-padding to max_local_N before gathering,
                then slicing each chunk to its actual size before concatenation.
                """
                B, loc_n, D = local_t.shape
                if loc_n < max_local_N:
                    pad = local_t.new_zeros(B, max_local_N - loc_n, D)
                    padded = torch.cat([local_t, pad], dim=1).contiguous()
                else:
                    padded = local_t.contiguous()
                chunks = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(chunks, padded)
                return torch.cat(
                    [chunks[r][:, : all_local_N[r], :] for r in range(world_size)],
                    dim=1,
                )  # (B, N, D)

            # Local slice of the initial boundary / layer input.
            local_layer_input = node_features[:, local_start:local_end, :].clone()

            for layer in self.layers:
                if separate_grad:
                    edge_weight = edge_weight.clone().requires_grad_()

                # AllGather â†’ full source-node states on each rank.
                global_input = _allgather(local_layer_input)  # (B, N, D)

                # Layer forward with full input but local-target edges.
                # hidden[v] is correct for v in [local_start, local_end);
                # non-local positions contain boundary-only values (discarded).
                hidden = layer(
                    global_input,
                    query,
                    node_features,  # boundary: full (B, N, D), same on all ranks
                    data.edge_index,
                    data.edge_type,
                    size,
                    edge_weight,
                )

                # Slice to local, apply residual, then store local hidden.
                local_hidden = hidden[:, local_start:local_end, :]  # (B, local_N, D)
                if self.short_cut and local_hidden.shape == local_layer_input.shape:
                    local_hidden = local_hidden + local_layer_input
                hiddens.append(local_hidden)
                edge_weights.append(edge_weight)
                local_layer_input = local_hidden

            # Concatenate local hidden slices (+ local node_query) then AllGather.
            node_query_local = (
                query.unsqueeze(1).expand(-1, local_N, -1).contiguous()
            )  # (B, local_N, input_dim)
            if self.concat_hidden:
                local_output = torch.cat(hiddens + [node_query_local], dim=-1)
            else:
                local_output = torch.cat([hiddens[-1], node_query_local], dim=-1)
            # local_output: (B, local_N, out_dim)

            output = _allgather(local_output)  # (B, N, out_dim)

        else:
            # ------------------------------------------------------------------
            # Standard single-process path (unchanged)
            # ------------------------------------------------------------------
            layer_input = node_features
            # --- CCMP: contrastive continuation message passing -------------------
            # A node's predicted responsibility scales its OUTGOING messages. Scaling
            # `layer_input` does exactly that: every message leaving v is computed from
            # layer_input[v]. `node_features` is left alone on purpose -- that is the
            # query's own seed injection (the boundary condition), not a routing choice.
            #
            # THE GATE IS MEAN-NORMALISED, and that is a deliberate departure from
            # d + (1-d)*yhat as specified. That form lies in [d, 1], so it can only
            # ATTENUATE, and the factors compound over L layers: even a perfect gate
            # emitting 0.9 everywhere leaves a 6-hop route at 0.53 and a 3-hop route at
            # 0.73. That is a short-path prior applied on top of a model whose measured
            # failure is a hop2->hop3 rank cliff, i.e. it pushes the wrong way. Dividing
            # by the mean keeps the relative selectivity, which is the whole mechanism,
            # and drops the depth-dependent global shrink, which is an artefact.
            # CCMP_GATE_NORM=0 restores the attenuating form for the ablation.
            _rp = []
            _rp_stat = []
            _heads = getattr(self, "resp_proj", None)
            # STRUCTURAL REACH, not "the state is nonzero". Early-late entity fusion puts a
            # static text embedding on EVERY node before propagation, so a nonzero-state
            # test is true almost everywhere from layer 1 and the "activated set" was in
            # practice the whole graph. Then the gate's mean was a graph-wide mean and the
            # normalisation said nothing about what the query had reached.
            _reach = None
            _seed = getattr(self, "_ccmp_seeds", None)
            # Path interpretation (trainer.interpret) asks for the per-layer frontier so a
            # node's responsibility can be read against the frontier mean it was normalised
            # by. Off by default; costs one [B, N] clone per layer when on.
            self._reach_layers = []
            # GATE ATTRIBUTION (interpretation only): keep each layer's gate as a leaf so the trainer can
            # take d score / d g per node and layer; the responsibility head is detached on purpose, the
            # gate is treated as the intervention variable.
            self._gate_tensors = []
            if _heads is not None and _seed is not None:
                if getattr(self, "_ccmp_adj_key", None) != id(data):
                    _ei = data.edge_index
                    _r2 = torch.cat([_ei[0], _ei[1]])
                    _c2 = torch.cat([_ei[1], _ei[0]])
                    self._ccmp_adj = torch.sparse_coo_tensor(
                        torch.stack([_r2, _c2]),
                        torch.ones(_r2.numel(), device=_ei.device),
                        (data.num_nodes, data.num_nodes)).coalesce()
                    self._ccmp_adj_key = id(data)
                _reach = (_seed.to(layer_input.device) > 0).float()
            for _li, layer in enumerate(self.layers):
                if getattr(self, "_keep_reach", False):
                    self._reach_layers.append(None if _reach is None else _reach.detach().clone())
                if _heads is not None:
                    # Pinning the weights to fp32 is not sufficient on its own:
                    # under AMP an nn.Linear emits bf16 whatever its parameter dtype.
                    # The projection stays autocast (big matmul into a 64-dim
                    # bottleneck); the trunk and the sigmoid run in fp32, which is where
                    # the gate's precision actually lives -- bf16's ULP near 1.0 is
                    # 0.0039 and the gate only spans about [0.5, 1.5] at eta=0.5.
                    _z = (_heads[_li](layer_input).float()
                          + self.resp_emb.weight[_li].float())
                    with torch.autocast(device_type=_z.device.type, enabled=False):
                        _yh = torch.sigmoid(self.resp_head(_z).squeeze(-1))  # [B, N]
                    _rp.append(_yh)
                    if getattr(self, "route_mode", "") == "astar":
                        # A*Net-STYLE HARD ROUTING. Keep the top-K reached nodes by the
                        # head's priority and drop the rest of the frontier; kept nodes
                        # are weighted by priority / mean(priority over kept) so the head
                        # receives a gradient from the ranking loss (A*Net multiplies
                        # selected messages by the priority for the same reason) and so
                        # the selection is mean-preserving like CCMP's gate. yhat starts
                        # at 0.5 everywhere, so at initialisation this is a pure top-K.
                        # Off the frontier the gate is 1, exactly as for CCMP.
                        _k = int(getattr(self, "route_k", 1024))
                        _act = (_reach if _reach is not None
                                else torch.ones_like(_yh)).to(_yh.dtype)
                        _sc = torch.where(_act.bool(), _yh, torch.full_like(_yh, -1.0))
                        _top = _sc.topk(min(_k, _sc.shape[-1]), dim=-1).indices
                        _keep = torch.zeros_like(_yh).scatter_(-1, _top, 1.0) * _act
                        _num = _yh * _keep
                        _den = (_num.sum(-1, keepdim=True)
                                / _keep.sum(-1, keepdim=True).clamp(min=1.0)).clamp(min=1e-6)
                        _gt = torch.where(_keep.bool(), _num / _den, torch.zeros_like(_yh))
                        if _reach is not None:
                            _gt = torch.where(_reach.bool(), _gt, torch.ones_like(_gt))
                        with torch.no_grad():
                            _sel = _gt[_keep.bool()]
                            if _sel.numel():
                                _rp_stat.append((float(_sel.mean()), float(_sel.max()),
                                                 float(_sel.quantile(0.95))))
                        _msg = layer_input * _gt.unsqueeze(-1).to(layer_input.dtype)
                    elif getattr(self, "resp_gate", False):
                        _eta = getattr(self, "resp_eta", 1.0)
                        _num = 1e-6 + _yh
                        if getattr(self, "resp_gate_norm", True):
                            # NORMALISE FIRST, THEN INTERPOLATE. mean(ybar) = 1 by
                            # construction, so mean(g) = (1-eta) + eta = 1 for EVERY eta:
                            # the strength knob and the mean-preservation are independent,
                            # and eta = 0 recovers the ungated GNN exactly. Doing it the
                            # other way round (floor, then divide by the mean) also gives
                            # mean 1 but makes the floor a range-compression whose meaning
                            # changes with the spread of yhat.
                            #
                            # The mean is over the ACTIVATED set: nodes the query has
                            # actually reached at this layer. Averaging over all ~60k nodes
                            # would divide by a number dominated by unreached nodes, and
                            # the gate would scale with how far propagation has spread
                            # rather than with what it found.
                            _act = (_reach if _reach is not None
                                    else (layer_input.detach().abs().sum(-1) > 0).float()
                                    ).to(_num.dtype)
                            _den = ((_num * _act).sum(-1, keepdim=True)
                                    / _act.sum(-1, keepdim=True).clamp(min=1.0))
                            _num = _num / _den.clamp(min=1e-6)
                        _gt = (1.0 - _eta) + _eta * _num
                        # GATE ONLY WHAT IS SUPERVISED. The loss trains yhat on A^(l)
                        # alone, so off the frontier the prediction is whatever the head
                        # happens to emit -- and under early fusion those nodes still hold
                        # static text embeddings and still send messages. Applying an
                        # unsupervised gate to them lets CCMP perturb the graph in a
                        # direction no gradient ever checked. Gate 1 there leaves the
                        # background computation exactly as the control computes it.
                        if _reach is not None:
                            _gt = torch.where(_reach.bool(), _gt, torch.ones_like(_gt))
                        # GATE DECOMPOSITION (interpretation only, never set in training): restrict
                        # the gate to a set of layers and/or a set of sender nodes, gate 1 elsewhere.
                        # trainer.gate_decomposition() sets these to ask which gates move a route's
                        # weight: the ones on the route's own senders, at the attributed layers, or
                        # the ones on everybody else.
                        _glm = getattr(self, "_gate_layers", None)
                        if _glm is not None and _li not in _glm:
                            _gt = torch.ones_like(_gt)
                        _gnm = getattr(self, "_gate_nodes", None)
                        if _gnm is not None:
                            _gt = torch.where(_gnm.to(_gt.device).unsqueeze(0).expand_as(_gt).bool(),
                                              _gt, torch.ones_like(_gt))
                        if getattr(self, "_gate_grad", False):
                            _gt = _gt.detach().float().requires_grad_(True)
                            self._gate_tensors.append(_gt)
                        # Unbounded above: the normaliser is a MEAN, so if most reached
                        # nodes sit near 0 a few can be amplified hard, and six layers
                        # compound it. Recorded rather than clipped -- a clip would hide
                        # the instability instead of showing it.
                        with torch.no_grad():
                            _sel = _gt[_reach.bool()] if _reach is not None else _gt
                            if _sel.numel():
                                _rp_stat.append((float(_sel.mean()), float(_sel.max()),
                                                 float(_sel.quantile(0.95))))
                        # GATE THE MESSAGES ONLY. `_raw` is what the residual adds back.
                        # Overwriting layer_input here would put the gate on the residual
                        # too, i.e. h^(l+1) = Conv(g h) + g h instead of Conv(g h) + h, so
                        # the scaling would compound through all six layers and the method
                        # would be re-weighting states rather than routing messages --
                        # which is not what CCMP claims to do.
                        _msg = layer_input * _gt.unsqueeze(-1).to(layer_input.dtype)
                    else:
                        _msg = layer_input
                    if _reach is not None:
                        # AUTOCAST OFF. torch.sparse.mm is on the autocast list, so under
                        # bf16 AMP both operands are cast and there is no
                        # addmm_sparse_cuda kernel for BFloat16 -- it raises rather than
                        # falling back. The frontier is pure topology, so nothing here
                        # should be cast or differentiated in the first place.
                        with torch.no_grad(), torch.autocast(
                                device_type=_reach.device.type, enabled=False):
                            _nx = torch.sparse.mm(
                                self._ccmp_adj, _reach.t().contiguous().float()).t()
                            _reach = ((_nx > 0) | (_reach > 0)).float()
                else:
                    _msg = layer_input
                if separate_grad:
                    edge_weight = edge_weight.clone().requires_grad_()

                # RED-GNN-STYLE EDGE ATTENTION (ROUTE=attn): per-query weights on every
                # edge, computed from the sender state, the relation and the query. Passed
                # as a [B, E] edge_weight, which routes the layer through the unfused
                # message/aggregate path (the rspmm kernel takes one weight per edge).
                _ew = edge_weight
                # ROUTE_CKPT defaults OFF since 8 Sep: layers.py now runs per-query weights
                # through the fused rspmm kernel (O(N d) memory, kernel backward gives the
                # weight gradient), so the checkpoint's recompute only costs time. Set
                # ROUTE_CKPT=1 to get the old unfused+checkpoint path back.
                if getattr(self, "attn_node", None) is not None and \
                        os.environ.get("ROUTE_CKPT", "0") == "1" and torch.is_grad_enabled():
                    # GRADIENT CHECKPOINT THE ATTENTION LAYER. The unfused path keeps every
                    # [B, E+N, D] intermediate (gathered senders, gathered relations, the
                    # product, the boundary concat, the weighted copy) alive for backward:
                    # ~1.5 GB each in bf16 on TOMATO's 253k-edge train graph, x6 layers, which
                    # is how the attn smoke reached 75 GB on an 80 GB A100 (8 Sep) while the
                    # fused arms need a fraction of that. Recompute each layer's forward in
                    # backward instead: peak memory becomes one layer's transient, the
                    # numbers are identical, and the forward is paid twice. The attention
                    # logits are inside the checkpoint so their [B, E, H] tensors are not kept
                    # either. ROUTE_CKPT=0 restores the plain path.
                    from torch.utils.checkpoint import checkpoint as _ckpt

                    def _attn_layer(_x, _m, _q, _nf, _w, _layer=layer, _l=_li):
                        _w2 = self._route_attention(_l, _x, _q, _layer, data, _w)
                        return _layer(_m, _q, _nf, data.edge_index, data.edge_type, size, _w2)

                    hidden = _ckpt(_attn_layer, layer_input, _msg, query, node_features,
                                   edge_weight, use_reentrant=False)
                else:
                    if getattr(self, "attn_node", None) is not None:
                        _ew = self._route_attention(_li, layer_input, query, layer, data, edge_weight)
                    # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
                    hidden = layer(
                        _msg,
                        query,
                        node_features,
                        data.edge_index,
                        data.edge_type,
                        size,
                        _ew,
                    )
                if self.short_cut and hidden.shape == layer_input.shape:
                    # residual connection here. UNGATED `layer_input`, not `_msg`: the
                    # gate belongs on the messages this node sends, not on the state it
                    # keeps.
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                edge_weights.append(edge_weight)
                layer_input = hidden
            # Cached for the loss. A list of [B, N] per layer, or empty when CCMP is off,
            # so a control run pays nothing and the attribute always exists.
            self._resp_pred = _rp
            self._resp_gstat = _rp_stat

            # original query (relation type) embeddings
            node_query = query.unsqueeze(1).expand(
                -1, data.num_nodes, -1
            )  # (batch_size, num_nodes, input_dim)
            if self.concat_hidden:
                output = torch.cat(hiddens + [node_query], dim=-1)
            else:
                output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, node_features, relation_representations, query):
        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # we already did traversal_dropout in the outer loop of UltraQuery
        # if self.training:
        #     # Edge dropout in the training mode
        #     # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
        #     # to make NBFNet iteration learn non-trivial paths
        #     data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # node features arrive in shape (bs, num_nodes, dim)
        # NBFNet needs batch size on the first place
        output = self.bellmanford(
            data, node_features, query
        )  # (num_nodes, batch_size, feature_dim）
        if self.return_hidden:
            return output["node_feature"]
        else:
            score = self.mlp(output["node_feature"]).squeeze(-1)  # (bs, num_nodes)
            # return only the score
            return score

    def visualize(self, data, sample, node_features, relation_representations, query):
        for layer in self.layers:
            layer.relation = relation_representations

        output = self.bellmanford(
            data, node_features, query, separate_grad=True
        )  # (num_nodes, batch_size, feature_dim）
        node_feature = output["node_feature"]
        edge_weights = output["edge_weights"]
        question_entities_mask = sample["start_nodes_mask"]
        target_entities_mask = sample["target_nodes_mask"]
        query_entities_index = question_entities_mask.nonzero(as_tuple=True)[1]
        target_entities_index = target_entities_mask.nonzero(as_tuple=True)[1]

        paths_results = {}
        for t_index in target_entities_index:
            index = (
                t_index.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(-1, -1, node_feature.shape[-1])
            )
            feature = node_feature.gather(1, index).squeeze(0)
            score = self.mlp(feature).squeeze(-1)

            edge_grads = autograd.grad(score, edge_weights, retain_graph=True)
            distances, back_edges = self.beam_search_distance(
                data, edge_grads, query_entities_index, t_index, self.num_beam
            )
            paths, weights = self.topk_average_length(
                distances, back_edges, t_index, self.path_topk
            )
            paths_results[t_index.item()] = (paths, weights)
        return paths_results
