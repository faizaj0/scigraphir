"""
fusion_trainer.py — SFTTrainer for the CARGO fusion (FusionGraphReasoner / RoutedFusionReasoner).

Two objectives, selected by env FUSION_OBJECTIVE:

- "bce_pcr" (default, legacy): the config losses (bce + pcr) act on the FUSED document scores, plus a
  small graph-alone ListCE aux (weight AUX_W). This is the run whose graph stayed inert (aux frozen at
  ~log N): the operator already satisfies bce/pcr on the easy queries, so the GNN never gets a gradient.

- "hardneg" (the report's Eq 3.9 objective): a softmax cross-entropy over a per-query lineup whose
  negatives are the OPERATOR'S OWN top-ranked hubs — {gold} u operator-top-`HARDNEG_HUB` u `HARDNEG_RAND`
  random docs. Ranking the gold above the docs the operator already loves can only be done with the graph,
  so this is what actually teaches the graph to fix the operator's cross-domain misses. It is applied to
  BOTH the fused score (trains gate/router + operator scalars + graph jointly) AND the graph-alone score
  (weight AUX_W — trains the GNN DIRECTLY, so it still learns even when the fusion is initialised
  near-operator and the fused-path gradient into the graph is tiny).

LOSS V2 — three flag-gated, dissim-targeted edits to the hardneg objective.
All default OFF, giving bit-identical legacy behavior:

1. PER_GOLD=1 — per-gold contrastive. Legacy pools golds in one logsumexp, which is a
   soft-max: one easy gold satisfies the query and a buried dissim gold free-rides with
   ~zero gradient. Per-gold, EVERY gold must individually beat the lineup:
       L = sum_g w_g * [ logsumexp(negs u {g}) - s_g ] / sum_g w_g
2. MISS_W_AUX=1 / MISS_W_FUSED=1 — miss-weighting: w_g = log1p(operator rank of gold g),
   capped at MISS_W_CAP (default 8), normalised by the weight sum so the loss scale is
   stable. Concentrates gradient on the golds the operator buries (68% of dissim golds
   sit past rank 100). Recommended always-on for the graph-alone aux (no gate/router in
   that path); on the FUSED term only under convex routing — with the per-query additive
   gate, amplifying the gradient of graph-noisy queries is what taught the gate backwards.
   With PER_GOLD=0 the pooled query term is weighted by its WORST-ranked gold.
3. HARDNEG_GRAPH=K — graph-mined negatives: the graph-alone top-K (detached, golds
   removed) join the lineup, so the contrastive also pushes DOWN docs the graph
   over-scores (PPR domain-hub flooding) instead of only pushing golds up past operator
   hubs. Early in training the GNN is ~random so these are just extra random negatives;
   the term becomes self-adversarial as the graph learns.

Only train_step is overridden; evaluate()/predict() are inherited unchanged and already consume the
fused score, because the fusion lives inside the model's forward().
"""
import os
import json

import torch

from gfmrag.losses import ListCELoss
from gfmrag.models import cqig as cqig_mod
from gfmrag.models.ultra import query_utils

from .sft_trainer import SFTTrainer


def _emb(batch):
    """A single [1, D] question vector for a batch of any size."""
    e = batch["question_embeddings"].float()
    return e.reshape(-1, e.shape[-1]).mean(0, keepdim=True)


class FusionSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aux_loss_fn = ListCELoss()
        self._aux_w = float(os.environ.get("AUX_W", "0.1"))
        self._objective = os.environ.get("FUSION_OBJECTIVE", "bce_pcr")
        self._hn_hub = int(os.environ.get("HARDNEG_HUB", "50"))    # operator-top-K hubs per query
        self._hn_rand = int(os.environ.get("HARDNEG_RAND", "50"))  # random negatives per query
        # Anchor weight for the learned semantic scorer's popularity predictor. Matches
        # --pop_lambda in semantic_scorer.py so the fusion continues training the model
        # under the objective it was selected under, not a different one.
        self._sem_pop_lambda = float(os.environ.get("SEM_POP_LAMBDA", "1.0"))
        # --- Cross-Query Informativeness Gating (inert unless the model has a gate) ---
        self._cqig_m = int(os.environ.get("CQIG_M", "16"))            # reference queries
        self._cqig_pool_size = int(os.environ.get("CQIG_POOL", "64"))  # pool to choose from
        self._cqig_recal = int(os.environ.get("CQIG_RECAL", "1000"))   # steps; 0 = calibrate once
        # Calibration rounds. 1 is exact for a single gated layer; 2 re-runs the whole
        # calibration with the first round's gates active, which is what removes the stale
        # centering when several layers are gated. Costs one more pair of reference passes.
        self._cqig_rounds = int(os.environ.get("CQIG_ROUNDS", "1"))
        # Queries per reference forward. 1 is the safe default: the calibration state is
        # [B, N, D] and the graphs here have ~60k nodes.
        self._cqig_ref_batch = int(os.environ.get("CQIG_REF_BATCH", "1"))
        # Hard negatives per query for the gold-vs-negative informativeness AUC. Matches
        # HARDNEG_HUB so the diagnostic asks about the same documents the loss does.
        self._cqig_hn_k = int(os.environ.get("CQIG_HN_K", os.environ.get("HARDNEG_HUB", "50")))
        # How the frozen source references are re-linked to a target graph.
        #   exact     string equality on entity names, i.e. what the indexer does. Correct
        #             on the source graph and near-useless off it: physics resolved 501/501
        #             on its train graph and 25/501 on its own test graph.
        #   semantic  exact first, then the K nearest target nodes to the seed's frozen
        #             source embedding, above a source-selected cosine floor, weighted by
        #             softmax(cos / T) and normalised so the seed's total weight is
        #             unchanged. Still zero-shot: frozen encoder, source-chosen floor, no
        #             target label or query touched.
        self._cqig_link = os.environ.get("CQIG_LINK", "exact").strip().lower()
        assert self._cqig_link in ("exact", "semantic"), (
            f"CQIG_LINK={self._cqig_link!r}; expected 'exact' or 'semantic'")
        self._cqig_link_k = int(os.environ.get("CQIG_LINK_K", "3"))
        # Softmax temperature over cosines. At 0.05 a 0.1 gap in cosine is a ~7x weight
        # ratio: sharp enough that the best match dominates, soft enough that a genuine
        # near-tie splits rather than being decided by encoder noise.
        self._cqig_link_temp = float(os.environ.get("CQIG_LINK_T", "0.05"))
        # The rejection floor. "auto" measures it on the SOURCE graph (see _cqig_link_floor);
        # a number pins it. Either way it is fixed before any target graph is seen.
        self._cqig_link_min = os.environ.get("CQIG_LINK_MIN", "auto").strip().lower()
        self._cqig_link_q = float(os.environ.get("CQIG_LINK_Q", "0.5"))
        # --- loss v2 flags (all default to exact legacy behavior) ---
        self._per_gold = os.environ.get("PER_GOLD", "0") == "1"
        self._hn_graph = int(os.environ.get("HARDNEG_GRAPH", "0"))       # graph-top-K negatives
        self._miss_w_fused = os.environ.get("MISS_W_FUSED", "0") == "1"
        self._miss_w_aux = os.environ.get("MISS_W_AUX", "0") == "1"
        self._miss_w_cap = float(os.environ.get("MISS_W_CAP", "8.0"))
        # --- RESID_PRIOR: residual supervision against the graph's own structural prior ---
        #
        # MEASURED PROBLEM. A zero-parameter random walk on these graphs scores 0.245/0.273
        # nDCG@5; the trained 6-layer GNN scores 0.213-0.273. The graph-alone contrastive
        # asks the GNN to rank golds above operator hubs, which personalised PageRank from
        # the same seeds largely does already, so the term is near-satisfied at init and the
        # surviving gradient points back at the structure the walk exploits. The learned
        # component is close to free-lunch topology.
        #
        # THE EDIT. Restrict each gold's negatives to the documents the PRIOR already ranks
        # ABOVE it. Documents the prior ordered correctly contribute exactly zero gradient,
        # so the loss can only be reduced by capacity the walk does not have. This is
        # orthogonal to MISS_W_*, which reweights by the SEMANTIC channel's rank and
        # therefore never told the graph anything about what the graph already knew.
        #
        # Costs one cached PPR per query: T sparse mat-muls, computed once and reused for
        # every later epoch, against 6 dense GNN layers per step.
        self._resid_prior = os.environ.get("RESID_PRIOR", "0") == "1"
        self._resid_k = int(os.environ.get("RESID_K", "200"))       # cap on |negatives|
        self._resid_T = int(os.environ.get("RESID_T", "3"))         # walk length
        self._resid_alpha = float(os.environ.get("RESID_ALPHA", "0.15"))   # restart prob
        # Golds the prior already ranks first have an EMPTY negative set and would vanish
        # from the loss entirely, leaving the GNN unconstrained on everything it gets right.
        # They keep a small term against the ordinary lineup instead.
        self._resid_anchor = float(os.environ.get("RESID_ANCHOR", "0.1"))
        self._ppr_A = None          # row-normalised transition, one per graph
        self._ppr_A_key = None
        self._prior_cache = {}      # (graph key, query id) -> document prior, float16 CPU

        # --- CCMP: contrastive continuation message passing ------------------------
        self._ccmp = os.environ.get("CCMP", "0") == "1"
        self._ccmp_w = float(os.environ.get("CCMP_W", "0.1"))     # lambda_r
        self._ccmp_neg = int(os.environ.get("CCMP_NEG", "64"))    # |H^sem_q|
        # Over-fetch semantic errors before applying the graph-reachability filter.
        # CCMP_POOL=256 with CCMP_NEG=64 is the paper setting: it preserves semantic
        # hardness while giving the graph four times as many candidates from which to
        # find errors it can actually reach within the GNN horizon.
        self._ccmp_pool = max(
            self._ccmp_neg, int(os.environ.get("CCMP_POOL", "256"))
        )
        # Select the two sides independently. CCMP_M remains a backwards-compatible
        # total cap for old runs; when supplied alone it is split evenly. New runs use
        # 512 gold-favouring + 512 error-favouring nodes per layer.
        _legacy_m = os.environ.get("CCMP_M")
        _legacy_side = max(1, int(_legacy_m) // 2) if _legacy_m else 512
        self._ccmp_m_pos = int(os.environ.get("CCMP_M_POS", str(_legacy_side)))
        self._ccmp_m_neg = int(os.environ.get("CCMP_M_NEG", str(_legacy_side)))
        self._ccmp_cmin = float(os.environ.get("CCMP_CMIN", "0.2"))
        # RESIDUAL CCMP. Default OFF, so every run made before this is byte-identical.
        # ON: supervision is restricted to the golds the semantic scorer has NOT already
        # resolved, U_q = {g in G_q : exists d not in G_q with s_sem(q,d) >= s_sem(q,g)},
        # and the negatives to the reachable errors that outrank them. Queries with
        # U_q empty are supervised to leave propagation unchanged (g_qv = 1) instead of
        # being asked to improve a ranking that is already correct. Nothing changes at
        # inference: the head still predicts from node states alone.
        self._ccmp_resid = os.environ.get("CCMP_RESIDUAL", "0") == "1"
        # Weight of the identity (neutral) bucket in the class-balanced mean. 0 drops the
        # identity supervision and keeps only the restriction to unresolved golds, which
        # separates the two halves of the change.
        self._ccmp_id_w = float(os.environ.get("CCMP_IDENTITY_W", "1.0"))
        self._ccmp_P = None
        self._ccmp_key = None
        # The selected endpoint set and its continuation targets are fixed on first use,
        # then cached. This keeps CCMP's supervision stationary while the semantic and
        # graph scorers continue to train.
        self._ccmp_cache = {}        # (graph key, query id) -> (idx, y, c, meta) CPU
        if self._ccmp:
            print(
                f"[ccmp targets] reachable semantic errors: pool={self._ccmp_pool} "
                f"keep={self._ccmp_neg}; graph fallback on shortfall; "
                f"nodes/layer=+{self._ccmp_m_pos}/-{self._ccmp_m_neg}"
                + (f"; RESIDUAL on (identity_w={self._ccmp_id_w:g})"
                   if self._ccmp_resid else "; residual off"),
                flush=True,
            )

    def build_lineups(self, target_doc, s_op, g_mine=None):
        """One candidate lineup per query, built ONCE and reused by both losses.

        The fused and graph-alone terms are meant to face the SAME lineup, so that the
        only difference between them is which score is being trained. Building the
        negatives inside each call gave them the same hard negatives (top-k is
        deterministic) but freshly drawn randoms, which is not the design and adds
        variance for nothing.
        """
        B, n_doc = target_doc.shape
        dev = target_doc.device
        out = []
        for b in range(B):
            pos = target_doc[b].nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                out.append(None)
                continue
            # A UNION, NOT A CONCATENATION. The semantic and graph top-K overlap heavily
            # and torch.randint samples WITH replacement, so cat() put the same paper in
            # the denominator two or three times and silently gave it two or three times
            # the negative weight -- worst for exactly the papers both rankers over-score.
            taken = torch.zeros(n_doc, dtype=torch.bool, device=dev)
            taken[pos] = True
            # Over-fetch by the most that could already be claimed, so a fresh top-k is
            # always available without a .item() sync inside the training loop.
            budget = pos.numel() + self._hn_hub + max(self._hn_graph, 0)

            def _top_fresh(scores, k):
                """Top-k of `scores` not already claimed; marks what it returns."""
                if scores is None or k <= 0:
                    return None
                idx = scores.topk(min(k + budget, n_doc)).indices
                idx = idx[~taken[idx]][:k]
                taken[idx] = True
                return idx

            negs = [x for x in (
                _top_fresh(s_op[b], self._hn_hub),          # the semantic scorer's own hubs
                _top_fresh(g_mine[b] if g_mine is not None else None, self._hn_graph),
            ) if x is not None and x.numel() > 0]
            # EXACTLY `_hn_rand` DISTINCT random non-golds, rather than "50 draws minus
            # whatever collided", which quietly delivered fewer. A permutation is exact;
            # at 1k-20k documents it costs far less than the forward pass it rides on.
            if self._hn_rand > 0:
                pool = torch.randperm(n_doc, device=dev)
                pool = pool[~taken[pool]][: self._hn_rand]
                if pool.numel():
                    negs.append(pool)
            out.append(torch.cat(negs) if negs else pos.new_empty(0))
        return out

    def _contrastive_hardneg(self, doc_scores, target_doc, s_op, g_mine=None, miss_w=False,
                             lineups=None):
        """Contrastive over {gold(s)} u operator-top-K hubs [u graph-top-K] u random negatives.

        doc_scores : [B, n_doc] scores to train (fused or graph-alone), require grad.
        target_doc : [B, n_doc] {0,1} gold mask over document nodes.
        s_op       : [B, n_doc] operator scores (detached) — mines hard-negative hubs + miss weights.
        g_mine     : [B, n_doc] graph-alone scores (DETACHED) — mines HARDNEG_GRAPH extra negatives.
        miss_w     : weight each gold by log1p(its operator rank), capped at MISS_W_CAP.
        """
        B, n_doc = doc_scores.shape
        dev = doc_scores.device
        loss = doc_scores.new_zeros(())
        wsum = 0.0
        for b in range(B):
            pos = target_doc[b].nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            # Built once per step and shared by both losses. Falls back to building its
            # own only when called without one, which keeps older callers working.
            neg = (lineups[b] if lineups is not None
                   else self.build_lineups(target_doc, s_op, g_mine)[b])
            neg_logits = doc_scores[b, neg]
            # per-gold miss weights: how badly does the operator rank each gold?
            if miss_w:
                ranks = (s_op[b].unsqueeze(0) > s_op[b, pos].unsqueeze(1)).sum(1).float() + 1.0
                w = torch.log1p(ranks).clamp(max=self._miss_w_cap)
            else:
                w = torch.ones(pos.numel(), device=dev)
            if self._per_gold:
                # every gold must individually beat the lineup (no free-riding behind a sibling)
                for j in range(pos.numel()):
                    lg = torch.cat([neg_logits, doc_scores[b, pos[j : j + 1]]])
                    loss = loss + w[j] * (torch.logsumexp(lg, 0) - doc_scores[b, pos[j]])
                    wsum += float(w[j])
            else:
                # legacy pooled multi-positive; miss_w weights the query by its worst-ranked gold
                logits = torch.cat([doc_scores[b, pos], neg_logits])
                log_z = torch.logsumexp(logits, 0)
                log_pos = torch.logsumexp(logits[: pos.numel()], 0)
                wq = float(w.max())
                loss = loss + wq * (log_z - log_pos)
                wsum += wq
        return loss / max(wsum, 1e-6)

    # ------------------------------------------------- RESID_PRIOR: the structural prior
    def _prior_doc(self, graph, batch, doc_ids):
        """Personalised PageRank over the KG from this query's seeds, restricted to documents.

        Parameter-free and detached: this is the baseline the GNN has to beat, so nothing
        about it may depend on anything the GNN learns. Cached per (graph, seed set), which
        makes it a first-epoch cost only. Cached on CPU in float16 because the train graph
        holds 1304 queries x 4676 documents and that is 12 MB rather than 24.
        """
        dev = batch["start_nodes_mask"].device
        N = graph.num_nodes
        gkey = f"{id(graph)}:{N}"
        if self._ppr_A_key != gkey:
            ei = graph.edge_index
            # SYMMETRISED. The KG is directed, and a walk that can only travel head->tail
            # cannot reach a document from an entity the document mentions, which is the
            # dominant path in this corpus. Both directions, then row-normalise.
            r = torch.cat([ei[0], ei[1]])
            c = torch.cat([ei[1], ei[0]])
            deg = torch.zeros(N, device=dev).index_add_(
                0, r, torch.ones(r.numel(), device=dev))
            w = 1.0 / deg.clamp(min=1.0)[r]
            self._ppr_A = torch.sparse_coo_tensor(
                torch.stack([c, r]), w, (N, N), device=dev).coalesce()
            self._ppr_A_key = gkey
            self._prior_cache.clear()

        seeds = batch["start_nodes_mask"].float()                    # [B, N]
        out = seeds.new_zeros(seeds.shape[0], doc_ids.numel())
        for b in range(seeds.shape[0]):
            idx = seeds[b].nonzero(as_tuple=True)[0]
            key = (gkey, tuple(idx.tolist()))
            hit = self._prior_cache.get(key)
            if hit is None:
                x0 = seeds[b : b + 1]
                s = x0.sum()
                if float(s) <= 0:
                    # no seed resolved: a uniform prior, which makes EVERY document a
                    # negative for every gold. Skipped by the loss instead.
                    hit = torch.zeros(doc_ids.numel(), dtype=torch.float16)
                else:
                    # AUTOCAST OFF and no grad. train_step runs under bfloat16 AMP and
                    # sparse.mm has no bf16 kernel on every build; a crash here would cost
                    # a whole run. The prior is data, not a learned quantity, so neither
                    # autocast nor autograd has any business touching it.
                    with torch.autocast(device_type=x0.device.type, enabled=False), \
                            torch.no_grad():
                        x0 = (x0 / s).float()
                        x = x0
                        for _ in range(self._resid_T):
                            x = ((1.0 - self._resid_alpha)
                                 * torch.sparse.mm(self._ppr_A, x.t()).t()
                                 + self._resid_alpha * x0)
                    hit = x[0, doc_ids].detach().half().cpu()
                self._prior_cache[key] = hit
            out[b] = hit.to(dev).float()
        return out

    def _contrastive_resid(self, doc_scores, target_doc, prior_doc, lineups):
        """Negatives = the documents the PRIOR already ranks above this gold.

        A gold the walk already places above every non-gold has nothing to fix and drops
        out (except for the anchor); a gold the walk buries under 300 documents is asked to
        climb past the worst of them. The loss therefore measures only what the GNN adds
        to the topology, which is the quantity the run is trying to move.
        """
        B = doc_scores.shape[0]
        loss = doc_scores.new_zeros(())
        wsum, n_gold, n_touch, n_neg = 0.0, 0, 0, 0
        for b in range(B):
            pos = target_doc[b].nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            pr = prior_doc[b]
            if float(pr.max()) <= 0:
                continue                        # unseeded query: the prior says nothing
            for j in range(pos.numel()):
                g = pos[j]
                n_gold += 1
                above = (pr > pr[g]).nonzero(as_tuple=True)[0]
                if above.numel():
                    # sibling golds are not negatives, whatever the prior thinks of them
                    above = above[~torch.isin(above, pos)]
                if above.numel() == 0:
                    if self._resid_anchor > 0 and lineups is not None and lineups[b] is not None:
                        lg = torch.cat([doc_scores[b, lineups[b]], doc_scores[b, g : g + 1]])
                        loss = loss + self._resid_anchor * (
                            torch.logsumexp(lg, 0) - doc_scores[b, g])
                        wsum += self._resid_anchor
                    continue
                if above.numel() > self._resid_k:
                    # keep the prior's HIGHEST-scored offenders. Those are the documents
                    # actually occupying the top of the ranking the gold has to enter;
                    # the tail of a 3000-long set is noise and would dominate by count.
                    above = above[pr[above].topk(self._resid_k).indices]
                n_touch += 1
                n_neg += int(above.numel())
                lg = torch.cat([doc_scores[b, above], doc_scores[b, g : g + 1]])
                loss = loss + (torch.logsumexp(lg, 0) - doc_scores[b, g])
                wsum += 1.0
        stats = {
            # what fraction of golds the prior actually gets wrong, i.e. how much of the
            # data this loss can even see. If it is near zero the walk is already right
            # and there is nothing to learn; if it is near one the prior is useless here.
            "resid_cover": n_touch / max(n_gold, 1),
            "resid_negs": n_neg / max(n_touch, 1),
        }
        return loss / max(wsum, 1e-6), stats

    # ------------------------------------------------------------------- CCMP targets
    def _ccmp_hit(self, onehot, h):
        """Pr[a walk from v reaches this column's target within h hops], in [0, 1].

        Absorbing recurrence x <- 1_T + (1 - 1_T) P x. NOT sum_t P^t 1_T, which is the
        expected number of visits: unbounded, and it over-counts nodes sitting on short
        cycles, so it is not the probability the ratio is supposed to be a ratio of.
        """
        # SELF-GUARDING. Correct today only because the one caller wraps it, and there is
        # no bf16 addmm_sparse_cuda kernel: under autocast this raises rather than falling
        # back, so a second caller would be a crash, not a slow path.
        with torch.no_grad(), torch.autocast(
                device_type=onehot.device.type, enabled=False):
            x = onehot.clone().float()
            keep = 1.0 - x
            for _ in range(h):
                x = onehot.float() + keep * torch.sparse.mm(self._ccmp_P.float(), x)
        return x

    def _ccmp_targets(self, graph, batch, doc_ids, s_op, g_mine=None):
        """(idx, y, c, meta) per query, with idx/y/c each [L, M]. The endpoint
        set is selected on the query's first occurrence and then cached, making CCMP's
        intermediate supervision stationary while the two retrieval channels train.

        PER-TARGET COLUMNS, REDUCED PER SIDE. Seeding B+ from |G| nodes and B- from |H|
        nodes -- the form as originally written -- makes y ~ |G|/(|G|+|H|) ~ 0.01 and
        c = |2y-1| ~ 1 at essentially EVERY node, so the objective collapses to "predict
        0, confidently", the gate shuts globally and delta is the only thing propagating.
        Reducing each side separately is what makes a hub land at y ~ 0.5, c ~ 0, which
        is the ambiguity the confidence weight exists to express.

        ONE TARGET PER (q, l, v), over the gold UNION. yhat carries no gold index, so a
        per-gold target would only ever be fit to its c-weighted mean over golds -- i.e.
        exactly the "one easy gold satisfies the supervision" failure that per-gold
        computation is supposed to prevent.
        """
        dev = doc_ids.device
        N = graph.num_nodes
        gkey = f"{id(graph)}:{N}"
        if self._ccmp_key != gkey:
            ei = graph.edge_index
            r = torch.cat([ei[0], ei[1]])
            c = torch.cat([ei[1], ei[0]])
            deg = torch.zeros(N, device=dev).index_add_(
                0, r, torch.ones(r.numel(), device=dev))
            self._ccmp_P = torch.sparse_coo_tensor(
                torch.stack([r, c]), 1.0 / deg.clamp(min=1.0)[r],
                (N, N), device=dev).coalesce()
            self._ccmp_key = gkey
            self._ccmp_cache.clear()
        L = len(getattr(self.model.base.entity_model, "resp_proj", []) or [])
        tgt = batch["target_nodes_mask"]
        seeds = batch["start_nodes_mask"]
        out = []
        for b in range(tgt.shape[0]):
            gp = tgt[b, doc_ids].nonzero(as_tuple=True)[0]
            if gp.numel() == 0 or float(seeds[b].sum()) <= 0:
                out.append(None)
                continue
            # KEYED ON THE QUERY ID. Golds + seed COUNT is not a query identity: two
            # queries sharing golds and seed count would collide, and the negatives are
            # the SEMANTIC scorer's top-K, which differ per query, so the second query
            # would silently train against the first one's distractors.
            key = (gkey, str(batch["id"][b]))
            hit = self._ccmp_cache.get(key)
            if hit is None:
                with torch.autocast(device_type=dev.type, enabled=False), torch.no_grad():
                    # STRUCTURAL REACH. A^(0) = the query's seeds, A^(l+1) = A^(l) u
                    # N(A^(l)). Besides restricting layer-l supervision to A^(l), the
                    # final `reach` identifies documents the graph can reach within the
                    # complete L-layer horizon. An unreachable negative has B-=0 from
                    # every query-conditioned state and therefore cannot teach CCMP which
                    # route to suppress.
                    reach = (seeds[b] > 0).float()
                    reaches = []
                    for _ in range(L):
                        reaches.append(reach.clone())
                        nxt = torch.sparse.mm(
                            self._ccmp_P, reach.unsqueeze(1)
                        ).squeeze(1)
                        reach = ((nxt > 0) | (reach > 0)).float()
                    reachable_doc = reach[doc_ids] > 0

                    # Start from the scorer's most convincing errors, but retain only
                    # errors the graph could propagate to in L layers. We over-fetch
                    # CCMP_POOL (256 by default) and keep the first CCMP_NEG (64).
                    gold_doc = torch.zeros(
                        doc_ids.numel(), dtype=torch.bool, device=dev
                    )
                    gold_doc[gp] = True
                    s1 = s_op[b].detach().float().clone()
                    s1[gold_doc] = -torch.inf

                    # WHICH GOLDS THE SEMANTIC SCORER HAS NOT RESOLVED. A gold is
                    # unresolved when at least one non-gold scores at least as highly,
                    # i.e. s_sem(q,g) <= max_{d not in G_q} s_sem(q,d). With CCMP_RESIDUAL
                    # off, gp_use is every gold and no threshold is applied, which is the
                    # historical behaviour exactly.
                    gp_use, neutral = gp, False
                    if self._ccmp_resid:
                        max_neg = s1.max()
                        unres = gp[s_op[b].detach().float()[gp] <= max_neg]
                        if unres.numel() == 0:
                            # Every gold already outranks every non-gold. There is no
                            # error for the graph to correct, so CCMP is supervised to be
                            # a no-op here rather than perturbing a correct ranking. Node
                            # SELECTION is left untouched (below) so the represented set
                            # is the same one the contrastive queries use; only the target
                            # becomes neutral.
                            neutral = True
                        else:
                            gp_use = unres
                            # Negatives become the errors that stand between the query and
                            # its missed golds: non-golds scoring at least as highly as the
                            # lowest unresolved gold. Documents ranked below every missed
                            # gold are not obstructing anything and carry no signal about
                            # which route to suppress.
                            thr = s_op[b].detach().float()[gp_use].min()
                            s1[s1 < thr] = -torch.inf
                    pool_k = min(self._ccmp_pool, s1.numel())
                    sem_pool = s1.topk(pool_k).indices
                    # torch.isfinite drops the -inf padding topk returns once the
                    # residual threshold has masked most of the corpus.
                    sem_ok = (reachable_doc[sem_pool] & ~gold_doc[sem_pool]
                              & torch.isfinite(s1[sem_pool]))
                    sem_reachable = sem_pool[sem_ok]
                    neg_idx = sem_reachable[: self._ccmp_neg]
                    n_sem = int(neg_idx.numel())

                    # Some queries have fewer than 64 reachable documents among the
                    # semantic top 256. Fill only the missing slots with the graph's
                    # highest-ranked reachable errors. Never re-add a gold or a semantic
                    # negative already selected. If the reachable subgraph itself holds
                    # fewer than 64 non-golds, use the smaller honest endpoint set rather
                    # than reintroducing unreachable papers.
                    n_graph = 0
                    missing = self._ccmp_neg - n_sem
                    if missing > 0 and g_mine is not None:
                        eligible = reachable_doc & ~gold_doc
                        if self._ccmp_resid and not neutral:
                            # The graph-score fallback respects the same threshold, or the
                            # "errors that outrank the missed golds" definition would leak.
                            eligible = eligible & torch.isfinite(s1)
                        if neg_idx.numel() > 0:
                            eligible[neg_idx] = False
                        fill_k = min(missing, int(eligible.sum()))
                        if fill_k > 0:
                            g1 = g_mine[b].detach().float().clone()
                            g1[~eligible] = -torch.inf
                            graph_fill = g1.topk(fill_k).indices
                            neg_idx = torch.cat([neg_idx, graph_fill])
                            n_graph = int(graph_fill.numel())

                    # A seed component can contain only gold documents. In that rare
                    # case no valid negative continuation exists, so omitting CCMP for
                    # this query is preferable to taking a mean over an empty B- side.
                    if neg_idx.numel() == 0:
                        out.append(None)
                        continue

                    # B+ is seeded from the UNRESOLVED golds under CCMP_RESIDUAL, and
                    # from every gold otherwise.
                    gi, ni = doc_ids[gp_use], doc_ids[neg_idx]
                    oh = torch.zeros(N, gi.numel() + ni.numel(), device=dev)
                    oh[gi, torch.arange(gi.numel(), device=dev)] = 1.0
                    oh[ni, torch.arange(ni.numel(), device=dev) + gi.numel()] = 1.0
                    # Supervision at layer l is restricted to A^(l), because
                    # h^(l) at an unreached node carries no query information at all: it
                    # is still the static entity embedding that early-late fusion put
                    # there. Asking the head to predict a query-specific target from a
                    # query-independent state is asking it to fit noise, and on this graph
                    # it is nearly all of the sampled supervision.
                    I, Y, C = [], [], []
                    cand_pos = cand_neg = sel_pos = sel_neg = 0
                    for l in range(L):
                        x = self._ccmp_hit(oh, L - l)
                        bp = x[:, :gi.numel()].mean(1)
                        bn = x[:, gi.numel():].mean(1)
                        y = bp / (bp + bn + 1e-6)
                        # c = |B+ - B-| / (B+ + B-), NOT |2y - 1|.
                        #
                        # The two agree exactly wherever the node reaches something. They
                        # differ on the case that dominates this graph: a node reaching
                        # NEITHER set has B+ = B- = 0, so y = 0/eps = 0 and |2y-1| = 1.
                        # The old form therefore labelled every unreachable node
                        # "confidently negative" with full weight, and since the M nodes
                        # are chosen by top-c, the supervision was almost entirely those
                        # nodes: measured at 90% of all nodes passing at layer 1 rising to
                        # 100% by layer 6, with ~0% gold-side. The loss reduced to "predict
                        # zero everywhere", after which a mean-normalised gate returns ~1
                        # and CCMP is a no-op. This form sends those nodes to c = 0.
                        cc = (bp - bn).abs() / (bp + bn + 1e-6)
                        cc = cc * reaches[l]
                        # BALANCED NODE SELECTION. A single top-M over confidence was
                        # dominated by error-favouring nodes (~93% on this graph). The
                        # BCE was class-balanced afterwards, but the representation set
                        # itself contained very few positive routes. Rank the two sides
                        # independently so scarce gold-favouring states cannot be crowded
                        # out before the loss sees them. y=0.5 is genuinely ambiguous and
                        # occupies neither quota.
                        pos_all = ((y > 0.5) & (cc > 0)).nonzero(
                            as_tuple=True
                        )[0]
                        neg_all = ((y < 0.5) & (cc > 0)).nonzero(
                            as_tuple=True
                        )[0]
                        kp = min(self._ccmp_m_pos, int(pos_all.numel()))
                        kn = min(self._ccmp_m_neg, int(neg_all.numel()))
                        cand_pos += int(pos_all.numel())
                        cand_neg += int(neg_all.numel())
                        sel_pos += kp
                        sel_neg += kn
                        pos_idx = (
                            pos_all[cc[pos_all].topk(kp).indices]
                            if kp > 0 else pos_all
                        )
                        neg_idx_nodes = (
                            neg_all[cc[neg_all].topk(kn).indices]
                            if kn > 0 else neg_all
                        )
                        idx = torch.cat([pos_idx, neg_idx_nodes])
                        yy, csel = y[idx], cc[idx]
                        if neutral:
                            # IDENTITY SUPERVISION. A constant prediction across the
                            # reached nodes makes the mean-normalised gate
                            # ybar = (eps+yhat)/mean(eps+yhat) equal 1, hence
                            # g = (1-eta) + eta*1 = 1 and propagation is unchanged. 0.5 is
                            # the constant that also minimises the BCE at p = 0.5, and it
                            # is the sentinel _ccmp_loss routes to the neutral bucket.
                            # Full confidence, because "do not move" is not an ambiguous
                            # instruction; padding stays at c = 0 and is still dropped.
                            yy = torch.full_like(yy, 0.5)
                            csel = torch.ones_like(csel)

                        # Keep fixed [L, M_pos+M_neg] tensors for the CPU cache. Padding
                        # has c=0 and is removed by CCMP_CMIN before the BCE, so it cannot
                        # affect the loss or the reported supervised-node counts.
                        cap = self._ccmp_m_pos + self._ccmp_m_neg
                        pad = cap - idx.numel()
                        if pad > 0:
                            idx = torch.cat([
                                idx,
                                torch.zeros(pad, dtype=torch.long, device=dev),
                            ])
                            yy = torch.cat([yy, torch.full(
                                (pad,), 0.5, dtype=y.dtype, device=dev
                            )])
                            csel = torch.cat([csel, torch.zeros(
                                pad, dtype=cc.dtype, device=dev
                            )])
                        I.append(idx); Y.append(yy); C.append(csel)
                hit = (torch.stack(I).cpu(), torch.stack(Y).half().cpu(),
                       torch.stack(C).half().cpu(),
                       # Counts make the reachability filter auditable in the existing
                       # step and epoch logs without storing any document identifiers.
                       (n_sem, n_graph, int(sem_reachable.numel()), pool_k,
                        cand_pos, cand_neg, sel_pos, sel_neg,
                        # residual diagnostics: is this query neutral, and how many of
                        # its golds the semantic scorer left unresolved.
                        bool(neutral), int(gp_use.numel()), int(gp.numel())))
                self._ccmp_cache[key] = hit
            out.append(hit)
        return out

    def _ccmp_loss(self, preds, targets):
        """Confidence-weighted BCE between yhat^(l)_v and the continuation target."""
        if not preds:
            return None, {}
        dev = preds[0].device
        pn = preds[0].new_zeros(())      # positive-side numerator (y > 1/2)
        nn_ = preds[0].new_zeros(())     # negative-side numerator
        un_ = preds[0].new_zeros(())    # neutral / identity numerator (y == 1/2)
        pd = nd = ud = 0.0
        nsup, csum, npos = 0, 0.0, 0
        nquery = n_sem = n_graph = n_neg = n_pool_reach = n_pool = 0
        n_cand_pos = n_cand_neg = n_sel_pos = n_sel_neg = 0
        n_neutral = n_unres_gold = n_gold = 0
        for b, hit in enumerate(targets):
            if hit is None:
                continue
            idx, y, c, meta = hit
            # 11-tuple under CCMP_RESIDUAL, 8-tuple on a cache written before it. Reading
            # both keeps a warm _ccmp_cache from an earlier cell usable.
            (sem_count, graph_count, pool_reach, pool_count,
             cand_pos, cand_neg, sel_pos, sel_neg) = meta[:8]
            is_neutral, n_used, n_all = (meta[8:] if len(meta) >= 11
                                         else (False, 0, 0))
            n_neutral += int(bool(is_neutral))
            n_unres_gold += (0 if is_neutral else n_used)
            n_gold += n_all
            nquery += 1
            n_sem += sem_count
            n_graph += graph_count
            n_neg += sem_count + graph_count
            n_pool_reach += pool_reach
            n_pool += pool_count
            n_cand_pos += cand_pos
            n_cand_neg += cand_neg
            n_sel_pos += sel_pos
            n_sel_neg += sel_neg
            for l in range(min(len(preds), idx.shape[0])):
                ii = idx[l].to(dev)
                yy = y[l].to(dev).float()
                cc = c[l].to(dev).float()
                keep = cc > self._ccmp_cmin
                if not bool(keep.any()):
                    continue
                ii, yy, cc = ii[keep], yy[keep], cc[keep]
                p = preds[l][b, ii].float().clamp(1e-6, 1 - 1e-6)
                bce = -(yy * p.log() + (1 - yy) * (1 - p).log())
                # CLASS-BALANCED. Even after the confidence fix the reached targets run
                # ~93% negative on this graph, so an unbalanced mean is minimised by
                # "predict 0 everywhere" -- the same degenerate solution by a slower route.
                # Each side is normalised by its own confidence mass and the two are
                # averaged, so the head cannot buy the loss down by collapsing.
                # THREE BUCKETS, NOT TWO. y == 0.5 is the identity sentinel written by
                # _ccmp_targets for queries the semantic scorer already resolved. Left in
                # the negative bucket it would read as "predict 0", which is the collapse
                # the class balance exists to prevent, and it would drag the gate down on
                # exactly the same-field queries this change is meant to protect.
                # Ordinary targets never land on 0.5: pos_all/neg_all are strict.
                u = yy == 0.5
                m = yy > 0.5
                if bool(m.any()):
                    pn = pn + (cc[m] * bce[m]).sum(); pd += float(cc[m].sum())
                    npos += int(m.sum())
                neg = (~m) & (~u)
                if bool(neg.any()):
                    nn_ = nn_ + (cc[neg] * bce[neg]).sum(); nd += float(cc[neg].sum())
                if bool(u.any()):
                    un_ = un_ + (cc[u] * bce[u]).sum(); ud += float(cc[u].sum())
                nsup += int(ii.numel())
                csum += float(cc.sum())
        # Mean over whichever buckets carry mass, each normalised by its own confidence.
        # With no neutral targets this is 0.5*(pn/pd) + 0.5*(nn_/nd), identical to before.
        # CCMP_IDENTITY_W=0 drops the identity term, isolating the restriction to
        # unresolved golds from the instruction to stand still.
        terms, weights = [], []
        if pd > 0:
            terms.append(pn / pd); weights.append(1.0)
        if nd > 0:
            terms.append(nn_ / nd); weights.append(1.0)
        if ud > 0 and self._ccmp_id_w > 0:
            terms.append(un_ / ud); weights.append(self._ccmp_id_w)
        if not terms:
            return None, {}
        wsum = sum(weights)
        loss = sum(w * t for w, t in zip(weights, terms)) / wsum
        return loss, {"ccmp_nodes": nsup / max(len(targets), 1),
                      "ccmp_conf": csum / max(nsup, 1),
                      # share of supervised nodes on the gold side. If this is ~0 the
                      # target has collapsed and nothing downstream can work.
                      "ccmp_pos": npos / max(nsup, 1),
                      "ccmp_nodes_pos": npos / max(nquery, 1),
                      "ccmp_nodes_neg": (nsup - npos) / max(nquery, 1),
                      # Pre-selection prevalence remains the collapse diagnostic now
                      # that the selected representation set is deliberately balanced.
                      "ccmp_cand_pos": n_cand_pos / max(nquery, 1),
                      "ccmp_cand_neg": n_cand_neg / max(nquery, 1),
                      "ccmp_cand_pos_frac": n_cand_pos / max(
                          n_cand_pos + n_cand_neg, 1
                      ),
                      "ccmp_sel_pos": n_sel_pos / max(nquery, 1),
                      "ccmp_sel_neg": n_sel_neg / max(nquery, 1),
                      "ccmp_neg_sem": n_sem / max(nquery, 1),
                      "ccmp_neg_graph": n_graph / max(nquery, 1),
                      "ccmp_neg_count": n_neg / max(nquery, 1),
                      "ccmp_pool_reach": n_pool_reach / max(n_pool, 1),
                      # RESIDUAL DIAGNOSTICS. Read these first on a residual run.
                      # ccmp_resolved is the share of queries the semantic scorer already
                      # ranks perfectly, i.e. the share on which CCMP is now a supervised
                      # no-op. If it is ~0 the change cannot do anything and the arm is
                      # the old CCMP; if it is ~1 nothing is being corrected.
                      # ccmp_unres_frac is the share of golds still contested on the
                      # remaining queries, i.e. how much narrower B+ has become.
                      "ccmp_resolved": n_neutral / max(nquery, 1),
                      "ccmp_unres_gold": n_unres_gold / max(nquery - n_neutral, 1),
                      "ccmp_unres_frac": n_unres_gold / max(n_gold, 1)}

    # ------------------------------------------------------------------ CQIG plumbing
    #
    # PROTOCOL. One bank of M source problems is chosen ONCE, from the training queries, and
    # stored on the model in a graph-independent form (frozen question embedding + start-node
    # NAMES). Every graph, including graphs the model has never seen, is calibrated by
    # re-linking those SAME problems against its own vocabulary. Nothing is ever read from
    # the target corpus's query set, so the evaluation is not transductive.
    @staticmethod
    def _cqig_key(task_dataset):
        """Identify a graph. NOT the node count: two corpora could share one."""
        return getattr(task_dataset, "name", None) or f"graph@{id(task_dataset)}"

    def _cqig_gate(self):
        m = getattr(self, "model", None)
        return getattr(m, "cqig", None) if m is not None else None

    @staticmethod
    def _cqig_vocab(ds, which):
        v = getattr(ds, f"cqig_{which}", None)
        assert v, (
            f"task dataset {getattr(ds, 'name', ds)!r} carries no {which}; CQIG re-links its "
            f"reference problems by entity NAME, so it needs the graph's vocabulary. "
            f"_create_task_dataset attaches it.")
        return v

    @staticmethod
    def _cqig_node_emb(ds):
        """This graph's frozen node-text embeddings, or None if it was indexed without them.

        `graph.x` is what the indexer wrote by encoding every node name with the same frozen
        text encoder the questions went through, so a seed name and a target node name are
        already in one space and no encoder has to be carried to inference time.
        """
        graph = getattr(ds, "graph", None)
        return getattr(graph, "x", None) if graph is not None else None

    @staticmethod
    def _cqig_refs_from_batch(batch, id2node, node_emb=None):
        """Split a loader batch into PORTABLE per-query references.

        Portable means nothing in it is sized to, or indexed by, this graph: the question
        embedding is frozen text, and the seeds are the entity names the indexer looked up
        in node2id, recovered here from the mask through id2node together with their
        attachment weights. One entry per QUERY, never per batch.

        `node_emb` is the SOURCE graph's node features. When given, each seed also carries
        its frozen embedding, which is what the semantic linker matches against a target
        vocabulary. It is a property of the seed's text under a frozen encoder, not of the
        source graph's topology, so it travels as cleanly as the name does.
        """
        e = batch["question_embeddings"].detach().float().cpu()
        m = batch["start_nodes_mask"].detach().float().cpu()
        ids = batch.get("id")
        out = []
        for k in range(e.shape[0]):
            nz = torch.nonzero(m[k], as_tuple=False).flatten().tolist()
            rows = [j for j in nz if j in id2node]
            seeds = [(id2node[j], float(m[k, j])) for j in rows]
            se = None
            if node_emb is not None and rows:
                # Same order as `seeds`; set_reference_bank asserts on that pairing.
                se = node_emb[torch.tensor(rows, dtype=torch.long)].detach().float().cpu()
            out.append({"qemb": e[k:k + 1].clone(), "seeds": seeds, "seed_emb": se,
                        "id": (str(ids[k]) if ids is not None else None)})
        return out

    def _cqig_link_floor(self, refs, node_emb):
        """Choose the semantic linker's rejection threshold from SOURCE data only.

        The question the floor has to answer is "how similar are two node names that mean
        the same thing, in THIS encoder's geometry?", and that is answerable without any
        target corpus. For every reference seed, take its nearest OTHER node in the source
        graph: those are the encoder's own near-synonyms, at the scale this graph's
        vocabulary actually produces. The floor is a quantile of that distribution.

        Deliberately not a round number like 0.7. A hand-set constant is a hyperparameter
        tuned on whatever corpus it was first tried on, and it would not survive a change of
        encoder; this rescales with the encoder because it is measured in it.
        """
        emb = torch.cat([r["seed_emb"] for r in refs if r.get("seed_emb") is not None], 0)
        if emb.numel() == 0:
            return 0.0, 0
        # top-2: the first is the seed's own node at cosine 1, the second is its neighbour.
        cos, _ = cqig_mod.topk_cosine(emb.to(node_emb.device), node_emb, 2)
        nn_cos = cos[:, 1].float().cpu()
        q = min(max(self._cqig_link_q, 0.0), 1.0)
        return float(torch.quantile(nn_cos, q)), int(nn_cos.numel())

    @staticmethod
    def _cqig_farthest_point(refs, m):
        """Pick m maximally spread references by farthest-point on question embeddings."""
        embs = torch.cat([r["qemb"] for r in refs], 0)
        embs = embs / embs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        picked = [0]
        while len(picked) < min(m, len(refs)):
            d = (1.0 - embs @ embs[picked].T).min(dim=1).values
            d[torch.tensor(picked)] = -1.0
            picked.append(int(d.argmax()))
        return [refs[i] for i in picked]

    def _cqig_select_reference(self, batch, ds):
        """Fill a pool of individual TRAINING queries, then freeze M of them as the bank.

        The bank is handed to the MODEL, not kept on the trainer, so it rides in the
        checkpoint and a reloaded model can calibrate a corpus the trainer never saw.
        """
        g = self._cqig_gate()
        if g is None:
            return False
        if g.ref_bank:
            return True
        id2node = self._cqig_vocab(ds, "id2node")
        # SOURCE node features. Only needed for CQIG_LINK=semantic, but stored whenever the
        # graph has them: the bank rides in the checkpoint, and a bank frozen without seed
        # embeddings cannot be linked semantically later without retraining.
        src_emb = self._cqig_node_emb(ds)
        self._cqig_pool = getattr(self, "_cqig_pool", [])
        self._cqig_pool.extend(self._cqig_refs_from_batch(batch, id2node, src_emb))
        if len(self._cqig_pool) < self._cqig_pool_size:
            return False
        pool = self._cqig_pool[: self._cqig_pool_size]
        refs = self._cqig_farthest_point(pool, self._cqig_m)
        # EXACTLY M, OR SAY SO. Every graph is calibrated from this one bank, so M is the
        # same on all of them by construction -- but only if the bank really holds M.
        assert len(refs) == self._cqig_m, (
            f"asked for M={self._cqig_m} reference queries but the pool of "
            f"{len(pool)} yielded {len(refs)}; raise CQIG_POOL above CQIG_M")
        # THE FLOOR IS FROZEN WITH THE BANK, on the source graph, before any target graph
        # exists. Measuring it later against a target would make the threshold a function
        # of the corpus being transferred to, which is exactly what zero-shot forbids.
        floor, n_meas = 0.0, 0
        if self._cqig_link == "semantic":
            assert src_emb is not None, (
                "CQIG_LINK=semantic needs the source graph's node embeddings (graph.x), "
                "which are absent; this dataset was indexed without node features")
            if self._cqig_link_min == "auto":
                floor, n_meas = self._cqig_link_floor(refs, src_emb.to(self.device))
            else:
                floor = float(self._cqig_link_min)
        g.set_reference_bank(refs, meta={
            "m": len(refs), "pool": self._cqig_pool_size,
            "source": self._cqig_key(ds),
            "selection": "farthest-point on question embeddings, per query",
            "link": self._cqig_link, "link_k": self._cqig_link_k,
            "link_temp": self._cqig_link_temp, "link_floor": floor})
        self._cqig_pool = []
        seeds = [len(r["seeds"]) for r in refs]
        print(f"[cqig] source bank frozen: M={len(refs)} individual training queries from "
              f"'{self._cqig_key(ds)}' (pool {self._cqig_pool_size}); "
              f"seeds/query min={min(seeds)} median={sorted(seeds)[len(seeds) // 2]} "
              f"max={max(seeds)}; stored on the model", flush=True)
        if self._cqig_link == "semantic":
            how = (f"source nearest-neighbour cosine, q={self._cqig_link_q:g} over "
                   f"{n_meas} seeds" if self._cqig_link_min == "auto"
                   else f"pinned by CQIG_LINK_MIN={self._cqig_link_min}")
            print(f"[cqig] link=semantic: K={self._cqig_link_k} T={self._cqig_link_temp:g} "
                  f"floor={floor:.4f} [{how}]; seed weight is split by softmax(cos/T) over "
                  f"the survivors, so each seed's total attachment weight is unchanged",
                  flush=True)
        return True

    def _cqig_calibrate(self, graph, ds, why):
        """Re-link the source bank to THIS graph, calibrate, then measure what the gate does."""
        g = self._cqig_gate()
        key = self._cqig_key(ds)
        node2id = self._cqig_vocab(ds, "node2id")
        # The floor travels with the BANK, not with the trainer: a checkpoint calibrated on
        # a new corpus must reject at the same threshold the bank was frozen under, even in
        # a process where CQIG_LINK_MIN was never set.
        meta = g.ref_meta or {}
        batches, cov = g.materialise(
            node2id, int(graph.num_nodes), device=self.device,
            batch_size=self._cqig_ref_batch,
            node_emb=(graph.x if self._cqig_link == "semantic" else None),
            link=self._cqig_link, link_k=self._cqig_link_k,
            link_temp=meta.get("link_temp", self._cqig_link_temp),
            link_floor=meta.get("link_floor", 0.0))
        # SAME PRECISION CONTEXT AS EVERY OTHER FORWARD IN THIS TRAINER. Training, eval and
        # predict all wrap their forwards in autocast. The train-side calibration inherited
        # that by accident, being called from inside the training loop; the inference-side
        # call sits outside it, so float32 activations met bfloat16 weights.
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype,
                                enabled=self.use_amp):
            rep = self.model.cqig_calibrate(graph, key, ref_batches=batches,
                                            device=self.device,
                                            rounds=self._cqig_rounds)
        print(f"[cqig] {why}: graph='{key}' M={cov['queries']} queries "
              f"({cov['batches']} forwards x {self._cqig_rounds} round(s) x 2 passes) "
              f"seed-coverage={cov['seed_coverage']:.1%} "
              f"({cov['seeds_found']}/{cov['seeds_total']}) "
              f"queries_with_no_seed={cov['queries_with_no_seed']} "
              f"stats={g.stats_bytes() / 2**20:.0f} MB", flush=True)
        # THE SPLIT, ALWAYS. `exact` is directly comparable to every run made before the
        # semantic linker existed, so the two arms can be read against each other rather
        # than against an aggregate that moved for an unstated reason.
        if cov["link"] == "semantic":
            how = (f"K={cov['link_k']} floor={cov['link_floor']:.4f}, median best cos "
                   f"{cov['sem_cos_median']:.3f}, {cov['sem_nodes_per_seed']:.2f} nodes/seed"
                   if cov["seeds_semantic"] else
                   f"K={cov['link_k']} floor={cov['link_floor']:.4f}, none needed")
            print(f"[cqig]   linking: exact {cov['seeds_exact']} "
                  f"({cov['exact_coverage']:.1%})  + semantic {cov['seeds_semantic']} "
                  f"({how})  = {cov['seeds_found']}/{cov['seeds_total']}; "
                  f"unmatched {cov['seeds_unmatched']}", flush=True)
        # EVERY GATED LAYER, not just the first. With a multi-layer arm the interesting
        # failure is one layer doing the work and the rest sitting at a constant.
        for j in sorted(rep):
            r = rep[j]
            v, iq, gq = r["var_q"], r["I"], r["gate"]
            print(f"[cqig]   layer {j + 1}/{g.n_layers}: nodes={r['nodes']} "
                  f"responding={r['live']} ({r['responding_frac']:.2%})  "
                  f"V p10/p50/p90 = {v['p10']:.4e} {v['p50']:.4e} {v['p90']:.4e}  "
                  f"scale={r['scale']:.4e}  tau_ref={r['tau_ref']:.4e} "
                  f"[{r['tau_src']}]", flush=True)
            print(f"[cqig]     I    min/p10/p50/p90/max = {iq['min']:.4e} {iq['p10']:.4e} "
                  f"{iq['p50']:.4e} {iq['p90']:.4e} {iq['max']:.4e}  "
                  f"std={iq['std']:.4e}  responded={r['resp_frac']:.2%} over "
                  f"{r['sample_queries']} reference queries", flush=True)
            print(f"[cqig]     g    min/p10/p50/p90/max = {gq['min']:.4e} {gq['p10']:.4e} "
                  f"{gq['p50']:.4e} {gq['p90']:.4e} {gq['max']:.4e}  "
                  f"std={gq['std']:.4e}  span={gq['max'] - gq['min']:.4e}", flush=True)
            # THE COEFFICIENT THE ARM ACTUALLY APPLIES. Identical to g under op="gate", so
            # the line is only printed when it would say something different -- and there it
            # is the one that matters, because g is not what multiplies anything.
            if r.get("mu_source", "ref") == "zero":
                # The ablation must be visible in the calibration block itself, not only in
                # the startup banner, because that block is what gets pasted around.
                print(f"[cqig]     mu=ZERO ABLATION: mu is 0 on all {r['mu_zero']} nodes, "
                      f"so I = ||h||^2/scale and the reference bank contributes nothing to "
                      f"this arm. Its score is the magnitude-gate floor, not the method.",
                      flush=True)
            if r.get("op", "gate") != "gate":
                cq = r["coef"]
                print(f"[cqig]     op={r['op']}: h - c*mu, c min/p10/p50/p90/max = "
                      f"{cq['min']:.4e} {cq['p10']:.4e} {cq['p50']:.4e} {cq['p90']:.4e} "
                      f"{cq['max']:.4e}   mu=0 on {r['mu_zero']} nodes "
                      f"({r['mu_zero_frac']:.2%}), which this op cannot touch at all",
                      flush=True)
            print(f"[cqig]     lam={r['lam']:.6e} alpha={r['alpha']:.6e} "
                  f"tau={r['tau']:.6e} gate_at_I=0 {r['gate_unreached']:.4e}", flush=True)
            if r["degenerate"]:
                print(f"[cqig]     WARNING: no node responded to any reference query at "
                      f"layer {j + 1}; the scale fell back to 1.0 and I is meaningless.",
                      flush=True)
            elif gq["max"] - gq["min"] < 1e-3:
                print(f"[cqig]     WARNING: the gate spans {gq['max'] - gq['min']:.2e} "
                      f"across nodes. That is a near-constant rescaling, not a gate, and "
                      f"this arm will read as a no-op whatever the final metrics say.",
                      flush=True)
        if cov["seed_coverage"] < 0.2:
            fix = ("" if cov["link"] == "semantic" else
                   " Exact name lookup is the likely cause, not the corpus: set "
                   "CQIG_LINK=semantic to link the remainder by embedding.")
            print(f"[cqig]   WARNING: only {cov['seed_coverage']:.1%} of the source seeds "
                  f"resolve in '{key}'. The reference problems barely touch this graph, so "
                  f"its statistics are not measuring a response to them.{fix}", flush=True)
        elif cov["link"] == "semantic" and cov["exact_coverage"] < 0.2:
            print(f"[cqig]   note: exact lookup alone would have reached "
                  f"{cov['exact_coverage']:.1%} on '{key}'; the semantic linker carried it "
                  f"to {cov['seed_coverage']:.1%}. That gap is what this arm is testing.",
                  flush=True)
        gb = g.stats_bytes() / 2**30
        if gb > 4.0:
            print(f"[cqig]   WARNING: the reference statistics now hold {gb:.1f} GiB of "
                  f"GPU memory ({len(g.gate_layers)} gated layers x "
                  f"{len(g.known_graphs())} graphs). mu is one float32 row per node per "
                  f"gated layer, so gating fewer layers is the lever if this run runs out "
                  f"of memory.", flush=True)
        return key

    def _cqig_report_live(self, why):
        """What the gate did on the queries that were actually scored, plus the AUC that
        decides whether informativeness has anything to do with relevance."""
        g = self._cqig_gate()
        if g is None or not g.recording():
            return
        rep = g.live_report()
        if not rep:
            g.stop_live()
            return
        for j in sorted(rep):
            r = rep[j]
            what = "gate" if r.get("op", "gate") == "gate" else f"{r['op']} coef"
            print(f"[cqig] {why} {what}, layer {j + 1}: n={r['n']:.4e}  "
                  f"min/p10/p50/p90/max = {r['min']:.4e} {r['p10']:.4e} {r['p50']:.4e} "
                  f"{r['p90']:.4e} {r['max']:.4e}  mean={r['mean']:.4e} "
                  f"std={r['std']:.4e}  span={r['max'] - r['min']:.4e}  "
                  f"lam={float(g.lam(j).detach()):.6e} "
                  f"alpha={float(g.alpha(j).detach()):.6e} "
                  f"tau={g.tau(j):.6e}", flush=True)
            # HOW BIG THE EDIT WAS, measured on the states. A coefficient far from 1 that
            # moves the state by 1e-3 is a no-op wearing a large number, and with
            # `layer_norm: yes` downstream that is the failure worth being able to see.
            print(f"[cqig]   edit ||h~-h||/||h||: mean={r['edit_rel_mean']:.4e} "
                  f"max={r['edit_rel_max']:.4e} on {r['edit_frac']:.2%} of live nodes",
                  flush=True)
            if r["auc_n"]:
                print(f"[cqig]   informativeness AUC, gold vs semantic-top-"
                      f"{self._cqig_hn_k}: {r['auc']:.4f} over {r['auc_n']} queries  "
                      f"(mean I gold {r['I_gold']:.4e} vs negative {r['I_neg']:.4e})",
                      flush=True)
                if abs(r["auc"] - 0.5) < 0.02:
                    print(f"[cqig]   NOTE: an AUC of {r['auc']:.4f} says a high I is no "
                          f"likelier on a gold paper than on a hard negative. The gate's "
                          f"premise, not its calibration, is what is failing.", flush=True)
        g.stop_live()

    def evaluate(self) -> dict:
        """Measure the gate on the real evaluation queries, around the inherited eval.

        Reset before and report after, so the numbers belong to one evaluation rather than
        accumulating across epochs. Purely observational: `reset_live` switches on a
        histogram and a label-side AUC, neither of which is read by any calibration.
        """
        g = self._cqig_gate()
        if g is not None and g.ref_bank:
            g.reset_live()
        m = super().evaluate()
        self._cqig_report_live("eval")
        return m

    def predict(self) -> dict:
        """Same measurement around the pass that writes the predictions file.

        The AUC stays empty here, because prediction batches carry no gold mask; the gate
        distribution is the part that matters, since these are the scores the benchmark
        numbers are computed from.
        """
        g = self._cqig_gate()
        if g is not None and g.ref_bank:
            g.reset_live()
        out = super().predict()
        self._cqig_report_live("predict")
        return out

    # ------------------------------------------------------------ path interpretation
    @torch.no_grad()
    def _channel_scores(self, graph, batch):
        """One forward; returns doc node ids and the four per-document score vectors
        (fused, graph-alone raw, multi-view scorer, Qwen3 cosine), all in nodes.csv doc order."""
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
            pred = self.model(graph, batch)
        did = self.model._doc_ids
        fused = pred[0, did].float()
        graph_raw = self.model._raw_doc[0].float()
        sem = self.model._s_op[0].float()
        if getattr(self.model, "semantic", "mlp") == "operator":
            # operator-scorer checkpoint (the July v1 fusion, e.g. TOMATO on the OpenIE graph): there is
            # no multi-view table, the 'scorer' channel is S_op and the Qwen3 cosine comes from the
            # operator components; callers get (None, None) instead of a views table.
            tag, row = self.model._row[str(batch["id"][0])]
            dense = self.model._dense[tag][row].float().to(fused.device)
            return did, {"fused": fused, "graph": graph_raw, "scorer": sem, "dense": dense}, (None, None)
        tag, row = self.model._sem_row[str(batch["id"][0])]
        tab = self.model._sem_tab[tag]
        dense = tab["dense"][row].float().to(fused.device)
        return did, {"fused": fused, "graph": graph_raw, "scorer": sem, "dense": dense}, (tab, row)

    def _rank_without(self, graph, batch, triples, node, n_random=0):
        """Rank of `node` under the graph and fused channels after dropping the given (h, t, r)
        edges from the graph (or `n_random` random edges when triples is None). Edits the graph
        in place for one forward pass and restores it."""
        ei, et = graph.edge_index, graph.edge_type
        keep = torch.ones(ei.shape[1], dtype=torch.bool, device=ei.device)
        if triples is None:
            if n_random > 0:
                drop = torch.randperm(ei.shape[1], device=ei.device)[:n_random]
                keep[drop] = False
        else:
            for h, t, r in set(tuple(int(x) for x in tr) for tr in triples):
                m = (ei[0] == h) & (ei[1] == t) & (et == r)
                if m.sum() == 0:
                    m = (ei[0] == h) & (ei[1] == t)
                keep &= ~m
        removed = int((~keep).sum().item())
        em = self.model.base.entity_model
        try:
            graph.edge_index, graph.edge_type = ei[:, keep], et[keep]
            em._ccmp_adj_key = None
            with torch.no_grad():
                did, ch, _ = self._channel_scores(graph, batch)
            j = (did == node).nonzero(as_tuple=True)[0]
            out = {"removed": removed}
            if j.numel():
                j = int(j[0])
                for k in ("graph", "fused"):
                    out[k] = int((ch[k] > ch[k][j]).sum().item()) + 1
        finally:
            graph.edge_index, graph.edge_type = ei, et
            em._ccmp_adj_key = None
        return out

    @staticmethod
    def _valid_paths(paths, weights, seeds):
        """Drop decoded paths that do not start at a seed or whose hops do not chain. The beam
        backtrack can emit a placeholder first hop (node 0 -> node 0) when a node's layer-0 beam
        slot was never filled; those are artefacts, not routes. Returns (paths, weights, n_dropped)."""
        keep_p, keep_w = [], []
        for path, w in zip(paths, weights):
            ok = len(path) > 0 and int(path[0][0]) in seeds and all(int(a[1]) == int(b[0]) for a, b in zip(path, path[1:]))
            if ok:
                keep_p.append(path); keep_w.append(w)
        return keep_p, keep_w, len(paths) - len(keep_p)

    def _interpret_target(self, graph, batch, node, em, eta, id2node, id2rel):
        """Gradient beam search from the query's seeds to one node; returns (paths with the CCMP
        gate per hop, frontier-mean responsibility per layer, raw (h, t, r) paths, n_dropped)."""
        seeds = set(batch["start_nodes_mask"][0].nonzero(as_tuple=True)[0].tolist())
        sample = dict(batch); tm = torch.zeros_like(batch["target_nodes_mask"]); tm[0, node] = 1.0
        sample["target_nodes_mask"] = tm
        if getattr(em, "resp_proj", None) is not None:
            em._ccmp_seeds = sample["start_nodes_mask"]
        em._keep_reach = True
        with torch.enable_grad(), torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
            pr = self.model.base.visualize(graph, sample)
        em._keep_reach = False
        rp = [x[0].detach().float() for x in getattr(em, "_resp_pred", [])]
        reach = getattr(em, "_reach_layers", [])
        fm = []
        for l, y in enumerate(rp):
            r_ = reach[l] if l < len(reach) and reach[l] is not None else None
            fm.append(float(y[r_[0].bool()].mean()) if r_ is not None and r_.sum() > 0 else float(y.mean()))
        paths, weights = pr.get(int(node), ([], []))
        paths, weights, dropped = self._valid_paths(paths, weights, seeds)
        paths_out = []
        for path, w in zip(paths, weights):
            hops = []
            for l, (h, t, r) in enumerate(path):
                hop = {"layer": l, "head": id2node[h], "rel": id2rel.get(r, str(r)), "tail": id2node[t]}
                if l < len(rp):
                    y = float(rp[l][h]); hop["resp"] = y; hop["frontier_mean"] = fm[l]
                    hop["gate"] = (1 - eta) + eta * (y / max(fm[l], 1e-6))
                    if l < len(reach) and reach[l] is not None:
                        hop["reached"] = bool(reach[l][0, h] > 0)
                hops.append(hop)
            paths_out.append({"weight": float(w), "hops": hops})
        return paths_out, fm, paths, dropped

    def _edge_grads_for(self, graph, batch, node, em):
        """Per-layer d(graph score of `node`)/d(edge weight), [E] per layer, captured from the same
        forward the beam search uses (base.visualize), so the numbers are exactly the beam's."""
        sample = dict(batch); tm = torch.zeros_like(batch["target_nodes_mask"]); tm[0, node] = 1.0
        sample["target_nodes_mask"] = tm
        if getattr(em, "resp_proj", None) is not None:
            em._ccmp_seeds = sample["start_nodes_mask"]
        cap = {}
        orig = em.beam_search_distance
        def spy(data, edge_grads, h_index, t_index, num_beam=10):
            cap["eg"] = [g.detach().float() for g in edge_grads]
            return orig(data, edge_grads, h_index, t_index, num_beam)
        em.beam_search_distance = spy
        try:
            with torch.enable_grad(), torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                pr = self.model.base.visualize(graph, sample)
        finally:
            em.beam_search_distance = orig
        return cap["eg"], pr

    def _direct_path_weights(self, graph, edge_grads, paths):
        """Weight of each given path [(h, t, r), ...] under the given per-layer edge gradients: the
        beam's own definition (hop i taken at layer i, mean over hops), evaluated on a fixed route so
        the same route can be compared across gating conditions. None when an edge is missing."""
        key = getattr(self, "_edge_key_cache", None)
        if key is None or key[0] != id(graph):
            ei = graph.edge_index.cpu(); et = graph.edge_type.cpu()
            lut = {}
            for e, (h, t, r) in enumerate(zip(ei[0].tolist(), ei[1].tolist(), et.tolist())):
                lut.setdefault((h, t, r), []).append(e)
            key = (id(graph), lut); self._edge_key_cache = key
        lut = key[1]; out = []
        for path in paths:
            vals = []
            for i, (h, t, r) in enumerate(path):
                es = lut.get((int(h), int(t), int(r)))
                if es is None or i >= len(edge_grads):
                    vals = None; break
                vals.append(max(float(edge_grads[i][e]) for e in es))
            out.append(None if vals is None else sum(vals) / len(vals))
        return out

    def _gate_attribution(self, graph, batch, node, em, id2node, raw_paths, topn=15):
        """Path interpretation through the gate: d(graph score of `node`)/d g_u^(l) for every node u and
        layer l, and the first-order CCMP contribution c_u^(l) = d s/d g * (g - 1). Summed over layers
        this says how much each node's gating moved the gold's score; along a route it says where on the
        route CCMP acted, at every layer, not only the one the hop was attributed to."""
        em.resp_gate = True; em._gate_layers = None; em._gate_nodes = None; em._gate_grad = True
        try:
            with torch.enable_grad(), torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                self.model(graph, batch)
                did = self.model._doc_ids
                j = (did == node).nonzero(as_tuple=True)[0]
                score = self.model._raw_doc[0, j].float().sum()
                gates = list(em._gate_tensors)
                grads = torch.autograd.grad(score, gates, allow_unused=True)
        finally:
            em._gate_grad = False; em._gate_tensors = []
        G = [g[0].detach().float().cpu() for g in gates]
        D = [(torch.zeros_like(G[l]) if grads[l] is None else grads[l][0].detach().float().cpu()) for l in range(len(G))]
        C = [D[l] * (G[l] - 1.0) for l in range(len(G))]
        total = torch.stack(C).sum(0)
        senders = sorted({int(h) for p_ in raw_paths for (h, t, r) in p_})
        per_sender = {id2node[u]: {"gate": [float(G[l][u]) for l in range(len(G))],
                                   "dscore_dg": [float(D[l][u]) for l in range(len(G))],
                                   "contrib": [float(C[l][u]) for l in range(len(G))],
                                   "contrib_total": float(total[u])} for u in senders}
        routes = [{"senders": [id2node[int(h)] for (h, t, r) in p_], "contrib_total": float(sum(total[int(h)] for (h, t, r) in p_))} for p_ in raw_paths]
        top = torch.topk(total.abs(), min(topn, total.numel())).indices.tolist()
        top_nodes = [{"node": id2node[u], "contrib_total": float(total[u]), "on_route": u in set(senders),
                      "gate": [round(float(G[l][u]), 3) for l in range(len(G))]} for u in top]
        return {"score_gate_on": float(score), "sum_contrib_all_nodes": float(total.sum()),
                "sum_contrib_route_senders": float(sum(total[u] for u in senders)),
                "per_layer_sum_all": [float(c.sum()) for c in C], "route_senders": per_sender, "routes": routes, "top_nodes": top_nodes}

    def gate_decomposition(self, qids, out_path, golds=None, num_beam=10, path_topk=5, max_golds=2):
        """Which gates move a route's weight? For each (query, gold): the top routes under the full
        CCMP gate, then the SAME routes' weights, the gold's graph score and its graph / fused ranks
        under: gate off; gate only at the layers the route is attributed to (0..L-1); gate only at
        the later layers; gate only on the route's own sender nodes; gate on every node except them.
        Same trained weights throughout; only the inference-time gate mask changes."""
        import numpy as np
        self.model.eval()
        em = self.model.base.entity_model
        assert getattr(em, "resp_proj", None) is not None, "gate_decomposition needs a CCMP checkpoint"
        em.num_beam, em.path_topk = int(num_beam), int(path_topk)
        eta = float(getattr(em, "resp_eta", 0.5))
        want = [str(q) for q in qids]
        results = []
        for test_dataset in self.eval_graph_dataset_loader:
            src = test_dataset.data
            graph = src.graph.to(self.device); data = src.test_data
            id2node = src.id2node; id2rel = {v: k for k, v in src.rel2id.items()}
            raw = {str(x["id"]): x for x in src.raw_test_data}
            pos = {str(data[i]["id"]): i for i in range(len(data)) if str(data[i]["id"]) in set(want)}
            n_layers = len(em.layers)
            for sid in want:
                if sid not in pos: continue
                item = data[pos[sid]]
                batch = {"question_embeddings": item["question_embeddings"].unsqueeze(0).to(self.device),
                         "start_nodes_mask": item["start_nodes_mask"].unsqueeze(0).to(self.device),
                         "target_nodes_mask": item["target_nodes_mask"].unsqueeze(0).to(self.device), "id": [sid]}
                did, ch, _ = self._channel_scores(graph, batch)
                gold_pos = (batch["target_nodes_mask"][0, did] > 0).nonzero(as_tuple=True)[0]
                pinned = set((golds or {}).get(sid, []))
                forced = [j for j in gold_pos.tolist() if id2node[int(did[j])] in pinned]
                best = forced if forced else sorted(gold_pos.tolist(), key=lambda j: int((ch["fused"] > ch["fused"][j]).sum()))[:max_golds]
                rec = {"id": sid, "question": raw.get(sid, {}).get("question", ""), "stratum": raw.get(sid, {}).get("stratum"),
                       "seeds": [id2node[x] for x in batch["start_nodes_mask"][0].nonzero(as_tuple=True)[0].tolist()], "targets": []}
                for j in best:
                    node = int(did[j]); gname = id2node[node]
                    # baseline: full gate -> the routes we follow through every condition
                    em.resp_gate = True; em._gate_layers = None; em._gate_nodes = None
                    paths_out, fm, raw_paths, dropped = self._interpret_target(graph, batch, node, em, eta, id2node, id2rel)
                    seeds = set(batch["start_nodes_mask"][0].nonzero(as_tuple=True)[0].tolist())
                    route_senders = sorted({int(h) for p_ in raw_paths for (h, t, r) in p_})
                    L = max((len(p_) for p_ in raw_paths), default=1)
                    mask = torch.zeros(graph.num_nodes, dtype=torch.bool); mask[route_senders] = True
                    conds = [("gate_on", dict(resp_gate=True, _gate_layers=None, _gate_nodes=None)),
                             ("gate_off", dict(resp_gate=False, _gate_layers=None, _gate_nodes=None)),
                             ("gate_layers_attributed", dict(resp_gate=True, _gate_layers=set(range(L)), _gate_nodes=None)),
                             ("gate_layers_later", dict(resp_gate=True, _gate_layers=set(range(L, n_layers)), _gate_nodes=None)),
                             ("gate_route_senders_only", dict(resp_gate=True, _gate_layers=None, _gate_nodes=mask)),
                             ("gate_all_but_route_senders", dict(resp_gate=True, _gate_layers=None, _gate_nodes=~mask))]
                    tgt = {"doc": gname, "attributed_layers": L, "n_route_senders": len(route_senders),
                           "frontier_mean_resp": fm, "routes": [{"hops": p["hops"], "weight_beam": p["weight"]} for p in paths_out],
                           "conditions": {}}
                    try:
                        tgt["gate_attribution"] = self._gate_attribution(graph, batch, node, em, id2node, raw_paths)
                    except Exception as _e:   # attribution is additive to the decomposition; never lose the rest
                        tgt["gate_attribution"] = {"error": str(_e)[:300]}
                        print(f"[gate-decomp] attribution failed for {sid} -> {gname}: {_e}", flush=True)
                    for cname, cfg in conds:
                        for k_, v_ in cfg.items(): setattr(em, k_, v_)
                        did_c, ch_c, _ = self._channel_scores(graph, batch)
                        ranks = {k: int((v > v[j]).sum().item()) + 1 for k, v in ch_c.items()}
                        eg, pr = self._edge_grads_for(graph, batch, node, em)
                        w_direct = self._direct_path_weights(graph, eg, raw_paths)
                        top_here, top_w = pr.get(node, ([], []))
                        top_here, top_w, _ = self._valid_paths(top_here, top_w, seeds)
                        tgt["conditions"][cname] = {"rank": ranks, "graph_score": float(ch_c["graph"][j]),
                                                    "graph_score_gap_to_top": float(ch_c["graph"].max() - ch_c["graph"][j]),
                                                    "route_weights_direct": w_direct,
                                                    "top_route_here": ([{"head": id2node[h], "rel": id2rel.get(r, str(r)), "tail": id2node[t]} for (h, t, r) in top_here[0]] if top_here else None),
                                                    "top_route_here_weight": (float(top_w[0]) if top_here else None)}
                    em.resp_gate = True; em._gate_layers = None; em._gate_nodes = None
                    ga = tgt.get("gate_attribution", {})
                    if "score_gate_on" in ga:
                        ga["delta_score_on_minus_off"] = float(tgt["conditions"]["gate_on"]["graph_score"] - tgt["conditions"]["gate_off"]["graph_score"])
                    rec["targets"].append(tgt)
                    c = tgt["conditions"]
                    print(f"[gate-decomp] {sid} -> {gname}: graph rank on {c['gate_on']['rank']['graph']} off {c['gate_off']['rank']['graph']} "
                          f"| route-1 weight on {c['gate_on']['route_weights_direct'][:1]} off {c['gate_off']['route_weights_direct'][:1]} "
                          f"L01 {c['gate_layers_attributed']['route_weights_direct'][:1]} later {c['gate_layers_later']['route_weights_direct'][:1]} "
                          f"own {c['gate_route_senders_only']['route_weights_direct'][:1]} others {c['gate_all_but_route_senders']['route_weights_direct'][:1]}", flush=True)
                results.append(rec)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        json.dump(results, open(out_path, "w"), indent=1)
        print(f"[gate-decomp] wrote {len(results)} queries -> {out_path}")
        return results

    def _min_hops(self, graph, seeds, targets, max_hops):
        """Fewest hops from any seed to each target node (undirected BFS, sparse matmul frontier),
        None when the target is not reached within max_hops."""
        if not seeds or not targets:
            return {}
        adj = getattr(self, "_bfs_adj", None)
        if adj is None or getattr(self, "_bfs_adj_key", None) != id(graph):
            ei = graph.edge_index
            r2 = torch.cat([ei[0], ei[1]]); c2 = torch.cat([ei[1], ei[0]])
            adj = torch.sparse_coo_tensor(torch.stack([r2, c2]), torch.ones(r2.numel(), device=ei.device),
                                          (graph.num_nodes, graph.num_nodes)).coalesce()
            self._bfs_adj, self._bfs_adj_key = adj, id(graph)
        n = graph.num_nodes
        visited = torch.zeros(n, device=adj.device); visited[seeds] = 1.0
        frontier = visited.clone()
        out = {int(t): (0 if visited[t] > 0 else None) for t in targets}
        for h in range(1, int(max_hops) + 1):
            if all(v is not None for v in out.values()):
                break
            frontier = ((torch.sparse.mm(adj, frontier.unsqueeze(1)).squeeze(1) > 0).float() * (1 - visited))
            if frontier.sum() == 0:
                break
            visited = visited + frontier
            for t in out:
                if out[t] is None and frontier[t] > 0:
                    out[t] = h
        return out

    def interpret(self, qids, out_path, probes_path=None, num_beam=10, path_topk=5,
                  max_golds=2, top_views=3, do_paths=True, golds=None, necessity=False, distractor=False, dump_k=0):
        """Path interpretations, NBFNet-style, for the graph channel of the fusion model.

        For each requested query: the rank of every gold under each channel (fused, graph
        alone, multi-view scorer, raw Qwen3 cosine), the top-k paths from the query's seed
        frames to its best-ranked gold (beam search over the gradient of the graph score
        w.r.t. each layer's edge weights, exactly the GFM-RAG / NBFNet recipe), and along
        every path the CCMP responsibility of each hop's sender node at the layer it was
        used, normalised by that layer's frontier mean (i.e. the gate the model applied).
        Also the scorer's views that matched the gold best, with their probe text.
        """
        import numpy as np
        self.model.eval()
        em = self.model.base.entity_model
        em.num_beam, em.path_topk = int(num_beam), int(path_topk)
        eta = float(getattr(em, "resp_eta", 0.5))
        probes = {}
        if probes_path and os.path.exists(probes_path):
            for line in open(probes_path):
                if line.strip():
                    r = json.loads(line); probes[str(r["id"])] = r.get("probes", [])
        want = [str(q) for q in qids]
        results = []
        for test_dataset in self.eval_graph_dataset_loader:
            src = test_dataset.data
            graph = src.graph.to(self.device)
            data = src.test_data
            id2node = src.id2node
            id2rel = {v: k for k, v in src.rel2id.items()}
            raw = {str(x["id"]): x for x in src.raw_test_data}
            pos = {}
            for i in range(len(data)):
                sid = str(data[i]["id"])
                if sid in want: pos[sid] = i
            missing = [q for q in want if q not in pos]
            if missing:
                print(f"[interpret] {len(missing)} requested ids not in {test_dataset.name}: {missing[:5]}")
            for sid in want:
                if sid not in pos: continue
                item = data[pos[sid]]
                batch = {"question_embeddings": item["question_embeddings"].unsqueeze(0).to(self.device),
                         "start_nodes_mask": item["start_nodes_mask"].unsqueeze(0).to(self.device),
                         "target_nodes_mask": item["target_nodes_mask"].unsqueeze(0).to(self.device),
                         "id": [sid]}
                did, ch, (tab, row) = self._channel_scores(graph, batch)
                n_doc = did.numel()
                gold_pos = (batch["target_nodes_mask"][0, did] > 0).nonzero(as_tuple=True)[0]
                doc_name = [id2node[int(did[j])] for j in gold_pos.tolist()]
                ranks = {}
                for k, v in ch.items():
                    ranks[k] = {doc_name[a]: int((v > v[j]).sum().item()) + 1 for a, j in enumerate(gold_pos.tolist())}
                # `golds` pins the documents to interpret (e.g. the one gold the picker chose); otherwise
                # the query's best-ranked golds under the fused score.
                pinned = set((golds or {}).get(sid, []))
                forced = [j for j in gold_pos.tolist() if id2node[int(did[j])] in pinned]
                best = forced if forced else sorted(gold_pos.tolist(), key=lambda j: ranks["fused"][id2node[int(did[j])]])[:max_golds]
                # scorer views for each gold
                if tab is not None:
                    Hq = np.asarray(tab["H"][row])[:, tab["col"]].astype(np.float32)   # [J, n_doc]
                    vmask = tab["mask"][row].numpy() > 0
                else:                   # operator-scorer checkpoint: no hypothetical-answer views
                    Hq, vmask = None, None
                seeds = batch["start_nodes_mask"][0].nonzero(as_tuple=True)[0].tolist()
                # Structural floor for the hop-count figure: the fewest hops from ANY seed node to each
                # gold, undirected BFS on the graph, capped at the reasoner's depth. The reasoner's top
                # path can only be this long or longer; the gap between the two is what the figure shows.
                min_hops = self._min_hops(graph, seeds, [int(did[j]) for j in gold_pos.tolist()], len(em.layers))
                rec = {"id": sid, "question": raw.get(sid, {}).get("question", ""),
                       "stratum": raw.get(sid, {}).get("stratum"), "golds": doc_name, "ranks": ranks,
                       "n_doc": int(n_doc), "seeds": [id2node[x] for x in seeds], "targets": []}
                if dump_k:
                    # per-channel top-K (doc, score) lists, for post-hoc fusion experiments
                    rec["channels"] = {k: [[id2node[int(did[jj])], float(v[jj])] for jj in torch.topk(v, min(int(dump_k), v.numel())).indices.tolist()] for k, v in ch.items()}
                    rec["gold_scores"] = {k: {doc_name[a]: float(v[j]) for a, j in enumerate(gold_pos.tolist())} for k, v in ch.items()}
                    _g = getattr(self.model, "_last_gamma", None); rec["gamma_q"] = float(_g[0]) if _g is not None else None
                for j in best:
                    gname = id2node[int(did[j])]
                    tv = [(int(a), float(Hq[a, j])) for a in np.argsort(-Hq[:, j]) if vmask[a]][:top_views] if Hq is not None else []
                    views = [{"view": a, "match": m, "text": (probes.get(sid, [None] * (a + 1))[a] if a < len(probes.get(sid, [])) else None)}
                             for a, m in tv]
                    if not do_paths:        # scan mode: ranks and views only, no gradient beam search
                        rec["targets"].append({"doc": gname, "rank": {k: ranks[k][gname] for k in ranks},
                                               "dense_cos": float(ch["dense"][j]), "views": views, "min_hops": min_hops.get(int(did[j])),
                                               "frontier_mean_resp": [], "paths": []})
                        continue
                    # ---- paths: gradient beam search on the GRAPH channel's score for this gold
                    paths_out, fm, paths, dropped = self._interpret_target(graph, batch, int(did[j]), em, eta, id2node, id2rel)
                    nec = None
                    if necessity and paths:
                        nec = {}
                        for kk in (1, 3):
                            nec[f"top{kk}"] = self._rank_without(graph, batch, [x for p_ in paths[:kk] for x in p_], int(did[j]))
                        nec["random"] = self._rank_without(graph, batch, None, int(did[j]), n_random=nec["top3"]["removed"])
                    rec["targets"].append({"doc": gname, "rank": {k: ranks[k][gname] for k in ranks},
                                           "dense_cos": float(ch["dense"][j]), "views": views, "min_hops": min_hops.get(int(did[j])),
                                           "frontier_mean_resp": fm, "paths": paths_out, "dropped_paths": dropped, "necessity": nec})
                    print(f"[interpret] {sid} -> {gname}: ranks {rec['targets'][-1]['rank']} | {len(paths_out)} paths ({dropped} artefacts dropped)")
                if do_paths and distractor:
                    order = torch.argsort(ch["graph"], descending=True).tolist(); gset = set(gold_pos.tolist())
                    jd = next((o for o in order if o not in gset), None)
                    if jd is not None:
                        dp, dfm, draw, ddrop = self._interpret_target(graph, batch, int(did[jd]), em, eta, id2node, id2rel)
                        rec["distractor"] = {"doc": id2node[int(did[jd])], "rank": {k: int((v > v[jd]).sum().item()) + 1 for k, v in ch.items()},
                                             "frontier_mean_resp": dfm, "paths": dp, "dropped_paths": ddrop}
                        if necessity and draw:
                            rec["distractor"]["necessity"] = {"top1": self._rank_without(graph, batch, [x for p_ in draw[:1] for x in p_], int(did[jd]))}
                if not do_paths and len(results) % 25 == 0:
                    print(f"[interpret] scanned {len(results) + 1}/{len(want)}", flush=True)
                results.append(rec)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        json.dump(results, open(out_path, "w"), indent=1)
        print(f"[interpret] wrote {len(results)} queries -> {out_path}")
        return results

    def _cqig_maybe_calibrate(self, graph, batch, task_dataset):
        """Called from train_step. Keeps the TRAIN graph's statistics current; mu drifts as
        the model trains, so it is recomputed every CQIG_RECAL steps."""
        g = self._cqig_gate()
        if g is None:
            return
        self._cqig_step = getattr(self, "_cqig_step", 0) + 1
        if not self._cqig_select_reference(batch, task_dataset):
            g.mode = "off"                          # no bank yet: run ungated, not wrongly
            return
        key = self._cqig_key(task_dataset)
        due = (not g.calibrated(key)) or (
            self._cqig_recal > 0 and self._cqig_step % self._cqig_recal == 0)
        if due:
            self._cqig_train_key = self._cqig_calibrate(graph, task_dataset, "train")
        else:
            self.model.cqig_use_graph(key)          # eval may have left another graph active

    def _cqig_for_inference(self, task_dataset):
        """Calibrate the graph about to be scored, from the SOURCE bank re-linked to it.

        This is the zero-shot path: the target corpus supplies its vocabulary and its graph,
        and nothing else. Its own queries are never inspected.
        """
        g = self._cqig_gate()
        if g is None:
            return
        if not g.ref_bank:
            # The bank is frozen a few hundred training steps in. An evaluation before that
            # must run UNGATED: the only calibrated graph is the train graph, and its mu has
            # the wrong number of rows for this one.
            g.mode = "off"
            return
        graph = task_dataset.graph.to(self.device)
        key = self._cqig_key(task_dataset)
        due = (not g.calibrated(key)) or (
            self._cqig_recal > 0
            and getattr(self, "_cqig_infer_at", -1) != getattr(self, "_cqig_step", 0))
        if due:
            self._cqig_calibrate(graph, task_dataset, "inference (source refs re-linked)")
            self._cqig_infer_at = getattr(self, "_cqig_step", 0)
        else:
            self.model.cqig_use_graph(key)

    def _create_task_dataset(self, dataset, is_train=True, **kw):
        """The one seam that train, evaluate AND predict all pass through.

        Evaluation and prediction run on a different graph from training, and mu has one
        row per node, so scoring the test graph against the train graph's statistics is not
        merely wrong but shape-invalid. Calibrating here means every consumer gets the
        right graph's statistics without each one needing its own hook.
        """
        ds = super()._create_task_dataset(dataset, is_train=is_train, **kw)
        g = self._cqig_gate()
        if g is not None:
            # The vocabulary rides along so CQIG can re-link its source problems by NAME.
            src = getattr(dataset, "data", dataset)
            ds.cqig_node2id = getattr(src, "node2id", None)
            ds.cqig_id2node = getattr(src, "id2node", None)
            if is_train:
                self._cqig_train_key = self._cqig_key(ds)
            else:
                self._cqig_for_inference(ds)
        return ds

    def train_step(self, batch, task_dataset):
        graph = task_dataset.graph.to(self.device)
        batch = query_utils.cuda(batch, device=self.device)
        self._cqig_maybe_calibrate(graph, batch, task_dataset)

        pred = self.parallel_model(graph, batch)          # FUSED [B, N]
        target = batch["target_nodes_mask"]

        total = torch.tensor(0.0, device=self.device, requires_grad=True)
        step_metrics = {}

        if self._objective == "hardneg":
            did = self.model._doc_ids
            # Whichever scorer the model's `semantic` setting selected: handcrafted
            # operator or the learned sorted-MLP. Both contrastive terms below mine
            # their negatives from it, so the graph is always trained to fix the misses
            # of the scorer actually in use.
            s_op = self.model._s_op                        # [B, n_doc] detached
            tgt_doc = target[:, did]
            raw = getattr(self.model, "_raw_doc", None)
            # THE AUXILIARY LOSS MUST TRAIN WHAT THE FUSION RANKS ON. The fused score
            # uses z(s_graph); z is scale-invariant, so a contrastive loss on the RAW
            # graph score can be driven down simply by scaling every score up, a
            # direction that leaves the fused ranking untouched. That is a free way for
            # hn_graph to fall while hn_fused and validation do not move. Mining still
            # uses the raw score, because top-k is scale-free either way.
            raw_l = getattr(self.model, "_raw_doc_z", None)
            if raw_l is None:
                raw_l = raw                     # older reasoners that cache only the raw score
            g_mine = raw.detach() if (raw is not None and self._hn_graph > 0) else None
            # ONE LINEUP FOR BOTH TERMS. Same golds, same hubs, same randoms, so the only
            # thing that differs between the fused and graph-alone losses is which score
            # is being trained -- which is the whole point of having both.
            lineups = self.build_lineups(tgt_doc, s_op, g_mine)
            # (1) fused-score contrastive: trains gate/router + operator scalars + graph jointly
            l_fused = self._contrastive_hardneg(
                pred[:, did], tgt_doc, s_op, g_mine=g_mine, miss_w=self._miss_w_fused,
                lineups=lineups
            )
            step_metrics["hn_fused"] = l_fused.item()
            total = total + l_fused
            # (2) graph-alone contrastive: trains the GNN DIRECTLY (survives a near-zero gate/alpha)
            # `_aux_ok` is False for the semantic-prior reasoner, which has no standalone
            # graph ranking: its `_raw_doc` is the belief CORRECTION, kept for diagnostics.
            # Training a contrastive on it would demand that the correction be a ranker in
            # its own right, which is not what it is.
            if self._aux_w > 0 and raw_l is not None and getattr(self.model, "_aux_ok", True):
                if self._resid_prior:
                    # REPLACES the standard graph-alone term rather than adding to it, so
                    # the arm differs from its control in exactly one thing: which
                    # documents count as negatives for the graph channel.
                    prior_doc = self._prior_doc(graph, batch, did)
                    l_graph, rstats = self._contrastive_resid(
                        raw_l, tgt_doc, prior_doc, lineups)
                    step_metrics.update(rstats)
                else:
                    l_graph = self._contrastive_hardneg(
                        raw_l, tgt_doc, s_op, g_mine=g_mine, miss_w=self._miss_w_aux,
                        lineups=lineups
                    )
                step_metrics["hn_graph"] = l_graph.item()
                total = total + self._aux_w * l_graph
            # (3) CCMP: intermediate responsibility. ADDED to the endpoint terms, not
            # substituted for them -- the graph must still be trained against the
            # scorer's own mistakes, which is its job. This term only says WHICH
            # intermediate nodes should be carrying the query while it does that.
            if self._ccmp and self._ccmp_w > 0:
                preds = getattr(self.model.base.entity_model, "_resp_pred", None)
                if preds:
                    tg = self._ccmp_targets(graph, batch, did, s_op, g_mine=g_mine)
                    l_crp, cstats = self._ccmp_loss(preds, tg)
                    if l_crp is not None:
                        step_metrics["ccmp"] = l_crp.item()
                        step_metrics.update(cstats)
                        _gs = getattr(self.model.base.entity_model,
                                      "_resp_gstat", None)
                        if _gs:
                            step_metrics["ccmp_gmax"] = max(x[1] for x in _gs)
                            step_metrics["ccmp_gp95"] = max(x[2] for x in _gs)
                        total = total + self._ccmp_w * l_crp
        else:
            for sft_loss in self.loss_functions:
                tids = graph.nodes_by_type[sft_loss.target_node_type]
                loss = sft_loss.loss_fn(pred[:, tids], target[:, tids])
                step_metrics[sft_loss.name] = loss.item()
                total = total + sft_loss.weight * loss

            # graph-alone auxiliary term (teach the GNN independently of the gate)
            if self._aux_w > 0 and getattr(self.model, "_raw_doc", None) is not None:
                did = self.model._doc_ids
                aux = self._aux_loss_fn(self.model._raw_doc, target[:, did])
                step_metrics["aux_graph"] = aux.item()
                total = total + self._aux_w * aux

        # POPULARITY ANCHOR, only under semantic='mlp'. The learned scorer's popularity
        # term is a network, and the fusion's ranking gradient reaches it exactly as the
        # standalone run's did -- so it needs the same anchor, or p_hat stops meaning "how
        # generally matchable is this paper" and starts meaning "whatever lowers this loss".
        # Zero for the operator channel, which has no predictor.
        if self._sem_pop_lambda > 0 and hasattr(self.model, "semantic_aux_loss"):
            l_pop = self.model.semantic_aux_loss()
            if torch.is_tensor(l_pop):
                step_metrics["pop_anchor"] = l_pop.item()
                total = total + self._sem_pop_lambda * l_pop

        step_metrics["loss"] = total
        return step_metrics
