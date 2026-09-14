"""
fusion_reasoner.py — CARGO fusion: v16sc G-Reasoner + operator, EVERYTHING learned jointly in one run.

Wraps the standard GraphReasoner. The operator is recomputed LIVE from its raw ingredients so its
weights and exponent are trainable (matches the interim report: beta and the fusion weights are learned):

    S_op      = w0 z(dense) + w1 z(S / dem^beta) + w2 z(M / dem^beta)   # dem = total_S - S (anti-hub)
    fused_doc = z(S_op) + gamma_q * relu( z(graph_doc) )
    gamma_q   = softplus( gate(coverage) )                             # per-query gate >= 0

TWO FUSION FORMS, selected by FUSION_FORM (default 'additive' = the arithmetic above,
bit-identical to every run made before the option existed).

    FUSION_FORM=mixture   fused_doc = log( (1 - a_q) p_s + a_q p_g )
                          p_s = softmax( z(S_op) )            # tau_s fixed at 1
                          p_g = softmax( z(graph_doc)/tau_g ) # tau_g learned in [0.25, 4]
                          a_q = a_max * sigmoid( router(phi_q) )

The additive form cannot promote a document, at any gamma it actually learns: a z-score
over this corpus tops out near 5-10 and the converged gamma is 0.036, so the graph's
largest possible contribution to any document is ~0.36 z-units against a semantic top-50
spread of 1-3. It can reorder neighbours, never rescue a gold the semantic scorer missed,
which is exactly the cross-domain case. Mixing calibrated DISTRIBUTIONS instead of scores
makes the response exponential, so the graph's few confident documents can move hundreds
of places, while a flat p_g leaves the ranking exactly unchanged rather than adding noise
to every document. The cost is that the graph's confident MISTAKES are promoted just as
hard, and the graph ranking is substantially weaker overall, so a concentrated p_g puts
large mass on wrong documents on many queries; a_max bounds that. Full detail, including
why tau_s is frozen and why tau_g is bounded rather than free, in __init__.

Trained end-to-end from one ranking loss on fused_doc:
  - the GNN weights (the v16sc graph reasoner, from scratch),
  - the gate (when to trust the graph), and
  - the operator scalars w=[w0,w1,w2] and beta.
The BGE encoder that produced dense/S/M is frozen (we only learn the handful of combination scalars).
w, beta are warm-started at the fitted values and learn at the base LR (op_lr_scale=1.0) — 4 params on
a near-convex objective, so they converge fast. relu => the graph can only promote a doc (aggregate floor).

Ingredients come from env OPERATOR_COMPONENTS (train) / OPERATOR_COMPONENTS_TEST (test): npz with
{dense,S,M: float16 [Q x n_doc] in nodes.csv doc order, total_S: float32 [n_doc], query_ids:[...]}.
Query ids are split-unique, so both tables are merged and looked up by batch['id'].

MULTI-CORPUS TRAINING. Either variable also accepts a COMMA-SEPARATED list of npz
paths, which is what joint Physics+Biology training needs: the trainer walks a list
of graphs and each carries its own corpus, so `dense` has a different document-column
count per graph (physics 10,349 vs biology 15,588) and one merged table cannot serve
both. Each file becomes its own table, and because query ids are unique across SIR-4
datasets the existing id -> (tag, row) lookup already routes a batch to the right one.
GraphDatasetLoader keeps one graph resident at a time, so a batch never spans two
corpora and the single-tag assert in _operator still holds.

THE SEMANTIC CHANNEL IS SELECTABLE (`semantic:` in the config).

  "operator" (default)  the handcrafted scorer above. Unchanged, bit-identical.
  "mlp"                 the LEARNED scorer from semantic_scorer.py: one MLP reading the
                        SORTED vector of per-answer match scores, with a popularity
                        discount predicted from the document embedding alone.

Only the semantic channel changes. The gate, the relu floor, the fusion arithmetic and the
hard-negative mining are identical either way, so a run pair isolates "handcrafted vs learned
semantic scorer" with the graph half held fixed. The learned scorer is warm-started from a
trained 5d checkpoint (SEMANTIC_CKPT + SEMANTIC_POPNET) and keeps training under the fusion's
ranking loss unless semantic_train=False.

WHY THE POPULARITY PREDICTOR AND NOT THE LEAVE-ONE-OUT TERM. The operator's anti-hub
denominator is total_S - S, i.e. a document's popularity measured from the OTHER queries'
hypothetical answers in the same split. Inside the fusion that is the same transductive
dependency it has always been. The learned scorer replaces it with a function of the
document's own embedding, so the fused model can score one query against an unseen corpus.

Ingredients come from env SEMANTIC_COMPONENTS / SEMANTIC_COMPONENTS_TEST (see
precompute_semantic_components.py). The per-answer matrix H is NOT copied into those files:
they carry the memmap's path plus the corpus->nodes.csv column permutation, and the rows for
one batch are sliced and permuted on demand.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from gfmrag.models.gfm_reasoner import GraphReasoner

W_INIT = (1.05, 1.05, 0.25)   # fitted operator fusion weights
BETA_INIT = 0.95              # fitted anti-hub exponent


class FusionGraphReasoner(nn.Module):
    def __init__(self, entity_model, feat_dim, gamma_init=0.5, gate_hidden=8,
                 op_lr_scale=1.0, semantic="operator", semantic_train=True,
                 cqig=False, cqig_lam=0.1, cqig_layers=None, cqig_norm=None,
                 cqig_rho=1e-3, cqig_grad_I=None, cqig_op=None, cqig_mu=None, **kwargs):
        # The operator was first proposed as `cqig_mode`. Accepted as an alias rather than
        # left to fall through **kwargs into GraphReasoner, where it would be a silent
        # no-op and the arm would train as the default operator under the other one's name.
        if "cqig_mode" in kwargs:
            cqig_op = kwargs.pop("cqig_mode") if cqig_op is None else cqig_op
        super().__init__()
        self.base = GraphReasoner(entity_model, feat_dim, **kwargs)
        assert semantic in ("operator", "mlp"), f"unknown semantic channel {semantic!r}"
        self.semantic = semantic

        # --- Cross-Query Informativeness Gating (off by default, see cqig.py) ---
        # Attaches a forward pre-hook to each conv layer. With cqig=False nothing is
        # registered at all, so the ungated arm is bit-identical.
        #
        # The three knobs that define an arm of the controlled experiment come from the
        # ENVIRONMENT when the config leaves them unset, for the same reason every other
        # run knob does: a Hydra override of a string like "3-6" has to survive quoting
        # through subprocess, an env var does not.
        # --- CCMP responsibility head (off by default) --------------------------
        # One projection per layer because self.dims may differ layer to layer, then a
        # SHARED trunk plus a layer embedding, so the predictor is one function of
        # (state, remaining depth) rather than L unrelated predictors. Built here, not
        # lazily on first forward, because a module created after the optimiser is
        # constructed never receives a gradient step and the arm would silently run as
        # its own control.
        _em = self.base.entity_model
        # --- Routing baselines for the CCMP comparison (off by default) ----------
        # ROUTE=astar  A*Net-style: the SAME responsibility head CCMP uses, but trained
        #              only through the ranking loss (no continuation targets) and applied
        #              as a hard top-K selection over the reached frontier per layer.
        # ROUTE=attn   RED-GNN-style: query-conditioned attention over each receiver's
        #              incoming edges, trained only through the ranking loss.
        # Either is a REPLACEMENT for CCMP, never an addition, so the three arms differ
        # in exactly one thing: where the routing signal comes from.
        _route = os.environ.get("ROUTE", "").strip().lower()
        assert _route in ("", "none", "astar", "attn"), f"unknown ROUTE={_route!r}"
        _route = "" if _route == "none" else _route
        _ccmp_on = os.environ.get("CCMP", "0") == "1"
        assert not (_ccmp_on and _route), "ROUTE replaces CCMP; unset one of CCMP / ROUTE"
        _em.route_mode = _route
        if _ccmp_on or _route == "astar":
            # RNG STATE SAVED AND RESTORED AROUND THIS BLOCK. Constructing these modules
            # draws from the global generator, so every parameter initialised AFTER this
            # point -- the fusion gate, the router, the operator scalars -- would start
            # from different values in the CCMP arm than in the control. The pair would
            # then differ in the loss AND in the initialisation, and the control's own
            # reruns already move 0.0118 nDCG@5.
            _rng = torch.get_rng_state()
            _dims = list(_em.dims)[:-1]
            _h = int(os.environ.get("CCMP_HID", "64"))
            _em.resp_proj = nn.ModuleList([nn.Linear(_d, _h) for _d in _dims])
            _em.resp_emb = nn.Embedding(len(_dims), _h)
            nn.init.zeros_(_em.resp_emb.weight)
            _em.resp_head = nn.Sequential(nn.ReLU(), nn.Linear(_h, _h),
                                          nn.ReLU(), nn.Linear(_h, 1))
            # Output bias at 0 => yhat starts at 0.5 => a mean-normalised gate starts at
            # exactly 1.0 at every node, so the CCMP arm's epoch 0 is the control's epoch 0
            # and any difference later is the loss, not a different initialisation.
            nn.init.zeros_(_em.resp_head[-1].bias)
            nn.init.zeros_(_em.resp_head[-1].weight)
            _em.resp_gate = os.environ.get("CCMP_GATE", "1") == "1"
            _em.resp_gate_norm = os.environ.get("CCMP_GATE_NORM", "1") == "1"
            _em.resp_eta = float(os.environ.get("CCMP_ETA", "0.5"))
            torch.set_rng_state(_rng)
            if _route == "astar":
                _em.route_k = int(os.environ.get("ROUTE_K", "1024"))
                _em.resp_gate = True
                print(f"[route] mode=astar: same head ({len(_dims)} layers, hid={_h}), "
                      f"hard top-{_em.route_k} of the reached frontier per layer, priority "
                      f"trained by the ranking loss only (no CCMP targets)", flush=True)
            else:
                print(f"[ccmp] responsibility head on {len(_dims)} layers, hid={_h}, "
                      f"gate={_em.resp_gate} mean_norm={_em.resp_gate_norm} "
                      f"eta={_em.resp_eta}", flush=True)
        else:
            _em.resp_proj = None
            _em.resp_gate = False
        if _route == "attn":
            # Same RNG discipline as the CCMP head, same hidden width, and a zero-initialised
            # output layer so attention starts UNIFORM: after degree normalisation every
            # edge weight is exactly 1 and epoch 0 is the control's epoch 0.
            _rng = torch.get_rng_state()
            _dims = list(_em.dims)[:-1]
            _h = int(os.environ.get("ROUTE_HID", os.environ.get("CCMP_HID", "64")))
            _em.attn_node = nn.ModuleList([nn.Linear(_d, _h) for _d in _dims])
            _em.attn_rel = nn.Linear(int(_em.dims[0]), _h)
            _em.attn_query = nn.Linear(int(_em.dims[0]), _h)
            _em.attn_emb = nn.Embedding(len(_dims), _h)
            nn.init.zeros_(_em.attn_emb.weight)
            _em.attn_out = nn.Linear(_h, 1)
            nn.init.zeros_(_em.attn_out.weight)
            nn.init.zeros_(_em.attn_out.bias)
            _em.attn_norm = os.environ.get("ROUTE_ATTN_NORM", "1") == "1"
            torch.set_rng_state(_rng)
            print(f"[route] mode=attn: query-conditioned edge attention on {len(_dims)} layers "
                  f"(hid={_h}, degree-normalised={_em.attn_norm}), trained by the ranking "
                  f"loss only", flush=True)
        else:
            _em.attn_node = None

        self.cqig = None
        if cqig:
            from gfmrag.models.cqig import CQIGate
            spec = cqig_layers if cqig_layers is not None else os.environ.get("CQIG_LAYERS")
            norm = cqig_norm if cqig_norm is not None else os.environ.get("CQIG_NORM", "layer")
            grad = (cqig_grad_I if cqig_grad_I is not None
                    else os.environ.get("CQIG_GRAD_I", "0") == "1")
            op = cqig_op if cqig_op is not None else os.environ.get("CQIG_OP", "gate")
            mu = cqig_mu if cqig_mu is not None else os.environ.get("CQIG_MU", "ref")
            self.cqig = CQIGate(self.base.entity_model.layers, lam_init=cqig_lam,
                                rho=cqig_rho, gate_layers=spec, norm=norm,
                                grad_through_I=grad, op=op, mu=mu)
            print(f"[cqig] attached to {self.cqig.n_layers} layers; gating "
                  f"{sorted(i + 1 for i in self.cqig.gate_layers)} (1-indexed) "
                  f"= positions {sorted(self.cqig.gate_layers)}; norm={norm!r} "
                  f"lam_init={cqig_lam} grad_through_I={grad} op={self.cqig.op!r} "
                  f"mu={self.cqig.mu_source!r}")
            if self.cqig.mu_zero:
                print("[cqig] mu=zero: THIS IS THE ABLATION, NOT THE METHOD. mu is forced to "
                      "0, so I = ||h||^2/s is an activation-MAGNITUDE gate and the reference "
                      "bank contributes nothing. The references are still run and discarded "
                      "so the two arms differ in mu alone. If this reproduces the method's "
                      "score, the bank was never load-bearing.")
            if self.cqig.mu_zero and self.cqig.centring:
                raise AssertionError(
                    "cqig mu='zero' with a centring op is a literal no-op: the operator "
                    "subtracts c*mu and mu is 0, so the model is the ungated reasoner. "
                    "Ablate mu against op='gate', which is the arm whose result is in doubt.")
            if self.cqig.centring:
                print(f"[cqig] op={self.cqig.op!r}: the state is CENTRED, h - c*mu, not "
                      f"scaled. c is "
                      + ("lam (constant, query-independent; alpha and dtau get no "
                         "gradient in this arm)" if self.cqig.op == "centre-fixed"
                         else "1-g, so an uninformative node loses more of its background")
                      + ". Nodes no reference reached have mu=0 and are left ALONE here, "
                        "where a gating arm damps them hardest.")

        # --- per-query gate: gamma_q = softplus(gate(phi_q)) ---
        #
        # ROUTER FEATURES ARE NOW SHARED BY BOTH FUSION FORMS. This used to be read only
        # inside the mixture branch, so on an ADDITIVE arm `FUSION_ROUTER` was accepted,
        # logged, and never used: the gate was nn.Linear(1, ...) on the semantic top-5
        # mean alone. Every additive run recorded before this change was therefore a `cov`
        # run whatever its flag said, and the matsci two-stage 2x2's legacy-vs-phi5 axis
        # compared two identical configurations (measured: -0.0004 / +0.0021 / -0.0019 /
        # -0.0038 against a 0.0118 noise floor, i.e. the null the wiring predicts).
        #
        # `full` is the default because a gate on semantic confidence alone cannot express
        # the one thing worth conditioning on: whether the GRAPH is worth listening to on
        # this query. Set FUSION_ROUTER=cov to reproduce any pre-existing additive run.
        self.router_feats = os.environ.get("FUSION_ROUTER", "full").lower()
        assert self.router_feats in ("full", "cov"), self.router_feats
        n_feat = 5 if self.router_feats == "full" else 1
        self.gate = nn.Sequential(
            nn.Linear(n_feat, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1)
        )
        # INIT IS UNCHANGED BY n_feat. The last layer's weight is zeroed, so gamma_q is
        # exactly softplus(bias) = gamma_init for every query at step 0 no matter how many
        # features feed it. A `full` arm and a `cov` arm therefore start from the identical
        # ranking and diverge only as the gate learns, which is what makes them a pair.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(math.expm1(max(gamma_init, 1e-3))))
        self._gamma_lowp_note = False

        # --- FUSION FORM: additive z-scores (default) or a mixture of calibrated
        #     distributions (FUSION_FORM=mixture) ------------------------------------
        #
        # WHY A SECOND FORM AT ALL. The additive form cannot promote a document, at any
        # gamma it actually learns. A z-score over a 15k-60k document corpus tops out
        # near 5-10 even for the graph's single most confident paper, and the converged
        # gamma is 0.036, so the largest contribution the graph can make to ANY document
        # is about 0.36 z-units. The semantic channel's own spread across its top 50 is
        # 1-3 z-units. The graph is therefore structurally unable to lift a paper from
        # outside the top 50 into the top 5 however certain it is; it can only reorder
        # documents the semantic scorer already placed next to each other. That is a
        # property of the functional form, not of the gate, the gating authority lam, or
        # the bf16 freeze, and it caps the measured contribution at +0.017 nDCG@5.
        #
        # THE MIXTURE. Combine calibrated distributions instead of scores:
        #
        #     log p_s = z(s_op)      - logsumexp(z(s_op))        # tau_s FIXED at 1
        #     log p_g = z(g)/tau_g   - logsumexp(z(g)/tau_g)
        #     a_q     = a_max * sigmoid(router(phi_q))           # in [0, a_max]
        #     fused   = log( (1-a_q) p_s + a_q p_g )
        #
        # Four properties, each of which the additive form lacks:
        #
        #  1. EXPONENTIAL RESPONSE. p_g for a document the graph ranks first out of 60k
        #     is O(0.1); p_s in the semantic tail is O(1e-5). Even at a_q = 0.05 the
        #     second term dominates and that document moves hundreds of places. This is
        #     the only mechanism by which a 0.27-nDCG channel can rescue a gold the
        #     semantic scorer missed outright, which is the cross-domain case.
        #  2. AN UNINFORMATIVE GRAPH IS EXACTLY INERT. A flat p_g is a constant, and
        #     logaddexp(const, log(1-a) + log p_s) is strictly increasing in p_s, so the
        #     ranking does not move AT ALL -- not approximately, exactly. The weak-channel
        #     problem is handled by the algebra instead of by keeping gamma small. tau_g
        #     controls how sharp the graph's vote is relative to the semantic's, which is
        #     the one quantity that matters, and it is the only new scalar.
        #  3. MIXTURE, NOT PRODUCT. A product of experts (log p_s + b log p_g) would let
        #     the weak channel VETO a document the semantic scorer got right, driving its
        #     score toward zero. A mixture can only ADD mass, so no document's score ever
        #     falls, a_q -> 0 recovers the semantic ranking and a_max bounds the worst case.
        #     READ THAT AS A STATEMENT ABOUT SCORES, NOT RANKS. Rank is zero-sum: a
        #     confidently-wrong graph promotes other documents OVER a correct one and
        #     demotes it just as effectively as a veto would. Measured: a gold at semantic
        #     rank 1 falls to rank 3 when the graph is certain about six wrong documents,
        #     its own score gain being exactly 0.0 while theirs are 0.13 to 9.75. The
        #     additive form's relu is non-negative too, so this floor is NOT what
        #     distinguishes the two forms; a_max and property 2 are.
        #  4. GRADIENT FROM THE RANKING LOSS EVERYWHERE. relu(z(g)) zeroes the FUSED
        #     loss's gradient to the GNN for every document below the graph's own mean --
        #     precisely the buried golds. Under the mixture the graph softmax couples the
        #     whole corpus, so every document gets a gradient from the final ranking loss,
        #     attenuated rather than deleted. NOTE the precise claim: the graph is not
        #     gradient-starved today, because AUX_W=1.0 runs a separate graph-only
        #     contrastive on _raw_doc_z that never passes through the relu. What changes
        #     is that the RANKING loss itself now reaches every graph logit, instead of
        #     the graph learning about its buried documents only from the auxiliary term.
        #
        # THE RISK, STATED PLAINLY. Property 1 works just as well on the graph's confident
        # MISTAKES. nDCG@5 = 0.27 does NOT establish a top-1 error rate -- with ~1.9 golds
        # per query it is not a Hits@1 measurement and nothing here should be read as one
        # -- but the graph ranking is substantially weaker overall, so a concentrated p_g
        # will put large mass on wrong documents on many queries. The additive form is
        # safe because it is inert; this one is useful because it is not. a_max is the
        # only thing bounding that, which is why it defaults to 0.5 rather than 1.0 and
        # why a_q is learned per query rather than fixed. Whether the rescues outnumber
        # the false promotions is the empirical question the run exists to answer.
        #
        # tau_s IS DEFINED AND FROZEN AT 1, NOT LEARNED. The training loss is already
        # `logsumexp(lineup) - gold`, a softmax cross-entropy, so a learnable tau_s would
        # be softmaxing a softmax: shrinking it sharpens log p_s, which lowers the loss
        # whenever the gold already leads, without improving any ranking. Frozen at 1 it
        # buys something instead: log p_s = z(s_op) - const, cross-entropy is shift
        # invariant per query, so AT INITIALISATION (a_q small) this arm's loss is
        # numerically the same objective the additive arm trains under. No LR retuning,
        # and the two arms stay comparable.
        self.fusion_form = os.environ.get("FUSION_FORM", "additive").lower()
        assert self.fusion_form in ("additive", "mixture"), (
            f"FUSION_FORM={self.fusion_form!r}; expected 'additive' or 'mixture'")
        self._fusion_note = ""
        if self.fusion_form == "mixture":
            self.a_max = float(os.environ.get("FUSION_AMAX", "0.5"))
            a_init = float(os.environ.get("FUSION_AINIT", "0.1"))
            assert 0.0 < a_init < self.a_max <= 1.0, (
                f"need 0 < FUSION_AINIT ({a_init}) < FUSION_AMAX ({self.a_max}) <= 1")
            # a_q must not start AT zero: sigmoid'(-inf) = 0 and the router would get no
            # gradient to tell queries apart with. 0.1 is small enough that epoch 1 is
            # still essentially the semantic ranking.
            # router_feats and n_feat are set once above, for BOTH forms; the mixture just
            # builds a second head on the same input.
            self.router = nn.Sequential(
                nn.Linear(n_feat, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1)
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.constant_(self.router[-1].bias,
                              math.log(a_init / (self.a_max - a_init)))
            # tau_g = exp(log_tau_g), so it stays positive without a clamp. log_tau_g = 0
            # is tau_g = 1, i.e. the graph's vote starts exactly as sharp as z(g) makes it.
            #
            # tau_g IS BOUNDED, NOT FREE. Two separate problems, only one of which is
            # already handled:
            #
            #   SCALE IDENTIFIABILITY -- handled, and this is why the graph channel is
            #   z-scored before it gets here. softmax(c*g / (c*tau_g)) = softmax(g/tau_g),
            #   so with RAW graph logits the readout could inflate its own scale while
            #   tau_g grew to match and the pair would not be separately identifiable.
            #   z() is exactly scale-invariant (z(c*g) = z(g)), so that degeneracy cannot
            #   arise. Feeding raw logits here would reintroduce it, and would also expose
            #   tau_g to the GNN's score scale drifting across epochs.
            #
            #   OVERCONFIDENCE -- NOT handled by z(), and this is the reason for the
            #   bounds. Shrinking tau_g sharpens p_g, which lowers the training
            #   cross-entropy on any query whose gold already leads, WITHOUT the graph
            #   ranking any better. That is the same degenerate direction that tau_s is
            #   frozen to avoid, and z-scoring does nothing about it. tau_g is therefore
            #   parameterised as tau_min + (tau_max - tau_min) * sigmoid(tau_hat), so the
            #   sharpest and flattest the graph's vote can get are both fixed in advance.
            #   The default [0.25, 4] is a factor of 4 either side of neutral.
            #
            # The stronger version of this is post-hoc calibration: train the reasoner,
            # freeze its readout, then fit tau_g and the router on held-out SOURCE queries
            # and freeze both for target inference, which makes tau_g a genuine
            # calibration parameter instead of one more jointly-optimised scale. That
            # costs an extra stage; the bounds are the cheap version of the same
            # protection. FUSION_TAUG=<float> freezes tau_g outright if that stage is
            # ever run and its fitted value is known.
            self.tau_min = float(os.environ.get("FUSION_TAU_MIN", "0.25"))
            self.tau_max = float(os.environ.get("FUSION_TAU_MAX", "4.0"))
            assert 0 < self.tau_min < self.tau_max, (self.tau_min, self.tau_max)
            tg = os.environ.get("FUSION_TAUG", "learn").lower()
            if tg == "learn":
                # tau_hat such that tau_g starts at exactly 1: the graph's vote begins
                # neither sharpened nor flattened relative to z(g).
                f = (1.0 - self.tau_min) / (self.tau_max - self.tau_min)
                assert 0.0 < f < 1.0, (
                    f"FUSION_TAU_MIN/MAX = [{self.tau_min}, {self.tau_max}] must bracket "
                    f"tau_g = 1, or the arm cannot start at the neutral sharpness")
                self.tau_g_hat = nn.Parameter(torch.tensor(math.log(f / (1.0 - f))))
            else:  # frozen at a given value, for the ablation or a calibrated fit
                v = float(tg)
                assert self.tau_min <= v <= self.tau_max, (
                    f"FUSION_TAUG={v} is outside [{self.tau_min}, {self.tau_max}]")
                f = (v - self.tau_min) / (self.tau_max - self.tau_min)
                self.register_buffer("tau_g_hat", torch.tensor(math.log(f / (1.0 - f))))
            print(f"[fusion] form=mixture: fused = log((1-a_q) p_s + a_q p_g), "
                  f"a_max={self.a_max} a_init={a_init} router={self.router_feats!r} "
                  f"({n_feat} features) tau_s=1 (frozen) tau_g={tg}. The diagnostic "
                  f"column printed as 'gamma' is now a_q, the mixing weight in "
                  f"[0, {self.a_max}] -- NOT the additive arm's gamma, and the two are "
                  f"not on the same scale.")
        else:
            # SAY WHICH ROUTER RAN. The additive arm used to print nothing about the
            # router, which is how FUSION_ROUTER stayed dead in this path unnoticed
            # across a whole 2x2. A run that does not log the knob it varied cannot be
            # audited from its own console output.
            print(f"[fusion] form=additive: fused = z(s_op) + gamma_q * relu(z(graph)), "
                  f"gamma_q = softplus(gate(phi_q)), router={self.router_feats!r} "
                  f"({n_feat} feature{'s' if n_feat > 1 else ''}), gamma_init={gamma_init}. "
                  + ("phi_q = [semantic confidence, semantic peakedness, graph confidence, "
                     "graph peakedness, channel agreement]." if n_feat > 1 else
                     "phi_q = the semantic top-5 mean alone (the pre-2026-08-16 behaviour)."))

        # --- trainable operator scalars (warm-started at fitted values; base LR via op_lr_scale) ---
        # effective value = init + op_lr_scale * delta, delta starts at 0. With Adam this makes the
        # operator learn at op_lr_scale x the graph/gate LR (1.0 = same rate).
        self.op_lr_scale = float(op_lr_scale)
        self.register_buffer("w_init", torch.tensor(W_INIT, dtype=torch.float32))
        self.w_delta = nn.Parameter(torch.zeros(3))
        self.beta_delta = nn.Parameter(torch.zeros(()))

        # --- operator raw ingredients (CPU float16; rows moved to GPU per batch) ---
        self._dense: dict[str, torch.Tensor] = {}
        self._S: dict[str, torch.Tensor] = {}
        self._M: dict[str, torch.Tensor] = {}
        self._totS: dict[str, torch.Tensor] = {}
        self._row: dict[str, tuple] = {}
        seen_paths: dict[str, str] = {}          # abspath -> tag already holding it
        for split, ev in ((("train", "OPERATOR_COMPONENTS"), ("test", "OPERATOR_COMPONENTS_TEST"))
                          if semantic == "operator" else ()):
            # Not loaded under semantic='mlp'. dense+S+M is 660 MB of resident CPU
            # memory on CS, and nothing in that arm reads it.
            raw = os.environ.get(ev)
            if not raw:
                continue
            paths = [x.strip() for x in raw.split(",") if x.strip()]
            for j, p in enumerate(paths):
                # THE SAME FILE FOR BOTH VARIABLES IS THE ZERO-SHOT IDIOM. Predict runs
                # have one corpus and set OPERATOR_COMPONENTS=OPERATOR_COMPONENTS_TEST=x,
                # which must stay legal: the duplicate check below exists to catch two
                # DIFFERENT tables claiming the same query ids (a real routing bug), not
                # one table registered twice (the same rows either way).
                ap = os.path.abspath(p)
                if ap in seen_paths:
                    print(f"[fusion] {ev}[{j}] is the same file as '{seen_paths[ap]}', reusing it")
                    continue
                # One table per file. The tag stays "train"/"test" for the single-file
                # case so existing runs and checkpoints are byte-identical; only a list
                # introduces the suffixed tags.
                tag = split if len(paths) == 1 else f"{split}#{j}"
                seen_paths[ap] = tag
                d = np.load(p, allow_pickle=True)
                self._dense[tag] = torch.from_numpy(np.asarray(d["dense"], dtype=np.float16))
                self._S[tag] = torch.from_numpy(np.asarray(d["S"], dtype=np.float16))
                self._M[tag] = torch.from_numpy(np.asarray(d["M"], dtype=np.float16))
                self._totS[tag] = torch.from_numpy(np.asarray(d["total_S"], dtype=np.float32))
                # A query id appearing in two tables would make routing order-dependent
                # and silently score half the batch against the wrong corpus.
                dup = [str(q) for q in d["query_ids"] if str(q) in self._row]
                assert not dup, (
                    f"{p}: {len(dup)} query ids already claimed by another components "
                    f"table (e.g. {dup[:3]}); tables must cover disjoint query sets")
                for i, q in enumerate(d["query_ids"]):
                    self._row[str(q)] = (tag, i)
                print(f"[fusion] loaded {ev}[{j}] as '{tag}': dense/S/M "
                      f"{tuple(self._dense[tag].shape)} ({len(d['query_ids'])} queries) "
                      f"{os.path.basename(p)}")
        if semantic == "operator":
            assert self._row, ("no operator components: set OPERATOR_COMPONENTS / "
                               "OPERATOR_COMPONENTS_TEST")
        else:
            self._init_semantic(semantic_train)

        self._raw_doc = None   # graph-alone doc scores, cached each forward for the aux loss
        self._doc_ids = None
        # [B, n_doc] detached SEMANTIC scores, cached for hard-negative mining. Named
        # _s_op for back-compatibility with the trainer and the routed reasoner, but it
        # holds whichever channel `semantic` selected, not necessarily the operator.
        self._s_op = None

    # ------------------------------------------------------------------ precision
    def _apply(self, *args, **kwargs):
        """Keep the gamma gate's weights in float32 through `model.to(dtype=...)`.

        SAME BUG AS CQIGate._apply, DIFFERENT MODULE. The trainer casts the whole model
        with `model.to(dtype=torch.bfloat16)` (utils/setup_training.py). A bfloat16 value
        has a 2^-8 relative ULP, so any parameter above |w| ~= 256*lr = 0.128 at lr 5e-4
        has every AdamW step rounded straight back, permanently, since round-to-nearest
        keeps no remainder.

        `self.gate[-1].bias` is initialised at log(expm1(gamma_init)) = -4.6 for the
        configured gamma_init=0.01. That is 36x the freeze line, so gamma's output bias
        could never move at all, and gate[0]'s weights are drawn from
        U(-1/sqrt(8), 1/sqrt(8)) = U(-0.354, 0.354), so most of those froze as well.

        The measured consequence: gamma stops dead at epoch 4 in EVERY arm and holds to
        four decimals for the next seven epochs while the graph channel's own nDCG still
        moves by 0.017. Whatever gamma reached in those four epochs is the run's final
        answer, because fused = z(s_op) + gamma*relu(z(graph)) and gamma is the only term
        deciding how loud the graph is. matsci: gamma froze at 0.0279 -> 0.5207,
        0.0359 -> 0.5340, 0.0288 -> 0.5218. The score is a function of the freeze point.

        This makes the lam=0.9 result unsafe to attribute to lam: graph-channel nDCG and
        informativeness AUC are the same in all three arms, so what lam changed was where
        gamma happened to stop, not the quality of the gating. CQIG_ALLOW_LOWP=1
        reproduces the old frozen behaviour for both this and the CQIG scalars.

        EVERY FUSION-HEAD PARAMETER IS PINNED, not just the gate. The mixture arm's
        router carries the same -1.39 output bias and its tau_g would drift past the
        freeze line within an epoch, so pinning the gate alone would reintroduce exactly
        this bug under a new name. These are a few dozen scalars in total; float32 for
        all of them costs nothing measurable.
        """
        heads = [("gate", self.gate)]
        # THE CCMP HEAD SITS ON THE SAME CLIFF AS GAMMA. The resp_head trunk initialises
        # at +-0.125 against a 256*lr = 0.128 freeze line, so roughly half its weights are
        # one small drift away from never updating again -- the exact failure that decided
        # every CQIG arm. Empty in the control, where these attributes are None.
        _rm = self.base.entity_model
        # ROUTE=attn's head is on the same cliff, and `_route_attention` runs it with
        # autocast off on float32 inputs, so leaving it in bfloat16 is not a precision
        # loss but a crash: "mat1 and mat2 must have the same dtype" on the first step
        # (tomato attn smoke, 2026-09-07). Empty unless ROUTE=attn.
        for _t in ("resp_proj", "resp_emb", "resp_head",
                   "attn_node", "attn_rel", "attn_query", "attn_emb", "attn_out"):
            if getattr(_rm, _t, None) is not None:
                heads.append((_t, getattr(_rm, _t)))
        if getattr(self, "fusion_form", "additive") == "mixture":
            heads.append(("router", self.router))
            if isinstance(getattr(self, "tau_g_hat", None), nn.Parameter):
                # `self` with recurse=False is tau_g_hat AND the operator scalars
                # w_delta/beta_delta. Pinning those too is deliberate: w_delta starts at
                # 0 but drifts, and W_INIT is 1.05, so they sit on the same cliff. Only
                # reached under form=mixture, so the additive arm is untouched.
                heads.append(("fusion", self))
        saved = {f"{tag}.{n}": p.detach().clone().float()
                 for tag, mod in heads
                 for n, p in (mod.named_parameters(recurse=False)
                              if mod is self else mod.named_parameters())}
        out = super()._apply(*args, **kwargs)
        if os.environ.get("CQIG_ALLOW_LOWP", "0") == "1":
            return out
        pinned = []
        for tag, mod in heads:
            it = (mod.named_parameters(recurse=False) if mod is self
                  else mod.named_parameters())
            for n, p in it:
                k = f"{tag}.{n}"
                if k in saved and p.is_floating_point() and p.dtype != torch.float32:
                    was = p.dtype
                    p.data = saved[k].to(device=p.device)
                    pinned.append((k, was))
        if pinned and not self._gamma_lowp_note:
            self._gamma_lowp_note = True
            print(f"[fusion] fusion head pinned to float32 against a {pinned[0][1]} model "
                  f"cast: {', '.join(n for n, _ in pinned)}. The output bias starts at "
                  f"{float(self.gate[-1].bias.flatten()[0]):.3f}, far above the "
                  f"|w| > 256*lr freeze line, so in {pinned[0][1]} gamma stopped moving "
                  f"at epoch ~4 and the run's score was fixed by whatever it reached.")
        return out

    def _router_phi(self, sz, gz, cov):
        """The per-query router input, shared by the additive gate and the mixture router.

        ONE DEFINITION, TWO CONSUMERS. These features were written for the mixture and
        lived inside it, which is how the additive arm ended up accepting FUSION_ROUTER
        and ignoring it. Both forms now call this, so "router=full" means the same five
        numbers whichever fusion is running and the two forms stay comparable.

        `cov` (the semantic top-5 mean) is feature 1 under `full` and the whole vector
        under `cov`, so the ablation is a strict narrowing rather than a different input.

        Divisors are NOMINAL, not tuned: a top-5 mean of a z-score over a corpus this size
        runs about 3-6 and a top1-to-top5 margin about 1-3, so these put all five features
        on the same O(1) footing and stop the [0,1] overlap feature from starting with a
        hundredth of the gradient of the others. They are constants, not parameters, so
        nothing here can be fitted to the test set.
        """
        if self.router_feats == "cov":
            return cov
        k = min(5, sz.shape[1])
        st, gt = sz.topk(k, dim=-1).values, gz.topk(k, dim=-1).values
        s_cov, g_cov = st.mean(-1, keepdim=True), gt.mean(-1, keepdim=True)
        ko = min(10, sz.shape[1])
        si = sz.topk(ko, dim=-1).indices                        # [B, ko]
        gi = gz.topk(ko, dim=-1).indices
        # agreement: what fraction of the semantic top-10 the graph also ranks top-10.
        # Low overlap with a confident graph is the only configuration in which the
        # graph has something to say that the semantic channel has not already said.
        ov = (si.unsqueeze(2) == gi.unsqueeze(1)).any(-1).float().mean(-1, keepdim=True)
        return torch.cat([s_cov / 5.0,                          # semantic confidence
                          (st[:, :1] - s_cov) / 2.0,            # semantic peakedness
                          g_cov / 5.0,                          # graph confidence
                          (gt[:, :1] - g_cov) / 2.0,            # graph peakedness
                          ov], dim=-1)                          # channel agreement

    def _fuse_mixture(self, sz, gz, cov):
        """log( (1-a_q) p_s + a_q p_g ) from the two standardised channels.

        sz, gz : [B, n_doc] z-scored semantic and graph document scores (float32).
        cov    : [B, 1] the additive arm's router feature, reused as feature 1.
        returns (a_q [B], fused [B, n_doc]).
        """
        # ROUTER FEATURES. The additive gate saw one number, the semantic top-5 mean, so
        # it could tell whether the SEMANTIC scorer was confident but had no way to know
        # whether the graph was worth listening to on this query. Under a mixture the
        # flat-graph case is already handled exactly by the algebra, so these features are
        # not load-bearing for safety -- they are what lets a_q open further on the
        # queries where the graph is peaked AND disagrees, which is where a promote-only
        # channel can actually change the answer. FUSION_ROUTER=cov drops back to the one
        # feature if the richer input ever needs to be ablated out.
        #
        # Divisors are NOMINAL, not tuned: a top-5 mean of a z-score over a corpus this
        # size runs about 3-6 and a top1-to-top5 margin about 1-3, so these put all five
        # features on the same O(1) footing and stop the [0,1] overlap feature from
        # starting with a hundredth of the gradient of the others. They are constants, not
        # parameters, so nothing here can be fitted to the test set.
        phi = self._router_phi(sz, gz, cov)
        # AUTOCAST OFF, same reason as the additive gate: under AMP an nn.Linear emits
        # bfloat16 whatever its parameter dtype, and the output bias sits at -1.39 where
        # a bfloat16 ULP is 0.005, which is noise of the same order as the quantity being
        # learned. Pinning the weights in `_apply` fixes the master copy, not this.
        with torch.autocast(device_type=sz.device.type, enabled=False):
            a = self.a_max * torch.sigmoid(self.router(phi.float())).squeeze(-1).float()
        # FUSION_AFIX=<float> REPLACES THE ROUTER WITH A CONSTANT. The first mixture run
        # (matsci, a_max=0.5) is why this exists: a_q climbed to 0.371 and spanned only
        # [0.309, 0.437] over 331 queries, i.e. the router did not route, it just opened.
        # It opened because it is trained on the TRAINING loss, where the GNN fits its own
        # queries far better than the 0.27 nDCG@5 it scores at test, so it calibrated to a
        # graph quality that does not exist at inference. Fused came out at 0.4265 against
        # a 0.5178 semantic channel: -0.091, exactly the predicted damage at that weight.
        # A constant a removes the train/test mismatch and makes the arm a one-parameter
        # sweep, which is the only honest way to find out whether ANY a > 0 helps here.
        if os.environ.get("FUSION_AFIX", "") != "":
            a = torch.full_like(a, float(os.environ["FUSION_AFIX"]))
        # DELIBERATELY OUTSIDE THE BRANCH ABOVE. tau_g used to be assigned inside it, so
        # the router path -- i.e. every mixture arm run so far -- reached the division
        # below with the name unbound. It has not fired yet only because those runs
        # predate the FUSION_AFIX block; the next router run would raise UnboundLocalError.
        tau_g = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(
            self.tau_g_hat.float())
        # tau_s is 1 and not learned, so log p_s = sz - logsumexp(sz) exactly (see the
        # note in __init__). The per-query logsumexp is a constant within a query, and
        # both the ranking and the cross-entropy loss are invariant to it, which is why
        # this arm starts from the same objective the additive arm trains under.
        lg = gz / tau_g.clamp(min=1e-3)
        log_ps = sz - torch.logsumexp(sz, dim=-1, keepdim=True)
        log_pg = lg - torch.logsumexp(lg, dim=-1, keepdim=True)
        # FUSION_TOPK=<int> CONFINES THE GRAPH TO THE SEMANTIC HEAD. This is the one
        # structural difference between the two forms, and it is why the additive arm wins
        # despite its arithmetic ceiling. gamma*relu(z(g)) can move a document a few places
        # WITHIN the semantic head and can essentially never lift one out of the deep tail;
        # it is a safe local reordering. The mixture is the opposite: every document with
        # a*p_g > p_s is scored log a + log p_g, i.e. purely by graph order, so they arrive
        # as a contiguous block whose depth is set by a. nDCG@5 has five slots, and the
        # graph's own top-5 is worth 0.245 against the semantic channel's 0.517, so once
        # that block reaches rank 5 the trade is losing by construction. Masking p_g outside
        # the semantic top-K keeps the calibrated reweighting (which the additive form can
        # only approximate, capped at 0.36 z-units) and drops the flooding.
        #
        # Out-of-set documents get logaddexp(l1a + log_ps, -inf) = l1a + log_ps exactly:
        # the semantic order shifted by one per-query constant, so their relative order is
        # untouched and in-set documents can only rise past them, never the reverse. The
        # renormalise keeps a's meaning ("fraction of the mixture's mass taken from the
        # graph") rather than silently shrinking it by the mass that was masked away.
        _k = int(os.environ.get("FUSION_TOPK", "0"))
        if 0 < _k < sz.shape[1]:
            keep = torch.zeros_like(sz, dtype=torch.bool)
            keep.scatter_(1, sz.topk(_k, dim=-1).indices, True)
            log_pg = log_pg.masked_fill(~keep, float("-inf"))
            log_pg = log_pg - torch.logsumexp(log_pg, dim=-1, keepdim=True)
        # clamp_min on a, and log1p(-a) rather than log(1-a), so the two mixing logs stay
        # finite if the router saturates at either end. a_max <= 1 keeps 1-a positive.
        la = torch.log(a.clamp(min=1e-8))[:, None]
        l1a = torch.log1p(-a.clamp(max=1.0 - 1e-6))[:, None]
        fused = torch.logaddexp(l1a + log_ps, la + log_pg)
        # ONE FLOAT32 CAVEAT, measured. d/dx log((1-a)e^x + a e^k) > 0, so a constant
        # (uninformative) p_g preserves the semantic order EXACTLY in real arithmetic.
        # In float32 it preserves it only until the a*p_g floor swamps the semantic term
        # and two documents COLLIDE onto the same value -- a tie, not a reorder. Measured
        # on a 15,588-document corpus at a_q=0.1: top-300 bit-identical to the semantic
        # order, first collision at rank ~3,100 between documents whose semantic z-scores
        # differ by 4e-7, ~40 tied of 15,588. Far below anything reported (@5, @10, @100),
        # so the invariance holds where it is read. It would NOT hold at recall@5000, and
        # a metric that deep would need the tie broken by the semantic score.
        with torch.no_grad():
            self._fusion_note = (f"mix a_q {a.mean().item():.4f} "
                                 f"[{a.min().item():.4f},{a.max().item():.4f}] "
                                 f"/{self.a_max:g}  tau_g {tau_g.item():.4f}")
        return a, fused

    @staticmethod
    def _z(x):
        return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)

    def _operator(self, ids, n_doc, device):
        """Recompute S_op live from cached ingredients with the current (trainable) w, beta."""
        order = [self._row[str(x.item() if hasattr(x, "item") else x)] for x in ids]
        tag = order[0][0]
        # One graph is resident per batch, so one table serves the whole batch. A
        # mixed batch means the loader changed, and index_select below would silently
        # read rows of the wrong corpus rather than fail.
        assert all(t == tag for t, _ in order), (
            f"batch mixes operator tables {sorted({t for t, _ in order})}")
        idx = torch.tensor([r for _, r in order], dtype=torch.long)

        dense = self._dense[tag].index_select(0, idx).to(device, torch.float32)
        S = self._S[tag].index_select(0, idx).to(device, torch.float32)
        M = self._M[tag].index_select(0, idx).to(device, torch.float32)
        totS = self._totS[tag].to(device)                       # [n_doc]
        assert dense.shape[1] == n_doc, f"operator cols {dense.shape[1]} != {n_doc} doc nodes (alignment)"

        w = self.w_init + self.op_lr_scale * self.w_delta       # [3]
        beta = (BETA_INIT + self.op_lr_scale * self.beta_delta).clamp(min=0.05)
        dem = (totS.unsqueeze(0) - S).clamp(min=1e-6)           # [B, n_doc] leave-one-out popularity
        degb = dem.pow(beta)
        return w[0] * self._z(dense) + w[1] * self._z(S / degb) + w[2] * self._z(M / degb)

    # ---------------------------------------------------------------- learned scorer
    @staticmethod
    def _load_semantic_module():
        """Import semantic_scorer.py from the repo rather than vendoring a copy of it.

        The scorer, its normalisation and its popularity object are one design. A second
        copy living here would drift from the one that produced the checkpoint, and that
        drift would surface as a quietly different score rather than an import error.
        """
        import importlib.util
        import sys
        root = os.environ.get("SCIGRAPHIR_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
        path = f"{root}/experiments/eval/semantic_scorer.py"
        assert os.path.exists(path), (
            f"semantic scorer source not found at {path}; set SCIGRAPHIR_ROOT to the repo root")
        for p in (root, f"{root}/retriever"):
            if p not in sys.path:
                sys.path.insert(0, p)
        spec = importlib.util.spec_from_file_location("cargo_semantic_scorer", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _init_semantic(self, trainable):
        sem = self._load_semantic_module()
        self._sem = sem
        self._sem_tab, self._sem_row, self._sem_last_tag = {}, {}, None
        seen: dict[str, str] = {}
        for split, ev in (("train", "SEMANTIC_COMPONENTS"), ("test", "SEMANTIC_COMPONENTS_TEST")):
            raw = os.environ.get(ev)
            if not raw:
                continue
            for j, p in enumerate([x.strip() for x in raw.split(",") if x.strip()]):
                ap = os.path.abspath(p)
                if ap in seen:
                    print(f"[fusion] {ev}[{j}] is the same file as '{seen[ap]}', reusing it")
                    continue
                tag = split if raw.count(",") == 0 else f"{split}#{j}"
                seen[ap] = tag
                d = np.load(p, allow_pickle=True)
                Q, jmax, D = (int(x) for x in d["h_shape"])
                h_path = str(d["h_path"])
                assert os.path.exists(h_path), (
                    f"{p} references the per-answer matrix at {h_path}, which is missing. "
                    "It lives in the semantic scorer's own cache, which does not survive a "
                    "runtime reset; rerun section 5d or the precompute with --force.")
                col = np.asarray(d["col"], dtype=np.int64)
                nb = max(float(np.asarray(d["mask"]).sum()), 1.0)
                self._sem_tab[tag] = {
                    # memmap: only the batch's rows and this graph's columns are ever read
                    "H": np.memmap(h_path, np.float16, "r", shape=(Q, jmax, D)),
                    "col": col,
                    "dense": torch.from_numpy(np.asarray(d["dense"], dtype=np.float16)),
                    "mask": torch.from_numpy(np.asarray(d["mask"]).astype(np.float32)),
                    "doc_emb": torch.from_numpy(np.asarray(d["doc_emb"], dtype=np.float32)),
                    # the anchor target: total_S / (number of answers) IS the bank mean
                    "pop_target": torch.from_numpy(
                        np.asarray(d["total_S"], dtype=np.float32) / nb),
                    "doc_emb_gpu": None, "pop_target_gpu": None,
                }
                dup = [str(q) for q in d["query_ids"] if str(q) in self._sem_row]
                assert not dup, (f"{p}: {len(dup)} query ids already claimed by another "
                                 f"semantic components table (e.g. {dup[:3]})")
                for i, q in enumerate(d["query_ids"]):
                    self._sem_row[str(q)] = (tag, i)
                print(f"[fusion] loaded {ev}[{j}] as '{tag}': H {(Q, jmax, D)} -> {len(col)} "
                      f"doc nodes ({Q} queries) {os.path.basename(p)}")
        assert self._sem_row, ("semantic='mlp' needs SEMANTIC_COMPONENTS / "
                               "SEMANTIC_COMPONENTS_TEST from precompute_semantic_components.py")

        # --- the trained scorer and its predictor, as ONE warm start ---
        # Loading the scorer from a joint run without its predictor would pair a trained
        # readout with an untrained popularity, which is a model that never existed.
        ck, pn = os.environ.get("SEMANTIC_CKPT", ""), os.environ.get("SEMANTIC_POPNET", "")
        assert ck and os.path.exists(ck), (
            "semantic='mlp' needs SEMANTIC_CKPT: a params_semantic_mlp_*.json from section 5d")
        assert pn and os.path.exists(pn), (
            "semantic='mlp' needs SEMANTIC_POPNET: the popnet_semantic_mlp_*.pt written "
            "alongside it by a joint run (--mlp_pop_joint 1)")
        st = json.load(open(ck))
        scorer = sem.SortedMLPScorer("cpu", int(st["jmax"]), hidden=len(st["net"]["0.weight"]))
        scorer.load(st)
        psd = torch.load(pn, map_location="cpu")
        p_hidden, p_dim = psd["net.0.weight"].shape
        popnet = sem.MatchabilityPredictor(int(p_dim), int(p_hidden))
        popnet.load_state_dict(psd)

        # Registered as attributes so the optimiser sees them and .to(device) moves them.
        # These are the SAME objects the scorer holds, so updates propagate both ways.
        self.sem_net, self.sem_logbeta, self.sem_popnet = scorer.net, scorer.logbeta, popnet
        self._sem_scorer = scorer
        if not trainable:
            for prm in self.sem_net.parameters():
                prm.requires_grad_(False)
            self.sem_logbeta.requires_grad_(False)
            for prm in self.sem_popnet.parameters():
                prm.requires_grad_(False)
        print(f"[fusion] semantic='mlp' warm start: jmax={st['jmax']} beta={st['beta']:.4f} "
              f"popnet {p_dim}->{p_hidden}->1, trainable={trainable}")

    def _semantic(self, ids, n_doc, device):
        """Score with the learned sorted-MLP, recomputed live so it keeps training."""
        order = [self._sem_row[str(x.item() if hasattr(x, "item") else x)] for x in ids]
        tag = order[0][0]
        assert all(t == tag for t, _ in order), (
            f"batch mixes semantic tables {sorted({t for t, _ in order})}")
        tab, rows = self._sem_tab[tag], [r for _, r in order]
        self._sem_last_tag = tag

        # Slice rows from disk, then permute to nodes.csv document order. The resident
        # cost is B x Jmax x n_doc, not the whole 0.1-1.8 GB matrix.
        H = torch.from_numpy(
            np.asarray(tab["H"][rows])[:, :, tab["col"]].astype(np.float32)).to(device)
        idx = torch.tensor(rows, dtype=torch.long)
        dense = tab["dense"].index_select(0, idx).to(device, torch.float32)
        mask = tab["mask"].index_select(0, idx).to(device, torch.float32)
        assert H.shape[-1] == n_doc, (
            f"semantic cols {H.shape[-1]} != {n_doc} doc nodes (alignment)")

        if tab["doc_emb_gpu"] is None or tab["doc_emb_gpu"].device != device:
            tab["doc_emb_gpu"] = tab["doc_emb"].to(device, torch.float32)
            tab["pop_target_gpu"] = tab["pop_target"].to(device, torch.float32)
        pop = self._sem.Pop("joint", predictor=self.sem_popnet, doc_emb=tab["doc_emb_gpu"])
        return self._sem_scorer(H, mask, dense, pop)

    # ---------------------------------------------------------------- CQIG calibration
    def _cqig_run_refs(self, graph, batches, dev, count=False):
        for b in batches:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            self.base(graph, b)
            if count:
                # COUNT QUERIES, NOT BATCHES. `len(b["id"])` counted whatever the loader
                # happened to pack, so a bank of 16 items meant 16 queries at train batch
                # size 1 and 123 at eval batch size 8, and the two calibrations were not
                # comparable. Re-linked reference batches carry no "id" at all.
                self.cqig.note_reference_batch(int(b["question_embeddings"].shape[0]))

    def cqig_calibrate(self, graph, graph_key, ref_batches=None, device=None, rounds=1):
        """Calibrate THIS graph from the fixed reference problems, in two passes per round.

        Pass "mean" accumulates mu. Pass "dev" measures ||h - mu||^2 directly, which gives
        both the variance -- exactly zero for a node whose state does not move, unlike the
        one-pass identity -- and the deviations tau_ref is the median of.

        `rounds` > 1 repeats both passes with the previous round's gates active. That
        matters only when more than one layer is gated: a layer's own gate cannot change
        its own input, so a single gated layer is already exact at rounds=1.

        Unlabelled and target-query-free: this is the indexing-time pass that gives an
        unseen graph its own reference statistics. Statistics are per graph because mu has
        one row per node, so the training graph's mu is not even shape-compatible with a
        different corpus. `ref_batches` defaults to the bank stored on the gate, which is
        what makes a reloaded checkpoint able to calibrate a corpus it has never seen.
        """
        assert self.cqig is not None, "cqig is not enabled on this model"
        batches = ref_batches if ref_batches is not None else self.cqig.ref_bank
        assert batches, ("no reference bank: the trainer sets one during training and it "
                         "is carried in the checkpoint; cannot calibrate without it")
        dev = device or next(self.parameters()).device
        was_training = self.training
        self.eval()
        report = {}
        with torch.no_grad():
            for r in range(max(1, int(rounds))):
                gated = r > 0
                for phase in ("mean", "dev"):
                    self.cqig.start_calibration(graph_key, phase=phase,
                                                apply_existing_gates=gated)
                    self._cqig_run_refs(graph, batches, dev, count=True)
                    if phase == "mean":
                        self.cqig.finish_mean()
                    else:
                        report = self.cqig.finish_calibration()
                for row in report.values():
                    row["round"] = r + 1
        if was_training:
            self.train()
        self.cqig.use_graph(graph_key)
        self.cqig.mode = "gate"
        return report

    def cqig_use_graph(self, graph_key):
        """Point the gate at an already-calibrated graph, e.g. back to train after eval."""
        assert self.cqig is not None and self.cqig.calibrated(graph_key), (
            f"graph {graph_key!r} has not been calibrated "
            f"(known: {self.cqig.known_graphs() if self.cqig else []})")
        self.cqig.use_graph(graph_key)
        self.cqig.mode = "gate"

    def semantic_aux_loss(self):
        """Log-space anchor holding p_hat near measured popularity. Mirrors Pop.aux_loss.

        Without it the fusion's ranking gradient is free to repurpose the predictor as extra
        scorer capacity, exactly as it would in the standalone run. Uses the tag of the last
        forward, which is the corpus the current batch came from.
        """
        tab = self._sem_tab.get(self._sem_last_tag) if self.semantic == "mlp" else None
        if tab is None or tab["doc_emb_gpu"] is None:
            return 0.0
        eps = 1e-6
        p = self.sem_popnet(tab["doc_emb_gpu"])
        return ((torch.log(eps + p) - torch.log(eps + tab["pop_target_gpu"])) ** 2).mean()

    def forward(self, graph, batch, entities_weight=None):
        # The frontier A^(l) is expanded inside bellmanford, which sees the edge index but
        # not the batch. Handed over here rather than threaded through GraphReasoner's
        # signature, which every other reasoner shares.
        if getattr(self.base.entity_model, "resp_proj", None) is not None:
            self.base.entity_model._ccmp_seeds = batch.get("start_nodes_mask")
        g = self.base(graph, batch, entities_weight)            # [B, N] per-node scores
        doc = graph.nodes_by_type["document"].to(g.device)      # nodes.csv document order
        gdoc = g.index_select(1, doc).float()                   # [B, n_doc] graph-alone doc scores

        # THE ONLY LINE THAT DIFFERS BETWEEN THE TWO ARMS. Everything below -- gate,
        # relu floor, fusion arithmetic, hard-negative mining -- is shared, so a run
        # pair isolates the semantic scorer with the graph half held fixed.
        s_op = (self._operator(batch["id"], doc.numel(), g.device)
                if self.semantic == "operator"
                else self._semantic(batch["id"], doc.numel(), g.device))
        # IS A HIGH INFORMATIVENESS ACTUALLY ON THE GOLD PAPERS? Measurement only, and only
        # while the trainer has recording switched on (around evaluate()). Nothing here
        # feeds back into the statistics, so it is not a transductive path -- but it is the
        # test that decides whether the gate's premise holds at all, so it needs the labels
        # and the hard negatives, which exist only here.
        if self.cqig is not None and self.cqig.recording():
            tgt = batch["target_nodes_mask"] if "target_nodes_mask" in batch else None
            self.cqig.note_alignment(tgt, doc, s_op.detach())
        # THE GATE READS THE STANDARDISED SCORE, like the fusion does. On the raw score
        # the semantic scorer could add a constant or scale everything up and change
        # gamma_q while leaving z(s_op) and its own ranking untouched, so the gate would
        # be keyed to a quantity with no semantic content. That is live for the learned
        # arm in particular: an MLP's output scale is free, unlike the operator's
        # warm-started w. After z(), a peaked top-5 genuinely means a confident scorer.
        sz = self._z(s_op).float()
        cov = sz.topk(min(5, sz.shape[1]), dim=-1).values.mean(-1, keepdim=True)
        # .float() on both operands, explicitly. Under AMP autocast the gate and the
        # learned scorer are nn.Linear and emit bfloat16 regardless of input dtype, and
        # a bfloat16 sum would round away a graph contribution of order gamma=0.01 for
        # the same reason the bfloat16 output write did. This currently lands in float32
        # by accident, via promotion from `gz`; relying on that is one config change away
        # from silently losing the effect.
        # AUTOCAST OFF FOR THE GATE. Pinning the weights to float32 (see `_apply`) is not
        # sufficient on its own: under AMP an nn.Linear is autocast to bfloat16 whatever
        # its parameter dtype, so the pre-activation would be quantised at an ULP of
        # 4.6 * 2^-8 = 0.018 around the -4.6 output bias, roughly 0.6% of gamma itself.
        # The master weights would still update in float32, but gamma would carry
        # avoidable noise. This gate is a 1->8->1 MLP, so float32 here costs nothing.
        gz = self._z(gdoc).float()                              # what the ranking uses
        if self.fusion_form == "mixture":
            gamma, fused = self._fuse_mixture(sz, gz, cov)
        else:
            # phi_q, not cov. Under FUSION_ROUTER=cov this is exactly the old one-feature
            # input; under the default `full` the gate additionally sees graph confidence,
            # graph peakedness and channel agreement, so it can condition gamma on whether
            # the GRAPH is worth listening to rather than only on semantic confidence.
            # Needs `gz`, which is why this sits after the gz line above and not with cov.
            with torch.autocast(device_type=cov.device.type, enabled=False):
                phi = self._router_phi(sz, gz, cov)
                gamma = F.softplus(self.gate(phi.float())).squeeze(-1).float()  # [B] per-query gate
            # FUSION_GAMMAFIX=<float>, the additive twin of FUSION_AFIX, and the only way
            # to find out whether the 0.5340 arm is a result or a coincidence. That number
            # came from a run whose gamma was bf16-FROZEN at 0.0359; the fp32-learnable
            # rerun froze at 0.0288 and scored 0.5218, and 0.012 nDCG@5 on n=331 is inside
            # what two training runs differ by anyway. Pinning gamma at 0.0359 makes the
            # comparison paired instead of accidental. Everything the mixture is being
            # asked to beat rests on that one number reproducing.
            if os.environ.get("FUSION_GAMMAFIX", "") != "":
                gamma = torch.full_like(gamma, float(os.environ["FUSION_GAMMAFIX"]))
            fused = sz + gamma[:, None] * torch.relu(gz)
            self._last_gamma = gamma.detach()   # exposed for interpret() dumps

        # FLOAT32 OUT. `g` is bfloat16, whose epsilon near 1.0 is 2^-8 = 0.0039 -- and
        # the graph's contribution is gamma_q * relu(z(graph)), which at the gate's 0.01
        # init is the same order. Writing the fused score back into a bfloat16 tensor
        # therefore rounds away most of the graph bonus and creates ranking ties, i.e.
        # it destroys precisely the signal being measured. clone() first so a float32
        # `g` is never mutated in place.
        out = g.clone().float()
        out[:, doc] = fused
        self._raw_doc = gdoc          # RAW graph scores: mining and diagnostics (both top-k, scale-free)
        # STANDARDISED graph scores: what the auxiliary loss must train, because it is
        # what the fused ranking consumes. z() is scale-invariant, so a loss on the raw
        # score has a free direction -- multiply every score by a constant and the loss
        # falls while the fused ranking does not move at all. Kept pre-relu on purpose:
        # relu would zero the gradient for exactly the golds sitting below the graph
        # mean, which are the ones that most need lifting.
        #
        # DELIBERATELY z(g) UNDER BOTH FORMS, not lg = z(g)/tau_g. Cross-entropy is
        # shift-invariant, so handing it lg would differ from gz only by the 1/tau_g
        # factor, i.e. it would couple the auxiliary loss's temperature to a fusion
        # parameter and give tau_g a gradient path that has nothing to do with fusion
        # quality. Keeping gz makes the graph-alone objective IDENTICAL in the additive
        # and mixture arms, so a run pair isolates the fusion form and nothing else.
        self._raw_doc_z = gz
        self._doc_ids = doc
        # The SELECTED scorer, so switching `semantic` also switches which documents the
        # contrastive loss makes the graph beat. One variable, every consumer.
        self._s_op = s_op.detach()
        # The gate, cached for diagnostics. Whether gamma_q ever leaves its 0.01 init is
        # the first thing to know about a fusion run: if it does not, the fused score IS
        # the semantic score and no loss change can matter.
        #
        # UNDER form=mixture THIS IS a_q, THE MIXING WEIGHT IN [0, a_max], NOT gamma. The
        # eval hook prints it in the same column, so a mixture run's "gamma mean" is on a
        # different scale from every additive run's and the two must not be read off the
        # same axis. a_q ~ 0.1 is its initialisation, not a stall.
        self._gamma = gamma.detach()
        return out
