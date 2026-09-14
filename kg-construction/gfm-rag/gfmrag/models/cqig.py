"""
cqig.py — Cross-Query Informativeness Gating.

A node earns the right to send a message by responding to THIS query differently from how
the nodes of this graph TYPICALLY respond to unrelated scientific problems. A node that
answers every query the same way is broadcasting background graph activity, and under sum
aggregation that broadcast reaches every one of its neighbours identically, so it shifts a
whole domain rather than discriminating within it.

    mu_v^l    = (1/M) sum_m h_{a_m v}^l                       reference mean,     PER NODE
    V_v^l     = (1/(M-1)) sum_m || h_{a_m v}^l - mu_v ||^2    reference variance, PER NODE
    s^l       = median_{v : V_v^l > delta} V_v^l              reference scale,    PER LAYER
    I_qv^l    = || h_qv^l - mu_v^l ||^2 / (s^l + eps)
    tau_l(G)  = tau_ref_l(G) + dtau_l                         graph term + LEARNED term
    g_qv^l    = 1 - lam_l [ 1 - sigmoid( alpha_l ( log(1+I) - tau_l(G) ) ) ]
    m~        = g_qv^l * m_{v->u}^l                           op="gate", the default

WHY THE DENOMINATOR IS A LAYER-WIDE SCALE AND NOT THE NODE'S OWN VARIANCE. Two dead ends
came first and both are kept as ablations, because the difference between them IS the
hypothesis:

  norm="energy"  divide by E_v = mean ||h||^2. Dead on arrival under
                 `use_ent_emb: early-late-fusion`: h_qv is dominated by the node's static
                 entity embedding and the query-dependent part is ~1% of it in norm, so
                 I ~ 1e-4 for EVERY node and the sigmoid never leaves its midpoint.
  norm="node"    divide by the node's OWN variance V_v. Numerically alive (I ~ 1) but it
                 cancels the very comparison the method is built on: a generic node that
                 always moves by 0.01 and a discriminative node that always moves by 1.0
                 both come out at 0.01/0.01 = 1.0/1.0 = 1. Every node looks equally
                 informative and the gate is a near-constant rescaling.
  norm="layer"   divide by ONE robust scale for the whole layer. A weak generic response is
                 now small, a strong query-specific response is large, and a node that
                 never responds stays at zero. This is the only form in which I is
                 comparable BETWEEN nodes, which is what "this node is informative" has to
                 mean. Default.

WHY CALIBRATION IS TWO PASSES AND NOT ONE. The one-pass identity E||h-mu||^2 = E||h||^2 -
||mu||^2 is a catastrophic cancellation here. The deviation is ~1e-4 of the energy under
early-late fusion, float32 carries ~1e-7 relative, and the accumulation itself rounds. At
the real proportions (D=1024, deviation 1% of the state norm) that leaves a node which
never responds with a variance up to 4e-3 of a genuine one -- ragged, node-dependent, and
close enough to a real response to survive any sane cut. The second pass computes
||h - mu||^2 directly: the residue drops to ~1e-10 of the working scale and, more useful
still, becomes UNIFORM across the unreachable part of the graph, so one threshold removes
all of it. It costs one more forward over M references.

WHY THE THRESHOLD IS MEASURED, NOT DERIVED. The same second pass records the deviations
themselves, so tau_ref is the median of log(1+I) over the (node, reference query) pairs
that respond at all. Deriving it from V instead would compare a per-query deviation against
a mean-over-queries deviation, which sit at systematically different points of the same
distribution. Measuring puts the operating point exactly at "a typical reference query's
typical responding node", so dtau is a learned offset in interpretable units. Both the
variance and the recorded deviations are corrected to leave-one-out, since mu is built from
the same M references: V uses the M-1 denominator, and a stored deviation is scaled by
(M/(M-1))^2, which is exactly ||h_m - mu_{-m}||^2. A query the bank has never seen then
lands in the distribution tau_ref was measured on.

WHY I IS DETACHED FROM THE AUTOGRAD GRAPH BY DEFAULT. With a gradient path through I the
GNN can open every gate at once by drifting h away from mu -- mu is only recomputed every
CQIG_RECAL steps, so that direction is free, it raises I for every query equally, and it
destroys the mechanism while lowering the loss. Detached, the gate is a modulation whose
shape (lam, alpha, dtau) is still learned, and h still receives gradient through the
product h*g. Set grad_through_I=True to restore the differentiable version as an ablation.
Detaching also lets I be computed in float32 for free, which is worth having but is not a
cure. h is bfloat16 under autocast and the deviation is a small fraction of it, so most of
each component's significant digits are gone before the gate sees them. Measured against
float64 at D=1024: with the deviation at 1% of the state norm, doing the subtraction and
the sum in float32 gives ~2.6% median error on I against ~5.2% for the all-bfloat16 path
the earlier version used -- a factor of about two across M and across deviation ratios from
0.3% to 3%. The remainder is h's OWN quantisation and cannot be recovered here. What that
error scales with is the deviation ratio, not the arithmetic: at 0.3% of norm even the
float32 path carries ~29%, at 3% it carries ~0.3%. If a layer's states are nearly identical
across queries, I is noise there whatever the dtype, which is what `responding` in the
calibration report is for.

WHY A FORWARD PRE-HOOK AND NOT A FORK OF THE MESSAGE PASSING. The layer computes
`message = input_j * relation_j` for DistMult, which is LINEAR in the source state, and the
boundary condition arrives as a separate argument while the residual is added outside the
layer. Scaling `input` before the call is therefore exactly equivalent to scaling every
message leaving that node, and touches neither the boundary nor the shortcut.

THREE OPERATORS, ONE COEFFICIENT (`op`). The same pre-hook and the same statistic support a
second family, selected by `op`. Write d_qv = h_qv - mu_v for the query-specific residual.
Because a DistMult message is LINEAR in the source state and the relation embedding is a
function of the graph's relation text alone -- `rel_mlp(graph.rel_attr)` is expanded across
the batch and never sees the query -- E_a[ m(h_av, r) ] = m(mu_v, r) holds EXACTLY, so
"propagate the residual and retain a kappa-fraction of the background" is a subtraction on
the input and the hook implements it exactly rather than approximately:

    m(d_qv, r) + kappa m(mu_v, r) = m( h_qv - (1 - kappa) mu_v , r )

    op="gate"          h_qv * g_qv                 scale the whole state, the original
    op="centre"        h_qv - (1 - g_qv) mu_v      subtract background in proportion to
                                                   how UNinformative the node is
    op="centre-fixed"  h_qv - lam_l mu_v           the same subtraction at a constant rate

with kappa_qv = g_qv, so one coefficient covers all of it and lam_l = 0 recovers the ungated
reasoner exactly under every op. The ladder is nested: fixed centring is adaptive centring
with the coefficient frozen, and `op="centre-fixed"` at lam=1 is full centring, residual only.

WHAT ACTUALLY DIFFERS BETWEEN GATING AND CENTRING, given both are driven by the same g:

  * DIRECTION. Gating is a scalar multiple, so the message keeps its direction and only its
    magnitude moves. Centring removes a component, so the direction moves too. At full
    damping a gated node sends nothing while a centred node sends its residual d_qv. Whether
    silence or the residual is better is an empirical question about what a generic node's
    small d_qv contains -- encoder noise, or a weak but real cross-domain cue -- and it is
    the one thing the pair of arms genuinely tests.
  * WHICH NODES ARE TOUCHED. A node no reference query reaches has mu_v = 0 EXACTLY, so
    centring leaves it alone, while gating sees I = 0 and damps it to the floor. When seed
    coverage is partial these are not small print: they are opposite treatments of whatever
    fraction of the graph the references never entered, which `mu_zero_frac` reports.
  * QUERY-CONDITIONING. mu_v does not depend on the query, so `op="centre-fixed"` subtracts a
    query-INDEPENDENT background from every node. It still changes rankings -- a node fed by
    many broadly-active neighbours loses more -- but it is a hub correction, not
    query-conditioned reasoning, and alpha and dtau receive no gradient in that arm. Only
    the adaptive coefficient makes the operator a function of the query.
  * SIZE OF THE EDIT. Under `use_ent_emb: early-late-fusion` mu is dominated by the node's
    static entity embedding, so at a given lam the subtraction can be a far larger
    perturbation than the multiplication. `edit_rel` in the live report measures it directly
    as ||h~ - h|| / ||h||, which is also the number to read against `layer_norm: yes`: the
    norm renormalises much of a magnitude change away and leaves the direction change, so a
    centring arm should be expected to act through the direction.

GATING MORE THAN ONE LAYER. Calibration of layer l assumes the state entering it was
produced by the same computation at inference. With layers upstream of l gated, that holds
only if their gates were also active while l was calibrated. `rounds=1` calibrates every
gated layer with no gate active anywhere, which is exact for a single gated layer -- a
layer's own gate cannot change its own input -- and leaves the upstream mismatch in place
for a multi-layer arm. `rounds=2` repeats both passes with the round-1 gates live, which
removes it. Anything above 2 is a fixed-point iteration and has not been needed.

STATISTICS ARE PER GRAPH AND MUST BE SELECTED BEFORE A FORWARD. mu has one row per node, so
the train graph's mu is not even shape-compatible with the test graph's states. `_stats` is
keyed by a caller-supplied graph key and `use_graph(key)` must be called before any gated
forward. Reading statistics for the wrong graph raises rather than broadcasting.

ZERO-SHOT, AND WHAT THAT ACTUALLY REQUIRES. The reference bank is stored in a
GRAPH-INDEPENDENT form: a frozen question embedding plus the entity-linked start nodes as
NAMES with their attachment weights. `materialise(node2id, num_nodes)` re-links those same
source problems against any target graph, exactly as the indexer does
(`start_mask[node2id[name]] = weight`), and returns both the rebuilt batches and the
coverage of the source seeds in the target vocabulary. Every graph is therefore calibrated
with the SAME M individual reference problems, and nothing is ever read from the target
corpus's own query set. A reloaded checkpoint carries the bank in its state dict, so an
unseen graph needs one unlabelled calibration pass at indexing time and nothing else.

EXACT NAME LOOKUP IS NOT ENOUGH, AND THE FAILURE IS SILENT. Re-linking by string equality
assumes the target graph spells its concepts the way the source graph does. It does not:
on sir4_physics the source seeds resolved 501/501 against the train graph and 25/501
against the TEST graph, same domain, different papers. mu was then estimated from
references that barely entered the graph, and nothing failed loudly -- coverage was a
printed number, not an error. `link="semantic"` fixes the linker rather than the bank:

    exact match if the name exists; otherwise take the K nearest target nodes to the
    seed's FROZEN source embedding, reject any below a source-selected cosine floor,
    and split the seed's attachment weight across the survivors by softmax(cos / T).

The seed's total starting weight is preserved, so a semantically linked reference injects
exactly as much probability mass as an exactly linked one and the two are comparable. This
is still zero-shot: the reference questions are fixed on source data, the encoder is frozen,
the floor is chosen from source-side statistics, the target's labels and queries are never
touched, and nothing is fine-tuned. What changes is only that a reference problem can now
enter a target graph that does not happen to use the source's exact phrasing.

lam_l = 0 recovers the ungated reasoner exactly, so the ablation is one scalar.
"""
import math
import os

import torch
import torch.nn as nn

# A normalised response below this counts as "this node did not respond to this query".
# I is ~1 for a typical responding node by construction, so 1e-6 is six orders below the
# working range. With the deviation computed directly it only ever catches exact zeros.
RESP = 1e-6

# Bound on the transient float32 buffer used to compute ||h - mu||^2 without ever
# subtracting in bfloat16. ~256 MB.
_CHUNK_BYTES = 1 << 28

# Bound on the reference deviations kept for the threshold and the gate report. Whole
# rows are kept, never a flattened subsample, so a per-node scale stays aligned.
_SAMPLE_ELEMS = 8_000_000

# Target nodes per chunk when matching seeds by cosine. A seed set is a few hundred rows
# and a graph is a few hundred thousand, so the full product is ~0.5 GB in float32 and is
# built one slice at a time instead.
_LINK_CHUNK = 65_536


def topk_cosine(q, node_emb, k, chunk=_LINK_CHUNK):
    """Top-k cosine of each row of `q` against every row of `node_emb`.

    Both are L2-normalised here rather than by the caller, so a caller cannot pass one
    normalised and one not and get a silently rescaled similarity. Chunked over target
    nodes: the full [seeds, nodes] product is what makes this expensive, and it is never
    needed all at once.
    """
    qn = q.float()
    qn = qn / qn.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    n = int(node_emb.shape[0])
    k = max(1, min(int(k), n))
    best_c = torch.full((qn.shape[0], k), -2.0, device=qn.device)
    best_i = torch.zeros((qn.shape[0], k), dtype=torch.long, device=qn.device)
    for a in range(0, n, chunk):
        blk = node_emb[a:a + chunk].to(qn.device).float()
        blk = blk / blk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        c = qn @ blk.T                                       # [S, chunk]
        kk = min(k, c.shape[1])
        cv, ci = torch.topk(c, kk, dim=1)
        best_c = torch.cat([best_c, cv], 1)
        best_i = torch.cat([best_i, ci + a], 1)
        best_c, order = torch.topk(best_c, k, dim=1)
        best_i = torch.gather(best_i, 1, order)
    return best_c, best_i


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _describe(t, qs=(0.1, 0.5, 0.9)):
    """min / quantiles / max / mean / std of a tensor, as plain floats.

    min, max, mean and std are exact over the whole tensor; only the quantiles fall back
    to a random subsample, because torch.quantile has a hard size ceiling.
    """
    keys = [f"p{int(round(q * 100))}" for q in qs]
    if t is None or t.numel() == 0:
        return dict({"n": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
                    **{k: 0.0 for k in keys})
    t = t.detach().float().flatten()
    out = {"n": int(t.numel()), "min": float(t.min()), "max": float(t.max()),
           "mean": float(t.mean()),
           "std": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0}
    s = t
    if s.numel() > 1_000_000:
        s = s[torch.randperm(s.numel(), device=s.device)[:1_000_000]]
    qv = torch.quantile(s, torch.tensor(list(qs), device=s.device, dtype=s.dtype))
    for k, v in zip(keys, qv):
        out[k] = float(v)
    return out


def parse_layers(spec, n_layers):
    """Resolve a gated-layer specification to 0-indexed layer positions.

    NUMBERS ARE 1-INDEXED, matching how the layers are talked about: with six layers,
    "6" is the last one and "3-6" is the second half. Accepted forms:

        None / "" / "last"     the final layer only (default, exact at rounds=1)
        "all"                  every layer
        "6" / "3-6" / "3,5,6"  1-indexed layer numbers, ranges inclusive
        [3, 4, 5, 6]           the same, as a list

    Returns 0-indexed positions, which is what the hook and the checkpoint use.
    """
    if spec is None:
        return {n_layers - 1}
    if isinstance(spec, str):
        s = spec.strip().strip("'\"").lower()
        if s in ("", "none", "null", "last"):
            return {n_layers - 1}
        if s == "all":
            return set(range(n_layers))
        nums = set()
        for part in s.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part.lstrip("-"):
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                assert lo <= hi, f"cqig layer range {part!r} runs backwards"
                nums.update(range(lo, hi + 1))
            else:
                nums.add(int(part))
    else:
        nums = {int(x) for x in spec}
    assert nums, f"cqig layer spec {spec!r} selects no layers"
    bad = sorted(x for x in nums if not 1 <= x <= n_layers)
    assert not bad, (f"cqig layer spec {spec!r} names layer(s) {bad}, but the model has "
                     f"{n_layers} layers, numbered 1..{n_layers} (1-indexed)")
    return {x - 1 for x in nums}


class CQIGate(nn.Module):
    """Per-layer informativeness gate, attached to an entity model's conv layers.

    `mode` is WHAT THE HOOK IS DOING RIGHT NOW and `op` is WHICH OPERATOR the arm runs.
    They are independent and the names are close, so they are spelled out here:

    mode: "off"        pass through untouched; the model is bit-identical to the ungated one
          "calibrate"  accumulate reference statistics for the active graph (two phases)
          "gate"       compute I against the active graph's statistics and edit the state

    op:   "gate"          h * g                 the original, and the default
          "centre"        h - (1 - g) * mu      adaptive centring
          "centre-fixed"  h - lam * mu          query-independent centring

    mu:   "ref"           mu from the reference bank; the method
          "zero"          mu forced to 0; THE ABLATION, see mu_zero below
    """

    OPS = ("gate", "centre", "centre-fixed")
    MUS = ("ref", "zero")

    def __init__(self, layers, lam_init=0.1, alpha_init=1.0, rho=1e-3, eps=1e-12,
                 gate_layers=None, norm="layer", grad_through_I=False, op="gate",
                 mu="ref"):
        super().__init__()
        self.n_layers = len(layers)
        self.gate_layers = parse_layers(gate_layers, self.n_layers)
        self.rho, self.eps = rho, eps
        assert norm in ("layer", "node", "energy"), (
            f"unknown cqig norm {norm!r}; one of 'layer' (default), 'node', 'energy'")
        self.norm = norm
        self.grad_through_I = bool(grad_through_I)
        op = str(op or "gate").strip().lower()
        assert op in self.OPS, (
            f"unknown cqig op {op!r}; one of {', '.join(map(repr, self.OPS))}. This is the "
            f"OPERATOR (what is done to the state), not `mode` (what the hook is doing).")
        self.op = op
        self.centring = op != "gate"
        mu = str(mu or "ref").strip().lower()
        assert mu in self.MUS, f"unknown cqig mu {mu!r}; one of {', '.join(map(repr, self.MUS))}"
        self.mu_source = mu
        self.mu_zero = mu == "zero"

        # lam in [0,1] via sigmoid, alpha > 0 via softplus.
        # lam_init is deliberately NOT 0: at exactly 0 the sigmoid derivative is negligible
        # and the gate could never learn to open, which would present as "it did nothing".
        self.lam_raw = nn.Parameter(torch.full((self.n_layers,), _logit(lam_init)))
        self.alpha_raw = nn.Parameter(torch.full((self.n_layers,), _inv_softplus(alpha_init)))
        # LEARNED OFFSET ONLY. The absolute threshold is graph-specific and recomputed at
        # every calibration; a single learned tau was silently overwritten by each recal,
        # so it never actually learned anything.
        self.dtau = nn.Parameter(torch.zeros(self.n_layers))

        # The reference bank travels with the model, in a form that is not tied to any
        # graph, so a reloaded checkpoint can calibrate a corpus it has never seen.
        self.ref_bank = None                  # list[dict]: {"qemb", "seeds", "id"}
        self.ref_meta = {}

        self.mode = "off"
        self._stats: dict = {}                # graph_key -> {layer: {mu, scale, ...}}
        self._active = None                   # graph_key currently selected
        self._acc: dict = {}                  # layer -> running sums for the current phase
        self._mu: dict = {}                   # layer -> mu, between the two phases
        self._energy: dict = {}               # layer -> mean ||h||^2, for norm="energy"
        self._acc_key, self._n_ref, self._m_ref = None, 0, 0
        self._phase = "mean"
        self._calib_gate = False              # apply existing gates while calibrating
        self._live = None                     # layer -> running gate histogram, when on
        self._live_bins = 256
        self._align: dict = {}                # layer -> gold-vs-hard-negative tally
        self._edit: dict = {}                 # layer -> ||h~ - h|| / ||h|| tally
        self.last_I: dict = {}                # layer -> [B, N] I of the last forward
        self._handles = [
            layer.register_forward_pre_hook(self._make_hook(i))
            for i, layer in enumerate(layers)
        ]
        self.last_gate = {}
        self._lowp_note = False

    # ------------------------------------------------------------------ precision
    def _apply(self, *args, **kwargs):
        """Keep lam_raw, alpha_raw and dtau in float32 through `model.to(dtype=...)`.

        THIS IS A CORRECTNESS FIX, NOT AN OPTIMISATION. The trainer casts the whole model
        with `model.to(dtype=torch.bfloat16)` (utils/setup_training.py), which catches these
        three scalars. A bfloat16 value has a 2^-8 relative ULP, so half an ULP is
        |w| * 2^-9 .. |w| * 2^-8, while an AdamW step is at most `lr`. At lr 5e-4 every
        parameter above |w| ~= 0.128 has its update rounded straight back to the old value
        on EVERY step, and because round-to-nearest keeps no remainder it never accumulates.

        lam_raw = logit(0.9) = 2.197 and alpha_raw = inv_softplus(1) = 0.541 are both above
        that line, so neither ever learned. The logs are their own proof: float32 init would
        report lam exactly 0.9000000 and alpha exactly 1.0000000, and what every run to date
        actually reported was lam = 0.9023438 = sigmoid(bf16(2.197)) and alpha = 1.00104 =
        softplus(bf16(0.541)). Those digits ARE the rounding error. dtau starts at 0 where
        bfloat16 resolves finely so it did move, but it stalls hard at |dtau| = 0.25, where
        half an ULP overtakes lr again.

        `.to()`, `.cuda()`, `.float()` and DDP all funnel through `_apply`, so refusing the
        cast here covers every call site and travels with this file alone. Device moves are
        unaffected: the saved copy is restored onto the parameter's NEW device. The values
        kept are the pre-cast float32 ones, so lam_init is exactly logit(0.9) again rather
        than its bfloat16 neighbour.

        Nothing downstream needs a matching change. I is already float32 out of `_sqdist`,
        and both operator branches cast the coefficient back themselves -- `g.to(h.dtype)`
        in `_apply_op` and the per-chunk `.to(h.dtype)` in `_centre`. So the gate is
        computed in float32 and only the coefficient crosses into bfloat16, which is what
        mixed precision was supposed to be doing. Set CQIG_ALLOW_LOWP=1 to reproduce the old
        frozen-scalar runs.
        """
        saved = {n: p.detach().clone().float()
                 for n, p in self.named_parameters(recurse=False)}
        out = super()._apply(*args, **kwargs)
        if os.environ.get("CQIG_ALLOW_LOWP", "0") == "1":
            return out
        pinned = []
        for n, p in self.named_parameters(recurse=False):
            if n in saved and p.is_floating_point() and p.dtype != torch.float32:
                was = p.dtype
                p.data = saved[n].to(device=p.device)
                pinned.append((n, was))
        if pinned and not self._lowp_note:
            self._lowp_note = True
            print(f"[cqig] gate scalars pinned to float32 against a "
                  f"{pinned[0][1]} model cast: {', '.join(n for n, _ in pinned)}. "
                  f"These are learnable; in {pinned[0][1]} an lr-5e-4 step is below half "
                  f"an ULP for lam_raw and alpha_raw, so they would never move.")
        return out

    # ------------------------------------------------------------------ parameters
    def lam(self, i):
        return torch.sigmoid(self.lam_raw[i])

    def alpha(self, i):
        return nn.functional.softplus(self.alpha_raw[i])

    def tau(self, i, key=None):
        st = self._stats.get(key if key is not None else self._active) or {}
        return st.get(i, {}).get("tau_ref", 0.0) + float(self.dtau[i].detach())

    def _g_of(self, i, I, tau):
        return 1.0 - self.lam(i) * (
            1.0 - torch.sigmoid(self.alpha(i) * (torch.log1p(I) - tau))
        )

    @staticmethod
    def _g_scalars(I, lam, alpha, tau):
        """The same gate from plain floats, for reporting off-device."""
        return 1.0 - lam * (1.0 - torch.sigmoid(alpha * (torch.log1p(I) - tau)))

    # ------------------------------------------------------------------ graph selection
    def use_graph(self, key):
        """Select which graph's statistics the gate reads. Required before a gated forward."""
        self._active = key

    def calibrated(self, key):
        return key in self._stats

    def known_graphs(self):
        return sorted(self._stats)

    # ------------------------------------------------------------------ reference bank
    def set_reference_bank(self, refs, meta=None):
        """Store M source reference problems in a GRAPH-INDEPENDENT form.

        Each ref is {"qemb": [1, D] float tensor, "seeds": [(name, weight), ...], "id": str}
        and optionally "seed_emb": [len(seeds), D], the FROZEN source-graph embedding of each
        seed name. Names alone re-link only where the target spells a concept identically;
        the embeddings are what let `link="semantic"` find "thin-film growth" from "thin film
        deposition". They are source-side artefacts of a frozen encoder, so carrying them
        costs one float32 row per seed (~1.8 MB at M=16) and concedes nothing about zero-shot.
        """
        bank = []
        for r in refs:
            e = r["qemb"].detach().float().cpu()
            assert e.dim() == 2 and e.shape[0] == 1, (
                f"a reference must be ONE query, got question_embeddings of shape "
                f"{tuple(e.shape)}; batches of queries are not references")
            seeds = [(str(n), float(w)) for n, w in r["seeds"]]
            se = r.get("seed_emb")
            if se is not None:
                se = se.detach().float().cpu()
                # ROW k OF seed_emb IS SEED k. If that correspondence slips, every semantic
                # link is to the wrong concept and nothing downstream can detect it.
                assert se.dim() == 2 and se.shape[0] == len(seeds), (
                    f"seed_emb has {tuple(se.shape)} rows for {len(seeds)} seeds; they are "
                    f"positionally paired and must be built together")
            bank.append({"qemb": e, "seeds": seeds, "seed_emb": se, "id": r.get("id")})
        assert len(bank) >= 2, (
            f"a reference bank of {len(bank)} cannot give a cross-query variance; "
            f"raise CQIG_M")
        self.ref_bank = bank
        self.ref_meta = dict(meta or {})

    def materialise(self, node2id, num_nodes, device=None, dtype=None, batch_size=1,
                    node_emb=None, link="exact", link_k=3, link_temp=0.05, link_floor=0.0):
        """Re-link the stored source problems against THIS graph's vocabulary.

        `link="exact"` is the operation the indexer performs when it builds start_nodes_mask:
        `start_mask[node2id[name]] = weight`, silently dropping names the graph does not
        contain. That silent drop is the failure mode: 5% coverage and 100% coverage produce
        the same shapes and the same absence of errors.

        `link="semantic"` keeps every exact match and rescues the rest. A seed the target
        does not spell is embedded (its frozen source embedding is in the bank), matched to
        its `link_k` nearest target nodes, filtered at `link_floor`, and its weight split
        across the survivors by softmax(cos / link_temp). The SPLIT IS NORMALISED OVER THE
        SURVIVORS, so the seed contributes exactly its original weight whether it landed on
        one node or three, and an exactly linked seed and a semantically linked one inject
        the same mass. `link_floor` comes from the source side (see the trainer), so no
        target statistic sets it.

        Returns (batches, coverage). Coverage separates exact from semantic hits, because
        an aggregate that mixes them cannot answer the question the arm exists to ask.

        Every graph gets the SAME M problems, so `coverage["queries"]` is M on every graph
        and two graphs' statistics are computed from equal-sized reference sets.
        """
        assert self.ref_bank, "no reference bank; call set_reference_bank first"
        assert link in ("exact", "semantic"), f"unknown cqig link mode {link!r}"
        semantic = link == "semantic"
        if semantic:
            assert node_emb is not None, (
                "link='semantic' needs the TARGET graph's node embeddings (graph.x); it is "
                "None, so this graph was indexed without node features")
            assert int(node_emb.shape[0]) == int(num_nodes), (
                f"node_emb has {int(node_emb.shape[0])} rows for a {num_nodes}-node graph")
            missing = [r["id"] for r in self.ref_bank if r.get("seed_emb") is None]
            assert not missing, (
                f"{len(missing)} reference(s) carry no seed embeddings, so they cannot be "
                f"linked semantically. The bank was frozen before this feature existed; "
                f"retrain, or run this arm with CQIG_LINK=exact")
        masks, embs, total, empty = [], [], 0, 0
        n_exact = n_sem = 0
        sem_cos, sem_deg = [], []
        for r in self.ref_bank:
            mask = torch.zeros(num_nodes, dtype=torch.float32)
            hit, pend = 0, []
            for k, (name, w) in enumerate(r["seeds"]):
                j = node2id.get(name)
                if j is not None and 0 <= int(j) < num_nodes:
                    # ACCUMULATE. Exact hits have distinct ids so this matches the old
                    # assignment, but a semantic link can land on an already-seeded node
                    # and overwriting there would quietly destroy the other seed's weight.
                    mask[int(j)] += w
                    n_exact += 1
                    hit += 1
                elif semantic:
                    pend.append((k, w))
            if pend:
                # Onto the GRAPH's device, not the bank's: the bank is a few hundred CPU
                # rows and the graph is hundreds of thousands, so moving the small side is
                # the only direction that does not stream the whole graph through host RAM.
                q = r["seed_emb"][[k for k, _ in pend]].to(node_emb.device)
                cos, idx = topk_cosine(q, node_emb, link_k)
                cos, idx = cos.cpu(), idx.cpu()
                for t, (_, w) in enumerate(pend):
                    keep = cos[t] >= link_floor
                    if not bool(keep.any()):
                        continue                      # below the floor: a real miss, not a link
                    c, ix = cos[t][keep], idx[t][keep]
                    # Softmax over the SURVIVORS, times w: total attachment weight preserved.
                    wt = torch.softmax(c / max(float(link_temp), 1e-6), dim=0) * w
                    mask.index_add_(0, ix, wt)
                    n_sem += 1
                    hit += 1
                    sem_cos.append(float(c.max()))
                    sem_deg.append(int(ix.numel()))
            total += len(r["seeds"])
            empty += int(hit == 0)
            masks.append(mask)
            embs.append(r["qemb"])
        bs = max(1, int(batch_size))
        batches = []
        for a in range(0, len(masks), bs):
            b = {"question_embeddings": torch.cat(embs[a:a + bs], 0).clone(),
                 "start_nodes_mask": torch.stack(masks[a:a + bs], 0)}
            if device is not None:
                b = {k: v.to(device) for k, v in b.items()}
            if dtype is not None:
                b = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in b.items()}
            batches.append(b)
        found = n_exact + n_sem
        med = sorted(sem_cos)[len(sem_cos) // 2] if sem_cos else float("nan")
        cov = {"queries": len(masks),
               "batches": len(batches),
               "seeds_found": found,
               "seeds_total": total,
               "seed_coverage": (found / total) if total else 0.0,
               # KEPT SEPARATE ON PURPOSE. The semantic arm's whole claim is that it raises
               # coverage, so an aggregate that hides which half moved cannot test it.
               "seeds_exact": n_exact,
               "seeds_semantic": n_sem,
               "seeds_unmatched": total - found,
               "exact_coverage": (n_exact / total) if total else 0.0,
               "link": link,
               "link_k": int(link_k) if semantic else 0,
               "link_temp": float(link_temp) if semantic else 0.0,
               "link_floor": float(link_floor) if semantic else 0.0,
               "sem_cos_median": med,
               "sem_nodes_per_seed": (sum(sem_deg) / len(sem_deg)) if sem_deg else 0.0,
               "queries_with_no_seed": empty}
        return batches, cov

    # ------------------------------------------------------------------ calibration
    def start_calibration(self, key, phase="mean", apply_existing_gates=False):
        """Begin one reference pass for `key`.

        phase="mean" accumulates sum h and sum ||h||^2; phase="dev" accumulates
        sum ||h - mu||^2 and keeps a sample of the per-query deviations. Both passes must
        see the same M references, which finish_calibration checks.

        apply_existing_gates: run the already-calibrated gates while re-accumulating. This
        is what makes a second round consistent when more than one layer is gated -- see
        the module docstring. It is a no-op on the first round, when nothing is calibrated.
        """
        assert phase in ("mean", "dev"), f"unknown calibration phase {phase!r}"
        if phase == "dev":
            assert self._mu, "phase 'dev' needs phase 'mean' to have finished first"
        else:
            self._mu, self._energy = {}, {}
        self.mode, self._acc, self._acc_key, self._n_ref = "calibrate", {}, key, 0
        self._phase = phase
        self._active = key
        self._calib_gate = bool(apply_existing_gates) and key in self._stats
        if phase == "mean" and not self._calib_gate:
            # DROP THE OUTGOING STATISTICS BEFORE BUILDING THEIR REPLACEMENT. mu is one
            # float32 row per node, which on the CS train graph is 950 MB per gated layer;
            # holding the old set alive through a recalibration doubles the peak for no
            # reason, since nothing reads it until the new one is installed. Kept only when
            # apply_existing_gates asked for it, which is the round-2 path.
            self._stats.pop(key, None)

    def note_reference_batch(self, b):
        self._n_ref += b

    def finish_mean(self):
        """Close phase "mean": turn the running sums into mu (and the energy)."""
        assert self._n_ref > 0, "no reference queries were run"
        assert self._acc, "calibration collected nothing; are any layers in gate_layers?"
        # div_, not div: the accumulator IS the mean once divided, and nothing else holds a
        # reference to it. Allocating a second [N, D] float32 here would be another 950 MB
        # per gated layer on the CS graph, at the moment memory is already at its peak.
        #
        # mu="zero" IS THE ABLATION THAT DECIDES WHETHER THE REFERENCE BANK DOES ANYTHING.
        # With mu = 0 the statistic degenerates to I = ||h||^2 / s, an activation-MAGNITUDE
        # gate, and CQIG's whole claim -- that a node earns its message by responding to
        # THIS query unlike it responds to unrelated ones -- is gone. If this arm reproduces
        # the method's score, the reference bank was never load-bearing and the mechanism is
        # magnitude gating under another name.
        #
        # The mean pass still RUNS and its result is discarded. That wastes M forwards, and
        # it is the point: the two arms then differ in the value of mu and in nothing else,
        # not in code path, not in RNG consumption. Zeroed in place, so no extra memory, and
        # every quantity derived downstream (V, the layer scale s, tau_ref) is recomputed
        # consistently against mu = 0 rather than being inherited from a real one.
        self._mu = {i: (a["sum_h"].zero_() if self.mu_zero else a["sum_h"].div_(self._n_ref))
                    for i, a in self._acc.items()}
        self._energy = {i: a["sum_e"].div_(self._n_ref) for i, a in self._acc.items()}
        self._m_ref, self._acc, self.mode = self._n_ref, {}, "off"
        return len(self._mu)

    def finish_calibration(self):
        """Close phase "dev": variance, layer scale, threshold, and what the gate will do.

        s is the MEDIAN reference variance over responding nodes, found in two steps: a
        median over V > 0 fixes the order of magnitude, then nodes more than six orders
        below it are dropped and the median is retaken. A single median over V > 0 would be
        pulled down by whatever residue the barely-reachable part of the graph leaves.
        """
        M = self._n_ref
        assert M >= 2, f"the variance needs at least 2 reference queries, got {M}"
        assert M == self._m_ref, (
            f"the mean pass saw {self._m_ref} reference queries and the deviation pass "
            f"{M}; the two must be the same set")
        assert self._acc, "the deviation pass collected nothing"
        loo = (M / (M - 1.0)) ** 2         # in-sample deviation -> leave-one-out deviation
        report, per_layer = {}, {}
        for i, a in self._acc.items():
            mu = self._mu[i]
            V = a["sum_sq"] / (M - 1.0)                    # unbiased cross-query variance
            # WHICH NODES COUNT AS RESPONDING, AND WHY THE CUT IS ANCHORED AT THE TOP.
            # mu is a mean of M float32 values, so a node whose state never moves still
            # lands ~1e-10 of the working scale away from it -- uniformly, on every
            # unreachable node. Most of a KG is unreachable from any one query's seeds, so
            # a cut anchored at a MEDIAN over V > 0 sits inside that dust cloud, keeps
            # every node "live", and hands back a scale ten orders too small. The dust is
            # many orders below any real response and the response range spans at most a
            # few, so anchoring six orders under the largest variance separates them with
            # room on both sides.
            vmax = float(V.max()) if V.numel() else 0.0
            n_pos = int((V > 0).sum())
            live = (V > RESP * vmax) if vmax > 0 else (V > 0)
            s = float(V[live].median()) if int(live.sum()) else 0.0
            n_live = int(live.sum())
            vbar = float(V[live].mean()) if n_live else 0.0
            degenerate = not (s > 0)
            if self.norm == "layer":
                # ONE robust scale for the whole layer, so I is comparable BETWEEN nodes.
                # s == 0 means no node responded to any reference query; 1.0 keeps I finite
                # and the report says so rather than dividing by eps.
                scale = torch.full((1, 1), (s if s > 0 else 1.0) + self.eps,
                                   device=V.device, dtype=torch.float32)
            elif self.norm == "node":                      # ablation: self-normalising
                scale = (V + (self.rho * vbar + self.eps)).unsqueeze(0)
            else:                                          # ablation: the original energy
                scale = (self._energy[i] + (self.rho + self.eps)).unsqueeze(0)
            per_layer[i] = {"mu": mu, "scale": scale, "live": live, "V": V, "s": s}

            # --- what a reference query actually scores here, and what the gate does ----
            sample = torch.cat(a["rows"], 0) * loo         # [rows, N] on cpu
            I = sample / scale.detach().to(sample.device)
            resp = I[I > RESP]
            if resp.numel() >= 100:
                tau_ref = float(torch.log1p(resp).median())
                tau_src = "reference queries"
            else:
                # Too few responding pairs for a median to mean anything: fall back to the
                # variance, which is the same quantity averaged over queries.
                iota = (V.unsqueeze(0) / scale).flatten()
                tau_ref = float(torch.log1p(iota[live]).median()) if n_live else 0.0
                tau_src = f"reference variance (only {int(resp.numel())} responding pairs)"
            per_layer[i]["tau_ref"] = tau_ref
            per_layer[i]["tau_src"] = tau_src
            lam, alpha = float(self.lam(i).detach()), float(self.alpha(i).detach())
            tau = tau_ref + float(self.dtau[i].detach())
            gv = self._g_scalars(I, lam, alpha, tau)
            # WHAT THE ACTIVE OP MULTIPLIES BY, not what "gate" would have. Reporting g
            # under a centring arm would describe an operator that arm does not run.
            cv = (gv if self.op == "gate" else
                  torch.full_like(gv, lam) if self.op == "centre-fixed" else 1.0 - gv)
            # mu_v = 0 EXACTLY is "no reference query ever reached this node". A centring op
            # cannot touch those nodes at all while a gating op damps them hardest, so this
            # fraction is how much of the graph the two arms treat oppositely.
            mu_zero = int((mu.norm(dim=-1) == 0).sum())
            report[i] = {
                "nodes": int(V.numel()), "responding": n_pos, "live": n_live,
                "responding_frac": n_live / max(int(V.numel()), 1),
                "scale": s, "var_max": vmax, "var_mean_live": vbar,
                "var_q": _describe(V[live]) if n_live else _describe(None),
                "I": _describe(I), "gate": _describe(gv),
                "op": self.op, "coef": _describe(cv),
                "mu_source": self.mu_source, "mu_zero": mu_zero,
                "mu_zero_frac": mu_zero / max(int(V.numel()), 1),
                "gate_unreached": float(self._g_scalars(
                    torch.zeros(()), lam, alpha, tau)),
                "resp_frac": float(resp.numel()) / max(int(I.numel()), 1),
                "sample_queries": int(sample.shape[0]),
                "degenerate": degenerate, "lam": lam, "alpha": alpha, "tau": tau,
                "tau_ref": tau_ref, "tau_src": tau_src, "M": M,
            }
        self._stats[self._acc_key] = per_layer
        self._active = self._acc_key
        self._acc, self._mu, self._energy = {}, {}, {}
        self._acc_key, self.mode, self._calib_gate = None, "off", False
        return report

    # ------------------------------------------------------------------ live measurement
    def reset_live(self, bins=1024):
        """Start accumulating the gate distribution over REAL forwards, and the
        gold-vs-hard-negative alignment. Both are measurement only: nothing is calibrated
        from them and no target query or label enters the statistics."""
        self._live, self._align, self._edit, self.last_I = {}, {}, {}, {}
        self._live_bins = int(bins)

    def stop_live(self):
        self._live, self._align, self._edit, self.last_I = None, {}, {}, {}

    def recording(self):
        return self._live is not None

    def _note_live(self, i, g):
        a = self._live.get(i)
        if a is None:
            a = self._live[i] = {
                "n": 0,
                "sum": torch.zeros((), device=g.device, dtype=torch.float64),
                "sqs": torch.zeros((), device=g.device, dtype=torch.float64),
                "min": torch.full((), float("inf"), device=g.device),
                "max": torch.full((), float("-inf"), device=g.device),
                "hist": torch.zeros(self._live_bins, device=g.device),
            }
        f = g.flatten().float()
        a["n"] += int(f.numel())
        a["sum"] += f.sum().double()
        a["sqs"] += (f * f).sum().double()
        a["min"] = torch.minimum(a["min"], f.min())
        a["max"] = torch.maximum(a["max"], f.max())
        a["hist"] += torch.histc(f, bins=self._live_bins, min=0.0, max=1.0)

    def _note_edit(self, i, hh, h):
        """Running ||h~ - h|| / ||h|| over nodes, for whichever op is active.

        Measured on the states themselves rather than derived from the coefficient, because
        the two ops move a state by different amounts at the same coefficient and the point
        of the number is to compare them. Nodes with a numerically zero state are dropped
        rather than clamped: most of a KG is unreachable from one query's seeds, and a ratio
        of 0/eps there would dominate the mean with an artefact.
        """
        a = self._edit.get(i)
        if a is None:
            a = self._edit[i] = {
                "n": 0,
                "sum": torch.zeros((), device=h.device, dtype=torch.float64),
                "max": torch.zeros((), device=h.device),
                "moved": 0,
            }
        hn = h.float().norm(dim=-1)                              # [B, N]
        dn = (hh.float() - h.float()).norm(dim=-1)
        live = hn > 0
        if not bool(live.any()):
            return
        rel = dn[live] / hn[live]
        a["n"] += int(rel.numel())
        a["sum"] += rel.sum().double()
        a["max"] = torch.maximum(a["max"], rel.max())
        a["moved"] += int((dn[live] > 0).sum())

    def note_alignment(self, target_mask, doc_idx, doc_scores, k=50):
        """Is a high I actually concentrated on the gold papers?

        The hypothesis behind the gate is that informative nodes lie on the paths that
        matter. This is its most direct falsifiable form: for every query, compare I at the
        GOLD document nodes against I at the `k` documents the semantic scorer ranks
        highest among the non-golds -- the same hard negatives the training objective uses.
        The statistic is the AUC, i.e. the probability that a random gold outscores a random
        hard negative in informativeness, with ties counted as half. 0.5 is no alignment,
        and no amount of tuning lam/alpha/tau can rescue a gate whose statistic sits there.
        """
        if self._live is None or not self.last_I or target_mask is None:
            return
        with torch.no_grad():
            tgt = target_mask.index_select(1, doc_idx.to(target_mask.device)).bool()
            for i, I in self.last_I.items():
                Id = I.index_select(1, doc_idx.to(I.device)).float()
                a = self._align.setdefault(
                    i, {"wins": 0.0, "n": 0, "gold": 0.0, "neg": 0.0})
                for b in range(min(Id.shape[0], tgt.shape[0])):
                    gold = tgt[b].nonzero(as_tuple=False).flatten()
                    if gold.numel() == 0:
                        continue
                    s = doc_scores[b].detach().float().clone()
                    s[gold] = float("-inf")
                    kk = int(min(k, s.numel() - gold.numel()))
                    if kk <= 0:
                        continue
                    neg = s.topk(kk).indices
                    gi = Id[b].index_select(0, gold.to(Id.device)).unsqueeze(1)
                    ni = Id[b].index_select(0, neg.to(Id.device)).unsqueeze(0)
                    a["wins"] += float((gi > ni).float().mean()
                                       + 0.5 * (gi == ni).float().mean())
                    a["n"] += 1
                    a["gold"] += float(gi.mean())
                    a["neg"] += float(ni.mean())

    def live_report(self):
        """Gate distribution over the real forwards, plus the alignment AUC."""
        out = {}
        for i, a in (self._live or {}).items():
            n = max(a["n"], 1)
            mean = float(a["sum"]) / n
            var = max(float(a["sqs"]) / n - mean * mean, 0.0)
            lo, hi = float(a["min"]), float(a["max"])
            h = a["hist"]
            cdf = torch.cumsum(h, 0) / max(float(h.sum()), 1.0)
            edges = (torch.arange(self._live_bins, device=h.device).float() + 0.5) \
                / self._live_bins
            qv = {}
            for q in (0.1, 0.5, 0.9):
                j = int(torch.searchsorted(cdf, torch.tensor(q, device=h.device)).clamp(
                    max=self._live_bins - 1))
                # CLAMPED INTO THE OBSERVED RANGE. min and max are exact while the
                # quantiles are bin centres, so an unclamped p10 can print below the min
                # and read as a bug in the gate rather than in the histogram.
                qv[f"p{int(q * 100)}"] = min(max(float(edges[j]), lo), hi)
            al = self._align.get(i)
            ok = bool(al and al["n"])
            ed = self._edit.get(i)
            out[i] = {"n": a["n"], "min": float(a["min"]), "max": float(a["max"]),
                      "mean": mean, "std": var ** 0.5, **qv,
                      "op": self.op,
                      "auc": (al["wins"] / al["n"]) if ok else float("nan"),
                      "auc_n": (al["n"] if al else 0),
                      "I_gold": (al["gold"] / al["n"]) if ok else float("nan"),
                      "I_neg": (al["neg"] / al["n"]) if ok else float("nan"),
                      "edit_rel_mean": (float(ed["sum"]) / max(ed["n"], 1)) if ed else 0.0,
                      "edit_rel_max": float(ed["max"]) if ed else 0.0,
                      # What fraction of live nodes the operator moved AT ALL. Under a
                      # centring op this is the coverage question in its bluntest form:
                      # mu_v = 0 means the node is untouched however uninformative it is.
                      "edit_frac": (ed["moved"] / max(ed["n"], 1)) if ed else 0.0}
        return out

    # ------------------------------------------------------------------ the hook
    @staticmethod
    def _chunk(n_rows, d):
        return max(1024, int(_CHUNK_BYTES // max(n_rows * d * 4, 1)))

    def _accumulate_mean(self, i, h):
        """Running sum_h and sum_||h||^2, in float32 and in chunks over the node axis.

        FLOAT32 ACCUMULATORS. Under autocast h is bfloat16, whose ~8-bit mantissa would
        lose most of a 64-term running sum. Chunked because the float32 upcast of a
        [B, N, D] state is 2 GB at B=8 on this graph.
        """
        b, n, d = h.shape
        acc = self._acc.setdefault(i, {
            "sum_h": torch.zeros(n, d, device=h.device, dtype=torch.float32),
            "sum_e": torch.zeros(n, device=h.device, dtype=torch.float32),
        })
        step = self._chunk(b, d)
        for a0 in range(0, n, step):
            a1 = min(a0 + step, n)
            hd = h[:, a0:a1].detach().float()
            acc["sum_h"][a0:a1] += hd.sum(0)
            acc["sum_e"][a0:a1] += (hd * hd).sum(-1).sum(0)

    def _accumulate_dev(self, i, h):
        """Running sum ||h - mu||^2, plus a bounded sample of the deviations themselves."""
        mu = self._mu.get(i)
        if mu is None or mu.shape[0] != h.shape[1]:
            return
        sq = self._sqdist(h.detach(), mu)                  # [B, N] float32, exact
        acc = self._acc.setdefault(i, {
            "sum_sq": torch.zeros(h.shape[1], device=h.device, dtype=torch.float32),
            "rows": [], "kept": 0})
        acc["sum_sq"] += sq.sum(0)
        cap = max(1, _SAMPLE_ELEMS // max(h.shape[1], 1))
        if acc["kept"] < cap:
            take = min(sq.shape[0], cap - acc["kept"])
            acc["rows"].append(sq[:take].cpu())
            acc["kept"] += take

    def _sqdist(self, h, mu):
        """||h - mu||^2 per node, in float32, chunked over the node axis.

        The subtraction must not happen in bfloat16: the deviation is ~1% of the state
        under early-late fusion and bfloat16 resolves ~0.4%, so a bfloat16 difference is
        ~40% rounding noise. Chunking keeps the float32 temporary bounded.
        """
        b, n, d = h.shape
        out = torch.empty(b, n, device=h.device, dtype=torch.float32)
        step = self._chunk(b, d)
        for a0 in range(0, n, step):
            a1 = min(a0 + step, n)
            df = h[:, a0:a1].float() - mu[a0:a1].unsqueeze(0)
            out[:, a0:a1] = (df * df).sum(-1)
        return out

    def _centre(self, h, mu, c):
        """h - c * mu per node, chunked over the node axis, never in float32 at full size.

        `c` is the subtraction coefficient: [B, N] for an adaptive op, 0-dim for a fixed one.
        Written naively as `h - c.unsqueeze(-1) * mu` this allocates a float32 [B, N, D]
        because mu is float32 -- 1.9 GB at B=1 on the CS graph, twice the state it is
        editing, and then another copy to cast back. Chunked, with the cast to h's dtype
        INSIDE the chunk, the transient is bounded by _CHUNK_BYTES the same way _sqdist is.
        The slice assignments are autograd-tracked copies into a fresh tensor, so gradient
        still reaches h, lam, alpha and dtau; the loop covers 0..n so nothing is left
        uninitialised.
        """
        b, n, d = h.shape
        out = torch.empty_like(h)
        step = self._chunk(b, d)
        fixed = c.dim() == 0
        for a0 in range(0, n, step):
            a1 = min(a0 + step, n)
            cc = (c if fixed else c[:, a0:a1].unsqueeze(-1)).to(h.dtype)
            out[:, a0:a1] = h[:, a0:a1] - cc * mu[a0:a1].to(h.dtype).unsqueeze(0)
        return out

    def _coef(self, i, g):
        """The coefficient the active op multiplies by, given the gate value.

        Returned as the quantity actually applied, so the report and the live histogram
        describe what happened rather than what would have happened under `op="gate"`.
        """
        if self.op == "gate":
            return g
        if self.op == "centre-fixed":
            return self.lam(i)                 # 0-dim: query-independent by construction
        return 1.0 - g                         # "centre": damp the background, not the state

    def _apply_op(self, i, h, st, I, detach_dtau=False):
        """Edit the state under the active op. Returns (h~, g), g for the live report."""
        dt = float(self.dtau[i].detach()) if detach_dtau else self.dtau[i]
        g = self._g_of(i, I, st["tau_ref"] + dt)
        if self.op == "gate":
            return h * g.to(h.dtype).unsqueeze(-1), g
        # mu is the statistic being subtracted, so a centring op is a NO-OP wherever the
        # references never reached (mu_v = 0 exactly), which is the opposite of what gating
        # does there. See the module docstring; mu_zero_frac reports how much of the graph
        # that is.
        return self._centre(h, st["mu"], self._coef(i, g)), g

    def _stats_for(self, i, h):
        assert self._active is not None, (
            "cqig is in gate mode with no active graph; call use_graph(key) first")
        st = self._stats.get(self._active)
        assert st is not None and i in st, (
            f"cqig has no statistics for graph {self._active!r} at layer {i}; "
            f"calibrate this graph before scoring it (known: {self.known_graphs()})")
        # A wrong-graph lookup would otherwise fail as a bare broadcast error deep in
        # autograd. Say which graph and which shapes.
        assert st[i]["mu"].shape[0] == h.shape[1], (
            f"cqig statistics for graph {self._active!r} have {st[i]['mu'].shape[0]} nodes "
            f"but this forward has {h.shape[1]}; the wrong graph's statistics are active")
        return st[i]

    def _make_hook(self, i):
        def hook(_module, args):
            # args = (input, query, boundary, edge_index, edge_type, size, edge_weight)
            if i not in self.gate_layers or self.mode == "off":
                return None                    # not gated: never calibrated either
            h = args[0]                                    # [B, N, D] = h^(l)
            if self.mode == "calibrate":
                with torch.no_grad():
                    if self._phase == "mean":
                        self._accumulate_mean(i, h)
                    else:
                        self._accumulate_dev(i, h)
                    if not self._calib_gate:
                        return None                        # pass through unchanged
                    st = (self._stats.get(self._active) or {}).get(i)
                    if st is None or st["mu"].shape[0] != h.shape[1]:
                        return None
                    I = self._sqdist(h, st["mu"]) / st["scale"]
                    hh, _ = self._apply_op(i, h, st, I, detach_dtau=True)
                return (hh,) + tuple(args[1:])
            if self.mode != "gate":
                return None
            st = self._stats_for(i, h)
            # DETACHED BY DEFAULT: see the module docstring. A gradient path through I is a
            # free direction that opens every gate at once, and detaching also lets the
            # float32 chunked subtraction cost nothing in stored activations.
            if self.grad_through_I:
                I = self._sqdist(h, st["mu"]) / st["scale"]
            else:
                with torch.no_grad():
                    I = self._sqdist(h, st["mu"]) / st["scale"]
            # Only the messages are edited. The boundary condition is a separate argument
            # and the residual is added outside the layer, so both are left alone. The
            # coefficient is cast back to h's dtype inside the op: a float32 one would
            # silently promote the whole state and double every gated layer's activations.
            hh, g = self._apply_op(i, h, st, I)
            # KEPT AS A TENSOR. float() here is a device sync on every gated layer of every
            # training step, which at four gated layers is four stalls per step for a
            # number only summary() ever reads.
            self.last_gate[i] = g.detach().mean()
            if self._live is not None:
                with torch.no_grad():
                    self._note_live(i, self._coef(i, g).detach().expand_as(g))
                    self.last_I[i] = I.detach()
                    # HOW BIG THE EDIT ACTUALLY IS, relative to the state it edits. Under
                    # early-late fusion h is dominated by the static entity embedding, so
                    # the same lam is a very different perturbation under the two ops, and
                    # `layer_norm: yes` renormalises much of a magnitude change away. This
                    # is the number that says whether the operator did anything at all.
                    self._note_edit(i, hh.detach(), h.detach())
            return (hh,) + tuple(args[1:])
        return hook

    # ------------------------------------------------------------------ checkpointing
    def get_extra_state(self):
        return {"ref_bank": self.ref_bank, "ref_meta": self.ref_meta,
                "gate_layers": sorted(self.gate_layers), "norm": self.norm,
                "op": self.op, "mu": self.mu_source}

    def set_extra_state(self, state):
        if not state:
            return
        # STATISTICS BELONG TO A (WEIGHTS, GRAPH) PAIR, SO NEW WEIGHTS INVALIDATE THEM.
        # `load_best_model_at_end` defaults to true, so the final evaluate() and predict()
        # run on the BEST checkpoint while _stats still held mu from the LAST epoch's
        # weights -- and the recalibration guard, which keys on the training step, saw no
        # reason to refresh them. The reported numbers would have been the best model
        # gated by a different model's reference statistics. Clearing here forces one
        # calibration pass on the next inference, which is the cheap and correct answer.
        if self._stats:
            print(f"[cqig] checkpoint loaded: dropping reference statistics for "
                  f"{self.known_graphs()}; they belong to the previous weights")
        self._stats, self._active = {}, None
        self.ref_bank = state.get("ref_bank")
        self.ref_meta = state.get("ref_meta", {})
        if state.get("gate_layers"):
            want = set(state["gate_layers"])
            if want != self.gate_layers:
                print(f"[cqig] checkpoint gates layers {sorted(want)} (0-indexed), the "
                      f"config asked for {sorted(self.gate_layers)}; using the checkpoint's")
            self.gate_layers = want
        if state.get("norm"):
            if state["norm"] != self.norm:
                print(f"[cqig] checkpoint was normalised by {state['norm']!r}, the config "
                      f"asked for {self.norm!r}; using the checkpoint's")
            self.norm = state["norm"]
        # THE OPERATOR IS PART OF THE TRAINED MODEL, not a scoring-time choice: lam, alpha
        # and dtau were fitted under one of them. Absent on a checkpoint written before this
        # existed, which can only have been "gate".
        op = state.get("op", "gate")
        if op != self.op:
            print(f"[cqig] checkpoint was trained with op={op!r}, the config asked for "
                  f"{self.op!r}; using the checkpoint's")
        self.op, self.centring = op, op != "gate"
        # Same argument for mu: an ablated model recalibrating with a real mu at scoring
        # time would report the method's number under the ablation's name.
        mu = state.get("mu", "ref")
        if mu != self.mu_source:
            print(f"[cqig] checkpoint was trained with mu={mu!r}, the config asked for "
                  f"{self.mu_source!r}; using the checkpoint's")
        self.mu_source, self.mu_zero = mu, mu == "zero"

    # ------------------------------------------------------------------ diagnostics
    def n_reference(self):
        return self._n_ref

    def stats_bytes(self):
        return sum(v["mu"].numel() * v["mu"].element_size()
                   for st in self._stats.values() for v in st.values())

    def summary(self):
        return {
            "gate_layers_1indexed": sorted(i + 1 for i in self.gate_layers),
            "op": self.op,
            "mu": self.mu_source,
            "norm": self.norm,
            "grad_through_I": self.grad_through_I,
            "lam": [round(float(self.lam(i).detach()), 4) for i in sorted(self.gate_layers)],
            "alpha": [round(float(self.alpha(i).detach()), 4) for i in sorted(self.gate_layers)],
            "dtau": [round(float(self.dtau[i].detach()), 4) for i in sorted(self.gate_layers)],
            "active_graph": self._active,
            "calibrated_graphs": self.known_graphs(),
            "reference_queries": len(self.ref_bank or []),
            "stats_mb": round(self.stats_bytes() / 2**20, 1),
            "mean_gate": {k + 1: round(float(v), 4)
                          for k, v in sorted(self.last_gate.items())},
        }

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
