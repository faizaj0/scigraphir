# CCMP engine fixes -- paste as ONE cell, AFTER the CONTROL finishes, BEFORE the CCMP arm.
# Safe there: with ccmp=False every branch below is skipped, so the control you just
# ran is byte-identical to a control run with these applied.
import ast, os
R = "/content/gfm-rag/gfmrag"
P = []
P.append(("trainers/fusion_trainer.py", r"""
                    I, Y, C = [], [], []
""", r"""
                    # STRUCTURAL REACH. A^(0) = the query's seeds, A^(l+1) = A^(l) u
                    # N(A^(l)). Supervision at layer l is restricted to A^(l), because
                    # h^(l) at an unreached node carries no query information at all: it
                    # is still the static entity embedding that early-late fusion put
                    # there. Asking the head to predict a query-specific target from a
                    # query-independent state is asking it to fit noise, and on this graph
                    # it is nearly all of the sampled supervision.
                    reach = (seeds[b] > 0).float()
                    reaches = []
                    for _ in range(L):
                        reaches.append(reach.clone())
                        nxt = torch.sparse.mm(self._ccmp_P, reach.unsqueeze(1)).squeeze(1)
                        reach = ((nxt > 0) | (reach > 0)).float()
                    I, Y, C = [], [], []
"""))

P.append(("trainers/fusion_trainer.py", r"""
                        cc = (2.0 * y - 1.0).abs()
""", r"""
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
"""))

P.append(("trainers/fusion_trainer.py", r"""
        loss = preds[0].new_zeros(())
        wsum, nsup, csum = 0.0, 0, 0.0
        for b, hit in enumerate(targets):
            if hit is None:
                continue
            idx, y, c = hit
            dev = preds[0].device
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
                loss = loss + (cc * bce).sum()
                wsum += float(cc.sum())
                nsup += int(ii.numel())
                csum += float(cc.sum())
        if wsum <= 0:
            return None, {}
        return loss / wsum, {"ccmp_nodes": nsup / max(len(targets), 1),
                             "ccmp_conf": csum / max(nsup, 1)}
""", r"""
        dev = preds[0].device
        pn = preds[0].new_zeros(())      # positive-side numerator (y > 1/2)
        nn_ = preds[0].new_zeros(())     # negative-side numerator
        pd = nd = 0.0
        nsup, csum, npos = 0, 0.0, 0
        for b, hit in enumerate(targets):
            if hit is None:
                continue
            idx, y, c = hit
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
                m = yy > 0.5
                if bool(m.any()):
                    pn = pn + (cc[m] * bce[m]).sum(); pd += float(cc[m].sum())
                    npos += int(m.sum())
                if bool((~m).any()):
                    nn_ = nn_ + (cc[~m] * bce[~m]).sum(); nd += float(cc[~m].sum())
                nsup += int(ii.numel())
                csum += float(cc.sum())
        if pd <= 0 and nd <= 0:
            return None, {}
        if pd <= 0:
            loss = nn_ / nd
        elif nd <= 0:
            loss = pn / pd
        else:
            loss = 0.5 * (pn / pd) + 0.5 * (nn_ / nd)
        return loss, {"ccmp_nodes": nsup / max(len(targets), 1),
                      "ccmp_conf": csum / max(nsup, 1),
                      # share of supervised nodes on the gold side. If this is ~0 the
                      # target has collapsed and nothing downstream can work.
                      "ccmp_pos": npos / max(nsup, 1)}
"""))

P.append(("models/ultra/models.py", r"""
            for _li, layer in enumerate(self.layers):
""", r"""
            # STRUCTURAL REACH, not "the state is nonzero". Early-late entity fusion puts a
            # static text embedding on EVERY node before propagation, so a nonzero-state
            # test is true almost everywhere from layer 1 and the "activated set" was in
            # practice the whole graph. Then the gate's mean was a graph-wide mean and the
            # normalisation said nothing about what the query had reached.
            _reach = None
            _seed = getattr(self, "_ccmp_seeds", None)
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
"""))

P.append(("models/ultra/models.py", r"""
                            _act = (layer_input.detach().abs().sum(-1) > 0).to(_num.dtype)
                            _den = ((_num * _act).sum(-1, keepdim=True)
                                    / _act.sum(-1, keepdim=True).clamp(min=1.0))
                            _num = _num / _den.clamp(min=1e-6)
                        _gt = (1.0 - _eta) + _eta * _num
                        layer_input = layer_input * _gt.unsqueeze(-1).to(layer_input.dtype)

""", r"""
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
                        _nx = torch.sparse.mm(self._ccmp_adj, _reach.t()).t()
                        _reach = ((_nx > 0) | (_reach > 0)).to(_reach.dtype)
                else:
                    _msg = layer_input

"""))

P.append(("models/ultra/models.py", r"""
                hidden = layer(
                    layer_input,
""", r"""
                hidden = layer(
                    _msg,
"""))

P.append(("models/ultra/models.py", r"""
                    # residual connection here
                    hidden = hidden + layer_input
""", r"""
                    # residual connection here. UNGATED `layer_input`, not `_msg`: the
                    # gate belongs on the messages this node sends, not on the state it
                    # keeps.
                    hidden = hidden + layer_input
"""))

P.append(("models/fusion_reasoner.py", r"""
            _dims = list(_em.dims)[:-1]
""", r"""
            # RNG STATE SAVED AND RESTORED AROUND THIS BLOCK. Constructing these modules
            # draws from the global generator, so every parameter initialised AFTER this
            # point -- the fusion gate, the router, the operator scalars -- would start
            # from different values in the CCMP arm than in the control. The pair would
            # then differ in the loss AND in the initialisation, and the control's own
            # reruns already move 0.0118 nDCG@5.
            _rng = torch.get_rng_state()
            _dims = list(_em.dims)[:-1]
"""))

P.append(("models/fusion_reasoner.py", r"""
            _em.resp_eta = float(os.environ.get("CCMP_ETA", "0.1"))
            print(f"[crp] responsibility head on {len(_dims)} layers, hid={_h}, "
                  f"gate={_em.resp_gate} mean_norm={_em.resp_gate_norm} "
                  f"delta={_em.resp_eta}", flush=True)
""", r"""
            _em.resp_eta = float(os.environ.get("CCMP_ETA", "0.5"))
            torch.set_rng_state(_rng)
            print(f"[ccmp] responsibility head on {len(_dims)} layers, hid={_h}, "
                  f"gate={_em.resp_gate} mean_norm={_em.resp_gate_norm} "
                  f"eta={_em.resp_eta}", flush=True)
"""))

P.append(("models/fusion_reasoner.py", r"""
        g = self.base(graph, batch, entities_weight)            # [B, N] per-node scores
""", r"""
        # The frontier A^(l) is expanded inside bellmanford, which sees the edge index but
        # not the batch. Handed over here rather than threaded through GraphReasoner's
        # signature, which every other reasoner shares.
        if getattr(self.base.entity_model, "resp_proj", None) is not None:
            self.base.entity_model._ccmp_seeds = batch.get("start_nodes_mask")
        g = self.base(graph, batch, entities_weight)            # [B, N] per-node scores
"""))

P.append(("trainers/base_trainer.py", r"""
                # Update progress bar
                progress_bar.set_postfix(loss=step_metrics.get("loss", 0.0))

        # Log epoch averages
        if utils.get_rank() == 0:
            epoch_avg_metrics = {
                f"epoch_{k}": np.mean(v) for k, v in epoch_metrics.items()
            }
            epoch_avg_metrics["epoch"] = self.state["epoch"]

""", r"""
                # Update progress bar. EVERY term, not just the total: the total is a
                # weighted sum of three or four objectives and a change in it says nothing
                # about which one moved. `ccmp_pos` in particular is the collapse detector
                # -- if the gold-side share of supervised nodes goes to zero the target has
                # degenerated and the run is already dead, which is worth seeing at step 50
                # rather than after four epochs.
                progress_bar.set_postfix(
                    **{k: round(float(v), 4)
                       for k, v in step_metrics.items()
                       if k in ("loss", "hn_fused", "hn_graph", "ccmp",
                                "ccmp_conf", "ccmp_pos", "pop_anchor")})

        # Log epoch averages
        if utils.get_rank() == 0:
            epoch_avg_metrics = {
                f"epoch_{k}": np.mean(v) for k, v in epoch_metrics.items()
            }
            epoch_avg_metrics["epoch"] = self.state["epoch"]

"""))

P.append(("trainers/base_trainer.py", r"""
            self._log_metrics(epoch_avg_metrics, prefix="train")
""", r"""
            self._log_metrics(epoch_avg_metrics, prefix="train")
            # ONE LINE PER EPOCH, EVERY COMPONENT. _log_metrics goes to wandb, which the
            # runs disable, so without this the per-term averages are computed and thrown
            # away and only the total reaches stdout.
            _ord = ["loss", "hn_fused", "hn_graph", "ccmp", "aux_graph", "pop_anchor",
                    "ccmp_conf", "ccmp_pos", "ccmp_nodes", "resid_cover", "resid_negs"]
            _have = [k for k in _ord if k in epoch_metrics] + \
                    [k for k in sorted(epoch_metrics) if k not in _ord]
            print("[loss] epoch %d  " % self.state["epoch"]
                  + "  ".join("%s %.4f" % (k, np.mean(epoch_metrics[k])) for k in _have),
                  flush=True)
"""))

P.append(("trainers/fusion_trainer.py", r"""
            key = (gkey, tuple(gp.tolist()), int(seeds[b].sum()))
""", r"""
            # KEYED ON THE QUERY ID. Golds + seed COUNT is not a query identity: two
            # queries sharing golds and seed count would collide, and the negatives are
            # the SEMANTIC scorer's top-K, which differ per query, so the second query
            # would silently train against the first one's distractors.
            key = (gkey, str(batch["id"][b]))
"""))

P.append(("trainers/fusion_trainer.py", r"""
                        step_metrics.update(cstats)
""", r"""
                        step_metrics.update(cstats)
                        _gs = getattr(self.model.base.entity_model,
                                      "_resp_gstat", None)
                        if _gs:
                            step_metrics["ccmp_gmax"] = max(x[1] for x in _gs)
                            step_metrics["ccmp_gp95"] = max(x[2] for x in _gs)
"""))

P.append(("models/ultra/models.py", r"""
            _rp = []
            _heads = getattr(self, "resp_proj", None)
""", r"""
            _rp = []
            _rp_stat = []
            _heads = getattr(self, "resp_proj", None)
"""))

P.append(("models/ultra/models.py", r"""
            self._resp_pred = _rp
""", r"""
            self._resp_pred = _rp
            self._resp_gstat = _rp_stat
"""))

P.append(("models/fusion_reasoner.py", r"""
        heads = [("gate", self.gate)]
""", r"""
        heads = [("gate", self.gate)]
        # THE CCMP HEAD SITS ON THE SAME CLIFF AS GAMMA. The resp_head trunk initialises
        # at +-0.125 against a 256*lr = 0.128 freeze line, so roughly half its weights are
        # one small drift away from never updating again -- the exact failure that decided
        # every CQIG arm. Empty in the control, where these attributes are None.
        _rm = self.base.entity_model
        for _t in ("resp_proj", "resp_emb", "resp_head"):
            if getattr(_rm, _t, None) is not None:
                heads.append((_t, getattr(_rm, _t)))
"""))

P.append(("models/ultra/models.py", r"""
                    _z = _heads[_li](layer_input) + self.resp_emb.weight[_li]
                    _yh = torch.sigmoid(self.resp_head(_z).squeeze(-1))   # [B, N]
""", r"""
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
"""))

s, dup = {}, []
for f, o, n in P:
    o, n = o.strip("\n"), n.strip("\n")
    s.setdefault(f, open(os.path.join(R, f)).read())
    c = s[f].count(n)
    if c > 1:
        dup.append((f, o[:40])); continue
    if c == 1:
        print("already applied :", f, repr(o[:40])); continue
    assert s[f].count(o) == 1, "anchor x%d in %s: %r" % (s[f].count(o), f, o[:70])
    s[f] = s[f].replace(o, n, 1); print("patched         :", f, repr(o[:40]))
assert not dup, ("DUPLICATED by an earlier run: %r\nRe-run Section 3, then this cell." % dup)
for f, t in s.items():
    ast.parse(t); open(os.path.join(R, f), "w").write(t)
print()
_bad = 0
for k, f in [('reaches[l]', 'trainers/fusion_trainer.py'),
             ('ccmp_pos', 'trainers/fusion_trainer.py'),
             ('str(batch["id"][b])', 'trainers/fusion_trainer.py'),
             ('ccmp_gmax', 'trainers/fusion_trainer.py'),
             ('_rp_stat', 'models/ultra/models.py'),
             ('_resp_gstat', 'models/ultra/models.py'),
             ('torch.ones_like(_gt)', 'models/ultra/models.py'),
             ('_msg', 'models/ultra/models.py'),
             ('not sufficient on its own', 'models/ultra/models.py'),
             ('_ccmp_seeds', 'models/fusion_reasoner.py'),
             ('set_rng_state', 'models/fusion_reasoner.py'),
             ('SAME CLIFF AS GAMMA', 'models/fusion_reasoner.py'),
             ('[loss] epoch', 'trainers/base_trainer.py')]:
    ok = k in open(os.path.join(R, f)).read()
    _bad += (not ok); print(("OK  " if ok else "MISSING"), k, "in", f)
assert not _bad, "%d marker(s) missing" % _bad
try:
    import inspect, re
    _s = inspect.getsource(run_model)
    if 'env.pop("CCMP"' in _s:
        print("\nrun_model already clears CCMP_*")
    else:
        exec(re.sub(r"(\n    if ccmp:\n)",
             '\n    for _k in ("CCMP","CCMP_W","CCMP_NEG","CCMP_M","CCMP_GATE",'
             '"CCMP_GATE_NORM","CCMP_ETA","CCMP_HID","CCMP_CMIN"):\n'
             '        env.pop(_k, None)\\1', _s, count=1), globals())
        print("\nrun_model now clears CCMP_* unconditionally")
except NameError:
    print("\nrun_model not defined -- run Section 7, then re-run this cell")