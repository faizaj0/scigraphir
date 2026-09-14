#!/usr/bin/env python3
"""
faith_patch.py -- add two path-faithfulness measurements to FusionSFTTrainer.interpret():

  necessity : after the top paths to a gold are found, remove their edges from the graph and
              re-score the query; records the gold's graph-channel and fused rank without the
              top-1 and top-3 paths, plus a control that removes the same number of random edges.
  distractor: run the same gradient beam search towards the top-ranked NON-gold document of
              the query, so the CCMP gate along gold routes can be compared with the gate along
              the routes to the strongest wrong answer.

Idempotent. Run as  python faith_patch.py <gfmrag root>  (locally) or exec the same body in a
Colab cell with ROOT = "/content/gfm-rag/gfmrag". Requires the pinned-golds and min_hops patches
to be present already (both are in the current trainer).
"""
import sys


HELPERS = '''    def _rank_without(self, graph, batch, triples, node, n_random=0):
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

    def _interpret_target(self, graph, batch, node, em, eta, id2node, id2rel):
        """Gradient beam search from the query's seeds to one node; returns (paths with the CCMP
        gate per hop, frontier-mean responsibility per layer, raw (h, t, r) paths)."""
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
        return paths_out, fm, paths

'''

EDITS = [
    # signature
    ("max_golds=2, top_views=3, do_paths=True, golds=None):",
     "max_golds=2, top_views=3, do_paths=True, golds=None, necessity=False, distractor=False):"),
    # necessity: after the gold's paths are built, before the record is appended
    ('''                    rec["targets"].append({"doc": gname, "rank": {k: ranks[k][gname] for k in ranks},
                                           "dense_cos": float(ch["dense"][j]), "views": views, "min_hops": min_hops.get(int(did[j])),
                                           "frontier_mean_resp": fm, "paths": paths_out})
''',
     '''                    nec = None
                    if necessity and paths:
                        nec = {}
                        for kk in (1, 3):
                            nec[f"top{kk}"] = self._rank_without(graph, batch, [x for p_ in paths[:kk] for x in p_], int(did[j]))
                        nec["random"] = self._rank_without(graph, batch, None, int(did[j]), n_random=nec["top3"]["removed"])
                    rec["targets"].append({"doc": gname, "rank": {k: ranks[k][gname] for k in ranks},
                                           "dense_cos": float(ch["dense"][j]), "views": views, "min_hops": min_hops.get(int(did[j])),
                                           "frontier_mean_resp": fm, "paths": paths_out, "necessity": nec})
'''),
    # distractor: once per query, after the golds
    ('''                if not do_paths and len(results) % 25 == 0:
''',
     '''                if do_paths and distractor:
                    order = torch.argsort(ch["graph"], descending=True).tolist(); gset = set(gold_pos.tolist())
                    jd = next((o for o in order if o not in gset), None)
                    if jd is not None:
                        dp, dfm, draw = self._interpret_target(graph, batch, int(did[jd]), em, eta, id2node, id2rel)
                        rec["distractor"] = {"doc": id2node[int(did[jd])], "rank": {k: int((v > v[jd]).sum().item()) + 1 for k, v in ch.items()},
                                             "frontier_mean_resp": dfm, "paths": dp}
                        if necessity and draw:
                            rec["distractor"]["necessity"] = {"top1": self._rank_without(graph, batch, [x for p_ in draw[:1] for x in p_], int(did[jd]))}
                if not do_paths and len(results) % 25 == 0:
'''),
    # helpers
    ("    def _min_hops(self, graph, seeds, targets, max_hops):",
     HELPERS + "    def _min_hops(self, graph, seeds, targets, max_hops):"),
]

WF_EDIT = ('do_paths=bool(int(cfg.interp.get("paths", 1))), golds=golds)',
           'do_paths=bool(int(cfg.interp.get("paths", 1))), golds=golds,\n'
           '                      necessity=bool(int(cfg.interp.get("necessity", 0))), distractor=bool(int(cfg.interp.get("distractor", 0))))')


def apply(root):
    tf, wf = f"{root}/trainers/fusion_trainer.py", f"{root}/workflow/interpret_paths.py"
    t = open(tf).read()
    if "def _rank_without(" in t:
        print("trainer already patched")
    else:
        for old, new in EDITS:
            assert t.count(old) == 1, f"anchor not found exactly once: {old[:70]!r}"
            t = t.replace(old, new, 1)
        open(tf, "w").write(t)
        print("trainer patched: necessity + distractor")
    w = open(wf).read()
    if "necessity=" in w:
        print("workflow already patched")
    else:
        assert w.count(WF_EDIT[0]) == 1, "workflow anchor not found"
        w = w.replace(WF_EDIT[0], WF_EDIT[1], 1)
        open(wf, "w").write(w)
        print("workflow patched")
    t = open(tf).read(); w = open(wf).read()
    assert "def _rank_without(" in t and "def _interpret_target(" in t and '"necessity": nec' in t and 'rec["distractor"]' in t
    assert "necessity=bool" in w
    compile(t, tf, "exec"); compile(w, wf, "exec")
    print("ok")


if __name__ == "__main__":
    apply(sys.argv[1] if len(sys.argv) > 1 else "/content/gfm-rag/gfmrag")
