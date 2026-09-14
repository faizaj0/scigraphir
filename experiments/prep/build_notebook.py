"""
build_notebook.py -- generate the SIR-4 CS Colab training notebook.

Reuses the engine-install, fusion-model and Qwen3-fetch cells VERBATIM from
colab_train_v16sc_fusion_greasoner.ipynb, so the model code and the adapted
G-Reasoner fork cannot drift between the TOMATO run and this one. Everything
else is written fresh for SIR-4.

ISOLATION IS ENFORCED, NOT ASSUMED. This session lost several hours to caches
keyed by split rather than by corpus, which silently loaded TOMATO data into a
SIR-4 run and reported success. The generated notebook therefore:

  * reads `gfm-rag-adapted.zip` and `qwen3-embedding-0.6b/` from Drive and
    NOTHING else that TOMATO uses;
  * writes only under {DRIVE}/sir4_cs/ and {DRIVE}/outputs/sir4_cs/;
  * asserts, before training, that no dataset name contains "tomato" and that
    the output directory does not already hold a TOMATO run;
  * prints every path it will read and write, and stops if any output path
    would land inside a directory a previous run owns.

    python3 prep/build_notebook.py
"""
from __future__ import annotations

import json
import os

import argparse
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CARGO = os.path.dirname(ROOT)
SRC = f"{CARGO}/retriever/train/colab_train_v16sc_fusion_greasoner.ipynb"

# One notebook per domain. Everything downstream keys off DOMAIN/DATASET, so two
# domains can run in two Colab sessions at once without sharing a single path.
sys.path.insert(0, HERE)
from stage_sir4 import DOMAINS                                    # noqa: E402
_ap = argparse.ArgumentParser()
_ap.add_argument("--domain", default="cs", choices=sorted(DOMAINS))
# THE FIFTH DATASET, AND THE ONLY ONE WITH PUBLISHED BASELINES. Everything the SIR-4
# notebooks do is corpus-agnostic once DATASET is bound, so TOMATO is a flag rather than
# a second generator -- a second generator is exactly how the fusion implementation came
# to be a stale frozen copy once already. What actually differs is stated in the emitted
# notebook: 1 gold per row (so --loss operator and fixed are arithmetically identical),
# the same question text repeating across rows (3,132 rows over 1,658 questions), and no
# sets.json, so CompleteSet@k is undefined and dropped rather than reported as zeros.
_ap.add_argument("--tomato", action="store_true",
                 help="build for TOMATO-Star (tomato_{train,test}_v16sc) instead of a "
                      "SIR-4 domain. Inverts the isolation guard: the foreign corpus to "
                      "keep out becomes sir4, not tomato.")
_ap.add_argument("--batch", type=int, default=None,
                 help="train batch size. Default 1 on SIR-4 and 2 on --tomato. Batch is NOT "
                      "just gradient noise at a fixed epoch count: it sets the "
                      "optimizer-step count, so every arm being compared must share one. "
                      "TOMATO at 4 (what the historical run used, before CCMP and the "
                      "learned scorer existed) OOMs on an 80GB A100 at 76.97 GiB in use.")
# TWO NOTEBOOKS, ONE GENERATOR. A second generator file is how the fusion code came
# to be a stale frozen copy in the first place; a flag cannot drift from itself.
_ap.add_argument("--variant", default="fusion", choices=["fusion", "belief"],
                 help="fusion = operator/multi-view graph fusion; "
                      "belief = H3 semantic-prior graph reasoning, on its own")
# ONE-ARM VARIANT. The full notebook trains six models; re-running it to change a single
# scalar would repeat five finished experiments and overwrite their directories. With
# this flag every training cell that has already run is emitted COMMENTED OUT, with its
# variable bound to the Drive directory it produced, so section 10 still scores every arm
# and exactly one new model trains.
_ap.add_argument("--cqig-lam", type=float, default=None, metavar="LAM",
                 help="build the single-arm CQIG lambda variant at this lam_init "
                      "(e.g. 0.9). Omit for the full notebook.")
# THE OPERATOR ARM. `--cqig-lam 0.9` alone builds the two lambda/linker runs; adding
# `--cqig-op centre` deadens those two as well and leaves exactly ONE live training cell,
# so the third arm can be added to a finished pair without repeating either of them.
_ap.add_argument("--cqig-op", default=None, choices=["centre", "centre-fixed"],
                 help="build the CQIG operator arm on top of an already-run --cqig-lam "
                      "pair. centre = h - (1-g)*mu; centre-fixed = h - lam*mu.")
# THE ABLATION THAT DECIDES WHETHER THE REFERENCE BANK DOES ANYTHING. Combines with
# --cqig-op, adding a SECOND live cell rather than modifying the first: mu=0 under a
# centring op is a literal no-op, so the ablation only makes sense against op=gate.
_ap.add_argument("--cqig-mu-zero", action="store_true",
                 help="add the mu=0 control arm (always op=gate) alongside whatever else "
                      "this invocation builds. Needs --cqig-lam.")
# ONE ARM, BOTH FIXES. matsci separated lambda from the linker and found lambda is the whole
# gain; a domain being checked for replication does not need to pay for that separation
# again, so this drops the exact-linking arm and runs the semantic one alone.
_ap.add_argument("--cqig-link-only", action="store_true",
                 help="build ONLY the semantic-linking arm at --cqig-lam, dropping the "
                      "exact-linking arm. Both fixes move at once, so read it against 9b.")
# THE FUSION FORM. Everything above varies the GATE; this varies how the gated graph
# channel is combined with the semantic one, which is a different axis and the one the
# measured ceiling sits on. Deadens every gate arm, so exactly one new model trains and
# section 10 still scores the finished ones.
_ap.add_argument("--miss-weighted-graph", action="store_true",
                 help="build ONLY the semantic-error-weighted graph-loss ablation "
                      "(MISS_W_AUX=1): weight each gold's graph-alone contrastive term by "
                      "w=min(cap, log1p(r_sem)), normalised by sum(w). Control + two caps, "
                      "all with gamma PINNED so the fused column is readable.")
# The graph channel sits AT its own parameter-free prior (walk 0.245/0.273 nDCG@5 vs a
# trained 34M-parameter GNN's 0.213-0.273), because the standard graph-alone contrastive is
# nearly satisfied by personalised PageRank from the same seeds. This arm restricts each
# gold's negatives to the documents the prior already ranks above it, so the only way to
# lower the loss is with capacity the walk does not have.
_ap.add_argument("--resid-prior", action="store_true",
                 help="build ONLY the residual-prior graph-loss arm (RESID_PRIOR=1): "
                      "control + residual, gamma PINNED in both, plus the parameter-free "
                      "PPR baseline scored as its own retrieval arm.")
_ap.add_argument("--no-cqig-arm", action="store_true",
                 help="in a --ccmp build, omit the third arm (CQIG alone at lam 0.9, exact "
                      "links, against the same control). The arm is included by default so "
                      "the two hypotheses share one control instead of each getting its own.")
_ap.add_argument("--ccmp", action="store_true",
                 help="CONTRASTIVE RESPONSIBILITY PROPAGATION. Two arms: a control, and a "
                      "run whose intermediate nodes are supervised by whether their "
                      "remaining bounded-hop paths lead more strongly to a gold than to "
                      "the semantic scorer's own hard negatives, with the predicted "
                      "responsibility gating outgoing messages. Drops the ladder.")
_ap.add_argument("--fusion-mixture", action="store_true",
                 help="build ONLY the mixture-fusion arm at --cqig-lam: "
                      "log((1-a_q) p_s + a_q p_g) instead of z(s) + gamma*relu(z(g)). "
                      "Read it against the additive arm at the same lambda.")
_ap.add_argument("--fusion-amax", type=float, default=0.05, metavar="A",
                 help="cap on the mixing weight a_q (default 0.05). The graph's confident "
                      "MISTAKES are promoted as hard as its rescues, and this is the only "
                      "thing bounding that. 0.5 was the first default and it FAILED on "
                      "matsci: the router ran a_q to 0.371 and fused lost 0.091 nDCG@5.")
_ap.add_argument("--fusion-afix", type=float, default=None, metavar="A",
                 help="replace the router with this CONSTANT a. The matsci run showed the "
                      "router does not route (a_q spanned only 0.31-0.44 over 331 queries) "
                      "and calibrates to train-time graph quality, so a constant removes a "
                      "confound and makes the arm a clean one-parameter sweep.")
_ap.add_argument("--fusion-topk", type=int, default=0, metavar="K",
                 help="confine the graph channel to the semantic top-K (0 = whole corpus, "
                      "the current behaviour). This is the property the ADDITIVE form has "
                      "for free and the mixture does not: a 0.36 z-unit push cannot lift a "
                      "document out of the tail, whereas the mixture scores everything with "
                      "a*p_g > p_s purely by graph order, as a block. Since the graph's own "
                      "top-5 is worth 0.245 against the semantic channel's 0.517, that block "
                      "reaching rank 5 is a losing trade by construction.")
_ap.add_argument("--fusion-gammafix", type=float, default=None, metavar="G",
                 help="pin the ADDITIVE arm's gamma to this constant. Use 0.0359 to test "
                      "whether the 0.5340 result reproduces: that arm's gamma was bf16-FROZEN "
                      "there, the fp32-learnable rerun froze at 0.0288 and scored 0.5218, and "
                      "0.012 nDCG@5 on n=331 is inside run-to-run noise. Nothing should be "
                      "tuned toward 0.5340 until this says it is real.")
_ap.add_argument("--run-set", default="", metavar="NAME",
                 help="write into outputs/<NAME>/<dataset>/ instead of outputs/<dataset>/, "
                      "and emit ONLY the two ladder rows (graph+gate-off, graph+lam) as live "
                      "cells. A fresh namespace with no history: no already_run, no MISSING "
                      "lines, no chance of a new result landing next to a superseded one.")
_ap.add_argument("--noent", action="store_true",
                 help="TOMATO only: drop the `entity` seed channel, so query seeding uses the "
                      "same four role channels SIR-4 uses. SIR-4 is already built with "
                      "--no_entity_seeds, so this is a no-op there and is rejected. On TOMATO "
                      "the channel is 53.6%% of all seeds (76.9 -> 35.7 per query) and is raw "
                      "term matching, which is the one construction difference between the two "
                      "corpora. Emits a cell that strips the channel from the UNPACKED graph "
                      "rather than shipping a second dataset: nodes/edges/relations are "
                      "byte-identical either way, so a rebuild would spend an hour reproducing "
                      "files it cannot change.")
_a = _ap.parse_args()
RUN_SET = _a.run_set.strip("/")
# TWO DIFFERENT THINGS THAT USED TO SHARE ONE NAME. `RUN_SET` is a BUILD MODE: non-empty
# means "emit only the ladder rows, drop every ablation cell". `NB_RUN_SET` is only the
# output NAMESPACE stamped into the notebook. They coincide on SIR-4, which is why one
# variable served both, but a TOMATO build wants a private namespace WITHOUT ladder mode,
# so it needs the namespace on its own.
#
# Why TOMATO needs one: with an empty namespace the run writes into outputs/tomato/, where
# any earlier TOMATO work also lands, and a rerun that reuses a suffix replaces a finished
# arm in place. That has already cost one result in this project.
# A --ccmp build needs the same thing TOMATO needs: a private namespace WITHOUT ladder mode.
# It cannot set RUN_SET, because non-empty RUN_SET drops every non-ladder cell, which is all
# of section 9c. With an empty namespace these runs land in outputs/sir4_<domain>/ while
# sir4_all_report.ipynb reads outputs/ladder/sir4_<domain>/, so the report found none of them.
NB_RUN_SET = RUN_SET or (("tomato_ablations_noent" if _a.noent else "tomato_ablations_v1")
                         if _a.tomato else "ccmp_control" if _a.ccmp else "")
# the two rows of the arm table that a ladder notebook trains. Everything else is an
# ablation and is simply absent from a --run-set build.
LADDER_VARS = {"RUN_DIR_MLP", "RUN_DIR_CQIG_LAM",      # additive: gate off, gate lam
               "RUN_DIR_MIX", "RUN_DIR_MIX_LAM"}       # mixture:  the same two, trained
TOMATO = _a.tomato
NOENT = _a.noent
DOMAIN = _a.domain
# DSET is what "sir4_cs" is rewritten to at the bottom of this file, and TRAIN/TEST are
# derived from it inside the notebook as f"{DATASET}_{split}_v16sc". Binding it to
# "tomato" therefore points every path, cache key, bundle name and run directory at
# tomato_{train,test}_v16sc without touching anything in between.
DSET = "tomato" if TOMATO else f"sir4_{DOMAIN}"
# TOMATO has no sets.json, so there is no sets directory to substitute. The token is left
# unmatched rather than mapped to a lookalike: a wrong sets dir would silently score
# CompleteSet@k against another corpus's sets instead of reporting the metric as absent.
SETS_DIR = "" if TOMATO else DOMAINS[DOMAIN][1]   # e.g. cs_test_final / physics_test_low
# The corpus this build must NOT touch. On SIR-4 that is TOMATO (caches keyed by split
# rather than corpus contaminated a run during development); on a TOMATO build the same
# guard has to point the other way or it fires on the dataset being built.
FOREIGN = "sir4" if TOMATO else "tomato"
VARIANT = _a.variant
BELIEF = VARIANT == "belief"
LAM = _a.cqig_lam
# SINGLE-ARM MODE EXISTS TO AVOID REPEATING FINISHED RUNS. On a SIR-4 domain the full
# notebook has usually already been run, so --cqig-lam deadens those cells and leaves one
# live arm. TOMATO has no finished runs to protect, so the same flag there means "build the
# full notebook WITH the gate at this lambda" and everything stays live in ONE notebook.
# The default lam=0.1 caps the gate at a 5% modulation, which has measured a no-op in every
# arm ever run at it, so a TOMATO CQIG arm that did not take a lambda would be dead on
# arrival.
LAM_ONLY = LAM is not None and not TOMATO
OP = _a.cqig_op
OP_ONLY = OP is not None
MU0 = _a.cqig_mu_zero
LINK_ONLY = _a.cqig_link_only
MIX = _a.fusion_mixture
MISSW = _a.miss_weighted_graph
RESIDP = _a.resid_prior
CCMP = _a.ccmp
# The gsweep optimum on matsci, and where every bf16 arm happened to freeze. Pinned across
# all three arms so a change in the loss is not read through a gamma that runs to 0.26.
MISSW_GFIX = 0.0359
AMAX = _a.fusion_amax
AFIX = _a.fusion_afix
TOPK = _a.fusion_topk
GFIX = _a.fusion_gammafix
assert TOPK >= 0, "--fusion-topk is a candidate-set size; 0 means unrestricted"
assert not (TOPK and not MIX), "--fusion-topk only applies to --fusion-mixture"
assert GFIX is None or not MIX, (
    "--fusion-gammafix pins the ADDITIVE gamma; it does nothing under --fusion-mixture")
assert not (LAM_ONLY and BELIEF), "--cqig-lam applies to the fusion variant; belief has no gate"
# The single-arm flags all mean "add ONE live cell to a notebook whose other arms have
# already been run". A TOMATO build has no already-run arms and emits one notebook with
# every arm live, so combining them would silently deaden most of what was asked for.
assert not (TOMATO and (OP_ONLY or MU0 or LINK_ONLY or MIX or MISSW or RESIDP or CCMP
                        or RUN_SET)), (
    "--tomato builds ONE notebook with every arm live. The single-arm flags "
    "(--cqig-op/--cqig-mu-zero/--cqig-link-only/--fusion-mixture/--miss-weighted-graph/"
    "--resid-prior/--ccmp/--run-set) add one cell to an already-run set, which TOMATO "
    "does not have. Use --tomato with at most --cqig-lam.")
assert not (NOENT and not TOMATO), (
    "--noent is TOMATO-only. run_domain.py already passes --no_entity_seeds to every SIR-4 "
    "graph build, so a SIR-4 notebook has no entity channel to strip and this flag would "
    "silently produce a notebook identical to the default one.")
assert not LAM_ONLY or 0.0 < LAM < 1.0, "lam is sigmoid-parameterised, so 0 < lam < 1"
assert not OP_ONLY or LAM_ONLY, (
    "--cqig-op needs --cqig-lam: the operator arm is defined AT a lambda, and its control "
    "is the semantic-linking run at that same lambda")
assert not MU0 or LAM_ONLY, (
    "--cqig-mu-zero needs --cqig-lam: the ablation has to sit at the lambda whose result "
    "it is testing, or it measures a different arm's floor")
assert not LINK_ONLY or LAM_ONLY, "--cqig-link-only needs --cqig-lam"
assert not (LINK_ONLY and (OP_ONLY or MU0)), (
    "--cqig-link-only drops the arms that --cqig-op and --cqig-mu-zero build ON TOP of; "
    "run the link arm first, then add those against it")
assert not MIX or LAM_ONLY, (
    "--fusion-mixture needs --cqig-lam: the fusion form is varied AT a lambda, against "
    "the additive arm at that same lambda, or the pair differs in two things at once")
assert not MIX or not (OP_ONLY or MU0 or LINK_ONLY), (
    "--fusion-mixture changes the FUSION, the others change the GATE. Building both in "
    "one notebook makes a two-variable arm whose result attributes to neither.")
assert not MIX or 0.0 < AMAX <= 1.0, "--fusion-amax is a mixing weight cap in (0, 1]"
assert not RESIDP or LAM_ONLY, (
    "--resid-prior needs --cqig-lam: the loss is varied AT the best known gate setting, "
    "against its own control at that same setting, or the pair differs in two things")
assert not (RESIDP and (MISSW or MIX or OP_ONLY or MU0 or LINK_ONLY)), (
    "--resid-prior rewrites the graph-alone loss; combining it with another single-knob "
    "ablation makes a two-variable arm that attributes to neither")
assert not (CCMP and (RESIDP or MISSW or MIX or OP_ONLY or MU0 or LINK_ONLY)), (
    "--ccmp changes what the graph is trained on; combining it with another "
    "single-knob ablation makes a two-variable arm that attributes to neither")
# Any of these alone is enough to deaden the two lambda arms: they all build on top.
EXTRA = OP_ONLY or MU0 or MIX
# 0.9 -> "09", 0.5 -> "05". Used in the run-directory suffix, so it has to be filename-safe
# AND stable: two lambdas must never collide onto one directory.
# Keyed off LAM, not LAM_ONLY: on a TOMATO build lam rides on the main CQIG arm without
# single-arm mode, and the slug is what keeps two lambdas from landing in one run dir.
# Identical for SIR-4, where LAM_ONLY is true exactly when LAM is set.
LAM_SLUG = ("%g" % LAM).replace(".", "").replace("-", "") if LAM is not None else ""
# Distinct per operator, so the adaptive and the fixed arm cannot land in one directory.
# NOT `OP_SLUG`: that name is already taken inside the notebook for the operator ENCODER's
# filename suffix, and two different meanings under one name is how the components files
# came to be loaded from the wrong place once already.
OP_SUFFIX = {"centre": "ctr", "centre-fixed": "ctf"}.get(OP, "")
CTR_SLUG = f"{OP_SUFFIX}{LAM_SLUG}" if OP_ONLY else ""

# THE THIRD ARM IN A --ccmp BUILD: CQIG on its own, against the same control.
#   control  ccmp off, cqig off
#   CCMP     ccmp on,  cqig off
#   CQIG     ccmp off, cqig on
# Never both on. CCMP rescales a layer input and CQIG gates on a squared distance from a
# calibrated mean of that same input, so an arm with both on cannot attribute its difference
# to either. Sharing one control is what makes the two arms comparable to each other as well
# as to the baseline: a second notebook would train a second control and the two hypotheses
# would then be measured through different baselines.
#
# SUPPRESSED when --cqig-lam is given, because that build already turns CQIG on in the
# control, so a "CQIG only" arm would be the control with a different name.
CQIG_ARM = CCMP and not LAM_ONLY and not _a.no_cqig_arm
# 0.9, not the 0.1 default. lam sets the gate's span (span = lam/2), and every arm before the
# sweep measured a no-op at 5%. 0.9 is the value the sweep moved on.
CQIG_ARM_LAM = 0.9
# a_max is IN the directory name. It is the one knob that changes what the arm risks, so
# two caps must never overwrite each other's run: 0.5 -> "mix05lam09". top-K is in it for
# the same reason -- it changes the operator, not just its strength -- and because the
# a_max=0.3 run already wrote into the a_max=0.5 directory once.
TOPK_SLUG = f"k{TOPK}" if TOPK else ""
MIX_SLUG = (f"mix{('%g' % AMAX).replace('.', '').replace('-', '')}{TOPK_SLUG}{LAM_SLUG}"
            if MIX else "")
# A SEPARATE FILE, not an overwrite. The full notebook is the record of the finished
# experiments; regenerating over it would lose it, and this tree is not under git.
# NAMED BY ROLE, NOT BY FLAGS. The old scheme spelled the whole configuration into the
# filename -- colab_train_sir4_matsci_fusion_cqiglam09_mix03k200.ipynb -- which mixed the
# variant, the gate setting and three fusion knobs into one string and still did not say
# which row of the table it ran. The settings now live in each run's arm.json, so the
# filename only has to answer "what is this notebook FOR":
#
#   sir4_<domain>_ladder.ipynb        rows 2 and 3 of the arm table. Both fusion forms come
#                                     off the same eval via [sweep]/[gsweep], so one file
#                                     per domain covers 2A/2B/3A/3B.
#   sir4_<domain>_abl_<what>.ipynb    a single-knob ablation. Not a row; says so in the name.
#
# `colab_` is dropped: everything here is a Colab notebook, so the prefix sorted 20 files
# under one letter and carried no information.
_ABL = ("link" if LINK_ONLY else "") or (OP_SUFFIX if OP_ONLY else "") \
    or ("missw" if MISSW else "") \
    or ("residprior" if RESIDP else "") \
    or ("ccmp" if CCMP else "") \
    or ("muz" if MU0 else "") \
    or ((f"mixtrained{('%g' % AMAX).replace('.', '')}{TOPK_SLUG}") if MIX else "") \
    or ("" if LAM_ONLY else "nogate")
# ONE FILE FOR TOMATO. The _ABL scheme names a notebook by which single arm it adds to an
# already-run set, which is exactly what a TOMATO build is not: every arm is live in one
# notebook, so naming it "abl_nogate" would say the opposite of what it contains.
_ABL = "" if TOMATO else _ABL
# READABLE NAME FOR THE CCMP BUILD. `abl_ccmp` says which flag was passed, not what the
# notebook contains, and it collides in meaning across bases: four files called
# sir4_*_abl_ccmp.ipynb existed with different fusion routers and one of them silently
# stacked CQIG, which is unreadable from the filename. The name now states the comparison
# and, when something is stacked underneath it, what that is.
_NAME = None
if CCMP and not TOMATO:
    _NAME = "ccmp_vs_control" + (f"_over_cqig_lam{LAM_SLUG}" if LAM_ONLY else "")
# ITS OWN FILE, per the rule at the top of this block. --noent is not a variant of the
# ablations notebook, it is the other half of a one-variable comparison against it, and
# both have to stay readable side by side. Writing over tomato_ablations.ipynb would
# destroy the entity arm's record, which is the control.
if NOENT:
    _NAME = "ablations_noent"
OUT = (f"{ROOT}/notebooks/{DSET}_" + (_NAME if _NAME else
       "ablations" if TOMATO else f"abl_{_ABL}" if _ABL else "ladder")
       + ("" if VARIANT == "fusion" else f"_{VARIANT}") + ".ipynb")
if CCMP and LAM_ONLY:
    print("NOTE: this build stacks CCMP on CQIG. The CCMP-minus-control difference is then\n"
          "      conditional on CQIG, not a measurement of CCMP on the plain reasoner.")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def run_cell(body, var=None, suffix_expr=None, dead=None):
    """A training cell, emitted live in the full notebook and DEAD in the one-arm variant.

    Commenting a run out is not enough on its own: section 10 reads each arm's predictions
    out of `RUN_DIR_*`, so a commented cell would leave the variable undefined and silently
    drop a finished arm from the comparison table. The dead form therefore binds the same
    variable to the directory that run already wrote. Nothing is faked -- `already_run`
    returns None when the predictions are genuinely absent, and section 10 skips the arm
    and says so.

    `dead` overrides the default predicate for cells whose "already run" is a different
    question from the rest: the two lambda arms are live under `--cqig-lam` and dead once
    `--cqig-op` adds a third arm on top of them.
    """
    if dead is None:
        dead = LAM_ONLY
    # A RESIDUAL-PRIOR NOTEBOOK TRAINS ITS OWN TWO ARMS AND NOTHING ELSE. Leaving the
    # ladder's cells in, live or commented, is how a notebook built for one question ends
    # up holding six half-run arms and a table nobody can read. Section 9r supplies its
    # own cells directly, so they survive this.
    if RESIDP or CCMP:
        return None
    # A FRESH RUN SET HAS NO HISTORY, so nothing in it can be "already run". Under
    # --run-set the two ladder rows are both live and every ablation cell is dropped
    # entirely rather than left commented pointing at an empty directory, which is the
    # state that produced four MISSING lines and the impression that runs had vanished.
    if RUN_SET:
        if var in LADDER_VARS:
            return code(body)
        return None
    if not dead or var is None:
        return code(body)
    dead = "\n".join(("# " + ln) if ln.strip() else "#" for ln in body.split("\n"))
    # "ALREADY RUN" was a claim, and it was FALSE the moment the same flags were used
    # to build a second domain: --cqig-link-only was written for matsci, where the lam
    # arm really had run, and carried that sentence verbatim into biology, where it had
    # not. The header now states the condition instead of asserting the fact, and
    # already_run's MISSING line is the only thing that speaks to whether it holds.
    return code(
        f"# NOT RUN BY THIS NOTEBOOK. Expected on Drive from an earlier {DOMAIN} run.\n"
        "# If the line below says MISSING, it never ran FOR THIS DOMAIN: uncomment the\n"
        "# call and run it, or rebuild without the flag that deadened it.\n"
        + dead + "\n"
        + f"{var} = already_run({var!r}, {suffix_expr})")


src = json.load(open(SRC))
reuse = {i: src["cells"][i] for i in (4, 5, 7)}     # engine install, fusion files, Qwen3

cells = []

cells.append(md((
"""# SIR-4 CS — the 2x2 ladder (CQIG on/off x additive/mixture)

Within-domain CS run. Train on `sir4_cs_train_v16sc`, evaluate on
`sir4_cs_test_v16sc`. The title used to say "additive-gate fusion", which was true when
the notebook trained one arm and is wrong now that the fusion form is one of the two
axes being varied.""" if RUN_SET else
"""# SIR-4 CS — additive-gate fusion (operator ⊕ v16sc graph)

Within-domain CS run. Train on `sir4_cs_train_v16sc`, evaluate on
`sir4_cs_test_v16sc`. **Additive gate with the original operator-hard-negative
objective** — not lever D, not loss v2.""") + """

""" + ("""This corpus, measured from the shipped graphs rather than quoted:

| | train | test |
|---|---|---|
| queries | 9,669 | 3,132 |
| golds per query | **1.00** | **1.00** |
| query seeds | 76.85 | 76.96 |
| graph nodes | 106,653 | 47,751 |

**Three things follow from one gold per query, and all three are easy to get wrong:**

- `PER_GOLD=1` is a **no-op here**. It exists so a buried gold cannot free-ride behind an
  easy sibling, and there are no siblings. It is left on so the objective is identical to
  the SIR-4 arms, not because it does anything.
- **CompleteSet@k is undefined** and is dropped from every table below. There is no
  `sets.json` for TOMATO, and the SIR-4 one belongs to another corpus whose query ids
  cannot match. A borrowed sets file would not error, it would silently score zero.
- The same **question text repeats across rows** (3,132 rows over 1,658 distinct
  questions, 1.89 golds each). Scoring per row is the convention every published TOMATO
  number in this project uses; pooling by question roughly doubles the absolutes and
  leaves the ordering unchanged.

**The objective is the STRENGTHENED contrastive, not the original.** `HARDNEG_GRAPH=50`
puts the graph's own top-50 (detached, golds removed) into the lineup, so the loss also
pushes DOWN what the graph over-scores. That was not in the original semantic-hard-negative
objective, and on this corpus it is the only one of the two loss flags that changes
anything.""" if TOMATO else """Differences from the TOMATO run, all deliberate:

| | TOMATO | here |
|---|---|---|
| golds per query | 1 | **4.45** (multi-gold scoring) |
| query seeds | ~66, five sources | ~27, role channels only |
| metrics | MRR, Recall@k | **plus CompleteSet@k** |
| model, objective, hyperparameters | | **unchanged** |""") + """

**Isolation.** Reads only `gfm-rag-adapted.zip` and the cached Qwen3 model from
Drive. Writes only under `sir4_cs/` and `outputs/sir4_cs/`. Cell 2 asserts this
before anything runs."""))

if LAM_ONLY:
    _ROWS = ([f"| **9e-centre** | CQIG, `lam_init={LAM}` + semantic seeding + "
              f"`cqig_op={OP}` | the operator only, vs 9e-link |"] if OP_ONLY else [])
    _ROWS += ([f"| **9e-mu0** | CQIG, `lam_init={LAM}`, `cqig_mu=zero` "
               f"| mu only, vs 9e-lam. **The ablation.** |"] if MU0 else [])
    if not EXTRA:
        _ROWS = [f"| **9e-lam** | CQIG, `lam_init={LAM}` | lambda only, vs `RUN_DIR_CQIG` |",
                 f"| **9e-link** | CQIG, `lam_init={LAM}` + semantic seed linking "
                 f"| the linker only, vs 9e-lam |"]
        if LINK_ONLY:
            # BOTH FIXES AT ONCE. matsci already separated them, so a replication run does
            # not need to; but the row must say so or the delta will be read as lambda's.
            _ROWS = [f"| **9e-link** | CQIG, `lam_init={LAM}` + semantic seed linking "
                     f"| **both** fixes at once, vs `RUN_DIR_MLP` |"]
    _HOWMANY = "ONE MODEL" if len(_ROWS) == 1 else f"{len(_ROWS)} MODELS"
    _DEADSEC = ("sections 8, 9, 9b, 9e, 9e-lam and 9e-link" if EXTRA
                else "sections 8, 9, 9b and 9e")
    _NEWROWS = "\n".join(_ROWS)
    _NEWCOUNT = ("One new model trains, differing" if len(_ROWS) == 1
                 else f"{len(_ROWS)} new models train, each differing")
    # UNDER --run-set THE WHOLE PARAGRAPH ABOVE IS FALSE. There is no history in a fresh
    # namespace, so nothing is commented out, nothing is on Drive, and no variable is bound
    # to a previous run. This is the same failure as the old "ALREADY RUN" header: prose
    # asserting a state the code no longer produces, and it is worse in a header because it
    # is the first thing read. Both branches are generated from the same flags rather than
    # one being written by hand, so they cannot drift apart again.
    if RUN_SET:
        cells.append(md(f"""---

## THIS NOTEBOOK TRAINS 4 MODELS, FROM SCRATCH

Nothing is commented out and nothing is read from a previous run. All four arms below train
here, in this notebook, and write into `outputs/{RUN_SET}/{DSET}/`, which is a namespace of
its own. The exploratory runs under `outputs/{DSET}/` are neither read nor written.

| arm | section | CQIG | fusion | writes |
|---|---|---|---|---|
| **2A** | 9b | off | additive | `{DSET}_fusion_qwenmlp_epoch10_b1` |
| **2B** | 9m | off | mixture | `{DSET}_fusion_qwenmlp_mix_epoch10_b1` |
| **3A** | 9e-lam | `lam_init={LAM}` | additive | `{DSET}_fusion_qwenmlp_cqiglam{LAM_SLUG}_epoch10_b1` |
| **3B** | 9m | `lam_init={LAM}` | mixture | `{DSET}_fusion_qwenmlp_cqiglam{LAM_SLUG}_mix_epoch10_b1` |

Arm 1 (multi-view alone) is the `semantic` column every run prints, so it costs nothing.

**Every cell of the 2x2 is trained.** The fusion form changes the gradient the GNN receives
-- `relu(z(g))` zeroes it for every document below the graph's own mean and the mixture does
not -- so the two forms fit different graph channels, and scoring one form off the other's
checkpoint would compare combiners rather than methods. The `[sweep]`/`[gsweep]` lines still
print as a secondary paired read; they are not what the table reports.

The **Results** section at the end of this notebook reads both runs back off Drive and
prints the table, the trajectories and a paired bootstrap. It trains nothing, so it is safe
to re-run, and it reports MISSING rather than guessing if a run has not finished.

Sections 5b to 5e still run. They are not experiments being repeated: the operator fit, the
BGE baseline and the learned scorer all write under `/content`, which does not survive a
Colab reset, and 5e asserts on a memmap that 5d produces. They are seeded and deterministic.
"""))
    else:
        cells.append(md(f"""---

## THIS NOTEBOOK TRAINS {_HOWMANY}

Everything in {_DEADSEC} is **commented out**. Those runs are finished, their
results are on Drive, and each variable is bound to the directory its run produced so
section 10 still scores every arm. {_NEWCOUNT} from its baseline in
exactly one thing:

| section | arm | changed vs. its baseline |
|---|---|---|
{_NEWROWS}

Sections 5b to 5e still run. They are not experiments being repeated: the operator fit, the
BGE baseline and the learned scorer all write under `/content`, which does not survive a
Colab reset, and 5e asserts on a memmap that 5d produces. They are seeded and deterministic,
so they rebuild the same scorer the finished runs used.
{'''
The two defects below are what 9e-lam and 9e-link were built to test, and both of those runs
are finished. They are kept here because they are what `lam=''' + str(LAM) + '''` and semantic
seeding MEAN, and this arm inherits both — it changes only the operator on top of them. Read
9e-centre for what is new.
''' if EXTRA else ''}
### The two things that are actually wrong

**1. The gate has no authority.** The gate is

$$g = 1-\\lambda\\bigl[1-\\sigma(\\alpha(\\log(1+I)-\\tau))\\bigr]$$

so `g` can never fall below $1-\\lambda$, and with $\\tau$ sitting where calibration puts it
(the median of $\\log(1+I)$ over responding reference pairs) the realised span is about
$\\lambda/2$. At the config default `cqig_lam: 0.1` that is a **5% modulation**, applied to a
graph channel the fusion already weights at $\\gamma\\approx0.03$. The gate could not move a
ranking however well `I` ranked nodes, and $\\lambda$ could not grow out of it either,
because its own gradient carries the same factor. The physics run confirms both halves:
`lam` finished at `9.96e-02` from an init of `0.1`, and the gate span was `4.9e-02`.

**2. The references barely reach the target graph.** Physics resolved its source seeds
`501/501` on the train graph and `25/501` on the **test** graph. Same domain, different
papers, and 95% of the entry points do not exist under those names. $\\mu_v$ at test time was
therefore averaged over reference runs that never entered the graph.

### Why either is worth a run

The statistic itself is not the problem. On physics the
`informativeness AUC, gold vs semantic-top-50` came out at **0.675 / 0.709 / 0.717 / 0.720**
for layers 3-6 over 834 queries, with mean `I` about five times higher at gold papers than at
the hardest negatives -- and that was measured *with* the 5% seeding. Run 1 gives that
statistic authority (`g` spans roughly {1 - LAM:.2f} to 1.00 instead of 0.95 to 1.00). Run 2
gives it a properly estimated baseline to be informative *against*.

**What to read, in order.** The `[cqig] eval gate` line first, to confirm `span` is now about
`{LAM / 2:.2f}` and not `5e-02`. Then the AUC, which should be *unchanged* by lambda -- it
does not touch `I`, so a moved AUC there means something else changed. Then **graph-only
nDCG@5** from the `[diag]` line against the multi-view+graph control, because the gate acts
on the graph channel and the fused number cannot improve before it does. In 9e-link, read the
new `[cqig]   linking:` line on the TEST graph against the 5% above.

---"""))

cells.append(md("## 1. GPU + Drive"))
cells.append(code('''!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import os
from google.colab import drive
drive.mount('/content/drive')

DRIVE   = "/content/drive/MyDrive/cargo-gfmrag"
DATASET = "sir4_cs"
TRAIN, TEST = f"{DATASET}_train_v16sc", f"{DATASET}_test_v16sc"

BUNDLE   = f"{DRIVE}/{DATASET}_bundle.zip"   # the uploaded bundle (one zip)
# RUN_SET is stamped in at build time by --run-set. Empty string keeps the historical flat
# layout; a name puts this notebook's output in its own namespace so a clean set of runs
# cannot be confused with, or overwrite, the exploratory ones underneath.
RUN_SET  = "__RUN_SET__"
OUT_ROOT = f"{DRIVE}/outputs/{RUN_SET + '/' if RUN_SET else ''}{DATASET}"

# The bundle zip MIRRORS the repo, so unpacking it here and pointing SCIGRAPHIR_ROOT
# at it makes every pipeline script resolve exactly as it does on the laptop.
# No files are copied into a second layout, which is one whole class of
# "which copy is stale?" bug that cannot happen.
SCIGRAPHIR_ROOT = "/content/scigraphir"                # local only, never on Drive
os.environ["SCIGRAPHIR_ROOT"] = SCIGRAPHIR_ROOT
os.environ["SCIGRAPHIR_DATASET"] = DATASET

# scigraphir_paths puts built graphs at {SCIGRAPHIR_ROOT}/retriever/data/<graph>,
# and that is precisely the layout the trainer wants for datasets.cfgs.root,
# so DATA_ROOT is not a separate staging area -- it IS the repo data dir.
DATA_ROOT = f"{SCIGRAPHIR_ROOT}/retriever/data"

# Defined HERE, not in the BGE cell. The scoring cell guards on
# os.path.exists(BGE_PRED) so a missing baseline degrades to "similar/dissimilar
# skipped" -- but only if the NAME exists. Defining it in an optional cell made
# that guard unreachable.
S4       = f"{SCIGRAPHIR_ROOT}/experiments"
BGE_PRED = f"{S4}/data/predictions_bge_{DATASET}_test.json"

# THE OPERATOR'S ENCODER. Everything the operator produces -- cached embeddings,
# the fitted w/beta, the components npz -- is that encoder's output, so all three
# are filename-scoped by it. Empty slug for BGE keeps every pre-Qwen path intact;
# switching encoders can therefore never overwrite a calibration that trained
# checkpoints were warm-started from.
OP_MODEL = "/content/qwen3"
import re as _re
OP_SLUG = ("" if OP_MODEL == "BAAI/bge-large-en-v1.5"
           else "_" + _re.sub(r"[^a-z0-9]+", "-", OP_MODEL.lower()).strip("-"))
print("operator encoder", OP_MODEL, "-> slug", OP_SLUG or "(none)")

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    os.environ.setdefault("HF_TOKEN", "")
print("DRIVE     ", DRIVE)
print("reads     ", f"{os.path.basename(BUNDLE)} + gfm-rag-adapted.zip + qwen3-embedding-0.6b/")
print("unpacks to", SCIGRAPHIR_ROOT)
print("writes    ", OUT_ROOT)'''))

cells.append(md("## 2. Isolation guard — run before anything else"))
cells.append(code('''# Fails LOUDLY rather than quietly reusing TOMATO artefacts. Every check here
# corresponds to a real contamination that happened during development: caches
# keyed by split not corpus, a back-compat fallback that fired for any dataset,
# and a graph named for one corpus but built from another.
import os, glob

problems = []

# 1. no dataset name may resemble the foreign corpus
for n in (DATASET, TRAIN, TEST):
    if "__FOREIGN__" in n.lower():
        problems.append(f"dataset name {n!r} contains '__FOREIGN__'")

# 2. the output root must not already contain a foreign run
if os.path.isdir(OUT_ROOT):
    stale = [p for p in os.listdir(OUT_ROOT) if "__FOREIGN__" in p.lower() or "v16sc_fusion" == p]
    if stale:
        problems.append(f"{OUT_ROOT} already holds {stale} — pick a fresh OUT_ROOT")

# 3. the TOMATO output tree must exist and be untouched by us
tom = f"{DRIVE}/outputs/v16sc"
if os.path.isdir(tom):
    print(f"[guard] TOMATO runs at {tom}: {sorted(os.listdir(tom))}")
    print("[guard] this notebook never writes there")

# 4. the bundle must actually be present, and must contain what both phases need
for p in (BUNDLE, f"{DRIVE}/gfm-rag-adapted.zip"):
    if not os.path.exists(p):
        problems.append(f"missing {p}")
if os.path.exists(BUNDLE):
    import zipfile
    names = set(zipfile.ZipFile(BUNDLE).namelist())
    need = [
        f"retriever/data/{DATASET}_train/raw/documents.json",
        f"retriever/data/{DATASET}_test/raw/documents.json",
        f"retriever/data/{TRAIN}/processed/stage1/nodes.csv",
        f"retriever/data/{TEST}/processed/stage1/nodes.csv",
        # DATASET-SCOPED ON SIR-4, UNSCOPED ON TOMATO. scigraphir_paths.is_legacy() is
        # `DATASET == "tomato"`, and _scoped() drops the subdirectory in that case, so
        # TOMATO's probes live at cache/probes_*.jsonl with no tomato/ under it. bundle.py
        # already gets this right (it calls cp.probes_path), so hardcoding the SIR-4 layout
        # here would fail the guard on a bundle that is in fact complete.
        f"retriever/probes/cache/__PROBEDIR__probes_train.jsonl",
        f"retriever/probes/cache/__PROBEDIR__probes_test.jsonl",
        "retriever/eval/operator_scorer.py",
        "experiments/eval/bge_sir4.py",
        "experiments/eval/score_sir4.py",
        "experiments/eval/semantic_scorer.py",
        "scigraphir_paths.py",
    ]
    for p in need:
        if p not in names:
            problems.append(f"bundle is missing {p}")
    # Not fatal: without it CompleteSet@k silently reports 0, so say so now.
    if not any(n.endswith("sets.json") for n in names):
        print("[guard] WARNING: no sets.json in the bundle -> CompleteSet@k will be 0")

assert not problems, "ISOLATION GUARD FAILED:\\n  " + "\\n  ".join(problems)
os.makedirs(OUT_ROOT, exist_ok=True)
print("\\n[guard] OK — isolated from TOMATO")'''))

cells.append(md("## 3. Install the adapted G-Reasoner engine\n"
                "*(verbatim from the TOMATO notebook so the engine cannot drift)*"))
cells.append(reuse[4])
# CHECK THE IMPORT THAT ACTUALLY MATTERS, AND CHECK IT HERE. The engine cell above ends
# with sys.path.insert(0, "/content/gfm-rag") and then imports GraphReasoner, so it prints
# "G-Reasoner OK" whether or not the editable install worked. Training does not run in this
# kernel; it runs in a fresh interpreter that never sees that sys.path. Without this cell
# the first symptom of a failed install is a training cell dying after the whole setup
# section, the bundle unpack and the Qwen3 index have already run.
cells.append(code('# Does a FRESH interpreter -- the one training actually uses -- see gfmrag?\nimport os, sys, subprocess\ndef _sub_import():\n    r = subprocess.run([sys.executable, "-c", "import gfmrag; print(gfmrag.__file__)"],\n                       capture_output=True, text=True, env=dict(os.environ))\n    return r.returncode == 0, (r.stdout.strip() or r.stderr[-1500:])\n\n_ok, _msg = _sub_import()\nif _ok:\n    print("subprocess import OK |", _msg)\nelse:\n    # REPAIR, DO NOT RAISE. The fix is one environment variable, it is exactly what\n    # run_model passes anyway, and raising would strand an otherwise healthy runtime.\n    os.environ["PYTHONPATH"] = os.pathsep.join(\n        ["/content/gfm-rag"] + ([os.environ["PYTHONPATH"]]\n                               if os.environ.get("PYTHONPATH") else []))\n    _ok, _msg = _sub_import()\n    assert _ok, (\n        "gfmrag imports in this kernel but NOT in a subprocess, and PYTHONPATH did not "\n        "fix it. The editable install failed AND /content/gfm-rag is not a usable "\n        "package root. Check that the unzip produced /content/gfm-rag/gfmrag/.\\n" + _msg)\n    print("editable install did not register; PYTHONPATH set instead |", _msg)'))
# THE FUSION IMPLEMENTATION COMES FROM THE REPO, NOT FROM A FROZEN COPY INSIDE THE
# SOURCE NOTEBOOK. Reusing cell 5 verbatim meant every SIR-4 notebook shipped a
# snapshot that had drifted behind retriever/gfm-rag: multi-corpus operator
# tables, the loss-v2 flags and the learned semantic channel all existed in the repo
# and none of them reached Colab, so editing the repo file changed nothing that ran.
# Emitting the cell from the files on disk makes the repo the single source of truth.
FORK = f"{CARGO}/retriever/gfm-rag"
FUSION_FILES = {f"/content/gfm-rag/{rel}": open(f"{FORK}/{rel}").read()
                for rel in ["gfmrag/models/fusion_reasoner.py",
                            "gfmrag/models/cqig.py",
                            # THE CCMP GATE LIVES IN THE VENDORED ULTRA MODEL, not in the
                            # fusion wrapper: the responsibility scaling is applied to
                            # `layer_input` inside QueryNBFNet.bellmanford, because that is
                            # the only place a node's OUTGOING messages can be scaled
                            # before the conv sees them. Without this line the head trains
                            # against the target and nothing gates, so the CCMP arm would
                            # run as a second control and the run would be wasted.
                            "gfmrag/models/ultra/models.py",
                            "gfmrag/trainers/fusion_trainer.py",
                            # base_trainer.py ships for the same reason the others do: the
                            # engine is unzipped from gfm-rag-adapted.zip on Drive, so a fix
                            # made in this repo does not reach Colab until that zip is
                            # rebuilt and re-uploaded. _load_checkpoint called
                            # self.scaler.load_state_dict() unguarded, and self.scaler does
                            # not exist yet on the resume path -- so EVERY eval-from-
                            # checkpoint died with AttributeError and every model_best.pth
                            # on Drive was effectively write-only. Shipping the file makes
                            # the repo the source of truth and needs no zip re-upload.
                            "gfmrag/trainers/base_trainer.py",
                            "gfmrag/workflow/config/gfm_reasoner/sft_training_fusion.yaml"]
                + (["gfmrag/models/belief_reasoner.py",
                    "gfmrag/workflow/config/gfm_reasoner/sft_training_belief.yaml"]
                   if BELIEF else [])}
# The payload is embedded in an r''' ''' literal, so a triple quote inside any of
# these files would terminate it early and emit a notebook that cannot parse.
assert not any("'''" in v for v in FUSION_FILES.values()), "fusion source contains '''"
cells.append(code(
    "# === write the CARGO-fusion files into the fork (generated from the repo copies) ===\n"
    "# Objective = OPERATOR-HARD-NEGATIVE contrastive (report Eq 3.9): negatives are the\n"
    "# semantic scorer's own top hubs, so the graph is forced to fix its misses. Gate\n"
    "# gamma_init=0.01 (opt-in). `model.semantic` selects handcrafted vs learned scorer.\n"
    "import json, os\n"
    f"FILES = json.loads(r'''{json.dumps(FUSION_FILES)}''')\n"
    "for p, c in FILES.items():\n"
    "    os.makedirs(os.path.dirname(p), exist_ok=True)\n"
    "    open(p, 'w').write(c)\n"
    "    print('wrote', p, f'({len(c)} bytes)')\n"
    "import importlib, sys\n"
    "sys.path.insert(0, '/content/gfm-rag')\n"
    "for m in ['gfmrag.models.fusion_reasoner', 'gfmrag.trainers.fusion_trainer']:\n"
    "    importlib.import_module(m); print('import OK:', m)\n"
    "from gfmrag.models.fusion_reasoner import FusionGraphReasoner\n"
    "from gfmrag.trainers.fusion_trainer import FusionSFTTrainer\n"
    "print('fusion files ready')\n"))
cells.append(md("### 3b. Rewrite the config's dataset defaults"))
cells.append(code("""# Cell 3a is reused VERBATIM from the TOMATO notebook so the fusion model code
# cannot drift. That file also carries a config whose DEFAULT train_names is
# `tomato_train_v16sc`. run_model always overrides it on the command line, but a
# default naming another corpus is exactly the shape of every contamination we
# hit today, so it gets rewritten rather than trusted.
import re
CFGS = [p for p in ["/content/gfm-rag/gfmrag/workflow/config/gfm_reasoner/sft_training_fusion.yaml",
                    "/content/gfm-rag/gfmrag/workflow/config/gfm_reasoner/sft_training_belief.yaml"]
        if os.path.exists(p)]
# BOTH configs, in one loop. The belief config is a copy of the fusion one with a
# different model target, so a rewrite applied to only the first would leave the
# second pointing at a TOMATO corpus -- the exact contamination this cell exists to
# prevent, reintroduced by adding a second file.
for cfg in CFGS:
    t = open(cfg).read()
    t = t.replace("tomato_train_v16sc", TRAIN).replace("tomato_test_v16sc", TEST)
    open(cfg, "w").write(t)
    # ASSERT ON WHAT MUST BE PRESENT, NOT ON WHAT MUST BE ABSENT. The old check counted
    # surviving `tomato_\\w+` matches and required zero, which is correct only while the
    # target corpus is NOT tomato. On a TOMATO build TRAIN is itself `tomato_train_v16sc`,
    # so the replace is the identity, two names legitimately survive, and the assert fired
    # on a config that was in fact right. Checking that both intended names are present and
    # that no foreign corpus leaked is the same guarantee and holds for either target.
    assert TRAIN in t, f"{TRAIN} missing from {cfg}"
    assert TEST in t, f"{TEST} missing from {cfg}"
    assert "__FOREIGN__" not in t, f"a __FOREIGN__ dataset reference survived in {cfg}"
    print(f"{cfg.split('/')[-1]}: train_names/valid_names -> {TRAIN} / {TEST}")
cfg = CFGS[0]
print("  train_names:", TRAIN, "\\n  valid_names:", TEST)

# METRICS REPORTED PER EPOCH. The stock list is twelve columns of hits@k and
# recall@k, none of which is what gets reported, and selection ran on
# document_mrr. Narrow it to the four metrics that appear in the results table,
# and select on nDCG@5.
#
# CompleteSet@5 is deliberately NOT here. utils.evaluate receives gold RANKS, not
# ranked document ids, and knows nothing about sets.json, so the closest thing it
# could compute is "all of this query's golds in the top 5" -- a much harsher
# metric wearing the same name. It comes from score_sir4.py afterwards instead.
for cfg in CFGS:
    t = open(cfg).read()
    t = re.sub(r"^  metrics: \\[.*\\]$", "  metrics: [mrr, ndcg@5, recall@3, recall@5]",
               t, count=1, flags=re.M)
    t = t.replace("metric_for_best_model: document_mrr",
                  "metric_for_best_model: document_ndcg@5")
    open(cfg, "w").write(t)
    assert "metrics: [mrr, ndcg@5, recall@3, recall@5]" in t, f"metrics not rewritten in {cfg}"
    assert "document_ndcg@5" in t, f"selection metric not rewritten in {cfg}"
cfg = CFGS[0]
print("  metrics:  mrr, ndcg@5, recall@3, recall@5")
print("  best on:  document_ndcg@5")

# nDCG DOES NOT EXIST IN THE ENGINE. utils.evaluate handles mrr / recall@k /
# hits@k / mape and raises ValueError on anything else, so selecting on ndcg@5
# would die on the first evaluation. Patch the branch in. Binary relevance with
# the same ideal-DCG treatment as score_sir4.py, verified identical on seven
# cases including more golds than k and all golds outside the cutoff.
QA = "/content/gfm-rag/gfmrag/utils/qa_utils.py"
q = open(QA).read()
if '_metric.startswith("ndcg@")' in q:
    print("  qa_utils: ndcg already patched")
else:
    assert '        elif _metric == "mape":' in q, "mape anchor missing in qa_utils"
    _nd = [
        '        elif _metric.startswith("ndcg@"):',
        '            threshold = int(_metric[5:])',
        '            gain = torch.where(',
        '                answer_ranking <= threshold,',
        '                1.0 / torch.log2(answer_ranking.float() + 1.0),',
        '                torch.zeros_like(answer_ranking, dtype=torch.float),',
        '            )',
        '            dcg = variadic.variadic_sum(gain, num_hard)',
        '            disc = 1.0 / torch.log2(',
        '                torch.arange(1, threshold + 1, device=gain.device).float() + 1.0',
        '            )',
        '            idcg_table = torch.cat([torch.zeros(1, device=gain.device), disc.cumsum(0)])',
        '            idcg = idcg_table[num_hard.clamp(max=threshold)]',
        '            query_score = dcg / idcg.clamp(min=1e-9)',
        '        elif _metric == "mape":',
    ]
    q = q.replace('        elif _metric == "mape":', chr(10).join(_nd), 1)
    open(QA, "w").write(q)
    print("  qa_utils: ndcg@k branch added")
"""))

cells.append(md("### 3c. Fix the vendored PyG version check"))
cells.append(code('''# The ULTRA layers vendored inside the engine parse the PyG version as:
#     pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]
# Colab now resolves torch-geometric to 2.6.1.post1, and int("post1") raises
# ValueError deep inside the first message-passing call -- after the Qwen3 index
# has already been built, so it costs a full setup to discover.
#
# Pinning PyG would also work, but it fights Colab's resolver and will drift
# again at the next release. Parsing the version properly will not. Taking the
# first three dotted components and keeping only the numeric ones handles
# "2.6.1.post1" and "2.7.0+pt24cu121" alike.
import os, torch_geometric
OLD = 'pyg_version = [int(i) for i in torch_geometric.__version__.split(".")]'
NEW = ('pyg_version = [int(i) for i in torch_geometric.__version__.split(".")[:3] '
       'if i.isdigit()]')
hits = []
for root, _, files in os.walk("/content/gfm-rag/gfmrag"):
    for f in files:
        if not f.endswith(".py"):
            continue
        fp = os.path.join(root, f)
        t = open(fp).read()
        if OLD in t:
            open(fp, "w").write(t.replace(OLD, NEW))
            hits.append(fp)
if not hits:
    # Notebooks that ship the fork's ultra/layers.py already carry the fix; nothing to patch.
    fixed = [os.path.join(r, f) for r, _, fs in os.walk("/content/gfm-rag/gfmrag") for f in fs
             if f.endswith(".py") and "pyg_version = " in open(os.path.join(r, f)).read()
             and "if i.isdigit()" in open(os.path.join(r, f)).read()]
    assert fixed, "version-check pattern not found and no fixed copy either -- has the engine changed?"
    print("  already fixed:", *fixed)
parsed = [int(i) for i in torch_geometric.__version__.split(".")[:3] if i.isdigit()]
for h in hits:
    print("  patched", h)
print(f"torch_geometric {torch_geometric.__version__} now parses to {parsed}")'''))

cells.append(md("""### 3c-ii. Restore `torchvision.io.VideoReader` for `datasets`

Recent torchvision removed the legacy video API. `datasets`' torch formatter still does
`from torchvision.io import VideoReader` whenever torchvision is in `sys.modules`, and
`transformers` puts it there, so **every batch fetch** in the training subprocess raises

    ImportError: cannot import name 'VideoReader' from 'torchvision.io'

after the Qwen3 index is built, i.e. after the expensive part. This is a different fault
from the torch/torchvision mismatch the engine cell guards: there torchvision is broken,
here it imports fine and one symbol is gone, so deleting the runtime does not help.

The name is only needed for an `isinstance` check against tensor data, so a placeholder
class is enough. It downloads nothing, which is the point: `pip install torchvision`
resolves to the latest and drags torch and CUDA up with it."""))
cells.append(code('''# THE TRAINING SUBPROCESS IS A FRESH PYTHON. `python -m gfmrag.workflow.sft_training` does
# not inherit this kernel's patched modules, so the shim has to live in a file that the
# subprocess imports. sft_training.py is that file, and is already patched twice below.
import torch, torchvision, torchvision.io, datasets
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | "
      f"datasets {datasets.__version__} | VideoReader "
      f"{hasattr(torchvision.io, 'VideoReader')}")

_SHIM = """# PATCH (notebook): datasets' torch formatter imports torchvision.io.VideoReader
# whenever torchvision is in sys.modules. Recent torchvision removed the legacy video API, so
# that import raises inside every batch fetch. The name is only needed for an isinstance
# check against tensor data, so a placeholder is sufficient and downloads nothing.
import torchvision.io as _tvio
if not hasattr(_tvio, "VideoReader"):
    class _NoVideoReader:
        def __init__(self, *a, **k):
            raise RuntimeError("torchvision video API removed; this shim exists only so "
                               "the datasets torch formatter can import the name")
    _tvio.VideoReader = _NoVideoReader
"""

_STF = "/content/gfm-rag/gfmrag/workflow/sft_training.py"
_t = open(_STF).read()
if "_NoVideoReader" in _t:
    print("[skip] sft_training.py already carries the shim")
else:
    # PREPENDED, not injected at an anchor. The other two patches of this file search for a
    # line inside main(); this one must run before `datasets` is imported anywhere, so it goes
    # above every import. Prepending also leaves their anchors untouched.
    open(_STF, "w").write(_SHIM + _t)
    print("shimmed", _STF)

# This kernel too: the in-kernel gfmrag import and any in-notebook dataset use hit the same
# formatter.
exec(_SHIM)
print("VideoReader present now:", hasattr(torchvision.io, "VideoReader"))'''))

cells.append(md("""### 3d. Per-epoch component diagnostics

The trainer reports the **fused** metric and nothing else, which cannot tell "the graph
helped" apart from "the graph did nothing and you are reading the semantic scorer". This
adds three nDCG@5 numbers per epoch, plus the gate:

- **semantic-only** — the scorer's own ranking, the floor the fusion must beat
- **graph-only** — whether the GNN is learning anything at all
- **fused** — what the trainer already selects on
- **`gamma_q` mean/min/max** — if the mean never leaves its 0.01 init, the fused score
  *is* the semantic score and no loss change can matter

Costs no extra forward pass: a hook reads the scores already cached on the model during
`evaluate()`. The per-component training losses (`hn_fused`, `hn_graph`) are already in
the tqdm postfix each epoch from the `base_trainer` patch above."""))
cells.append(code(r'''STF = "/content/gfm-rag/gfmrag/workflow/sft_training.py"
_src = open(STF).read()
assert "_evaluate_stratified" in _src, "run the engine-install cell first"
if "_evaluate_with_diag" in _src:
    print("component diagnostics already patched")
else:
    _blk = "\n".join([
        "    # --- injected: component diagnostics, ZERO extra forward passes ---",
        "    import torch as _t",
        "    _diag = {'sem': [], 'gph': [], 'fus': [], 'gam': [], 'pri': []}",
        "    # THE STRUCTURAL PRIOR, scored as a channel in its own right. Personalised",
        "    # PageRank from the same seeds, no parameters, no training. It is the number",
        "    # the trained GNN has to beat for the learned component to be doing anything,",
        "    # and it was never in the table: a walk scores 0.245-0.273 nDCG@5 on these",
        "    # graphs while the 34M-parameter GNN scores 0.213-0.273. Three sparse mat-muls",
        "    # per eval batch, so it is free next to six dense GNN layers.",
        "    _PR = {'A': None, 'k': None}",
        "    # a=0 is NOT redundant with 'sem': it is the mixture's own zero point, so if the",
        "    # two disagree the arithmetic is wrong rather than the graph. Grid is log-spaced",
        "    # because the interesting region is a <= 0.05 and the learned 0.371 already lost.",
        "    _ASWEEP = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35]",
        "    # GRAPH RECALL BUCKETED BY THE GOLD'S SEMANTIC RANK. The miss-weighted graph",
        "    # loss (MISS_W_AUX) claims to spend graph capacity on the golds the scorer",
        "    # buries. An aggregate cannot see that trade; this can. Buckets are on r_sem,",
        "    # the SAME quantity the loss weights by, so a working loss must lift the deep",
        "    # buckets while holding the shallow one. If the deep buckets stay flat and the",
        "    # shallow one falls, the graph cannot fit those golds and the reweighting is",
        "    # just moving gradient onto examples the architecture does not support.",
        "    _BUCKETS = [(1, 10), (11, 100), (101, 1000), (1001, 10**9)]",
        "    _BNAME = ['1-10', '11-100', '101-1k', '1k+']",
        "    _diag.update({'b%d' % _i: [] for _i in range(len(_BUCKETS))})",
        "    _diag.update({'a%g' % _a: [] for _a in _ASWEEP})",
        "    # THE SAME SWEEP FOR THE ADDITIVE FORM, and it runs in BOTH arms. The two forms",
        "    # are being compared as a 2x2 against the gate, so each one's free scalar has to",
        "    # be chosen by the same procedure or the comparison measures the tuning effort",
        "    # rather than the form. gamma is learned in the additive arm, but it froze at",
        "    # 0.0279 / 0.0288 / 0.0359 in three runs that scored 0.5207 / 0.5218 / 0.5340,",
        "    # so what gamma converges to is not obviously what gamma should be. Both channels",
        "    # are cached here, so this curve costs nothing and one run of EITHER arm now",
        "    # returns both curves. Grid brackets every gamma any arm has actually reached.",
        "    _GSWEEP = [0.0, 0.01, 0.02, 0.0359, 0.06, 0.1, 0.2, 0.4]",
        "    _diag.update({'g%g' % _g: [] for _g in _GSWEEP})",
        "    # SPLICE sweep. The mixture's a controls the DEPTH at which the graph's block",
        "    # lands; what nDCG@5 actually cares about is its SIZE. _ISWEEP controls size",
        "    # directly: keep semantic's top (5-m) untouched and hand ranks (5-m+1)..5 to the",
        "    # graph's own top m. Displacement is exactly m, known before the run, and the",
        "    # win condition is a hit-rate comparison rather than an emergent property of a",
        "    # scalar. m=0 is the semantic control. Uses only the graph's ORDER, so it is",
        "    # immune to the overconfident calibration that z()/tau_g are fighting.",
        "    _ISWEEP = [1, 2, 3]",
        "    _diag.update({'i%d' % _m: [] for _m in _ISWEEP})",
        "    _diag.update({'gh%d' % _j: [] for _j in range(3)})",
        "    _diag.update({'sh%d' % _j: [] for _j in range(5)})",
        "    _drec = {'on': False}",
        "    def _ndcg5(_sc, _tgt):",
        "        # binary relevance, same ideal-DCG convention as score_sir4.py",
        "        _k = min(5, _sc.shape[-1])",
        "        _top = _sc.topk(_k, dim=-1).indices",
        "        _hit = _tgt.gather(1, _top).float()",
        "        _disc = 1.0 / _t.log2(_t.arange(2, _k + 2, device=_hit.device).float())",
        "        _dcg = (_hit * _disc).sum(-1)",
        "        _n = _tgt.sum(-1).clamp(max=_k).long()",
        "        _tab = _t.cat([_t.zeros(1, device=_hit.device), _disc.cumsum(0)])",
        "        return _dcg / _tab[_n].clamp(min=1e-9)",
        "    def _dhook(_mod, _inp, _out):",
        "        if not _drec['on'] or len(_inp) < 2:",
        "            return",
        "        try:",
        "            _g, _b = _inp[0], _inp[1]",
        "            _did = _g.nodes_by_type['document']",
        "            _tgt = _b['target_nodes_mask'][:, _did]",
        "            _diag['fus'] += _ndcg5(_out[:, _did].float(), _tgt).tolist()",
        "            # prior: symmetrised, degree-normalised walk with restart from the",
        "            # query's seeds. Symmetrised because the KG is directed and a walk that",
        "            # only travels head->tail cannot reach a document from an entity that",
        "            # document mentions, which is the dominant path in this corpus.",
        "            if _b.get('start_nodes_mask') is not None:",
        "                _N = _g.num_nodes",
        "                if _PR['k'] != id(_g):",
        "                    _ei = _g.edge_index",
        "                    _rr = _t.cat([_ei[0], _ei[1]]); _cc = _t.cat([_ei[1], _ei[0]])",
        "                    _dg = _t.zeros(_N, device=_ei.device).index_add_(",
        "                        0, _rr, _t.ones(_rr.numel(), device=_ei.device))",
        "                    _PR['A'] = _t.sparse_coo_tensor(",
        "                        _t.stack([_cc, _rr]), 1.0 / _dg.clamp(min=1.0)[_rr],",
        "                        (_N, _N)).coalesce()",
        "                    _PR['k'] = id(_g)",
        "                _x0 = _b['start_nodes_mask'].float()",
        "                _x0 = _x0 / _x0.sum(-1, keepdim=True).clamp(min=1e-9)",
        "                _x = _x0",
        "                # autocast off: eval runs under bfloat16 AMP and sparse.mm has no",
        "                # bf16 kernel on every build. The prior is data, not a learned",
        "                # quantity, so nothing here should be cast or differentiated.",
        "                with _t.autocast(device_type=_x0.device.type, enabled=False):",
        "                    for _ in range(3):",
        "                        _x = 0.85 * _t.sparse.mm(_PR['A'], _x.t()).t() + 0.15 * _x0",
        "                _diag['pri'] += _ndcg5(_x[:, _did].float(), _tgt).tolist()",
        "            _sem = getattr(_mod, '_s_op', None)",
        "            _gph = getattr(_mod, '_raw_doc', None)",
        "            _gam = getattr(_mod, '_gamma', None)",
        "            if _sem is not None:",
        "                _diag['sem'] += _ndcg5(_sem.detach().float(), _tgt).tolist()",
        "            if _gph is not None:",
        "                _diag['gph'] += _ndcg5(_gph.detach().float(), _tgt).tolist()",
        "            if _gam is not None:",
        "                _diag['gam'] += _gam.detach().flatten().tolist()",
        "            if _sem is not None and _gph is not None:",
        "                _sd, _gd = _sem.detach().float(), _gph.detach().float()",
        "                _gtop = _gd.topk(min(10, _gd.shape[-1]), dim=-1).indices",
        "                for _bi in range(_tgt.shape[0]):",
        "                    _pos = _tgt[_bi].nonzero(as_tuple=True)[0]",
        "                    if _pos.numel() == 0:",
        "                        continue",
        "                    # true corpus rank of each gold under the semantic scorer,",
        "                    # matching how _contrastive_hardneg computes its weights",
        "                    _r = (_sd[_bi].unsqueeze(0) > _sd[_bi, _pos].unsqueeze(1)).sum(1) + 1",
        "                    _in = _t.isin(_pos, _gtop[_bi]).float()",
        "                    for _i, (_lo, _hi) in enumerate(_BUCKETS):",
        "                        _m = (_r >= _lo) & (_r <= _hi)",
        "                        if bool(_m.any()):",
        "                            _diag['b%d' % _i] += _in[_m].tolist()",
        "            # THE a-SWEEP. Costs a few tensor ops per batch and answers the only",
        "            # question the mixture arm exists to answer: is there ANY mixing weight",
        "            # at which the graph helps? The first run tested a_q=0.371 (learned) and",
        "            # lost 0.091 nDCG@5. One point is not a curve. Both channels are already",
        "            # cached here, so every a is free. tau_g is read off the model when it is",
        "            # there, so the sweep matches what the fusion actually computes.",
        "            if _sem is not None and _gph is not None and _ASWEEP:",
        "                _zz = lambda x: (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)",
        "                _th = getattr(_mod, 'tau_g_hat', None)",
        "                _tg = 1.0 if _th is None else float(",
        "                    _mod.tau_min + (_mod.tau_max - _mod.tau_min) * _t.sigmoid(_th.float()))",
        "                _ls = _zz(_sem.detach().float())",
        "                _lg = _zz(_gph.detach().float()) / max(_tg, 1e-3)",
        "                _lps = _ls - _t.logsumexp(_ls, -1, keepdim=True)",
        "                _lpg = _lg - _t.logsumexp(_lg, -1, keepdim=True)",
        "                for _a in _ASWEEP:",
        "                    _f = _t.logaddexp(_t.log(_t.tensor(1.0 - _a)) + _lps,",
        "                                      _t.log(_t.tensor(_a)) + _lpg)",
        "                    _diag['a%g' % _a] += _ndcg5(_f, _tgt).tolist()",
        "                # the additive curve, on the SAME cached channels and the same queries,",
        "                # so a=x and gamma=y are paired per query and the two forms can be",
        "                # bootstrapped against each other rather than compared as two run-level",
        "                # point estimates 0.013 apart on n=331.",
        "                _rg = _t.relu(_zz(_gph.detach().float()))",
        "                for _g in _GSWEEP:",
        "                    _diag['g%g' % _g] += _ndcg5(_ls + _g * _rg, _tgt).tolist()",
        "                # splice: inject the graph's top m between semantic ranks (5-m) and",
        "                # (5-m+1). maximum() rather than scatter so a graph pick already in",
        "                # semantic's kept head is left where it is instead of demoted.",
        "                _gr = _zz(_gph.detach().float())",
        "                _sv = _ls.sort(-1, descending=True).values",
        "                for _m in _ISWEEP:",
        "                    _c = 5 - _m",
        "                    _mid = (_sv[:, _c - 1:_c] + _sv[:, _c:_c + 1]) / 2.0",
        "                    _off = _t.arange(_m, device=_ls.device).float() * 1e-4",
        "                    _inj = _t.full_like(_ls, -1e9)",
        "                    _inj.scatter_(1, _gr.topk(_m, -1).indices, _mid - _off)",
        "                    _diag['i%d' % _m] += _ndcg5(_t.maximum(_ls, _inj), _tgt).tolist()",
        "                # the decisive number: is the graph's own top-1 a gold more often",
        "                # than the semantic doc it would evict? if not, NO reordering of",
        "                # these two lists can raise nDCG@5 and the fusion form is not the",
        "                # problem. positions are per-rank hit rates, not nDCG.",
        "                _gh = _tgt.gather(1, _gr.topk(3, -1).indices).float()",
        "                _sh = _tgt.gather(1, _ls.topk(5, -1).indices).float()",
        "                for _j in range(3):",
        "                    _diag['gh%d' % _j] += _gh[:, _j].tolist()",
        "                for _j in range(5):",
        "                    _diag['sh%d' % _j] += _sh[:, _j].tolist()",
        "        except Exception as _e:",
        "            print('[diag] hook skipped:', _e)",
        "    trainer.model.register_forward_hook(_dhook)",
        "    _prev_eval = trainer.evaluate",
        "    def _evaluate_with_diag():",
        "        for _v in _diag.values():",
        "            _v.clear()",
        "        _drec['on'] = True",
        "        m = _prev_eval()",
        "        _drec['on'] = False",
        "        _mu = lambda x: (sum(x) / len(x)) if x else float('nan')",
        "        _lo = lambda x: min(x) if x else float('nan')",
        "        _hi = lambda x: max(x) if x else float('nan')",
        "        print(f\"[diag] nDCG@5  __L1__ {_mu(_diag['sem']):.4f} | \"",
        "              f\"__L2__ {_mu(_diag['gph']):.4f} | __L3__ {_mu(_diag['fus']):.4f}   \"",
        "              f\"gamma mean {_mu(_diag['gam']):.4f} min {_lo(_diag['gam']):.4f} \"",
        "              f\"max {_hi(_diag['gam']):.4f}\", flush=True)",
        "        _bhave = [_i for _i in range(len(_BUCKETS)) if _diag['b%d' % _i]]",
        "        if _bhave:",
        "            print('[bucket] graph recall@10 by gold semantic rank:  ' + '  '.join(",
        "                  '%s %.3f (n=%d)' % (_BNAME[_i], _mu(_diag['b%d' % _i]),",
        "                                      len(_diag['b%d' % _i])) for _i in _bhave), flush=True)",
        "            for _i in _bhave:",
        "                m['bucket/graph_r10_%s' % _BNAME[_i]] = _mu(_diag['b%d' % _i])",
        "        if _diag['pri']:",
        "            _pv, _gv = _mu(_diag['pri']), _mu(_diag['gph'])",
        "            print('[prior] parameter-free walk nDCG@5 %.4f   trained graph %.4f   '",
        "                  '(%+.4f)  -> %s' % (_pv, _gv, _gv - _pv,",
        "                  'the GNN adds something' if _gv - _pv > 0.02 else",
        "                  'THE GNN IS AT ITS OWN PRIOR: the learned component is ~free-lunch "
        "topology'), flush=True)",
        "            m['prior/ndcg@5_walk'] = _pv",
        "            m['prior/gnn_minus_walk'] = _gv - _pv",
        "        m['diag/ndcg@5_semantic'] = _mu(_diag['sem'])",
        "        m['diag/ndcg@5_graph'] = _mu(_diag['gph'])",
        "        m['diag/ndcg@5_fused'] = _mu(_diag['fus'])",
        "        m['diag/gamma_mean'] = _mu(_diag['gam'])",
        "        if _ASWEEP and _diag['a%g' % _ASWEEP[0]]:",
        "            _row = [(_a, _mu(_diag['a%g' % _a])) for _a in _ASWEEP]",
        "            _bi = max(range(len(_row)), key=lambda i: _row[i][1])",
        "            print('[sweep] fused nDCG@5 vs constant a:  '",
        "                  + '  '.join('a=%g %.4f' % r for r in _row), flush=True)",
        "            print('[sweep] best a=%g -> %.4f   vs semantic %.4f   (%+.4f)   '",
        "                  'a=0 check %.4f'",
        "                  % (_row[_bi][0], _row[_bi][1], _mu(_diag['sem']),",
        "                     _row[_bi][1] - _mu(_diag['sem']), _row[0][1]), flush=True)",
        "            for _a, _v in _row:",
        "                m['sweep/a%g' % _a] = _v",
        "            # the additive curve, printed next to it. These two lines ARE the 2x2's",
        "            # fusion axis at this gate setting: same queries, same checkpoint, same",
        "            # cached channels, so the only thing that differs between them is the form.",
        "            _grow = [(_g, _mu(_diag['g%g' % _g])) for _g in _GSWEEP]",
        "            _gi = max(range(len(_grow)), key=lambda i: _grow[i][1])",
        "            print('[gsweep] fused nDCG@5 vs constant gamma (additive):  '",
        "                  + '  '.join('g=%g %.4f' % r for r in _grow), flush=True)",
        "            print('[gsweep] best gamma=%g -> %.4f   vs semantic %.4f   (%+.4f)   '",
        "                  'gamma=0 check %.4f'",
        "                  % (_grow[_gi][0], _grow[_gi][1], _mu(_diag['sem']),",
        "                     _grow[_gi][1] - _mu(_diag['sem']), _grow[0][1]), flush=True)",
        "            print('[form] best mixture %+.4f  vs  best additive %+.4f   over semantic'",
        "                  % (_row[_bi][1] - _mu(_diag['sem']),",
        "                     _grow[_gi][1] - _mu(_diag['sem'])), flush=True)",
        "            for _g, _v in _grow:",
        "                m['sweep/g%g' % _g] = _v",
        "            # per-query vectors for the paired bootstrap. Aggregates cannot be",
        "            # bootstrapped after the fact and these are the only place the per-query",
        "            # numbers exist, so they get written next to the checkpoint.",
        "            try:",
        "                import json as _json, os as _os",
        "                _pq = {'semantic': _diag['sem'], 'graph': _diag['gph'],",
        "                       'fused': _diag['fus']}",
        "                _pq.update({'a%g' % _a: _diag['a%g' % _a] for _a in _ASWEEP})",
        "                _pq.update({'g%g' % _g: _diag['g%g' % _g] for _g in _GSWEEP})",
        "                with open(_os.path.join(_os.getcwd(), 'per_query_ndcg5.json'), 'w') as _fh:",
        "                    _json.dump(_pq, _fh)",
        "            except Exception as _e:",
        "                print('[diag] per-query dump skipped:', _e, flush=True)",
        "        if _diag['i%d' % _ISWEEP[0]]:",
        "            _ir = [(_m, _mu(_diag['i%d' % _m])) for _m in _ISWEEP]",
        "            _ib = max(range(len(_ir)), key=lambda i: _ir[i][1])",
        "            print('[splice] fused nDCG@5 vs graph block size m:  m=0 %.4f  ' % _mu(_diag['sem'])",
        "                  + '  '.join('m=%d %.4f' % r for r in _ir), flush=True)",
        "            print('[splice] best m=%d -> %.4f   vs semantic %.4f   (%+.4f)'",
        "                  % (_ir[_ib][0], _ir[_ib][1], _mu(_diag['sem']),",
        "                     _ir[_ib][1] - _mu(_diag['sem'])), flush=True)",
        "            for _m, _v in _ir:",
        "                m['splice/m%d' % _m] = _v",
        "        if _diag['gh0']:",
        "            _g1 = _mu(_diag['gh0'])",
        "            _s5 = _mu(_diag['sh4'])",
        "            print('[hit] gold rate by rank -- graph 1..3: '",
        "                  + ' '.join('%.3f' % _mu(_diag['gh%d' % _j]) for _j in range(3))",
        "                  + '   semantic 1..5: '",
        "                  + ' '.join('%.3f' % _mu(_diag['sh%d' % _j]) for _j in range(5)),",
        "                  flush=True)",
        "            print('[hit] graph@1 %.3f vs semantic@5 %.3f -> %s' % (_g1, _s5,",
        "                  'a swap at rank 5 gains in expectation' if _g1 > _s5 else",
        "                  'NO reordering of these two lists can raise nDCG@5'), flush=True)",
        "            m['hit/graph@1'] = _g1",
        "            m['hit/semantic@5'] = _s5",
        "        return m",
        "    trainer.evaluate = _evaluate_with_diag",
        "    trainer.train()",
    ])
    # Injected AFTER the stratified block, so _prev_eval wraps that one rather than
    # replacing it and silently dropping the per-slice metrics.
    assert _src.count("    trainer.train()") == 1, "unexpected number of trainer.train() calls"
    open(STF, "w").write(_src.replace("    trainer.train()", _blk, 1))
    print("patched sft_training.py -> per-epoch semantic/graph/fused nDCG@5 + gamma")'''))

cells.append(md("""### 3e. CCMP head learning rate (`CCMP_LR`)

The responsibility head is a fresh `6 x (1024 -> 64 -> 1)` MLP fitting a target from
scratch, while every other parameter around it is warm-started. At the trunk's `5e-4` it
fits slowly, and a null CCMP result then cannot be told apart from an under-trained head.
This gives it its own optimizer parameter group.

**Unset = one parameter group, byte-identical to every CCMP run made before this.** The
control arm has `CCMP_LR` popped by `run_model`, so the pair stays one-variable. The head
is already float32-pinned by `configure_model_precision`, so this is about how fast it
FITS, not about the bf16 freeze line."""))
cells.append(code(r'''# Idempotent: re-running is a no-op. Patches the same file section 3d just rewrote,
# so it must run AFTER it.
STF = "/content/gfm-rag/gfmrag/workflow/sft_training.py"
_src = open(STF).read()
_ANCHOR = "    optimizer = instantiate(cfg.optimizer, model.parameters())"
if "_ccmp_lr = os.environ.get" in _src:
    print("CCMP_LR parameter group already patched")
else:
    assert _src.count(_ANCHOR) == 1, (
        f"expected exactly one optimizer construction, found {_src.count(_ANCHOR)}")
    _NEW = '\n'.join([
        '    _ccmp_lr = os.environ.get("CCMP_LR", "")',
        '    _head, _rest = [], []',
        '    if _ccmp_lr != "":',
        '        for _n, _p in model.named_parameters():',
        '            if not _p.requires_grad:',
        '                continue',
        '            (_head if any(k in _n for k in ("resp_proj", "resp_emb", "resp_head"))',
        '             else _rest).append(_p)',
        '        # Raise only when the head is genuinely missing on a CCMP arm. CCMP_LR',
        '        # reaches a control through dict(os.environ, ...) with no resp_* to attach',
        '        # to, and that is not an error -- it falls back to one group and says so.',
        '        if not _head and os.environ.get("CCMP", "0") == "1":',
        '            raise AssertionError(',
        '                "CCMP_LR set and CCMP=1 but no resp_* parameters exist. The head "',
        '                "was not built; check the [ccmp] construction line above.")',
        '    if _head:',
        '        optimizer = instantiate(cfg.optimizer,',
        '                                [{"params": _rest},',
        '                                 {"params": _head, "lr": float(_ccmp_lr)}])',
        '        logger.info(f"[ccmp] head lr {float(_ccmp_lr):g} on {len(_head)} tensors | "',
        '                    f"trunk lr {cfg.optimizer.lr:g} on {len(_rest)} tensors")',
        '    else:',
        '        optimizer = instantiate(cfg.optimizer, model.parameters())',
        '        if _ccmp_lr != "":',
        '            logger.info("[ccmp] CCMP_LR set but no responsibility head in this arm "',
        '                        "(control); one parameter group, unchanged.")',
    ])
    open(STF, "w").write(_src.replace(_ANCHOR, _NEW, 1))
    print("patched sft_training.py -> CCMP_LR parameter group")
    print("read the [ccmp] head lr line in the training log to confirm it attached")'''))

cells.append(md("## 4. Qwen3-Embedding-0.6B (cached on Drive)"))
cells.append(reuse[7])

cells.append(md("## 5. Unpack the bundle to an isolated local root"))
cells.append(code('''import os, shutil, zipfile, sys, json
# Unpacked FRESH every run so a re-upload is picked up, and LOCAL so neither the
# Qwen3 stage-2 index nor any cache lands on Drive next to the TOMATO ones.
#
# BUT the BGE embeddings are the expensive artefact in this notebook (~24k
# documents), and they live under the tree we are about to delete. Re-running
# this cell to pick up a new bundle would silently throw away 20 minutes of GPU
# encoding. So they are parked outside SCIGRAPHIR_ROOT and restored afterwards; a
# stale one cannot survive because cached_encode re-encodes on a row-count
# mismatch, and the cache is already scoped by dataset and query-set hash.
EMB = f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb"
PARK = "/content/_op_emb_keep"
if os.path.isdir(EMB):
    if os.path.isdir(PARK):
        shutil.rmtree(PARK)
    shutil.move(EMB, PARK)
    print(f"parked embeddings: {sum(len(f) for _, _, f in os.walk(PARK))} files")

if os.path.exists(SCIGRAPHIR_ROOT):
    shutil.rmtree(SCIGRAPHIR_ROOT)
os.makedirs(SCIGRAPHIR_ROOT, exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    z.extractall(SCIGRAPHIR_ROOT)

if os.path.isdir(PARK):
    if os.path.isdir(EMB):
        shutil.rmtree(EMB)
    os.makedirs(os.path.dirname(EMB), exist_ok=True)
    shutil.move(PARK, EMB)
    print("restored embeddings")

# CODE OVERLAY, applied before anything imports or shells out.
# The bundles are 12-64 MB each and rebuilt rarely, so the scripts inside them go
# stale against the repo. Right now all four ship a operator_scorer.py,
# precompute_operator_components.py and score_sir4.py that predate the encoder
# scoping and the metric changes. Re-uploading 138 MB of zips to fix three files
# is the wrong trade. Anything at {DRIVE}/code_overlay/<repo-relative path> is
# copied over the unpacked bundle and wins, so one small upload fixes every domain
# at once.
_ov = f"{DRIVE}/code_overlay"
if os.path.isdir(_ov):
    _n = 0
    for _dp, _, _fs in os.walk(_ov):
        for _f in _fs:
            _src = os.path.join(_dp, _f)
            _dst = os.path.join(SCIGRAPHIR_ROOT, os.path.relpath(_src, _ov))
            os.makedirs(os.path.dirname(_dst), exist_ok=True)
            shutil.copy(_src, _dst); _n += 1
            print("  overlay:", os.path.relpath(_src, _ov))
    print(f"applied {_n} overlay file(s)")
else:
    print("no code_overlay on Drive; using the bundle's own copies")

# Fail on the missing capability, not on a bare argparse exit code 2 twenty
# minutes into a run.
_need = {
    f"{SCIGRAPHIR_ROOT}/retriever/eval/operator_scorer.py": ["--model", "model_slug"],
    f"{SCIGRAPHIR_ROOT}/retriever/precompute/precompute_operator_components.py": ["--model"],
    # "semantic-scorer comparison" is the comment on the KS tuple that added the
    # 25 cutoff. Probing for the literal "recall@25" would ALWAYS fail: the metric
    # keys are built by f-string from KS and never appear in the source.
    f"{SCIGRAPHIR_ROOT}/experiments/eval/score_sir4.py": ["--cols", "--gold-strata", "mrr_best",
                                                        "semantic-scorer comparison"],
    f"{SCIGRAPHIR_ROOT}/experiments/eval/semantic_scorer.py": ["--arms", "GatedScorer",
                                                             "multigold_loss"],
}
for _f, _toks in _need.items():
    _src = open(_f).read()
    _miss = [t for t in _toks if t not in _src]
    assert not _miss, (f"{os.path.basename(_f)} on this runtime lacks {_miss}. It came "
                       f"from a bundle predating them. Upload the current file to "
                       f"{DRIVE}/code_overlay/" + os.path.relpath(_f, SCIGRAPHIR_ROOT))
print("bundled scripts are current")

sys.path.insert(0, SCIGRAPHIR_ROOT)
import scigraphir_paths as cp
cp.set_dataset(DATASET)
print(cp.banner())
assert cp.ROOT == SCIGRAPHIR_ROOT, f"scigraphir_paths resolved {cp.ROOT}, expected {SCIGRAPHIR_ROOT}"

# Resolve every input through scigraphir_paths rather than by hand, so a path that is
# wrong here is wrong in the pipeline too and shows up now, not mid-run.
for s in ("train", "test"):
    for label, p in ((f"corpus {s}", f"{cp.corpus_dir(s)}/raw/documents.json"),
                     (f"queries {s}", f"{cp.corpus_dir(s)}/raw/{s}.json"),
                     (f"probes {s}", cp.probes_path(s)),
                     (f"graph {s}", f"{cp.graph_dir(s)}/processed/stage1/nodes.csv")):
        assert os.path.exists(p), f"missing {label}: {p}"
        print(f"  ok  {label:14} {p.replace(SCIGRAPHIR_ROOT, '<root>')}")
# The loader opens {graph}/raw/documents.json, i.e. a copy of the corpus INSIDE
# the graph directory, not the corpus directory next to it. The graph build never
# writes one, so it has to be placed here or GraphIndexDataset dies with a bare
# FileNotFoundError several minutes into the run.
for split, g in (("train", TRAIN), ("test", TEST)):
    dst = f"{cp.graph_dir(split)}/raw"
    os.makedirs(dst, exist_ok=True)
    shutil.copy(f"{cp.corpus_dir(split)}/raw/documents.json", f"{dst}/documents.json")
    n = len(json.load(open(f"{dst}/documents.json")))
    print(f"  ok  graph corpus  {g}/raw/documents.json  ({n:,} docs)")

SETS = f"{SCIGRAPHIR_ROOT}/benchmark/data/benchmark/__SETSDIR__/sets.json"
print("  sets.json:", "present" if os.path.exists(SETS) else "ABSENT (CompleteSet@k -> 0)")
!du -sh {SCIGRAPHIR_ROOT}

# ---------------------------------------------------------------- persistence
# Colab resets its runtime and /content does not survive it. Without this, every
# reset costs the operator fit (~6 min), the BGE baseline (~2 min) and the Qwen3
# node index (30-45 min). Those all live under /content, so they are copied to
# Drive as they are produced and copied back on the next run.
# NOT under OUT_ROOT. Everything in here is keyed to the GRAPH, not to which arms a
# notebook trains: the Qwen3 node index, the operator embeddings, the BGE predictions.
# Putting it under the run-set namespace orphaned the existing cache and demanded a
# 45-minute index rebuild per run set for an identical artefact. graph_fingerprint.json
# is what makes sharing safe -- it refuses to restore across a rebuilt graph.
CACHE = f"{DRIVE}/outputs/{DATASET}/cache"

def _fingerprint():
    \"\"\"Identify the GRAPHS the derived artefacts were built against.

    operator_components.npz is column-ordered to the graph's nodes.csv document
    order, and the Qwen3 index is built over the graph's node names. Restoring
    either against a REBUILT graph would silently misalign every score with no
    error -- the same shape as every contamination this project has hit. So both
    are gated on the graph files being byte-identical; everything else depends
    only on the corpus and restores unconditionally.
    \"\"\"
    fp = {}
    for g in (TRAIN, TEST):
        s1 = f"{DATA_ROOT}/{g}/processed/stage1"
        fp[g] = {f: os.path.getsize(f"{s1}/{f}")
                 for f in sorted(os.listdir(s1)) if f.endswith((".csv", ".json"))}
    return fp

def _index_dirs(g):
    pr = f"{DATA_ROOT}/{g}/processed"
    return [d for d in os.listdir(pr) if d != "stage1"] if os.path.isdir(pr) else []

def save_cache():
    \"\"\"Copy whatever exists to Drive. Safe to call at any point, repeatedly.\"\"\"
    os.makedirs(CACHE, exist_ok=True)
    json.dump(_fingerprint(), open(f"{CACHE}/graph_fingerprint.json", "w"), indent=1)
    saved = []
    src = f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb__EMBDS__"
    if os.path.isdir(src):
        shutil.copytree(src, f"{CACHE}/op_emb", dirs_exist_ok=True); saved.append("op_emb")
    for f in (f"{SCIGRAPHIR_ROOT}/retriever/eval/operator_params__PARAMDS__{OP_SLUG}.json", BGE_PRED):
        if os.path.exists(f):
            shutil.copy(f, f"{CACHE}/{os.path.basename(f)}"); saved.append(os.path.basename(f))
    for g in (TRAIN, TEST):
        p = f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
        if os.path.exists(p):
            shutil.copy(p, f"{CACHE}/{g}_operator_components{OP_SLUG}.npz"); saved.append(f"{g} components")
        for d in _index_dirs(g):
            shutil.copytree(f"{DATA_ROOT}/{g}/processed/{d}", f"{CACHE}/index/{g}/{d}",
                            dirs_exist_ok=True)
            saved.append(f"{g} index/{d}")
    print("[cache] saved:", ", ".join(saved) or "nothing yet")
    !du -sh {CACHE}

def restore_cache():
    if not os.path.isdir(CACHE):
        print("[cache] nothing on Drive yet -- run 5b/5c, they save as they go")
        return
    fpf = f"{CACHE}/graph_fingerprint.json"
    same = os.path.exists(fpf) and json.load(open(fpf)) == _fingerprint()
    got = []
    src = f"{CACHE}/op_emb"
    if os.path.isdir(src):
        shutil.copytree(src, f"{SCIGRAPHIR_ROOT}/outputs/caches/op_emb__EMBDS__", dirs_exist_ok=True)
        got.append("op_emb")
    pj = f"{CACHE}/operator_params__PARAMDS__{OP_SLUG}.json"
    if os.path.exists(pj):
        shutil.copy(pj, f"{SCIGRAPHIR_ROOT}/retriever/eval/"); got.append("operator params")
    bp = f"{CACHE}/{os.path.basename(BGE_PRED)}"
    if os.path.exists(bp):
        os.makedirs(os.path.dirname(BGE_PRED), exist_ok=True)
        shutil.copy(bp, BGE_PRED); got.append("bge predictions")
    if same:
        for g in (TRAIN, TEST):
            c = f"{CACHE}/{g}_operator_components{OP_SLUG}.npz"
            if os.path.exists(c):
                shutil.copy(c, f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz")
                got.append(f"{g} components")
            if os.path.isdir(f"{CACHE}/index/{g}"):
                shutil.copytree(f"{CACHE}/index/{g}", f"{DATA_ROOT}/{g}/processed",
                                dirs_exist_ok=True)
                got.append(f"{g} index")
    elif os.path.exists(fpf):
        print("[cache] GRAPHS CHANGED since this cache was written -- components and "
              "index NOT restored (they would misalign). Re-run 5b and rebuild the index.")
    print("[cache] restored:", ", ".join(got) or "nothing")
    print("[cache] force_reload should be:", "False" if same and
          all(_index_dirs(g) for g in (TRAIN, TEST)) else "True")

restore_cache()'''))

# ---------------------------------------------------------------- --noent
# Emitted ONLY under --noent, and deliberately placed AFTER restore_cache().
#
# WHY AFTER, NOT BEFORE. _fingerprint() records the SIZE of every stage1 .csv and
# .json, so stripping first changes {split}.json, fails the gate, and costs a
# 30-45 minute rebuild of the Qwen3 index. That rebuild would be pure waste: the
# index is built over node NAMES and operator_components.npz is column-ordered to
# nodes.csv document order, and the entity channel touches neither. Restoring
# against the entity fingerprint and stripping afterwards is not a trick past the
# guard, it is the guard applied to what it actually protects.
#
# WHY NOT SHIP A SECOND DATASET. In build_greasoner_dataset.py the node map and
# the edge set are finalised before the first read of `a.entity_seeds`, which
# lives entirely in the query-seeding block. nodes/edges/relations are therefore
# byte-identical with or without the flag, so a rebuild-and-reupload spends an
# hour of embedding and ~800 MB of Drive to reproduce files it cannot change.
if NOENT:
    cells.append(md(
        "## 5a-noent. Drop the `entity` seed channel\n"
        "The one construction difference between TOMATO-Star and the four SIR-4 corpora. "
        "`run_domain.py` builds every SIR-4 graph with `--no_entity_seeds`; TOMATO was built "
        "with the builder's default, which is on. The channel snaps raw query terms to the "
        "nearest node of any type and supplies **53.6% of all seeds** (76.9 to 35.7 per "
        "query). It is term-level topical matching, so it is also the seeding that most "
        "undercuts the claim that a retrieved path shows a document *supplies something*, "
        "rather than merely discussing the same topic.\n\n"
        "After this cell the remaining channels are `function, limitation, method, task`, "
        "which is exactly SIR-4's four."))
    cells.append(code('''import os, json, glob

for _g, _split in ((TRAIN, "train"), (TEST, "test")):
    _p = f"{DATA_ROOT}/{_g}/processed/stage1/{_split}.json"
    _rows = json.load(open(_p))
    _before = sum(sum(len(v) for v in r["start_nodes"].values()) for r in _rows)
    for r in _rows:
        r["start_nodes"].pop("entity", None)
    _after = sum(sum(len(v) for v in r["start_nodes"].values()) for r in _rows)
    # A seedless query is DROPPED by the loader with only a log line, so the final
    # metric would be computed over fewer queries than the table reports.
    _dead = [r["id"] for r in _rows if not any(r["start_nodes"].values())]
    assert not _dead, f"{len(_dead)} queries left seedless in {_g}: {_dead[:5]}"
    json.dump(_rows, open(_p, "w"))
    _ch = sorted({k for r in _rows for k in r["start_nodes"]})
    assert "entity" not in _ch
    print(f"{_g:26} {len(_rows):>6} queries  seeds/query "
          f"{_before/len(_rows):6.2f} -> {_after/len(_rows):5.2f}  channels={_ch}")

# THE PART THAT IS EASY TO MISS, AND SILENT IF MISSED.
# start_nodes_mask is baked into processed/stage2/<fingerprint>/{train,test}.pt, and
# load_qa_data only regenerates those on force_reload or a graph rebuild. That stage2
# fingerprint is an md5 of the TEXT EMBEDDING MODEL CONFIG, nothing about the data, so
# it does not move when the seeds change and cannot protect you here. Leave the tensors
# in place and the run trains on the ENTITY seeds while every path and label says
# no-entity: a fake null, indistinguishable from a real one.
#
# Deleting ONLY the QA tensors is what makes this cheap. graph.pt is the 30-45 minute
# artefact and is unchanged, so it stays and the next run reprocesses QA alone.
for _g in (TRAIN, TEST):
    for _pt in sorted(glob.glob(f"{DATA_ROOT}/{_g}/processed/stage2/*/*.pt")):
        if os.path.basename(_pt) in ("train.pt", "test.pt"):
            os.remove(_pt); print(f"  dropped stale QA tensors  {_g}  {os.path.basename(_pt)}")
        else:
            print(f"  kept                      {_g}  {os.path.basename(_pt)}")

# NEVER WRITE THIS BACK TO DRIVE. save_cache() copies processed/ wholesale under the
# GRAPH NAME, which this ablation does not change, so one call would overwrite the
# entity run's train.pt/test.pt with these and every later entity run would silently
# load no-entity seeds. Nothing new needs saving: everything restore_cache() brought
# back is unchanged, and a runtime reset is recovered by re-running section 5 and then
# this cell.
_real_save_cache = save_cache
def save_cache():
    print("[cache] save DISABLED for the no-entity arm "
          "(it would clobber the entity run's stage2 tensors)")

print("\\nstripped. Every later cell must run with force_reload=False: the graph is "
      "unchanged and only the QA tensors need rebuilding.")'''))

cells.append(md("## 5b. Phase 3 — Qwen3 operator (was hours on a laptop, minutes here)\n"
                "Encodes 24,384 documents, 6,497 questions and ~49k probes, fits the "
                "operator's four parameters, then caches the raw components so `w0,w1,w2,β` "
                "stay trainable inside the fusion. Run locally this thrashed a 17 GB "
                "machine into swap; on an A100 the encoding is the only real cost."))
cells.append(code('''import os, sys, subprocess, time
KGDIR = f"{SCIGRAPHIR_ROOT}/retriever"
# PYTHONUNBUFFERED so the child's print() reaches us as it happens. Without it
# Python block-buffers stdout whenever it is a pipe rather than a terminal, so
# even a live reader sees nothing for minutes at a time.
env = dict(os.environ, SCIGRAPHIR_ROOT=SCIGRAPHIR_ROOT, SCIGRAPHIR_DATASET=DATASET,
           PYTHONUNBUFFERED="1")

def sh(cmd, cwd):
    """Run a command, echoing its output LIVE.

    The first version used subprocess.run(stdout=PIPE) and printed the last 25
    lines afterwards. That meant a ten-minute encode showed an empty cell and
    then a wall of text. Reading by LINE would not have fixed it either: tqdm
    redraws a bar with a carriage return and emits no newline until the bar is
    finished, so a line-buffered reader stays silent for the entire bar. Read
    raw chunks and forward them byte for byte.
    """
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = []
    while True:
        chunk = os.read(p.stdout.fileno(), 8192)     # returns as soon as any byte lands
        if not chunk:
            break
        s = chunk.decode("utf-8", "replace")
        sys.stdout.write(s)
        sys.stdout.flush()
        out.append(s)
    p.wait()
    print(f"\\n[{time.time() - t0:.0f}s, exit {p.returncode}]")
    assert p.returncode == 0, f"FAILED: {cmd}"
    return "".join(out)

# 1. Fit w0,w1,w2,beta in the same Qwen3 space used by the graph model.
#    -u belt-and-braces alongside PYTHONUNBUFFERED.
sh(f"python3 -u eval/operator_scorer.py --dataset $SCIGRAPHIR_DATASET --model {OP_MODEL} "
   "--train_fit 2500 --dev 600", KGDIR)

# 1b. WARM-START THE FUSION FROM THIS CORPUS'S FIT.
#     fusion_reasoner.py hardcodes W_INIT=(1.05, 1.05, 0.25) and BETA_INIT=0.95,
#     both fitted on TOMATO. Inside the model w and beta are learnable deltas on
#     top of those, so training CAN move them -- but a SIR-4 run would still
#     begin from another corpus's calibration, and operator_scorer.py used to
#     print its own fitted values and save them nowhere. Now it writes them and
#     they get patched in here.
import json, re
P = json.load(open(f"{KGDIR}/eval/operator_params__PARAMDS__{OP_SLUG}.json"))
assert "qwen" in P.get("encoder", "").lower(), P
FR = "/content/gfm-rag/gfmrag/models/fusion_reasoner.py"
t = open(FR).read()
t2 = re.sub(r"^W_INIT = \\([^)]*\\)",
            "W_INIT = (" + ", ".join(f"{x:.4f}" for x in P["w"]) + ")", t, count=1, flags=re.M)
t2 = re.sub(r"^BETA_INIT = [\\d.]+", f"BETA_INIT = {P['beta']:.4f}", t2, count=1, flags=re.M)
assert t2 != t, "W_INIT/BETA_INIT not found in fusion_reasoner.py -- check the anchors"
open(FR, "w").write(t2)
print(f"[warm start] {DATASET}: W_INIT={tuple(round(x, 4) for x in P['w'])} "
      f"BETA_INIT={P['beta']:.4f}  (dev nDCG@10 {P['dev_ndcg10']:.3f})")
for l in open(FR):
    if l.startswith(("W_INIT", "BETA_INIT")):
        print("   ", l.rstrip())

# 2. cache the RAW ingredients (dense, S, M, total_S), column-ordered to the
#    graph's document nodes. This is what lets the four parameters stay
#    learnable inside FusionGraphReasoner instead of being frozen upstream.
for split, graph in (("train", TRAIN), ("test", TEST)):
    sh(f"python3 -u precompute/precompute_operator_components.py --model {OP_MODEL} "
       f"--dataset $SCIGRAPHIR_DATASET --model /content/qwen3 "
       f"--graph {graph} --split {split}", KGDIR)

for g in (TRAIN, TEST):
    p = f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
    assert os.path.exists(p), f"missing {p}"
    z = __import__("numpy").load(p, allow_pickle=True)
    assert "qwen" in str(z["encoder"]).lower(), f"non-Qwen operator components: {p}"
    print(f"  {g}: {os.path.getsize(p)/1e6:.0f} MB")
save_cache()'''))

cells.append(md("## 5c. Phase 4 — plain BGE baseline\n"
                "Not optional. `score_sir4.py` defines the **dissimilar** slice as "
                "*plain BGE ranks this query's best gold below 100*, so without this "
                "there is no similar/dissimilar axis at all. It reuses the document "
                "matrix Phase 3 just cached, so it is mostly a matmul."))
cells.append(code('''sh("python3 -u eval/bge_sir4.py --dataset $SCIGRAPHIR_DATASET --split test", S4)

assert os.path.exists(BGE_PRED), BGE_PRED

# Score the baseline immediately. It is the reference every fusion number is read
# against, and a broken baseline is far cheaper to spot here than after training.
args = (f"--pred {BGE_PRED} "
        f"--queries {SCIGRAPHIR_ROOT}/retriever/data/{DATASET}_test/raw/test.json "
        f"--bge {BGE_PRED} --name 'BGE baseline ({DATASET} test)' "
        f"--json-out {S4}/data/metrics_bge_{DATASET}_test.json")
if os.path.exists(SETS):
    args += f" --sets {SETS}"
sh(f"python3 -u eval/score_sir4.py {args}", S4)
!cp {S4}/data/metrics_bge_{DATASET}_test.json {OUT_ROOT}/metrics_bge_{DATASET}_test.json
!cp {BGE_PRED} {OUT_ROOT}/predictions_bge_test.json
save_cache()'''))

# THE SECOND TRAINING-FREE BASELINE. BGE alone cannot separate "the graph helps" from
# "a stronger encoder helps": every learned arm here runs on Qwen3, so a BGE-only floor
# credits the encoder swap to the method. Same script, same scorer, same slices, only
# --model differs, and the document matrix is already cached by Phase 3 so it is a matmul.
# The dissimilar slice stays defined by BGE (score_sir4.py's --bge), because that is the
# published convention and changing the slice definition would make every prior number
# incomparable.
if TOMATO:
    cells.append(md("## 5c-ii. Phase 4b — plain Qwen3 dense baseline\n"
                    "The **encoder control**. Every learned arm below runs on Qwen3, so "
                    "without this row a gain over BGE cannot be attributed: it could be "
                    "the graph, or it could be that Qwen3 is simply a better encoder than "
                    "BGE. This is the same dense retrieval as 5c with `--model` swapped, "
                    "so the difference between the two rows is the encoder and nothing "
                    "else.\n\n"
                    "The `dissimilar` slice stays defined by **BGE**, not by this run. "
                    "That definition is the published convention and every earlier number "
                    "in this project uses it; redefining it here would silently move the "
                    "slice boundary under the comparison."))
    cells.append(code('''QWEN_PRED = f"{S4}/data/predictions_qwen3_{DATASET}_test.json"
sh(f"python3 -u eval/bge_sir4.py --dataset $SCIGRAPHIR_DATASET --split test "
   f"--model {OP_MODEL} --out {QWEN_PRED}", S4)
assert os.path.exists(QWEN_PRED), QWEN_PRED

# --bge stays BGE_PRED: the slice definition must not move with the arm being scored.
args = (f"--pred {QWEN_PRED} "
        f"--queries {SCIGRAPHIR_ROOT}/retriever/data/{DATASET}_test/raw/test.json "
        f"--bge {BGE_PRED} --name 'Qwen3 dense baseline ({DATASET} test)' "
        f"--json-out {S4}/data/metrics_qwen3_{DATASET}_test.json")
if os.path.exists(SETS):
    args += f" --sets {SETS}"
sh(f"python3 -u eval/score_sir4.py {args}", S4)
!cp {S4}/data/metrics_qwen3_{DATASET}_test.json {OUT_ROOT}/metrics_qwen3_{DATASET}_test.json
!cp {QWEN_PRED} {OUT_ROOT}/predictions_qwen3_test.json
save_cache()'''))

cells.append(md("""## 5d. The proposed scorer vs the current one

**Self-contained. Sections 6 onwards are the graph run — stop after this cell.**

A popularity network predicts a paper's general matchability from its embedding
alone, `p_hat(d) = softplus(g(E(d)))`. That discounts every hypothetical-answer
match, `H~ = H / (eps + p_hat(d))**beta`. After normalisation the answer scores
are sorted descending, and one MLP reads the whole vector:

`s(q,d) = f([ x_dir, x_(1) >= x_(2) >= ... >= x_(J), J/J_max ])`

Both networks train together under **`L = L_rank + lambda_pop * L_pop`**, where
`L_rank` teaches the ranking and `L_pop` keeps `p_hat` close to measured
training-bank popularity.

| arm | pooling | popularity | zero-shot | params |
|---|---|---|---|--:|
| `dense` | none, cosine only — the floor | — | yes | 0 |
| `current` | handcrafted `sum` + `max` — the target | leave-one-query-out | **no** | 4 |
| `mlp` | sorted-input MLP | predictor trained **jointly** | yes | ~200 + 66k |

`mlp` is the proposed architecture: the predictor trains with the scorer under
`L_rank + lambda_pop * L_pop`, and it is the arm the graph fusion in 9b loads.

Two popularity variants exist for attribution and are **not run here**: `mlpbank`
(popularity measured directly against the frozen train-answer bank) and `mlp2s`
(that bank distilled into a network, then frozen). Add them to `--arms` if you want
to split credit between the pooling and the popularity term. Neither can drive the
fusion, which needs a jointly-trained predictor.

**Why popularity is part of the arm, not a separate axis.** `current` discounts a
paper using the *other test queries'* hypothetical answers, so it cannot score a
single query and is not zero-shot. The proposal replaces that with a network
reading only the paper's own embedding. The two rows therefore compare baseline
and proposal *as each is actually meant to be deployed*.

**Why `L_pop` is not optional.** Without the anchor the ranking gradient is free to
repurpose `g` as extra scorer capacity: it would stop meaning "how generally
matchable is this paper" and start meaning "whatever lowers the loss".
`--pop_lambda` sets how much drift is tolerated. The predictor is warm-started
from a two-stage fit to the bank targets, so it begins at general matchability
rather than at noise, and is rebuilt for every seed.

**Why sorting.** Hypothetical answer 3 means something different for every query,
so raw order carries no information. Sorting descending is a canonical order —
position 1 always means "best-matching answer" — which makes the input
permutation invariant while discarding nothing, unlike `sum` and `max`, which
throw the other J-2 numbers away. The arm therefore **generalises** fixed
summary-based scoring by exposing the complete ordered match profile. It does not
*contain* the baseline: each answer is centred separately before the shared
scale, so the largest sorted coordinate is not the same quantity `current`
computes its max from.

**Reading it.** `dense` says whether the hypothetical answers are worth anything at
all on this corpus. The `mlp` minus `current` row is the claim. Three seeds, and
the log prints which popularity source each arm used so you can check rather than
trust."""))
cells.append(code('''# NO GRAPH, NO REASONER, NO FUSION. Reuses only the embedding caches 5b produced.
import os, shutil, json
SEM = f"{S4}/results/semantic_{DATASET}"
os.makedirs(SEM, exist_ok=True)

# --train_fit 0 means EVERY train query left after the dev slice. The old 2,500
# cap came from operator_scorer.py, where it fitted four parameters and more data
# bought nothing; the learned arms here have 34-200 parameters and their measured
# failure mode is overfitting, so the cap was discarding 2,645 CS queries for no
# reason. --dev 300, not the operator's 600, for the same reason: dev is sliced
# first, so on matsci it is the difference between 704 and 1,004 fit queries, and
# 300 is still ample for a stable nDCG@10 selection signal.
sh(f"python3 -u eval/semantic_scorer.py --dataset $SCIGRAPHIR_DATASET --model {OP_MODEL} "
   f"--arms dense,current,mlp --loss fixed "
   f"--train_fit 0 --dev 300 --epochs 30 --patience 10 "
   f"--lr 1e-3 --weight_decay 1e-2 --mlp_hidden 16 "
   f"--mlp_pop_joint 1 --pop_lambda 1.0 "
   f"--select_on loss --qbatch 32 --seed 0 --seeds 0,1,2 --out {SEM}", S4)

# For the TWO-STAGE variant (predictor fitted to the bank targets and frozen
# before the scorer trains), rerun with --mlp_pop_joint 0. That is the cleaner
# attribution if you need to split credit between the pooling and the popularity
# term; joint is the proposed architecture.'''))

cells.append(code('''# Official evaluator on every arm, same flags, same slices.
COLS = "mrr,ndcg@5,recall@3,recall@5,recall@10,recall@25,recall@100__CSET__"
QS   = f"{SCIGRAPHIR_ROOT}/retriever/data/{DATASET}_test/raw/test.json"
LOSS = "fixedloss"           # matches --loss fixed above; "operatorloss" for --loss operator
# `dense` is untrained, so its file carries no loss tag.
# `dense` is untrained, so its file carries no loss tag.
ARMS = (("dense",   "dense query-document only",      "dense"),
        ("current", "current scorer",                 f"current_{LOSS}"),
        ("mlp",     "sorted-MLP, joint predictor",    f"mlp_{LOSS}"))

for arm, label, tag in ARMS:
    pred = f"{SEM}/predictions_semantic_{tag}_{DATASET}_test.json"
    assert os.path.exists(pred), f"missing {pred} -- did the previous cell finish?"
    args = (f"--pred {pred} --queries {QS} --cols {COLS} --name '{label}' "
            f"--json-out {SEM}/scores_semantic_{tag}.json")
    if os.path.exists(SETS):
        args += f" --sets {SETS}"
    if os.path.exists(BGE_PRED):
        args += f" --bge {BGE_PRED}"
    else:
        print("no BGE predictions -> similar/dissimilar slices skipped (run 5c)")
    sh(f"python3 -u eval/score_sir4.py {args}", S4)'''))

cells.append(code('''# One table, all three arms. Deltas are each learned arm minus `current`.
import json, os, shutil
SC = {a: json.load(open(f"{SEM}/scores_semantic_{t}.json")) for a, _, t in ARMS}
META = json.load(open(f"{SEM}/semantic_comparison_{DATASET}_{LOSS}.json"))
ROWS = ["mrr", "ndcg@5", "recall@3", "recall@5", "recall@10",
        "recall@25", "recall@100", "__CSETL__"]
# CompleteSet@k is undefined without sets.json, and the builder blanks the entry above
# rather than deleting it, so that the cell template stays valid Python either way.
ROWS = [r for r in ROWS if r]
BASE = "current"
OTHER = [a for a, _, _ in ARMS if a != BASE]

print(f"### {DATASET} test — semantic scorer, {' vs '.join(a for a, _, _ in ARMS)}")
print(f"objective: {META['loss']}")
sp = META["actual_split"]
print(f"fit={sp['fit']} dev={sp['dev']} of {sp['train_queries']} train queries; "
      f"test={sp['test_queries']}   seed={META['hyperparameters']['seed']}")
for arm, v in META["quick_summary"].items():
    print(f"  {arm:12} best dev nDCG@10 {v['dev_ndcg@10']:.4f}")

lines = []
for sl in ("all", "same", "cross", "similar", "dissimilar"):
    if any(sl not in SC[a] for a in SC):
        continue
    lines.append(f"\\n[{sl}]  n={SC[BASE][sl]['n']}")
    lines.append(f"{'metric':14}{BASE:>10}"
                 + "".join(f"{a:>11}{'delta':>9}" for a in OTHER))
    for m in ROWS:
        if any(m not in SC[a][sl] for a in SC):
            continue
        row = f"{m:14}{SC[BASE][sl][m]:10.4f}"
        for a in OTHER:
            v = SC[a][sl][m]
            row += f"{v:11.4f}{v - SC[BASE][sl][m]:+9.4f}"
        lines.append(row)
table = "\\n".join(lines)
print(table)

# A one-line verdict per arm, so the answer is not buried in the grid above.
print("\\n--- all-slice summary vs current ---")
for a in OTHER:
    d = {m: SC[a]["all"][m] - SC[BASE]["all"][m] for m in ROWS if m in SC[a]["all"]}
    won = sum(v > 0 for v in d.values())
    print(f"  {a:10} beats current on {won}/{len(d)} metrics; "
          f"mrr {d.get('mrr', 0):+.4f}  ndcg@5 {d.get('ndcg@5', 0):+.4f}  "
          f"recall@10 {d.get('recall@10', 0):+.4f}")

open(f"{SEM}/semantic_comparison_{DATASET}_{LOSS}.md", "w").write(
    f"# {DATASET} semantic scorer: {', '.join(a for a, _, _ in ARMS)}\\n"
    f"\\nfit={sp['fit']} dev={sp['dev']} test={sp['test_queries']} "
    f"seed={META['hyperparameters']['seed']} encoder={META['encoder']}\\n"
    f"\\nobjective: {META['loss']}\\n"
    f"\\n```{table}\\n```\\n")

# Persist everything to Drive: /content does not survive a runtime reset.
dst = f"{OUT_ROOT}/semantic"
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(SEM)):
    shutil.copy(f"{SEM}/{f}", f"{dst}/{f}")
print(f"\\ncopied {len(os.listdir(SEM))} file(s) to {dst}")'''))

cells.append(md("""## 5e. Cache the learned scorer so the graph fusion can use it

Section 5d trained the sorted-MLP and its popularity predictor. This packages what the
fusion needs to run that scorer as its semantic channel, aligned to the graph's document
order: the per-answer match matrix, the direct similarities, the answer-validity mask,
and the document embeddings the popularity predictor reads.

**The match matrix is referenced, not copied.** `semantic_scorer.py` already wrote it as a
memmap (0.1 GB on matsci, 1.8 GB on CS), and duplicating it per graph would cost more disk
than every other artefact here combined, plus give it a second copy that can go stale. The
npz carries its path and the permutation from corpus order into `nodes.csv` order, and the
fusion slices and permutes only the rows in each batch.

Both halves of the checkpoint are required. A scorer loaded without its predictor is a
trained readout paired with an untrained popularity, which is a model that never existed."""))
cells.append(code('''# 5d must have run: the checkpoint and the memmap both come from it.
import numpy as np          # section 7 imports it too; 5e runs first
SEM_TAG   = f"mlp_{LOSS}" if "LOSS" in dir() else "mlp_fixedloss"
SEM_CKPT  = f"{SEM}/params_semantic_{SEM_TAG}_{DATASET}.json"
SEM_POP   = f"{SEM}/popnet_semantic_{SEM_TAG}_{DATASET}.pt"
for _p, _what in ((SEM_CKPT, "scorer"), (SEM_POP, "popularity predictor")):
    assert os.path.exists(_p), (
        f"missing the trained {_what}: {_p}\\nRun section 5d first (it needs "
        f"--mlp_pop_joint 1, which writes the predictor next to the scorer).")
import json as _json
_st = _json.load(open(SEM_CKPT))
print(f"[5e] scorer: jmax={_st['jmax']} beta={_st['beta']:.4f} "
      f"hidden={len(_st['net']['0.weight'])}")

for split, graph in (("train", TRAIN), ("test", TEST)):
    sh(f"python3 -u precompute/precompute_semantic_components.py "
       f"--dataset $SCIGRAPHIR_DATASET --model {OP_MODEL} --graph {graph} --split {split}", KGDIR)

SEM_COMP  = f"{DATA_ROOT}/{TRAIN}/semantic_components{OP_SLUG}.npz"
SEM_COMP_T = f"{DATA_ROOT}/{TEST}/semantic_components{OP_SLUG}.npz"
for p in (SEM_COMP, SEM_COMP_T):
    assert os.path.exists(p), f"missing {p}"
    z = np.load(p, allow_pickle=True)
    assert "qwen" in str(z["encoder"]).lower(), f"non-Qwen semantic components: {p}"
    # The referenced memmap has to exist NOW, not at training time forty minutes in.
    assert os.path.exists(str(z["h_path"])), (
        f"{p} references {z['h_path']}, which is gone. It lives in the semantic "
        f"scorer's cache, which does not survive a runtime reset -- rerun 5d.")
    print(f"  {os.path.basename(p)}: {os.path.getsize(p)/1e6:.0f} MB, "
          f"dense{z['dense'].shape}, H at {os.path.basename(str(z['h_path']))}")'''))

cells.append(md("## 6. Structural audit — must pass before training"))
cells.append(code('''import csv, json, io
csv.field_size_limit(10 ** 7)

_CTRL = set(range(32)) - {9, 10, 13}

def rows(p):
    """DictReader over a stage1 CSV that may carry mangled unicode escapes.

    Some of these CSVs contain control bytes where a \\\\uXXXX escape was unescaped two hex
    digits at a time: "\\\\u00b7" (MIDDLE DOT) became NUL + "b7", and "\\\\u03b1" (alpha)
    became \\\\x03 + "b1". `csv.DictReader(open(p))` raises `_csv.Error: line contains NUL`
    on the ones that produced a NUL and silently accepts every other control byte, so a
    plain open() dies on some domains and quietly carries corrupt node names on the rest.
    Reading through this keeps the audit runnable AND reports the count, because a corrupt
    name cannot merge with the same concept spelled correctly elsewhere.
    """
    raw = open(p, "rb").read()
    n = sum(1 for b in raw if b in _CTRL)
    txt = raw.decode("utf-8", "replace")
    if n:
        txt = "".join(c for c in txt if ord(c) not in _CTRL)
    return list(csv.DictReader(io.StringIO(txt))), n

for name, split in ((TRAIN, "train"), (TEST, "test")):
    s1 = f"{DATA_ROOT}/{name}/processed/stage1"
    _nrows, _cn = rows(f"{s1}/nodes.csv")
    edges,  _ce = rows(f"{s1}/edges.csv")
    _rrows, _cr = rows(f"{s1}/relations.csv")
    names = {r["name"] for r in _nrows}
    rels  = {r["name"] for r in _rrows}
    q = json.load(open(f"{s1}/{split}.json"))
    assert all(e["source"] in names and e["target"] in names for e in edges), "dangling edge"
    assert {e["relation"] for e in edges} <= rels, "unknown relation"
    bad = sum(any(n not in names for v in x["start_nodes"].values() for n in v) for x in q)
    seedless = sum(not any(x["start_nodes"].values()) for x in q)
    golds = [len(x["supporting_documents"]) for x in q]
    print(f"{split}: nodes={len(names):,} edges={len(edges):,} queries={len(q):,} "
          f"unresolved_seeds={bad} seedless={seedless} "
          f"golds/query={sum(golds)/len(golds):.2f}")
    if _cn or _ce or _cr:
        print(f"  MANGLED ESCAPES: {_cn} control bytes in nodes.csv, {_ce} in edges.csv, "
              f"{_cr} in relations.csv -- stripped for this audit only; the graph the "
              f"trainer builds still carries them")
    # EVERY GOLD MUST BE A DOCUMENT NODE, even one with no extracted relations. A gold that
    # is not in the graph is unreachable by any walk, so the graph channel is scored on a
    # corpus the semantic channel does not have to share. It hits both arms equally and so
    # does not explain a CCMP-minus-control difference, but it does bias graph against
    # no-graph. Reported, not asserted: two documents should not block a run.
    _docs = {r["name"] for r in _nrows if r.get("type") == "document"}
    _gold = {g for x in q for g in x["supporting_documents"]}
    _miss = _gold - _docs
    print(f"  golds={len(_gold):,} document nodes={len(_docs):,} "
          f"golds MISSING from the graph={len(_miss)}"
          + (f"  e.g. {sorted(_miss)[:3]}" if _miss else ""))
    # A seedless query is DROPPED by the loader with only a log line, so the
    # final metric would be computed over fewer queries than reported.
    assert seedless == 0, f"{seedless} seedless queries would be silently dropped"
print("structural checks passed")'''))

cells.append(md("## 7. Run function"))
cells.append(code('''import os, subprocess, numpy as np

# BATCH SIZE. TOMATO trained at 4. SIR-4 CS cannot: measured 76.4 GB allocated on
# an 80 GB A100 at batch 4, OOM on the first step. The activations dominate and
# scale with the batch --  boundary is [batch, 243390 nodes, 1024] in bf16, i.e.
# 2 GB per copy, and return_hidden keeps all six layers for the early-late
# fusion.
#
# Halving the batch is the mildest deviation available. The alternative,
# split_graph_training=true, keeps batch 4 by partitioning the graph, but that
# severs paths crossing partition boundaries -- i.e. it damages the multi-hop
# reachability this whole method is about. Batch size only changes gradient
# noise: the objective's negatives are explicit (HARDNEG_HUB / HARDNEG_RAND),
# not drawn from the batch, so nothing else about the loss shifts.
BATCH = 2

# LOSS SETTINGS, READ FROM HERE BY BOTH ARMS. Deliberately not per-call arguments:
# section 9 and 9b must run the same objective, or "learned scorer beats operator"
# is confounded with "loss v2 beats loss v1" and neither claim survives.
#
# PER_GOLD=1        every gold must individually beat the lineup. The legacy loss
#                   pools golds in one logsumexp, which is a soft max: one easy gold
#                   satisfies the query and the buried ones get ~zero gradient. That
#                   is harmless on TOMATO (1 gold/query) and wrong on SIR-4 (3.9 to
#                   4.5), and it is the same correction `--loss fixed` already makes
#                   on the scorer side. Without it the two halves of the model train
#                   under different multi-gold assumptions.
# HARDNEG_GRAPH=50  the graph's own top-50 (detached, golds removed) join the
#                   negatives. The fusion is z(s) + gamma*relu(z(graph)), so the graph
#                   can only PROMOTE a document and its failure mode is over-promoting
#                   hubs; this trains against exactly that failure. Early on the GNN is
#                   near-random so these are just extra random negatives, and the term
#                   becomes self-adversarial as it learns.
# MISS_W_* stay OFF. The trainer's own notes record that amplifying the gradient of
#                   graph-noisy queries under the additive gate is what taught the gate
#                   backwards; MISS_W_AUX is the safer half if it is ever wanted.
PER_GOLD, HARDNEG_GRAPH = "1", "50"

# Epoch budget. Named once so the two arms, their run directories and the
# scoring cell cannot drift to different lengths -- an arm trained longer than
# the other is not a comparison.
EPOCHS = 10

def run_model(epochs, force_reload, suffix, batch=None, semantic="operator", overwrite=False,
              config="sft_training_fusion", extra=None, cqig=False,
              cqig_pool=64, cqig_m=16, cqig_norm="layer", cqig_layers="last",
              cqig_rounds=1, cqig_lam=None, cqig_link="exact", cqig_link_k=3,
              cqig_link_t=0.05, cqig_link_min="auto", cqig_op="gate", cqig_mu="ref",
              fusion_form="additive", fusion_amax=0.5, fusion_ainit=0.1,
              fusion_router="full", fusion_taug="learn", fusion_afix=None,
              fusion_topk=0, fusion_gammafix=None,
              miss_w_aux=False, miss_w_cap=None,
              resid_prior=False, resid_k=200, resid_anchor=0.1,
              ccmp=False, ccmp_w=0.1, ccmp_neg=64, ccmp_m=2000,
              ccmp_gate=True, ccmp_gate_norm=True, ccmp_eta=0.5, ccmp_lr=None,
              ccmp_residual=False, ccmp_identity_w=1.0):
    """semantic='operator' is the handcrafted scorer; 'mlp' is the learned one from 5d.

    Both arms are the SAME fusion: same gate, same relu floor, same hard-negative
    objective, same graph. Only which scorer supplies the semantic channel changes, so
    the pair is a controlled comparison rather than two different models.
    """
    batch = BATCH if batch is None else batch
    assert semantic in ("operator", "mlp"), semantic
    for g in (TRAIN, TEST):
        component_path = f"{DATA_ROOT}/{g}/operator_components{OP_SLUG}.npz"
        assert os.path.exists(component_path), f"missing {component_path}; run Section 5b"
        component_encoder = str(np.load(component_path, allow_pickle=True)["encoder"])
        assert "qwen" in component_encoder.lower(), (
            f"{component_path} was built with {component_encoder!r}, not Qwen3; rerun Section 5b")
    if semantic == "mlp":
        # Fail here, not forty minutes into a run: the learned arm needs 5d's trained
        # checkpoint AND 5e's aligned components, and neither is produced by 5b.
        for g in (TRAIN, TEST):
            p = f"{DATA_ROOT}/{g}/semantic_components{OP_SLUG}.npz"
            assert os.path.exists(p), f"missing {p}; run Section 5e"
            assert os.path.exists(str(np.load(p, allow_pickle=True)["h_path"])), (
                f"{p} references a per-answer matrix that no longer exists; rerun 5d then 5e")
        for p in (SEM_CKPT, SEM_POP):
            assert os.path.exists(p), f"missing {p}; run Section 5d with --mlp_pop_joint 1"
    run_dir = f"{OUT_ROOT}/{DATASET}_{suffix}"
    # COLLISION GUARD. A rerun with the same suffix used to overwrite the previous run's
    # weights, log, console capture and predictions in place, with no warning. That is how
    # the cqiglink09 result was lost: a later run reused the suffix and the 18:57 arm was
    # gone by 19:30. Nothing is deleted here -- an existing finished run either stops the
    # call, or is overwritten because the caller passed overwrite=True and meant it.
    if os.path.isfile(f"{run_dir}/model_best.pth") and not overwrite:
        raise FileExistsError(
            f"{run_dir} already holds a finished run (model_best.pth).\\n"
            f"Pass overwrite=True to replace it, or change `suffix` to keep both.\\n"
            f"Reruns used to clobber silently; this is the guard that stops that.")
    os.makedirs(run_dir, exist_ok=True)
    cmd = ["python","-u","-m","gfmrag.workflow.sft_training",
        "--config-path","config/gfm_reasoner","--config-name",config,
        "text_emb_model=qwen3_st", f"datasets.cfgs.root={DATA_ROOT}",
        f"datasets.cfgs.force_reload={str(force_reload)}",
        f"datasets.train_names=[{TRAIN}]", f"datasets.valid_names=[{TEST}]",
        f"trainer.args.num_epoch={epochs}",f"trainer.args.train_batch_size={batch}",
        f"model.semantic={semantic}", f"model.cqig={str(bool(cqig)).lower()}",
        "+trainer.args.do_predict=true","+trainer.args.predict_top_k=100",
        f"hydra.run.dir={run_dir}"]
    # lam_init is the gate's AUTHORITY, and it is the one CQIG knob no environment
    # variable reaches: it is a model argument, so it goes on the hydra command line.
    # g cannot fall below 1-lam, so the config default 0.1 caps the gate at a 5%
    # modulation no matter what I, alpha or tau do. Appended only when asked for, so
    # the default path is byte-identical to every run made before this argument existed.
    if cqig_lam is not None:
        assert cqig, "cqig_lam without cqig=True sets a parameter nothing reads"
        cmd.append(f"model.cqig_lam={cqig_lam}")
    cmd += list(extra or [])
    env = dict(os.environ, WANDB_MODE="disabled", HYDRA_FULL_ERROR="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               # THE KERNEL AND THE SUBPROCESS RESOLVE gfmrag DIFFERENTLY. The engine cell
               # ends with sys.path.insert(0, "/content/gfm-rag"), which fixes the
               # IN-KERNEL import and makes that cell print "G-Reasoner OK" -- but sys.path
               # does not cross a process boundary, so `python -m gfmrag.workflow
               # .sft_training` depends solely on the editable install, which runs under
               # `-q` and is therefore silent when it fails. The result is a setup section
               # that looks completely healthy and a training cell that dies with
               #   ModuleNotFoundError: No module named 'gfmrag'
               # after the run directory and manifest have already been written. Setting
               # PYTHONPATH makes the subprocess resolve the package the way the kernel
               # does, so a run no longer depends on pip's editable install having worked.
               # Harmless when it did: it is the same directory either way.
               PYTHONPATH=os.pathsep.join(
                   ["/content/gfm-rag"] + ([os.environ["PYTHONPATH"]]
                                           if os.environ.get("PYTHONPATH") else [])),
               OPERATOR_COMPONENTS=f"{DATA_ROOT}/{TRAIN}/operator_components{OP_SLUG}.npz",
               OPERATOR_COMPONENTS_TEST=f"{DATA_ROOT}/{TEST}/operator_components{OP_SLUG}.npz",
               # Per-epoch stratified eval. The slice patch in cell 6 reads BOTH of
               # these from the environment and falls back to an empty dict when a
               # path is missing or unset. Empty dicts are not a soft failure: with
               # no metadata every query takes the else-branch of BOTH tests, so it
               # is labelled `cross` AND `dissim` AND `cross+dissim`, the three
               # buckets end up holding the identical 331 queries, and `same`/`sim`
               # are never created at all. The matsci run printed three identical
               # numbers for exactly this reason. The values are still correct
               # whole-set best-gold metrics; only the slice labels are meaningless.
               STRAT_TEST=f"{DATA_ROOT}/{DATASET}_test/raw/test.json",
               STRAT_BGE=BGE_PRED,
               # the LEARNED scorer's inputs and its warm start. Harmless under
               # semantic=operator, which never reads them.
               SEMANTIC_COMPONENTS=f"{DATA_ROOT}/{TRAIN}/semantic_components{OP_SLUG}.npz",
               SEMANTIC_COMPONENTS_TEST=f"{DATA_ROOT}/{TEST}/semantic_components{OP_SLUG}.npz",
               SEMANTIC_CKPT=SEM_CKPT, SEMANTIC_POPNET=SEM_POP,
               # Same anchor weight 5d selected the checkpoint under. Dropping it here
               # would keep training the predictor under a DIFFERENT objective than the
               # one it was chosen by, and p_hat would drift off "general matchability".
               SEM_POP_LAMBDA="1.0",
               # Cross-Query Informativeness Gating. Read only when model.cqig=true.
               # CQIG_M reference queries chosen by farthest-point from a CQIG_POOL pool;
               # recalibrated every CQIG_RECAL steps because mu drifts as the model trains.
               # The SAME M individual problems calibrate every graph, so train and test
               # statistics are computed from equal-sized reference sets.
               CQIG_M=str(cqig_m), CQIG_POOL=str(cqig_pool), CQIG_RECAL="1000",
               # The two knobs that define an arm of the controlled experiment.
               #   CQIG_NORM   layer  = one median-variance scale per layer (I is
               #                        comparable BETWEEN nodes -- the actual hypothesis)
               #               node   = each node divided by its own variance, which makes
               #                        every node look equally informative (ablation)
               #               energy = the original mean-||h||^2 denominator (ablation)
               #   CQIG_LAYERS 1-INDEXED. "last", "all", "6", "3-6", "3,5,6".
               # CQIG_ROUNDS>1 re-runs calibration with the previous round's gates live,
               # which only matters when more than one layer is gated.
               CQIG_NORM=str(cqig_norm), CQIG_LAYERS=str(cqig_layers),
               CQIG_ROUNDS=str(cqig_rounds),
               #   CQIG_LINK   how the frozen source references are re-linked to a graph.
               #               exact    = string equality on entity names, which is what
               #                          the indexer does. Correct on the SOURCE graph and
               #                          near-useless off it: physics resolved 501/501 on
               #                          its train graph and 25/501 on its own test graph,
               #                          so mu was estimated from references that barely
               #                          entered the graph and nothing failed loudly.
               #               semantic = exact first, then the CQIG_LINK_K nearest target
               #                          nodes to the seed's frozen source embedding, above
               #                          a source-selected cosine floor, split by
               #                          softmax(cos / CQIG_LINK_T) and normalised so each
               #                          seed's TOTAL attachment weight is unchanged.
               # Still zero-shot: frozen encoder, floor measured on the source graph before
               # any target is seen, no target label or query touched, nothing fine-tuned.
               CQIG_LINK=str(cqig_link), CQIG_LINK_K=str(cqig_link_k),
               CQIG_LINK_T=str(cqig_link_t), CQIG_LINK_MIN=str(cqig_link_min),
               #   CQIG_OP     which OPERATOR the same statistic drives. Not a variant of
               #               the gate; a different thing done with the same mu, alpha,
               #               tau and lam, via the same forward pre-hook.
               #               gate         = h * g. The original. A damped node sends a
               #                              quieter copy of itself, direction unchanged.
               #               centre       = h - (1-g)*mu. The message is linear in the
               #                              source state and e_r never sees the query, so
               #                              this is EXACTLY "propagate the residual and
               #                              retain a g-fraction of the background". A
               #                              damped node sends its residual, not silence.
               #               centre-fixed = h - lam*mu. The same subtraction at a constant
               #                              rate: a query-INDEPENDENT hub correction, and
               #                              the rung of the ladder where alpha and dtau
               #                              stop receiving gradient.
               # lam=0 recovers the ungated reasoner under all three, so the ablation
               # remains one scalar. A centring op cannot touch a node with mu=0, i.e. one
               # no reference query reached -- exactly the nodes a gating op damps hardest.
               CQIG_OP=str(cqig_op),
               #   CQIG_MU     ref  = mu from the reference bank. The method.
               #               zero = mu forced to 0, so I = ||h||^2/s. THE ABLATION: an
               #                      activation-magnitude gate with no reference bank in
               #                      it. The reference forwards still run and are
               #                      discarded, so the pair differs in mu alone. If this
               #                      reproduces the method's score, the bank was never
               #                      load-bearing and CQIG's premise is not what improved
               #                      anything. Illegal with a centring op (h - c*0 = h).
               CQIG_MU=str(cqig_mu),
               # additive gate, operator-hard-negative objective, plus the two
               # multi-gold / graph-negative corrections defined above.
               FUSION_OBJECTIVE="hardneg", HARDNEG_HUB="50", HARDNEG_RAND="50", AUX_W="1.0",
               PER_GOLD=PER_GOLD, HARDNEG_GRAPH=HARDNEG_GRAPH,
               #   FUSION_FORM  how the two channels are COMBINED. Orthogonal to every
               #                CQIG knob above, which all change the graph channel itself.
               #     additive = z(s) + gamma_q * relu(z(graph)). The default and every
               #                run made before this option existed. Its ceiling is
               #                arithmetic, not tuning: a z-score tops out near 5-10 and
               #                the converged gamma is 0.036, so the graph can push a
               #                document by at most ~0.36 z-units against a top-50
               #                semantic spread of 1-3. It reorders neighbours; it cannot
               #                rescue a gold the semantic scorer buried.
               #     mixture  = log((1-a_q) p_s + a_q p_g) over softmaxed channels. The
               #                response is exponential, so the graph's few confident
               #                documents move hundreds of places -- and so do its
               #                confident MISTAKES, which at 0.27 nDCG@5 is most of its
               #                top-1s. FUSION_AMAX is the only bound on that.
               # FUSION_TAUG=learn learns the graph channel's sharpness; tau_s is fixed at
               # 1 on purpose (the training loss is already a softmax, so a learnable
               # tau_s would lower the loss by sharpening rather than by ranking better).
               # FUSION_ROUTER=cov drops the router back to the additive gate's single
               # feature, for isolating the form from the richer routing input.
               # FUSION_TOPK confines the graph channel to the semantic top-K, which is the
               # property the additive form has for free (a 0.36 z-unit push cannot lift a
               # document out of the tail) and the mixture does not. FUSION_GAMMAFIX pins
               # the additive arm's gamma so the 0.5340 number can be reproduced on purpose
               # rather than by whatever the bf16 freeze happened to land on.
               FUSION_FORM=str(fusion_form), FUSION_AMAX=str(fusion_amax),
               FUSION_AINIT=str(fusion_ainit), FUSION_ROUTER=str(fusion_router),
               FUSION_TAUG=str(fusion_taug),
               FUSION_TOPK=str(fusion_topk),
               FUSION_AFIX=("" if fusion_afix is None else str(fusion_afix)))
    # FUSION_GAMMAFIX is set only when asked for, NOT defaulted to "" in the update above.
    # The additive arms' run_model calls are generated elsewhere and do not pass it, so
    # writing "" here would clobber an os.environ value set from a notebook cell, which is
    # the only way to pin those arms without regenerating them.
    if fusion_gammafix is not None:
        env["FUSION_GAMMAFIX"] = str(fusion_gammafix)
    # MISS_W_FUSED stays off unconditionally. Weighting the FUSED loss by the semantic
    # miss amplifies the gradient of graph-noisy queries on the term the GATE reads, and
    # that is what taught the gate backwards. MISS_W_AUX weights only the graph-alone
    # term -- w_qg = min(cap, log1p(r_sem)) per gold, normalised by sum(w), see
    # _contrastive_hardneg -- which the gate does not read. It is exposed as an argument
    # so the experiment can run without editing this file; default stays off.
    env.pop("MISS_W_FUSED", None)
    if miss_w_aux:
        env["MISS_W_AUX"] = "1"
        if miss_w_cap is not None:
            env["MISS_W_CAP"] = str(miss_w_cap)
    else:
        env.pop("MISS_W_AUX", None)
        env.pop("MISS_W_CAP", None)
    # RESID_PRIOR REPLACES the graph-alone term's negative set with "documents the
    # parameter-free PPR already ranks above this gold". It is not a reweighting: golds the
    # walk orders correctly contribute no gradient at all, so the loss can only fall by
    # capacity the topology does not already supply. Mutually exclusive with MISS_W_AUX,
    # which reweights the same term by the OTHER channel's rank; running both would leave
    # two things different between arm and control.
    # CLEARED UNCONDITIONALLY, THEN SET. A CCMP cell run earlier in the same Colab
    # session leaves CCMP=1 in os.environ, which `dict(os.environ, ...)` copies into the
    # control's subprocess -- so the control would build a responsibility head and gate,
    # and the "one variable" pair would be two CCMP runs. Popping here makes arm order
    # irrelevant.
    for _k in ("CCMP", "CCMP_W", "CCMP_NEG", "CCMP_M", "CCMP_GATE",
               "CCMP_GATE_NORM", "CCMP_ETA", "CCMP_HID", "CCMP_CMIN", "CCMP_LR",
               # Same reason as the rest: a residual cell run earlier in the session
               # leaves these in os.environ, and dict(os.environ, ...) would carry them
               # into a plain-CCMP arm, making the pair two variables instead of one.
               "CCMP_RESIDUAL", "CCMP_IDENTITY_W"):
        env.pop(_k, None)
    if ccmp:
        assert not resid_prior, (
            "CCMP and RESID_PRIOR both rewrite what the graph is trained on; running "
            "them together makes a two-variable arm that attributes to neither")
        env.update(CCMP="1", CCMP_W=str(ccmp_w), CCMP_NEG=str(ccmp_neg), CCMP_M=str(ccmp_m),
                   CCMP_GATE="1" if ccmp_gate else "0",
                   CCMP_GATE_NORM="1" if ccmp_gate_norm else "0",
                   CCMP_ETA=str(ccmp_eta))
        # RESIDUAL CCMP. Restricts supervision to the golds the semantic scorer has not
        # already resolved, and supervises the gate to 1 on queries where it has. Default
        # off, so an unmodified run_model call reproduces the previous CCMP exactly.
        if ccmp_residual:
            env["CCMP_RESIDUAL"] = "1"
            env["CCMP_IDENTITY_W"] = str(ccmp_identity_w)
        # HEAD-SPECIFIC LEARNING RATE. The responsibility head is a fresh
        # 6 x (1024 -> 64 -> 1) MLP fitting a target from scratch while every other
        # parameter around it is warm-started, so at the trunk's 5e-4 it fits slowly and
        # a null result cannot be told apart from an under-trained head. Unset = one
        # parameter group, i.e. byte-identical to every CCMP run made before this.
        # It is popped above for the control arm, so the pair stays one-variable.
        if ccmp_lr is not None:
            env["CCMP_LR"] = str(ccmp_lr)
    if resid_prior:
        assert not miss_w_aux, "RESID_PRIOR and MISS_W_AUX both rewrite the graph-alone term"
        env["RESID_PRIOR"] = "1"
        env["RESID_K"] = str(resid_k)
        env["RESID_ANCHOR"] = str(resid_anchor)
    else:
        for _k in ("RESID_PRIOR", "RESID_K", "RESID_ANCHOR"):
            env.pop(_k, None)
    print(f"[loss] per_gold={PER_GOLD} hardneg_graph={HARDNEG_GRAPH} "
          f"hub=50 rand=50 aux_w=1.0 | semantic={semantic} cqig={cqig}"
          + (f" | FUSION={fusion_form} a_max={fusion_amax} a_init={fusion_ainit} "
             f"router={fusion_router} tau_g={fusion_taug}"
             if fusion_form != "additive" else "")
          + (f" op={cqig_op} mu={cqig_mu} norm={cqig_norm} layers={cqig_layers} rounds={cqig_rounds} "
             f"M={cqig_m} lam_init={cqig_lam if cqig_lam is not None else 'config default'}"
             f" link={cqig_link}"
             + (f" (K={cqig_link_k} T={cqig_link_t} floor={cqig_link_min})"
                if cqig_link == "semantic" else "")
             if cqig else ""))
    print("writing to", run_dir)
    # SELF-DESCRIBING RUN. Until now the only record of what an arm was, was its directory
    # name, decoded by convention: `cqiglam09` means lam 0.9, `cqigmix0509` means mixture at
    # a_max 0.5 AND lam 0.9, `qwenmlp` vs `qwenop` is the semantic scorer. Eleven suffixes
    # into matsci that is not a system. This writes the settings down so a run can be read
    # months later, and so the comparison notebook can group by ARM rather than by string
    # matching on filenames. .hydra/ already has the full config; this is the short version
    # that answers "which row of the table is this" without parsing yaml.
    import json as _json, datetime as _dt
    _manifest = {
        "suffix": suffix, "dataset": DATASET, "epochs": epochs, "batch": batch,
        "semantic": semantic, "cqig": bool(cqig), "cqig_lam": cqig_lam,
        "cqig_link": cqig_link if cqig else None, "cqig_op": cqig_op,
        "fusion_form": fusion_form, "fusion_amax": fusion_amax if fusion_form != "additive" else None,
        "fusion_afix": fusion_afix, "fusion_topk": fusion_topk or None,
        "fusion_gammafix": fusion_gammafix,
        "ccmp": ccmp or None, "ccmp_w": ccmp_w if ccmp else None,
        "ccmp_gate": ccmp_gate if ccmp else None,
        # PART OF THE ARM IDENTITY, for the same reason ccmp itself is: without it a
        # residual run and a plain CCMP run write the same manifest and a later table
        # cannot tell which one it loaded.
        "ccmp_residual": (ccmp_residual or None) if ccmp else None,
        "ccmp_identity_w": (ccmp_identity_w if ccmp_residual else None) if ccmp else None,
        # the five-arm ladder: semantic / graph / gate / fusion. Anything that does not map
        # onto a row is an ablation and says so, rather than being silently filed as one.
        # ccmp IS PART OF THE ARM IDENTITY. Without it a CCMP run and its control both
        # record "2A", so the manifest cannot tell them apart even though they are the two
        # halves of a one-variable comparison. The reporting cell reads explicit run
        # directories and is unaffected, but anything aggregating arm.json would merge them.
        "arm": ("2A+ccmp" if (not cqig and fusion_form == "additive" and ccmp) else
                "2A" if (not cqig and fusion_form == "additive") else
                "2B" if (not cqig and fusion_form == "mixture") else
                "3A" if (cqig and cqig_lam == 0.9 and fusion_form == "additive") else
                "3B" if (cqig and cqig_lam == 0.9 and fusion_form == "mixture") else
                "ablation"),
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(f"{run_dir}/arm.json", "w") as _fh:
        _json.dump(_manifest, _fh, indent=2)
    print("arm:", _manifest["arm"], "|", {k: v for k, v in _manifest.items()
                                          if v not in (None, False)})
    with open(f"{run_dir}/console.log","w") as log:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout: print(line, end=""); log.write(line); log.flush()
        rc = p.wait()
        # Report the CODE, not just "failed". A Python exception prints a
        # traceback; a process killed by the host OOM killer prints nothing at
        # all and exits -9. Without the number those two look identical in the
        # notebook, and they need completely different fixes.
        # NAME THE CODE. -9 is the host OOM killer and prints nothing; 1 is a Python
        # exception whose traceback is already above; 127 is a missing binary. Reading
        # "out of memory" off a code-1 failure sends you to the batch size when the
        # actual message is three lines up, which has cost real time.
        assert rc == 0, (
            f"training failed, exit code {rc}: " + {
                -9: "killed by the host OOM killer (no traceback above; lower BATCH)",
                1: "the subprocess raised — read the traceback printed ABOVE this line. "
                   "'No module named gfmrag' means section 3 has not run in this session",
                127: "command not found (the engine is not installed)",
            }.get(rc, "see the streamed output above"))
    return run_dir'''))

# Section 7b resolves arms from PREVIOUS runs. A --run-set build has no previous runs
# and calls already_run() nowhere, so emitting it would define a helper nobody uses under a
# heading that contradicts the header three cells above.
if LAM_ONLY and not RUN_SET:
    cells.append(md("### 7b. Resolving the runs this notebook does not repeat"))
    cells.append(code('''# Every training cell below is commented out. This binds its variable to the
# directory that run already wrote, so section 10 scores it from its saved
# predictions. It returns None when the predictions are genuinely absent, which
# makes section 10 skip the arm and say so rather than point at an empty path.
def already_run(var, suffix):
    p = f"{OUT_ROOT}/{DATASET}_{suffix}"
    ok = os.path.exists(f"{p}/predictions_{TEST}.json")
    print(f"  {'found  ' if ok else 'MISSING'}  {var:20} {p}")
    return p if ok else None'''))

cells.append(md("## 8. Smoke run (epochs=0) — builds the Qwen3 index, catches shape errors\n"
                "The index build encodes ~250k node names and is the slow one-off, 30-45 min."))
if LAM_ONLY:
    # Not run_cell: the smoke run writes no predictions, so there is nothing for
    # section 10 to bind to. What matters is that its ARTEFACT is present, because
    # skipping the only cell with force_reload=True is safe exactly when the cached
    # index restored in section 5 matches these graphs.
    cells.append(code('''# "From scratch" in the header means the four MODELS. The Qwen3 node index is not a
# model: it is a property of the GRAPH, identical for every arm, and 30-45 minutes to
# build. Section 5 restores it from the SHARED cache at outputs/<dataset>/cache, which
# is deliberately not under the run-set namespace -- putting it there once orphaned a
# perfectly good index and demanded a rebuild for a byte-identical artefact.
#
# BUILD IT IF IT IS NOT THERE. This used to be a commented-out call next to a bare
# assert, so a cache miss stopped the whole notebook and no graph arm could start --
# an overnight run that produced nothing. The index is a PREREQUISITE, so it builds
# itself; what stays manual is only the expensive-and-unnecessary case of rebuilding
# one that already matches.
def _index_of(g):
    _pr = f"{DATA_ROOT}/{g}/processed"
    return [d for d in os.listdir(_pr) if d != "stage1"] if os.path.isdir(_pr) else []

_need = [g for g in (TRAIN, TEST) if not _index_of(g)]
if _need:
    print("Qwen3 index MISSING for:", ", ".join(_need))
    print("Building it now (30-45 min, once). It is cached to Drive immediately after.")
    SMOKE = run_model(epochs=0, force_reload=True, suffix="smoke")
    save_cache()
else:
    print("Qwen3 index restored from the shared cache; nothing to build.")

for _g in (TRAIN, TEST):
    _ix = _index_of(_g)
    assert _ix, (f"index build did not produce anything under {DATA_ROOT}/{_g}/processed. "
                 f"Read the smoke run's output above -- this is a real failure, not a "
                 f"missing cache.")
    print("  index ok ", _g, "->", ", ".join(_ix))'''))
elif NOENT:
    # force_reload=True would rebuild the graph index, and the graph did not change --
    # only the query seeds did. Section 5a-noent already deleted the QA tensors, which
    # is precisely the reprocessing that IS needed, and load_qa_data regenerates a
    # missing {split}.pt without force_reload. So this is minutes, not 45 of them.
    cells.append(code('''SMOKE = run_model(epochs=0, force_reload=False, suffix="smoke")
# NO save_cache() HERE. 5a-noent disabled it on purpose; calling the original would
# push no-entity QA tensors over the entity run's under the same graph name.

# PROVE THE STRIP REACHED THE MODEL. prior/ndcg@5_walk is a parameter-free random walk
# from the query's seed nodes, so it depends on the seeding and on nothing that trains.
# The entity arms print 0.1442. An identical value here means a stale {split}.pt was
# loaded and this whole arm is measuring the entity seeds under a no-entity label.
import re as _re
_w = _re.findall(r"prior/ndcg@5_walk:\\s*([0-9.]+)", open(f"{SMOKE}/console.log").read())
print("\\nprior/ndcg@5_walk:", _w[:1] or "NOT PRINTED -- check the log by hand")
assert not _w or abs(float(_w[0]) - 0.1442) > 1e-4, (
    f"walk prior is {_w[0]}, identical to the ENTITY runs. The stripped seeds did not "
    f"reach the model: a cached processed/stage2/*/{{train,test}}.pt was reused. Re-run "
    f"section 5a-noent, confirm it reports the .pt files as dropped, and retry.")'''))
else:
    cells.append(code('SMOKE = run_model(epochs=0, force_reload=True, suffix="smoke")\n'
                      '# the Qwen3 index is the 30-45 min artefact; get it onto Drive immediately\n'
                      'save_cache()'))

if not BELIEF:
    # Under a run set run_cell() drops the section-9 operator arm, leaving this heading
    # over nothing -- a section that appears to train something and does not.
    # DROPPED ON TOMATO. The operator is a fitted scorer, so an operator+graph row is a
    # third semantic channel competing with the two training-free baselines and the
    # multi-view arm, and it costs a full training run to say something the SIR-4
    # notebooks already say. The floors here are BGE and Qwen3 dense; the graph arms all
    # run on the multi-view scorer.
    cells.append(None if (RUN_SET or TOMATO) else md("## 9. Train Qwen3-operator fusion"))
    cells.append(None if TOMATO else
                 run_cell('RUN_DIR = run_model(epochs=EPOCHS, force_reload=False,\n'
                          '                    suffix=f"fusion_qwenop_epoch{EPOCHS}_b{BATCH}")',
                          "RUN_DIR", 'f"fusion_qwenop_epoch{EPOCHS}_b{BATCH}"'))

    cells.append(md((
        "## 9b. ARM 2A — multi-view + graph, gate OFF, ADDITIVE fusion\n\nTrains `RUN_DIR_MLP`. "
        "One of the four ladder arms, not \"the main arm\": 9m trains 2B and 3B, 9e-lam trains "
        "3A, and the four are read together in the Results section at the bottom.\n" if RUN_SET
        else "## 9b. Multi-view + graph — the main arm\n") + """

    The graph half is untouched: same gate, same relu floor, same operator-hard-negative
    objective, same v16sc graph, same epochs. The only change is which scorer produces the
    semantic channel that the gate reads and the negatives are mined from.

    | | section 9 | here |
    |---|---|---|
    | semantic channel | `w0 z(dense) + w1 z(S/p^β) + w2 z(M/p^β)` | sorted-input MLP over the full match profile |
    | popularity | leave-one-out over the other test queries | predicted from the paper's embedding |
    | zero-shot | no | yes |
    | warm start | fitted w, β from 5b | trained checkpoint from 5d |

    Both the scorer and its popularity predictor keep training under the fusion's ranking
    loss, with the same `λ_pop` anchor 5d selected the checkpoint under. Set
    `model.semantic_train=false` in `run_model` to freeze them and attribute any gain to the
    graph alone.

    The 0-epoch run first: it is cheap once the index exists, and it is what catches a
    document-order or shape mismatch before a 20-epoch run does."""))
    cells.append(run_cell('SMOKE_MLP = run_model(epochs=0, force_reload=False, semantic="mlp",\n'
                          '                      suffix="fusion_qwenmlp_smoke")',
                          "SMOKE_MLP", '"fusion_qwenmlp_smoke"'))
    cells.append(run_cell('RUN_DIR_MLP = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
                          '                        suffix=f"fusion_qwenmlp_epoch{EPOCHS}_b{BATCH}")',
                          "RUN_DIR_MLP", 'f"fusion_qwenmlp_epoch{EPOCHS}_b{BATCH}"'))
    # THE CCMP ABLATION. One variable against RUN_DIR_MLP: same scorer, same fusion, same
    # gate, same objective, CCMP on. ccmp_w=1.0 matches the matsci two-stage run, whose
    # ccmp term sat at ~0.66 of a ~9.5 total loss, so the weight is doing real work rather
    # than riding along at a rounding error.
    if TOMATO:
        cells.append(md("### 9b-ii. CCMP ablation\n"
                        "One variable against 9b: identical scorer, fusion, gate and "
                        "objective, with contrastive responsibility propagation on. "
                        "Intermediate nodes are supervised by whether their remaining "
                        "bounded-hop paths lead more strongly to a gold than to the "
                        "semantic scorer's own hard negatives, and the predicted "
                        "responsibility gates outgoing messages.\n\n"
                        "On matsci this was **+0.0070 graph / +0.0025 fused**, both inside "
                        "that run's 0.0118 noise floor at a single seed. So this arm is "
                        "here to replicate a null on a second corpus, not to confirm a "
                        "gain: read it as a wash unless it clears the floor."))
        cells.append(run_cell(
            'RUN_DIR_MLP_CCMP = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
            '                             ccmp=True, ccmp_w=1.0, ccmp_neg=64, ccmp_m=2000,\n'
            '                             ccmp_gate=True, ccmp_gate_norm=True, ccmp_eta=0.5,\n'
            '                             ccmp_lr=2e-3,\n'
            '                             suffix=f"fusion_qwenmlp_ccmp_epoch{EPOCHS}_b{BATCH}")',
            "RUN_DIR_MLP_CCMP", 'f"fusion_qwenmlp_ccmp_epoch{EPOCHS}_b{BATCH}"'))
        # ---------------------------------------------------------------- residual CCMP
        # ONE VARIABLE AGAINST 9b-ii, NOT AGAINST 9b. Same scorer, fusion, gate,
        # objective, ccmp_w, ccmp_lr and eta; the only difference is which queries and
        # which golds the responsibility loss is computed on. Reading it against 9b
        # instead would confound "residual" with "CCMP at all".
        cells.append(md(
            "### 9b-iii. Residual CCMP\n"
            "Plain CCMP is supervised on **every** training query, including the ones "
            "where the semantic scorer already ranks every gold above every non-gold. On "
            "those queries there is no error for the graph to correct, so the "
            "responsibility loss can only perturb a ranking that is already right, and "
            "same-field queries are where the scorer is most often already right.\n\n"
            "This arm makes the supervision *residual*. First find the golds the scorer "
            "has left unresolved,\n\n"
            "$$\\mathcal U_q=\\left\\{g\\in\\mathcal G_q:\\ \\exists\\,"
            "d\\notin\\mathcal G_q,\\ s_{\\mathrm{sem}}(q,d)\\ge "
            "s_{\\mathrm{sem}}(q,g)\\right\\}.$$\n\n"
            "If $\\mathcal U_q\\neq\\varnothing$, $B^{+}$ is seeded from those "
            "missed golds only and $B^{-}$ from the reachable errors that outrank them, "
            "so the target asks which routes would fix *this* query's actual mistake. If "
            "$\\mathcal U_q=\\varnothing$, the head is supervised toward a constant, "
            "which makes the mean-normalised gate $g_{qv}=1$ and leaves propagation "
            "exactly as G-Reasoner would have it.\n\n"
            "**Nothing changes at inference.** No gold labels and no graph walks are "
            "needed then either way: the head has simply learned from training examples "
            "when a node state indicates a useful corrective route, and predicts a near-"
            "uniform gate otherwise.\n\n"
            "**Read `ccmp_resolved` first.** It is the share of queries now supervised as "
            "a no-op. Near 0 means this arm is the old CCMP and the comparison is empty; "
            "near 1 means almost nothing is being corrected. `ccmp_unres_frac` is how "
            "much narrower $B^{+}$ became on the rest. Then read same-field, which is "
            "what this change is meant to protect, before reading cross-field."))
        cells.append(run_cell(
            'RUN_DIR_MLP_CCMP_RESID = run_model(epochs=EPOCHS, force_reload=False,\n'
            '                             semantic="mlp",\n'
            '                             ccmp=True, ccmp_w=1.0, ccmp_neg=64, ccmp_m=2000,\n'
            '                             ccmp_gate=True, ccmp_gate_norm=True, ccmp_eta=0.5,\n'
            '                             ccmp_lr=2e-3,\n'
            '                             ccmp_residual=True, ccmp_identity_w=1.0,\n'
            '                             suffix=f"fusion_qwenmlp_ccmpresid_epoch{EPOCHS}_b{BATCH}")',
            "RUN_DIR_MLP_CCMP_RESID",
            'f"fusion_qwenmlp_ccmpresid_epoch{EPOCHS}_b{BATCH}"'))

if BELIEF:
    cells.append(md("""## 9c. H3 — semantic-prior graph reasoning (no fusion)

    The semantic scorer stops producing a ranking. It produces the graph's **initial belief**
    over document nodes, and the graph revises it:

    $$b^{(0)}_{qd}=z\\bigl(s_{\\text{sem}}(q,d)\\bigr),\\qquad
      h^{(0)}_{qd}\\mathrel{+}= W_b\\,b^{(0)}_{qd},\\qquad
      S(q,d)=b^{(0)}_{qd}+R_\\theta\\bigl(h^{(L)}_{qd}\\bigr)$$

    No gate, no budget, no weighted average, no rerank stage. The difference from section 9b
    is not the arithmetic at the end but that **the graph's computation now depends on the
    semantic belief**: `b^(0)` enters at layer 0 and propagates, so a plausible paper can
    raise the belief of a textually distant paper through a shared method or mechanism. In
    the fusion the two channels were formed in ignorance of each other and met only at the
    end, so no such path existed.

    It also removes a defect in the stock reasoner: document nodes currently begin with **no
    query-specific information at all**, so the graph must transport relevance across several
    hops before it can rank anything. Here every document starts at the best available
    estimate and the graph computes only the correction.

    `R_theta` starts near zero, so at step 0 the model *is* the semantic scorer and departs
    only where that lowers the retrieval loss. That is a property of the architecture, not a
    gate that has to learn to stay shut.

    **Two ablations, same class**: `model.use_prior=false` gives the stock graph reasoner,
    `model.use_graph=false` gives the semantic scorer alone. Both nest, so all three arms
    come from one implementation."""))
    cells.append(code('SMOKE_BEL = run_model(epochs=0, force_reload=False, semantic="mlp",\n'
                      '                      config="sft_training_belief",\n'
                      '                      suffix="belief_smoke")'))
    cells.append(code('RUN_DIR_BEL = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
                      '                        config="sft_training_belief",\n'
                      '                        suffix=f"belief_epoch{EPOCHS}_b{BATCH}")'))

    cells.append(md("""### 9d. Control — the same model without the prior

`model.use_prior=false` removes the injection, so document nodes start at zero exactly as
they do in the stock reasoner, and the residual readout is left with nothing but the
graph's own propagation to revise. Everything else is identical: same graph, same
objective, same negatives, same epochs, same seed.

This is the arm that isolates the hypothesis. If H3 beats it, the gain came from seeding
documents with semantic belief and not from anything else about the architecture. Run it
before believing any headline number."""))
    cells.append(code('RUN_DIR_NOPRIOR = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
                      '                            config="sft_training_belief",\n'
                      '                            extra=["model.use_prior=false"],\n'
                      '                            suffix=f"belief_noprior_epoch{EPOCHS}_b{BATCH}")'))

# Which layers the multi-layer CQIG arm gates, 1-INDEXED. Named once so the arm label,
# the run cell and the memory note in 9e cannot drift to different depths. The same value
# on every domain, so the multi-layer result means the same thing in each notebook; the
# cost of that is memory, which 9e tabulates and the trainer warns about.
CQIG_ML = "3-6"

ARMS = ([("semantic-prior graph (H3)", 'globals().get("RUN_DIR_BEL")', 'f"belief_epoch{EPOCHS}"'),
  ("no-prior control",        'globals().get("RUN_DIR_NOPRIOR")', 'f"belief_noprior_epoch{EPOCHS}"')]
 if BELIEF else
 ([] if TOMATO else
  [("operator + graph",   'globals().get("RUN_DIR")', 'f"fusion_qwenop_epoch{EPOCHS}"')])
 + [("multi-view + graph", 'globals().get("RUN_DIR_MLP")', 'f"fusion_qwenmlp_epoch{EPOCHS}"')]
 + ([("multi-view + graph + CCMP", 'globals().get("RUN_DIR_MLP_CCMP")',
      'f"fusion_qwenmlp_ccmp_epoch{EPOCHS}"'),
     ("multi-view + graph + CCMP (residual)",
      'globals().get("RUN_DIR_MLP_CCMP_RESID")',
      'f"fusion_qwenmlp_ccmpresid_epoch{EPOCHS}"')] if TOMATO else [])
 + [("multi-view + graph + CQIG", 'globals().get("RUN_DIR_CQIG")',
     'f"fusion_qwenmlp_%s_epoch{EPOCHS}"' % ("cqig" + LAM_SLUG if TOMATO else "cqig"))]
 # The two CQIG ablation cells are not emitted on a TOMATO build, so naming them here
 # would put two rows in the results table that can never fill.
 + ([] if TOMATO else
    [("CQIG ablation: per-node norm", 'globals().get("RUN_DIR_CQIG_NODE")',
      'f"fusion_qwenmlp_cqignode_epoch{EPOCHS}"'),
     ("CQIG multi-layer (%s)" % CQIG_ML, 'globals().get("RUN_DIR_CQIG_ML")',
      'f"fusion_qwenmlp_cqigml_epoch{EPOCHS}"')]))
A4_FUSION = """ARMS4 = [("operator",         f"{SEMD}/scores_semantic_current_{LOSSD}.json"),
         ("multi-view",       f"{SEMD}/scores_semantic_mlp_{LOSSD}.json"),
         ("operator+graph",   f"{globals().get('RUN_DIR') or '_none_'}/scores.json"),
         ("multi-view+graph", f"{globals().get('RUN_DIR_MLP') or '_none_'}/scores.json")]
# The CQIG arms, in the order of the 9e-i table. Any that was not run is skipped.
ARMS4 += [("mv+graph+CQIG",        f"{globals().get('RUN_DIR_CQIG') or '_none_'}/scores.json"),
          ("mv+graph+CQIG node",   f"{globals().get('RUN_DIR_CQIG_NODE') or '_none_'}/scores.json"),
          ("mv+graph+CQIG ML",     f"{globals().get('RUN_DIR_CQIG_ML') or '_none_'}/scores.json")]"""
# TOMATO ARM SET. Two training-free dense floors (BGE, Qwen3), then the multi-view scorer
# with the graph off and on, then the gate. No operator row: the floors are the untrained
# encoders, so every delta below them is attributable to something this project built.
# BGE vs Qwen3 is the encoder control -- without it, "graph beats BGE" is confounded with
# "Qwen3 beats BGE", since every learned arm runs on Qwen3.
A4_FUSION_TOMATO = """ARMS4 = [("BGE dense",        f"{S4}/data/metrics_bge_{DATASET}_test.json"),
         ("Qwen3 dense",      f"{S4}/data/metrics_qwen3_{DATASET}_test.json"),
         ("multi-view",       f"{SEMD}/scores_semantic_mlp_{LOSSD}.json"),
         ("multi-view+graph", f"{globals().get('RUN_DIR_MLP') or '_none_'}/scores.json"),
         ("mv+graph+CCMP",    f"{globals().get('RUN_DIR_MLP_CCMP') or '_none_'}/scores.json"),
         ("mv+graph+CCMPresid", f"{globals().get('RUN_DIR_MLP_CCMP_RESID') or '_none_'}/scores.json"),
         ("mv+graph+CQIG",    f"{globals().get('RUN_DIR_CQIG') or '_none_'}/scores.json")]"""
C_FUSION_TOMATO = """CONTRASTS = [("BGE dense",       "Qwen3 dense",      "ENCODER CONTROL: BGE -> Qwen3"),
             ("Qwen3 dense",    "multi-view",       "learned scorer over dense"),
             ("multi-view",     "multi-view+graph", "THE GRAPH"),
             ("multi-view+graph", "mv+graph+CCMP",  "ABLATION: CCMP"),
             # Against plain CCMP, so the contrast isolates "residual" from "CCMP at all".
             ("mv+graph+CCMP", "mv+graph+CCMPresid", "ABLATION: CCMP residual"),
             ("multi-view+graph", "mv+graph+CCMPresid", "ABLATION: CCMP residual vs no CCMP"),
             ("multi-view+graph", "mv+graph+CQIG",  "THE HYPOTHESIS: CQIG")]"""
A4_BELIEF = """ARMS4 = [("multi-view",        f"{SEMD}/scores_semantic_mlp_{LOSSD}.json"),
         ("no-prior control",  f"{globals().get('RUN_DIR_NOPRIOR') or '_none_'}/scores.json"),
         ("semantic-prior H3", f"{globals().get('RUN_DIR_BEL') or '_none_'}/scores.json")]"""
C_FUSION = """CONTRASTS = [("operator",       "operator+graph",   "graph adds, operator"),
             ("multi-view",     "multi-view+graph", "graph adds, multi-view"),
             ("operator",       "multi-view",       "scorer swap, no graph"),
             ("operator+graph", "multi-view+graph", "scorer swap, with graph")]
CONTRASTS += [("multi-view+graph",  "mv+graph+CQIG",     "THE HYPOTHESIS: CQIG"),
              ("mv+graph+CQIG node", "mv+graph+CQIG",    "normalisation: node -> layer"),
              ("mv+graph+CQIG",      "mv+graph+CQIG ML", "gating earlier layers too")]"""

if LAM_ONLY:
    # THE TWO NEW ARMS, and the three one-variable contrasts they support. Each new arm
    # differs from its own baseline in exactly one thing, so neither delta has to be
    # attributed by argument: lam alone, then the linker alone on top of that lam.
    _LAB_LAM = f"CQIG lam{LAM_SLUG}"
    _LAB_LNK = f"CQIG lam{LAM_SLUG}+link"
    ARMS += [(f"CQIG lam={LAM}", 'globals().get("RUN_DIR_CQIG_LAM")',
              'f"fusion_qwenmlp_cqiglam%s_epoch{EPOCHS}"' % LAM_SLUG),
             (f"CQIG lam={LAM} + semantic seeding", 'globals().get("RUN_DIR_CQIG_LINK")',
              'f"fusion_qwenmlp_cqiglink%s_epoch{EPOCHS}"' % LAM_SLUG)]
    A4_FUSION += (
        "\nARMS4 += [(%r, f\"{globals().get('RUN_DIR_CQIG_LAM') or '_none_'}/scores.json\"),"
        "\n          (%r, f\"{globals().get('RUN_DIR_CQIG_LINK') or '_none_'}/scores.json\")]"
        % (_LAB_LAM, _LAB_LNK))
    C_FUSION += (
        "\nCONTRASTS += [(%r, %r, %r),"
        "\n              (%r, %r, %r),"
        "\n              (%r, %r, %r)]"
        % ("mv+graph+CQIG", _LAB_LAM, f"lam 0.1 -> {LAM} (authority)",
           _LAB_LAM, _LAB_LNK, "exact -> semantic seeding",
           "multi-view+graph", _LAB_LNK, "THE HYPOTHESIS, both fixes"))
    if OP_ONLY:
        # The operator arm's control is the LINK arm, not the lam arm: both carry this lam
        # and semantic seeding, so the pair differs in the operator alone. Against 9e-lam it
        # would differ in two things and the delta could not be attributed.
        _LAB_CTR = f"CQIG {OP_SUFFIX}{LAM_SLUG}"
        ARMS += [(f"CQIG {OP} lam={LAM}", 'globals().get("RUN_DIR_CQIG_CTR")',
                  'f"fusion_qwenmlp_cqig%s_epoch{EPOCHS}"' % CTR_SLUG)]
        A4_FUSION += (
            "\nARMS4 += [(%r, f\"{globals().get('RUN_DIR_CQIG_CTR') or '_none_'}/scores.json\")]"
            % _LAB_CTR)
        C_FUSION += (
            "\nCONTRASTS += [(%r, %r, %r)]"
            % (_LAB_LNK, _LAB_CTR, f"scale -> centre (op={OP})"))
    if MU0:
        # The ablation's control is the EXACT-linking lam arm, which it matches in every
        # respect but mu. Signed so a NEGATIVE delta means the bank was load-bearing.
        _LAB_MU0 = f"CQIG muz{LAM_SLUG}"
        ARMS += [(f"CQIG mu=0 lam={LAM}", 'globals().get("RUN_DIR_CQIG_MU0")',
                  'f"fusion_qwenmlp_cqigmuz%s_epoch{EPOCHS}"' % LAM_SLUG)]
        A4_FUSION += (
            "\nARMS4 += [(%r, f\"{globals().get('RUN_DIR_CQIG_MU0') or '_none_'}/scores.json\")]"
            % _LAB_MU0)
        C_FUSION += (
            "\nCONTRASTS += [(%r, %r, %r)]"
            % (_LAB_LAM, _LAB_MU0, "ABLATION: remove the reference bank"))
    if MIX:
        # The mixture arm's control is the LINK arm: same lam, same linker, same graph,
        # same objective, so the pair differs in the fusion form alone. Against the lam
        # arm it would differ in the linker as well and the delta could not be attributed.
        _LAB_MIX = f"fusion mix a_max={AMAX:g}"
        ARMS += [(f"mixture fusion (a_max={AMAX:g}, lam={LAM})",
                  'globals().get("RUN_DIR_MIX")',
                  'f"fusion_qwenmlp_cqig%s_epoch{EPOCHS}"' % MIX_SLUG)]
        A4_FUSION += (
            "\nARMS4 += [(%r, f\"{globals().get('RUN_DIR_MIX') or '_none_'}/scores.json\")]"
            % _LAB_MIX)
        C_FUSION += (
            "\nCONTRASTS += [(%r, %r, %r),"
            "\n              (%r, %r, %r)]"
            % (_LAB_LNK, _LAB_MIX,
               "additive z-scores -> mixture of distributions (FUSION FORM)",
               # Second row is the one to quote at anyone asking "is this better than
               # the fusion we had". It is NOT an attribution: it stacks lam 0.9, the
               # semantic linker AND the fusion form, so a win here does not say which
               # of the three did it. The row above is the attributable one.
               "multi-view+graph", _LAB_MIX,
               "vs plain additive fusion (3 changes, not attributable)"))

if RUN_SET and not BELIEF:
    # OVERRIDE, not append. Everything accumulated above names RUN_DIR_CQIG /
    # RUN_DIR_CQIG_NODE / RUN_DIR_CQIG_ML / RUN_DIR_CQIG_LINK, whose training cells
    # run_cell() drops under a run set, and it omits RUN_DIR_MIX / RUN_DIR_MIX_LAM
    # entirely -- so the two mixture arms trained above never reached the official
    # scorer and never got a scores.json. A ladder notebook scores the ladder: the
    # four trained arms, and nothing that is not in it.
    ARMS = [("2A  graph, no gate, additive", 'globals().get("RUN_DIR_MLP")',
             'f"fusion_qwenmlp_epoch{EPOCHS}"'),
            ("2B  graph, no gate, mixture", 'globals().get("RUN_DIR_MIX")',
             'f"fusion_qwenmlp_mix_epoch{EPOCHS}"'),
            ("3A  graph, lam=%s, additive" % LAM, 'globals().get("RUN_DIR_CQIG_LAM")',
             'f"fusion_qwenmlp_cqiglam%s_epoch{EPOCHS}"' % LAM_SLUG),
            ("3B  graph, lam=%s, mixture" % LAM, 'globals().get("RUN_DIR_MIX_LAM")',
             'f"fusion_qwenmlp_cqiglam%s_mix_epoch{EPOCHS}"' % LAM_SLUG)]
    # ARM 1 IS 5d's STANDALONE mlp SCORE, not a column of a fusion run. Section 5d
    # fits and scores the multi-view scorer with no graph anywhere in the objective,
    # which is what "graph reasoning: off" means. Reading arm 1 off 2A's semantic
    # channel instead would report a scorer that trained jointly with a graph, so the
    # 2A-minus-1 delta would understate the graph by whatever the joint training gave
    # the scorer -- the baseline would be contaminated by the arm it is the control for.
    A4_FUSION = """ARMS4 = [("1   multi-view, no graph", f"{SEMD}/scores_semantic_mlp_{LOSSD}.json"),
         ("2A  no gate, additive", f"{globals().get('RUN_DIR_MLP')     or '_none_'}/scores.json"),
         ("2B  no gate, mixture",  f"{globals().get('RUN_DIR_MIX')     or '_none_'}/scores.json"),
         ("3A  lam, additive",     f"{globals().get('RUN_DIR_CQIG_LAM') or '_none_'}/scores.json"),
         ("3B  lam, mixture",      f"{globals().get('RUN_DIR_MIX_LAM') or '_none_'}/scores.json")]
# The operator row is kept only if section 9 was run; under a run set it is not.
if globals().get("RUN_DIR"):
    ARMS4.insert(1, ("operator, no graph", f"{SEMD}/scores_semantic_current_{LOSSD}.json"))"""
    # Every contrast moves exactly one thing. The 2x2 supports four such pairs plus the
    # two "does the graph help at all" rows; nothing here stacks two changes.
    C_FUSION = """CONTRASTS = [
    ("1   multi-view, no graph", "2A  no gate, additive", "graph adds (additive)"),
    ("1   multi-view, no graph", "2B  no gate, mixture",  "graph adds (mixture)"),
    ("2A  no gate, additive",   "3A  lam, additive",      "CQIG, additive held fixed"),
    ("2B  no gate, mixture",    "3B  lam, mixture",       "CQIG, mixture held fixed"),
    ("2A  no gate, additive",   "2B  no gate, mixture",   "FUSION FORM, gate off"),
    ("3A  lam, additive",       "3B  lam, mixture",       "FUSION FORM, gate on")]"""

# The first row is the hypothesis: identical everything except the prior injection.
C_BELIEF = """CONTRASTS = [("no-prior control", "semantic-prior H3", "THE HYPOTHESIS: prior injection"),
             ("multi-view",       "semantic-prior H3", "graph revision over the scorer"),
             ("multi-view",       "no-prior control",  "graph alone, no prior")]"""
NEED_FUSION = '"no arms available: run 5d, then 9 and 9b"'
NEED_BELIEF = '"no arms available: run 5d, then 9c and 9d"'
TN = ('"semantic prior ablation"' if BELIEF else
      '"five-arm ladder"' if RUN_SET else '"scorer x graph"')
TS = ('"prior_ablation"' if BELIEF else
      '"ladder"' if RUN_SET else '"scorer_x_graph"')
if CCMP:
    # OVERRIDE. Section 9c trains the control plus one arm per hypothesis, and the table has
    # exactly those rows plus the semantic floor. Everything the ladder accumulated above
    # names run directories this notebook never creates.
    _CSFX = "fusion_qwenmlp%s_ccmp" % (("_cqiglam" + LAM_SLUG) if LAM_ONLY else "")
    # NO _b{BATCH} IN THE out_tag. _resolve() appends "_b{BATCH}" itself, so a tag that
    # already carries it built ".._epoch10_b1_b1" and the restart-recovery path never
    # existed: a disconnected session would report the arm as not run and retrain it. The
    # ladder's tags (f"fusion_qwenmlp_epoch{EPOCHS}") are the convention this now matches.
    ARMS = [("control (endpoint loss only)", 'globals().get("RUN_DIR_CCMP_CTRL")',
             'f"%sctrl_epoch{EPOCHS}"' % _CSFX),
            ("CCMP (+ responsibility, gated)", 'globals().get("RUN_DIR_CCMP")',
             # ccmpSTRONG: ccmp_w=1.0 with the head at 2e-3 is a different experiment from
             # the ccmp_w=0.1 runs that already carry the ccmpon name. One name for both
             # would let a table load either and label it "CCMP".
             'f"%sstrong_epoch{EPOCHS}"' % _CSFX)]
    A4_FUSION = """ARMS4 = [("multi-view, no graph", f"{SEMD}/scores_semantic_mlp_{LOSSD}.json"),
         ("control", f"{globals().get('RUN_DIR_CCMP_CTRL') or '_none_'}/scores.json"),
         ("CCMP",     f"{globals().get('RUN_DIR_CCMP')      or '_none_'}/scores.json")]"""
    # (lo, hi, label), NOT (label, lo, hi). The consumer tests `c[0] in S and c[1] in S` and
    # prints S[hi] - S[lo] with c[2] as the caption. With the label first, c[0] was never an
    # arm name, so EVERY contrast landed in `dead` and the table printed
    # "NO CONTRAST ... needs CCMP - control" instead of a single delta.
    C_FUSION = """CONTRASTS = [("control", "CCMP", "CCMP - control"),
             ("multi-view, no graph", "control", "control - no graph"),
             ("multi-view, no graph", "CCMP", "CCMP - no graph")]"""
    NEED_FUSION = '"no arms available: run 5d, then 9c-i and 9c-ii"'
    TN, TS = '"CCMP vs control"', '"ccmp"'
    if CQIG_ARM:
        # THIRD ROW, SAME CONTROL. Added here and not as its own notebook so that the two
        # hypotheses are measured against one control on one checkpointing schedule. A
        # separate notebook would retrain a second control and the two arms would then be
        # compared through different baselines.
        ARMS.append(("CQIG lam=%s, exact links" % CQIG_ARM_LAM,
                     'globals().get("RUN_DIR_CQIG")',
                     'f"fusion_qwenmlp_cqigonly_epoch{EPOCHS}"'))
        # WRITTEN OUT, not sliced onto the two-arm version. Trimming "]" off the end of a
        # generated code string and appending a row is how the emitted ARMS4 lost a closing
        # parenthesis: the builder's own parse check runs over `cells` BEFORE these blobs are
        # substituted in, so it cannot see a break introduced here.
        A4_FUSION = """ARMS4 = [("multi-view, no graph", f"{SEMD}/scores_semantic_mlp_{LOSSD}.json"),
         ("control", f"{globals().get('RUN_DIR_CCMP_CTRL') or '_none_'}/scores.json"),
         ("CCMP",    f"{globals().get('RUN_DIR_CCMP')      or '_none_'}/scores.json"),
         ("CQIG",    f"{globals().get('RUN_DIR_CQIG')      or '_none_'}/scores.json")]"""
        # (lo, hi, label): the printed value is S[hi] - S[lo]. See the note above.
        C_FUSION = """CONTRASTS = [("control", "CCMP", "CCMP - control"),
             ("control", "CQIG", "CQIG - control"),
             ("CQIG", "CCMP", "CCMP - CQIG"),
             ("multi-view, no graph", "control", "control - no graph"),
             ("multi-view, no graph", "CCMP", "CCMP - no graph"),
             ("multi-view, no graph", "CQIG", "CQIG - no graph")]"""
        NEED_FUSION = '"no arms available: run 5d, then 9c-i, 9c-ii and 9c-iii"'
        TN, TS = '"CCMP and CQIG vs one control"', '"ccmp_cqig"'

if RESIDP:
    # OVERRIDE. Section 9r trains exactly two models and the table has exactly two rows
    # plus the semantic floor. Everything the ladder accumulated above names run
    # directories this notebook never creates.
    _RSFX = "fusion_qwenmlp_cqiglam%s_resid" % LAM_SLUG
    ARMS = [("control (standard graph loss)", 'globals().get("RUN_DIR_RESID_CTRL")',
             'f"%sctrl_epoch{EPOCHS}"' % _RSFX),
            ("residual-prior graph loss", 'globals().get("RUN_DIR_RESID")',
             'f"%son_epoch{EPOCHS}"' % _RSFX),
            ("residual-prior + mixture", 'globals().get("RUN_DIR_RESID_MIX")',
             'f"%smix_epoch{EPOCHS}"' % _RSFX)]
    A4_FUSION = """ARMS4 = [("multi-view, no graph", f"{SEMD}/scores_semantic_mlp_{LOSSD}.json"),
         ("control",  f"{globals().get('RUN_DIR_RESID_CTRL') or '_none_'}/scores.json"),
         ("residual", f"{globals().get('RUN_DIR_RESID')      or '_none_'}/scores.json"),
         ("resid+mix", f"{globals().get('RUN_DIR_RESID_MIX') or '_none_'}/scores.json")]"""
    C_FUSION = """CONTRASTS = [
    ("multi-view, no graph", "control",  "what the graph adds at all"),
    ("control",              "residual", "THE ARM: negatives restricted to the prior's mistakes"),
    ("residual",             "resid+mix", "same loss, mixture fusion instead of additive")]"""
    NEED_FUSION = '"no arms available: run 5d, then 9r"'
    TN, TS = '"residual-prior graph loss"', '"resid_prior"'

ARMS_SRC = "[" + ",\n        ".join(f'("{n}", {g}, {t})' for n, g, t in ARMS) + "]"


if not BELIEF:
    cells.append(md("""## 9e. Multi-view + graph + CQIG

Background for arms 3A and 3B. **The control is arm 2A in 9b**, gate off, and the two
gated arms below are read against it. This section trains nothing on its own; 9e-lam
trains 3A and 9m trains 3B.

Identical to 9b in every respect except that each node's outgoing messages are damped when
the node responds to this query much as the nodes of this graph *typically* respond to a
fixed set of unrelated scientific problems.

$$\\mu_v^{(\\ell)}=\\tfrac1M\\sum_m h_{a_mv}^{(\\ell)},\\qquad
V_v^{(\\ell)}=\\tfrac1M\\sum_m\\bigl\\lVert h_{a_mv}^{(\\ell)}-\\mu_v^{(\\ell)}\\bigr\\rVert^2,\\qquad
s^{(\\ell)}=\\operatorname*{median}_{v\\,:\\,V_v^{(\\ell)}>\\delta}V_v^{(\\ell)}$$

$$I_{qv}^{(\\ell)}=\\frac{\\lVert h_{qv}^{(\\ell)}-\\mu_v^{(\\ell)}\\rVert^2}{s^{(\\ell)}+\\epsilon},
\\qquad g_{qv}^{(\\ell)}=1-\\lambda_\\ell\\bigl[1-\\sigma\\bigl(\\alpha_\\ell(\\log(1+I_{qv}^{(\\ell)})-\\tau_\\ell)\\bigr)\\bigr],
\\qquad \\tilde m_{v\\to u}^{(\\ell)}=g_{qv}^{(\\ell)}\\,m_{v\\to u}^{(\\ell)}$$

**The denominator is one scale per layer, not per node.** Dividing by a node's own variance
is self-cancelling: a generic node that always moves by 0.01 and a discriminative node that
always moves by 1.0 both come out at 1, every node looks equally informative, and the gate
degenerates into a constant rescaling. Against a layer-wide scale a weak generic response is
small, a strong query-specific one is large, and a node that never responds stays at zero.
`CQIG_NORM=node` and `=energy` keep the two earlier denominators as ablations.

**The threshold is measured, not derived.** Calibration runs two reference passes: the
first builds $\\mu$, $V$ and $s$; the second runs the same references back through the
calibrated gate and sets $\\tau_{\\text{ref}}$ to the median $\\log(1+I)$ over the
(node, query) pairs that respond at all. So $\\Delta\\tau$ is a learned offset in units of
"a typical reference query at a typical responding node".

**What it adds:** 3 scalars per gated layer. Nothing is keyed to a node id, so the same
trained model transfers to an unseen graph after one unlabelled calibration pass, the way
BM25 needs collection statistics when you index a new corpus.

**Where it attaches:** a forward pre-hook on each conv layer. The DistMult message is
`input_j * relation_j`, linear in the source state, so scaling `h_v` before the layer is
exactly equivalent to scaling every message leaving `v`. The boundary condition is a
separate argument and the residual is added outside the layer, so neither is touched.

**Reference set (zero-shot protocol).** The first `CQIG_POOL=64` training queries are
pooled, then exactly `CQIG_M=16` are chosen by greedy farthest-point on the frozen question
embeddings and frozen for good. They are stored as *question embedding + start-node names*,
so every other graph is calibrated by re-linking those same problems against its own
vocabulary. The target corpus supplies its graph and its vocabulary and nothing else; its
queries are never inspected and its labels never enter anything. `mu` drifts as the model
trains, so calibration repeats every `CQIG_RECAL=1000` steps.

**What to read in the log.** Every calibration now prints, per gated layer:
`V` quantiles and the layer `scale`; the distribution of `I`; and the full gate
distribution `min/p10/p50/p90/max`, `std` and `span`, in scientific notation, next to
$\\lambda,\\alpha,\\tau$. A gate whose whole range is 0.95 to 0.95 is a constant, however
healthy `I` looks. Each `evaluate()` then prints the same distribution over the queries
actually scored, plus the decisive number:

> `informativeness AUC, gold vs semantic-top-50`

the probability that a random gold paper carries higher `I` than a random hard negative.
**0.5 means the premise is wrong**, not the tuning: no $\\lambda,\\alpha,\\tau$ can turn an
uninformative statistic into a useful gate. Judge this arm on **graph-only nDCG@5** from the
`[diag]` line first; the fusion weight $\\gamma$ cannot be expected to grow while the graph
signal itself is no better."""))
    cells.append(run_cell(
        '# epochs=0 would run no train_step: no reference pool, no calibration, no gating,\n'
        '# i.e. it would silently verify the UNGATED model. One short epoch with a small\n'
        '# pool builds the bank, calibrates the train graph, then evaluates on the TEST\n'
        '# graph -- the path where train/test statistics must not be confused.\n'
        '# Expect: the reference bank line, a calibration block for the train graph, one\n'
        '# for the test graph, and an [cqig] eval gate + AUC line.\n'
        'SMOKE_CQIG = run_model(epochs=1, force_reload=False, semantic="mlp", cqig=True,\n'
        '                       cqig_pool=8, cqig_m=4,\n'
        '                       extra=["trainer.args.max_steps_per_epoch=12"],\n'
        '                       suffix="fusion_qwenmlp_cqig_smoke")',
        "SMOKE_CQIG", '"fusion_qwenmlp_cqig_smoke"'))
    # This table names RUN_DIR_CQIG_NODE / RUN_DIR_CQIG / RUN_DIR_CQIG_ML, whose cells
    # run_cell() drops under --run-set. Left in, it described a quartet that does not
    # exist in the notebook, next to a 9m section describing the quartet that does --
    # three different "four arms" in one file. Suppressed rather than reworded: under a
    # run set the ladder table in the header is the only arm inventory.
    cells.append(None if RUN_SET else md("""### 9e-i. The controlled comparison

Four matched models, same `seed: 1024`, same data, same objective, same epoch budget. Only
the normalisation and the set of gated layers move.

| arm | normalisation | gated layers | cell |
|---|---|---|---|
| graph control | none (no gate at all) | none | 9b, `RUN_DIR_MLP` |
| CQIG, per-node | node's own variance | 6 | `RUN_DIR_CQIG_NODE` |
| **CQIG, corrected** | **layer median variance** | **6** | **`RUN_DIR_CQIG`** |
| CQIG, multi-layer | layer median variance | __CQIG_ML__ | `RUN_DIR_CQIG_ML` |

Layer numbers are 1-indexed, so "6" is the last of six and "__CQIG_ML__" is the second half.
The multi-layer arm gates upstream of where it calibrates, so it runs `cqig_rounds=2`: the
second round re-runs the whole calibration with the first round's gates live, which removes
the stale centering. The single-layer arms are exact at one round, because a layer's own
gate cannot change its own input.

The corrected arm is the one to run first. The other two exist to attribute whatever it
does: without the per-node arm you cannot say the normalisation was the fix, and without the
multi-layer arm you cannot say one gated layer was too late to matter.

**Memory.** The reference statistics are one float32 row per node per gated layer per graph,
held on the GPU. That is negligible for the single-layer arms and not for the multi-layer
one on the larger corpora, so the next cell prints this corpus's number before you commit
40 minutes to a run that may not fit. If it does not, gate fewer layers (`cqig_layers="5-6"`)
rather than dropping the arm."""))
    cells.append(None if RUN_SET else code(
        'import csv\n'
        '# mu is [n_nodes, 1024] float32 per gated layer, per graph, resident on the GPU.\n'
        '# Both graphs are calibrated (train during training, test at every evaluate), so\n'
        '# the resident cost is the sum of the two.\n'
        'def _cqig_mem(layers):\n'
        '    tot = 0\n'
        '    for g in (TRAIN, TEST):\n'
        '        p = f"{DATA_ROOT}/{g}/processed/stage1/nodes.csv"\n'
        '        n = sum(1 for _ in open(p)) - 1\n'
        '        tot += n * 1024 * 4 * layers\n'
        '        print(f"  {g:34} {n:>8,} nodes")\n'
        '    return tot / 2**30\n'
        'for _L in (1, 4):\n'
        '    print(f"{_L} gated layer(s):")\n'
        '    print(f"  -> {_cqig_mem(_L):.2f} GiB of GPU memory for the CQIG statistics\\n")'))
    # lam rides on the MAIN arm when this is a one-notebook build (TOMATO). Without it the
    # gate sits at the config default 0.1, whose span is lam/2 = 5%, and every arm measured
    # at that default has been a no-op -- so the CQIG row would cost a full training run to
    # reproduce a known null.
    # SEMANTIC LINKING ON THE MAIN TOMATO ARM. cqig_link defaults to "exact", which
    # resolves the reference bank by entity NAME against the target graph's vocabulary.
    # On a graph the bank was not chosen from, that resolves very few seeds, and the gate
    # then applies a train-calibrated threshold to statistics computed from almost nothing
    # (measured elsewhere at 5.6% inference coverage against 100% at train). Embedding-kNN
    # relinking is the mechanism the zero-shot claim actually rests on, so the arm that
    # carries the claim should use it rather than inherit the exact-match default.
    # Read the printed `[cqig] inference: seed-coverage=` line before trusting the arm.
    _CQ_LAM = (f'\n                         cqig_lam={LAM}, cqig_link="semantic",'
               if (LAM is not None and TOMATO) else '')
    _CQ_SFX = (f'cqig{LAM_SLUG}' if (LAM is not None and TOMATO) else 'cqig')
    cells.append(run_cell(
        '# THE MAIN ARM: layer-wide median-variance normalisation, last layer only.\n'
        'RUN_DIR_CQIG = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
        '                         cqig=True, cqig_norm="layer", cqig_layers="last",'
        + _CQ_LAM + '\n'
        '                         suffix=f"fusion_qwenmlp_%s_epoch{EPOCHS}_b{BATCH}")' % _CQ_SFX,
        "RUN_DIR_CQIG", 'f"fusion_qwenmlp_%s_epoch{EPOCHS}_b{BATCH}"' % _CQ_SFX))
    # THE TWO CQIG ABLATIONS ARE SIR-4 ONLY. They answer "which normaliser, how many
    # layers", which is already settled; on TOMATO they would be two extra 10-epoch runs
    # outside the 2x2 the notebook is for.
    if not TOMATO:
        cells.append(run_cell(
            '# ABLATION: the self-normalising denominator, which is what made the gate a\n'
            '# near-constant rescaling. Same everything else, so the pair isolates the fix.\n'
            'RUN_DIR_CQIG_NODE = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
            '                              cqig=True, cqig_norm="node", cqig_layers="last",\n'
            '                              suffix=f"fusion_qwenmlp_cqignode_epoch{EPOCHS}_b{BATCH}")',
            "RUN_DIR_CQIG_NODE", 'f"fusion_qwenmlp_cqignode_epoch{EPOCHS}_b{BATCH}"'))
        cells.append(run_cell(
            '# MULTI-LAYER: gating only the last layer lets five layers of uninformative\n'
            '# messages propagate first. rounds=2 keeps calibration consistent with the gates.\n'
            '# The heaviest arm; see the memory cell above before starting it.\n'
            'RUN_DIR_CQIG_ML = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
            '                            cqig=True, cqig_norm="layer", cqig_layers="__CQIG_ML__",\n'
            '                            cqig_rounds=2,\n'
            '                            suffix=f"fusion_qwenmlp_cqigml_epoch{EPOCHS}_b{BATCH}")',
            "RUN_DIR_CQIG_ML", 'f"fusion_qwenmlp_cqigml_epoch{EPOCHS}_b{BATCH}"'))

if LAM_ONLY:
    cells.append(md((f"""### 9e-lam. NOT RUN HERE — what `lam_init={LAM}` means

This section is background: the arm below is commented out because this notebook runs the
semantic-linking arm alone. It is kept because `lam` is half of what that arm changes, and
the numbers here are what the one live run has to be read against.

""" if LINK_ONLY else f"""### 9e-lam. ARM 3A — graph, CQIG `lam_init={LAM}`, ADDITIVE fusion

Trains `RUN_DIR_CQIG_LAM`, the third of this notebook's four arms. Its control is **arm 2A
in 9b**: same scorer, same graph, same objective, same epochs, gate off. Its partner under
the mixture is **arm 3B in 9m**.

**What lambda is.** It is the gate's authority, not its aim. `I` decides *which* nodes are
damped; lambda decides *how much*, and it is the only thing standing between a good ranking
statistic and no effect on the ranking. At the config default 0.1 the realised span is ~0.05
and the gate is a near-constant rescaling; at {LAM} the span is ~{LAM / 2:.2f} and the floor on
`g` drops from 0.90 to {1 - LAM:.2f}. Every CQIG arm before this one measured a no-op for that
reason, so this is the first setting at which the hypothesis is actually on trial.

Read the log in this order: `[cqig] eval gate ... span=` (should be ~{LAM / 2:.2e}, not ~5e-02),
then `informativeness AUC` (must be UNCHANGED -- lam does not touch `I`, so a moved AUC means
something else changed), then `[diag] ... graph`, since the fused number cannot move before
the graph-only one does.
""" if RUN_SET else f"""### 9e-lam. The one new run — CQIG with `lam_init={LAM}`

Identical to `RUN_DIR_CQIG` above in every respect except `lam_init`: same layer-wide
median-variance normalisation, same single gated layer, same exact-name reference linking,
same `seed: 1024`, same data, same objective, same epoch budget. One scalar moves, so
whatever this does is attributable to it.

**What lambda is.** It is the gate's authority, not its aim. `I` decides *which* nodes are
damped; lambda decides *how much*, and it is the only thing standing between a good ranking
statistic and no effect on the ranking.

| | `RUN_DIR_CQIG` | this run |
|---|---|---|
| `lam_init` | 0.1 (config default) | **{LAM}** |
| floor on `g` | 0.90 | **{1 - LAM:.2f}** |
| realised span, tau at the calibrated median | ~0.05 | **~{LAM / 2:.2f}** |
| effect on a graph channel at gamma≈0.03 | ~0.2% | **~{LAM / 2 * 3:.0f}%** |

**Three outcomes and what each means.**

- **Graph-only nDCG@5 rises** over the multi-view+graph control. The hypothesis survives:
  damping uninformative messages helps, and the earlier flat results were a lambda problem.
- **Graph-only nDCG@5 falls.** The statistic ranks documents well (AUC 0.72) but the nodes
  it damps are load-bearing *intermediates*, not distractors. That is a real finding and it
  is the hub-cutting concern made falsifiable: the AUC is measured at document nodes and
  says nothing about the hops on the way there.
- **Still flat at span ~{LAM / 2:.2f}.** Then the gate is genuinely inert and neither
  lambda nor the normalisation is the lever, which retires this parameterisation.

Note that `lam` remains learnable. This changes where it starts, and the run's final `lam=`
in the log says which direction the loss then pushed it -- itself informative, since from
0.1 it did not move at all.""")))
    cells.append(run_cell(
        ('# ARM 3A of 4: gate lam=%s, ADDITIVE fusion.\n' % LAM if RUN_SET else
         '# THE FIRST NEW RUN.\n') +
        '# lam_init is a model argument, not an environment variable, so\n'
        '# it goes through hydra as model.cqig_lam; run_model appends it only when set.\n'
        '#\n'
        '# Read the log in this order:\n'
        '#   1. "[cqig] eval gate ... span="  should be ~%.2e now, not ~5e-02.\n'
        '#   2. "informativeness AUC"         should be UNCHANGED: lam does not touch I,\n'
        '#                                    so a moved AUC means something else changed.\n'
        '#   3. "[diag] ... %s"           the gate acts on the graph channel, and the\n'
        '#                                    fused number cannot move before that one does.\n'
        '#   4. the final "lam=" in the last calibration block: from 0.1 it never moved.\n'
        'RUN_DIR_CQIG_LAM = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
        '                             cqig=True, cqig_norm="layer", cqig_layers="last",\n'
        '                             cqig_lam=%s,\n'
        '                             suffix=f"fusion_qwenmlp_cqiglam%s_epoch{EPOCHS}_b{BATCH}")'
        % (LAM / 2, "__L2__", LAM, LAM_SLUG),
        "RUN_DIR_CQIG_LAM",
        'f"fusion_qwenmlp_cqiglam%s_epoch{EPOCHS}_b{BATCH}"' % LAM_SLUG,
        dead=EXTRA or LINK_ONLY))

    # THE B COLUMN, TRAINED. The eval-time [sweep]/[gsweep] lines score both fusion forms
    # off one set of channels, which is cheap and pairs per query, but those channels were
    # fitted under the ADDITIVE objective. Read off them the mixture is being judged on a
    # representation that was never optimised for it, so a mixture loss there is confounded
    # and means nothing. The fusion form changes the gradient the GNN sees -- relu(z(g))
    # zeroes it for every document below the graph's own mean, the mixture does not -- so
    # the two forms train different graph channels and the only honest comparison trains
    # both. The sweeps stay in the output as a secondary, paired read; these two runs are
    # the primary one.
    if RUN_SET:
        cells.append(md("""### 9m. ARMS 2B and 3B — the same two rows, trained under the MIXTURE

`fusion_form="mixture"` is passed to `run_model`, so these two train with the mixture in the
loss rather than having it applied afterwards. Together with 9b and 9e-lam this is the full
2x2: {gate off, gate lam} x {additive, mixture}, four trained models, no post-hoc scoring in
the headline table.

**`a_max=0.3`, router live.** These are the settings that produced the only measured
non-negative mixture result on real channels:

| setting | measured | result |
|---|---|---|
| router, `a_max=0.5` | yes | fused 0.4265 vs semantic 0.5178, **-0.091** |
| **router, `a_max=0.3`** | **yes** | ep6 fused 0.5239 vs semantic 0.5189, **+0.0050** (best 0.5240, `a_q` 0.2246) |
| constant `a=0.05` | no | never run |

The router's per-query span is only ~0.030 wide, so it is fair to say it is not really
routing: it learns one number and ramps it toward the cap. But that is a claim about the
**mechanism**, not about the result, and pinning `a` to a constant on the strength of it
swapped a measured setting for an unmeasured one roughly four times smaller. It also broke
the pairing this table depends on: 2A learns `gamma`, so 2B has to learn its mixing scalar
too, or the two arms differ in two things at once and the "fusion form" row means nothing.

The `[sweep]` line still prints fused nDCG@5 at constant
`a` in {0, .005, .01, .02, .05, .1, .2, .35} from the cached channels at every epoch. That
is where to look for whether a constant would have beaten the router at this checkpoint. It
is a diagnostic, not an arm: those channels were fitted with the router live, and the `a=0`
row is the control that must equal the `semantic` column."""))
        cells.append(run_cell(
            '# MIXTURE, gate off. Pairs with RUN_DIR_MLP: same data, same objective weights,\n'
            '# same epochs -- the ONLY difference is how the two channels are combined.\n'
            '#\n'
            '# a_max=0.3 WITH THE ROUTER LIVE. This is the only mixture setting with a\n'
            '# measured non-negative result on real channels: matsci ep6 fused 0.5239 vs\n'
            '# semantic 0.5189, at a_q 0.2246. a_max=0.5 is the one that failed (-0.091).\n'
            '# There is no FUSION_AFIX here on purpose. Pinning a to a constant looked like\n'
            '# a control -- the router\'s per-query span is only 0.030 wide, so it is not\n'
            '# really routing -- but "not routing" is a claim about the MECHANISM, not about\n'
            '# the result, and the constant that replaced it (0.05) had never been run and is\n'
            '# 4x below the value that produced the decent number. It also broke the pairing:\n'
            '# 2A learns gamma, so 2B must learn its mixing scalar or the two arms differ in\n'
            '# two things at once.\n'
            'RUN_DIR_MIX = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
            '                        fusion_form="mixture", fusion_amax=0.3,\n'
            '                        fusion_ainit=0.1, fusion_router="full", fusion_taug="learn",\n'
            '                        suffix=f"fusion_qwenmlp_mix_epoch{EPOCHS}_b{BATCH}")',
            "RUN_DIR_MIX", 'f"fusion_qwenmlp_mix_epoch{EPOCHS}_b{BATCH}"'))
        cells.append(run_cell(
            '# MIXTURE + gate. Pairs with RUN_DIR_CQIG_LAM the same way. The 2x2 closes here:\n'
            '# (this - RUN_DIR_MIX) is the gate under the mixture, and (this - RUN_DIR_CQIG_LAM)\n'
            '# is the fusion form under the gate. Neither delta is contaminated by the other.\n'
            'RUN_DIR_MIX_LAM = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
            '                            cqig=True, cqig_norm="layer", cqig_layers="last",\n'
            '                            cqig_lam=%s,\n'
            '                            fusion_form="mixture", fusion_amax=0.3,\n'
            '                            fusion_ainit=0.1, fusion_router="full", fusion_taug="learn",\n'
            '                            suffix=f"fusion_qwenmlp_cqiglam%s_mix_epoch{EPOCHS}_b{BATCH}")'
            % (LAM, LAM_SLUG),
            "RUN_DIR_MIX_LAM",
            'f"fusion_qwenmlp_cqiglam%s_mix_epoch{EPOCHS}_b{BATCH}"' % LAM_SLUG))

    # The 9e-link prose describes an arm that --run-set does not emit (semantic linking
    # is an ablation of the gate, not a row of the table). Leaving it in was how the
    # header came to promise "9e-lam and 9e-link" in a notebook that trains neither.
    if not RUN_SET:
        cells.append(md(f"""### 9e-link. The second new run — fix the seeding as well

The gate's authority is one of two things wrong. The other is that the references barely
reach the target graph at all.

```
physics TRAIN graph:  seed-coverage 100.0%  (501/501)
physics TEST  graph:  seed-coverage   5.0%  (25/501)   queries_with_no_seed=3
```

Same domain, different papers, and 95% of the reference problems' entry points simply do not
exist under those names. Everything downstream inherits that: $\\mu_v$ is the average
response *to the reference problems*, and on the test graph it was averaged over runs that
never entered the graph. The statistic the AUC vindicated was measured despite this, not
because of it.

**The linker, not the bank, is what was wrong.** For each seed $c$:

1. keep an exact target-node match when one exists;
2. otherwise embed the concept, $\\mathbf e_c = E(c)$, using the frozen encoder that already
   embedded every node name (`graph.x`), so no encoder has to be carried to inference;
3. take its $K$ nearest target nodes,
   $\\mathcal N_K(c)=\\operatorname{{TopK}}_{{v\\in V_{{\\text{{target}}}}}}\\cos(\\mathbf e_c,\\mathbf e_v)$;
4. weight them
   $w_{{cv}}=\\dfrac{{\\exp(\\cos(\\mathbf e_c,\\mathbf e_v)/T)}}{{\\sum_{{u\\in\\mathcal N_K(c)}}\\exp(\\cos(\\mathbf e_c,\\mathbf e_u)/T)}}$;
5. reject anything below a fixed, source-selected similarity floor.

`K=3`, `T=0.05`. **The softmax is normalised over the survivors**, so a seed that links to
three nodes injects exactly the weight it injected when it linked to one. An exactly linked
reference and a semantically linked one therefore start with the same mass, which is what
makes the two arms comparable rather than one of them simply louder.

So "thin film deposition" can enter through "thin-film growth", "physical vapour deposition"
and "film fabrication" instead of not entering at all.

**The floor is measured, not chosen.** `CQIG_LINK_MIN=auto` takes each reference seed's
nearest *other* node in the **source** graph and uses the median of those cosines. That is
what a genuine near-synonym scores in this encoder's geometry, at this vocabulary's scale.
A hand-set 0.7 would be a hyperparameter tuned on whichever corpus it was first tried on and
would not survive an encoder change; this rescales with the encoder because it is measured
in it. It is frozen with the bank and rides in the checkpoint, so a later calibration on a
new corpus rejects at the same threshold even in a process where the variable is unset.

**This is still zero-shot.** The reference questions are fixed on source data; the encoder is
frozen; the floor comes from source statistics; nearest-neighbour linking reads only the
target's node names, which are the same public artefact the index is built from; no target
label, no target query, and no fine-tuning. What changes is only that a reference problem can
enter a target graph that does not use the source's exact phrasing.

**What to read.** A new `[cqig]   linking:` line splits exact from semantic hits at every
calibration. The number that matters is the TEST graph's, against the 5% above. Then the same
sequence as 9e-lam: gate span, AUC, graph-only nDCG@5.

{f'''**This is the only run in this notebook, and it moves TWO things at once** relative to
`RUN_DIR_CQIG`: lambda 0.1 to {LAM}, and exact to semantic seeding. That is deliberate, not
sloppy. matsci separated them at cost and the answer was one-sided: lambda carried the whole
gain (fused nDCG@5 0.5213 to 0.5340, graph contribution +0.0018 to +0.0170), and the linker
on top of it was a wash (-0.0006). So the question here is only whether the pair replicates
off matsci, and paying for the separation again would buy a number we already have.

**Read it against `RUN_DIR_MLP` (multi-view + graph), which is the row that matters**, and
against `RUN_DIR_CQIG` for how much of that gap lambda opened. If this arm does NOT beat 9b
on this domain, the matsci result does not generalise, and that is the finding.''' if LINK_ONLY
else f'''**Read this arm against 9e-lam, not against 9b.** Both new runs carry `lam={LAM}`, so the pair
isolates the seeding; 9e-lam against `RUN_DIR_CQIG` isolates lambda. Two runs, two one-variable
contrasts, and the finished arms supply both baselines.'''}"""))
    cells.append(run_cell(
        '# THE SECOND NEW RUN: same lam as 9e-lam, semantic seed linking on top. Everything\n'
        '# else is 9e\'s main arm. The pair (9e-lam, this) differs in ONE thing, the linker.\n'
        '#\n'
        '# Read the [cqig]   linking: line at EVERY calibration, and note there are two that\n'
        '# matter and they should not agree:\n'
        '#   train graph: exact should already be ~100%%, so semantic adds ~nothing. If it\n'
        '#                adds a lot there, the floor is too low and the linker is inventing\n'
        '#                matches for names that were found.\n'
        '#   TEST graph:  exact is the ~5%% failure. This is the number the run exists to move.\n'
        '#\n'
        '# CQIG_LINK_MIN=auto measures the floor on the SOURCE graph when the bank is frozen;\n'
        '# the "[cqig] link=semantic:" line prints it and how it was obtained. Pin it with\n'
        '# cqig_link_min=0.72 if you want the threshold fixed rather than measured.\n'
        'RUN_DIR_CQIG_LINK = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
        '                              cqig=True, cqig_norm="layer", cqig_layers="last",\n'
        '                              cqig_lam=%s,\n'
        '                              cqig_link="semantic", cqig_link_k=3, cqig_link_t=0.05,\n'
        '                              cqig_link_min="auto",\n'
        '                              suffix=f"fusion_qwenmlp_cqiglink%s_epoch{EPOCHS}_b{BATCH}")'
        % (LAM, LAM_SLUG),
        "RUN_DIR_CQIG_LINK",
        'f"fusion_qwenmlp_cqiglink%s_epoch{EPOCHS}_b{BATCH}"' % LAM_SLUG, dead=EXTRA))

if MISSW:
    cells.append(md(r"""### 9w. Semantic-error-weighted graph loss

Every gold currently supervises the graph equally. This arm weights each gold's
**graph-alone** contrastive term by how badly the semantic scorer ranked it:

$$w_{qg}=\min\bigl(c,\;\log(1+r_{\mathrm{sem}}(q,g))\bigr)$$

$$\mathcal L_{\mathrm{graph}}^{\mathrm{miss}}=
\frac{\sum_{g} w_{qg}\Bigl[\log\bigl(e^{z_g(q,g)}+\sum_{n\in\mathcal N_q}e^{z_g(q,n)}\bigr)-z_g(q,g)\Bigr]}
{\sum_{g} w_{qg}}$$

$$\mathcal L=\mathcal L_{\mathrm{fused}}+\lambda_{\mathrm{graph}}\mathcal L_{\mathrm{graph}}^{\mathrm{miss}}+\lambda_{\mathrm{pop}}\mathcal L_{\mathrm{pop}}$$

This is `MISS_W_AUX=1`, already implemented in `_contrastive_hardneg` and never run.
$r_{\mathrm{sem}}$ is the gold's **true corpus rank** under whichever scorer is live, detached.

**Only the graph term is weighted, and that is the whole safety argument.** Weighting the
FUSED loss the same way is `MISS_W_FUSED`, and amplifying graph-noisy queries on the term
the *gate* reads is what taught the gate backwards once already. The gate never sees this term.

**Normalising by the weight sum makes it a pure reallocation.** The loss scale is invariant
to $c$, so $\lambda_{\mathrm{graph}}$ needs no retuning and a uniform shift in ranks is a
no-op. It can only help by moving capacity between golds, never by asking for more graph
overall -- so it cannot fix a graph that is simply too quiet.

| arm | $c$ | bites at $r_{\mathrm{sem}}$ |
|---|---|---|
| control | -- | uniform weights |
| loose | 8.0 | about 2980, so effectively uncapped on a 24k corpus |
| tight | 4.0 | about 54 |

**$\gamma$ is pinned at __GFIX__ in all three.** Left free it runs to 0.26 and costs 0.07
nDCG@5 on its own, which would swamp any loss effect. Pinned, the fused column is readable
and the three arms differ in exactly one thing.

**Read `[bucket]`, not the aggregate.** The mechanism claims to spend graph capacity on the
golds the scorer buries, so the test is graph recall@10 split by $r_{\mathrm{sem}}$:

- **deep buckets rise, shallow holds** -- the hypothesis survives.
- **deep flat, shallow falls** -- the graph cannot fit those golds, and the reweighting is
  moving gradient onto examples the architecture does not support. This is what the
  construction results predict: dissimilar golds share no frame node with the query, so they
  are reachable but not rankable.
- **everything flat** -- $r_{\mathrm{sem}}$ on TRAIN queries is not $r_{\mathrm{sem}}$ at
  test. The scorer trains jointly and fits its own queries, so train ranks are systematically
  shallower and the weights end up flatter than the test distribution warrants. Freezing the
  scorer (`model.semantic_train=false`) is the follow-up, not a bigger $c$.
""".replace("__GFIX__", str(MISSW_GFIX))))
    for _lab, _cap in (("control", None), ("loose", 8.0), ("tight", 4.0)):
        _sfx = "missw%s" % ("off" if _cap is None else ("%g" % _cap).replace(".", ""))
        cells.append(code(
            ("# CONTROL: uniform per-gold weights. Identical to the two arms below in every\n"
             "# other respect, including the pinned gamma.\n" if _cap is None else
             "# w = min(%g, log1p(r_sem)) on the GRAPH-ALONE term only.\n" % _cap) +
            'RUN_DIR_MISSW_%s = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
            '        fusion_gammafix=%s,\n'
            '        miss_w_aux=%s,%s\n'
            '        suffix=f"fusion_qwenmlp_%s_epoch{EPOCHS}_b{BATCH}")'
            % (_lab.upper(), MISSW_GFIX, _cap is not None,
               "" if _cap is None else " miss_w_cap=%g," % _cap, _sfx)))

if RESIDP:
    cells.append(md(r"""### 9r. Residual supervision against the structural prior

**The measured problem.** A zero-parameter random walk on these graphs scores
0.245-0.273 nDCG@5. The trained 34M-parameter query-conditioned GNN scores 0.213-0.273.
The learned component is at, or below, its own untrained structural prior.

Why: the graph-alone contrastive asks the GNN to rank each gold above the scorer's top
hubs, and personalised PageRank from the same seed entities largely does that already. The
term is close to satisfied at initialisation, so the gradient that survives points back at
the topology the walk exploits. Every loss edit tried so far reweighted this term
(`MISS_W_AUX` by the *semantic* channel's rank) and none of them changed what it is
measuring.

**The edit.** Restrict each gold's negative set to the documents the **prior itself**
already ranks above that gold:

$$\mathcal N_q(g)=\{d: r(q,d) > r(q,g)\}\ \text{truncated to the prior's top-}K,\qquad
\mathcal L_{\mathrm{resid}}=\sum_g\Bigl[\log\!\!\sum_{n\in\mathcal N_q(g)\cup\{g\}}\!\!e^{z_g(q,n)}-z_g(q,g)\Bigr]$$

with $r$ a symmetrised degree-normalised walk with restart, $T=3$, $\alpha=0.15$, detached.

Documents the prior ordered correctly contribute **exactly zero** gradient, so the loss can
only fall by capacity the topology does not already supply. It is not a reweighting of the
old term, it is a different question asked of the same model.

**Cost is a first-epoch one.** One cached PPR per query, three sparse mat-muls, reused for
every later epoch, against six dense GNN layers per step. Scales with edges, not with
parameters.

**Two guards inside the loss.** Golds the walk already ranks above everything have an empty
negative set and would leave the model unconstrained on what it gets right; they keep a
small term (`RESID_ANCHOR=0.1`) against the ordinary lineup. Golds the walk buries under
thousands of documents are truncated to the prior's top `RESID_K=200`, so one pathological
query cannot dominate by sheer count.

**Read these three lines, in this order:**

1. `[prior]` — the walk's own nDCG@5 next to the trained graph's. This is the baseline the
   whole arm exists to beat, and it is now printed by every run including the control.
2. `[diag] graph` — graph-alone nDCG@5. **This is the arm's outcome measure, not the fused
   column.** The fusion cannot express a better graph ranking it is not allowed to act on.
3. `resid_cover` in the training postfix — the fraction of golds the prior actually gets
   wrong, i.e. how much of the data this loss can even see. Near zero means the walk is
   already right and there is nothing to learn; near one means the prior is useless here
   and the restriction is not doing what it claims.

**Falsification.** If graph-alone nDCG@5 does not clear the walk by more than the control
already does, the GNN has no capacity to exceed its prior on this graph and the answer is
graph construction, not the loss. That is a publishable negative with a number attached.

$\gamma$ is pinned at __GFIX__ in both additive arms, so a change in the loss is not read
through a fusion weight that drifts to 0.26 and costs 0.07 nDCG@5 on its own.
""".replace("__GFIX__", str(MISSW_GFIX))))
    _RCOMMON = ('epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
                '        cqig=True, cqig_norm="layer", cqig_layers="last", cqig_lam=%s,\n'
                '        cqig_link="semantic",\n' % LAM)
    cells.append(code(
        "# CONTROL. The standard graph-alone contrastive: negatives are the scorer's top\n"
        "# hubs plus randoms. Identical to the arm below in every other respect.\n"
        "RUN_DIR_RESID_CTRL = run_model(%s"
        "        fusion_gammafix=%s,\n"
        "        resid_prior=False,\n"
        '        suffix=f"fusion_qwenmlp_cqiglam%sresidctrl_epoch{EPOCHS}_b{BATCH}")'
        % (_RCOMMON, MISSW_GFIX, LAM_SLUG)))
    cells.append(code(
        "# THE ARM. Negatives restricted to documents the parameter-free walk already ranks\n"
        "# above the gold, so the only way down is capacity the topology does not have.\n"
        "RUN_DIR_RESID = run_model(%s"
        "        fusion_gammafix=%s,\n"
        "        resid_prior=True, resid_k=200, resid_anchor=0.1,\n"
        '        suffix=f"fusion_qwenmlp_cqiglam%sresidon_epoch{EPOCHS}_b{BATCH}")'
        % (_RCOMMON, MISSW_GFIX, LAM_SLUG)))
    cells.append(md("""**Third run, optional.** Same loss, mixture fusion instead of additive.

Be aware what it can and cannot show. The mixture's ceiling was measured directly: a
32-cell $a\\times\\tau_g$ grid came back identical to four decimal places, because
$a\\,p_g^{(1)}$ never reaches $(1-a)\\,p_s^{(5)}$ until $a\\approx0.9$, at which point the
ranking is the graph's. If the residual loss does not sharpen the graph channel, this run
returns the control's number again.

`a` is pinned rather than routed: the learned router spanned only 0.13 across 331 queries
and ran to whatever cap it was given, so it was a constant with extra steps.

**If you only have three runs, a second seed of the arm above is probably worth more.** The
control moved 0.0118 nDCG@5 between two identical reruns, which is larger than every effect
measured on this project so far. Change the suffix to `...residon2_...` and run it twice."""))
    cells.append(code(
        "# OPTIONAL third run.\n"
        "RUN_DIR_RESID_MIX = run_model(%s"
        "        fusion_form=\"mixture\", fusion_afix=0.05,\n"
        "        resid_prior=True, resid_k=200, resid_anchor=0.1,\n"
        '        suffix=f"fusion_qwenmlp_cqiglam%sresidmix_epoch{EPOCHS}_b{BATCH}")'
        % (_RCOMMON, LAM_SLUG)))

if MIX:
    cells.append(md(rf"""### 9f. The fusion form — mixing distributions instead of scores

Every arm above changes the **graph channel**. This one changes how that channel is
**combined** with the semantic one, and that is where the measured ceiling actually is.

**The additive form cannot promote a document, at any $\gamma$ it learns.** A $z$-score over
a corpus this size tops out near 5–10 even for the graph's single most confident paper, and
the converged $\gamma$ is $0.036$ in all three finished arms. The largest contribution the
graph can make to *any* document is therefore about $0.36$ $z$-units, against a semantic
spread across its own top 50 of 1–3. The graph can reorder documents the semantic scorer
already placed next to each other; it cannot lift a buried gold. That is a property of the
arithmetic, not of the gate, of $\lambda$, or of the bfloat16 freeze, and it is why the
contribution is pinned at $+0.017$ nDCG@5 no matter what the gate does.

**Mix calibrated distributions instead of scores:**

$$p_s=\operatorname{{softmax}}\bigl(z(s)\bigr),\qquad
  p_g=\operatorname{{softmax}}\bigl(z(g)/\tau_g\bigr),\qquad
  \text{{fused}}=\log\bigl((1-a_q)\,p_s+a_q\,p_g\bigr)$$

with $a_q = a_{{\max}}\,\sigma(u^{{\top}}\phi_q)\in[0,{AMAX:g}]$.

Four properties the additive form lacks:

1. **Exponential response.** $p_g$ for the graph's top document is $O(10^{{-1}})$; $p_s$ in the
   semantic tail is $O(10^{{-5}})$. Even at $a_q=0.05$ the graph term dominates there. On a
   synthetic 15,588-document corpus a gold at semantic rank 1,560 that the graph ranks first
   goes to **rank 3** under the mixture and to **rank 1,019** under the additive form.
2. **An uninformative graph is exactly inert.** $\partial_x\log((1-a)e^x+ae^k)>0$, so a flat
   $p_g$ is a constant and the ranking does not move at all. The weak-channel problem is
   handled by the algebra rather than by keeping $\gamma$ small.
3. **Mixture, not product.** A product of experts would let the weak channel *veto* a document
   the semantic scorer got right, driving its score toward zero. A mixture only adds mass, so no
   document's score ever falls, $a_q\to0$ recovers the semantic ranking and $a_{{\max}}$ bounds
   the worst case. **Read that as a claim about scores, not ranks.** Rank is zero-sum: a
   confidently wrong graph promotes other documents *over* a correct one and demotes it as
   effectively as a veto would. Measured — a gold at semantic rank 1 falls to **rank 3** when the
   graph is certain about six wrong documents, its own score gain being exactly $0.0$ while
   theirs are $0.13$ to $9.75$. The additive form's $\mathrm{{relu}}$ is non-negative too, so this
   floor is **not** what separates the two forms; $a_{{\max}}$ and property 2 are.
4. **The ranking loss reaches every graph logit.** $\mathrm{{relu}}(z(g))$ zeroes the *fused*
   loss's gradient for **half** the corpus — every document below the graph's own mean, which
   is exactly the buried golds (measured: 50.2% of documents get identically zero gradient on
   that path, 0.0% under the mixture). Be precise about what this does and does not say: the
   graph is *not* gradient-starved today, because `AUX_W=1.0` runs a separate graph-only
   contrastive that never passes through the relu. What changes is that the **final ranking
   loss** now reaches every graph logit, instead of the graph learning about its buried
   documents only from the auxiliary term.

**The risk, stated plainly.** Property 1 works just as well on the graph's confident
*mistakes*. Note what nDCG@5 $=0.27$ does **not** establish: with $\approx1.9$ golds per query
it is not a Hits@1 measurement, so it does not pin down how often the graph's top document is
wrong. The defensible statement is that the graph ranking is substantially weaker overall, so
a concentrated $p_g$ will place large mass on incorrect documents on many queries. In the same
synthetic test a document the graph is confidently *wrong* about goes from semantic rank 9,661
to rank 2. The additive form is safe because it is inert; this one is useful because it is not.
`a_max={AMAX:g}` is the only bound on that. **Whether the rescues outnumber the false
promotions is exactly the empirical question**, and it is why this is worth one run and no
conclusions before it.

**$\tau_s$ is fixed at 1 and not learned.** The training loss is already
$\log\sum\exp(\text{{lineup}})-\text{{gold}}$, a softmax cross-entropy, so a learnable $\tau_s$
would be softmaxing a softmax: shrinking it lowers the loss whenever the gold already leads,
without improving any ranking. Fixed at 1 it buys something instead — $\log p_s = z(s) -
\text{{const}}$, and cross-entropy is shift-invariant per query, so **at initialisation this arm
trains under numerically the same objective the additive arm does** (verified: identical to
6 d.p.). The learning rate and loss scale carry over unchanged, and only $\tau_g$ is learned.
($a_q$ starts at $0.1$, not at $0$ — a sigmoid is never exactly zero, so "recovers the
semantic ranking" is the limit $a_q\to0$, not the initialisation. A bias of $-7$ would make it
$\approx0.0009$ and mathematically closer, but it also puts $\sigma'$ at $\approx9\times10^{{-4}}$,
so the router would learn ~100× slower — and $\gamma$ starting at $0.01$ and needing four
epochs to climb is precisely what the bfloat16 freeze killed in all three finished arms.)

**$\tau_g$ is bounded, not free: $\tau_g = 0.25 + 3.75\,\sigma(\hat\tau_g)$.** Two distinct
problems, and only one of them is handled by the $z$-score. *Scale identifiability* is handled:
$\operatorname{{softmax}}(cg/c\tau_g) = \operatorname{{softmax}}(g/\tau_g)$, so with **raw** graph
logits the readout could inflate its scale while $\tau_g$ grew to match, and the two would not
be separately identifiable — but $z$ is exactly scale-invariant, $z(cg)=z(g)$, so the
degeneracy cannot arise. This is why the graph channel is standardised here and why feeding raw
logits would be a step backwards. *Overconfidence* is **not** handled by $z$: shrinking $\tau_g$
sharpens $p_g$ and lowers the training cross-entropy on any query whose gold already leads,
with no better ranking — the same degenerate direction $\tau_s$ is frozen to avoid. Hence the
bounds. The stronger version is post-hoc calibration (train the reasoner, freeze the readout,
fit $\tau_g$ and the router on held-out **source** queries, freeze both for target inference),
which makes $\tau_g$ a genuine calibration parameter; that costs an extra stage, and
`fusion_taug=<float>` pins it if that stage is ever run.

**What to read.** The `[diag]` line's `gamma` column now prints $a_q$, the mixing weight in
$[0,{AMAX:g}]$ — *not* the additive arm's $\gamma$, and the two are not on the same axis.
$a_q\approx0.1$ is its initialisation, not a stall. The `[fusion] form=mixture:` banner at
startup confirms which form is live, and `[fusion] fusion head pinned to float32` must appear
or the router bias at $-1.386$ freezes exactly as $\gamma$ did.

**Read it against `RUN_DIR_CQIG_LINK`**, which is the same $\lambda={LAM}$, the same linker, the
same graph and the same objective. The one thing that differs is the fusion form."""))
    cells.append(code(
        '# THE ONE LIVE RUN. Same lam, same linker, same graph, same hard-negative\n'
        '# objective as 9e-link -- the ONLY difference is fusion_form.\n'
        '#\n'
        '# Three things to check in the first minute, before letting it run:\n'
        '#   1. "[fusion] form=mixture: ..."      the form is actually live\n'
        '#   2. "[fusion] fusion head pinned to float32"   or the router bias at -1.386\n'
        '#      freezes on step one, exactly as gamma did, and a_q never leaves 0.1\n'
        '#   3. epoch-1 [diag] fused ~ semantic   a_q starts at 0.1 and the graph is\n'
        '#      near-random, so a large epoch-1 SWING either way means the graph is being\n'
        '#      trusted before it can rank, and a_max is too high for this corpus\n'
        '#\n'
        '# a_max is the risk knob, not a tuning knob: it caps how far the graph can promote\n'
        '# its own confident errors. Rebuild with --fusion-amax 0.25 for a more conservative\n'
        '# arm; it lands in its own run directory, so the two never overwrite each other.\n'
        'RUN_DIR_MIX = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
        '                        cqig=True, cqig_norm="layer", cqig_layers="last",\n'
        '                        cqig_lam=%s,\n'
        '                        cqig_link="semantic", cqig_link_k=3, cqig_link_t=0.05,\n'
        '                        cqig_link_min="auto",\n'
        '                        fusion_form="mixture", fusion_amax=%g,\n'
        '                        fusion_afix=%s, fusion_topk=%d,\n'
        '                        fusion_ainit=0.1, fusion_router="full", fusion_taug="learn",\n'
        '                        suffix=f"fusion_qwenmlp_cqig%s_epoch{EPOCHS}_b{BATCH}")'
        % (LAM, AMAX, repr(AFIX), TOPK, MIX_SLUG)))

if OP_ONLY:
    _ADAPT = OP == "centre"
    cells.append(md(f"""### 9e-centre. The third run — the same statistic, a different operator

The two arms above both spend $I$ the same way: they **scale** the state. This one spends it
by **subtracting** the background instead. It is not a variant of the gate; it is the other
thing you can do with $\\mu$.

Write $\\boldsymbol\\delta_{{qv}}=\\mathbf h_{{qv}}-\\boldsymbol\\mu_v$ for the query-specific
residual. "Propagate the residual, and keep a $\\kappa$-fraction of the background" is

$$\\tilde m = m(\\boldsymbol\\delta_{{qv}},r)+\\kappa\\,m(\\boldsymbol\\mu_v,r)
= m\\bigl(\\mathbf h_{{qv}}-(1-\\kappa)\\boldsymbol\\mu_v,\\;r\\bigr)$$

so it is a **subtraction on the input**, which is exactly what the existing forward pre-hook
can do. Two conditions make that identity exact rather than approximate, and both hold here:
a DistMult message is linear in the source state, and

```python
relation_representations = self.rel_mlp(graph.rel_attr).unsqueeze(0).expand(batch_size, -1, -1)
```

is a function of the graph's relation text alone, expanded across the batch, so it is
query-independent and $\\mathbb E_a[m(\\mathbf h_{{av}},r)]=m(\\boldsymbol\\mu_v,r)$ holds
exactly. With $\\kappa_{{qv}}=g_{{qv}}$ the whole family is one coefficient:

| `cqig_op` | operator | what it is |
|---|---|---|
| — (`cqig_lam=0`) | $\\mathbf h$ | the ungated reasoner, under every op |
| `centre-fixed` | $\\mathbf h-\\lambda\\boldsymbol\\mu_v$ | fixed centring, **query-independent** |
| `centre` | $\\mathbf h-(1-g_{{qv}})\\boldsymbol\\mu_v$ | adaptive centring |
| `centre-fixed`, $\\lambda=1$ | $\\boldsymbol\\delta_{{qv}}$ | full centring, residual only |
| `gate` | $g_{{qv}}\\mathbf h$ | 9e-lam and 9e-link |

**This run is `{OP}`.** Its control is **9e-link, not 9e-lam**: both carry `lam={LAM}` and
semantic seeding, so the pair differs in the operator and nothing else.

### Two corrections to how this was pitched

**The invariance theorem does not survive the adaptive version.** At constant $\\lambda$,

$$\\tilde m(q_1)-\\tilde m(q_2)=m(\\mathbf h_{{q_1v}}-\\mathbf h_{{q_2v}},r)
-(\\kappa_{{q_2}}-\\kappa_{{q_1}})\\,m(\\boldsymbol\\mu_v,r)$$

and the last term vanishes only when $\\kappa_{{q_1}}=\\kappa_{{q_2}}$. So the pairwise-difference
property belongs to `centre-fixed`, not to `centre`. **And $\\lambda$ fixed is the weaker
claim**, because $\\boldsymbol\\mu_v$ does not depend on $q$: constant-$\\lambda$ centring
subtracts a query-independent background, which is a real hub correction but is not
query-conditioned reasoning, and it invites the GraphNorm comparison rather than escaping it.
The query-conditioning enters only through the adaptive $\\kappa$ — the version without the
theorem. Do not present the two together.

**The parameter saving is smaller than the pitch.** $\\kappa$ still needs $I$, so it still
needs the layer scale $s^{{(\\ell)}}$, the responder cut and the calibrated $\\tau$. Three
scalars per layer become two; the calibration machinery is unchanged.

### The two things to watch, which the run now measures

**1. $\\mu=0$ is a no-op, and $\\mu=0$ is most of the graph.** A node no reference query
reached has $\\boldsymbol\\mu_v=0$ **exactly**, so centring leaves it untouched — while
gating sees $I=0$ and damps it to the floor. These are opposite treatments of the same nodes.
At the test-graph seed coverage this pair of runs saw, that is a large fraction of the graph,
and it may be the whole difference between the arms. The calibration block now prints
`mu=0 on N nodes (x%), which this op cannot touch at all`.

**2. The edit may be much bigger than the gate's at the same $\\lambda$.** Under
`use_ent_emb: early-late-fusion` $\\mathbf h$ is dominated by the node's static entity
embedding, so $\\boldsymbol\\mu$ is large and $\\boldsymbol\\delta$ is ~1% of it in norm.
Subtracting {'about half of' if _ADAPT else str(LAM)} $\\boldsymbol\\mu$ removes a large,
*directional* part of the state, where gating at the same coefficient only shortens it.
`layer_norm: yes` then renormalises much of a magnitude change away and leaves the direction
change, so this operator acts through direction whether or not that was the intent. A new log
line measures it directly:

```
[cqig]   edit ||h~-h||/||h||: mean=... max=... on ...% of live nodes
```

**Read it before the metrics.** A mean near zero means the operator did nothing and the arm
is uninformative rather than negative. A mean near 1 means the state was largely destroyed,
and the honest next step is {OP} at a lower $\\lambda$, not "centring fails".

**One thing that is genuinely lost:** with $\\tau$ at the calibrated median, $\\kappa\\approx0.5$
at init, so step 0 already subtracts half the background everywhere. `gamma_init=0.01` was
chosen so the fusion *begins* as the semantic scorer; the graph operator no longer begins
where the gated arms began. The comparison is therefore between two training trajectories,
not a perturbation of one. That is not a flaw, but it should be stated rather than discovered."""))
    cells.append(code(
        '# THE THIRD NEW RUN: same lam, same semantic seeding, DIFFERENT operator.\n'
        '# cqig_op goes through the environment as CQIG_OP (the model reads it when the\n'
        '# config leaves cqig_op null), so nothing about the other two arms changes.\n'
        '#\n'
        '# Read the log in this order, and the first two are new:\n'
        '#   1. "[cqig]     op=%s: h - c*mu, c ..."  the coefficient actually applied,\n'
        '#      and "mu=0 on N nodes (x%%)" -- the part of the graph this op CANNOT touch,\n'
        '#      which is precisely where the gating arms damp hardest. If that fraction is\n'
        '#      large, the two arms are not disagreeing about damping, they are disagreeing\n'
        '#      about coverage, and the contrast means less than it looks.\n'
        '#   2. "[cqig]   edit ||h~-h||/||h||"       how much the state actually moved.\n'
        '#      Near 0 = the arm is uninformative, not negative. Near 1 = the state was\n'
        '#      destroyed; rerun at a lower lam before concluding anything.\n'
        '#   3. "informativeness AUC"                should be UNCHANGED from 9e-link: the\n'
        '#      operator does not touch I. A moved AUC means something else changed.\n'
        '#   4. "[diag] ... graph"                   the operator acts on the graph channel.\n'
        'RUN_DIR_CQIG_CTR = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
        '                             cqig=True, cqig_norm="layer", cqig_layers="last",\n'
        '                             cqig_lam=%s, cqig_op="%s",\n'
        '                             cqig_link="semantic", cqig_link_k=3, cqig_link_t=0.05,\n'
        '                             cqig_link_min="auto",\n'
        '                             suffix=f"fusion_qwenmlp_cqig%s_epoch{EPOCHS}_b{BATCH}")'
        % (OP, LAM, OP, CTR_SLUG)))

if MU0:
    cells.append(md(f"""### 9e-mu0. The ablation — is the reference bank load-bearing at all?

Everything so far has assumed that what `lam={LAM}` gave authority to was
*cross-query informativeness*. This run tests that assumption directly by removing the only
thing that makes it cross-query.

$$\\mu_v^{{(\\ell)}}\\equiv\\mathbf 0
\\quad\\Longrightarrow\\quad
I_{{qv}}^{{(\\ell)}}=\\frac{{\\lVert\\mathbf h_{{qv}}^{{(\\ell)}}\\rVert^2}}{{s^{{(\\ell)}}+\\epsilon}}$$

With $\\boldsymbol\\mu=0$ the statistic is not "this node responds to *this* query unlike it
responds to unrelated ones". It is **how large this node's activation is**. No reference
problem enters it. Same gate, same $\\lambda$, same $\\alpha$, same measured $\\tau$, same
single layer, same seeding, same data, same seed — and no bank.

| | 9e-lam | this run |
|---|---|---|
| $\\boldsymbol\\mu_v$ | mean over M reference problems | $\\mathbf 0$ |
| $I_{{qv}}$ | $\\lVert\\mathbf h-\\boldsymbol\\mu\\rVert^2/s$ | $\\lVert\\mathbf h\\rVert^2/s$ |
| what the gate damps | nodes that answer everything alike | nodes with small activations |
| M reference forwards | used | **still run, then discarded** |

That last row is deliberate. The mean pass executes and its result is zeroed in place, so the
two arms differ in the *value* of $\\boldsymbol\\mu$ and in nothing else: not the code path,
not the number of forwards, not RNG consumption. And $V$, the layer scale $s$ and
$\\tau_{{\\text{{ref}}}}$ are all recomputed against $\\boldsymbol\\mu=0$ rather than inherited
from a real one, so the gate sits at the median of *this* statistic, not the other one's.

### What each outcome means, and neither is a bad result

- **It reproduces ~0.534.** The reference bank was never load-bearing. `lam` gave authority
  to an activation-magnitude gate, the AUC of 0.68 was measuring node reachability rather
  than cross-query informativeness, and H1 as written is not what improved the score. That
  is a clean negative and it is publishable as one: it says the gain is real and the
  mechanism is not the stated one.
- **It drops toward the `lam=0.1` control (~0.521).** The bank IS load-bearing, the premise
  survives, and the 34.6%-coverage linker result becomes a statement about *how much* of
  $\\boldsymbol\\mu$ you need rather than about whether you need it.
- **It rises above 0.534.** The reference subtraction was actively costing something, which
  would make `centre` (which subtracts *more* of it) the wrong direction and is worth
  knowing before that arm is interpreted.

**Why this is the cheapest experiment left.** It is one flag, it reuses everything, and it is
the only run that can distinguish the method from its own floor. Every number reported for
CQIG so far is uninterpretable without it.

**What to read.** `[cqig]     mu=ZERO ABLATION: mu is 0 on all N nodes` confirms the flag
took effect, and `mu_zero_frac` should be exactly 1.00. Then the AUC: if it stays near 0.68
with no bank at all, the AUC was never evidence for the premise. Then the `[diag]` graph
column, then fused."""))
    cells.append(code(
        '# THE ABLATION. Same lam as 9e-lam, exact linking as 9e-lam, mu forced to zero.\n'
        '# Its control is 9e-lam (RUN_DIR_CQIG_LAM), which differs in mu ALONE.\n'
        '#\n'
        '# NOT combined with a centring op: h - c*0 = h, so that pair would be the ungated\n'
        '# reasoner rather than an ablation. The model raises if you try.\n'
        '#\n'
        '# Read the log in this order:\n'
        '#   1. "[cqig]     mu=ZERO ABLATION"   the flag took effect, mu_zero_frac == 1.00.\n'
        '#   2. "informativeness AUC"           THE NUMBER THAT MATTERS. 9e-lam measured\n'
        '#      0.678 WITH a bank. If this arm measures ~0.68 with no bank at all, the AUC\n'
        '#      was reading node reachability, not cross-query informativeness, and it was\n'
        '#      never evidence for the premise.\n'
        '#   3. "[diag] ... graph"              then fused, against 9e-lam\'s 0.5346.\n'
        'RUN_DIR_CQIG_MU0 = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",\n'
        '                             cqig=True, cqig_norm="layer", cqig_layers="last",\n'
        '                             cqig_lam=%s, cqig_mu="zero",\n'
        '                             suffix=f"fusion_qwenmlp_cqigmuz%s_epoch{EPOCHS}_b{BATCH}")'
        % (LAM, LAM_SLUG)))

cells.append(md("## 10. Score — multi-gold, with CompleteSet@k\n"
                "The trainer uses filtered knowledge-graph MRR/Hit metrics. Thesis results "
                "come from the benchmark scorer here: standard IR MRR, nDCG, multi-gold "
                "Recall, and CompleteSet."))
cells.append(code('''import os, json
# Query file: the CORPUS one, not the graph's stage1 copy. Both carry the same
# ids, but the corpus file is the benchmark as staged, so the scorer's
# denominator is the benchmark's and not something the graph build produced.
qs = f"{SCIGRAPHIR_ROOT}/retriever/data/{DATASET}_test/raw/test.json"

# THE SAME METRIC SET 5d USED. score_sir4's default cols are mrr, ndcg@5,
# recall@3, recall@5, completeset@5 -- and --json-out writes only the requested
# cols. Without this the fusion score files silently lack recall@10/25/100 and
# cannot be put in one table with 5d's semantic-only files afterwards.
COLS4 = "mrr,ndcg@5,recall@3,recall@5,recall@10,recall@25,recall@100__CSET__"

# Both fusion arms through the SAME scorer with the SAME flags. A run that was not
# executed is skipped rather than faked, and says so.
RUNS = __ARMS__

# SURVIVE A RUNTIME RESTART. A Colab disconnect wipes RUN_DIR_*, but the run itself is on
# Drive, and the directory is a pure function of (OUT_ROOT, DATASET, suffix, BATCH). So a
# lost variable is a naming problem, not lost work: rebind it rather than retraining.
import glob
def _resolve(name, rd, out_tag):
    if rd and os.path.exists(f"{rd}/predictions_{TEST}.json"):
        return rd
    want = f"{OUT_ROOT}/{DATASET}_{out_tag}_b{BATCH}"
    if os.path.exists(f"{want}/predictions_{TEST}.json"):
        print(f"[recover] {name}: rebound from Drive -> {os.path.basename(want)}")
        return want
    # A FINISHED RUN AT ANOTHER BATCH IS NOT A SUBSTITUTE, so it is reported and NOT used.
    # Batch sets the optimizer-step count at a fixed epoch budget, so arms trained at
    # different batches are two variables and attribute to neither.
    other = [d for d in sorted(glob.glob(f"{OUT_ROOT}/{DATASET}_{out_tag}_b*"))
             if os.path.exists(f"{d}/predictions_{TEST}.json")]
    if other:
        print(f"[recover] {name}: found {[os.path.basename(d) for d in other]} but this "
              f"notebook is at BATCH={BATCH}, so it is NOT used. Re-run this arm at "
              f"BATCH={BATCH}, or set BATCH to match and re-run EVERY arm.")
    return None

RUNS = [(n, _resolve(n, rd, t), t) for n, rd, t in RUNS]
for name, rd, out_tag in RUNS:
    if not rd or not os.path.exists(f"{rd}/predictions_{TEST}.json"):
        print(f"skipping {name}: no predictions (run its training cell first)")
        continue
    args = f"--pred {rd}/predictions_{TEST}.json --queries {qs}"
    if os.path.exists(SETS):     args += f" --sets {SETS}"
    if os.path.exists(BGE_PRED): args += f" --bge {BGE_PRED}"
    else: print("no BGE predictions -> similar/dissimilar slices will be skipped")
    sh(f"python3 -u eval/score_sir4.py {args} --cols {COLS4} --name '{name}' "
       f"--json-out {rd}/scores.json", S4)
    os.system(f"cp {rd}/scores.json {OUT_ROOT}/scores_{out_tag}.json")

# Same scorer, same slices, every arm side by side. The learned-scorer row is the
# claim; the operator row is what it has to beat with the graph half held fixed.
rows = [("BGE baseline", f"{S4}/data/metrics_bge_{DATASET}_test.json")]
rows += [(n, f"{rd}/scores.json") for n, rd, _ in RUNS if rd]
for tag, p in rows:
    if not os.path.exists(p):
        continue
    m = json.load(open(p)).get("all", {})
    # CS@5 ONLY IF IT WAS ACTUALLY COMPUTED. `.get(..., 0)` printed a hard 0.0000 for any
    # corpus without a sets.json, which reads as "measured, and zero" rather than "not
    # defined for this dataset". An absent metric should be absent from the line.
    _cs = m.get("completeset@5")
    print(f"{tag:22} n={m.get('n')}  MRR {m.get('mrr', 0):.4f}  "
          f"nDCG@5 {m.get('ndcg@5', 0):.4f}  R@10 {m.get('recall@10', 0):.4f}"
          + (f"  CS@5 {_cs:.4f}" if _cs is not None else "  CS@5 n/a"))'''))

MD_10B = ("""## 10b. Did the prior do the work? — three arms in one table

Everything needed is already on disk; this cell computes nothing and trains nothing.

| arm | what it is |
|---|---|
| **multi-view** | 5d `mlp`. The prior `b^(0)` alone, never revised. |
| **no-prior control** | 9d. Same architecture with `use_prior=false`, so documents start at zero. |
| **semantic-prior H3** | 9c. Documents seeded with `b^(0)`, graph revises. |

Three contrasts, printed as explicit deltas:

- **H3 minus no-prior control** is the hypothesis. Both arms have the identical graph,
  objective, negatives, epochs and seed; the only difference is whether documents begin
  with their semantic belief. A win here is attributable to the prior and nothing else.
- **H3 minus multi-view** is what graph revision buys over the scorer alone. This is the
  number that decides whether the graph earns its place at all.
- **no-prior control minus multi-view** says whether the graph is worth anything without
  the prior, which is the arm closest to the stock reasoner.

The fusion arms are deliberately absent. They live in the `_fusion` notebook and are a
different experiment; putting them here would invite a comparison across two training
runs that differ in more than one thing.

An arm whose training cell was never run is reported as missing and dropped from the
table. It is never silently filled with a stale file."""
           if BELIEF else
           """## 10b. The five-arm ladder, official metrics

Every arm through the same scorer, the same slices, the same flags. This cell computes
nothing and trains nothing: it reads the `scores.json` section 10 just wrote.

| arm | semantic scorer | graph | CQIG | fusion |
|---|---|---|---|---|
| **1** | multi-view | off | — | — |
| **2A** | multi-view | on | off | additive |
| **2B** | multi-view | on | off | mixture |
| **3A** | multi-view | on | lam | additive |
| **3B** | multi-view | on | lam | mixture |

**Arm 1 is 5d's standalone `mlp` score.** It is not the `semantic` column of a fusion
run. 5d fits the multi-view scorer with no graph in the objective at all, which is what
"graph reasoning: off" means; a fusion run's semantic channel keeps training under the
fused ranking loss, so reading arm 1 off it would report a scorer that had already seen
the graph and would shrink every graph delta below it.

The contrasts printed underneath each move exactly one thing: 1 to 2A/2B is the graph,
2A to 3A and 2B to 3B is the gate, 2A to 2B and 3A to 3B is the fusion form. Nothing
stacks two changes.

An arm whose training cell was never run is reported as missing and dropped from the
table. It is never silently filled with a stale file."""
           if RUN_SET else
           """## 10b. Scorer x graph — a DIFFERENT 2x2 (not the ladder arms)

Everything needed is already on disk; this cell computes nothing and trains nothing.

| | graph off | graph on |
|---|---|---|
| **operator** | 5d `current` | section 9 |
| **multi-view scorer** | 5d `mlp` | section 9b |

Read the rows for what the graph adds with the scorer held fixed. Read the columns for
what the scorer swap buys with the graph half held fixed. Those are the only two
contrasts the 2x2 supports, and both are printed as explicit deltas so neither has to be
eyeballed.

**The operator row is 5d's `current`, not `operator_scorer.py` from 5b.** 5d refits the
operator's own formula on the same split under the same objective as the multi-view arm,
so the column comparison isolates the architecture. Using 5b would confound it with a
different loss (operator, not multi-gold-corrected) and a different fit slice, and the
delta would not be attributable to the scorer.

An arm whose training cell was never run is reported as missing and dropped from the
table. It is never silently filled with a stale file.""")
cells.append(md(MD_10B))
cells.append(code('''import json, os
TABLE_NAME = __TABLE_NAME__
TABLE_SLUG = __TABLE_SLUG__
SEMD  = f"{S4}/results/semantic_{DATASET}"
LOSSD = globals().get("LOSS", "fixedloss")

# (label, score json). Graph paths come from the run dirs the training cells
# returned, so a re-train points here automatically.
__ARMS4__

S, missing = {}, []
for nm, p in ARMS4:
    if os.path.exists(p):
        S[nm] = json.load(open(p))
    else:
        missing.append((nm, p))
for nm, p in missing:
    hint = "  (its RUN_DIR global is unset: re-run its training cell, or set it by hand)" \\
        if p.startswith("_none_") else ""
    print(f"MISSING  {nm:18} {p}{hint}")
if missing:
    print("   -> dropped from the table rather than faked\\n")

# AN ARM TRAINED LONGER IS NOT A COMPARISON. The epoch count is in the run directory
# name, so a table assembled from run dirs left over from an earlier EPOCHS setting
# silently compares 20 epochs against 10 -- and the longer arm also gets twice as many
# best-epoch draws on the eval split, so it is advantaged twice over.
_ep = {}
for nm, p in ARMS4:
    m = re.search(r"epoch(\\d+)", p)
    if m and nm in S:
        _ep.setdefault(int(m.group(1)), []).append(nm)
if len(_ep) > 1:
    print("EPOCH MISMATCH across arms in this table:")
    for e in sorted(_ep):
        print(f"  {e:>3} epochs: {', '.join(_ep[e])}")
    print("   -> the longer arm is advantaged; retrain the short ones or discount "
          "every delta below\\n")

names = [n for n, _ in ARMS4 if n in S]
assert names, __NEEDRUN__

# Intersect the metrics actually present, so an arm scored with different --cols
# narrows the table instead of raising a KeyError halfway down it.
common = None
for n in names:
    ks = {k for v in S[n].values() if isinstance(v, dict) for k in v if k != "n"}
    common = ks if common is None else (common & ks)
WANT = ("mrr", "ndcg@5", "recall@3", "recall@5", "recall@10",
        "recall@25", "recall@100", "completeset@5")
ROWS = [m for m in WANT if m in common]
gone = [m for m in WANT if m not in common]
if gone:
    print(f"note: {', '.join(gone)} missing from at least one arm; rescore it "
          f"with --cols {','.join(WANT)}\\n")

__CONTRASTS__

live = [c for c in CONTRASTS if c[0] in S and c[1] in S]
dead = [c for c in CONTRASTS if c not in live]
for lo, hi, tag in dead:
    absent = [x for x in (lo, hi) if x not in S]
    print(f"NO CONTRAST  {tag:34} needs {', '.join(absent)}")
if dead:
    print("   -> run the missing arm; the remaining deltas do NOT stand in for it\\n")

# WIDTHS FROM THE CONTENT, NOT A GUESS. A name exactly as wide as the column ran
# straight into its neighbour in the header ("mv+graph+CQIGmv+graph+CQIG node"),
# which reads as a typo in the arm name rather than as two columns.
W = max(18, max(len(n) for n in names) + 2)
CW = max(26, max((len(c[2]) for c in live), default=0) + 2)
out = [f"### {DATASET} test - {TABLE_NAME}"]
for sl in ("all", "same", "cross", "similar", "dissimilar"):
    if any(sl not in S[n] for n in names):
        continue
    n_q = S[names[0]][sl]["n"]
    thin = "   (n is too small to read a difference from)" if n_q < 30 else ""
    out.append(f"\\n[{sl}]  n={n_q}{thin}")
    out.append(f"{'metric':16}" + "".join(f"{n:>{W}}" for n in names))
    for m in ROWS:
        out.append(f"{m:16}" + "".join(f"{S[n][sl][m]:>{W}.4f}" for n in names))
    if live:
        out.append(f"\\n  {'contrast':{CW}}" + "".join(f"{m:>14}" for m in ROWS))
        for lo, hi, tag in live:
            out.append(f"  {tag:{CW}}"
                       + "".join(f"{S[hi][sl][m] - S[lo][sl][m]:>+14.4f}" for m in ROWS))
print("\\n".join(out))

# Distinct filename per experiment: both notebooks share OUT_ROOT, so a single
# name would have each overwrite the other's table.
dst = f"{OUT_ROOT}/{TABLE_SLUG}_{DATASET}.md"
open(dst, "w").write("\\n".join(out) + "\\n")
print(f"\\nwritten to {dst}")'''))

cells.append(md("""## 11. Per-epoch trajectory

Two tables per run. The **component** table is the one that answers whether the graph
is doing anything: `graph` is the graph-alone ranking, computed with the same nDCG
convention as `semantic` and `fused`, so the three are directly comparable.

Read it as:

- **`graph` flat while `hn_graph` falls** — the graph is growing its score *scale*, not
  its ranking. The fusion consumes `z(s_graph)`, which is scale-invariant, so that
  buys nothing.
- **`gamma mean` stuck at its 0.01 init** — the gate never opened and the fused score
  *is* the semantic score.
- **`fused` below `semantic`** — the graph is actively hurting.

The trainer's own metrics are shown underneath. They describe the fused ranking only,
which is why they cannot distinguish any of the cases above."""))
cells.append(code(r'''import re, collections, os, glob

MET  = re.compile(r"\[([\d\-]+ [\d:]+),\d+\].*?" + TEST + r"/document_([a-z@0-9]+): ([0-9.]+)")
NUM  = r"(nan|-?[0-9.]+)"
# The [diag] line printed by the cell-3d patch, which carries all six numbers at once.
DIAG = re.compile(r"\[diag\] nDCG@5\s+semantic " + NUM + r" \| graph " + NUM +
                  r" \| fused " + NUM + r"\s+gamma mean " + NUM + r" min " + NUM +
                  r" max " + NUM)
# Fallback: the same values logged as metric keys, if the print is ever lost.
KEY  = re.compile(r"diag/(ndcg@5_semantic|ndcg@5_graph|ndcg@5_fused|gamma_mean):\s*" + NUM)
NAN  = float("nan")

def _f(x):
    return NAN if x == "nan" else float(x)

def _from_keys(txt):
    """Rebuild rows from diag/* keys; a repeated key starts the next epoch."""
    rows, cur = [], {}
    for k, v in KEY.findall(txt):
        if k in cur:
            rows.append(cur); cur = {}
        cur[k] = _f(v)
    if cur:
        rows.append(cur)
    return [(r.get("ndcg@5_semantic", NAN), r.get("ndcg@5_graph", NAN),
             r.get("ndcg@5_fused", NAN), r.get("gamma_mean", NAN), NAN, NAN) for r in rows]

logs = sorted(glob.glob(f"{OUT_ROOT}/*/console.log"), key=os.path.getmtime)
if not logs:
    print(f"no runs found under {OUT_ROOT}")
for log in logs:
    name = os.path.basename(os.path.dirname(log))
    txt  = open(log, errors="ignore").read()
    diag = [tuple(_f(v) for v in m) for m in DIAG.findall(txt)] or _from_keys(txt)
    ev   = collections.defaultdict(dict)
    for ts, metric, val in MET.findall(txt):
        ev[ts][metric] = float(val)
    if not diag and not ev:
        print(f"\n=== {name} ===\n  no per-epoch metrics in this log")
        continue
    print(f"\n=== {name} ===")

    if diag:
        print("  epoch | semantic |  graph |  fused | gamma mean |    min |    max")
        for i, (sm, gp, fu, gm, gl, gh) in enumerate(diag, 1):
            print(f"  {i:5} |   {sm:6.4f} | {gp:6.4f} | {fu:6.4f} | {gm:10.4f} "
                  f"| {gl:6.4f} | {gh:6.4f}")
        g0, g1 = diag[0][1], diag[-1][1]
        a0, a1 = diag[0][3], diag[-1][3]
        s1, f1 = diag[-1][0], diag[-1][2]
        print(f"\n  graph-only nDCG@5  {g0:.4f} -> {g1:.4f}   (delta {g1 - g0:+.4f})")
        print(f"  gamma mean         {a0:.4f} -> {a1:.4f}   (delta {a1 - a0:+.4f})")
        print(f"  fused vs semantic at the last epoch: {f1 - s1:+.4f}")
    else:
        print("  no component diagnostics: this run predates cell 3d")

    if ev:
        cols = sorted({m for row in ev.values() for m in row})
        print("\n  trainer metrics (fused ranking; MRR is filtered per-gold)")
        print("  epoch | " + " | ".join(f"{c:>9}" for c in cols))
        for i, ts in enumerate(sorted(ev), 1):
            print(f"  {i:5} | " + " | ".join(f"{ev[ts].get(c, NAN):9.4f}" for c in cols))

print("\nBest epoch is selected on the TEST split. Cell 10 has the official "
      "benchmark metrics.")'''))

# run_cell() returns None for ablation cells under --run-set; drop them here so a
# ladder notebook contains only the two rows it claims to run.
cells = [c for c in cells if c is not None]
# THE REPORT LIVES IN THE SAME NOTEBOOK. Splitting "run it" and "read it" into two files
# meant finishing an overnight run and then having to find, upload and mount a second
# notebook to see what happened. The report cells train nothing and are safe to re-run, so
# they belong at the bottom of the notebook that produced the runs. Generated by
# build_compare_notebook.py rather than duplicated here, so the standalone cross-domain
# report and this appendix can never drift apart.
if RUN_SET:
    import subprocess as _sp
    import tempfile as _tf
    _tmp = _tf.NamedTemporaryFile(suffix=".ipynb", delete=False).name
    _sp.run([sys.executable,
             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "build_compare_notebook.py"),
             "--domain", DOMAIN, "--run-set", RUN_SET, "--out", _tmp], check=True,
            stdout=_sp.DEVNULL)
    _rep = json.load(open(_tmp))["cells"]
    # Drive is already mounted by section 1; mounting twice is harmless but prints a
    # confusing second prompt, and the report's title cell duplicates this notebook's.
    _rep = [c for c in _rep
            if "drive.mount" not in "".join(c["source"])
            and not "".join(c["source"]).startswith("# SIR-4 arms:")]
    cells.append(md("---\n\n# Results\n\nEverything below reads the two runs above off "
                    "Drive and trains nothing, so it is safe to re-run at any point. If a "
                    "run has not finished yet it reports MISSING rather than guessing."))
    cells += _rep
    os.unlink(_tmp)
# stamp the build-time run set into the emitted OUT_ROOT.
for _c in cells:
    if _c.get('cell_type') == 'code':
        # NB_RUN_SET, not RUN_SET: the namespace, not the build mode. See their definition.
        _c['source'] = [ln.replace('__RUN_SET__', NB_RUN_SET) for ln in _c['source']]



# ============================================================================ CCMP ARMS
# TWO ARMS AND A TABLE. Everything from section 9 down names run directories this
# notebook never creates, so it is dropped rather than left commented pointing at
# nothing -- the state that produced four MISSING lines in the last ablation.
if CCMP:
    _cut = next((_i for _i, _c in enumerate(cells)
                 if _c and _c["cell_type"] == "markdown"
                 and "".join(_c["source"]).lstrip().startswith(("## 9", "### 9"))), None)
    assert _cut is not None, "no section-9 heading found; the layout changed"
    # Section 10 is the score table -- the entire point of this notebook -- and it sits
    # BELOW section 9 in the cell list, so a plain truncation silently removes it.
    _cut10 = next((_i for _i, _c in enumerate(cells)
                   if _c and _c["cell_type"] == "markdown"
                   and "".join(_c["source"]).lstrip().startswith(("## 10", "## 11"))), None)
    assert _cut10 is not None and _cut10 > _cut, "no scoring section found below section 9"
    _tail = cells[_cut10:]
    cells = cells[:_cut]

    cells.append(md(r"""## 9c. Contrastive Continuation Message Passing (CCMP)

The endpoint loss says *rank this document first*. Six layers of message passing can
produce that same document score through many different internal routes, and nothing in
the objective separates a node that is specifically useful for the gold from a hub wired
to half the corpus. CCMP supervises that choice directly.

**The target.** For an intermediate node $v$ with $L-\ell$ hops of budget left, compare
where its remaining paths actually go:

$$y_{q,\ell}(v)=\frac{B^{+}_{\ell}(v)}{B^{+}_{\ell}(v)+B^{-}_{\ell}(v)},\qquad
c_{q,\ell}(v)=\bigl|2y_{q,\ell}(v)-1\bigr|$$

$B^{+}$ is the probability a walk from $v$ hits a **gold** within the remaining budget,
$B^{-}$ the same for the **semantic scorer's own top-ranked mistakes**. So $y>\tfrac12$
means the node leads more specifically to the gold than to the documents the scorer
already prefers, $y\approx\tfrac12$ means it does not distinguish them, and $c$ down-weights
exactly the nodes that cannot tell the two apart. Hubs land there by construction.

**The head, and the gate.** A shared head predicts $\widehat y^{(\ell)}_{q}(v)$ from the
layer's own state, and that prediction scales the node's outgoing messages. Responsibility
is not an explanation produced after retrieval; it decides what propagates. At inference
no labels are needed, which is the point.

$$\mathcal L=\mathcal L_{\text{fused}}+\lambda_g\mathcal L_{\text{graph-endpoint}}
+\lambda_r\mathcal L_{\text{responsibility}}$$

The endpoint terms are **kept**. The graph still has to rank golds above the scorer's
mistakes; CCMP only says which intermediate nodes should be carrying the query while it
does that.

**Three departures from the formulation as first written**, each of which is a run that
would otherwise be wasted:

1. $\Pr[\text{reach within }h]$ is a **hitting** probability, computed by the absorbing
   recurrence $x\leftarrow\mathbf 1_T+(1-\mathbf 1_T)Px$. The written form
   $\sum_t P^t\mathbf 1_T$ is expected visit mass: unbounded, and it over-counts nodes on
   short cycles, so the ratio is not a ratio of probabilities.
2. $B^{+}$ seeded from $|\mathcal G|$ nodes against $B^{-}$ from $|\mathcal H|$ nodes gives
   $y\approx|\mathcal G|/(|\mathcal G|+|\mathcal H|)\approx0.01$ and $c\approx1$ at nearly
   every node. The objective would collapse to "predict 0, confidently", the gate would shut
   globally and $\delta$ would be the only thing propagating. Each target gets its own
   column here and the two sides reduce **per side**, which is what puts a hub at
   $y\approx0.5$.
3. The gate is **normalised before it is interpolated**:
   $\bar y=(\epsilon+\widehat y)/\operatorname{mean}_{u\in\mathcal A_{q\ell}}(\epsilon+\widehat y)$,
   then $g=(1-\eta)+\eta\bar y$. So $\operatorname{mean}(g)=1$ for **every** $\eta$, and
   $\eta=0$ recovers the ungated GNN exactly, which makes $\eta$ a clean ablation knob.
   The form in the earlier draft, $\delta+(1-\delta)\widehat y$, lies in $[\delta,1]$ and
   can therefore only attenuate: the factors compound with depth, so a perfect gate
   emitting $0.9$ everywhere still leaves a 3-hop route at $0.754$ and a 6-hop route at
   $0.568$. That is a short-path prior imposed on a model whose measured failure is a hop2
   to hop3 rank cliff. The mean is taken over the **activated** set, not all 60k nodes, or
   the gate would scale with how far propagation has spread rather than with what it found.
   At initialisation the head outputs $\widehat y=0.5$ everywhere, so $g$ is **exactly**
   $1.0$ and the CCMP arm's epoch 0 is the control's epoch 0.

Two runs. `gamma` is learned in fp32 in both, so if CCMP makes the graph channel better the
gate can turn it up and the fused score can move."""))

    # CQIG OFF unless explicitly asked for. Its pre-hook edits the same layer input CCMP
    # scales, and it gates on a squared distance from a CALIBRATED mean, so a rescaled
    # state moves CQIG's statistic for reasons that have nothing to do with
    # informativeness. Both arms would still carry it, but part of the CCMP-minus-control
    # difference would then be "CCMP perturbing CQIG", which attributes to neither.
    _CQ = ('cqig=True, cqig_norm="layer", cqig_layers="last", cqig_lam=%s,\n'
           '        cqig_link="semantic",\n        ' % LAM) if LAM_ONLY else ""
    _SFX = "fusion_qwenmlp%s_ccmp" % (("_cqiglam" + LAM_SLUG) if LAM_ONLY else "")
    cells.append(md("### 9c-i. Control"))
    cells.append(code('''RUN_DIR_CCMP_CTRL = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",
        %sccmp=False,
        suffix=f"%sctrl_epoch{EPOCHS}_b{BATCH}")''' % (_CQ, _SFX)))

    cells.append(md(r"""### 9c-ii. CCMP

`ccmp_w` is $\lambda_r$. `ccmp_neg` is $|\mathcal H^{\mathrm{sem}}_q|$. `ccmp_eta` is $\\eta$, the gate strength ($0$ = ungated). `ccmp_m` is how many
nodes per layer are supervised, taken as the top-$M$ by confidence, because everything
below that has $c\approx0$ and contributes no gradient for $L\times N$ of cost.

Targets are cached per query, keyed on `(graph, query id)`, so the sparse propagations are a
first-epoch cost.

**The cache has a known inconsistency, stated here rather than implied away.** An earlier
version of this sentence said the targets depend on "the graph, the seeds and the endpoint
sets, none of which move during training". The graph and the seeds do not move. The negative
endpoints DO: they are the top-$|\mathcal H^{\mathrm{sem}}_q|$ documents under the *current*
semantic score at the moment that query is first seen (`fusion_trainer.py`, the
`_ccmp_cache` miss branch), and the semantic MLP is training jointly. Every query is first
seen during epoch 1, so a query in the first batch is mined against a nearly warm-start
scorer and one in the last batch against a scorer that has had an epoch of updates, and
neither is ever refreshed.

This does not crash and does not break the control comparison: the control has no CCMP
term at all, so nothing about the mining leaks into it. What it does mean is that CCMP's
supervision is not identically defined across training queries. The fix, if this arm is
kept, is to pre-mine all queries from one frozen snapshot of the semantic scorer before
joint training starts, which freezes the CCMP targets without freezing the model."""))
    # STRENGTH MATCHED TO sir4_matsci_ccmp_twostage.ipynb's joint full-router CCMP arm:
    # ccmp_w 1.0 (not 0.1) and the head at 2e-3 against the trunk's 5e-4.
    #
    # ccmp_lr AS A KEYWORD, not os.environ. run_model builds env = dict(os.environ, ...) and
    # then POPS every CCMP_* key before re-setting them from its own arguments, so the
    # two-stage notebook's `os.environ["CCMP_LR"] = "2e-3"` was discarded and that run's head
    # actually trained at 5e-4 despite the comment above it. The keyword is the path that
    # survives the pop. To reproduce what the two-stage run EXECUTED rather than what it
    # intended, set ccmp_lr=None here.
    #
    # This is not a second variable against the control: CCMP_LR only reaches resp_*
    # parameters, which exist only in this arm, and the control has the key popped because
    # its ccmp=False skips the block that sets it.
    cells.append(code('''RUN_DIR_CCMP = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",
        %sccmp=True, ccmp_w=1.0, ccmp_neg=64, ccmp_m=2000,
        ccmp_gate=True, ccmp_gate_norm=True, ccmp_eta=0.5, ccmp_lr=2e-3,
        suffix=f"%sstrong_epoch{EPOCHS}_b{BATCH}")''' % (_CQ, _SFX)))

    if CQIG_ARM:
        cells.append(md(r"""### 9c-iii. CQIG, on its own

The **second hypothesis, against the same control**, never stacked on the first. CCMP
supervises intermediate nodes; CQIG rescales the layer input by how far a node's state sits
from a calibrated reference mean. They edit the same tensor, so an arm with both on cannot
attribute its difference to either one. Two independent arms sharing one control can.

`cqig_lam` is the lever: it sets the gate's span, and every arm before the lambda sweep
measured a no-op because the span was 5%%. This runs at **%s**.

**No semantic linking.** `cqig_link` is left at its default `exact`, so the seeds are the
graph's own exact matches. The embedding-kNN relinker (`cqig_link="semantic"`) was measured
as a wash on top of the lambda lever, and it is a second variable inside what is meant to be
a one-variable arm.

Read the `[cqig]` lines in the log: `eval gate ... span=` should be about %s, not 5e-02, and
`inference: seed-coverage=` is the number to distrust the arm by if it is low.""" % (
            CQIG_ARM_LAM, "%.2e" % (CQIG_ARM_LAM / 2))))
        cells.append(code('''RUN_DIR_CQIG = run_model(epochs=EPOCHS, force_reload=False, semantic="mlp",
        cqig=True, cqig_norm="layer", cqig_layers="last", cqig_lam=%s,
        # cqig_link omitted on purpose -> defaults to "exact". Do NOT add
        # cqig_link="semantic" here: it is a separate lever and a wash on top of lambda.
        ccmp=False,
        suffix=f"fusion_qwenmlp_cqigonly_epoch{EPOCHS}_b{BATCH}")''' % CQIG_ARM_LAM))

    cells.append(md(r"""### 9c-iv. Reading it

**The headline is the fused `nDCG@5` in section 10**, each arm against the shared control, on
`all` and on `dissimilar`. That is the question: does either hypothesis improve results.

Three supporting lines, in the training log:

- **`ccmp`** the continuation loss. If it does not fall, the head cannot fit the target
  and nothing downstream can work.
- **`ccmp_conf`** mean confidence of the supervised nodes. Near 1 across the board means
  the target has collapsed to one side and correction 2 above has not held on this graph.
- **`[diag] graph`** graph-alone nDCG@5, against **`[prior]`**, the parameter-free walk.
  This is where CCMP should show up first, before any of it reaches the fused score. If
  graph-alone does not move, the gate is not changing what the graph ranks and the fused
  column will not move either.

One caveat worth holding on to: the control moved **0.0118** nDCG@5 between two identical
reruns earlier in this project. Treat anything smaller than that as noise, and if CCMP lands
inside it, a second seed of the CCMP arm is worth more than any third variant."""))

    cells.extend(_tail)

nb = {"cells": cells, "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

# ONE substitution pass over the finished notebook. Doing this by f-string in
# the cell bodies would have interpolated every {DATASET}/{DRIVE} placeholder
# that has to survive into the generated code.
# ARMS_SRC carries real newlines and quotes, so it has to be JSON-escaped before it is
# spliced into an already-serialised notebook. json.dumps(x)[1:-1] is that escaping
# without the surrounding quotes.
esc = lambda x: json.dumps(x)[1:-1]
blob = json.dumps(nb).replace("__ARMS__", esc(ARMS_SRC))
blob = blob.replace("__ARMS4__", esc(A4_BELIEF if BELIEF else
                                     A4_FUSION_TOMATO if TOMATO else A4_FUSION))
blob = blob.replace("__CONTRASTS__", esc(C_BELIEF if BELIEF else
                                         C_FUSION_TOMATO if TOMATO else C_FUSION))
blob = blob.replace("__NEEDRUN__", esc(NEED_BELIEF if BELIEF else NEED_FUSION))
blob = blob.replace("__TABLE_NAME__", esc(TN)).replace("__TABLE_SLUG__", esc(TS))
# One source for the multi-layer arm's depth: the prose, the table and the run cell.
blob = blob.replace("__CQIG_ML__", esc(CQIG_ML))
# The belief arm has no fusion: the columns are the prior, the correction ALONE (which is
# not a ranking), and the final score. Calling the last one "fused" invites reading it as
# a combination of two rankings, which is exactly what H3 is not.
L1, L2, L3 = ("prior", "correct", "final") if BELIEF else ("semantic", "graph", "fused")
for k, v in (("__L1__", L1), ("__L2__", L2), ("__L3__", L3)):
    blob = blob.replace(k, v)
blob = blob.replace("__FOREIGN__", FOREIGN)
blob = blob.replace("__PROBEDIR__", "" if TOMATO else "{DATASET}/")
# THE SAME LEGACY RULE, THIRD TIME. operator_scorer.py:279 builds its params filename as
#     sfx = ("" if a.dataset == "tomato" else f"_{a.dataset}") + model_slug(a.model)
# so on TOMATO it writes `operator_params_content-qwen3.json` with NO dataset segment,
# while the notebook read `operator_params_{DATASET}{OP_SLUG}.json`. Phase 3 then finished
# successfully and the next cell died on FileNotFoundError. The placeholder absorbs the
# separating underscore so both spellings stay exact.
blob = blob.replace("__PARAMDS__", "" if TOMATO else "_{DATASET}")
# FOURTH INSTANCE, and the expensive one. scigraphir_paths.emb_dir() is UNSCOPED on TOMATO
# (outputs/caches/op_emb) and scoped elsewhere (.../op_emb/sir4_cs). save_cache() looked
# in the scoped path unconditionally, so on TOMATO it found nothing, copied no embeddings
# to Drive, and restore_cache() put them back somewhere the encoder never reads. Every
# TOMATO rerun therefore re-encoded the whole corpus from scratch, silently.
blob = blob.replace("__EMBDS__", "" if TOMATO else "/{DATASET}")
# SETS. TOMATO HAS NO sets.json AND MUST NOT BORROW ONE. Leaving the SIR-4 default in
# place pointed a TOMATO run at `cs_test_final/sets.json`, a CS resource whose query ids
# cannot match, so CompleteSet@k would either be silently zero or, worse, computed against
# another corpus's sets. Point it at a path under the dataset's own name: it does not
# exist, os.path.exists(SETS) is False, and every caller already guards on that.
blob = blob.replace("__SETSDIR__", "{DATASET}_test_sets_DO_NOT_EXIST" if TOMATO
                    else (SETS_DIR or "cs_test_final"))
# CompleteSet@k needs sets.json, which TOMATO does not have. Requesting the column anyway
# would print a row of zeros that reads as a measured result rather than an undefined one.
blob = blob.replace("__CSET__", "" if TOMATO else ",completeset@5")
blob = blob.replace("__CSETL__", "" if TOMATO else "completeset@5")
for a, b in (("sir4_cs", DSET),
             # The sets dir is now carried by the __SETSDIR__ placeholder above, which is
             # the only place it appears, so there is no literal left to substitute here.
             # TOMATO's train graph is 106,653 nodes against SIR-4 CS's 243,390, so it has
             # headroom SIR-4 does not -- but NOT enough for 4. Measured: batch 4 OOMs on an
             # 80GB A100 on the CCMP arm, at 76.97 GiB in use before an 834 MiB request.
             # The historical TOMATO run that used 4 had no CCMP head (six resp_proj layers
             # over [B, N, 1024]), no learned MLP scorer, and no 5-feature gate.
             #
             # 2 FOR EVERY ARM, NOT 4-FOR-SOME-2-FOR-CCMP. Batch is not a free knob at a
             # fixed epoch count: it sets the optimizer-step count, so a CCMP arm at 2
             # against its control at 4 would vary two things and attribute to neither.
             ("BATCH = 2", "BATCH = %d" % (_a.batch if _a.batch is not None
                                           else (2 if TOMATO else 1))),
             # PROSE, not just paths. Every generated notebook was titled "SIR-4 CS"
             # regardless of domain, because only the snake_case name was substituted,
             # so the physics run carried a CS heading into its own results.
             ("SIR-4 CS", "TOMATO-Star" if TOMATO else f"SIR-4 {DOMAIN.upper()}"),
             # The heading has to name the EXPERIMENT, not the shared machinery. Every
             # arm-specific build inherited "additive-gate fusion", so a CCMP notebook and
             # a mixture notebook opened with the same sentence and the reader had to find
             # a run_model call forty cells down to tell them apart.
             ("additive-gate fusion (operator \u2295 v16sc graph)",
              "H3 semantic-prior graph reasoning" if BELIEF else
              ("CCMP over CQIG \u03bb=%s: control vs CCMP" % LAM if LAM_ONLY else
               "CCMP: control vs contrastive responsibility propagation") if CCMP else
              "additive-gate fusion (operator \u2295 v16sc graph)"),
             ("Within-domain CS run", "TOMATO-Star run" if TOMATO
              else f"Within-domain {DOMAIN.upper()} run")):
    blob = blob.replace(a, b)
# The reused fusion implementation predates the Qwen operator. Its executable
# code is encoder-agnostic, but make the generated notebook's documentation
# describe the components actually supplied above.
blob = blob.replace("The BGE encoder that produced dense/S/M is frozen",
                    "The Qwen3 encoder that produced dense/S/M is frozen")
nb = json.loads(blob)

# Same reasoning as the CCMP block below, and rewritten here for the same reason: the
# blob table runs over json.dumps output where U+2295 is already escaped, so a key with
# the literal character cannot match.
if NOENT:
    _h = "".join(nb["cells"][0]["source"])
    _old = "additive-gate fusion (operator ⊕ v16sc graph)"
    assert _old in _h, "the ladder heading is not where this expected it"
    # The corpus table quotes the AS-BUILT seed counts, which this notebook no longer
    # runs on. Leaving them would put 76.9 at the top of a run whose whole point is 35.7.
    _seeds = "| query seeds | 76.85 | 76.96 |"
    assert _seeds in _h, "the corpus table's seed row moved; fix this replacement"
    nb["cells"][0]["source"] = (_h.replace(
        _seeds, "| query seeds (as built) | 76.85 | 76.96 |\n"
                "| **query seeds (this run)** | **35.29** | **35.74** |").replace(
        _old, "no entity seeds (SIR-4's seeding recipe on TOMATO)") + """

**THE ONE VARIABLE.** Identical graph, identical corpus, identical objective; the only
change is that section 5a-noent removes the `entity` seed channel from `start_nodes`
before anything reads it. `run_domain.py` builds every SIR-4 graph with
`--no_entity_seeds`; TOMATO was built with the builder's default, which is on.

| | TOMATO as built | this notebook | SIR-4 |
|---|---|---|---|
| seed channels | task, method, function, limitation, **entity** | task, method, function, limitation | task, method, function, limitation |
| seeds per query (test) | 76.96 | **35.74** | — |

The entity channel snaps raw query terms to the nearest node of any type, so it is
term-level topical matching rather than role-typed structure. It is 53.6% of all seeds,
and it is the seeding that most undercuts the claim that a retrieved path shows a
document *supplies something relevant*, rather than merely discussing the same topic.

**The graph is not rebuilt, and does not need to be.** In `build_greasoner_dataset.py`
the node map and edge set are finalised before the first read of `entity_seeds`, which
lives entirely in the query-seeding block, so `nodes.csv`, `edges.csv` and
`relations.csv` are byte-identical either way. Verified locally: identical hashes, and
zero queries left seedless.

**Read `prior/ndcg@5_walk` first.** It is parameter-free, so it depends on the seeding
and nothing that trains. The entity arms print 0.1442. Section 8 asserts this run does
not, because an identical value would mean a cached `stage2/*/{train,test}.pt` was
reused and the arm is measuring entity seeds under a no-entity label.
""").splitlines(True)

# WHAT THIS NOTEBOOK IS, at the top, in the reader's language. Cell 0 describes the shared
# ladder because that is what the source notebook is. An arm-specific build that does not
# restate its own purpose is unreadable six weeks later, and worse, unreadable side by side:
# four CCMP notebooks existed, one on a different fusion router and one stacked on CQIG, and
# nothing above cell 40 distinguished them.
if CCMP and not TOMATO:
    # THE HEADING, rewritten here and not in the blob replacement table above. That table
    # runs over json.dumps(nb) output, where ensure_ascii has turned the title's U+2295 into
    # the six characters ⊕, so a key containing the literal character cannot match. The
    # entry for it has therefore never fired -- the BELIEF build is mistitled for the same
    # reason, which is a separate bug this does not touch.
    _h = "".join(nb["cells"][0]["source"])
    _old = "additive-gate fusion (operator ⊕ v16sc graph)"
    _new = ("CCMP over CQIG λ=%s: control vs CCMP" % LAM) if LAM_ONLY else \
           ("CCMP and CQIG, each against one shared control" if CQIG_ARM else
            "CCMP vs control (contrastive responsibility propagation)")
    assert _old in _h, "the ladder heading is not where this expected it"
    nb["cells"][0]["source"] = _h.replace(_old, _new).splitlines(True)

    _base = ("CQIG at λ=%s with semantic relinking, enabled identically in BOTH arms"
             % LAM) if LAM_ONLY else (
        "the plain fusion reasoner. CQIG is off in the control and in the CCMP arm, and is "
        "the single thing the CQIG arm adds" if CQIG_ARM else
        "the plain fusion reasoner, no CQIG")
    _reads = ("`ccmp` (the continuation loss, must fall), `ccmp_conf` (mean confidence of "
              "the supervised nodes, near 1 everywhere means the target collapsed), and "
              "`[diag] graph` against `[prior]`")
    _rows = ("""| control | `..._ccmpctrl_epoch{EPOCHS}_b{BATCH}` | off | off | nothing added |
| CCMP | `..._ccmpstrong_epoch{EPOCHS}_b{BATCH}` | **on** | off | + the responsibility loss and gate |
| CQIG | `fusion_qwenmlp_cqigonly_epoch{EPOCHS}_b{BATCH}` | off | **on** | + the informativeness gate, λ=%s |"""
             % CQIG_ARM_LAM) if CQIG_ARM else \
            """| control | `..._ccmpctrl_epoch{EPOCHS}_b{BATCH}` | off | off | nothing added |
| CCMP | `..._ccmpstrong_epoch{EPOCHS}_b{BATCH}` | **on** | off | + the responsibility loss and gate |"""
    _n = "three runs" if CQIG_ARM else "twice"
    _cqig_para = ("""
**What CQIG does.** It rescales each layer's input by how far a node's state sits from a
calibrated reference mean, so uninformative states are attenuated before they propagate.
`cqig_lam` sets the gate's span and is the whole lever: every arm before the sweep measured
a no-op because the span was 5%%. This runs at λ=%s with `cqig_link` left at its default
`exact`, so **no semantic relinking** -- that is a separate lever, measured as a wash on top
of λ, and it would be a second variable inside a one-variable arm.

**Never both on.** CCMP rescales a layer input and CQIG gates on a squared distance from a
calibrated mean of that same input. An arm with both on cannot attribute its difference to
either one, so the two are separate arms over one shared control.
""" % CQIG_ARM_LAM) if CQIG_ARM else ""
    nb["cells"].insert(1, {"cell_type": "markdown", "metadata": {}, "source":
                           ("""## What this notebook is

**A controlled comparison on %s. It trains %s and scores them through one scorer.**

| arm | run dir suffix | ccmp | cqig | differs how |
|---|---|---|---|---|
%s

**Base for every arm:** %s.

**What CCMP does.** The graph reasoner is trained only on its final document scores, so the
intermediate nodes a path runs through get no supervision at all. CCMP supervises them:
each node is labelled by whether its remaining bounded-hop paths lead more strongly to a
gold than to the semantic scorer's own hard negatives, and the predicted responsibility
then gates that node's outgoing messages. The claim being tested is that this makes the
graph channel better, not that it makes the fusion better.
%s
**Settings, identical in every arm except the one block being tested:** additive fusion,
learned per-query γ in fp32, the learned multi-view semantic MLP training jointly (not
frozen), per-gold contrastive loss, 50 semantic + 50 graph + 50 random hard negatives, graph
auxiliary weight 1.0. The CCMP arm adds λ_r=1.0, 64 semantic negatives per query, M=2000
supervised nodes per layer, η=0.5, and a responsibility-head learning rate of 2e-3 against
the trunk's 5e-4. Its gate starts as the identity, so epoch 0 is the same ranking as the
control's, and the head learning rate reaches only `resp_*` parameters, which exist in no
other arm. Strength matched to `sir4_matsci_ccmp_twostage.ipynb`.

**How to read it.** The headline is the fused `nDCG@5` in section 10, each arm against the
shared control, on `all` and on `dissimilar`. Three supporting lines in the training log: %s.

**What this notebook is not.** Not a ladder rung and not a table row: each arm varies one
thing against a common control. Section 9c has the arms; sections 1-8 are the shared setup.
""" % (DOMAIN.upper(), _n, _rows, _base, _cqig_para, _reads)).splitlines(True)})

assert DSET in json.dumps(nb), "domain substitution did not take"
if TOMATO or DOMAIN != "cs":
    assert "sir4_cs" not in json.dumps(nb), "a cs reference survived"
# The guard is the one cell whose correctness cannot be checked at runtime: if the
# placeholder survived, the emitted guard tests for the literal string "__FOREIGN__",
# which never matches, so a contaminated run would sail through it silently.
assert "__FOREIGN__" not in json.dumps(nb), "the isolation guard's placeholder survived"


# GENERATED CELLS MUST PARSE BEFORE THIS FILE IS WRITTEN.
# A cell template is a Python string inside a Python file, so an escape correct in
# one layer is wrong in the other: "\n" inside a non-raw triple-quoted template
# becomes a REAL newline in the emitted cell and leaves a string literal
# unterminated. That has now shipped twice. IPython magics are not Python, so blank
# them (and their backslash continuations) rather than skipping whole cells.
import ast as _ast
_broken = []
for _i, _c in enumerate(cells):
    if _c["cell_type"] != "code":
        continue
    _out, _mag = [], False
    for _ln in "".join(_c["source"]).split(chr(10)):
        if _ln.lstrip().startswith(("!", "%")) or _mag:
            _mag = _ln.rstrip().endswith(chr(92)); _out.append("pass")
        else:
            _out.append(_ln)
    try:
        _ast.parse(chr(10).join(_out))
    except SyntaxError as _e:
        _broken.append((_i, _e.lineno, _e.msg))
if _broken:
    for _i, _l, _m in _broken:
        print(f"  cell {_i} line {_l}: {_m}")
    raise SystemExit(f"{len(_broken)} generated cell(s) do not parse; {OUT} not written")

json.dump(nb, open(OUT, "w"), indent=1)
print(f"wrote {OUT}  ({len(cells)} cells, {os.path.getsize(OUT)/1024:.0f} KB)")
print("reused verbatim from the TOMATO notebook: engine install, fusion model files, Qwen3 fetch")
