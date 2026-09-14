#!/usr/bin/env python3
"""How QUARTET decides a gold inspiration is same-domain or cross-domain.

ONE module, imported by Stage 4, Stage 5 and build/relabel.py, so the definition
cannot drift between the file that writes the labels, the file that exports them
and the script that changes them later.

WHY THE DEFINITION IS NOT A CONSTANT.

The first version compared each paper's OpenAlex `primary_topic.field`, one
label per side. An audit of 544 gold occurrences found that rule unusable:

  15%  of labels contradicted their own topic list. `primary_topic` is the field
       of the single top-scoring topic, so "Lieb-Robinson Bounds..." was labelled
       Physics while 2 of its 3 topics said Computer Science.
  33%  of cross rows were a coin flip: the target's own field was tied for top
       among the gold's topics, e.g. {Mathematics 1, Engineering 1, CS 1}.
   7%  claimed purity 1.00 on a single topic, which is not evidence.

All three come from the same mistake: forcing an interdisciplinary paper into one
of 26 buckets and then comparing buckets. Comparing the topic SETS removes the
forced choice, needs no tiebreak, and drops nothing.

WHICH CUT, AND WHY.

Chosen the way TOMATO-Star justified its own: by whether the bands separate
retrieval difficulty. BM25 over the 234-document pilot corpus, 544 golds:

    band                        n    R@5   R@10   R@25   med
    primary fields equal      287   62.4   72.8   80.1     4
    2 or 3 shared topics       65   61.5   69.2   73.8     4   <- same as above
    exactly 1 shared          120   45.8   58.3   64.2     6   <- intermediate
    no shared topic            72   37.5   48.6   59.7    14

Sharing 2 of 3 topic fields behaves exactly like sharing the primary one, which
is why the "which of the three did OpenAlex call primary" question stops
mattering. As a binary, the two candidate cuts separate identically:

    cross = no shared field       same 472 / cross 72    gap 9.1 pts (1.28x)
    cross = at most 1 shared      same 352 / cross 192   gap 9.3 pts (1.27x)

DISJOINT is the default, because it separates just as well and states in one
sentence with no threshold to defend: cross means the two papers share no field.

CAVEAT, and it is on the record deliberately: 234 documents is a small corpus and
compresses every score toward the ceiling, so those absolute numbers are not
comparable to TOMATO's 3,033-document table. The ORDERING is monotonic at every
cutoff, which is what the choice rests on. Recompute the table at full scale
before quoting a gap.

CHANGING YOUR MIND LATER IS FREE.

Stage 4 stores the EVIDENCE, not just the verdict: the topic-field distribution
of both sides, and which fields they share. Every policy below is a pure function
of that, so switching costs no API calls, no LLM calls and no rerun:

    python3 build/relabel.py --benchmark data/benchmark/cs_test --policy overlap2
"""
from __future__ import annotations

import re
import unicodedata

__all__ = ["POLICIES", "DEFAULT_POLICY", "BANDS", "shared_fields",
           "label_evidence", "apply_policy", "norm_field"]

DEFAULT_POLICY = "disjoint"

# The graded band, always recorded whatever the binary policy is. A policy
# chooses where to cut this scale; it never changes the scale itself.
BANDS = ["same_primary", "overlap_2plus", "overlap_1", "disjoint"]


def norm_field(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def shared_fields(target_topics: dict | None, insp_topics: dict | None) -> list[str]:
    """The OpenAlex fields both papers' topics touch, in a stable order.

    Keyed on the normalised name but returned with the target's spelling, so the
    stored evidence reads the way OpenAlex writes it.
    """
    t = {norm_field(k): k for k in (target_topics or {}) if k}
    i = {norm_field(k) for k in (insp_topics or {}) if k}
    return sorted(t[k] for k in (set(t) & i))


def label_evidence(target_topics: dict | None, insp_topics: dict | None,
                   target_primary: str | None, insp_primary: str | None,
                   target_domain: str | None = None,
                   insp_domain: str | None = None) -> dict:
    """Everything any policy needs, computed once and stored.

    This is what makes relabelling free. Nothing here is a verdict; it is the
    two topic distributions reduced to the facts a policy asks about.
    """
    sh = shared_fields(target_topics, insp_topics)
    both_primary = bool(target_primary and insp_primary)
    labelled = bool(target_topics and insp_topics and both_primary)
    return {
        "shared_fields": sh,
        "n_shared": len(sh),
        "n_target_fields": len(target_topics or {}),
        "n_insp_fields": len(insp_topics or {}),
        "primary_equal": bool(both_primary
                              and norm_field(target_primary) == norm_field(insp_primary)),
        "same_openalex_domain": (None if not (target_domain and insp_domain)
                                 else target_domain == insp_domain),
        "labelled": labelled,
        # Enough for the FALLBACK below, though not for the topic-set policies.
        # Two things cause the gap, both metadata rather than method:
        #   a 2026 target OpenAlex has filed under a field but not yet given
        #   topics, and an inspiration resolved through Semantic Scholar, which
        #   has no field taxonomy of its own.
        "primary_known": both_primary,
    }


def band_of(ev: dict) -> str | None:
    """The graded scale. Policy-independent."""
    if not ev.get("labelled"):
        return None
    if ev["primary_equal"]:
        return "same_primary"
    n = ev["n_shared"]
    return "overlap_2plus" if n >= 2 else ("overlap_1" if n == 1 else "disjoint")


# --------------------------------------------------------------------- policies
# Each returns "same" or "cross" from the evidence alone. Add one here and every
# stage and the relabel script pick it up with no other change.

def _disjoint(ev: dict) -> str:
    """DEFAULT. Cross means the two papers share no OpenAlex field at all.

    The strictest reading, so it cannot be accused of inflating the headline.
    Measured at 13% of golds on the pilot, against TOMATO's 9.2%, which makes
    the two numbers comparable for the first time.
    """
    return "cross" if ev["n_shared"] == 0 else "same"


def _overlap2(ev: dict) -> str:
    """Cross unless the papers agree on their primary field or share two fields.

    Separates as well as `disjoint` (9.3 pts against 9.1) and puts the
    intermediate `overlap_1` regime on the cross side, which is defensible: that
    band's R@5 of 45.8 sits closer to disjoint's 37.5 than to same's 62.4.
    Costs you a threshold you then have to justify.
    """
    return "same" if (ev["primary_equal"] or ev["n_shared"] >= 2) else "cross"


def _primary(ev: dict) -> str:
    """The ORIGINAL rule, kept only so the audit is reproducible.

    Compares one forced label per side. Do not report from this: 15% of its
    labels contradict their own topic list and a third of its cross rows are a
    tiebreak. It is here to regenerate the "as shipped" column of the audit.
    """
    return "same" if ev["primary_equal"] else "cross"


def _domain_jump(ev: dict) -> str:
    """Cross only when the papers also sit in different OpenAlex DOMAINS.

    The coarsest option, four buckets rather than 26. Included because it is the
    level TOMATO's published split uses, so it is the like-for-like comparison
    if a reviewer asks for one. Too coarse for QUARTET on its own: cs, physics,
    maths and matsci all live inside Physical Sciences, so four of the five
    splits would collapse into one bucket.
    """
    if ev.get("same_openalex_domain") is None:
        return _disjoint(ev)
    return "cross" if (ev["n_shared"] == 0
                       and not ev["same_openalex_domain"]) else "same"


POLICIES = {
    "disjoint": _disjoint,
    "overlap2": _overlap2,
    "primary": _primary,
    "domain_jump": _domain_jump,
}


def apply_policy(ev: dict, policy: str = DEFAULT_POLICY) -> dict:
    """{domain_relation, domain_distance, label_basis, label_policy} for one gold.

    `domain_distance` is the graded band and does not depend on the policy, so
    switching the binary never silently changes the scale underneath it.

    THE FALLBACK, and why `label_basis` is not optional.

    The topic-set policies need a topic distribution from BOTH papers, and
    OpenAlex does not always have one: a 2026 paper can be filed under a field
    before it is given topics, and an inspiration resolved through Semantic
    Scholar carries no OpenAlex taxonomy at all. Those rows used to come out
    unlabelled even when both PRIMARY fields were sitting right there.

    So when the full evidence is missing but both primary fields are known, the
    label falls back to comparing them. That recovers most of the gap, and it is
    recorded as `label_basis = "primary_fallback"` rather than blended in,
    because the primary rule is a MEASURABLY DIFFERENT rule: on the pilot it
    called 40.8% of golds cross where `disjoint` called 12.4%. Counting a
    fallback row as though a topic-set policy had produced it would import that
    bias into the headline split. Filter on `label_basis == "topics"` for a
    clean number, and report the fallback rows separately.
    """
    if policy not in POLICIES:
        raise KeyError(f"unknown label policy {policy!r}. "
                       f"Known: {', '.join(sorted(POLICIES))}")
    if ev.get("labelled"):
        return {"domain_relation": POLICIES[policy](ev),
                "domain_distance": band_of(ev),
                "label_basis": "topics",
                "label_policy": policy}
    if ev.get("primary_known"):
        return {"domain_relation": "same" if ev["primary_equal"] else "cross",
                # The graded band needs topic SETS, so only the equal case
                # survives the fallback. Claiming an overlap band here would be
                # inventing evidence.
                "domain_distance": "same_primary" if ev["primary_equal"] else None,
                "label_basis": "primary_fallback",
                "label_policy": policy}
    return {"domain_relation": None, "domain_distance": None,
            "label_basis": None, "label_policy": policy}
