"""Shared normalisation for cross-benchmark matching.

ResearchBench candidates carry a title but no DOI; TOMATO-Star carries both. Matching is
therefore title-first, DOI-confirmed. Both keys have to be aggressive enough to survive
publisher formatting differences (smart quotes, Greek letters, trailing periods, HTML
entities) without collapsing genuinely different papers onto one key.
"""
import re
import unicodedata

_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9 ]+")
_DOI_PREFIX = re.compile(r"^(https?://)?(dx\.)?(doi\.org/)?(doi:)?", re.I)


def norm_title(t) -> str:
    """NFKD-fold, drop non-ASCII, lowercase, keep alphanumerics, collapse whitespace.

    'π-Extended Tetraphenylethylene' -> 'extended tetraphenylethylene'
    'Cardiac Measures of Cognitive Workload: A Meta-Analysis.' -> 'cardiac measures of
    cognitive workload a meta analysis'
    """
    if not t:
        return ""
    t = unicodedata.normalize("NFKD", str(t))
    t = t.encode("ascii", "ignore").decode("ascii").lower()
    t = _NONALNUM.sub(" ", t)
    return _WS.sub(" ", t).strip()


def norm_doi(d) -> str:
    """Strip resolver prefixes and case. DOIs are case-insensitive by specification."""
    if not d:
        return ""
    d = _DOI_PREFIX.sub("", str(d).strip())
    return d.lower().rstrip(".")


def title_tokens(t) -> frozenset:
    """Token set for near-duplicate detection, stopwords dropped."""
    stop = {"a", "an", "the", "of", "in", "on", "for", "and", "to", "with", "by",
            "from", "at", "as", "is", "are", "via", "using"}
    return frozenset(w for w in norm_title(t).split() if w not in stop and len(w) > 2)
