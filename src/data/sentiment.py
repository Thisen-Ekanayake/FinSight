# ═══════════════════════════════════════════════════════
# FinSight — Headline Sentiment
# ═══════════════════════════════════════════════════════
#
# Purpose : Score a news headline in [-1, +1] with a finance-specific lexicon.
#
# Public API:
#   score_sentiment(headline, summary="") -> float
#   matched_terms(text) -> tuple[list[str], list[str]]
#
# ══ WHY THIS EXISTS AT ALL ══
#   NewsItem.sentiment was specified as provider-supplied. It is not available:
#   Finnhub's per-article sentiment sits behind their paid tier, and the Yahoo
#   RSS fallback has never had one. Both providers were returning None for
#   every article, which meant the news monitor filtered on
#   `sentiment is not None` and could NEVER emit a candidate.
#
#   Measured on a live cycle: 250 NVDA articles, 0 scored, 0 candidates. One of
#   the four monitors was structurally dead, and it looked exactly like a quiet
#   news week.
#
# ══ WHY A LEXICON AND NOT A MODEL ══
#   A cycle can pull 250 articles per ticker. Scoring those with an LLM would
#   be 1,250 calls per cycle against a subsystem whose entire budget is ~26
#   external requests — a hundredfold increase to rank headlines the severity
#   rules then reduce to three buckets.
#
#   A local classifier would need weights, a download, and a warm-up on the
#   critical path of every cycle. This is a dictionary lookup.
#
# ══ THE SCORE IS DELIBERATELY CRUDE, AND THE DESIGN ALREADY KNOWS ══
#   Keyword sentiment misreads sarcasm, negation, and context. That is why
#   HIGH severity requires NEWS_HIGH_MIN_SOURCES independent outlets rather
#   than one confident number: the corroboration rule exists precisely because
#   this score is noisy. A single mis-scored headline is capped at MED and
#   cannot page anyone.
#
#   Treat it as a ranking signal, not a measurement. If it ever needs to be
#   better, the honest upgrade is FinBERT on the ~5 headlines that survive
#   NEWS_MAX_CANDIDATES_PER_TICKER, not a model over all 250.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Weight 2.0 — terms that are near-unambiguous in a financial headline.
STRONG_NEGATIVE: frozenset[str] = frozenset(
    {
        "bankruptcy",
        "fraud",
        "fraudulent",
        "delisting",
        "delisted",
        "subpoena",
        "subpoenaed",
        "indicted",
        "indictment",
        "probe",
        "investigation",
        "investigating",
        "lawsuit",
        "sues",
        "sued",
        "recall",
        "recalls",
        "plunge",
        "plunges",
        "plunged",
        "crash",
        "crashes",
        "halted",
        "resigns",
        "resigned",
        "ousted",
        "layoffs",
        "guilty",
        "collapse",
        "collapses",
        "breach",
        "outage",
        "impairment",
        "restatement",
        "restate",
        "insolvency",
        "default",
        "downgrade",
        "downgrades",
        "downgraded",
    }
)

# Weight 1.0 — directional but context-dependent.
WEAK_NEGATIVE: frozenset[str] = frozenset(
    {
        "falls",
        "fell",
        "drops",
        "dropped",
        "declines",
        "declined",
        "slides",
        "slumps",
        "misses",
        "missed",
        "miss",
        "cuts",
        "cut",
        "slashes",
        "warns",
        "warning",
        "weak",
        "weaker",
        "weakness",
        "concerns",
        "concern",
        "scrutiny",
        "delay",
        "delays",
        "delayed",
        "loss",
        "losses",
        "fine",
        "fined",
        "penalty",
        "shortfall",
        "headwind",
        "headwinds",
        "slowdown",
        "sluggish",
        "disappointing",
        "disappoints",
        "risk",
        "risks",
        "pressure",
    }
)

STRONG_POSITIVE: frozenset[str] = frozenset(
    {
        "record",
        "surges",
        "surged",
        "soars",
        "soared",
        "breakthrough",
        "approval",
        "approved",
        "acquires",
        "acquisition",
        "wins",
        "won",
        "upgrade",
        "upgrades",
        "upgraded",
        "rally",
        "rallies",
        "outperform",
        "blowout",
    }
)

WEAK_POSITIVE: frozenset[str] = frozenset(
    {
        "beats",
        "beat",
        "raises",
        "raised",
        "rises",
        "rose",
        "gains",
        "gained",
        "climbs",
        "expands",
        "expansion",
        "partnership",
        "launch",
        "launches",
        "growth",
        "strong",
        "stronger",
        "improves",
        "improved",
        "boosts",
        "boosted",
        "optimistic",
        "tailwind",
    }
)

_WEIGHTS: list[tuple[frozenset[str], float]] = [
    (STRONG_NEGATIVE, -2.0),
    (WEAK_NEGATIVE, -1.0),
    (STRONG_POSITIVE, 2.0),
    (WEAK_POSITIVE, 1.0),
]

_TOKEN = re.compile(r"[a-z]+")

# Smoothing constant. Keeps a single weak hit from reading as a strong signal:
# one weak negative scores -1/(1+1.5) = -0.40, one strong -2/(2+1.5) = -0.57,
# two strong -4/(4+1.5) = -0.73. Those land sensibly either side of the
# NEWS_MED_SENTIMENT (-0.30) and NEWS_HIGH_SENTIMENT (-0.60) thresholds.
SMOOTHING: float = 1.5

# The headline is the claim; the summary is supporting detail that repeats and
# hedges it. Weighting them equally lets a long body outvote the headline.
SUMMARY_WEIGHT: float = 0.4

# Only the first part of a summary is read. Feed summaries frequently end in
# boilerplate ("...nothing in this article constitutes investment advice"),
# which is lexically negative and identical across every article.
MAX_SUMMARY_CHARS: int = 400


def matched_terms(text: str) -> tuple[list[str], list[str]]:
    """
    Return the (negative, positive) lexicon terms found in text.

    Exposed so a surprising score can be explained rather than argued with.

    Parameters
    ----------
    text : str

    Returns
    -------
    tuple
        ``(negative_terms, positive_terms)``, each sorted.
    """
    tokens = set(_TOKEN.findall(text.lower()))
    negative = sorted(tokens & (STRONG_NEGATIVE | WEAK_NEGATIVE))
    positive = sorted(tokens & (STRONG_POSITIVE | WEAK_POSITIVE))
    return negative, positive


def _weighted(text: str) -> tuple[float, float]:
    """Total negative and positive weight in one piece of text."""
    tokens = set(_TOKEN.findall(text.lower()))

    negative = 0.0
    positive = 0.0
    for lexicon, weight in _WEIGHTS:
        hits = len(tokens & lexicon)
        if not hits:
            continue
        if weight < 0:
            negative += hits * abs(weight)
        else:
            positive += hits * weight

    return negative, positive


def score_sentiment(headline: str, summary: str = "") -> float:
    """
    Score a headline in [-1, +1]; negative is bad news.

    Terms are counted as a SET rather than by occurrence, so a headline that
    repeats "probe" three times scores the same as one that says it once. A
    repeated word is emphasis, not evidence, and counting it would let one
    long article dominate a corroboration count.

    Parameters
    ----------
    headline : str
        The article headline. Carries most of the weight.
    summary : str, optional
        Body or standfirst. Down-weighted and truncated — feed summaries end in
        boilerplate disclaimers that are lexically negative and identical
        across every article from that source.

    Returns
    -------
    float
        0.0 when no lexicon term matched, which is the common case and means
        "no signal" rather than "neutral news". The monitor's
        NEWS_MIN_ABS_SENTIMENT floor discards it either way.
    """
    negative, positive = _weighted(headline)

    if summary:
        summary_negative, summary_positive = _weighted(summary[:MAX_SUMMARY_CHARS])
        negative += summary_negative * SUMMARY_WEIGHT
        positive += summary_positive * SUMMARY_WEIGHT

    total = negative + positive
    if total == 0:
        return 0.0

    return round((positive - negative) / (total + SMOOTHING), 4)
