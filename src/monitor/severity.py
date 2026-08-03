# ═══════════════════════════════════════════════════════
# FinSight — Deterministic Severity
# ═══════════════════════════════════════════════════════
#
# Purpose : Score a candidate LOW / MED / HIGH from rules, never from a model.
#
# Public API:
#   severity_for(candidate) -> Severity
#   explain(candidate)      -> str        which rule fired, for the audit trail
#
# ══ WHY THIS IS NOT AN LLM CALL ══
#   Three reasons, in ascending order of importance.
#
#   1. Testable. "An 8-K carrying Item 4.02 is HIGH" is an assertion a unit
#      test can hold the code to. "The model judged it serious" is not.
#   2. Stable. The same filing scores the same in March and September, which
#      is what makes an alert history comparable to itself.
#   3. Honest. A model asked to rate urgency inflates it — sounding alarmed is
#      how a chat assistant demonstrates that it understood you. An alerting
#      system whose severity drifts upward is one nobody reads, and a system
#      nobody reads has a false-negative rate of 100%.
#
#   The LLM's job in this subsystem is exactly one thing: writing the
#   qualitative summary that gets embedded. It never decides what matters.
#
# ══ THE RULES ESCALATE, THEY DO NOT AVERAGE ══
#   Every rule below returns a floor, and the HIGHEST floor wins. A 5% drop
#   that is also a 3-sigma move is HIGH, not "MED-ish". Averaging severity
#   signals is how a genuinely alarming event gets smoothed into a shrug.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging

from src.core.schemas import Severity
from src.monitor.config import (
    FILING_HIGH_8K_ITEMS,
    FILING_MED_8K_ITEMS,
    FILING_PERIODIC_FORMS,
    MACRO_THRESHOLDS,
    NEWS_HIGH_MIN_SOURCES,
    NEWS_HIGH_SENTIMENT,
    NEWS_MED_SENTIMENT,
    PRICE_HIGH_PCT,
    PRICE_HIGH_ZSCORE,
    PRICE_MED_PCT,
    PRICE_MED_ZSCORE,
)
from src.monitor.state import CandidateAlert

logger = logging.getLogger(__name__)

# Ordered weakest to strongest. Severity is compared through this map and
# never as a string: "MED" > "HIGH" is True lexically, which would silently
# downgrade every genuinely alarming price move.
_RANK: dict[str, int] = {"LOW": 0, "MED": 1, "HIGH": 2}


# ── Per-type rules ──────────────────────────────────────
def _price_severity(metrics: dict) -> tuple[Severity, str]:
    """
    Score a price move on BOTH magnitude and unusualness.

    Percentage alone is the wrong measure on its own: a 5% day in NVDA is a
    Tuesday and a 5% day in JPM is a headline. ``vol_zscore`` measures the move
    against the ticker's own 60-day realised volatility, so a small move in a
    quiet name can still escalate. Whichever route reads worse wins.
    """
    change = abs(float(metrics.get("change_pct_1d") or 0.0))
    raw_z = metrics.get("vol_zscore")
    zscore = abs(float(raw_z)) if raw_z is not None else 0.0

    by_pct = "HIGH" if change >= PRICE_HIGH_PCT else "MED" if change >= PRICE_MED_PCT else "LOW"
    by_z = "HIGH" if zscore >= PRICE_HIGH_ZSCORE else "MED" if zscore >= PRICE_MED_ZSCORE else "LOW"

    # Compare by RANK, not lexically — "MED" > "HIGH" is true as a string.
    if _RANK[by_z] > _RANK[by_pct]:
        return by_z, f"{zscore:.1f}-sigma move against 60-day realised volatility"  # type: ignore[return-value]
    return by_pct, f"{change:.1f}% single-day move"  # type: ignore[return-value]


def _filing_severity(metrics: dict) -> tuple[Severity, str]:
    """
    Score a filing on its FORM and, for an 8-K, its ITEM CODES.

    The item codes are what carry the information. An 8-K is just "something
    happened that shareholders must be told about promptly"; Item 4.02 is
    "the auditor says our previously issued financials cannot be relied upon",
    which is one of the most serious things a public company ever files.
    """
    form = str(metrics.get("form_type") or "").upper()
    items = {str(item).strip() for item in metrics.get("items") or []}

    high = sorted(items & FILING_HIGH_8K_ITEMS)
    if high:
        return "HIGH", f"8-K item {', '.join(high)}"

    med = sorted(items & FILING_MED_8K_ITEMS)
    if med:
        return "MED", f"8-K item {', '.join(med)}"

    if form in FILING_PERIODIC_FORMS:
        # Material by definition, but scheduled — its arrival is never news.
        return "MED", f"periodic report ({form})"

    return "LOW", f"{form or 'filing'} with no escalating item code"


def _news_severity(metrics: dict) -> tuple[Severity, str]:
    """
    Score a news item on sentiment AND corroboration.

    HIGH requires more than one independent outlet. Without that clause a
    single mis-scored headline pages you at 3am, and provider sentiment is
    exactly the kind of number that is wrong often enough to matter.
    """
    raw = metrics.get("sentiment")
    sentiment = float(raw) if raw is not None else 0.0
    sources = int(metrics.get("source_count") or 1)

    if sentiment <= NEWS_HIGH_SENTIMENT and sources >= NEWS_HIGH_MIN_SOURCES:
        return "HIGH", f"sentiment {sentiment:+.2f} corroborated by {sources} outlets"

    if sentiment <= NEWS_HIGH_SENTIMENT:
        # Strongly negative but uncorroborated — capped at MED on purpose.
        return "MED", f"sentiment {sentiment:+.2f} from a single outlet"

    if sentiment <= NEWS_MED_SENTIMENT:
        return "MED", f"sentiment {sentiment:+.2f}"

    return "LOW", f"sentiment {sentiment:+.2f}"


def _macro_severity(metrics: dict) -> tuple[Severity, str]:
    """
    Score a macro release against a PER-SERIES threshold.

    A single threshold cannot serve this set: DFF, UNRATE, DGS10 and T10Y2Y are
    already expressed in percent, so their meaningful moves are fractions of a
    unit, while CPIAUCSL is an index level whose meaningful moves are fractions
    of a percent. Each series declares which measure applies — see
    MACRO_THRESHOLDS.

    A level CROSSING escalates on its own. The move that takes the 10Y-2Y
    spread from +0.02 to -0.01 is trivially small and historically the most
    watched signal on the list.
    """
    series_id = str(metrics.get("series_id") or "")

    if metrics.get("crossing"):
        return "HIGH", f"{series_id} crossed {metrics['crossing']}"

    measure, med, high = MACRO_THRESHOLDS.get(series_id, ("pct", 1.0, 2.0))
    key = "abs_change" if measure == "abs" else "pct_change"
    magnitude = abs(float(metrics.get(key) or 0.0))
    unit = "" if measure == "abs" else "%"

    if magnitude >= high:
        return "HIGH", f"{series_id} moved {magnitude:.2f}{unit} ({measure} threshold {high})"
    if magnitude >= med:
        return "MED", f"{series_id} moved {magnitude:.2f}{unit} ({measure} threshold {med})"
    return "LOW", f"{series_id} moved {magnitude:.2f}{unit}"


_RULES = {
    "PRICE_MOVE": _price_severity,
    "NEW_FILING": _filing_severity,
    "NEWS_SENTIMENT": _news_severity,
    "MACRO_EVENT": _macro_severity,
}


# ── Public API ──────────────────────────────────────────
def score(candidate: CandidateAlert) -> tuple[Severity, str]:
    """
    Score a candidate and say which rule decided it.

    Parameters
    ----------
    candidate : CandidateAlert
        Its ``metrics`` carry everything the rules read. Monitors are
        responsible for populating them; a missing key scores as absent rather
        than raising, so a partially-degraded data source produces a quieter
        alert instead of no cycle.

    Returns
    -------
    tuple
        ``(severity, reason)``. The reason goes into the audit trail — a
        severity nobody can explain is one nobody can trust or tune.
    """
    rule = _RULES.get(candidate["alert_type"])
    if rule is None:
        logger.warning("No severity rule for alert_type %r — defaulting to LOW", candidate["alert_type"])
        return "LOW", "no rule for this alert type"

    return rule(candidate.get("metrics") or {})


def severity_for(candidate: CandidateAlert) -> Severity:
    """Severity only, for callers that do not need the explanation."""
    return score(candidate)[0]


def explain(candidate: CandidateAlert) -> str:
    """The reason string only."""
    return score(candidate)[1]
