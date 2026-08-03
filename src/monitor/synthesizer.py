# ═══════════════════════════════════════════════════════
# FinSight — Alert Canonicalization
# ═══════════════════════════════════════════════════════
#
# Purpose : Turn a candidate into the text that gets EMBEDDED, plus the exact
#           key that gets hashed. Everything the dedup engine compares is
#           produced here.
#
# Public API:
#   dedup_key(candidate)          sha256 of (ticker, type, natural_key)
#   alert_id_for(dedup_key)       deterministic uuid5 — also the Qdrant point id
#   contains_volatile(text)       does this text carry a magnitude or a date?
#   strip_volatile(text)          remove them
#   template_summary(candidate)   the deterministic fallback
#   summarize(candidates)         the batched LLM call
#   canonical_text(...)           the final embedded string
#   build_alert(...)              assemble an Alert
#
# ══ WHY THE DISPLAY TEXT IS THE WRONG THING TO EMBED ══
#   Consider two pairs.
#
#     "AAPL fell 5.2%"  vs  "AAPL fell 5.4%"     SAME event, different strings
#     "AAPL fell 5.2%"  vs  "MSFT fell 5.2%"     DIFFERENT events, near-identical
#
#   Cosine similarity over the display text gets both backwards. The magnitude
#   is the most volatile token in the sentence and contributes most of the
#   lexical difference; the ticker is the most discriminating token and
#   contributes almost none.
#
#   So the fix is structural, not a matter of tuning a threshold:
#
#     * the embedded text is QUALITATIVE — what happened, not how much
#     * ticker and alert_type are HARD PAYLOAD FILTERS, not soft signals
#
#   After that split, the only thing similarity has to judge is whether two
#   descriptions of the same company's same kind of event describe the same
#   event. That is a question embeddings are actually good at.
#
# ══ AND WHY THE LLM'S OUTPUT IS VERIFIED, NOT TRUSTED ══
#   The prompt forbids numbers. Models comply most of the time. One leaked
#   "5.2%" silently reintroduces exactly the failure this module exists to
#   prevent, and it would show up as a mysteriously low similarity score
#   months later.
#
#   So the output is checked with a regex, and a summary that carries a
#   magnitude is DISCARDED rather than repaired — a model that ignored the one
#   rule that matters has not earned partial credit, and the deterministic
#   template is always available and always compliant.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Literal

from src.core.schemas import Severity
from src.monitor.config import (
    CANONICAL_SUMMARY_PROMPT_SYSTEM,
    CANONICAL_SUMMARY_PROMPT_USER,
    PRICE_NOTABLE_VOLUME_RATIO,
)
from src.monitor.monitors.filings import ITEM_DESCRIPTIONS
from src.monitor.state import Alert, CandidateAlert

logger = logging.getLogger(__name__)

# Model tier for the summary. It is a one-line rewrite under a rigid rule, on
# the critical path of every candidate that survives the exact-key check —
# exactly the shape `flash` is for.
SUMMARY_MODEL_TIER: Literal["flash", "pro"] = "flash"

# Namespace for deterministic alert ids. Fixed forever: changing it would give
# every existing alert a new id and silently orphan the entire dedup index.
ALERT_NAMESPACE: uuid.UUID = uuid.NAMESPACE_URL

MAX_SUMMARY_WORDS: int = 15


# ══ VOLATILE-NUMERIC DETECTION ══
#
# What counts as volatile is narrower than "contains a digit", and getting the
# boundary right matters in both directions:
#
#   VOLATILE, must go     5.2%   $210.11   391,035   2026-08-03   Q3   FY2026
#   STABLE, must stay     8-K    10-Q      20-day moving average   RSI
#
# A form type is not a magnitude — every 8-K is an 8-K — and stripping it would
# make "8-K reporting an auditor change" and "10-Q filed" collapse toward each
# other. So the pattern targets the shapes magnitudes and dates actually take,
# not digits in general.
# "may" is deliberately absent from the always-stripped list and present in the
# adjacent-to-a-day list instead. It is an ordinary English modal — "may face a
# regulatory review" is not a date, and flagging it would send a perfectly good
# LLM summary back to the template for no reason.
_MONTHS = "january|february|march|april|june|july|august|september|october|november|december"
_MONTH_ABBR = "jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec"
_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"

# Order matters. Alternatives are tried left to right at each position, so the
# percentage rule must precede the bare-decimal rule — otherwise "5.2%" has its
# "5.2" eaten by the decimal branch and leaves a stranded "%" behind.
_VOLATILE = re.compile(
    r"""
      \$\s?\d[\d,]*(?:\.\d+)?                  # currency amount
    | \d[\d,]*(?:\.\d+)?\s?%                   # percentage — BEFORE the decimal rule
    | \d[\d,]*\.\d                             # any decimal — 210.11, 4.02
    | \b\d{1,3}(?:,\d{3})+\b                   # comma-grouped thousands
    | \bFY\s?\d{2,4}\b                         # FY2026 — no word boundary inside it
    | \b(?:19|20)\d{2}\b                       # a bare year
    | \bQ[1-4]\b                               # fiscal quarter
    | \bFY\b                                   # bare fiscal-year marker
    | \b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b        # a written date
    | \b(?:"""
    + _MONTHS
    + r""")\b
    # Abbreviations only when adjacent to a day number: "Aug 3" is a date,
    # "may" is an ordinary word, and "Mar" appears inside plenty of names.
    | \b(?:"""
    + _MONTH_ABBR
    + r""")\.?\s+\d{1,2}\b
    | \b\d{1,2}\s+(?:"""
    + _MONTH_ABBR
    + r""")\.?\b
    | \b(?:"""
    + _WEEKDAYS
    + r""")\b
    | \b\d{1,2}(?:st|nd|rd|th)\b               # ordinal day
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Applied only after the volatile shapes are gone, to catch bare magnitudes
# like "fell 12 points". Runs second so it cannot eat the digits inside "8-K".
_BARE_NUMBER = re.compile(r"(?<![\w-])\d+(?![\w-])")


def contains_volatile(text: str) -> bool:
    """
    Report whether text carries a magnitude, a date, or a fiscal period.

    Used to audit the LLM's compliance with the no-numbers rule. Deliberately
    does NOT flag form types (``8-K``) or indicator windows (``20-day``), which
    are stable identifiers rather than volatile measurements.
    """
    return bool(_VOLATILE.search(text))


def strip_volatile(text: str) -> str:
    """
    Remove magnitudes, dates, and fiscal periods, leaving the prose.

    Parameters
    ----------
    text : str
        Free text, typically a news headline.

    Returns
    -------
    str
        The same text with volatile tokens removed and whitespace tidied.
        Punctuation left stranded by a removal is cleaned up, because
        ``"fell , to"`` embeds differently from ``"fell to"`` for no reason.
    """
    cleaned = _VOLATILE.sub(" ", text)
    cleaned = _BARE_NUMBER.sub(" ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])\s*([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,;:-—–")


# ── Keys ────────────────────────────────────────────────
def dedup_key(candidate: CandidateAlert) -> str:
    """
    The exact identity of a candidate's underlying event.

    Hashed rather than concatenated so the key is a fixed width whatever a
    natural key looks like — a URL, an accession number, or a composite date
    string — and so it is safe as both a payload value and a point id seed.
    """
    raw = f"{candidate['ticker'].upper()}|{candidate['alert_type']}|{candidate['natural_key']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def alert_id_for(key: str) -> str:
    """
    Deterministic alert id, derived from the dedup key.

    ══ WHY THIS DOUBLES AS THE QDRANT POINT ID ══
    Qdrant point ids must be a UUID or an unsigned integer, and deriving the
    UUID from the dedup key makes the exact-match path a direct ``retrieve()``
    by id — O(1), no filter, no payload index consulted at all — instead of a
    filtered scroll over the collection.

    It also makes re-upsert idempotent: the same event re-observed writes to
    the same point rather than accumulating near-duplicate vectors that would
    then compete with each other in search results.
    """
    return str(uuid.uuid5(ALERT_NAMESPACE, key))


# ── The deterministic fallback ──────────────────────────
def template_summary(candidate: CandidateAlert) -> str:
    """
    Build a canonical summary from STRUCTURED FIELDS, with no model involved.

    Built from fields rather than by stripping the headline, wherever the
    fields exist. Stripping is lossy in the one direction that hurts: an 8-K's
    item code is a number, and removing it would make "non-reliance on
    previously issued financials" and "results of operations" collapse into the
    same generic "8-K filed" — a false suppression of the single most serious
    filing type on the list. The item code is therefore translated into WORDS
    (see ITEM_DESCRIPTIONS) rather than deleted.

    News is the exception. There is no structured description of a story, so
    its headline is stripped instead.

    Parameters
    ----------
    candidate : CandidateAlert

    Returns
    -------
    str
        Lower-case, numeric-free, at most MAX_SUMMARY_WORDS words.
    """
    alert_type = candidate["alert_type"]
    metrics = candidate.get("metrics") or {}

    if alert_type == "NEW_FILING":
        form = str(metrics.get("form_type") or "filing")
        described = [ITEM_DESCRIPTIONS[code] for code in metrics.get("items") or [] if code in ITEM_DESCRIPTIONS]
        summary = f"{form} reporting {'; '.join(described)}" if described else f"new {form} filed"

    elif alert_type == "PRICE_MOVE":
        change = float(metrics.get("change_pct_1d") or 0.0)
        direction = "decline" if change < 0 else "advance"
        ratio = metrics.get("volume_ratio")
        volume = " on elevated volume" if ratio and ratio >= PRICE_NOTABLE_VOLUME_RATIO else ""
        summary = f"sharp single-day {direction}{volume}"

    elif alert_type == "MACRO_EVENT":
        series = str(metrics.get("series_id") or "series")
        if metrics.get("crossing"):
            summary = f"{series} crossed a watched threshold"
        else:
            direction = "increase" if float(metrics.get("abs_change") or 0.0) > 0 else "decrease"
            summary = f"{series} release showing a material {direction}"

    else:  # NEWS_SENTIMENT — no structured description exists for a story.
        summary = strip_volatile(candidate["headline"])
        # The ticker and company name are attached separately and filtered
        # exactly, so leaving them in the embedded text only adds a term that
        # is identical across every candidate it could ever be compared to.
        for token in (candidate["ticker"], candidate.get("company_name", "")):
            if token:
                summary = re.sub(rf"\b{re.escape(token)}\b", " ", summary, flags=re.IGNORECASE)

    words = re.sub(r"\s{2,}", " ", summary).strip(" ,;:-").lower().split()
    return " ".join(words[:MAX_SUMMARY_WORDS])


# ── The LLM path ────────────────────────────────────────
def _render_events(candidates: list[CandidateAlert]) -> str:
    """Render candidates for the batched summary prompt."""
    lines = []
    for index, item in enumerate(candidates, start=1):
        lines.append(f"{index}. [{item['alert_type']}] {item['headline']}\n   {item['detail']}")
    return "\n\n".join(lines)


def summarize(
    candidates: list[CandidateAlert],
    *,
    _mock_response: list[str] | None = None,
) -> list[str]:
    """
    Write one canonical summary per candidate, in a single batched call.

    Batched rather than one call per candidate: a busy cycle can produce
    fifteen survivors, and fifteen sequential `flash` calls is fifteen times
    the latency for a rewrite that fits comfortably in one prompt.

    EVERY returned summary is audited. A summary that is missing, empty, or
    carries a volatile number falls back to ``template_summary`` for that
    candidate alone — the batch is not discarded wholesale, because one bad
    line says nothing about the other fourteen.

    Parameters
    ----------
    candidates : list of CandidateAlert
        Survivors of the exact-key check. Empty is valid and costs no call.
    _mock_response : list of str, optional
        Bypass the LLM in tests.

    Returns
    -------
    list of str
        Exactly ``len(candidates)`` summaries, aligned by position.
    """
    if not candidates:
        return []

    if _mock_response is not None:
        raw = list(_mock_response)
    else:
        raw = _invoke_summary_model(candidates)

    summaries: list[str] = []
    for index, item in enumerate(candidates):
        text = (raw[index] if index < len(raw) else "").strip()

        if not text:
            reason = "model returned nothing for this position"
        elif contains_volatile(text):
            # The one rule that matters was broken. Do not repair it — a
            # partially-stripped summary is a summary written under a rule the
            # model was not following.
            reason = f"model leaked a volatile numeric: {text!r}"
        else:
            summaries.append(text.lower().strip(" .").strip())
            continue

        logger.warning("Canonical summary fell back to template (%s)", reason)
        summaries.append(template_summary(item))

    return summaries


def _invoke_summary_model(candidates: list[CandidateAlert]) -> list[str]:
    """
    Call the model, degrading to templates rather than failing the cycle.

    A dedup engine that stops working when the LLM is unavailable is worse
    than one that deduplicates slightly less well: the fallback direction is a
    duplicate ping, which is mild, while the alternative is a cycle that
    reports nothing.
    """
    from pydantic import BaseModel, Field

    from src.core.llm import get_llm
    from src.core.tracing import trace_metadata

    class Summaries(BaseModel):
        """Flat and fully required — nested or optional fields fight Gemini's schema."""

        summaries: list[str] = Field(description="One canonical description per event, in order")

    user = CANONICAL_SUMMARY_PROMPT_USER.format(events=_render_events(candidates), count=len(candidates))

    try:
        model = get_llm(SUMMARY_MODEL_TIER, temperature=0.0).with_structured_output(Summaries)
        result = model.invoke(
            [("system", CANONICAL_SUMMARY_PROMPT_SYSTEM), ("human", user)],
            config={"metadata": trace_metadata(phase="P6"), "tags": ["subsystem2", "canonicalize"]},
        )
        return [str(s) for s in getattr(result, "summaries", [])]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Canonical summary model unavailable (%s) — using templates for all %d", exc, len(candidates))
        return []


# ── Assembly ────────────────────────────────────────────
def canonical_text(candidate: CandidateAlert, summary: str) -> str:
    """
    Assemble the string that actually gets embedded.

    ``{TICKER} {Company} | {ALERT_TYPE} | {qualitative summary}``

    The ticker and type appear here even though both are ALSO hard payload
    filters, and the redundancy is deliberate: it keeps the text readable in
    the decision log, where a human re-labelling pairs for the Phase 7 sweep
    needs to see what was compared. The filters do the discriminating; these
    two fields cost a couple of tokens and make the log legible.
    """
    company = candidate.get("company_name") or candidate["ticker"]
    scope = f"{candidate['ticker']} {company}".strip() or "MACRO"
    return f"{scope} | {candidate['alert_type']} | {summary}"


def build_alert(
    candidate: CandidateAlert,
    *,
    severity: Severity,
    summary: str,
    status: str = "FIRED",
    now: str = "",
    parent_alert_id: str | None = None,
) -> Alert:
    """
    Assemble a fully-formed Alert from a scored, canonicalized candidate.

    Parameters
    ----------
    candidate : CandidateAlert
    severity : Severity
        From src.monitor.severity — rules, never a model.
    summary : str
        The canonical qualitative summary.
    status : str, default "FIRED"
    now : str, optional
        ISO timestamp; defaults to now. Injected for tests.
    parent_alert_id : str, optional
        Set when this alert escalates an existing one.

    Returns
    -------
    Alert
    """
    key = dedup_key(candidate)
    stamp = now or datetime.now(timezone.utc).isoformat()

    return Alert(
        alert_id=alert_id_for(key),
        ticker=candidate["ticker"],
        company_name=candidate.get("company_name") or candidate["ticker"],
        alert_type=candidate["alert_type"],
        severity=severity,
        status=status,
        headline=candidate["headline"],
        detail=candidate["detail"],
        canonical_text=canonical_text(candidate, summary),
        dedup_key=key,
        metrics=candidate.get("metrics") or {},
        evidence=candidate.get("evidence") or [],
        occurrence_count=1,
        first_seen_at=stamp,
        last_seen_at=stamp,
        fired_at=stamp,
        parent_alert_id=parent_alert_id,
    )
