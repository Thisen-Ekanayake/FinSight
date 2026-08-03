# ═══════════════════════════════════════════════════════
# FinSight — Alert Deduplication
# ═══════════════════════════════════════════════════════
#
# Purpose : Decide, for each candidate, whether it is news or a repeat.
#
# Public API:
#   Decision                      the outcome enum-ish
#   DedupOutcome                  what decide() returns
#   decide(candidate, ...)        one candidate -> one decision
#   deduplicate(candidates, ...)  a whole cycle, in order
#
# ══ THE SHAPE OF THE PROBLEM ══
#   An alerting system's failure modes are wildly asymmetric.
#
#     A false FIRE     you get told twice. Mildly annoying.
#     A false SUPPRESS you are never told. The system silently did not work.
#
#   Every threshold and every guardrail below leans the same way because of
#   that. Suppression precision is optimised, NOT F1 — F1 treats those two
#   errors as equally bad, and they are not remotely equally bad.
#
# ══ THE ALGORITHM ══
#   1. EXACT KEY      free. ~90% of duplicates. No embedding, no LLM.
#   2. SEVERITY       computed BEFORE dedup, because step 5 needs it.
#   3. EMBED          symmetric — alert vs alert, so no bge query prefix.
#   4. FILTERED SEARCH ticker + type + status + time window, all indexed.
#   5. DECIDE
#        s >= TAU_HIGH             SUPPRESS  — same event, nothing new
#        TAU_LOW <= s < TAU_HIGH   MERGE     — same event, new information
#        s <  TAU_LOW  / no hits   FIRE      — new event
#   6. GUARDRAIL      a HIGH-severity candidate below TAU_HIGH_SEVERITY_FORCE_FIRE
#                     fires regardless of what step 5 concluded.
#
# ══ WHY THE MERGE BAND EXISTS AT ALL ══
#   Two thresholds rather than one, because "is this the same event?" has three
#   answers, not two. A second outlet covering the same lawsuit is neither a
#   duplicate to discard nor a new event to report — it is corroboration, and
#   it can carry information the first report did not (a named regulator, a
#   larger scope). The merge band captures that: the parent absorbs the new
#   evidence, and only ESCALATES into a fresh alert if the newcomer scores
#   higher severity than the parent did.
#
# ══ WHY THRESHOLDS ARE NOT THE TUTORIAL'S 0.7 ══
#   Two UNRELATED financial sentences score 0.65-0.78 with bge-small. And the
#   negatives that matter here are not random sentences — the payload filter
#   has already constrained candidates to the same ticker AND the same alert
#   type, so the hard case is two genuinely different events sharing both, and
#   those still score ~0.73. A 0.7 threshold would suppress essentially
#   everything. See src/vectorstore/config.py for the measured bands.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timezone
from typing import Any, NamedTuple

from src.core.schemas import Severity
from src.monitor.alert_store import bump_occurrence, find_exact, search_similar, update_centroid, upsert_alert
from src.monitor.config import WARMUP_STATUS
from src.monitor.severity import score as severity_score
from src.monitor.state import Alert, CandidateAlert, DecisionRecord, SuppressionRecord
from src.monitor.synthesizer import alert_id_for, build_alert, dedup_key, summarize
from src.vectorstore.config import TAU_HIGH, TAU_HIGH_SEVERITY_FORCE_FIRE, TAU_LOW

logger = logging.getLogger(__name__)

_RANK: dict[str, int] = {"LOW": 0, "MED": 1, "HIGH": 2}


class Decision:
    """The outcomes decide() can reach. Strings, so they land in SQLite as-is."""

    FIRE = "FIRE"
    SUPPRESS_EXACT = "SUPPRESS_EXACT"
    SUPPRESS_SEMANTIC = "SUPPRESS_SEMANTIC"
    MERGE = "MERGE"
    ESCALATE = "ESCALATE"
    WARMUP = "WARMUP"


class DedupOutcome(NamedTuple):
    """
    What happened to one candidate.

    ``alert`` is set for anything that produced or updated a reportable alert;
    ``suppression`` for anything that did not. Exactly one is populated, which
    is what lets the caller sort outcomes without re-deriving the decision.
    """

    decision: str
    reason: str
    score: float
    alert: Alert | None
    suppression: SuppressionRecord | None
    record: DecisionRecord


def _decision_record(
    candidate: CandidateAlert,
    *,
    decision: str,
    reason: str,
    severity: str,
    key: str,
    canonical: str,
    score: float = 0.0,
    parent: dict[str, Any] | None = None,
) -> DecisionRecord:
    """Build the audit row. Written for EVERY candidate, including the fires."""
    return DecisionRecord(
        ticker=candidate["ticker"],
        alert_type=candidate["alert_type"],
        severity=severity,
        decision=decision,
        reason=reason,
        dedup_key=key,
        candidate_text=canonical,
        parent_alert_id=(parent or {}).get("alert_id"),
        parent_text=(parent or {}).get("canonical_text", ""),
        score=score,
    )


def decide(
    candidate: CandidateAlert,
    *,
    summary: str,
    warmup: bool = False,
    now: datetime | None = None,
) -> DedupOutcome:
    """
    Run one candidate through the whole algorithm.

    Parameters
    ----------
    candidate : CandidateAlert
        From a monitor.
    summary : str
        Its canonical qualitative summary — see src/monitor/synthesizer.py.
    warmup : bool, default False
        Observe-only. The point is still written so the NEXT cycle can dedup
        against it, but nothing is reported.
    now : datetime, optional
        Injected for tests, and threaded into the time-window filter.

    Returns
    -------
    DedupOutcome
    """
    from src.vectorstore.embeddings import get_embedder

    moment = now or datetime.now(timezone.utc)
    key = dedup_key(candidate)
    alert_id = alert_id_for(key)

    # ── 1. Exact-key fast path — free ───────────────────
    # Before embedding, before the LLM, before any search. The same accession
    # number, the same article URL, the same day's price move in the same band.
    existing = find_exact(key, alert_id=alert_id)
    if existing is not None:
        count = int(existing.get("occurrence_count", 1)) + 1
        stamp = moment.isoformat()
        bump_occurrence(alert_id, count=count, last_seen_at=stamp)

        canonical = existing.get("canonical_text", "")
        reason = f"exact key match on {candidate['alert_type']} natural key"
        logger.info("SUPPRESS_EXACT %s %s (seen %dx)", candidate["ticker"], candidate["alert_type"], count)

        return DedupOutcome(
            decision=Decision.SUPPRESS_EXACT,
            reason=reason,
            score=1.0,
            alert=None,
            suppression=SuppressionRecord(
                ticker=candidate["ticker"],
                alert_type=candidate["alert_type"],
                headline=candidate["headline"],
                canonical_text=canonical,
                parent_alert_id=alert_id,
                parent_headline=existing.get("headline", ""),
                score=1.0,
                reason=reason,
            ),
            record=_decision_record(
                candidate,
                decision=Decision.SUPPRESS_EXACT,
                reason=reason,
                severity=existing.get("severity", "LOW"),
                key=key,
                canonical=canonical,
                score=1.0,
                parent=existing,
            ),
        )

    # ── 2. Severity BEFORE dedup — step 6 depends on it ──
    severity, severity_reason = severity_score(candidate)

    # ── 3. Embed — symmetric, no bge query prefix ───────
    # This compares an alert to other alerts. Both sides are the same kind of
    # text, so the asymmetric query instruction would only add noise.
    alert = build_alert(
        candidate,
        severity=severity,
        summary=summary,
        status=WARMUP_STATUS if warmup else "FIRED",
        now=moment.isoformat(),
    )
    vector = get_embedder().embed_symmetric(alert["canonical_text"])

    # ── 4. Filtered vector search ───────────────────────
    neighbours = search_similar(
        vector,
        ticker=candidate["ticker"],
        alert_type=candidate["alert_type"],
        now=moment,
        score_threshold=TAU_LOW,
    )
    best_score, best = neighbours[0] if neighbours else (0.0, {})

    # ── 6. Asymmetric-cost guardrail — checked BEFORE 5 ──
    # Placed ahead of the suppression branches deliberately: a HIGH-severity
    # event has to escape them, not be rescued afterwards. Missing a real 8-K
    # Item 4.02 because it read like last week's costs far more than one
    # duplicate ping.
    #
    # Only meaningful when there IS a neighbour — search_similar already floors
    # its results at TAU_LOW, so an empty result set would reach FIRE anyway and
    # labelling that a guardrail save would overstate what the rule did.
    if neighbours and severity == "HIGH" and best_score < TAU_HIGH_SEVERITY_FORCE_FIRE:
        reason = (
            f"HIGH severity ({severity_reason}) at {best_score:.3f}, below the "
            f"{TAU_HIGH_SEVERITY_FORCE_FIRE} force-fire floor"
        )
        logger.info("FIRE (guardrail) %s %s @ %.3f", candidate["ticker"], candidate["alert_type"], best_score)
        return _fire(candidate, alert, vector, key, warmup, best_score, reason, severity)

    # ── 5. Decide on the best score ─────────────────────
    if best_score >= TAU_HIGH:
        count = int(best.get("occurrence_count", 1)) + 1
        bump_occurrence(best["alert_id"], count=count, last_seen_at=moment.isoformat())
        reason = f"semantic duplicate at {best_score:.3f} (>= TAU_HIGH {TAU_HIGH})"
        logger.info(
            "SUPPRESS_SEMANTIC %s %s @ %.3f -> %s",
            candidate["ticker"],
            candidate["alert_type"],
            best_score,
            best["alert_id"],
        )
        return DedupOutcome(
            decision=Decision.SUPPRESS_SEMANTIC,
            reason=reason,
            score=best_score,
            alert=None,
            suppression=SuppressionRecord(
                ticker=candidate["ticker"],
                alert_type=candidate["alert_type"],
                headline=candidate["headline"],
                canonical_text=alert["canonical_text"],
                parent_alert_id=best["alert_id"],
                parent_headline=best.get("headline", ""),
                score=best_score,
                reason=reason,
            ),
            record=_decision_record(
                candidate,
                decision=Decision.SUPPRESS_SEMANTIC,
                reason=reason,
                severity=severity,
                key=key,
                canonical=alert["canonical_text"],
                score=best_score,
                parent=best,
            ),
        )

    if best_score >= TAU_LOW:
        return _merge(candidate, alert, vector, key, best, best_score, severity, severity_reason, moment)

    # ── No neighbour: a new event ───────────────────────
    reason = "no prior alert within the dedup window" if not neighbours else f"nearest was {best_score:.3f}"
    return _fire(candidate, alert, vector, key, warmup, best_score, reason, severity)


def _fire(
    candidate: CandidateAlert,
    alert: Alert,
    vector: list[float],
    key: str,
    warmup: bool,
    score: float,
    reason: str,
    severity: Severity,
) -> DedupOutcome:
    """Write the point and report the alert — unless this is a warmup cycle."""
    upsert_alert(alert, vector)

    decision = Decision.WARMUP if warmup else Decision.FIRE
    if warmup:
        reason = f"warmup cycle — indexed but not reported ({reason})"

    return DedupOutcome(
        decision=decision,
        reason=reason,
        score=score,
        alert=None if warmup else alert,
        suppression=None,
        record=_decision_record(
            candidate,
            decision=decision,
            reason=reason,
            severity=severity,
            key=key,
            canonical=alert["canonical_text"],
            score=score,
        ),
    )


def _merge(
    candidate: CandidateAlert,
    alert: Alert,
    vector: list[float],
    key: str,
    parent: dict[str, Any],
    score: float,
    severity: Severity,
    severity_reason: str,
    moment: datetime,
) -> DedupOutcome:
    """
    Same event, new information.

    The parent absorbs it either way: its occurrence count rises and its vector
    drifts toward the newcomer. What differs is whether anything is REPORTED —
    and that turns on severity alone. A second outlet saying the same thing is
    corroboration; a second outlet revealing it is a criminal probe rather than
    a civil one is an escalation, and escalations must reach the reader.
    """
    parent_id = parent["alert_id"]
    parent_severity = str(parent.get("severity", "LOW"))

    count = int(parent.get("occurrence_count", 1)) + 1
    bump_occurrence(parent_id, count=count, last_seen_at=moment.isoformat())
    update_centroid(parent_id, vector)

    escalating = _RANK.get(severity, 0) > _RANK.get(parent_severity, 0)

    if escalating:
        reason = f"merged at {score:.3f} and escalated {parent_severity} -> {severity} ({severity_reason})"
        logger.info(
            "ESCALATE %s %s @ %.3f (%s -> %s)",
            candidate["ticker"],
            alert["alert_type"],
            score,
            parent_severity,
            severity,
        )

        escalation = dict(alert)
        escalation["parent_alert_id"] = parent_id
        escalation["occurrence_count"] = count

        return DedupOutcome(
            decision=Decision.ESCALATE,
            reason=reason,
            score=score,
            alert=escalation,  # type: ignore[arg-type]
            suppression=None,
            record=_decision_record(
                candidate,
                decision=Decision.ESCALATE,
                reason=reason,
                severity=severity,
                key=key,
                canonical=alert["canonical_text"],
                score=score,
                parent=parent,
            ),
        )

    reason = f"merged into {parent_id} at {score:.3f} — same event, no severity change"
    logger.info("MERGE %s %s @ %.3f -> %s", candidate["ticker"], alert["alert_type"], score, parent_id)

    return DedupOutcome(
        decision=Decision.MERGE,
        reason=reason,
        score=score,
        alert=None,
        suppression=SuppressionRecord(
            ticker=candidate["ticker"],
            alert_type=candidate["alert_type"],
            headline=candidate["headline"],
            canonical_text=alert["canonical_text"],
            parent_alert_id=parent_id,
            parent_headline=parent.get("headline", ""),
            score=score,
            reason=reason,
        ),
        record=_decision_record(
            candidate,
            decision=Decision.MERGE,
            reason=reason,
            severity=severity,
            key=key,
            canonical=alert["canonical_text"],
            score=score,
            parent=parent,
        ),
    )


def deduplicate(
    candidates: list[CandidateAlert],
    *,
    warmup: bool = False,
    now: datetime | None = None,
    summaries: list[str] | None = None,
    log_distribution: bool = False,
) -> list[DedupOutcome]:
    """
    Run a whole cycle's candidates through the engine, in order.

    ══ ORDER MATTERS, AND SEQUENTIAL IS THE POINT ══
    Each decision writes to the same index the next decision searches. Three
    outlets covering one story in a single cycle must collapse into one alert
    plus two suppressions, and that only works if the first one is INDEXED
    before the second one is searched. Parallelising this would race them all
    against an empty index and fire three times.

    Parameters
    ----------
    candidates : list of CandidateAlert
        In whatever order the monitors produced them.
    warmup : bool, default False
        Index everything, report nothing.
    now : datetime, optional
    summaries : list of str, optional
        Pre-computed canonical summaries, aligned by position. Computed here
        when omitted.
    log_distribution : bool, default False
        Log the full score distribution at INFO. Used for the first few cycles:
        if p90 sits above TAU_LOW, canonicalization has collapsed and every
        alert is reading as similar to every other.

    Returns
    -------
    list of DedupOutcome
        One per candidate, in input order.
    """
    if not candidates:
        return []

    # Summaries are computed for the WHOLE batch in one call, before the loop.
    # Candidates that turn out to be exact duplicates will have theirs go
    # unused — which is the cheap direction: one extra line in one batched
    # `flash` call, versus serialising the loop behind N separate calls.
    texts = summaries if summaries is not None else summarize(candidates)

    outcomes: list[DedupOutcome] = []
    for index, item in enumerate(candidates):
        summary = texts[index] if index < len(texts) else ""
        outcomes.append(decide(item, summary=summary, warmup=warmup, now=now))

    if log_distribution:
        _log_distribution(outcomes)

    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.decision] = counts.get(outcome.decision, 0) + 1
    logger.info("Dedup: %d candidates -> %s", len(candidates), counts)

    return outcomes


def _log_distribution(outcomes: list[DedupOutcome]) -> None:
    """
    Report the cycle's similarity scores, for cold-start diagnosis.

    Exact-key hits are excluded: they score a synthetic 1.0 that says nothing
    about the embedding, and leaving them in would drag every percentile toward
    1.0 and hide exactly the problem this is meant to expose.
    """
    scores = sorted(o.score for o in outcomes if o.decision != Decision.SUPPRESS_EXACT and o.score > 0)
    if not scores:
        logger.info("Dedup score distribution: no semantic comparisons this cycle")
        return

    p50 = statistics.median(scores)
    p90 = scores[int(len(scores) * 0.9)] if len(scores) > 1 else scores[0]

    logger.info(
        "Dedup score distribution: n=%d p50=%.3f p90=%.3f max=%.3f (TAU_LOW=%.2f TAU_HIGH=%.2f)",
        len(scores),
        p50,
        p90,
        scores[-1],
        TAU_LOW,
        TAU_HIGH,
    )
    if p90 > TAU_LOW:
        logger.warning(
            "p90 similarity %.3f exceeds TAU_LOW %.2f — canonical summaries may be too generic to "
            "tell distinct events apart. Check that the qualitative summaries name the KIND of event.",
            p90,
            TAU_LOW,
        )
