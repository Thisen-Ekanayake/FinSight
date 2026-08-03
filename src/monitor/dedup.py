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
#   6. GUARDRAIL      a HIGH candidate whose matched parent is NOT already a
#                     reported HIGH fires regardless of what step 5 concluded.
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
from src.monitor.config import PENDING_APPROVAL_STATUS, WARMUP_STATUS
from src.monitor.severity import score as severity_score
from src.monitor.state import Alert, CandidateAlert, DecisionRecord, SuppressionRecord
from src.monitor.synthesizer import alert_id_for, build_alert, dedup_key, summarize
from src.vectorstore.config import TAU_HIGH, TAU_LOW

logger = logging.getLogger(__name__)

_RANK: dict[str, int] = {"LOW": 0, "MED": 1, "HIGH": 2}


def _initial_status(severity: Severity, warmup: bool) -> str:
    """
    The status a newly-written alert point starts in.

    ══ WHY THIS AND NOT A BARE "FIRED" ══
    A HIGH-severity finding does not get to page anyone until a human has
    seen it — that is Phase 7's approval gate. Routing it through
    PENDING_APPROVAL here, at the moment the point is upserted, means the
    Qdrant index already reflects the pending decision before the graph even
    reaches human_approval_node: a second candidate describing the same event
    while the first is still awaiting approval correctly suppresses against
    it, because PENDING_APPROVAL is in DEDUP_ACTIVE_STATUSES.

    warmup outranks severity: an observe-only cycle reports nothing at any
    severity, so there is nothing to approve.
    """
    if warmup:
        return WARMUP_STATUS
    if severity == "HIGH":
        return PENDING_APPROVAL_STATUS
    return "FIRED"


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


def check_exact(candidate: CandidateAlert, *, now: datetime | None = None) -> DedupOutcome | None:
    """
    Step 1 on its own: the free exact-key path.

    Split out so ``deduplicate`` can run it across the WHOLE batch before
    anything is summarised. That ordering is what makes the "zero cost" claim
    true in practice rather than only per-candidate — see deduplicate.

    Parameters
    ----------
    candidate : CandidateAlert
    now : datetime, optional

    Returns
    -------
    DedupOutcome or None
        None means this event has not been seen and must go through the
        semantic path.
    """
    moment = now or datetime.now(timezone.utc)
    key = dedup_key(candidate)
    alert_id = alert_id_for(key)

    # Before embedding, before the LLM, before any search. The same accession
    # number, the same article URL, the same day's price move in the same band.
    existing = find_exact(key, alert_id=alert_id)
    if existing is None:
        return None

    count = int(existing.get("occurrence_count", 1)) + 1
    bump_occurrence(alert_id, count=count, last_seen_at=moment.isoformat())

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


def decide(
    candidate: CandidateAlert,
    *,
    summary: str,
    warmup: bool = False,
    now: datetime | None = None,
    skip_exact: bool = False,
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
    skip_exact : bool, default False
        Skip step 1 because the caller has already run it. Only ``deduplicate``
        sets this, and only for candidates check_exact already returned None
        for — running it twice would be harmless but would double the point
        fetch on every survivor.

    Returns
    -------
    DedupOutcome
    """
    from src.vectorstore.embeddings import get_embedder

    moment = now or datetime.now(timezone.utc)
    key = dedup_key(candidate)

    # ── 1. Exact-key fast path — free ───────────────────
    if not skip_exact:
        exact = check_exact(candidate, now=moment)
        if exact is not None:
            return exact

    # ── 2. Severity BEFORE dedup — step 6 depends on it ──
    severity, severity_reason = severity_score(candidate)

    # ── 3. Embed — symmetric, no bge query prefix ───────
    # This compares an alert to other alerts. Both sides are the same kind of
    # text, so the asymmetric query instruction would only add noise.
    alert = build_alert(
        candidate,
        severity=severity,
        summary=summary,
        status=_initial_status(severity, warmup),
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

    # ── 6. Asymmetric-cost guardrail — evaluated BEFORE 5 acts ──
    # The rule is not "a HIGH alert always fires". It is: A HIGH-SEVERITY EVENT
    # MUST PRODUCE A REPORT UNLESS IT IS NEAR-IDENTICAL TO SOMETHING ALREADY
    # REPORTED. Missing a real 8-K Item 4.02 because it read like last week's
    # costs far more than one duplicate ping.
    #
    # So it has to know which branches would REPORT NOTHING. Two do: a
    # suppression, and a merge whose severity did not rise above its parent's.
    # An ESCALATING merge already reaches the reader AND links to its parent
    # AND moves the cluster centroid — overriding it with a bare FIRE would
    # report the same thing while throwing all three away.
    #
    # ══ AND IT GATES ON THE PARENT'S SEVERITY, NOT ON SIMILARITY ══
    # The plan specified a similarity floor: never suppress a HIGH alert below
    # 0.96, whatever the thresholds say. The first live run of the semantic
    # path refuted it — three outlets on one DOJ probe scored 0.898 and 0.913,
    # so a 0.96 floor does not prevent a missed event, it guarantees one page
    # per outlet for every HIGH story. See src/vectorstore/config.py.
    #
    # The guarantee being reached for is "the reader learns about this HIGH
    # event". If the parent was itself HIGH and fired, they already have.
    #
    # Evaluated before step 5 runs so the override happens instead of the
    # merge's side effects, not on top of them.
    parent_severity = str(best.get("severity", "LOW"))
    parent_rank = _RANK.get(parent_severity, 0)

    if best_score >= TAU_HIGH:
        reports_nothing = True
    elif best_score >= TAU_LOW:
        reports_nothing = _RANK.get(severity, 0) <= parent_rank
    else:
        # No neighbour worth considering — step 5 fires anyway, and calling
        # that a guardrail save would overstate what the rule did.
        reports_nothing = False

    if reports_nothing and severity == "HIGH" and parent_rank < _RANK["HIGH"]:
        reason = (
            f"HIGH severity ({severity_reason}) matched a {parent_severity} alert at {best_score:.3f} — "
            "the reader has not been told about this at HIGH severity"
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
    Run a whole cycle's candidates through the engine, in two passes.

    ══ PASS 1 IS FREE, AND HAS TO RUN FIRST ══
    Every candidate gets the exact-key check before ANYTHING is summarised or
    embedded. In steady state that resolves most of a cycle: the same filing
    seen again, the same article re-served by the feed, the same day's price
    move re-measured.

    Running the batched summary call up front instead would be simpler and
    would spend a `flash` call on every cycle whose candidates are all exact
    duplicates — which, measured on a live run, is the ordinary case: 5 of 5
    candidates resolved on the exact path, and the model had already been
    called for all five. Cheap per cycle, and permanently wrong about what the
    fast path costs.

    ══ PASS 2 IS SEQUENTIAL, AND THAT IS THE POINT ══
    Each decision writes to the same index the next decision searches. Three
    outlets covering one story in a single cycle must collapse into one alert
    plus two suppressions, and that only works if the first is INDEXED before
    the second is SEARCHED. Parallelising would race them against an empty
    index and fire three times.

    Parameters
    ----------
    candidates : list of CandidateAlert
        In whatever order the monitors produced them.
    warmup : bool, default False
        Index everything, report nothing.
    now : datetime, optional
    summaries : list of str, optional
        Pre-computed canonical summaries, aligned with ``candidates`` by
        position. Computed for the survivors when omitted.
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

    # ── Pass 1: the free exact-key sweep ────────────────
    resolved: dict[int, DedupOutcome] = {}
    survivors: list[int] = []

    for index, item in enumerate(candidates):
        outcome = check_exact(item, now=now)
        if outcome is not None:
            resolved[index] = outcome
        else:
            survivors.append(index)

    # ── Pass 2: one batched summary call, survivors only ──
    texts: list[str]
    if not survivors:
        # Guarded rather than left to a summarize([]) that happens to return
        # early. "No survivors means no model call" is a contract worth stating
        # in the control flow.
        logger.info("Dedup: %d candidates, all resolved on the exact path — no model call", len(candidates))
        texts = []
    elif summaries is not None:
        # Indexed by ORIGINAL position: a caller supplying summaries aligned
        # them with `candidates`, not with whichever subset survived pass 1.
        texts = [summaries[i] if i < len(summaries) else "" for i in survivors]
    else:
        texts = summarize([candidates[i] for i in survivors])

    for position, index in enumerate(survivors):
        summary = texts[position] if position < len(texts) else ""
        resolved[index] = decide(
            candidates[index],
            summary=summary,
            warmup=warmup,
            now=now,
            skip_exact=True,
        )

    outcomes = [resolved[index] for index in range(len(candidates))]

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
