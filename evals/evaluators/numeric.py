# ═══════════════════════════════════════════════════════
# FinSight — Numeric Evaluators (deterministic, ground-truth)
# ═══════════════════════════════════════════════════════
#
# Purpose : Score the answer against figures taken from SEC XBRL, and score
#           the unanswerable archetype on whether it declined to answer.
#
# Public API:
#   numeric_accuracy(inputs, outputs, reference_outputs)
#   refusal_correctness(inputs, outputs)
#
# ══ WHY THIS IS THE EVALUATOR THAT CANNOT BE FOOLED ══
#   citation_coverage asks "does this number match something a tool returned?"
#   — a question the system answers with its own output. If a tool returns the
#   WRONG figure, the answer cites it faithfully, coverage reads 1.0, and the
#   number is four years stale.
#
#   numeric_accuracy asks a different question: does the number match what SEC
#   XBRL actually filed? Ground truth comes from evals/build_dataset.py, which
#   reads companyfacts directly and never touches the data layer under test.
#   This is the pair the plan calls "right citation, wrong number", and it is
#   the only metric here that can fail while every other one passes.
#
# ══ WHY UNANSWERABLE IS SCORED INVERTED ══
#   For eight of the forty questions the correct output is a refusal. Scoring
#   them like the rest would reward a confident fabrication and punish the one
#   behaviour that makes a research assistant safe to use.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any

from evals.config import FACT_REL_TOLERANCE, REFUSAL_MARKERS
from src.research.citation_verifier import extract_numeric_claims


def _appears_in(value: float, answer: str) -> bool:
    """
    True if ``answer`` states ``value`` within tolerance.

    Comparison is on magnitude, matching the verifier: prose carries direction
    in words ("fell", "a decline of") and in parentheses, and an extractor that
    guessed at sign would fail correct answers.
    """
    target = abs(value)
    for claim in extract_numeric_claims(answer):
        candidate = abs(claim.value)
        if abs(candidate - target) / max(candidate, target, 1e-9) <= FACT_REL_TOLERANCE:
            return True
    return False


def numeric_accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> Any:
    """
    Share of the example's XBRL-sourced figures the answer actually states.

    Parameters
    ----------
    inputs : dict
        Needs ``archetype``.
    outputs : dict
        Needs ``answer``.
    reference_outputs : dict
        Needs ``expected_facts``.

    Returns
    -------
    dict
        ``{"results": []}`` — no feedback at all — for examples with no pinned
        figures. Narrative questions have no numbers to check and macro
        questions cannot have any, since a live FRED series is not ground
        truth. Emitting 1.0 for them would inflate the mean with examples the
        evaluator never examined.
    """
    expected: list[dict] = (reference_outputs or {}).get("expected_facts") or []
    if not expected:
        return {"results": []}

    answer: str = outputs.get("answer") or ""
    hits = [fact for fact in expected if _appears_in(float(fact["value"]), answer)]
    missing = [fact["label"] for fact in expected if fact not in hits]

    return {
        "key": "numeric_accuracy",
        "score": len(hits) / len(expected),
        "comment": (
            f"{len(hits)}/{len(expected)} filed figures stated correctly"
            + (f" — missing or wrong: {', '.join(missing)}" if missing else "")
        ),
    }


def refusal_correctness(inputs: dict, outputs: dict) -> Any:
    """
    Unanswerable archetype: did the system decline instead of inventing?

    Two independent failure modes, scored together because either one alone is
    a wrong answer:

    * No hedge at all — the answer reads as authoritative.
    * A large figure present — the specific fabrication risk, since a model
      asked for an unavailable number reaches for a plausible one. Small
      numbers are exempt: an answer may legitimately reference a fiscal year
      or an item number while refusing.

    Returns
    -------
    dict
        ``{"results": []}`` for the four answerable archetypes.
    """
    if inputs.get("archetype") != "unanswerable":
        return {"results": []}

    answer: str = outputs.get("answer") or ""
    lowered = answer.lower()

    hedged = any(marker in lowered for marker in REFUSAL_MARKERS)
    # extract_numeric_claims already drops years, dates, and bare integers
    # under 100, so anything left is a financial assertion.
    fabricated = [claim.text for claim in extract_numeric_claims(answer)]

    if hedged and not fabricated:
        return {
            "key": "refusal_correctness",
            "score": 1.0,
            "comment": "declined, and stated no figure",
        }
    if hedged and fabricated:
        return {
            "key": "refusal_correctness",
            "score": 0.5,
            "comment": f"declined but still stated {len(fabricated)} figure(s): {', '.join(fabricated[:3])}",
        }
    return {
        "key": "refusal_correctness",
        "score": 0.0,
        "comment": ("answered a question with no available data" if fabricated else "did not signal the data gap"),
    }
