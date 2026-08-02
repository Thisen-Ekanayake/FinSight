# ═══════════════════════════════════════════════════════
# FinSight — Citation Evaluators (deterministic)
# ═══════════════════════════════════════════════════════
#
# Purpose : Score how well an answer's numbers and source markers are backed by
#           what the tools actually returned. Zero LLM calls, zero quota.
#
# Public API:
#   citation_coverage(outputs)     ★ headline metric, scored on the DRAFT
#   answer_groundedness(outputs)     the same check on the shipped answer
#   source_validity(outputs, reference_outputs)
#
# ══ WHY THE HEADLINE METRIC IS SCORED ON THE DRAFT ══
#   finalize() STRIPS ungrounded claims, so coverage measured on the final
#   answer is close to 1.0 by construction — it measures whether stripping
#   works, not whether the system grounds what it writes.
#
#   The interesting quantity is how much of the model's FIRST attempt was
#   already grounded, because that is what a better synthesis prompt or better
#   retrieval actually moves. So citation_coverage scores the draft, and
#   answer_groundedness scores the final answer as a separate check: it should
#   sit at 1.0, and any dip is a stripping bug rather than a grounding one.
#
# ══ WHY THIS REUSES THE VERIFIER'S EXTRACTOR ══
#   A second number extractor written for the evaluator would drift from the
#   first, and then a disagreement between them would say nothing about the
#   system. What makes these numbers meaningful is that the SAME rule is
#   applied to every experiment, so the deltas are attributable to the change
#   under test. Ground-truth independence lives in numeric_accuracy, which
#   checks against XBRL rather than against anything the system produced.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any

from src.core.schemas import AgentFinding, Citation
from src.research.citation_verifier import (
    build_evidence_index,
    extract_numeric_claims,
    find_support,
    validate_source_markers,
)


def _grounded_fraction(answer: str, findings: list[AgentFinding]) -> tuple[float, int, int]:
    """
    Fraction of the answer's numbers that some finding supports.

    Returns
    -------
    tuple
        ``(coverage, grounded, total)``. An answer with no numbers at all
        scores 1.0 — a purely narrative answer has made no numeric claim to
        fail, and scoring it 0 would punish the narrative archetype for being
        narrative.
    """
    claims = extract_numeric_claims(answer or "")
    if not claims:
        return 1.0, 0, 0

    index = build_evidence_index(findings or [])
    grounded = sum(1 for claim in claims if find_support(claim, index) is not None)
    return grounded / len(claims), grounded, len(claims)


def citation_coverage(outputs: dict) -> dict:
    """
    ★ Headline metric: share of the drafted answer's numbers backed by evidence.

    Parameters
    ----------
    outputs : dict
        The target's return value; needs ``draft_answer`` and ``findings``.

    Returns
    -------
    dict
        LangSmith feedback with the count in the comment, because 3/4 and
        30/40 are the same score and very different situations.
    """
    draft = outputs.get("draft_answer") or outputs.get("answer") or ""
    coverage, grounded, total = _grounded_fraction(draft, outputs.get("findings") or [])

    return {
        "key": "citation_coverage",
        "score": coverage,
        "comment": f"{grounded}/{total} numeric claims grounded in retrieved findings",
    }


def answer_groundedness(outputs: dict) -> dict:
    """
    The same check on the answer a user actually sees.

    Expected to be 1.0: finalize removes anything stage 1 could not ground.
    A score below 1.0 means an ungrounded number survived stripping, which is
    a defect in finalize rather than in the synthesizer.
    """
    answer = outputs.get("answer") or ""
    coverage, grounded, total = _grounded_fraction(answer, outputs.get("findings") or [])

    return {
        "key": "answer_groundedness",
        "score": coverage,
        "comment": (
            f"{grounded}/{total} grounded in the shipped answer"
            + ("" if coverage >= 1.0 else " — an ungrounded number survived finalize")
        ),
    }


def source_validity(outputs: dict, reference_outputs: dict) -> Any:
    """
    Every inline marker must be well-formed AND resolve to a retrieved source.

    This is what catches a fabricated accession number: a plausible
    ``0000320193-99-000001`` has the right shape, cites nothing the system ever
    fetched, and reads as authoritative.

    When the example pins expected accession numbers, a second score reports
    how many of them the answer actually cited — a correct figure attributed to
    the wrong filing is still a citation failure.

    Parameters
    ----------
    outputs : dict
        Needs ``answer`` and ``citations``.
    reference_outputs : dict
        May carry ``expected_sources``.

    Returns
    -------
    dict
        ``{"results": [...]}`` — one or two feedback entries.
    """
    answer: str = outputs.get("answer") or ""
    citations: list[Citation] = outputs.get("citations") or []

    invalid = validate_source_markers(answer, citations)
    results: list[dict] = [
        {
            "key": "source_validity",
            "score": 0.0 if invalid else 1.0,
            "comment": "; ".join(invalid) if invalid else "all source markers resolve",
        }
    ]

    expected: list[str] = (reference_outputs or {}).get("expected_sources") or []
    if expected:
        cited = sum(1 for source_id in expected if source_id in answer)
        results.append(
            {
                "key": "expected_source_recall",
                "score": cited / len(expected),
                "comment": f"{cited}/{len(expected)} expected filings cited by accession number",
            }
        )

    return {"results": results}
