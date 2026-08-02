# ═══════════════════════════════════════════════════════
# FinSight — LLM-Judge Evaluators
# ═══════════════════════════════════════════════════════
#
# Purpose : The two questions no regex can answer — is a real citation attached
#           to a claim it does not support, and does the answer mean the same
#           thing as the reference.
#
# Public API:
#   citation_faithfulness(inputs, outputs)
#   answer_correctness(inputs, outputs, reference_outputs)
#
# ══ WHY THESE RUN LAST AND COST MONEY ══
#   Both use `pro`, so a 40-example run adds ~80 graded calls on top of the
#   graph's own traffic. Every deterministic evaluator runs first and for free;
#   these two exist only for what deterministic code provably cannot see.
#
#   citation_faithfulness catches CITED BUT IRRELEVANT — the marker is
#   well-formed, resolves to a retrieved source, and the number matches
#   something in the findings, so every deterministic check passes. Only
#   reading the claim beside its source reveals they are about different
#   things. That is a real failure mode of a citation-enforcing prompt: the
#   model learns to attach markers, not to attach the RIGHT markers.
#
# ══ ON GRADING A MODEL WITH A MODEL ══
#   Circular in the citation verifier, where the model would be checking its
#   own arithmetic. Not circular here: the judge sees the answer next to
#   evidence and a human-written reference it did not produce, and it is
#   scored on questions with fixed verdict vocabularies rather than asked
#   whether the answer is good.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from typing import Any

from evals.config import (
    CORRECTNESS_PROMPT_SYSTEM,
    CORRECTNESS_PROMPT_USER,
    FAITHFULNESS_PROMPT_SYSTEM,
    FAITHFULNESS_PROMPT_USER,
)
from src.core.schemas import AgentFinding

logger = logging.getLogger(__name__)

# Findings rendered into a judge prompt. Enough to check a citation against,
# far short of pasting an entire filing into every graded call.
MAX_JUDGE_FINDINGS: int = 24
MAX_JUDGE_CLAIM_CHARS: int = 400


def _render_findings(findings: list[AgentFinding]) -> str:
    """Render findings as a compact evidence block for a judge prompt."""
    if not findings:
        return "(no findings were retrieved)"

    lines: list[str] = []
    for finding in findings[:MAX_JUDGE_FINDINGS]:
        sources = ", ".join(f"{c['source_type']}:{c['source_id']}" for c in finding.get("citations") or []) or "-"
        claim = (finding.get("claim") or "")[:MAX_JUDGE_CLAIM_CHARS]
        lines.append(f"- [{finding.get('agent')}/{finding.get('ticker') or 'macro'}] {claim}  (sources: {sources})")

    if len(findings) > MAX_JUDGE_FINDINGS:
        lines.append(f"- ... and {len(findings) - MAX_JUDGE_FINDINGS} further findings")

    return "\n".join(lines)


def _grade(system: str, user: str, *, _mock_response: tuple[float, str] | None = None) -> tuple[float, str]:
    """
    Run one graded call and return ``(score, reasoning)``.

    The schema is two flat, required scalars for the same reason the router's
    is: Gemini's structured output goes through native JSON-schema validation
    and fights nested or optional fields.

    A failed call returns ``(-1.0, ...)`` rather than raising. One judge timing
    out should cost one score, not the whole experiment — and the caller turns
    -1.0 into "no feedback" rather than a zero, because an ungraded example is
    not a failing example.
    """
    if _mock_response is not None:
        return _mock_response

    from pydantic import BaseModel, Field

    from src.core.llm import get_llm
    from src.core.tracing import trace_metadata

    class Grade(BaseModel):
        """Flat and fully required — nested or optional fields fight Gemini's schema."""

        score: float = Field(description="0.0, 0.5, or 1.0")
        reasoning: str = Field(description="One sentence")

    try:
        model = get_llm("pro", temperature=0.0).with_structured_output(Grade)
        result = model.invoke(
            [("system", system), ("human", user)],
            config={"metadata": trace_metadata(phase="P5"), "tags": ["eval", "judge"]},
        )
        score = float(getattr(result, "score", 0.0))
        reasoning = str(getattr(result, "reasoning", ""))
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Judge call failed, leaving the example ungraded: %s", exc)
        return -1.0, f"judge unavailable: {exc}"

    # Clamp: a model asked for 0/0.5/1 occasionally returns 0.8 or 3.
    return max(0.0, min(1.0, score)), reasoning


def _feedback(key: str, score: float, reasoning: str) -> Any:
    """Wrap a grade as LangSmith feedback, dropping ungraded examples entirely."""
    if score < 0:
        return {"results": []}
    return {"key": key, "score": score, "comment": reasoning}


def citation_faithfulness(inputs: dict, outputs: dict, *, _mock_response: tuple[float, str] | None = None) -> Any:
    """
    Is each source marker attached to a claim its source genuinely supports?

    Parameters
    ----------
    inputs : dict
        Needs ``question``.
    outputs : dict
        Needs ``answer`` and ``findings``.
    _mock_response : tuple, optional
        ``(score, reasoning)``, bypassing the LLM in tests.

    Returns
    -------
    dict
        LangSmith feedback, or ``{"results": []}`` when the answer carries no
        markers to audit — an answer that cites nothing cannot cite wrongly,
        and source_validity already scores whether markers are present.
    """
    answer: str = outputs.get("answer") or ""
    if "[SRC:" not in answer:
        return {"results": []}

    user = FAITHFULNESS_PROMPT_USER.format(
        question=inputs.get("question", ""),
        answer=answer,
        findings=_render_findings(outputs.get("findings") or []),
    )
    score, reasoning = _grade(FAITHFULNESS_PROMPT_SYSTEM, user, _mock_response=_mock_response)
    return _feedback("citation_faithfulness", score, reasoning)


def answer_correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
    *,
    _mock_response: tuple[float, str] | None = None,
) -> Any:
    """
    Does the answer convey the same substance as the reference?

    Applies to every archetype including unanswerable, where the reference
    itself declines — so declining scores 1.0 and supplying a confident figure
    scores 0.0. That inversion is in the prompt rather than in a branch here,
    because it is a grading rule, not control flow.

    Parameters
    ----------
    inputs : dict
        Needs ``question``.
    outputs : dict
        Needs ``answer``.
    reference_outputs : dict
        Needs ``reference_answer``.
    _mock_response : tuple, optional
        ``(score, reasoning)``, bypassing the LLM in tests.

    Returns
    -------
    dict
    """
    reference: str = (reference_outputs or {}).get("reference_answer") or ""
    if not reference:
        return {"results": []}

    user = CORRECTNESS_PROMPT_USER.format(
        question=inputs.get("question", ""),
        reference=reference,
        answer=outputs.get("answer") or "(the system produced no answer)",
    )
    score, reasoning = _grade(CORRECTNESS_PROMPT_SYSTEM, user, _mock_response=_mock_response)
    return _feedback("answer_correctness", score, reasoning)
