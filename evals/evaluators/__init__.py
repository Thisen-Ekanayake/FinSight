# ═══════════════════════════════════════════════════════
# FinSight — Evaluators
# ═══════════════════════════════════════════════════════
#
# Deterministic evaluators first, LLM judges second — the same ordering
# principle as the citation verifier itself, for the same reason. Code that
# checks arithmetic costs nothing and cannot be talked out of its answer.
#
# Public API:
#   DETERMINISTIC_EVALUATORS, JUDGE_EVALUATORS, all_evaluators(judges=True)
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any

from evals.evaluators.citation import (
    answer_groundedness,
    citation_coverage,
    source_validity,
)
from evals.evaluators.judge import answer_correctness, citation_faithfulness
from evals.evaluators.numeric import numeric_accuracy, refusal_correctness

DETERMINISTIC_EVALUATORS: tuple[Any, ...] = (
    citation_coverage,
    answer_groundedness,
    source_validity,
    numeric_accuracy,
    refusal_correctness,
)

JUDGE_EVALUATORS: tuple[Any, ...] = (
    citation_faithfulness,
    answer_correctness,
)


def all_evaluators(*, judges: bool = True) -> list[Any]:
    """
    Assemble the evaluator list for one experiment.

    Parameters
    ----------
    judges : bool, default True
        Include the two LLM judges. Dropping them makes a run free and fast
        while leaving every ground-truth metric intact — the right setting for
        checking the harness works before spending on a real experiment.

    Returns
    -------
    list
    """
    return [*DETERMINISTIC_EVALUATORS, *(JUDGE_EVALUATORS if judges else ())]


__all__ = [
    "DETERMINISTIC_EVALUATORS",
    "JUDGE_EVALUATORS",
    "all_evaluators",
    "answer_correctness",
    "answer_groundedness",
    "citation_coverage",
    "citation_faithfulness",
    "numeric_accuracy",
    "refusal_correctness",
    "source_validity",
]
