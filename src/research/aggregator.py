# ═══════════════════════════════════════════════════════
# FinSight — Aggregator / Synthesizer
# ═══════════════════════════════════════════════════════
#
# Purpose : Merge the specialists' findings, resolve numeric disagreements,
#           and write the grounded answer.
#
# Public API:
#   detect_conflicts(findings)
#   format_findings(findings)
#   synthesize(query, findings, conflicts)
#   aggregator_node(state)
#
# ══ DISAGREEMENT IS SURFACED, NOT SILENTLY RESOLVED ══
#   Two sources reporting different revenue is INFORMATION, not noise. Picking
#   one quietly is how a system produces a confident wrong number.
#
#   So: values within CONFLICT_REL_TOLERANCE are treated as agreeing and the
#   higher-trust source wins; beyond it a Conflict is recorded and injected
#   into the synthesis prompt with an explicit instruction to state both.
#
#     "EDGAR (accession 0000320193-25-000079) reports $416.2B; yfinance
#      reports $408.6B. Using the filed figure."
#
#   This is the behaviour Kensho's grounding layer exists to provide, and it
#   is the difference between a demo and something an analyst would trust.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from collections import defaultdict

from src.core.llm import get_llm
from src.core.schemas import SOURCE_TRUST, AgentFinding, Conflict
from src.core.tracing import trace_metadata
from src.research.config import (
    CONFLICT_REL_TOLERANCE,
    CONFLICTS_BLOCK_TEMPLATE,
    NO_FINDINGS_NOTE,
    SYNTHESIS_PROMPT_SYSTEM,
    SYNTHESIS_PROMPT_USER,
)
from src.research.state import ResearchState

logger = logging.getLogger(__name__)


def _trust(finding: AgentFinding) -> int:
    """Trust score for a finding, taken from its first citation's source type."""
    if not finding["citations"]:
        return 0
    return SOURCE_TRUST.get(finding["citations"][0]["source_type"], 0)


def detect_conflicts(findings: list[AgentFinding]) -> tuple[list[Conflict], set[int]]:
    """
    Find numeric disagreements about the same metric.

    Groups findings by (ticker, metric) and compares numeric values. Anything
    outside CONFLICT_REL_TOLERANCE is a Conflict; the higher-trust source is
    chosen, and the losing findings are marked for exclusion so the
    synthesizer is not handed two contradictory numbers to reconcile.

    Parameters
    ----------
    findings : list of AgentFinding
        All findings collected from the fan-out.

    Returns
    -------
    tuple
        ``(conflicts, superseded_indexes)`` — the second being positions in
        ``findings`` that lost a conflict and should be dropped.
    """
    groups: dict[tuple[str, str], list[tuple[int, AgentFinding]]] = defaultdict(list)

    for index, item in enumerate(findings):
        metric = item.get("metric")
        value = item.get("value")
        if metric and isinstance(value, (int, float)):
            groups[(item.get("ticker") or "", metric)].append((index, item))

    conflicts: list[Conflict] = []
    superseded: set[int] = set()

    for (ticker, metric), members in groups.items():
        if len(members) < 2:
            continue

        best_index, best = max(members, key=lambda pair: (_trust(pair[1]), pair[1]["confidence"]))
        best_value = float(best["value"])  # type: ignore[arg-type]

        disagreeing = []
        for index, item in members:
            if index == best_index:
                continue
            value = float(item["value"])  # type: ignore[arg-type]
            denominator = max(abs(best_value), abs(value), 1e-9)
            if abs(value - best_value) / denominator > CONFLICT_REL_TOLERANCE:
                disagreeing.append((index, item, value))
            else:
                # Agrees within tolerance — redundant, so drop it to avoid
                # the same figure appearing twice in the prompt.
                superseded.add(index)

        if not disagreeing:
            continue

        # Annotated as plain str: SourceType is a Literal, and list invariance
        # would otherwise reject assigning list[tuple[Literal, float]] to the
        # Conflict field's list[tuple[str, float]].
        values: list[tuple[str, float]] = [
            (str(best["citations"][0]["source_type"]) if best["citations"] else "UNKNOWN", best_value)
        ]
        worst_delta = 0.0
        for index, item, value in disagreeing:
            source = str(item["citations"][0]["source_type"]) if item["citations"] else "UNKNOWN"
            values.append((source, value))
            superseded.add(index)
            worst_delta = max(worst_delta, abs(value - best_value) / max(abs(best_value), 1e-9))

        conflict = Conflict(
            metric=metric,
            ticker=ticker or None,
            values=values,
            chosen_source=values[0][0],
            chosen_value=best_value,
            rel_difference=worst_delta,
        )
        conflicts.append(conflict)
        logger.warning(
            "Conflict on %s %s: %s — using %s (%.2f%% spread)",
            ticker or "macro",
            metric,
            values,
            conflict["chosen_source"],
            worst_delta * 100,
        )

    return conflicts, superseded


def format_findings(findings: list[AgentFinding]) -> str:
    """
    Render findings for the synthesis prompt, grouped by agent.

    Each line carries its source marker in the exact ``[SRC:TYPE:ID]`` form
    the synthesizer is instructed to reproduce — which is what makes the
    Phase 4 deterministic verifier possible at all.
    """
    if not findings:
        return NO_FINDINGS_NOTE

    by_agent: dict[str, list[AgentFinding]] = defaultdict(list)
    for item in findings:
        by_agent[item["agent"]].append(item)

    blocks = []
    for agent in sorted(by_agent):
        lines = [f"## {agent}"]
        for item in by_agent[agent]:
            markers = "".join(f" [SRC:{c['source_type']}:{c['source_id']}]" for c in item["citations"][:1])
            lines.append(f"- {item['claim']}{markers}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _format_conflicts(conflicts: list[Conflict]) -> str:
    """Render the conflicts block injected into the synthesis prompt."""
    if not conflicts:
        return ""

    lines = []
    for conflict in conflicts:
        reported = "; ".join(f"{source} reports {value:,.2f}" for source, value in conflict["values"])
        lines.append(
            f"- {conflict['ticker'] or 'macro'} {conflict['metric']}: {reported}. "
            f"Use {conflict['chosen_source']} ({conflict['chosen_value']:,.2f}) — it is the more authoritative "
            f"source — but state the disagreement explicitly."
        )
    return CONFLICTS_BLOCK_TEMPLATE.format(conflicts="\n".join(lines))


def synthesize(
    query: str,
    findings: list[AgentFinding],
    conflicts: list[Conflict],
    *,
    _mock_response: str | None = None,
) -> str:
    """
    Write the grounded answer from the collected findings.

    Uses the ``pro`` tier: this is the one step where reasoning quality
    materially changes the output, and it runs once per query rather than once
    per branch.

    Parameters
    ----------
    query : str
        The original user question.
    findings : list of AgentFinding
        Merged, conflict-resolved findings.
    conflicts : list of Conflict
        Disagreements to surface in the answer.
    _mock_response : str, optional
        Bypass the LLM in tests.

    Returns
    -------
    str
        The answer, with inline ``[SRC:...]`` markers.
    """
    if _mock_response is not None:
        return _mock_response

    user = SYNTHESIS_PROMPT_USER.format(
        query=query,
        findings=format_findings(findings),
        conflicts_block=_format_conflicts(conflicts),
    )

    response = get_llm("pro", temperature=0.0).invoke(
        [("system", SYNTHESIS_PROMPT_SYSTEM), ("human", user)],
        config={"metadata": trace_metadata(phase="P3"), "tags": ["subsystem1", "synthesizer"]},
    )
    return str(response.content).strip()


def aggregator_node(state: ResearchState) -> dict:
    """
    Graph node: resolve conflicts and synthesize the answer.

    Returns a partial state with ``conflicts`` and ``draft_answer``. Both have
    a single writer, so neither needs a reducer.
    """
    findings = state.get("findings", [])
    conflicts, superseded = detect_conflicts(findings)

    kept = [f for i, f in enumerate(findings) if i not in superseded]
    if superseded:
        logger.info("Aggregator: dropped %d superseded findings, kept %d", len(superseded), len(kept))

    errors = state.get("errors", [])
    if errors:
        logger.warning("Aggregator: %d specialist(s) failed — %s", len(errors), errors[:2])

    answer = synthesize(state["query"], kept, conflicts)
    return {"conflicts": conflicts, "draft_answer": answer}
