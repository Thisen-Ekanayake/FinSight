# ═══════════════════════════════════════════════════════
# FinSight — Tests: Aggregator / Synthesizer
# ═══════════════════════════════════════════════════════
#
# Conflict resolution is pure logic and tested offline. Synthesis uses
# _mock_response, so no Vertex spend.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import pytest

from src.core.schemas import AgentFinding, Citation
from src.research.aggregator import (
    _format_conflicts,
    detect_conflicts,
    format_findings,
    synthesize,
)


def _citation(source_type: str, source_id: str = "acc-1") -> Citation:
    return Citation(
        source_type=source_type,  # type: ignore[typeddict-item]
        source_id=source_id,
        url="https://example.com",
        as_of="2025-09-27",
        excerpt=None,
    )


def _finding(
    agent: str = "fundamentals",
    *,
    metric: str | None = "revenue",
    value: float | None = 100.0,
    source: str = "EDGAR",
    ticker: str = "AAPL",
    confidence: float = 1.0,
) -> AgentFinding:
    return AgentFinding(
        agent=agent,
        ticker=ticker,
        claim=f"{ticker} {metric} is {value}",
        metric=metric,
        period=None,
        value=value,
        unit="USD",
        citations=[_citation(source)],
        confidence=confidence,
        error=None,
    )


class TestConflictDetection:
    """
    Two sources disagreeing is INFORMATION. Silently picking one is how a
    system produces a confident wrong number.
    """

    def test_single_source_is_never_a_conflict(self):
        conflicts, superseded = detect_conflicts([_finding()])
        assert conflicts == []
        assert superseded == set()

    def test_agreement_within_tolerance_is_not_a_conflict(self):
        # 100.0 vs 100.5 is 0.5%, inside the 1% tolerance.
        findings = [_finding(value=100.0), _finding(value=100.5, source="YFINANCE")]
        conflicts, superseded = detect_conflicts(findings)
        assert conflicts == []
        # The redundant one is still dropped so the prompt does not carry the
        # same figure twice.
        assert len(superseded) == 1

    def test_disagreement_beyond_tolerance_is_a_conflict(self):
        findings = [_finding(value=100.0), _finding(value=150.0, source="YFINANCE")]
        conflicts, _ = detect_conflicts(findings)
        assert len(conflicts) == 1
        assert conflicts[0]["metric"] == "revenue"

    def test_higher_trust_source_wins(self):
        # EDGAR is authoritative; yfinance re-publishes with normalisation.
        findings = [_finding(value=150.0, source="YFINANCE"), _finding(value=100.0, source="EDGAR")]
        conflicts, _ = detect_conflicts(findings)
        assert conflicts[0]["chosen_source"] == "EDGAR"
        assert conflicts[0]["chosen_value"] == 100.0

    def test_losing_finding_is_superseded(self):
        findings = [_finding(value=100.0, source="EDGAR"), _finding(value=150.0, source="YFINANCE")]
        _, superseded = detect_conflicts(findings)
        assert 1 in superseded

    def test_conflict_records_every_reported_value(self):
        findings = [_finding(value=100.0, source="EDGAR"), _finding(value=150.0, source="YFINANCE")]
        conflicts, _ = detect_conflicts(findings)
        sources = {source for source, _ in conflicts[0]["values"]}
        assert sources == {"EDGAR", "YFINANCE"}

    def test_relative_difference_is_recorded(self):
        findings = [_finding(value=100.0, source="EDGAR"), _finding(value=150.0, source="YFINANCE")]
        conflicts, _ = detect_conflicts(findings)
        assert conflicts[0]["rel_difference"] == pytest.approx(0.5)

    def test_different_tickers_never_conflict(self):
        findings = [_finding(ticker="AAPL", value=100.0), _finding(ticker="MSFT", value=999.0)]
        conflicts, _ = detect_conflicts(findings)
        assert conflicts == []

    def test_different_metrics_never_conflict(self):
        findings = [_finding(metric="revenue", value=100.0), _finding(metric="net_income", value=999.0)]
        conflicts, _ = detect_conflicts(findings)
        assert conflicts == []

    def test_different_periods_are_not_a_conflict(self):
        """
        REGRESSION GUARD. The fundamentals agent returns several fiscal years,
        so the metric key carries the period. Without that, FY2024 revenue and
        FY2025 revenue would look like two sources disagreeing and one year
        would be silently dropped.
        """
        findings = [
            _finding(metric="revenue@2024 FY", value=391035.0),
            _finding(metric="revenue@2025 FY", value=416161.0),
        ]
        conflicts, superseded = detect_conflicts(findings)
        assert conflicts == []
        assert superseded == set()

    def test_non_numeric_findings_are_ignored(self):
        # Narrative findings carry value=None and must not be compared.
        findings = [_finding(value=None, metric=None), _finding(value=None, metric=None)]
        conflicts, _ = detect_conflicts(findings)
        assert conflicts == []

    def test_confidence_breaks_a_trust_tie(self):
        findings = [
            _finding(value=100.0, source="EDGAR", confidence=0.5),
            _finding(value=150.0, source="EDGAR", confidence=1.0),
        ]
        conflicts, _ = detect_conflicts(findings)
        assert conflicts[0]["chosen_value"] == 150.0


class TestFindingFormatting:
    def test_source_markers_use_the_verifiable_form(self):
        # The Phase 4 deterministic verifier parses exactly this shape.
        rendered = format_findings([_finding(source="EDGAR")])
        assert "[SRC:EDGAR:acc-1]" in rendered

    def test_findings_are_grouped_by_agent(self):
        rendered = format_findings([_finding(agent="fundamentals"), _finding(agent="macro", metric="DFF")])
        assert "## fundamentals" in rendered
        assert "## macro" in rendered

    def test_empty_findings_produce_an_explicit_note(self):
        # The model must be told the data was unavailable, not left to invent
        # coverage from silence.
        assert "unavailable" in format_findings([]).lower()


class TestConflictBlock:
    def test_empty_when_there_are_no_conflicts(self):
        assert _format_conflicts([]) == ""

    def test_instructs_the_model_to_state_the_disagreement(self):
        findings = [_finding(value=100.0, source="EDGAR"), _finding(value=150.0, source="YFINANCE")]
        conflicts, _ = detect_conflicts(findings)
        block = _format_conflicts(conflicts)
        assert "EDGAR" in block and "YFINANCE" in block
        assert "explicitly" in block


class TestSynthesize:
    def test_mock_response_bypasses_the_llm(self):
        assert synthesize("q", [], [], _mock_response="mocked answer") == "mocked answer"


@pytest.mark.llm
class TestLiveSynthesis:
    """Real Gemini pro tier."""

    def test_produces_source_markers(self):
        answer = synthesize("What was Apple's revenue?", [_finding(value=416161000000.0)], [])
        assert "[SRC:EDGAR:acc-1]" in answer

    def test_surfaces_a_conflict_rather_than_hiding_it(self):
        findings = [_finding(value=100.0, source="EDGAR"), _finding(value=150.0, source="YFINANCE")]
        conflicts, superseded = detect_conflicts(findings)
        kept = [f for i, f in enumerate(findings) if i not in superseded]

        answer = synthesize("What was AAPL revenue?", kept, conflicts)
        assert "EDGAR" in answer and "YFINANCE" in answer

    def test_admits_missing_data_rather_than_inventing_it(self):
        answer = synthesize("What was Apple's revenue in fiscal 2019?", [], [])
        assert any(word in answer.lower() for word in ("not", "unavailable", "no ", "cannot"))
