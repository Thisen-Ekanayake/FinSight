# ═══════════════════════════════════════════════════════
# FinSight — Tests: Citation Verifier
# ═══════════════════════════════════════════════════════
#
# Stage 1 is pure code, so every test here runs offline at zero quota. That is
# the point of doing the numeric check deterministically.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from src.core.schemas import AgentFinding, Citation
from src.research.citation_verifier import (
    Evidence,
    build_evidence_index,
    extract_numeric_claims,
    find_support,
    validate_source_markers,
    verify,
)


def _citation(source_type: str = "EDGAR", source_id: str = "0000320193-25-000079") -> Citation:
    return Citation(
        source_type=source_type,  # type: ignore[typeddict-item]
        source_id=source_id,
        url="https://example.com",
        as_of="2025-09-27",
        excerpt=None,
    )


def _finding(
    claim: str,
    *,
    agent: str = "fundamentals",
    ticker: str = "AAPL",
    metric: str | None = None,
    value: float | str | None = None,
    unit: str | None = "USD",
    citations: list[Citation] | None = None,
) -> AgentFinding:
    return AgentFinding(
        agent=agent,
        ticker=ticker,
        claim=claim,
        metric=metric,
        value=value,
        unit=unit,
        citations=citations if citations is not None else [_citation()],
        confidence=1.0,
        error=None,
    )


def _values(claims) -> list[float]:
    return [c.value for c in claims]


class TestNumberExtraction:
    def test_plain_integer_with_thousands_separators(self):
        assert _values(extract_numeric_claims("Revenue was 416,161,000,000 dollars")) == [416161000000.0]

    def test_currency_and_scale_suffix(self):
        assert _values(extract_numeric_claims("Revenue was $416.2B")) == [416.2e9]

    def test_spelled_out_scale(self):
        assert _values(extract_numeric_claims("Revenue was $391.0 billion")) == [391.0e9]

    def test_percentage_is_a_distinct_kind(self):
        claims = extract_numeric_claims("Gross margin was 46.91%")
        assert claims[0].kind == "percent"
        assert claims[0].value == 46.91

    def test_absolute_figure_is_a_distinct_kind(self):
        assert extract_numeric_claims("Revenue was $416.2B")[0].kind == "absolute"

    def test_parentheses_mean_negative(self):
        assert _values(extract_numeric_claims("Operating loss of (2,100.5)")) == [-2100.5]

    def test_basis_points_convert_to_percentage_points(self):
        claims = extract_numeric_claims("Margin expanded by 70 bps")
        assert claims[0].kind == "percent"
        assert claims[0].value == 0.70

    def test_offsets_point_into_the_original_text(self):
        text = "Gross margin was 46.91% in the period"
        claim = extract_numeric_claims(text)[0]
        assert text[claim.start : claim.end].strip().startswith("46.91")


class TestExtractionExclusions:
    """
    Digits that assert nothing must not become claims — every false positive
    here becomes an unsupported claim and a wasted repair branch.
    """

    def test_accession_numbers_inside_markers_are_not_claims(self):
        text = "Revenue was $416.2B [SRC:EDGAR:0000320193-25-000079]"
        assert _values(extract_numeric_claims(text)) == [416.2e9]

    def test_fiscal_year_labels_are_not_claims(self):
        assert extract_numeric_claims("FY2025 and fiscal 2024 and CY2023") == []

    def test_bare_years_are_not_claims(self):
        assert extract_numeric_claims("Between 2023 and 2025 the trend held") == []

    def test_quarter_labels_are_not_claims(self):
        assert extract_numeric_claims("Q3 2025 was strong") == []

    def test_iso_dates_are_not_claims(self):
        assert extract_numeric_claims("As of 2025-09-27 the figure stood") == []

    def test_small_bare_integers_are_prose(self):
        # "over 5 sessions", "the 10-year Treasury" — grounding these is noise.
        assert extract_numeric_claims("up over 5 sessions and the 10-year yield") == []

    def test_a_year_adjacent_to_a_decimal_is_still_a_number(self):
        # REGRESSION GUARD. Masking bare years must not eat the "2024" out of a
        # real figure like a share price of 2024.50.
        assert _values(extract_numeric_claims("The share price was $2024.50")) == [2024.50]

    def test_small_integer_with_a_currency_symbol_is_a_claim(self):
        assert _values(extract_numeric_claims("EPS of $7.42")) == [7.42]


class TestEvidenceIndex:
    def test_structured_value_becomes_evidence(self):
        index = build_evidence_index([_finding("AAPL revenue", metric="revenue@2025 FY", value=416161000000.0)])
        assert any(e.value == 416161000000.0 and not e.derived for e in index)

    def test_numbers_in_claim_text_become_evidence(self):
        """
        Claim strings are built by specialists with f-strings from tool output,
        so every number in them is grounded by construction. The technical
        agent reports its percentage changes only in claim text.
        """
        index = build_evidence_index(
            [
                _finding(
                    "AAPL closed at 254.43 on 2025-09-26, +1.20% on the day",
                    agent="technical",
                    metric="last_close",
                    value=254.43,
                )
            ]
        )
        assert any(e.kind == "percent" and abs(e.value - 1.20) < 1e-9 for e in index)

    def test_percent_unit_classifies_evidence_as_percent(self):
        index = build_evidence_index([_finding("Fed funds", metric="DFF", value=4.33, unit="Percent")])
        assert any(e.kind == "percent" and e.value == 4.33 for e in index)

    def test_ratio_between_same_period_figures_is_derived(self):
        # gross_profit / revenue = 46.91% — a number in no finding.
        index = build_evidence_index(
            [
                _finding("gross profit", metric="gross_profit@2025 FY", value=195201000000.0),
                _finding("revenue", metric="revenue@2025 FY", value=416161000000.0),
            ]
        )
        margins = [e.value for e in index if e.derived and e.kind == "percent"]
        assert any(abs(m - 46.9051) < 0.01 for m in margins)

    def test_ratios_do_not_cross_periods(self):
        index = build_evidence_index(
            [
                _finding("gross profit", metric="gross_profit@2025 FY", value=195201000000.0),
                _finding("revenue", metric="revenue@2024 FY", value=391035000000.0),
            ]
        )
        assert not [e for e in index if e.derived and e.kind == "percent" and 49.0 < e.value < 50.5]

    def test_period_over_period_growth_is_derived(self):
        index = build_evidence_index(
            [
                _finding("revenue", metric="revenue@2024 FY", value=391035000000.0),
                _finding("revenue", metric="revenue@2025 FY", value=416161000000.0),
            ]
        )
        # (416161 - 391035) / 391035 = 6.42%
        assert any(e.derived and abs(e.value - 6.4255) < 0.01 for e in index)

    def test_derived_evidence_inherits_both_operands_citations(self):
        index = build_evidence_index(
            [
                _finding("gp", metric="gross_profit@2025 FY", value=100.0, citations=[_citation()]),
                _finding(
                    "rev",
                    metric="revenue@2025 FY",
                    value=200.0,
                    citations=[_citation(source_id="0000320193-24-000123")],
                ),
            ]
        )
        derived = next(e for e in index if e.derived and abs(e.value - 50.0) < 1e-9)
        assert len({c["source_id"] for c in derived.citations}) == 2

    def test_narrative_findings_contribute_no_structured_value(self):
        index = build_evidence_index([_finding("supply chain concentration risk", agent="filings_rag", value=None)])
        assert all(e.derived or e.label.endswith("claim text") for e in index)


class TestMatching:
    def test_rounding_within_tolerance_is_supported(self):
        # "$391.0B" against a filed 391,035,000,000 is 0.009% apart.
        claim = extract_numeric_claims("Revenue was $391.0B")[0]
        assert find_support(claim, [Evidence(391035000000.0, "absolute", "revenue")]) is not None

    def test_a_genuinely_different_number_is_unsupported(self):
        claim = extract_numeric_claims("Revenue was $450.0B")[0]
        assert find_support(claim, [Evidence(391035000000.0, "absolute", "revenue")]) is None

    def test_percent_does_not_match_an_absolute_of_the_same_magnitude(self):
        claim = extract_numeric_claims("Margin was 46.91%")[0]
        assert find_support(claim, [Evidence(46.91, "absolute", "some_dollar_figure")]) is None

    def test_small_percentages_use_the_absolute_tolerance(self):
        # 0.50 vs 0.51 is 2% relative — rejected by the relative test alone.
        claim = extract_numeric_claims("Yield rose 0.50%")[0]
        assert find_support(claim, [Evidence(0.51, "percent", "spread")]) is not None

    def test_magnitude_matching_ignores_direction(self):
        # "fell 5.2%" is a positive token; the tool recorded -5.2.
        claim = extract_numeric_claims("The stock fell 5.2%")[0]
        assert find_support(claim, [Evidence(-5.2, "percent", "change_pct_1d")]) is not None

    def test_direct_evidence_is_preferred_over_derived(self):
        claim = extract_numeric_claims("Margin was 46.91%")[0]
        direct = Evidence(46.91, "percent", "reported_margin")
        derived = Evidence(46.91, "percent", "gp/rev", derived=True)
        assert find_support(claim, [derived, direct]) is direct


class TestSourceMarkerValidation:
    def test_a_marker_matching_a_real_citation_passes(self):
        answer = "Revenue was $416.2B [SRC:EDGAR:0000320193-25-000079]"
        assert validate_source_markers(answer, [_citation()]) == []

    def test_a_malformed_accession_number_is_rejected(self):
        answer = "Revenue was $416.2B [SRC:EDGAR:123-456]"
        assert "malformed" in validate_source_markers(answer, [_citation()])[0]

    def test_a_well_formed_but_unretrieved_source_is_rejected(self):
        # The shape is right and the model invented it anyway.
        answer = "Revenue was $416.2B [SRC:EDGAR:9999999999-99-999999]"
        assert "no such source" in validate_source_markers(answer, [_citation()])[0]

    def test_fred_series_ids_are_format_checked(self):
        answer = "The rate was 4.33% [SRC:FRED:not a series]"
        assert validate_source_markers(answer, []) != []

    def test_an_answer_with_no_markers_has_no_invalid_markers(self):
        assert validate_source_markers("No sources cited here.", []) == []


class TestVerify:
    def test_a_fully_grounded_answer_passes(self):
        findings = [_finding("AAPL revenue", metric="revenue@2025 FY", value=416161000000.0)]
        report = verify("Apple reported revenue of $416.2B [SRC:EDGAR:0000320193-25-000079]", findings, [_citation()])
        assert report["passed"]
        assert report["citation_coverage"] == 1.0

    def test_an_invented_number_is_caught(self):
        findings = [_finding("AAPL revenue", metric="revenue@2025 FY", value=416161000000.0)]
        report = verify("Apple reported revenue of $500.0B [SRC:EDGAR:0000320193-25-000079]", findings, [_citation()])
        assert not report["passed"]
        assert len(report["unsupported_claims"]) == 1

    def test_coverage_is_the_grounded_share(self):
        findings = [_finding("AAPL revenue", metric="revenue@2025 FY", value=416161000000.0)]
        report = verify("Revenue was $416.2B and net income was $99.9B", findings, [_citation()])
        assert report["citation_coverage"] == 0.5

    def test_an_answer_with_no_numbers_scores_one(self):
        # Nothing numeric to get wrong — a qualitative answer is stage 2's job.
        report = verify("Management flagged supply chain concentration as a risk.", [], [])
        assert report["citation_coverage"] == 1.0
        assert report["passed"]

    def test_a_computed_margin_is_not_flagged(self):
        """
        REGRESSION GUARD. The synthesizer computes gross margin from two filed
        figures. A verifier knowing only raw values would flag every ratio it
        ever wrote and repair-loop on correct answers.
        """
        findings = [
            _finding("gross profit", metric="gross_profit@2025 FY", value=195201000000.0),
            _finding("revenue", metric="revenue@2025 FY", value=416161000000.0),
        ]
        report = verify("Gross margin was 46.91% in fiscal 2025.", findings, [])
        assert report["passed"]

    def test_the_unsupported_claim_carries_the_whole_sentence(self):
        report = verify("Apple reported revenue of $500.0B in the period. Margins held.", [], [])
        assert "Apple reported revenue" in report["unsupported_claims"][0]["claim"]

    def test_repair_targets_an_agent_the_router_selected(self):
        report = verify("Revenue was $500.0B", [], [], selected_agents=["technical", "macro"])
        assert report["unsupported_claims"][0]["origin_agent"] == "technical"

    def test_repair_falls_back_to_fundamentals_when_the_plan_is_empty(self):
        report = verify("Revenue was $500.0B", [], [], selected_agents=[])
        assert report["unsupported_claims"][0]["origin_agent"] == "fundamentals"

    def test_a_bad_marker_fails_verification_even_with_grounded_numbers(self):
        findings = [_finding("AAPL revenue", metric="revenue@2025 FY", value=416161000000.0)]
        report = verify("Revenue was $416.2B [SRC:EDGAR:1111111111-11-111111]", findings, [_citation()])
        assert report["citation_coverage"] == 1.0
        assert not report["passed"]
