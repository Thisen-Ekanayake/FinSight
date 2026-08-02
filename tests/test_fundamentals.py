# ═══════════════════════════════════════════════════════
# FinSight — Tests: Fundamentals Period & Concept Selection
# ═══════════════════════════════════════════════════════
#
# These pin two traps in the SEC companyfacts shape that Phase 5's eval suite
# found in production data. Both produce the same symptom — a perfectly-formed
# citation on a figure that is years out of date — which no amount of citation
# checking can catch, because the citation is genuine.
#
# Offline: companyfacts is stubbed, so nothing here touches the network.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from unittest.mock import patch

from src.data.fundamentals import (
    _freshest_concept,
    _period_key,
    _period_label,
    _period_matches,
    get_fundamentals_history,
)
from src.data.schemas import XBRLFact


def _fact(
    *,
    value: float,
    start: str | None,
    end: str,
    fiscal_year: int,
    accession: str,
    fiscal_period: str = "FY",
    form: str = "10-K",
    filed: str = "",
    concept: str = "Revenues",
) -> XBRLFact:
    return XBRLFact(
        concept=concept,
        label=None,
        value=value,
        unit="USD",
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_start=start,
        period_end=end,
        form_type=form,
        accession_no=accession,
        filed_date=filed or f"{end[:4]}-03-01",
    )


# NVIDIA's FY2026 10-K, as companyfacts actually returns it: three fiscal years
# of revenue from the comparative income statement, ALL stamped fiscal_year=2026
# because that is the year of the FILING.
NVDA_COMPARATIVES = [
    _fact(
        value=60_922_000_000.0, start="2023-01-30", end="2024-01-28", fiscal_year=2026, accession="0001045810-26-000021"
    ),
    _fact(
        value=130_497_000_000.0,
        start="2024-01-29",
        end="2025-01-26",
        fiscal_year=2026,
        accession="0001045810-26-000021",
    ),
    _fact(
        value=215_938_000_000.0,
        start="2025-01-27",
        end="2026-01-25",
        fiscal_year=2026,
        accession="0001045810-26-000021",
    ),
]


class TestPeriodIdentity:
    """`fiscal_year` belongs to the filing; only the period span belongs to the fact."""

    def test_three_comparatives_share_one_fiscal_year(self):
        # The premise. If this ever stops being true the rest is unnecessary.
        assert {f["fiscal_year"] for f in NVDA_COMPARATIVES} == {2026}

    def test_but_they_are_three_distinct_periods(self):
        assert len({_period_key(f) for f in NVDA_COMPARATIVES}) == 3

    def test_the_label_comes_from_the_facts_own_end_date(self):
        assert [_period_label(f) for f in NVDA_COMPARATIVES] == ["2024 FY", "2025 FY", "2026 FY"]

    def test_a_quarter_keeps_the_filings_own_period_label(self):
        quarter = _fact(
            value=57_006_000_000.0,
            start="2025-06-29",
            end="2025-09-27",
            fiscal_year=2025,
            accession="0000320193-25-000073",
            fiscal_period="Q4",
            form="10-Q",
        )
        assert _period_label(quarter) == "2025 Q4"

    def test_a_balance_sheet_instant_in_a_ten_k_is_annual(self):
        instant = _fact(
            value=359_241_000_000.0,
            start=None,
            end="2025-09-27",
            fiscal_year=2025,
            accession="0000320193-25-000079",
        )
        assert _period_matches(instant, annual_only=True)
        assert _period_label(instant) == "2025 FY"

    def test_a_quarter_tagged_fy_is_not_an_annual_period(self):
        # A 10-K carries its own Q4 figures, and a Q4 duration ENDS ON THE SAME
        # DATE as the fiscal year — so period_end alone cannot separate them.
        # Passing one off as a year is a 4x error in a filed figure.
        q4 = _fact(
            value=102_466_000_000.0,
            start="2025-06-29",
            end="2025-09-27",
            fiscal_year=2025,
            accession="0000320193-25-000079",
        )
        year = _fact(
            value=416_161_000_000.0,
            start="2024-09-29",
            end="2025-09-27",
            fiscal_year=2025,
            accession="0000320193-25-000079",
        )

        assert not _period_matches(q4, annual_only=True)
        assert _period_matches(year, annual_only=True)
        assert _period_key(q4) != _period_key(year)

    def test_a_ten_q_fact_is_never_annual(self):
        fact = _fact(
            value=1.0,
            start="2024-09-29",
            end="2025-09-27",
            fiscal_year=2025,
            accession="x",
            form="10-Q",
        )
        assert not _period_matches(fact, annual_only=True)


class TestConceptFreshness:
    """Filers switch tags between years; the first tag with data may be stale."""

    OLD = [
        _fact(
            value=26_914_000_000.0, start="2021-02-01", end="2022-01-30", fiscal_year=2024, accession="a", concept="Old"
        ),
    ]

    def test_the_concept_reaching_furthest_forward_wins(self):
        facts = {"Old": self.OLD, "Revenues": NVDA_COMPARATIVES}

        assert _freshest_concept(facts, ["Old", "Revenues"], annual_only=True) == "Revenues"

    def test_preference_order_only_breaks_ties(self):
        facts = {"Preferred": NVDA_COMPARATIVES, "Alternate": list(NVDA_COMPARATIVES)}

        assert _freshest_concept(facts, ["Preferred", "Alternate"], annual_only=True) == "Preferred"

    def test_a_concept_the_company_never_reports_yields_nothing(self):
        # A bank asked about gross profit. Returning None is the honest answer;
        # the unanswerable archetype in the eval suite is built on this.
        assert _freshest_concept({}, ["GrossProfit"], annual_only=True) is None

    def test_a_concept_with_only_quarterly_data_does_not_win_an_annual_query(self):
        quarterly = [
            _fact(
                value=1.0,
                start="2026-04-01",
                end="2026-06-30",
                fiscal_year=2026,
                accession="q",
                fiscal_period="Q2",
                form="10-Q",
                concept="Newer",
            )
        ]
        facts = {"Revenues": NVDA_COMPARATIVES, "Newer": quarterly}

        assert _freshest_concept(facts, ["Revenues", "Newer"], annual_only=True) == "Revenues"


class TestHistorySelection:
    """End to end over stubbed companyfacts."""

    def _history(self, facts: dict, **kwargs):
        with (
            patch("src.data.fundamentals.resolve_cik", return_value="0001045810"),
            patch("src.data.fundamentals.get_company_facts", return_value=facts),
        ):
            return get_fundamentals_history("NVDA", metrics=["revenue"], **kwargs)

    def test_comparatives_become_three_years_not_one(self):
        # The regression. Keyed on fiscal_year, all three collapse to a single
        # entry and the series silently stops at whichever was written last.
        history = self._history({"Revenues": NVDA_COMPARATIVES}, periods=3)
        revenue = history["revenue"]

        assert [r["period"] for r in revenue] == ["2024 FY", "2025 FY", "2026 FY"]
        assert [r["value"] for r in revenue] == [60_922_000_000.0, 130_497_000_000.0, 215_938_000_000.0]

    def test_the_series_is_ordered_oldest_first(self):
        revenue = self._history({"Revenues": NVDA_COMPARATIVES}, periods=3)["revenue"]

        assert revenue[-1]["value"] > revenue[0]["value"]

    def test_a_stale_first_choice_concept_does_not_freeze_the_series(self):
        stale = [
            _fact(
                value=26_914_000_000.0,
                start="2021-02-01",
                end="2022-01-30",
                fiscal_year=2024,
                accession="stale",
                concept="RevenueFromContractWithCustomerExcludingAssessedTax",
            )
        ]
        facts = {"RevenueFromContractWithCustomerExcludingAssessedTax": stale, "Revenues": NVDA_COMPARATIVES}

        latest = self._history(facts, periods=1)["revenue"][-1]

        assert latest["value"] == 215_938_000_000.0
        assert latest["period"] == "2026 FY"

    def test_the_earliest_filing_of_a_period_is_the_one_cited(self):
        # A figure restated in two later filings should still cite the filing
        # that first reported it.
        original = _fact(
            value=130_497_000_000.0,
            start="2024-01-29",
            end="2025-01-26",
            fiscal_year=2025,
            accession="0001045810-25-000023",
            filed="2025-02-26",
        )
        restated = _fact(
            value=130_497_000_000.0,
            start="2024-01-29",
            end="2025-01-26",
            fiscal_year=2026,
            accession="0001045810-26-000021",
            filed="2026-02-25",
        )

        selected = self._history({"Revenues": [restated, original]}, periods=1)["revenue"][-1]

        assert selected["source_id"] == "0001045810-25-000023"

    def test_periods_caps_the_series_from_the_recent_end(self):
        revenue = self._history({"Revenues": NVDA_COMPARATIVES}, periods=2)["revenue"]

        assert [r["period"] for r in revenue] == ["2025 FY", "2026 FY"]
