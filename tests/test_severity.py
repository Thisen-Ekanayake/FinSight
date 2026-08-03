# ═══════════════════════════════════════════════════════
# FinSight — Tests: Deterministic Severity
# ═══════════════════════════════════════════════════════
#
# These tests are the reason severity is rules rather than a model. Every
# assertion below is a statement about the SYSTEM that stays true next month;
# none of them could be written against "the model judged it serious".
#
# No network, no LLM, no database.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import pytest

from src.monitor.config import FILING_HIGH_8K_ITEMS
from src.monitor.severity import explain, score, severity_for


def candidate(alert_type, **metrics):
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "alert_type": alert_type,
        "monitor": "test",
        "headline": "",
        "detail": "",
        "natural_key": "k",
        "metrics": metrics,
        "evidence": [],
        "observed_at": "2026-08-03",
    }


class TestPrice:
    @pytest.mark.parametrize(
        "change,expected",
        [(-1.0, "LOW"), (-3.9, "LOW"), (-4.0, "MED"), (-6.9, "MED"), (-7.0, "HIGH"), (12.0, "HIGH")],
    )
    def test_percentage_bands(self, change, expected):
        assert severity_for(candidate("PRICE_MOVE", change_pct_1d=change)) == expected

    def test_direction_does_not_change_severity(self):
        # A 9% rally is as unusual as a 9% drop. Alerting only on declines
        # would make the system a bear-market tool.
        assert severity_for(candidate("PRICE_MOVE", change_pct_1d=9.0)) == severity_for(
            candidate("PRICE_MOVE", change_pct_1d=-9.0)
        )

    def test_a_small_move_in_a_quiet_name_can_still_be_high(self):
        """
        Percentage alone is the wrong measure. 3% is nothing in NVDA and a
        headline in a utility — the z-score is what carries that.
        """
        quiet = candidate("PRICE_MOVE", change_pct_1d=-3.0, vol_zscore=-3.4)
        assert severity_for(quiet) == "HIGH"
        assert "sigma" in explain(quiet)

    def test_a_large_move_in_a_volatile_name_still_escalates_on_percentage(self):
        # The two routes are a max, not an average: a 9% move is HIGH even
        # when the name is volatile enough that it is only 1.2 sigma.
        volatile = candidate("PRICE_MOVE", change_pct_1d=-9.0, vol_zscore=-1.2)
        assert severity_for(volatile) == "HIGH"
        assert "%" in explain(volatile)

    def test_severity_is_ranked_not_compared_as_a_string(self):
        """
        Regression guard. "MED" > "HIGH" is True lexically, so comparing the
        two routes as strings silently downgrades every alarming move that
        happens to be less unusual than it is large.
        """
        # by_pct=HIGH, by_z=MED — the lexical bug would have picked MED.
        assert severity_for(candidate("PRICE_MOVE", change_pct_1d=-8.0, vol_zscore=-2.5)) == "HIGH"

    def test_a_missing_zscore_is_absent_not_zero(self):
        # A degraded data source must produce a quieter alert, not no cycle.
        assert severity_for(candidate("PRICE_MOVE", change_pct_1d=-5.0)) == "MED"


class TestFiling:
    @pytest.mark.parametrize("item", sorted(FILING_HIGH_8K_ITEMS))
    def test_every_configured_high_item_is_high(self, item):
        assert severity_for(candidate("NEW_FILING", form_type="8-K", items=[item])) == "HIGH"

    def test_item_402_is_high(self):
        """
        The scary one: the auditor has told the company its previously issued
        financials should not be relied upon. It arrives in exactly the same
        envelope as a routine press release.
        """
        alarming = candidate("NEW_FILING", form_type="8-K", items=["4.02"])
        routine = candidate("NEW_FILING", form_type="8-K", items=["8.01"])

        assert severity_for(alarming) == "HIGH"
        assert severity_for(routine) == "MED"

    def test_the_most_serious_item_wins_when_several_are_present(self):
        mixed = candidate("NEW_FILING", form_type="8-K", items=["8.01", "5.02", "2.02"])
        assert severity_for(mixed) == "HIGH"
        assert "5.02" in explain(mixed)

    def test_a_periodic_report_is_med_because_it_is_scheduled(self):
        # Material by definition, but its arrival is never a surprise.
        assert severity_for(candidate("NEW_FILING", form_type="10-K", items=[])) == "MED"
        assert severity_for(candidate("NEW_FILING", form_type="10-Q", items=[])) == "MED"

    def test_an_8k_with_no_recognised_item_is_low(self):
        assert severity_for(candidate("NEW_FILING", form_type="8-K", items=["9.99"])) == "LOW"

    def test_explanation_names_the_item_that_decided_it(self):
        assert "4.02" in explain(candidate("NEW_FILING", form_type="8-K", items=["4.02"]))


class TestNews:
    def test_high_requires_corroboration(self):
        """
        Provider sentiment is wrong often enough that a single badly-scored
        headline must not be able to page anyone at 3am.
        """
        alone = candidate("NEWS_SENTIMENT", sentiment=-0.8, source_count=1)
        corroborated = candidate("NEWS_SENTIMENT", sentiment=-0.8, source_count=3)

        assert severity_for(alone) == "MED"
        assert severity_for(corroborated) == "HIGH"

    def test_corroboration_alone_does_not_escalate_mild_sentiment(self):
        assert severity_for(candidate("NEWS_SENTIMENT", sentiment=-0.35, source_count=5)) == "MED"

    def test_positive_sentiment_is_low(self):
        assert severity_for(candidate("NEWS_SENTIMENT", sentiment=0.7, source_count=4)) == "LOW"

    def test_a_missing_source_count_assumes_one_outlet(self):
        # Absence of corroboration evidence is not evidence of corroboration.
        assert severity_for(candidate("NEWS_SENTIMENT", sentiment=-0.9)) == "MED"


class TestMacro:
    def test_rate_series_are_scored_on_absolute_change(self):
        """
        A 25bp move in the Fed funds rate is 0.25 in the series' own units and
        roughly 6% in relative terms. Scoring it on percent change would rank
        it alongside a 6% move in the CPI index, which would be historic.
        """
        assert severity_for(candidate("MACRO_EVENT", series_id="DFF", abs_change=0.25, pct_change=6.0)) == "MED"
        assert severity_for(candidate("MACRO_EVENT", series_id="DFF", abs_change=0.50, pct_change=12.0)) == "HIGH"

    def test_index_series_are_scored_on_percent_change(self):
        # CPIAUCSL is an index level near 310; its absolute change is
        # meaningless as a severity signal.
        assert severity_for(candidate("MACRO_EVENT", series_id="CPIAUCSL", abs_change=1.0, pct_change=0.35)) == "MED"
        assert severity_for(candidate("MACRO_EVENT", series_id="CPIAUCSL", abs_change=2.0, pct_change=0.7)) == "HIGH"

    def test_a_crossing_is_high_however_small_the_move(self):
        """
        The move taking the 10Y-2Y spread from +0.02 to -0.01 is 0.03 wide and
        is the most-watched recession signal on the list.
        """
        tiny = candidate("MACRO_EVENT", series_id="T10Y2Y", abs_change=-0.03, crossing="below 0.0")
        assert severity_for(tiny) == "HIGH"
        assert "crossed" in explain(tiny)

    def test_an_unknown_series_falls_back_to_a_percent_threshold(self):
        assert severity_for(candidate("MACRO_EVENT", series_id="NOPE", pct_change=0.1)) == "LOW"
        assert severity_for(candidate("MACRO_EVENT", series_id="NOPE", pct_change=3.0)) == "HIGH"


class TestUnknownTypes:
    def test_an_unrecognised_alert_type_is_low_not_an_exception(self):
        # A new monitor shipping before its rule must degrade quietly, not
        # abort the cycle for every other monitor's candidates.
        level, reason = score(candidate("SOMETHING_NEW"))
        assert level == "LOW"
        assert "no rule" in reason

    def test_empty_metrics_never_raise(self):
        for alert_type in ("PRICE_MOVE", "NEW_FILING", "NEWS_SENTIMENT", "MACRO_EVENT"):
            assert severity_for(candidate(alert_type)) in {"LOW", "MED", "HIGH"}
