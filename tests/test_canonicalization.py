# ═══════════════════════════════════════════════════════
# FinSight — Tests: Alert Canonicalization
# ═══════════════════════════════════════════════════════
#
# The dedup engine can only be as good as the text it compares. These tests
# hold the boundary between VOLATILE tokens (magnitudes, dates — must go) and
# STABLE ones (form types, indicator windows — must stay), because getting it
# wrong in either direction breaks dedup silently:
#
#   too little stripped   the same event looks different       -> duplicate fires
#   too much stripped     different events look the same       -> REAL ALERT LOST
#
# No network, no LLM.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import pytest

from src.monitor.synthesizer import (
    canonical_text,
    contains_volatile,
    dedup_key,
    strip_volatile,
    summarize,
    template_summary,
)


def candidate(alert_type="PRICE_MOVE", *, ticker="AAPL", headline="", metrics=None, company="Apple Inc."):
    return {
        "ticker": ticker,
        "company_name": company,
        "alert_type": alert_type,
        "monitor": "test",
        "headline": headline,
        "detail": headline,
        "natural_key": "k",
        "metrics": metrics or {},
        "evidence": [],
        "observed_at": "2026-08-03",
    }


class TestVolatileDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "fell 5.2%",
            "up 12%",
            "closed at $210.11",
            "$1.2",
            "revenue of 391,035",
            "on 2026-08-03",
            "in Q3",
            "FY2026 results",
            "FY 2026",
            "August 3",
            "Aug 3",
            "3 Aug",
            "on Monday",
            "filed on the 3rd",
            "Item 4.02",
        ],
    )
    def test_magnitudes_and_dates_are_flagged(self, text):
        assert contains_volatile(text)

    @pytest.mark.parametrize(
        "text",
        [
            "8-K reporting departure of the principal financial officer",
            "10-Q filed",
            "breaking below the 20-day moving average",
            "sharp single-day decline on elevated volume",
            "auditor notified the company of non-reliance",
            "may face a regulatory review",
        ],
    )
    def test_stable_identifiers_are_not_flagged(self, text):
        """
        A form type is not a magnitude — every 8-K is an 8-K. Flagging these
        would send perfectly good summaries back to the template, and stripping
        them would collapse distinct filings into each other.
        """
        assert not contains_volatile(text)

    def test_may_the_modal_is_not_may_the_month(self):
        # Regression: "may" was in the always-stripped month list, so
        # "may face a probe" read as a date.
        assert not contains_volatile("the company may face a probe")
        assert contains_volatile("filed May 3")


class TestStripping:
    def test_a_percentage_leaves_no_stranded_sign(self):
        # Regression: with the decimal rule ahead of the percent rule, "5.2%"
        # lost its "5.2" and left a bare "%" behind.
        assert "%" not in strip_volatile("Apple fell 5.2% today")

    def test_dates_and_prices_go(self):
        stripped = strip_volatile("Apple fell 5.2% to $210.11 on Monday, August 3, 2026")
        assert not contains_volatile(stripped)
        assert "Apple" in stripped and "fell" in stripped

    def test_the_prose_survives(self):
        stripped = strip_volatile("AAPL faces DOJ antitrust probe over 30% App Store fee")
        assert "DOJ antitrust probe" in stripped
        assert "App Store fee" in stripped

    def test_stripping_is_idempotent(self):
        once = strip_volatile("NVDA up 12.4% after Q3 beat, FY2026 guidance raised")
        assert strip_volatile(once) == once


class TestTemplateSummary:
    def test_item_codes_become_words_not_holes(self):
        """
        THE case that makes template_summary build from structured fields
        rather than by stripping the headline. An item code is a number; if it
        were deleted, the most serious filing type on the list would collapse
        into a generic "8-K filed" and be suppressed against a routine one.
        """
        alarming = template_summary(candidate("NEW_FILING", metrics={"form_type": "8-K", "items": ["4.02"]}))
        routine = template_summary(candidate("NEW_FILING", metrics={"form_type": "8-K", "items": ["8.01"]}))

        assert "non-reliance" in alarming
        assert alarming != routine

    def test_every_template_output_is_numeric_free(self):
        cases = [
            candidate("NEW_FILING", metrics={"form_type": "8-K", "items": ["4.02", "5.02"]}),
            candidate("NEW_FILING", metrics={"form_type": "10-K", "items": []}),
            candidate("PRICE_MOVE", metrics={"change_pct_1d": -5.2, "volume_ratio": 2.1}),
            candidate("MACRO_EVENT", ticker="", metrics={"series_id": "T10Y2Y", "crossing": "below 0.0"}),
            candidate("MACRO_EVENT", ticker="", metrics={"series_id": "CPIAUCSL", "abs_change": 0.9}),
            candidate("NEWS_SENTIMENT", headline="Apple Inc. sued for $2.1B on Aug 3 over 30% fees"),
        ]
        for item in cases:
            summary = template_summary(item)
            assert summary, item["alert_type"]
            assert not contains_volatile(summary), (item["alert_type"], summary)

    def test_price_direction_survives_but_magnitude_does_not(self):
        down = template_summary(candidate("PRICE_MOVE", metrics={"change_pct_1d": -5.2}))
        up = template_summary(candidate("PRICE_MOVE", metrics={"change_pct_1d": 5.2}))

        assert "decline" in down and "advance" in up

    def test_volume_confirmation_appears_only_when_notable(self):
        heavy = template_summary(candidate("PRICE_MOVE", metrics={"change_pct_1d": -5.0, "volume_ratio": 2.4}))
        normal = template_summary(candidate("PRICE_MOVE", metrics={"change_pct_1d": -5.0, "volume_ratio": 1.0}))

        assert "elevated volume" in heavy
        assert "elevated volume" not in normal

    def test_news_drops_the_ticker_and_company_name(self):
        """
        Both are attached separately and filtered exactly. Leaving them in the
        embedded text adds a term identical across every candidate it could
        ever be compared against — pure noise in the similarity.
        """
        summary = template_summary(
            candidate("NEWS_SENTIMENT", headline="AAPL: Apple Inc. faces a DOJ probe", company="Apple Inc.")
        )
        assert summary == "faces a doj probe"

    @pytest.mark.parametrize(
        "headline,company,expected",
        [
            # The registered name in full, and the bare form headlines use.
            ("Apple Inc. faces a DOJ probe", "Apple Inc.", "faces a doj probe"),
            ("Apple sued over App Store fees", "Apple Inc.", "sued over app store fees"),
            ("JPMorgan Chase & Co. fined by regulators", "JPMorgan Chase & Co.", "fined by regulators"),
            ("NVIDIA shares slide on export curbs", "NVIDIA CORP", "shares slide on export curbs"),
        ],
    )
    def test_corporate_suffixes_do_not_survive_as_fragments(self, headline, company, expected):
        """
        Regression: a trailing "." defeats \\b, so "Apple Inc." matched nothing
        and a bare "Apple" ate the front, leaving "inc." in the embedded text.
        """
        summary = template_summary(candidate("NEWS_SENTIMENT", headline=headline, company=company))
        assert summary == expected

    def test_output_is_bounded(self):
        long_headline = " ".join(["word"] * 60)
        assert len(template_summary(candidate("NEWS_SENTIMENT", headline=long_headline)).split()) <= 15


class TestSummarizeAuditsTheModel:
    def test_a_clean_model_response_is_used(self):
        items = [candidate("PRICE_MOVE", metrics={"change_pct_1d": -5.0})]
        assert summarize(items, _mock_response=["sharp single-day decline"]) == ["sharp single-day decline"]

    def test_a_leaked_number_is_discarded_not_repaired(self):
        """
        A model that ignored the one rule that matters has not earned partial
        credit. Repairing its output would produce a summary written under a
        rule it was not following.
        """
        items = [candidate("PRICE_MOVE", metrics={"change_pct_1d": -5.2})]
        result = summarize(items, _mock_response=["sharp decline of 5.2% on the day"])

        assert result == [template_summary(items[0])]
        assert not contains_volatile(result[0])

    def test_one_bad_line_does_not_discard_the_batch(self):
        items = [
            candidate("PRICE_MOVE", metrics={"change_pct_1d": -5.0}),
            candidate("NEW_FILING", metrics={"form_type": "8-K", "items": ["4.02"]}),
        ]
        result = summarize(items, _mock_response=["fell 5.0% today", "auditor flagged non-reliance"])

        assert result[0] == template_summary(items[0])
        assert result[1] == "auditor flagged non-reliance"

    def test_a_short_response_pads_from_the_template(self):
        items = [candidate("PRICE_MOVE", metrics={"change_pct_1d": -5.0}), candidate("NEW_FILING")]
        result = summarize(items, _mock_response=["sharp single-day decline"])

        assert len(result) == 2
        assert result[1] == template_summary(items[1])

    def test_an_empty_string_falls_back(self):
        items = [candidate("NEW_FILING", metrics={"form_type": "10-K"})]
        assert summarize(items, _mock_response=["   "]) == [template_summary(items[0])]

    def test_an_empty_batch_never_calls_the_model(self):
        assert summarize([], _mock_response=None) == []


class TestCanonicalText:
    def test_scope_and_type_are_present_for_the_decision_log(self):
        text = canonical_text(candidate("PRICE_MOVE"), "sharp single-day decline")
        assert text == "AAPL Apple Inc. | PRICE_MOVE | sharp single-day decline"

    def test_a_tickerless_macro_alert_reads_as_macro(self):
        text = canonical_text(candidate("MACRO_EVENT", ticker="", company=""), "policy rate raised")
        assert text.startswith("MACRO | MACRO_EVENT |")

    def test_the_ticker_falls_back_when_the_company_name_is_unknown(self):
        text = canonical_text(candidate("PRICE_MOVE", company=""), "decline")
        assert text.startswith("AAPL AAPL |")


class TestDedupKey:
    def test_the_key_is_stable_for_the_same_event(self):
        assert dedup_key(candidate()) == dedup_key(candidate())

    def test_ticker_type_and_natural_key_all_participate(self):
        base = candidate()
        by_ticker = {**base, "ticker": "MSFT"}
        by_type = {**base, "alert_type": "NEW_FILING"}
        by_key = {**base, "natural_key": "other"}

        keys = {dedup_key(x) for x in (base, by_ticker, by_type, by_key)}
        assert len(keys) == 4

    def test_the_key_does_not_depend_on_the_headline(self):
        """
        Two observations of one event can be worded differently — the same
        price move re-reported, the same filing summarised twice. Only the
        natural key decides identity.
        """
        a = candidate(headline="AAPL fell 5.2%")
        b = candidate(headline="Apple slid 5.4% in afternoon trade")
        assert dedup_key(a) == dedup_key(b)
