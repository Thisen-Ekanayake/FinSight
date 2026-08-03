# ═══════════════════════════════════════════════════════
# FinSight — Tests: Headline Sentiment
# ═══════════════════════════════════════════════════════
#
# The scorer is deliberately crude — see src/data/sentiment.py for why. These
# tests hold it to what it actually claims: correct SIGN on unambiguous
# financial headlines, and a magnitude that lands on the right side of the
# severity thresholds. They do not assert precise scores, which would be
# testing the lexicon's weights rather than its behaviour.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import pytest

from src.data.sentiment import matched_terms, score_sentiment
from src.monitor.config import NEWS_HIGH_SENTIMENT, NEWS_MED_SENTIMENT, NEWS_MIN_ABS_SENTIMENT


class TestSign:
    @pytest.mark.parametrize(
        "headline",
        [
            "Apple faces DOJ antitrust probe over App Store practices",
            "JPMorgan sued over trading practices",
            "Tesla recalls 200,000 vehicles after software fault",
            "Boeing CFO resigns amid accounting restatement",
            "NVIDIA misses expectations, cuts guidance",
            "Analysts downgrade the stock on demand weakness",
        ],
    )
    def test_bad_news_is_negative(self, headline):
        assert score_sentiment(headline) < 0

    @pytest.mark.parametrize(
        "headline",
        [
            "NVIDIA beats expectations, raises guidance",
            "Alphabet shares surge on record cloud revenue",
            "Microsoft wins regulatory approval for the acquisition",
            "Analysts upgrade the stock on strong demand",
        ],
    )
    def test_good_news_is_positive(self, headline):
        assert score_sentiment(headline) > 0

    @pytest.mark.parametrize(
        "headline",
        [
            "Microsoft announces quarterly dividend",
            "Apple to report results next Thursday",
            "The company will present at a conference",
        ],
    )
    def test_neutral_headlines_score_zero(self, headline):
        """
        Zero means NO SIGNAL, not "neutral news". The monitor's
        NEWS_MIN_ABS_SENTIMENT floor discards it either way.
        """
        assert score_sentiment(headline) == 0.0


class TestMagnitude:
    def test_the_range_is_bounded(self):
        piled_on = "fraud bankruptcy probe lawsuit recall plunge collapse indicted subpoena delisting"
        assert -1.0 <= score_sentiment(piled_on) <= 0.0

    def test_a_single_strong_term_clears_the_candidate_floor(self):
        """
        Below NEWS_MIN_ABS_SENTIMENT the news monitor emits nothing at all, so
        an unambiguous headline has to clear it or the monitor is dead again.
        """
        assert abs(score_sentiment("Company under federal investigation")) >= NEWS_MIN_ABS_SENTIMENT

    def test_one_strong_term_reaches_med_but_not_high(self):
        # HIGH needs corroboration too, so a single article capping at MED is
        # the intended shape rather than a limitation.
        score = score_sentiment("Company under federal investigation")
        assert score <= NEWS_MED_SENTIMENT

    def test_several_strong_terms_reach_high(self):
        score = score_sentiment("Auditor resigns amid fraud probe and restatement")
        assert score <= NEWS_HIGH_SENTIMENT

    def test_more_evidence_scores_more_strongly(self):
        mild = score_sentiment("Shares fell on weak guidance")
        severe = score_sentiment("Shares plunged as regulators opened a fraud investigation")
        assert severe < mild


class TestCounting:
    def test_terms_are_counted_as_a_set_not_by_occurrence(self):
        """
        A repeated word is emphasis, not evidence. Counting occurrences would
        let one long article dominate a corroboration count.
        """
        once = score_sentiment("Regulators opened a probe")
        thrice = score_sentiment("Regulators opened a probe; the probe follows an earlier probe")
        assert once == thrice

    def test_the_summary_is_weighted_below_the_headline(self):
        headline_only = score_sentiment("NVIDIA beats expectations")
        with_negative_body = score_sentiment("NVIDIA beats expectations", "Shares fell on weak guidance")

        # The body drags it down but must not flip a clear headline on its own.
        assert with_negative_body < headline_only
        assert with_negative_body > -1.0

    def test_a_long_boilerplate_summary_is_truncated(self):
        """
        Feed summaries end in disclaimers that are lexically negative and
        identical across every article from that source. Left unbounded they
        would give every article the same downward bias.
        """
        boilerplate = "x " * 400 + "this involves risk of loss and is not advice"
        assert score_sentiment("Microsoft announces quarterly dividend", boilerplate) == 0.0


class TestExplainability:
    def test_matched_terms_shows_what_drove_the_score(self):
        negative, positive = matched_terms("Shares plunged after the fraud probe, though revenue beat estimates")

        assert {"plunged", "fraud", "probe"} <= set(negative)
        assert "beat" in positive

    def test_no_matches_is_two_empty_lists(self):
        assert matched_terms("The company will present at a conference") == ([], [])


class TestIntegrationWithNews:
    def test_news_items_carry_a_score_rather_than_none(self):
        """
        Regression guard for the gap this module closed. Both providers had
        sentiment hard-coded to None, so the news monitor filtered on
        `sentiment is not None` and could never emit a candidate — 250 live
        articles, 0 scored, 0 candidates, indistinguishable from a quiet week.
        """
        import inspect

        from src.data import news

        source = inspect.getsource(news)
        assert "sentiment=None" not in source
        assert source.count("sentiment=score_sentiment(headline, summary)") == 2
