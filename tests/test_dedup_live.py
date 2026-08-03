# ═══════════════════════════════════════════════════════
# FinSight — Tests: Dedup Against the Real Embedder
# ═══════════════════════════════════════════════════════
#
# tests/test_dedup.py injects the similarity score so each branch can be
# exercised exactly. That is the right way to test the ALGORITHM and it cannot
# tell you the one thing that actually decides whether the engine works:
#
#     do the calibrated thresholds sit in the right place for text the
#     canonicalizer really produces, embedded by the model really in use?
#
# These tests answer that. Real bge-small, real Qdrant, throwaway collection.
#
# ══ THIS FILE FOUND THE GUARDRAIL BUG ══
#   The plan specified "never suppress a HIGH alert below 0.96 similarity".
#   Run for the first time, three outlets covering one DOJ probe scored 0.898
#   and 0.913 against the first report — so the 0.96 floor did not protect
#   against a missed event, it guaranteed one page per outlet for every
#   HIGH-severity story.
#
#   Nothing in the mocked suite could have caught that, because the mocked
#   suite chooses the numbers. See src/vectorstore/config.py.
#
# Integration + slow, so `make test` skips them. Run with:
#   .venv/bin/python -m pytest tests/test_dedup_live.py -m "" -q
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.vectorstore.config import TAU_HIGH, TAU_LOW

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def candidate(ticker, company, alert_type, headline, key, metrics):
    return {
        "ticker": ticker,
        "company_name": company,
        "alert_type": alert_type,
        "monitor": "test",
        "headline": headline,
        "detail": headline,
        "natural_key": key,
        "metrics": metrics,
        "evidence": [],
        "observed_at": NOW.isoformat(),
    }


# Severity HIGH: sentiment past NEWS_HIGH_SENTIMENT with corroboration.
HIGH_NEWS = {"sentiment": -0.7, "source_count": 3}
MED_NEWS = {"sentiment": -0.35, "source_count": 1}


@pytest.fixture
def live_collection():
    """A throwaway alerts collection. Never touches finsight_alerts."""
    import src.monitor.alert_store as store
    from src.vectorstore.collections import drop_collection, ensure_collection
    from src.vectorstore.config import ALERTS_PAYLOAD_INDEXES

    name = f"finsight_alerts_test_{uuid.uuid4().hex[:8]}"
    original = store.ALERT_COLLECTION
    store.ALERT_COLLECTION = name
    ensure_collection(name, ALERTS_PAYLOAD_INDEXES)

    try:
        yield name
    finally:
        store.ALERT_COLLECTION = original
        drop_collection(name)


def run(cases, **kwargs):
    from src.monitor.dedup import deduplicate

    return deduplicate(
        [item for item, _ in cases],
        now=NOW,
        summaries=[summary for _, summary in cases],
        **kwargs,
    )


class TestRealSimilarityBands:
    def test_paraphrases_of_one_story_land_above_tau_high(self, live_collection):
        """
        THE measurement the thresholds rest on. If real paraphrases scored
        below TAU_HIGH, the engine would fire once per outlet and every claim
        made for it would be false.
        """
        cases = [
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "DOJ antitrust probe", "u1", MED_NEWS),
                "regulatory antitrust investigation into app store practices",
            ),
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "DOJ opens inquiry", "u2", MED_NEWS),
                "justice department inquiry into app store competition practices",
            ),
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "Federal scrutiny of rules", "u3", MED_NEWS),
                "federal scrutiny of app store rules and developer terms",
            ),
        ]
        outcomes = run(cases)

        assert outcomes[0].decision == "FIRE"
        for outcome in outcomes[1:]:
            assert outcome.score >= TAU_HIGH, f"paraphrase scored {outcome.score:.3f}, below TAU_HIGH {TAU_HIGH}"
            assert outcome.decision == "SUPPRESS_SEMANTIC"

    def test_a_different_event_on_the_same_ticker_stays_below_tau_low(self, live_collection):
        """
        The negative that matters. The payload filter has already constrained
        this to the same ticker AND type, so this is the hard case: two
        genuinely different events that share both.
        """
        cases = [
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "DOJ probe", "u1", MED_NEWS),
                "regulatory antitrust investigation into app store practices",
            ),
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "MacBook recall", "u2", MED_NEWS),
                "product recall of power adapters over a safety defect",
            ),
        ]
        outcomes = run(cases)

        assert outcomes[1].decision == "FIRE"
        assert outcomes[1].score < TAU_LOW, f"distinct events scored {outcomes[1].score:.3f}, above TAU_LOW {TAU_LOW}"

    def test_the_same_story_about_a_different_company_never_matches(self, live_collection):
        """
        Lexically near-identical, completely unrelated. This is answered by the
        payload filter, not by similarity — which is the whole reason ticker is
        a hard filter rather than an embedded term.
        """
        summary = "regulatory antitrust investigation into app store practices"
        cases = [
            (candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "Apple probe", "u1", MED_NEWS), summary),
            (candidate("MSFT", "Microsoft Corp", "NEWS_SENTIMENT", "Microsoft probe", "u2", MED_NEWS), summary),
        ]
        outcomes = run(cases)

        assert outcomes[1].decision == "FIRE"
        assert outcomes[1].score == 0.0  # filtered out entirely, never scored


class TestHighSeverityGuardrailAgainstRealScores:
    def test_three_high_severity_outlets_produce_exactly_one_page(self, live_collection):
        """
        ══ THE REGRESSION THIS FILE EXISTS FOR ══
        Under the planned 0.96 similarity floor all three FIRED, because real
        paraphrases score ~0.90. The rule is now about information — a HIGH
        candidate fires unless the matched parent is already a reported HIGH —
        so the reader is told once.
        """
        cases = [
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "DOJ antitrust probe", "u1", HIGH_NEWS),
                "regulatory antitrust investigation into app store practices",
            ),
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "DOJ opens inquiry", "u2", HIGH_NEWS),
                "justice department inquiry into app store competition practices",
            ),
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "Federal scrutiny", "u3", HIGH_NEWS),
                "federal scrutiny of app store rules and developer terms",
            ),
        ]
        outcomes = run(cases)

        assert [o.decision for o in outcomes] == ["FIRE", "SUPPRESS_SEMANTIC", "SUPPRESS_SEMANTIC"]
        assert outcomes[0].alert["severity"] == "HIGH"

    def test_an_escalation_still_reaches_the_reader(self, live_collection):
        """
        The guardrail must not silence a severity increase. A MED parent
        followed by a HIGH report of the same event has to be reported, because
        the reader was told a milder version.
        """
        cases = [
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "Regulator asks questions", "u1", MED_NEWS),
                "regulatory antitrust investigation into app store practices",
            ),
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "DOJ opens formal probe", "u2", HIGH_NEWS),
                "justice department inquiry into app store competition practices",
            ),
        ]
        outcomes = run(cases)

        assert outcomes[0].alert["severity"] == "MED"
        assert outcomes[1].decision in {"FIRE", "ESCALATE"}
        assert outcomes[1].alert is not None
        assert outcomes[1].alert["severity"] == "HIGH"


class TestExactPathAgainstRealQdrant:
    def test_the_same_event_twice_is_suppressed_by_id(self, live_collection):
        cases = [
            (
                candidate(
                    "AAPL", "Apple Inc.", "NEW_FILING", "8-K filed", "0000320193-26-000010", {"form_type": "8-K"}
                ),
                "8-k reporting other events",
            )
        ]
        assert run(cases)[0].decision == "FIRE"
        assert run(cases)[0].decision == "SUPPRESS_EXACT"

    def test_an_all_duplicate_cycle_makes_no_model_call(self, live_collection):
        from unittest.mock import patch

        from src.monitor import dedup as dedup_module

        cases = [
            (
                candidate("AAPL", "Apple Inc.", "NEW_FILING", "8-K filed", "acc-1", {"form_type": "8-K"}),
                "8-k reporting other events",
            )
        ]
        run(cases)

        with patch.object(dedup_module, "summarize", side_effect=AssertionError("summarised a duplicate")):
            from src.monitor.dedup import deduplicate

            outcomes = deduplicate([cases[0][0]], now=NOW)

        assert outcomes[0].decision == "SUPPRESS_EXACT"


class TestCentroidDrift:
    def test_a_merge_moves_the_stored_vector_and_keeps_it_normalised(self, live_collection):
        """
        Cosine normalises at INDEX time only, so an un-normalised running mean
        would shrink with every merge until the cluster stopped matching
        anything.
        """
        import math

        from src.monitor.alert_store import update_centroid
        from src.vectorstore.client import get_qdrant_client

        cases = [
            (
                candidate("AAPL", "Apple Inc.", "NEWS_SENTIMENT", "DOJ probe", "u1", MED_NEWS),
                "regulatory antitrust investigation into app store practices",
            )
        ]
        alert_id = run(cases)[0].alert["alert_id"]

        assert update_centroid(alert_id, [0.05] * 384)

        point = get_qdrant_client().retrieve(collection_name=live_collection, ids=[alert_id], with_vectors=True)[0]
        norm = math.sqrt(sum(value * value for value in point.vector))
        assert norm == pytest.approx(1.0, abs=1e-4)

    def test_updating_a_missing_point_is_reported_not_raised(self, live_collection):
        # A point can be pruned between the search and the update. That is a
        # race to log, not an error to abort a cycle with.
        from src.monitor.alert_store import update_centroid

        assert update_centroid("00000000-0000-0000-0000-000000000000", [0.1] * 384) is False
