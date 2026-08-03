# ═══════════════════════════════════════════════════════
# FinSight — Tests: Alert Deduplication
# ═══════════════════════════════════════════════════════
#
# The biggest test file in the project, because dedup is the piece whose
# failure is SILENT. A broken retriever returns nothing and you notice; a
# broken suppressor eats a real alert and looks exactly like a quiet week.
#
# ══ WHY A FAKE QDRANT AND A FAKE EMBEDDER ══
#   Both are substituted, but for opposite reasons.
#
#   The store is faked because these tests are about the ALGORITHM — which
#   branch a score lands in, what gets written, what gets counted. Running
#   them against a live Qdrant would make them slow, order-dependent, and
#   dependent on a container being up.
#
#   The embedder is faked because a test that asserts "these two texts score
#   0.94" is really asserting a property of bge-small, not of this code. The
#   thresholds themselves were calibrated against the real model and live in
#   vectorstore/config.py; here the score is an INPUT so each branch can be
#   exercised exactly.
#
#   What is NOT faked: the canonicalization, the severity rules, and every
#   decision boundary. Those are the parts that can be wrong.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.monitor import dedup as dedup_module
from src.monitor.dedup import Decision, decide, deduplicate
from src.monitor.synthesizer import alert_id_for, dedup_key
from src.vectorstore.config import TAU_HIGH, TAU_LOW

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def candidate(
    *,
    ticker="AAPL",
    alert_type="NEWS_SENTIMENT",
    natural_key="k1",
    headline="Apple faces a regulatory probe",
    metrics=None,
):
    return {
        "ticker": ticker,
        "company_name": "Apple Inc." if ticker else "",
        "alert_type": alert_type,
        "monitor": "test",
        "headline": headline,
        "detail": headline,
        "natural_key": natural_key,
        "metrics": metrics if metrics is not None else {"sentiment": -0.4, "source_count": 1},
        "evidence": [],
        "observed_at": NOW.isoformat(),
    }


class FakeStore:
    """
    A stand-in for the alert collection, recording every mutation.

    Deliberately keeps points in a plain dict keyed by id, because that is
    exactly the contract alert_store relies on: the id is derived from the
    dedup key, so an exact match is a dict lookup.
    """

    def __init__(self, *, next_score=0.0):
        self.points: dict[str, dict] = {}
        self.next_score = next_score
        self.bumps: list[tuple[str, int]] = []
        self.centroid_updates: list[str] = []
        self.upserts: list[str] = []

    # ── the four functions dedup.py imports ──
    def find_exact(self, key, *, alert_id):
        payload = self.points.get(alert_id)
        if payload and payload.get("dedup_key") == key:
            return payload
        return None

    def search_similar(self, vector, *, ticker, alert_type, now=None, **kwargs):
        matches = [
            payload
            for payload in self.points.values()
            if payload["ticker"] == ticker.upper() and payload["alert_type"] == alert_type
        ]
        if not matches or self.next_score < TAU_LOW:
            return []
        return [(self.next_score, matches[-1])]

    def upsert_alert(self, alert, vector):
        self.upserts.append(alert["alert_id"])
        self.points[alert["alert_id"]] = {
            "alert_id": alert["alert_id"],
            "ticker": alert["ticker"],
            "alert_type": alert["alert_type"],
            "severity": alert["severity"],
            "status": alert["status"],
            "dedup_key": alert["dedup_key"],
            "occurrence_count": alert["occurrence_count"],
            "canonical_text": alert["canonical_text"],
            "headline": alert["headline"],
        }

    def bump_occurrence(self, alert_id, *, count, last_seen_at):
        self.bumps.append((alert_id, count))
        if alert_id in self.points:
            self.points[alert_id]["occurrence_count"] = count

    def update_centroid(self, alert_id, vector):
        self.centroid_updates.append(alert_id)
        return True


@pytest.fixture
def store():
    """Patch the store functions WHERE THEY ARE USED, not where they are defined."""
    fake = FakeStore()

    class FakeEmbedder:
        # A constant vector: the score is injected through FakeStore, so the
        # embedding's content is irrelevant and a constant makes that obvious.
        def embed_symmetric(self, text):
            return [0.1] * 384

    with (
        patch.object(dedup_module, "find_exact", fake.find_exact),
        patch.object(dedup_module, "search_similar", fake.search_similar),
        patch.object(dedup_module, "upsert_alert", fake.upsert_alert),
        patch.object(dedup_module, "bump_occurrence", fake.bump_occurrence),
        patch.object(dedup_module, "update_centroid", fake.update_centroid),
        patch("src.vectorstore.embeddings.get_embedder", return_value=FakeEmbedder()),
    ):
        yield fake


SUMMARY = "regulatory probe into sales practices"


class TestExactKeyFastPath:
    def test_the_same_event_twice_is_suppressed_for_free(self, store):
        item = candidate()

        first = decide(item, summary=SUMMARY, now=NOW)
        second = decide(item, summary=SUMMARY, now=NOW)

        assert first.decision == Decision.FIRE
        assert second.decision == Decision.SUPPRESS_EXACT

    def test_the_exact_path_never_embeds(self, store):
        """
        The whole value of this path is that it costs nothing. If it ever
        reaches the embedder, ~90% of duplicates start costing an embedding.
        """
        item = candidate()
        decide(item, summary=SUMMARY, now=NOW)

        with patch("src.vectorstore.embeddings.get_embedder", side_effect=AssertionError("embedded on the fast path")):
            assert decide(item, summary=SUMMARY, now=NOW).decision == Decision.SUPPRESS_EXACT

    def test_suppression_increments_the_parent_rather_than_inserting(self, store):
        item = candidate()
        decide(item, summary=SUMMARY, now=NOW)
        decide(item, summary=SUMMARY, now=NOW)
        decide(item, summary=SUMMARY, now=NOW)

        assert len(store.points) == 1
        assert store.bumps == [(alert_id_for(dedup_key(item)), 2), (alert_id_for(dedup_key(item)), 3)]

    def test_a_different_natural_key_is_a_different_event(self, store):
        decide(candidate(natural_key="article-1"), summary=SUMMARY, now=NOW)
        outcome = decide(candidate(natural_key="article-2"), summary=SUMMARY, now=NOW)

        assert outcome.decision == Decision.FIRE


class TestCrossContamination:
    def test_the_same_event_on_two_tickers_does_not_collide(self, store):
        """
        The dedup key is (ticker, type, natural_key). Two companies filing on
        the same day must never suppress each other.
        """
        assert dedup_key(candidate(ticker="AAPL")) != dedup_key(candidate(ticker="MSFT"))

        decide(candidate(ticker="AAPL"), summary=SUMMARY, now=NOW)
        assert decide(candidate(ticker="MSFT"), summary=SUMMARY, now=NOW).decision == Decision.FIRE

    def test_the_same_key_under_two_alert_types_does_not_collide(self, store):
        a = candidate(alert_type="NEWS_SENTIMENT", natural_key="x")
        b = candidate(alert_type="PRICE_MOVE", natural_key="x", metrics={"change_pct_1d": -5.0})
        assert dedup_key(a) != dedup_key(b)

    def test_semantic_search_is_filtered_by_ticker(self, store):
        # A near-identical story about a different company scores high
        # lexically and must never be reachable, whatever the threshold says.
        store.next_score = 0.99
        decide(candidate(ticker="AAPL"), summary=SUMMARY, now=NOW)

        outcome = decide(candidate(ticker="MSFT", natural_key="k2"), summary=SUMMARY, now=NOW)
        assert outcome.decision == Decision.FIRE


class TestSemanticBands:
    def test_at_or_above_tau_high_suppresses(self, store):
        store.next_score = TAU_HIGH
        decide(candidate(natural_key="a"), summary=SUMMARY, now=NOW)

        outcome = decide(candidate(natural_key="b"), summary=SUMMARY, now=NOW)
        assert outcome.decision == Decision.SUPPRESS_SEMANTIC
        assert outcome.score == TAU_HIGH

    def test_just_below_tau_high_merges_rather_than_suppressing(self, store):
        store.next_score = TAU_HIGH - 0.01
        decide(candidate(natural_key="a"), summary=SUMMARY, now=NOW)

        assert decide(candidate(natural_key="b"), summary=SUMMARY, now=NOW).decision == Decision.MERGE

    def test_below_tau_low_fires(self, store):
        store.next_score = TAU_LOW - 0.01
        decide(candidate(natural_key="a"), summary=SUMMARY, now=NOW)

        assert decide(candidate(natural_key="b"), summary=SUMMARY, now=NOW).decision == Decision.FIRE

    def test_no_neighbours_at_all_fires(self, store):
        outcome = decide(candidate(), summary=SUMMARY, now=NOW)
        assert outcome.decision == Decision.FIRE
        assert outcome.score == 0.0

    def test_a_suppressed_candidate_reports_the_parent_it_collapsed_into(self, store):
        store.next_score = 0.95
        first = decide(candidate(natural_key="a"), summary=SUMMARY, now=NOW)

        outcome = decide(candidate(natural_key="b"), summary=SUMMARY, now=NOW)
        assert outcome.suppression is not None
        assert outcome.suppression["parent_alert_id"] == first.alert["alert_id"]
        assert outcome.suppression["score"] == 0.95


class TestMergeBand:
    def test_a_merge_moves_the_parent_centroid(self, store):
        """
        A cluster's meaning is better described by the average of its members
        than by whichever arrived first. Without this, cluster identity is an
        accident of arrival order and later paraphrases drift out of range.
        """
        store.next_score = (TAU_LOW + TAU_HIGH) / 2
        first = decide(candidate(natural_key="a"), summary=SUMMARY, now=NOW)
        decide(candidate(natural_key="b"), summary=SUMMARY, now=NOW)

        assert store.centroid_updates == [first.alert["alert_id"]]

    def test_a_merge_does_not_move_the_centroid_of_a_suppression(self, store):
        # A pure duplicate carries no new information, so averaging it in
        # would only pull the cluster toward whatever the feed repeated most.
        store.next_score = 0.97
        decide(candidate(natural_key="a"), summary=SUMMARY, now=NOW)
        decide(candidate(natural_key="b"), summary=SUMMARY, now=NOW)

        assert store.centroid_updates == []

    def test_a_merge_at_equal_severity_reports_nothing(self, store):
        store.next_score = (TAU_LOW + TAU_HIGH) / 2
        decide(candidate(natural_key="a"), summary=SUMMARY, now=NOW)

        outcome = decide(candidate(natural_key="b"), summary=SUMMARY, now=NOW)
        assert outcome.decision == Decision.MERGE
        assert outcome.alert is None

    def test_a_merge_that_raises_severity_escalates_and_is_reported(self, store):
        """
        A second outlet saying the same thing is corroboration. A second outlet
        revealing it is a criminal probe rather than a civil one is an
        escalation, and escalations must reach the reader.
        """
        store.next_score = (TAU_LOW + TAU_HIGH) / 2
        mild = candidate(natural_key="a", metrics={"sentiment": -0.35, "source_count": 1})
        severe = candidate(natural_key="b", metrics={"sentiment": -0.9, "source_count": 3})

        parent = decide(mild, summary=SUMMARY, now=NOW)
        outcome = decide(severe, summary=SUMMARY, now=NOW)

        assert parent.alert["severity"] == "MED"
        assert outcome.decision == Decision.ESCALATE
        assert outcome.alert is not None
        assert outcome.alert["severity"] == "HIGH"
        assert outcome.alert["parent_alert_id"] == parent.alert["alert_id"]

    def test_a_merge_that_lowers_severity_does_not_escalate(self, store):
        store.next_score = (TAU_LOW + TAU_HIGH) / 2
        severe = candidate(natural_key="a", metrics={"sentiment": -0.9, "source_count": 3})
        mild = candidate(natural_key="b", metrics={"sentiment": -0.35, "source_count": 1})

        decide(severe, summary=SUMMARY, now=NOW)
        assert decide(mild, summary=SUMMARY, now=NOW).decision == Decision.MERGE


class TestHighSeverityGuardrail:
    """
    ══ THE RULE IS ABOUT INFORMATION, NOT SIMILARITY ══
    The plan specified a similarity floor — never suppress a HIGH alert below
    0.96, whatever the thresholds say. The first live run of the semantic path
    refuted it: three outlets on one DOJ probe scored 0.898 and 0.913, so a
    0.96 floor guarantees one page per outlet for every HIGH story rather than
    protecting against anything.

    What it means to be safe is that the READER LEARNS about the HIGH event.
    If the matched parent was itself HIGH and fired, they already have.
    """

    def test_a_high_alert_matching_a_lower_severity_parent_fires(self, store):
        """
        The asymmetric-cost case. The reader was told a MED version; the event
        is worse than they know, so suppressing it would leave them wrong.
        """
        store.next_score = TAU_HIGH + 0.01  # would otherwise SUPPRESS

        routine = {"form_type": "10-K", "items": []}  # MED
        alarming = {"form_type": "8-K", "items": ["4.02"]}  # HIGH

        parent = decide(candidate(alert_type="NEW_FILING", natural_key="a", metrics=routine), summary=SUMMARY, now=NOW)
        assert parent.alert["severity"] == "MED"

        outcome = decide(
            candidate(alert_type="NEW_FILING", natural_key="b", metrics=alarming),
            summary=SUMMARY,
            now=NOW,
        )
        assert outcome.decision == Decision.FIRE
        assert "not been told about this at HIGH" in outcome.reason

    def test_a_high_alert_matching_a_high_parent_is_suppressed(self, store):
        """
        THE case the 0.96 floor got wrong. Three outlets on one story score
        0.90-ish against each other; the reader learned about it from the
        first, so the second and third are noise.
        """
        store.next_score = TAU_HIGH + 0.01
        alarming = {"form_type": "8-K", "items": ["4.02"]}

        decide(candidate(alert_type="NEW_FILING", natural_key="a", metrics=alarming), summary=SUMMARY, now=NOW)
        outcome = decide(
            candidate(alert_type="NEW_FILING", natural_key="b", metrics=alarming),
            summary=SUMMARY,
            now=NOW,
        )
        assert outcome.decision == Decision.SUPPRESS_SEMANTIC

    def test_three_high_severity_outlets_produce_one_page(self, store):
        # The end-to-end shape of the same rule, at the score real paraphrases
        # were measured at.
        store.next_score = 0.91
        alarming = {"form_type": "8-K", "items": ["4.02"]}
        candidates = [candidate(alert_type="NEW_FILING", natural_key=f"outlet-{i}", metrics=alarming) for i in range(3)]

        outcomes = deduplicate(candidates, now=NOW, summaries=[SUMMARY] * 3)
        assert [o.decision for o in outcomes] == [
            Decision.FIRE,
            Decision.SUPPRESS_SEMANTIC,
            Decision.SUPPRESS_SEMANTIC,
        ]

    def test_a_high_alert_in_the_merge_band_escalates_rather_than_firing_bare(self, store):
        # An escalating merge already reports AND links to its parent AND moves
        # the centroid; a bare guardrail FIRE would report the same thing while
        # discarding all three.
        store.next_score = (TAU_LOW + TAU_HIGH) / 2
        routine = {"form_type": "10-K", "items": []}
        alarming = {"form_type": "8-K", "items": ["4.02"]}

        decide(candidate(alert_type="NEW_FILING", natural_key="a", metrics=routine), summary=SUMMARY, now=NOW)
        outcome = decide(
            candidate(alert_type="NEW_FILING", natural_key="b", metrics=alarming),
            summary=SUMMARY,
            now=NOW,
        )
        assert outcome.decision == Decision.ESCALATE

    def test_the_guardrail_does_not_rescue_med_alerts(self, store):
        store.next_score = TAU_HIGH + 0.01
        routine = {"form_type": "10-K", "items": []}

        decide(candidate(alert_type="NEW_FILING", natural_key="a", metrics=routine), summary=SUMMARY, now=NOW)
        outcome = decide(
            candidate(alert_type="NEW_FILING", natural_key="b", metrics=routine),
            summary=SUMMARY,
            now=NOW,
        )
        assert outcome.decision == Decision.SUPPRESS_SEMANTIC

    def test_an_exact_duplicate_is_suppressed_even_at_high_severity(self, store):
        """
        The guardrail is about SEMANTIC uncertainty. The same accession number
        twice carries no uncertainty at all — firing it would page someone for
        a filing they have already been told about.
        """
        alarming = {"form_type": "8-K", "items": ["4.02"]}
        item = candidate(alert_type="NEW_FILING", natural_key="0000320193-26-000010", metrics=alarming)

        decide(item, summary=SUMMARY, now=NOW)
        assert decide(item, summary=SUMMARY, now=NOW).decision == Decision.SUPPRESS_EXACT


class TestWarmup:
    def test_warmup_indexes_but_reports_nothing(self, store):
        outcome = decide(candidate(), summary=SUMMARY, warmup=True, now=NOW)

        assert outcome.decision == Decision.WARMUP
        assert outcome.alert is None
        assert len(store.points) == 1  # indexed anyway

    def test_a_warmup_point_is_matchable_next_cycle(self, store):
        """
        The entire purpose of a warmup. A status the dedup filter excluded
        would produce a system that looks like it has a cold-start guard and
        does not.
        """
        decide(candidate(natural_key="a"), summary=SUMMARY, warmup=True, now=NOW)

        store.next_score = 0.95
        assert decide(candidate(natural_key="b"), summary=SUMMARY, now=NOW).decision == Decision.SUPPRESS_SEMANTIC

    def test_warmup_still_suppresses_exact_duplicates_within_itself(self, store):
        item = candidate()
        decide(item, summary=SUMMARY, warmup=True, now=NOW)
        assert decide(item, summary=SUMMARY, warmup=True, now=NOW).decision == Decision.SUPPRESS_EXACT


class TestCycleOrdering:
    def test_three_outlets_on_one_story_collapse_to_one_alert(self, store):
        """
        THE case the engine exists for. It only works because decisions are
        sequential: the first must be INDEXED before the second is SEARCHED.
        Parallelising would race all three against an empty index.
        """
        store.next_score = 0.95
        candidates = [candidate(natural_key=f"outlet-{i}") for i in range(3)]

        outcomes = deduplicate(candidates, now=NOW, summaries=[SUMMARY] * 3)

        assert [o.decision for o in outcomes] == [
            Decision.FIRE,
            Decision.SUPPRESS_SEMANTIC,
            Decision.SUPPRESS_SEMANTIC,
        ]

    def test_an_empty_cycle_costs_nothing(self, store):
        with patch.object(dedup_module, "summarize", side_effect=AssertionError("summarised an empty batch")):
            assert deduplicate([], now=NOW) == []

    def test_summaries_are_aligned_by_position(self, store):
        candidates = [candidate(ticker="AAPL", natural_key="a"), candidate(ticker="MSFT", natural_key="b")]
        outcomes = deduplicate(candidates, now=NOW, summaries=["apple probe", "microsoft outage"])

        assert "apple probe" in outcomes[0].alert["canonical_text"]
        assert "microsoft outage" in outcomes[1].alert["canonical_text"]

    def test_a_missing_summary_does_not_shift_the_others(self, store):
        # Short lists are padded at the END, so position 0 keeps its summary.
        outcomes = deduplicate(
            [candidate(natural_key="a"), candidate(natural_key="b")],
            now=NOW,
            summaries=["only one"],
        )
        assert "only one" in outcomes[0].alert["canonical_text"]


class TestDecisionLog:
    def test_every_candidate_produces_a_record_including_the_fires(self, store):
        """
        Phase 7's threshold sweep needs negatives. A log of only suppressions
        can justify the threshold that produced it and nothing else.
        """
        store.next_score = 0.95
        outcomes = deduplicate(
            [candidate(natural_key="a"), candidate(natural_key="b")],
            now=NOW,
            summaries=[SUMMARY] * 2,
        )

        assert len(outcomes) == 2
        assert all(o.record for o in outcomes)
        assert {o.record["decision"] for o in outcomes} == {Decision.FIRE, Decision.SUPPRESS_SEMANTIC}

    def test_a_suppression_record_carries_both_texts_and_the_score(self, store):
        store.next_score = 0.94
        outcomes = deduplicate(
            [candidate(natural_key="a"), candidate(natural_key="b")],
            now=NOW,
            summaries=["first description", "second description"],
        )

        record = outcomes[1].record
        assert record["score"] == 0.94
        assert "second description" in record["candidate_text"]
        assert "first description" in record["parent_text"]
        assert record["parent_alert_id"] == outcomes[0].alert["alert_id"]


class TestTimeWindows:
    def test_the_window_is_per_alert_type(self):
        """
        The right dedup horizon is a property of the event's natural frequency:
        a 5% drop tomorrow IS a new event; a 10-K filed today is the same 10-K
        in three weeks.
        """
        from src.vectorstore.config import DEDUP_WINDOW_SECONDS

        assert DEDUP_WINDOW_SECONDS["PRICE_MOVE"] < DEDUP_WINDOW_SECONDS["NEWS_SENTIMENT"]
        assert DEDUP_WINDOW_SECONDS["NEWS_SENTIMENT"] < DEDUP_WINDOW_SECONDS["MACRO_EVENT"]
        assert DEDUP_WINDOW_SECONDS["MACRO_EVENT"] < DEDUP_WINDOW_SECONDS["NEW_FILING"]

    def test_the_window_floor_is_passed_to_the_search(self):
        """
        The filter is what stops a search matching an alert from months ago.
        Asserted against the real search_similar rather than the fake, because
        this is a claim about the Qdrant call, not about the algorithm.
        """
        from src.monitor.alert_store import search_similar

        captured = {}

        class FakeResponse:
            points: list = []

        class FakeClient:
            def query_points(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        with patch("src.vectorstore.client.get_qdrant_client", return_value=FakeClient()):
            search_similar([0.1] * 384, ticker="AAPL", alert_type="PRICE_MOVE", now=NOW)

        conditions = captured["query_filter"].must
        range_condition = next(c for c in conditions if getattr(c, "range", None))
        expected = int((NOW - timedelta(hours=24)).timestamp())
        assert range_condition.range.gte == expected


class TestPointIdentity:
    def test_the_alert_id_is_derived_from_the_dedup_key(self):
        """
        This is what makes the exact-match path an O(1) retrieve instead of a
        filtered scroll, and what makes re-upsert idempotent.
        """
        key = dedup_key(candidate())
        assert alert_id_for(key) == alert_id_for(key)
        assert alert_id_for(key) != alert_id_for(dedup_key(candidate(natural_key="other")))

    def test_the_alert_id_is_a_valid_uuid(self):
        import uuid

        uuid.UUID(alert_id_for(dedup_key(candidate())))  # raises if malformed


class TestTwoPassBatching:
    def test_an_all_duplicate_cycle_never_calls_the_model(self, store):
        """
        Measured on a live run: 5 of 5 candidates resolved on the exact path,
        and the batched summary call had already been made for all five.
        Cheap per cycle, and permanently wrong about what the fast path costs.
        """
        candidates = [candidate(natural_key=f"k{i}") for i in range(3)]
        deduplicate(candidates, now=NOW, summaries=[SUMMARY] * 3)

        with patch.object(dedup_module, "summarize", side_effect=AssertionError("summarised a duplicate")):
            outcomes = deduplicate(candidates, now=NOW)

        assert [o.decision for o in outcomes] == [Decision.SUPPRESS_EXACT] * 3

    def test_only_the_survivors_are_summarised(self, store):
        deduplicate([candidate(natural_key="seen")], now=NOW, summaries=[SUMMARY])

        summarised: list[int] = []

        def record(items, **kwargs):
            summarised.append(len(items))
            return [SUMMARY] * len(items)

        with patch.object(dedup_module, "summarize", side_effect=record):
            deduplicate(
                [candidate(natural_key="seen"), candidate(natural_key="new")],
                now=NOW,
            )

        assert summarised == [1]

    def test_outcomes_stay_in_input_order_across_both_passes(self, store):
        deduplicate([candidate(natural_key="seen")], now=NOW, summaries=[SUMMARY])

        outcomes = deduplicate(
            [
                candidate(natural_key="new-a"),
                candidate(natural_key="seen"),
                candidate(natural_key="new-b"),
            ],
            now=NOW,
            summaries=["a summary", "unused", "b summary"],
        )

        assert [o.decision for o in outcomes] == [Decision.FIRE, Decision.SUPPRESS_EXACT, Decision.FIRE]
        assert "a summary" in outcomes[0].alert["canonical_text"]
        assert "b summary" in outcomes[2].alert["canonical_text"]

    def test_supplied_summaries_stay_aligned_with_the_original_positions(self, store):
        """
        Regression guard: pass 2 iterates survivors, so a supplied summary list
        has to be indexed by ORIGINAL position, not by survivor position.
        """
        deduplicate([candidate(ticker="AAPL", natural_key="seen")], now=NOW, summaries=["parent"])

        outcomes = deduplicate(
            [
                candidate(ticker="AAPL", natural_key="seen"),
                candidate(ticker="MSFT", natural_key="fresh"),
            ],
            now=NOW,
            summaries=["belongs to the duplicate", "belongs to the new one"],
        )

        assert "belongs to the new one" in outcomes[1].alert["canonical_text"]
