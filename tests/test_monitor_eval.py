# ═══════════════════════════════════════════════════════
# FinSight — Tests: Suite B Sweep Logic
# ═══════════════════════════════════════════════════════
#
# The sweep MATH is tested here with injected scores — fast, deterministic,
# no embedder. score_pairs() itself (real bge-small) is exercised by actually
# running `./shell_scripts/run_evals.sh alerts`, not by a unit test, the same split
# test_dedup.py / test_dedup_live.py already draws for the algorithm it
# feeds.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from evals.run_monitor_eval import (
    Confusion,
    GoldenText,
    ScoredPair,
    choose_threshold,
    confusion_at,
    load_golden,
    sweep,
)


def pair(same_event: bool, score: float, **overrides) -> ScoredPair:
    base = dict(ticker="AAPL", alert_type="NEWS_SENTIMENT", a_id="a", b_id="b", same_event=same_event, score=score)
    base.update(overrides)
    return ScoredPair(**base)


class TestConfusionAt:
    def test_a_true_duplicate_above_threshold_is_a_true_positive(self):
        result = confusion_at([pair(True, 0.95)], 0.89)
        assert (result.true_positive, result.false_positive, result.false_negative, result.true_negative) == (
            1,
            0,
            0,
            0,
        )

    def test_a_distinct_event_above_threshold_is_a_false_positive(self):
        result = confusion_at([pair(False, 0.95)], 0.89)
        assert result.false_positive == 1

    def test_a_true_duplicate_below_threshold_is_a_false_negative(self):
        result = confusion_at([pair(True, 0.5)], 0.89)
        assert result.false_negative == 1

    def test_a_distinct_event_below_threshold_is_a_true_negative(self):
        result = confusion_at([pair(False, 0.5)], 0.89)
        assert result.true_negative == 1

    def test_the_threshold_itself_counts_as_above(self):
        result = confusion_at([pair(True, 0.89)], 0.89)
        assert result.true_positive == 1


class TestPrecisionRecall:
    def test_precision_is_true_positives_over_everything_flagged(self):
        result = Confusion(threshold=0.89, true_positive=3, false_positive=1, false_negative=0, true_negative=0)
        assert result.precision == 0.75

    def test_precision_is_none_when_nothing_is_flagged(self):
        result = Confusion(threshold=0.89, true_positive=0, false_positive=0, false_negative=5, true_negative=10)
        assert result.precision is None

    def test_recall_is_true_positives_over_every_real_duplicate(self):
        result = Confusion(threshold=0.89, true_positive=3, false_positive=0, false_negative=1, true_negative=0)
        assert result.recall == 0.75

    def test_recall_is_none_when_there_are_no_true_duplicates_at_all(self):
        result = Confusion(threshold=0.89, true_positive=0, false_positive=2, false_negative=0, true_negative=10)
        assert result.recall is None


class TestSweep:
    def test_sweeps_every_step_in_the_range(self):
        rows = sweep([pair(True, 0.8)], low=0.70, high=0.90, step=0.10)
        assert [r.threshold for r in rows] == [0.70, 0.80, 0.90]

    def test_is_monotonically_ordered_ascending(self):
        rows = sweep([pair(True, 0.8), pair(False, 0.75)])
        assert [r.threshold for r in rows] == sorted(r.threshold for r in rows)


class TestChooseThreshold:
    def test_picks_the_lowest_threshold_clearing_the_precision_floor(self):
        rows = [
            Confusion(0.70, true_positive=10, false_positive=10, false_negative=0, true_negative=0),  # 0.50
            Confusion(0.80, true_positive=8, false_positive=1, false_negative=2, true_negative=9),  # 0.889
            Confusion(0.90, true_positive=2, false_positive=0, false_negative=8, true_negative=10),  # 1.0
        ]
        chosen = choose_threshold(rows, min_precision=0.85)
        assert chosen.threshold == 0.80

    def test_none_when_no_threshold_clears_the_floor(self):
        rows = [Confusion(0.70, true_positive=1, false_positive=9, false_negative=0, true_negative=0)]
        assert choose_threshold(rows, min_precision=0.97) is None

    def test_a_threshold_with_undefined_precision_is_skipped_not_crashed_on(self):
        rows = [
            Confusion(0.95, true_positive=0, false_positive=0, false_negative=5, true_negative=10),  # None
            Confusion(0.96, true_positive=1, false_positive=0, false_negative=4, true_negative=10),  # 1.0
        ]
        assert choose_threshold(rows, min_precision=0.9).threshold == 0.96


class TestLoadGolden:
    def test_the_committed_dataset_parses_and_is_well_formed(self):
        rows = load_golden()
        assert len(rows) > 0
        for row in rows:
            assert set(row) == set(GoldenText.__annotations__)

    def test_every_group_has_at_least_two_clusters(self):
        """
        A group (ticker, alert_type) with only ONE cluster can only ever
        produce true-positive pairs — no hard negatives — which would make
        precision trivially 1.0 at every threshold and hide exactly the
        failure mode this sweep exists to measure.
        """
        rows = load_golden()
        clusters_by_group: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            key = (row["ticker"], row["alert_type"])
            clusters_by_group.setdefault(key, set()).add(row["cluster"])

        for group, clusters in clusters_by_group.items():
            assert len(clusters) >= 2, f"{group} has only one cluster — no hard negatives"

    def test_every_cluster_has_at_least_two_texts(self):
        """A singleton cluster contributes zero same-event pairs — dead weight in the dataset."""
        rows = load_golden()
        counts: dict[tuple[str, str, str], int] = {}
        for row in rows:
            key = (row["ticker"], row["alert_type"], row["cluster"])
            counts[key] = counts.get(key, 0) + 1

        for key, count in counts.items():
            assert count >= 2, f"cluster {key} has only one text"
