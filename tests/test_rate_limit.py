# ═══════════════════════════════════════════════════════
# FinSight — Tests: Rate Limiting & Daily Budgets
# ═══════════════════════════════════════════════════════
# Offline. No network, no LLM.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from src.core.errors import BudgetExhausted
from src.data import rate_limit
from src.data.rate_limit import TokenBucket


class TestTokenBucket:
    """Pacing must be real — the SEC bans clients that burst past 10 req/s."""

    def test_burst_up_to_capacity_is_immediate(self):
        bucket = TokenBucket(rate=10.0, capacity=5.0)
        started = time.monotonic()
        for _ in range(5):
            bucket.acquire()
        assert time.monotonic() - started < 0.05

    def test_exceeding_capacity_blocks(self):
        bucket = TokenBucket(rate=20.0, capacity=1.0)
        bucket.acquire()  # drain
        waited = bucket.acquire()
        assert waited > 0
        assert waited == pytest.approx(1 / 20.0, abs=0.03)

    def test_tokens_refill_over_time(self):
        bucket = TokenBucket(rate=100.0, capacity=2.0)
        bucket.acquire()
        bucket.acquire()
        time.sleep(0.05)  # ~5 tokens regenerate, capped at 2
        assert bucket.acquire() == 0.0

    def test_is_thread_safe(self):
        # Parallel graph branches share one bucket; without the lock they would
        # race and collectively exceed the configured rate.
        bucket = TokenBucket(rate=500.0, capacity=50.0)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(10):
                    bucket.acquire()
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_default_capacity_is_one_second_of_rate(self):
        assert TokenBucket(rate=8.0).capacity == 8.0

    def test_capacity_floors_at_one_for_slow_rates(self):
        # 0.9 req/s must still permit a single call rather than deadlocking.
        assert TokenBucket(rate=0.9).capacity == 1.0


class TestBucketRegistry:
    """One shared bucket per provider, built from configured rates."""

    def setup_method(self):
        rate_limit.reset_buckets()

    def teardown_method(self):
        rate_limit.reset_buckets()

    def test_same_provider_shares_one_bucket(self):
        assert rate_limit.get_bucket("sec") is rate_limit.get_bucket("sec")

    def test_providers_get_separate_buckets(self):
        assert rate_limit.get_bucket("sec") is not rate_limit.get_bucket("fred")

    def test_sec_rate_stays_under_the_published_ceiling(self):
        # SEC publishes 10 req/s and blocks aggressive clients outright.
        assert rate_limit.get_bucket("sec").rate < 10.0


class TestDailyBudgets:
    """FMP gives 250 calls/day total. Stop at the soft cap, not the hard one."""

    def setup_method(self):
        rate_limit.reset_buckets()

    def teardown_method(self):
        rate_limit.reset_buckets()

    def test_unmetered_provider_never_raises(self):
        for _ in range(1000):
            rate_limit.check_budget("sec")

    def test_zero_budget_provider_is_disabled(self):
        # Alpha Vantage is configured to 0 — 25 req/day is unusable.
        with pytest.raises(BudgetExhausted, match="disabled"):
            rate_limit.check_budget("alphavantage")

    def test_budget_allows_calls_below_the_soft_limit(self):
        with patch.dict(rate_limit.DAILY_BUDGETS, {"fmp": 100}):
            for _ in range(79):
                rate_limit.record_call("fmp")
            rate_limit.check_budget("fmp")

    def test_budget_raises_at_the_soft_limit(self):
        with patch.dict(rate_limit.DAILY_BUDGETS, {"fmp": 100}):
            for _ in range(80):  # soft limit = 80% of 100
                rate_limit.record_call("fmp")
            with pytest.raises(BudgetExhausted, match="soft limit"):
                rate_limit.check_budget("fmp")

    def test_soft_limit_leaves_headroom_below_the_hard_cap(self):
        assert rate_limit.BUDGET_SOFT_LIMIT < 1.0

    def test_counter_resets_on_a_new_utc_day(self):
        from datetime import timedelta

        with patch.dict(rate_limit.DAILY_BUDGETS, {"fmp": 10}):
            rate_limit.record_call("fmp")
            # Backdate the stored day to simulate the rollover. This MUST be
            # relative to the module's own UTC date, not date.today(): in any
            # timezone ahead of UTC, between local midnight and UTC midnight,
            # "local yesterday" IS "UTC today" — so the counter would not reset
            # and the test would fail for 5.5 hours a day in UTC+5:30.
            rate_limit._BUDGET_STATE["fmp"] = (rate_limit._today() - timedelta(days=1), 999)
            rate_limit.check_budget("fmp")

    def test_status_snapshot_reports_every_metered_provider(self):
        snapshot = {row["provider"]: row for row in rate_limit.budget_status()}
        assert "fmp" in snapshot
        assert snapshot["alphavantage"]["exhausted"] is True

    def test_guard_checks_budget_before_pacing(self):
        # An exhausted provider should fail instantly, not sleep in the bucket
        # only to be rejected anyway.
        started = time.monotonic()
        with pytest.raises(BudgetExhausted):
            rate_limit.guard("alphavantage")
        assert time.monotonic() - started < 0.05
