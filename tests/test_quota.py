# ═══════════════════════════════════════════════════════
# FinSight — Tests: Free-tier metering
# ═══════════════════════════════════════════════════════
#
# Every test runs against a temporary SQLite file and a stub graph, so nothing
# touches data/finsight.db and nothing spends an LLM token.
#
# ══ THE TEST THAT MATTERS MOST ══
#   test_concurrent_spends_never_exceed_the_limit. The whole feature is one
#   increment, and a read-modify-write increment loses races — which is a free
#   query each time it does. That test is why consume_free_query is a single
#   conditional UPDATE rather than the select-mutate-commit pattern used for
#   API budgets, and it is the one that fails if anyone "simplifies" it back.
#
# ══ THE SECOND MOST IMPORTANT ══
#   test_a_stranger_is_refused_with_402_on_the_stream_route. The status line
#   of a StreamingResponse is fixed the moment the response is constructed, so
#   a charge that happens inside the generator can only report failure as an
#   SSE error frame — a 200 that looks exactly like a graph crash. Nothing
#   else in the suite would notice that regression.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.auth import ANONYMOUS_USER, Identity, require_identity
from src.core import config as core_config
from src.persistence import db as db_module
from src.persistence.db import init_db, reset_engine, session_scope
from src.persistence.models import FreeQueryQuota, ResearchRun
from src.persistence.repository import (
    consume_free_query,
    get_free_quota,
    refund_free_query,
)

SUBJECT = "google-sub-112233"
FREE = Identity(subject=SUBJECT, email="stranger@example.com", unlimited=False)
PAID = Identity(subject="owner-sub", email="owner@example.com", unlimited=True)


@pytest.fixture
def temp_db(tmp_path: Any) -> Iterator[str]:
    """Point the engine at a throwaway database for one test."""
    url = f"sqlite:///{tmp_path / 'quota.db'}"
    reset_engine()
    with patch.object(db_module, "DATABASE_URL", url):
        init_db()
        yield url
    reset_engine()


def _stored_used(subject: str = SUBJECT) -> int | None:
    """Read the counter straight from the table, bypassing the repository."""
    with session_scope() as session:
        row = session.query(FreeQueryQuota).filter_by(subject=subject).one_or_none()
        return None if row is None else row.used


# ═══════════════════════════════════════════════════════
# The repository: spending, refusing, refunding
# ═══════════════════════════════════════════════════════
class TestConsume:
    def test_the_first_spend_creates_the_row(self, temp_db: str) -> None:
        """
        Signing in does not create a row; the first successful spend does.

        The row is inserted with used=0 and then incremented by the same
        conditional UPDATE as every later spend, so the insert path cannot
        itself hand out a query.
        """
        status = consume_free_query(SUBJECT, limit=5, email="stranger@example.com")

        assert status is not None
        assert status["used"] == 1
        assert status["remaining"] == 4
        assert status["exhausted"] is False
        assert _stored_used() == 1

    def test_spending_the_whole_allowance_then_refusing(self, temp_db: str) -> None:
        for expected in range(1, 6):
            status = consume_free_query(SUBJECT, limit=5)
            assert status is not None and status["used"] == expected

        assert consume_free_query(SUBJECT, limit=5) is None
        # The refusal must not have incremented anything.
        assert _stored_used() == 5

    def test_a_zero_limit_refuses_and_writes_nothing(self, temp_db: str) -> None:
        """FREE_QUERY_LIMIT=0 restores the old deployment: only the unlimited tier queries."""
        assert consume_free_query(SUBJECT, limit=0) is None
        assert _stored_used() is None

    def test_the_email_is_a_label_and_never_the_key(self, temp_db: str) -> None:
        """
        An account that changes its address keeps one row and one counter.

        Keying by email would reset the lifetime allowance for the price of an
        email change, which is the whole reason the key is Google's `sub`.
        """
        consume_free_query(SUBJECT, limit=5, email="before@example.com")
        consume_free_query(SUBJECT, limit=5, email="after@example.com")

        assert _stored_used() == 2
        assert get_free_quota(SUBJECT, limit=5)["email"] == "after@example.com"

    def test_raising_the_limit_later_grants_the_difference(self, temp_db: str) -> None:
        """The allowance is configuration, not a frozen column — that is why there is no limit field."""
        for _ in range(5):
            consume_free_query(SUBJECT, limit=5)
        assert consume_free_query(SUBJECT, limit=5) is None

        status = consume_free_query(SUBJECT, limit=8)
        assert status is not None and status["used"] == 6

    def test_lowering_the_limit_below_what_was_spent_is_not_a_debt(self, temp_db: str) -> None:
        for _ in range(5):
            consume_free_query(SUBJECT, limit=5)

        status = get_free_quota(SUBJECT, limit=2)
        assert status["used"] == 5
        assert status["remaining"] == 0  # floored, never negative
        assert status["exhausted"] is True


class TestConcurrency:
    def test_concurrent_spends_never_exceed_the_limit(self, temp_db: str) -> None:
        """
        Twelve simultaneous requests against an allowance of five yield five.

        A select-then-increment implementation loses increments under this
        load, and every lost increment is a free query. The conditional UPDATE
        makes the database the arbiter: `WHERE used < limit` is evaluated and
        applied in one statement, so exactly five callers can win.
        """
        # Seed the row first, so this measures the increment race specifically
        # rather than the insert race (which the next test covers).
        consume_free_query(SUBJECT, limit=5)
        refund_free_query(SUBJECT)

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(lambda _: consume_free_query(SUBJECT, limit=5), range(12)))

        granted = [r for r in results if r is not None]
        assert len(granted) == 5
        assert _stored_used() == 5
        # Each winner saw a distinct count — no two callers were told "you are #3".
        assert sorted(r["used"] for r in granted) == [1, 2, 3, 4, 5]

    def test_a_concurrent_first_spend_creates_exactly_one_row(self, temp_db: str) -> None:
        """
        The unique index on `subject` turns the insert race into an
        IntegrityError that consume_free_query retries, rather than into two
        rows and two allowances for one account.
        """
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(lambda _: consume_free_query(SUBJECT, limit=5), range(12)))

        with session_scope() as session:
            assert session.query(FreeQueryQuota).filter_by(subject=SUBJECT).count() == 1

        assert len([r for r in results if r is not None]) == 5


class TestRefund:
    def test_a_refund_returns_one_unit(self, temp_db: str) -> None:
        consume_free_query(SUBJECT, limit=5)
        consume_free_query(SUBJECT, limit=5)
        refund_free_query(SUBJECT)

        assert _stored_used() == 1

    def test_a_refund_cannot_mint_credit(self, temp_db: str) -> None:
        """The `used > 0` floor is a correctness rule, not an optimisation."""
        consume_free_query(SUBJECT, limit=5)
        refund_free_query(SUBJECT)
        refund_free_query(SUBJECT)
        refund_free_query(SUBJECT)

        assert _stored_used() == 0

    def test_refunding_an_unknown_account_is_a_no_op(self, temp_db: str) -> None:
        refund_free_query("never-seen")
        assert _stored_used("never-seen") is None


class TestRead:
    def test_an_account_with_no_row_reads_as_unspent(self, temp_db: str) -> None:
        """Not an error: the row appears on first spend, not on first sign-in."""
        status = get_free_quota("brand-new", limit=5)

        assert status == {
            "subject": "brand-new",
            "email": "",
            "used": 0,
            "limit": 5,
            "remaining": 5,
            "exhausted": False,
        }

    def test_reading_spends_nothing(self, temp_db: str) -> None:
        get_free_quota(SUBJECT, limit=5)
        get_free_quota(SUBJECT, limit=5)
        assert _stored_used() is None


# ═══════════════════════════════════════════════════════
# The charge layer: who is metered at all
# ═══════════════════════════════════════════════════════
class TestIsMetered:
    def test_the_unlimited_tier_is_never_metered(self) -> None:
        from src.api.quota import is_metered

        with patch.object(core_config, "AUTH_ENABLED", True):
            assert is_metered(PAID) is False
            assert is_metered(FREE) is True

    def test_auth_off_is_never_metered(self) -> None:
        """
        Load-bearing. The CLIs, the eval harness and this entire suite run
        headless with auth off; a meter that engaged for them would write
        quota rows during pytest and cap the eval harness at five queries.
        """
        from src.api.quota import is_metered

        anonymous = Identity(subject=ANONYMOUS_USER, email=ANONYMOUS_USER, unlimited=True)
        with patch.object(core_config, "AUTH_ENABLED", False):
            assert is_metered(anonymous) is False
            # Even an identity that claims to be metered is not, with auth off.
            assert is_metered(FREE) is False


# ═══════════════════════════════════════════════════════
# The HTTP surface
# ═══════════════════════════════════════════════════════
class _StubGraph:
    """A graph that answers instantly, or raises on demand."""

    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.calls = 0

    def _state(self, query: str, thread_id: str) -> dict[str, Any]:
        return {
            "query": query,
            "final_answer": "Revenue was $416.2B [SRC:EDGAR:0000320193-25-000079]",
            "citations": [],
            "conflicts": [],
            "findings": [],
            "tool_calls": [],
            "errors": [],
            "repair_count": 0,
            "plan": {"selected_agents": ["fundamentals"], "tickers": ["AAPL"]},
            "verification": {
                "verified_claims": [],
                "unsupported_claims": [],
                "invalid_source_ids": [],
                "citation_coverage": 1.0,
                "passed": True,
            },
        }

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.fails:
            raise RuntimeError("the graph exploded")
        return self._state(state["query"], config["configurable"]["thread_id"])

    async def astream(self, state: dict[str, Any], config: dict[str, Any], stream_mode: Any = None):
        self.calls += 1
        if self.fails:
            raise RuntimeError("the graph exploded")
        yield "values", self._state(state["query"], config["configurable"]["thread_id"])


@pytest.fixture
def metered_client(tmp_path: Any) -> Iterator[TestClient]:
    """
    An app running as a free-tier stranger, with a five-query allowance.

    AUTH_ENABLED is patched on because is_metered checks it independently of
    the Identity — the bypass is deliberately belt-and-braces, so a test that
    left it off would meter nothing and pass vacuously.
    """
    import src.api.main as main_module

    graph = _StubGraph()

    @asynccontextmanager
    async def fake_checkpointer() -> Any:
        yield object()

    db_module.reset_engine()
    with (
        patch.object(db_module, "DATABASE_URL", f"sqlite:///{tmp_path / 'quota_api.db'}"),
        patch.object(core_config, "AUTH_ENABLED", True),
        # The lifespan runs validate_auth_config, which is fatal without one.
        patch.object(core_config, "GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com"),
        patch.object(core_config, "AUTH_ALLOWED_EMAILS", frozenset({PAID.email})),
        patch.object(core_config, "FREE_QUERY_LIMIT", 5),
        patch.object(core_config, "CONTACT_URL", "https://example.test/contact"),
        patch.object(main_module, "async_checkpointer", fake_checkpointer),
        patch.object(main_module, "build_research_graph", lambda checkpointer=None: graph),
        patch.object(main_module, "_connect_qdrant", lambda: (None, "unavailable")),
        patch("src.vectorstore.collections.ensure_collections", lambda: {}),
    ):
        app = main_module.create_app()
        app.dependency_overrides[require_identity] = lambda: FREE

        with TestClient(app) as client:
            client.graph = graph  # type: ignore[attr-defined]
            yield client

    db_module.reset_engine()


def _ask(client: TestClient) -> Any:
    return client.post("/research/query", json={"query": "What was Apple's revenue?"})


class TestMeteredRoutes:
    def test_five_queries_then_a_402(self, metered_client: TestClient) -> None:
        for _ in range(5):
            assert _ask(metered_client).status_code == 200

        response = _ask(metered_client)
        assert response.status_code == 402

    def test_the_402_body_carries_what_the_ui_needs(self, metered_client: TestClient) -> None:
        for _ in range(5):
            _ask(metered_client)

        detail = _ask(metered_client).json()["detail"]
        # Machine-readable on purpose: the browser renders a CTA from this, so
        # it must not have to pattern-match prose to decide to show it.
        assert detail["error"] == "free_quota_exhausted"
        assert detail["used"] == 5
        assert detail["limit"] == 5
        assert detail["remaining"] == 0
        assert detail["contact_url"] == "https://example.test/contact"

    def test_a_refused_query_never_reaches_the_graph(self, metered_client: TestClient) -> None:
        """The point of the meter is not spending the tokens, not reporting afterwards."""
        for _ in range(5):
            _ask(metered_client)
        before = metered_client.graph.calls  # type: ignore[attr-defined]

        _ask(metered_client)
        assert metered_client.graph.calls == before  # type: ignore[attr-defined]

    def test_a_stranger_is_refused_with_402_on_the_stream_route(self, metered_client: TestClient) -> None:
        """
        A real 402, not a 200 carrying an error frame.

        StreamingResponse fixes the status line when it is constructed, so a
        charge inside the generator could only report failure as SSE — which
        every status-code check in every client would read as success.
        """
        for _ in range(5):
            assert (
                metered_client.post("/research/query/stream", json={"query": "What was Apple's revenue?"}).status_code
                == 200
            )

        response = metered_client.post("/research/query/stream", json={"query": "One too many"})
        assert response.status_code == 402
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["detail"]["error"] == "free_quota_exhausted"

    def test_an_invalid_body_is_not_charged(self, metered_client: TestClient) -> None:
        """
        422 before the meter.

        FastAPI resolves dependencies before validating the body, so charging
        in a Depends would bill a query that never ran. This is why
        charge_query is called in the handler instead.
        """
        for _ in range(10):
            assert metered_client.post("/research/query", json={"query": "no"}).status_code == 422

        assert metered_client.get("/auth/quota").json()["used"] == 0

    def test_only_the_query_routes_are_metered(self, metered_client: TestClient) -> None:
        """A spent account keeps the whole dashboard; it just cannot ask anything new."""
        for _ in range(5):
            _ask(metered_client)
        assert _ask(metered_client).status_code == 402

        assert metered_client.get("/research/runs").status_code == 200
        assert metered_client.get("/watchlist").status_code == 200


class TestThreadHijack:
    """
    Refusing another account's thread must not cost the caller a query.

    ══ THIS IS THE TEST THAT PINS THE ORDERING ══
      _resolve_thread and charge_query sit next to each other in both query
      handlers, and only one order is correct. Swap them and the graph still
      never runs, the 404 is still returned, and every other test in the suite
      still passes — but a free account has silently been billed for a request
      that was refused. Nothing else would notice.
    """

    def _plant_a_foreign_run(self) -> None:
        with session_scope() as session:
            session.add(
                ResearchRun(
                    thread_id="research:victim",
                    query="somebody else's question",
                    subject="another-account",
                )
            )

    def test_a_hijacked_thread_is_refused_without_spending_a_query(self, metered_client: TestClient) -> None:
        self._plant_a_foreign_run()

        response = metered_client.post(
            "/research/query", json={"query": "give me that", "thread_id": "research:victim"}
        )

        assert response.status_code == 404
        assert metered_client.get("/auth/quota").json()["used"] == 0

    def test_the_same_holds_on_the_stream_route(self, metered_client: TestClient) -> None:
        self._plant_a_foreign_run()

        response = metered_client.post(
            "/research/query/stream", json={"query": "give me that", "thread_id": "research:victim"}
        )

        # A real 404 status line, not a 200 carrying an SSE error frame.
        assert response.status_code == 404
        assert metered_client.get("/auth/quota").json()["used"] == 0

    def test_the_victims_row_is_untouched(self, metered_client: TestClient) -> None:
        self._plant_a_foreign_run()
        metered_client.post("/research/query", json={"query": "give me that", "thread_id": "research:victim"})

        with session_scope() as session:
            row = session.query(ResearchRun).filter_by(thread_id="research:victim").one()
            assert row.subject == "another-account"
            assert row.query == "somebody else's question"

    def test_a_free_account_cannot_replay_a_thread_it_does_not_own(self, metered_client: TestClient) -> None:
        self._plant_a_foreign_run()
        assert metered_client.get("/research/threads/research:victim").status_code == 404

    def test_an_own_thread_is_still_reusable(self, metered_client: TestClient) -> None:
        """The rule refuses other people's threads, not the caller's own."""
        first = _ask(metered_client)
        thread_id = first.json()["thread_id"]

        again = metered_client.post("/research/query", json={"query": "a follow-up", "thread_id": thread_id})

        assert again.status_code == 200
        assert metered_client.get("/auth/quota").json()["used"] == 2


class TestRefundOnFailure:
    def test_a_failed_run_is_refunded(self, metered_client: TestClient) -> None:
        """A five-query trial must not be spent on our own crash."""
        assert _ask(metered_client).status_code == 200
        assert metered_client.get("/auth/quota").json()["used"] == 1

        metered_client.graph.fails = True  # type: ignore[attr-defined]
        assert _ask(metered_client).status_code == 500

        assert metered_client.get("/auth/quota").json()["used"] == 1

    def test_a_failed_stream_is_refunded_and_still_reports_the_error(self, metered_client: TestClient) -> None:
        metered_client.graph.fails = True  # type: ignore[attr-defined]

        response = metered_client.post("/research/query/stream", json={"query": "What was Apple's revenue?"})

        assert response.status_code == 200  # the stream opened before the failure
        assert "event: error" in response.text
        assert metered_client.get("/auth/quota").json()["used"] == 0


class TestQuotaRoute:
    def test_it_reports_the_standing_without_spending(self, metered_client: TestClient) -> None:
        _ask(metered_client)

        body = metered_client.get("/auth/quota").json()
        assert body == {
            "metered": True,
            "used": 1,
            "limit": 5,
            "remaining": 4,
            "contact_url": "https://example.test/contact",
        }
        # Reading it twice must not move the counter.
        assert metered_client.get("/auth/quota").json()["used"] == 1

    def test_it_requires_a_token(self, tmp_path: Any) -> None:
        """
        /auth/quota is authenticated even though /auth/config is not.

        auth_router is included without the app-wide guard, so this route's
        dependency lives in its own signature — easy to lose, hence the test.
        """
        import src.api.main as main_module

        @asynccontextmanager
        async def fake_checkpointer() -> Any:
            yield object()

        db_module.reset_engine()
        with (
            patch.object(db_module, "DATABASE_URL", f"sqlite:///{tmp_path / 'anon.db'}"),
            patch.object(core_config, "AUTH_ENABLED", True),
            patch.object(core_config, "GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com"),
            patch.object(main_module, "async_checkpointer", fake_checkpointer),
            patch.object(main_module, "build_research_graph", lambda checkpointer=None: _StubGraph()),
            patch.object(main_module, "_connect_qdrant", lambda: (None, "unavailable")),
            patch("src.vectorstore.collections.ensure_collections", lambda: {}),
        ):
            with TestClient(main_module.create_app()) as client:
                assert client.get("/auth/quota").status_code == 401
                assert client.get("/auth/config").status_code == 200
        db_module.reset_engine()


class TestUnlimitedTier:
    @pytest.fixture
    def paid_client(self, tmp_path: Any) -> Iterator[TestClient]:
        import src.api.main as main_module

        @asynccontextmanager
        async def fake_checkpointer() -> Any:
            yield object()

        db_module.reset_engine()
        with (
            patch.object(db_module, "DATABASE_URL", f"sqlite:///{tmp_path / 'paid.db'}"),
            patch.object(core_config, "AUTH_ENABLED", True),
            patch.object(core_config, "GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com"),
            patch.object(core_config, "AUTH_ALLOWED_EMAILS", frozenset({PAID.email})),
            patch.object(core_config, "FREE_QUERY_LIMIT", 5),
            patch.object(main_module, "async_checkpointer", fake_checkpointer),
            patch.object(main_module, "build_research_graph", lambda checkpointer=None: _StubGraph()),
            patch.object(main_module, "_connect_qdrant", lambda: (None, "unavailable")),
            patch("src.vectorstore.collections.ensure_collections", lambda: {}),
        ):
            app = main_module.create_app()
            app.dependency_overrides[require_identity] = lambda: PAID
            with TestClient(app) as client:
                yield client
        db_module.reset_engine()

    def test_it_runs_past_the_limit_and_writes_no_row(self, paid_client: TestClient) -> None:
        for _ in range(10):
            assert _ask(paid_client).status_code == 200

        assert paid_client.get("/auth/quota").json()["metered"] is False
        with session_scope() as session:
            assert session.query(FreeQueryQuota).count() == 0

    def test_the_reserved_routes_are_reachable(self, paid_client: TestClient) -> None:
        assert paid_client.get("/admin/config").status_code == 200
        assert paid_client.get("/monitor/alerts").status_code == 200


class TestFreeTierIsRefusedTheReservedRoutes:
    def test_monitor_and_admin_are_403(self, metered_client: TestClient) -> None:
        """
        Opening sign-in to the world opened every route behind require_user.
        A monitoring cycle spends unbounded LLM tokens and is NOT metered;
        resuming one dispatches an alert; /admin/config publishes the
        deployment's configuration. None of that belongs to a free trial.
        """
        assert metered_client.get("/admin/config").status_code == 403
        assert metered_client.get("/admin/budgets").status_code == 403
        assert metered_client.get("/monitor/alerts").status_code == 403
        assert metered_client.post("/monitor/cycles", json={}).status_code == 403
