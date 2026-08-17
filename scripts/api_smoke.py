#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════
# FinSight — Live API Smoke Test
# ═══════════════════════════════════════════════════════
#
# Purpose : Call every endpoint the web dashboard uses, against a RUNNING
#           backend, and check the response actually has the shape
#           frontend/src/api/types.ts was written against.
#
# ══ WHY THIS EXISTS ALONGSIDE tests/test_api.py ══
#   Those tests run against a TestClient with the graph stubbed out. They
#   prove the routing and the serialisation are right. They cannot prove that
#   the deployed container is the build you think it is, that Qdrant is
#   reachable from inside it, or that the SSE frames survive whatever proxy
#   sits in front — which is exactly the class of problem that only shows up
#   between a frontend and a real backend.
#
# ══ SPEND ══
#   The read-only sweep is free. Two endpoints are not:
#     --query   POST /research/query/stream   real Vertex tokens, 30-60s
#     --cycle   POST /monitor/cycles          real provider quota, ~1 min
#   Both are OFF by default and each prints what it is about to spend.
#
# Usage:
#   python scripts/api_smoke.py                     # free endpoints only
#   python scripts/api_smoke.py --query --cycle     # everything
#   API_URL=http://localhost:8000 python scripts/api_smoke.py
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Iterable

import requests

API_URL: str = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT: int = 30
LONG_TIMEOUT: int = 180

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m–\033[0m"

_results: list[tuple[bool, str, str]] = []


def check(name: str, fn: Callable[[], str]) -> Any:
    """
    Run one check. `fn` returns a one-line summary; raising means failure.

    Every check is independent — a failing one records and moves on rather
    than aborting, because the whole point is a full picture of what the
    running backend does and does not serve.
    """
    started = time.monotonic()
    try:
        detail = fn()
        ms = int((time.monotonic() - started) * 1000)
        print(f"  {PASS} {name}  \033[2m{detail} · {ms}ms\033[0m")
        _results.append((True, name, detail))
    except Exception as exc:  # noqa: BLE001
        ms = int((time.monotonic() - started) * 1000)
        print(f"  {FAIL} {name}  \033[31m{type(exc).__name__}: {exc}\033[0m \033[2m· {ms}ms\033[0m")
        _results.append((False, name, str(exc)))


def skip(name: str, why: str) -> None:
    print(f"  {SKIP} {name}  \033[2m{why}\033[0m")


def get(path: str, **params: Any) -> Any:
    response = requests.get(f"{API_URL}{path}", params=params or None, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def post(path: str, body: Any, timeout: int = TIMEOUT) -> Any:
    response = requests.post(f"{API_URL}{path}", json=body, timeout=timeout)
    response.raise_for_status()
    return response.json() if response.content else None


def require(obj: Any, fields: Iterable[str], what: str) -> None:
    """Assert every field the frontend's type declares is actually present."""
    missing = [f for f in fields if f not in obj]
    if missing:
        raise AssertionError(f"{what} is missing {missing} — frontend/src/api/types.ts expects them")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# ── Admin ───────────────────────────────────────────────
def check_admin() -> None:
    section("admin")

    def health() -> str:
        body = get("/health")
        require(
            body,
            (
                "status",
                "environment",
                "llm_backend",
                "database",
                "checkpointer",
                "qdrant",
                "qdrant_detail",
                "scheduler_enabled",
                "scheduler_next_run_at",
            ),
            "HealthResponse",
        )
        degraded = [k for k in ("database", "checkpointer", "qdrant") if not body[k]]
        return f"{body['status']} · {body['llm_backend']}" + (f" · DEGRADED {degraded}" if degraded else "")

    def budgets() -> str:
        body = get("/admin/budgets")
        assert isinstance(body, list), "expected a list"
        for row in body:
            require(
                row, ("provider", "day", "used", "limit", "remaining", "soft_limit_reached", "exhausted"), "BudgetOut"
            )
        spent = [r["provider"] for r in body if r["exhausted"] and r["limit"] > 0]
        return f"{len(body)} providers" + (f" · spent: {', '.join(spent)}" if spent else "")

    def config() -> str:
        body = get("/admin/config")
        require(
            body,
            (
                "environment",
                "llm_backend",
                "model_flash",
                "model_pro",
                "rpm_limits",
                "qdrant_url",
                "embedding_model",
                "embedding_dim",
                "tracing_enabled",
                "langsmith_project",
                "numeric_tolerance",
                "max_repair_attempts",
                "verify_qualitative_claims",
                # Added for the web dashboard, which explains its own dedup
                # thresholds in prose rather than hardcoding them.
                "dedup_tau_high",
                "dedup_tau_low",
                "monitor_cadence_hours",
                "notification_sinks",
            ),
            "ConfigOut",
        )
        leaked = [
            word
            for word in ("api_key", "apikey", "secret", "token", "credential", "password")
            if word in json.dumps(body).lower()
        ]
        assert not leaked, f"/admin/config may be leaking a secret: {leaked}"
        return f"tau {body['dedup_tau_low']}/{body['dedup_tau_high']} · sinks {','.join(body['notification_sinks'])}"

    check("GET  /health", health)
    check("GET  /admin/budgets", budgets)
    check("GET  /admin/config", config)


# ── Watchlist ───────────────────────────────────────────
PROBE_TICKER = "ZZZZ"


def check_watchlist() -> None:
    section("watchlist")

    def listing() -> str:
        body = get("/watchlist")
        assert isinstance(body, list), "expected a list"
        for row in body:
            require(row, ("ticker", "company_name", "warmed_up", "added_at", "last_checked"), "WatchItemOut")
        cold = [r["ticker"] for r in body if not r["warmed_up"]]
        return f"{len(body)} watched" + (f" · {len(cold)} cold" if cold else "")

    def include_inactive() -> str:
        body = get("/watchlist", include_inactive=True)
        return f"{len(body)} including soft-deleted"

    def add_then_remove() -> str:
        # A deliberately unlistable symbol so a real name is never touched.
        created = post("/watchlist", {"ticker": PROBE_TICKER, "company_name": "Smoke Test Probe"})
        require(created, ("ticker", "company_name", "warmed_up", "added_at", "last_checked"), "WatchItemOut")
        assert created["ticker"] == PROBE_TICKER, f"echoed back {created['ticker']}"

        listed = {row["ticker"] for row in get("/watchlist")}
        assert PROBE_TICKER in listed, "added ticker did not appear in the listing"

        deleted = requests.delete(f"{API_URL}/watchlist/{PROBE_TICKER}", timeout=TIMEOUT)
        assert deleted.status_code == 204, f"DELETE returned {deleted.status_code}, expected 204"

        remaining = {row["ticker"] for row in get("/watchlist")}
        assert PROBE_TICKER not in remaining, "soft-deleted ticker is still listed as active"
        return "POST 201 -> GET -> DELETE 204, round trip clean"

    def unknown_is_404() -> str:
        response = requests.delete(f"{API_URL}/watchlist/NOSUCHTICKER", timeout=TIMEOUT)
        assert response.status_code == 404, f"expected 404, got {response.status_code}"
        return "404 as expected"

    check("GET  /watchlist", listing)
    check("GET  /watchlist?include_inactive", include_inactive)
    check("POST /watchlist + DELETE /watchlist/{ticker}", add_then_remove)
    check("DELETE /watchlist/{unknown} -> 404", unknown_is_404)


# ── Monitor (read-only) ─────────────────────────────────
def check_monitor_cycles() -> dict[str, Any]:
    """Cycle endpoints. Returns the newest cycle, so alerts can probe by id."""
    section("monitor — cycles")
    seen_cycle: dict[str, Any] = {}

    def cycles() -> str:
        body = get("/monitor/cycles", limit=10)
        assert isinstance(body, list), "expected a list"
        for row in body:
            require(
                row,
                (
                    "cycle_id",
                    "status",
                    "warmup",
                    "tickers",
                    "candidate_count",
                    "fired_count",
                    "suppressed_count",
                    "merged_count",
                    "error_count",
                    "api_call_count",
                    "started_at",
                    "duration_ms",
                ),
                "CycleOut",
            )
        if body:
            seen_cycle.update(body[0])
        held = [r for r in body if r["status"] == "PENDING_APPROVAL"]
        return f"{len(body)} cycles" + (f" · {len(held)} held for approval" if held else " · none held")

    def paused_filter() -> str:
        body = get("/monitor/cycles", limit=25, status="PENDING_APPROVAL")
        assert all(r["status"] == "PENDING_APPROVAL" for r in body), "filter returned a non-paused cycle"
        return f"{len(body)} paused"

    def one_cycle() -> str:
        if not seen_cycle:
            return "no cycles to fetch (skipped)"
        body = get(f"/monitor/cycles/{seen_cycle['cycle_id']}")
        assert body["cycle_id"] == seen_cycle["cycle_id"], "returned a different cycle"
        return f"{body['cycle_id']} · {body['status']}"

    def unknown_cycle() -> str:
        response = requests.get(f"{API_URL}/monitor/cycles/no-such-cycle", timeout=TIMEOUT)
        assert response.status_code == 404, f"expected 404, got {response.status_code}"
        return "404 as expected"

    def pending() -> str:
        if not seen_cycle:
            return "no cycles to fetch (skipped)"
        body = get(f"/monitor/cycles/{seen_cycle['cycle_id']}/pending")
        assert isinstance(body, list), "expected a list"
        # A COMPLETE cycle correctly returns [] — nothing is left to decide.
        # It must NOT 500: this reads the checkpointer, and a sync call to an
        # AsyncSqliteSaver from the event loop thread is exactly how this
        # endpoint used to fail while every unit test stayed green.
        return f"{len(body)} alerts awaiting a decision"

    check("GET  /monitor/cycles", cycles)
    check("GET  /monitor/cycles?status=PENDING_APPROVAL", paused_filter)
    check("GET  /monitor/cycles/{id}", one_cycle)
    check("GET  /monitor/cycles/{unknown} -> 404", unknown_cycle)
    check("GET  /monitor/cycles/{id}/pending", pending)

    return seen_cycle


def check_monitor_alerts() -> None:
    section("monitor — alerts")

    def alerts() -> str:
        body = get("/monitor/alerts", limit=20)
        assert isinstance(body, list), "expected a list"
        for row in body:
            require(
                row,
                (
                    "alert_id",
                    "cycle_id",
                    "ticker",
                    "alert_type",
                    "severity",
                    "status",
                    "headline",
                    "detail",
                    "canonical_text",
                    "occurrence_count",
                    "first_seen_at",
                    "last_seen_at",
                    "fired_at",
                    "parent_alert_id",
                    "evidence",
                ),
                "AlertOut",
            )
            assert row["severity"] in {"LOW", "MED", "HIGH"}, f"unknown severity {row['severity']}"
        by_sev = {s: sum(1 for r in body if r["severity"] == s) for s in ("HIGH", "MED", "LOW")}
        return f"{len(body)} alerts · " + " ".join(f"{k}:{v}" for k, v in by_sev.items())

    def alert_severity_filter() -> str:
        body = get("/monitor/alerts", limit=20, severity="LOW")
        assert all(r["severity"] == "LOW" for r in body), "severity filter leaked another level"
        return f"{len(body)} LOW"

    def one_alert() -> str:
        listing = get("/monitor/alerts", limit=1)
        if not listing:
            return "no alerts to fetch (skipped)"
        body = get(f"/monitor/alerts/{listing[0]['alert_id']}")
        assert body["alert_id"] == listing[0]["alert_id"], "returned a different alert"
        return body["alert_id"][:8]

    check("GET  /monitor/alerts", alerts)
    check("GET  /monitor/alerts?severity=LOW", alert_severity_filter)
    check("GET  /monitor/alerts/{id}", one_alert)


def check_monitor_decisions() -> None:
    section("monitor — dedup decisions")

    def decisions() -> str:
        body = get("/monitor/decisions", limit=100)
        assert isinstance(body, list), "expected a list"
        for row in body:
            require(
                row,
                (
                    "cycle_id",
                    "ticker",
                    "alert_type",
                    "severity",
                    "decision",
                    "reason",
                    "dedup_key",
                    "candidate_text",
                    "parent_alert_id",
                    "parent_text",
                    "score",
                    "decided_at",
                ),
                "DedupDecisionOut",
            )
        kinds: dict[str, int] = {}
        for row in body:
            kinds[row["decision"]] = kinds.get(row["decision"], 0) + 1
        # The Findings view hangs suppressed candidates under their parent, so
        # a suppression with no parent id would render nowhere.
        orphans = [r for r in body if r["decision"] != "FIRE" and not r["parent_alert_id"]]
        assert not orphans, f"{len(orphans)} suppressions carry no parent_alert_id"
        return f"{len(body)} decisions · " + " ".join(f"{k}:{v}" for k, v in sorted(kinds.items()))

    def decision_filter() -> str:
        body = get("/monitor/decisions", limit=50, decision="FIRE")
        assert all(r["decision"] == "FIRE" for r in body), "decision filter leaked another kind"
        return f"{len(body)} FIRE"

    check("GET  /monitor/decisions", decisions)
    check("GET  /monitor/decisions?decision=FIRE", decision_filter)


# ── Research (read-only) ────────────────────────────────
def check_research_reads() -> None:
    """
    Note that /research/runs is scoped to the caller.

    This script sends no Authorization header, so it is aimed at a deployment
    with AUTH_ENABLED=false — where every caller is an unlimited
    ANONYMOUS_USER and therefore sees its own runs plus every unattributed
    one. Against a deployment with auth ON these checks would 401 long before
    the scoping mattered.
    """
    section("research (read-only)")
    seen: dict[str, Any] = {}

    def runs() -> str:
        body = get("/research/runs", limit=10)
        assert isinstance(body, list), "expected a list"
        for row in body:
            require(
                row,
                (
                    "thread_id",
                    "query",
                    "citation_coverage",
                    "verification_passed",
                    "repair_count",
                    "agents_used",
                    "tickers",
                    "latency_ms",
                    "created_at",
                ),
                "RunSummaryOut",
            )
        if body:
            seen.update(body[0])
        failed = sum(1 for r in body if not r["verification_passed"])
        return f"{len(body)} runs · {failed} failed verification"

    def thread() -> str:
        if not seen:
            return "no runs to replay (skipped)"
        body = get(f"/research/threads/{seen['thread_id']}")
        require(
            body, ("thread_id", "query", "answer", "summary", "verification", "tool_calls", "steps"), "ThreadResponse"
        )
        if body["verification"]:
            require(
                body["verification"],
                ("citation_coverage", "passed", "verified_count", "unsupported_claims", "invalid_source_ids"),
                "VerificationOut",
            )
        for step in body["steps"]:
            require(step, ("step", "nodes", "next", "findings_total", "citations_total", "created_at"), "ThreadStep")
        return f"{len(body['steps'])} supersteps · {len(body['tool_calls'])} tool calls"

    def unknown_thread() -> str:
        response = requests.get(f"{API_URL}/research/threads/research:nosuchthread", timeout=TIMEOUT)
        assert response.status_code == 404, f"expected 404, got {response.status_code}"
        return "404 as expected"

    def short_query_rejected() -> str:
        response = requests.post(f"{API_URL}/research/query", json={"query": "hi"}, timeout=TIMEOUT)
        assert response.status_code == 422, f"expected 422 before any spend, got {response.status_code}"
        return "422 before any LLM spend"

    check("GET  /research/runs", runs)
    check("GET  /research/threads/{id}", thread)
    check("GET  /research/threads/{unknown} -> 404", unknown_thread)
    check("POST /research/query (too short) -> 422", short_query_rejected)


# ── Research (spends) ───────────────────────────────────
DEFAULT_QUESTION = "What was Apple's most recent closing price and its 50-day moving average?"


def check_research_stream(question: str) -> None:
    section("research — streamed query \033[33m(spends Vertex tokens)\033[0m")

    def stream() -> str:
        """
        Consume the SSE stream frame by frame, the same way the frontend's
        askStream does — including the "\\n\\n" framing, because a proxy that
        buffers would still return 200 with the whole body at the end and
        only a frame-arrival check catches it.
        """
        started = time.monotonic()
        frames: list[tuple[str, dict[str, Any]]] = []
        arrival: list[float] = []

        with requests.post(
            f"{API_URL}/research/query/stream",
            json={"query": question},
            stream=True,
            timeout=LONG_TIMEOUT,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            assert "text/event-stream" in response.headers.get(
                "content-type", ""
            ), f"wrong content-type: {response.headers.get('content-type')}"

            buffer = ""
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk
                while "\n\n" in buffer:
                    raw, buffer = buffer.split("\n\n", 1)
                    name = next((ln[6:].strip() for ln in raw.splitlines() if ln.startswith("event:")), None)
                    data = next((ln[5:].strip() for ln in raw.splitlines() if ln.startswith("data:")), None)
                    if name and data is not None:
                        frames.append((name, json.loads(data)))
                        arrival.append(time.monotonic() - started)
                        if name == "node":
                            print(f"      \033[2m{arrival[-1]:5.1f}s  {frames[-1][1]['node']}\033[0m")

        names = [n for n, _ in frames]
        assert names, "no frames arrived at all"
        assert names[0] == "start", f"first frame was {names[0]!r}, expected 'start'"
        assert "error" not in names, f"error frame: {dict(frames[names.index('error')][1])}"
        assert names[-1] == "final", f"last frame was {names[-1]!r}, expected 'final'"

        # If everything landed in the same instant, something in front of the
        # API buffered the stream and the progress UI would be a lie.
        span = arrival[-1] - arrival[0]
        assert span > 0.5, f"every frame arrived within {span:.2f}s — the stream is being buffered somewhere"

        final = frames[-1][1]
        require(
            final,
            (
                "thread_id",
                "query",
                "answer",
                "citations",
                "conflicts",
                "verification",
                "agents_used",
                "tickers",
                "branch_count",
                "repair_count",
                "errors",
                "latency_ms",
            ),
            "QueryResponse",
        )
        assert final["answer"].strip(), "the final frame carried an empty answer"

        verification = final["verification"]
        nodes = [f["node"] for n, f in frames if n == "node"]
        return (
            f"{len(nodes)} nodes ({', '.join(nodes)}) · "
            f"{verification['citation_coverage']:.0%} traced · "
            f"{len(verification['unsupported_claims'])} flagged · "
            f"{final['latency_ms'] / 1000:.1f}s"
        )

    check("POST /research/query/stream", stream)


# ── Monitor (spends) ────────────────────────────────────
def check_monitor_cycle() -> None:
    section("monitor — one cycle \033[33m(spends provider quota)\033[0m")

    def run() -> str:
        body = post("/monitor/cycles", {"warmup": False, "tickers": []}, timeout=LONG_TIMEOUT)
        require(
            body,
            (
                "cycle_id",
                "status",
                "warmup",
                "candidate_count",
                "fired",
                "merged",
                "suppressed",
                "pending_approval",
                "errors",
                "duration_ms",
            ),
            "CycleRunResponse",
        )
        assert body["status"] in {"COMPLETE", "PENDING_APPROVAL"}, f"unexpected status {body['status']}"

        held = body["pending_approval"]
        if held:
            # The pending endpoint must agree with what the run reported, or
            # the Desk would offer a decision the resume call then rejects.
            from_endpoint = get(f"/monitor/cycles/{body['cycle_id']}/pending")
            ids_run = {a["alert_id"] for a in held}
            ids_endpoint = {a["alert_id"] for a in from_endpoint}
            assert ids_run == ids_endpoint, f"run reported {ids_run}, /pending reports {ids_endpoint}"

        return (
            f"{body['status']} · {body['candidate_count']} weighed · "
            f"{len(body['fired'])} reported · {len(body['suppressed'])} folded · "
            f"{len(held)} held · {body['duration_ms'] / 1000:.1f}s"
            + (f" · errors: {body['errors']}" if body["errors"] else "")
        )

    check("POST /monitor/cycles", run)

    def resume_guard() -> str:
        """
        A resume with decisions that do not match the pending set must be
        REFUSED, not partially applied. Checked against a cycle that is not
        paused at all, which is the cheapest way to prove the guard exists.
        """
        listing = get("/monitor/cycles", limit=1)
        if not listing:
            return "no cycle to probe (skipped)"
        response = requests.post(
            f"{API_URL}/monitor/cycles/{listing[0]['cycle_id']}/resume",
            json={"decisions": {"not-a-real-alert-id": "approve"}},
            timeout=TIMEOUT,
        )
        assert (
            response.status_code >= 400
        ), f"a mismatched resume returned {response.status_code} — it should have been refused"
        return f"refused with {response.status_code}"

    check("POST /monitor/cycles/{id}/resume (mismatched) -> 4xx", resume_guard)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running FinSight API.")
    parser.add_argument("--query", action="store_true", help="also run a real research query (spends Vertex tokens)")
    parser.add_argument("--cycle", action="store_true", help="also run a real monitoring cycle (spends provider quota)")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="the question to ask with --query")
    args = parser.parse_args()

    print(f"\n\033[1mFinSight API smoke test\033[0m  \033[2m{API_URL}\033[0m")

    try:
        requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"\n  {FAIL} Cannot reach {API_URL} — is the API running? ({exc})\n")
        return 2

    check_admin()
    check_watchlist()
    check_monitor_cycles()
    check_monitor_alerts()
    check_monitor_decisions()
    check_research_reads()

    if args.query:
        check_research_stream(args.question)
    else:
        section("research — streamed query")
        skip("POST /research/query/stream", "costs Vertex tokens — pass --query to run it")

    if args.cycle:
        check_monitor_cycle()
    else:
        section("monitor — one cycle")
        skip("POST /monitor/cycles", "costs provider quota — pass --cycle to run it")

    passed = sum(1 for ok, _, _ in _results if ok)
    failed = [name for ok, name, _ in _results if not ok]

    print(f"\n\033[1m{passed}/{len(_results)} checks passed\033[0m")
    if failed:
        print("\033[31mfailed:\033[0m")
        for name in failed:
            print(f"  · {name}")
        print()
        return 1

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
