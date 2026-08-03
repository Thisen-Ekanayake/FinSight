# ═══════════════════════════════════════════════════════
# FinSight — Monitoring: Cycles & Alerts
# ═══════════════════════════════════════════════════════
#
# Trigger a cycle, and browse what past cycles found, fired, and suppressed.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import streamlit as st

from src.ui import client
from src.ui.components import safe_call, severity_badge

st.title("Cycles & alerts")
st.caption("Watches the ticker watchlist, scores what it finds by rule, and deduplicates it semantically.")

with st.form("run_cycle_form"):
    cols = st.columns([1, 2])
    warmup = cols[0].checkbox(
        "Warmup (index only, report nothing)",
        help="Run this once before the very first real cycle — a cold dedup index would otherwise "
        "report every open filing, article, and price move at once.",
    )
    tickers_raw = cols[1].text_input("Override tickers (comma-separated, blank = use the watchlist)")
    run = st.form_submit_button("Run a cycle now", type="primary")

if run:
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()] or None
    with st.spinner("Fetching from up to four providers and deduplicating..."):
        result = safe_call(client.run_cycle, warmup=warmup, tickers=tickers)
    if result is not None:
        if result["status"] == "PENDING_APPROVAL":
            st.warning(
                f"{len(result['pending_approval'])} HIGH-severity alert(s) are paused awaiting a decision "
                "— nothing has been dispatched yet. Resolve them on the **Approvals** page."
            )
        else:
            st.success(
                f"{result['candidate_count']} candidate(s) → {len(result['fired'])} fired, "
                f"{len(result['suppressed'])} suppressed, {len(result['merged'])} escalated."
            )
        for alert in [*result["fired"], *result["merged"]]:
            st.write(f"**{severity_badge(alert['severity'])}** {alert['ticker'] or 'MACRO'} — {alert['headline']}")
        for error in result.get("errors", []):
            st.error(error)

st.divider()

tab_cycles, tab_alerts, tab_decisions = st.tabs(["Cycles", "Fired alerts", "Dedup decisions"])

with tab_cycles:
    status_filter = st.selectbox("Status", ["(any)", "COMPLETE", "PENDING_APPROVAL"], key="cycle_status_filter")
    cycles = safe_call(client.list_cycles, limit=25, status=None if status_filter == "(any)" else status_filter)
    if cycles:
        st.dataframe(
            [
                {
                    "cycle_id": row["cycle_id"],
                    "status": row["status"],
                    "tickers": ", ".join(row["tickers"]),
                    "candidates": row["candidate_count"],
                    "fired": row["fired_count"],
                    "suppressed": row["suppressed_count"],
                    "started": row["started_at"][:16].replace("T", " "),
                }
                for row in cycles
            ],
            use_container_width=True,
            hide_index=True,
        )
    elif cycles is not None:
        st.caption("No cycles recorded yet.")

with tab_alerts:
    a_cols = st.columns(3)
    ticker_filter = a_cols[0].text_input("Ticker", key="alert_ticker_filter")
    severity_filter = a_cols[1].selectbox("Severity", ["(any)", "HIGH", "MED", "LOW"], key="alert_severity_filter")
    alert_status_filter = a_cols[2].selectbox(
        "Status", ["(any)", "FIRED", "PENDING_APPROVAL", "REJECTED"], key="alert_status_filter"
    )
    alerts = safe_call(
        client.list_alerts,
        limit=50,
        ticker=ticker_filter or None,
        severity=None if severity_filter == "(any)" else severity_filter,
        status=None if alert_status_filter == "(any)" else alert_status_filter,
    )
    if alerts:
        st.dataframe(
            [
                {
                    "severity": row["severity"],
                    "status": row["status"],
                    "ticker": row["ticker"] or "MACRO",
                    "headline": row["headline"],
                    "occurrences": row["occurrence_count"],
                    "fired_at": row["fired_at"][:16].replace("T", " "),
                }
                for row in alerts
            ],
            use_container_width=True,
            hide_index=True,
        )
    elif alerts is not None:
        st.caption("No alerts match these filters.")

with tab_decisions:
    st.caption(
        "Every dedup decision, including the FIREs — a log of only suppressions would have no negatives "
        "to measure precision against."
    )
    decisions = safe_call(client.list_decisions, limit=100)
    if decisions:
        st.dataframe(
            [
                {
                    "decision": row["decision"],
                    "score": f"{row['score']:.3f}" if row["score"] else "-",
                    "ticker": row["ticker"] or "MACRO",
                    "text": row["candidate_text"],
                    "matched": row["parent_text"] or "-",
                }
                for row in decisions
            ],
            use_container_width=True,
            hide_index=True,
        )
    elif decisions is not None:
        st.caption("No dedup decisions recorded yet.")
