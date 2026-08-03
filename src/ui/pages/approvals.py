# ═══════════════════════════════════════════════════════
# FinSight — Approvals
# ═══════════════════════════════════════════════════════
#
# The Phase 7 HITL queue: every cycle paused on a HIGH-severity finding,
# waiting for a human to approve or reject it before anything is dispatched.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import streamlit as st

from src.ui import client
from src.ui.components import safe_call, severity_badge

st.title("Approvals")
st.caption(
    "A HIGH-severity finding pauses the monitoring graph before it reaches console, file, or email — "
    "durably, via a checkpointed LangGraph interrupt(). Nothing here has been dispatched yet."
)

pending_cycles = safe_call(client.list_cycles, limit=25, status="PENDING_APPROVAL")

if pending_cycles is not None and not pending_cycles:
    st.success("Nothing is waiting on a decision.")

for cycle in pending_cycles or []:
    cycle_id = cycle["cycle_id"]
    alerts = safe_call(client.get_pending_alerts, cycle_id)
    if not alerts:
        # Resolved between the list call and this one, or a stale summary row.
        continue

    with st.container(border=True):
        st.subheader(f"Cycle {cycle_id}")
        st.caption(f"tickers: {', '.join(cycle['tickers']) or '(none)'} · started {cycle['started_at'][:16]}")

        decisions: dict[str, str] = {}
        for alert in alerts:
            st.markdown(f"**{severity_badge(alert['severity'])} {alert['ticker'] or 'MACRO'}** — {alert['headline']}")
            st.write(alert["detail"])
            choice = st.radio(
                "Decision",
                ["Reject", "Approve"],
                horizontal=True,
                index=0,
                key=f"decision-{cycle_id}-{alert['alert_id']}",
                label_visibility="collapsed",
            )
            decisions[alert["alert_id"]] = "approve" if choice == "Approve" else "reject"
            st.divider()

        if st.button("Submit decisions", key=f"submit-{cycle_id}", type="primary"):
            result = safe_call(client.resume_cycle, cycle_id, decisions)
            if result is not None:
                approved = sum(1 for v in decisions.values() if v == "approve")
                st.success(f"Cycle resolved: {approved}/{len(decisions)} approved.")
                st.rerun()
