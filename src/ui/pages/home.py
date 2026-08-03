# ═══════════════════════════════════════════════════════
# FinSight — Home
# ═══════════════════════════════════════════════════════
#
# The landing page: is the backend up, and is anything waiting on a human.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import streamlit as st

from src.ui import client
from src.ui.components import safe_call

st.title("FinSight")
st.caption("Multi-agent financial research, plus autonomous portfolio surveillance.")

status = safe_call(client.health)

if status is None:
    st.warning(f"Cannot reach the API at `{client.API_URL}`. Set the `API_URL` environment variable if it moved.")
    st.stop()

cols = st.columns(4)
cols[0].metric("Status", status["status"].upper())
cols[1].metric("Database", "up" if status["database"] else "down")
cols[2].metric("Qdrant", status["qdrant_detail"])
cols[3].metric("LLM backend", status["llm_backend"])

if status["scheduler_enabled"]:
    next_run = status.get("scheduler_next_run_at")
    st.info(f"Automatic monitoring is ON — next cycle at {next_run or 'unknown'}.")
else:
    st.caption("Automatic monitoring is off (`MONITOR_SCHEDULER_ENABLED=false`). Run cycles from the Monitoring page.")

pending = safe_call(client.list_cycles, limit=5, status="PENDING_APPROVAL")
if pending:
    st.warning(
        f"{len(pending)} cycle(s) are paused awaiting a HIGH-severity decision. "
        "See the **Approvals** page under Monitoring."
    )

st.divider()
st.markdown("""
**Research** — ask a natural-language question; every number in the answer traces back to a primary source.

**Monitoring** — watches a ticker list, deduplicates what it finds, and pauses every HIGH-severity
finding for a human to approve before it is dispatched anywhere.
""")
