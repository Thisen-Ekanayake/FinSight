# ═══════════════════════════════════════════════════════
# FinSight — Admin: Health & Budgets
# ═══════════════════════════════════════════════════════
#
# Operational visibility — is it up, what is it configured as, and how much
# of each free tier is left today.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import streamlit as st

from src.ui import client
from src.ui.components import safe_call

st.title("Health & budgets")

status = safe_call(client.health)
if status:
    cols = st.columns(5)
    cols[0].metric("Status", status["status"].upper())
    cols[1].metric("Database", "up" if status["database"] else "down")
    cols[2].metric("Checkpointer", "up" if status["checkpointer"] else "down")
    cols[3].metric("Qdrant", status["qdrant_detail"])
    cols[4].metric("Scheduler", "on" if status["scheduler_enabled"] else "off")
    if status["scheduler_enabled"]:
        st.caption(f"Next automatic cycle: {status.get('scheduler_next_run_at') or 'unknown'}")

st.divider()
st.subheader("Today's API usage")
usage = safe_call(client.budgets)
if usage:
    st.dataframe(
        [
            {
                "provider": row["provider"],
                "used": row["used"],
                "limit": row["limit"] or "unlimited",
                "remaining": row["remaining"],
                "status": "EXHAUSTED" if row["exhausted"] else ("near limit" if row["soft_limit_reached"] else "ok"),
            }
            for row in usage
        ],
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("Effective configuration")
st.caption("Secrets excluded — names, URLs, and numbers only.")
cfg = safe_call(client.config)
if cfg:
    st.json(cfg)
