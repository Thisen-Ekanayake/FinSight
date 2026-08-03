# ═══════════════════════════════════════════════════════
# FinSight — Watchlist
# ═══════════════════════════════════════════════════════
#
# Manage which tickers the monitoring subsystem watches.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import streamlit as st

from src.ui import client
from src.ui.components import safe_call

st.title("Watchlist")

with st.form("add_ticker_form", clear_on_submit=True):
    cols = st.columns([1, 2, 1])
    ticker = cols[0].text_input("Ticker", placeholder="AAPL")
    company_name = cols[1].text_input("Company name (resolved from EDGAR if left blank)")
    add = cols[2].form_submit_button("Add", type="primary")

if add:
    if not ticker.strip():
        st.warning("Enter a ticker symbol first.")
    else:
        added = safe_call(client.add_ticker, ticker.strip(), company_name=company_name.strip())
        if added is not None:
            st.success(f"Watching {added['ticker']} ({added['company_name']}).")
            st.rerun()

st.divider()

include_inactive = st.checkbox("Show removed tickers too")
items = safe_call(client.get_watchlist, include_inactive=include_inactive)

if not items:
    if items is not None:
        st.caption("Watchlist is empty.")
else:
    for item in items:
        cols = st.columns([1, 2, 2, 1])
        cols[0].write(f"**{item['ticker']}**")
        cols[1].write(item["company_name"])
        warm = "warm" if item["warmed_up"] else "COLD — run a --warmup cycle"
        checked = ", ".join(f"{monitor} {stamp[:16]}" for monitor, stamp in sorted(item["last_checked"].items()))
        cols[2].write(f"{warm}" + (f" · {checked}" if checked else " · never checked"))
        if cols[3].button("Remove", key=f"remove-{item['ticker']}"):
            if safe_call(client.remove_ticker, item["ticker"]):
                st.rerun()
