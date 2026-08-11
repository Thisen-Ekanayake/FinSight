# ═══════════════════════════════════════════════════════
# FinSight — Streamlit Dashboard Entrypoint
# ═══════════════════════════════════════════════════════
#
# Purpose : Page config, navigation, and the connection banner every page
#           shares. Individual pages live in src/ui/pages/ and know nothing
#           about each other — this file is the only place that wires them
#           into one app.
#
# Usage:
#   ./shell_scripts/run_ui.sh
#   streamlit run src/ui/app.py
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="FinSight", layout="wide")

pages = {
    "Overview": [st.Page("pages/home.py", title="Home", default=True)],
    "Research": [st.Page("pages/research.py", title="Ask a Question")],
    "Monitoring": [
        st.Page("pages/monitor.py", title="Cycles & Alerts"),
        st.Page("pages/approvals.py", title="Approvals"),
        st.Page("pages/watchlist.py", title="Watchlist"),
    ],
    "Admin": [st.Page("pages/admin.py", title="Health & Budgets")],
}

st.navigation(pages).run()
