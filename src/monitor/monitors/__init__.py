# ═══════════════════════════════════════════════════════
# FinSight — Monitor Nodes
# ═══════════════════════════════════════════════════════
#
# Four nodes, two shapes. price and macro take the WHOLE watchlist in one
# branch because their data sources are batchable; filings and news take one
# ticker per branch because EDGAR and Finnhub are per-symbol.
#
# That asymmetry is the rate-limit strategy written into the graph topology
# rather than hidden in a helper — see src/monitor/graph.py.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from src.monitor.monitors.filings import filing_monitor_node
from src.monitor.monitors.macro import macro_monitor_node
from src.monitor.monitors.news import news_monitor_node
from src.monitor.monitors.price import price_monitor_node

MONITOR_NODES = {
    "price_monitor": price_monitor_node,
    "filing_monitor": filing_monitor_node,
    "news_monitor": news_monitor_node,
    "macro_monitor": macro_monitor_node,
}

__all__ = [
    "MONITOR_NODES",
    "filing_monitor_node",
    "macro_monitor_node",
    "news_monitor_node",
    "price_monitor_node",
]
