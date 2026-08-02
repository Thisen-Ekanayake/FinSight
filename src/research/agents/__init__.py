# ═══════════════════════════════════════════════════════
# FinSight — Specialist Agents
# ═══════════════════════════════════════════════════════
#
# Four narrow agents, each dispatched by the router via Send(). Every one
# receives a SpecialistInput and returns a PARTIAL ResearchState that the
# fan-in reducers merge.
#
# Public API:
#   fundamentals_node  filed financials from XBRL
#   filings_rag_node   narrative disclosure from Qdrant
#   macro_node         FRED time series
#   technical_node     price action and indicators
#   AGENT_NODES        name -> callable, consumed by graph.py
#
# Design note:
#   None of these call an LLM. They fetch, shape, and cite. Keeping them
#   deterministic means the fan-out is cheap, fast, and exactly reproducible
#   in tests — the reasoning happens once, in the synthesizer.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from typing import Callable

from src.research.agents.filings_rag import filings_rag_node
from src.research.agents.fundamentals import fundamentals_node
from src.research.agents.macro import macro_node
from src.research.agents.technical import technical_node

AGENT_NODES: dict[str, Callable] = {
    "fundamentals": fundamentals_node,
    "filings_rag": filings_rag_node,
    "macro": macro_node,
    "technical": technical_node,
}

__all__ = [
    "fundamentals_node",
    "filings_rag_node",
    "macro_node",
    "technical_node",
    "AGENT_NODES",
]
