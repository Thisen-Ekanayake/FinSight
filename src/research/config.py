# ═══════════════════════════════════════════════════════
# FinSight — Research Subsystem Configuration
# ═══════════════════════════════════════════════════════
#
# Purpose : All prompts and tunable thresholds for Subsystem 1, as module
#           constants. Nothing here imports an LLM — this file stays cheap
#           and diffable, so a prompt change is a one-line review.
#
# Public API:
#   ROUTER_PROMPT_SYSTEM / _USER
#   SYNTHESIS_PROMPT_SYSTEM / _USER
#   AGENT_CAPABILITIES, DEFAULT_TICKER_LIMIT, MAX_REPAIR_ATTEMPTS, ...
#
# Mirrors the per-domain config.py pattern from Hemas-Nexus-AI's risk_agent.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

# ── Routing behaviour ───────────────────────────────────
# Cap on tickers per query. The fan-out is (agents x tickers), so a query
# naming 12 companies would spawn 48 concurrent branches and burn quota.
DEFAULT_TICKER_LIMIT: int = 4

# Bounded self-correction. Infinite repair loops are how these systems spend
# real money on one question; finalize strips unsupported claims instead.
MAX_REPAIR_ATTEMPTS: int = 1

# Guard against a cyclic graph running away.
RECURSION_LIMIT: int = 25

# Filing chunks retrieved per filings_rag branch.
FILINGS_TOP_K: int = 8

# What each specialist can actually answer. Injected into the router prompt so
# routing decisions stay in sync with the code rather than drifting from it.
AGENT_CAPABILITIES: dict[str, str] = {
    "fundamentals": (
        "Reported financial figures from SEC XBRL data: revenue, net income, gross profit, "
        "operating income, total assets, liabilities, equity, cash, EPS, R&D expense, "
        "operating cash flow. Exact filed values with accession numbers. "
        "Use for any question about what a company actually reported."
    ),
    "filings_rag": (
        "Narrative disclosure text from 10-K and 10-Q filings: risk factors, business "
        "description, legal proceedings, and management's discussion (MD&A). "
        "Use for qualitative questions — what management said, what risks were flagged, "
        "how the company describes its strategy or competition."
    ),
    "macro": (
        "US macroeconomic time series from FRED: Fed funds rate, CPI/inflation, "
        "unemployment, 10-year Treasury yield, yield-curve spread. "
        "Use for questions about the economic environment, rates, or inflation. "
        "NOT company-specific — this agent ignores tickers."
    ),
    "technical": (
        "Price history and technical indicators: last close, 1d/5d/20d change, RSI, MACD, "
        "moving averages, Bollinger bands, volume ratio, volatility z-score. "
        "Use for questions about price action, momentum, or how a stock has moved."
    ),
}

# ── Conflict resolution ─────────────────────────────────
# Two values within this relative tolerance are treated as agreeing; the
# higher-trust source wins. Beyond it, a Conflict is recorded and SURFACED in
# the answer rather than silently resolved.
CONFLICT_REL_TOLERANCE: float = 0.01

# ── Citation verification ───────────────────────────────
# How close a number in the answer must be to a number a tool actually returned
# to count as grounded. 0.5% absorbs legitimate rounding — "$391.0B" against a
# filed 391,035,000,000 is a 0.009% difference — without letting a genuinely
# different figure through.
NUMERIC_REL_TOLERANCE: float = 0.005

# Percentages need an absolute fallback: 0.50% vs 0.51% is only 0.01
# percentage points apart but 2% in relative terms, which the tolerance above
# would reject. Either test passing is enough.
PERCENT_ABS_TOLERANCE: float = 0.05

# Bare integers below this, carrying no currency symbol, scale suffix, or
# percent sign, are prose rather than financial claims — list counts, item
# numbers, "the 10-year Treasury", "over 5 sessions". Grounding them produces
# noise and pointless repair attempts.
IGNORE_BARE_INTEGERS_BELOW: float = 100.0

# Cap on the pairwise ratio derivation in the evidence index. Groups are
# (ticker, period) so they are small in practice; this bounds the O(n^2).
MAX_DERIVATION_GROUP: int = 10

# Repair fan-out ceiling. Five unsupported claims should not spawn five
# branches — they are usually the same missing data seen from five angles.
MAX_REPAIR_BRANCHES: int = 3

# Appended by finalize when a claim could not be grounded and was removed.
CAVEAT_TEMPLATE = (
    "\n\n---\n*Note: {count} statement(s) could not be grounded in the retrieved "
    "source data and were removed from this answer: {claims}*"
)

# ── Prompts ─────────────────────────────────────────────

ROUTER_PROMPT_SYSTEM = """You are the routing layer of a financial research system. \
Your job is to decompose a user's question into work for specialist agents.

You do NOT answer the question. You only plan.

Available specialists:
{capabilities}

Rules:
1. Select ONLY the specialists whose data can actually help. Selecting an
   irrelevant specialist wastes a real API call and adds noise to the answer.
2. Extract US-listed ticker symbols. Resolve company names to tickers
   (Apple -> AAPL, Microsoft -> MSFT, Nvidia -> NVDA, Alphabet/Google -> GOOGL,
   JPMorgan -> JPM). Maximum {ticker_limit} tickers.
3. If the question is purely macroeconomic with no company mentioned, return an
   empty ticker list and select only "macro".
4. For EACH selected specialist, write a focused sub-question answerable from
   that specialist's data alone. Do not just repeat the user's question.
5. timeframe: a short phrase if the question is time-scoped ("last two fiscal
   years", "Q3 2025"), otherwise null.

Be decisive. Prefer fewer, better-targeted specialists over selecting everything."""

ROUTER_PROMPT_USER = """User question:
{query}

Produce the routing plan."""


SYNTHESIS_PROMPT_SYSTEM = """You are a financial research analyst writing a grounded answer.

You are given FINDINGS collected by specialist agents. Each finding carries a
source identifier. Your answer must rest entirely on those findings.

ABSOLUTE RULES:
1. Use ONLY the values present in the findings. Never introduce a number from
   your own knowledge, and never estimate, extrapolate, or round beyond what
   is given.
2. Every numeric claim MUST carry an inline source marker in the exact form
   [SRC:<SOURCE_TYPE>:<SOURCE_ID>], placed immediately after the claim.
   Example: Apple reported revenue of $416.2B [SRC:EDGAR:0000320193-25-000079].
3. If the findings do not answer part of the question, say so plainly. An
   honest gap is far more useful than a plausible guess.
4. If two sources disagree, state both figures, name the sources, and say which
   you are using and why. Do not silently pick one.
5. No preamble. No "Based on the findings...". Start with the answer.

Style: precise, compact, and specific. Write for an analyst who wants the
number and its provenance, not an essay."""

SYNTHESIS_PROMPT_USER = """Question:
{query}

Findings from specialist agents:
{findings}

{conflicts_block}

Write the grounded answer."""


CONFLICTS_BLOCK_TEMPLATE = """CONFLICTING VALUES DETECTED — you must surface these in your answer:
{conflicts}
"""

# Rendered into SYNTHESIS_PROMPT_USER when a specialist returned nothing, so
# the model is told explicitly rather than left to invent coverage.
NO_FINDINGS_NOTE = "(No findings were returned. Say clearly that the data was unavailable.)"
