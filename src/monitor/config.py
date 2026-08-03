# ═══════════════════════════════════════════════════════
# FinSight — Monitoring Subsystem Configuration
# ═══════════════════════════════════════════════════════
#
# Purpose : Severity rules, monitor lookbacks, and the canonicalization
#           prompt, as module constants. Nothing here imports an LLM or
#           touches the network.
#
# Public API:
#   MONITOR_NAMES, DEFAULT_WATCHLIST, MONITOR_CADENCE_HOURS
#   PRICE_*, FILING_*, NEWS_*, MACRO_*        severity thresholds
#   CANONICAL_SUMMARY_PROMPT_SYSTEM / _USER
#   DEDUP_ACTIVE_STATUSES, WARMUP_STATUS, PENDING_APPROVAL_STATUS
#   NOTIFICATION_SINKS, EMAIL_*, ALERTS_LOG_PATH_NAME   (Phase 7 dispatcher)
#   MONITOR_SCHEDULER_ENABLED                            (Phase 7 scheduler)
#
# ══ SEVERITY IS RULES, NOT LLM JUDGEMENT ══
#   Only the one-line qualitative SUMMARY is generated. Severity itself is
#   computed from thresholds in this file, for three reasons:
#
#     1. It is testable. "An 8-K carrying Item 4.02 is HIGH" is an assertion;
#        "the model thought it was serious" is not.
#     2. It is stable. The same input produces the same severity in March and
#        in September, which is what makes the alert history comparable.
#     3. A model asked to rate urgency will inflate it. Being alarmed is how a
#        chat assistant demonstrates that it understood you — and an alerting
#        system whose severity drifts upward stops being read at all.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import os

# The four monitor nodes. graph.py Sends to these names, so a typo here is a
# dispatch to a node that does not exist.
MONITOR_NAMES: tuple[str, ...] = ("price_monitor", "filing_monitor", "news_monitor", "macro_monitor")

# Which checkpoint key each monitor advances. Short names because they are a
# composite key component in SQLite, not a display string.
MONITOR_CHECKPOINT_KEYS: dict[str, str] = {
    "price_monitor": "price",
    "filing_monitor": "filing",
    "news_monitor": "news",
    "macro_monitor": "macro",
}

DEFAULT_WATCHLIST: list[str] = [
    t.strip().upper() for t in os.getenv("MONITOR_WATCHLIST", "AAPL,MSFT,NVDA,JPM,GOOGL").split(",") if t.strip()
]

# Cycle cadence. Four cycles a day over ten tickers is ~26 external calls per
# cycle with price and macro batched — see src/data/config.py.
MONITOR_CADENCE_HOURS: int = int(os.getenv("MONITOR_CADENCE_HOURS", "6"))

# How far back a monitor looks when it has never run for a ticker. Deliberately
# short: the first cycle for a new ticker is a WARMUP that fires nothing, so a
# long lookback would only cost API calls.
DEFAULT_LOOKBACK_DAYS: int = 3

# Ceiling on the lookback however stale the checkpoint is. A process down for a
# month must not ask EDGAR for a month of filings across the whole watchlist in
# one burst on restart.
MAX_LOOKBACK_DAYS: int = 14


# ── Alert status vocabulary ─────────────────────────────
WARMUP_STATUS: str = "WARMUP"

# A HIGH-severity, non-warmup alert lands here instead of FIRED, and stays
# there until a human resolves it — see src/monitor/graph.py human_approval_node.
PENDING_APPROVAL_STATUS: str = "PENDING_APPROVAL"
REJECTED_STATUS: str = "REJECTED"

# Statuses whose Qdrant points participate in dedup.
#
# WARMUP is included deliberately, and the plan's original filter omitted it.
# The whole point of a warmup cycle is that cycle 2 dedups against what cycle 1
# saw; excluding WARMUP would upsert those points and then never match them,
# which produces a system that looks like it has a cold-start guard and does
# not. PENDING_APPROVAL is included because an alert awaiting a human decision
# has already been raised — a duplicate of it is still a duplicate.
#
# REJECTED is deliberately EXCLUDED. A human declining to raise an alert says
# nothing about whether the next candidate describing it is worth raising —
# unlike a suppression, a rejection carries no information about what the
# event WAS, only that this particular report of it should not have paged
# anyone. Keeping it dedup-active would let one rejection silently veto every
# future report of the same event, which is the false-suppress failure this
# whole engine is built to avoid.
DEDUP_ACTIVE_STATUSES: tuple[str, ...] = ("FIRED", PENDING_APPROVAL_STATUS, WARMUP_STATUS)


# ── Price severity ──────────────────────────────────────
# Below this, nothing is even a candidate. A 1% day is not an event, and
# emitting it would put the dedup engine to work on noise.
PRICE_MIN_PCT: float = 2.5

PRICE_MED_PCT: float = 4.0
PRICE_HIGH_PCT: float = 7.0

# The percentage thresholds alone are wrong for a volatile name: a 5% day in
# NVDA is ordinary and a 5% day in JPM is not. vol_zscore measures the move
# against the ticker's OWN 60-day realised volatility, so either route can
# raise severity and the more alarming one wins.
PRICE_MED_ZSCORE: float = 2.0
PRICE_HIGH_ZSCORE: float = 3.0

# Volume confirmation. Not a severity input on its own — it only ever appears
# in the qualitative summary, because "on heavy volume" changes what the event
# MEANS without changing how urgent it is.
PRICE_NOTABLE_VOLUME_RATIO: float = 1.5


# ── Filing severity ─────────────────────────────────────
FILING_WATCHED_FORMS: list[str] = ["8-K", "10-K", "10-Q"]

# 8-K item codes that are automatically HIGH. These are the disclosures a
# company files because it has to, not because it wants to.
FILING_HIGH_8K_ITEMS: frozenset[str] = frozenset(
    {
        "1.03",  # bankruptcy or receivership
        "2.06",  # material impairment
        "4.01",  # change in certifying accountant
        "4.02",  # non-reliance on previously issued financials  <- the scary one
        "5.02",  # departure of principal officers
    }
)

# Material but expected: earnings releases, results of shareholder votes,
# material agreements. Worth knowing about, not worth waking up for.
FILING_MED_8K_ITEMS: frozenset[str] = frozenset(
    {
        "1.01",  # entry into a material definitive agreement
        "2.02",  # results of operations and financial condition
        "3.01",  # delisting notice
        "5.07",  # submission of matters to a vote
        "8.01",  # other events
    }
)

# A periodic report is MED: material by definition, but scheduled, so its
# arrival is never a surprise.
FILING_PERIODIC_FORMS: frozenset[str] = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F"})


# ── News severity ───────────────────────────────────────
# Sentiment scores are provider-supplied and noisy, so these are wide.
NEWS_MED_SENTIMENT: float = -0.30
NEWS_HIGH_SENTIMENT: float = -0.60

# HIGH additionally requires corroboration. One outlet running a negative story
# is a story; several independent outlets running it is an event. Without this,
# a single badly-scored headline pages you.
NEWS_HIGH_MIN_SOURCES: int = 2

# Below this magnitude the article is not a candidate at all.
NEWS_MIN_ABS_SENTIMENT: float = 0.25

# Cap on candidates per ticker per cycle, most negative first. A bad news day
# can return forty articles, and forty near-identical candidates is a dedup
# workload, not information.
NEWS_MAX_CANDIDATES_PER_TICKER: int = 5


# ── Macro severity ──────────────────────────────────────
# ══ PERCENT CHANGE IS THE WRONG MEASURE FOR HALF OF THESE ══
#   DFF, UNRATE, DGS10 and T10Y2Y are already expressed IN PERCENT. A 25bp
#   policy move is 0.25 in the series' own units but ~6% in relative terms,
#   while the same 6% relative move on the CPI index would be a historic
#   inflation shock. One threshold cannot cover both, so each series declares
#   which measure applies.
#
#   ("abs", med, high)  compare the change in the series' own units
#   ("pct", med, high)  compare the percent change in the level
MACRO_THRESHOLDS: dict[str, tuple[str, float, float]] = {
    "DFF": ("abs", 0.20, 0.45),  # 25bp is a decision, 50bp is a shock
    "UNRATE": ("abs", 0.20, 0.40),  # 0.4pp in a month is recessionary
    "DGS10": ("abs", 0.20, 0.40),
    "T10Y2Y": ("abs", 0.15, 0.30),
    "CPIAUCSL": ("pct", 0.30, 0.60),  # m/m; 0.6% m/m is ~7% annualised
}

# Level crossings that are events in themselves, independent of magnitude.
# The yield curve inverting is the canonical example: the move that takes
# T10Y2Y from +0.02 to -0.01 is trivially small and historically significant.
#
# ``(series_id, direction)`` where direction is the sign of the crossing.
# Recorded as a CROSSING, not a state, so it fires once rather than every
# cycle for however many months the curve stays inverted.
MACRO_LEVEL_CROSSINGS: dict[str, float] = {"T10Y2Y": 0.0}


# ── Cycle behaviour ─────────────────────────────────────
# Cycles whose full dedup score distribution is logged at INFO. If p90 sits
# above TAU_LOW, canonicalization has collapsed — every alert is reading as
# similar to every other, which means the summaries are too generic to
# discriminate. That is a real diagnostic and it is only visible early.
DIAGNOSTIC_CYCLES: int = 3

# Hard ceiling on candidates processed in one cycle. A data-source glitch that
# marks every price bar as a 40% move must not turn into 500 embeddings and
# 500 alerts.
MAX_CANDIDATES_PER_CYCLE: int = 60


# ── Prompts ─────────────────────────────────────────────
# The ONLY LLM call in the whole subsystem. It writes the qualitative summary
# that gets embedded for dedup — see src/monitor/synthesizer.py for why the
# numbers must be stripped before embedding.
CANONICAL_SUMMARY_PROMPT_SYSTEM = """You write one-line canonical descriptions of financial \
events, used to recognise when two alerts describe the SAME event.

Your output is not shown to a human. It is embedded and compared against other
descriptions, so it must describe the SITUATION, never the MAGNITUDE.

ABSOLUTE RULES:
1. NEVER include a number, percentage, price, date, quarter, or fiscal year.
   "fell 5.2% to $210.11 on Monday" must become "sharp single-day decline".
   A number in your output makes two reports of the same event look different,
   which is the exact failure this description exists to prevent.
2. Describe WHAT HAPPENED and WHY IT MATTERS, in that order, in one clause each.
3. Be specific about KIND and vague about SIZE. "departure of the chief
   financial officer" is right; "a significant executive change" is too vague
   to tell two different events apart, and "the CFO resigned on the 3rd" is too
   specific to match the same event reported differently.
4. No ticker symbol and no company name — those are attached separately and
   filtered exactly.
5. Lower case, no trailing period, at most 15 words.

Examples:
  sharp single-day decline breaking below the 20-day moving average on elevated volume
  8-K reporting departure of the principal financial officer
  auditor notified the company that previously issued financials should not be relied upon
  multiple outlets reporting a regulatory investigation into sales practices
  federal funds rate raised at the scheduled policy meeting"""

CANONICAL_SUMMARY_PROMPT_USER = """Write one canonical description for each numbered event, \
in the same order.

{events}

Return exactly {count} description(s)."""


# ── Dispatch (Phase 7) ──────────────────────────────────
# Where a FIRED (or approved) alert is actually delivered. Comma-separated so
# a deploy can run several at once — console and file need no configuration,
# which is why they are the default.
NOTIFICATION_SINKS: tuple[str, ...] = tuple(
    s.strip().lower() for s in os.getenv("NOTIFICATION_SINKS", "console,file").split(",") if s.strip()
)

# Durable, greppable notification record independent of SQLite — a `tail -f`
# story for a process nobody is watching interactively.
ALERTS_LOG_PATH_NAME: str = "alerts.log"

# Gmail SMTP via an app password. Off by default: NOTIFICATION_SINKS can list
# "email" without EMAIL_ENABLED=true, and dispatch.py treats that as a sink
# FAILURE rather than a silent no-op — a misconfigured sink that reported
# success would let dispatch_alert claim an alert was delivered when it was
# not, which is worse than a loud error in the log.
EMAIL_ENABLED: bool = os.getenv("EMAIL_ENABLED", "false").lower() in {"1", "true", "yes"}
EMAIL_FROM: str = os.getenv("EMAIL_FROM", "")
EMAIL_TO: str = os.getenv("EMAIL_TO", "")
EMAIL_APP_PASSWORD: str = os.getenv("EMAIL_APP_PASSWORD", "")
EMAIL_SMTP_HOST: str = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
EMAIL_SMTP_PORT: int = int(os.getenv("EMAIL_SMTP_PORT", "465"))


# ── Scheduler (Phase 7) ─────────────────────────────────
# Off by default so `make api` does not silently start hitting EDGAR,
# Finnhub, and yfinance on a timer the first time someone runs the server.
# Enable once the watchlist has been warmed up — see run_monitor.sh --once
# --warmup — and see src/monitor/scheduler.py for why it runs cadence off
# MONITOR_CADENCE_HOURS inside the API process rather than a separate cron.
MONITOR_SCHEDULER_ENABLED: bool = os.getenv("MONITOR_SCHEDULER_ENABLED", "false").lower() in {"1", "true", "yes"}
