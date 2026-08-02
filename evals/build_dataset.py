# ═══════════════════════════════════════════════════════
# FinSight — Golden Dataset Builder
# ═══════════════════════════════════════════════════════
#
# Purpose : Turn the hand-authored question spec into
#           evals/datasets/research_golden.jsonl, resolving every expected
#           figure straight from SEC XBRL companyfacts.
#
# Public API:
#   GOLDEN_SPEC          the 40 questions, as module constants
#   resolve_fact(...)    one exact filed figure, with its accession number
#   build_examples()     spec -> list[GoldenExample]
#   main()               write the .jsonl
#
# Usage:
#   python -m evals.build_dataset            # rebuild and diff
#   python -m evals.build_dataset --check    # verify the committed file matches
#
# ══ WHY GROUND TRUTH IS RESOLVED, NOT TYPED ══
#   Typing 60 nine-digit figures by hand guarantees a transcription error, and
#   a wrong expected value is worse than no evaluator: it reports failures that
#   are the dataset's fault and sends you debugging working code.
#
# ══ WHY THIS DOES NOT CALL src.data.fundamentals ══
#   That module is under test. It applies its own concept-preference and
#   period-labelling rules, and if those rules are wrong the dataset would
#   inherit the same wrong answers and score them correct. So this reads
#   companyfacts directly and keys facts on their OWN period_end — the only
#   field that identifies which year a figure actually covers.
#
#   companyfacts' `fiscal_year` is the fiscal year of the FILING, not of the
#   fact. A 10-K restates two prior years in its comparative income statement
#   and stamps all three with the filing's year, so keying on `fiscal_year`
#   silently collapses three different years into one. That is the single
#   sharpest edge in the XBRL API and it is why this file exists.
#
# ══ WHY MACRO QUESTIONS CARRY NO EXPECTED FIGURE ══
#   A filed figure is immutable — Apple's FY2025 revenue will be $416.161B
#   forever. The Fed funds rate is not: pinning today's value into a committed
#   dataset means the eval starts failing next month for a reason that has
#   nothing to do with the system. Macro questions are therefore scored on
#   citation coverage, source validity, and the judges — never on a number.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, TypedDict

from src.data.edgar import get_company_facts, resolve_cik
from src.data.schemas import XBRLFact

logger = logging.getLogger(__name__)

# US-GAAP concepts, in preference order per metric. Filers differ in which tag
# they use and SWITCH between years, so every candidate is collected and the
# one covering the requested period wins — not the first one with any data.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    "net_income": ("NetIncomeLoss",),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "rd_expense": ("ResearchAndDevelopmentExpense",),
    "total_assets": ("Assets",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}

# An annual fact's period must span roughly a year. 10-Ks also carry quarterly
# durations tagged FY, and a quarter passed off as a year is a 4x error.
ANNUAL_MIN_DAYS: int = 340
ANNUAL_MAX_DAYS: int = 400


class ExpectedFact(TypedDict):
    """One exact filed figure the answer is expected to contain."""

    label: str
    value: float
    unit: str
    source_type: str
    source_id: str


class GoldenExample(TypedDict):
    """One dataset row: the question, and everything needed to grade it."""

    question: str
    archetype: str
    tickers: list[str]
    answerable: bool
    expected_facts: list[ExpectedFact]
    expected_sources: list[str]
    reference_answer: str
    notes: str


# ── Fact resolution ─────────────────────────────────────
def _duration_days(fact: XBRLFact) -> int:
    """Length of the period a fact covers, in days. 0 for instant facts."""
    from datetime import date

    start = fact.get("period_start")
    if not start:
        return 0
    return (date.fromisoformat(fact["period_end"]) - date.fromisoformat(start)).days


def _is_annual(fact: XBRLFact) -> bool:
    """
    True for a fact covering a full fiscal year, or a year-end balance.

    Balance-sheet concepts (total assets) are instants with no period_start, so
    duration cannot filter them — the 10-K form type already does.
    """
    if fact["form_type"] != "10-K":
        return False
    days = _duration_days(fact)
    return days == 0 or ANNUAL_MIN_DAYS <= days <= ANNUAL_MAX_DAYS


def resolve_fact(ticker: str, metric: str, fiscal_year: int) -> ExpectedFact | None:
    """
    Resolve one exact filed figure from XBRL companyfacts.

    The fiscal year of a fact is taken from the calendar year of its own
    ``period_end``, which is how all five watchlist companies label their years
    (NVDA's FY2026 ends January 2026, Apple's FY2025 ends September 2025).

    When several filings report the same period — the original 10-K and the two
    later ones carrying it as a comparative — the EARLIEST filing wins. That is
    the filing that first reported the figure, so it is the accession number a
    correct answer should cite.

    Parameters
    ----------
    ticker : str
        US-listed symbol.
    metric : str
        Key of ``CONCEPTS``.
    fiscal_year : int
        Fiscal year, as the company labels it.

    Returns
    -------
    ExpectedFact or None
        None when the company does not report the concept at all — which is a
        legitimate answer for a bank asked about gross profit, and is exactly
        what the unanswerable archetype is built on.
    """
    cik = resolve_cik(ticker)
    facts = get_company_facts(cik, concepts=list(CONCEPTS[metric]))

    candidates: list[XBRLFact] = []
    for concept in CONCEPTS[metric]:
        for fact in facts.get(concept, []):
            if _is_annual(fact) and int(fact["period_end"][:4]) == fiscal_year:
                candidates.append(fact)

    if not candidates:
        return None

    # Earliest filing that reported this period.
    fact = min(candidates, key=lambda f: (f["filed_date"], f["accession_no"]))

    return ExpectedFact(
        label=f"{ticker.upper()} FY{fiscal_year} {metric}",
        value=fact["value"],
        unit=fact["unit"],
        source_type="EDGAR",
        source_id=fact["accession_no"],
    )


# ── The spec ────────────────────────────────────────────
# (question, archetype, tickers, [(ticker, metric, fy), ...], reference, notes)
#
# Only the five ingested tickers appear: AAPL, MSFT, NVDA, GOOGL, JPM. A
# question about a company whose filings were never ingested measures the
# corpus, not the graph.
FactRef = tuple[str, str, int]
SpecRow = tuple[str, str, list[str], list[FactRef], str, str]

GOLDEN_SPEC: tuple[SpecRow, ...] = (
    # ── single_metric (8) ───────────────────────────────
    (
        "What revenue did Apple report for fiscal year 2025?",
        "single_metric",
        ["AAPL"],
        [("AAPL", "revenue", 2025)],
        "Apple reported total net sales of $416.161 billion for fiscal 2025.",
        "The floor case: one figure, one filing, one citation.",
    ),
    (
        "What was Microsoft's net income in fiscal 2026?",
        "single_metric",
        ["MSFT"],
        [("MSFT", "net_income", 2026)],
        "Microsoft reported net income of $133.749 billion for fiscal 2026.",
        "",
    ),
    (
        "How much did NVIDIA spend on research and development in fiscal 2026?",
        "single_metric",
        ["NVDA"],
        [("NVDA", "rd_expense", 2026)],
        "NVIDIA reported research and development expense of $18.497 billion in fiscal 2026.",
        "",
    ),
    (
        "What were Alphabet's total assets at the end of fiscal 2025?",
        "single_metric",
        ["GOOGL"],
        [("GOOGL", "total_assets", 2025)],
        "Alphabet reported total assets of $595.281 billion at the end of fiscal 2025.",
        "Balance-sheet instant rather than a duration — different XBRL shape.",
    ),
    (
        "What was JPMorgan's diluted earnings per share in 2025?",
        "single_metric",
        ["JPM"],
        [("JPM", "eps_diluted", 2025)],
        "JPMorgan reported diluted earnings per share of $20.02 for 2025.",
        "Small-magnitude figure — the verifier must not confuse it with a percentage.",
    ),
    (
        "What was NVIDIA's revenue in fiscal year 2026?",
        "single_metric",
        ["NVDA"],
        [("NVDA", "revenue", 2026)],
        "NVIDIA reported revenue of $215.938 billion for fiscal 2026.",
        "NVDA switched revenue concepts after FY2022; a stale-concept bug shows up here first.",
    ),
    (
        "What was Apple's operating cash flow in fiscal 2025?",
        "single_metric",
        ["AAPL"],
        [("AAPL", "operating_cash_flow", 2025)],
        "Apple generated $111.482 billion of cash from operating activities in fiscal 2025.",
        "",
    ),
    (
        "What operating income did Microsoft report for fiscal 2025?",
        "single_metric",
        ["MSFT"],
        [("MSFT", "operating_income", 2025)],
        "Microsoft reported operating income of $128.528 billion for fiscal 2025.",
        "A prior year, so the answer must not drift to the latest filing.",
    ),
    # ── multi_source (8) ────────────────────────────────
    (
        "How did Apple's gross margin change between fiscal 2024 and fiscal 2025?",
        "multi_source",
        ["AAPL"],
        [
            ("AAPL", "gross_profit", 2024),
            ("AAPL", "revenue", 2024),
            ("AAPL", "gross_profit", 2025),
            ("AAPL", "revenue", 2025),
        ],
        "Apple's gross margin rose from about 46.2% in fiscal 2024 ($180.683B on $391.035B) "
        "to about 46.9% in fiscal 2025 ($195.201B on $416.161B).",
        "Every percentage here is DERIVED — it appears in no finding, so it tests the "
        "evidence index's ratio derivation directly.",
    ),
    (
        "What was NVIDIA's operating margin in fiscal 2026, and how does it compare to fiscal 2025?",
        "multi_source",
        ["NVDA"],
        [
            ("NVDA", "operating_income", 2026),
            ("NVDA", "operating_income", 2025),
        ],
        "NVIDIA's operating income grew from $81.453 billion in fiscal 2025 to "
        "$130.387 billion in fiscal 2026, on revenue of $130.497B and $215.938B "
        "respectively — an operating margin of roughly 62% in fiscal 2025 and 60% in fiscal 2026.",
        "Margin is flat-to-down while the absolute figure grows — an answer that only "
        "reports growth has missed the question.",
    ),
    (
        "How much of Alphabet's fiscal 2025 revenue went to research and development?",
        "multi_source",
        ["GOOGL"],
        [("GOOGL", "rd_expense", 2025), ("GOOGL", "revenue", 2025)],
        "Alphabet spent $61.087 billion on research and development in fiscal 2025.",
        "Ratio of two filed figures from the same filing.",
    ),
    (
        "What did Apple earn per diluted share in fiscal 2025, and how did net income change from 2024?",
        "multi_source",
        ["AAPL"],
        [
            ("AAPL", "eps_diluted", 2025),
            ("AAPL", "net_income", 2025),
            ("AAPL", "net_income", 2024),
        ],
        "Apple reported diluted EPS of $7.46 in fiscal 2025. Net income rose from "
        "$93.736 billion in fiscal 2024 to $112.010 billion in fiscal 2025.",
        "Mixes a per-share figure with two absolutes — three different magnitudes in one answer.",
    ),
    (
        "How did Microsoft's revenue and R&D spending both change from fiscal 2025 to fiscal 2026?",
        "multi_source",
        ["MSFT"],
        [
            ("MSFT", "revenue", 2025),
            ("MSFT", "revenue", 2026),
            ("MSFT", "rd_expense", 2025),
            ("MSFT", "rd_expense", 2026),
        ],
        "Microsoft's revenue grew from $281.724 billion in fiscal 2025 to $331.839 billion "
        "in fiscal 2026, while research and development expense rose from $32.488 billion "
        "to $35.562 billion.",
        "Four figures, two trends — the archetype where markers get attached to the wrong claim.",
    ),
    (
        "What is the current Federal Funds rate, and how has it moved recently?",
        "multi_source",
        [],
        [],
        "The answer should report the current effective federal funds rate from FRED "
        "series DFF with its observation date, and describe the recent direction of travel.",
        "No pinned figure — a live series cannot be ground truth in a committed dataset. "
        "Scored on coverage, source validity, and the judges.",
    ),
    (
        "How does NVIDIA's R&D spending compare to its operating income in fiscal 2026?",
        "multi_source",
        ["NVDA"],
        [("NVDA", "rd_expense", 2026), ("NVDA", "operating_income", 2026)],
        "NVIDIA spent $18.497 billion on research and development in fiscal 2026 against "
        "operating income of $130.387 billion — R&D is roughly 14% of operating income.",
        "",
    ),
    (
        "What was JPMorgan's net income in 2025 relative to its total assets?",
        "multi_source",
        ["JPM"],
        [("JPM", "net_income", 2025), ("JPM", "total_assets", 2025)],
        "JPMorgan reported net income of $57.048 billion in 2025 on total assets of "
        "$4.4249 trillion — a return on assets of roughly 1.3%.",
        "Trillion-scale denominator against a billion-scale numerator; a scale slip is visible.",
    ),
    # ── cross_ticker (8) ────────────────────────────────
    (
        "Compare Apple's and Microsoft's most recent full-year revenue.",
        "cross_ticker",
        ["AAPL", "MSFT"],
        [("AAPL", "revenue", 2025), ("MSFT", "revenue", 2026)],
        "Apple reported $416.161 billion of revenue in fiscal 2025; Microsoft reported "
        "$331.839 billion in fiscal 2026. The fiscal years do not align.",
        "The fiscal calendars differ — an answer implying a like-for-like comparison is wrong.",
    ),
    (
        "Which of Apple, Microsoft, and NVIDIA had the highest net income in their most recent fiscal year?",
        "cross_ticker",
        ["AAPL", "MSFT", "NVDA"],
        [
            ("AAPL", "net_income", 2025),
            ("MSFT", "net_income", 2026),
            ("NVDA", "net_income", 2026),
        ],
        "Microsoft, at $133.749 billion in fiscal 2026, ahead of NVIDIA's $120.067 billion "
        "in fiscal 2026 and Apple's $112.010 billion in fiscal 2025.",
        "Three tickers x fundamentals = a three-branch fan-out that must not cross-attribute.",
    ),
    (
        "Compare R&D spending at Alphabet and Microsoft in their latest fiscal years.",
        "cross_ticker",
        ["GOOGL", "MSFT"],
        [("GOOGL", "rd_expense", 2025), ("MSFT", "rd_expense", 2026)],
        "Alphabet spent $61.087 billion on research and development in fiscal 2025; "
        "Microsoft spent $35.562 billion in fiscal 2026.",
        "",
    ),
    (
        "How do Apple's and Alphabet's total assets compare?",
        "cross_ticker",
        ["AAPL", "GOOGL"],
        [("AAPL", "total_assets", 2025), ("GOOGL", "total_assets", 2025)],
        "Alphabet held $595.281 billion of total assets at the end of fiscal 2025, "
        "against Apple's $359.241 billion.",
        "",
    ),
    (
        "Compare diluted EPS for JPMorgan and Alphabet in their most recent fiscal years.",
        "cross_ticker",
        ["JPM", "GOOGL"],
        [("JPM", "eps_diluted", 2025), ("GOOGL", "eps_diluted", 2025)],
        "JPMorgan reported diluted EPS of $20.02 for 2025; Alphabet reported $10.81 for fiscal 2025.",
        "Two small figures of similar magnitude — the evidence index must not ground one with the other.",
    ),
    (
        "Which grew revenue faster over its last two fiscal years, Apple or Microsoft?",
        "cross_ticker",
        ["AAPL", "MSFT"],
        [
            ("AAPL", "revenue", 2024),
            ("AAPL", "revenue", 2025),
            ("MSFT", "revenue", 2025),
            ("MSFT", "revenue", 2026),
        ],
        "Microsoft grew faster: revenue rose from $281.724 billion to $331.839 billion "
        "(about 17.8%), against Apple's $391.035 billion to $416.161 billion (about 6.4%).",
        "Both growth rates are derived; neither appears in any finding.",
    ),
    (
        "Compare operating cash flow at Apple and Alphabet for their latest fiscal years.",
        "cross_ticker",
        ["AAPL", "GOOGL"],
        [("AAPL", "operating_cash_flow", 2025), ("GOOGL", "operating_cash_flow", 2025)],
        "Alphabet generated $164.713 billion of operating cash flow in fiscal 2025, "
        "against Apple's $111.482 billion in fiscal 2025.",
        "",
    ),
    (
        "How does NVIDIA's most recent revenue compare with Alphabet's?",
        "cross_ticker",
        ["NVDA", "GOOGL"],
        [("NVDA", "revenue", 2026), ("GOOGL", "revenue", 2025)],
        "NVIDIA reported $215.938 billion of revenue in fiscal 2026; Alphabet reported "
        "$402.836 billion in fiscal 2025.",
        "Also exercises the NVDA concept-switch path, under a comparison.",
    ),
    # ── narrative (8) ───────────────────────────────────
    (
        "What supply chain risks does Apple flag in its most recent annual report?",
        "narrative",
        ["AAPL"],
        [],
        "Apple's risk factors describe dependence on outsourced manufacturing and "
        "component suppliers concentrated in a small number of locations, single-source "
        "components, and exposure to disruption from geopolitical events and natural disasters.",
        "Pure Item 1A retrieval. Stage 1 has almost nothing to check — this is the judges' archetype.",
    ),
    (
        "How does NVIDIA describe its competitive landscape in its latest 10-K?",
        "narrative",
        ["NVDA"],
        [],
        "NVIDIA describes competition from other semiconductor and platform companies, "
        "from customers developing their own in-house silicon, and from alternative "
        "computing architectures.",
        "",
    ),
    (
        "What does Microsoft say about the risks of its AI investments?",
        "narrative",
        ["MSFT"],
        [],
        "Microsoft's risk factors discuss AI-related risks including datacenter and "
        "capital expenditure commitments, model quality and safety issues, evolving "
        "regulation, and intellectual property and competitive uncertainty.",
        "",
    ),
    (
        "What regulatory and legal proceedings does Alphabet disclose?",
        "narrative",
        ["GOOGL"],
        [],
        "Alphabet discloses antitrust proceedings brought by the U.S. Department of "
        "Justice and state attorneys general, European Commission actions, and privacy "
        "and consumer-protection investigations across multiple jurisdictions.",
        "Item 3 rather than Item 1A — a different section filter.",
    ),
    (
        "What credit and market risks does JPMorgan highlight in its annual report?",
        "narrative",
        ["JPM"],
        [],
        "JPMorgan discusses credit risk from its lending and wholesale exposures, market "
        "risk in its trading portfolios, interest-rate risk, liquidity risk, and the "
        "regulatory capital requirements that constrain them.",
        "A bank's risk narrative is unlike a technology company's — tests corpus breadth.",
    ),
    (
        "How does Apple describe its dependence on the iPhone?",
        "narrative",
        ["AAPL"],
        [],
        "Apple states that the iPhone accounts for a large majority of net sales and that "
        "its business is substantially dependent on continued iPhone demand.",
        "A specific claim the filing states plainly — a judge should score this confidently.",
    ),
    (
        "What does NVIDIA say about export controls affecting its business?",
        "narrative",
        ["NVDA"],
        [],
        "NVIDIA discusses U.S. government export licensing requirements on advanced "
        "computing products to China and other markets, and describes the resulting "
        "revenue impact and compliance uncertainty.",
        "A named, checkable disclosure — cited-but-irrelevant is easy to spot here.",
    ),
    (
        "What does Microsoft disclose about competition in cloud services?",
        "narrative",
        ["MSFT"],
        [],
        "Microsoft describes intense competition in cloud infrastructure and productivity "
        "services from other large-scale providers, and notes pricing pressure and the "
        "capital intensity of competing at scale.",
        "",
    ),
    # ── unanswerable (8) ────────────────────────────────
    (
        "What gross profit did JPMorgan report in 2025?",
        "unanswerable",
        ["JPM"],
        [],
        "JPMorgan does not report a gross profit line. Banks present net interest income "
        "and noninterest revenue rather than a cost-of-sales gross margin, and the concept "
        "is absent from its XBRL filings.",
        "REAL absence, not a trick: the US-GAAP GrossProfit tag genuinely does not appear "
        "in JPM's filings. A confident figure here is fabrication.",
    ),
    (
        "What was Alphabet's gross profit in fiscal 2025?",
        "unanswerable",
        ["GOOGL"],
        [],
        "Alphabet does not tag a GrossProfit concept in its XBRL filings, so the figure "
        "is not available from the filed data.",
        "Also genuinely absent — Alphabet reports cost of revenues without a gross profit subtotal.",
    ),
    (
        "What will Apple's revenue be in fiscal 2030?",
        "unanswerable",
        ["AAPL"],
        [],
        "That is a forecast. No filed figure exists for a future period and none should " "be produced.",
        "The temptation is a trend extrapolation presented with a citation on it.",
    ),
    (
        "How many employees does Tesla have?",
        "unanswerable",
        ["TSLA"],
        [],
        "Tesla is outside the ingested universe and its filings have not been retrieved, "
        "so the question cannot be answered from this system's data.",
        "Out-of-universe ticker. The correct answer names the gap rather than reaching " "for general knowledge.",
    ),
    (
        "What did Microsoft's CEO say on the fiscal 2026 fourth-quarter earnings call?",
        "unanswerable",
        ["MSFT"],
        [],
        "Earnings call transcripts are not among this system's sources, which cover SEC "
        "filings, XBRL facts, FRED series, prices, and news.",
        "A source type that does not exist here — the answer must say which sources it has.",
    ),
    (
        "What is NVIDIA's market share in datacenter GPUs?",
        "unanswerable",
        ["NVDA"],
        [],
        "Market share is not a filed figure. NVIDIA does not report it in its financial "
        "statements and no retrieved source supplies it.",
        "A widely-known number that appears in no source — the strongest pull toward "
        "answering from parametric memory.",
    ),
    (
        "What was Apple's revenue in fiscal 1985?",
        "unanswerable",
        ["AAPL"],
        [],
        "EDGAR's structured XBRL data does not reach back to 1985, so the figure is not "
        "available from the filed data this system retrieves.",
        "Before EDGAR's coverage. Plausible-sounding and unretrievable.",
    ),
    (
        "Should I buy NVIDIA stock?",
        "unanswerable",
        ["NVDA"],
        [],
        "That is investment advice rather than a research question. The system reports "
        "filed figures and disclosures; it does not make recommendations.",
        "Out of scope by design, not by data gap. Included because a research assistant "
        "that answers this is a liability.",
    ),
)


# ── Build ───────────────────────────────────────────────
def build_examples() -> list[GoldenExample]:
    """
    Resolve the spec into dataset rows.

    Returns
    -------
    list of GoldenExample

    Raises
    ------
    RuntimeError
        If a fact the spec expects to exist cannot be resolved. Failing loudly
        is the point — a silently-dropped expected value would make the
        evaluator report a perfect score on a question it stopped checking.
    """
    examples: list[GoldenExample] = []

    for question, archetype, tickers, fact_refs, reference, notes in GOLDEN_SPEC:
        facts: list[ExpectedFact] = []
        for ticker, metric, fiscal_year in fact_refs:
            fact = resolve_fact(ticker, metric, fiscal_year)
            if fact is None:
                raise RuntimeError(
                    f"{question!r}: expected {ticker} FY{fiscal_year} {metric} but XBRL has no such fact. "
                    "Either the spec is wrong or the company does not report the concept."
                )
            facts.append(fact)

        # De-duplicated, order-stable: several facts often share one filing.
        sources: list[str] = []
        for fact in facts:
            if fact["source_id"] not in sources:
                sources.append(fact["source_id"])

        examples.append(
            GoldenExample(
                question=question,
                archetype=archetype,
                tickers=tickers,
                answerable=archetype != "unanswerable",
                expected_facts=facts,
                expected_sources=sources,
                reference_answer=reference,
                notes=notes,
            )
        )

    return examples


def write_jsonl(examples: list[GoldenExample], path: Any) -> None:
    """Write examples one-per-line, sorted keys, so the file diffs cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")


def load_jsonl(path: Any) -> list[GoldenExample]:
    """Read a golden dataset file."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    """Rebuild the golden dataset from XBRL, or verify the committed copy."""
    from evals.config import GOLDEN_PATH
    from src.core.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Build the research golden dataset")
    parser.add_argument("--check", action="store_true", help="verify the committed file is up to date")
    args = parser.parse_args(argv)

    configure_logging()
    examples = build_examples()

    if args.check:
        if not GOLDEN_PATH.exists():
            print(f"MISSING {GOLDEN_PATH}")
            return 1
        if load_jsonl(GOLDEN_PATH) != examples:
            print(f"STALE {GOLDEN_PATH} — rerun: python -m evals.build_dataset")
            return 1
        print(f"OK {GOLDEN_PATH} ({len(examples)} examples)")
        return 0

    write_jsonl(examples, GOLDEN_PATH)

    counts: dict[str, int] = {}
    for example in examples:
        counts[example["archetype"]] = counts.get(example["archetype"], 0) + 1

    print(f"\nWrote {len(examples)} examples -> {GOLDEN_PATH}")
    for archetype in sorted(counts):
        print(f"  {archetype:16s} {counts[archetype]:>3d}")
    print(f"  {'expected facts':16s} {sum(len(e['expected_facts']) for e in examples):>3d}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
