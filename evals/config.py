# ═══════════════════════════════════════════════════════
# FinSight — Evaluation Configuration
# ═══════════════════════════════════════════════════════
#
# Purpose : Dataset names, archetypes, scoring thresholds, and the quota
#           estimator. Mirrors the per-domain config.py pattern — prompts and
#           tunables as module constants, nothing here imports an LLM.
#
# Public API:
#   DATASET_RESEARCH, EVAL_PROJECT, ARCHETYPES
#   GOLDEN_PATH, MAX_CONCURRENCY
#   LLM_CALLS_PER_EXAMPLE, estimate_run(n) -> RunEstimate
#   JUDGE_PROMPT_SYSTEM / _USER
#   DEDUP_GOLDEN_PATH, SWEEP_LOW, SWEEP_HIGH, SWEEP_STEP, MIN_SUPPRESSION_PRECISION  (Suite B)
#
# ══ WHY AN ESTIMATOR LIVES IN CONFIG ══
#   An eval run is the single largest quota spike in this project: 40 examples
#   times ~6 LLM calls is more traffic than a week of interactive use. On the
#   Vertex backend that is billed per token with no free tier, so a casual
#   re-run is a real charge. run_evals.sh prints this estimate and waits for
#   confirmation before spending anything.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from src.core.config import PROJECT_ROOT

# ── Dataset identity ────────────────────────────────────
# The LangSmith dataset name is stable across experiments on purpose: every
# experiment must run against the SAME examples or the comparison view is
# comparing datasets rather than changes.
DATASET_RESEARCH: str = "finsight-research-golden"

# Experiments go to their own LangSmith project so eval history is not buried
# under interactive CLI traffic in finsight-dev.
EVAL_PROJECT: str = "finsight-eval"

EVALS_DIR: Path = PROJECT_ROOT / "evals"
GOLDEN_PATH: Path = EVALS_DIR / "datasets" / "research_golden.jsonl"
RESULTS_DOC: Path = PROJECT_ROOT / "docs" / "eval_results.md"

# ── Suite B: the dedup threshold sweep ──────────────────
# Hand-authored canonical-style text, clustered by the real-world event each
# paraphrases — NOT LLM-generated, NOT pulled from a live run (the system has
# not yet produced enough real HIGH-severity volume to fill 60+ pairs). See
# evals/run_monitor_eval.py for how clusters become labelled pairs.
DEDUP_GOLDEN_PATH: Path = EVALS_DIR / "datasets" / "dedup_golden.jsonl"

# The exact commitment docs/dedup_algorithm.md made: sweep TAU_HIGH across
# this range and pick the LOWEST value clearing the precision floor — not
# max F1, because a false suppress costs more than a false fire.
SWEEP_LOW: float = 0.70
SWEEP_HIGH: float = 0.99
SWEEP_STEP: float = 0.01
MIN_SUPPRESSION_PRECISION: float = 0.97

# ── Archetypes ──────────────────────────────────────────
# Five question shapes that fail in different ways. Splitting the score by
# archetype is what turns "coverage is 0.78" into "coverage is 0.78 because
# cross-ticker comparisons are at 0.51" — one is a number, the other is a lead.
ARCHETYPES: dict[str, str] = {
    "single_metric": (
        "One filed figure, one source. The floor case — if this archetype is not "
        "near-perfect, nothing downstream can be trusted."
    ),
    "multi_source": (
        "Two or more specialists must both contribute and the synthesizer must "
        "keep their citations distinct. Where markers get attached to the wrong claim."
    ),
    "cross_ticker": (
        "The same metric across companies. Exercises the (agent x ticker) fan-out, "
        "and is where a number from one ticker gets attributed to another."
    ),
    "narrative": (
        "Qualitative disclosure from filings. Stage 1 has almost nothing to check, "
        "so this archetype is what the LLM judge exists for."
    ),
    "unanswerable": (
        "The data genuinely does not exist — a bank has no gross profit line, a "
        "filing predates the event. Scored INVERTED: saying so is the correct answer, "
        "and producing a confident figure is the worst possible failure."
    ),
}

# ── Scoring ─────────────────────────────────────────────
# A run is "passing" at this coverage. Not a target to tune toward — it is the
# line below which the answer stops being safe to show a user.
COVERAGE_PASS_THRESHOLD: float = 0.95

# Ground-truth numeric comparison. Deliberately the SAME tolerance the verifier
# uses: an evaluator stricter than the system under test reports failures the
# system was never designed to avoid.
FACT_REL_TOLERANCE: float = 0.005

# Hedging vocabulary the unanswerable archetype is scored on, matched
# case-insensitively against the final answer.
#
# A phrase list is a PROXY, and a deliberately generous one. English has
# unbounded ways to decline — "no filed revenue figure is available" and "the
# figure is not available" mean the same thing and share no substring — so the
# list errs toward recall: missing a real refusal scores a correct answer zero,
# which is the more damaging error. The authoritative refusal score is
# answer_correctness, whose reference for these eight examples is itself a
# refusal; this one is the free check that runs first.
REFUSAL_MARKERS: tuple[str, ...] = (
    "not available",
    "is available",
    "unavailable",
    "no data",
    "no filed",
    "no such",
    "not report",
    "does not",
    "did not",
    "do not",
    "cannot",
    "can not",
    "could not",
    "not disclose",
    "not among",
    "not present",
    "not provide",
    "no findings",
    "not been",
    "unable to",
    "insufficient",
    "outside",
    "beyond the scope",
    "not a filed",
    "forecast",
    "investment advice",
)

# ── Run economics ───────────────────────────────────────
# Measured from Phase 4 runs, not guessed:
#   1 router (flash) + ~2.5 specialist branches (flash, some make 0 LLM calls)
#   + 1 synthesis (pro) + ~0.6 qualitative judge (pro, only when there are
#   qualitative claims) + ~0.3 repair round-trip.
LLM_CALLS_PER_EXAMPLE: float = 6.0

# Two graders per example, both `pro`, and only on examples that produce an
# answer at all.
JUDGE_CALLS_PER_EXAMPLE: float = 2.0

# Rough per-example cost on Vertex at gemini-2.5-pro list price, dominated by
# the two pro calls. Flash traffic rounds to nothing next to it. Revisit if the
# published price moves — this is for a warning banner, not accounting.
USD_PER_EXAMPLE: float = 0.05

# Concurrent examples. The per-tier InMemoryRateLimiter in src/core/llm.py is
# the real throttle; this only bounds how many graphs are in flight at once, so
# a failure does not take 40 half-finished runs with it.
MAX_CONCURRENCY: int = 4


class RunEstimate(TypedDict):
    """What one eval run is expected to cost before it is allowed to start."""

    examples: int
    llm_calls: int
    judge_calls: int
    usd: float
    minutes: float


def estimate_run(examples: int, *, judges: bool = True) -> RunEstimate:
    """
    Estimate the quota, spend, and wall time of an eval run.

    Parameters
    ----------
    examples : int
        Number of dataset examples the run will cover.
    judges : bool, default True
        Whether the two LLM-judge evaluators are enabled. Turning them off
        roughly halves the spend and leaves the deterministic metrics intact,
        which is the right setting for a smoke run.

    Returns
    -------
    RunEstimate
    """
    llm_calls = round(examples * LLM_CALLS_PER_EXAMPLE)
    judge_calls = round(examples * JUDGE_CALLS_PER_EXAMPLE) if judges else 0
    usd = examples * USD_PER_EXAMPLE * (1.0 if judges else 0.55)

    # ~35s per example end-to-end, divided by the concurrency ceiling.
    minutes = (examples * 35.0) / max(MAX_CONCURRENCY, 1) / 60.0

    return RunEstimate(
        examples=examples,
        llm_calls=llm_calls,
        judge_calls=judge_calls,
        usd=round(usd, 2),
        minutes=round(minutes, 1),
    )


# ── LLM-judge prompts ───────────────────────────────────
# Deliberately narrow. A judge asked "is this a good answer?" returns an
# aesthetic opinion; a judge asked one factual question with a fixed verdict
# vocabulary returns something you can put in a table.

FAITHFULNESS_PROMPT_SYSTEM = """You are auditing a financial answer for CITATION FAITHFULNESS.

You are given an answer containing inline source markers of the form
[SRC:<TYPE>:<ID>], and the findings the system actually retrieved.

Judge ONE thing: is each marker attached to a claim that its source genuinely
supports? You are NOT judging whether the claim is true in the world, whether
the writing is good, or whether the answer is complete.

The specific failure you exist to catch is CITED BUT IRRELEVANT — a real,
correctly-formatted source attached to a claim it does not back. A deterministic
checker cannot see this, because the marker resolves and the number matches
something; only reading both can tell.

Score 0 to 1:
  1.0  every marker supports its claim.
  0.5  markers are broadly right but at least one is attached to the wrong claim.
  0.0  markers are decorative — they resolve, but do not support what they follow.

Return the score and one sentence of reasoning. Be strict: when a marker's
source has nothing to do with its sentence, that is 0.0, not 0.5."""

FAITHFULNESS_PROMPT_USER = """Question:
{question}

Answer under audit:
{answer}

Findings the system actually retrieved:
{findings}

Score citation faithfulness."""


CORRECTNESS_PROMPT_SYSTEM = """You are grading a financial answer against a reference answer.

Judge whether the answer conveys the same substance as the reference. Wording,
ordering, and extra detail do not matter. A different NUMBER does matter. A
missing part of a multi-part question matters.

Score 0 to 1:
  1.0  same substance, no contradictions, nothing important missing.
  0.5  partially correct — one part right and one part missing or wrong.
  0.0  contradicts the reference, or does not answer the question.

If the reference says the data is unavailable and the answer supplies a
confident figure, that is 0.0. Refusing to answer when the reference refuses is
1.0 — declining to invent data is the correct behaviour, not a failure.

Return the score and one sentence of reasoning."""

CORRECTNESS_PROMPT_USER = """Question:
{question}

Reference answer:
{reference}

Answer under grading:
{answer}

Score answer correctness."""
