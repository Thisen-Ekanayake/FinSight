# ═══════════════════════════════════════════════════════
# FinSight — Citation Verifier
# ═══════════════════════════════════════════════════════
#
# Purpose : Check that every number in the drafted answer traces back to a
#           value some tool actually returned, and that every inline source
#           marker resolves to a real citation.
#
# Public API:
#   extract_numeric_claims(text)      -> list[NumericClaim]
#   build_evidence_index(findings)    -> list[Evidence]
#   verify(answer, findings, citations, selected_agents) -> VerificationReport
#   citation_verifier_node(state)
#
# ══ DETERMINISTIC FIRST, LLM SECOND ══
#   An LLM asked whether its own numbers are grounded is checking its work with
#   the faculty that produced the error. So stage 1 is pure code: regex every
#   number out of the answer, normalise it, and match it against the values the
#   tools returned within a 0.5% tolerance. No model is consulted and no quota
#   is spent. The LLM judge (stage 2) only handles qualitative assertions,
#   where there is no number to compare and code genuinely cannot decide.
#
# ══ WHY CLAIM TEXT COUNTS AS EVIDENCE ══
#   A finding carries one structured `value`, but its `claim` string often
#   contains several numbers — "AAPL closed at 254.43, +1.20% on the day".
#   Those strings are built by the specialist from tool output with an f-string;
#   no model writes them. Every number in a claim is therefore grounded by
#   construction, and the evidence index harvests them too. Without that the
#   verifier would reject figures the system itself produced correctly.
#
# ══ WHY DERIVED VALUES ARE EVIDENCE TOO ══
#   The synthesizer legitimately computes gross margin as gross_profit/revenue.
#   That percentage appears in no finding, so a verifier that knew only the raw
#   filed values would flag every ratio it ever wrote. The index therefore
#   derives the arithmetic an analyst would actually do — ratios between
#   figures from the same period, and period-over-period growth — and nothing
#   beyond it. A bounded derivation set is the point: allowing arbitrary
#   arithmetic would make any number reachable from any two others.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from itertools import permutations

from src.core.schemas import AgentFinding, Citation
from src.research.config import (
    IGNORE_BARE_INTEGERS_BELOW,
    MAX_DERIVATION_GROUP,
    MAX_EVIDENCE_CHARS,
    MAX_EVIDENCE_PER_CLAIM,
    NUMERIC_REL_TOLERANCE,
    PERCENT_ABS_TOLERANCE,
    VERIFY_QUALITATIVE_CLAIMS,
)
from src.research.state import ResearchState, UnsupportedClaim, VerificationReport

logger = logging.getLogger(__name__)

# ── Source marker grammar ───────────────────────────────
# The synthesis prompt mandates this exact shape. That mandate is what makes
# deterministic verification possible at all — a free-form "(source: Apple's
# 10-K)" cannot be resolved back to a citation record.
MARKER_RE = re.compile(r"\[SRC:([A-Z_]+):([^\]]+)\]")

# Format rules per source type. Catches a fabricated identifier that happens to
# look plausible — the failure mode where a model invents an accession number
# with the right number of digits.
SOURCE_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "EDGAR": re.compile(r"^\d{10}-\d{2}-\d{6}$"),
    "FRED": re.compile(r"^[A-Z0-9]{2,24}$"),
    "YFINANCE": re.compile(r"^[A-Z.\-]{1,10}@\d{4}-\d{2}-\d{2}$"),
}

# ── Masks applied before number extraction ──────────────
# Each masked span is replaced by spaces of equal length so character offsets
# into the original answer stay valid.
_MASKS: tuple[re.Pattern[str], ...] = (
    # Source markers first: accession numbers are full of digits.
    re.compile(r"\[SRC:[^\]]*\]"),
    re.compile(r"\b(?:FY|CY|fiscal(?:\s+year)?)\s*'?\d{2,4}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}\s+(?:FY|Q[1-4])\b", re.IGNORECASE),
    re.compile(r"\bQ[1-4](?:\s+\d{4})?\b", re.IGNORECASE),
    re.compile(r"\bH[12]\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # A bare year, but only where it cannot be part of a real figure: not
    # preceded by a digit, comma, dot or currency symbol, and not followed by
    # one. Without those guards this would eat the "2024" out of "2024.50".
    re.compile(r"(?<![\d.,$€£])\b(?:19|20)\d{2}\b(?![\d.,%])"),
)

# Scale suffixes, longest alternatives first so "million" is not consumed by
# the single-letter "M" branch.
_SCALE_FACTORS: dict[str, float] = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "million": 1e6,
    "b": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

_PERCENT_SCALES = {"%", "percent", "percentage point", "percentage points", "pp", "bps"}

NUMBER_RE = re.compile(
    r"""
    (?P<open>\()?                               # (2.1) is finance for -2.1
    (?P<sign>-\s*)?
    (?P<currency>[$€£])?\s*
    (?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    # The separator lives INSIDE the optional group. Outside it, a trailing
    # \s* would run past a masked source marker — which is now a run of
    # spaces — and the reported token text would swallow the next line.
    (?:\s{0,2}
      (?P<scale>
          %|percentage\ points?|percent|pp\b|bps\b
        | thousand|million|billion|trillion
        | [KMBT](?![A-Za-z])
      )
    )?
    (?P<close>\))?
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ── Records ─────────────────────────────────────────────
@dataclass
class NumericClaim:
    """
    One number lifted out of the drafted answer.

    ``kind`` separates the two things that must never be matched against each
    other: an absolute figure (416,161,000,000) and a percentage (46.91).
    """

    text: str
    value: float
    kind: str
    start: int
    end: int


@dataclass
class Evidence:
    """
    One number a tool actually produced, with the citations backing it.

    ``derived`` marks values the index computed rather than received — a ratio
    or a growth rate — so a report can distinguish "this was filed" from "this
    follows arithmetically from what was filed".
    """

    value: float
    kind: str
    label: str
    citations: list[Citation] = field(default_factory=list)
    derived: bool = False


# ── Stage 1a: pull the numbers out of the answer ────────
def _mask(text: str) -> str:
    """Blank out spans that contain digits but assert nothing, preserving offsets."""
    masked = text
    for pattern in _MASKS:
        masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)
    return masked


def _normalise(match: re.Match[str]) -> tuple[float, str] | None:
    """
    Turn one regex match into ``(value, kind)``.

    Returns None for tokens that are prose rather than financial claims.
    """
    raw = match.group("digits").replace(",", "")
    try:
        value = float(raw)
    except ValueError:  # pragma: no cover - the pattern cannot produce this
        return None

    scale = (match.group("scale") or "").lower().strip()
    currency = match.group("currency")

    if scale in _PERCENT_SCALES:
        # Basis points are a hundredth of a percentage point.
        return (value / 100.0 if scale == "bps" else value), "percent"

    if scale in _SCALE_FACTORS:
        return value * _SCALE_FACTORS[scale], "absolute"

    # No scale and no currency: keep only figures large enough to be a claim.
    # "over 5 sessions" and "the 10-year Treasury" are prose, and grounding
    # them produces noise and pointless repair attempts.
    if not currency and value < IGNORE_BARE_INTEGERS_BELOW:
        return None

    return value, "absolute"


def extract_numeric_claims(text: str) -> list[NumericClaim]:
    """
    Extract every groundable number from an answer.

    Parameters
    ----------
    text : str
        The drafted answer, including its inline ``[SRC:...]`` markers.

    Returns
    -------
    list of NumericClaim
        In document order. Source markers, fiscal periods, and dates are
        excluded — they carry digits but assert nothing about a value.
    """
    masked = _mask(text)
    claims: list[NumericClaim] = []

    for match in NUMBER_RE.finditer(masked):
        parsed = _normalise(match)
        if parsed is None:
            continue

        value, kind = parsed
        negative = bool(match.group("sign")) or (match.group("open") and match.group("close"))
        claims.append(
            NumericClaim(
                text=text[match.start() : match.end()].strip(),
                value=-value if negative else value,
                kind=kind,
                start=match.start(),
                end=match.end(),
            )
        )

    return claims


# ── Stage 1b: what the tools actually returned ──────────
def _split_metric(metric: str | None) -> tuple[str, str]:
    """Split ``revenue@2025 FY`` into ``("revenue", "2025 FY")``."""
    name, _, period = (metric or "").partition("@")
    return name, period


def _kind_for_unit(unit: str | None) -> str:
    """Classify a finding's unit as a percentage or an absolute quantity."""
    lowered = (unit or "").lower()
    return "percent" if "%" in lowered or "percent" in lowered else "absolute"


def _derive(structured: list[tuple[str, str, str, Evidence]]) -> list[Evidence]:
    """
    Compute the arithmetic an analyst would legitimately do.

    Two families only:

    * **Ratios** between absolute figures from the same ticker and period —
      this is gross margin, operating margin, and every other percentage the
      synthesizer computes rather than reads.
    * **Growth** in one metric between two periods.

    Parameters
    ----------
    structured : list of tuple
        ``(ticker, metric_name, period, evidence)`` for each structured value.

    Returns
    -------
    list of Evidence
        All marked ``derived=True``.
    """
    derived: list[Evidence] = []

    by_period: dict[tuple[str, str], list[tuple[str, Evidence]]] = {}
    by_metric: dict[tuple[str, str], list[tuple[str, Evidence]]] = {}

    for ticker, name, period, evidence in structured:
        if evidence.kind != "absolute":
            continue
        by_period.setdefault((ticker, period), []).append((name, evidence))
        by_metric.setdefault((ticker, name), []).append((period, evidence))

    for (ticker, period), members in by_period.items():
        if len(members) > MAX_DERIVATION_GROUP:
            logger.debug("Skipping ratio derivation for %s %s: %d members", ticker, period, len(members))
            continue
        for (name_a, ev_a), (name_b, ev_b) in permutations(members, 2):
            if not ev_b.value:
                continue
            derived.append(
                Evidence(
                    value=ev_a.value / ev_b.value * 100.0,
                    kind="percent",
                    label=f"{name_a}/{name_b} {period}".strip(),
                    citations=[*ev_a.citations, *ev_b.citations],
                    derived=True,
                )
            )

    for (ticker, name), members in by_metric.items():
        if len(members) > MAX_DERIVATION_GROUP:
            continue
        for (period_a, ev_a), (period_b, ev_b) in permutations(members, 2):
            if not ev_a.value:
                continue
            derived.append(
                Evidence(
                    value=(ev_b.value - ev_a.value) / abs(ev_a.value) * 100.0,
                    kind="percent",
                    label=f"{name} change {period_a}->{period_b}",
                    citations=[*ev_a.citations, *ev_b.citations],
                    derived=True,
                )
            )

    return derived


def build_evidence_index(findings: list[AgentFinding]) -> list[Evidence]:
    """
    Collect every number the system can legitimately assert.

    Three sources, in descending directness:

    1. Each finding's structured ``value`` — what the tool returned.
    2. Every number inside each finding's ``claim`` text, which the specialist
       built from tool output with an f-string. No model writes those strings.
    3. Ratios and growth rates derived from (1) — see ``_derive``.

    Parameters
    ----------
    findings : list of AgentFinding
        Everything the fan-out collected.

    Returns
    -------
    list of Evidence
    """
    index: list[Evidence] = []
    structured: list[tuple[str, str, str, Evidence]] = []

    for item in findings:
        citations = list(item["citations"])
        name, period = _split_metric(item.get("metric"))
        value = item.get("value")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            evidence = Evidence(
                value=float(value),
                kind=_kind_for_unit(item.get("unit")),
                label=item.get("metric") or item["agent"],
                citations=citations,
            )
            index.append(evidence)
            structured.append((item.get("ticker") or "", name, period, evidence))

        for claim in extract_numeric_claims(item["claim"]):
            index.append(
                Evidence(
                    value=claim.value,
                    kind=claim.kind,
                    label=f"{item['agent']} claim text",
                    citations=citations,
                )
            )

    index.extend(_derive(structured))
    logger.debug("Evidence index: %d values (%d derived)", len(index), sum(1 for e in index if e.derived))
    return index


# ── Stage 1c: match ─────────────────────────────────────
def _agrees(claim: NumericClaim, evidence: Evidence) -> bool:
    """
    Decide whether one number grounds another.

    Comparison is on MAGNITUDE, not signed value. Prose conveys direction with
    words — "fell 5.2%", "a decline of 5.2%", "(5.2)" — and no extractor
    recovers that reliably. Whether the answer got the direction right is a
    qualitative question, and qualitative questions are stage 2's job.
    """
    if claim.kind != evidence.kind:
        return False

    left, right = abs(claim.value), abs(evidence.value)
    difference = abs(left - right)

    if evidence.kind == "percent" and difference <= PERCENT_ABS_TOLERANCE:
        return True

    return difference / max(left, right, 1e-9) <= NUMERIC_REL_TOLERANCE


def find_support(claim: NumericClaim, index: list[Evidence]) -> Evidence | None:
    """Return the first evidence grounding ``claim``, preferring direct over derived."""
    for evidence in index:
        if not evidence.derived and _agrees(claim, evidence):
            return evidence
    for evidence in index:
        if evidence.derived and _agrees(claim, evidence):
            return evidence
    return None


# ── Stage 1d: source markers ────────────────────────────
def validate_source_markers(answer: str, citations: list[Citation]) -> list[str]:
    """
    Check every inline marker against the citations the tools produced.

    Two distinct failures, both reported:

    * The identifier is malformed for its type — a fabricated accession number
      with the wrong shape.
    * The identifier is well-formed but appears in no citation, meaning the
      model attached a source the system never retrieved.

    Parameters
    ----------
    answer : str
        The drafted answer.
    citations : list of Citation
        Every citation collected during the run.

    Returns
    -------
    list of str
        Human-readable descriptions of the invalid markers.
    """
    known = {(c["source_type"], c["source_id"]) for c in citations}
    invalid: list[str] = []

    for source_type, source_id in MARKER_RE.findall(answer):
        pattern = SOURCE_ID_PATTERNS.get(source_type)
        if pattern and not pattern.match(source_id):
            invalid.append(f"[SRC:{source_type}:{source_id}] — malformed {source_type} identifier")
        elif (source_type, source_id) not in known:
            invalid.append(f"[SRC:{source_type}:{source_id}] — no such source was retrieved")

    return invalid


# ── Repair targeting ────────────────────────────────────
# Which specialist to re-query for an ungrounded number, in preference order.
# A percentage the system could not derive is usually a missing filed figure,
# so fundamentals leads.
_NUMERIC_AGENT_PRIORITY: tuple[str, ...] = ("fundamentals", "technical", "macro")


def _infer_origin(selected_agents: list[str]) -> str:
    """
    Pick the specialist most likely to close an unsupported numeric claim.

    Restricted to agents the router actually selected: re-querying an agent
    that was never in the plan asks for data the query does not need, and for
    filings_rag it would search a ticker that may not be ingested.
    """
    for agent in _NUMERIC_AGENT_PRIORITY:
        if agent in selected_agents:
            return agent
    return selected_agents[0] if selected_agents else "fundamentals"


def _sentence_around(text: str, position: int) -> str:
    """Return the sentence containing ``position`` — the unit a repair asks about."""
    start = max(text.rfind(". ", 0, position), text.rfind("\n", 0, position)) + 1
    end_candidates = [i for i in (text.find(". ", position), text.find("\n", position)) if i != -1]
    end = min(end_candidates) + 1 if end_candidates else len(text)
    return text[start:end].strip()


# ── Stage 2: qualitative claims ─────────────────────────
# Sentence boundary: a terminator followed by whitespace and something that
# starts a new sentence. The lookahead is what keeps "$416.2B." and "Item 1A."
# from splitting mid-figure.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])|\n+")


@dataclass
class QualitativeClaim:
    """
    A cited sentence with nothing numeric in it.

    This is the shape stage 1 structurally cannot judge: there is no value to
    compare, only an assertion pointing at a source. It is also where the
    interesting failure lives — a real citation attached to a claim it does
    not actually support.
    """

    text: str
    source_ids: list[tuple[str, str]]


def extract_qualitative_claims(answer: str) -> list[QualitativeClaim]:
    """
    Find sentences that cite a source but assert nothing numeric.

    A sentence with no marker at all is not audited: the synthesis prompt
    requires markers on numeric claims, so an uncited sentence is framing or
    summary rather than a sourced assertion.

    Parameters
    ----------
    answer : str
        The drafted answer.

    Returns
    -------
    list of QualitativeClaim
    """
    claims: list[QualitativeClaim] = []

    for sentence in _SENTENCE_SPLIT_RE.split(answer):
        sentence = sentence.strip()
        if not sentence:
            continue

        markers = MARKER_RE.findall(sentence)
        if not markers or extract_numeric_claims(sentence):
            continue

        claims.append(QualitativeClaim(text=sentence, source_ids=[(t, i) for t, i in markers]))

    return claims


def _evidence_text(claim: QualitativeClaim, findings: list[AgentFinding], citations: list[Citation]) -> str:
    """
    Gather what the cited sources actually said, for the judge to read.

    Pulls the retrieved excerpt from each cited citation and the claim text of
    any finding carrying that citation — the specialist's own rendering of
    what the tool returned.
    """
    wanted = set(claim.source_ids)
    parts: list[str] = []

    for citation in citations:
        if (citation["source_type"], citation["source_id"]) in wanted and citation.get("excerpt"):
            parts.append(str(citation["excerpt"]))

    for finding in findings:
        if any((c["source_type"], c["source_id"]) in wanted for c in finding["citations"]):
            parts.append(finding["claim"])

    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            unique.append(part)

    if not unique:
        return "(no supporting text was retrieved for this source)"

    return "\n".join(f"  - {p[:MAX_EVIDENCE_CHARS]}" for p in unique[:MAX_EVIDENCE_PER_CLAIM])


def judge_qualitative_claims(
    claims: list[QualitativeClaim],
    findings: list[AgentFinding],
    citations: list[Citation],
    *,
    _mock_response: list[tuple[bool, str]] | None = None,
) -> list[tuple[bool, str]]:
    """
    Ask the LLM whether each cited sentence is supported by its evidence.

    ONE batched ``pro`` call for all claims, not one call per claim. The
    structured-output schema is two parallel ``list[str]`` fields because
    Gemini's schema handling is strict about nesting and optionality —
    the same constraint the router works around.

    Parameters
    ----------
    claims : list of QualitativeClaim
        Cited, non-numeric sentences.
    findings : list of AgentFinding
        Used to reconstruct what each cited source said.
    citations : list of Citation
        Used for retrieved excerpts.
    _mock_response : list of tuple, optional
        Bypass the LLM in tests.

    Returns
    -------
    list of tuple
        ``(supported, reason)`` aligned with ``claims``. A claim the model
        skipped defaults to supported: stage 2 is an additional check, and an
        incomplete response must not manufacture failures stage 1 never saw.
    """
    if _mock_response is not None:
        return _mock_response
    if not claims:
        return []

    from pydantic import BaseModel, Field

    from src.core.llm import get_llm
    from src.core.tracing import trace_metadata
    from src.research.config import (
        MAX_QUALITATIVE_CLAIMS,
        QUALITATIVE_VERIFIER_PROMPT_SYSTEM,
        QUALITATIVE_VERIFIER_PROMPT_USER,
    )

    batch = claims[:MAX_QUALITATIVE_CLAIMS]

    class Verdicts(BaseModel):
        """Flat and fully required — nested or optional fields fight Gemini's schema."""

        verdicts: list[str] = Field(description="SUPPORTED or UNSUPPORTED, one per statement, in order")
        reasons: list[str] = Field(description="One short sentence per statement, in order")

    rendered = "\n\n".join(
        f"{i}. STATEMENT: {claim.text}\n   EVIDENCE:\n{_evidence_text(claim, findings, citations)}"
        for i, claim in enumerate(batch, start=1)
    )
    user = QUALITATIVE_VERIFIER_PROMPT_USER.format(claims=rendered, count=len(batch))

    model = get_llm("pro", temperature=0.0).with_structured_output(Verdicts)
    result = model.invoke(
        [("system", QUALITATIVE_VERIFIER_PROMPT_SYSTEM), ("human", user)],
        config={"metadata": trace_metadata(phase="P4"), "tags": ["subsystem1", "citation_verifier"]},
    )

    verdicts = list(getattr(result, "verdicts", []))
    reasons = list(getattr(result, "reasons", []))

    judged: list[tuple[bool, str]] = []
    for index in range(len(batch)):
        verdict = verdicts[index].strip().upper() if index < len(verdicts) else "SUPPORTED"
        reason = reasons[index] if index < len(reasons) else ""
        judged.append((not verdict.startswith("UNSUPPORTED"), reason))

    # Claims past the batch cap were never looked at, so they are not failures.
    judged.extend((True, "not audited — beyond the per-query batch cap") for _ in claims[len(batch) :])
    return judged


# ── The verifier ────────────────────────────────────────
def verify(
    answer: str,
    findings: list[AgentFinding],
    citations: list[Citation],
    *,
    selected_agents: list[str] | None = None,
    use_llm: bool = False,
) -> VerificationReport:
    """
    Check an answer's numbers, source markers, and cited assertions.

    Stage 1 (always) is deterministic and free. Stage 2 (``use_llm``) judges
    the cited sentences stage 1 structurally cannot: no number to compare, only
    an assertion pointing at a source.

    Parameters
    ----------
    answer : str
        The drafted answer.
    findings : list of AgentFinding
        Everything the specialists returned.
    citations : list of Citation
        Every citation collected during the run.
    selected_agents : list of str, optional
        The router's plan, used to target repairs at an agent that was
        actually in it.
    use_llm : bool, default False
        Run stage 2. Off by default so ``verify`` stays free and reproducible;
        the graph node turns it on from config.

    Returns
    -------
    VerificationReport
        ``citation_coverage`` is the headline metric: the share of numeric
        claims traceable to tool output. An answer with no numbers scores 1.0
        — there is nothing to get wrong.
    """
    agents = selected_agents or []
    index = build_evidence_index(findings)
    claims = extract_numeric_claims(answer)

    verified: list[str] = []
    unsupported: list[UnsupportedClaim] = []

    for claim in claims:
        support = find_support(claim, index)
        if support is not None:
            verified.append(f"{claim.text} -> {support.label}")
            continue

        sentence = _sentence_around(answer, claim.start)
        unsupported.append(
            UnsupportedClaim(
                claim=sentence or claim.text,
                reason=f"'{claim.text}' matches no value any tool returned",
                origin_agent=_infer_origin(agents),
                suggested_requery=f"Provide the exact reported figure for: {sentence or claim.text}",
            )
        )

    invalid = validate_source_markers(answer, citations)
    # Coverage stays a stage-1 metric on purpose: it must be comparable across
    # eval runs, and a number that silently changes meaning when an LLM switch
    # is flipped is not a metric.
    coverage = len(verified) / len(claims) if claims else 1.0

    if use_llm:
        qualitative = extract_qualitative_claims(answer)
        verdicts = judge_qualitative_claims(qualitative, findings, citations)
        for audited, (supported, reason) in zip(qualitative, verdicts):
            if supported:
                verified.append(f"(qualitative) {audited.text[:80]}")
                continue
            unsupported.append(
                UnsupportedClaim(
                    claim=audited.text,
                    reason=reason or "the cited source does not support this statement",
                    origin_agent="filings_rag" if "filings_rag" in agents else _infer_origin(agents),
                    suggested_requery=f"Find disclosure text that directly addresses: {audited.text}",
                )
            )

    report = VerificationReport(
        verified_claims=verified,
        unsupported_claims=unsupported,
        invalid_source_ids=invalid,
        citation_coverage=round(coverage, 4),
        passed=not unsupported and not invalid,
    )

    logger.info(
        "Verification: %d/%d numeric claims grounded (coverage %.2f), %d invalid markers",
        len(verified),
        len(claims),
        coverage,
        len(invalid),
    )
    return report


def citation_verifier_node(state: ResearchState) -> dict:
    """
    Graph node: verify the drafted answer.

    Returns a partial state with ``verification``. Single writer, so no reducer.
    """
    plan: dict = dict(state.get("plan") or {})
    report = verify(
        state.get("draft_answer", ""),
        state.get("findings", []),
        state.get("citations", []),
        selected_agents=list(plan.get("selected_agents", [])),
        use_llm=VERIFY_QUALITATIVE_CLAIMS,
    )
    return {"verification": report}
