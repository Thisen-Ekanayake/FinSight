# ═══════════════════════════════════════════════════════
# FinSight — Experiment Variants
# ═══════════════════════════════════════════════════════
#
# Purpose : Each named variant changes EXACTLY ONE thing about the system, so a
#           delta in the LangSmith comparison view has one candidate cause.
#
# Public API:
#   VARIANTS                  {name: Variant}
#   apply_variant(name)       context manager
#   describe(name) -> str
#
# ══ THE ONE-VARIABLE RULE ══
#   Two changes in one experiment produce one number and no information. If
#   coverage moves from 0.71 to 0.93 after both a prompt rewrite and a
#   retrieval change, the honest report is "something helped" — which is what
#   you already believed before running anything.
#
#   So every variant here touches a single constant, and the runner refuses to
#   compose them.
#
# ══ PATCH WHERE IT IS USED, NOT WHERE IT IS DEFINED ══
#   `from src.research.config import FILINGS_TOP_K` binds the VALUE into the
#   importing module at import time. Rebinding src.research.config.FILINGS_TOP_K
#   afterwards changes nothing — filings_rag is still holding the integer it
#   copied. The patch has to land on src.research.agents.filings_rag.FILINGS_TOP_K.
#
#   Getting this backwards is silent: the experiment runs, reports a number,
#   and the number is the baseline's. Every variant is therefore verified on
#   entry — if the attribute is missing or already equal to the target value,
#   this raises rather than quietly measuring nothing.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager
from typing import Any, Iterator, NamedTuple

from src.vectorstore.config import COLLECTION_FILINGS_ABLATION

logger = logging.getLogger(__name__)


class Patch(NamedTuple):
    """One attribute rebind: ``module.attribute = value`` for the run's duration."""

    module: str
    attribute: str
    value: Any


class Variant(NamedTuple):
    """A named single-variable change, with the reason it is worth measuring."""

    description: str
    hypothesis: str
    patches: tuple[Patch, ...]
    # Checked before the patches are applied. Only needed when a variant
    # depends on state outside the process — see require_ablation_corpus.
    precondition: Any = None


def require_ablation_corpus() -> None:
    """
    Refuse to run the header ablation against a missing or partial index.

    An empty collection does not error. Retrieval simply returns nothing, the
    filings specialist contributes no findings, and the narrative archetype
    collapses — which reads exactly like a dramatic confirmation that
    contextual headers matter enormously.

    That is the most dangerous shape of failure in this whole suite: a broken
    setup and a real effect produce the same numbers. So parity is asserted,
    not assumed.

    Raises
    ------
    RuntimeError
        If the ablation collection is absent, or holds a different number of
        points than production. Same corpus, same chunking, same payloads —
        only the vectors may differ.
    """
    from src.vectorstore.collections import collection_stats
    from src.vectorstore.config import COLLECTION_FILINGS

    production = collection_stats(COLLECTION_FILINGS)
    ablation = collection_stats(COLLECTION_FILINGS_ABLATION)

    if not ablation["exists"]:
        raise RuntimeError(
            f"{COLLECTION_FILINGS_ABLATION} does not exist. Build it first:\n"
            "  .venv/bin/python -m src.vectorstore.ingest --watchlist --limit 4 --no-headers"
        )

    if ablation["points"] != production["points"]:
        raise RuntimeError(
            f"Corpus mismatch: {COLLECTION_FILINGS} has {production['points']:,} points, "
            f"{COLLECTION_FILINGS_ABLATION} has {ablation['points']:,}. An ablation over a different "
            "corpus measures the corpus, not the header."
        )


# ── The tightened synthesis prompt ──────────────────────
# Baseline rule 2 requires a marker on every NUMERIC claim. Qualitative
# sentences are left unmarked, so stage 2 of the verifier — which only audits
# CITED sentences — has almost nothing to look at on a narrative answer.
#
# This asks for a marker on every factual claim of either kind. The cost is a
# busier answer and more tokens; the hoped-for gain is that the LLM judge
# finally has jurisdiction over the narrative archetype.
STRICT_SRC_PROMPT_SYSTEM = """You are a financial research analyst writing a grounded answer.

You are given FINDINGS collected by specialist agents. Each finding carries a
source identifier. Your answer must rest entirely on those findings.

ABSOLUTE RULES:
1. Use ONLY the values present in the findings. Never introduce a number from
   your own knowledge, and never estimate, extrapolate, or round beyond what
   is given.
2. EVERY factual claim MUST carry an inline source marker in the exact form
   [SRC:<SOURCE_TYPE>:<SOURCE_ID>], placed immediately after the claim. This
   applies to qualitative statements exactly as it does to numbers:
     Apple reported revenue of $416.2B [SRC:EDGAR:0000320193-25-000079].
     Management flagged supplier concentration as a risk [SRC:EDGAR:0000320193-25-000079].
   A sentence asserting something about a company with no marker after it is
   an error, regardless of whether it contains a number.
3. If the findings do not answer part of the question, say so plainly. An
   honest gap is far more useful than a plausible guess. Sentences that state
   a gap need no marker — there is no source for an absence.
4. If two sources disagree, state both figures, name the sources, and say which
   you are using and why. Do not silently pick one.
5. No preamble. No "Based on the findings...". Start with the answer.

Style: precise, compact, and specific. Write for an analyst who wants the
number and its provenance, not an essay."""


VARIANTS: dict[str, Variant] = {
    "baseline": Variant(
        description="The system exactly as Phase 4 left it.",
        hypothesis="Reference point. Every other number in docs/eval_results.md is a delta from this.",
        patches=(),
    ),
    "strict-src": Variant(
        description="Synthesis prompt requires a [SRC:...] marker on every factual claim, not only numeric ones.",
        hypothesis=(
            "Stage 2 of the verifier only audits CITED sentences, so today it barely engages with narrative "
            "answers. Marking qualitative claims should widen its jurisdiction and raise citation faithfulness "
            "on the narrative archetype. Risk: markers become decorative and faithfulness falls instead."
        ),
        patches=(Patch("src.research.aggregator", "SYNTHESIS_PROMPT_SYSTEM", STRICT_SRC_PROMPT_SYSTEM),),
    ),
    "k12": Variant(
        description="Filing retrieval returns 12 chunks per branch instead of 8.",
        hypothesis=(
            "More context should help the narrative archetype, where a single risk factor can span chunks. "
            "Risk: 50% more retrieved text dilutes the synthesis prompt and gives the model more material "
            "to misattribute."
        ),
        patches=(Patch("src.research.agents.filings_rag", "FILINGS_TOP_K", 12),),
    ),
    "no-headers": Variant(
        description="Filing retrieval reads an index built WITHOUT contextual chunk headers.",
        hypothesis=(
            "A bare chunk loses both its entity and its section — 'we face intense competition' is nearly "
            "meaningless without knowing it is Apple's Risk Factors — so removing the header should degrade "
            "retrieval and show up as lower answer_correctness on the narrative archetype. This is the only "
            "ablation here that measures a decision already made rather than proposing a new one: if the "
            "metrics do not move, the header is costing ingest complexity for nothing."
        ),
        patches=(Patch("src.research.agents.filings_rag", "SEARCH_COLLECTION", COLLECTION_FILINGS_ABLATION),),
        precondition=require_ablation_corpus,
    ),
    "pro-router": Variant(
        description="Routing runs on gemini-2.5-pro instead of gemini-2.5-flash.",
        hypothesis=(
            "Routing is a short classification, so flash should be enough. If pro does not move the metrics, "
            "that is a real result: it says the money belongs in synthesis, and pro on the critical path of "
            "every query would be the most expensive place to spend it."
        ),
        patches=(Patch("src.research.router", "ROUTER_MODEL_TIER", "pro"),),
    ),
}


def describe(name: str) -> str:
    """One-line summary of a variant, for the run banner."""
    variant = VARIANTS[name]
    return f"{name}: {variant.description}"


@contextmanager
def apply_variant(name: str) -> Iterator[Variant]:
    """
    Apply a variant's patches for the duration of the block, then restore them.

    Parameters
    ----------
    name : str
        Key of ``VARIANTS``.

    Yields
    ------
    Variant

    Raises
    ------
    KeyError
        Unknown variant name.
    AttributeError
        The target module does not have the attribute — almost always because
        it imports the constant under a different name, or not at all.
    ValueError
        The attribute already equals the target value, so the experiment would
        measure nothing while appearing to work. Louder than a silent no-op.
    RuntimeError
        The variant's precondition failed — see require_ablation_corpus.
    """
    variant = VARIANTS[name]
    restore: list[tuple[Any, str, Any]] = []

    # Before anything is patched: a variant depending on external state must
    # prove that state is there. A missing ablation index does not error, it
    # just returns nothing — and "no retrieval" looks identical to "the change
    # had an enormous effect".
    if variant.precondition is not None:
        variant.precondition()

    try:
        for patch in variant.patches:
            module = importlib.import_module(patch.module)
            if not hasattr(module, patch.attribute):
                raise AttributeError(
                    f"{patch.module} has no attribute {patch.attribute!r}. Patch the module that USES the "
                    "constant, not the config module that defines it."
                )

            current = getattr(module, patch.attribute)
            if current == patch.value:
                raise ValueError(
                    f"{patch.module}.{patch.attribute} is already {patch.value!r}; variant {name!r} would "
                    "measure the baseline under a different name."
                )

            restore.append((module, patch.attribute, current))
            setattr(module, patch.attribute, patch.value)
            logger.info("variant %s: %s.%s -> %r", name, patch.module, patch.attribute, patch.value)

        yield variant

    finally:
        for module, attribute, value in reversed(restore):
            setattr(module, attribute, value)
