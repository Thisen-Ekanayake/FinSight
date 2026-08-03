# ═══════════════════════════════════════════════════════
# FinSight — Macro Monitor
# ═══════════════════════════════════════════════════════
#
# Purpose : Notice new FRED releases that moved enough to matter.
#
# Public API:
#   macro_monitor_node(payload)
#   detect_crossing(series_id, previous, latest)
#
# ══ BATCHED, AND TICKERLESS ══
#   FRED series are economy-wide. This monitor is dispatched ONCE per cycle
#   regardless of watchlist size — fanning it out per ticker would issue five
#   identical CPI requests to answer one question. The research subsystem's
#   macro specialist makes exactly the same call for exactly the same reason.
#
#   The alerts it produces carry no ticker, which the dedup engine has to
#   accommodate: its filter is (ticker, alert_type), and every macro alert
#   shares the empty ticker. That is correct — two CPI releases ARE comparable
#   to each other and to nothing else.
#
#   It is also the one monitor with no watermark. The others ask "what is new
#   since I last looked"; this one compares the last two observations FRED
#   publishes, so the series itself carries the state and there is nothing to
#   remember between cycles.
#
# ══ A CROSSING IS AN EVENT; A LEVEL IS NOT ══
#   The yield curve inverting is the canonical case. The move from +0.02 to
#   -0.01 is trivially small and is the single most-watched recession signal
#   on the list. But "the curve is inverted" is a STATE, and alerting on a
#   state fires every cycle for however many months it persists. So the
#   crossing is detected from the last two observations and fires once.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time

from src.data.config import WATCHED_FRED_SERIES
from src.monitor.config import MACRO_LEVEL_CROSSINGS, MACRO_THRESHOLDS
from src.monitor.monitors._common import candidate, monitor
from src.monitor.state import CandidateAlert

logger = logging.getLogger(__name__)

MONITOR_NAME = "macro_monitor"

# Two observations is all the change detection needs; a couple more give the
# series' own units and title without a second request.
OBSERVATION_LIMIT: int = 4


def detect_crossing(series_id: str, previous: float, latest: float) -> str:
    """
    Report a threshold crossing between two consecutive observations.

    Parameters
    ----------
    series_id : str
        FRED series id.
    previous, latest : float
        The last two observed values, in order.

    Returns
    -------
    str
        A description like ``"below 0.0"``, or ``""`` if nothing was crossed.
        Only the transition counts: a series that was already below the level
        and stayed there returns ``""``, which is what stops an inverted curve
        re-alerting every cycle for a year.
    """
    level = MACRO_LEVEL_CROSSINGS.get(series_id)
    if level is None:
        return ""

    if previous >= level > latest:
        return f"below {level}"
    if previous <= level < latest:
        return f"above {level}"
    return ""


def _change(series_id: str, previous: float, latest: float) -> tuple[float, float]:
    """Absolute and percent change between two observations."""
    absolute = latest - previous
    percent = (absolute / abs(previous) * 100.0) if previous else 0.0
    return absolute, percent


@monitor(MONITOR_NAME)
def macro_monitor_node(payload: dict) -> tuple[list[CandidateAlert], list]:
    """
    Check every watched FRED series for a new, material observation.

    Returns
    -------
    tuple
        ``(candidates, api_calls)``.
    """
    from src.core.schemas import make_citation
    from src.data.fred import get_series
    from src.research.agents._common import tool_record

    candidates: list[CandidateAlert] = []
    calls = []

    for series_id, label in WATCHED_FRED_SERIES.items():
        started = time.monotonic()
        try:
            series = get_series(series_id, limit=OBSERVATION_LIMIT)
        except Exception as exc:  # noqa: BLE001 - one bad series must not lose the others
            calls.append(
                tool_record(
                    MONITOR_NAME,
                    "get_series",
                    args={"series_id": series_id},
                    provider="fred",
                    latency_ms=int((time.monotonic() - started) * 1000),
                    ok=False,
                )
            )
            logger.warning("%s: series %s failed — %s", MONITOR_NAME, series_id, exc)
            continue

        calls.append(
            tool_record(
                MONITOR_NAME,
                "get_series",
                args={"series_id": series_id},
                provider="fred",
                latency_ms=int((time.monotonic() - started) * 1000),
                ok=True,
            )
        )

        observations = series.get("observations") or []
        if len(observations) < 2:
            logger.debug("%s: %s has too few observations to compare", MONITOR_NAME, series_id)
            continue

        previous = float(observations[-2]["value"])
        latest = float(observations[-1]["value"])
        as_of = observations[-1]["date"]

        absolute, percent = _change(series_id, previous, latest)
        crossing = detect_crossing(series_id, previous, latest)

        measure, med, _high = MACRO_THRESHOLDS.get(series_id, ("pct", 1.0, 2.0))
        magnitude = abs(absolute) if measure == "abs" else abs(percent)

        # Below the MED floor and no crossing: a routine reading, not an event.
        if magnitude < med and not crossing:
            continue

        citation = make_citation("FRED", series_id, as_of=as_of, url=series["url"])
        direction = "rose" if absolute > 0 else "fell"
        units = series.get("units") or ""

        candidates.append(
            candidate(
                # Macro alerts are economy-wide. An empty ticker is not a
                # missing value — it is the correct scope, and the dedup filter
                # groups every macro alert of a type together because of it.
                "",
                "MACRO_EVENT",
                monitor_name=MONITOR_NAME,
                headline=f"{label} {direction} to {latest:g}",
                detail=(
                    f"{series['title']} ({series_id}) {direction} from {previous:g} to {latest:g} {units} "
                    f"as of {as_of}" + (f"; crossed {crossing}" if crossing else "") + "."
                ),
                # series + release date. A revision to an already-released
                # observation keeps the same key and is therefore correctly
                # treated as the same event.
                natural_key=f"{series_id}:{as_of}",
                metrics={
                    "series_id": series_id,
                    "latest": latest,
                    "previous": previous,
                    "abs_change": absolute,
                    "pct_change": percent,
                    "crossing": crossing,
                    "units": units,
                    "as_of": as_of,
                },
                evidence=[citation],
                observed_at=as_of,
            )
        )

    return candidates, calls
