# ═══════════════════════════════════════════════════════
# FinSight — Shared Dashboard Components
# ═══════════════════════════════════════════════════════
#
# Purpose : The handful of patterns every page in src/ui/pages/ repeats —
#           calling the API without leaking a traceback into the page, and
#           the severity/status vocabulary the backend returns as plain
#           strings.
#
# Public API:
#   safe_call(fn, *args, **kwargs)   run an API call, st.error() on failure
#   severity_badge(severity)         "HIGH" -> "🔺 HIGH" -ish plain-text marker
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any, Callable, TypeVar

import streamlit as st

from src.ui.client import ApiError

T = TypeVar("T")

# Plain-text markers rather than emoji or colour alone — a severity that
# only reads as a colour fails anyone on a monochrome terminal-style theme
# or with colour vision deficiency, and this is a financial alert stream
# where missing a HIGH is the failure mode the whole backend is built around.
_SEVERITY_MARKERS: dict[str, str] = {"HIGH": "[!!]", "MED": "[~]", "LOW": "[.]"}


def safe_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
    """
    Call an API client function, turning ApiError into a visible st.error
    instead of a stack trace.

    Returns
    -------
    The call's result, or None if it failed — callers are expected to check
    for None before using the result, the same shape `requests` itself would
    hand back from a call that raised.
    """
    try:
        return fn(*args, **kwargs)
    except ApiError as exc:
        st.error(str(exc))
        return None


def severity_badge(severity: str) -> str:
    """A plain-text severity marker, prefixed to a headline."""
    return f"{_SEVERITY_MARKERS.get(severity.upper(), '[?]')} {severity.upper()}"
