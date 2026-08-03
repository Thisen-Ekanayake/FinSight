# ═══════════════════════════════════════════════════════
# FinSight — Monitoring CLI
# ═══════════════════════════════════════════════════════
#
# Purpose : Run a cycle, manage the watchlist, and inspect the dedup index
#           from a shell.
#
# Usage:
#   python -m src.monitor.cli --once --warmup     # cold start: index, report nothing
#   python -m src.monitor.cli --once              # a real cycle
#   python -m src.monitor.cli --once --ticker NVDA
#   python -m src.monitor.cli --watchlist         # show it
#   python -m src.monitor.cli --add TSLA --remove JPM
#   python -m src.monitor.cli --decisions         # why things were suppressed
#   python -m src.monitor.cli --prune
#
# ══ RUN --warmup FIRST ══
#   A cold dedup index has nothing to match against, so the first real cycle
#   would report every open filing, every recent article, and every price move
#   in one burst. --warmup does the same work and dispatches nothing, leaving
#   cycle 2 with something to compare against.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def _print_watchlist() -> None:
    from src.monitor.watchlist import current_watchlist
    from src.persistence.repository import get_checkpoints

    items = current_watchlist()
    if not items:
        print("\n  Watchlist is empty. Add one: --add AAPL\n")
        return

    checkpoints = get_checkpoints([item["ticker"] for item in items])

    print(f"\n  {len(items)} ticker(s) watched\n")
    for item in items:
        warm = "warm" if item["warmed_up"] else "COLD"
        print(f"  {item['ticker']:<6} {warm:<5} {item['company_name']}")
        seen = {key.split(":")[1]: value for key, value in checkpoints.items() if key.startswith(item["ticker"] + ":")}
        if seen:
            stamps = ", ".join(f"{monitor} {value[:16]}" for monitor, value in sorted(seen.items()))
            print(f"  {'':<6} last checked: {stamps}")
        else:
            print(f"  {'':<6} never checked")
    print()


def _print_decisions(limit: int) -> None:
    """
    Show recent dedup decisions.

    The scores are the interesting part: this is the table Phase 7's threshold
    sweep is calibrated against, and eyeballing it is how you notice that the
    bands have drifted before the sweep tells you so.
    """
    from src.persistence.repository import list_dedup_decisions

    rows = list_dedup_decisions(limit=limit)
    if not rows:
        print("\n  No dedup decisions recorded yet. Run a cycle first.\n")
        return

    print(f"\n  {len(rows)} most recent dedup decisions\n")
    print(f"  {'score':>6}  {'decision':<18} {'ticker':<7} text")
    for row in rows:
        score = f"{row['score']:.3f}" if row["score"] else "  -  "
        print(f"  {score:>6}  {row['decision']:<18} {row['ticker'] or 'MACRO':<7} {row['candidate_text'][:70]}")
        if row["parent_alert_id"]:
            print(f"  {'':>6}  {'':<18} {'':<7} -> {row['parent_text'][:70]}")
    print()


def main(argv: list[str] | None = None) -> int:
    """Run the monitoring subsystem from a shell."""
    parser = argparse.ArgumentParser(description="FinSight autonomous monitoring")
    parser.add_argument("--once", action="store_true", help="run a single cycle")
    parser.add_argument("--warmup", action="store_true", help="observe-only: index candidates, dispatch nothing")
    parser.add_argument("--ticker", action="append", help="override the watchlist for this run (repeatable)")

    parser.add_argument("--watchlist", action="store_true", help="show the watchlist and its watermarks")
    parser.add_argument("--add", action="append", help="add a ticker (repeatable)")
    parser.add_argument("--remove", action="append", help="remove a ticker (repeatable)")

    parser.add_argument("--decisions", action="store_true", help="show recent dedup decisions and scores")
    parser.add_argument("--limit", type=int, default=25, help="rows for --decisions")
    parser.add_argument("--stats", action="store_true", help="show the alert index")
    parser.add_argument("--prune", action="store_true", help="delete alert points past the retention window")

    args = parser.parse_args(argv)

    from src.core.logging_setup import configure_logging
    from src.core.tracing import configure_tracing
    from src.persistence.db import init_db

    configure_logging()
    configure_tracing()
    init_db()

    if args.add or args.remove:
        from src.monitor.watchlist import add_ticker, remove_ticker

        for ticker in args.add or []:
            item = add_ticker(ticker)
            print(f"  + {item['ticker']:<6} {item['company_name']}")
        for ticker in args.remove or []:
            print(f"  - {ticker.upper():<6} {'removed' if remove_ticker(ticker) else 'was not being watched'}")
        print()

    if args.watchlist:
        _print_watchlist()

    if args.decisions:
        _print_decisions(args.limit)

    if args.prune:
        from src.monitor.alert_store import prune_expired

        print(f"\n  Pruned {prune_expired()} expired alert point(s)\n")

    if args.stats:
        from src.monitor.alert_store import alert_stats

        stats = alert_stats()
        print(f"\n  {stats['name']:<24} {stats['status']:<10} points={stats['points']:,}")
        print(f"  {'':<24} indexed: {', '.join(stats['indexed_fields']) or 'none'}\n")

    if args.once:
        from src.monitor.graph import cycle_report, run_cycle
        from src.vectorstore.collections import ensure_collections

        # The alerts collection may not exist yet on a fresh install, and a
        # dedup search against a missing collection raises rather than
        # returning nothing — which would look like a monitor bug.
        ensure_collections()

        state = run_cycle(tickers=args.ticker, warmup=args.warmup)
        print(cycle_report(state))

        if state.get("warmup"):
            print("  Warmup complete. Run again WITHOUT --warmup to start alerting.\n")

        return 1 if state.get("monitor_errors") else 0

    if not any([args.once, args.watchlist, args.add, args.remove, args.decisions, args.stats, args.prune]):
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
