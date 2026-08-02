# ═══════════════════════════════════════════════════════
# FinSight — Persistence
# ═══════════════════════════════════════════════════════
#
# Two stores with different owners, deliberately kept apart:
#
#   finsight.db        application records — research runs, API budgets, and
#                      from Phase 6 the watchlist, alerts, and cycles.
#                      Schema is ours, via SQLAlchemy.
#
#   checkpoints.sqlite LangGraph's checkpointer. Schema belongs to LangGraph;
#                      we never read or write it directly, only through the
#                      graph's own state-history API.
#
# Mixing them would couple our migrations to a library's internal schema.
# ═══════════════════════════════════════════════════════
