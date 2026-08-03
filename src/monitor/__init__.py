# ═══════════════════════════════════════════════════════
# FinSight — Monitoring Subsystem (Subsystem 2)
# ═══════════════════════════════════════════════════════
#
# Autonomous portfolio surveillance: watch a ticker list, turn what changed
# into candidate alerts, score their severity by rule, deduplicate them
# semantically, and persist the cycle.
#
# The contrast with src/research/ is the point. That subsystem is
# request/response and stateless between queries; this one is scheduled,
# long-running, and its whole value depends on remembering what it already
# told you.
# ═══════════════════════════════════════════════════════
