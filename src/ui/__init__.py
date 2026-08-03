# ═══════════════════════════════════════════════════════
# FinSight — Streamlit Dashboard (Phase 8)
# ═══════════════════════════════════════════════════════
#
# A thin client over the FastAPI backend. The dashboard never imports the
# graph, the repository, or the vector store directly — every read and write
# goes through the same HTTP surface a curl command or another integration
# would use, which is what CORS in src/api/main.py already assumes ("Phase
# 8's Streamlit UI is a separate origin").
# ═══════════════════════════════════════════════════════
