# ═══════════════════════════════════════════════════════
# FinSight — Research
# ═══════════════════════════════════════════════════════
#
# Ask a question, see the answer with its citations and verification report.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import streamlit as st

from src.ui import client
from src.ui.components import safe_call

st.title("Ask a research question")
st.caption("Every number in the answer is matched against a value a tool actually returned before it is shown.")

with st.form("ask_form"):
    query = st.text_area(
        "Question",
        placeholder="How did Apple's gross margin trend over the last three fiscal years?",
        height=80,
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Routing, researching, and verifying — this can take a minute..."):
            result = safe_call(client.ask, query.strip())
        if result is not None:
            st.session_state["last_result"] = result

result = st.session_state.get("last_result")
if result:
    st.divider()
    st.subheader("Answer")
    st.write(result["answer"])

    verification = result.get("verification") or {}
    v_cols = st.columns(3)
    v_cols[0].metric("Citation coverage", f"{verification.get('citation_coverage', 0):.0%}")
    v_cols[1].metric("Verified", "yes" if verification.get("passed") else "no")
    v_cols[2].metric("Latency", f"{result['latency_ms'] / 1000:.1f}s")

    if verification.get("unsupported_claims"):
        with st.expander(f"{len(verification['unsupported_claims'])} unsupported claim(s)"):
            for claim in verification["unsupported_claims"]:
                st.write(f"- **{claim['reason']}** — {claim['claim']} ({claim['origin_agent']})")

    if result.get("citations"):
        st.subheader("Sources")
        seen = set()
        for citation in result["citations"]:
            key = (citation["source_type"], citation["source_id"])
            if key in seen:
                continue
            seen.add(key)
            st.write(f"[{citation['source_type']}:{citation['source_id']}]({citation['url']})")

    if result.get("conflicts"):
        st.subheader("Conflicts resolved")
        for conflict in result["conflicts"]:
            reported = ", ".join(f"{source}={value:,.2f}" for source, value in conflict["values"])
            st.write(
                f"**{conflict.get('ticker') or 'macro'} {conflict['metric']}**: {reported} "
                f"→ used **{conflict['chosen_source']}** ({conflict['rel_difference']:.1%} spread)"
            )

    if result.get("errors"):
        st.subheader("Specialist failures (answer is partial)")
        for error in result["errors"]:
            st.write(f"- {error}")

    st.caption(f"thread: `{result['thread_id']}` · agents: {', '.join(result.get('agents_used', []))}")

st.divider()
st.subheader("Recent runs")
runs = safe_call(client.list_runs, limit=10)
if runs:
    st.dataframe(
        [
            {
                "query": run["query"],
                "coverage": f"{run['citation_coverage']:.0%}",
                "verified": run["verification_passed"],
                "tickers": ", ".join(run["tickers"]),
                "asked": run["created_at"][:16].replace("T", " "),
            }
            for run in runs
        ],
        use_container_width=True,
        hide_index=True,
    )
elif runs is not None:
    st.caption("No research runs yet.")
