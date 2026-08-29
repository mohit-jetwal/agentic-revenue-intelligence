"""Streamlit demo UI, wired to the live API.

Run with::

    uv run uvicorn app.main:app --reload      # in one terminal
    uv run streamlit run app/ui/streamlit_app.py

**Talks HTTP, not Python.** It would be shorter to import the container and call
the service directly, and that shortcut would make the UI a second consumer of
the internals rather than a client of the API - so an endpoint could break
without the demo noticing. Going over HTTP means the UI exercises the contract a
real client would.

Design constraint carried through from the brief: the trace shows the plan, the
tool calls, their structured results, re-planning events and the critic verdict.
It shows concise reasoning *summaries*. It never exposes private
chain-of-thought - the API does not return it, and the UI could not display it
if it wanted to.

**What the UI must not do is soften the output.** A recommendation that carries
risks, an unsourced-figure warning or an approval flag is displayed with them
attached. A demo that shows the headline and hides the caveats is a
misrepresentation of what the system actually concluded.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

from app.config.settings import get_settings

DEFAULT_API = os.getenv("ARI_API_URL", "http://127.0.0.1:8000")

EXAMPLE_QUESTIONS = [
    "Did the promotion on product P00091 between 2024-04-08 and 2024-04-28 "
    "generate incremental profit, and was it worth running?",
    "We raised the price of product P00013 on 2024-06-21. How did demand "
    "respond, and was the increase the right call?",
    "Sales of product P00245 fell sharply in July 2025. What caused the decline?",
]

#: Trace event type -> (icon, label). Unlisted types still render, with a
#: neutral marker: a UI that silently dropped an event it did not recognise
#: would show an incomplete history of how the answer was reached.
_EVENT_STYLE: dict[str, tuple[str, str]] = {
    "intent_classified": ("🧭", "Understood"),
    "plan_created": ("📋", "Planned"),
    "tool_called": ("🔧", "Tool"),
    "tool_failed": ("⚠️", "Tool failed"),
    "observation": ("👁", "Observed"),
    "replanned": ("🔁", "Re-planned"),
    "critic_verdict": ("⚖️", "Critic"),
    "recommendation": ("✅", "Recommendation"),
    "error": ("❌", "Error"),
}


def _api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API).rstrip("/")


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = httpx.post(f"{_api_url()}{path}", json=payload, timeout=300.0)
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the API at {_api_url()}: {exc}")
        return None
    if response.status_code >= 400:
        st.error(f"{response.status_code}: {response.text[:500]}")
        return None
    return response.json()


def _get(path: str) -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{_api_url()}{path}", timeout=60.0)
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the API at {_api_url()}: {exc}")
        return None
    if response.status_code >= 400:
        return None
    return response.json()


def _sidebar() -> None:
    settings = get_settings()
    with st.sidebar:
        st.header("Connection")
        st.session_state["api_url"] = st.text_input("API URL", value=_api_url())

        health = _get("/health")
        if health is None:
            st.error("API unreachable")
        else:
            st.success(f"API {health['status']}")
            for dependency in health.get("dependencies", []):
                marker = "🟢" if dependency["status"] == "ok" else "🟡"
                st.caption(f"{marker} {dependency['name']} — {dependency['status']}")

        st.divider()
        st.header("Environment")
        st.write(f"**Mode:** `{settings.app.environment.value}`")
        st.write(f"**Planner:** `{settings.llm.planner_model}`")
        st.divider()
        st.caption(
            "Every number in a recommendation is checked against the tool "
            "results that produced it. Figures that do not appear in any result "
            "are flagged rather than removed."
        )


def _render_recommendation(recommendation: dict[str, Any]) -> None:
    if recommendation.get("requires_human_approval"):
        st.warning(
            "**Approval required.** The projected impact crosses the approval "
            "threshold, so this recommendation is not cleared to act on."
        )

    st.markdown(f"### {recommendation['executive_summary']}")
    if recommendation.get("root_cause"):
        st.markdown(f"**Root cause.** {recommendation['root_cause']}")
    st.markdown(f"**Recommended action.** {recommendation['recommended_action']}")

    confidence = recommendation.get("confidence", 0.0)
    st.progress(min(1.0, max(0.0, confidence)), text=f"Confidence {confidence:.0%}")

    evidence = recommendation.get("evidence") or []
    if evidence:
        st.markdown("**Evidence**")
        for item in evidence:
            st.markdown(f"- `{item['source_tool']}` — {item['claim']}")

    # Risks and assumptions are expanded by default, not tucked behind a
    # collapsed section. They are the conditions under which the number above
    # means what it says.
    risks = recommendation.get("risks") or []
    if risks:
        st.markdown("**Risks and warnings**")
        for risk in risks:
            st.markdown(f"- {risk}")

    assumptions = recommendation.get("assumptions") or []
    if assumptions:
        with st.expander(f"Assumptions ({len(assumptions)})"):
            for assumption in assumptions:
                st.markdown(f"- {assumption}")


def _render_trace(events: list[dict[str, Any]]) -> None:
    for event in events:
        icon, label = _EVENT_STYLE.get(event["event_type"], ("•", event["event_type"]))
        header = f"{icon} **{label}**"
        if event.get("tool_name"):
            header += f" · `{event['tool_name']}`"
        st.markdown(f"{header} — {event['summary']}")

        payload = event.get("payload") or {}
        if payload:
            with st.expander("Detail", expanded=False):
                st.json(payload)


def _render_feedback(investigation_id: str) -> None:
    st.divider()
    st.markdown("**Was this useful?**")
    left, right = st.columns(2)
    if left.button("👍 Helpful", key=f"up-{investigation_id}"):
        _post("/feedback", {"investigation_id": investigation_id, "helpful": True})
        st.success("Recorded.")
    if right.button("👎 Not helpful", key=f"down-{investigation_id}"):
        _post("/feedback", {"investigation_id": investigation_id, "helpful": False})
        st.success("Recorded.")


def main() -> None:
    st.set_page_config(page_title="Agentic Revenue Intelligence", layout="wide")
    st.title("Agentic Revenue Intelligence")
    st.caption("CPG/Retail revenue, pricing and promotion decision intelligence")

    _sidebar()

    st.subheader("Ask a business question")
    example = st.selectbox(
        "Examples", ["(write your own)", *EXAMPLE_QUESTIONS], index=1
    )
    default = "" if example == "(write your own)" else example
    question = st.text_area("Question", value=default, height=90)

    if st.button("Investigate", type="primary", disabled=not question.strip()):
        with st.spinner("Planning, running tools, and checking the evidence..."):
            answer = _post("/chat", {"question": question})
        if answer:
            st.session_state["last"] = answer

    result = st.session_state.get("last")
    if not result:
        st.info(
            "Ask a question to run a live investigation. The third example is "
            "one the platform deliberately cannot answer — no tool diagnoses "
            "availability — so it shows what declining looks like."
        )
        return

    st.divider()
    recommendation = result.get("recommendation")
    if recommendation:
        _render_recommendation(recommendation)
    else:
        st.warning(result.get("answer", "No conclusion was reached."))

    trace = _get(f"/investigation/{result['investigation_id']}/trace")
    if trace and trace.get("events"):
        st.divider()
        st.subheader("How this answer was reached")
        _render_trace(trace["events"])

    _render_feedback(result["investigation_id"])
    st.caption(
        f"investigation `{result['investigation_id']}` · trace `{result['trace_id']}`"
    )


if __name__ == "__main__":
    main()
