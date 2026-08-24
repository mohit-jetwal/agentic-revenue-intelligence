"""Streamlit demo UI - agentic trace layout.

Renders the trace structure from section 22 against static placeholder data so
the intended shape is reviewable before the agent exists. Wired to the live API
in Stage 1 Step 20.

Run with::

    streamlit run app/ui/streamlit_app.py

Design constraint carried through from the brief: the trace shows the plan, the
tool calls, their structured results, re-planning events and the critic verdict.
It shows concise reasoning *summaries*. It never exposes private
chain-of-thought.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.config.settings import get_settings

PLACEHOLDER_NOTICE = (
    "**Placeholder trace.** The layout below is static sample data illustrating "
    "the intended agentic trace. Live investigations are wired in Stage 1 Step 20."
)

SAMPLE_QUESTION = "Revenue declined 12% this month. Should we reduce prices or increase promotions?"

SAMPLE_PLAN: list[dict[str, str]] = [
    {"tool": "baseline_sales", "why": "Establish whether the decline exceeds normal variation."},
    {"tool": "price_elasticity", "why": "Price rose 8%; quantify demand sensitivity."},
    {"tool": "cross_price_elasticity", "why": "Check whether volume moved to a substitute."},
    {"tool": "promo_uplift", "why": "Assess whether current promotions are incremental."},
    {"tool": "scenario_simulation", "why": "Compare a price cut against added promotion spend."},
]

SAMPLE_RESULTS: list[dict[str, Any]] = [
    {
        "tool": "baseline_sales",
        "model": "baseline_sales:v1.0",
        "duration_ms": 420,
        "result": {"baseline_revenue": 100_000, "actual_revenue": 88_000, "revenue_gap_pct": -0.12},
        "confidence": 0.91,
    },
    {
        "tool": "price_elasticity",
        "model": "price_elasticity:v1.0",
        "duration_ms": 1_180,
        "result": {"elasticity": -1.42, "confidence_interval": [-1.65, -1.20], "p_value": 0.001},
        "confidence": 0.88,
    },
]


def _sidebar() -> None:
    settings = get_settings()
    with st.sidebar:
        st.header("Environment")
        st.write(f"**Mode:** `{settings.app.environment.value}`")
        st.write(f"**Version:** `{settings.app.version}`")
        st.write(f"**Planner:** `{settings.llm.planner_model}`")
        st.write(f"**Worker:** `{settings.llm.model}`")
        st.divider()
        st.caption(
            "Numbers shown in a trace always originate from a tool result and "
            "carry the model version that produced them."
        )


def main() -> None:
    st.set_page_config(page_title="Agentic Revenue Intelligence", layout="wide")
    st.title("Agentic Revenue Intelligence")
    st.caption("CPG/Retail revenue, pricing and promotion decision intelligence")

    _sidebar()
    st.info(PLACEHOLDER_NOTICE)

    st.subheader("Question")
    st.text_input("Ask a business question", value=SAMPLE_QUESTION, disabled=True)

    st.subheader("Investigation plan")
    for i, step in enumerate(SAMPLE_PLAN, start=1):
        st.markdown(f"**{i}. `{step['tool']}`** — {step['why']}")

    st.subheader("Tool calls and results")
    for entry in SAMPLE_RESULTS:
        with st.expander(f"`{entry['tool']}` — {entry['duration_ms']} ms", expanded=False):
            st.caption(f"model `{entry['model']}` · confidence {entry['confidence']}")
            st.json(entry["result"])

    st.subheader("Re-planning")
    st.markdown(
        "_No re-planning events in this sample. When the Critic finds evidence "
        "insufficient, the revised plan and the reason for revision appear here._"
    )

    st.subheader("Critic verdict")
    st.markdown("_Populated in Stage 1 Step 18._")

    st.subheader("Recommendation")
    st.markdown("_Populated in Stage 1 Step 18._")


if __name__ == "__main__":
    main()
