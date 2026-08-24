"""Agent node logic.

This package holds *what each agent does*: prompt construction, structured
output parsing, and the decision it returns. The graph that connects them -
edges, routing conditions, loops, checkpoints - lives in ``app.workflows``.

Keeping those apart matters because they change for different reasons. Tuning
how the Critic judges evidence should not touch the graph; adding a re-planning
edge should not touch the Critic's prompt. When both live in one module, every
routing change becomes a diff against agent behaviour and vice versa.

Four agents, deliberately, not one per model (brief section 3):

* **Supervisor**      - intent, objective, planning, tool selection, re-planning.
* **Root Cause**      - hypothesis generation and evidence interpretation.
* **Critic**          - validation, contradiction detection, sufficiency check.
* **Recommendation**  - synthesis, trade-off comparison, final business output.

The eight analytical models are *tools*, not agents. They are deterministic:
given the same inputs they must return the same numbers, which is precisely the
property an LLM does not have. Wrapping each in its own agent would add a
non-deterministic layer between the caller and a deterministic computation,
buying latency and token cost in exchange for nothing.

Implemented across Stage 1 Steps 15-18.
"""
