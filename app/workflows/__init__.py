"""LangGraph workflow assembly.

This package holds *how the agents connect*: the state graph, conditional edges,
the plan-act-observe-evaluate loop, re-planning routes, checkpointing and the
human-approval interrupt. Agent behaviour itself lives in ``app.agents``.

The graph shape the brief asks for (sections 5 and 18)::

    classify_intent
         |
      plan  <-----------------+
         |                    |
      execute_step            |  re-plan
         |                    |  (bounded by BudgetTracker)
      observe                 |
         |                    |
      evaluate --------------- + insufficient evidence
         |
      critic
         |
      +--+-- invalid --> re-plan (bounded by max_replans)
      |
    recommend  <-- interrupt_before, when a checkpointer is supplied
         |
      finish

Two properties this layer is responsible for:

*Minimum sufficient workflow.* A forecast question must reach the forecasting
tool and stop. Fanning out to elasticity and optimisation because the graph
happens to contain those nodes is the failure mode section 6 warns about - it
looks impressive in a demo and is wrong.

*Bounded loops.* Every path back to ``plan`` passes through the budget check.
An agent that can re-plan is an agent that can loop forever, and the Critic
returning "insufficient evidence" indefinitely is the realistic way it happens.
Bounded twice, in fact: by ``max_replans`` on the critic edge and by the budget
check inside ``plan``.
"""
