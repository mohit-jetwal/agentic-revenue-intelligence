"""Guardrails.

Implemented in Step 1:

* ``budget`` - execution limits that bound the agentic loop.

Added in Stage 1 Step 20:

* SQL validation - read-only, allowlisted schemas, row and time limits.
* Prompt-injection screening on retrieved documents and user input.
* Tool permission enforcement against the caller's role.
* PII and sensitive-field filtering on tool output.
* Hallucination detection - every numerical claim in a recommendation must
  resolve to a ``ToolResult`` trace id, or the Critic rejects it.
"""
