"""Guardrails.

Built:

* ``budget`` - execution limits that bound the agentic loop: iterations, tool
  calls, tokens and wall clock, each independent.
* ``output_validation`` - every numeral in a recommendation is matched against
  the tool results that produced it. The hallucination control that is
  architectural rather than prompted.

Deliberately not built, and the reasons differ:

* SQL validation - agents never author SQL. Tools take typed Pydantic inputs and
  the repository owns every query, so there is no injection surface to guard.
* Prompt-injection screening - screens retrieved documents, and there is no
  document corpus. It arrives with agentic RAG or not at all.
* Role-based tool permissions - ``ToolSpec`` carries a permission and
  ``registry.specs(permissions=...)`` filters on it, but nothing populates a
  caller role yet. The filter is the mechanism; authentication is the gap.
* PII filtering - the dataset is synthetic and carries no personal data.
"""
