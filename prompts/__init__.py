"""Versioned prompts.

Prompt text lives in ``<agent>/<version>.md``, loaded through
:mod:`prompts.registry`. Keeping prompts in files rather than string literals
means a prompt change shows up as a reviewable diff, and lets an evaluation
result be tied to the exact prompt version that produced it.
"""
