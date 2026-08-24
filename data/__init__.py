"""Data generation and access.

``data.generation`` produces the synthetic CPG/Retail dataset (Stage 1 Step 2).
``data.repositories`` provides the storage abstraction every model reads through.

Generated artifacts are written to ``data/local/`` and are git-ignored. Code and
output are kept apart deliberately: mixing them is the failure mode of a
top-level package that shares its name with a data directory.
"""
