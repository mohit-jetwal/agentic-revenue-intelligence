"""Synthetic CPG/Retail data generation (Stage 1 Step 2).

Generated *output* is written to ``data/local/`` and is git-ignored. Only the
generator code lives here.

The design constraint for Step 2, stated now so it is not forgotten: the data
must be produced by a causal simulation, not by sampling each column
independently. Price must actually move demand, promotions must actually lift
it, stockouts must actually censor it, and a substitute's price must actually
shift a product's volume - with known underlying parameters written alongside
the data.

Without that, every model in Steps 4-11 is unfalsifiable. With it, each one can
be tested against the parameter it is supposed to recover, and a naive
specification can be shown to be biased. That test is the difference between a
demo and evidence the models work.
"""
