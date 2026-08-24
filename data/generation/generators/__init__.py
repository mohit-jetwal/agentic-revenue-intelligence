"""Per-entity generators.

Each module owns one part of the simulation and is independently testable. The
pipeline composes them in dependency order:

    calendar -> products -> relationships -> stores -> listings -> customers
             -> ground truth -> cost index -> prices -> competitor -> promotions
             -> scenario injection -> demand -> inventory -> sales -> transactions

The ordering is not arbitrary. Ground truth is drawn *before* any driver series
exists, and scenarios are injected into the driver matrices *before* demand is
simulated - so injected effects propagate through the same causal chain as
everything else rather than being pasted onto the output.
"""
