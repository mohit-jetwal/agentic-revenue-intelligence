"""Seeded random number generation.

Reproducibility is an acceptance criterion (brief sections 22 and 35): the same
seed and configuration must yield the same dataset, byte for byte.

The subtlety is *stream independence*. A single shared ``default_rng`` would
make every draw positionally dependent on every draw before it, so adding one
store would shift the random numbers used for products, prices and promotions -
and the entire dataset would change. That makes debugging nearly impossible:
you could not vary one dimension and compare.

``SeedSequence.spawn`` fixes this. Each entity family gets its own independent
stream derived from the master seed, so changing the store count perturbs the
store stream and nothing else.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np


class Stream(StrEnum):
    """Independent random streams, one per entity family.

    Order matters and must not be rearranged: streams are spawned positionally,
    so reordering this enum silently changes every previously generated dataset.
    Append new members at the end.
    """

    PRODUCT = "product"
    RELATIONSHIP = "relationship"
    STORE = "store"
    CUSTOMER = "customer"
    CALENDAR = "calendar"
    GROUND_TRUTH = "ground_truth"
    PRICING = "pricing"
    COST_INDEX = "cost_index"
    COMPETITOR = "competitor"
    PROMOTION = "promotion"
    TRADE_PROMOTION = "trade_promotion"
    LISTING = "listing"
    DEMAND = "demand"
    INVENTORY = "inventory"
    TRANSACTION = "transaction"
    SCENARIO = "scenario"
    DATA_QUALITY = "data_quality"


class RngFactory:
    """Hands out independent, reproducible generators keyed by :class:`Stream`."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._positions = {stream: index for index, stream in enumerate(Stream)}
        root = np.random.SeedSequence(seed)
        children = root.spawn(len(Stream))
        self._sequences = dict(zip(Stream, children, strict=True))
        self._generators: dict[Stream, np.random.Generator] = {}

    def get(self, stream: Stream) -> np.random.Generator:
        """Return the generator for a stream, creating it on first use.

        Cached, so repeated calls within one stream continue the same sequence
        rather than restarting it.
        """
        if stream not in self._generators:
            self._generators[stream] = np.random.default_rng(self._sequences[stream])
        return self._generators[stream]

    def fresh(self, stream: Stream, chunk: int) -> np.random.Generator:
        """A generator for one chunk of a partitioned pass.

        Derived from an explicit ``spawn_key`` rather than by calling ``spawn``
        on the parent sequence. That distinction matters: ``spawn`` mutates the
        parent's internal child counter, so asking for chunk 2 after chunks 0
        and 1 would yield different numbers than asking for chunk 2 first. The
        data would then depend on iteration order and on how many chunks a
        profile happened to use, which would quietly make ``output.chunk_months``
        a data-changing setting rather than a performance knob.

        Keying on ``(stream_position, chunk)`` makes each chunk's randomness a
        pure function of its coordinates.
        """
        sequence = np.random.SeedSequence(
            entropy=self.seed, spawn_key=(self._positions[stream], chunk)
        )
        return np.random.default_rng(sequence)


def sample_range(
    rng: np.random.Generator,
    bounds: tuple[float, float],
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Uniform draw within an inclusive ``[low, high]`` band."""
    low, high = bounds
    return np.asarray(rng.uniform(low, high, size=size))


def sample_choice_weighted(
    rng: np.random.Generator,
    options: list[str],
    weights: list[float],
    size: int,
) -> np.ndarray:
    """Weighted categorical draw; weights are normalised for you."""
    probabilities = np.asarray(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return rng.choice(np.asarray(options, dtype=object), size=size, p=probabilities)
