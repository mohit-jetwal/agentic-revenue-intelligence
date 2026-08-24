"""Vector store abstraction for enterprise RAG.

Chroma locally, Databricks Vector Search in production. The interface is
narrower than either backend offers on purpose - it exposes similarity search
with metadata filtering and nothing else, because that is all the RAG agent
needs and a wider surface would be harder to reimplement faithfully.

Metadata filtering is a required parameter rather than an optional nicety.
Retrieving the top-k chunks across an entire policy corpus regardless of region,
product or effective date is how a RAG system confidently cites a superseded
pricing policy. Filtering first is the fix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A chunk of enterprise knowledge, with the metadata needed to filter it."""

    document_id: str
    content: str
    #: document_type, business_domain, product, region, effective_date,
    #: source, access_level - see the RAG section of the brief.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float
    #: Populated when a reranking stage runs after retrieval.
    rerank_score: float | None = None


class VectorStore(ABC):
    """Similarity search over the enterprise document corpus."""

    @abstractmethod
    def upsert(self, documents: list[Document]) -> int:
        """Insert or replace documents. Returns the number written."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve the most similar documents, filtered by metadata."""

    @abstractmethod
    def delete(self, document_ids: list[str]) -> int:
        """Remove documents. Returns the number deleted."""

    @abstractmethod
    def count(self) -> int:
        """Number of indexed documents."""

    def health_check(self) -> tuple[bool, str]:
        try:
            n = self.count()
        except Exception as exc:  # noqa: BLE001 - health checks must not raise
            return False, f"{type(self).__name__}: {exc}"
        return True, f"{n} documents indexed"
