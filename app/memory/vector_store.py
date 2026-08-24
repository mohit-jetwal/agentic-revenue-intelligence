"""Vector store implementations.

Both are declared now so the container has something concrete to construct and
``GET /health`` reports honestly. Bodies land in Stage 1 Step 15 (local) and
Stage 2 (Databricks Vector Search).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.memory.base import Document, RetrievedDocument, VectorStore

_LOCAL_STEP = "Stage 1 Step 15 (enterprise RAG)"
_PROD_STAGE = "Stage 2 (Databricks Vector Search)"


class ChromaVectorStore(VectorStore):
    """Local persistent Chroma collection.

    Chroma over FAISS here because the corpus is small (policy documents, not
    web scale) and metadata filtering matters more than raw index speed. FAISS
    needs a separate metadata sidecar to filter at all; Chroma has it built in,
    and filtering is the property this design depends on.
    """

    def __init__(self, path: Path, collection: str) -> None:
        self.path = path
        self.collection = collection
        self._client: object | None = None

    def _not_yet(self, method: str) -> NotImplementedError:
        return NotImplementedError(f"ChromaVectorStore.{method}() is implemented in {_LOCAL_STEP}")

    def upsert(self, documents: list[Document]) -> int:
        raise self._not_yet("upsert")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        raise self._not_yet("search")

    def delete(self, document_ids: list[str]) -> int:
        raise self._not_yet("delete")

    def count(self) -> int:
        raise self._not_yet("count")

    def health_check(self) -> tuple[bool, str]:
        if not self.path.exists():
            return False, f"no index at {self.path} (built in {_LOCAL_STEP})"
        return True, f"chroma collection '{self.collection}' at {self.path}"


class DatabricksVectorSearchStore(VectorStore):
    """Databricks Vector Search index over a Delta-backed document table.

    Stage 2 notes: use a Delta Sync index so the corpus stays current as the
    source table changes, rather than a direct-access index that needs manual
    re-embedding. Access is governed by Unity Catalog grants on the index.
    """

    def __init__(self, *, endpoint: str, index_name: str) -> None:
        self.endpoint = endpoint
        self.index_name = index_name

    def _not_yet(self, method: str) -> NotImplementedError:
        return NotImplementedError(
            f"DatabricksVectorSearchStore.{method}() belongs to {_PROD_STAGE}"
        )

    def upsert(self, documents: list[Document]) -> int:
        raise self._not_yet("upsert")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        raise self._not_yet("search")

    def delete(self, document_ids: list[str]) -> int:
        raise self._not_yet("delete")

    def count(self) -> int:
        raise self._not_yet("count")

    def health_check(self) -> tuple[bool, str]:
        return False, f"Databricks Vector Search not implemented. {_PROD_STAGE}"
