from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chunker import build_chunks
from generator import FinancialGenerator, RagResponse
from indexer import build_indexes
from retriever import FinancialRetriever


@dataclass(frozen=True)
class QueryFilters:

    company_name: str | None = None
    filing_type: str | None = None


class FinancialRAGPipeline:

    def __init__(
        self,
        retriever: FinancialRetriever | None = None,
        generator: FinancialGenerator | None = None,
    ) -> None:

        self.retriever = retriever or FinancialRetriever()
        self.generator = generator or FinancialGenerator()

    def answer_question(self, question: str, filters: QueryFilters | None = None) -> RagResponse:
        """Retrieve evidence and generate a grounded answer."""

        active_filters = filters or QueryFilters()
        chunks = self.retriever.retrieve(
            query=question,
            company_name=active_filters.company_name,
            filing_type=active_filters.filing_type,
        )
        return self.generator.generate(question, chunks)


def build_default_pipeline() -> FinancialRAGPipeline:

    return FinancialRAGPipeline()


def rebuild_retrieval_artifacts() -> None:

    build_chunks()
    build_indexes()


def response_to_dict(response: RagResponse) -> dict[str, Any]:

    return {
        "answer": response.answer,
        "latency_ms": response.latency_ms,
        "sources": [source.__dict__ for source in response.sources],
    }
