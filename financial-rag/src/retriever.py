from __future__ import annotations

import re 
import pickle
import logging
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from chunker import TextChunk , DEFAULT_CHUNKS_PATH
from indexer import COLLECTION_NAME, EMBEDDING_MODEL_NAME, DEFAULT_CHROMA_DIR, DEFAULT_BM25_PATH

VECTOR_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
RERANK_INPUT_K = 10
FINAL_TOP_K = 3
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class RetrieverConfig:
    chunks_path: Path = DEFAULT_CHUNKS_PATH
    bm25_path: Path = DEFAULT_BM25_PATH
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    collection_name: str = COLLECTION_NAME
    embedding_model_name: str = EMBEDDING_MODEL_NAME
    reranker_model_name: str = RERANKER_MODEL_NAME

@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    vector_rank: int |None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

def load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except OSError as exc:
        raise OSError(f"Could not load retrieval artifact: {path}") from exc
    
def create_embedding_model(model_name:str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"Could not load embedding model: {model_name}") from exc
    
def create_reranker(model_name: str) -> Any:
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except Exception as exc:
        raise RuntimeError(f"Could not load reranker model: {model_name}") from exc  
    
def create_chroma_collection(config:RetrieverConfig):
    try:
        import chromadb 
        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        return client.get_collection(name= config.collection_name)
    except Exception as exc:
        raise RuntimeError("Could not open ChromaDB collection.") from exc
    
def tokenize(text:str):
    return TOKEN_PATTERN.findall(text.lower())

def metadata_matches_filters(
        metadata: dict[str, Any],
        company_name: str | None,
        filing_type: str|None,
) -> bool :
    company_ok = company_name is None or metadata.get("company_name") == company_name
    filing_ok = filing_type is None or metadata.get("filing_type") == filing_type
    return company_ok and filing_ok

def build_chroma_where(company_name: str, filing_type:str) -> dict[str, Any]:
    clauses: list[dict[str,str]] = []
    if company_name:
        clauses.append({"company_name": company_name})
    if filing_type:
        clauses.append({"filing_type": filing_type})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses} if clauses else None

class FinancialRetriever:
    def __init__(self, config: RetrieverConfig | None= None) -> None:
        self.config = config or RetrieverConfig()
        self.chunks: list[TextChunk] = load_pickle(self.config.chunks_path)
        bm25_payload = load_pickle(self.config.bm25_path)
        self.bm25 = bm25_payload["bm25"]
        self.collection = create_chroma_collection(self.config)
        self.embedding_model = create_embedding_model(self.config.embedding_model_name)
        self.reranker = create_reranker(self.config.reranker_model_name)

    def embed_query(self, query:str) -> list[float]:
        try:
            vector = self.embedding_model.encode([query], normalize_embeddings= True)[0]
            return vector.tolist()
        except Exception as exc:
           raise RuntimeError("Failed to embed retrieval query.") from exc

    def vector_search(self, query: str, company_name: str | None, filing_type: str| None) -> list[RetrievalResult]:
        where = build_chroma_where(company_name, filing_type)
        try:
            result = self.collection.query(
                query_embeddings = [self.embed_query(query)],
                n_results = VECTOR_TOP_K,
                where = where,
                include = ["documents","metadatas"]
            )
        except Exception as exc:
            raise RuntimeError("Chroma vector search failed.") from exc
        return self._vector_results_from_chroma(result)

    def _vector_results_from_chroma(self, result: dict[str, Any]) -> list[RetrievalResult]:
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            RetrievalResult(chunk_id , text, metadata, vector_rank=index + 1)
            for index , (chunk_id , text, metadata) in enumerate(zip(ids, documents, metadatas))
        ]
    
    def bm25_search(self, query: str, company_name: str | None, filing_type: str | None):
        scores = self.bm25.get_scores(tokenize(query))
        ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        results: list[RetrievalResult] = []
        for index in ranked_indexes:
            chunk = self.chunks[index]
            if metadata_matches_filters(chunk.metadata,company_name,filing_type):
                results.append(RetrievalResult(chunk.chunk_id, chunk.text, chunk.metadata, bm25_rank=len(results) + 1))
            if len(results) >= BM25_TOP_K:
                break
        return results
    
    def retrieve(self,query: str, company_name: str| None = None, filing_type: str| None = None) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty.")
        vector_results , bm25_results = self._run_parallel_searches(query, company_name, filing_type)
        fused_results = reciprocal_rank_fusion(vector_results, bm25_results)
        return self.rerank(query, fused_results[:RERANK_INPUT_K])[:FINAL_TOP_K]
    
    def _run_parallel_searches(self,query: str, company_name: str| None , filing_type: str| None) -> tuple[list[RetrievalResult],list[RetrievalResult]]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self.vector_search, query, company_name, filing_type)
            bm25_future = executor.submit(self.bm25_search, query, company_name, filing_type)
            return vector_future.result(), bm25_future.result()
    
    def rerank(self, query:str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        if not candidates:
            return []
        try:
            scores = self.reranker.predict([(query, candidate.text) for candidate in candidates])
        except Exception as exc:
            raise RuntimeError("Cross-encoder reranking failed.") from exc
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        return sorted(candidates, key=lambda item : item.rerank_score or 0.0 , reverse=True)
    
def reciprocal_rank_fusion(
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
) -> list[RetrievalResult]:
    fused: dict[str, RetrievalResult] = {}
    for result in vector_results:
        fused[result.chunk_id] = result
        result.rrf_score += 1 / (RRF_K + (result.vector_rank or VECTOR_TOP_K))
    for result in bm25_results:
        existing = fused.get(result.chunk_id, result)
        existing.bm25_rank = result.bm25_rank
        existing.rrf_score += 1 / (RRF_K + (result.bm25_rank or BM25_TOP_K))
        fused[result.chunk_id] = existing
    return sorted(fused.values(), key=lambda item: item.rrf_score, reverse=True)
