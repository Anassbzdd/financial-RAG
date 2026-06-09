from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import build_default_pipeline, QueryFilters

app = FastAPI(title= "Financial RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    company_name: str | None = None
    filing_type: str | None = None

class SourceResponse(BaseModel):
    source_id: int
    text: str
    metadata: dict[str,Any]

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    latency_ms: int

class HealthResponse(BaseModel):
    status: str

@lru_cache(maxsize=1)
def get_pipeline() -> Any:
    return build_default_pipeline()

@app.on_event("startup")
def warm_up_pipeline() -> None:
    get_pipeline()

@app.get("/health", response_model= HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.post("/query", response_model=QueryResponse)
def query(request:QueryRequest) -> QueryResponse:
    try:
        filters = QueryFilters(request.company_name, request.filing_type)
        response = get_pipeline().answer_question(request.question, filters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    sources = [SourceResponse(**source.__dict__) for source in response.sources]
    return QueryResponse(answer= response.answer ,sources=sources,latency_ms= response.latency_ms)
