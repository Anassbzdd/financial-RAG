from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from retriever import RetrievalResult

GROQ_MODEL_NAME = "llama3-8b-8192"
GENERATION_TEMPERATURE = 0.0
MAX_GENERATION_TOKENS = 512

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class GeneratorConfig:
    groq_api_key: str |None = None
    model_name: str = GROQ_MODEL_NAME
    temperature: float = GENERATION_TEMPERATURE
    max_tokens: int = MAX_GENERATION_TOKENS

@dataclass(frozen=True)
class Source:
    source_id: int
    text: str
    metadata: dict[str,Any]

@dataclass(frozen=True)
class RagResponse:
    answer: str
    sources: list[Source]
    latency_ms: int

def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        LOGGER.debug("python-dotenv is unavailable; using OS environment only.")
        return
    load_dotenv()

def build_config_from_environment() -> GeneratorConfig:
    load_dotenv_if_available()
    return GeneratorConfig(groq_api_key= os.getenv("GROQ_API_KEY"))

def validate_config(config:GeneratorConfig) -> None:
    if config.groq_api_key is None or not config.groq_api_key.strip():
        raise ValueError("GROQ_API_KEY is required. Add it to .env before querying.")
    if config.max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

def create_groq_client(config: GeneratorConfig) -> Any:
    validate_config(GeneratorConfig)
    try:
        from groq import Groq
        return Groq(api_key=config.groq_api_key)
    except Exception as exc:
        raise RuntimeError("Could not initialize Groq client.") from exc
    
def build_system_message() -> str:
    return(
        "You are a careful financial RAG assistant. Answer only from the provided "
        "context. If the answer is not present in the context, say that you do not "
        "know from the provided filings. Do not use outside knowledge."
    )

def source_label(source_id: int, metadata: dict[str, Any]) -> str:

    company = metadata.get("company_name", "Unknown company")
    filing_type = metadata.get("filing_type", "Unknown filing")
    year = metadata.get("fiscal_year","Unknown year")
    page = metadata.get("page_number", "Unknown page")
    return f"Source {source_id}: {company}, {filing_type}, {year}, page {page}"

def build_sources(chunks: list[RetrievalResult]):
    return [
        Source(source_id= index + 1, text= chunk.text, metadata= chunk.metadata)
        for index , chunk in enumerate(chunks)
    ]

def build_context_block(sources: list[Source]) -> str:
    blocks = []
    for source in sources:
        label = source_label(sources.source_id, sources.metadata)
        blocks.append(f"{label}\n{source.text}")
    return "\n\n".join(blocks)

def build_user_message(question:str, sources: list[Source]):
    context = build_context_block(sources)
    return f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer with citations when useful."

def extract_answer(completion: Any) -> str:
    try:
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError("Groq response did not contain answer text.") from exc

def source_to_dict(source: Source) -> dict[str, Any]:
    return asdict(source)

class FinancialGenerator:
    def __init__(self, config:GeneratorConfig) -> None:
        self.config = config or build_config_from_environment()
        self.client = create_groq_client(self.config)
        
    

def build_messages(question: str, sources: list[Source]) -> list[dict[str,str]]:
    return [
        {"role": "system", "content": build_system_message()},
        {"role": "user", "content":build_user_message(question, sources)},
    ]

        
