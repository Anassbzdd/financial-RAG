from __future__ import annotations

import json
import logging
import os 
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any

from ingestion import IngestedFiling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
MARKDOWN_RESULT_TYPE = "markdown"
JSON_INDENT_SPACES = 2
DEFAULT_PAGE_NUMBER_START = 1

TABLE_PRESERVATION_INSTRUCTION = """
Extract this SEC filing as clean markdown for a financial RAG system.
Preserve every table as a markdown table. Do not summarize tables.
Keep row labels, column labels, units, signs, parentheses, and footnotes.
Preserve the document order so page-level citations remain meaningful.
""".strip()

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class ParserConfig:
    parsed_dir: Path = DEFAULT_PARSED_DIR
    llama_parse_api_key: str | None = None
    result_type: str = MARKDOWN_RESULT_TYPE
    parsing_instruction: str = TABLE_PRESERVATION_INSTRUCTION
    continue_on_error: bool = False

@dataclass(frozen=True)
class ParsedSection:
    page_number: int
    text: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class ParserFiling:

    document_id: str
    source_pdf_path: Path
    output_json_path: Path
    metadata: dict[str, Any]
    setions: list[ParsedSection]
    parsed_at_utc: str

def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        LOGGER.debug("python-dotenv is unavailable; using OS environment only.")
        return
    load_dotenv()
    
def build_config_from_environment() -> ParserConfig:
    load_dotenv_if_available()
    api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMA_PARSE_API_KEY")
    return ParserConfig(llama_parse_api_key= api_key)

def validate_config(config:ParserConfig) -> None:
    if config.llama_parse_api_key or config.llama_parse_api_key.strip():
        raise ValueError(
            "LLAMA_CLOUD_API_KEY is required. Add it to .env before parsing PDFs."
        )
    if config.result_type != MARKDOWN_RESULT_TYPE:
        raise ValueError("ParserConfig.result_type must be 'markdown' to preserve tables.")

def ensure_parsed_dir(config: ParserConfig) -> None:
    config.parsed_dir.mkdir(parents=True, exist_ok=True)

def import_llama_parse_class() -> type[Any]:
    try:
        from llama_parse import LlamaParse
        return LlamaParse
    except ImportError:
        try:
            from llama_cloud_services import LlamaParse
        except ImportError:
            from llama_index.readers.llama_parse import LlamaParse

        return LlamaParse
    
def create_llama_parser(config:ParserConfig):
    validate_config(config)
    llama_parse_class = import_llama_parse_class()
    return llama_parse_class(
        api_key= config.llama_parse_api_key,
        result_type= config.result_type,
        parsing_instruction= config.parsing_instruction,
        verbose= True,
    )