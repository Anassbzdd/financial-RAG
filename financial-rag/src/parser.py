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
class ParsedFiling:

    document_id: str
    source_pdf_path: Path
    output_json_path: Path
    metadata: dict[str, Any]
    sections: list[ParsedSection]
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
    if config.llama_parse_api_key is None or not config.llama_parse_api_key.strip():
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

def validate_ingested_filing(filing: IngestedFiling) -> None:
    if not filing.output_pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF for parsing: {filing.output_pdf_path}")
    
    if filing.output_pdf_path.suffix != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {filing.output_pdf_path}")
    
def filing_metadata_to_dict(filing: IngestedFiling) -> dict[str,Any]:
    return asdict(filing.metadata)

def build_document_id(filing: IngestedFiling) -> str:
    accession = filing.metadata.accession_number 
    if accession:
        return accession.replace("-", "")
    return filing.output_pdf_path.stem

def build_output_json_path(config: ParserConfig,filing: IngestedFiling) -> Path:
    document_id = build_document_id(filing)
    return config.parsed_dir / f"{document_id}.json"

def extract_document_text(document: Any) -> str:
    if hasattr(document, "text") and isinstance(document.text, str):
        return document.text
    
    if hasattr(document,"get_content"):
        content = document.get_content()
        return content if isinstance(content,str) else str(content)
    
    return str(document)
    
def extract_document_metadata(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", {})
    return metadata if isinstance(metadata,dict) else {}

def normalize_markdown(markdown_text:str) -> str:
    normalized = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()

def extract_page_number(llama_metadata: dict[str, Any], fallback_index:int) -> int:
    for key in ("page_number", "page_label", "page"):
        value = llama_metadata.get(key)
        if value is not None:
            return int(value)
        
    return fallback_index + DEFAULT_PAGE_NUMBER_START

def build_section_metadata(
    filing_metadata: dict[str, Any],
    llama_metadata: dict[str, Any],
    page_number: int,
) -> dict[str,Any]:
    return {
        **filing_metadata,
        "llama_metadata": llama_metadata,
        "page_number": page_number
    }

def convert_llama_documents_to_sections(
    documents: Iterable[Any],
    filing_metadata: dict[str,Any],
) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    for index, document in enumerate(documents):
        llama_metadata = extract_document_metadata(document)
        page_number = extract_page_number(llama_metadata, index)
        text = normalize_markdown(extract_document_text(document))
        metadata = build_section_metadata(filing_metadata, llama_metadata, page_number)
        if text:
            sections.append(ParsedSection(page_number, text, metadata))
    return sections

def parse_pdf_with_llamaparse(parser:Any, filing: IngestedFiling) -> list[Any]:
    try:
        documents = parser.load_data(str(filing.output_pdf_path))
    except Exception as exc:
        raise RuntimeError(f"LlamaParse failed for {filing.output_pdf_path}") from exc
    
    if not documents:
        raise ValueError(f"LlamaParse returned no content for {filing.output_pdf_path}")

    return list(documents)

def parsed_filing_to_dict(parsed_filing:ParsedFiling) -> dict[str, Any]:

    return {
        "document_id": parsed_filing.document_id,
        "source_pdf_path": str(parsed_filing.source_pdf_path),
        "output_json_path": parsed_filing.output_json_path,
        "metadata": parsed_filing.metadata,
        "sections": [asdict(section) for section in parsed_filing.sections],
        "parsed_at_utc": parsed_filing.parsed_at_utc,
    }

def save_parsed_filing(parsed_filing: ParsedFiling) -> None:
    try:
        parsed_filing.output_json_path.write_text(
            json.dumps(parsed_filing_to_dict(parsed_filing), indent= JSON_INDENT_SPACES),
            encoding="utf-8",
        )
    except OSError as exc:
        raise OSError(f"Could not write parsed JSON: {parsed_filing.output_json_path}") from exc
    
def parse_filing(parser: Any, config: ParserConfig, filing: IngestedFiling) -> ParsedFiling:
    validate_ingested_filing(filing)
    filing_metadata = filing_metadata_to_dict(filing)
    documents = parse_pdf_with_llamaparse(parser, filing)
    sections = convert_llama_documents_to_sections(documents, filing_metadata)
    if not sections:
        raise ValueError(f"No non-empty markdown sections found for {filing.output_pdf_path}")

    parsed_filing = ParsedFiling(
        document_id =build_document_id(filing),
        source_pdf_path =filing.output_pdf_path,
        output_json_path =build_output_json_path(config, filing),
        metadata =filing_metadata,
        sections =sections,
        parsed_at_utc =datetime.now(timezone.utc).isoformat(),
    )
    save_parsed_filing(parsed_filing)
    return parsed_filing

def parse_filings(
        filings: Iterable[IngestedFiling],
        config: ParserConfig | None = None,
) -> list[ParsedFiling]:
    
    active_config = config or build_config_from_environment()
    ensure_parsed_dir(active_config)
    parser = create_llama_parser(active_config)
    parsed_filings: list[ParsedFiling] = []

    for filing in filings:
        try:
            parsed_filings.append(parse_filing(parser, active_config, filing))
            LOGGER.info("Parsed %s.", filing.output_pdf_path.name)
        except Exception as exc:
            if not active_config.continue_on_error:
                raise
            LOGGER.exception("Skipping %s after parse failure: %s", filing.output_pdf_path, exc)

    return parsed_filings