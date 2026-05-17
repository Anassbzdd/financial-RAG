from __future__ import annotations

import json 
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.pkl"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text:str
    metadata: dict[str, Any]

def import_text_splitter_class() -> type[Any]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter

def create_text_splitter() -> Any:
    splitter_class = import_text_splitter_class()
    return splitter_class(chunk_size= CHUNK_SIZE,chunk_overlap = CHUNK_OVERLAP)

def list_parsed_json_files(parsed_dir: Path) -> list[Path]:
    if not parsed_dir.exists():
        raise FileNotFoundError(f"Parsed directory does not exist: {parsed_dir}")
    return sorted(path for path in parsed_dir.glob("*.json") if path.is_file())

def load_parsed_json(path:Path) -> dict[str,Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid parsed JSON file: {path}") from exc
    except OSError as exc:
        raise OSError(f"Could not read parsed JSON file: {path}") from exc
    
def normalize_metadata(metadata: dict[str, Any]) -> dict[str,Any]:
    return {
        "company_name": metadata.get("company_name") or metadata.get("company"),
        "ticker": metadata.get("ticker"),
        "filing_type": metadata.get("filing_type"),
        "fiscal_year": metadata.get("fiscal_year"),
        "quarter": metadata.get("quarter"),
        "page_number" : metadata.get("page_number"),
    }
    
def iter_parsed_sections(parsed: dict[str,Any]) -> Iterable[tuple[str,dict[str,Any]]]:
    root_metadata = parsed.get("metadata", {})
    for section in parsed.get("sections", []):
        text = section.get("text", "")
        metadata = {**root_metadata,**section.get("metadata", {})}
        if text.strip():
            yield text, metadata

def build_chunk_id(document_id: str, chunk_index:int) -> str:
    return f"{document_id}::chunk-{chunk_index:05d}"

def split_section(splitter:Any, text:str) -> list[str]:
    try:
        return splitter.split_text(text)
    except Exception as exc:
        raise RuntimeError("RecursiveCharacterTextSplitter failed on parsed section.") from exc

def chunk_parsed_document(path:Path, splitter: Any) -> list[TextChunk]:
    parsed = load_parsed_json(path)
    document_id = parsed.get("document_id") or path.stem
    chunks: list[TextChunk] = []

    for section_text, metadata in iter_parsed_sections(parsed):
        for chunk_text in split_section(splitter,section_text):
            chunk_index = len(chunk_text)
            chunk_metadata = normalize_metadata(metadata)
            chunk_metadata.update({"document_id": document_id, "chunk_index": chunk_index})
            chunks.append(TextChunk(build_chunk_id(document_id, chunk_index),chunk_text,chunk_metadata))

    return chunks

def save_chunks(output_path:Path, chunks: list[TextChunk]):
    try:
        output_path.parent.mkdir(parents= True, exist_ok = True)
        with output_path.open("wb") as file:
            pickle.dump(chunks, file)
    except OSError as exc:
        raise OSError(f"Could not save chunks to {output_path}") from exc
    
def build_chunks(
        parsed_dir: Path = DEFAULT_PARSED_DIR,
        output_path: Path = DEFAULT_CHUNKS_PATH,
):
    splitter = create_text_splitter()
    all_chunks: list[TextChunk] = []
    for path in list_parsed_json_files(parsed_dir):
        document_chunks = chunk_parsed_document(path,splitter)
        LOGGER.info("Chunked %s into %s chunks.", path.name, len(document_chunks))
        all_chunks.extend(document_chunks)
    
    if not all_chunks:
        raise ValueError(f"No chunks were created from parsed files in {parsed_dir}")
    
    save_chunks(output_path, all_chunks)
    return all_chunks
