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
    
