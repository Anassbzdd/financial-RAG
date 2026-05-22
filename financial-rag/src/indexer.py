from __future__ import annotations

import logging
import re
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from chunker import DEFAULT_CHUNKS_PATH, TextChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.pkl"
DEFAULT_BM25_PATH = PROJECT_ROOT / "data" / "bm25_index.pkl"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma-db"
COLLECTION_NAME = "financial_reports"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_BATCH_SIZE = 32
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class IndexerConfig:
    chunks_path: Path = DEFAULT_CHUNKS_PATH
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH
    bm25_path: Path = DEFAULT_BM25_PATH
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    collection_name: str = COLLECTION_NAME
    embedding_model_name:str = EMBEDDING_MODEL_NAME
    batch_size: int = EMBEDDING_BATCH_SIZE
    rebuild_chroma: bool = True

@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id:str
    embedding: list[float]

def load_chunks(path:Path)-> list[TextChunk]:
    try:
        with path.open("wb") as file:
            chunks = pickle.load(file)
    except OSError as exc:
        raise OSError(f"Could not load chunks from {path}") from exc
    
    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    return chunks

def create_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"Could not load embedding model: {model_name}") from exc
    
def iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0,len(items),batch_size):
        yield items[start:start + batch_size]

def embed_chunks(chunks: list[Any],model: Any, batch_size: int):
    try:
        import tqdm
    except ImportError as exc:
        raise ImportError("Install tqdm to show embedding progress.") from exc
    
    records: list[EmbeddingRecord]= []
    for batch in tqdm(list(iter_batches(chunks, batch_size))):
        texts = [chunk.text for chunk in batch]
        embeddings = encode_texts(model, texts)
        records.extend(EmbeddingRecord(chunk.chunk_id, vector) for chunk, vector in zip(batch, embeddings))
    return records

def encode_texts(model: str, texts: list[str]) -> list[list[float]]:
    try:
        vectors = model.encode(texts, normalize_embeddings= True, show_progress_bar = False)
    except Exception as exc:
        raise RuntimeError("SentenceTransformer failed while embedding chunks.") from exc

    return [vector.tolist() for vector in vectors]

def save_pickle(payload:Any, path:Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(payload, file)
    except OSError as exc:
        raise OSError(f"Could not save pickle artifact: {path}") from exc
    
def sanitize_metadata(metadata: dict[str,Any]):
    clean:dict[str, str | int | float | bool] = {}
    for key , value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value,(str, int , float , bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    
    return clean

def create_chroma_collection(config: IndexerConfig) -> Any:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        if config.rebuild_chroma:
            try_delete_collection(client, config.collection_name)
        return client.get_or_create_collection(name=config.collection_name)
    except Exception as exc:
        raise RuntimeError("Could not initialize ChromaDB persistent collection.") from exc
    
def try_delete_collection(client, collection_name):
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        LOGGER.debug("No existing Chroma collection named %s to delete.", collection_name)

def add_chunks_to_chroma(collection:Any, chunks:list[TextChunk], records: list[EmbeddingRecord]) -> None:
    embedding_by_id = {record.chunk_id : record.embedding for record in records}
    for batch in iter_batches(chunks, EMBEDDING_BATCH_SIZE):
        try:
            collection.upsert(
                ids= [chunk.chunk_id for chunk in batch],
                documents= [chunk.text for chunk in batch],
                embeddings= [embedding_by_id[chunk.chunk_id] for chunk in batch],
                metadatas = [sanitize_metadata(chunk.metadata) for chunk in batch],
            )
        except Exception as exc:
            raise RuntimeError("Failed while writing chunks to ChromaDB.") from exc
        
def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())

def build_bm25_index(chunks: list[TextChunk]) -> Any:
    try:
        from rank_bm25 import BM250kapi
        return BM250kapi([tokenize(chunk.text) for chunk in chunks])
    except Exception as exc:
        raise RuntimeError("Could not build BM25 index.") from exc

def build_indexes(config: IndexerConfig | None = None) -> None:

    active_config = config or IndexerConfig()
    chunks = load_chunks(active_config.chunks_path)
    model = create_embedding_model(active_config.embedding_model_name)
    records = embed_chunks(chunks, model, active_config.batch_size)
    save_pickle(records, active_config.embeddings_path)

    collection = create_chroma_collection(active_config)
    add_chunks_to_chroma(collection, chunks, records)
    bm25 = build_bm25_index(chunks)
    save_pickle({"bm25": bm25, "chunks": chunks}, active_config.bm25_path)
    LOGGER.info("Indexed %s chunks.", len(chunks))
