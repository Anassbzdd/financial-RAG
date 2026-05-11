from __future__ import annotations

import re
import os
import logging
import shutil
import time
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sec_edgar_downloader import _Downloader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_SEC_CACHE_DIR = PROJECT_ROOT / "data" / "sec-edgar-cache"

ANNUAL_FILING_TYPE = "10-K"
QUARTERLY_FILING_TYPE = "10-Q"
DEFAULT_ANNUAL_LIMIT = 3
DEFAULT_QUARTERLY_LIMIT = 2
SECONDS_BETWEEN_COMPANY_REQUESTS = 0.2
MINIMUM_VALID_PDF_BYTES = 1024

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class Company:
    name: str 
    ticker: str
    edgar_identifier: str

@dataclass(frozen=True)
class FilingMetadata:
    company_name: str
    ticker: str
    filing_type: str
    fiscal_year: int
    quarter: int | None
    filing_date: str | None
    report_period: str | None
    accesion_number: str | None

@dataclass(frozen=True)
class IngestedFiling:
    source_directory: Path
    primary_document_path: Path
    output_pdf_path: Path
    metadata: FilingMetadata

@dataclass(frozen=True)
class IngestionConfig:
    raw_dir: Path = DEFAULT_RAW_DIR
    sec_cache_dir: Path = DEFAULT_SEC_CACHE_DIR
    downloader_company_name: str = "Financial RAG Portfolio Project"
    downloader_email: str | None = None
    annual_limit: int = DEFAULT_ANNUAL_LIMIT
    quarterly_limit: int = DEFAULT_QUARTERLY_LIMIT
    overwrite_existing_pdfs: bool = False
    keep_sec_cache: bool = True



