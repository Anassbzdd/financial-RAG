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

from sec_edgar_downloader import Downloader

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


DEFAULT_COMPANIES: tuple[Company, ...] = (
    Company("Apple", "AAPL", "0000320193"),
    Company("Microsoft", "MSFT", "0000789019"),
    Company("Nvidia", "NVDA", "0001045810"),
    Company("Amazon", "AMZN", "0001018724"),
    Company("Alphabet", "GOOGL", "0001652044"),
    Company("Meta", "META", "0001326801"),
    Company("Tesla", "TSLA", "0001318605"),
    Company("Berkshire Hathaway", "BRK-B", "0001067983"),
    Company("JPMorgan Chase", "JPM", "0000019617"),
    Company("Johnson and Johnson", "JNJ", "0000200406"),
)

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        LOGGER.debug("python-dotenv is not installed; reading OS environment only.")
        return
    load_dotenv()

def build_config_from_environment() -> IngestionConfig:
    load_dotenv_if_available()
    downloader_email = os.getenv("SEC_DOWNLOADER_EMAIL")
    downloader_company = os.getenv(
        "SEC_DOWNLOADER_COMPANY",
        "Financial RAG Portfolio Project",
    )
    return IngestionConfig(
        downloader_company_name = downloader_company,
        downloader_email = downloader_email,
    )

def validate_config(config:IngestionConfig) -> None:
    if config.downloader_email is None or not config.downloader_email.strip():
        raise ValueError(
            "SEC_DOWNLOADER_EMAIL is required. Add it to .env, for example: "
            "SEC_DOWNLOADER_EMAIL=your.email@example.com"
        )
    
    if config.annual_limit <= 0:
        raise ValueError("annual_limit must be a positive integer.")
    
    if config.quarterly_limit <= 0:
        raise ValueError("quarterly_limit must be a positive integer.")
    
def ensure_directories(config:IngestionConfig):
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.sec_cache_dir.mkdir(parents=True, exist_ok=True)

def create_downloader(config:IngestionConfig):
    validate_config(config)
    ensure_directories(config)
    return Downloader(
        config.downloader_company_name,
        config.downloader_email,
        config.sec_cache_dir,
    )

def slugify_company_name(company_name) -> str:
    normalized = company_name.lower().replace("&", "and")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")

def infer_quarter_from_report_period(report_period:str| None) -> int| None:
    if not report_period or re.fullmatch(r"\d{8}", report_period):
        return None
    month = report_period[4:6]
    return ((month - 1) // 3) + 1

def infer_fiscal_year(report_period:str|None, filing_date:str|None):
    candidate_date = report_period or filing_date
    if not candidate_date or re.fullmatch(r"\d{8}", report_period):
        return None
    return int(candidate_date[:4])

def build_output_pdf_path(
    raw_dir: Path,
    company: str,
    filing_type:str,
    fiscal_year:int,
    quarter:int | None,
)-> Path:
    company_slug = slugify_company_name(company.name)
    quarter_suffix = f"_q{quarter}" if filing_type == QUARTERLY_FILING_TYPE and quarter else ""
    filename = f"{company_slug}_{filing_type}_{fiscal_year}{quarter_suffix}.pdf"
    return raw_dir / filename

def parse_sec_header_value(full_submission_text:str, header_name:str) -> str | None:
    pattern = rf"^\s*{re.escape(header_name)}:\s*(?P<value>.+?)\s*$"
    match = re.search(pattern, full_submission_text, flags= re.MULTILINE)
    return match.group("value") if match else None

def read_full_submission_text(filing_directory: Path) -> str:
    full_submission_path = filing_directory / "full-submission.txt"
    if not full_submission_path.exists():
        raise FileNotFoundError(f"Missing SEC full submission file: {full_submission_path}")
    return full_submission_path.read_text(encoding="utf-8", errors="replace")

def extract_metadata(
        filing_directory: Path,
        company: Company,
        filing_type: str,
) -> FilingMetadata:
    
    full_submission_text = read_full_submission_text(filing_directory)
    report_period = parse_sec_header_value(full_submission_text,"CONFORMED PERIOD OF REPORT")
    filing_date = parse_sec_header_value(full_submission_text,"FILED AS OF DATE")
    accesion_number = parse_sec_header_value(full_submission_text,"ACCESSION NUMBER")
    fiscal_year = infer_fiscal_year(report_period=report_period, filing_date=filing_date)
    quarter = (
        infer_quarter_from_report_period(report_period)
        if filing_type == QUARTERLY_FILING_TYPE
        else None
    )
    return FilingMetadata(
        company_name = company.name,
        ticker = company.ticker,
        filing_type = filing_type,
        fiscal_year = fiscal_year,
        quarter = quarter,
        filing_date = filing_date,
        report_period = report_period,
        accesion_number = accesion_number,
    )  

def find_company_filing_root(config:IngestionConfig, company:Company, filing_type:str) -> Path:
    return config.sec_cache_dir / "sec-edgar-filings" / company.name / filing_type

def list_downloaded_filing_directories(config:IngestionConfig, company:Company, filing_type:str) -> list[Path]:
    filing_root = find_company_filing_root(config, company, filing_type)
    if not filing_root.exists():
        return []
    
    return sorted(path for path in filing_root.iterdir() if path.is_dir() )

def sor_filing_directories_by_report_date(
        filing_directories: Iterable[Path],
        company: Company,
        filing_type: str
):
    sortable_directories: list[tuple[str, Path]] = []
    for filing_directory in filing_directories:
        try:
            metadata = extract_metadata(filing_directory, company, filing_type)
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.warning("Skipping %s because metadata could not be read: %s", filing_directory, exc)
            continue
        data_key = metadata.report_period or metadata.filing_date
        sortable_directories.append((data_key,filing_directory))
        return [path for _, path in sorted(sortable_directories, reverse = True)]
    
def find_primary_document(filing_directory: Path) -> Path:
    preferred_names= (
        "primary-document.html",
        "primary-document.htm",
        "primary-document.txt",
    )
    for filename in preferred_names:
        candidate = filing_directory / filename
        if candidate.exists():
            return candidate
    
    html_candidate = sorted(
        path
        for path in filing_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".html",".htm",".txt"}
    )

    if not html_candidate:
        raise FileNotFoundError(f"No primary HTML/TXT filing document found in {filing_directory}")
    
    return html_candidate[0]

def convert_document_to_pdf(source_path: Path, output_pdf_path: Path):
    from weasyprint import HTML
    output_pdf_path.mkdir(parents=True, exist_ok=True)

    if source_path.suffix.lower() == '.txt':
        text = source_path.read_text(encoding="utf-8", errors="replace")
        html_string = "<html><body><pre>{text}</pre></body></html>"
        HTML(string = html_string, base_url=str(source_path.parent)).write_pdf(str(output_pdf_path))
    else:
        HTML(filename=str(source_path), base_url=str(source_path.parent)).write_pdf(str(output_pdf_path))
    validate_pdf(output_pdf_path)

def validate_pdf(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected PDF was not created: {pdf_path}")
    if pdf_path.stat().st_size < MINIMUM_VALID_PDF_BYTES:
        raise ValueError(f"Generated PDF appears too small to be valid: {pdf_path}")


    



