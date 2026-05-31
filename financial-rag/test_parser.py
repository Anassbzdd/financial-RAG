from dataclasses import replace

from src.ingestion import Company, build_config_from_environment, create_downloader, ingest_company
from src.parser import parse_filings

company = Company("Apple", "AAPL", "0000320193")

config = replace(
    build_config_from_environment(),
    annual_limit=1,
    quarterly_limit=1,
)

downloader = create_downloader(config)
filings = ingest_company(downloader, config, company)

parsed_filings = parse_filings(filings)

print(f"Parsed {len(parsed_filings)} filings")

for parsed in parsed_filings:
    print(parsed.output_json_path)