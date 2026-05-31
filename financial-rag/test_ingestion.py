from dataclasses import replace

from src.ingestion import (
    Company,
    build_config_from_environment,
    configure_logging,
    create_downloader,
    ingest_company,
)

configure_logging()

company = Company("Apple", "AAPL", "0000320193")

config = replace(
    build_config_from_environment(),
    annual_limit=1,
    quarterly_limit=1,
)

downloader = create_downloader(config)
filings = ingest_company(downloader, config, company)

print(f"Created {len(filings)} filing PDFs")

for filing in filings:
    print(filing.output_pdf_path)
