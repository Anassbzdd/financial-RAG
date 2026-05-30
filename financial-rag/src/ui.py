from __future__ import annotations

from functools import lru_cache
from typing import Any

import gradio as gr

from pipeline import build_default_pipeline, QueryFilters

COMPANY_OPTIONS = [
    "All companies",
    "Apple",
    "Microsoft",
    "Nvidia",
    "Amazon",
    "Alphabet",
    "Meta",
    "Tesla",
    "Berkshire Hathaway",
    "JPMorgan Chase",
    "Johnson and Johnson",
]
FILING_OPTIONS = ["All filings", "10-K", "10-Q"]

CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; margin: auto !important; }
#title { text-align: center; margin-bottom: 8px; }
#subtitle { text-align: center; color: #475569; margin-bottom: 24px; }
"""

@lru_cache(maxsize=1)
def get_pipeline() -> Any :
    return build_default_pipeline()

def normalize_filter(value: str) -> str | None:
    return None if value.startswith("All") else value

def format_sources(sources: list[Any]) -> str:

    if not sources:
        return "No sources returned."
    lines: list[str] = []
    for source in sources:
        metadata = source.metadata
        label = build_source_label(source.source_id,metadata)
        lines.append(f"**{label}**\n\n{source.text[:900]}...")
    return "\n\n---\n\n".join(lines)

def build_source_label(source_id: int, metadata: dict[str, Any]) -> str:
    company = metadata.get("company_name", "Unknown company")
    filing = metadata.get("filing_type", "Unknown filing")
    year = metadata.get("fiscal_year", "Unknown year")
    page = metadata.get("page_number", "Unknown page")
    return f"Source {source_id}: {company} | {filing} | {year} | page {page}"

def answer_question(question:str, company:str, filing_type:str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""
    filters = QueryFilters(normalize_filter(company), normalize_filter(filing_type))
    response = get_pipeline().answer_question(question, filters)
    return response.answer, format_sources(response.sources)

def build_interface() -> gr.Blocks:
    """Build a clean Gradio interface for portfolio demos."""

    with gr.Blocks(css=CUSTOM_CSS, title="Financial RAG") as demo:
        gr.Markdown("# Financial RAG", elem_id="title")
        gr.Markdown("Ask SEC filing questions with hybrid retrieval and cited sources.", elem_id="subtitle")
        with gr.Row():
            company = gr.Dropdown(COMPANY_OPTIONS, value="All companies", label="Company")
            filing = gr.Dropdown(FILING_OPTIONS, value="All filings", label="Filing type")
        question = gr.Textbox(label="Question", lines=3, placeholder="What drove Nvidia's data center revenue?")
        submit = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", lines=8)
        sources = gr.Markdown(label="Sources")
        submit.click(answer_question, inputs=[question, company, filing], outputs=[answer, sources])
    return demo


if __name__ == "__main__":
    build_interface().launch()

